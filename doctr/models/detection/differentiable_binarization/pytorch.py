# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.deform_conv import DeformConv2d
from torchvision.models import resnet34, resnet50
from typing import List, Dict, Any, Optional

from .base import DBPostProcessor, _DBNet
from ...backbones import mobilenet_v3_large
from ...utils import load_pretrained_params

__all__ = ['DBNet', 'db_resnet50', 'db_resnet34', 'db_mobilenet_v3_large']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'db_resnet50': {
        'backbone': resnet50,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'input_shape': (3, 1024, 1024),
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.1/db_resnet50-ac60cadc.pt',
    },
    'db_resnet34': {
        'backbone': resnet34,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'input_shape': (3, 1024, 1024),
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'url': None,
    },
    'db_mobilenet_v3_large': {
        'backbone': mobilenet_v3_large,
        'backbone_submodule': 'features',
        'fpn_layers': ['3', '6', '12', '16'],
        'input_shape': (3, 1024, 1024),
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.1/db_mobilenet_v3_large-fd62154b.pt',
    },
}


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        deform_conv: bool = False,
    ) -> None:

        super().__init__()

        out_chans = out_channels // len(in_channels)

        conv_layer = DeformConv2d if deform_conv else nn.Conv2d

        self.in_branches = nn.ModuleList([
            nn.Sequential(
                conv_layer(chans, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for idx, chans in enumerate(in_channels)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_branches = nn.ModuleList([
            nn.Sequential(
                conv_layer(out_channels, out_chans, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2 ** idx, mode='bilinear', align_corners=True),
            ) for idx, chans in enumerate(in_channels)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) != len(self.out_branches):
            raise AssertionError
        # Conv1x1 to get the same number of channels
        _x: List[torch.Tensor] = [branch(t) for branch, t in zip(self.in_branches, x)]
        out: List[torch.Tensor] = [_x[-1]]
        for t in _x[:-1][::-1]:
            out.append(self.upsample(out[-1]) + t)

        # Conv and final upsampling
        out = [branch(t) for branch, t in zip(self.out_branches, out[::-1])]

        return torch.cat(out, dim=1)


class DBNet(_DBNet, nn.Module):

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        head_chans: int = 256,
        deform_conv: bool = False,
        num_classes: int = 1,
        rotated_bbox: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.cfg = cfg

        conv_layer = DeformConv2d if deform_conv else nn.Conv2d

        self.rotated_bbox = rotated_bbox

        self.feat_extractor = feat_extractor
        # Identify the number of channels for the head initialization
        _is_training = self.feat_extractor.training
        self.feat_extractor = self.feat_extractor.eval()
        with torch.no_grad():
            out = self.feat_extractor(torch.zeros((1, 3, 224, 224)))
            fpn_channels = [v.shape[1] for _, v in out.items()]

        if _is_training:
            self.feat_extractor = self.feat_extractor.train()

        self.fpn = FeaturePyramidNetwork(fpn_channels, head_chans, deform_conv)
        # Conv1 map to channels

        self.prob_head = nn.Sequential(
            conv_layer(head_chans, head_chans // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, head_chans // 4, 2, stride=2, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, 1, 2, stride=2),
        )
        self.thresh_head = nn.Sequential(
            conv_layer(head_chans, head_chans // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, head_chans // 4, 2, stride=2, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, num_classes, 2, stride=2),
        )

        self.postprocessor = DBPostProcessor(rotated_bbox=rotated_bbox)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, DeformConv2d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[np.ndarray]] = None,
        return_model_output: bool = False,
        return_boxes: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Extract feature maps at different stages
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        # Pass through the FPN
        feat_concat = self.fpn(feats)
        logits = self.prob_head(feat_concat)

        out: Dict[str, Any] = {}
        if return_model_output or target is None or return_boxes:
            prob_map = torch.sigmoid(logits)

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_boxes:
            # Post-process boxes
            out["preds"] = self.postprocessor(prob_map.squeeze(1).detach().cpu().numpy())

        if target is not None:
            thresh_map = self.thresh_head(feat_concat)
            loss = self.compute_loss(logits, thresh_map, target)
            out['loss'] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        thresh_map: torch.Tensor,
        target: List[np.ndarray]
    ) -> torch.Tensor:
        """Compute a batch of gts, masks, thresh_gts, thresh_masks from a list of boxes
        and a list of masks for each image. From there it computes the loss with the model output

        Args:
            out_map: output feature map of the model of shape (N, C, H, W)
            thresh_map: threshold map of shape (N, C, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry

        Returns:
            A loss tensor
        """

        prob_map = torch.sigmoid(out_map.squeeze(1))
        thresh_map = torch.sigmoid(thresh_map.squeeze(1))

        targets = self.compute_target(target, prob_map.shape)  # type: ignore[arg-type]

        seg_target, seg_mask = torch.from_numpy(targets[0]), torch.from_numpy(targets[1])
        seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)
        thresh_target, thresh_mask = torch.from_numpy(targets[2]), torch.from_numpy(targets[3])
        thresh_target, thresh_mask = thresh_target.to(out_map.device), thresh_mask.to(out_map.device)

        # Compute balanced BCE loss for proba_map
        bce_scale = 5.
        bce_loss = F.binary_cross_entropy_with_logits(out_map.squeeze(1), seg_target, reduction='none')[seg_mask]

        neg_target = 1 - seg_target[seg_mask]
        positive_count = seg_target[seg_mask].sum()
        negative_count = torch.minimum(neg_target.sum(), 3. * positive_count)
        negative_loss = bce_loss * neg_target
        negative_loss = negative_loss.sort().values[-int(negative_count.item()):]
        sum_losses = torch.sum(bce_loss * seg_target[seg_mask]) + torch.sum(negative_loss)
        balanced_bce_loss = sum_losses / (positive_count + negative_count + 1e-6)

        # Compute dice loss for approxbin_map
        bin_map = 1 / (1 + torch.exp(-50. * (prob_map[seg_mask] - thresh_map[seg_mask])))

        bce_min = bce_loss.min()
        weights = (bce_loss - bce_min) / (bce_loss.max() - bce_min) + 1.
        inter = torch.sum(bin_map * seg_target[seg_mask] * weights)
        union = torch.sum(bin_map) + torch.sum(seg_target[seg_mask]) + 1e-8
        dice_loss = 1 - 2.0 * inter / union

        # Compute l1 loss for thresh_map
        l1_scale = 10.
        if torch.any(thresh_mask):
            l1_loss = torch.mean(torch.abs(thresh_map[thresh_mask] - thresh_target[thresh_mask]))
        else:
            l1_loss = torch.zeros(1)

        return l1_scale * l1_loss + bce_scale * balanced_bce_loss + dice_loss


def _dbnet(arch: str, pretrained: bool, pretrained_backbone: bool = True, **kwargs: Any) -> DBNet:

    # Starting with Imagenet pretrained params introduces some NaNs in layer3 & layer4 of resnet50
    pretrained_backbone = pretrained_backbone and not arch.split('_')[1].startswith('resnet')
    pretrained_backbone = pretrained_backbone and not pretrained

    # Feature extractor
    backbone = default_cfgs[arch]['backbone'](pretrained=pretrained_backbone)
    if isinstance(default_cfgs[arch]['backbone_submodule'], str):
        backbone = getattr(backbone, default_cfgs[arch]['backbone_submodule'])
    feat_extractor = IntermediateLayerGetter(
        backbone,
        {layer_name: str(idx) for idx, layer_name in enumerate(default_cfgs[arch]['fpn_layers'])},
    )

    # Build the model
    model = DBNet(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def db_resnet34(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-34 backbone.

    Example::
        >>> import torch
        >>> from doctr.models import db_resnet34
        >>> model = db_resnet34(pretrained=True)
        >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _dbnet('db_resnet34', pretrained, **kwargs)


def db_resnet50(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-50 backbone.

    Example::
        >>> import torch
        >>> from doctr.models import db_resnet50
        >>> model = db_resnet50(pretrained=True)
        >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _dbnet('db_resnet50', pretrained, **kwargs)


def db_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a MobileNet V3 Large backbone.

    Example::
        >>> import torch
        >>> from doctr.models import db_mobilenet_v3_large
        >>> model = db_mobilenet_v3_large(pretrained=True)
        >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _dbnet('db_mobilenet_v3_large', pretrained, **kwargs)
