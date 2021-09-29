# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import List, Dict, Any, Optional

from .base import LinkNetPostProcessor, _LinkNet
from ...utils import load_pretrained_params

__all__ = ['LinkNet', 'linknet16']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'linknet16': {
        'layout': [64, 128, 256, 512],
        'input_shape': (3, 1024, 1024),
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'url': None,
    },
}


class LinkNetEncoder(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_chans),
        ) if in_chans != out_chans else None

        self.stage2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage1(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        out = self.stage2(out) + out

        return out


def linknet_backbone(layout: List[int], in_channels: int = 3, stem_channels: int = 64) -> nn.Sequential:
    # Stem
    _layers: List[nn.Module] = [
        nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]
    # Encoders
    for in_chan, out_chan in zip([stem_channels] + layout[:-1], layout):
        _layers.append(LinkNetEncoder(in_chan, out_chan))

    return nn.Sequential(*_layers)


class LinkNetFPN(nn.Module):
    def __init__(self, layout: List[int], in_channels: int = 64) -> None:
        super().__init__()
        _decoder_layers = [
            self.decoder_block(out_chan, in_chan) for in_chan, out_chan in zip([in_channels] + layout[:-1], layout)
        ]
        self.decoders = nn.ModuleList(_decoder_layers)

    @staticmethod
    def decoder_block(in_chan: int, out_chan: int) -> nn.Sequential:
        """Creates a LinkNet decoder block"""

        return nn.Sequential(
            nn.Conv2d(in_chan, in_chan // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_chan // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chan // 4, in_chan // 4, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_chan // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan // 4, out_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:

        out = feats[-1]
        for decoder, fmap in zip(self.decoders[::-1], feats[:-1][::-1]):
            out = decoder(out) + fmap

        out = self.decoders[0](out)

        return out


class LinkNet(nn.Module, _LinkNet):

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        num_classes: int = 1,
        rotated_bbox: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.cfg = cfg

        self.feat_extractor = feat_extractor
        # Identify the number of channels for the FPN initialization
        _is_training = self.feat_extractor.training
        self.feat_extractor = self.feat_extractor.eval()
        with torch.no_grad():
            out = self.feat_extractor(torch.zeros((1, 3, 224, 224)))
            fpn_channels = [v.shape[1] for _, v in out.items()]

        if _is_training:
            self.feat_extractor = self.feat_extractor.train()

        self.fpn = LinkNetFPN(fpn_channels, fpn_channels[0])

        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(fpn_channels[0], 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),
        )

        self.postprocessor = LinkNetPostProcessor(rotated_bbox=rotated_bbox)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
        focal_loss: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        feats = self.feat_extractor(x)
        logits = self.fpn([feats[str(idx)] for idx in range(len(feats))])
        logits = self.classifier(logits)

        out: Dict[str, Any] = {}
        if return_model_output or target is None or return_boxes:
            prob_map = torch.sigmoid(logits)
        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_boxes:
            # Post-process boxes
            out["preds"] = self.postprocessor(prob_map.squeeze(1).detach().cpu().numpy())

        if target is not None:
            loss = self.compute_loss(logits, target, focal_loss)
            out['loss'] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        target: List[np.ndarray],
        focal_loss: bool = False,
        alpha: float = .5,
        gamma: float = 2.,
        edge_factor: float = 2.,
    ) -> torch.Tensor:
        """Compute linknet loss, BCE with boosted box edges or focal loss. Focal loss implementation based on
        <https://github.com/tensorflow/addons/>`_.

        Args:
            out_map: output feature map of the model of shape N x H x W x 1
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            focal_loss: if True, use focal loss instead of BCE
            edge_factor: boost factor for box edges (in case of BCE)
            alpha: balancing factor in the focal loss formula
            gamma: modulating factor in the focal loss formula

        Returns:
            A loss tensor
        """
        targets = self.compute_target(target, out_map.shape)  # type: ignore[arg-type]

        seg_target, seg_mask = torch.from_numpy(targets[0]).to(dtype=out_map.dtype), torch.from_numpy(targets[1])
        seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)
        edge_mask = torch.from_numpy(targets[2]).to(out_map.device)

        # Get the cross_entropy for each entry
        bce = F.binary_cross_entropy_with_logits(out_map, seg_target, reduction='none')[seg_mask]

        if focal_loss:
            if gamma and gamma < 0:
                raise ValueError("Value of gamma should be greater than or equal to zero.")

            # Convert logits to prob, compute gamma factor
            pred_prob = torch.sigmoid(out_map)[seg_mask]
            p_t = (seg_target[seg_mask] * pred_prob) + ((1 - seg_target[seg_mask]) * (1 - pred_prob))

            # Compute alpha factor
            alpha_factor = seg_target[seg_mask] * alpha + (1 - seg_target[seg_mask]) * (1 - alpha)

            # compute the final loss
            loss = (alpha_factor * (1. - p_t) ** gamma * bce).mean()

        else:
            # Compute BCE loss with highlighted edges
            loss = ((1 + (edge_factor - 1) * edge_mask) * bce).mean()

        return loss


def _linknet(arch: str, pretrained: bool, pretrained_backbone: bool = False, **kwargs: Any) -> LinkNet:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Build the feature extractor
    backbone = linknet_backbone(default_cfgs[arch]['layout'])
    if pretrained_backbone:
        load_pretrained_params(backbone, None)

    feat_extractor = IntermediateLayerGetter(
        backbone,
        {str(layer): str(idx) for idx, layer in enumerate(range(4, 4 + len(default_cfgs[arch]['layout'])))},
    )

    # Build the model
    model = LinkNet(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def linknet16(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Example::
        >>> import torch
        >>> from doctr.models import linknet16
        >>> model = linknet16(pretrained=True).eval()
        >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
        >>> with torch.no_grad(): out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet('linknet16', pretrained, **kwargs)
