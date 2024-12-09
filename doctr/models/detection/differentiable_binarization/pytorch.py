# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.deform_conv import DeformConv2d

from doctr.file_utils import CLASS_NAME

from ...classification import mobilenet_v3_large
from ...utils import _bf16_to_float32, load_pretrained_params
from .base import DBPostProcessor, _DBNet

__all__ = ["DBNet", "db_resnet50", "db_resnet34", "db_mobilenet_v3_large"]


default_cfgs: dict[str, dict[str, Any]] = {
    "db_resnet50": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet50-79bd7d70.pt&src=0",
    },
    "db_resnet34": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet34-cb6aed9e.pt&src=0",
    },
    "db_mobilenet_v3_large": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/db_mobilenet_v3_large-21748dd0.pt&src=0",
    },
}


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
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
            )
            for idx, chans in enumerate(in_channels)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.out_branches = nn.ModuleList([
            nn.Sequential(
                conv_layer(out_channels, out_chans, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2**idx, mode="bilinear", align_corners=True),
            )
            for idx, chans in enumerate(in_channels)
        ])

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        if len(x) != len(self.out_branches):
            raise AssertionError
        # Conv1x1 to get the same number of channels
        _x: list[torch.Tensor] = [branch(t) for branch, t in zip(self.in_branches, x)]
        out: list[torch.Tensor] = [_x[-1]]
        for t in _x[:-1][::-1]:
            out.append(self.upsample(out[-1]) + t)

        # Conv and final upsampling
        out = [branch(t) for branch, t in zip(self.out_branches, out[::-1])]

        return torch.cat(out, dim=1)


class DBNet(_DBNet, nn.Module):
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        head_chans: the number of channels in the head
        deform_conv: whether to use deformable convolution
        bin_thresh: threshold for binarization
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        head_chans: int = 256,
        deform_conv: bool = False,
        bin_thresh: float = 0.3,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
        class_names: list[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg

        conv_layer = DeformConv2d if deform_conv else nn.Conv2d

        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

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
            nn.ConvTranspose2d(head_chans // 4, num_classes, 2, stride=2),
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

        self.postprocessor = DBPostProcessor(
            assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh, box_thresh=box_thresh
        )

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, (nn.Conv2d, DeformConv2d)):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        target: list[np.ndarray] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Extract feature maps at different stages
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        # Pass through the FPN
        feat_concat = self.fpn(feats)
        logits = self.prob_head(feat_concat)

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(torch.sigmoid(logits))

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable  # type: ignore[attr-defined]
            def _postprocess(prob_map: torch.Tensor) -> list[dict[str, Any]]:
                return [
                    dict(zip(self.class_names, preds))
                    for preds in self.postprocessor(prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy())
                ]

            # Post-process boxes (keep only text predictions)
            out["preds"] = _postprocess(prob_map)

        if target is not None:
            thresh_map = self.thresh_head(feat_concat)
            loss = self.compute_loss(logits, thresh_map, target)
            out["loss"] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        thresh_map: torch.Tensor,
        target: list[np.ndarray],
        gamma: float = 2.0,
        alpha: float = 0.5,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute a batch of gts, masks, thresh_gts, thresh_masks from a list of boxes
        and a list of masks for each image. From there it computes the loss with the model output

        Args:
            out_map: output feature map of the model of shape (N, C, H, W)
            thresh_map: threshold map of shape (N, C, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            gamma: modulating factor in the focal loss formula
            alpha: balancing factor in the focal loss formula
            eps: epsilon factor in dice loss

        Returns:
            A loss tensor
        """
        if gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")

        prob_map = torch.sigmoid(out_map)
        thresh_map = torch.sigmoid(thresh_map)

        targets = self.build_target(target, out_map.shape[1:], False)  # type: ignore[arg-type]

        seg_target, seg_mask = torch.from_numpy(targets[0]), torch.from_numpy(targets[1])
        seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)
        thresh_target, thresh_mask = torch.from_numpy(targets[2]), torch.from_numpy(targets[3])
        thresh_target, thresh_mask = thresh_target.to(out_map.device), thresh_mask.to(out_map.device)

        if torch.any(seg_mask):
            # Focal loss
            focal_scale = 10.0
            bce_loss = F.binary_cross_entropy_with_logits(out_map, seg_target, reduction="none")

            p_t = prob_map * seg_target + (1 - prob_map) * (1 - seg_target)
            alpha_t = alpha * seg_target + (1 - alpha) * (1 - seg_target)
            # Unreduced version
            focal_loss = alpha_t * (1 - p_t) ** gamma * bce_loss
            # Class reduced
            focal_loss = (seg_mask * focal_loss).sum((0, 1, 2, 3)) / seg_mask.sum((0, 1, 2, 3))

            # Compute dice loss for each class or for approx binary_map
            if len(self.class_names) > 1:
                dice_map = torch.softmax(out_map, dim=1)
            else:
                # compute binary map instead
                dice_map = 1 / (1 + torch.exp(-50.0 * (prob_map - thresh_map)))  # type: ignore[assignment]
            # Class reduced
            inter = (seg_mask * dice_map * seg_target).sum((0, 2, 3))
            cardinality = (seg_mask * (dice_map + seg_target)).sum((0, 2, 3))
            dice_loss = (1 - 2 * inter / (cardinality + eps)).mean()

        # Compute l1 loss for thresh_map
        if torch.any(thresh_mask):
            l1_loss = (torch.abs(thresh_map - thresh_target) * thresh_mask).sum() / (thresh_mask.sum() + eps)

        return l1_loss + focal_scale * focal_loss + dice_loss


def _dbnet(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    fpn_layers: list[str],
    backbone_submodule: str | None = None,
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> DBNet:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Feature extractor
    backbone = (
        backbone_fn(pretrained_backbone)
        if not arch.split("_")[1].startswith("resnet")
        # Starting with Imagenet pretrained params introduces some NaNs in layer3 & layer4 of resnet50
        else backbone_fn(weights=None)  # type: ignore[call-arg]
    )
    if isinstance(backbone_submodule, str):
        backbone = getattr(backbone, backbone_submodule)
    feat_extractor = IntermediateLayerGetter(
        backbone,
        {layer_name: str(idx) for idx, layer_name in enumerate(fpn_layers)},
    )

    if not kwargs.get("class_names", None):
        kwargs["class_names"] = default_cfgs[arch].get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])
    # Build the model
    model = DBNet(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of class_names is not the same as the number of classes in the pretrained model =>
        # remove the layer weights
        _ignore_keys = (
            ignore_keys if kwargs["class_names"] != default_cfgs[arch].get("class_names", [CLASS_NAME]) else None
        )
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def db_resnet34(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-34 backbone.

    >>> import torch
    >>> from doctr.models import db_resnet34
    >>> model = db_resnet34(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _dbnet(
        "db_resnet34",
        pretrained,
        resnet34,
        ["layer1", "layer2", "layer3", "layer4"],
        None,
        ignore_keys=[
            "prob_head.6.weight",
            "prob_head.6.bias",
            "thresh_head.6.weight",
            "thresh_head.6.bias",
        ],
        **kwargs,
    )


def db_resnet50(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-50 backbone.

    >>> import torch
    >>> from doctr.models import db_resnet50
    >>> model = db_resnet50(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _dbnet(
        "db_resnet50",
        pretrained,
        resnet50,
        ["layer1", "layer2", "layer3", "layer4"],
        None,
        ignore_keys=[
            "prob_head.6.weight",
            "prob_head.6.bias",
            "thresh_head.6.weight",
            "thresh_head.6.bias",
        ],
        **kwargs,
    )


def db_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a MobileNet V3 Large backbone.

    >>> import torch
    >>> from doctr.models import db_mobilenet_v3_large
    >>> model = db_mobilenet_v3_large(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _dbnet(
        "db_mobilenet_v3_large",
        pretrained,
        mobilenet_v3_large,
        ["3", "6", "12", "16"],
        "features",
        ignore_keys=[
            "prob_head.6.weight",
            "prob_head.6.bias",
            "thresh_head.6.weight",
            "thresh_head.6.bias",
        ],
        **kwargs,
    )
