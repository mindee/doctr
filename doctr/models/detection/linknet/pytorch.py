# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.file_utils import CLASS_NAME
from doctr.models.classification import resnet18, resnet34, resnet50

from ...utils import _bf16_to_float32, load_pretrained_params
from .base import LinkNetPostProcessor, _LinkNet

__all__ = ["LinkNet", "linknet_resnet18", "linknet_resnet34", "linknet_resnet50"]


default_cfgs: dict[str, dict[str, Any]] = {
    "linknet_resnet18": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/linknet_resnet18-e47a14dc.pt&src=0",
    },
    "linknet_resnet34": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/linknet_resnet34-9ca2df3e.pt&src=0",
    },
    "linknet_resnet50": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/linknet_resnet50-6cf565c1.pt&src=0",
    },
}


class LinkNetFPN(nn.Module):
    def __init__(self, layer_shapes: list[tuple[int, int, int]]) -> None:
        super().__init__()
        strides = [
            1 if (in_shape[-1] == out_shape[-1]) else 2
            for in_shape, out_shape in zip(layer_shapes[:-1], layer_shapes[1:])
        ]

        chans = [shape[0] for shape in layer_shapes]

        _decoder_layers = [
            self.decoder_block(ochan, ichan, stride) for ichan, ochan, stride in zip(chans[:-1], chans[1:], strides)
        ]

        self.decoders = nn.ModuleList(_decoder_layers)

    @staticmethod
    def decoder_block(in_chan: int, out_chan: int, stride: int) -> nn.Sequential:
        """Creates a LinkNet decoder block"""
        mid_chan = in_chan // 4
        return nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_chan, mid_chan, 3, padding=1, output_padding=stride - 1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        out = feats[-1]
        for decoder, fmap in zip(self.decoders[::-1], feats[:-1][::-1]):
            out = decoder(out) + fmap

        out = self.decoders[0](out)

        return out


class LinkNet(nn.Module, _LinkNet):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        bin_thresh: threshold for binarization of the output feature map
        box_thresh: minimal objectness score to consider a box
        head_chans: number of channels in the head layers
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        bin_thresh: float = 0.1,
        box_thresh: float = 0.1,
        head_chans: int = 32,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
        class_names: list[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor
        # Identify the number of channels for the FPN initialization
        self.feat_extractor.eval()
        with torch.no_grad():
            in_shape = (3, 512, 512)
            out = self.feat_extractor(torch.zeros((1, *in_shape)))
            # Get the shapes of the extracted feature maps
            _shapes = [v.shape[1:] for _, v in out.items()]
            # Prepend the expected shapes of the first encoder
            _shapes = [(_shapes[0][0], in_shape[1] // 4, in_shape[2] // 4)] + _shapes
        self.feat_extractor.train()

        self.fpn = LinkNetFPN(_shapes)

        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(
                _shapes[0][0], head_chans, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm2d(head_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_chans, head_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_chans),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans, num_classes, kernel_size=2, stride=2),
        )

        self.postprocessor = LinkNetPostProcessor(
            assume_straight_pages=self.assume_straight_pages, bin_thresh=bin_thresh, box_thresh=box_thresh
        )

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        feats = self.feat_extractor(x)
        logits = self.fpn([feats[str(idx)] for idx in range(len(feats))])
        logits = self.classifier(logits)

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
            loss = self.compute_loss(logits, target)
            out["loss"] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        target: list[np.ndarray],
        gamma: float = 2.0,
        alpha: float = 0.5,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute linknet loss, BCE with boosted box edges or focal loss. Focal loss implementation based on
        <https://github.com/tensorflow/addons/>`_.

        Args:
            out_map: output feature map of the model of shape (N, num_classes, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            gamma: modulating factor in the focal loss formula
            alpha: balancing factor in the focal loss formula
            eps: epsilon factor in dice loss

        Returns:
            A loss tensor
        """
        _target, _mask = self.build_target(target, out_map.shape[1:], False)  # type: ignore[arg-type]

        seg_target, seg_mask = torch.from_numpy(_target).to(dtype=out_map.dtype), torch.from_numpy(_mask)
        seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)
        seg_mask = seg_mask.to(dtype=torch.float32)

        bce_loss = F.binary_cross_entropy_with_logits(out_map, seg_target, reduction="none")
        proba_map = torch.sigmoid(out_map)

        # Focal loss
        if gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")
        p_t = proba_map * seg_target + (1 - proba_map) * (1 - seg_target)
        alpha_t = alpha * seg_target + (1 - alpha) * (1 - seg_target)
        # Unreduced version
        focal_loss = alpha_t * (1 - p_t) ** gamma * bce_loss
        # Class reduced
        focal_loss = (seg_mask * focal_loss).sum((0, 1, 2, 3)) / seg_mask.sum((0, 1, 2, 3))

        # Compute dice loss for each class
        dice_map = torch.softmax(out_map, dim=1) if len(self.class_names) > 1 else proba_map
        # Class reduced
        inter = (seg_mask * dice_map * seg_target).sum((0, 2, 3))
        cardinality = (seg_mask * (dice_map + seg_target)).sum((0, 2, 3))
        dice_loss = (1 - 2 * inter / (cardinality + eps)).mean()

        # Return the full loss (equal sum of focal loss and dice loss)
        return focal_loss + dice_loss


def _linknet(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    fpn_layers: list[str],
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> LinkNet:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Build the feature extractor
    backbone = backbone_fn(pretrained_backbone)
    feat_extractor = IntermediateLayerGetter(
        backbone,
        {layer_name: str(idx) for idx, layer_name in enumerate(fpn_layers)},
    )
    if not kwargs.get("class_names", None):
        kwargs["class_names"] = default_cfgs[arch].get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])

    # Build the model
    model = LinkNet(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of class_names is not the same as the number of classes in the pretrained model =>
        # remove the layer weights
        _ignore_keys = (
            ignore_keys if kwargs["class_names"] != default_cfgs[arch].get("class_names", [CLASS_NAME]) else None
        )
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def linknet_resnet18(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import torch
    >>> from doctr.models import linknet_resnet18
    >>> model = linknet_resnet18(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _linknet(
        "linknet_resnet18",
        pretrained,
        resnet18,
        ["layer1", "layer2", "layer3", "layer4"],
        ignore_keys=[
            "classifier.6.weight",
            "classifier.6.bias",
        ],
        **kwargs,
    )


def linknet_resnet34(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import torch
    >>> from doctr.models import linknet_resnet34
    >>> model = linknet_resnet34(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _linknet(
        "linknet_resnet34",
        pretrained,
        resnet34,
        ["layer1", "layer2", "layer3", "layer4"],
        ignore_keys=[
            "classifier.6.weight",
            "classifier.6.bias",
        ],
        **kwargs,
    )


def linknet_resnet50(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import torch
    >>> from doctr.models import linknet_resnet50
    >>> model = linknet_resnet50(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _linknet(
        "linknet_resnet50",
        pretrained,
        resnet50,
        ["layer1", "layer2", "layer3", "layer4"],
        ignore_keys=[
            "classifier.6.weight",
            "classifier.6.bias",
        ],
        **kwargs,
    )
