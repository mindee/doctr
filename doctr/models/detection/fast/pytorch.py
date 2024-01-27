# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.file_utils import CLASS_NAME

from ...classification import textnet_base, textnet_small, textnet_tiny
from ...modules.layers import FASTConvLayer
from ...utils import _bf16_to_float32, load_pretrained_params
from .base import _FAST, FASTPostProcessor

__all__ = ["FAST", "fast_tiny", "fast_small", "fast_base"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
    "fast_small": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
    "fast_base": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
}

class FastNeck(nn.Module):
    """Neck of the FAST architecture, composed of a series of 3x3 convolutions and upsampling layers.

    Args:
    ----
        in_channels: number of input channels
        out_channels: number of output channels
        upsample_size: size of the upsampling layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        upsample_size: int = 256,
    ) -> None:
        super().__init__()
        self.reduction = nn.ModuleList([
            FASTConvLayer(in_channels * scale, out_channels, kernel_size=3) for scale in [1, 2, 4, 8]
        ])
        self.upsample = nn.Upsample(size=upsample_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = x
        f1, f2, f3, f4 = [reduction(f) for reduction, f in zip(self.reduction, (f1, f2, f3, f4))]
        f2, f3, f4 = [self.upsample(f) for f in (f2, f3, f4)]
        f = torch.cat((f1, f2, f3, f4), 1)
        return f


class FastHead(nn.Sequential):
    """Head of the FAST architecture

    Args:
    ----
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        _layers: List[nn.Module] = [
                FASTConvLayer(in_channels, out_channels, kernel_size=3),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False)
            ]
        super().__init__(*_layers)



class FAST(_FAST, nn.Module):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.

    Args:
    ----
        feat extractor: the backbone serving as feature extractor
        bin_thresh: threshold for binarization
        dropout_prob: dropout probability
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        bin_thresh: float = 0.3,
        dropout_prob: float = 0.1,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
        class_names: List[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg

        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor
        # Identify the number of channels for the neck & head initialization
        _is_training = self.feat_extractor.training
        self.feat_extractor = self.feat_extractor.eval()
        with torch.no_grad():
            out = self.feat_extractor(torch.zeros((1, 3, 32, 32)))
            feat_out_channels = [v.shape[1] for _, v in out.items()]

        if _is_training:
            self.feat_extractor = self.feat_extractor.train()

        # Initialize the neck
        self.neck = FastNeck(feat_out_channels[0], feat_out_channels[1], feat_out_channels[0] * 4)
        self.prob_head = FastHead(feat_out_channels[-1], num_classes, feat_out_channels[1], dropout_prob)

        self.postprocessor = FASTPostProcessor(assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode="bilinear")

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[np.ndarray]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Extract feature maps at different stages
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        # Pass through the Neck
        feat_concat = self.neck(feats)
        logits = self.prob_head(feat_concat)

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(torch.sigmoid(self._upsample(logits, x.shape)))  # TODO: correct?

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes (keep only text predictions)
            out["preds"] = [
                dict(zip(self.class_names, preds))
                for preds in self.postprocessor(prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy())
            ]

        if target is not None:
            loss = self.compute_loss(logits, target)
            out["loss"] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        target: List[np.ndarray],
    ) -> torch.Tensor:

        # TODO: Tversky Loss for multi-class segmentation (alpha=0.5, beta=0.5 -> same as Dice Loss)
        return torch.tensor(0.0, device=out_map.device, dtype=out_map.dtype)


def _fast(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    feat_layers: List[str],
    pretrained_backbone: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> FAST:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Build the feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone),
        {layer_name: str(idx) for idx, layer_name in enumerate(feat_layers)},
    )

    if not kwargs.get("class_names", None):
        kwargs["class_names"] = default_cfgs[arch].get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])
    # Build the model
    model = FAST(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of class_names is not the same as the number of classes in the pretrained model =>
        # remove the layer weights
        _ignore_keys = (
            ignore_keys if kwargs["class_names"] != default_cfgs[arch].get("class_names", [CLASS_NAME]) else None
        )
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def fast_tiny(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a tiny TextNet backbone.

    >>> import torch
    >>> from doctr.models import fast_tiny
    >>> model = fast_tiny(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_tiny",
        pretrained,
        textnet_tiny,
        ["3", "4", "5", "6"],
        ignore_keys=[],  # TODO: ignore_keys
        **kwargs,
    )


def fast_small(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a small TextNet backbone.

    >>> import torch
    >>> from doctr.models import fast_small
    >>> model = fast_small(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_small",
        pretrained,
        textnet_small,
        ["3", "4", "5", "6"],
        ignore_keys=[],  # TODO: ignore_keys
        **kwargs,
    )


def fast_base(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a base TextNet backbone.

    >>> import torch
    >>> from doctr.models import fast_base
    >>> model = fast_base(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_base",
        pretrained,
        textnet_base,
        ["3", "4", "5", "6"],
        ignore_keys=[],  # TODO: ignore_keys
        **kwargs,
    )
