# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.models.classification import resnet18, resnet34, resnet50

from ...utils import load_pretrained_params
from .base import LinkNetPostProcessor, _LinkNet

__all__ = ["LinkNet", "linknet_resnet18", "linknet_resnet34", "linknet_resnet50"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "linknet_resnet18": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.5, 0.5, 0.5),
        "std": (1.0, 1.0, 1.0),
        "url": None,
    },
    "linknet_resnet34": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.5, 0.5, 0.5),
        "std": (1.0, 1.0, 1.0),
        "url": None,
    },
    "linknet_resnet50": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.5, 0.5, 0.5),
        "std": (1.0, 1.0, 1.0),
        "url": None,
    },
}


class LinkNetFPN(nn.Module):
    def __init__(self, layer_shapes: List[Tuple[int, int, int]]) -> None:
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

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:

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
        num_classes: number of output channels in the segmentation map
        head_chans: number of channels in the head layers
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        num_classes: int = 1,
        head_chans: int = 32,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
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

        self.postprocessor = LinkNetPostProcessor(assume_straight_pages=self.assume_straight_pages)

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
        target: Optional[List[np.ndarray]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        feats = self.feat_extractor(x)
        logits = self.fpn([feats[str(idx)] for idx in range(len(feats))])
        logits = self.classifier(logits)

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = torch.sigmoid(logits)
        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = [
                preds[0] for preds in self.postprocessor(prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy())
            ]

        if target is not None:
            loss = self.compute_loss(logits, target)
            out["loss"] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        target: List[np.ndarray],
        gamma: float = 2.0,
        alpha: float = 0.5,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute linknet loss, BCE with boosted box edges or focal loss. Focal loss implementation based on
        <https://github.com/tensorflow/addons/>`_.

        Args:
            out_map: output feature map of the model of shape (N, 1, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            gamma: modulating factor in the focal loss formula
            alpha: balancing factor in the focal loss formula

        Returns:
            A loss tensor
        """
        _target, _mask = self.build_target(target, out_map.shape[-2:])  # type: ignore[arg-type]

        seg_target, seg_mask = torch.from_numpy(_target).to(dtype=out_map.dtype), torch.from_numpy(_mask)
        seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)
        seg_mask = seg_mask.to(dtype=torch.float32)

        bce_loss = bce_loss = F.binary_cross_entropy_with_logits(out_map, seg_target, reduction="none")
        proba_map = torch.sigmoid(out_map)

        # Focal loss
        if gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")
        p_t = proba_map * seg_target + (1 - proba_map) * (1 - seg_target)
        alpha_t = alpha * seg_target + (1 - alpha) * (1 - seg_target)
        # Unreduced version
        focal_loss = alpha_t * (1 - p_t) ** gamma * bce_loss
        # Class reduced
        focal_loss = (seg_mask * focal_loss).sum((0, 2, 3)) / seg_mask.sum((0, 2, 3))

        # Dice loss
        inter = (seg_mask * proba_map * seg_target).sum((0, 2, 3))
        cardinality = (seg_mask * (proba_map + seg_target)).sum((0, 2, 3))
        dice_loss = 1 - 2 * (inter + eps) / (cardinality + eps)

        # Return the full loss (equal sum of focal loss and dice loss)
        return focal_loss.mean() + dice_loss.mean()


def _linknet(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    fpn_layers: List[str],
    pretrained_backbone: bool = True,
    **kwargs: Any,
) -> LinkNet:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Build the feature extractor
    backbone = backbone_fn(pretrained_backbone)
    feat_extractor = IntermediateLayerGetter(
        backbone,
        {layer_name: str(idx) for idx, layer_name in enumerate(fpn_layers)},
    )

    # Build the model
    model = LinkNet(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

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

    Returns:
        text detection architecture
    """

    return _linknet("linknet_resnet18", pretrained, resnet18, ["layer1", "layer2", "layer3", "layer4"], **kwargs)


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

    Returns:
        text detection architecture
    """

    return _linknet("linknet_resnet34", pretrained, resnet34, ["layer1", "layer2", "layer3", "layer4"], **kwargs)


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

    Returns:
        text detection architecture
    """

    return _linknet("linknet_resnet50", pretrained, resnet50, ["layer1", "layer2", "layer3", "layer4"], **kwargs)
