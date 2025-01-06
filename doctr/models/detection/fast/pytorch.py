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

from ...classification import textnet_base, textnet_small, textnet_tiny
from ...modules.layers import FASTConvLayer
from ...utils import _bf16_to_float32, load_pretrained_params
from .base import _FAST, FASTPostProcessor

__all__ = ["FAST", "fast_tiny", "fast_small", "fast_base", "reparameterize"]


default_cfgs: dict[str, dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/fast_tiny-1acac421.pt&src=0",
    },
    "fast_small": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/fast_small-10952cc1.pt&src=0",
    },
    "fast_base": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/fast_base-688a8b34.pt&src=0",
    },
}


class FastNeck(nn.Module):
    """Neck of the FAST architecture, composed of a series of 3x3 convolutions and upsampling layers.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.reduction = nn.ModuleList([
            FASTConvLayer(in_channels * scale, out_channels, kernel_size=3) for scale in [1, 2, 4, 8]
        ])

    def _upsample(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = x
        f1, f2, f3, f4 = [reduction(f) for reduction, f in zip(self.reduction, (f1, f2, f3, f4))]
        f2, f3, f4 = [self._upsample(f, f1) for f in (f2, f3, f4)]
        f = torch.cat((f1, f2, f3, f4), 1)
        return f


class FastHead(nn.Sequential):
    """Head of the FAST architecture

    Args:
        in_channels: number of input channels
        num_classes: number of output classes
        out_channels: number of output channels
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        _layers: list[nn.Module] = [
            FASTConvLayer(in_channels, out_channels, kernel_size=3),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False),
        ]
        super().__init__(*_layers)


class FAST(_FAST, nn.Module):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.

    Args:
        feat extractor: the backbone serving as feature extractor
        bin_thresh: threshold for binarization
        box_thresh: minimal objectness score to consider a box
        dropout_prob: dropout probability
        pooling_size: size of the pooling layer
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
        dropout_prob: float = 0.1,
        pooling_size: int = 4,  # different from paper performs better on close text-rich images
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: dict[str, Any] = {},
        class_names: list[str] = [CLASS_NAME],
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

        # Initialize neck & head
        self.neck = FastNeck(feat_out_channels[0], feat_out_channels[1])
        self.prob_head = FastHead(feat_out_channels[-1], num_classes, feat_out_channels[1], dropout_prob)

        # NOTE: The post processing from the paper works not well for text-rich images
        # so we use a modified version from DBNet
        self.postprocessor = FASTPostProcessor(
            assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh, box_thresh=box_thresh
        )

        # Pooling layer as erosion reversal as described in the paper
        self.pooling = nn.MaxPool2d(kernel_size=pooling_size // 2 + 1, stride=1, padding=(pooling_size // 2) // 2)

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
        # Pass through the Neck & Head & Upsample
        feat_concat = self.neck(feats)
        logits = F.interpolate(self.prob_head(feat_concat), size=x.shape[-2:], mode="bilinear")

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(torch.sigmoid(self.pooling(logits)))

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
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute fast loss, 2 x Dice loss where the text kernel loss is scaled by 0.5.

        Args:
            out_map: output feature map of the model of shape (N, num_classes, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            eps: epsilon factor in dice loss

        Returns:
            A loss tensor
        """
        targets = self.build_target(target, out_map.shape[1:], False)  # type: ignore[arg-type]

        seg_target, seg_mask = torch.from_numpy(targets[0]), torch.from_numpy(targets[1])
        shrunken_kernel = torch.from_numpy(targets[2]).to(out_map.device)
        seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)

        def ohem_sample(score: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            masks = []
            for class_idx in range(gt.shape[0]):
                pos_num = int(torch.sum(gt[class_idx] > 0.5)) - int(
                    torch.sum((gt[class_idx] > 0.5) & (mask[class_idx] <= 0.5))
                )
                neg_num = int(torch.sum(gt[class_idx] <= 0.5))
                neg_num = int(min(pos_num * 3, neg_num))

                if neg_num == 0 or pos_num == 0:
                    masks.append(mask[class_idx])
                    continue

                neg_score_sorted, _ = torch.sort(-score[class_idx][gt[class_idx] <= 0.5])
                threshold = -neg_score_sorted[neg_num - 1]

                selected_mask = ((score[class_idx] >= threshold) | (gt[class_idx] > 0.5)) & (mask[class_idx] > 0.5)
                masks.append(selected_mask)
            # combine all masks to shape (len(masks), H, W)
            return torch.stack(masks).unsqueeze(0).float()

        if len(self.class_names) > 1:
            kernels = torch.softmax(out_map, dim=1)
            prob_map = torch.softmax(self.pooling(out_map), dim=1)
        else:
            kernels = torch.sigmoid(out_map)
            prob_map = torch.sigmoid(self.pooling(out_map))

        # As described in the paper, we use the Dice loss for the text segmentation map and the Dice loss scaled by 0.5.
        selected_masks = torch.cat(
            [ohem_sample(score, gt, mask) for score, gt, mask in zip(prob_map, seg_target, seg_mask)], 0
        ).float()
        inter = (selected_masks * prob_map * seg_target).sum((0, 2, 3))
        cardinality = (selected_masks * (prob_map + seg_target)).sum((0, 2, 3))
        text_loss = (1 - 2 * inter / (cardinality + eps)).mean() * 0.5

        # As described in the paper, we use the Dice loss for the text kernel map.
        selected_masks = seg_target * seg_mask
        inter = (selected_masks * kernels * shrunken_kernel).sum((0, 2, 3))  # noqa
        cardinality = (selected_masks * (kernels + shrunken_kernel)).sum((0, 2, 3))  # noqa
        kernel_loss = (1 - 2 * inter / (cardinality + eps)).mean()

        return text_loss + kernel_loss


def reparameterize(model: FAST | nn.Module) -> FAST:
    """Fuse batchnorm and conv layers and reparameterize the model

    Args:
        model: the FAST model to reparameterize

    Returns:
        the reparameterized model
    """
    last_conv = None
    last_conv_name = None

    for module in model.modules():
        if hasattr(module, "reparameterize_layer"):
            module.reparameterize_layer()

    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # fuse batchnorm only if it is followed by a conv layer
            if last_conv is None:
                continue
            conv_w = last_conv.weight
            conv_b = last_conv.bias if last_conv.bias is not None else torch.zeros_like(child.running_mean)

            factor = child.weight / torch.sqrt(child.running_var + child.eps)
            last_conv.weight = nn.Parameter(conv_w * factor.reshape([last_conv.out_channels, 1, 1, 1]))
            last_conv.bias = nn.Parameter((conv_b - child.running_mean) * factor + child.bias)
            model._modules[last_conv_name] = last_conv
            model._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            reparameterize(child)

    return model  # type: ignore[return-value]


def _fast(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    feat_layers: list[str],
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
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
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_tiny",
        pretrained,
        textnet_tiny,
        ["3", "4", "5", "6"],
        ignore_keys=["prob_head.2.weight"],
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
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_small",
        pretrained,
        textnet_small,
        ["3", "4", "5", "6"],
        ignore_keys=["prob_head.2.weight"],
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
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_base",
        pretrained,
        textnet_base,
        ["3", "4", "5", "6"],
        ignore_keys=["prob_head.2.weight"],
        **kwargs,
    )
