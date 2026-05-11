# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params
from .layers import PatchEmbed, WindowedCAETransformerBlock

__all__ = ["vit_det_s", "vit_det_m"]


default_cfgs: dict[str, dict[str, Any]] = {
    "vit_det_s": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://github.com/mindee/doctr/releases/download/v1.0.1/vit_det_s-56a33dee.pt",
    },
    "vit_det_m": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://github.com/mindee/doctr/releases/download/v1.0.1/vit_det_m-669daf92.pt",
    },
}


class ClassifierHead(nn.Module):
    """Classifier head for Vision Detection Transformer

    Args:
        in_channels: number of input channels
        num_classes: number of output classes
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W)
        x = x.flatten(2).mean(dim=-1)  # global average pooling (B, C)
        x = self.norm(x)
        return self.fc(x)


class TakeLastFeature(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[-1]


class ViTInput(nn.Module):
    """ViT input module, which includes the patch embedding and positional embedding

    Args:
        patch_embed: the patch embedding module
        pos_embed: the positional embedding module
        has_cls_token: whether the input includes a cls token
    """

    def __init__(self, patch_embed: PatchEmbed, pos_embed: torch.Tensor, has_cls_token: bool = True):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.has_cls_token = has_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        H, W = x.shape[1], x.shape[2]

        pos_embed = self.pos_embed
        if self.has_cls_token:
            pos_embed = pos_embed[:, 1:]

        xy_num = pos_embed.shape[1]
        size = int(math.sqrt(xy_num))

        if size != H or size != W:  # pragma: no cover
            pos_embed = F.interpolate(
                pos_embed.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            pos_embed = pos_embed.reshape(1, H, W, -1)

        return x + pos_embed


class ViTTokenize(nn.Module):
    """ViT tokenize module, which reshapes the input feature map into a sequence of tokens"""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int]]:
        B, H, W, C = x.shape
        h, w = H // 4, W // 4
        x = x.reshape(B, 4, h, 4, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B * 16, h * w, C)
        return x, (B, H, W, h, w, C)


class ViTStage(nn.Module):
    """ "ViT stage module, which includes a list of transformer blocks and outputs the features of the specified blocks

    Args:
        blocks: a list of transformer blocks
        out_features_mask: a list of booleans indicating which blocks' features to output
    """

    def __init__(self, blocks: list[WindowedCAETransformerBlock], out_features_mask: list[bool]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.out_features_mask = out_features_mask

    def forward(self, inputs: tuple[torch.Tensor, tuple[int, int, int, int, int, int]]) -> list[torch.Tensor]:
        x, meta = inputs
        B, H, W, h, w, C = meta

        outputs = []

        for idx, blk in enumerate(self.blocks):
            x = blk(x, mask=None)

            if self.out_features_mask[idx]:
                feat = x.reshape(B, 4, 4, h, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
                outputs.append(feat)

        return outputs


class VisionDetectionTransformer(nn.Sequential):
    """VisionDetectionTransformer architecture as described in
    `"Exploring Plain Vision Transformer Backbones for Object Detection",
    <https://arxiv.org/abs/2203.16527>`_.

    Args:
        d_model: dimension of the transformer layers
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffd_ratio: multiplier for the hidden dimension of the feedforward layer
        patch_size: size of the patches
        input_shape: size of the input image
        dropout: dropout rate
        window_block_indexes: list of block indices that use window attention
        out_feature_indexes: list of block indices whose features are output
        num_classes: number of output classes
        include_top: whether the classifier head should be instantiated
        cfg: additional configuration dictionary
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        patch_size: tuple[int, int] = (4, 4),
        input_shape: tuple[int, int, int] = (3, 32, 32),
        dropout: float = 0.1,
        window_block_indexes: list[int] = [0, 1, 3, 6, 7, 9],
        out_feature_indexes: list[int] = [2, 4, 5, 9],
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:

        in_chans = input_shape[0]

        patch_embed = PatchEmbed(
            kernel_size=patch_size,
            stride=patch_size,
            in_chans=in_chans,
            embed_dim=d_model,
        )

        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (input_shape[1] // patch_size[0]) * (input_shape[2] // patch_size[1])
        pos_embed = nn.Parameter(torch.zeros(1, (num_patches + 1), d_model))
        nn.init.trunc_normal_(pos_embed, std=0.02)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, dropout, num_layers)]

        # blocks
        blocks = [
            WindowedCAETransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_prob=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window=(i in window_block_indexes),
                use_cae=True,
            )
            for i in range(num_layers)
        ]

        # normalize feature indices
        out_feature_indexes = [i if i >= 0 else i + num_layers for i in out_feature_indexes]
        out_mask = [i in out_feature_indexes for i in range(num_layers)]

        _layers = [
            ViTInput(patch_embed, pos_embed, has_cls_token=True),
            ViTTokenize(),
            ViTStage(blocks, out_mask),
        ]
        if include_top:
            _layers.extend([
                TakeLastFeature(),
                ClassifierHead(d_model, num_classes),
            ])

        super().__init__(*_layers)
        self.cfg = cfg
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, ViTInput):
            if isinstance(m.pos_embed, torch.nn.Parameter):
                nn.init.trunc_normal_(m.pos_embed, mean=0.0, std=0.02)
        elif isinstance(m, WindowedCAETransformerBlock):
            if m.use_cae:
                nn.init.constant_(m.gamma_1, 0.1)
                nn.init.constant_(m.gamma_2, 0.1)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)


def _vit_det(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> VisionDetectionTransformer:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = VisionDetectionTransformer(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def vit_det_s(pretrained: bool = False, **kwargs: Any) -> VisionDetectionTransformer:
    """VisionDetectionTransformer-S architecture
    `"Exploring Plain Vision Transformer Backbones for Object Detection",
    <https://arxiv.org/abs/2203.16527>`_.

    NOTE: Modified for LW-DETR

    >>> import torch
    >>> from doctr.models import vit_det_s
    >>> model = vit_det_s(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionDetectionTransformer architecture

    Returns:
        A feature extractor model
    """
    return _vit_det(
        "vit_det_s",
        pretrained,
        d_model=192,
        num_layers=10,
        num_heads=12,
        mlp_ratio=4.0,
        ignore_keys=["4.norm.weight", "4.norm.bias", "4.fc.weight", "4.fc.bias"],
        **kwargs,
    )


def vit_det_m(pretrained: bool = False, **kwargs: Any) -> VisionDetectionTransformer:
    """VisionDetectionTransformer-B architecture as described in
    `"Exploring Plain Vision Transformer Backbones for Object Detection",
    <https://arxiv.org/abs/2203.16527>`_.

    NOTE: Modified for LW-DETR

    >>> import torch
    >>> from doctr.models import vit_det_m
    >>> model = vit_det_m(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionTransformer architecture

    Returns:
        A feature extractor model
    """
    return _vit_det(
        "vit_det_m",
        pretrained,
        d_model=384,
        num_layers=10,
        num_heads=12,
        mlp_ratio=4.0,
        ignore_keys=["4.norm.weight", "4.norm.bias", "4.fc.weight", "4.fc.bias"],
        **kwargs,
    )
