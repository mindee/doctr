# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from doctr.datasets import VOCABS
from doctr.models.modules.layers import AdaptiveAvgPool2d

from ...utils import load_pretrained_params
from .layers import (
    CrossShapedWindowAttention,
    MultiHeadSelfAttention,
    OSRABlock,
    PatchEmbed,
    PatchMerging,
    PermuteLayer,
    SqueezeLayer,
)

__all__ = ["vip_tiny", "vip_base"]

default_cfgs: dict[str, dict[str, Any]] = {
    "vip_tiny": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.11.0/vip_tiny-033ed51c.pt&src=0",
    },
    "vip_base": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.11.0/vip_base-f6ea2ff5.pt&src=0",
    },
}


class ClassifierHead(nn.Module):
    """Classification head which averages the features and applies a linear layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=1))


class VIPBlock(nn.Module):
    """Unified block for Local, Global, and Mixed feature mixing in VIP architecture."""

    def __init__(
        self,
        embed_dim: int,
        local_unit: nn.ModuleList,
        global_unit: nn.ModuleList | None = None,
        proj: nn.Module | None = None,
        downsample: bool = False,
        out_dim: int | None = None,
    ):
        """
        Args:
            embed_dim: dimension of embeddings
            local_unit: local mixing block(s)
            global_unit: global mixing block(s)
            proj: projection layer used for mixed mixing
            downsample: whether to downsample at the end
            out_dim: out channels if downsampling
        """
        super().__init__()
        if downsample and out_dim is None:  # pragma: no cover
            raise ValueError("`out_dim` must be specified if `downsample=True`")

        self.local_unit = local_unit
        self.global_unit = global_unit
        self.proj = proj
        self.downsample = PatchMerging(dim=embed_dim, out_dim=out_dim) if downsample else None  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VIPBlock.

        Args:
            x: input tensor (B, H, W, C)

        Returns:
            Transformed tensor
        """
        b, h, w, C = x.shape

        # Local or Mixed
        if self.global_unit is None:
            # local or global only
            for blk in self.local_unit:
                # Flatten to (B, H*W, C)
                x = x.reshape(b, -1, C)
                x = blk(x, (h, w))
                x = x.reshape(b, h, w, -1)
        else:
            # Mixed
            for lblk, gblk in zip(self.local_unit, self.global_unit):
                x = x.reshape(b, -1, C)
                # chunk into two halves
                x1, x2 = torch.chunk(x, chunks=2, dim=2)
                x1 = lblk(x1, (h, w))
                x2 = gblk(x2, (h, w))
                x = torch.cat([x1, x2], dim=2)
                x = x.transpose(1, 2).contiguous().reshape(b, -1, h, w)
                x = self.proj(x) + x  # type: ignore[misc]
                x = x.permute(0, 2, 3, 1).contiguous()

        if isinstance(self.downsample, nn.Module):
            x = self.downsample(x)

        return x


class VIPNet(nn.Sequential):
    """
    VIP (Vision Permutable) encoder architecture, adapted for text recognition.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        embed_dims: list[int],
        depths: list[int],
        num_heads: list[int],
        mlp_ratios: list[int],
        split_sizes: list[int],
        sr_ratios: list[int],
        input_shape: tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            in_channels: number of input channels
            out_dim: final embedding dimension
            embed_dims: list of embedding dims per stage
            depths: number of blocks per stage
            num_heads: number of heads for attention blocks
            mlp_ratios: ratio for MLP expansion
            split_sizes: local window split sizes
            sr_ratios: used for some global block adjustments
            input_shape: (C, H, W)
            num_classes: number of output classes
            include_top: if True, append a classification head
            cfg: optional config dictionary
        """
        self.cfg = cfg

        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        drop_paths = [dpr[sum(depths[:i]) : sum(depths[: i + 1])] for i in range(len(depths))]
        layers: list[Any] = [PatchEmbed(in_channels=in_channels, embed_dim=embed_dims[0])]

        # Construct mixers
        # e.g. local, mixed, global
        mixer_functions = [
            _vip_local_mixer,
            _vip_mixed_mixer,
            _vip_global_mha_mixer,
        ]

        for i, mixer_fn in enumerate(mixer_functions):
            embed_dim = embed_dims[i]
            depth_i = depths[i]
            num_head = num_heads[i]
            mlp_ratio = mlp_ratios[i]
            sp_size = split_sizes[i]
            sr_ratio = sr_ratios[i]
            drop_path = drop_paths[i]

            next_dim = embed_dims[i + 1] if i < len(embed_dims) - 1 else None

            block = mixer_fn(
                embed_dim=embed_dim,
                depth=depth_i,
                num_heads=num_head,
                mlp_ratio=mlp_ratio,
                split_size=sp_size,
                sr_ratio=sr_ratio,
                drop_path=drop_path,
                downsample=(next_dim is not None),
                out_dim=next_dim,
            )
            layers.append(block)

        # LN -> permute -> GAP -> squeeze -> MLP
        layers.append(
            nn.Sequential(
                nn.LayerNorm(embed_dims[-1], eps=1e-6),
                PermuteLayer((0, 2, 3, 1)),
                AdaptiveAvgPool2d((embed_dims[-1], 1)),
                SqueezeLayer(dim=3),
            )
        )

        mlp_head = nn.Sequential(
            nn.Linear(embed_dims[-1], out_dim, bias=False),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
        )
        layers.append(mlp_head)
        if include_top:
            layers.append(ClassifierHead(out_dim, num_classes))

        super().__init__(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)


def vip_tiny(pretrained: bool = False, **kwargs: Any) -> VIPNet:
    """
    VIP-Tiny encoder architecture.Corresponds to SVIPTRv2-T variant in the paper (VIPTRv2 function
    in the official implementation:
    https://github.com/cxfyxl/VIPTR/blob/main/modules/VIPTRv2.py)

    Args:
        pretrained: whether to load pretrained weights
        **kwargs: optional arguments

    Returns:
        VIPNet model
    """
    return _vip(
        "vip_tiny",
        pretrained,
        in_channels=3,
        out_dim=192,
        embed_dims=[64, 128, 256],
        depths=[3, 3, 3],
        num_heads=[2, 4, 8],
        mlp_ratios=[3, 4, 4],
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],
        ignore_keys=["6.fc.weight", "6.fc.bias"],
        **kwargs,
    )


def vip_base(pretrained: bool = False, **kwargs: Any) -> VIPNet:
    """
    VIP-Base encoder architecture. Corresponds to SVIPTRv2-B variant in the paper (VIPTRv2B function
    in the official implementation:
    https://github.com/cxfyxl/VIPTR/blob/main/modules/VIPTRv2.py)

    Args:
        pretrained: whether to load pretrained weights
        **kwargs: optional arguments

    Returns:
        VIPNet model
    """
    return _vip(
        "vip_base",
        pretrained,
        in_channels=3,
        out_dim=256,
        embed_dims=[128, 256, 384],
        depths=[3, 6, 9],
        num_heads=[4, 8, 12],
        mlp_ratios=[4, 4, 4],
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],
        ignore_keys=["6.fc.weight", "6.fc.bias"],
        **kwargs,
    )


def _vip(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str],
    **kwargs: Any,
) -> VIPNet:
    """
    Internal constructor for the VIPNet models.

    Args:
        arch: architecture key
        pretrained: load pretrained weights?
        ignore_keys: layer keys to ignore
        **kwargs: arguments passed to VIPNet

    Returns:
        VIPNet instance
    """
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    model = VIPNet(cfg=_cfg, **kwargs)
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)
    return model


############################################
# _vip_local_mixer
############################################
def _vip_local_mixer(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    drop_path: list[float],
    split_size: int = 1,
    sr_ratio: int = 1,
    downsample: bool = False,
    out_dim: int | None = None,
) -> nn.Module:
    """Builds a VIPBlock performing local (cross-shaped) window attention.

    Args:
        embed_dim: embedding dimension.
        depth: number of attention blocks in this stage.
        num_heads: number of attention heads.
        mlp_ratio: ratio used to expand the hidden dimension in MLP.
        split_size: size of the local window splits.
        sr_ratio: parameter needed for cross-compatibility between different mixers
        drop_path: list of per-block drop path rates.
        downsample: whether to apply PatchMerging at the end.
        out_dim: output embedding dimension if downsampling.

    Returns:
        A VIPBlock (local attention) for one stage of the VIP network.
    """
    blocks = nn.ModuleList([
        CrossShapedWindowAttention(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            split_size=split_size,
            drop_path=drop_path[i],
        )
        for i in range(depth)
    ])
    return VIPBlock(embed_dim, local_unit=blocks, downsample=downsample, out_dim=out_dim)


############################################
# _vip_global_mha_mixer
############################################
def _vip_global_mha_mixer(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    drop_path: list[float],
    split_size: int = 1,
    sr_ratio: int = 1,
    downsample: bool = False,
    out_dim: int | None = None,
) -> nn.Module:
    """Builds a VIPBlock performing global multi-head self-attention.

    Args:
        embed_dim: embedding dimension.
        depth: number of attention blocks in this stage.
        num_heads: number of attention heads.
        mlp_ratio: ratio used to expand the hidden dimension in MLP.
        drop_path: list of per-block drop path rates.
        split_size: parameter needed for cross-compatibility between different mixers
        sr_ratio: parameter needed for cross-compatibility between different mixers
        downsample: whether to apply PatchMerging at the end.
        out_dim: output embedding dimension if downsampling.

    Returns:
        A VIPBlock (global MHA) for one stage of the VIP network.
    """
    blocks = nn.ModuleList([
        MultiHeadSelfAttention(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_path_rate=drop_path[i],
        )
        for i in range(depth)
    ])
    return VIPBlock(
        embed_dim,
        local_unit=blocks,  # In this context, they are "global" blocks but stored in local_unit
        downsample=downsample,
        out_dim=out_dim,
    )


############################################
# _vip_mixed_mixer
############################################
def _vip_mixed_mixer(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    drop_path: list[float],
    split_size: int = 1,
    sr_ratio: int = 1,
    downsample: bool = False,
    out_dim: int | None = None,
) -> nn.Module:
    """Builds a VIPBlock performing mixed local+global attention.

    Args:
        embed_dim: embedding dimension.
        depth: number of attention blocks in this stage.
        num_heads: total number of attention heads.
        mlp_ratio: ratio used to expand the hidden dimension in MLP.
        drop_path: list of per-block drop path rates.
        split_size: size of the local window splits (for the local half).
        sr_ratio: reduce spatial resolution in the global half (OSRA).
        downsample: whether to apply PatchMerging at the end.
        out_dim: output embedding dimension if downsampling.

    Returns:
        A VIPBlock (mixed local+global) for one stage of the VIP network.
    """
    # an inner dimension for the conv-projection
    inner_dim = max(16, embed_dim // 8)
    proj = nn.Sequential(
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
        nn.GELU(),
        nn.BatchNorm2d(embed_dim),
        nn.Conv2d(embed_dim, inner_dim, kernel_size=1),
        nn.GELU(),
        nn.BatchNorm2d(inner_dim),
        nn.Conv2d(inner_dim, embed_dim, kernel_size=1),
        nn.BatchNorm2d(embed_dim),
    )

    # local half blocks
    local_unit = nn.ModuleList([
        CrossShapedWindowAttention(
            dim=embed_dim // 2,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            split_size=split_size,
            drop_path=drop_path[i],
        )
        for i in range(depth)
    ])

    # global half blocks
    global_unit = nn.ModuleList([
        OSRABlock(
            dim=embed_dim // 2,
            sr_ratio=sr_ratio,
            num_heads=num_heads // 2,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path[i],
        )
        for i in range(depth)
    ])

    return VIPBlock(
        embed_dim,
        local_unit=local_unit,
        global_unit=global_unit,
        proj=proj,
        downsample=downsample,
        out_dim=out_dim,
    )
