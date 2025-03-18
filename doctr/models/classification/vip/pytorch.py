from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# If needed
# from itertools import groupby
from doctr.datasets import VOCABS

from ...utils import load_pretrained_params
from .layers import CrossShapedWindowAttention, MultiHeadSelfAttention, OSRABlock, PatchEmbed, PatchMerging

__all__ = ["vip_tiny", "vip_base", "default_cfgs"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "vip_tiny": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "vip_base": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
}


class VIPBlock(nn.Module):
    """Unified block for Local, Global, and Mixed feature mixing in VIP architecture."""

    def __init__(
        self,
        embed_dim: int,
        local_unit: nn.ModuleList,
        size: Tuple[int, int],
        global_unit: nn.ModuleList = None,
        proj: nn.Module = None,
        downsample: bool = False,
        out_dim: int = None,
    ):
        """
        Args:
            embed_dim: dimension of embeddings
            local_unit: local mixing block(s)
            size: (H, W) size
            global_unit: global mixing block(s)
            proj: projection layer used for mixed mixing
            downsample: whether to downsample at the end
            out_dim: out channels if downsampling
        """
        super().__init__()
        if downsample and out_dim is None:
            raise ValueError("`out_dim` must be specified if `downsample=True`")

        self.local_unit = local_unit
        self.global_unit = global_unit
        self.proj = proj
        self.downsample = PatchMerging(dim=embed_dim, out_dim=out_dim) if downsample else None
        self.size = size

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
                x = self.proj(x) + x
                x = x.permute(0, 2, 3, 1).contiguous()

        if self.downsample:
            x = self.downsample(x)

        return x


class VIPNet(nn.Sequential):
    """
    VIP (Vision Permutable) encoder architecture, adapted for text recognition.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 384,
        embed_dims: List[int] = [64, 128, 256],
        depths: List[int] = [3, 3, 3],
        num_heads: List[int] = [2, 4, 8],
        mlp_ratios: List[int] = [3, 4, 4],
        split_sizes: List[int] = [1, 2, 4],
        sr_ratios: List[int] = [4, 2, 2],
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 1000,
        include_top: bool = False,
        cfg: Dict[str, Any] = None,
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
        self.include_top = include_top
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

        layers = [PatchEmbed(in_channels=in_channels, embed_dim=embed_dims[0])]

        # Construct mixers
        # e.g. local, mixed, global
        mixer_functions = [
            _vip_local_mixer,
            _vip_mixed_mixer,
            _vip_global_mha_mixer,
        ]

        H, W = input_shape[1], input_shape[2]

        dpr_splits = []
        idx = 0
        for d in depths:
            dpr_splits.append(dpr[idx : idx + d])
            idx += d

        for i, mixer_fn in enumerate(mixer_functions):
            embed_dim = embed_dims[i]
            depth_i = depths[i]
            num_head = num_heads[i]
            mlp_ratio = mlp_ratios[i]
            sp_size = split_sizes[i]
            sr_ratio = sr_ratios[i]
            drop_path = dpr_splits[i]
            next_dim = embed_dims[i + 1] if i < len(embed_dims) - 1 else None

            block = mixer_fn(
                embed_dim=embed_dim,
                depth=depth_i,
                num_heads=num_head,
                mlp_ratio=mlp_ratio,
                split_size=sp_size,
                sr_ratio=sr_ratio,
                drop_path=drop_path,
                size=(H, W),
                downsample=(next_dim is not None),
                out_dim=next_dim,
            )
            layers.append(block)
            if next_dim is not None:
                H //= 2  # PatchMerging reduces height by 2
                W //= 2

        # LN -> permute -> GAP -> squeeze -> MLP
        layers.append(nn.LayerNorm(embed_dims[-1], eps=1e-6))
        layers.append(PermuteLayer((0, 2, 3, 1)))
        layers.append(nn.AdaptiveAvgPool2d((embed_dims[-1], 1)))
        layers.append(SqueezeLayer(dim=3))

        mlp_head = nn.Sequential(
            nn.Linear(embed_dims[-1], out_dim, bias=False),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
        )
        layers.append(mlp_head)

        if include_top:
            layers.append(nn.Linear(out_dim, num_classes))

        super().__init__(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def vip_tiny(pretrained: bool = False, **kwargs: Any) -> VIPNet:
    """
    VIP-Tiny encoder architecture.

    Args:
        pretrained: whether to load pretrained weights
        **kwargs: optional arguments

    Returns:
        VIPNet model
    """
    return _vip(
        "vip_tiny",
        pretrained,
        out_dim=384,
        embed_dims=[64, 128, 256],
        depths=[3, 3, 3],
        num_heads=[2, 4, 8],
        mlp_ratios=[3, 4, 4],
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )


def vip_base(pretrained: bool = False, **kwargs: Any) -> VIPNet:
    """
    VIP-Base encoder architecture.

    Args:
        pretrained: whether to load pretrained weights
        **kwargs: optional arguments

    Returns:
        VIPNet model
    """
    return _vip(
        "vip_base",
        pretrained,
        out_dim=384,
        embed_dims=[128, 256, 384],
        depths=[3, 6, 9],
        num_heads=[4, 8, 12],
        mlp_ratios=[4, 4, 4],
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )


def _vip(
    arch: str,
    pretrained: bool,
    ignore_keys: List[str],
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
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs.get("num_classes", len(_cfg["classes"]))
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    _cfg["classes"] = kwargs.get("classes", _cfg["classes"])
    kwargs.pop("classes", None)

    model = VIPNet(cfg=_cfg, **kwargs)
    if pretrained:
        # If #classes differs, ignore final layers
        model_keys = ignore_keys if kwargs.get("num_classes", len(_cfg["classes"])) != len(_cfg["classes"]) else None
        load_pretrained_params(model, _cfg["url"], ignore_keys=model_keys)
    return model


############################################
# _vip_local_mixer
############################################
def _vip_local_mixer(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    drop_path: List[float],
    size: Tuple[int, int],
    split_size: int = 1,
    sr_ratio: int = 1,
    downsample: bool = False,
    out_dim: Optional[int] = None,
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
        size: the (H, W) dimensions of the input feature map at this stage.
        downsample: whether to apply PatchMerging at the end.
        out_dim: output embedding dimension if downsampling.

    Returns:
        A VIPBlock (local attention) for one stage of the VIP network.
    """
    blocks = nn.ModuleList([
        CrossShapedWindowAttention(
            dim=embed_dim,
            num_heads=num_heads,
            patches_resolution=25,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            split_size=split_size,
            drop_path=drop_path[i],
        )
        for i in range(depth)
    ])
    return VIPBlock(embed_dim, local_unit=blocks, size=size, downsample=downsample, out_dim=out_dim)


############################################
# _vip_global_mha_mixer
############################################
def _vip_global_mha_mixer(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    drop_path: List[float],
    size: Tuple[int, int],
    split_size: int = 1,
    sr_ratio: int = 1,
    downsample: bool = False,
    out_dim: Optional[int] = None,
) -> nn.Module:
    """Builds a VIPBlock performing global multi-head self-attention.

    Args:
        embed_dim: embedding dimension.
        depth: number of attention blocks in this stage.
        num_heads: number of attention heads.
        mlp_ratio: ratio used to expand the hidden dimension in MLP.
        drop_path: list of per-block drop path rates.
        split_size: parameter needed for cross-compatibility between different mixers
        sr_ratio:parameter needed for cross-compatibility between different mixers
        size: the (H, W) dimensions of the input feature map at this stage.
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
        size=size,
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
    drop_path: List[float],
    size: Tuple[int, int],
    split_size: int = 1,
    sr_ratio: int = 1,
    downsample: bool = False,
    out_dim: Optional[int] = None,
) -> nn.Module:
    """Builds a VIPBlock performing mixed local+global attention.

    Args:
        embed_dim: embedding dimension.
        depth: number of attention blocks in this stage.
        num_heads: total number of attention heads.
        mlp_ratio: ratio used to expand the hidden dimension in MLP.
        drop_path: list of per-block drop path rates.
        size: the (H, W) dimensions of the input feature map at this stage.
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
            patches_resolution=25,
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
        size=size,
        global_unit=global_unit,
        proj=proj,
        downsample=downsample,
        out_dim=out_dim,
    )


class PermuteLayer(nn.Module):
    """Custom layer to permute dimensions in a Sequential model."""

    def __init__(self, dims=(0, 2, 3, 1)):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims).contiguous()


class SqueezeLayer(nn.Module):
    """Custom layer to squeeze out a dimension in a Sequential model."""

    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)
