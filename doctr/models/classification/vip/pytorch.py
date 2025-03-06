import torch
import torch.nn as nn
from copy import deepcopy
from typing import Any

from doctr.datasets import VOCABS
from ...utils import load_pretrained_params
from .layers import PatchEmbed, PatchMerging, CrossShapedWindowAttention, MultiHeadSelfAttention, OSRABlock

__all__ = ["vip_tiny", "vip_base"]

default_cfgs: dict[str, dict[str, Any]] = {
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
    """Unified block for Local, Global, and Mixed feature mixing"""

    def __init__(self, embed_dim, local_unit, size, global_unit=None, proj=None, downsample=False, out_dim=None):
        super().__init__()
        if downsample:
            assert out_dim is not None, "out_dim cannot be None if downsample=True"

        self.local_unit = local_unit  # Always set for both local/global mixing and mixed mixing
        self.global_unit = global_unit  # Only set for mixed mixing
        self.proj = proj  # Projection layer (only for mixed mixing)
        self.downsample = PatchMerging(dim=embed_dim, out_dim=out_dim) if downsample else None
        self.size = size  # Store the computed (H, W) size

    def forward(self, x):
        """Now uses stored `self.size` instead of needing `size` as an argument"""
        b, _, _, C = x.shape  # We no longer need to extract (H, W)
        h, w = self.size  # Retrieve stored size

        if self.global_unit is None:
            # Local or Global mixing case (only local_unit is set)
            for blk in self.local_unit:
                x = x.flatten(1).reshape(b, -1, C)  # Ensure correct shape
                x = blk(x, (h, w))
                x = x.reshape(b, h, w, -1)  # Restore spatial dimensions

        else:
            # Mixed mixing case (both local_unit and global_unit are set)
            for lblk, gblk in zip(self.local_unit, self.global_unit):
                x = x.flatten(1).reshape(b, -1, C)  # Ensure correct shape
                x1, x2 = torch.chunk(x, chunks=2, dim=2)  # Each (B, H * W, C/2)
                x1 = lblk(x1, (h, w))  # (B, H * W, C'/2)
                x2 = gblk(x2, (h, w))  # (B, H * W, C'/2)
                x = torch.cat([x1, x2], dim=2)  # (B, H * W, C')

                x = x.transpose(1, 2).contiguous().reshape(b, -1, h, w)
                x = self.proj(x) + x
                x = x.permute(0, 2, 3, 1).contiguous()
                x = x.reshape(b, h, w, -1)

        if self.downsample:
            x = self.downsample(x)

        return x


def _vip_local_mixer(
    embed_dim, depth, num_heads, mlp_ratio, split_size, drop_path, size, downsample=False, out_dim=None
):
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
    return VIPBlock(embed_dim, blocks, size, downsample=downsample, out_dim=out_dim)


def _vip_global_mha_mixer(embed_dim, depth, num_heads, mlp_ratio, drop_path, size, downsample=False, out_dim=None):
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
    return VIPBlock(embed_dim, blocks, size, downsample=downsample, out_dim=out_dim)


def _vip_mixed_mixer(
    embed_dim,
    depth,
    num_heads,
    mlp_ratio,
    drop_path,
    size,
    split_size: int = 1,
    sr_ratio: int = 1,
    downsample=False,
    out_dim=None,
):
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
        embed_dim, local_unit, size, global_unit=global_unit, proj=proj, downsample=downsample, out_dim=out_dim
    )


class PermuteLayer(nn.Module):
    """A custom layer to permute dimensions in a Sequential model."""

    def __init__(self, dims=(0, 2, 3, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims).contiguous()


class SqueezeLayer(nn.Module):
    """A custom layer to squeeze the last dimension in a Sequential model."""

    def __init__(self, dim=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class VIPNet(nn.Sequential):
    """VIP (Vision Permutable) encoder architecture as described in
    `"SVIPTR: Fast and Efficient Scene Text Recognition with Vision Permutable Extractor",
    <https://arxiv.org/pdf/2401.10110>`_.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 384,
        embed_dims: list[int] = [64, 128, 256],
        depths: list[int] = [3, 3, 3],
        num_heads: list[int] = [2, 4, 8],
        mlp_ratios: list[int] = [3, 4, 4],
        split_sizes: list[int] = [1, 2, 4],
        sr_ratios: list[int] = [4, 2, 2],
        input_shape: tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.include_top = include_top

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

        # Patch embedding (initial feature extraction)
        layers = [PatchEmbed(in_channels=in_channels, embed_dim=embed_dims[0])]

        # Define the sequence of mixers
        mixer_functions = [_vip_local_mixer, _vip_mixed_mixer, _vip_global_mha_mixer]

        # Compute size values before passing to VIPBlock
        H, W = input_shape[1], input_shape[2]  # Initial input image size

        for mixer_fn, embed_dim, depth, num_head, mlp_ratio, split_size, sr_ratio, drop_path, next_embed_dim in zip(
            mixer_functions,
            embed_dims,
            depths,
            num_heads,
            mlp_ratios,
            split_sizes,
            sr_ratios,
            [dpr[sum(depths[:i]) : sum(depths[: i + 1])] for i in range(len(depths))],
            embed_dims[1:] + [None],  # Ensure last stage has no out_dim
        ):
            # Create mixer block and pass the precomputed size
            block = mixer_fn(
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_head,
                mlp_ratio=mlp_ratio,
                split_size=split_size,
                sr_ratio=sr_ratio,
                drop_path=drop_path,
                downsample=(next_embed_dim is not None),
                out_dim=next_embed_dim,
                size=(H, W),  # Pass computed size
            )
            layers.append(block)

            # If downsample=True, update the H, W size
            if next_embed_dim is not None:
                H = H // 2  # Corrected: PatchMerging reduces H by half

        # Add normalization before permutation
        layers.append(nn.LayerNorm(embed_dims[-1], eps=1e-6))
        layers.append(PermuteLayer())  # Equivalent to x.permute(0, 2, 3, 1)
        layers.append(nn.AdaptiveAvgPool2d((embed_dims[-1], 1)))
        layers.append(SqueezeLayer())  # Equivalent to x.squeeze(3)

        # MLP Head for classification
        mlp_head = nn.Sequential(
            nn.Linear(embed_dims[-1], out_dim, bias=False),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
        )
        layers.append(mlp_head)

        if include_top:
            layers.append(nn.Linear(out_dim, num_classes))

        # Initialize sequential model
        super().__init__(*layers)

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _vip(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> VIPNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = VIPNet(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def vip_tiny(pretrained: bool = False, **kwargs: Any) -> VIPNet:
    """VIP-Tiny encoder architecture as described in
    `"SVIPTR: Fast and Efficient Scene Text Recognition with Vision Permutable Extractor",
    <https://arxiv.org/pdf/2401.10110>`_.
    >>> import torch
    >>> from doctr.models import vip_tiny
    >>> model = vip_tiny(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=tf.float32)
    >>> out = model(input_tensor)
    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionTransformer architecture
    Returns:
        A feature extractor model
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
    """VIP-Base encoder architecture as described in
    `"SVIPTR: Fast and Efficient Scene Text Recognition with Vision Permutable Extractor",
    <https://arxiv.org/pdf/2401.10110>`_.
    >>> import torch
    >>> from doctr.models import vip_base
    >>> model = vib_base(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=tf.float32)
    >>> out = model(input_tensor)
    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionTransformer architecture
    Returns:
        A feature extractor model
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
