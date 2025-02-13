# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params
from .layers import CrossShapedWindowAttention, MultiHeadSelfAttention, OSRABlock, PatchEmbed, PatchMerging

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


# TODO: Refactor and cleanup !
class BasicLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        split_size=1,
        sr_ratio=1,
        qkv_bias=True,
        drop_path=0.0,
        downsample=False,
        mixer_type="Global",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.depth = depth
        self.mixer_type = mixer_type
        if mixer_type == "Local1":
            self.blocks = nn.ModuleList([
                CrossShapedWindowAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    patches_resolution=25,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    split_size=split_size,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ])

        elif mixer_type == "Global2":
            self.blocks = nn.ModuleList([
                MultiHeadSelfAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=drop_path[i],
                )
                for i in range(depth)
            ])
        elif mixer_type == "LG1":
            inner_dim = max(16, embed_dim // 8)
            self.proj = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
                nn.GELU(),
                nn.BatchNorm2d(embed_dim),
                nn.Conv2d(embed_dim, inner_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(inner_dim),
                nn.Conv2d(inner_dim, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
            )

            self.local_unit = nn.ModuleList([
                CrossShapedWindowAttention(
                    dim=embed_dim // 2,
                    num_heads=num_heads,
                    patches_resolution=25,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    split_size=split_size,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ])
            self.global_unit = nn.ModuleList([
                OSRABlock(
                    dim=embed_dim // 2,
                    sr_ratio=sr_ratio,
                    num_heads=num_heads // 2,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ])

        self.downsample = PatchMerging(dim=embed_dim, out_dim=out_dim) if downsample else None

    def forward(self, x, size):
        b, h, w, d = x.size()
        if self.mixer_type == "Local1" or self.mixer_type == "Global2":
            for blk in self.blocks:
                x = x.flatten(1).reshape(b, -1, d)
                x = blk(x, size)
                x = x.reshape(b, h, w, -1)

        elif self.mixer_type == "LG1":
            for lblk, gblk in zip(self.local_unit, self.global_unit):
                x = x.flatten(1).reshape(b, -1, d)
                x1, x2 = torch.chunk(x, chunks=2, dim=2)
                x1 = lblk(x1, size)
                x2 = gblk(x2, size)
                x = torch.cat([x1, x2], dim=2)
                x = x.transpose(1, 2).contiguous().reshape(b, -1, h, w)
                x = self.proj(x) + x
                x = x.permute(0, 2, 3, 1).contiguous()
                x = x.reshape(b, h, w, -1)

        if self.downsample:
            x = self.downsample(x)
        return x


class VIPNet(nn.Sequential):
    # TODO
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
        d_model: dimension of the transformer layers
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffd_ratio: multiplier for the hidden dimension of the feedforward layer
        patch_size: size of the patches
        input_shape: size of the input image
        dropout: dropout rate
        num_classes: number of output classes
        include_top: whether the classifier head should be instantiated
    """

    # TODO: This should be a Sequential model
    def __init__(
        self,
        in_chans=3,
        out_dim=192,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[3, 3, 3, 3],
        split_sizes=[1, 2, 2, 4],
        sr_ratios=[8, 4, 2, 1],
        mixer_types=["Local1", "LG1", "Global2"],
        input_shape: tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.out_dim = out_dim
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        self.include_top = include_top

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_channels=in_chans,
            embed_dim=embed_dims[0],
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                split_size=split_sizes[i_layer],
                sr_ratio=sr_ratios[i_layer],
                qkv_bias=True,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=True if (i_layer in [0, 1]) else None,
                mixer_type=mixer_types[i_layer],
            )
            self.layers.append(layer)

        self.pooling = nn.AdaptiveAvgPool2d((embed_dims[self.num_layers - 1], 1))
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dims[self.num_layers - 1], out_dim, bias=False),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
        )
        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)
        if self.include_top:
            self.head = nn.Linear(out_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        _, H, W, _ = x.shape
        for layer in self.layers:
            x = layer(x, (H, W))
            H = x.shape[1]
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # nwch
        x = self.pooling(x)
        x = x.squeeze(3)
        x = self.mlp_head(x)
        if self.include_top:
            x = self.head(x.mean(dim=1))

        return x


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
        ignore_keys=["2.head.weight", "2.head.bias"],  # TODO
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
        ignore_keys=["2.head.weight", "2.head.bias"],  # TODO
        **kwargs,
    )
