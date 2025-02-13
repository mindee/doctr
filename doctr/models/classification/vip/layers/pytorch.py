# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import torch
import torch.nn as nn
import torch.nn.functional as F

from doctr.models.modules.layers import DropPath
from doctr.models.modules.transformer import PositionwiseFeedForward
from doctr.models.utils import conv_sequence_pt

__all__ = [
    "PatchEmbed",
    "Attention",
    "MultiHeadSelfAttention",
    "OverlappingShiftedRelativeAttention",
    "OSRABlock",
    "PatchMerging",
    "LePEAttention",
    "CrossShapedWindowAttention",
]


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()

        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            *conv_sequence_pt(
                in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False, bn=True, relu=False
            ),
            nn.GELU(),
            *conv_sequence_pt(
                embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False, bn=True, relu=False
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        _, N, C = x.shape
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q.matmul(k.permute((0, 1, 3, 2)))
        attn = nn.functional.softmax(attn, -1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute((0, 2, 1, 3)).contiguous().reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = PositionwiseFeedForward(d_model=dim, ffd=mlp_hidden_dim, dropout=0.0, activation_fct=nn.GELU())

    def forward(self, x, size=None):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlappingShiftedRelativeAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=1,
        qk_scale=None,
        attn_drop=0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                *conv_sequence_pt(
                    dim,
                    dim,
                    kernel_size=sr_ratio + 3,
                    stride=sr_ratio,
                    padding=(sr_ratio + 3) // 2,
                    groups=dim,
                    bias=False,
                    bn=True,
                    relu=False,
                ),
                # GELU now
                nn.GELU(),
                *conv_sequence_pt(dim, dim, kernel_size=1, groups=dim, bias=False, bn=True, relu=False),
            )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, size, relative_pos_enc=None):
        # B, C, H, W = x.shape
        B, N, C = x.shape
        H, W = size
        x = x.permute(0, 2, 1).contiguous().reshape(B, -1, H, W)
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(
                    relative_pos_enc, size=attn.shape[2:], mode="bicubic", align_corners=False
                )
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2).contiguous().reshape(B, C, -1)

        x = x.permute(0, 2, 1).contiguous()
        return x  # .reshape(B, C, H, W)


class OSRABlock(nn.Module):
    def __init__(
        self,
        dim=64,
        sr_ratio=1,
        num_heads=1,
        mlp_ratio=4,
        drop_path=0,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = OverlappingShiftedRelativeAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = PositionwiseFeedForward(d_model=dim, ffd=mlp_hidden_dim, dropout=0.0, activation_fct=nn.GELU())
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, relative_pos_enc=None):
        x = x + self.drop_path(self.token_mixer(self.norm1(x), relative_pos_enc))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer"""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, (2, 1), 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H // 2, W, C)
        return self.norm(self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))


class LePEAttention(nn.Module):
    def __init__(
        self,
        dim,
        resolution,
        idx,
        split_size=7,
        dim_out=None,
        num_heads=8,
        attn_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        head_dim = dim // num_heads
        # NOTE scale factor can set manually to be compat with prev weights
        self.scale = head_dim**-0.5

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def img2windows(self, img: torch.Tensor, H_sp: int, W_sp: int) -> torch.Tensor:
        B, C, H, W = img.shape
        img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
        return img_perm

    def windows2img(self, img_splits_hw, H_sp, W_sp, H, W):
        """
        img_splits_hw: B' H W C
        """
        B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

        img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
        img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return img

    def _get_split(self, size: tuple[int, int]) -> tuple[int, int]:
        H, W = size
        if self.idx == -1:
            return H, W
        elif self.idx == 0:
            return H, self.split_size
        elif self.idx == 1:
            return self.split_size, W

    def im2cswin(self, x, size):
        B, N, C = x.shape
        H, W = size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        H_sp, W_sp = self._get_split(size)

        x = self.img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, size):
        B, N, C = x.shape
        H, W = size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        H_sp, W_sp = self._get_split(size)

        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  # B', C, H', W'

        lepe = self.get_v(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2)
        return x, lepe

    def forward(self, qkv: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Img2Window
        H, W = size
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        H_sp, W_sp = self._get_split(size)

        q = self.im2cswin(q, size)
        k = self.im2cswin(k, size)
        v, lepe = self.get_lepe(v, size)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C
        # Window2Img
        x = self.windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C
        return x


class CrossShapedWindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        patches_resolution,
        num_heads,
        split_size=7,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.0,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

        self.attns = nn.ModuleList([
            LePEAttention(
                dim // 2,
                resolution=patches_resolution,
                idx=i,
                split_size=split_size,
                num_heads=num_heads // 2,
                dim_out=dim // 2,
            )
            for i in range(2)
        ])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = PositionwiseFeedForward(d_model=dim, ffd=mlp_hidden_dim, dropout=0.0, activation_fct=nn.GELU())
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        B, _, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:, :, :, : C // 2], size)
        x2 = self.attns[1](qkv[:, :, :, C // 2 :], size)
        merged = self.proj(torch.cat([x1, x2], dim=2))
        x = x + self.drop_path(merged)
        return x + self.drop_path(self.mlp(self.norm2(x)))
