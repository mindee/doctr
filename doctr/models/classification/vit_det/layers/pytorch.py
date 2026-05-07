# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import torch
import torch.nn as nn
import torch.nn.functional as F

from doctr.models.modules import DropPath

__all__ = ["PatchEmbed", "MLP", "AttentionWithCAE", "WindowedCAETransformerBlock"]


class PatchEmbed(nn.Module):
    """Simple 2D convolutional patch embedding layer for ViT Det

    Args:
        kernel_size: kernel size of the projection layer.
        stride: stride of the projection layer.
        padding: padding size of the projection layer.
        in_chans: Number of input image channels.
        embed_dim:  embed_dim (int): Patch embedding dimension.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B C H W -> B H W C
        return self.proj(x).permute(0, 2, 3, 1)


class MLP(nn.Module):
    """Simple Multilayer Perceptron (MLP)

    Args:
        in_features: number of input features
        hidden_features: number of hidden features (default: in_features)
        out_features: number of output features (default: in_features)
        act_layer: activation layer (default: nn.GELU)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
    ):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionWithCAE(nn.Module):
    """Multi-head Attention block with CAE bias construction.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to query, key, value.
        use_cae: If True, use CAE bias construction (separate q and v bias).
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, use_cae: bool = False):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_cae = use_cae

        self.qkv = nn.Linear(dim, dim * 3, bias=(qkv_bias and not use_cae))

        # CAE bias
        if use_cae:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape

        # QKV projection
        if self.use_cae:
            zeros = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat([self.q_bias, zeros, self.v_bias])
            qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        # Reshape to multi-head
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            attn = attn.masked_fill(mask.view(B, 1, 1, N).expand_as(attn), float("-inf"))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class WindowedCAETransformerBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        drop_prob (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
        window (bool): If True, use window attention. Otherwise, use global attention.
        use_cae (bool): If True, use CAE bias construction (separate q and v bias).
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_prob=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window=False,
        use_cae=False,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = AttentionWithCAE(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_cae=use_cae,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

        self.window = window
        self.use_cae = use_cae

        if use_cae:
            self.gamma_1 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(0.1 * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, HW, C = x.shape
        shortcut = x

        x = self.norm1(x)
        mask_r = mask

        # Window partitioning logic
        if not self.window:
            x = x.reshape(B // 16, 16 * HW, C)
            shortcut_r = shortcut.reshape(B // 16, 16 * HW, C)

            if mask is not None:
                mask_r = mask.reshape(B // 16, 16 * HW)
            else:
                mask_r = None
        else:
            shortcut_r = shortcut

        # Attention
        attn_out = self.attn(x, mask_r)

        if self.use_cae:
            attn_out = self.gamma_1 * attn_out

        x = shortcut_r + self.drop_path(attn_out)

        # Reshape back if needed
        if not self.window:
            x = x.reshape(B, HW, C)
            if mask is not None:
                mask = mask.reshape(B, HW)

        x = x + self.drop_path((self.gamma_2 * self.mlp(self.norm2(x))) if self.use_cae else self.mlp(self.norm2(x)))

        return x
