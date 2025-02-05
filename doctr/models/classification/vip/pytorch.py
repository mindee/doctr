import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from doctr.models.modules.layers import DropPath
from doctr.models.utils import conv_sequence_pt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class ConvBNLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias_attr=False, groups=1, act=nn.GELU
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class RelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        """
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        """
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_2d_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        """
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        """
        if activate_recurrent:
            retention_rel_pos = self.decay.exp()

        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = (mask_h, mask_w)

        else:
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
            retention_rel_pos = mask

        return retention_rel_pos


class MaSAd(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        """
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        """
        bsz, h, w, _ = x.size()

        mask_h, mask_w = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        """
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        """

        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)  # (b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # (b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # (b h n w w)
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # (b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # (b w n h h)
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output


class MaSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        """
        x: (b h w c)
        rel_pos: mask: (n l l)
        """
        bsz, h, w, _ = x.size()
        mask = rel_pos

        assert h * w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        qr = qr.flatten(2, 3)  # (b n l d1)
        kr = kr.flatten(2, 3)  # (b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d2)
        vr = vr.flatten(2, 3)  # (b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2)  # (b n l l)
        qk_mat = qk_mat + mask  # (b n l l)
        qk_mat = torch.softmax(qk_mat, -1)  # (b n l l)
        output = torch.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # self.out_dim = out_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_chans=None, act_layer=nn.GELU, dropout=0.0):
        super().__init__()

        out_chans = out_chans or in_dim
        hidden_dim = hidden_dim or in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_chans),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        _, N, C = x.shape
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute((2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.permute((0, 1, 3, 2)).contiguous())).contiguous()
        attn = nn.functional.softmax(attn, -1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute((0, 2, 1, 3)).contiguous().reshape((-1, N, C)).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MHSA_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
        prenorm=False,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)

        self.mixer = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        # self.drop_path = DropPath(local_rank,drop_path) if drop_path > 0. else Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = FeedForward(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, dropout=drop)
        self.prenorm = prenorm

    def forward(self, x, size=None):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OSRA_Attention(nn.Module):  # OSRA
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


class OSRA_Block(nn.Module):
    def __init__(
        self,
        dim=64,
        sr_ratio=1,
        num_heads=1,
        mlp_ratio=4,
        norm_cfg=nn.LayerNorm,  # dict(type='GN', num_groups=1),
        act_cfg=nn.GELU,  # dict(type='GELU'),
        drop=0,
        drop_path=0,
        layer_scale_init_value=1e-5,
        grad_checkpoint=False,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        # self.pos_embed = DWConv2d(dim, 3, 1, 1)
        # self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.norm1 = norm_cfg(dim)
        self.token_mixer = OSRA_Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = norm_cfg(dim)

        self.mlp = FeedForward(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_cfg, dropout=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, relative_pos_enc=None):
        # print(x.shape)
        # x = x + self.pos_embed(x)
        x = x + self.drop_path(self.token_mixer(self.norm1(x), relative_pos_enc))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RetBlock(nn.Module):
    def __init__(
        self,
        retention: str,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        out_dim,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        # self.out_dim = out_dim if out_dim is not None else embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ["chunk", "whole"]
        if retention == "chunk":
            self.retention = MaSAd(embed_dim, num_heads)
        else:
            self.retention = MaSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(self, x: torch.Tensor, incremental_state=None, chunkwise_recurrent=False, retention_rel_pos=None):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1
                * self.retention(
                    self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state)
            )
            # print(x.shape)
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, (2, 1), 1)
        # self.norm = nn.BatchNorm2d(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B H W C
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b oh ow oc)
        x = self.norm(x)

        return x


class LePEAttention(nn.Module):
    def __init__(
        self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0.0, proj_drop=0.0, qk_scale=None
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
        self.scale = qk_scale or head_dim**-0.5

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, size):
        B, N, C = x.shape
        H, W = size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size

        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func, size):
        B, N, C = x.shape
        H, W = size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size

        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  # B', C, H', W'

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, size):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Img2Window
        H, W = size
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size

        q = self.im2cswin(q, size)
        k = self.im2cswin(k, size)
        v, lepe = self.get_lepe(v, self.get_v, size)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C
        # Window2Img
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C
        return x


class CSWinBlock(nn.Module):
    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=7,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        last_stage=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        last_stage = False
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            LePEAttention(
                dim // 2,
                resolution=self.patches_resolution,
                idx=i,
                split_size=split_size,
                num_heads=num_heads // 2,
                dim_out=dim // 2,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            for i in range(self.branch_num)
        ])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = FeedForward(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, dropout=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        """
        x: B, H*W, C
        """
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, : C // 2], size)
            x2 = self.attns[1](qkv[:, :, :, C // 2 :], size)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv, size)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class BasicLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value: float,
        heads_range: float,
        mlp_ratio=4.0,
        split_size=1,
        sr_ratio=1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        chunkwise_recurrent=False,
        downsample: PatchMerging = None,
        use_checkpoint=False,
        mixer_type="Global",
        layerscale=False,
        layer_init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        self.mixer_type = mixer_type
        if mixer_type == "Local1":
            self.blocks = nn.ModuleList([
                CSWinBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    reso=25,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    split_size=split_size,
                    drop=drop_rate,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ])

        elif mixer_type == "Local2":
            if chunkwise_recurrent:
                flag = "chunk"
            else:
                flag = "whole"
            self.Relpos = RelPos2d(embed_dim, num_heads, init_value, heads_range)

            # build blocks
            self.blocks = nn.ModuleList([
                RetBlock(
                    flag,
                    embed_dim,
                    num_heads,
                    int(mlp_ratio * embed_dim),
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale,
                    layer_init_values,
                )
                for i in range(depth)
            ])

        elif mixer_type == "Global1":
            self.blocks = nn.ModuleList([
                OSRA_Block(
                    dim=embed_dim,
                    sr_ratio=sr_ratio,
                    num_heads=num_heads // 2,
                    mlp_ratio=mlp_ratio,
                    norm_cfg=norm_layer,
                    drop=drop_rate,
                    drop_path=drop_path[i],
                    act_cfg=nn.GELU,
                )
                for i in range(depth)
            ])
        elif mixer_type == "Global2":
            self.blocks = nn.ModuleList([
                MHSA_Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop,
                    drop_path_rate=drop_path[i],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
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
                CSWinBlock(
                    dim=embed_dim // 2,
                    num_heads=num_heads,
                    reso=25,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    split_size=split_size,
                    drop=drop_rate,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ])
            self.global_unit = nn.ModuleList([
                OSRA_Block(
                    dim=embed_dim // 2,
                    sr_ratio=sr_ratio,
                    num_heads=num_heads // 2,
                    mlp_ratio=mlp_ratio,
                    norm_cfg=norm_layer,
                    drop=drop_rate,
                    drop_path=drop_path[i],
                    act_cfg=nn.GELU,
                )
                for i in range(depth)
            ])
            # self.global_unit = nn.ModuleList([
            #     MHSA_Block(
            #         dim=embed_dim//2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            #         drop=drop_rate, attn_drop=attn_drop, drop_path_rate=drop_path[i], act_layer=nn.GELU,
            #         norm_layer=norm_layer,
            #     ) for i in range(depth)]
            # )

            # if chunkwise_recurrent:
            #     flag = 'chunk'
            # else:
            #     flag = 'whole'
            # self.Relpos = RelPos2d(embed_dim//2, num_heads, init_value, heads_range)
            #
            # self.global_unit = nn.ModuleList([
            #     RetBlock(flag, embed_dim//2, num_heads, int(mlp_ratio*embed_dim),
            #             drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            #     for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, size):
        b, h, w, d = x.size()
        # print(x.size())
        if self.mixer_type == "Local1":
            for blk in self.blocks:
                x = x.flatten(1).reshape(b, -1, d)
                x = blk(x, size)
                x = x.reshape(b, h, w, -1)

        elif self.mixer_type == "Local2":
            rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
            for blk in self.blocks:
                x = blk(
                    x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos
                )

        elif self.mixer_type == "Global1":
            for blk in self.blocks:
                x = x.flatten(1).reshape(b, -1, d)
                # x = x.permute(0, 3, 1, 2).contiguous()
                x = blk(x, size)
                x = x.reshape(b, h, w, -1)

        elif self.mixer_type == "Global2":
            for blk in self.blocks:
                x = x.flatten(1).reshape(b, -1, d)
                x = blk(x, size)
                x = x.reshape(b, h, w, -1)

        elif self.mixer_type == "LG1":
            # print(self.mixer_type)
            # print(x.shape)
            # rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
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

                # x = x.flatten(1).reshape(b, -1, d)
                # x1, x2 = torch.chunk(x, chunks=2, dim=2)
                # x1 = lblk(x1, size)
                # x2 = gblk(x2, size)
                # x = torch.cat([x1, x2], dim=2)
                # x = x.transpose(1, 2).contiguous().reshape(b, -1, h, w)
                # x = self.proj(x) + x
                # x = x.permute(0, 2, 3, 1).contiguous()
                # x = x.reshape(b, h, w, -1)

                # x1, x2 = torch.chunk(x, chunks=2, dim=3)
                # x1 = x1.flatten(1).reshape(b, -1, d//2)
                # x1 = lblk(x1, size)
                # x2 = gblk(x2, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                #           retention_rel_pos=rel_pos)
                # x2 = x2.flatten(1).reshape(b, -1, d//2)
                # # print(x2.shape)
                # x = torch.cat([x1, x2], dim=2)
                # x = x.transpose(1, 2).contiguous().reshape(b, -1, h, w)
                # x = self.proj(x) + x
                # x = x.permute(0, 2, 3, 1).contiguous()

        # print(x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            ConvBNLayer(
                in_channels=in_chans,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias_attr=False,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias_attr=False,
            ),
        )

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1).contiguous()
        return x


class VIPTRNet(nn.Module):
    def __init__(
        self,
        in_chans=3,
        out_dim=192,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        init_values=[1, 1, 1, 1],
        heads_ranges=[3, 3, 3, 3],
        mlp_ratios=[3, 3, 3, 3],
        split_sizes=[1, 2, 2, 4],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoints=[False, False, False, False],
        mixer_types=["Local1", "LG1", "Global2"],
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                split_size=split_sizes[i_layer],
                sr_ratio=sr_ratios[i_layer],
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop=0.0,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                # norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging
                if (i_layer in [0, 1])
                else None,  # PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                mixer_type=mixer_types[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        self.pooling = nn.AdaptiveAvgPool2d((embed_dims[self.num_layers - 1], 1))
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dims[self.num_layers - 1], out_dim, bias=False), nn.Hardswish(), nn.Dropout(p=0.1)
        )
        self.norm = nn.LayerNorm(embed_dims[-1], eps=layer_init_values)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward_features(self, x):
        x = self.patch_embed(x)
        _, H, W, _ = x.shape
        for layer in self.layers:
            x = layer(x, (H, W))
            H = x.shape[1]
            # print(x.shape)  # nhwc
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # nwch
        x = self.pooling(x)
        x = x.squeeze(3)  # .reshape(b, W, -1)
        x = self.mlp_head(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class VIPTRRecHead(nn.Module):
    def __init__(self, out_dim: 384, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


def VIPTRv2():
    model = VIPTRNet(
        out_dim=384,
        embed_dims=[64, 128, 256],
        depths=[3, 3, 3],
        num_heads=[2, 4, 8],
        init_values=[2, 2, 2],
        heads_ranges=[4, 4, 6],
        mlp_ratios=[3, 4, 4],  # 4 4 4
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],  # [8, 4, 2]
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, False, False],
        layerscales=[False, False, False],
    )
    return model


def VIPTRv2B():
    model = VIPTRNet(
        out_dim=384,
        embed_dims=[128, 256, 384],
        depths=[3, 6, 9],
        num_heads=[4, 8, 12],
        init_values=[2, 2, 2],
        heads_ranges=[6, 6, 6],
        mlp_ratios=[4, 4, 4],  # 4 4 4
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],  # [8, 4, 2]
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, False, False],
        layerscales=[False, False, False],
    )
    return model


if __name__ == "__main__":
    model = VIPTRv2B().to(device)
    a = torch.randn(1, 3, 32, 128).to(device)
    print(model(a).shape)
    # b = torch.randn(1, 3, 32, 320).to(device)
    # print(model(b).shape)
    model.eval()
    with torch.no_grad():
        start = time.time()
        for i in range(5):
            a = torch.randn(1, 3, 32, 128).to(device)
            y = model(a)
            print(y.shape)
        infer_time = time.time() - start
    print(infer_time / 5)
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    # test with HEAD
    model = VIPTRv2B().to(device)
    a = torch.randn(1, 3, 32, 128).to(device)
    print(model(a).shape)
    head = VIPTRRecHead(384, 128).to(device)
    y = model(a)
    print(head(y).shape)

    # TODO: Head is only linear - so in general we can implement the model as is as classification arch and then use it for recognition and adjust
