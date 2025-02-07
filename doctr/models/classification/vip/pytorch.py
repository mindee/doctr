import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from doctr.models.modules.layers import DropPath
from doctr.models.modules.transformer import PositionwiseFeedForward
from doctr.models.modules.transformer.pytorch import PositionwiseFeedForward
from doctr.models.utils import conv_sequence_pt


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
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute((2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.permute((0, 1, 3, 2)).contiguous())).contiguous()
        attn = nn.functional.softmax(attn, -1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute((0, 2, 1, 3)).contiguous().reshape((-1, N, C)).contiguous()
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

        x = self.img2windows(x, H_sp, W_sp)
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

    def forward(self, qkv: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
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

    def forward(self, x, size):
        """
        x: B, H*W, C
        """
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:, :, :, : C // 2], size)
        x2 = self.attns[1](qkv[:, :, :, C // 2 :], size)
        attened_x = torch.cat([x1, x2], dim=2)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


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


class VIPTRNet(nn.Module):
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
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

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

    def forward(self, x):
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
        mlp_ratios=[3, 4, 4],  # 4 4 4
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],  # [8, 4, 2]
    )
    return model


def VIPTRv2B():
    model = VIPTRNet(
        out_dim=384,
        embed_dims=[128, 256, 384],
        depths=[3, 6, 9],
        num_heads=[4, 8, 12],
        mlp_ratios=[4, 4, 4],  # 4 4 4
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2],  # [8, 4, 2]
    )
    return model


if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VIPTRv2B().to(device)
    a = torch.randn(1, 3, 32, 128).to(device)
    # b = torch.randn(1, 3, 32, 320).to(device)
    # print(model(b).shape)
    model.eval()
    with torch.no_grad():
        start = time.time()
        for i in range(5):
            a = torch.randn(1, 3, 32, 128).to(device)
            y = model(a)
        infer_time = time.time() - start
    print(infer_time / 5)
    print("Base model Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    # test with HEAD
    model = VIPTRv2B().to(device)
    a = torch.randn(1, 3, 32, 128).to(device)
    print(model(a).shape)
    head = VIPTRRecHead(384, 128).to(device)
    y = model(a)
    print(head(y).shape)

    model = VIPTRv2().to(device)
    a = torch.randn(1, 3, 32, 128).to(device)
    # b = torch.randn(1, 3, 32, 320).to(device)
    # print(model(b).shape)
    model.eval()
    with torch.no_grad():
        start = time.time()
        for i in range(5):
            a = torch.randn(1, 3, 32, 128).to(device)
            y = model(a)
        infer_time = time.time() - start
    print(infer_time / 5)
    print("Tiny model Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    # test with HEAD
    a = torch.randn(1, 3, 32, 128).to(device)
    print(model(a).shape)
    head = VIPTRRecHead(384, 128).to(device)
    y = model(a)
    print(head(y).shape)

    # TODO: Head is only linear - so in general we can implement the model as is as classification arch and then use it for recognition and adjust
