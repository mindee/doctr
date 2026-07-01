# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FASTConvLayer", "DropPath", "AdaptiveAvgPool2d", "ChannelLayerNorm", "DCNv2"]


class DropPath(nn.Module):
    """
    DropPath (Drop Connect) layer. This is a stochastic version of the identity layer.
    """

    # Borrowed from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with different dimensions
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class AdaptiveAvgPool2d(nn.Module):
    """
    Custom AdaptiveAvgPool2d implementation which is ONNX and `torch.compile` compatible.
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        H_out, W_out = self.output_size
        N, C, H, W = x.shape

        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
        for oh in range(H_out):
            start_h = (oh * H) // H_out
            end_h = ((oh + 1) * H + H_out - 1) // H_out  # ceil((oh+1)*H / H_out)
            for ow in range(W_out):
                start_w = (ow * W) // W_out
                end_w = ((ow + 1) * W + W_out - 1) // W_out  # ceil((ow+1)*W / W_out)
                # average over the window
                out[:, :, oh, ow] = x[:, :, start_h:end_h, start_w:end_w].mean(dim=(-2, -1))
        return out


class ChannelLayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def _deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Modulated deformable convolution (DCNv2).

    Numerically equivalent to `torchvision.ops.deform_conv2d` (same offset/mask channel layout and bilinear
    convention) but built only from `grid_sample` + `conv2d`, so the model is ONNX-exportable (the
    `torchvision::deform_conv2d` operator has no ONNX symbolic). `offset` is laid out as torchvision
    expects: for kernel position `k = kh * Kw + kw`, `offset[:, 2 * k]` is the vertical offset and
    `offset[:, 2 * k + 1]` the horizontal one; `mask[:, k]` is the modulation.

    Args:
        x: input feature map, shape (N, C, H, W)
        offset: sampling offsets, shape (N, 2 * Kh * Kw, Ho, Wo)
        weight: convolution weight, shape (Cout, C, Kh, Kw)
        bias: convolution bias, shape (Cout,)
        stride: convolution stride (sh, sw)
        padding: convolution padding (ph, pw)
        dilation: convolution dilation (dh, dw)
        mask: modulation mask, shape (N, Kh * Kw, Ho, Wo)

    Returns:
        the output feature map, shape (N, Cout, Ho, Wo)
    """
    _, _, h, w = x.shape
    cout, _, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    ho, wo = offset.shape[-2], offset.shape[-1]

    # Base sampling location of each output position in the input plane (before kernel/dilation and offset)
    base_y = (torch.arange(ho, device=x.device, dtype=x.dtype) * sh - ph).view(1, ho, 1)
    base_x = (torch.arange(wo, device=x.device, dtype=x.dtype) * sw - pw).view(1, 1, wo)
    norm_y, norm_x = max(h - 1, 1), max(w - 1, 1)

    out = x.new_zeros((x.shape[0], cout, ho, wo))
    for kh_i in range(kh):
        for kw_i in range(kw):
            k = kh_i * kw + kw_i
            sample_y = base_y + kh_i * dh + offset[:, 2 * k, :, :]
            sample_x = base_x + kw_i * dw + offset[:, 2 * k + 1, :, :]
            # Normalize to [-1, 1] for grid_sample with align_corners=True (zero padding out of bounds)
            grid = torch.stack((2.0 * sample_x / norm_x - 1.0, 2.0 * sample_y / norm_y - 1.0), dim=-1)
            sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled = sampled * mask[:, k : k + 1, :, :]
            out = out + F.conv2d(sampled, weight[:, :, kh_i, kw_i].unsqueeze(-1).unsqueeze(-1))
    return out + bias.view(1, -1, 1, 1)


class DCNv2(nn.Module):
    """Modulated deformable convolution (v2).

    Args:
        in_channels: number of channels in the input feature map
        out_channels: number of channels produced by the convolution
        kernel_size: size of the convolving kernel
        stride: stride of the convolution
        padding: zero-padding added to both sides of the input
        dilation: spacing between kernel elements
        deformable_groups: number of deformable group partitions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int,
        padding: int,
        dilation: int = 1,
        deformable_groups: int = 1,
    ):
        super().__init__()
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        channels_ = deformable_groups * 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(in_channels, channels_, kernel_size, stride, padding, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard DCN initialization: the regular conv weight is initialized like a vanilla conv, while
        # the offset/mask predictor is zero-initialized so the layer starts as a plain convolution
        # (offsets = 0, modulation = 0.5). Without this, weight/bias keep their uninitialized
        # torch.empty values, which makes the deformable conv explode and the loss diverge to NaN.
        n = self.weight.shape[1]
        for k in self.weight.shape[2:]:
            n *= k
        stdv = 1.0 / (n**0.5)
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return _deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)


class FASTConvLayer(nn.Module):
    """Convolutional layer used in the TextNet and FAST architectures"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.groups = groups
        self.in_channels = in_channels
        self.converted_ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.hor_conv, self.hor_bn = None, None
        self.ver_conv, self.ver_bn = None, None

        padding = (int(((self.converted_ks[0] - 1) * dilation) / 2), int(((self.converted_ks[1] - 1) * dilation) / 2))

        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.converted_ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if self.converted_ks[1] != 1:
            self.ver_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(self.converted_ks[0], 1),
                padding=(int(((self.converted_ks[0] - 1) * dilation) / 2), 0),
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.ver_bn = nn.BatchNorm2d(out_channels)

        if self.converted_ks[0] != 1:
            self.hor_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, self.converted_ks[1]),
                padding=(0, int(((self.converted_ks[1] - 1) * dilation) / 2)),
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.hor_bn = nn.BatchNorm2d(out_channels)

        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "fused_conv"):
            return self.activation(self.fused_conv(x))

        main_outputs = self.bn(self.conv(x))
        vertical_outputs = self.ver_bn(self.ver_conv(x)) if self.ver_conv is not None and self.ver_bn is not None else 0
        horizontal_outputs = (
            self.hor_bn(self.hor_conv(x)) if self.hor_bn is not None and self.hor_conv is not None else 0
        )
        id_out = self.rbr_identity(x) if self.rbr_identity is not None else 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)

    # The following logic is used to reparametrize the layer
    # Borrowed from: https://github.com/czczup/FAST/blob/main/models/utils/nas_utils.py
    def _identity_to_conv(self, identity: nn.BatchNorm2d | None) -> tuple[torch.Tensor, torch.Tensor] | tuple[int, int]:
        if identity is None or identity.running_var is None:
            return 0, 0
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        std = (identity.running_var + identity.eps).sqrt()
        t = (identity.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, identity.bias - identity.running_mean * identity.weight / std  # type: ignore[operator]

    def _fuse_bn_tensor(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        std = (bn.running_var + bn.eps).sqrt()  # type: ignore
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std  # type: ignore[operator]

    def _get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_mxn, bias_mxn = self._fuse_bn_tensor(self.conv, self.bn)
        if self.ver_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)  # type: ignore[arg-type]
        else:
            kernel_mx1, bias_mx1 = 0, 0  # type: ignore[assignment]
        if self.hor_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)  # type: ignore[arg-type]
        else:
            kernel_1xn, bias_1xn = 0, 0  # type: ignore[assignment]
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel: torch.Tensor) -> torch.Tensor:
        kernel_height, kernel_width = self.converted_ks
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right, pad_top_down, pad_top_down], value=0)

    def reparameterize_layer(self):
        if hasattr(self, "fused_conv"):
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,  # type: ignore[arg-type]
            stride=self.conv.stride,  # type: ignore[arg-type]
            padding=self.conv.padding,  # type: ignore[arg-type]
            dilation=self.conv.dilation,  # type: ignore[arg-type]
            groups=self.conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias  # type: ignore[union-attr]
        for para in self.parameters():
            para.detach_()
        for attr in ["conv", "bn", "ver_conv", "ver_bn", "hor_conv", "hor_bn"]:
            if hasattr(self, attr):
                self.__delattr__(attr)

        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
