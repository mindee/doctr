from typing import Any, Union

import torch
import torch.nn as nn
import numpy as np

__all__ = ["RepConvLayer"]


class RepConvLayer(nn.Module):
    """Reparameterized Convolutional Layer"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[Any],groups: int =1, deploy: bool = False,  **kwargs: Any) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        
        
        dilation = kwargs.get("dilation", 1)
        stride = kwargs.get("stride", 1)
        kwargs.pop("padding", None)
        kwargs.pop("bias", None)

        self.hor_conv, self.hor_bn = None, None
        self.ver_conv, self.ver_bn = None, None

        padding = (int(((kernel_size[0] - 1) * dilation) / 2), int(((kernel_size[1] - 1) * dilation) / 2))

        self.activation = nn.ReLU(inplace=True)
        self.main_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            **kwargs,
        )

        self.main_bn = nn.BatchNorm2d(out_channels)

        if kernel_size[1] != 1:
            self.ver_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                padding=(int(((kernel_size[0] - 1) * dilation) / 2), 0),
                bias=False,
                **kwargs,
            )
            self.ver_bn = nn.BatchNorm2d(out_channels)

        if kernel_size[0] != 1:
            self.hor_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size[1]),
                padding=(0, int(((kernel_size[1] - 1) * dilation) / 2)),
                bias=False,
                **kwargs,
            )
            self.hor_bn = nn.BatchNorm2d(out_channels)

        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_outputs = self.main_bn(self.main_conv(x))

        if self.ver_conv is not None and self.ver_bn is not None:
            vertical_outputs = self.ver_bn(self.ver_conv(x))
        else:
            vertical_outputs = 0

        if self.hor_bn is not None and self.hor_conv is not None:
            horizontal_outputs = self.hor_bn(self.hor_conv(x))
        else:
            horizontal_outputs = 0

        if self.rbr_identity is not None and self.ver_bn is not None:
            id_out = self.rbr_identity(x)
        else:
            id_out = 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)

    def _identity_to_conv(self, identity):
        if identity is None:
            return 0, 0
        assert isinstance(identity, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        running_mean = identity.running_mean
        running_var = identity.running_var
        gamma = identity.weight
        beta = identity.bias
        eps = identity.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel_mxn, bias_mxn = self._fuse_bn_tensor(self.main_conv, self.main_bn)
        if self.ver_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        else:
            kernel_mx1, bias_mx1 = 0, 0
        if self.hor_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        else:
            kernel_1xn, bias_1xn = 0, 0
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel):
        kernel_height, kernel_width = self.kernel_size
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right,
                                                pad_top_down, pad_top_down])

    def switch_to_deploy(self):
        if hasattr(self, 'fused_conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(in_channels=self.main_conv.in_channels,
                                    out_channels=self.main_conv.out_channels,
                                    kernel_size=self.main_conv.kernel_size, stride=self.main_conv.stride,
                                    padding=self.main_conv.padding, dilation=self.main_conv.dilation,
                                    groups=self.main_conv.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        self.deploy = True
        for para in self.parameters():
            para.detach_()
        for attr in ['main_conv', 'main_bn', 'ver_conv', 'ver_bn', 'hor_conv', 'hor_bn']:
            if hasattr(self, attr):
                self.__delattr__(attr)

        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

    def switch_to_test(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(in_channels=self.main_conv.in_channels,
                                    out_channels=self.main_conv.out_channels,
                                    kernel_size=self.main_conv.kernel_size, stride=self.main_conv.stride,
                                    padding=self.main_conv.padding, dilation=self.main_conv.dilation,
                                    groups=self.main_conv.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.fused_conv.parameters():
            para.detach_()
        self.deploy = True

    def switch_to_train(self):
        if hasattr(self, 'fused_conv'):
            self.__delattr__('fused_conv')
        self.deploy = False

    @staticmethod
    def is_zero_layer():
        return False

    @property
    def module_str(self):
        return 'Rep_%dx%d' % (self.kernel_size[0], self.kernel_size[1])

    @property
    def config(self):
        return {'name': RepConvLayer.__name__,
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'dilation': self.dilation,
                'groups': self.groups}

    @staticmethod
    def build_from_config(config):
        return RepConvLayer(**config)
