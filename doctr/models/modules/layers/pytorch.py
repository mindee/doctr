import torch.nn as nn

class RepConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  dilation=1, groups=1):
        super(RepConvLayer, self).__init__()
        
        padding = (int(((kernel_size[0] - 1) * dilation) / 2),
                   int(((kernel_size[1] - 1) * dilation) / 2))
        
        self.activation = nn.ReLU(inplace=True)
        self.main_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation, 
            groups=groups,
            bias=False,
        )
        self.main_bn = nn.BatchNorm2d(out_channels)

        ver_pad = (int(((kernel_size[0] - 1) * dilation) / 2), 0)
        hor_pad = (0, int(((kernel_size[1] - 1) * dilation) / 2))
        
        
        if kernel_size[1] != 1:
            self.ver_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=stride,
                padding=ver_pad,
                dilation=dilation, 
                groups=groups,
                bias=False
            )
            self.ver_bn = nn.BatchNorm2d(out_channels)
        else:
             self.ver_conv, self.ver_bn = None, None
             
        if kernel_size[0] != 1:
            self.hor_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size[1]),
                stride=stride,
                padding=hor_pad,
                dilation=dilation, 
                groups=groups,
                bias=False
            )
            self.hor_bn = nn.BatchNorm2d(out_channels)
        else:
            self.hor_conv, self.hor_bn = None, None
        
        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, input):
    
        main_outputs = self.main_bn(self.main_conv(input))
        vertical_outputs = self.ver_bn(self.ver_conv(input)) if self.ver_conv is not None else 0
        horizontal_outputs = self.hor_bn(self.hor_conv(input)) if self.hor_conv is not None else 0
        id_out = self.rbr_identity(input) if self.rbr_identity is not None else 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)
