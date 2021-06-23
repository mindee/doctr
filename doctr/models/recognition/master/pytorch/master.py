# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch import nn
from torch.nn import functional as F
from ...datasets import VOCABS

__all__ = ['MASTER', 'MASTERPostProcessor', 'master']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'master': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'input_shape': (3, 48, 160),
        'post_processor': 'MASTERPostProcessor',
        'vocab': VOCABS['french'],
        'url': None,
    },
}


class MAGC(nn.Module):

    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        att_scale: if True, re-scale attention to counteract the variance distibutions
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 1,
        att_scale: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.headers = headers  # h
        self.inplanes = inplanes  # C
        self.att_scale = att_scale

        self.single_header_inplanes = int(inplanes / headers)  # C / h

        self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
        )

    def context_modeling(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        # [N*headers, C', H , W] C = headers * C'
        x = inputs.view(batch * self.headers, self.single_header_inplanes, height, width)
        shortcut = x

        # [N*headers, C', H * W] C = headers * C'
        # input_x = input_x.view(batch, channel, height * width)
        shortcut = shortcut.view(batch * self.headers, self.single_header_inplanes, height * width)

        # [N*headers, 1, C', H * W]
        shortcut = shortcut.unsqueeze(1)
        # [N*headers, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N*headers, 1, H * W]
        context_mask = context_mask.view(batch * self.headers, 1, height * width)

        # scale variance
        if self.att_scale and self.headers > 1:
            context_mask = context_mask / torch.sqrt(self.single_header_inplanes)

        # [N*headers, 1, H * W]
        context_mask = self.softmax(context_mask)

        # [N*headers, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
        context = torch.matmul(shortcut, context_mask)

        # [N, headers * C', 1, 1]
        context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)

        return context

    def call(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        # Context modeling: B, C, H, W  ->  B, C, 1, 1
        context = self.context_modeling(inputs)
        # Transform: B, C, 1, 1 ->  B, C, 1, 1
        transformed = self.channel_add_conv(context)
        return inputs + transformed


decoder_layer = nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-05,
    batch_first=False,
)

decoder = nn.TransformerDecoder(
    decoder_layer,
    num_layers=3,
    norm=None
)