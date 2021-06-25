# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Tuple, Optional, List

from ....datasets import VOCABS
from ...utils import conv_sequence_pt
from ...backbones import resnet_stage
from .transformer_pt import Decoder, positional_encoding, create_look_ahead_mask, create_padding_mask
from .base import _MASTER

__all__ = ['MASTER']


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
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),
            nn.LayerNorm([self.inplanes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1)
        )

    def context_modeling(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = inputs.size()
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
            context_mask = context_mask / math.sqrt(self.single_header_inplanes)

        # [N*headers, 1, H * W]
        context_mask = self.softmax(context_mask)

        # [N*headers, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
        context = torch.matmul(shortcut, context_mask)

        # [N, headers * C', 1, 1]
        context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)

        return context

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        # Context modeling: B, C, H, W  ->  B, C, 1, 1
        context = self.context_modeling(inputs)
        # Transform: B, C, 1, 1 ->  B, C, 1, 1
        transformed = self.channel_add_conv(context)
        return inputs + transformed


class MAGCResnet(nn.Sequential):

    """Implements the modified resnet with MAGC layers, as described in paper.

    Args:
        headers: number of header to split channels in MAGC layers
        input_shape: shape of the model input (without batch dim)
    """

    def __init__(
        self,
        headers: int = 1,
    ) -> None:
        _layers = [
            # conv_1x
            *conv_sequence_pt(3, 64, relu=True, bn=True, kernel_size=3, padding=1),
            *conv_sequence_pt(64, 128, relu=True, bn=True, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # conv_2x
            *resnet_stage(128, 256, num_blocks=1),
            MAGC(inplanes=256, headers=headers, att_scale=True),
            *conv_sequence_pt(256, 256, relu=True, bn=True, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # conv_3x
            *resnet_stage(256, 512, num_blocks=2),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence_pt(512, 512, relu=True, bn=True, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 1)),
            # conv_4x
            *resnet_stage(512, 512, num_blocks=5),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence_pt(512, 512, relu=True, bn=True, kernel_size=3, padding=1),
            # conv_5x
            *resnet_stage(512, 512, num_blocks=3),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence_pt(512, 512, relu=True, bn=True, kernel_size=3, padding=1),
        ]
        super().__init__(*_layers)


class MASTER(_MASTER, nn.Module):

    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/wenwenyu/MASTER-pytorch>`_.

    Args:
        vocab: vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        headers: headers for the MAGC module
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        input_size: size of the image inputs
    """

    def __init__(
        self,
        vocab: str,
        d_model: int = 512,
        headers: int = 1,
        dff: int = 2048,
        num_heads: int = 8,
        num_layers: int = 3,
        max_length: int = 50,
        input_shape: Tuple[int, int, int] = (3, 48, 160),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.max_length = max_length
        self.vocab = vocab
        self.cfg = cfg
        self.vocab_size = len(vocab)
        self.num_heads = num_heads

        self.feature_extractor = MAGCResnet(headers=headers)
        self.seq_embedding = nn.Embedding(self.vocab_size + 1, d_model)  # One additional class for EOS

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=self.vocab_size,
            maximum_position_encoding=max_length,
        )
        self.feature_pe = positional_encoding(input_shape[1] * input_shape[2], d_model)
        self.linear = nn.Linear(d_model, self.vocab_size + 1)
        nn.init.kaiming_uniform_(self.linear.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_mask(self, target: torch.Tensor) -> torch.Tensor:
        look_ahead_mask = create_look_ahead_mask(target.size(1))
        target_padding_mask = create_padding_mask(target, self.vocab_size)  # Pad with EOS
        combined_mask = target_padding_mask & look_ahead_mask
        return torch.tile(combined_mask.permute(1, 0, 2), (self.num_heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Call function for training

        Args:
            x: images
            target: list of str labels
            return_model_output: if True, return logits
            return_preds: if True, decode logits

        Return:
            A torch tensor, containing logits
        """

        # Encode
        feature = self.feature_extractor(x, **kwargs)
        b, c, h, w = (feature.size(i) for i in range(4))
        feature = torch.reshape(feature, shape=(b, c, h * w))
        feature = feature.permute(0, 2, 1)  # shape (b, h*w, c)
        encoded = feature + self.feature_pe[:, :h * w, :]

        if target is not None:
            # Compute target: tensor of gts and sequence lengths
            gt, seq_len = self.compute_target(target)
            tgt_mask = self.make_mask(torch.from_numpy(gt))
            # Compute logits
            output = self.decoder(torch.from_numpy(gt), encoded, tgt_mask, None)
            logits = self.linear(output)

        else:
            _, logits = self.decode(encoded)

        return logits

    def decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode function for prediction

        Args:
            x: input tensor

        Return:
            A Tuple of torch.Tensor: predictions, logits
        """
        # Encode
        feature = self.feature_extractor(x)
        b, c, h, w = (feature.size(i) for i in range(4))
        feature = torch.reshape(feature, shape=(b, c, h * w))
        feature = feature.permute(0, 2, 1)  # shape (b, h*w, c)
        encoded = feature + self.feature_pe[:, :h * w, :]

        ys = torch.ones((b, self.max_length - 1), dtype=torch.long) * self.vocab_size  # padding symbol
        start_vector = torch.ones((b, 1), dtype=torch.long) * self.vocab_size + 1  # SOS
        ys = torch.cat((start_vector, ys), axis=-1)

        final_logits = torch.zeros((b, self.max_length - 1, self.vocab_size + 1), dtype=torch.long)  # EOS
        # max_len = len + 2
        for i in range(self.max_length - 1):
            ys_mask = self.make_mask(ys)
            output = self.decoder(ys, encoded, ys_mask, None)
            logits = self.linear(output)
            prob = F.softmax(logits, dim=-1)
            max_prob, next_word = torch.max(prob, dim=-1)
            ys[:, i + 1] = next_word[:, i]

            if i == (self.max_length - 2):
                final_logits = logits

        # ys predictions of shape B x max_length, final_logits of shape B x max_length x vocab_size + 1
        return ys, final_logits
