# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from doctr.datasets import VOCABS
from doctr.models import backbones

from ...utils import load_pretrained_params
from ..transformer import Decoder, positional_encoding
from .base import _MASTER, _MASTERPostProcessor

__all__ = ['MASTER', 'master']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'master': {
        'backbone': 'magc_resnet31',
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'input_shape': (3, 48, 160),
        'vocab': VOCABS['french'],
        'url': None,
    },
}


class MASTER(_MASTER, nn.Module):
    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/wenwenyu/MASTER-pytorch>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        dropout: dropout probability of the decoder
        input_shape: size of the image inputs
        cfg: dictionary containing information about the model
    """

    feature_pe: torch.Tensor

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        d_model: int = 512,
        dff: int = 2048,
        num_heads: int = 8,  # number of heads in the transformer decoder
        num_layers: int = 3,
        max_length: int = 50,
        dropout: float = 0.2,
        input_shape: Tuple[int, int, int] = (3, 48, 160),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.max_length = max_length
        self.vocab = vocab
        self.cfg = cfg
        self.vocab_size = len(vocab)
        self.num_heads = num_heads

        self.feat_extractor = feature_extractor
        self.seq_embedding = nn.Embedding(self.vocab_size + 3, d_model)  # 3 more for EOS/SOS/PAD

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=self.vocab_size,
            maximum_position_encoding=max_length,
            dropout=dropout,
        )
        self.register_buffer('feature_pe', positional_encoding(input_shape[1] * input_shape[2], d_model))
        self.linear = nn.Linear(d_model, self.vocab_size + 3)

        self.postprocessor = MASTERPostProcessor(vocab=self.vocab)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_mask(self, target: torch.Tensor) -> torch.Tensor:
        size = target.size(1)
        look_ahead_mask = ~ (torch.triu(torch.ones(size, size, device=target.device)) == 1).transpose(0, 1)[:, None]
        target_padding_mask = torch.eq(target, self.vocab_size + 2)  # Pad symbol
        combined_mask = target_padding_mask | look_ahead_mask
        return torch.tile(combined_mask.permute(1, 0, 2), (self.num_heads, 1, 1))

    @staticmethod
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of timesteps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token (sos disappear in shift!)
        seq_len = seq_len + 1
        # Compute loss: don't forget to shift gt! Otherwise the model learns to output the gt[t-1]!
        # The "masked" first gt char is <sos>. Delete last logit of the model output.
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:], reduction='none')
        # Compute mask, remove 1 timestep here as well
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """Call function for training

        Args:
            x: images
            target: list of str labels
            return_model_output: if True, return logits
            return_preds: if True, decode logits

        Returns:
            A torch tensor, containing logits
        """

        # Encode
        feature = self.feat_extractor(x)
        b, c, h, w = (feature.size(i) for i in range(4))
        feature = torch.reshape(feature, shape=(b, c, h * w))
        feature = feature.permute(0, 2, 1)  # shape (b, h*w, c)
        encoded = feature + self.feature_pe[:, :h * w, :]

        out: Dict[str, Any] = {}

        if target is not None:
            # Compute target: tensor of gts and sequence lengths
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training:
            if target is None:
                raise AssertionError("In training mode, you need to pass a value to 'target'")
            tgt_mask = self.make_mask(gt)
            # Compute logits
            output = self.decoder(gt, encoded, tgt_mask, None)
            logits = self.linear(output)

        else:
            logits = self.decode(encoded)

        if target is not None:
            out['loss'] = self.compute_loss(logits, gt, seq_len)

        if return_model_output:
            out['out_map'] = logits

        if return_preds:
            predictions = self.postprocessor(logits)
            out['preds'] = predictions

        return out

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode function for prediction

        Args:
            encoded: input tensor

        Return:
            A Tuple of torch.Tensor: predictions, logits
        """
        b = encoded.size(0)

        # Padding symbol + SOS at the beginning
        ys = torch.full((b, self.max_length), self.vocab_size + 2, dtype=torch.long, device=encoded.device)
        ys[:, 0] = self.vocab_size + 1

        # Final dimension include EOS/SOS/PAD
        for i in range(self.max_length - 1):
            ys_mask = self.make_mask(ys)
            output = self.decoder(ys, encoded, ys_mask, None)
            logits = self.linear(output)
            prob = F.softmax(logits, dim=-1)
            next_word = torch.max(prob, dim=-1).indices
            ys[:, i + 1] = next_word[:, i + 1]

        # Shape (N, max_length, vocab_size + 1)
        return logits


class MASTERPostProcessor(_MASTERPostProcessor):
    """Post processor for MASTER architectures
    """

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = logits.argmax(-1)
        # N x L
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        # Take the minimum confidence of the sequence
        probs = probs.min(dim=1).values.detach().cpu()

        # Manual decoding
        word_values = [
            ''.join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().tolist()))


def _master(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    input_shape: Tuple[int, int, int] = None,
    **kwargs: Any
) -> MASTER:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])

    kwargs['vocab'] = _cfg['vocab']

    # Build the model
    model = MASTER(
        backbones.__dict__[_cfg['backbone']](pretrained=pretrained_backbone),
        cfg=_cfg,
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def master(pretrained: bool = False, **kwargs: Any) -> MASTER:
    """MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Example::
        >>> import torch
        >>> from doctr.models import master
        >>> model = master(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 48, 160))
        >>> out = model(input_tensor)
    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
    Returns:
        text recognition architecture
    """

    return _master('master', pretrained, **kwargs)
