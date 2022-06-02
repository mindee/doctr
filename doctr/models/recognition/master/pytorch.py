# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS
from doctr.models.classification import magc_resnet31

from ...utils.pytorch import load_pretrained_params
from ..transformer.pytorch import Decoder, PositionalEncoding
from .base import _MASTER, _MASTERPostProcessor

__all__ = ['MASTER', 'master']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'master': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
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
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.max_length = max_length
        self.d_model = d_model
        self.vocab = vocab
        self.cfg = cfg
        self.vocab_size = len(vocab)

        self.feat_extractor = feature_extractor
        self.positional_encoding = PositionalEncoding(self.d_model, dropout, max_len=input_shape[1] * input_shape[2])
        self.embed = nn.Embedding(self.vocab_size + 3, self.d_model)

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=self.d_model,
            num_heads=num_heads,
            dff=dff,
            dropout=dropout,
        )

        self.linear = nn.Linear(self.d_model, self.vocab_size + 3)
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

    def make_source_and_target_mask(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # borrowed from  https://github.com/wenwenyu/MASTER-pytorch
        target_pad_mask = (target != self.vocab_size + 2).unsqueeze(1).unsqueeze(3)  # (b, 1, len_src, 1)
        target_length = target.size(1)
        target_sub_mask = torch.tril(
            torch.ones((target_length, target_length), dtype=torch.uint8, device=source.device)
        )
        source_mask = torch.ones((target_length, source.size(1)), dtype=torch.uint8, device=source.device)
        target_mask = target_pad_mask & target_sub_mask.bool()
        return source_mask, target_mask

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
        features = self.feat_extractor(x)['features']
        b, c, h, w = features.shape
        # (N, C, H, W) --> (N, H * W, C)
        features = features.view(b, c, h * w).permute((0, 2, 1))
        # add positional encoding to features
        encoded = self.positional_encoding(features)

        out: Dict[str, Any] = {}

        if target is not None:
            # Compute target: tensor of gts and sequence lengths
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

            # Compute logits
            source_mask, target_mask = self.make_source_and_target_mask(encoded, gt)
            target = self.embed(gt) * math.sqrt(self.d_model)
            target = self.positional_encoding(target)
            output = self.decoder(target, encoded, source_mask, target_mask)
            logits = self.linear(output)
        else:
            logits = self.decode(encoded)

        if target is not None:
            out['loss'] = self.compute_loss(logits, gt, seq_len)

        if return_model_output:
            out['out_map'] = logits

        if return_preds:
            out['preds'] = self.postprocessor(logits)

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
        ys = torch.zeros((b, self.max_length), dtype=torch.long, device=encoded.device) * self.vocab_size + 2  # pad
        ys[:, 0] = self.vocab_size + 1  # sos

        # Final dimension include EOS/SOS/PAD
        for i in range(self.max_length - 1):

            source_mask, target_mask = self.make_source_and_target_mask(encoded, ys)
            target = self.embed(ys) * math.sqrt(self.d_model)
            target = self.positional_encoding(target)
            output = self.decoder(target, encoded, source_mask, target_mask)
            logits = self.linear(output)
            prob = torch.softmax(logits, dim=-1)
            next_word = torch.max(prob, dim=-1).indices
            ys[:, i + 1] = next_word[:, i]

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
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    **kwargs: Any
) -> MASTER:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = kwargs.get('input_shape', _cfg['input_shape'])
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])

    kwargs['vocab'] = _cfg['vocab']
    kwargs['input_shape'] = _cfg['input_shape']

    # Build the model
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone),
        {layer: 'features'},
    )
    model = MASTER(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

        # TODO: filter vocab_size specific weights and biases

    return model


def master(pretrained: bool = False, **kwargs: Any) -> MASTER:
    """MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.

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

    return _master('master', pretrained, magc_resnet31, '10', **kwargs)
