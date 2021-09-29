# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Dict, List, Any, Optional

from ...backbones import resnet31
from ...utils import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor
from ....datasets import VOCABS


__all__ = ['SAR', 'sar_resnet31']

default_cfgs: Dict[str, Dict[str, Any]] = {
    'sar_resnet31': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'backbone': resnet31, 'rnn_units': 512, 'max_length': 30, 'num_decoders': 2,
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['legacy_french'],
        'url': None,
    },
}


class AttentionModule(nn.Module):

    def __init__(self, feat_chans: int, state_chans: int, attention_units: int) -> None:
        super().__init__()
        self.feat_conv = nn.Conv2d(feat_chans, attention_units, 3, padding=1)
        # No need to add another bias since both tensors are summed together
        self.state_conv = nn.Conv2d(state_chans, attention_units, 1, bias=False)
        self.attention_projector = nn.Conv2d(attention_units, 1, 1, bias=False)

    def forward(self, features: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        # shape (N, vgg_units, H, W) -> (N, attention_units, H, W)
        feat_projection = self.feat_conv(features)
        # shape (N, rnn_units, 1, 1) -> (N, attention_units, 1, 1)
        state_projection = self.state_conv(hidden_state)
        projection = torch.tanh(feat_projection + state_projection)
        # shape (N, attention_units, H, W) -> (N, 1, H, W)
        attention = self.attention_projector(projection)
        # shape (N, 1, H, W) -> (N, H * W)
        attention = torch.flatten(attention, 1)
        # shape (N, H * W) -> (N, 1, H, W)
        attention = torch.softmax(attention, 1).reshape(-1, 1, features.shape[-2], features.shape[-1])

        glimpse = (features * attention).sum(dim=(2, 3))

        return glimpse


class SARDecoder(nn.Module):
    """Implements decoder module of the SAR model

    Args:
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        vocab_size: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units
        num_decoder_layers: number of LSTM layers to stack

    """
    def __init__(
        self,
        rnn_units: int,
        max_length: int,
        vocab_size: int,
        embedding_units: int,
        attention_units: int,
        num_decoder_layers: int = 2,
        feat_chans: int = 512,
    ) -> None:

        super().__init__()
        self.vocab_size = vocab_size
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(rnn_units, rnn_units) for _ in range(num_decoder_layers)
        ])
        self.embed = nn.Linear(self.vocab_size + 1, embedding_units, bias=False)
        self.attention_module = AttentionModule(feat_chans, rnn_units, attention_units)
        self.output_dense = nn.Linear(2 * rnn_units, vocab_size + 1)
        self.max_length = max_length

    def forward(
        self,
        features: torch.Tensor,
        holistic: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # initialize states (each of shape (N, rnn_units))
        hx = [None, None]
        # Initialize with the index of virtual START symbol (placed after <eos>)
        symbol = torch.zeros((features.shape[0], self.vocab_size + 1), device=features.device, dtype=features.dtype)
        logits_list = []
        for t in range(self.max_length + 1):  # keep 1 step for <eos>

            # one-hot symbol with depth vocab_size + 1
            # embeded_symbol: shape (N, embedding_units)
            embeded_symbol = self.embed(symbol)

            hx[0] = self.lstm_cells[0](embeded_symbol, hx[0])
            hx[1] = self.lstm_cells[1](hx[0][0], hx[1])  # type: ignore[index]
            logits, _ = hx[1]  # type: ignore[misc]

            glimpse = self.attention_module(
                features, logits.unsqueeze(-1).unsqueeze(-1),  # type: ignore[has-type]
            )
            # logits: shape (N, rnn_units), glimpse: shape (N, 1)
            logits = torch.cat([logits, glimpse], 1)  # type: ignore[has-type]
            # shape (N, rnn_units + 1) -> (N, vocab_size + 1)
            logits = self.output_dense(logits)
            # update symbol with predicted logits for t+1 step
            if gt is not None:
                _symbol = gt[:, t]  # type: ignore[index]
            else:
                _symbol = logits.argmax(-1)
            symbol = F.one_hot(_symbol, self.vocab_size + 1).to(dtype=features.dtype)
            logits_list.append(logits)
        outputs = torch.stack(logits_list, 1)  # shape (N, max_length + 1, vocab_size + 1)

        return outputs


class SAR(nn.Module, RecognitionModel):
    """Implements a SAR architecture as described in `"Show, Attend and Read:A Simple and Strong Baseline for
    Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        num_decoders: number of LSTM to stack in decoder layer
        dropout_prob: dropout probability of the encoder LSTM
        cfg: default setup dict of the model
    """

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        rnn_units: int = 512,
        embedding_units: int = 512,
        attention_units: int = 512,
        max_length: int = 32,
        num_decoders: int = 2,
        dropout_prob: float = 0.,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.vocab = vocab
        self.cfg = cfg

        self.max_length = max_length + 1  # Add 1 timestep for EOS after the longest word

        self.feat_extractor = feature_extractor

        self.encoder = nn.LSTM(32, rnn_units, 2, batch_first=True, dropout=dropout_prob)

        self.decoder = SARDecoder(
            rnn_units, max_length, len(vocab), embedding_units, attention_units, num_decoders, 512,
        )

        self.postprocessor = SARPostProcessor(vocab=vocab)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:

        features = self.feat_extractor(x)
        pooled_features = features.max(dim=-2).values  # vertical max pooling
        _, (encoded, _) = self.encoder(pooled_features)
        encoded = encoded[-1]
        if target is not None:
            _gt, _seq_len = self.compute_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)  # type: ignore[assignment]
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)
        decoded_features = self.decoder(features, encoded, gt=None if target is None else gt)

        out: Dict[str, Any] = {}
        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(decoded_features)

        if target is not None:
            out['loss'] = self.compute_loss(decoded_features, gt, seq_len)  # type: ignore[arg-type]

        return out

    def compute_loss(
        self,
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
        # Add one for additional <eos> token
        seq_len = seq_len + 1
        # Compute loss
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt, reduction='none')
        # Compute mask
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] < seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures"""

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
            for encoded_seq in out_idxs.detach().cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().tolist()))


def _sar(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    input_shape: Tuple[int, int, int] = None,
    **kwargs: Any
) -> SAR:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])
    _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])
    _cfg['embedding_units'] = kwargs.get('embedding_units', _cfg['rnn_units'])
    _cfg['attention_units'] = kwargs.get('attention_units', _cfg['rnn_units'])
    _cfg['max_length'] = kwargs.get('max_length', _cfg['max_length'])
    _cfg['num_decoders'] = kwargs.get('num_decoders', _cfg['num_decoders'])

    # Feature extractor
    feat_extractor = default_cfgs[arch]['backbone'](pretrained=pretrained_backbone)
    # Trick to keep only the features while it's not unified between both frameworks
    if arch.split('_')[1] == "mobilenet":
        feat_extractor = feat_extractor.features

    kwargs['vocab'] = _cfg['vocab']
    kwargs['rnn_units'] = _cfg['rnn_units']
    kwargs['embedding_units'] = _cfg['embedding_units']
    kwargs['attention_units'] = _cfg['attention_units']
    kwargs['max_length'] = _cfg['max_length']
    kwargs['num_decoders'] = _cfg['num_decoders']

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def sar_resnet31(pretrained: bool = False, **kwargs: Any) -> SAR:
    """SAR with a resnet-31 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Example:
        >>> import torch
        >>> from doctr.models import sar_resnet31
        >>> model = sar_resnet31(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 32, 128))
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _sar('sar_resnet31', pretrained, **kwargs)
