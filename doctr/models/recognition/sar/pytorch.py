# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS

from ...classification import resnet31
from ...utils.pytorch import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ['SAR', 'sar_resnet31']

default_cfgs: Dict[str, Dict[str, Any]] = {
    'sar_resnet31': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['legacy_french'],
        'url': None,
    },
}


class SAREncoder(nn.Module):

    def __init__(self, in_feats: int, rnn_units: int, dropout_prob: float = 0.) -> None:

        super().__init__()
        self.rnn = nn.LSTM(in_feats, rnn_units, 2, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(rnn_units, rnn_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (N, L, C) --> (N, T, C)
        encoded = self.rnn(x)[0]
        # (N, C)
        return self.linear(encoded[:, -1, :])


class AttentionModule(nn.Module):

    def __init__(self, feat_chans: int, state_chans: int, attention_units: int) -> None:
        super().__init__()
        self.feat_conv = nn.Conv2d(feat_chans, attention_units, 3, padding=1)
        # No need to add another bias since both tensors are summed together
        self.state_conv = nn.Linear(state_chans, attention_units, bias=False)
        self.attention_projector = nn.Linear(attention_units, 1, bias=False)

    def forward(self, features: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        # shape (N, C, H, W) -> (N, attention_units, H, W)
        feat_projection = self.feat_conv(features)
        # shape (N, L, rnn_units) -> (N, L, attention_units)
        state_projection = self.state_conv(hidden_state).unsqueeze(-1).unsqueeze(-1)
        # (N, L, attention_units, H, W)
        projection = torch.tanh(feat_projection.unsqueeze(1) + state_projection)
        # (N, L, H, W, 1)
        attention = self.attention_projector(projection.permute(0, 1, 3, 4, 2))
        # shape (N, L, H, W, 1) -> (N, L, H * W)
        attention = torch.flatten(attention, 2)
        attention = torch.softmax(attention, -1)
        # shape (N, L, H * W) -> (N, L, 1, H, W)
        attention = attention.reshape(-1, hidden_state.shape[1], features.shape[-2], features.shape[-1])

        # (N, L, C)
        return (features.unsqueeze(1) * attention.unsqueeze(2)).sum(dim=(3, 4))


class SequentialSARDecoder(nn.Module):

    def __init__(self,
                 num_classes=37,
                 d_k=64,
                 d_model=512,
                 d_enc=512,
                 pred_dropout=0.0,
                 mask=True,
                 max_seq_len=40,
                 start_idx=0,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.d_k = d_k
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len
        self.mask = mask

        encoder_rnn_out_size = d_enc * 1
        decoder_rnn_out_size = encoder_rnn_out_size * 1
        # 2D attention layer
        self.conv1x1_1 = nn.Conv2d(decoder_rnn_out_size, d_k, kernel_size=1, stride=1)
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2d(d_k, 1, kernel_size=1, stride=1)


        self.rnn_decoder_layer1 = nn.LSTMCell(encoder_rnn_out_size, encoder_rnn_out_size)
        self.rnn_decoder_layer2 = nn.LSTMCell(encoder_rnn_out_size, encoder_rnn_out_size)

        # Decoder input embedding
        self.embedding = nn.Embedding(self.num_classes + 1, encoder_rnn_out_size)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_class = self.num_classes + 1
        self.prediction = nn.Linear(d_model, pred_num_class)

    def _2d_attention(self,
                      y_prev,
                      feat,
                      holistic_feat,
                      hx1,
                      cx1,
                      hx2,
                      cx2):
        _, _, h_feat, w_feat = feat.size()
        hx1, cx1 = self.rnn_decoder_layer1(y_prev, (hx1, cx1))
        hx2, cx2 = self.rnn_decoder_layer2(hx1, (hx2, cx2))

        tile_hx2 = hx2.view(hx2.size(0), hx2.size(1), 1, 1)
        attn_query = self.conv1x1_1(tile_hx2)  # bsz * attn_size * 1 * 1
        attn_query = attn_query.expand(-1, -1, h_feat, w_feat)
        attn_key = self.conv3x3_1(feat)
        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        attn_weight = self.conv1x1_2(attn_weight)
        bsz, c, h, w = attn_weight.size()
        assert c == 1


        attn_weight = F.softmax(attn_weight.view(bsz, -1), dim=-1)
        attn_weight = attn_weight.view(bsz, c, h, w)

        attn_feat = torch.sum(
            torch.mul(feat, attn_weight), (2, 3), keepdim=False)  # n * c

        # linear transformation
        y = self.prediction(attn_feat)

        return y, hx1, hx1, hx2, hx2

    def forward(self, feat, out_enc, gt):
        if gt is not None:
            tgt_embedding = self.embedding(gt)

        outputs = []
        start_token = torch.full((feat.size(0), ),
                                 0,
                                 device=feat.device,
                                 dtype=torch.long)
        start_token = self.embedding(start_token)
        print(start_token.size())
        for i in range(-1, self.max_seq_len + 1):
            if i == -1:
                hx1, cx1 = self.rnn_decoder_layer1(out_enc)
                hx2, cx2 = self.rnn_decoder_layer2(hx1)
                if gt is None:
                    y_prev = start_token
            else:
                if gt is not None:
                    y_prev = tgt_embedding[:, i, :]
                y, hx1, cx1, hx2, cx2 = self._2d_attention(y_prev, feat, out_enc, hx1, cx1, hx2, cx2)
                if gt is not None:
                    y = self.pred_dropout(y)
                else: # TODO: returned gerade immer 31 Zeichen
                    y = F.softmax(y, -1)
                    print(y.size())
                    _, max_idx = torch.max(y, dim=1, keepdim=False)
                    char_embedding = self.embedding(max_idx)
                    y_prev = char_embedding
                outputs.append(y)

        print(len(outputs))
        outputs = torch.stack(outputs, 1)
        print(outputs.size())

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
        max_length: int = 30,
        num_decoders: int = 2,
        dropout_prob: float = 0.,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.vocab = vocab
        self.cfg = cfg

        self.max_length = max_length + 1  # Add 1 timestep for EOS after the longest word

        self.feat_extractor = feature_extractor

        # Size the LSTM
        self.feat_extractor.eval()
        with torch.no_grad():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape)))['features'].shape
        # Switch back to original mode
        self.feat_extractor.train()

        self.encoder = SAREncoder(out_shape[1], rnn_units, dropout_prob)

        #self.decoder = SARDecoder(
        #    rnn_units, max_length, len(vocab), embedding_units, attention_units, num_decoders, out_shape[1],
        #)
        print(out_shape[1])
        print(out_shape)
        self.decoder = SequentialSARDecoder(num_classes=len(vocab), d_k=out_shape[1], max_seq_len=max_length)

        self.postprocessor = SARPostProcessor(vocab=vocab)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:

        features = self.feat_extractor(x)['features']
        # Vertical max pooling --> (N, C, W)
        pooled_features = F.max_pool2d(features, kernel_size=(features.shape[2], 1), stride=(1, 1))
        pooled_features = pooled_features.squeeze(2)
        # (N, W, C)
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()
        # (N, C)
        encoded = self.encoder(pooled_features)
        if target is not None:
            _gt, _seq_len = self.build_target(target)
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
        # Add one for additional <eos> token
        seq_len = seq_len + 1
        # Compute loss
        # (N, L, vocab_size + 1)
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt, reduction='none')
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
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
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    **kwargs: Any
) -> SAR:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])
    _cfg['input_shape'] = kwargs.get('input_shape', _cfg['input_shape'])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(False), # pretrained_backbone
        {layer: 'features'},
    )
    kwargs['vocab'] = _cfg['vocab']
    kwargs['input_shape'] = _cfg['input_shape']

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def sar_resnet31(pretrained: bool = False, **kwargs: Any) -> SAR:
    """SAR with a resnet-31 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

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

    return _sar('sar_resnet31', pretrained, resnet31, '10', **kwargs)
