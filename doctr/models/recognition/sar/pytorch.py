# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS

from ...classification import resnet31
from ...utils.pytorch import _bf16_to_float32, load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ["SAR", "sar_resnet31"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "sar_resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/sar_resnet31-9a1deedf.pt&src=0",
    },
}


class SAREncoder(nn.Module):
    def __init__(self, in_feats: int, rnn_units: int, dropout_prob: float = 0.0) -> None:
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
        self.feat_conv = nn.Conv2d(feat_chans, attention_units, kernel_size=3, padding=1)
        # No need to add another bias since both tensors are summed together
        self.state_conv = nn.Conv2d(state_chans, attention_units, kernel_size=1, bias=False)
        self.attention_projector = nn.Conv2d(attention_units, 1, kernel_size=1, bias=False)

    def forward(
        self,
        features: torch.Tensor,  # (N, C, H, W)
        hidden_state: torch.Tensor,  # (N, C)
    ) -> torch.Tensor:
        H_f, W_f = features.shape[2:]

        # (N, feat_chans, H, W) --> (N, attention_units, H, W)
        feat_projection = self.feat_conv(features)
        # (N, state_chans, 1, 1) --> (N, attention_units, 1, 1)
        hidden_state = hidden_state.view(hidden_state.size(0), hidden_state.size(1), 1, 1)
        state_projection = self.state_conv(hidden_state)
        state_projection = state_projection.expand(-1, -1, H_f, W_f)
        # (N, attention_units, 1, 1) --> (N, attention_units, H_f, W_f)
        attention_weights = torch.tanh(feat_projection + state_projection)
        # (N, attention_units, H_f, W_f) --> (N, 1, H_f, W_f)
        attention_weights = self.attention_projector(attention_weights)
        B, C, H, W = attention_weights.size()

        # (N, H, W) --> (N, 1, H, W)
        attention_weights = torch.softmax(attention_weights.view(B, -1), dim=-1).view(B, C, H, W)
        # fuse features and attention weights (N, C)
        return (features * attention_weights).sum(dim=(2, 3))


class SARDecoder(nn.Module):
    """Implements decoder module of the SAR model

    Args:
    ----
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        vocab_size: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units

    """

    def __init__(
        self,
        rnn_units: int,
        max_length: int,
        vocab_size: int,
        embedding_units: int,
        attention_units: int,
        feat_chans: int = 512,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.embed = nn.Linear(self.vocab_size + 1, embedding_units)
        self.embed_tgt = nn.Embedding(embedding_units, self.vocab_size + 1)
        self.attention_module = AttentionModule(feat_chans, rnn_units, attention_units)
        self.lstm_cell = nn.LSTMCell(rnn_units, rnn_units)
        self.output_dense = nn.Linear(2 * rnn_units, self.vocab_size + 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        features: torch.Tensor,  # (N, C, H, W)
        holistic: torch.Tensor,  # (N, C)
        gt: Optional[torch.Tensor] = None,  # (N, L)
    ) -> torch.Tensor:
        if gt is not None:
            gt_embedding = self.embed_tgt(gt)

        logits_list: List[torch.Tensor] = []

        for t in range(self.max_length + 1):  # 32
            if t == 0:
                # step to init the first states of the LSTMCell
                hidden_state_init = cell_state_init = torch.zeros(
                    features.size(0), features.size(1), device=features.device
                )
                hidden_state, cell_state = hidden_state_init, cell_state_init
                prev_symbol = holistic
            elif t == 1:
                # step to init a 'blank' sequence of length vocab_size + 1 filled with zeros
                # (N, vocab_size + 1) --> (N, embedding_units)
                prev_symbol = torch.zeros(features.size(0), self.vocab_size + 1, device=features.device)
                prev_symbol = self.embed(prev_symbol)
            else:
                if gt is not None:
                    # (N, embedding_units) -2 because of <bos> and <eos> (same)
                    prev_symbol = self.embed(gt_embedding[:, t - 2])
                else:
                    # -1 to start at timestep where prev_symbol was initialized
                    index = logits_list[t - 1].argmax(-1)
                    # update prev_symbol with ones at the index of the previous logit vector
                    # (N, embedding_units)
                    prev_symbol = prev_symbol.scatter_(1, index.unsqueeze(1), 1)

            # (N, C), (N, C)  take the last hidden state and cell state from current timestep
            hidden_state_init, cell_state_init = self.lstm_cell(prev_symbol, (hidden_state_init, cell_state_init))
            hidden_state, cell_state = self.lstm_cell(hidden_state_init, (hidden_state, cell_state))
            # (N, C, H, W), (N, C) --> (N, C)
            glimpse = self.attention_module(features, hidden_state)
            # (N, C), (N, C) --> (N, 2 * C)
            logits = torch.cat([hidden_state, glimpse], dim=1)
            logits = self.dropout(logits)
            # (N, vocab_size + 1)
            logits_list.append(self.output_dense(logits))

        # (max_length + 1, N, vocab_size + 1) --> (N, max_length + 1, vocab_size + 1)
        return torch.stack(logits_list[1:]).permute(1, 0, 2)


class SAR(nn.Module, RecognitionModel):
    """Implements a SAR architecture as described in `"Show, Attend and Read:A Simple and Strong Baseline for
    Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
    ----
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        rnn_units: int = 512,
        embedding_units: int = 512,
        attention_units: int = 512,
        max_length: int = 30,
        dropout_prob: float = 0.0,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg

        self.max_length = max_length + 1  # Add 1 timestep for EOS after the longest word

        self.feat_extractor = feature_extractor

        # Size the LSTM
        self.feat_extractor.eval()
        with torch.no_grad():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape)))["features"].shape
        # Switch back to original mode
        self.feat_extractor.train()

        self.encoder = SAREncoder(out_shape[1], rnn_units, dropout_prob)
        self.decoder = SARDecoder(
            rnn_units,
            self.max_length,
            len(self.vocab),
            embedding_units,
            attention_units,
            dropout_prob=dropout_prob,
        )

        self.postprocessor = SARPostProcessor(vocab=vocab)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        features = self.feat_extractor(x)["features"]
        # NOTE: use max instead of functional max_pool2d which leads to ONNX incompatibility (kernel_size)
        # Vertical max pooling (N, C, H, W) --> (N, C, W)
        pooled_features = features.max(dim=-2).values
        # (N, W, C)
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()
        # (N, C)
        encoded = self.encoder(pooled_features)
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training for teacher forcing")

        decoded_features = _bf16_to_float32(self.decoder(features, encoded, gt=None if target is None else gt))

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(decoded_features)

        if target is not None:
            out["loss"] = self.compute_loss(decoded_features, gt, seq_len)

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
        ----
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
        -------
            The loss of the model on the batch
        """
        # Input length : number of timesteps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token
        seq_len = seq_len + 1  # type: ignore[assignment]
        # Compute loss
        # (N, L, vocab_size + 1)
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt, reduction="none")
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
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
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.detach().cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().clip(0, 1).tolist()))


def _sar(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> SAR:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone),
        {layer: "features"},
    )
    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

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
    ----
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the SAR architecture

    Returns:
    -------
        text recognition architecture
    """
    return _sar(
        "sar_resnet31",
        pretrained,
        resnet31,
        "10",
        ignore_keys=[
            "decoder.embed.weight",
            "decoder.embed_tgt.weight",
            "decoder.output_dense.weight",
            "decoder.output_dense.bias",
        ],
        **kwargs,
    )
