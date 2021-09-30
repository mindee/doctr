# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from itertools import groupby
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Dict, Any, Optional, List

from ...backbones import vgg16_bn, resnet31, mobilenet_v3_small_r, mobilenet_v3_large_r
from ...utils import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor
from ....datasets import VOCABS

__all__ = ['CRNN', 'crnn_vgg16_bn', 'CTCPostProcessor', 'crnn_mobilenet_v3_small',
           'crnn_mobilenet_v3_large']

default_cfgs: Dict[str, Dict[str, Any]] = {
    'crnn_vgg16_bn': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'backbone': vgg16_bn, 'rnn_units': 128, 'lstm_features': 512,
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['legacy_french'],
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.1/crnn_vgg16_bn-9762b0b0.pt',
    },
    'crnn_mobilenet_v3_small': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'backbone': mobilenet_v3_small_r, 'rnn_units': 128, 'lstm_features': 576,
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'],
        'url': "https://github.com/mindee/doctr/releases/download/v0.3.1/crnn_mobilenet_v3_small_pt-3b919a02.pt",
    },
    'crnn_mobilenet_v3_large': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'backbone': mobilenet_v3_large_r, 'rnn_units': 128, 'lstm_features': 960,
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'],
        'url': "https://github.com/mindee/doctr/releases/download/v0.3.1/crnn_mobilenet_v3_large_pt-f5259ec2.pt",
    },
}


class CTCPostProcessor(RecognitionPostProcessor):
    """
    Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
    """
    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor, vocab: str = VOCABS['french'], blank: int = 0
    ) -> List[Tuple[str, float]]:
        """Implements best path decoding as shown by Graves (Dissertation, p63), highly inspired from
        <https://github.com/githubharald/CTCDecoder>`_.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            blank: index of blank label

        Returns:
            A list of tuples: (word, confidence)
        """
        # compute softmax
        probs = F.softmax(logits, dim=-1)
        # get char indices along best path
        best_path = torch.argmax(probs, dim=-1)
        # define word proba as min proba of sequence
        probs, _ = torch.max(probs, dim=-1)
        probs, _ = torch.min(probs, dim=1)

        words = []
        for sequence in best_path:
            # collapse best path (using itertools.groupby), map to chars, join char list to string
            collapsed = [vocab[k] for k, _ in groupby(sequence) if k != blank]
            res = ''.join(collapsed)
            words.append(res)

        return list(zip(words, probs.tolist()))

    def __call__(  # type: ignore[override]
        self,
        logits: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Decode CTC
        return self.ctc_best_path(logits=logits, vocab=self.vocab, blank=len(self.vocab))


class CRNN(RecognitionModel, nn.Module):
    """Implements a CRNN architecture as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        cfg: configuration dictionary
    """

    _children_names: List[str] = ['feat_extractor', 'decoder', 'linear', 'postprocessor']

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        lstm_features: int,
        rnn_units: int = 128,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.max_length = 32
        self.feat_extractor = feature_extractor

        self.decoder = nn.LSTM(
            input_size=lstm_features, hidden_size=rnn_units, batch_first=True, num_layers=2, bidirectional=True
        )

        # features units = 2 * rnn_units because bidirectional layers
        self.linear = nn.Linear(in_features=2 * rnn_units, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: List[str],
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.compute_target(target)
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)
        ctc_loss = F.ctc_loss(
            probs,
            torch.from_numpy(gt),
            input_length,
            torch.tensor(seq_len, dtype=torch.int),
            len(self.vocab),
            zero_infinity=True,
        )

        return ctc_loss

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:

        features = self.feat_extractor(x)
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        logits, _ = self.decoder(features_seq)
        logits = self.linear(logits)

        out: Dict[str, Any] = {}
        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits)

        if target is not None:
            out['loss'] = self.compute_loss(logits, target)

        return out


def _crnn(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any
) -> CRNN:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])
    _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])

    # Feature extractor
    feat_extractor = _cfg['backbone'](pretrained=pretrained_backbone)
    # Trick to keep only the features while it's not unified between both frameworks
    if arch.split('_')[1] == "mobilenet":
        feat_extractor = feat_extractor.features

    kwargs['vocab'] = _cfg['vocab']
    kwargs['rnn_units'] = _cfg['rnn_units']
    kwargs['lstm_features'] = _cfg['lstm_features']

    # Build the model
    model = CRNN(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def crnn_vgg16_bn(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a VGG-16 backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Example::
        >>> import torch
        >>> from doctr.models import crnn_vgg16_bn
        >>> model = crnn_vgg16_bn(pretrained=True)
        >>> input_tensor = torch.rand(1, 3, 32, 128)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn('crnn_vgg16_bn', pretrained, **kwargs)


def crnn_mobilenet_v3_small(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Small backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Example::
        >>> import torch
        >>> from doctr.models import crnn_mobilenet_v3_small
        >>> model = crnn_mobilenet_v3_small(pretrained=True)
        >>> input_tensor = torch.rand(1, 3, 32, 128)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn('crnn_mobilenet_v3_small', pretrained, **kwargs)


def crnn_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Large backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Example::
        >>> import torch
        >>> from doctr.models import crnn_mobilenet_v3_large
        >>> model = crnn_mobilenet_v3_large(pretrained=True)
        >>> input_tensor = torch.rand(1, 3, 32, 128)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn('crnn_mobilenet_v3_large', pretrained, **kwargs)
