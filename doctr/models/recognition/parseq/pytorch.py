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
from ...utils.pytorch import load_pretrained_params_local

import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import _PARSeq, _PARSeqPostProcessor
from ...classification import parseq as parseq_model
# MODIFIER INIT_WEIGTHS selon la fonction load_pretrained_params
#from strhub.models.utils import init_weights



__all__ = ["parseq"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "charset_train": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        "charset_test": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" ,
        "max_label_length": 25 ,
        "batch_size": 384,
        "lr": 7e-4,
        "warmup_pct": 0.075,
        "weight_decay": 0.0,
        "img_size": [ 32, 128 ],
        "patch_size": [ 4, 8 ] ,
        "embed_dim": 384 ,
        "enc_num_heads": 6,
        "enc_mlp_ratio": 4,
        "enc_depth": 12,
        "dec_num_heads": 12,
        "dec_mlp_ratio": 4 ,
        "dec_depth": 1,
        "perm_num": 6 ,
        "perm_forward": True ,
        "perm_mirrored": True ,
        "decode_ar": True,
        "refine_iters": 1,
        "dropout": 0.1,
        "vocab": VOCABS["french"],
        "input_shape": (3, 32, 128),
        "classes": list(VOCABS["french"]),
        "url": "/home/nikkokks/Desktop/github/parseq-bb5792a6.pt",
        }
}


class PARSeq(_PARSeq,nn.Module):
    """
    Implements a PARSeq architecture as described in `"Scene Text Recognition 
    with Permuted Autoregressive Sequence Models" 
    <https://arxiv.org/pdf/2207.06966>`_.
    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """
    def __init__(
        self,
        feature_extractor,
        vocab: str,
        max_length: int = 25,
        input_shape: Tuple[int, int, int] = (3, 32, 128),  # different from paper
        exportable: bool = False,
        cfg: Dict[str, Any] = default_cfgs,
    ) -> None:
    
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg

        self.max_length = max_length + 3  # Add 1 step for EOS, 1 for SOS, 1 for PAD

        self.feat_extractor = feature_extractor

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        
        features = self.feat_extractor(x)

        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        features = features[:, : self.max_length + 1]  # add 1 for unused cls token (ViT)

        B, N, E = features.size()
        #features = features.reshape(B * N, E)
        logits = features
        decoded_features = logits  # remove cls_token

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Post-process boxes
            logits = decoded_features
            
            pred = logits.softmax(-1)

            label, confidence = self.feat_extractor.tokenizer.decode(pred)
            
            out_idxs = logits.argmax(-1)     
            probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
            # Take the minimum confidence of the sequence
            probs = probs.min(dim=1).values.detach().cpu()

            # Manual decoding
        
            out['preds']=  list(zip(label, probs.numpy().tolist()))
            
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
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of steps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token (sos disappear in shift!)
        seq_len = seq_len + 1
        # Compute loss: don't forget to shift gt! Otherwise the model learns to output the gt[t-1]!
        # The "masked" first gt char is <sos>. Delete last logit of the model output.
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:], reduction="none")
        # Compute mask, remove 1 timestep here as well
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()
        

def _parseq(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = False,  # NOTE: training from scratch without a pretrained backbone works better
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
):
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Feature extractor
    #feat_extractor = IntermediateLayerGetter(
    #    backbone_fn(pretrained_backbone, input_shape=_cfg["input_shape"]),  # type: ignore[call-arg]
    #    {layer: "features"},
    #)
    feat_extractor = backbone_fn(_cfg)
    # Build the model
    model = PARSeq(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params_local(model, default_cfgs[arch]["url"])

    return model

def parseq(pretrained: bool = False, **kwargs: Any):   
    """
    parseq as described in `"Scene Text Recognition with Permuted Autoregressive Sequence Models"
    <https://arxiv.org/pdf/2207.06966>`_.
    >>> import torch
    >>> from doctr.models import parseq
    >>> model = parseq(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)
    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
    Returns:
        text recognition architecture
    """
    return _parseq(
        "parseq",
        pretrained,
        parseq_model,
        "1",
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )

    



        
        

