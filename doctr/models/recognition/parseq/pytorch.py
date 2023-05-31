# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS


from ...utils.pytorch import load_pretrained_params_local
from .base import _PARSeq, _PARSeqPostProcessor
from ...classification.parseq.base import *
from ...classification import vit_s
__all__ = ["PARSeq", "parseq"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "charset_train": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        "charset_test": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
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
        "vocab": VOCABS["latin"],
        "input_shape": (3, 32, 128),
        "classes": list(VOCABS["latin"]),
        "url": "/home/nikkokks/Desktop/github/parseq-bb5792a6.pt",
        }
}


class PARSeq(_PARSeq, nn.Module):
    """Implements a PARSeq architecture as described in `"Scene Text Recognition
    with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        embedding_units: int,
        max_length: int = 25,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.charset_train=cfg['charset_train']
        self.charset_test=cfg['charset_test']
        self.batch_size=cfg['batch_size']
        self.lr=cfg['lr']
        self.warmup_pct=cfg['warmup_pct']
        self.weight_decay=cfg['weight_decay']
        self.embed_dim=cfg['embed_dim']
        self.dec_num_heads=cfg['dec_num_heads']
        self.dec_mlp_ratio=cfg['dec_mlp_ratio']
        self.dec_depth=cfg['dec_depth']
        self.perm_num=cfg['perm_num']
        self.perm_forward=cfg['perm_forward']
        self.perm_mirrored=cfg['perm_mirrored']
        self.decode_ar=cfg['decode_ar']
        self.refine_iters=cfg['refine_iters']
        self.dropout=cfg['dropout']
        self._device='cpu'
        self.vocab = vocab
        self.bos_id=len(self.vocab)+1
        self.pad_id=len(self.vocab)+2
        self.eos_id=0 

        self.exportable = exportable
        self.cfg = cfg
        self.max_label_length = max_length +3  # Add 1 step for EOS, 1 for SOS, 1 for PAD
        
        super().__init__()

        self.encoder = feature_extractor
        #self.encoder = Encoder(self.img_size, self.patch_size, embed_dim=self.embed_dim, depth=self.enc_depth, num_heads=self.enc_num_heads,
        #                       mlp_ratio=self.enc_mlp_ratio)
        decoder_layer = DecoderLayer(self.embed_dim, self.dec_num_heads, self.embed_dim * self.dec_mlp_ratio, self.dropout)
        self.decoder = Decoder(decoder_layer, num_layers=self.dec_depth, norm=nn.LayerNorm(self.embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = self.perm_num // 2 if self.perm_mirrored else self.perm_num
        
        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(self.embed_dim, len(self.vocab)+1)
        self.text_embed = TokenEmbedding(len(self.vocab)+3, self.embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_label_length -2, self.embed_dim))
        
        self.dropout = nn.Dropout(p=self.dropout)
        # Encoder has its own init.

        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)
        
        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forwards(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length - 3 if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)['features']
        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                print(pos_queries.shape,query_mask.shape)
                print(tgt_in[:,:j].shape,memory.shape, tgt_mask[:j, :j].shape, pos_queries[:, i:j].shape,query_mask[i:j, :j].shape)
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        # for the moment comment this line
        features = self.forwards(x) # (batch_size, patches_seqlen, d_model)

        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        decoded_features = features[:, : self.max_label_length + 1] 

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


class PARSeqPostProcessor(_PARSeqPostProcessor):
    """Post processor for PARSeq architecture

    Args:
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
            for encoded_seq in out_idxs.cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().tolist()))


def _parseq(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> PARSeq:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone, input_shape=_cfg["input_shape"]),  # type: ignore[call-arg]
        {layer: "features"},)

    # Build the model
    model = PARSeq(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        #load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)
        load_pretrained_params_local(model, default_cfgs[arch]["url"])
    return model


def parseq(pretrained: bool = False, **kwargs: Any) -> PARSeq:
    """PARSeq architecture from
    `"Scene Text Recognition with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

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
        vit_s,
        "1",
        embedding_units=384,
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
