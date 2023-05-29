# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch import Tensor

import numpy as np

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import EncoderBlock

from ...utils.pytorch import load_pretrained_params_local

from .modules import *
from functools import partial
from .visiontransformers import named_apply,init_weights

from .base import CrossEntropySystem

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

class PARSeq(CrossEntropySystem):

    def __init__(self, cfg, **kwargs):

        self.charset_train=cfg['charset_train']
        self.charset_test=cfg['charset_test']
        self.max_label_length=cfg['max_label_length']
        self.batch_size=cfg['batch_size']
        self.lr=cfg['lr']
        self.warmup_pct=cfg['warmup_pct']
        self.weight_decay=cfg['weight_decay']
        self.img_size=cfg['img_size']
        self.patch_size=cfg['patch_size']
        self.embed_dim=cfg['embed_dim']
        self.enc_num_heads=cfg['enc_num_heads']
        self.enc_mlp_ratio=cfg['enc_mlp_ratio']
        self.enc_depth=cfg['enc_depth']
        self.dec_num_heads=cfg['dec_num_heads']
        self.dec_mlp_ratio=cfg['dec_mlp_ratio']
        self.dec_depth=cfg['dec_depth']
        self.perm_num=cfg['perm_num']
        self.perm_forward=cfg['perm_forward']
        self.perm_mirrored=cfg['perm_mirrored']
        self.decode_ar=cfg['decode_ar']
        self.refine_iters=cfg['refine_iters']
        self.dropout=cfg['dropout']
        
        self.cfg = cfg
     
        super().__init__(self.charset_train, self.charset_test, self.batch_size, self.lr, self.warmup_pct, self.weight_decay)

        self.encoder = Encoder(self.img_size, self.patch_size, embed_dim=self.embed_dim, depth=self.enc_depth, num_heads=self.enc_num_heads,
                               mlp_ratio=self.enc_mlp_ratio)
        decoder_layer = DecoderLayer(self.embed_dim, self.dec_num_heads, self.embed_dim * self.dec_mlp_ratio, self.dropout)
        self.decoder = Decoder(decoder_layer, num_layers=self.dec_depth, norm=nn.LayerNorm(self.embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = self.perm_num // 2 if self.perm_mirrored else self.perm_num

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(self.embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), self.embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_label_length + 1, self.embed_dim))
        self.dropout = nn.Dropout(p=self.dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

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

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

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

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx):
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        self.log('loss', loss)
        return loss

def _parseq(
    arch: str,
    pretrained: bool,
    **kwargs: Any,
):
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = kwargs["input_shape"]

    # Build the model
    model = PARSeq(cfg=_cfg, **kwargs)
    _cfg["num_classes"] = len(model.tokenizer)-2
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        load_pretrained_params_local(model, default_cfgs[arch]["url"])
        
    return model

def parseq(pretrained: bool = False, **kwargs: Any):
    """VisionTransformer-S architecture
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_. Patches: (H, W) -> (H/8, W/8)

    NOTE: unofficial config used in ViTSTR and ParSeq

    >>> import torch
    >>> from doctr.models import vit_s
    >>> model = vit_s(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A feature extractor model
    """

    return _parseq(
        "parseq",
        pretrained,
        **kwargs,
    )
