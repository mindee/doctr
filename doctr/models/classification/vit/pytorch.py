# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import EncoderBlock
from doctr.models.modules.vision_transformer import PatchEmbedding

from ...utils.pytorch import load_pretrained_params

__all__ = ["vit"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "vit": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
}


class VisionTransformer(nn.Module):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
        img_size: size of the input image
        patch_size: size of the patches to be extracted from the input
        channels: number of channels in the input image
        d_model: dimension of the transformer layers
        num_layers: number of transformer layers
        num_heads: number of attention heads
        dropout: dropout rate
        num_classes: number of output classes
        include_top: whether the classifier head should be instantiated
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        channels: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.include_top = include_top

        self.patch_embedding = PatchEmbedding(img_size, patch_size, channels, d_model)
        self.encoder = EncoderBlock(num_layers, num_heads, d_model, dropout, use_gelu=True)

        if self.include_top:
            self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embeddings = self.patch_embedding(x)
        encoded = self.encoder(embeddings)

        if self.include_top:
            # (batch_size, num_classes) cls token
            return self.head(encoded[:, 0])

        return encoded


def _vit(
    arch: str,
    pretrained: bool,
    img_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    channels: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float = 0.0,
    include_top: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> VisionTransformer:

    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        channels=channels,
        d_model=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        cfg=_cfg,
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def vit(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    >>> import torch
    >>> from doctr.models import vit
    >>> model = vit(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A feature extractor model
    """

    return _vit(
        "vit",
        pretrained,
        img_size=(32, 32),
        patch_size=(4, 4),
        channels=3,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
