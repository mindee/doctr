# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional

from torch import nn

from doctr.datasets import VOCABS
from doctr.models.modules import Encoder, PatchEmbedding

from ...utils import load_pretrained_params

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
    """Implements An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, as described in
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
        TODO !!
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # TODO

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # TODO
        return model


def _vit(
    arch: str,
    pretrained: bool,
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
    model = VisionTransformer(cfg=_cfg, **kwargs)
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
    >>> input_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A vit model
    """

    return _vit(
        "vit",
        pretrained,
        ignore_keys=["13.weight", "13.bias"],
        **kwargs,
    )  # TODO !!
