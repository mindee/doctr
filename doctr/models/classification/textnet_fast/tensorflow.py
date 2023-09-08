# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from doctr.datasets import VOCABS
from doctr.models.modules.layers.tensorflow import RepConvLayer
from doctr.models.utils.tensorflow import (
    conv_sequence,
    fuse_module,
    rep_model_convert,
    rep_model_convert_deploy,
    rep_model_unconvert,
    unfuse_module,
)

from ...utils import load_pretrained_params

__all__ = ["textnetfast_tiny", "textnetfast_small", "textnetfast_base"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "textnetfast_tiny": {
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "textnetfast_small": {
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "textnetfast_base": {
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
}


class TextNetFast(tf.keras.Sequential):
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    Args:
        stage1 (Dict[str, Union[int, List[int]]]): Configuration for stage 1
        stage2 (Dict[str, Union[int, List[int]]]): Configuration for stage 2
        stage3 (Dict[str, Union[int, List[int]]]): Configuration for stage 3
        stage4 (Dict[str, Union[int, List[int]]]): Configuration for stage 4
        include_top (bool, optional): Whether to include the classifier head. Defaults to True.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        cfg (Optional[Dict[str, Any]], optional): Additional configuration. Defaults to None.
    """

    def __init__(
        self,
        stage1: List[Dict[str, Union[int, List[int]]]],
        stage2: List[Dict[str, Union[int, List[int]]]],
        stage3: List[Dict[str, Union[int, List[int]]]],
        stage4: List[Dict[str, Union[int, List[int]]]],
        include_top: bool = True,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, int, int]] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        first_conv = tf.keras.Sequential(
            conv_sequence(out_channels=64, activation="relu", bn=True, kernel_size=3, strides=2)
        )
        _layers = [first_conv]

        for stage in [stage1, stage2, stage3, stage4]:
            stage_ = tf.keras.Sequential([RepConvLayer(**params) for params in stage])
            _layers.extend([stage_])

        if include_top:
            classif_block = [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(num_classes, activation="softmax", use_bias=True, kernel_initializer="he_normal"),
            ]
            classif_block_ = tf.keras.Sequential(classif_block)
            _layers.append(classif_block_)

        super().__init__(_layers)
        self.cfg = cfg

    def eval(self, mode=False):
        self = rep_model_convert(self)
        self = fuse_module(self)
        self.trainable = mode
        return self

    def train(self, mode=True):
        self = unfuse_module(self)
        self = rep_model_unconvert(self)
        self.trainable = mode
        return self

    def test(self, mode=False):
        self = rep_model_convert_deploy(self)
        self = fuse_module(self)
        self.trainable = mode
        return self


def _textnetfast(
    arch: str,
    pretrained: bool,
    arch_fn,
    **kwargs: Any,
) -> TextNetFast:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    kwargs.pop("classes")

    # Build the model
    model = arch_fn(**kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def textnetfast_tiny(pretrained: bool = False, **kwargs: Any) -> TextNetFast:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnetfast_tiny
    >>> model = textnetfast_tiny(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNet model
    """

    return _textnetfast(
        "textnetfast_tiny",
        pretrained,
        TextNetFast,
        stage1=[
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
        ],
        stage2=[
            {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
        ],
        stage3=[
            {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
        ],
        stage4=[
            {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 3], "stride": 1},
        ],
        **kwargs,
    )


def textnetfast_small(pretrained: bool = False, **kwargs: Any) -> TextNetFast:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnetfast_small
    >>> model = textnetfast_small(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNetFast model
    """

    return _textnetfast(
        "textnetfast_small",
        pretrained,
        TextNetFast,
        stage1=[
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2},
        ],
        stage2=[
            {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
        ],
        stage3=[
            {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
        ],
        stage4=[
            {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
        ],
        **kwargs,
    )


def textnetfast_base(pretrained: bool = False, **kwargs: Any) -> TextNetFast:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnetfast_base
    >>> model = textnetfast_base(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNetFast model
    """

    return _textnetfast(
        "textnetfast_base",
        pretrained,
        TextNetFast,
        stage1=[
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
        ],
        stage2=[
            {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
        ],
        stage3=[
            {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
        ],
        stage4=[
            {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
        ],
        **kwargs,
    )
