# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from doctr.datasets import VOCABS

from ...utils import conv_sequence, load_pretrained_params

__all__ = ["VGG", "vgg16_bn_r"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "vgg16_bn_r": {
        "mean": (0.5, 0.5, 0.5),
        "std": (1.0, 1.0, 1.0),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/vgg16_bn_r-c5836cea.zip&src=0",
    },
}


class VGG(Sequential):
    """Implements the VGG architecture from `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        num_blocks: number of convolutional block in each stage
        planes: number of output channels in each stage
        rect_pools: whether pooling square kernels should be replace with rectangular ones
        include_top: whether the classifier head should be instantiated
        num_classes: number of output classes
        input_shape: shapes of the input tensor
    """

    def __init__(
        self,
        num_blocks: List[int],
        planes: List[int],
        rect_pools: List[bool],
        include_top: bool = False,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, int, int]] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        _layers = []
        # Specify input_shape only for the first layer
        kwargs = {"input_shape": input_shape}
        for nb_blocks, out_chan, rect_pool in zip(num_blocks, planes, rect_pools):
            for _ in range(nb_blocks):
                _layers.extend(conv_sequence(out_chan, "relu", True, kernel_size=3, **kwargs))  # type: ignore[arg-type]
                kwargs = {}
            _layers.append(layers.MaxPooling2D((2, 1 if rect_pool else 2)))

        if include_top:
            _layers.extend([layers.GlobalAveragePooling2D(), layers.Dense(num_classes)])
        super().__init__(_layers)
        self.cfg = cfg


def _vgg(
    arch: str, pretrained: bool, num_blocks: List[int], planes: List[int], rect_pools: List[bool], **kwargs: Any
) -> VGG:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    kwargs.pop("classes")

    # Build the model
    model = VGG(num_blocks, planes, rect_pools, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def vgg16_bn_r(pretrained: bool = False, **kwargs: Any) -> VGG:
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_, modified by adding batch normalization, rectangular pooling and a simpler
    classification head.

    >>> import tensorflow as tf
    >>> from doctr.models import vgg16_bn_r
    >>> model = vgg16_bn_r(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        VGG feature extractor
    """

    return _vgg(
        "vgg16_bn_r", pretrained, [2, 2, 3, 3, 3], [64, 128, 256, 512, 512], [False, False, True, True, True], **kwargs
    )
