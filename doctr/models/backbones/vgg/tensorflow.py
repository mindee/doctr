# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple, Dict, Any

from ...utils import conv_sequence, load_pretrained_params


__all__ = ['VGG', 'vgg16_bn']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'vgg16_bn': {'num_blocks': (2, 2, 3, 3, 3), 'planes': (64, 128, 256, 512, 512),
                 'rect_pools': (False, False, True, True, True),
                 'url': None},
}


class VGG(Sequential):
    """Implements the VGG architecture from `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        num_blocks: number of convolutional block in each stage
        planes: number of output channels in each stage
        rect_pools: whether pooling square kernels should be replace with rectangular ones
        input_shape: shapes of the input tensor
        include_top: whether the classifier head should be instantiated
    """
    def __init__(
        self,
        num_blocks: Tuple[int, int, int, int, int],
        planes: Tuple[int, int, int, int, int],
        rect_pools: Tuple[bool, bool, bool, bool, bool],
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        include_top: bool = False,
    ) -> None:

        _layers = []
        # Specify input_shape only for the first layer
        kwargs = {"input_shape": input_shape}
        for nb_blocks, out_chan, rect_pool in zip(num_blocks, planes, rect_pools):
            for _ in range(nb_blocks):
                _layers.extend(conv_sequence(out_chan, 'relu', True, kernel_size=3, **kwargs))  # type: ignore[arg-type]
                kwargs = {}
            _layers.append(layers.MaxPooling2D((2, 1 if rect_pool else 2)))
        super().__init__(_layers)


def _vgg(arch: str, pretrained: bool, **kwargs: Any) -> VGG:

    # Build the model
    model = VGG(default_cfgs[arch]['num_blocks'], default_cfgs[arch]['planes'],
                default_cfgs[arch]['rect_pools'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def vgg16_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_, modified by adding batch normalization.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import vgg16_bn
        >>> model = vgg16_bn(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 224, 224, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        VGG feature extractor
    """

    return _vgg('vgg16_bn', pretrained, **kwargs)
