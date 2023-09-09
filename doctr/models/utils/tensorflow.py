# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow.keras import Model, layers

from doctr.models.modules.layers.tensorflow import RepConvLayer
from doctr.utils.data import download_from_url

logging.getLogger("tensorflow").setLevel(logging.DEBUG)


__all__ = ["load_pretrained_params", "conv_sequence", "IntermediateLayerGetter", "export_model_to_onnx", "_copy_tensor"]


def _copy_tensor(x: tf.Tensor) -> tf.Tensor:
    return tf.identity(x)


def load_pretrained_params(
    model: Model,
    url: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    overwrite: bool = False,
    internal_name: str = "weights",
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the keras model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        overwrite: should the zip extraction be enforced if the archive has already been extracted
        internal_name: name of the ckpt files
    """

    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        archive_path = download_from_url(url, hash_prefix=hash_prefix, cache_subdir="models", **kwargs)

        # Unzip the archive
        params_path = archive_path.parent.joinpath(archive_path.stem)
        if not params_path.is_dir() or overwrite:
            with ZipFile(archive_path, "r") as f:
                f.extractall(path=params_path)

        # Load weights
        model.load_weights(f"{params_path}{os.sep}{internal_name}")


def conv_sequence(
    out_channels: int,
    activation: Optional[Union[str, Callable]] = None,
    bn: bool = False,
    padding: str = "same",
    kernel_initializer: str = "he_normal",
    **kwargs: Any,
) -> List[layers.Layer]:
    """Builds a convolutional-based layer sequence

    >>> from doctr.models.utils import conv_sequence
    >>> module = Sequential(conv_sequence(32, 'relu', True, kernel_size=3, input_shape=[224, 224, 3]))

    Args:
        out_channels: number of output channels
        activation: activation to be used (default: no activation)
        bn: should a batch normalization layer be added
        padding: padding scheme
        kernel_initializer: kernel initializer

    Returns:
        list of layers
    """
    # No bias before Batch norm
    kwargs["use_bias"] = kwargs.get("use_bias", not bn)
    # Add activation directly to the conv if there is no BN
    kwargs["activation"] = activation if not bn else None
    conv_seq = [layers.Conv2D(out_channels, padding=padding, kernel_initializer=kernel_initializer, **kwargs)]

    if bn:
        conv_seq.append(layers.BatchNormalization())

    if (isinstance(activation, str) or callable(activation)) and bn:
        # Activation function can either be a string or a function ('relu' or tf.nn.relu)
        conv_seq.append(layers.Activation(activation))

    return conv_seq


class IntermediateLayerGetter(Model):
    """Implements an intermediate layer getter

    >>> from tensorflow.keras.applications import ResNet50
    >>> from doctr.models import IntermediateLayerGetter
    >>> target_layers = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    >>> feat_extractor = IntermediateLayerGetter(ResNet50(include_top=False, pooling=False), target_layers)

    Args:
        model: the model to extract feature maps from
        layer_names: the list of layers to retrieve the feature map from
    """

    def __init__(self, model: Model, layer_names: List[str]) -> None:
        intermediate_fmaps = [model.get_layer(layer_name).get_output_at(0) for layer_name in layer_names]
        super().__init__(model.input, outputs=intermediate_fmaps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def export_model_to_onnx(
    model: Model, model_name: str, dummy_input: List[tf.TensorSpec], **kwargs: Any
) -> Tuple[str, List[str]]:
    """Export model to ONNX format.

    >>> import tensorflow as tf
    >>> from doctr.models.classification import resnet18
    >>> from doctr.models.utils import export_classification_model_to_onnx
    >>> model = resnet18(pretrained=True, include_top=True)
    >>> export_model_to_onnx(model, "my_model",
    >>> dummy_input=[tf.TensorSpec([None, 32, 32, 3], tf.float32, name="input")])

    Args:
        model: the keras model to be exported
        model_name: the name for the exported model
        dummy_input: the dummy input to the model
        kwargs: additional arguments to be passed to tf2onnx

    Returns:
        the path to the exported model and a list with the output layer names
    """
    large_model = kwargs.get("large_model", False)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=dummy_input,
        output_path=f"{model_name}.zip" if large_model else f"{model_name}.onnx",
        **kwargs,
    )
    # Get the output layer names
    output = [n.name for n in model_proto.graph.output]

    # models which are too large (weights > 2GB while converting to ONNX) needs to be handled
    # about an external tensor storage where the graph and weights are seperatly stored in a archive
    if large_model:
        logging.info(f"Model exported to {model_name}.zip")
        return f"{model_name}.zip", output

    logging.info(f"Model exported to {model_name}.zip")
    return f"{model_name}.onnx", output


def rep_model_convert(model):
    for layer in model.layers:
        if hasattr(layer, "switch_to_test"):
            layer.switch_to_test()
    return model


def rep_model_unconvert(model):
    for layer in model.layers:
        if hasattr(layer, "switch_to_train"):
            layer.switch_to_train()
    return model


def rep_model_convert_deploy(model):
    for layer in model.layers:
        if hasattr(layer, "switch_to_deploy"):
            layer.switch_to_deploy()
    return model


def fuse_conv_bn(conv, bn):
    """During inference, the functionality of batch norm layers is turned off but
    only the mean and variance along channels are used, which exposes the opportunity
    to fuse it with the preceding conv layers to save computations and simplify
    network structures."""

    bn_weights, bn_biases, bn_running_mean, bn_running_var = bn.get_weights()
    weights = conv.get_weights()
    if len(weights) == 1:
        conv_weights = weights[0]
        conv_biases = np.zeros_like(bn_running_mean)
    else:
        conv_weights, conv_biases = conv.get_weights()
    epsilon = bn.epsilon
    scale_factor = bn_weights / np.sqrt(bn_running_var + epsilon)

    # Reshape the scale factor to match the convolutional weights shape
    scale_factor = scale_factor.reshape((1, 1, 1, -1))

    # Update convolutional weights and biases
    fused_conv_weights = conv_weights * scale_factor
    fused_conv_biases = (conv_biases - bn_running_mean) * scale_factor.flatten() + bn_biases

    conv.use_bias = True
    conv.build(input_shape=conv.input_shape)
    conv.set_weights([fused_conv_weights, fused_conv_biases])
    conv.old_weight, conv.old_biais = conv_weights, conv_biases
    return conv


def fuse_module(model):
    last_conv = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.experimental.SyncBatchNormalization)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fuse_conv = fuse_conv_bn(last_conv, layer)
            new_layer = tf.keras.layers.Lambda(lambda x: x)
            model.layers[i] = new_layer

            setattr(layer, layer.name, new_layer)
            print(last_conv.name)
            print(fuse_conv.name)
            print(layer.name)
            print(new_layer.name)
            print(model.layers[i].name)
            print(model.layers[i])
            print()
        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
        elif isinstance(layer, (tf.keras.Sequential, RepConvLayer)):
            fuse_module(layer)
    return model


def unfuse_conv_bn(conv, bn):
    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""
    conv.set_weights([conv.old_weight, conv.old_biais])
    return conv


def unfuse_module(model):
    last_conv = None

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Layer):
            pass
        else:
            continue

        if isinstance(layer, tf.keras.layers.Lambda):
            if last_conv is None:
                continue
            unfused_conv, unfused_bn = unfuse_conv_bn(last_conv, layer)

            # In TensorFlow, we can't modify the model in-place like in PyTorch,
            # so you would need to create a new model with the modified layers.
            # Here, you'd replace the last_conv layer with unfused_conv and
            # the current layer with unfused_bn.

        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
        else:
            # Recursive call for potentially nested layers (e.g., in case of a nested model)
            unfuse_module(layer)
    return layer
