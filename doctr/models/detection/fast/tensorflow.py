# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from doctr.file_utils import CLASS_NAME
from doctr.models.utils import IntermediateLayerGetter, _bf16_to_float32, load_pretrained_params
from doctr.utils.repr import NestedObject

from ...classification import textnet_base, textnet_small, textnet_tiny
from .base import _FAST, FASTPostProcessor

__all__ = ["FAST", "fast_tiny", "fast_small", "fast_base"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
    "fast_small": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
    "fast_base": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": None,
    },
}



class FAST(_FAST, keras.Model, NestedObject):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.

    Args:
    ----
        feature extractor: the backbone serving as feature extractor
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    _children_names: List[str] = ["in_module", "feat_extractor", "postprocessor"]

    def __init__(
        self,
        feature_extractor: IntermediateLayerGetter,
        bin_thresh: float = 0.3,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
        class_names: List[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg

        self.feat_extractor = feature_extractor
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages



        self.postprocessor = FASTPostProcessor(assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh)

    def compute_loss(
        self,
        out_map: tf.Tensor,
        target: List[Dict[str, np.ndarray]],
    ) -> tf.Tensor:
        # TODO: same as pytorch

        return tf.constant(0.0)

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[Dict[str, np.ndarray]]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        feat_maps = self.feat_extractor(x, **kwargs)

        out: Dict[str, tf.Tensor] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(tf.math.sigmoid(logits))

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes (keep only text predictions)
            out["preds"] = [dict(zip(self.class_names, preds)) for preds in self.postprocessor(prob_map.numpy())]

        if target is not None:
            thresh_map = self.threshold_head(feat_concat, **kwargs)
            loss = self.compute_loss(logits, thresh_map, target)
            out["loss"] = loss

        return out


def _fast(
    arch: str,
    pretrained: bool,
    backbone_fn,
    feat_layers: List[str],
    pretrained_backbone: bool = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any,
) -> FAST:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    if not kwargs.get("class_names", None):
        kwargs["class_names"] = _cfg.get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(
            input_shape=_cfg["input_shape"],
            include_top=False,
            pretrained=pretrained_backbone,
        ),
        feat_layers,
    )

    # Build the model
    model = FAST(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg["url"])

    return model





def fast_tiny(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a tiny TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_tiny
    >>> model = fast_tiny(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_tiny",
        pretrained,
        textnet_tiny,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )


def fast_small(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a small TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_small
    >>> model = fast_small(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_small",
        pretrained,
        textnet_small,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )


def fast_base(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a base TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_base
    >>> model = fast_base(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _fast(
        "fast_base",
        pretrained,
        textnet_base,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )
