# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import numpy as np

from doctr.file_utils import requires_package
from doctr.utils.data import download_from_url


class _BasePredictor:
    """
    Base class for all predictors

    Args:
        batch_size: the batch size to use
        url: the url to use to download a model if needed
        model_path: the path to the model to use
        **kwargs: additional arguments to be passed to `download_from_url`
    """

    def __init__(self, batch_size: int, url: str | None = None, model_path: str | None = None, **kwargs) -> None:
        self.batch_size = batch_size
        self.session = self._init_model(url, model_path, **kwargs)

        self._inputs: list[np.ndarray] = []
        self._results: list[Any] = []

    def _init_model(self, url: str | None = None, model_path: str | None = None, **kwargs: Any) -> Any:
        """
        Download the model from the given url if needed

        Args:
            url: the url to use
            model_path: the path to the model to use
            **kwargs: additional arguments to be passed to `download_from_url`

        Returns:
            Any: the ONNX loaded model
        """
        requires_package("onnxruntime", "`.contrib` module requires `onnxruntime` to be installed.")
        import onnxruntime as ort

        if not url and not model_path:
            raise ValueError("You must provide either a url or a model_path")
        onnx_model_path = model_path if model_path else str(download_from_url(url, cache_subdir="models", **kwargs))  # type: ignore[arg-type]
        return ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image

        Args:
            img: the input image to preprocess

        Returns:
            np.ndarray: the preprocessed image
        """
        raise NotImplementedError

    def postprocess(self, output: list[np.ndarray], input_images: list[list[np.ndarray]]) -> Any:
        """
        Postprocess the model output

        Args:
            output: the model output to postprocess
            input_images: the input images used to generate the output

        Returns:
            Any: the postprocessed output
        """
        raise NotImplementedError

    def __call__(self, inputs: list[np.ndarray]) -> Any:
        """
        Call the model on the given inputs

        Args:
            inputs: the inputs to use

        Returns:
            Any: the postprocessed output
        """
        self._inputs = inputs
        model_inputs = self.session.get_inputs()

        batched_inputs = [inputs[i : i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        processed_batches = [
            np.array([self.preprocess(img) for img in batch], dtype=np.float32) for batch in batched_inputs
        ]

        outputs = [self.session.run(None, {model_inputs[0].name: batch}) for batch in processed_batches]
        return self.postprocess(outputs, batched_inputs)
