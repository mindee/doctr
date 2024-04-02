# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List, Optional

import numpy as np
import onnxruntime as ort

from doctr.utils.data import download_from_url


class _BasePredictor:
    """
    Base class for all predictors

    Args:
    ----
        batch_size: the batch size to use
        url: the url to use
        model_path: the path to the model to use
        **kwargs: additional arguments to be passed to `download_from_url`
    """

    def __init__(self, batch_size: int, url: Optional[str] = None, model_path: Optional[str] = None, **kwargs) -> None:
        self.batch_size = batch_size

        self.onnx_model_path = self._init_model(url, model_path, **kwargs)
        self.session = ort.InferenceSession(
            self.onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self._inputs: List[np.ndarray] = []
        self._results: List[Any] = []

    def _init_model(self, url: Optional[str] = None, model_path: Optional[str] = None, **kwargs: Any) -> str:
        """
        Download the model from the given url if needed

        Args:
        ----
            url: the url to use
            model_path: the path to the model to use
            **kwargs: additional arguments to be passed to `download_from_url`

        Returns:
        -------
            str: the path to the model
        """
        if not url and not model_path:
            raise ValueError("You must provide either a url or a model_path")
        return model_path if model_path else str(download_from_url(url, cache_subdir="models", **kwargs))  # type: ignore[arg-type]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image

        Args:
        ----
            img: the input image to preprocess

        Returns:
        -------
            np.ndarray: the preprocessed image
        """
        raise NotImplementedError

    def postprocess(self, output: List[np.ndarray], input_images: List[List[np.ndarray]]) -> Any:
        """
        Postprocess the model output

        Args:
        ----
            output: the model output to postprocess
            input_images: the input images used to generate the output

        Returns:
        -------
            Any: the postprocessed output
        """
        raise NotImplementedError

    def __call__(self, inputs: List[np.ndarray]) -> Any:
        """
        Call the model on the given inputs

        Args:
        ----
            inputs: the inputs to use

        Returns:
        -------
            Any: the postprocessed output
        """
        self._inputs = inputs
        model_inputs = self.session.get_inputs()

        # Get input shape to reuse it for postprocessing
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        batched_inputs = [inputs[i : i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        processed_batches = [
            np.array([self.preprocess(img) for img in batch], dtype=np.float32) for batch in batched_inputs
        ]

        outputs = [self.session.run(None, {model_inputs[0].name: batch}) for batch in processed_batches]
        return self.postprocess(outputs, batched_inputs)