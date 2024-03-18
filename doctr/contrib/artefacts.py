from typing import Any, Dict, List

import cv2
import numpy as np
import onnxruntime as ort
from altair import Optional

from doctr.models.preprocessor import PreProcessor
from doctr.utils.data import download_from_url

__all__ = ["ArtefactDetector"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "yolov8_artefact": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "labels": ["bar_code", "qr_code", "logo", "photo"],
        "url": None,
    },
}


class ArtefactDetector:
    """
    A class to detect artefacts in images

    Args:
    ----
        arch: the architecture to use
        batch_size: the batch size to use
        model_path: the path to the model to use
        labels: the labels to use
        mask_labels: the mask labels to use
        conf_threshold: the confidence threshold to use
        iou_threshold: the intersection over union threshold to use
        **kwargs: additional arguments to be passed to `doctr.models.preprocessor.PreProcessor`
    """

    def __init__(
        self,
        arch: str = "yolov8_artefact",
        batch_size: int = 2,
        model_path: Optional[str] = None,
        labels: Optional[List[str]] = default_cfgs["yolov8_artefact"]["labels"],
        mask_labels: Optional[List[str]] = None,
        conf_threshold: float = 0.9,
        iou_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.onnx_model = self._init_model(default_cfgs[arch]["url"], model_path)
        self.labels = labels or default_cfgs[arch]["labels"]
        self.mask_labels = mask_labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = ort.InferenceSession(
            self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.pre_processor = PreProcessor(
            output_size=(1024, 1024),
            batch_size=batch_size,
            mean=default_cfgs[arch]["mean"],
            std=default_cfgs[arch]["std"],
            **kwargs,
        )

    def _init_model(self, url: str, model_path: Optional[str] = None, **kwargs: Any):
        return model_path if model_path else download_from_url(url, cache_subdir="models", **kwargs)

    def _postprocess(self, output: np.ndarray, input_images: np.ndarray):
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # TODO: Finish all + Testing + Clean up + Documentation

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.conf_threshold:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        if self.mask_labels:
            pass
            # TODO: mask detected label in org image with black box
        return [
            {
                "boxes": [boxes[i] for i in indices],
                "scores": [scores[i] for i in indices],
                "labels": [self.labels[class_ids[i]] for i in indices],
            }
        ]

    def __call__(self, inputs: np.ndarray) -> Any:
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        processed_batches = self.pre_processor(inputs)
        outputs = [self.session.run(None, {model_inputs[0].name: batch}) for batch in processed_batches]
        return self._postprocess(outputs, inputs)
