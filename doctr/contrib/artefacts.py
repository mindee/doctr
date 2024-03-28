from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from doctr.utils.data import download_from_url

__all__ = ["ArtefactDetector"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "yolov8_artefact": {
        "input_shape": (3, 1024, 1024),
        "labels": ["bar_code", "qr_code", "logo", "photo"],
        "url": "https://github.com/mindee/doctr/releases/download/v0.8.1/yolo_artefact-f9d66f14.onnx",
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
        conf_threshold: float = 0.9,
        iou_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.onnx_model = self._init_model(default_cfgs[arch]["url"], model_path, **kwargs)
        self.labels = labels or default_cfgs[arch]["labels"]
        self.input_shape = default_cfgs[arch]["input_shape"]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.session = ort.InferenceSession(
            self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self._inputs: List[np.ndarray] = []
        self._results: List[List[Dict[str, Any]]] = []

    def _init_model(self, url: str, model_path: Optional[str] = None, **kwargs: Any) -> str:
        return model_path if model_path else str(download_from_url(url, cache_subdir="models", **kwargs))

    def _postprocess(
        self, output: List[np.ndarray], input_images: List[List[np.ndarray]]
    ) -> List[List[Dict[str, Any]]]:
        results = []

        for batch in zip(output, input_images):
            for out, img in zip(batch[0], batch[1]):
                org_height, org_width = img.shape[:2]
                width_scale, height_scale = org_width / self.input_shape[2], org_height / self.input_shape[1]
                for res in out:
                    sample_results = []
                    for row in np.transpose(np.squeeze(res)):
                        classes_scores = row[4:]
                        max_score = np.amax(classes_scores)
                        if max_score >= self.conf_threshold:
                            class_id = np.argmax(classes_scores)
                            x, y, w, h = row[0], row[1], row[2], row[3]
                            # to rescaled xmin, ymin, xmax, ymax
                            xmin = int((x - w / 2) * width_scale)
                            ymin = int((y - h / 2) * height_scale)
                            xmax = int((x + w / 2) * width_scale)
                            ymax = int((y + h / 2) * height_scale)

                            sample_results.append({
                                "label": self.labels[class_id],
                                "confidence": max_score,
                                "box": [xmin, ymin, xmax, ymax],
                            })

                    # Filter out overlapping boxes
                    boxes = [res["box"] for res in sample_results]
                    scores = [res["confidence"] for res in sample_results]
                    keep_indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
                    sample_results = [sample_results[i] for i in keep_indices]

                    results.append(sample_results)

        self._results = results
        return results

    def show(self):
        # visualize the results with matplotlib
        if self._results and self._inputs:
            for img, res in zip(self._inputs, self._results):
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                for obj in res:
                    xmin, ymin, xmax, ymax = obj["box"]
                    label = obj["label"]
                    plt.text(xmin, ymin, f"{label} {obj['confidence']:.2f}", color="red")
                    plt.gca().add_patch(
                        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="red", linewidth=2)
                    )
                plt.show()

    def _prepare_img(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(cv2.resize(img, (self.input_shape[2], self.input_shape[1])), (2, 0, 1)) / 255.0

    def __call__(self, inputs: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        self._inputs = inputs
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        batched_inputs = [inputs[i : i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        processed_batches = [
            np.array([self._prepare_img(img) for img in batch], dtype=np.float32) for batch in batched_inputs
        ]

        outputs = [self.session.run(None, {model_inputs[0].name: batch}) for batch in processed_batches]
        return self._postprocess(outputs, batched_inputs)
