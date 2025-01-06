# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import numpy as np
import tensorflow as tf

from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor

DET_ARCHS = [
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]
RECO_ARCHS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]


def load_predictor(
    det_arch: str,
    reco_arch: str,
    assume_straight_pages: bool,
    straighten_pages: bool,
    export_as_straight_boxes: bool,
    disable_page_orientation: bool,
    disable_crop_orientation: bool,
    bin_thresh: float,
    box_thresh: float,
    device: tf.device,
) -> OCRPredictor:
    """Load a predictor from doctr.models

    Args:
        det_arch: detection architecture
        reco_arch: recognition architecture
        assume_straight_pages: whether to assume straight pages or not
        straighten_pages: whether to straighten rotated pages or not
        export_as_straight_boxes: whether to export boxes as straight or not
        disable_page_orientation: whether to disable page orientation or not
        disable_crop_orientation: whether to disable crop orientation or not
        bin_thresh: binarization threshold for the segmentation map
        box_thresh: threshold for the detection boxes
        device: tf.device, the device to load the predictor on

    Returns:
        instance of OCRPredictor
    """
    with device:
        predictor = ocr_predictor(
            det_arch,
            reco_arch,
            pretrained=True,
            assume_straight_pages=assume_straight_pages,
            straighten_pages=straighten_pages,
            export_as_straight_boxes=export_as_straight_boxes,
            detect_orientation=not assume_straight_pages,
            disable_page_orientation=disable_page_orientation,
            disable_crop_orientation=disable_crop_orientation,
        )
        predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
        predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
    return predictor


def forward_image(predictor: OCRPredictor, image: np.ndarray, device: tf.device) -> np.ndarray:
    """Forward an image through the predictor

    Args:
        predictor: instance of OCRPredictor
        image: image to process as numpy array
        device: tf.device, the device to process the image on

    Returns:
        segmentation map
    """
    with device:
        processed_batches = predictor.det_predictor.pre_processor([image])
        out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
        seg_map = out["out_map"]

    with tf.device("/cpu:0"):
        seg_map = tf.identity(seg_map).numpy()

    return seg_map
