# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import cv2
import json
import os
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import tensorflow as tf
from typing import Union, List, Tuple, Optional, Any, Dict

MIN_SIZE_BOX = 5

def show_infer_pdf(
    filepaths: List[str],
    infer_dict: Dict[str, Dict[str, Any]]
) -> None:
    """
    Show pdf pages with detected bounding boxes
    :param path_to_pdfs: list of pdfs to show
    :param infer_dict: inference output for the images :
        img_name -> dict(boxes, scores -> list of boxes, list of scores)

    """
    documents_imgs, documents_names = prepare_pdf_documents(filepaths)
    for document_img, document_name in zip(documents_imgs, documents_names):
        for page_img, page_name in zip(document_img, document_name):
            if page_name in infer_dict.keys():
                boxes = infer_dict[page_name]["infer_boxes"]
                page_img = page_img.astype(np.uint8)
                for box in boxes:
                    cv2.drawContours(page_img, [np.array(box)], -1, (0, 255, 0), 5)
                cv2.namedWindow(page_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(page_name, 600, 600)
                cv2.imshow(page_name, page_img)
                cv2.waitKey(0)
            else:
                print("inference has not been performed on this file!")


def box_score(
    pred: np.ndarray,
    _box: np.ndarray
) -> float:
    """
    Compute the confidence score for a box : mean between p_map values on the box
    :param pred: p_map (output of the model)
    :param _box: box
    """

    h, w = pred.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

    return cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def polygon_to_box(
    points: np.ndarray,
    unclip_ratio: Union[float, int] = 1.5,
) -> Tuple[np.ndarray, int, int]:
    """
    Expand polygon (box) by a factor unclip_ratio
    :param poly: polygon to unclip
    :param unclip_ratio: dilatation ratio
    """

    poly = Polygon(points)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded_points = np.array(offset.Execute(distance))
    x, y, w, h = cv2.boundingRect(expanded_points)
    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return box, h, w


def bitmap_to_boxes(
    pred: np.ndarray,
    bitmap: np.ndarray,
    dest_width: int,
    dest_height: int,
    max_candidates: int = 100,
    box_thresh: float = 0.7
) -> Tuple[List[List[List[int]]], List[float]]:
    """
    predict scores and boxes from p_map and bin_map
    :param pred : probability map (np array)
    :param bitmap: bin_map (generated from p_map with a constant threshold at inference time), np array
    :param dest_width, dest_height: dims of the mask to output to scale the boxes
    :param max candidates: max boxes to look for in a document page
    :param box_thresh: min score to consider a box
    """

    height, width = bitmap.shape[:2]
    boxes, scores = [], []

    contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        box, h, w = polygon_to_box(points, unclip_ratio=2.0)
        if h < MIN_SIZE_BOX or w < MIN_SIZE_BOX:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


def postprocess_img_dbnet(
    pred: np.ndarray,
    heights: List[int],
    widths: List[int]
) -> Tuple[List[List[List[List[int]]]], List[List[float]]]:
    """
    postprocessing function which convert the model output to boxes and scores
    :param pred: raw output of the differentiable binarization model
    :param heights: images heights
    :param widths: images widths
    """

    p = tf.squeeze(pred, axis=-1)  # remove last dim
    bitmap = tf.cast(p > 0.3, tf.float32)

    p = tf.unstack(p, axis=0)
    bitmap = tf.unstack(bitmap, axis=0)

    boxes_batch, scores_batch = [], []

    for p_, bitmap_, w, h in zip(p, bitmap, widths, heights):
        p_ = p_.numpy()
        bitmap_ = bitmap_.numpy()
        boxes, scores = bitmap_to_boxes(p_, bitmap_, w, h, box_thresh=0.5)
        boxes_batch.append(boxes)
        scores_batch.append(scores)

    return boxes_batch, scores_batch
