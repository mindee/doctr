# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import numpy as np
from typing import List, Tuple

__all__ = ['BarCodeDetector']


class BarCodeDetector:

    """ Implements a Bar-code detector.
    For now, only horizontal (or with a small angle) bar-codes are supported

    Args:
        min_size: minimum relative size of a barcode on the page
        canny_minval: lower bound for canny hysteresis
        canny_maxval: upper-bound for canny hysteresis
    """
    def __init__(
        self,
        min_size: float = 1 / 6,
        canny_minval: int = 50,
        canny_maxval: int = 150
    ) -> None:
        self.min_size = min_size
        self.canny_minval = canny_minval
        self.canny_maxval = canny_maxval

    def __call__(
        self,
        img: np.array,
    ) -> List[Tuple[float, float, float, float]]:
        """Detect Barcodes on the image
        Args:
            img: np image

        Returns:
            A list of tuples: [(xmin, ymin, xmax, ymax), ...] containing barcodes rel. coordinates
        """
        # get image size and define parameters
        height, width = img.shape[:2]
        k = (1 + int(width / 512)) * 10  # spatial extension of kernels, 512 -> 20, 1024 -> 30, ...
        min_w = int(width * self.min_size)  # minimal size of a possible barcode

        # Detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_minval, self.canny_maxval, apertureSize=3)

        # Horizontal dilation to aggregate bars of the potential barcode
        # without aggregating text lines of the page vertically
        edges = cv2.dilate(edges, np.ones((1, k), np.uint8))

        # Instantiate a barcode-shaped kernel and erode to keep only vertical-bar structures
        bar_code_kernel = np.zeros((k, 3), np.uint8)
        bar_code_kernel[..., [0, 2]] = 1
        edges = cv2.erode(edges, bar_code_kernel, iterations=1)

        # Opening to remove noise
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

        # Dilation to retrieve vertical length (lost at the first dilation)
        edges = cv2.dilate(edges, np.ones((k, 1), np.uint8))

        # Find contours, and keep the widest as barcodes
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        barcodes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_w:
                barcodes.append((x / width, y / height, (x + w) / width, (y + h) / height))

        return barcodes
