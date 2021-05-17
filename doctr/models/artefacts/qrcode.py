# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import numpy as np
from typing import Tuple, Optional

__all__ = ['QRCodeDetector']


class QRCodeDetector(cv2.QRCodeDetector):

    """ Implements a QR-code detector.
    Based on open CV QRCodeDetector. Returns both localization and decoded text.
    """

    def __call__(
        self,
        img: np.array,
    ) -> Optional[Tuple[Tuple[float, float, float, float], str]]:
        """Detect QRcode on the image, if there is one.
        Args:
            img: np image

        Returns:
            A tuple: ((xmin, ymin, xmax, ymax), prediction) or None
        """
        decodedText, points, _ = self.detector.detectAndDecode(img)

        if points is None:
            return None
        xmin, ymin = np.min(points[0][:, 0]), np.min(points[0][:, 1])
        xmax, ymax = np.max(points[0][:, 0]), np.max(points[0][:, 1])
        return ((xmin, ymin, xmax, ymax), decodedText)
