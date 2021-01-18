# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict


def show_boxes_pdf(
    filepaths: List[str],
    postprocess_out: List[np.ndarray]
) -> None:
    """
    Show pdf pages with detected bounding boxes
    :param filepaths: list of pdfs to show
    :param postprocess_out: predictions, list of (Nx5) tensors containing N boxes
    """

    documents_imgs, documents_names, doc_shapes = read_documents(filepaths)
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