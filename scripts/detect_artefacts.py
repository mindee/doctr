# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import os

os.environ["USE_TORCH"] = "1"

import argparse
import logging

import cv2
import matplotlib.pyplot as plt
import torch

from doctr.io.image import read_img_as_tensor
from doctr.models import obj_detection

CLASSES = ["__background__", "QR Code", "Barcode", "Logo", "Photo"]
CM = [(255, 255, 255), (0, 0, 150), (0, 0, 0), (0, 150, 0), (150, 0, 0)]


def plot_predictions(image, boxes, labels):
    for box, label in zip(boxes, labels):
        # Bounding box around artefacts
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), CM[label], 2)
        text_size, _ = cv2.getTextSize(CLASSES[label], cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        text_w, text_h = text_size
        # Filled rectangle above bounding box
        cv2.rectangle(image, (box[0], box[1]), (box[0] + text_w, box[1] - text_h), CM[label], -1)
        # Text bearing the name of the artefact detected
        cv2.putText(image, CLASSES[label], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    plt.axis("off")
    plt.imshow(image)
    plt.show()


@torch.no_grad()
def main(args):
    print(args)

    model = obj_detection.__dict__[args.arch](pretrained=True, num_classes=5).eval()
    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        logging.warning("No accessible GPU, target device set to CPU.")
    img = read_img_as_tensor(args.img_path).unsqueeze(0)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()
        img = img.cuda()

    pred = model(img)
    labels = pred[0]["labels"].detach().cpu().numpy()
    labels = labels.round().astype(int)
    boxes = pred[0]["boxes"].detach().cpu().numpy()
    boxes = boxes.round().astype(int)
    img = img.cpu().permute(0, 2, 3, 1).numpy()[0].copy()
    plot_predictions(img, boxes, labels)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DocTR artefact detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("arch", type=str, help="Artefact detection model to use")
    parser.add_argument("img_path", type=str, help="path to the image")
    parser.add_argument("--device", default=None, type=int, help="device")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
