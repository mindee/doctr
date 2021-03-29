# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import numpy as np
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from doctr.utils.metrics import LocalizationConfusion, ExactMatch, OCRMetric
from doctr.datasets import FUNSD, SROIE
from doctr.models import ocr_predictor, extract_crops


def main(args):

    model = ocr_predictor(args.detection, args.recognition, pretrained=True)

    train_set = FUNSD(train=True, download=True)
    test_set = FUNSD(train=False, download=True)

    det_metric = LocalizationConfusion(iou_thresh=args.iou)
    reco_metric = ExactMatch()
    e2e_metric = OCRMetric(iou_thresh=args.iou)

    for dataset in (train_set, test_set):
        for page, target in tqdm(dataset):
            # GT
            gt_boxes = np.asarray(target['boxes'])
            gt_labels = list(target['labels'])

            # Forward
            out = model([page])
            # Crop GT
            crops = extract_crops(page, gt_boxes)
            reco_out = model.reco_predictor(crops)

            # Unpack preds
            pred_boxes = []
            pred_labels = []
            for page in out.pages:
                h, w = page.dimensions
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            (a, b), (c, d) = word.geometry
                            pred_boxes.append([int(a * w), int(b * h), int(c * w), int(d * h)])
                            pred_labels.append(word.value)

            # Update the metric
            det_metric.update(gt_boxes, np.asarray(pred_boxes))
            reco_metric.update(gt_labels, reco_out)
            e2e_metric.update(gt_boxes, np.asarray(pred_boxes), gt_labels, pred_labels)

    # Unpack aggregated metrics
    print(f"Model Evaluation (model= {args.detection} + {args.recognition}, dataset='FUNSD')")
    recall, precision, mean_iou = det_metric.summary()
    print(f"Text Detection - Recall: {recall:.2%}, Precision: {precision:.2%}, Mean IoU: {mean_iou:.2%}")
    acc = reco_metric.summary()
    print(f"Text Recognition - Accuracy: {acc:.2%}")
    recall, precision, mean_iou, _ = e2e_metric.summary()
    print(f"OCR - Recall: {recall:.2%}, Precision: {precision:.2%}, Mean IoU: {mean_iou:.2%}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('detection', type=str, help='Text detection model to use for analysis')
    parser.add_argument('recognition', type=str, help='Text recognition model to use for analysis')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold to match a pair of boxes')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
