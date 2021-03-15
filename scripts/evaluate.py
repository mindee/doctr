# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import numpy as np
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from doctr.utils import metrics
from doctr.datasets import FUNSD
from doctr.documents import read_img
from doctr.models import zoo


def main(args):

    if args.model not in zoo.__all__:
        raise ValueError('only the following end-to-end predictors are supported:', zoo.__all__)

    model = zoo.__dict__[args.model](pretrained=True)

    dataset = FUNSD(train=True, download=True)
    # Patch the dataset to use both subsets as they are not used for training
    test_set = FUNSD(train=False, download=True)
    dataset.data.extend(test_set.data)

    metric = metrics.OCRMetric(iou_thresh=args.iou)

    for page, target in tqdm(dataset):
        # GT
        gt_boxes = np.asarray(target['boxes'])
        gt_labels = list(target['labels'])

        # Forward
        out = model([[page]])

        # Unpack preds
        pred_boxes = []
        pred_labels = []
        for page in out[0].pages:
            h, w = page.dimensions
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        (a, b), (c, d) = word.geometry
                        pred_boxes.append([int(a * w), int(b * h), int(c * w), int(d * h)])
                        pred_labels.append(word.value)

        # Update the metric
        metric.update(gt_boxes, np.asarray(pred_boxes), gt_labels, pred_labels)

    # Unpack aggregated metrics
    recall, precision, mean_iou, _ = metric.summary()
    print(f"End-to-End Evaluation (model='{args.model}', dataset='{args.dataset}')")
    print(f"Recall: {recall:.2%}, Precision: {precision:.2%}, Mean IoU: {mean_iou:.2%}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model', type=str, help='OCR model to use for analysis')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold to match a pair of boxes')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
