# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tqdm import tqdm

from doctr import datasets
from doctr.file_utils import is_tf_available
from doctr.models import ocr_predictor
from doctr.models._utils import extract_crops
from doctr.utils.metrics import LocalizationConfusion, OCRMetric, TextMatch

# Enable GPU growth if using TF
if is_tf_available():
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if any(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    import torch


def _pct(val):
    return "N/A" if val is None else f"{val:.2%}"


def main(args):

    predictor = ocr_predictor(args.detection, args.recognition, pretrained=True, reco_bs=args.batch_size)

    if args.img_folder and args.label_file:
        testset = datasets.OCRDataset(
            img_folder=args.img_folder,
            label_file=args.label_file,
        )
        sets = [testset]
    else:
        train_set = datasets.__dict__[args.dataset](train=True, download=True, rotated_bbox=args.rotation)
        val_set = datasets.__dict__[args.dataset](train=False, download=True, rotated_bbox=args.rotation)
        sets = [train_set, val_set]

    reco_metric = TextMatch()
    if args.rotation and args.mask_shape:
        det_metric = LocalizationConfusion(
            iou_thresh=args.iou,
            rotated_bbox=args.rotation,
            mask_shape=(args.mask_shape, args.mask_shape)
        )
        e2e_metric = OCRMetric(
            iou_thresh=args.iou,
            rotated_bbox=args.rotation,
            mask_shape=(args.mask_shape, args.mask_shape)
        )
    else:
        det_metric = LocalizationConfusion(iou_thresh=args.iou, rotated_bbox=args.rotation)
        e2e_metric = OCRMetric(iou_thresh=args.iou, rotated_bbox=args.rotation)

    sample_idx = 0
    for dataset in sets:
        for page, target in tqdm(dataset):
            # GT
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            if args.img_folder and args.label_file:
                x, y, w, h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
                xmin, ymin = np.clip(x - w / 2, 0, 1), np.clip(y - h / 2, 0, 1)
                xmax, ymax = np.clip(x + w / 2, 0, 1), np.clip(y + h / 2, 0, 1)
                gt_boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)

            # Forward
            if is_tf_available():
                out = predictor(page[None, ...])
                crops = extract_crops(page, gt_boxes)
                reco_out = predictor.reco_predictor(crops)
            else:
                with torch.no_grad():
                    out = predictor(page[None, ...])
                    # We directly crop on PyTorch tensors, which are in channels_first
                    crops = extract_crops(page, gt_boxes, channels_last=False)
                    reco_out = predictor.reco_predictor(crops)

            if len(reco_out):
                reco_words, _ = zip(*reco_out)
            else:
                reco_words = []

            # Unpack preds
            pred_boxes = []
            pred_labels = []
            for page in out.pages:
                height, width = page.dimensions
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if not args.rotation:
                                (a, b), (c, d) = word.geometry
                            else:
                                x, y, w, h, alpha = word.geometry
                            if gt_boxes.dtype == int:
                                if not args.rotation:
                                    pred_boxes.append([int(a * width), int(b * height),
                                                       int(c * width), int(d * height)])
                                else:
                                    pred_boxes.append(
                                        [int(x * width), int(y * height), int(w * width), int(h * height), alpha]
                                    )
                            else:
                                if not args.rotation:
                                    pred_boxes.append([a, b, c, d])
                                else:
                                    pred_boxes.append([x, y, w, h, alpha])
                            pred_labels.append(word.value)

            # Update the metric
            det_metric.update(gt_boxes, np.asarray(pred_boxes))
            reco_metric.update(gt_labels, reco_words)
            e2e_metric.update(gt_boxes, np.asarray(pred_boxes), gt_labels, pred_labels)

            # Loop break
            sample_idx += 1
            if isinstance(args.samples, int) and args.samples == sample_idx:
                break
        if isinstance(args.samples, int) and args.samples == sample_idx:
            break

    # Unpack aggregated metrics
    print(f"Model Evaluation (model= {args.detection} + {args.recognition}, "
          f"dataset={'OCRDataset' if args.img_folder else args.dataset})")
    recall, precision, mean_iou = det_metric.summary()
    print(f"Text Detection - Recall: {_pct(recall)}, Precision: {_pct(precision)}, Mean IoU: {_pct(mean_iou)}")
    acc = reco_metric.summary()
    print(f"Text Recognition - Accuracy: {_pct(acc['raw'])} (unicase: {_pct(acc['unicase'])})")
    recall, precision, mean_iou = e2e_metric.summary()
    print(f"OCR - Recall: {_pct(recall['raw'])} (unicase: {_pct(recall['unicase'])}), "
          f"Precision: {_pct(precision['raw'])} (unicase: {_pct(precision['unicase'])}), Mean IoU: {_pct(mean_iou)}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('detection', type=str, help='Text detection model to use for analysis')
    parser.add_argument('recognition', type=str, help='Text recognition model to use for analysis')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold to match a pair of boxes')
    parser.add_argument('--dataset', type=str, default='FUNSD', help='choose a dataset: FUNSD, CORD')
    parser.add_argument('--img_folder', type=str, default=None, help='Only for local sets, path to images')
    parser.add_argument('--label_file', type=str, default=None, help='Only for local sets, path to labels')
    parser.add_argument('--rotation', dest='rotation', action='store_true', help='evaluate with rotated bbox')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size for recognition')
    parser.add_argument('--mask_shape', type=int, default=None, help='mask shape for mask iou (only for rotation)')
    parser.add_argument('--samples', type=int, default=None, help='evaluate only on the N first samples')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
