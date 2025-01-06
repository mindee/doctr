# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

from doctr.io.elements import KIEDocument

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tqdm import tqdm

from doctr import datasets
from doctr import transforms as T
from doctr.file_utils import is_tf_available
from doctr.models import kie_predictor
from doctr.utils.geometry import extract_crops, extract_rcrops
from doctr.utils.metrics import LocalizationConfusion, OCRMetric, TextMatch

# Enable GPU growth if using TF
if is_tf_available():
    import tensorflow as tf

    gpu_devices = tf.config.list_physical_devices("GPU")
    if any(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    import torch


def _pct(val):
    return "N/A" if val is None else f"{val:.2%}"


def main(args):
    if not args.rotation:
        args.eval_straight = True

    input_shape = (args.size, args.size)

    # We define a transformation function which does transform the annotation
    # to the required format for the Resize transformation
    def _transform(img, target):
        boxes = target["boxes"]
        transformed_img, transformed_boxes = T.Resize(
            input_shape, preserve_aspect_ratio=args.keep_ratio, symmetric_pad=args.symmetric_pad
        )(img, boxes)
        return transformed_img, {"boxes": transformed_boxes, "labels": target["labels"]}

    predictor = kie_predictor(
        args.detection,
        args.recognition,
        pretrained=True,
        reco_bs=args.batch_size,
        preserve_aspect_ratio=False,  # we handle the transformation directly in the dataset so this is set to False
        symmetric_pad=False,  # we handle the transformation directly in the dataset so this is set to False
        assume_straight_pages=not args.rotation,
    )

    if args.img_folder and args.label_file:
        testset = datasets.OCRDataset(
            img_folder=args.img_folder,
            label_file=args.label_file,
            sample_transforms=_transform,
        )
        sets = [testset]
    else:
        train_set = datasets.__dict__[args.dataset](
            train=True,
            download=True,
            use_polygons=not args.eval_straight,
            sample_transforms=_transform,
        )
        val_set = datasets.__dict__[args.dataset](
            train=False,
            download=True,
            use_polygons=not args.eval_straight,
            sample_transforms=_transform,
        )
        sets = [train_set, val_set]

    reco_metric = TextMatch()

    det_metric = LocalizationConfusion(iou_thresh=args.iou, use_polygons=not args.eval_straight)
    e2e_metric = OCRMetric(iou_thresh=args.iou, use_polygons=not args.eval_straight)

    sample_idx = 0
    extraction_fn = extract_crops if args.eval_straight else extract_rcrops

    for dataset in sets:
        for page, target in tqdm(dataset):
            # GT
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            if args.img_folder and args.label_file:
                x, y, w, h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
                xmin, ymin = np.clip(x - w / 2, 0, 1), np.clip(y - h / 2, 0, 1)
                xmax, ymax = np.clip(x + w / 2, 0, 1), np.clip(y + h / 2, 0, 1)
                gt_boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)

            # Forward
            out: KIEDocument
            if is_tf_available():
                out = predictor(page[None, ...])
                crops = extraction_fn(page, gt_boxes)
                reco_out = predictor.reco_predictor(crops)
            else:
                with torch.no_grad():
                    out = predictor(page[None, ...])
                    # We directly crop on PyTorch tensors, which are in channels_first
                    crops = extraction_fn(page, gt_boxes, channels_last=False)
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
                for predictions in page.predictions.values():
                    for prediction in predictions:
                        if not args.rotation:
                            (a, b), (c, d) = prediction.geometry
                        else:
                            (
                                [x1, y1],
                                [x2, y2],
                                [x3, y3],
                                [x4, y4],
                            ) = prediction.geometry
                        if np.issubdtype(gt_boxes.dtype, np.integer):
                            if not args.rotation:
                                pred_boxes.append([int(a * width), int(b * height), int(c * width), int(d * height)])
                            else:
                                if args.eval_straight:
                                    pred_boxes.append([
                                        int(width * min(x1, x2, x3, x4)),
                                        int(height * min(y1, y2, y3, y4)),
                                        int(width * max(x1, x2, x3, x4)),
                                        int(height * max(y1, y2, y3, y4)),
                                    ])
                                else:
                                    pred_boxes.append([
                                        [int(x1 * width), int(y1 * height)],
                                        [int(x2 * width), int(y2 * height)],
                                        [int(x3 * width), int(y3 * height)],
                                        [int(x4 * width), int(y4 * height)],
                                    ])
                        else:
                            if not args.rotation:
                                pred_boxes.append([a, b, c, d])
                            else:
                                if args.eval_straight:
                                    pred_boxes.append([
                                        min(x1, x2, x3, x4),
                                        min(y1, y2, y3, y4),
                                        max(x1, x2, x3, x4),
                                        max(y1, y2, y3, y4),
                                    ])
                                else:
                                    pred_boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                        pred_labels.append(prediction.value)

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
    print(
        f"Model Evaluation (model= {args.detection} + {args.recognition}, "
        f"dataset={'OCRDataset' if args.img_folder else args.dataset})"
    )
    recall, precision, mean_iou = det_metric.summary()
    print(f"Text Detection - Recall: {_pct(recall)}, Precision: {_pct(precision)}, Mean IoU: {_pct(mean_iou)}")
    acc = reco_metric.summary()
    print(f"Text Recognition - Accuracy: {_pct(acc['raw'])} (unicase: {_pct(acc['unicase'])})")
    recall, precision, mean_iou = e2e_metric.summary()
    print(
        f"KIE OCR - Recall: {_pct(recall['raw'])} (unicase: {_pct(recall['unicase'])}), "
        f"Precision: {_pct(precision['raw'])} (unicase: {_pct(precision['unicase'])}), Mean IoU: {_pct(mean_iou)}"
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR end-to-end evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("detection", type=str, help="Text detection model to use for analysis")
    parser.add_argument("recognition", type=str, help="Text recognition model to use for analysis")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold to match a pair of boxes")
    parser.add_argument("--dataset", type=str, default="FUNSD", help="choose a dataset: FUNSD, CORD")
    parser.add_argument("--img_folder", type=str, default=None, help="Only for local sets, path to images")
    parser.add_argument("--label_file", type=str, default=None, help="Only for local sets, path to labels")
    parser.add_argument("--rotation", dest="rotation", action="store_true", help="run rotated OCR + postprocessing")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch size for recognition")
    parser.add_argument("--size", type=int, default=1024, help="model input size, H = W")
    parser.add_argument("--keep_ratio", action="store_true", help="keep the aspect ratio of the input image")
    parser.add_argument("--symmetric_pad", action="store_true", help="pad the image symmetrically")
    parser.add_argument("--samples", type=int, default=None, help="evaluate only on the N first samples")
    parser.add_argument(
        "--eval-straight",
        action="store_true",
        help="evaluate on straight pages with straight bbox (to use the quick and light metric)",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
