# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

from doctr.file_utils import ensure_keras_v2

ensure_keras_v2()

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm

gpu_devices = tf.config.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr import datasets
from doctr import transforms as T
from doctr.datasets import VOCABS, DataLoader
from doctr.models import recognition
from doctr.utils.metrics import TextMatch


def evaluate(model, val_loader, batch_transforms, val_metric):
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    val_iter = iter(val_loader)
    for images, targets in tqdm(val_iter):
        try:
            images = batch_transforms(images)
            out = model(images, target=targets, return_preds=True, training=False)
            # Compute metric
            if len(out["preds"]):
                words, _ = zip(*out["preds"])
            else:
                words = []
            val_metric.update(targets, words)

            val_loss += out["loss"].numpy().mean()
            batch_cnt += 1
        except ValueError:
            print(f"unexpected symbol/s in targets:\n{targets} \n--> skip batch")
            continue

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"]


def main(args):
    print(args)

    # AMP
    if args.amp:
        mixed_precision.set_global_policy("mixed_float16")

    # Load doctr model
    model = recognition.__dict__[args.arch](
        pretrained=True if args.resume is None else False,
        input_shape=(args.input_size, 4 * args.input_size, 3),
        vocab=VOCABS[args.vocab],
    )

    # Resume weights
    if isinstance(args.resume, str):
        model.load_weights(args.resume)

    st = time.time()
    ds = datasets.__dict__[args.dataset](
        train=True,
        download=True,
        recognition_task=True,
        use_polygons=args.regular,
        img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )

    _ds = datasets.__dict__[args.dataset](
        train=False,
        download=True,
        recognition_task=True,
        use_polygons=args.regular,
        img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )
    ds.data.extend((np_img, target) for np_img, target in _ds.data)

    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )
    print(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in {len(test_loader)} batches)")

    mean, std = model.cfg["mean"], model.cfg["std"]
    batch_transforms = T.Normalize(mean=mean, std=std)

    # Metrics
    val_metric = TextMatch()

    print("Running evaluation")
    val_loss, exact_match, partial_match = evaluate(model, test_loader, batch_transforms, val_metric)
    print(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text recognition (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-recognition model to evaluate")
    parser.add_argument("--vocab", type=str, default="french", help="Vocab to be used for evaluation")
    parser.add_argument("--dataset", type=str, default="FUNSD", help="Dataset to evaluate on")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument(
        "--only_regular", dest="regular", action="store_true", help="test set contains only regular text"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
