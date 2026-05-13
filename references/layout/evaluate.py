# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import multiprocessing as mp
import os
import time

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import Normalize

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import transforms as T
from doctr.datasets import LayoutDataset
from doctr.models import layout
from doctr.utils.metrics import ObjectDetectionMetric


@torch.inference_mode()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    for images, targets in tqdm(val_loader):
        imgs, padding_masks = images
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            padding_masks = padding_masks.cuda()
        imgs = batch_transforms(imgs)
        if amp:
            with torch.cuda.amp.autocast():
                out = model(imgs, padding_masks, targets, return_preds=True)
        else:
            out = model(imgs, padding_masks, targets, return_preds=True)
        # Compute metric
        loc_preds = out["preds"]
        for target, pred in zip(targets, loc_preds):
            assert pred["boxes"].shape[0] == pred["scores"].shape[0]
            assert pred["boxes"].shape[0] == pred["labels"].shape[0]
            val_metric.update(
                gt_boxes=target["boxes"],
                pred_boxes=pred["boxes"],
                gt_labels=target["labels"],
                pred_labels=pred["labels"],
                pred_scores=pred["scores"],
            )

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    metrics = val_metric.summary()
    return (
        val_loss,
        metrics["mAP@[.5:.95]"],
        metrics["AP@[.5]"],
        metrics["AP@[.75]"],
    )


def main(args):
    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")
    pbar = tqdm(disable=False if slack_token and slack_channel else True)
    if slack_token and slack_channel:
        # Monkey patch tqdm write method to send messages directly to Slack
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(
            channel=slack_channel,
            text=msg,
        )
    pbar.write(str(args))

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    # Temporary model to recover configuration
    tmp_model = layout.__dict__[args.arch](
        pretrained=False,
        assume_straight_pages=not args.rotation,
    )

    if isinstance(args.size, int):
        input_shape = (args.size, args.size)
    else:
        input_shape = tmp_model.cfg["input_shape"][-2:]
    mean, std = tmp_model.cfg["mean"], tmp_model.cfg["std"]

    st = time.time()
    ds = LayoutDataset(
        img_folder=os.path.join(args.dataset_path, "images"),
        label_path=os.path.join(args.dataset_path, "labels.json"),
        use_polygons=args.rotation,
        sample_transforms=T.Resize(
            input_shape,
            preserve_aspect_ratio=args.keep_ratio,
            symmetric_pad=args.symmetric_pad,
            return_padding_mask=True,
        ),
    )
    class_names = ds.class_names

    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(ds),
        pin_memory=torch.cuda.is_available(),
        collate_fn=ds.collate_fn,
    )

    pbar.write(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in {len(test_loader)} batches)")

    # Load docTR model
    model = layout.__dict__[args.arch](
        pretrained=not isinstance(args.resume, str),
        assume_straight_pages=not args.rotation,
        class_names=class_names,
    ).eval()

    batch_transforms = Normalize(mean=mean, std=std)

    # Resume weights
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

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
        pbar.write("No accessible GPU, target device set to CPU.")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    # Metrics
    metric = ObjectDetectionMetric(
        num_classes=len(class_names),
        use_polygons=args.rotation,
    )

    pbar.write("Running evaluation")
    val_loss, map5095, ap50, ap75 = evaluate(
        model,
        test_loader,
        batch_transforms,
        metric,
        amp=args.amp,
    )
    pbar.write(
        f"Validation loss: {val_loss:.6f} | mAP@[.5:.95]: {map5095:.2%} | AP@[.5]: {ap50:.2%} | AP@[.75]: {ap75:.2%}"
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-detection model to evaluate")
    parser.add_argument("dataset_path", type=str, help="path to the dataset to evaluate on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for evaluation")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--size", type=int, default=None, help="model input size, H = W")
    parser.add_argument("--keep_ratio", action="store_true", help="keep the aspect ratio of the input image")
    parser.add_argument("--symmetric_pad", action="store_true", help="pad the image symmetrically")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--rotation", dest="rotation", action="store_true", help="inference with rotated bbox")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
