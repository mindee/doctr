# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

os.environ["USE_TORCH"] = "1"

import logging
import multiprocessing as mp
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import Normalize
from tqdm import tqdm

from doctr import datasets
from doctr import transforms as T
from doctr.models import detection
from doctr.utils.metrics import LocalizationConfusion


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    for images, targets in tqdm(val_loader):
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)
        targets = [t["boxes"] for t in targets]
        if amp:
            with torch.cuda.amp.autocast():
                out = model(images, targets, return_preds=True)
        else:
            out = model(images, targets, return_preds=True)
        # Compute metric
        loc_preds = out["preds"]
        for boxes_gt, boxes_pred in zip(targets, loc_preds):
            # Remove scores
            val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :-1])

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args):

    print(args)

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True
    system_available_memory = int(psutil.virtual_memory().available / 1024**3)

    # Load docTR model
    model = detection.__dict__[args.arch](
        pretrained=not isinstance(args.resume, str), assume_straight_pages=not args.rotation
    ).eval()

    if isinstance(args.size, int):
        input_shape = (args.size, args.size)
    else:
        input_shape = model.cfg["input_shape"][-2:]
    mean, std = model.cfg["mean"], model.cfg["std"]

    st = time.time()
    ds = datasets.__dict__[args.dataset](
        train=True,
        download=True,
        use_polygons=args.rotation,
        sample_transforms=T.Resize(input_shape),
    )
    # Monkeypatch
    subfolder = ds.root.split("/")[-2:]
    ds.root = str(Path(ds.root).parent.parent)
    ds.data = [(os.path.join(*subfolder, name), target) for name, target in ds.data]
    _ds = datasets.__dict__[args.dataset](
        train=False,
        download=True,
        use_polygons=args.rotation,
        sample_transforms=T.Resize(input_shape),
    )
    subfolder = _ds.root.split("/")[-2:]
    ds.data.extend([(os.path.join(*subfolder, name), target) for name, target in _ds.data])

    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(ds),
        pin_memory=torch.cuda.is_available(),
        collate_fn=ds.collate_fn,
    )
    print(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in " f"{len(test_loader)} batches)")

    batch_transforms = Normalize(mean=mean, std=std)

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

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
        logging.warning("No accessible GPU, targe device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    # Metrics
    metric = LocalizationConfusion(
        use_polygons=args.rotation,
        mask_shape=input_shape,
        use_broadcasting=True if system_available_memory > 62 else False,
    )

    print("Running evaluation")
    val_loss, recall, precision, mean_iou = evaluate(model, test_loader, batch_transforms, metric, amp=args.amp)
    print(
        f"Validation loss: {val_loss:.6} (Recall: {recall:.2%} | Precision: {precision:.2%} | "
        f"Mean IoU: {mean_iou:.2%})"
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-detection model to evaluate")
    parser.add_argument("--dataset", type=str, default="FUNSD", help="Dataset to evaluate on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for evaluation")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--size", type=int, default=None, help="model input size, H = W")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--rotation", dest="rotation", action="store_true", help="inference with rotated bbox")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
