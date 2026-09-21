# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import multiprocessing as mp
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import Normalize

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import transforms as T
from doctr.datasets import TableStructureDataset
from doctr.models import table_structure
from doctr.utils.metrics import TableCellMetric
from utils import amp_dtype, model_device, resolve_device

AMP_DTYPE = torch.float16  # set from --amp-dtype in main()


def _autocast():
    return torch.amp.autocast("cuda", dtype=AMP_DTYPE)


def _scaler():
    # bfloat16 has the range of float32: no loss scaling needed (GradScaler only makes sense for float16)
    return torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)


@torch.inference_mode()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    model.eval()
    val_metric.reset()
    val_loss, batch_cnt = 0, 0
    for images, targets in tqdm(val_loader):
        images = images.to(model_device(model), non_blocking=True)
        images = batch_transforms(images)
        if amp:
            with _autocast():
                out = model(images, target=targets, return_preds=True)
        else:
            out = model(images, target=targets, return_preds=True)

        for target, pred in zip(targets, out["preds"]):
            val_metric.update(
                np.asarray(target["cells"], dtype=np.float32),
                np.asarray(target["logic"], dtype=np.int64).reshape(-1, 4),
                pred["polygons"],
                pred["logical"],
            )

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    metrics = val_metric.summary()
    return val_loss, metrics["recall"], metrics["precision"], metrics["f1"], metrics["structure_acc"]


def main(args):
    global AMP_DTYPE
    AMP_DTYPE = amp_dtype(args.amp_dtype)
    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")
    pbar = tqdm(disable=False if slack_token and slack_channel else True)
    if slack_token and slack_channel:
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(channel=slack_channel, text=msg)
    pbar.write(str(args))
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp (automatic mixed precision) is only supported on CUDA devices")

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tmp_model = table_structure.__dict__[args.arch](pretrained=False, assume_straight_pages=not args.rotation)
    input_shape = (args.size, args.size) if isinstance(args.size, int) else tmp_model.cfg["input_shape"][-2:]
    mean, std = tmp_model.cfg["mean"], tmp_model.cfg["std"]

    st = time.time()
    ds = TableStructureDataset(
        img_folder=os.path.join(args.dataset_path, "images"),
        label_path=os.path.join(args.dataset_path, "labels.json"),
        use_polygons=args.rotation,
        sample_transforms=T.Resize(
            input_shape, preserve_aspect_ratio=args.keep_ratio, symmetric_pad=args.symmetric_pad
        ),
    )
    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(ds),
        pin_memory=device.type == "cuda",
        collate_fn=ds.collate_fn,
    )
    pbar.write(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in {len(test_loader)} batches)")

    model = table_structure.__dict__[args.arch](
        pretrained=not isinstance(args.resume, str), assume_straight_pages=not args.rotation
    ).eval()
    batch_transforms = Normalize(mean=mean, std=std)
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    if device.type == "cpu":
        pbar.write("No accessible GPU, target device set to CPU.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device)

    metric = TableCellMetric(iou_thresh=args.iou_thresh, use_polygons=args.rotation)
    pbar.write("Running evaluation")
    val_loss, recall, precision, f1, struct = evaluate(model, test_loader, batch_transforms, metric, amp=args.amp)
    pbar.write(
        f"Validation loss: {val_loss:.6f} | Recall: {(recall or 0):.2%} | Precision: {(precision or 0):.2%} "
        f"| F1: {(f1 or 0):.2%} | Structure acc: {(struct or 0):.2%}"
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for table structure recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("arch", type=str, help="table model to evaluate")
    parser.add_argument("dataset_path", type=str, help="path to the dataset folder (images/ + labels.json)")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for evaluation")
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device for single-process runs: a CUDA index (e.g. 0), 'cuda:N', 'mps' (Apple Silicon) or 'cpu'. "
        "Default: CUDA if available, else MPS, else CPU.",
    )
    parser.add_argument("--size", type=int, default=None, help="model input size, H = W")
    parser.add_argument("--keep_ratio", action="store_true", help="keep the aspect ratio of the input image")
    parser.add_argument("--symmetric_pad", action="store_true", help="pad the image symmetrically")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for cell matching")
    parser.add_argument("--rotation", action="store_true", help="use rotation augmentation")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="autocast dtype for --amp; bfloat16 (Ampere+ GPUs) avoids float16 overflows, no loss scaling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
