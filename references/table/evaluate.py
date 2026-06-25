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


@torch.inference_mode()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    model.eval()
    val_metric.reset()
    val_loss, batch_cnt = 0, 0
    for images, targets in tqdm(val_loader):
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)
        if amp:
            with torch.amp.autocast("cuda"):
                out = model(images, target=targets, return_preds=True)
        else:
            out = model(images, target=targets, return_preds=True)

        for target, pred in zip(targets, out["preds"]):
            val_metric.update(
                np.asarray(target["cells"], dtype=np.float32).reshape(-1, 4, 2),
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
    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")
    pbar = tqdm(disable=False if slack_token and slack_channel else True)
    if slack_token and slack_channel:
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(channel=slack_channel, text=msg)
    pbar.write(str(args))

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    tmp_model = table_structure.__dict__[args.arch](pretrained=False)
    input_shape = (args.size, args.size) if isinstance(args.size, int) else tmp_model.cfg["input_shape"][-2:]
    mean, std = tmp_model.cfg["mean"], tmp_model.cfg["std"]

    st = time.time()
    ds = TableStructureDataset(
        img_folder=os.path.join(args.dataset_path, "images"),
        label_path=os.path.join(args.dataset_path, "labels.json"),
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
        pin_memory=torch.cuda.is_available(),
        collate_fn=ds.collate_fn,
    )
    pbar.write(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in {len(test_loader)} batches)")

    model = table_structure.__dict__[args.arch](pretrained=not isinstance(args.resume, str)).eval()
    batch_transforms = Normalize(mean=mean, std=std)
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    elif torch.cuda.is_available():
        args.device = 0
    else:
        pbar.write("No accessible GPU, target device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    metric = TableCellMetric(iou_thresh=args.iou_thresh)
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
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--size", type=int, default=None, help="model input size, H = W")
    parser.add_argument("--keep_ratio", action="store_true", help="keep the aspect ratio of the input image")
    parser.add_argument("--symmetric_pad", action="store_true", help="pad the image symmetrically")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for cell matching")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
