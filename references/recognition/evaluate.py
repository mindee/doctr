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

from doctr import datasets
from doctr import transforms as T
from doctr.datasets import VOCABS
from doctr.models import recognition
from doctr.utils.metrics import TextMatch
from utils import amp_dtype, model_device, resolve_device

AMP_DTYPE = torch.float16  # set from --amp-dtype in main()


def _autocast():
    return torch.amp.autocast("cuda", dtype=AMP_DTYPE)


@torch.inference_mode()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    pbar = tqdm(val_loader)
    for images, targets in pbar:
        try:
            images = images.to(model_device(model), non_blocking=True)
            images = batch_transforms(images)
            if amp:
                with _autocast():
                    out = model(images, targets, return_preds=True)
            else:
                out = model(images, targets, return_preds=True)
            # Compute metric
            if len(out["preds"]):
                words, _ = zip(*out["preds"])
            else:
                words = []
            val_metric.update(targets, words)

            val_loss += out["loss"].item()
            batch_cnt += 1
        except ValueError:
            pbar.write(f"unexpected symbol/s in targets:\n{targets} \n--> skip batch")
            continue

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"]


def main(args):
    global AMP_DTYPE
    AMP_DTYPE = amp_dtype(args.amp_dtype)
    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")

    pbar = tqdm(disable=False if slack_token and slack_channel else True)
    if slack_token and slack_channel:
        # Monkey patch tqdm write method to send messages directly to Slack
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(channel=slack_channel, text=msg)
    pbar.write(str(args))
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp (automatic mixed precision) is only supported on CUDA devices")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    # Load doctr model
    model = recognition.__dict__[args.arch](
        pretrained=True if args.resume is None else False,
        input_shape=(3, args.input_size, 4 * args.input_size),
        vocab=VOCABS[args.vocab],
    ).eval()

    # Resume weights
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

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
        num_workers=args.workers,
        sampler=SequentialSampler(ds),
        pin_memory=device.type == "cuda",
        collate_fn=ds.collate_fn,
    )
    pbar.write(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in {len(test_loader)} batches)")

    mean, std = model.cfg["mean"], model.cfg["std"]
    batch_transforms = Normalize(mean=mean, std=std)

    # Metrics
    val_metric = TextMatch()

    # GPU
    if device.type == "cpu":
        pbar.write("No accessible GPU, target device set to CPU.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device)

    pbar.write("Running evaluation")
    val_loss, exact_match, partial_match = evaluate(model, test_loader, batch_transforms, val_metric, amp=args.amp)
    pbar.write(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-recognition model to evaluate")
    parser.add_argument("--vocab", type=str, default="french", help="Vocab to be used for evaluation")
    parser.add_argument("--dataset", type=str, default="FUNSD", help="Dataset to evaluate on")
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device for single-process runs: a CUDA index (e.g. 0), 'cuda:N', 'mps' (Apple Silicon) or 'cpu'. "
        "Default: CUDA if available, else MPS, else CPU.",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument(
        "--only_regular", dest="regular", action="store_true", help="test set contains only regular text"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="autocast dtype for --amp; bfloat16 (Ampere+ GPUs) avoids float16 overflows, no loss scaling",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
