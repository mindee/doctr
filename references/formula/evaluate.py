# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import multiprocessing as mp
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms.v2 import Normalize

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import transforms as T
from doctr.datasets import RecognitionDataset
from doctr.models import formula
from doctr.utils.metrics import FormulaMetric
from utils import amp_dtype, filter_samples, load_vocab, model_device, resolve_device

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
    val_loss, sample_cnt = 0.0, 0
    pbar = tqdm(val_loader)
    for images, targets in pbar:
        images = images.to(model_device(model), non_blocking=True)
        images = batch_transforms(images)
        if amp:
            with _autocast():
                out = model(images, targets, return_preds=True)
        else:
            out = model(images, targets, return_preds=True)
        # Compute metric
        val_metric.update(targets, [latex for latex, _ in out["preds"]])

        val_loss += out["loss"].item() * images.shape[0]
        sample_cnt += images.shape[0]

    val_loss /= sample_cnt
    return val_loss, val_metric.summary()


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

    vocab = load_vocab(args.vocab)

    # Load doctr model
    model = formula.__dict__[args.arch](
        pretrained=True if args.resume is None else False,
        input_shape=(3, *args.input_size),
        vocab=vocab,
        max_length=args.max_length,
        max_decode_length=args.max_decode_length,
    ).eval()

    # Resume weights
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    st = time.time()
    dataset_path = Path(args.dataset_path)
    ds = RecognitionDataset(
        img_folder=str(dataset_path.joinpath("images")),
        labels_path=str(dataset_path.joinpath("labels.json")),
        img_transforms=T.Resize(args.input_size, preserve_aspect_ratio=True),
    )
    unknown, too_long = filter_samples(ds, vocab, args.max_length)
    if unknown or too_long:
        pbar.write(f"{unknown} samples with unknown tokens and {too_long} too long samples skipped")

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

    batch_transforms = Normalize(mean=model.cfg["mean"], std=model.cfg["std"])

    # Metrics
    val_metric = FormulaMetric()

    # GPU
    if device.type == "cpu":
        pbar.write("No accessible GPU, target device set to CPU.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device)

    pbar.write("Running evaluation")
    val_loss, result = evaluate(model, test_loader, batch_transforms, val_metric, amp=args.amp)
    pbar.write(
        f"Validation loss: {val_loss:.6} (Exact match: {result['exact_match']:.2%} | "
        f"Edit similarity: {result['edit_similarity']:.2%} | BLEU: {result['bleu']:.4})"
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for formula recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="formula recognition model to evaluate")
    parser.add_argument("dataset_path", type=str, help="path to the evaluation data folder (images/ + labels.json)")
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="path to a JSON list of LaTeX tokens (cf. build_vocab.py), the default LaTeX tokens of docTR if omitted",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device for single-process runs: a CUDA index (e.g. 0), 'cuda:N', 'mps' (Apple Silicon) or 'cpu'. "
        "Default: CUDA if available, else MPS, else CPU.",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size for evaluation")
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=(256, 768),
        metavar=("H", "W"),
        help="input size (H, W) of the model",
    )
    parser.add_argument("--max_length", type=int, default=4096, help="maximum number of tokens of a formula")
    parser.add_argument(
        "--max_decode_length",
        type=int,
        default=2048,
        help="maximum number of tokens generated per formula by the greedy decoding of the predictions",
    )
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="autocast dtype for --amp; bfloat16 (Ampere+ GPUs) avoids float16 overflows, no loss scaling",
    )
    args = parser.parse_args()
    args.input_size = tuple(args.input_size)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
