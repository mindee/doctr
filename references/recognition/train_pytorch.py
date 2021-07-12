# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ['USE_TORCH'] = '1'

import time
import datetime
import multiprocessing as mp
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
import torch
from torchvision.transforms import Compose, Lambda, Normalize, ColorJitter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from contiguous_params import ContiguousParams
import wandb
from typing import List

from doctr.models import recognition
from doctr.utils.metrics import TextMatch
from doctr.datasets import RecognitionDataset, VOCABS
from doctr import transforms as T

from utils import plot_samples


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, mb):
    model.train()
    train_iter = iter(train_loader)
    # Iterate over the batches of the dataset
    for _ in progress_bar(range(len(train_loader)), parent=mb):
        images, targets = next(train_iter)

        images = batch_transforms(images)

        train_loss = model(images, targets)['loss']

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        mb.child.comment = f'Training loss: {train_loss.item():.6}'


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    val_iter = iter(val_loader)
    for images, targets in val_iter:
        images = batch_transforms(images)
        out = model(images, targets, return_preds=True)
        # Compute metric
        if len(out['preds']):
            words, _ = zip(*out['preds'])
        else:
            words = []
        val_metric.update(targets, words)

        val_loss += out['loss'].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result['raw'], result['unicase']


def main(args):

    print(args)

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    # Load val data generator
    st = time.time()
    val_set = RecognitionDataset(
        img_folder=os.path.join(args.val_data_path, 'images'),
        labels_path=os.path.join(args.val_data_path, 'labels.json'),
        sample_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(val_set),
        pin_memory=True,
        collate_fn=val_set.collate_fn,
    )
    print(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in "
          f"{len(val_loader)} batches)")

    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load doctr model
    model = recognition.__dict__[args.model](pretrained=args.pretrained, vocab=VOCABS['french'])

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)

    # Metrics
    val_metric = TextMatch()

    if args.test_only:
        print("Running evaluation")
        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric)
        print(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")
        return

    st = time.time()

    # Load train data generator
    train_set = RecognitionDataset(
        img_folder=os.path.join(args.train_data_path[0], 'images'),
        labels_path=os.path.join(args.train_data_path[0], 'labels.json'),
        sample_transforms=Compose([
            T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
            # Augmentations
            T.RandomApply(T.ColorInversion(), .1),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        ]),
    )

    # If multiple paths provided, 
    if len(args.train_data_path) > 1:
        for i in range(1, len(args.train_data_path)):
                train_set.merge_dataset(
                    RecognitionDataset(
                        img_folder=os.path.join(args.train_data_path[i], 'images'),
                        labels_path=os.path.join(args.train_data_path[i], 'labels.json'),
                        sample_transforms=Compose([
                            T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                            # Augmentations
                            T.RandomApply(T.ColorInversion(), .1),
                            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
                        ]),
                    )
                )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=RandomSampler(train_set),
        pin_memory=True,
        collate_fn=train_set.collate_fn,
    )
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
          f"{len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    # Optimizer
    model_params = ContiguousParams([p for p in model.parameters() if p.requires_grad]).contiguous()
    optimizer = torch.optim.Adam(model_params, args.lr,
                                 betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="text-recognition",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.model,
                "input_size": args.input_size,
                "optimizer": "adam",
                "exp_type": "text-recognition",
                "framework": "pytorch",
            }
        )

    # Create loss queue
    min_loss = np.inf
    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, mb)

        # Validation loop at the end of each epoch
        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            torch.save(model.state_dict(), f"./{exp_name}.pt")
            min_loss = val_loss
        mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                 f"(Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")
        # W&B
        if args.wb:
            wandb.log({
                'epochs': epoch + 1,
                'val_loss': val_loss,
                'exact_match': exact_match,
                'partial_match': partial_match,
            })
        #reset val metric
        val_metric.reset()

    if args.wb:
        run.finish()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR train text-recognition model (PyTorch)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_data_path', type=List[str], help='list of path(s) to train data folder')
    parser.add_argument('val_data_path', type=str, help='path to data folder')
    parser.add_argument('model', type=str, help='text-recognition model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('--input_size', type=int, default=32, help='input size H for the model, W = 4*H')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('-j', '--workers', type=int, default=None, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
    parser.add_argument("--test-only", dest='test_only', action='store_true', help="Run the validation loop")
    parser.add_argument('--show-samples', dest='show_samples', action='store_true',
                        help='Display unormalized training samples')
    parser.add_argument('--wb', dest='wb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Load pretrained parameters before starting the training')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
