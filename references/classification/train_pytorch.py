# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ['USE_TORCH'] = '1'

import datetime
import multiprocessing as mp
import time

import numpy as np
import torch
from contiguous_params import ContiguousParams
from fastprogress.fastprogress import master_bar, progress_bar
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import models
from torchvision.transforms import ColorJitter, Compose, Normalize, RandomPerspective
from utils import plot_samples

import wandb
from doctr import transforms as T
from doctr.datasets import VOCABS, CharacterGenerator


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb):
    model.train()
    train_iter = iter(train_loader)
    # Iterate over the batches of the dataset
    for _ in progress_bar(range(len(train_loader)), parent=mb):
        images, targets = next(train_iter)

        images = batch_transforms(images)

        out = model(images)
        train_loss = cross_entropy(out, targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        mb.child.comment = f'Training loss: {train_loss.item():.6}'


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms):
    # Model in eval mode
    model.eval()
    # Validation loop
    val_loss, correct, samples, batch_cnt = 0, 0, 0, 0
    val_iter = iter(val_loader)
    for images, targets in val_iter:
        images = batch_transforms(images)
        out = model(images)
        loss = cross_entropy(out, targets)
        # Compute metric
        correct += (out.argmax(dim=1) == targets).sum().item()

        val_loss += loss.item()
        batch_cnt += 1
        samples += images.shape[0]

    val_loss /= batch_cnt
    acc = correct / samples
    return val_loss, acc


def main(args):

    print(args)

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    vocab = VOCABS[args.vocab]

    # Load val data generator
    st = time.time()
    val_set = CharacterGenerator(
        vocab=vocab,
        num_samples=args.val_samples * len(vocab),
        cache_samples=True,
        sample_transforms=T.Resize((args.input_size, args.input_size)),
        font_family=args.font,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(val_set),
        pin_memory=True,
    )
    print(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in "
          f"{len(val_loader)} batches)")

    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load doctr model
    model = models.__dict__[args.model](pretrained=args.pretrained, num_classes=len(vocab))

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)

    if args.test_only:
        print("Running evaluation")
        val_loss, acc = evaluate(model, val_loader, batch_transforms)
        print(f"Validation loss: {val_loss:.6} (Acc: {acc:.2%})")
        return

    st = time.time()

    # Load train data generator
    train_set = CharacterGenerator(
        vocab=vocab,
        num_samples=args.train_samples * len(vocab),
        cache_samples=True,
        sample_transforms=Compose([
            T.Resize((args.input_size, args.input_size)),
            # Augmentations
            RandomPerspective(),
            T.RandomApply(T.ColorInversion(), .7),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        ]),
        font_family=args.font,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=RandomSampler(train_set),
        pin_memory=True,
    )
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
          f"{len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, list(map(vocab.__getitem__, target)))
        return

    # Optimizer
    model_params = ContiguousParams([p for p in model.parameters() if p.requires_grad]).contiguous()
    optimizer = torch.optim.Adam(model_params, args.lr,
                                 betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    # Scheduler
    if args.sched == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4)
    elif args.sched == 'onecycle':
        scheduler = OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader))

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="character-classification",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.model,
                "input_size": args.input_size,
                "optimizer": "adam",
                "exp_type": "character-classification",
                "framework": "pytorch",
            }
        )

    # Create loss queue
    min_loss = np.inf
    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb)

        # Validation loop at the end of each epoch
        val_loss, acc = evaluate(model, val_loader, batch_transforms)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            torch.save(model.state_dict(), f"./{exp_name}.pt")
            min_loss = val_loss
        mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} (Acc: {acc:.2%})")
        # W&B
        if args.wb:
            wandb.log({
                'epochs': epoch + 1,
                'val_loss': val_loss,
                'acc': acc,
            })

    if args.wb:
        run.finish()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR training script for character classification (PyTorch)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model', type=str, help='text-recognition model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('--input_size', type=int, default=32, help='input size H for the model, W = H')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('-j', '--workers', type=int, default=None, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
    parser.add_argument('--font', type=str, default="FreeMono.ttf", help='Font family to be used')
    parser.add_argument('--vocab', type=str, default="french", help='Vocab to be used for training')
    parser.add_argument(
        '--train-samples',
        dest='train_samples',
        type=int,
        default=1000,
        help='Multiplied by the vocab length gets you the number of training samples that will be used.'
    )
    parser.add_argument(
        '--val-samples',
        dest='val_samples',
        type=int,
        default=20,
        help='Multiplied by the vocab length gets you the number of validation samples that will be used.'
    )
    parser.add_argument("--test-only", dest='test_only', action='store_true', help="Run the validation loop")
    parser.add_argument('--show-samples', dest='show_samples', action='store_true',
                        help='Display unormalized training samples')
    parser.add_argument('--wb', dest='wb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Load pretrained parameters before starting the training')
    parser.add_argument('--sched', type=str, default='cosine', help='scheduler to use')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
