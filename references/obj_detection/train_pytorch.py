# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ['USE_TORCH'] = '1'

import torch
import torch.optim as optim
import torchvision
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.ops import MultiScaleRoIAlign
from utils import val_metric

from doctr.datasets import DocArtefacts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def absolute(images, targets):
    height, width = images.shape[2], images.shape[3]

    for idx in range(images.shape[0]):
        targets[idx]['boxes'][:, 0::2] = (targets[idx]['boxes'][:, 0::2] * width).round()
        targets[idx]['boxes'][:, 1::2] = (targets[idx]['boxes'][:, 1::2] * height).round()

    targets = [{
        "boxes": torch.from_numpy(t['boxes']).to(device, dtype=torch.float32),
        "labels": torch.tensor(t['labels']).to(device, dtype=torch.long)}
        for t in targets
    ]


    return targets


def fit_one_epoch(model_, train_loader, optimizer, scheduler, mb, ):
    model_.train()
    train_iter = iter(train_loader)
    # Iterate over the batches of the dataset
    for _ in progress_bar(range(len(train_loader)), parent=mb):
        images, targets = next(train_iter)
        optimizer.zero_grad()
        target_ = absolute(images, targets)
        
        # import ipdb;
        # ipdb.set_trace()
        loss_dict = model_(images.to(device), target_)
        loss = sum(v for v in loss_dict.values())
        loss.backward()
        optimizer.step()
        mb.child.comment = f'Train_loss: {loss.item()}'
    scheduler.step()


@torch.no_grad()
def evaluate(model_, val_loader, val_metric_, mb):
    model_.eval()
    val_iter = iter(val_loader)
    pq, seg, rec, pre = 0, 0, 0, 0
    for _ in progress_bar(range(len(val_loader)), parent=mb):
        images, targets = next(val_iter)
        images = images.to(device)
        targets = absolute(images, targets)
        pq_metric, seg_quality, recall, precision = val_metric_(targets=targets, model=model_, x=images)
        pq += pq_metric
        seg += seg_quality
        rec += recall
        pre += precision
    epch_pq = pq / len(val_loader)
    epch_precision = pre / len(val_loader)
    epch_recall = rec / len(val_loader)
    epch_seg = seg / len(val_loader)
    return epch_pq, epch_seg, epch_recall, epch_precision


def main(args):

    model = torchvision.models.detection.__dict__[args.arch](pretrained=True)

    # Filter keys
    state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('roi_heads.')}
    defaults = {"min_size": 800, "max_size": 1300,
                "box_fg_iou_thresh": 0.5,
                "box_bg_iou_thresh": 0.5,
                "box_detections_per_img": 150, "box_score_thresh": 0.15, "box_positive_fraction": 0.35,
                "box_nms_thresh": 0.2,
                "rpn_pre_nms_top_n_train": 2000, "rpn_pre_nms_top_n_test": 1000,
                "rpn_post_nms_top_n_train": 2000, "rpn_post_nms_top_n_test": 1000,
                "rpn_nms_thresh": 0.2,
                "rpn_batch_size_per_image": 250
                }
    kwargs = {**defaults}

    faster_model = torchvision.models.detection.__dict__[args.arch](pretrained=False, num_classes=5, **kwargs)
    faster_model.load_state_dict(state_dict, strict=False)
    faster_model.roi_heads.box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7),
                                                             sampling_ratio=2)
    anchor_sizes = ((16), (64), (128), (264))
    aspect_ratios = ((0.5, 1.0, 2.0, 3.0,)) * len(anchor_sizes)
    faster_model.rpn.anchor_generator.sizes = anchor_sizes
    faster_model.rpn.anchor_generator.aspect_ratios = aspect_ratios
    faster_model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    train_set = DocArtefacts(train=True, download=True)
    val_set = DocArtefacts(train=False, download=True)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                  sampler=RandomSampler(train_set), pin_memory=True, collate_fn=train_set.collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                sampler=SequentialSampler(val_set), pin_memory=True, collate_fn=val_set.collate_fn)

    # W&B
    if args.wb:
        run = wandb.init(
            name="Artefacts",
            project="object-detection",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "input_size": args.input_size,
                "optimizer": "sgd",
                "framework": "pytorch",
                "scheduler": args.sched,
                "pretrained": args.pretrained,
            }
        )

    mb = master_bar(range(args.epochs))

    for epoch in mb:
        fit_one_epoch(model_=faster_model, train_loader=train_dataloader, optimizer=optimizer,
                      scheduler=scheduler, mb=mb)
        epch_pq, epch_seg, epch_recall, epch_precision = evaluate(model_=faster_model, val_loader=val_dataloader,
                                                                  val_metric_=val_metric, mb=mb)
        mb.write(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Recall: {epch_recall:.2%} | Precision: {epch_precision:.2%} "
            f"|Segmentation: {epch_seg:.2%}| Panoptic_Q: {epch_pq:.2%}) |")
        # W&B
        if args.wb:
            wandb.log({
                'recall': epch_recall,
                'precision': epch_precision,
                'seg': epch_seg,
                'pq': epch_pq,
            })

    if args.wb:
        run.finish()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR training script for object detection (PyTorch)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('arch', type=str, help='text-detection model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--device', default=None, type=int, help='device')
    # parser.add_argument('--input_size', type=int, default=1024, help='model input size, H = W')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (SGD)')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('-j', '--workers', type=int, default=14, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
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
