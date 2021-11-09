# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ['USE_TORCH'] = '1'

import argparse
import json

import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from doctr.transforms.functional.pytorch import rotate
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import faster_model
from utilities import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train_dataset(Dataset):
    """
    Args:
        root_dir: location of image folder
        label_path: location of labels/annotations
    Returns:
        Iterable dataset accessible with integer indices.
    """

    def __init__(self, root_dir: str, label_path: str, val=False):
        self.root_dir = root_dir
        self.root = d_train if not val else d_val
        self.label_path = label_path
        with open(self.label_path) as f:
            self.my_json = dict(json.load(f))

    def __len__(self):
        return len(self.root)

    def __getitem__(self, idx):
        target_ = label_formatter({self.root[idx]: self.my_json[self.root[idx]]}, width, height)
        boxes = np.array(target_[self.root[idx]]['boxes'])
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        bbox = [torch.tensor(p, dtype=torch.float32) for p in boxes]
        return {"image": image,
                "target": {"boxes": bbox, "labels": target_[self.root[idx]]['labels']}}

class FocalLoss(nn.Module):
    """
    Focal Loss to be replaced with the BCE loss from the Faster Rcnn
    """

    def __init__(self, alpha=0.25, gamma=2, ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs):
        loss = [l for l in inputs.values()]
        pt = torch.exp(-loss[0])
        F_loss_1 = self.alpha * (1 - pt) ** self.gamma * loss[0]
        pt_2 = torch.exp(-loss[2])
        F_loss_2 = self.alpha * (1 - pt_2) ** self.gamma * loss[2]
        return sum([F_loss_1, loss[1], F_loss_2, loss[3]])


fl = FocalLoss()


class Faster_netv3(nn.Module):
    '''
    nn Module wrapping faster rcnn, mobilenet_v3_320_fpn
    '''

    def __init__(self):
        super().__init__()
        self.model = faster_model.to(device)

    def forward(self, x, y=False):
        return self.model(x) if y == False else self.model(x, y)


def train_faster(root_dir: str, label_path: str, num_epochs: int, early_stop=False):
    """Training script
    Args:
        root_dir: location of image folder
        label_path: location of label folder
        num_epochs: number of epochs in training
    Returns:
        torch checkpoint
    """
    model = Faster_netv3()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    epch_pq = 0.0
    epch_precision = 0.0
    epch_recall = 0.0
    epch_seg = 0.0
    epch_train_loss = None
    epch_loss = None
    for i in range(num_epochs):
        mean_loss = []
        mean_pq = []
        mean_precision = []
        mean_recall = []
        mean_seg = []
        model.train()
        tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
        for images, targets in tk0:
        for inpu in tk0:
            tk0.set_description(f"Epoch:{i}_Train")
            optimizer.zero_grad()
            x = inpu['image'].to(device)
            y = inpu['target']
            targets = [torch.stack(j, dim=1)[0].to(device) for i, j in y.items()]
            d = {}
            d.update({"boxes": targets[0], "labels": targets[1]})
            loss_dict = model(x, [d])
            loss = fl(loss_dict)
            mean_loss.append(loss)
            loss.backward()
            optimizer.step()
            # lo1 = loss.cpu().clone().detach().numpy()
            tk0.set_postfix(loss=loss, eph_loss=epch_loss, prec=epch_precision,
                            rec=epch_recall, seg=epch_seg)
            time.sleep(0.1)
        epch_loss = torch.sum(torch.stack(mean_loss, dim=0), dim=0) / len(train_dataloader)
        # epch_train_loss = epch_loss.cpu().clone().detach().numpy()
        tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
        scheduler.step()
        for inputs in tk1:
            tk1.set_description(f"Epoch:{i}_Val")
            model.eval()
            x, y = inputs['image'].to(device), inputs['target']
            targets = [torch.stack(j, dim=1)[0].to(device) for i, j in y.items()]
            pq_metric, seg_quality, recall, precision = val_metric(targets=targets, model=model, x=x)
            mean_pq.append(pq_metric)
            mean_seg.append(seg_quality)
            mean_recall.append(recall)
            mean_precision.append(precision)
        epch_pq = np.sum(np.array(mean_pq)) / len(val_dataloader)
        epch_precision = np.sum(np.array(mean_precision)) / len(val_dataloader)
        epch_recall = np.sum(np.array(mean_recall)) / len(val_dataloader)
        epch_seg = np.sum(np.array(mean_seg)) / len(val_dataloader)
    # torch.save(model, "/home/siddhant/PycharmProjects/pythonProject/sid/nd_small_qr_dm_30.ckpt")


def main(args):
    model = Faster_netv3()
    train_dir = args.train_path
    val_dir = args.val_path
    label_path = args.label_path
    new = os.listdir(root_dir)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    global d_train, d_val
    d_train, d_val = train_test_split(new, test_size=0.2, random_state=444, shuffle=True)
    global train_dataloader, val_dataloader, val_dataset, train_dataset
    train_dataset = Train_dataset(root_dir=root_dir, label_path=label_path, val=False)
    train_set = DocArtefaxcts(train=True, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=14, shuffle=False)
    val_dataset = Train_dataset(root_dir=root_dir, label_path=label_path, val=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=14, shuffle=False)
    train_faster(root_dir=root_dir, label_path=label_path, num_epochs=args.epochs, early_stop=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR training script for object detection (PyTorch)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_path', type=str, help='path to training data folder')
    parser.add_argument('val_path', type=str, help='path to validation data folder')
    parser.add_argument('arch', type=str, help='text-detection model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    # parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--device', default=None, type=int, help='device')
    # parser.add_argument('--input_size', type=int, default=1024, help='model input size, H = W')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('-j', '--workers', type=int, default=None, help='number of workers used for dataloading')
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
