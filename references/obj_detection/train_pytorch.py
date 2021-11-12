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

from doctr.datasets import DocArtefacts
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

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.root = os.listdir(root_dir)
        self.train_folder = "self.root_dir/images"
        self.im_names = os.listdir(self.train_folder)
        self.label_path = "self.root_dir/labels.json"
        with open(self.label_path) as f:
            self.my_json = dict(json.load(f))

    def __len__(self):
        return len(self.root)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.train_folder, self.im_names[idx]))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return (image,self.my_json,self.root[idx])

class Val_dataset(Dataset):
    """
    Args:
        root_dir: location of image folder
        label_path: location of labels/annotations
    Returns:
        Iterable dataset accessible with integer indices.
    """

    def __init__(self, root_dir: str,):
        self.root_dir = root_dir
        self.root = os.listdir(root_dir)
        self.val_folder = "self.root_dir/images"
        self.im_names = os.listdir(self.val_folder)
        self.label_path = "self.root_dir/labels.json"
        with open(self.label_path) as f:
            self.my_json = dict(json.load(f))

    def __len__(self):
        return len(self.root)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.val_folder, self.im_names[idx]))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return (image,self.my_json,self.root[idx])

def fit_one_epoch(mymodel,images,targets,names,label_formatter):
    x = images.to(device)
    height, width = x.shape[0], x.shape[1]
    target_ = label_formatter({names: targets[names]}, width, height)
    boxes = np.array(target_[names]['boxes'])
    bbox = [torch.tensor(p, dtype=torch.float32) for p in boxes]
    y = {"boxes": bbox, "labels": target_[names]['labels']}
    targets = [torch.stack(j, dim=1)[0].to(device) for i, j in y.items()]
    d = {}
    d.update({"boxes": targets[0], "labels": targets[1]})
    loss_dict = mymodel(x, [d])
    loss = sum([l for k, l in loss_dict.items()])
    return loss

def evaluate(images,targets,names,val_metricm,mymodel):
    x, y = images.to(device), targets
    targets = [torch.stack(j, dim=1)[0].to(device) for i, j in y.items()]
    pq_metric, seg_quality, recall, precision = val_metric(targets=targets, model=mymodel, x=x)
    return pq_metric,seg_quality,recall,precision


def train_faster(mymodel:nn.Module,root_dir: str, label_path: str, num_epochs: int, early_stop=False):
    """Training script
    Args:
        root_dir: location of image folder
        label_path: location of label folder
        num_epochs: number of epochs in training
    Returns:
        torch checkpoint
    """
    model = mymodel
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
        for images, targets, names in tk0:
            tk0.set_description(f"Epoch:{i}_Train")
            optimizer.zero_grad()
            loss = fit_one_epoch(mymodel, images, targets, names, label_formatter)
            mean_loss.append(loss)
            loss.backward()
            optimizer.step()
            lo1 = loss.cpu().clone().detach().numpy()
            tk0.set_postfix(loss=lo1, eph_loss=epch_train_loss, prec=epch_precision,
                            rec=epch_recall, seg=epch_seg)
            time.sleep(0.1)
        epch_loss = torch.sum(torch.stack(mean_loss, dim=0), dim=0) / len(train_dataloader)
        epch_train_loss = epch_loss.cpu().clone().detach().numpy()
        tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
        scheduler.step()
        for images,targets,names in tk1:
            tk1.set_description(f"Epoch:{i}_Val")
            model.eval()
            pq_metric, seg_quality, recall, precision= evaluate(images, targets, names, val_metric, mymodel)
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
    model = faster_model
    train_dir = args.train_path
    val_dir = args.val_path
    use_DocArtefacts = args.use_DocArtefacts
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    global train_dataloader, val_dataloader, val_dataset, train_dataset
    train_dataset = Train_dataset(root_dir=train_dir)
    if use_DocArtefacts:
        train_set = DocArtefacts(train=True, download=True)
        val_set = DocArtefacts(train=False, download=True)
        train_dataloader = DataLoader(train_set, batch_size=1, num_workers=14, shuffle=False)
        val_dataloader = DataLoader(val_set, batch_size=1, num_workers=14, shuffle=False)
    else:
        train_dataset=Train_dataset(root_dir=train_dir)
        val_dataset = Val_dataset(root_dir=val_dir)
        train_dataloader = DataLoader(train_dataset batch_size=1, num_workers=14, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=14, shuffle=False)
    train_faster(root_dir=root_dir, label_path=label_path, num_epochs=args.epochs, early_stop=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR training script for object detection (PyTorch)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_path', type=str, help='path to training data folder')
    parser.add_argument('val_path', type=str, help='path to validation data folder')
    #parser.add_argument('arch', type=str, help='text-detection model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('--use_DocArtefacts', nargs='?', const=False, type=bool,
                        help="Optional:If user wants to use DocArtefacts by DocTR")
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
    if args.use_DocArtefacts and (args.train_path is None):
        parser.error("Please provide the path to the train+val datasets since you have opted to not use datasets provided by docTR."

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
