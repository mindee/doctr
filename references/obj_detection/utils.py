# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from operator import itemgetter
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from doctr.utils.metrics import LocalizationConfusion

__all__ = ["val_metric", ]


def val_metric(targets: List[Dict], model: nn.Module, x: List[torch.Tensor]):
    """
    Args:
        targets: Ground truth labels
        model: Faster Rcnn module
        x: Input image
    Returns:
         Panoptic Metric, Segmentation Quality, Recall, Precision
    """
    gt_box = targets[0].cpu().numpy()
    gt_label = targets[1].cpu().numpy()
    pop1 = model(x)
    pop = pop1.copy()
    pr_label = pop[0]['labels'].detach().cpu().numpy()
    pr_box = pop[0]['boxes'].detach().cpu().numpy()
    pr_score = pop[0]['scores'].detach().cpu().numpy()
    true_positive = int()
    false_negative = int()
    false_positive = int()
    temp_list = []
    iou_list = []
    for i in range(len(gt_label)):
        good = []
        for j in range(len(pr_label)):
            c = LocalizationConfusion(iou_thresh=0.5, )
            c.update(np.asarray([list(gt_box[i])]), np.asarray(list([pr_box[j]])))
            if c.summary()[2] >= 0.5:
                iou = c.summary()[2]
                good.append((iou, pr_box[j], pr_label[j], pr_score[j]))
                temp_list.append(list(pr_box[j]))
            else:
                pass
        if good:
            good.sort(key=itemgetter(3))
            arr_ = good[-1]
            if gt_label[i] == arr_[2]:
                true_positive += 1
                iou_list.append(arr_[0])
            else:
                false_negative += 1
    if temp_list != []:
        for i in pr_box:
            if list(i) not in temp_list:
                false_positive += 1
    else:
        false_positive = len(pr_label)

    if iou_list:
        try:
            net_iou = sum(iou_list)
        except TypeError:
            net_iou = iou_list
        pq_metric = net_iou / (true_positive + 0.5 * false_positive + 0.5 * false_negative)
        seg_quality = net_iou / true_positive
    else:
        net_iou = 0.0
        pq_metric = 0.0
        seg_quality = 0.0
    if false_negative == 0 and true_positive == 0:
        recall = 0.0
    else:
        recall = true_positive / (true_positive + false_negative)
    if false_positive == 0 and true_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)
    return pq_metric, seg_quality, recall, precision
