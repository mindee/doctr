# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn as nn

from doctr.file_utils import CLASS_NAME
from doctr.models.classification.textnet_fast.pytorch import textnetfast_tiny, textnetfast_small, textnetfast_base

from ...utils import load_pretrained_params
from .base import FastPostProcessor

__all__ = ["fast_tiny", "fast_small", "fast_base"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (3, 1024, 1024),
        "url": None,
    },
    "fast_small": {
        "input_shape": (3, 1024, 1024),
        "url": None,
    },
    "fast_base": {
        "input_shape": (3, 1024, 1024),
        "url": None,
    },
}
# reimplement FAST Class
# NECK AND TEXTNET AND HEAD READY

class FAST(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(FAST, self).__init__()
        self.backbone = backbone # okay
        self.neck = neck #okay
        self.det_head = detection_head
        
    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs, gt_texts=None, gt_kernels=None, training_masks=None,
                gt_instances=None, img_metas=None, cfg=None):
        outputs = dict()

        if not self.training:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # reduce channel
        f = self.neck(f)
        
        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))

        if self.training:
            det_out = self._upsample(det_out, imgs.size(), scale=1)
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks, gt_instances)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), scale=4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg, scale=2)
            outputs.update(det_res)

        return outputs


class FASTHead(nn.Module):
    def __init__(self, conv, final, pooling_size):
        super(FASTHead, self).__init__()
        self.conv = RepConvLayer(in_channels=512, out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1)

        self.final = ConvLayer(kernel_size=1, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False, in_channels=128, out_channels=5, use_bn=False, act_func=None,
                               dropout_rate=0, ops_order="weight")

        self.pooling_size = pooling_size

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.final(x)
        return x


class FASTNeck(nn.Module):
    def __init__(self, reduce_layers = [64, 128, 256, 512]):
        super(FASTNeck, self).__init__()
        
        self.reduce_layer1 = RepConvLayer(in_channels=reduce_layers[0], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1)
        self.reduce_layer2 = RepConvLayer(in_channels=reduce_layers[1], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1)
        self.reduce_layer3 = RepConvLayer(in_channels=reduce_layers[2], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1)
        self.reduce_layer4 = RepConvLayer(in_channels=reduce_layers[3], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')
    
    def forward(self, x):
        f1, f2, f3, f4 = x
        f1 = self.reduce_layer1(f1)
        f2 = self.reduce_layer2(f2)
        f3 = self.reduce_layer3(f3)
        f4 = self.reduce_layer4(f4)

        f2 = self._upsample(f2, f1)
        f3 = self._upsample(f3, f1)
        f4 = self._upsample(f4, f1)
        f = torch.cat((f1, f2, f3, f4), 1)
        return f


def _fast(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    pretrained_backbone: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> Fast:
    pretrained_backbone = pretrained_backbone and not pretrained
    
    feat_extractor =  backbone_fn(pretrained_backbone)
    neck = FASTNeck()
    head = FASTHead()
    
    if not kwargs.get("class_names", None):
        kwargs["class_names"] = default_cfgs[arch].get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])

    # Build the model
    model = Fast(feat_extractor, neck, head, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of class_names is not the same as the number of classes in the pretrained model =>
        # remove the layer weights
        _ignore_keys = (
            ignore_keys if kwargs["class_names"] != default_cfgs[arch].get("class_names", [CLASS_NAME]) else None
        )
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def fast_tiny(pretrained: bool = False, **kwargs: Any) -> Fast:
    """Fast architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import fast_tiny
    >>> model = fast_tiny(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_tiny",
        pretrained,
        textnetfast_tiny,
        # change ignore keys
        ignore_keys=[
            "classifier.6.weight",
            "classifier.6.bias",
        ],
        **kwargs,
    )
# set fast_small and fast_base
