# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from PIL import Image

from doctr.file_utils import CLASS_NAME
from doctr.models.classification.textnet_fast.pytorch import textnetfast_tiny
from doctr.models.modules.layers.pytorch import ConvLayer, RepConvLayer
from doctr.utils.metrics import box_iou

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

# implement FastPostProcessing class with get_results head class


class FAST(nn.Module):
    def __init__(
        self,
        feat_extractor,
        bin_thresh: float = 0.1,
        head_chans: int = 32,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
        class_names: List[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.cfg = cfg
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages
        self.feat_extractor = feat_extractor
        self.feat_extractor.train()
        self.fpn = FASTNeck()
        self.classifier = FASTHead()
        self.postprocessor = FastPostProcessor(assume_straight_pages=self.assume_straight_pages, bin_thresh=bin_thresh)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[np.ndarray]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, torch.Tensor]:

        x, gt_texts, gt_kernels, training_masks, gt_instances, img_metas = self.prepare_data(x, target)

        feats = self.backbone(x)  
        logits = self.fpn(feats)
        logits = self.classifier(logits)
        logits = self._upsample(logits, x.size(), scale=1)

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = torch.sigmoid(logits)
            
        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = [
                dict(zip(self.class_names, preds))
                for preds in self.postprocessor(prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy(), img_metas, cfg, scale=2)
            ]

        if target is not None:
            loss = self.compute_loss(logits, gt_texts, gt_kernels, training_masks, gt_instances)
            out["loss"] = loss

        return out

    def compute_loss(self, out_map: torch.Tensor, target: List[np.ndarray]) -> torch.Tensor:
        # IL MANQUE CES PARAMATRES (gt_kernels, training_masks, gt_instances)

        # output
        kernels = out_map[:, 0, :, :]  # 4*640*640
        texts = self._max_pooling(kernels, scale=1)  # 4*640*640
        embs = out_map[:, 1:, :, :]  # 4*4*640*640

        # text loss
        loss_text = multiclass_dice_loss(texts, target, self.num_classes, loss_weight=0.25)
        iou_text = box_iou((texts > 0).long(), target)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        loss_kernel = multiclass_dice_loss(kernels, None, self.num_classes, loss_weight=1.0)
        loss_kernel = torch.mean(loss_kernel, dim=0)
        iou_kernel = box_iou((kernels > 0).long(), None)
        losses.update(dict(loss_kernels=loss_kernel, iou_kernel=iou_kernel))

        # auxiliary loss
        loss_emb = emb_loss_v2(embs, None, None, None)
        losses.update(dict(loss_emb=loss_emb))

        return losses

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode="bilinear")

    def prepare_data(self,
        x: torch.Tensor,
        target: Optional[List[np.ndarray]] = None):

        target = target[:self.num_classes]
        gt_instance = np.zeros(x.shape[0:2], dtype='uint8')
        training_mask = np.ones(x.shape[0:2], dtype='uint8')

        if target.shape[0] > 0:
            target = np.reshape(target * ([x.shape[1], x.shape[0]] * 4),
                                (target.shape[0], -1, 2)).astype('int32')
            for i in range(target.shape[0]):
                cv2.drawContours(gt_instance, [target[i]], -1, i + 1, -1)

        gt_kernels = np.array([np.zeros(x.shape[0:2], dtype='uint8')] * len(target)) # [instance_num, h, w]
        gt_kernel = self.min_pooling(gt_kernels)

        shrink_kernel_scale = 0.1
        gt_kernel_shrinked = np.zeros(x.shape[0:2], dtype='uint8')
        kernel_target = shrink(target, shrink_kernel_scale)
        
        for i in range(target.shape[0]):
            cv2.drawContours(gt_kernel_shrinked, [kernel_target[i]], -1, 1, -1)
        gt_kernel = np.maximum(gt_kernel, gt_kernel_shrinked)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1

        x = Image.fromarray(x)
        
        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
            img_size=np.array(img.shape[:2]),
            filename=filename))

        img = scale_aligned_short(img, self.short_size)
        x = x.convert('RGB')

        x = transforms.ToTensor()(x)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        return x, \ 
               torch.from_numpy(gt_text).long(), \
               torch.from_numpy(gt_kernel).long(), \
               torch.from_numpy(training_mask).long(), \
               torch.from_numpy(gt_instance).long()\
               img_meta
        
    # simplify this method
    def min_pooling(self, input):
        input = torch.tensor(input, dtype=torch.float)
        temp = input.sum(dim=0).to(torch.uint8)
        overlap = (temp > 1).to(torch.float32).unsqueeze(0).unsqueeze(0)
        overlap = self.overlap_pool(overlap).squeeze(0).squeeze(0)

        B = input.size(0)
        h_sum = input.sum(dim=2) > 0
        
        h_sum_ = h_sum.long() * torch.arange(h_sum.shape[1], 0, -1)
        h_min = torch.argmax(h_sum_, 1, keepdim=True)
        h_sum_ = h_sum.long() * torch.arange(1, h_sum.shape[1] + 1)
        h_max = torch.argmax(h_sum_, 1, keepdim=True)

        w_sum = input.sum(dim=1) > 0
        w_sum_ = w_sum.long() * torch.arange(w_sum.shape[1], 0, -1)
        w_min = torch.argmax(w_sum_, 1, keepdim=True)
        w_sum_ = w_sum.long() * torch.arange(1, w_sum.shape[1] + 1)
        w_max = torch.argmax(w_sum_, 1, keepdim=True)

        for i in range(B):
            region = input[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1]
            region = self.pad(region)
            region = -self.pooling(-region)
            input[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1] = region

        x = input.sum(dim=0).to(torch.uint8)
        x[overlap > 0] = 0  # overlapping regions
        return x.numpy()
       
class FASTHead(nn.Module):
    def __init__(self):
        super(FASTHead, self).__init__()
        self.conv = RepConvLayer(in_channels=512, out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1)

        self.final = ConvLayer(
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            has_shuffle=False,
            in_channels=128,
            out_channels=5,
            use_bn=False,
            act_func=None,
            dropout_rate=0,
            ops_order="weight",
        )

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
    def __init__(self, reduce_layers=[64, 128, 256, 512]):
        super(FASTNeck, self).__init__()

        self.reduce_layer1 = RepConvLayer(
            in_channels=reduce_layers[0], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1
        )
        self.reduce_layer2 = RepConvLayer(
            in_channels=reduce_layers[1], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1
        )
        self.reduce_layer3 = RepConvLayer(
            in_channels=reduce_layers[2], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1
        )
        self.reduce_layer4 = RepConvLayer(
            in_channels=reduce_layers[3], out_channels=128, kernel_size=[3, 3], stride=1, dilation=1, groups=1
        )
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
        return F.upsample(x, size=(H, W), mode="bilinear")

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
) -> FAST:
    pretrained_backbone = pretrained_backbone and not pretrained
 
    # corriger l'encapsulation du backbon neck et head
    backbone = backbone_fn(pretrained_backbone)
    FASTNeck()
    FASTHead()

    feat_extractor = backbone

    if not kwargs.get("class_names", None):
        kwargs["class_names"] = default_cfgs[arch].get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])

    # Build the model
    model = FAST(feat_extractor=feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of class_names is not the same as the number of classes in the pretrained model =>
        # remove the layer weights
        _ignore_keys = (
            ignore_keys if kwargs["class_names"] != default_cfgs[arch].get("class_names", [CLASS_NAME]) else None
        )
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def fast_tiny(pretrained: bool = False, **kwargs: Any) -> FAST:
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
            "classifier.final.conv.weight",
            "classifier.final.conv.bias",
        ],
        **kwargs,
    )


def fast_small(pretrained: bool = False, **kwargs: Any) -> FAST:
    """Fast architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import fast_small
    >>> model = fast_small(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_small",
        pretrained,
        textnetfast_tiny,
        # change ignore keys
        ignore_keys=[
            "classifier.final.conv.weight",
            "classifier.final.conv.bias",
        ],
        **kwargs,
    )


def fast_base(pretrained: bool = False, **kwargs: Any) -> FAST:
    """Fast architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import fast_base
    >>> model = fast_base(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_base",
        pretrained,
        textnetfast_tiny,
        # change ignore keys
        ignore_keys=[
            "classifier.final.conv.weight",
            "classifier.final.conv.bias",
        ],
        **kwargs,
    )


# verifier que le code fonction; cest le code de https://github.com/czczup/FAST/blob/main/models/loss/dice_loss.py
# faire en sorte d'inserer dans le code le selected_masks
def multiclass_dice_loss(inputs, targets, num_classes, loss_weight=1.0):
    # Convert targets to one-hot encoding
    targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Calculate intersection and union
    intersection = torch.sum(inputs * targets, dim=(2, 3))
    union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))

    # Calculate Dice coefficients for each class
    dice_coeffs = (2.0 * intersection + 1e-5) / (union + 1e-5)

    # Calculate the average Dice loss across all classes
    dice_loss = 1.0 - torch.mean(dice_coeffs)

    return loss_weight * dice_loss


# simplify emb_loss_v2
def emb_loss_v2(emb, instance, kernel, training_mask):
    training_mask = (training_mask > 0.5).long()
    kernel = (kernel > 0.5).long()
    instance = instance * training_mask
    instance_kernel = (instance * kernel).view(-1)
    instance = instance.view(-1)
    emb = emb.view(4, -1)

    unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
    num_instance = unique_labels.size(0)
    if num_instance <= 1:
        return 0

    emb_mean = emb.new_zeros((4, num_instance), dtype=torch.float32)
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind_k = instance_kernel == lb
        emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

    l_agg = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind = instance == lb
        emb_ = emb[:, ind]
        dist = (emb_ - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
        dist = F.relu(dist - 0.5) ** 2
        l_agg[i] = torch.mean(torch.log(dist + 1.0))
    l_agg = torch.mean(l_agg[1:])

    if num_instance > 2:
        emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
        emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, 4)

        mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, 4)
        mask = mask.view(num_instance, num_instance, -1)
        mask[0, :, :] = 0
        mask[:, 0, :] = 0
        mask = mask.view(num_instance * num_instance, -1)

        dist = emb_interleave - emb_band
        dist = dist[mask > 0].view(-1, 4).norm(p=2, dim=1)
        dist = F.relu(2 * 1.5 - dist) ** 2

        l_dis = [torch.log(dist + 1.0)]
        emb_bg = emb[:, instance == 0].view(4, -1)
        if emb_bg.size(1) > 100:
            rand_ind = np.random.permutation(emb_bg.size(1))[:100]
            emb_bg = emb_bg[:, rand_ind]
        if emb_bg.size(1) > 0:
            for i, lb in enumerate(unique_labels):
                if lb == 0:
                    continue
                dist = (emb_bg - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
                dist = F.relu(2 * 1.5 - dist) ** 2
                l_dis_bg = torch.mean(torch.log(dist + 1.0), 0, keepdim=True)
                l_dis.append(l_dis_bg)
        l_dis = torch.mean(torch.cat(l_dis))
    else:
        l_dis = 0
    l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
    loss = l_agg + l_dis + l_reg
    return loss

    def forward(self, emb, instance, kernel, training_mask, reduce=True):
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i], training_mask[i])

        loss_batch = 0.25 * loss_batch

        if reduce:
            loss_batch = torch.mean(loss_batch)

        return loss_batch
