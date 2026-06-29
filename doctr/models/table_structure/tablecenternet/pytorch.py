# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: architecture and loss ported from https://github.com/dreamy-xay/TableCenterNet

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from ....models.classification import starnet_s3
from ...modules.layers.pytorch import DCNv2
from ...utils import _bf16_to_float32, load_pretrained_params
from .base import TableCenterNetPostProcessor, _TableCenterNet

__all__ = ["TableCenterNet", "tablecenternet"]

default_cfgs: dict[str, dict[str, Any]] = {
    "tablecenternet": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/mindee/doctr/releases/download/v1.0.1/tablecenternet-27736590.pt",
    },
}

# Helper functions


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Gather features at specific indices."""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Transpose and gather features at specific indices."""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


# Layers


class DeformConv(nn.Module):
    """A deformable convolution layer, as described in `<https://arxiv.org/abs/1703.06211>`_.

    Args:
        chi: number of input channels
        cho: number of output channels
    """

    def __init__(self, chi: int, cho: int):
        super().__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=0.1), nn.ReLU(inplace=True))
        self.conv = DCNv2(chi, cho, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actf(self.conv(x))


class IDAUp(nn.Module):
    """Iterative Deep Aggregation for Upsampling, as described in `<https://arxiv.org/abs/1707.06484>`_.

    Args:
        o: number of output channels
        channels: list of number of channels for each input feature map
        up_f: list of upsampling factors for each input feature map
    """

    def __init__(self, o: int, channels: list[int], up_f: list[int]):
        super().__init__()
        for i in range(1, len(channels)):
            c, f = channels[i], int(up_f[i])
            setattr(self, "proj_" + str(i), DeformConv(c, o))
            setattr(self, "node_" + str(i), DeformConv(o, o))
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False)
            setattr(self, "up_" + str(i), up)

    def forward(self, layers: list[torch.Tensor], startp: int, endp: int) -> None:
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i - 1] + layers[i])


class DLAUp(nn.Module):
    """Deep Layer Aggregation for Upsampling, as described in `<https://arxiv.org/abs/1707.06484>`_.

    Args:
        startp: index of the first backbone map fed to the decoder
        channels: list of number of channels for each input feature map
        scales: list of upsampling factors for each input feature map
        in_channels: list of number of input channels for each input feature map (optional)
    """

    def __init__(self, startp: int, channels: list[int], scales: list[int], in_channels: list[int] | None = None):
        super().__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        np_scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, f"ida_{i}", IDAUp(channels[j], in_channels[j:], (np_scales[j:] // np_scales[j]).tolist()))
            np_scales[j + 1 :] = np_scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers: list[torch.Tensor]) -> list[torch.Tensor]:
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, f"ida_{i}")
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


# Model


class TableCenterNet(nn.Module, _TableCenterNet):
    """TableCenterNet for table-structure recognition, as described in the official implementation
    `<https://github.com/dreamy-xay/TableCenterNet>`_.

    A StarNet backbone feeds a deformable-convolution DLA decoder, followed by six dense heads
    (`hm`, `reg`, `ct2cn`, `cn2ct`, `lc`, `sp`) describing cell centers, corners and their
    logical coordinates.

    Args:
        feat_extractor: the StarNet backbone serving as feature extractor (returns the stem + 4 stage maps)
        heads: mapping from head name to its number of output channels
        head_conv: number of channels in the hidden layer of each head
        first_level: index of the first backbone map fed to the decoder
        last_level: index (exclusive) of the last backbone map fed to the decoder
        center_thresh: minimum score for a cell center to be kept
        corner_thresh: minimum score for a corner to be used during relocation
        center_k: maximum number of cell centers
        corner_k: maximum number of corners
        not_relocate: if True, skip the corner-relocation step
        assume_straight_pages: if True, the predictor will fit straight boxes to the cells
        exportable: onnx exportable returns only the raw head maps
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        heads: dict[str, int] = {"hm": 2, "reg": 2, "ct2cn": 8, "cn2ct": 8, "lc": 2, "sp": 2},
        head_conv: int = 256,
        first_level: int = 1,
        last_level: int = 4,
        center_thresh: float = 0.3,
        corner_thresh: float = 0.3,
        center_k: int = 3000,
        corner_k: int = 5000,
        not_relocate: bool = False,
        max_objects: int = 300,
        max_corners: int = 1200,
        assume_straight_pages: bool = False,
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages
        self.cfg = cfg
        self.first_level, self.last_level = first_level, last_level
        self.center_k, self.corner_k = center_k, corner_k
        self.max_objects, self.max_corners = max_objects, max_corners

        self.feat_extractor = feat_extractor
        # Identify the number of channels for the decoder initialization
        _is_training = self.feat_extractor.training
        self.feat_extractor = self.feat_extractor.eval()
        with torch.no_grad():
            out = self.feat_extractor(torch.zeros((1, 3, 256, 256)))
            channels = [v.shape[1] for v in out.values()]
        if _is_training:
            self.feat_extractor = self.feat_extractor.train()

        scales = [2**i for i in range(len(channels[first_level:]))]
        self.dla_up = DLAUp(first_level, channels[first_level:], scales)
        out_channel = channels[first_level]
        self.ida_up = IDAUp(
            out_channel, channels[first_level:last_level], [2**i for i in range(last_level - first_level)]
        )
        for head, out_ch in self.heads.items():
            fc = nn.Sequential(
                nn.Conv2d(channels[first_level], head_conv, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_ch, 1, stride=1, padding=0, bias=True),
            )
            # Reference head initialisation: detection-style bias for heatmaps, zeroed bias otherwise
            final = fc[2]
            if isinstance(final, nn.Conv2d) and final.bias is not None:
                nn.init.constant_(final.bias, -2.19 if "hm" in head else 0.0)
            self.__setattr__(head, fc)

        self.postprocessor = TableCenterNetPostProcessor(
            center_thresh=center_thresh,
            corner_thresh=corner_thresh,
            not_relocate=not_relocate,
            assume_straight_pages=self.assume_straight_pages,
        )

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)

    def _polygons_decode(self, heatmap: torch.Tensor, vec: torch.Tensor, reg: torch.Tensor, k: int):
        """Decode key-points (cell centers or corners) into the four points of a quadrilateral."""
        batch = heatmap.size(0)
        # NMS on heatmaps
        pad = (3 - 1) // 2
        hmax = F.max_pool2d(heatmap, (3, 3), stride=1, padding=pad)
        heatmap = heatmap * (hmax == heatmap).float()
        # Top-K key-points
        batch, cat, height, width = heatmap.size()
        k = min(k, height * width)  # never request more points than there are locations
        topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), k)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()
        scores, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
        indexes = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
        ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
        xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

        scores = scores.view(batch, k, 1)
        offset = _transpose_and_gather_feat(reg, indexes)
        xs = xs.view(batch, k, 1) + offset[:, :, 0:1]
        ys = ys.view(batch, k, 1) + offset[:, :, 1:2]
        v = _transpose_and_gather_feat(vec, indexes)
        polygons = torch.cat(
            [
                xs - v[..., 0:1],
                ys - v[..., 1:2],
                xs - v[..., 2:3],
                ys - v[..., 3:4],
                xs - v[..., 4:5],
                ys - v[..., 5:6],
                xs - v[..., 6:7],
                ys - v[..., 7:8],
            ],
            dim=2,
        )
        return scores, indexes, xs, ys, polygons

    def _forward_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model and return the raw head maps."""
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        layers = self.dla_up(feats)
        y = [layers[i].clone() for i in range(self.last_level - self.first_level)]
        self.ida_up(y, 0, len(y))
        return {head: self.__getattr__(head)(y[-1]) for head in self.heads}  # type: ignore[operator]

    @torch.compiler.disable
    def _decode(self, heads: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Decode the raw head maps into cell polygons, scores, logical coordinates and corner points."""
        hm = heads["hm"].sigmoid()
        reg = heads["reg"]
        c_scores, c_ind, _, _, c_poly = self._polygons_decode(hm[:, 0:1], heads["ct2cn"], reg, self.center_k)
        k_scores, k_ind, k_xs, k_ys, k_poly = self._polygons_decode(hm[:, 1:2], heads["cn2ct"], reg, self.corner_k)
        spans = _transpose_and_gather_feat(heads["sp"], c_ind)
        corner_logics = _transpose_and_gather_feat(heads["lc"], k_ind)
        feat_h, feat_w = hm.shape[2], hm.shape[3]

        def _np(t: torch.Tensor) -> np.ndarray:
            # Cast to float32 first: relevant under autocast/AMP
            return t.detach().float().cpu().numpy()

        return {
            "center_polygons": _np(c_poly),
            "center_scores": _np(c_scores.squeeze(-1)),
            "center_spans": _np(spans),
            "corner_polygons": _np(k_poly),
            "corner_scores": _np(k_scores.squeeze(-1)),
            "corner_points": _np(torch.cat([k_xs, k_ys], dim=2)),
            "corner_logics": _np(corner_logics),
            "lc": _np(heads["lc"]),
            "feat_size": (feat_h, feat_w),
        }

    def forward(
        self,
        x: torch.Tensor,
        target: list[dict[str, np.ndarray]] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        heads_out = self._forward_heads(x)

        heads_out = {head: _bf16_to_float32(heads_out[head]) for head in self.heads}  # cast to float32 (AMP safe-guard)

        out: dict[str, Any] = {}

        if self.exportable:
            return heads_out

        if return_model_output:
            # Cast to float32 (the heads can be bfloat16/float16 under autocast)
            out["out_map"] = heads_out

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable
            def _postprocess(heads_out):
                return self.postprocessor(self._decode(heads_out))

            out["preds"] = _postprocess(heads_out)

        if target is not None:
            # Disable for torch.compile compatibility (the target rendering relies on numpy/scipy)
            @torch.compiler.disable
            def _compute_loss(heads_out, target):
                return self.compute_loss(heads_out, target)

            out["loss"] = _compute_loss(heads_out, target)

        return out

    def compute_loss(
        self,
        output: dict[str, torch.Tensor],
        target: list[dict[str, np.ndarray]],
    ) -> torch.Tensor:
        """Compute the multi-task TableCenterNet loss.

        Args:
            output: the raw head maps returned by the model
            target: one `{"cells": (N, 4, 2) relative polygons, "logic": (N, 4)}` dict per image

        Returns:
            the scalar training loss
        """
        out_h, out_w = int(output["hm"].shape[-2]), int(output["hm"].shape[-1])
        # Render the dense targets (numpy/scipy) from the relative cell annotations
        dense_np = self.build_target(target, (out_h, out_w))
        device = output["hm"].device
        dense = {k: torch.from_numpy(v).to(device) for k, v in dense_np.items()}
        # AMP safe-guard: compute the loss in float32
        output = {k: v.float() for k, v in output.items()}
        return self._loss_from_dense(output, dense)

    def _loss_from_dense(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the multi-task TableCenterNet loss.

        Args:
            output: the raw head maps returned by the model
            target: dense targets matching the reference schema (
                keys:
                hm, reg, reg_ind, reg_mask, ct_ind, ct_mask,
                ct2cn, cn_ind, cn_mask, cn2ct, ct_cn_ind,
                lc, lc_mask, lc_ind, lc_span
            )

        Returns:
            the scalar training loss
        """
        eps = 1e-4
        hm = torch.clamp(output["hm"].sigmoid(), min=eps, max=1 - eps)

        # Focal loss on the center/corner heat-maps
        gt = target["hm"]
        pos_inds, neg_inds = gt.eq(1).float(), gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)
        pos_loss = (torch.log(hm) * torch.pow(1 - hm, 2) * pos_inds).sum()
        neg_loss = (torch.log(1 - hm) * torch.pow(hm, 2) * neg_weights * neg_inds).sum()
        num_pos = pos_inds.sum()
        hm_loss = -neg_loss if num_pos == 0 else -(pos_loss + neg_loss) / num_pos

        # L1 on the sub-pixel offsets
        reg_pred = _transpose_and_gather_feat(output["reg"], target["reg_ind"])
        reg_mask = target["reg_mask"].unsqueeze(2).expand_as(reg_pred).float()
        reg_loss = F.l1_loss(reg_pred * reg_mask, target["reg"] * reg_mask, reduction="sum") / (reg_mask.sum() + eps)

        # Vector-pair loss (center<->corner offsets)
        ct2cn_loss, cn2ct_loss, invalid_loss = self._vec_pair_loss(output, target, eps)

        # Logical-coordinate + span loss
        lc_coord_loss, span_diff_loss, span_loss = self._logic_coord_loss(output, target, eps)

        return (
            hm_loss + reg_loss + ct2cn_loss + (cn2ct_loss + invalid_loss) + (lc_coord_loss + span_diff_loss + span_loss)
        )

    @staticmethod
    def _vec_pair_loss(output, target, eps):
        ct2cn_pred = _transpose_and_gather_feat(output["ct2cn"], target["ct_ind"])
        cn2ct_pred = _transpose_and_gather_feat(output["cn2ct"], target["cn_ind"])
        cn2ct_pred_temp, cn2ct_gt_temp = cn2ct_pred, target["cn2ct"]
        b, m, n = ct2cn_pred.size(0), ct2cn_pred.size(1), cn2ct_pred.size(1)

        ct_cn_ind = target["ct_cn_ind"].unsqueeze(2).expand(b, 4 * m, 2)
        cn2ct_pred = cn2ct_pred.view(b, 4 * n, 2).gather(1, ct_cn_ind).view(b, m, 8)
        cn2ct_gt = target["cn2ct"].view(b, 4 * n, 2).gather(1, ct_cn_ind).view(b, m, 8)

        ct_mask = target["ct_mask"].unsqueeze(2).expand_as(ct2cn_pred).float()
        num_ct = ct_mask.sum() + eps
        cn_mask = target["cn_mask"].unsqueeze(2).expand_as(cn2ct_pred_temp)

        delta = (torch.abs(ct2cn_pred - target["ct2cn"]) + torch.abs(cn2ct_pred - cn2ct_gt)) / (
            torch.abs(target["ct2cn"]) + eps
        )
        weight = torch.sin(1.570796 * torch.min(delta, torch.tensor(1.0, device=delta.device)))
        ct2cn_loss = (
            F.l1_loss(ct2cn_pred * ct_mask * weight, target["ct2cn"] * ct_mask * weight, reduction="sum") / num_ct
        )
        cn2ct_loss = F.l1_loss(cn2ct_pred * ct_mask * weight, cn2ct_gt * ct_mask * weight, reduction="sum") / num_ct

        invalid_vec_mask = cn2ct_gt_temp == 0
        invalid_vec_cn_mask = (invalid_vec_mask == cn_mask).float()
        invalid_loss = F.l1_loss(
            cn2ct_pred_temp * invalid_vec_cn_mask, cn2ct_gt_temp * invalid_vec_cn_mask, reduction="sum"
        ) / (invalid_vec_cn_mask.sum() + eps)
        return ct2cn_loss, 0.5 * cn2ct_loss, 0.2 * invalid_loss

    @staticmethod
    def _logic_coord_loss(output, target, eps):
        coord, span = output["lc"], output["sp"]
        b, num = target["lc_span"].size(0), target["lc_span"].size(1)

        coords_pred = _transpose_and_gather_feat(coord, target["lc_ind"].view(b, num * 4)).view(b, num, 4, 2)
        cols_pred, rows_pred = coords_pred[..., 0], coords_pred[..., 1]
        span_pred = _transpose_and_gather_feat(span, target["ct_ind"])
        span_mask = target["ct_mask"].unsqueeze(2).expand(b, num, 2).float()
        num_span_mask = span_mask.sum() + eps

        coord_gt, coord_mask = target["lc"], target["lc_mask"]
        coord_weight = torch.square(1.0 - torch.abs(coord_gt - torch.round(coord_gt)))
        coord_loss = F.l1_loss(
            coord * coord_mask * coord_weight, coord_gt * coord_mask * coord_weight, reduction="sum"
        ) / (coord_mask.sum() + eps)

        col_span_diff_pred = cols_pred[..., [1, 2]] - cols_pred[..., [0, 3]]
        row_span_diff_pred = rows_pred[..., [3, 2]] - rows_pred[..., [0, 1]]
        col_span_pred = span_pred[..., 0].unsqueeze(2).expand(b, num, 2)
        row_span_pred = span_pred[..., 1].unsqueeze(2).expand(b, num, 2)
        col_span_gt = target["lc_span"][..., 0].unsqueeze(2).expand(b, num, 2)
        row_span_gt = target["lc_span"][..., 1].unsqueeze(2).expand(b, num, 2)

        def span_weight(out1, out2, tgt):
            scaled = (torch.abs(out1 - tgt) + torch.abs(out2 - tgt)) * 5.0
            delta = torch.min(scaled, torch.tensor(1.0, device=tgt.device))
            return torch.sin(1.570796 * delta)

        col_w = span_weight(col_span_pred, col_span_diff_pred, col_span_gt)
        row_w = span_weight(row_span_pred, row_span_diff_pred, row_span_gt)
        sp_weight = torch.stack([(col_w[..., 0] + col_w[..., 1]) / 2.0, (row_w[..., 0] + row_w[..., 1]) / 2.0], dim=-1)

        col_span_diff_loss = (
            F.l1_loss(col_span_diff_pred * span_mask * col_w, col_span_gt * span_mask * col_w, reduction="sum")
            / num_span_mask
        )
        row_span_diff_loss = (
            F.l1_loss(row_span_diff_pred * span_mask * row_w, row_span_gt * span_mask * row_w, reduction="sum")
            / num_span_mask
        )
        span_diff_loss = col_span_diff_loss + row_span_diff_loss

        span_loss = (
            F.l1_loss(span_pred * span_mask * sp_weight, target["lc_span"] * span_mask * sp_weight, reduction="sum")
            / num_span_mask
        )
        return coord_loss, span_diff_loss, span_loss


def _tablecenternet(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[..., nn.Module],
    pretrained_backbone: bool = True,
    **kwargs: Any,
) -> TableCenterNet:
    pretrained_backbone = pretrained_backbone and not pretrained
    backbone = backbone_fn(pretrained_backbone)
    # 4 stages - all starnet variants
    feat_extractor = IntermediateLayerGetter(backbone, {str(idx): str(idx) for idx in range(5)})

    model = TableCenterNet(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=None)

    return model


def tablecenternet(pretrained: bool = False, **kwargs: Any) -> TableCenterNet:
    """TableCenterNet with a StarNet-S3 backbone, matching the official checkpoint.

    >>> import torch
    >>> from doctr.models import tablecenternet
    >>> model = tablecenternet(pretrained=False)
    >>> out = model(torch.rand((1, 3, 1024, 1024), dtype=torch.float32), return_preds=True)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the TableCenterNet architecture

    Returns:
        A TableCenterNet model with a StarNet-S3 backbone
    """
    return _tablecenternet("tablecenternet", pretrained, starnet_s3, **kwargs)
