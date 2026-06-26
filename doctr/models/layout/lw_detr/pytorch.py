# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from doctr.models.classification import vit_det_m, vit_det_s

from ...utils import _bf16_to_float32, load_pretrained_params
from .base import _LWDETR, LWDETRPostProcessor
from .layers import (
    LWDETRDecoder,
    LWDETRHead,
    MultiScaleProjector,
    refine_obb_boxes,
)

__all__ = ["LWDETR", "lw_detr_s", "lw_detr_m"]


default_cfgs: dict[str, dict[str, Any]] = {
    "lw_detr_s": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "class_names": [
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
            "Checkbox-Selected",
            "Checkbox-Unselected",
        ],
        "url": None,
    },
    "lw_detr_m": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "class_names": [
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
            "Checkbox-Selected",
            "Checkbox-Unselected",
        ],
        "url": None,
    },
}


def _obb_covariance_components(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the components (a, b, c) of the Gaussian covariance matrix [[a, c], [c, b]] associated with
    oriented boxes in (cx, cy, w, h, sinθ, cosθ) format, following ProbIoU (https://arxiv.org/abs/2106.06072).

    Args:
        boxes: (..., 6) tensor of oriented boxes

    Returns:
        a, b, c: (...,) tensors of covariance components
    """
    w = boxes[..., 2].clamp(min=1e-6)
    h = boxes[..., 3].clamp(min=1e-6)
    rot = F.normalize(boxes[..., 4:6], dim=-1, eps=1e-6)
    sin, cos = rot[..., 0], rot[..., 1]

    var_w = w.pow(2) / 12.0
    var_h = h.pow(2) / 12.0
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)

    a = var_w * cos2 + var_h * sin2
    b = var_w * sin2 + var_h * cos2
    c = (var_w - var_h) * cos * sin
    return a, b, c


def _probiou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    pairwise: bool = False,
    scale: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the probabilistic IoU between oriented boxes in (cx, cy, w, h, sinθ, cosθ) format,
    as described in `"Gaussian Bounding Boxes and Probabilistic IoU" <https://arxiv.org/abs/2106.06072>`_.

    Args:
        boxes1: (N, 6) tensor of oriented boxes
        boxes2: (M, 6) tensor of oriented boxes (M = N if `pairwise` is False)
        pairwise: if True, return the (N, M) matrix of IoUs, otherwise the element-wise (N,) IoUs
        scale: factor applied to the spatial components (cx, cy, w, h) before computing the IoU.
            ProbIoU is mathematically scale-invariant, but the stabilizing `eps` terms are not: in
            normalized [0, 1] coordinates the covariance terms of small boxes (e.g. checkboxes) fall
            below `eps` and the result degenerates (non-overlapping small boxes can score ~1).
        eps: small value for numerical stability

    Returns:
        probiou: (N,) or (N, M) tensor of IoU-like similarities in [0, 1]
    """
    # AMP safe-guard
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    if scale != 1.0:
        boxes1 = torch.cat([boxes1[..., :4] * scale, boxes1[..., 4:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :4] * scale, boxes2[..., 4:]], dim=-1)

    x1, y1 = boxes1[..., 0], boxes1[..., 1]
    x2, y2 = boxes2[..., 0], boxes2[..., 1]
    a1, b1, c1 = _obb_covariance_components(boxes1)
    a2, b2, c2 = _obb_covariance_components(boxes2)

    if pairwise:
        x1, y1, a1, b1, c1 = (t.unsqueeze(-1) for t in (x1, y1, a1, b1, c1))  # (N, 1)
        x2, y2, a2, b2, c2 = (t.unsqueeze(-2) for t in (x2, y2, a2, b2, c2))  # (1, M)

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)).clamp(min=eps)
        / (4 * ((a1 * b1 - c1.pow(2)).clamp(min=eps) * (a2 * b2 - c2.pow(2)).clamp(min=eps)).sqrt() + eps)
        + eps
    ).log() * 0.5

    bhattacharyya = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    hellinger = (1.0 - (-bhattacharyya).exp() + eps).sqrt()
    return 1.0 - hellinger


class LWDETRBackbone(nn.Module):
    """Backbone of LW-DETR, based on a ViT Det architecture. The backbone is used as feature extractor.

    Args:
        encoder_fn: the function to build the encoder of the backbone, which is a ViT Det architecture
        out_channels: number of channels in the output feature maps of the backbone.
        num_blocks: number of blocks in the C2fBottleneck of the projector.
    """

    def __init__(
        self,
        encoder_fn: nn.Module,
        out_channels: int = 256,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = encoder_fn

        _is_training = self.encoder.training

        self.encoder.eval()
        with torch.no_grad():
            in_shape = (3, 512, 512)
            out = self.encoder(torch.zeros((1, *in_shape)))
            # Get the number of channels for each feature map output by the backbone
            _shapes = [feat.shape[1] for feat in out]
        self.encoder.train(_is_training)

        self.projector = MultiScaleProjector(
            in_channels=_shapes,
            out_channels=out_channels,
            num_blocks=num_blocks,
        )

    def _resize_padding_mask(self, mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize padding mask to feature-map size

        Args:
            mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            size: the target size (H', W') for the resized mask

        Returns:
            resized_mask: a binary mask of shape [batch_size x H' x W'], containing 1 on padded pixels
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()

        valid = (~mask).float().unsqueeze(1)  # True/1 = valid pixels

        # Data-dependent sanity checks
        if self.training:  # pragma: no cover
            if (valid.flatten(1).sum(dim=1) == 0).any():
                bad = torch.where(valid.flatten(1).sum(dim=1) == 0)[0].tolist()
                raise RuntimeError(f"Input masks are fully padded before resizing: {bad}")

        # Use max pooling to resize the valid mask:
        # a pixel in the resized mask is valid if at least one pixel
        # in the corresponding window in the input mask is valid
        h_in, w_in = int(mask.shape[-2]), int(mask.shape[-1])
        h_out, w_out = int(size[0]), int(size[1])
        kh, kw = h_in // h_out, w_in // w_out
        valid_resized = F.max_pool2d(valid, kernel_size=(kh, kw), stride=(kh, kw)) > 0

        resized_mask = ~valid_resized.squeeze(1)

        # Data-dependent sanity checks
        if self.training:  # pragma: no cover
            if resized_mask.flatten(1).all(dim=1).any():
                bad = torch.where(resized_mask.flatten(1).all(dim=1))[0].tolist()
                raise RuntimeError(f"Feature masks became fully padded after resizing: {bad}")

        return resized_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the backbone.

        Args:
            x: batched images, of shape [batch_size x 3 x H x W]
            mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A list of tuples (feat, mask) for each feature map, where:
            - feat is the feature map of shape [batch_size x out_channels x H' x W']
            - mask is the corresponding attention mask of shape [batch_size x H' x W'], containing 1 on padded pixels
        """
        # (H, W, B, C)
        feats = self.encoder(x)
        feats = self.projector(feats)
        # [(B, C, H, W)]
        if mask is None:  # pragma: no cover
            mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        return [(feat, self._resize_padding_mask(mask, feat.shape[2:])) for feat in feats]


class LWDETR(nn.Module, _LWDETR):
    """LW-DETR as described in `"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
    <https://arxiv.org/pdf/2406.03459v1>`_.

    Args:
        feat_extractor: the backbone of the model, used as feature extractor
        class_names: list of class names to be detected by the model
        score_thresh: the score threshold for post-processing the model outputs
        iou_thresh: the IoU threshold for post-processing the model outputs
        d_model: the dimension of the model
        num_queries: the number of object queries
        group_detr: the number of groups in the group DETR architecture
        dec_layers: the number of decoder layers
        sa_num_heads: the number of heads in the self-attention of the decoder
        ca_num_heads: the number of heads in the cross-attention of the decoder
        ff_dim: the dimension of the feedforward network in the decoder
        dec_n_points: the number of sampling points in the deformable attention of the decoder
        dropout_prob: the dropout probability in the decoder
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        feat_extractor: LWDETRBackbone,
        class_names: list[str],
        score_thresh: float = 0.25,
        iou_thresh: float = 0.5,
        d_model: int = 256,
        num_queries: int = 195,
        group_detr: int = 13,
        dec_layers: int = 3,
        sa_num_heads: int = 8,
        ca_num_heads: int = 16,
        ff_dim: int = 2048,
        dec_n_points: int = 2,
        dropout_prob: float = 0.0,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.class_names: list[str] = class_names
        # No background class: the model is trained with a sigmoid-based (IA-BCE) loss
        self.num_classes = len(self.class_names)
        self.cfg = cfg
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor

        self.group_detr = group_detr
        self.num_queries = num_queries
        self.d_model = d_model

        self.reference_point_embed = nn.Embedding(self.num_queries * self.group_detr, 6)

        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, self.d_model)

        self.class_embed = nn.Linear(self.d_model, self.num_classes)
        self.bbox_embed = LWDETRHead(self.d_model, self.d_model, 6, num_layers=3)

        self.decoder = LWDETRDecoder(
            num_layers=dec_layers,
            d_model=d_model,
            sa_num_heads=sa_num_heads,
            ca_num_heads=ca_num_heads,
            ff_dim=ff_dim,
            dec_n_points=dec_n_points,
            group_detr=group_detr,
            dropout_prob=dropout_prob,
            bbox_embed=self.bbox_embed,
        )

        self.enc_output = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(self.group_detr)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.group_detr)])

        self.enc_out_bbox_embed = nn.ModuleList([
            LWDETRHead(self.d_model, self.d_model, 6, num_layers=3) for _ in range(self.group_detr)
        ])
        self.enc_out_class_embed = nn.ModuleList([
            nn.Linear(self.d_model, self.num_classes) for _ in range(self.group_detr)
        ])

        self.postprocessor = LWDETRPostProcessor(
            num_classes=self.num_classes,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            assume_straight_pages=self.assume_straight_pages,
        )

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue

            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

            if isinstance(m, nn.Linear) and m.out_features == self.num_classes:
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias_value)

        # Initialize the iterative refinement heads to predict zero deltas (i.e. identity refinement)
        # at the start of training, to stabilize training in the early stages when the encoder proposals are still noisy
        with torch.no_grad():
            for head in [self.bbox_embed, *self.enc_out_bbox_embed]:
                last = head.layers[-1]
                last.weight.zero_()
                last.bias.zero_()
                last.bias[5] = 1.0  # cosθ of the rotation delta -> identity rotation

            # The reference point embedding acts as a learned delta composed with the encoder proposals
            self.reference_point_embed.weight.zero_()
            self.reference_point_embed.weight[:, 5] = 1.0  # cosθ

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)

    def gen_encoder_output_proposals(
        self, enc_output: torch.Tensor, padding_mask: torch.Tensor, spatial_shapes: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output: Output of the encoder
            padding_mask: Padding mask for `enc_output`
            spatial_shapes: Spatial shapes of the feature maps

        Returns:
            A tuple of feature map and bbox prediction.
            - object_query: Object query features. Later used to directly predict a bounding box.
            - output_proposals: Normalized proposals in [0, 1] space.
                Invalid positions (padding or out-of-bounds) are filled with 0.
            - invalid_mask: Boolean mask that is True for invalid positions
                (padded pixels or proposals whose coordinates fall outside (0.01, 0.99)).
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0

        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=enc_output.dtype, device=enc_output.device),
                torch.arange(width, dtype=enc_output.dtype, device=enc_output.device),
                indexing="ij",
            )

            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

            scale = torch.tensor(
                [width, height],
                dtype=enc_output.dtype,
                device=enc_output.device,
            ).view(1, 1, 1, 2)

            # Canvas-normalized center coordinates
            grid = (grid + 0.5) / scale
            width_height = torch.ones_like(grid) * 0.05 * (2.0**level)
            sin = torch.zeros_like(grid[..., :1])
            cos = torch.ones_like(grid[..., :1])
            proposal = torch.cat((grid, width_height, sin, cos), dim=-1).view(batch_size, -1, 6)
            proposals.append(proposal)
            _cur += height * width

        output_proposals = torch.cat(proposals, dim=1)

        spatial_valid = ((output_proposals[..., :4] > 0.01) & (output_proposals[..., :4] < 0.99)).all(
            dim=-1, keepdim=True
        )
        invalid_mask = padding_mask.unsqueeze(-1) | ~spatial_valid

        output_proposals = output_proposals.masked_fill(invalid_mask, 0.0)
        object_query = enc_output.masked_fill(invalid_mask, 0.0)

        return object_query, output_proposals, invalid_mask

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        target: list[dict[str, np.ndarray]] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        feats = self.feat_extractor(input, masks)

        sources: list[torch.Tensor] = []
        feats_masks: list[torch.Tensor] = []

        for source, mask in feats:
            sources.append(source)
            feats_masks.append(mask)
            if mask is None:  # pragma: no cover
                raise ValueError("No attention mask was provided")

        if self.training:
            reference_points = self.reference_point_embed.weight
            query_feat = self.query_feat.weight
        else:
            # only use one group in inference
            reference_points = self.reference_point_embed.weight[: self.num_queries]
            query_feat = self.query_feat.weight[: self.num_queries]

        # Prepare encoder inputs (by flattening)
        source_flatten_list: list[torch.Tensor] = []
        mask_flatten_list: list[torch.Tensor] = []
        spatial_shapes_list: list[tuple[int, int]] = []
        for source, mask in zip(sources, feats_masks):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            source_flatten_list.append(source)
            mask_flatten_list.append(mask)
        source_flatten = torch.cat(source_flatten_list, 1)
        mask_flatten = torch.cat(mask_flatten_list, 1)

        tgt = query_feat.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        object_query_embedding, output_proposals, invalid_mask = self.gen_encoder_output_proposals(
            source_flatten, mask_flatten, spatial_shapes_list
        )

        group_detr = self.group_detr if self.training else 1
        topk = self.num_queries

        topk_coords_logits_list: list[torch.Tensor] = []

        # encoder predictions on the selected top-k proposals, kept undetached for the auxiliary loss
        all_group_enc_logits: list[torch.Tensor] = []
        all_group_enc_coords: list[torch.Tensor] = []

        for group_id in range(group_detr):
            group_object_query = self.enc_output[group_id](object_query_embedding)
            group_object_query = self.enc_output_norm[group_id](group_object_query)

            group_enc_outputs_class = self.enc_out_class_embed[group_id](group_object_query)

            group_enc_outputs_class_masked = group_enc_outputs_class.masked_fill(invalid_mask, float("-inf"))

            group_delta_bbox = self.enc_out_bbox_embed[group_id](group_object_query)
            group_enc_outputs_coord = refine_obb_boxes(output_proposals, group_delta_bbox)

            group_topk_proposals = torch.topk(group_enc_outputs_class_masked.max(-1)[0], topk, dim=1)[1]

            group_topk_coords_logits_undetach = torch.gather(
                group_enc_outputs_coord,
                1,
                group_topk_proposals.unsqueeze(-1).repeat(1, 1, 6),
            )
            # the auxiliary loss supervises only the selected proposals,
            # so gather the matching class logits as well
            group_topk_logits_undetach = torch.gather(
                group_enc_outputs_class,
                1,
                group_topk_proposals.unsqueeze(-1).repeat(1, 1, self.num_classes),
            )
            all_group_enc_logits.append(group_topk_logits_undetach)
            all_group_enc_coords.append(group_topk_coords_logits_undetach)

            # the decoder consumes detached proposals as initial reference points
            topk_coords_logits_list.append(group_topk_coords_logits_undetach.detach())

        topk_coords_logits = torch.cat(topk_coords_logits_list, 1)
        reference_points = refine_obb_boxes(topk_coords_logits, reference_points)

        last_hidden_states, intermediate, intermediate_reference_points = self.decoder(
            inputs_embeds=tgt,
            reference_points=reference_points,
            spatial_shapes_list=spatial_shapes_list,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
        )

        logits = self.class_embed(last_hidden_states)
        pred_boxes_delta = self.bbox_embed(last_hidden_states)
        pred_boxes = refine_obb_boxes(intermediate_reference_points[-1], pred_boxes_delta)

        out: dict[str, Any] = {}

        logits = _bf16_to_float32(logits)
        pred_boxes = _bf16_to_float32(pred_boxes)

        if self.exportable:
            out["logits"] = logits
            out["pred_boxes"] = pred_boxes
            return out

        if return_model_output or target is None or return_preds:
            out["logits"] = logits

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable
            def _postprocess(logits, boxes):
                return self.postprocessor(logits, boxes)

            out["preds"] = _postprocess(logits.detach().cpu().numpy(), pred_boxes.detach().cpu().numpy())

        if target is not None:
            # Build target
            processed_targets = self.build_target(target, self.class_names)

            # ProbIoU is computed in pixel coordinates
            box_scale = float(max(input.shape[-2], input.shape[-1]))

            # Main loss from final decoder layer (group DETR: each group is matched independently)
            split_logits = logits.chunk(group_detr, dim=1)
            split_boxes = pred_boxes.chunk(group_detr, dim=1)

            main_loss: float | torch.Tensor = 0.0
            for g_logits, g_boxes in zip(split_logits, split_boxes):
                main_loss += self.compute_loss(g_logits, g_boxes, processed_targets, box_scale=box_scale)
            loss = main_loss / group_detr

            # Auxiliary losses from intermediate decoder layers
            # (`intermediate_reference_points[i]` is the reference INPUT to decoder layer i)
            for i in range(intermediate.shape[0] - 1):
                aux_logits = self.class_embed(intermediate[i])
                aux_boxes_delta = self.bbox_embed(intermediate[i])
                aux_boxes = refine_obb_boxes(intermediate_reference_points[i], aux_boxes_delta)

                split_aux_logits = aux_logits.chunk(group_detr, dim=1)
                split_aux_boxes = aux_boxes.chunk(group_detr, dim=1)

                aux_loss: float | torch.Tensor = 0.0
                for g_logits, g_boxes in zip(split_aux_logits, split_aux_boxes):
                    aux_loss += self.compute_loss(g_logits, g_boxes, processed_targets, box_scale=box_scale)
                loss += aux_loss / group_detr

            # Auxiliary losses for the selected encoder proposals
            enc_loss: float | torch.Tensor = 0.0
            for group_logits, group_coords in zip(all_group_enc_logits, all_group_enc_coords):
                enc_loss += self.compute_loss(group_logits, group_coords, processed_targets, box_scale=box_scale)
            loss += enc_loss / group_detr

            out["loss"] = loss

        return out

    def compute_loss(
        self,
        logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, np.ndarray]],
        cls_loss_weight: float = 1.0,
        l1_loss_weight: float = 5.0,
        iou_loss_weight: float = 2.0,
        box_scale: float = 1024.0,
    ) -> torch.Tensor:
        """Compute the LW-DETR loss for oriented bounding boxes.

        Predictions are matched one-to-one to the ground truth boxes with Hungarian matching,
        using a cost combining a focal-style classification cost, an L1 box cost
        and a (negated) ProbIoU cost. The loss then consists of:

        - an IoU-aware binary cross-entropy (IA-BCE) classification loss, as described in the LW-DETR paper,
          where the target of a matched (query, class) pair is `p**alpha * IoU**(1 - alpha)` and the rotated
          ProbIoU is used as IoU measure
        - an L1 regression loss on the normalized (cx, cy, w, h) of the matched pairs
        - a ProbIoU loss (1 - ProbIoU) on the matched oriented boxes, computed in absolute pixel
          coordinates (as in O2-RT-DETR). The box rotation is supervised solely through this term:
          ProbIoU is differentiable w.r.t. the angle even for non-overlapping boxes.

        All terms are normalized by the number of ground truth boxes in the batch.

        Args:
            logits: (B, Q, C) tensor containing the predicted class logits for each query
            pred_boxes: (B, Q, 6) tensor containing the predicted boxes (cx, cy, w, h, sinθ, cosθ) for each query
            targets: list of B dictionaries with keys "boxes" ((N, 6) array in OBB format)
                and "labels" ((N,) array of class indices)
            cls_loss_weight: weight of the classification loss
            l1_loss_weight: weight of the L1 box regression loss
            iou_loss_weight: weight of the ProbIoU loss
            box_scale: image size used to rescale normalized boxes to pixel coordinates for ProbIoU computation

        Returns:
            loss: the computed loss value
        """
        alpha, gamma, eps = 0.25, 2.0, 1e-8
        # AMP safe-guard
        logits = logits.float()
        pred_boxes = pred_boxes.float()
        device = logits.device
        batch_size = logits.shape[0]

        tgt_boxes_list: list[torch.Tensor] = []
        tgt_labels_list: list[torch.Tensor] = []
        for sample in targets:
            tgt_boxes_list.append(
                torch.as_tensor(sample["boxes"], device=device, dtype=pred_boxes.dtype).reshape(-1, 6)
            )
            tgt_labels_list.append(torch.as_tensor(sample["labels"], device=device, dtype=torch.long).reshape(-1))

        # Number of target boxes in the batch, for loss normalization
        num_boxes = max(sum(int(labels.numel()) for labels in tgt_labels_list), 1)

        # Hungarian matching (one-to-one), performed independently for each sample
        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            prob = logits.sigmoid()
            for b in range(batch_size):
                tgt_boxes, tgt_labels = tgt_boxes_list[b], tgt_labels_list[b]
                if tgt_labels.numel() == 0:
                    empty = torch.empty(0, dtype=torch.long, device=device)
                    indices.append((empty, empty))
                    continue

                out_prob = prob[b]
                out_boxes = pred_boxes[b]

                # Focal-style classification cost
                neg_cost = (1 - alpha) * out_prob.pow(gamma) * (-(1 - out_prob + eps).log())
                pos_cost = alpha * (1 - out_prob).pow(gamma) * (-(out_prob + eps).log())
                cost_class = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]

                # L1 cost on normalized (cx, cy, w, h)
                cost_bbox = torch.cdist(out_boxes[:, :4].float(), tgt_boxes[:, :4].float(), p=1)

                # Rotated IoU cost, computed in pixel coordinates
                # this term also carries the angle signal for the matching
                cost_iou = -_probiou(out_boxes, tgt_boxes, pairwise=True, scale=box_scale)

                cost = 2.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_iou

                query_idx, tgt_idx = linear_sum_assignment(cost.cpu().numpy())
                indices.append((
                    torch.as_tensor(query_idx, dtype=torch.long, device=device),
                    torch.as_tensor(tgt_idx, dtype=torch.long, device=device),
                ))

        # Flatten the matched pairs across the batch
        batch_idx = torch.cat([torch.full_like(src, b) for b, (src, _) in enumerate(indices)])
        query_idx = torch.cat([src for (src, _) in indices])
        matched_tgt_boxes = torch.cat([tgt_boxes_list[b][tgt] for b, (_, tgt) in enumerate(indices)])
        matched_tgt_labels = torch.cat([tgt_labels_list[b][tgt] for b, (_, tgt) in enumerate(indices)])
        matched_pred_boxes = pred_boxes[batch_idx, query_idx]

        prob = logits.sigmoid()

        # IoU-aware BCE classification loss (IA-BCE)
        pos_weights = torch.zeros_like(logits)
        neg_weights = prob.pow(gamma)
        if len(batch_idx) > 0:
            with torch.no_grad():
                ious = _probiou(matched_pred_boxes, matched_tgt_boxes, scale=box_scale).clamp(min=0.0, max=1.0)
                t = prob[batch_idx, query_idx, matched_tgt_labels].pow(alpha) * ious.pow(1 - alpha)
                t = t.clamp(min=0.01)
            pos_weights[batch_idx, query_idx, matched_tgt_labels] = t
            neg_weights[batch_idx, query_idx, matched_tgt_labels] = 1 - t

        cls_loss = -(pos_weights * (prob + eps).log() + neg_weights * (1 - prob + eps).log())
        loss = cls_loss_weight * cls_loss.sum() / num_boxes

        if len(batch_idx) == 0:
            return loss

        # L1 loss on normalized (cx, cy, w, h)
        l1_loss = F.l1_loss(matched_pred_boxes[:, :4], matched_tgt_boxes[:, :4], reduction="sum") / num_boxes
        # ProbIoU loss on the whole oriented box (position, size and rotation), in pixel coordinates
        probiou_loss = (1 - _probiou(matched_pred_boxes, matched_tgt_boxes, scale=box_scale)).sum() / num_boxes

        return loss + l1_loss_weight * l1_loss + iou_loss_weight * probiou_loss


def _lw_detr(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> LWDETR:
    # Patch the config
    kwargs["class_names"] = kwargs.get("class_names", default_cfgs[arch].get("class_names", []))

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["class_names"] = sorted(kwargs["class_names"])
    kwargs.pop("class_names")

    # Build the feature extractor
    backbone = backbone_fn(  # type: ignore[call-arg]
        False,
        include_top=False,
        input_shape=default_cfgs[arch]["input_shape"],
        patch_size=kwargs.pop("patch_size", (16, 16)),
    )
    feat_extractor = LWDETRBackbone(encoder_fn=backbone)

    # Build the model
    model = LWDETR(
        feat_extractor,
        cfg=_cfg,
        class_names=_cfg["class_names"],
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        # The number of class_names is not the same as the number of classes in the pretrained model =>
        # remove the layer weights
        _ignore_keys = ignore_keys if _cfg["class_names"] != sorted(default_cfgs[arch].get("class_names", [])) else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def lw_detr_s(pretrained: bool = False, **kwargs: Any) -> LWDETR:
    """LW-DETR as described in `"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
    <https://arxiv.org/pdf/2406.03459v1>`_.

    >>> import torch
    >>> from doctr.models import lw_detr_s
    >>> model = lw_detr_s(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _lw_detr(
        "lw_detr_s",
        pretrained,
        vit_det_s,
        ignore_keys=[
            "class_embed.weight",
            "class_embed.bias",
            *[f"enc_out_class_embed.{i}.weight" for i in range(kwargs.get("group_detr", 13))],
            *[f"enc_out_class_embed.{i}.bias" for i in range(kwargs.get("group_detr", 13))],
        ],
        **kwargs,
    )


def lw_detr_m(pretrained: bool = False, **kwargs: Any) -> LWDETR:
    """LW-DETR as described in `"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
    <https://arxiv.org/pdf/2406.03459v1>`_.

    >>> import torch
    >>> from doctr.models import lw_detr_m
    >>> model = lw_detr_m(pretrained=True).eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _lw_detr(
        "lw_detr_m",
        pretrained,
        vit_det_m,
        ignore_keys=[
            "class_embed.weight",
            "class_embed.bias",
            *[f"enc_out_class_embed.{i}.weight" for i in range(kwargs.get("group_detr", 13))],
            *[f"enc_out_class_embed.{i}.bias" for i in range(kwargs.get("group_detr", 13))],
        ],
        **kwargs,
    )
