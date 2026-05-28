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

from ...utils import load_pretrained_params
from .base import _LWDETR, LWDETRPostProcessor
from .layers import LWDETRDecoder, LWDETRHead, LWDETRMultiscaleDeformableAttention, MultiScaleProjector

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
        return [
            (feat, F.interpolate(mask.unsqueeze(1).float(), size=feat.shape[-2:], mode="nearest").squeeze(1).bool())
            for feat in feats
        ]


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
        score_thresh: float = 0.3,
        iou_thresh: float = 0.5,
        d_model: int = 256,
        num_queries: int = 300,
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
        self.num_classes = len(self.class_names) + 1  # +1 for background class
        self.cfg = cfg
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor

        self.group_detr = group_detr
        self.num_queries = num_queries
        self.d_model = d_model

        self.reference_point_embed = nn.Embedding(self.num_queries * self.group_detr, 6)
        # Initialize angle to (sin=0, cos=1)
        with torch.no_grad():
            self.reference_point_embed.weight[:, 0:2].uniform_(0.05, 0.95)
            self.reference_point_embed.weight[:, 2:4].fill_(0.1)
            self.reference_point_embed.weight[:, 4].zero_()
            self.reference_point_embed.weight[:, 5].fill_(1.0)

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
                if m is not self.reference_point_embed:
                    nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, LWDETRMultiscaleDeformableAttention):
                nn.init.constant_(m.sampling_offsets.weight, 0.0)

                thetas = torch.arange(m.n_heads, dtype=torch.float32) * (2.0 * math.pi / m.n_heads)
                grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
                grid_init = (
                    (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                    .view(m.n_heads, 1, 1, 2)
                    .repeat(1, m.n_levels, m.n_points, 1)
                )
                for i in range(m.n_points):
                    grid_init[:, :, i, :] *= i + 1
                with torch.no_grad():
                    m.sampling_offsets.bias.copy_(grid_init.view(-1))

                nn.init.constant_(m.attention_weights.weight, 0.0)
                nn.init.constant_(m.attention_weights.bias, 0.0)
                nn.init.xavier_uniform_(m.value_proj.weight)
                nn.init.zeros_(m.value_proj.bias)
                nn.init.xavier_uniform_(m.output_proj.weight)
                nn.init.zeros_(m.output_proj.bias)
            if isinstance(m, nn.Linear) and m.out_features == self.num_classes:
                if m.bias is not None:
                    with torch.no_grad():
                        # Focal-loss prior: foreground starts with low confidence (~0.01),
                        # preventing background from dominating gradients at the start of training.
                        prior_prob = 0.01
                        bias_value = -math.log((1 - prior_prob) / prior_prob)
                        nn.init.constant_(m.bias, 0.0)
                        m.bias[:-1].fill_(bias_value)
            if isinstance(m, LWDETRHead):
                last = m.layers[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
                    if last.bias.shape[0] == 6:
                        nn.init.constant_(last.bias[5], 1.0)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)

    def refine_bboxes(self, reference_points: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Refine bounding boxes by applying the predicted deltas to the reference points.
        The reference points are in the format (cx, cy, w, h, sinθ, cosθ), and the deltas are in the same format.
        The refined boxes are computed as follows:

        cx' = cx + delta_cx * w
        cy' = cy + delta_cy * h
        w' = w * exp(delta_w)
        h' = h * exp(delta_h)
        sinθ' = sinθ * cosΔ + cosθ * sinΔ
        cosθ' = cosθ * cosΔ - sinθ * sinΔ

        Args:
            reference_points: (N, S, 6) tensor containing the reference points
            deltas: (N, S, 6) tensor containing the predicted deltas

        Returns:
            refined_boxes: (N, S, 6) tensor containing the refined bounding boxes
        """
        reference_points = reference_points.to(deltas.device)
        cxcy = deltas[..., :2] * reference_points[..., 2:4] + reference_points[..., :2]
        # size
        wh = torch.clamp(deltas[..., 2:4], min=-10.0, max=10.0).exp() * reference_points[..., 2:4]
        # rotation
        delta_rot = F.normalize(deltas[..., 4:6], dim=-1, eps=1e-6)
        sin_delta = delta_rot[..., 0:1]
        cos_delta = delta_rot[..., 1:2]
        sin_ref = reference_points[..., 4:5]
        cos_ref = reference_points[..., 5:6]
        # compose rotations
        sin_new = sin_ref * cos_delta + cos_ref * sin_delta
        cos_new = cos_ref * cos_delta - sin_ref * sin_delta
        rot = F.normalize(torch.cat([sin_new, cos_new], dim=-1), dim=-1, eps=1e-6)
        return torch.cat((cxcy, wh, rot), dim=-1)

    def get_valid_ratio(self, mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get the valid ratio of all feature maps.

        Args:
            mask: (N, H, W) binary tensor containing 1 on padded pixels
            dtype: the desired data type of the output tensor

        Returns:
            valid_ratio: (N, 2) tensor containing the valid ratio of width and height for each image in the batch
        """
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

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
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0,
                    height - 1,
                    height,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                torch.linspace(
                    0,
                    width - 1,
                    width,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_height = torch.ones_like(grid) * 0.05 * (2.0**level)
            # add default rotation (sin=0, cos=1)
            sin = torch.zeros_like(grid[..., :1])
            cos = torch.ones_like(grid[..., :1])
            proposal = torch.cat((grid, width_height, sin, cos), -1).view(batch_size, -1, 6)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)

        spatial_valid = ((output_proposals[..., :4] > 0.01) & (output_proposals[..., :4] < 0.99)).all(-1, keepdim=True)
        output_proposals_valid = spatial_valid
        invalid_mask = padding_mask.unsqueeze(-1) | ~output_proposals_valid
        output_proposals = output_proposals.masked_fill(invalid_mask, float(0))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(invalid_mask, 0.0)

        return object_query, output_proposals, invalid_mask

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor,
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
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in feats_masks], 1)

        tgt = query_feat.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        object_query_embedding, output_proposals, invalid_mask = self.gen_encoder_output_proposals(
            source_flatten, mask_flatten, spatial_shapes_list
        )

        group_detr = self.group_detr if self.training else 1
        topk = self.num_queries

        topk_coords_logits_list: list[torch.Tensor] = []
        topk_content_list: list[torch.Tensor] = []

        # encoder predictions for auxiliary losses
        all_group_enc_logits: list[torch.Tensor] = []
        all_group_enc_coords: list[torch.Tensor] = []

        for group_id in range(group_detr):
            group_object_query = self.enc_output[group_id](object_query_embedding)
            group_object_query = self.enc_output_norm[group_id](group_object_query)

            group_enc_outputs_class = self.enc_out_class_embed[group_id](group_object_query)
            all_group_enc_logits.append(group_enc_outputs_class)

            group_enc_outputs_class_masked = group_enc_outputs_class.masked_fill(invalid_mask, float("-inf"))

            group_delta_bbox = self.enc_out_bbox_embed[group_id](group_object_query)
            group_enc_outputs_coord = self.refine_bboxes(output_proposals, group_delta_bbox)

            all_group_enc_coords.append(group_enc_outputs_coord)

            scores = group_enc_outputs_class_masked[..., :-1].max(-1).values
            group_topk_proposals = torch.topk(scores, topk, dim=1)[1]

            group_topk_coords_logits_undetach = torch.gather(
                group_enc_outputs_coord,
                1,
                group_topk_proposals.unsqueeze(-1).repeat(1, 1, 6),
            )
            group_topk_coords_logits = group_topk_coords_logits_undetach.detach()
            topk_coords_logits_list.append(group_topk_coords_logits)
            group_topk_content = torch.gather(
                group_object_query,
                1,
                group_topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )
            topk_content_list.append(group_topk_content)

        topk_coords_logits = torch.cat(topk_coords_logits_list, 1)

        reference_points = self.refine_bboxes(topk_coords_logits, reference_points)

        last_hidden_states, intermediate, intermediate_reference_points = self.decoder(
            inputs_embeds=tgt,
            reference_points=reference_points,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
        )

        logits = self.class_embed(last_hidden_states)
        pred_boxes = intermediate_reference_points[-1]

        out: dict[str, Any] = {}

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

            # Main loss from final decoder layer (group DETR)
            split_logits = logits.chunk(group_detr, dim=1)
            split_boxes = pred_boxes.chunk(group_detr, dim=1)

            main_loss: float | torch.Tensor = 0.0
            for g_logits, g_boxes in zip(split_logits, split_boxes):
                main_loss += self.compute_loss(g_logits, g_boxes, processed_targets)
            loss = main_loss / group_detr

            # Auxiliary losses from intermediate decoder layers (group DETR)
            for i in range(intermediate.shape[0] - 1):
                aux_logits = self.class_embed(intermediate[i])
                aux_boxes = intermediate_reference_points[i + 1]

                split_aux_logits = aux_logits.chunk(group_detr, dim=1)
                split_aux_boxes = aux_boxes.chunk(group_detr, dim=1)

                aux_loss: float | torch.Tensor = 0.0
                for g_logits, g_boxes in zip(split_aux_logits, split_aux_boxes):
                    aux_loss += self.compute_loss(g_logits, g_boxes, processed_targets)
                loss += aux_loss / group_detr

            # Auxiliary losses for encoder proposals
            enc_loss: float | torch.Tensor = 0.0
            for group_logits, group_coords in zip(all_group_enc_logits, all_group_enc_coords):
                enc_loss += self.compute_loss(group_logits, group_coords, processed_targets)
            loss += enc_loss / group_detr

            out["loss"] = loss

        return out

    def compute_loss(
        self,
        logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, np.ndarray]],
    ) -> torch.Tensor:
        """Compute the loss between predicted logits and boxes and target labels and boxes.

        Args:
            logits: (N, S, C) tensor containing the predicted class logits for each query
            pred_boxes: (N, S, 6) tensor containing the predicted boxes in (cx, cy, w, h, sinθ, cosθ) format
            targets: list of length N, where each element is a dict with keys "labels" and "boxes",
                containing the ground truth labels and boxes for each image in the batch.
                The boxes are in (cx, cy, w, h, sinθ, cosθ) format.

        Returns:
            A scalar tensor containing the computed loss.
        """

        def _rotated_boxes_to_gaussian(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Convert rotated boxes in (cx, cy, w, h, sinθ, cosθ) format
                to Gaussian distributions (mean and covariance).
                The mean is simply (cx, cy), and the covariance is computed from the width, height, and rotation angle.

            Args:
                boxes: (N, S, 6) tensor containing the rotated boxes in (cx, cy, w, h, sinθ, cosθ) format
            Returns:
                A tuple of (mean, covariance) where:
                - mean is a (N, S, 2) tensor containing the mean (cx, cy) of the Gaussian distributions
                - covariance is a (N, S, 2, 2) tensor containing the covariance matrices of the Gaussian distributions
            """
            cxcy = boxes[..., :2]

            w = boxes[..., 2].clamp(min=1e-6)
            h = boxes[..., 3].clamp(min=1e-6)

            sin = boxes[..., 4]
            cos = boxes[..., 5]

            R = torch.stack(
                [
                    torch.stack([cos, -sin], dim=-1),
                    torch.stack([sin, cos], dim=-1),
                ],
                dim=-2,
            )

            # Variance for a box half-width/half-height: σ² = (w/2)²
            # Using w²/12 (uniform distribution) produces ~8x smaller variance,
            # which collapses Bhattacharyya distance to the clamp ceiling and kills gradients.
            sx = (w / 2) ** 2
            sy = (h / 2) ** 2

            S = torch.zeros((*boxes.shape[:-1], 2, 2), device=boxes.device)
            S[..., 0, 0] = sx
            S[..., 1, 1] = sy

            covariance = R @ S @ R.transpose(-1, -2)
            return cxcy, covariance

        def _probiou_loss(pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor) -> torch.Tensor:
            """Compute the ProbIoU loss between predicted and target boxes,
                where boxes are represented as Gaussian distributions.
                The ProbIoU loss is defined as 1 - exp(-Bhattacharyya distance),
                where the Bhattacharyya distance is computed between the two Gaussian distributions.

            Args:
                pred_boxes: (N, S, 6) tensor containing the predicted boxes in (cx, cy, w, h, sinθ, cosθ) format
                tgt_boxes: (N, S, 6) tensor containing the target boxes in (cx, cy, w, h, sinθ, cosθ) format
            Returns:
                A (N, S) tensor containing the ProbIoU loss for each pair of predicted and target boxes
            """
            mu1, sigma1 = _rotated_boxes_to_gaussian(pred_boxes)
            mu2, sigma2 = _rotated_boxes_to_gaussian(tgt_boxes)

            delta = (mu1 - mu2).unsqueeze(-1)
            sigma = (sigma1 + sigma2) * 0.5

            eps = 1e-6
            eye = torch.eye(2, device=sigma.device) * eps

            sigma_safe = sigma + eye
            sigma1_safe = sigma1 + eye
            sigma2_safe = sigma2 + eye

            sigma_inv = torch.linalg.inv(sigma_safe)

            mahalanobis = (delta.transpose(-1, -2) @ sigma_inv @ delta).squeeze(-1).squeeze(-1)

            det_sigma = torch.linalg.det(sigma_safe).clamp(min=eps)
            det_sigma1 = torch.linalg.det(sigma1_safe).clamp(min=eps)
            det_sigma2 = torch.linalg.det(sigma2_safe).clamp(min=eps)

            bhattacharyya = 0.125 * mahalanobis + 0.5 * torch.log(det_sigma / torch.sqrt(det_sigma1 * det_sigma2))

            bhattacharyya = torch.clamp(bhattacharyya, min=0.0, max=10.0)
            probiou = torch.exp(-bhattacharyya)
            return 1 - probiou

        device = logits.device
        B, Q, C = logits.shape

        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)

        num_matched_total = 0

        for b in range(B):
            pred_logits = logits[b]
            pred_boxes_b = pred_boxes[b]

            boxes = targets[b]["boxes"]

            if len(boxes) == 0:
                # Penalize the model for any foreground boxes it guessed on this empty image
                background_idx = self.num_classes - 1
                target_classes = torch.full((Q,), background_idx, device=device, dtype=torch.long)
                total_cls += F.cross_entropy(pred_logits, target_classes)
                continue

            tgt_boxes = torch.as_tensor(boxes, device=device, dtype=pred_boxes.dtype)
            tgt_cls = torch.as_tensor(targets[b]["labels"], device=device, dtype=torch.long)

            if tgt_boxes.ndim == 1:
                tgt_boxes = tgt_boxes.unsqueeze(0)

            pred_rot = F.normalize(pred_boxes_b[:, 4:6], dim=-1)
            tgt_rot = F.normalize(tgt_boxes[:, 4:6], dim=-1)

            with torch.no_grad():
                out_logprob = pred_logits.log_softmax(-1)

                cost_cls = -out_logprob[:, tgt_cls]
                cost_l1 = torch.cdist(pred_boxes_b[:, :4], tgt_boxes[:, :4], p=1)
                cost_rot = 1.0 - torch.abs(pred_rot @ tgt_rot.T)

                total_cost = 2.0 * cost_cls + 5.0 * cost_l1 + 2.0 * cost_rot

                cost_np = total_cost.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)

                pos_idx = torch.as_tensor(row_ind, device=device)
                gt_idx = torch.as_tensor(col_ind, device=device)

            background_idx = self.num_classes - 1

            target_classes = torch.full((Q,), background_idx, device=device, dtype=torch.long)
            target_classes[pos_idx] = tgt_cls[gt_idx]

            cls_weights = torch.ones(self.num_classes, device=device)
            cls_weights[background_idx] = 0.1

            total_cls += F.cross_entropy(pred_logits, target_classes, weight=cls_weights)

            if pos_idx.numel() == 0:
                continue

            num_matched_total += pos_idx.numel()

            pred_sel = pred_boxes_b[pos_idx]
            tgt_sel = tgt_boxes[gt_idx]

            l1_loss = F.smooth_l1_loss(pred_sel[:, :4], tgt_sel[:, :4], reduction="sum", beta=0.1)
            probiou_loss = _probiou_loss(pred_sel, tgt_sel).sum()
            total_box += 5.0 * l1_loss + 2.0 * probiou_loss

        num_matched_total = max(num_matched_total, 1)
        loss_cls = total_cls / B
        loss_box = total_box / num_matched_total

        return loss_cls + loss_box


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
        patch_size=kwargs.get("patch_size", (16, 16)),
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
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
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
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
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
