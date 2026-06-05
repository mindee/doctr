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
        score_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        d_model: int = 256,
        num_queries: int = 195,  # This is different from the paper which uses 300 queries, but 195 queries is sufficient for document layout analysis)  # noqa: E501
        group_detr: int = 13,
        dec_layers: int = 3,
        sa_num_heads: int = 8,
        ca_num_heads: int = 16,
        ff_dim: int = 2048,
        dec_n_points: int = 2,
        dropout_prob: float = 0.1,
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
            self.reference_point_embed.weight[:, 0:2].uniform_(0.05, 0.95)  # cx, cy
            self.reference_point_embed.weight[:, 2].uniform_(0.1, 0.6)  # w
            self.reference_point_embed.weight[:, 3].uniform_(0.02, 0.3)  # h
            self.reference_point_embed.weight[:, 4].zero_()  # sinθ
            self.reference_point_embed.weight[:, 5].fill_(1.0)  # cosθ

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

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)

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

        spatial_valid = ((output_proposals[..., :2] > 0.01) & (output_proposals[..., :2] < 0.99)).all(-1, keepdim=True)
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
            group_enc_outputs_coord = self.decoder.refine_bboxes(output_proposals, group_delta_bbox)

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

        reference_points = self.decoder.refine_bboxes(topk_coords_logits, reference_points)

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

            # Disable mixed precision for loss computation to ensure numerical stability,
            # especially for the Bhattacharyya distance which involves
            # logarithms and determinants of covariance matrices.
            with torch.autocast(device_type=logits.device.type, enabled=False):
                # Main loss from final decoder layer
                loss = self.compute_loss(logits.float(), pred_boxes.float(), processed_targets)

                # Auxiliary losses from intermediate decoder layers
                for i in range(intermediate.shape[0] - 1):
                    aux_logits = self.class_embed(intermediate[i]).float()
                    aux_boxes = intermediate_reference_points[i + 1].float()
                    loss += self.compute_loss(aux_logits, aux_boxes, processed_targets)

                # Auxiliary losses for encoder proposals
                enc_logits = torch.cat(all_group_enc_logits, dim=1).float()
                enc_coords = torch.cat(all_group_enc_coords, dim=1).float()
                loss += 0.2 * self.compute_loss(enc_logits, enc_coords, processed_targets)

                out["loss"] = loss

        return out

    def compute_loss(
        self,
        logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, np.ndarray]],
    ) -> torch.Tensor:
        """Compute the loss using Grouped Hungarian Matching
        and consistent ProbIoU semantics for rotated bounding boxes.

        Args:
            logits: (B, Q, C) tensor containing the predicted class logits for each query
            pred_boxes: (B, Q, 6) tensor containing the predicted boxes in (cx, cy, w, h, sinθ, cosθ) format
            targets: list of length B, where each element is a dict with keys "labels" and "boxes",
                containing the ground truth labels and boxes for each image in the batch.

        Returns:
            A scalar tensor containing the computed loss.
        """
        device = logits.device
        dtype = logits.dtype
        B, Q, C = logits.shape

        # Consistent coefficients across matcher and loss components
        class_weight = 2.0
        bbox_weight = 5.0
        probiou_weight = 2.0
        rot_weight = 0.5

        # Focal Loss Params
        alpha = 0.25
        gamma = 2.0
        eps = 1e-7

        group_detr = getattr(self, "group_detr", 1)

        def _rotated_boxes_to_gaussian(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Convert rotated boxes to Gaussian distributions using the true
            variance of a uniform continuous rectangle (w^2 / 12)."""
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

            sx = (w**2) / 12.0
            sy = (h**2) / 12.0

            S = torch.zeros((*boxes.shape[:-1], 2, 2), device=boxes.device, dtype=boxes.dtype)
            S[..., 0, 0] = sx
            S[..., 1, 1] = sy

            covariance = R @ S @ R.transpose(-1, -2)
            return cxcy, covariance

        def _bhattacharyya_distance(
            mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor
        ) -> torch.Tensor:
            """Compute Bhattacharyya distance with broadcast support."""
            delta = (mu1 - mu2).unsqueeze(-1)
            sigma = (sigma1 + sigma2) * 0.5

            eye = torch.eye(2, device=sigma.device, dtype=sigma.dtype) * 1e-6
            sigma_safe = sigma + eye
            sigma1_safe = sigma1 + eye
            sigma2_safe = sigma2 + eye

            L = torch.linalg.cholesky(sigma_safe)
            sigma_inv = torch.cholesky_inverse(L)

            mahalanobis = (delta.transpose(-1, -2) @ sigma_inv @ delta).squeeze(-1).squeeze(-1)

            det_sigma = torch.linalg.det(sigma_safe).clamp(min=1e-6)
            det_sigma1 = torch.linalg.det(sigma1_safe).clamp(min=1e-6)
            det_sigma2 = torch.linalg.det(sigma2_safe).clamp(min=1e-6)

            bhattacharyya = 0.125 * mahalanobis + 0.5 * torch.log(det_sigma / torch.sqrt(det_sigma1 * det_sigma2))
            return bhattacharyya.clamp(min=0.0)

        # Prepare targets for matching
        target_labels = []
        target_boxes = []
        sizes = []
        for t in targets:
            lbls = torch.as_tensor(t["labels"], device=device, dtype=torch.long)
            bxs = torch.as_tensor(t["boxes"], device=device, dtype=pred_boxes.dtype)
            if bxs.ndim == 1 and bxs.numel() > 0:
                bxs = bxs.unsqueeze(0)
            target_labels.append(lbls)
            target_boxes.append(bxs)
            sizes.append(len(lbls))

        # Unified formulation for empty batches
        if sum(sizes) == 0:
            prob = logits.sigmoid()
            prob_safe = prob.clamp(min=eps, max=1.0 - eps)
            neg_weights = prob.pow(gamma)
            loss_ce = -neg_weights * (1.0 - prob_safe).log()
            return class_weight * (loss_ce.sum() / (B * Q))

        tgt_ids = torch.cat(target_labels)
        tgt_bbox = torch.cat(target_boxes)

        # Matcher: Grouped Hungarian Assignment with a balanced cost matrix
        with torch.no_grad():
            out_prob = logits.flatten(0, 1).sigmoid()
            out_bbox = pred_boxes.flatten(0, 1)

            # Classification Cost (Focal Loss based)
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + eps).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + eps).log())
            class_cost = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Box L1 Cost
            out_bbox_f = out_bbox.to(torch.float32)
            tgt_bbox_f = tgt_bbox.to(torch.float32)
            bbox_cost = torch.cdist(out_bbox_f[:, :4], tgt_bbox_f[:, :4], p=1).to(dtype)

            # ProbIoU Cost
            mu_pred, sig_pred = _rotated_boxes_to_gaussian(out_bbox_f)
            mu_tgt, sig_tgt = _rotated_boxes_to_gaussian(tgt_bbox_f)

            bhat_dist = _bhattacharyya_distance(
                mu_pred.unsqueeze(1), sig_pred.unsqueeze(1), mu_tgt.unsqueeze(0), sig_tgt.unsqueeze(0)
            )
            probiou_cost = (1.0 - torch.exp(-bhat_dist)).to(dtype)

            # Rotation Cost
            pred_rot = F.normalize(out_bbox_f[:, 4:6], dim=-1)
            tgt_rot = F.normalize(tgt_bbox_f[:, 4:6], dim=-1)
            rot_cost = (1.0 - torch.abs(pred_rot @ tgt_rot.T)).to(dtype)

            # Total balanced Cost Matrix
            cost_matrix = (
                class_weight * class_cost
                + bbox_weight * bbox_cost
                + probiou_weight * probiou_cost
                + rot_weight * rot_cost
            )
            cost_matrix = cost_matrix.view(B, Q, -1).cpu()

            # Grouped Hungarian Assignment
            indices = []
            group_num_queries = Q // group_detr
            cost_matrix_groups = cost_matrix.split(group_num_queries, dim=1)

            for group_id in range(group_detr):
                group_cost_matrix = cost_matrix_groups[group_id]

                # Split targets per batch element
                group_indices = []
                for i, c in enumerate(group_cost_matrix.split(sizes, -1)):
                    if sizes[i] == 0:
                        group_indices.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                    else:
                        row_ind, col_ind = linear_sum_assignment(c[i].numpy())
                        group_indices.append((row_ind, col_ind))

                if group_id == 0:
                    indices = group_indices
                else:
                    indices = [
                        (
                            np.concatenate([idx1[0], idx2[0] + group_num_queries * group_id]),
                            np.concatenate([idx1[1], idx2[1]]),
                        )
                        for idx1, idx2 in zip(indices, group_indices)
                    ]

        # Image lovel loss normalization: scale by the number of matched boxes,
        # and the number of active groups in group DETR
        # Scale denominator by the number of active assignment groups
        num_boxes = max(sum(sizes) * group_detr, 1)

        batch_idx = torch.cat([torch.full((len(src),), i, dtype=torch.long) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([torch.as_tensor(src, dtype=torch.long) for (src, _) in indices])

        flat_tgt_idx_list = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            flat_tgt_idx_list.append(torch.as_tensor(tgt, dtype=torch.long) + offset)
            offset += sizes[i]
        flat_tgt_idx = torch.cat(flat_tgt_idx_list)

        target_classes_o = tgt_ids[flat_tgt_idx]
        src_boxes = pred_boxes[batch_idx, src_idx]
        target_boxes_matched = tgt_bbox[flat_tgt_idx]

        # Label Loss with Quality Mapping
        prob = logits.sigmoid()

        mu1, sig1 = _rotated_boxes_to_gaussian(src_boxes.detach().to(torch.float32))
        mu2, sig2 = _rotated_boxes_to_gaussian(target_boxes_matched.detach().to(torch.float32))
        bhat_matched = _bhattacharyya_distance(mu1, sig1, mu2, sig2)
        pos_ious = torch.exp(-bhat_matched).clamp(min=0.0, max=1.0).to(dtype)

        pos_weights = torch.zeros_like(logits)
        neg_weights = prob.pow(gamma)
        pos_ind = (batch_idx, src_idx, target_classes_o)

        pos_quality = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
        pos_quality = torch.clamp(pos_quality, 0.01).detach()

        pos_weights[pos_ind] = pos_quality
        neg_weights[pos_ind] = 1 - pos_quality

        prob_safe = prob.clamp(min=eps, max=1.0 - eps)
        loss_ce = -pos_weights * prob_safe.log() - neg_weights * (1.0 - prob_safe).log()
        loss_ce = loss_ce.sum() / num_boxes

        # Bounding Box Loss
        loss_bbox = (
            F.smooth_l1_loss(src_boxes[:, :4], target_boxes_matched[:, :4], reduction="sum", beta=0.1) / num_boxes
        )

        # ProbIoU Loss
        mu1_l, sig1_l = _rotated_boxes_to_gaussian(src_boxes.to(torch.float32))
        mu2_l, sig2_l = _rotated_boxes_to_gaussian(target_boxes_matched.to(torch.float32))
        bhat_loss = _bhattacharyya_distance(mu1_l, sig1_l, mu2_l, sig2_l)
        loss_probiou = (1.0 - torch.exp(-bhat_loss)).to(dtype).sum() / num_boxes

        # Rotation Loss
        pred_rot = F.normalize(src_boxes[:, 4:6], dim=-1, eps=1e-6)
        tgt_rot = F.normalize(target_boxes_matched[:, 4:6], dim=-1, eps=1e-6)
        loss_rot = (1.0 - torch.abs((pred_rot * tgt_rot).sum(dim=-1))).sum() / num_boxes

        return class_weight * loss_ce + bbox_weight * loss_bbox + probiou_weight * loss_probiou + rot_weight * loss_rot


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
