# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from doctr.models.classification import vit_det_m, vit_det_s

from ...utils import load_pretrained_params
from .base import _LWDETR, LWDETRPostProcessor
from .layers import LWDETRDecoder, LWDETRHead, LWDETRMultiscaleDeformableAttention, MultiScaleProjector
from .loss import lw_detr_for_object_detection_loss

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
        score_thresh: float = 0.0,
        iou_thresh: float = 0.1,
        d_model: int = 256,
        num_queries: int = 130,
        group_detr: int = 1,
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
        self.num_classes = len(self.class_names) + 1  # +1 for background class (NO OBJECT)
        self.cfg = cfg
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor

        self.group_detr = group_detr
        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = dec_layers

        self.reference_point_embed = nn.Embedding(self.num_queries * self.group_detr, 4)
        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, self.d_model)

        self.decoder = LWDETRDecoder(
            num_layers=self.dec_layers,
            d_model=d_model,
            sa_num_heads=sa_num_heads,
            ca_num_heads=ca_num_heads,
            ff_dim=ff_dim,
            dec_n_points=dec_n_points,
            group_detr=group_detr,
            dropout_prob=dropout_prob,
        )

        self.enc_output = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(self.group_detr)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.group_detr)])

        self.enc_out_bbox_embed = nn.ModuleList([
            LWDETRHead(self.d_model, self.d_model, 4, num_layers=3) for _ in range(self.group_detr)
        ])
        self.enc_out_class_embed = nn.ModuleList([
            nn.Linear(self.d_model, self.num_classes) for _ in range(self.group_detr)
        ])

        self.class_embed = nn.Linear(self.d_model, self.num_classes)
        self.bbox_embed = LWDETRHead(self.d_model, self.d_model, 4, num_layers=3)

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
            if isinstance(m, LWDETRMultiscaleDeformableAttention):
                nn.init.constant_(m.sampling_offsets.weight, 0.0)
                thetas = torch.arange(m.n_heads, dtype=torch.int64).float() * (2.0 * math.pi / m.n_heads)
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
                nn.init.constant_(m.value_proj.bias, 0.0)
                nn.init.xavier_uniform_(m.output_proj.weight)
                nn.init.constant_(m.output_proj.bias, 0.0)
            if hasattr(m, "refpoint_embed") and m.refpoint_embed is not None:
                nn.init.constant_(m.refpoint_embed.weight, 0)
            if hasattr(m, "class_embed") and m.class_embed is not None:
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(m.class_embed.bias, bias_value)
            if hasattr(m, "bbox_embed") and m.bbox_embed is not None:
                nn.init.constant_(m.bbox_embed.layers[-1].weight, 0)
                nn.init.constant_(m.bbox_embed.layers[-1].bias, 0)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)

    def refine_bboxes(self, reference_points, deltas):
        reference_points = reference_points.to(deltas.device)
        new_reference_points_cxcy = deltas[..., :2] * reference_points[..., 2:] + reference_points[..., :2]
        new_reference_points_wh = deltas[..., 2:].exp() * reference_points[..., 2:]
        new_reference_points = torch.cat((new_reference_points_cxcy, new_reference_points_wh), -1)
        return new_reference_points

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""
        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (list[tuple[int, int]]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals in [0, 1] space.
                  Invalid positions (padding or out-of-bounds) are filled with 0.
                - invalid_mask (Tensor[batch_size, sequence_length, 1]): Boolean mask that is True for invalid positions
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
            proposal = torch.cat((grid, width_height), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        invalid_mask = padding_mask.unsqueeze(-1) | ~output_proposals_valid
        output_proposals = output_proposals.masked_fill(invalid_mask, float(0))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(invalid_mask, float(0))
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
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in feats_masks], 1)

        tgt = query_feat.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        object_query_embedding, output_proposals, invalid_mask = self.gen_encoder_output_proposals(
            source_flatten, mask_flatten, spatial_shapes_list
        )

        group_detr = self.group_detr if self.training else 1
        topk = self.num_queries

        topk_coords_logits = []
        topk_coords_logits_undetach = []
        object_query_undetach = []

        for group_id in range(group_detr):
            group_object_query = self.enc_output[group_id](object_query_embedding)
            group_object_query = self.enc_output_norm[group_id](group_object_query)

            group_enc_outputs_class = self.enc_out_class_embed[group_id](group_object_query)
            group_enc_outputs_class = group_enc_outputs_class.masked_fill(invalid_mask, float("-inf"))
            group_delta_bbox = self.enc_out_bbox_embed[group_id](group_object_query)
            group_enc_outputs_coord = self.refine_bboxes(output_proposals, group_delta_bbox)

            group_topk_proposals = torch.topk(group_enc_outputs_class.max(-1)[0], topk, dim=1)[1]
            group_topk_coords_logits_undetach = torch.gather(
                group_enc_outputs_coord,
                1,
                group_topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            group_topk_coords_logits = group_topk_coords_logits_undetach.detach()
            group_object_query_undetach = torch.gather(
                group_object_query, 1, group_topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )

            topk_coords_logits.append(group_topk_coords_logits)
            topk_coords_logits_undetach.append(group_topk_coords_logits_undetach)
            object_query_undetach.append(group_object_query_undetach)

        topk_coords_logits = torch.cat(topk_coords_logits, 1)
        topk_coords_logits_undetach = torch.cat(topk_coords_logits_undetach, 1)
        object_query_undetach = torch.cat(object_query_undetach, 1)

        enc_outputs_class_logits = object_query_undetach
        enc_outputs_boxes_logits = topk_coords_logits_undetach

        reference_points = self.refine_bboxes(topk_coords_logits, reference_points)

        init_reference_points = reference_points
        last_hidden_state, intermediate, intermediate_reference_points = self.decoder(
            inputs_embeds=tgt,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
        )

        logits = self.class_embed(last_hidden_state)
        pred_boxes_delta = self.bbox_embed(last_hidden_state)
        pred_boxes = self.refine_bboxes(intermediate_reference_points[-1], pred_boxes_delta)

        enc_outputs_class_logits_list = enc_outputs_class_logits.split(self.num_queries, dim=1)
        pred_class = []
        group_detr = self.group_detr if self.training else 1
        for group_index in range(group_detr):
            group_pred_class = self.enc_out_class_embed[group_index](enc_outputs_class_logits_list[group_index])
            pred_class.append(group_pred_class)
        enc_outputs_class_logits = torch.cat(pred_class, dim=1)

        if target is not None:
            outputs_class, outputs_coord = None, None
            intermediate_hidden_states = intermediate
            outputs_coord_delta = self.bbox_embed(intermediate_hidden_states)
            outputs_coord = self.refine_bboxes(intermediate_reference_points, outputs_coord_delta)
            outputs_class = self.class_embed(intermediate_hidden_states)

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
            out["loss"] = self.compute_loss(
                logits,
                processed_targets,
                pred_boxes,
                outputs_class,
                outputs_coord,
                enc_outputs_class_logits,
                enc_outputs_boxes_logits,
            )

        return out

    def compute_loss(
        self,
        logits,
        targets,
        pred_boxes,
        outputs_class,
        outputs_coord,
        enc_outputs_class_logits,
        enc_outputs_boxes_logits,
    ):

        loss_calc = lw_detr_for_object_detection_loss(
            logits=logits,
            device=logits.device,
            labels=targets,
            pred_boxes=pred_boxes,
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
            enc_outputs_class=enc_outputs_class_logits,
            enc_outputs_coord=enc_outputs_boxes_logits,
            use_aux_loss=True,
            group_detr=self.group_detr,
            num_decoder_layers=self.dec_layers,
            num_labels=self.num_classes,
        )
        return loss_calc[0]


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
