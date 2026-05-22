# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math

import torch
import torch.nn.functional as F
from torch import nn

from doctr.models.modules import ChannelLayerNorm
from doctr.models.utils import conv_sequence_pt

__all__ = ["MultiScaleProjector", "C2fBottleneck", "LWDETRHead", "LWDETRDecoder", "LWDETRMultiscaleDeformableAttention"]


class LWDETRHead(nn.Module):
    """
    Simple MLP used as the reference point head in LW-DETR.

    Args:
        input_dim: number of input features
        hidden_dim: number of hidden features
        output_dim: number of output features
        num_layers: number of layers in the MLP
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LWDETRMLP(nn.Module):
    """Simple MLP used in the decoder layers of LW-DETR.

    Args:
        d_model: number of input and output features
        ff_dim: number of hidden features
        dropout_prob: dropout probability
    """

    def __init__(
        self,
        d_model: int,
        ff_dim: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout_2(self.fc2(self.dropout_1(self.act(self.fc1(x)))))


class LWDETRAttention(nn.Module):
    """This module implements the self-attention mechanism used in LW-DETR.
    It performs multi-head self-attention on the input hidden states.
    The group detr technique is used during training to add more supervision by
    using multiple weight-sharing decoders at once for faster convergence.

    Args:
        sa_num_heads: number of attention heads for self-attention
        d_model: number of input and output features
        dropout_prob: dropout probability for attention weights
        group_detr: number of weight-sharing decoders to use during training
        layer_idx: index of the decoder layer (used for group detr)
    """

    def __init__(
        self,
        sa_num_heads: int = 8,
        d_model: int = 256,
        dropout_prob: float = 0.0,
        group_detr: int = 13,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = d_model // sa_num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout_prob
        self.group_detr = group_detr

        self.q_proj = nn.Linear(d_model, sa_num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(d_model, sa_num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(d_model, sa_num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(sa_num_heads * self.head_dim, d_model, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        if self.training:
            # Crash prevention: ensure seq_len is perfectly divisible
            assert seq_len % self.group_detr == 0, (
                f"Seq len {seq_len} must be divisible by group_detr {self.group_detr}"
            )  # noqa: E501

        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = hidden_states if position_embeddings is None else hidden_states + position_embeddings

        if self.training:
            # at training, we use group detr technique to
            # add more supervision by using multiple weight-sharing decoders at once for faster convergence
            # at inference, we only use one decoder
            hidden_states_original = torch.cat(hidden_states_original.split(seq_len // self.group_detr, dim=1), dim=0)
            hidden_states = torch.cat(hidden_states.split(seq_len // self.group_detr, dim=1), dim=0)

        attention_input_shape = hidden_states.shape[:-1]
        hidden_shape = (*attention_input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states_original).view(hidden_shape).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*attention_input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if self.training:
            attn_output = torch.cat(torch.split(attn_output, batch_size, dim=0), dim=1)

        return attn_output, attn_weights


class MultiScaleDeformableAttention(nn.Module):
    """This module implements MultiScaleDeformableAttention from Deformable DETR.
    It performs multi-scale deformable attention on the input feature maps.
    Borrowed from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/modeling_deformable_detr.py
    """

    def forward(
        self,
        value: torch.Tensor,
        value_spatial_shapes_list: list[tuple],
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
        value_list = value.split([height * width for height, width in value_spatial_shapes_list], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level_id, (height, width) in enumerate(value_spatial_shapes_list):
            # batch_size, height*width, num_heads, hidden_dim
            # -> batch_size, height*width, num_heads*hidden_dim
            # -> batch_size, num_heads*hidden_dim, height*width
            # -> batch_size*num_heads, hidden_dim, height, width
            value_l_ = (
                value_list[level_id]
                .flatten(2)
                .transpose(1, 2)
                .reshape(batch_size * num_heads, hidden_dim, height, width)
            )
            # batch_size, num_queries, num_heads, num_points, 2
            # -> batch_size, num_heads, num_queries, num_points, 2
            # -> batch_size*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
            # batch_size*num_heads, hidden_dim, num_queries, num_points
            sampling_value_l_ = nn.functional.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)
        # (batch_size, num_queries, num_heads, num_levels, num_points)
        # -> (batch_size, num_heads, num_queries, num_levels, num_points)
        # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
        attention_weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads, 1, num_queries, num_levels * num_points
        )
        output = (
            (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
            .sum(-1)
            .view(batch_size, num_heads * hidden_dim, num_queries)
        )
        return output.transpose(1, 2).contiguous()


class LWDETRMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.

    Args:
        d_model: number of input and output features
        ca_num_heads: number of attention heads for cross-attention
        dec_n_points: number of sampling points for each attention head
    """

    def __init__(
        self,
        d_model: int = 256,
        ca_num_heads: int = 16,
        dec_n_points: int = 2,
    ):
        super().__init__()

        self.attn = MultiScaleDeformableAttention()

        self.d_model = d_model
        self.n_levels = 1
        self.n_heads = ca_num_heads
        self.n_points = dec_n_points

        self.sampling_offsets = nn.Linear(d_model, self.n_heads * self.n_levels * dec_n_points * 2)
        self.attention_weights = nn.Linear(d_model, self.n_heads * self.n_levels * dec_n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states=None,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes_list=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]

        if num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        elif num_coordinates == 6:
            ref = reference_points[:, :, None, :, None, :]  # (..., 6)

            center = ref[..., :2]  # (cx, cy)
            wh = ref[..., 2:4]  # (w, h)
            sin = ref[..., 4:5]  # sinθ
            cos = ref[..., 5:6]  # cosθ

            # normalize offsets
            offsets = sampling_offsets / self.n_points * wh * 0.5

            dx = offsets[..., 0:1]
            dy = offsets[..., 1:2]

            # rotate offsets
            dx_rot = dx * cos - dy * sin
            dy_rot = dx * sin + dy * cos

            rotated_offsets = torch.cat([dx_rot, dy_rot], dim=-1)

            sampling_locations = center + rotated_offsets
        else:
            raise ValueError(f"Last dim of reference_points must be 4 or 6, but got {reference_points.shape[-1]}")

        output = self.attn(
            value,
            spatial_shapes_list,
            sampling_locations,
            attention_weights,
        )

        output = self.output_proj(output)

        return output, attention_weights


class LWDETRDecoderLayer(nn.Module):
    """This module implements a single decoder layer of LW-DETR,
    which consists of self-attention, cross-attention and an MLP.

    Args:
        d_model: number of input and output features
        ff_dim: number of hidden features in the MLP
        dropout_prob: dropout probability for the attention and MLP layers
        ca_num_heads: number of attention heads for cross-attention
        dec_n_points: number of sampling points for each attention head in cross-attention
        sa_num_heads: number of attention heads for self-attention
        group_detr: number of weight-sharing decoders to use during training for the group detr technique
        layer_idx: index of the decoder layer (used for group detr)
    """

    def __init__(
        self,
        d_model: int = 256,
        ff_dim: int = 2048,
        dropout_prob: float = 0.0,
        ca_num_heads: int = 16,
        dec_n_points: int = 2,
        sa_num_heads: int = 8,
        group_detr: int = 13,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.dropout = dropout_prob

        # self-attention
        self.self_attn = LWDETRAttention(
            sa_num_heads=sa_num_heads,
            d_model=d_model,
            dropout_prob=dropout_prob,
            group_detr=group_detr,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        # cross-attention
        self.cross_attn = LWDETRMultiscaleDeformableAttention(
            d_model=d_model,
            ca_num_heads=ca_num_heads,
            dec_n_points=dec_n_points,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(d_model)

        # mlp
        self.mlp = LWDETRMLP(d_model, ff_dim, dropout_prob)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple] | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attention_output, self_attn_weights = self.self_attn(
            hidden_states, position_embeddings=position_embeddings
        )

        self_attention_output = F.dropout(self_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + self_attention_output
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attention_output, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes_list=spatial_shapes_list,
        )
        cross_attention_output = F.dropout(cross_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + cross_attention_output
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


# function to generate sine positional embedding for 4d coordinates
# Borrowed from: https://github.com/Atten4Vis/LW-DETR/blob/main/models/transformer.py
def gen_sine_position_embeddings(pos_tensor: torch.Tensor, hidden_size: int = 256) -> torch.Tensor:
    """
    This function computes position embeddings using sine and cosine functions from the input positional tensor,
    which has a shape of (batch_size, num_queries, 4).
    The last dimension of `pos_tensor` represents the following coordinates:
    - 0: x-coord
    - 1: y-coord
    - 2: width
    - 3: height

    The output shape is (batch_size, num_queries, 512),
    where final dim (hidden_size*2 = 512) is the total embedding dimension
    achieved by concatenating the sine and cosine values for each coordinate.
    """
    scale = 2 * math.pi
    dim = hidden_size // 2
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos.to(pos_tensor.dtype)


class LWDETRDecoder(nn.Module):
    """This module implements the decoder of LW-DETR,
    which consists of multiple decoder layers and a reference point head.

    Args:
        num_layers: number of decoder layers
        d_model: number of input and output features for each decoder layer
        sa_num_heads: number of attention heads for self-attention in each decoder layer
        ca_num_heads: number of attention heads for cross-attention in each decoder layer
        ff_dim: number of hidden features in the MLP of each decoder layer
        dec_n_points: number of sampling points for each attention head in cross-attention of each decoder layer
        group_detr: number of weight-sharing decoders to use during training for the group detr technique
        dropout_prob: dropout probability for the attention and MLP layers in each decoder layer
        bbox_embed: module to predict bounding box deltas for iterative refinement of reference points
    """

    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 256,
        sa_num_heads: int = 8,
        ca_num_heads: int = 16,
        ff_dim: int = 2048,
        dec_n_points: int = 2,
        group_detr: int = 13,
        dropout_prob: float = 0.0,
        bbox_embed: nn.Module | None = None,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.d_model = d_model
        self.layers = nn.ModuleList([
            LWDETRDecoderLayer(
                d_model=self.d_model,
                sa_num_heads=sa_num_heads,
                ca_num_heads=ca_num_heads,
                ff_dim=ff_dim,
                dec_n_points=dec_n_points,
                group_detr=group_detr,
                dropout_prob=dropout_prob,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(self.d_model)
        self.bbox_embed = bbox_embed

        self.ref_point_head = LWDETRHead(2 * self.d_model, self.d_model, self.d_model, num_layers=2)
        self.angle_proj = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def get_reference(
        self, reference_points: torch.Tensor, valid_ratios: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This function computes the reference point inputs and positional embeddings for the decoder layers.

        Args:
            reference_points: (batch_size, num_queries, 6)
                tensor containing the current reference points in the format (cx, cy, w, h, sinθ, cosθ)
            valid_ratios: (batch_size, num_levels, 2)
                tensor containing the valid ratios for each level of the input feature maps

        Returns:
            reference_points_inputs: (batch_size, num_queries, 1, num_levels, 4)
                tensor containing the reference point inputs for the decoder layers,
                which are the normalized center coordinates,
                width and height of the bounding boxes w.r.t. the valid ratios of the input feature maps
            query_pos: (batch_size, num_queries, d_model)
                tensor containing the positional embeddings for the decoder layers,
                which are computed from the reference points using sine and cosine functions and a linear projection
        """
        obj_center = reference_points[..., :4]
        spatial_inputs = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        # Extract angles
        angle = reference_points[..., 4:6]  # (sin, cos)
        angle_expanded = angle[:, :, None]
        reference_points_inputs = torch.cat([spatial_inputs, angle_expanded], dim=-1)
        # DETR positional encoding
        query_sine_embed = gen_sine_position_embeddings(spatial_inputs[:, :, 0, :], self.d_model)
        base_query_pos = self.ref_point_head(query_sine_embed)
        # Angle embedding
        sin_t = angle[..., 0:1]
        cos_t = angle[..., 1:2]

        angle_feat = torch.cat(
            [
                sin_t,
                cos_t,
                2 * sin_t * cos_t,
                cos_t**2 - sin_t**2,
            ],
            dim=-1,
        )

        angle_emb = self.angle_proj(angle_feat)
        # Combine
        query_pos = base_query_pos + angle_emb
        return reference_points_inputs, query_pos

    def refine_boxes(self, reference_points: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        reference_points = reference_points.to(deltas.device)
        cxcy = deltas[..., :2] * reference_points[..., 2:4] + reference_points[..., :2]

        # Clamp deltas to prevent exp() from shooting to Infinity during early training
        wh = torch.clamp(deltas[..., 2:4], min=-4.0, max=2.0).exp() * reference_points[..., 2:4]

        # Add eps=1e-6 to avoid division-by-zero NaN creation
        delta_rot = F.normalize(deltas[..., 4:6], dim=-1, eps=1e-6)
        sin_delta = delta_rot[..., 0:1]
        cos_delta = delta_rot[..., 1:2]
        sin_ref = reference_points[..., 4:5]
        cos_ref = reference_points[..., 5:6]

        sin_new = sin_ref * cos_delta + cos_ref * sin_delta
        cos_new = cos_ref * cos_delta - sin_ref * sin_delta

        # Add eps=1e-6 here too
        rot = F.normalize(torch.cat([sin_new, cos_new], dim=-1), dim=-1, eps=1e-6)

        return torch.cat((cxcy, wh, rot), dim=-1)

    def forward(
        self,
        inputs_embeds: torch.Tensor | None,
        reference_points: torch.Tensor,
        spatial_shapes_list: torch.Tensor,
        valid_ratios: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        intermediate: list[torch.Tensor] = []

        intermediate_reference_points: list[torch.Tensor] = [reference_points]

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        reference_points_inputs, query_pos = self.get_reference(reference_points, valid_ratios)

        for lid, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_embeddings=query_pos,
                reference_points=reference_points_inputs,
                spatial_shapes_list=spatial_shapes_list,
            )

            hidden_states_norm = self.layernorm(hidden_states)

            # iterative refinement
            if self.bbox_embed is not None:
                delta = self.bbox_embed(hidden_states_norm)

                reference_points = self.refine_boxes(
                    reference_points.squeeze(2),
                    delta,
                )

                intermediate_reference_points.append(reference_points)

                reference_points_inputs, query_pos = self.get_reference(
                    reference_points,
                    valid_ratios,
                )

            intermediate.append(hidden_states_norm)

        intermediate_stack = torch.stack(intermediate)
        last_hidden_state = intermediate_stack[-1]

        intermediate_reference_points_stack = torch.stack(intermediate_reference_points)

        return last_hidden_state, intermediate_stack, intermediate_reference_points_stack


class MultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    This is modified from the original MultiScaleProjector to use only the levels used in LW-DETR small and medium.

    Args:
        in_channels (list[int]): list of input channels for each level of the input feature maps.
        out_channels (int): number of channels in the output feature maps.
        num_blocks (int): number of blocks in the C2fBottleneck.
    """

    def __init__(self, in_channels: list[int], out_channels: int, num_blocks: int = 3):
        super().__init__()

        self.use_extra_pool = False

        self.stages_sampling = nn.ModuleList()
        self.stages = nn.ModuleList()

        sampling_layers = nn.ModuleList()
        out_dim: int = 0

        for in_dim in in_channels:
            layers, out_dim = [nn.Identity()], in_dim
            sampling_layers.append(nn.Sequential(*layers))

        self.stages_sampling.append(sampling_layers)

        fusion_in_dim = out_dim * len(in_channels)

        self.stages.append(
            nn.Sequential(
                C2fBottleneck(fusion_in_dim, out_channels, num_blocks),
                ChannelLayerNorm(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        feats = [layer(xi) for layer, xi in zip(self.stages_sampling[0], x)]  # type: ignore[call-overload]
        fused = torch.cat(feats, dim=1)
        return [self.stages[0](fused)]


class C2fBottleneck(nn.Module):
    """Faster implementation of CSP bottleneck with 2 convolutions and 1 residual connection.

    Args:
        input_dim: number of input channels
        out_channels: number of output channels
        num_blocks: number of bottleneck blocks
    """

    def __init__(self, input_dim: int, out_channels: int, num_blocks: int):
        super().__init__()

        self.c = int(out_channels * 0.5)

        self.conv_seq_1 = nn.Sequential(
            *conv_sequence_pt(
                in_channels=input_dim,
                out_channels=2 * self.c,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                dilation=1,
                act=True,
                bias=False,
                bn=True,
                activation=nn.SiLU(inplace=True),
            )
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                *conv_sequence_pt(
                    in_channels=self.c,
                    out_channels=self.c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    dilation=1,
                    act=True,
                    bias=False,
                    bn=True,
                    activation=nn.SiLU(inplace=True),
                ),
                *conv_sequence_pt(
                    in_channels=self.c,
                    out_channels=self.c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    dilation=1,
                    act=True,
                    bias=False,
                    bn=True,
                    activation=nn.SiLU(inplace=True),
                ),
            )
            for _ in range(num_blocks)
        ])

        self.conv_seq_2 = nn.Sequential(
            *conv_sequence_pt(
                in_channels=(2 + num_blocks) * self.c,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                dilation=1,
                act=True,
                bias=False,
                bn=True,
                activation=nn.SiLU(inplace=True),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.conv_seq_1(x).split((self.c, self.c), dim=1))

        for block in self.blocks:
            y.append(block(y[-1]))

        return self.conv_seq_2(torch.cat(y, dim=1))
