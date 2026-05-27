import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch import Tensor

__all__ = ["lw_detr_for_object_detection_loss"]


def center_to_corners_format(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        num_boxes: Normalization factor, typically the number of target boxes in the batch. This is used to scale the
            loss to an absolute value, and is used in the original implementation of DETR and LW-DETR.
            It doesn't have to be
            exactly the number of target boxes, but it should be correlated to it for the loss to be meaningful.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`):
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        num_boxes (`int`):
            Normalization factor, typically the number of target boxes in the batch. This is used to scale the loss
            to an absolute value, and is used in the original implementation of DETR and LW-DETR. It doesn't have to be
            exactly the number of target boxes, but it should be correlated to it for the loss to be meaningful.
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# below: taken from https://github.com/facebookresearch/detr/blob/master/util/misc.py#L306
def _max_by_axis(the_list):
    # type: (list[list[int]]) -> list[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors, mask: Tensor | None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: list[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
def _set_aux_loss(outputs_class, outputs_coord):
    return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class LwDetrHungarianMatcher(nn.Module):
    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr):
        """
        Differences:
        - out_prob = outputs["logits"].flatten(0, 1).sigmoid() instead of softmax
        - class_cost uses alpha and gamma
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([torch.as_tensor(v["class_labels"], dtype=torch.int64) for v in targets]).to(
            out_prob.device
        )
        target_bbox = torch.cat([torch.as_tensor(v["boxes"], dtype=torch.float32) for v in targets]).to(out_bbox.device)

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes, cdist only supports float32
        dtype = out_bbox.dtype
        out_bbox = out_bbox.to(torch.float32)
        target_bbox = target_bbox.to(torch.float32)
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        bbox_cost = bbox_cost.to(dtype)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        group_num_queries = num_queries // group_detr
        cost_matrix_list = cost_matrix.split(group_num_queries, dim=1)
        for group_id in range(group_detr):
            group_cost_matrix = cost_matrix_list[group_id]
            group_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(group_cost_matrix.split(sizes, -1))]
            if group_id == 0:
                indices = group_indices
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + group_num_queries * group_id]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, group_indices)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class LwDetrImageLoss(nn.Module):
    def __init__(self, matcher, num_classes, focal_alpha, losses, group_detr):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.group_detr = group_detr

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]
        dtype = source_logits.dtype

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([
            torch.as_tensor(np.atleast_1d(t["class_labels"][J]), dtype=torch.int64)
            for t, (_, J) in zip(targets, indices)
        ]).to(source_logits.device)
        alpha = self.focal_alpha
        gamma = 2
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [torch.as_tensor(np.atleast_2d(t["boxes"][i]), dtype=torch.float32) for t, (_, i) in zip(targets, indices)],
            dim=0,
        ).to(src_boxes.device)
        iou_targets = torch.diag(
            box_iou(center_to_corners_format(src_boxes.detach()), center_to_corners_format(target_boxes))[0]
        )
        # Convert to the same dtype as the source logits as box_iou upcasts to float32
        iou_targets = iou_targets.to(dtype)
        pos_ious = iou_targets.clone().detach()
        prob = source_logits.sigmoid()
        # init positive weights and negative weights
        pos_weights = torch.zeros_like(source_logits)
        # pow promotes to float32 under float16 CUDA autocast; cast back to preserve original dtype
        neg_weights = prob.pow(gamma).to(dtype)
        pos_ind = idx + (target_classes_o,)

        pos_quality = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
        pos_quality = torch.clamp(pos_quality, 0.01).detach().to(dtype)

        pos_weights[pos_ind] = pos_quality
        neg_weights[pos_ind] = 1 - pos_quality
        loss_ce = -pos_weights * prob.log() - neg_weights * (1 - prob).log()
        loss_ce = loss_ce.sum() / num_boxes
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (sigmoid > 0.5 threshold)
        card_pred = (logits.sigmoid().max(-1).values > 0.5).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    # Copied from loss.loss_for_object_detection.ImageLoss.loss_boxes
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [torch.as_tensor(np.atleast_2d(t["boxes"][i]), dtype=torch.float32) for t, (_, i) in zip(targets, indices)],
            dim=0,
        ).to(source_boxes.device)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # Copied from loss.loss_for_object_detection.ImageLoss.loss_masks
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    # Copied from loss.loss_for_object_detection.ImageLoss._get_source_permutation_idx
    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # Copied from loss.loss_for_object_detection.ImageLoss._get_target_permutation_idx
    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`list[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux_and_enc = {
            k: v for k, v in outputs.items() if k != "enc_outputs" and k != "auxiliary_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux_and_enc, targets, group_detr)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets, group_detr)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def lw_detr_for_object_detection_loss(
    logits,
    labels,
    device,
    pred_boxes,
    outputs_class=None,
    outputs_coord=None,
    enc_outputs_class=None,
    enc_outputs_coord=None,
    use_aux_loss=False,
    group_detr=1,
    num_labels=None,
    num_decoder_layers=None,
    **kwargs,
):
    """Loss computation for LW-DETR for object detection."""
    # First: create the matcher
    matcher = LwDetrHungarianMatcher(class_cost=2.0, bbox_cost=5, giou_cost=2)
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality"]
    criterion = LwDetrImageLoss(
        matcher=matcher,
        num_classes=num_labels,
        focal_alpha=0.1,
        losses=losses,
        group_detr=group_detr,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    auxiliary_outputs = None
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    outputs_loss["enc_outputs"] = {
        "logits": enc_outputs_class,
        "pred_boxes": enc_outputs_coord,
    }
    if use_aux_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs
    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": 5}
    weight_dict["loss_giou"] = 2
    if use_aux_loss:
        aux_weight_dict = {}
        for i in range(num_decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    enc_weight_dict = {k + "_enc": v for k, v in weight_dict.items()}
    weight_dict.update(enc_weight_dict)
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs
