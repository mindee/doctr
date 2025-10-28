# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import cv2
import numpy as np

from doctr.utils.geometry import rotate_abs_geoms

__all__ = ["crop_boxes", "create_shadow_mask"]


def crop_boxes(
    boxes: np.ndarray,
    crop_box: tuple[int, int, int, int] | tuple[float, float, float, float],
) -> np.ndarray:
    """Crop localization boxes

    Args:
        boxes: ndarray of shape (N, 4) in relative or abs coordinates
        crop_box: box (xmin, ymin, xmax, ymax) to crop the image, in the same coord format that the boxes

    Returns:
        the cropped boxes
    """
    is_box_rel = boxes.max() <= 1
    is_crop_rel = max(crop_box) <= 1

    if is_box_rel ^ is_crop_rel:
        raise AssertionError("both the boxes and the crop need to have the same coordinate convention")

    xmin, ymin, xmax, ymax = crop_box
    # Clip boxes & correct offset
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(xmin, xmax) - xmin
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(ymin, ymax) - ymin
    # Rescale relative coords
    if is_box_rel:
        boxes[:, [0, 2]] /= xmax - xmin
        boxes[:, [1, 3]] /= ymax - ymin

    # Remove 0-sized boxes
    is_valid = np.logical_and(boxes[:, 1] < boxes[:, 3], boxes[:, 0] < boxes[:, 2])

    return boxes[is_valid]


def expand_line(line: np.ndarray, target_shape: tuple[int, int]) -> tuple[float, float]:
    """Expands a 2-point line, so that the first is on the edge. In other terms, we extend the line in
    the same direction until we meet one of the edges.

    Args:
        line: array of shape (2, 2) of the point supposed to be on one edge, and the shadow tip.
        target_shape: the desired mask shape

    Returns:
        2D coordinates of the first point once we extended the line (on one of the edges)
    """
    if any(coord == 0 or coord == size for coord, size in zip(line[0], target_shape[::-1])):
        return line[0]
    # Get the line equation
    _tmp = line[1] - line[0]
    _direction = _tmp > 0
    _flat = _tmp == 0
    # vertical case
    if _tmp[0] == 0:
        solutions = [
            # y = 0
            (line[0, 0], 0),
            # y = bot
            (line[0, 0], target_shape[0]),
        ]
    # horizontal
    elif _tmp[1] == 0:
        solutions = [
            # x = 0
            (0, line[0, 1]),
            # x = right
            (target_shape[1], line[0, 1]),
        ]
    else:
        alpha = _tmp[1] / _tmp[0]
        beta = line[1, 1] - alpha * line[1, 0]

        # Solve it for edges
        solutions = [
            # x = 0
            (0, beta),
            # y = 0
            (-beta / alpha, 0),
            # x = right
            (target_shape[1], alpha * target_shape[1] + beta),
            # y = bot
            ((target_shape[0] - beta) / alpha, target_shape[0]),
        ]
    for point in solutions:
        # Skip points that are out of the final image
        if any(val < 0 or val > size for val, size in zip(point, target_shape[::-1])):
            continue
        if all(
            val == ref if _same else (val < ref if _dir else val > ref)
            for val, ref, _dir, _same in zip(point, line[1], _direction, _flat)
        ):
            return point
    raise ValueError


def create_shadow_mask(
    target_shape: tuple[int, int],
    min_base_width=0.3,
    max_tip_width=0.5,
    max_tip_height=0.3,
) -> np.ndarray:
    """Creates a random shadow mask

    Args:
        target_shape: the target shape (H, W)
        min_base_width: the relative minimum shadow base width
        max_tip_width: the relative maximum shadow tip width
        max_tip_height: the relative maximum shadow tip height

    Returns:
        a numpy ndarray of shape (H, W, 1) with values in the range [0, 1]
    """
    # Default base is top
    _params = np.random.rand(6)
    base_width = min_base_width + (1 - min_base_width) * _params[0]
    base_center = base_width / 2 + (1 - base_width) * _params[1]
    # Ensure tip width is smaller for shadow consistency
    tip_width = min(_params[2] * base_width * target_shape[0] / target_shape[1], max_tip_width)
    tip_center = tip_width / 2 + (1 - tip_width) * _params[3]
    tip_height = _params[4] * max_tip_height
    tip_mid = tip_height / 2 + (1 - tip_height) * _params[5]
    _order = tip_center < base_center
    contour: np.ndarray = np.array(
        [
            [base_center - base_width / 2, 0],
            [base_center + base_width / 2, 0],
            [tip_center + tip_width / 2, tip_mid + tip_height / 2 if _order else tip_mid - tip_height / 2],
            [tip_center - tip_width / 2, tip_mid - tip_height / 2 if _order else tip_mid + tip_height / 2],
        ],
        dtype=np.float32,
    )

    # Convert to absolute coords
    abs_contour: np.ndarray = (
        np.stack(
            (contour[:, 0] * target_shape[1], contour[:, 1] * target_shape[0]),
            axis=-1,
        )
        .round()
        .astype(np.int32)
    )

    # Direction
    _params = np.random.rand(1)
    rotated_contour = (
        rotate_abs_geoms(
            abs_contour[None, ...],
            360 * _params[0],
            target_shape,
            expand=False,
        )[0]
        .round()
        .astype(np.int32)
    )
    # Check approx quadrant
    quad_idx = int(_params[0] / 0.25)
    # Top-bot
    if quad_idx % 2 == 0:
        intensity_mask = np.repeat(np.arange(target_shape[0])[:, None], target_shape[1], axis=1) / (target_shape[0] - 1)
        if quad_idx == 0:
            intensity_mask = 1 - intensity_mask
    # Left - right
    else:
        intensity_mask = np.repeat(np.arange(target_shape[1])[None, :], target_shape[0], axis=0) / (target_shape[1] - 1)
        if quad_idx == 1:
            intensity_mask = 1 - intensity_mask

    # Expand base
    final_contour = rotated_contour.copy()
    final_contour[0] = expand_line(final_contour[[0, 3]], target_shape)
    final_contour[1] = expand_line(final_contour[[1, 2]], target_shape)
    # If both base are not on the same side, add a point
    if not np.any(final_contour[0] == final_contour[1]):
        corner_x = 0 if max(final_contour[0, 0], final_contour[1, 0]) < target_shape[1] else target_shape[1]
        corner_y = 0 if max(final_contour[0, 1], final_contour[1, 1]) < target_shape[0] else target_shape[0]
        corner: np.ndarray = np.array([corner_x, corner_y])
        final_contour = np.concatenate((final_contour[:1], corner[None, ...], final_contour[1:]), axis=0)

    # Direction & rotate
    mask: np.ndarray = np.zeros((*target_shape, 1), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [final_contour], (255,), lineType=cv2.LINE_AA)[..., 0]

    return (mask / 255).astype(np.float32).clip(0, 1) * intensity_mask.astype(np.float32)
