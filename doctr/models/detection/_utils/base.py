# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np

__all__ = ["_remove_padding"]


def _adjust_coords(
    loc_pred: np.ndarray,
    ratio: float,
    symmetric_pad: bool,
    assume_straight_pages: bool,
    axis: int,
) -> None:
    """Adjust coordinates along a given axis to remove padding

    Args:
        loc_pred: localization predictions
        ratio: aspect ratio multiplier
        symmetric_pad: whether the padding was symmetric
        assume_straight_pages: whether the pages are assumed to be straight
        axis: 0 for x coordinates, 1 for y coordinates
    """
    if assume_straight_pages:
        cols = [axis, axis + 2]
        if symmetric_pad:
            loc_pred[:, cols] = (loc_pred[:, cols] - 0.5) * ratio + 0.5
        else:
            loc_pred[:, cols] *= ratio
    else:
        if symmetric_pad:
            loc_pred[:, :, axis] = (loc_pred[:, :, axis] - 0.5) * ratio + 0.5
        else:
            loc_pred[:, :, axis] *= ratio


def _remove_padding(
    pages: list[np.ndarray],
    loc_preds: list[dict[str, np.ndarray]],
    preserve_aspect_ratio: bool,
    symmetric_pad: bool,
    assume_straight_pages: bool,
) -> list[dict[str, np.ndarray]]:
    """Remove padding from the localization predictions

    Args:
        pages: list of pages
        loc_preds: list of localization predictions
        preserve_aspect_ratio: whether the aspect ratio was preserved during padding
        symmetric_pad: whether the padding was symmetric
        assume_straight_pages: whether the pages are assumed to be straight

    Returns:
        list of unpaded localization predictions
    """
    if preserve_aspect_ratio:
        rectified_preds = []
        for page, dict_loc_preds in zip(pages, loc_preds):
            for k, loc_pred in dict_loc_preds.items():
                h, w = page.shape[0], page.shape[1]
                if h > w:
                    _adjust_coords(loc_pred, h / w, symmetric_pad, assume_straight_pages, axis=0)
                elif w > h:
                    _adjust_coords(loc_pred, w / h, symmetric_pad, assume_straight_pages, axis=1)
                rectified_preds.append({k: np.clip(loc_pred, 0, 1)})
        return rectified_preds
    return loc_preds
