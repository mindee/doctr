# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np

__all__ = ["_remove_padding"]


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
        # Rectify loc_preds to remove padding
        rectified_preds = []
        for page, dict_loc_preds in zip(pages, loc_preds):
            for k, loc_pred in dict_loc_preds.items():
                h, w = page.shape[0], page.shape[1]
                if h > w:
                    # y unchanged, dilate x coord
                    if symmetric_pad:
                        if assume_straight_pages:
                            loc_pred[:, [0, 2]] = (loc_pred[:, [0, 2]] - 0.5) * h / w + 0.5
                        else:
                            loc_pred[:, :, 0] = (loc_pred[:, :, 0] - 0.5) * h / w + 0.5
                    else:
                        if assume_straight_pages:
                            loc_pred[:, [0, 2]] *= h / w
                        else:
                            loc_pred[:, :, 0] *= h / w
                elif w > h:
                    # x unchanged, dilate y coord
                    if symmetric_pad:
                        if assume_straight_pages:
                            loc_pred[:, [1, 3]] = (loc_pred[:, [1, 3]] - 0.5) * w / h + 0.5
                        else:
                            loc_pred[:, :, 1] = (loc_pred[:, :, 1] - 0.5) * w / h + 0.5
                    else:
                        if assume_straight_pages:
                            loc_pred[:, [1, 3]] *= w / h
                        else:
                            loc_pred[:, :, 1] *= w / h
                rectified_preds.append({k: np.clip(loc_pred, 0, 1)})
        return rectified_preds
    return loc_preds
