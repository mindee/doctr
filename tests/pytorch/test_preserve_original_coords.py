import numpy as np
import cv2
import pytest

from doctr.utils.geometry import compute_expanded_shape


@pytest.mark.parametrize("angle", [5, 12, -5, -12, 90 + 13, 180 + 13, 270 + 13])
@pytest.mark.parametrize("shape", [(800, 600), (600, 800)])
def test_straighten_inverse_fiducial(angle, shape):
    """M_inv = inv(C @ R @ P) is the exact inverse of the pad→rotate→crop pipeline.

    Coloured 3×3 dots at known positions go through the same warpAffine path as
    _straighten_pages.  After locating each dot in the cropped output by exact
    colour match, M_inv is applied and the distance to the original dot centre
    must be below 0.6 px (the interpolation noise floor of discrete fiducials).

    ±0.5° cases are excluded: at sub‑degree rotations every fiducial pixel is
    blended by interpolation, so exact‑colour matching finds zero pixels.
    """
    h, w = shape
    page = np.ones((h, w, 3), dtype=np.uint8) * 255
    # 3×3 dots centred on (dx, dy)  --  slice dx-1:dx+2, dy-1:dy+2
    page[99:102, 99:102] = (255, 0, 0)  # (100, 100)
    page[99:102, 499:502] = (0, 255, 0)  # (500, 100)
    page[699:702, 99:102] = (0, 0, 255)  # (100, 700)
    page[699:702, 499:502] = (255, 255, 0)  # (500, 700)
    page[399:402, 299:302] = (255, 0, 255)  # (300, 400)
    dots = [(100, 100), (500, 100), (100, 700), (500, 700), (300, 400)]
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # ---- Inline pad / rotate / aspect‑pad / crop  (same path as _straighten_pages) ----
    expand = h != w
    if expand:
        exp = compute_expanded_shape((h, w), angle)
        h_pad = int(max(0, np.ceil(exp[0] - h)))
        w_pad = int(max(0, np.ceil(exp[1] - w)))
    else:
        h_pad = w_pad = 0
    pt, pb = h_pad // 2, h_pad - h_pad // 2
    pl, pr = w_pad // 2, w_pad - w_pad // 2

    exp_img = np.pad(page, ((pt, pb), (pl, pr), (0, 0)))
    rot_mat = cv2.getRotationMatrix2D((exp_img.shape[1] / 2, exp_img.shape[0] / 2), angle, 1.0)
    rotated = cv2.warpAffine(exp_img, rot_mat, (exp_img.shape[1], exp_img.shape[0]))

    if expand:
        if rotated.shape[0] / rotated.shape[1] > h / w:
            w_pad2 = int(rotated.shape[0] * w / h - rotated.shape[1])
            rotated = np.pad(rotated, ((0, 0), (0, w_pad2), (0, 0)))
        else:
            h_pad2 = int(rotated.shape[1] * h / w - rotated.shape[0])
            rotated = np.pad(rotated, ((0, h_pad2), (0, 0), (0, 0)))

    corners = np.array([[pl, pt, 1], [pl + w, pt, 1], [pl + w, pt + h, 1], [pl, pt + h, 1]], dtype=np.float64).T
    rc = rot_mat @ corners
    cx, cy = int(np.floor(rc[0].min())), int(np.floor(rc[1].min()))
    cropped = rotated[cy:, cx:]

    # ---- Build M_inv ----
    C3 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R3 = np.vstack([rot_mat, [0, 0, 1]])
    P3 = np.array([[1, 0, pl], [0, 1, pt], [0, 0, 1]], dtype=np.float64)
    M_inv = np.linalg.inv(C3 @ R3 @ P3)

    # ---- Locate dots in cropped image by exact colour match ----
    errors = []
    for (dx, dy), colour in zip(dots, colours):
        mask = np.all(cropped == colour, axis=-1)
        found = np.argwhere(mask)
        if len(found) == 0:
            continue
        fy, fx = found.mean(axis=0)
        recovered = (np.array([float(fx), float(fy), 1.0]) @ M_inv.T)[:2]
        errors.append(np.linalg.norm(recovered - np.array([float(dx), float(dy)])))

    assert len(errors) > 0, "No fiducial dots found — interpolation blended all pixels"
    assert max(errors) < 0.6, f"Max remap error {max(errors):.4f}px exceeds 0.6px threshold"
