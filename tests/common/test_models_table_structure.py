import numpy as np
import pytest

from doctr.models.table_structure.tablecenternet import TableCenterNetPostProcessor, _TableCenterNet


def _grid_target(rows: int, cols: int, use_polygons: bool) -> dict[str, np.ndarray]:
    """A relative-coordinate ``{"cells", "logic"}`` target for a ``rows x cols`` grid."""
    xs, ys = np.linspace(0.1, 0.9, cols + 1), np.linspace(0.1, 0.9, rows + 1)
    cells, logic = [], []
    for r in range(rows):
        for c in range(cols):
            cells.append([[xs[c], ys[r]], [xs[c + 1], ys[r]], [xs[c + 1], ys[r + 1]], [xs[c], ys[r + 1]]])
            logic.append([c, c, r, r])

    cell_array = np.asarray(cells, dtype=np.float32).reshape(-1, 4, 2)
    if not use_polygons:
        cell_array = np.concatenate([cell_array.min(axis=1), cell_array.max(axis=1)], axis=1)
    return {"cells": cell_array, "logic": np.asarray(logic, dtype=np.int64).reshape(-1, 4)}


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_tablecenternet_postprocessor(assume_straight_pages):
    postprocessor = TableCenterNetPostProcessor(
        center_thresh=0.0,
        assume_straight_pages=assume_straight_pages,
    )
    kc, kn, feat = 12, 16, 64
    decoded = {
        "center_polygons": (np.random.rand(1, kc, 8) * feat).astype(np.float32),
        "center_scores": np.random.rand(1, kc).astype(np.float32),
        "center_spans": np.random.randint(1, 3, (1, kc, 2)).astype(np.float32),
        "corner_polygons": (np.random.rand(1, kn, 8) * feat).astype(np.float32),
        "corner_scores": np.random.rand(1, kn).astype(np.float32),
        "corner_points": (np.random.rand(1, kn, 2) * feat).astype(np.float32),
        "corner_logics": np.random.rand(1, kn, 2).astype(np.float32),
        "lc": (np.random.rand(1, 2, feat, feat) * 5).astype(np.float32),
        "feat_size": (feat, feat),
    }
    res = postprocessor(decoded)
    assert len(res) == 1
    assert res[0]["polygons"].shape[1:] == ((4,) if assume_straight_pages else (4, 2))
    assert res[0]["logical"].shape[1] == 4
    if res[0]["polygons"].size:
        assert res[0]["polygons"].max() <= 1.0
        assert res[0]["polygons"].min() >= 0.0

    # not_relocate path follows the same geometry contract
    simple_res = TableCenterNetPostProcessor(
        center_thresh=0.0,
        not_relocate=True,
        assume_straight_pages=assume_straight_pages,
    )(decoded)
    assert len(simple_res) == 1
    assert simple_res[0]["polygons"].shape[1:] == ((4,) if assume_straight_pages else (4, 2))


@pytest.mark.parametrize("use_polygons", [False, True])
def test_tablecenternet_build_target(use_polygons):
    model = _TableCenterNet()
    out_h, out_w = 64, 64
    # Two images of different sizes + one empty image
    target = [
        _grid_target(2, 3, use_polygons),
        _grid_target(1, 2, use_polygons),
        {
            "cells": np.zeros((0, 4, 2) if use_polygons else (0, 4), np.float32),
            "logic": np.zeros((0, 4), np.int64),
        },
    ]

    dense = model.build_target(target, (out_h, out_w))

    # The dense schema matches what compute_loss consumes
    expected_keys = {
        "hm",
        "reg",
        "reg_ind",
        "reg_mask",
        "ct_ind",
        "ct_mask",
        "cn_ind",
        "cn_mask",
        "ct2cn",
        "cn2ct",
        "ct_cn_ind",
        "lc",
        "lc_mask",
        "lc_ind",
        "lc_span",
    }
    assert set(dense) == expected_keys
    # Every entry is batched over the images
    assert all(v.shape[0] == len(target) for v in dense.values())
    # Heat-maps and logical-coordinate maps cover the 2 (center, corner) channels at output resolution
    assert dense["hm"].shape == (len(target), 2, out_h, out_w)
    assert dense["lc"].shape == (len(target), 2, out_h, out_w)
    assert dense["lc_mask"].shape == (len(target), 2, out_h, out_w)
    # Vector-pair / span widths
    assert dense["ct2cn"].shape[-1] == 8
    assert dense["cn2ct"].shape[-1] == 8
    assert dense["lc_span"].shape[-1] == 2
    # The heat-map is a valid Gaussian field in [0, 1]
    assert dense["hm"].min() >= 0.0 and dense["hm"].max() <= 1.0
    # One positive cell-center per ground-truth cell, none for the empty image
    assert dense["ct_mask"][0].sum() == 6  # 2 x 3 grid
    assert dense["ct_mask"][1].sum() == 2  # 1 x 2 grid
    assert dense["ct_mask"][2].sum() == 0  # empty image
    # Index tensors stay int64 (used for gather), map tensors stay float32
    assert dense["ct_ind"].dtype == np.int64
    assert dense["hm"].dtype == np.float32

    # Cells outside the [0, 1] relative range are rejected
    bad = [
        {
            "cells": (
                np.array([[0.0, 0.0, 2.0, 2.0]], np.float32)
                if not use_polygons
                else np.array([[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]], np.float32)
            ),
            "logic": np.array([[0, 0, 0, 0]], np.int64),
        }
    ]
    with pytest.raises(ValueError):
        model.build_target(bad, (out_h, out_w))


def test_tablecenternet_build_target_box_polygon_equivalence():
    model = _TableCenterNet()
    box_dense = model.build_target([_grid_target(2, 3, use_polygons=False)], (64, 64))
    polygon_dense = model.build_target([_grid_target(2, 3, use_polygons=True)], (64, 64))

    assert box_dense.keys() == polygon_dense.keys()
    for key in box_dense:
        np.testing.assert_array_equal(box_dense[key], polygon_dense[key])


@pytest.mark.parametrize(
    "cells",
    [
        np.zeros((1, 5), dtype=np.float32),
        np.zeros((1, 3, 2), dtype=np.float32),
        np.zeros((4,), dtype=np.float32),
    ],
)
def test_tablecenternet_build_target_rejects_invalid_cell_shape(cells):
    with pytest.raises(ValueError, match="cells are expected to have shape"):
        _TableCenterNet().build_target(
            [{"cells": cells, "logic": np.zeros((len(cells), 4), dtype=np.int64)}],
            (64, 64),
        )
