import numpy as np

from doctr.models.table_structure.tablecenternet import TableCenterNetPostProcessor


def test_tablecenternet_postprocessor():
    postprocessor = TableCenterNetPostProcessor(center_thresh=0.0)
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
    assert len(res) == 1 and res[0]["polygons"].shape[1:] == (4, 2)
    assert res[0]["logical"].shape[1] == 4
    if res[0]["polygons"].size:
        assert res[0]["polygons"].max() <= 1.0  # relative coordinates
    # not_relocate path
    assert len(TableCenterNetPostProcessor(center_thresh=0.0, not_relocate=True)(decoded)) == 1
