from doctr.utils import geometry


def test_bbox_to_polygon():
    assert geometry.bbox_to_polygon((.1, .1, .2, .2, 0)) == [[.2, .2], [0, 0.2], [0, 0], [.2, 0]]


def test_polygon_to_bbox():
    pred = geometry.polygon_to_bbox([[.2, 0], [0, 0], [0, .2], [.2, .2]])[:4]
    target = (.1, .1, .2, .2)
    assert all(abs(i - j) <= 1e-7 for (i, j) in zip(pred, target))


def test_resolve_enclosing_bbox():
    pred = geometry.resolve_enclosing_bbox([(.2, .2, .05, .05, 0), (.2, .2, .2, .2, 0)])[:4]
    target = (.2, .2, .2, .2)
    assert all(abs(i - j) <= 1e-7 for (i, j) in zip(pred, target))
