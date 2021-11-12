import numpy as np

from doctr.utils import geometry


def test_bbox_to_polygon():
    assert geometry.bbox_to_polygon(((0, 0), (1, 1))) == ((0, 0), (1, 0), (0, 1), (1, 1))


def test_polygon_to_bbox():
    assert geometry.polygon_to_bbox(((0, 0), (1, 0), (0, 1), (1, 1))) == ((0, 0), (1, 1))


def test_resolve_enclosing_bbox():
    assert geometry.resolve_enclosing_bbox([((0, 0.5), (1, 0)), ((0.5, 0), (1, 0.25))]) == ((0, 0), (1, 0.5))
    pred = geometry.resolve_enclosing_bbox(np.array([[0.1, 0.1, 0.2, 0.2, 0.9], [0.15, 0.15, 0.2, 0.2, 0.8]]))
    assert pred.all() == np.array([0.1, 0.1, 0.2, 0.2, 0.85]).all()


def test_rbbox_to_polygon():
    assert (
        geometry.rbbox_to_polygon((.1, .1, .2, .2, 0)) == np.array([[0, .2], [0, 0], [.2, 0], [.2, .2]], np.float32)
    ).all()


def test_polygon_to_rbbox():
    pred = geometry.polygon_to_rbbox([[.2, 0], [0, 0], [0, .2], [.2, .2]])[:4]
    target = (.1, .1, .2, .2)
    assert all(abs(i - j) <= 1e-7 for (i, j) in zip(pred, target))


def test_resolve_enclosing_rbbox():
    pred = geometry.resolve_enclosing_rbbox([(.2, .2, .05, .05, 0), (.2, .2, .2, .2, 0)])[:4]
    target = (.2, .2, .2, .2)
    assert all(abs(i - j) <= 1e-7 for (i, j) in zip(pred, target))


def test_rotate_boxes():
    boxes = np.array([[0.1, 0.1, 0.8, 0.3]])
    # Angle = 0
    rotated = geometry.rotate_boxes(boxes, angle=0.)
    assert rotated.all() == boxes.all()
    # Angle < 1:
    rotated = geometry.rotate_boxes(boxes, angle=0.5)
    assert rotated.all() == boxes.all()
    # Angle = 30
    rotated = geometry.rotate_boxes(boxes, angle=30)
    assert rotated.shape == (1, 5)
    assert rotated[0, 4] == 30.


def test_rotate_image():
    img = np.ones((32, 64, 3), dtype=np.float32)
    rotated = geometry.rotate_image(img, 30.)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, 0, 0] == 0
    assert rotated[0, :, 0].sum() > 1

    # Expand
    rotated = geometry.rotate_image(img, 30., expand=True)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, :, 0].sum() <= 1
