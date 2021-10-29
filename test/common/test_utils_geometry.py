import cv2
import numpy as np
from math import hypot

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


def test_bbox_to_rbbox():
    pred = geometry.bbox_to_rbbox((0, 0, 0.6, 0.4))
    target = (0.3, 0.2, 0.6, 0.4, 0)
    assert all(abs(i - j) <= 1e-7 for (i, j) in zip(pred, target))


def test_resolve_enclosing_rbbox():
    pred = geometry.resolve_enclosing_rbbox([(.2, .2, .05, .05, 0), (.2, .2, .2, .2, 0)])[:4]
    target = (.2, .2, .2, .2)
    assert all(abs(i - j) <= 1e-7 for (i, j) in zip(pred, target))


def test_remap_boxes():
    pred = geometry.remap_boxes(np.array([[0.5, 0.5, 0.1, 0.1, 0.]]), (10, 10), (20, 20))
    target = np.array([[0.5, 0.5, 0.05, 0.05, 0.]])
    assert pred.all() == target.all()

    pred = geometry.remap_boxes(np.array([[0.5, 0.5, 0.1, 0.1, 0.]]), (10, 10), (20, 10))
    target = np.array([[0.5, 0.5, 0.1, 0.05, 0.]])
    assert pred.all() == target.all()

    pred = geometry.remap_boxes(np.array([[0.25, 0.5, 0.5, 0.33, 0.]]), (80, 30), (160, 30))
    target = np.array([[0.375, 0.5, 0.25, 0.1, 0.]])
    assert pred.all() == target.all()

    orig_dimension = (100, 100)
    dest_dimensions = (100, 200)
    orig_box = np.array([[0.5, 0.5, 0.2, 0., 45]])
    # Unpack
    height_o, width_o = orig_dimension
    height_d, width_d = dest_dimensions
    pred = geometry.remap_boxes(orig_box, orig_dimension, dest_dimensions)

    x, y, w, h, a = orig_box[0]
    # Switch to absolute coords
    x, w = x * width_o, w * width_o
    y, h = y * height_o, h * height_o
    orig = cv2.boxPoints(((x, y), (w, h), a))

    x, y, w, h, a = pred[0]
    # Switch to absolute coords
    x, w = x * width_d, w * width_d
    y, h = y * height_d, h * height_d
    dest = cv2.boxPoints(((x, y), (w, h), a))

    len_orig = hypot(orig[0][0] - orig[2][0], orig[0][1] - orig[2][1])
    len_dest = hypot(dest[0][0] - dest[2][0], dest[0][1] - dest[2][1])
    assert len_orig == len_dest

    alpha_orig = np.rad2deg(np.arctan((orig[0][1] - orig[2][1])/ (orig[0][0] - orig[2][0])))
    alpha_dest = np.rad2deg(np.arctan((dest[0][1] - dest[2][1])/ (dest[0][0] - dest[2][0])))
    assert alpha_orig == alpha_dest


def test_rotate_boxes():
    boxes = np.array([[0.1, 0.1, 0.8, 0.3]])
    rboxes = np.apply_along_axis(geometry.bbox_to_rbbox, 1, boxes)
    # Angle = 0
    rotated = geometry.rotate_boxes(boxes, angle=0.)
    assert rotated.all() == rboxes.all()
    # Angle < 1:
    rotated = geometry.rotate_boxes(boxes, angle=0.5)
    assert rotated.all() == rboxes.all()
    # Angle = 30
    rotated = geometry.rotate_boxes(boxes, angle=30)
    assert rotated.shape == (1, 5)
    assert rotated[0, 4] == 30.

    boxes = np.array([[0., 0., 0.6, 0.2]])
    # Angle = -90:
    rotated = geometry.rotate_boxes(boxes, angle=-90, min_angle=0)
    assert rotated.all() == np.array([[0.1, 0.7, 0.6, 0.2, -90.]]).all()
    # Angle = 90
    rotated = geometry.rotate_boxes(boxes, angle=+90, min_angle=0)
    assert rotated.all() == np.array([[0.9, 0.3, 0.6, 0.2, 90.]]).all()


def test_rotate_image():
    img = np.ones((32, 64, 3), dtype=np.float32)
    rotated = geometry.rotate_image(img, 30.)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, 0, 0] == 0
    assert rotated[0, :, 0].sum() > 1

    # Expand
    rotated = geometry.rotate_image(img, 30., expand=True)
    assert rotated.shape[:-1] == (60, 72)
    assert rotated[0, :, 0].sum() <= 1

    # Expand
    rotated = geometry.rotate_image(img, 30., expand=True, preserve_aspect_ratio=True)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, :, 0].sum() <= 1

    # Expand with 90° rotation
    rotated = geometry.rotate_image(img, 90., expand=True)
    assert rotated.shape[:-1] == (64, 64)
    assert rotated[0, :, 0].sum() <= 1

