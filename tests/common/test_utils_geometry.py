from math import hypot

import cv2
import numpy as np
import pytest

from doctr.utils import geometry


def test_bbox_to_polygon():
    assert geometry.bbox_to_polygon(((0, 0), (1, 1))) == ((0, 0), (1, 0), (0, 1), (1, 1))


def test_polygon_to_bbox():
    assert geometry.polygon_to_bbox(((0, 0), (1, 0), (0, 1), (1, 1))) == ((0, 0), (1, 1))


def test_resolve_enclosing_bbox():
    assert geometry.resolve_enclosing_bbox([((0, 0.5), (1, 0)), ((0.5, 0), (1, 0.25))]) == ((0, 0), (1, 0.5))
    pred = geometry.resolve_enclosing_bbox(np.array([[0.1, 0.1, 0.2, 0.2, 0.9], [0.15, 0.15, 0.2, 0.2, 0.8]]))
    assert pred.all() == np.array([0.1, 0.1, 0.2, 0.2, 0.85]).all()


def test_resolve_enclosing_rbbox():
    pred = geometry.resolve_enclosing_rbbox([
        np.asarray([[.1, .1], [.2, .2], [.15, .25], [.05, .15]]),
        np.asarray([[.5, .5], [.6, .6], [.55, .65], [.45, .55]])
    ])
    target1 = np.asarray([[.55, .65], [.05, .15], [.1, .1], [.6, .6]])
    target2 = np.asarray([[.05, .15], [.1, .1], [.6, .6], [.55, .65]])
    assert np.all(target1 - pred <= 1e-3) or np.all(target2 - pred <= 1e-3)


def test_rotate_boxes():
    boxes = np.array([[0.1, 0.1, 0.8, 0.3, 0.5]])
    rboxes = np.array([[0.1, 0.1], [0.8, 0.1], [0.8, 0.3], [0.1, 0.3]])
    # Angle = 0
    rotated = geometry.rotate_boxes(boxes, angle=0., orig_shape=(1, 1))
    assert np.all(rotated == rboxes)
    # Angle < 1:
    rotated = geometry.rotate_boxes(boxes, angle=0.5, orig_shape=(1, 1))
    assert np.all(rotated == rboxes)
    # Angle = 30
    rotated = geometry.rotate_boxes(boxes, angle=30, orig_shape=(1, 1))
    assert rotated.shape == (1, 4, 2)

    boxes = np.array([[0., 0., 0.6, 0.2, 0.5]])
    # Angle = -90:
    rotated = geometry.rotate_boxes(boxes, angle=-90, orig_shape=(1, 1), min_angle=0)
    assert np.allclose(rotated, np.array([[[1, 0.], [1, 0.6], [0.8, 0.6], [0.8, 0.]]]))
    # Angle = 90
    rotated = geometry.rotate_boxes(boxes, angle=+90, orig_shape=(1, 1), min_angle=0)
    assert np.allclose(rotated, np.array([[[0, 1.], [0, 0.4], [0.2, 0.4], [0.2, 1.]]]))


def test_rotate_image():
    img = np.ones((32, 64, 3), dtype=np.float32)
    rotated = geometry.rotate_image(img, 30.)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, 0, 0] == 0
    assert rotated[0, :, 0].sum() > 1

    # Expand
    rotated = geometry.rotate_image(img, 30., expand=True)
    assert rotated.shape[:-1] == (60, 120)
    assert rotated[0, :, 0].sum() <= 1

    # Expand
    rotated = geometry.rotate_image(img, 30., expand=True, preserve_origin_shape=True)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, :, 0].sum() <= 1

    # Expand with 90° rotation
    rotated = geometry.rotate_image(img, 90., expand=True)
    assert rotated.shape[:-1] == (64, 128)
    assert rotated[0, :, 0].sum() <= 1
