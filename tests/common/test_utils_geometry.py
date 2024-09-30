from copy import deepcopy
from math import hypot

import numpy as np
import pytest

from doctr.io import DocumentFile
from doctr.utils import geometry


def test_bbox_to_polygon():
    assert geometry.bbox_to_polygon(((0, 0), (1, 1))) == ((0, 0), (1, 0), (0, 1), (1, 1))


def test_polygon_to_bbox():
    assert geometry.polygon_to_bbox(((0, 0), (1, 0), (0, 1), (1, 1))) == ((0, 0), (1, 1))


def test_detach_scores():
    # box test
    boxes = np.array([[0.1, 0.1, 0.2, 0.2, 0.9], [0.15, 0.15, 0.2, 0.2, 0.8]])
    pred = geometry.detach_scores([boxes])
    target1 = np.array([[0.1, 0.1, 0.2, 0.2], [0.15, 0.15, 0.2, 0.2]])
    target2 = np.array([0.9, 0.8])
    assert np.all(pred[0] == target1) and np.all(pred[1] == target2)
    # polygon test
    boxes = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15], [0.0, 0.9]],
        [[0.15, 0.15], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15], [0.0, 0.8]],
    ])
    pred = geometry.detach_scores([boxes])
    target1 = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
        [[0.15, 0.15], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
    ])
    target2 = np.array([0.9, 0.8])
    assert np.all(pred[0] == target1) and np.all(pred[1] == target2)


def test_resolve_enclosing_bbox():
    assert geometry.resolve_enclosing_bbox([((0, 0.5), (1, 0)), ((0.5, 0), (1, 0.25))]) == ((0, 0), (1, 0.5))
    pred = geometry.resolve_enclosing_bbox(np.array([[0.1, 0.1, 0.2, 0.2], [0.15, 0.15, 0.2, 0.2]]))
    assert pred.all() == np.array([0.1, 0.1, 0.2, 0.2]).all()


def test_resolve_enclosing_rbbox():
    pred = geometry.resolve_enclosing_rbbox([
        np.asarray([[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]]),
        np.asarray([[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]]),
    ])
    target1 = np.asarray([[0.55, 0.65], [0.05, 0.15], [0.1, 0.1], [0.6, 0.6]])
    target2 = np.asarray([[0.05, 0.15], [0.1, 0.1], [0.6, 0.6], [0.55, 0.65]])
    assert np.all(target1 - pred <= 1e-3) or np.all(target2 - pred <= 1e-3)


def test_remap_boxes():
    pred = geometry.remap_boxes(
        np.asarray([[[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]]), (10, 10), (20, 20)
    )
    target = np.asarray([[[0.375, 0.375], [0.375, 0.625], [0.625, 0.375], [0.625, 0.625]]])
    assert np.all(pred == target)

    pred = geometry.remap_boxes(
        np.asarray([[[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]]), (10, 10), (20, 10)
    )
    target = np.asarray([[[0.25, 0.375], [0.25, 0.625], [0.75, 0.375], [0.75, 0.625]]])
    assert np.all(pred == target)

    with pytest.raises(ValueError):
        geometry.remap_boxes(
            np.asarray([[[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]]), (80, 40, 150), (160, 40)
        )

    with pytest.raises(ValueError):
        geometry.remap_boxes(np.asarray([[[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]]), (80, 40), (160,))

    orig_dimension = (100, 100)
    dest_dimensions = (200, 100)
    # Unpack dimensions
    height_o, width_o = orig_dimension
    height_d, width_d = dest_dimensions

    orig_box = np.asarray([[[0.25, 0.25], [0.25, 0.25], [0.75, 0.75], [0.75, 0.75]]])

    pred = geometry.remap_boxes(orig_box, orig_dimension, dest_dimensions)

    # Switch to absolute coords
    orig = np.stack((orig_box[:, :, 0] * width_o, orig_box[:, :, 1] * height_o), axis=2)[0]
    dest = np.stack((pred[:, :, 0] * width_d, pred[:, :, 1] * height_d), axis=2)[0]

    len_orig = hypot(orig[0][0] - orig[2][0], orig[0][1] - orig[2][1])
    len_dest = hypot(dest[0][0] - dest[2][0], dest[0][1] - dest[2][1])
    assert len_orig == len_dest

    alpha_orig = np.rad2deg(np.arctan((orig[0][1] - orig[2][1]) / (orig[0][0] - orig[2][0])))
    alpha_dest = np.rad2deg(np.arctan((dest[0][1] - dest[2][1]) / (dest[0][0] - dest[2][0])))
    assert alpha_orig == alpha_dest


def test_rotate_boxes():
    boxes = np.array([[0.1, 0.1, 0.8, 0.3, 0.5]])
    rboxes = np.array([[0.1, 0.1], [0.8, 0.1], [0.8, 0.3], [0.1, 0.3]])
    # Angle = 0
    rotated = geometry.rotate_boxes(boxes, angle=0.0, orig_shape=(1, 1))
    assert np.all(rotated == rboxes)
    # Angle < 1:
    rotated = geometry.rotate_boxes(boxes, angle=0.5, orig_shape=(1, 1))
    assert np.all(rotated == rboxes)
    # Angle = 30
    rotated = geometry.rotate_boxes(boxes, angle=30, orig_shape=(1, 1))
    assert rotated.shape == (1, 4, 2)

    boxes = np.array([[0.0, 0.0, 0.6, 0.2, 0.5]])
    # Angle = -90:
    rotated = geometry.rotate_boxes(boxes, angle=-90, orig_shape=(1, 1), min_angle=0)
    assert np.allclose(rotated, np.array([[[1, 0.0], [1, 0.6], [0.8, 0.6], [0.8, 0.0]]]))
    # Angle = 90
    rotated = geometry.rotate_boxes(boxes, angle=+90, orig_shape=(1, 1), min_angle=0)
    assert np.allclose(rotated, np.array([[[0, 1.0], [0, 0.4], [0.2, 0.4], [0.2, 1.0]]]))


def test_rotate_image():
    img = np.ones((32, 64, 3), dtype=np.float32)
    rotated = geometry.rotate_image(img, 30.0)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, 0, 0] == 0
    assert rotated[0, :, 0].sum() > 1

    # Expand
    rotated = geometry.rotate_image(img, 30.0, expand=True)
    assert rotated.shape[:-1] == (60, 120)
    assert rotated[0, :, 0].sum() <= 1

    # Expand
    rotated = geometry.rotate_image(img, 30.0, expand=True, preserve_origin_shape=True)
    assert rotated.shape[:-1] == (32, 64)
    assert rotated[0, :, 0].sum() <= 1

    # Expand with 90° rotation
    rotated = geometry.rotate_image(img, 90.0, expand=True)
    assert rotated.shape[:-1] == (64, 128)
    assert rotated[0, :, 0].sum() <= 1


def test_remove_image_padding():
    img = np.ones((32, 64, 3), dtype=np.float32)
    padded = np.pad(img, ((10, 10), (20, 20), (0, 0)))
    cropped = geometry.remove_image_padding(padded)
    assert np.all(cropped == img)

    # No padding
    cropped = geometry.remove_image_padding(img)
    assert np.all(cropped == img)


@pytest.mark.parametrize(
    "abs_geoms, img_size, rel_geoms",
    [
        # Full image (boxes)
        [np.array([[0, 0, 32, 32]]), (32, 32), np.array([[0, 0, 1, 1]], dtype=np.float32)],
        # Full image (polygons)
        [
            np.array([[[0, 0], [32, 0], [32, 32], [0, 32]]]),
            (32, 32),
            np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32),
        ],
        # Quarter image (boxes)
        [np.array([[0, 0, 16, 16]]), (32, 32), np.array([[0, 0, 0.5, 0.5]], dtype=np.float32)],
        # Quarter image (polygons)
        [
            np.array([[[0, 0], [16, 0], [16, 16], [0, 16]]]),
            (32, 32),
            np.array([[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]], dtype=np.float32),
        ],
    ],
)
def test_convert_to_relative_coords(abs_geoms, img_size, rel_geoms):
    assert np.all(geometry.convert_to_relative_coords(abs_geoms, img_size) == rel_geoms)

    # Wrong format
    with pytest.raises(ValueError):
        geometry.convert_to_relative_coords(np.zeros((3, 5)), (32, 32))


def test_estimate_page_angle():
    straight_polys = np.array([
        [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4], [0.3, 0.4]],
        [[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]],
        [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6]],
    ])
    rotated_polys = geometry.rotate_boxes(straight_polys, angle=20, orig_shape=(512, 512))
    angle = geometry.estimate_page_angle(rotated_polys)
    assert np.isclose(angle, 20)
    # Test divide by zero / NaN
    invalid_poly = np.array([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]])
    angle = geometry.estimate_page_angle(invalid_poly)
    assert angle == 0.0


def test_extract_crops(mock_pdf):
    doc_img = DocumentFile.from_pdf(mock_pdf)[0]
    num_crops = 2
    rel_boxes = np.array(
        [[idx / num_crops, idx / num_crops, (idx + 1) / num_crops, (idx + 1) / num_crops] for idx in range(num_crops)],
        dtype=np.float32,
    )
    abs_boxes = np.array(
        [
            [
                int(idx * doc_img.shape[1] / num_crops),
                int(idx * doc_img.shape[0]) / num_crops,
                int((idx + 1) * doc_img.shape[1] / num_crops),
                int((idx + 1) * doc_img.shape[0] / num_crops),
            ]
            for idx in range(num_crops)
        ],
        dtype=np.float32,
    )

    with pytest.raises(AssertionError):
        geometry.extract_crops(doc_img, np.zeros((1, 5)))

    for boxes in (rel_boxes, abs_boxes):
        croped_imgs = geometry.extract_crops(doc_img, boxes)
        # Number of crops
        assert len(croped_imgs) == num_crops
        # Data type and shape
        assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
        assert all(crop.ndim == 3 for crop in croped_imgs)

    # Identity
    assert np.all(
        doc_img == geometry.extract_crops(doc_img, np.array([[0, 0, 1, 1]], dtype=np.float32), channels_last=True)[0]
    )
    torch_img = np.transpose(doc_img, axes=(-1, 0, 1))
    assert np.all(
        torch_img
        == np.transpose(
            geometry.extract_crops(doc_img, np.array([[0, 0, 1, 1]], dtype=np.float32), channels_last=False)[0],
            axes=(-1, 0, 1),
        )
    )

    # No box
    assert geometry.extract_crops(doc_img, np.zeros((0, 4))) == []


@pytest.mark.parametrize("assume_horizontal", [True, False])
def test_extract_rcrops(mock_pdf, assume_horizontal):
    doc_img = DocumentFile.from_pdf(mock_pdf)[0]
    num_crops = 2
    rel_boxes = np.array(
        [
            [
                [idx / num_crops, idx / num_crops],
                [idx / num_crops + 0.1, idx / num_crops],
                [idx / num_crops + 0.1, idx / num_crops + 0.1],
                [idx / num_crops, idx / num_crops],
            ]
            for idx in range(num_crops)
        ],
        dtype=np.float32,
    )
    abs_boxes = deepcopy(rel_boxes)
    abs_boxes[:, :, 0] *= doc_img.shape[1]
    abs_boxes[:, :, 1] *= doc_img.shape[0]
    abs_boxes = abs_boxes.astype(np.int64)

    with pytest.raises(AssertionError):
        geometry.extract_rcrops(doc_img, np.zeros((1, 8)), assume_horizontal=assume_horizontal)
    for boxes in (rel_boxes, abs_boxes):
        croped_imgs = geometry.extract_rcrops(doc_img, boxes, assume_horizontal=assume_horizontal)
        # Number of crops
        assert len(croped_imgs) == num_crops
        # Data type and shape
        assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
        assert all(crop.ndim == 3 for crop in croped_imgs)

    # No box
    assert geometry.extract_rcrops(doc_img, np.zeros((0, 4, 2)), assume_horizontal=assume_horizontal) == []
