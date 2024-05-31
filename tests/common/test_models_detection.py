import numpy as np
import pytest

from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
from doctr.models.detection.fast.base import FASTPostProcessor
from doctr.models.detection.linknet.base import LinkNetPostProcessor


def test_dbpostprocessor():
    postprocessor = DBPostProcessor(assume_straight_pages=True)
    r_postprocessor = DBPostProcessor(assume_straight_pages=False)
    with pytest.raises(AssertionError):
        postprocessor(np.random.rand(2, 512, 512).astype(np.float32))
    mock_batch = np.random.rand(2, 512, 512, 1).astype(np.float32)
    out = postprocessor(mock_batch)
    r_out = r_postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, list) and all(isinstance(v, np.ndarray) for v in sample) for sample in out)
    assert all(all(v.shape[1] == 5 for v in sample) for sample in out)
    assert all(all(v.shape[1] == 5 and v.shape[2] == 2 for v in sample) for sample in r_out)
    # Relative coords
    assert all(all(np.all(np.logical_and(v[:, :4] >= 0, v[:, :4] <= 1)) for v in sample) for sample in out)
    assert all(all(np.all(np.logical_and(v[:, :4] >= 0, v[:, :4] <= 1)) for v in sample) for sample in r_out)
    # Repr
    assert repr(postprocessor) == "DBPostProcessor(bin_thresh=0.3, box_thresh=0.1)"
    # Edge case when the expanded points of the polygon has two lists
    issue_points = np.array(
        [
            [869, 561],
            [923, 581],
            [925, 595],
            [915, 583],
            [889, 583],
            [905, 593],
            [882, 601],
            [901, 595],
            [904, 604],
            [876, 608],
            [915, 614],
            [911, 605],
            [925, 601],
            [930, 616],
            [911, 617],
            [900, 636],
            [931, 637],
            [904, 649],
            [932, 649],
            [932, 628],
            [918, 627],
            [934, 624],
            [935, 573],
            [909, 569],
            [934, 562],
        ],
        dtype=np.int32,
    )
    out = postprocessor.polygon_to_box(issue_points)
    r_out = r_postprocessor.polygon_to_box(issue_points)
    assert isinstance(out, tuple) and len(out) == 4
    assert isinstance(r_out, np.ndarray) and r_out.shape == (4, 2)


def test_linknet_postprocessor():
    postprocessor = LinkNetPostProcessor()
    r_postprocessor = LinkNetPostProcessor(assume_straight_pages=False)
    with pytest.raises(AssertionError):
        postprocessor(np.random.rand(2, 512, 512).astype(np.float32))
    mock_batch = np.random.rand(2, 512, 512, 1).astype(np.float32)
    out = postprocessor(mock_batch)
    r_out = r_postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, list) and all(isinstance(v, np.ndarray) for v in sample) for sample in out)
    assert all(all(v.shape[1] == 5 for v in sample) for sample in out)
    assert all(all(v.shape[1] == 5 and v.shape[2] == 2 for v in sample) for sample in r_out)
    # Relative coords
    assert all(all(np.all(np.logical_and(v[:4] >= 0, v[:4] <= 1)) for v in sample) for sample in out)


def test_fast_postprocessor():
    postprocessor = FASTPostProcessor()
    r_postprocessor = FASTPostProcessor(assume_straight_pages=False)
    with pytest.raises(AssertionError):
        postprocessor(np.random.rand(2, 512, 512).astype(np.float32))
    mock_batch = np.random.rand(2, 512, 512, 1).astype(np.float32)
    out = postprocessor(mock_batch)
    r_out = r_postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, list) and all(isinstance(v, np.ndarray) for v in sample) for sample in out)
    assert all(all(v.shape[1] == 5 for v in sample) for sample in out)
    assert all(all(v.shape[1] == 5 and v.shape[2] == 2 for v in sample) for sample in r_out)
    # Relative coords
    assert all(all(np.all(np.logical_and(v[:4] >= 0, v[:4] <= 1)) for v in sample) for sample in out)
