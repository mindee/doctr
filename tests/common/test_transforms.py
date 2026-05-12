import numpy as np
import pytest

from doctr.transforms import modules as T
from doctr.transforms.functional.base import expand_line


def test_imagetransform():
    transfo = T.ImageTransform(lambda x: 1 - x)
    assert transfo(0, 1) == (1, 1)


def test_samplecompose():
    transfos = [lambda x, y: (1 - x, y), lambda x, y: (x, 2 * y)]
    transfo = T.SampleCompose(transfos)
    assert transfo(0, 1) == (1, 2)


def test_oneof():
    transfos = [lambda x: 1 - x, lambda x: x + 10]
    transfo = T.OneOf(transfos)
    out = transfo(1)
    assert out == 0 or out == 11

    # test with ndarray target
    transfos = [lambda x, y: (1 - x, y), lambda x, y: (x + 10, y)]
    transfo = T.OneOf(transfos)
    out = transfo(1, np.array([2]))
    assert out == (0, 2) or out == (11, 2)
    assert isinstance(out[1], np.ndarray)

    # test with dict target
    dict_target = {
        "boxes": np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
    }
    transfos = [
        lambda x, y: (1 - x, y),
        lambda x, y: (x + 10, y),
    ]
    transfo = T.OneOf(transfos)
    out = transfo(1, dict_target)
    assert out[0] == 0 or out[0] == 11
    assert isinstance(out[1], dict)
    assert set(out[1].keys()) == {"boxes", "labels"}
    assert isinstance(out[1]["boxes"], np.ndarray)
    assert isinstance(out[1]["labels"], np.ndarray)
    np.testing.assert_array_equal(
        out[1]["boxes"],
        np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        out[1]["labels"],
        np.array([1], dtype=np.int64),
    )


def test_randomapply():
    transfo = T.RandomApply(lambda x: 1 - x)
    out = transfo(1)
    assert out == 0 or out == 1

    # test with ndarray target
    transfo = T.RandomApply(lambda x, y: (1 - x, 2 * y))
    out = transfo(1, np.array([2]))
    assert out == (0, 4) or out == (1, 2)
    assert isinstance(out[1], np.ndarray)

    # test with dict target
    dict_target = {
        "boxes": np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
    }
    transfo = T.RandomApply(
        lambda x, y: (
            1 - x,
            {
                "boxes": 2 * y["boxes"],
                "labels": y["labels"],
            },
        )
    )

    out = transfo(1, dict_target)
    assert out[0] == 0 or out[0] == 1
    assert isinstance(out[1], dict)
    assert set(out[1].keys()) == {"boxes", "labels"}
    assert isinstance(out[1]["boxes"], np.ndarray)
    assert isinstance(out[1]["labels"], np.ndarray)
    if out[0] == 0:
        np.testing.assert_array_equal(
            out[1]["boxes"],
            2 * np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        )
    else:
        np.testing.assert_array_equal(
            out[1]["boxes"],
            np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        )
    np.testing.assert_array_equal(
        out[1]["labels"],
        np.array([1], dtype=np.int64),
    )
    assert repr(transfo).endswith(", p=0.5)")


@pytest.mark.parametrize(
    "line",
    [
        # Horizontal
        np.array([[63, 1], [42, 1]]).astype(np.int32),
        # Vertical
        np.array([[1, 63], [1, 42]]).astype(np.int32),
        # Normal
        np.array([[1, 63], [12, 42]]).astype(np.int32),
    ],
)
def test_expand_line(line):
    out = expand_line(line, (100, 100))
    assert isinstance(out, tuple)
    assert all(isinstance(val, (float, int, np.int32, np.float64)) and val >= 0 for val in out)
