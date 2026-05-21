import numpy as np
import pytest

from doctr.transforms import modules as T
from doctr.transforms.functional.base import expand_line
from doctr.utils import Sample


def test_imagetransform():
    transfo = T.ImageTransform(lambda sample: 1 - sample.image)
    assert transfo(Sample(image=0, target=1)) == Sample(image=1, target=1)


def test_samplecompose():
    transfos = [
        lambda sample: Sample(
            image=1 - sample.image,
            target=sample.target,
            mask=sample.mask,
        ),
        lambda sample: Sample(
            image=sample.image,
            target=2 * sample.target,
            mask=sample.mask,
        ),
    ]
    transfo = T.SampleCompose(transfos)
    assert transfo(Sample(image=0, target=1)) == Sample(image=1, target=2)


def test_oneof():
    transfos = [lambda x: 1 - x, lambda x: x + 10]
    transfo = T.OneOf(transfos)
    out = transfo(1)
    assert out == 0 or out == 11

    # test with ndarray target
    transfos = [
        lambda sample: Sample(
            image=1 - sample.image,
            target=sample.target,
            mask=sample.mask,
        ),
        lambda sample: Sample(
            image=sample.image + 10,
            target=sample.target,
            mask=sample.mask,
        ),
    ]

    transfo = T.OneOf(transfos)
    out = transfo(Sample(image=1, target=np.array([2])))
    assert out.image == 0 or out.image == 11
    assert isinstance(out.target, np.ndarray)
    np.testing.assert_array_equal(out.target, np.array([2]))

    # test with dict target
    dict_target = {
        "boxes": np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
    }
    transfos = [
        lambda sample: Sample(
            image=1 - sample.image,
            target=sample.target,
            mask=sample.mask,
        ),
        lambda sample: Sample(
            image=sample.image + 10,
            target=sample.target,
            mask=sample.mask,
        ),
    ]
    transfo = T.OneOf(transfos)
    out = transfo(Sample(image=1, target=dict_target))
    assert out.image == 0 or out.image == 11
    assert isinstance(out.target, dict)
    assert set(out.target.keys()) == {"boxes", "labels"}
    np.testing.assert_array_equal(
        out.target["boxes"],
        np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        out.target["labels"],
        np.array([1], dtype=np.int64),
    )


def test_randomapply():
    transfo = T.RandomApply(lambda x: 1 - x)
    out = transfo(1)
    assert out == 0 or out == 1
    # test with ndarray target
    transfo = T.RandomApply(
        lambda sample: Sample(
            image=1 - sample.image,
            target=2 * sample.target,
            mask=sample.mask,
        )
    )
    out = transfo(Sample(image=1, target=np.array([2])))
    assert out.image == 0 or out.image == 1
    assert isinstance(out.target, np.ndarray)
    if out.image == 0:
        np.testing.assert_array_equal(out.target, np.array([4]))
    else:
        np.testing.assert_array_equal(out.target, np.array([2]))

    # test with dict target
    dict_target = {
        "boxes": np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
    }
    transfo = T.RandomApply(
        lambda sample: Sample(
            image=1 - sample.image,
            target={
                "boxes": 2 * sample.target["boxes"],
                "labels": sample.target["labels"],
            },
            mask=sample.mask,
        )
    )
    out = transfo(Sample(image=1, target=dict_target))
    assert out.image == 0 or out.image == 1
    assert isinstance(out.target, dict)
    if out.image == 0:
        np.testing.assert_array_equal(
            out.target["boxes"],
            2 * np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        )
    else:
        np.testing.assert_array_equal(
            out.target["boxes"],
            np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        )
    np.testing.assert_array_equal(
        out.target["labels"],
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
