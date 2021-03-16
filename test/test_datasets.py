import pytest
import os
import numpy as np
from doctr import datasets


@pytest.mark.parametrize(
    "dataset_name, size",
    [
        ['FUNSD', 149],
    ],
)
def test_dataset(dataset_name, size):

    with pytest.raises(ValueError):
        datasets.__dict__[dataset_name](download=False)

    ds = datasets.__dict__[dataset_name](download=True)

    assert len(ds) == size
    assert repr(ds) == f"{dataset_name}()"
    img, target = ds[0]
    assert isinstance(img, np.ndarray) and img.ndim == 3
    assert isinstance(target, dict)
