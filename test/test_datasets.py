import pytest
import os
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
    img_name, target = ds[0]
    assert isinstance(img_name, str) and os.path.exists(os.path.join(ds.root, img_name))
    assert isinstance(target, dict)
