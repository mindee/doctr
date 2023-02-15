import os

import pytest
from torch import nn

from doctr.models.utils import conv_sequence_pt, load_pretrained_params


def test_load_pretrained_params(tmpdir_factory):
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    # Retrieve this URL
    url = "https://github.com/mindee/doctr/releases/download/v0.2.1/tmp_checkpoint-6f0ce0e6.pt"
    # Temp cache dir
    cache_dir = tmpdir_factory.mktemp("cache")
    # Pass an incorrect hash
    with pytest.raises(ValueError):
        load_pretrained_params(model, url, "mywronghash", cache_dir=str(cache_dir))
    # Let it resolve the hash from the file name
    load_pretrained_params(model, url, cache_dir=str(cache_dir))
    # Check that the file was downloaded & the archive extracted
    assert os.path.exists(cache_dir.join("models").join(url.rpartition("/")[-1].split("&")[0]))
    # Check ignore keys
    load_pretrained_params(model, url, cache_dir=str(cache_dir), ignore_keys=["2.weight"])
    # non matching keys
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    with pytest.raises(ValueError):
        load_pretrained_params(model, url, cache_dir=str(cache_dir), ignore_keys=["2.weight"])


def test_conv_sequence():
    assert len(conv_sequence_pt(3, 8, kernel_size=3)) == 1
    assert len(conv_sequence_pt(3, 8, True, kernel_size=3)) == 2
    assert len(conv_sequence_pt(3, 8, False, True, kernel_size=3)) == 2
    assert len(conv_sequence_pt(3, 8, True, True, kernel_size=3)) == 3
