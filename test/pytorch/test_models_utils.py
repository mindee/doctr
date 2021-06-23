import pytest
import os

from torch import nn
from doctr.models import utils


def test_load_pretrained_params(tmpdir_factory):

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    # Retrieve this URL
    url = "https://github.com/mindee/doctr/releases/download/v0.2.1/tmp_checkpoint-6f0ce0e6.pt"
    # Temp cache dir
    cache_dir = tmpdir_factory.mktemp("cache")
    # Pass an incorrect hash
    with pytest.raises(ValueError):
        utils.load_pretrained_params(model, url, "mywronghash", cache_dir=str(cache_dir))
    # Let tit resolve the hash from the file name
    utils.load_pretrained_params(model, url, cache_dir=str(cache_dir))
    # Check that the file was downloaded & the archive extracted
    assert os.path.exists(cache_dir.join('models').join(url.rpartition("/")[-1]))


def test_conv_sequence():

    assert len(utils.conv_sequence(3, 8, kernel_size=3)) == 1
    assert len(utils.conv_sequence(3, 8, True, kernel_size=3)) == 2
    assert len(utils.conv_sequence(3, 8, False, True, kernel_size=3)) == 2
    assert len(utils.conv_sequence(3, 8, True, True, kernel_size=3)) == 3
