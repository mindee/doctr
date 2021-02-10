import pytest
import os
from tensorflow.keras import layers, Sequential

from doctr.models import utils


def test_load_pretrained_params(tmpdir_factory):

    model = Sequential([layers.Dense(8, activation='relu', input_shape=(4,)), layers.Dense(4)])
    # Retrieve this URL
    url = "https://github.com/mindee/doctr/releases/download/v0.1-models/tmp_checkpoint-4a98e492.zip"
    # Temp cache dir
    cache_dir = tmpdir_factory.mktemp("cache")
    # Pass an incorrect hash
    with pytest.raises(ValueError):
        utils.load_pretrained_params(model, url, "mywronghash", cache_dir=str(cache_dir), internal_name='')
    # Let tit resolve the hash from the file name
    utils.load_pretrained_params(model, url, cache_dir=str(cache_dir), internal_name='')
    # Check that the file was downloaded & the archive extracted
    assert os.path.exists(cache_dir.join('models').join("tmp_checkpoint-4a98e492"))
    # Check that archive was deleted
    assert os.path.exists(cache_dir.join('models').join("tmp_checkpoint-4a98e492.zip"))
