import os

import pytest
import torch
from torch import nn

from doctr.models.utils import (
    _bf16_to_float32,
    _copy_tensor,
    conv_sequence_pt,
    load_pretrained_params,
    set_device_and_dtype,
)


def test_copy_tensor():
    x = torch.rand(8)
    m = _copy_tensor(x)
    assert m.device == x.device and m.dtype == x.dtype and m.shape == x.shape and torch.allclose(m, x)


def test_bf16_to_float32():
    x = torch.randn([2, 2], dtype=torch.bfloat16)
    converted_x = _bf16_to_float32(x)
    assert x.dtype == torch.bfloat16 and converted_x.dtype == torch.float32 and torch.equal(converted_x, x.float())


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


def test_set_device_and_dtype():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    batches = [torch.rand(8) for _ in range(2)]
    model, batches = set_device_and_dtype(model, batches, device="cpu", dtype=torch.float32)
    assert model[0].weight.device == torch.device("cpu")
    assert model[0].weight.dtype == torch.float32
    assert batches[0].device == torch.device("cpu")
    assert batches[0].dtype == torch.float32
    model, batches = set_device_and_dtype(model, batches, device="cpu", dtype=torch.float16)
    assert model[0].weight.dtype == torch.float16
    assert batches[0].dtype == torch.float16
