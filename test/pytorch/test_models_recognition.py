import pytest
import numpy as np
import torch

from doctr.models import recognition


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (3, 32, 128)],
    ],
)
def test_recognition_models(arch_name, input_shape):
    batch_size = 4
    reco_model = recognition.__dict__[arch_name](pretrained=False, input_shape=input_shape).eval()
    assert isinstance(reco_model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = ["i", "am", "a", "jedi"]

    out = reco_model(input_tensor, target, return_model_output=True, return_preds=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    assert isinstance(out['preds'], list)
    assert len(out['preds']) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out['preds'])
    assert isinstance(out['out_map'], torch.Tensor)
    assert isinstance(out['loss'], torch.Tensor)


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        ["CTCPostProcessor", [2, 119, 30]],
    ],
)
def test_reco_postprocessors(post_processor, input_shape, mock_vocab):
    processor = recognition.__dict__[post_processor](mock_vocab)
    decoded = processor(torch.rand(*input_shape))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == input_shape[0]
    assert all(char in mock_vocab for word, _ in decoded for char in word)
    # Repr
    assert repr(processor) == f'{post_processor}(vocab_size={len(mock_vocab)})'
