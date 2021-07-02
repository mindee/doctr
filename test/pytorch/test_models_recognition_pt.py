import pytest
import torch

from doctr.models import recognition


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (3, 32, 128)],
        ["sar_resnet31", (3, 32, 128)],
    ],
)
def test_recognition_models(arch_name, input_shape, mock_vocab):
    batch_size = 4
    reco_model = recognition.__dict__[arch_name](vocab=mock_vocab, pretrained=False, input_shape=input_shape).eval()
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


def test_master(mock_vocab, max_len=50, batch_size=4):
    master = recognition.MASTER(vocab=mock_vocab, input_shape=(3, 32, 128))
    input_tensor = torch.rand((batch_size, 3, 32, 128))
    target = ["i", "am", "a", "jedi"]
    logits = master(input_tensor, target, return_model_output=True)['out_map']
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, max_len, 3 + len(mock_vocab))
    prediction, logits = master.decode(input_tensor)
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(logits, torch.Tensor)
    assert prediction.shape == (batch_size, max_len)
    assert logits.shape == (batch_size, max_len, 3 + len(mock_vocab))
