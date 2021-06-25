import pytest
import torch

from doctr.models import recognition


def test_master(mock_vocab, max_len=50, batch_size=16):
    master = recognition.MASTER(vocab=mock_vocab, d_model=512, dff=512, num_heads=2, input_shape=(3, 48, 160))
    input_tensor = torch.rand((batch_size, 3, 48, 160))
    mock_labels = (len(mock_vocab) * torch.rand((batch_size, max_len))).int()
    logits = master(input_tensor, mock_labels)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, max_len, 1 + len(mock_vocab))  # 1 more for EOS
    prediction, logits = master.decode(input_tensor)
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(logits, torch.Tensor)
    assert prediction.shape == (batch_size, max_len)
    assert logits.shape == (batch_size, max_len, 1 + len(mock_vocab))
