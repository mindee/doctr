import os
import tempfile

import onnxruntime
import pytest
import torch

from doctr.models import recognition
from doctr.models.recognition.crnn.pytorch import CTCPostProcessor
from doctr.models.recognition.master.pytorch import MASTERPostProcessor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.sar.pytorch import SARPostProcessor
from doctr.models.recognition.vitstr.pytorch import ViTSTRPostProcessor
from doctr.models.utils import export_model_to_onnx


@pytest.mark.parametrize(
    "arch_name, input_shape, pretrained",
    [
        ["crnn_vgg16_bn", (3, 32, 128), True],
        ["crnn_mobilenet_v3_small", (3, 32, 128), True],
        ["crnn_mobilenet_v3_large", (3, 32, 128), True],
        ["sar_resnet31", (3, 32, 128), False],
        ["master", (3, 32, 128), False],
        ["vitstr_small", (3, 32, 128), False],
        ["vitstr_base", (3, 32, 128), False],
    ],
)
def test_recognition_models(arch_name, input_shape, pretrained, mock_vocab):
    batch_size = 4
    model = recognition.__dict__[arch_name](vocab=mock_vocab, pretrained=pretrained, input_shape=input_shape).eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = ["i", "am", "a", "jedi"]

    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor, target, return_model_output=True, return_preds=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    assert isinstance(out["preds"], list)
    assert len(out["preds"]) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out["preds"])
    assert isinstance(out["out_map"], torch.Tensor)
    assert out["out_map"].dtype == torch.float32
    assert isinstance(out["loss"], torch.Tensor)
    # test model in train mode needs targets
    with pytest.raises(ValueError):
        model.train()
        model(input_tensor, None)


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        [CTCPostProcessor, [2, 119, 30]],
        [SARPostProcessor, [2, 119, 30]],
        [ViTSTRPostProcessor, [2, 119, 30]],
        [MASTERPostProcessor, [2, 119, 30]],
    ],
)
def test_reco_postprocessors(post_processor, input_shape, mock_vocab):
    processor = post_processor(mock_vocab)
    decoded = processor(torch.rand(*input_shape))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == input_shape[0]
    assert all(char in mock_vocab for word, _ in decoded for char in word)
    # Repr
    assert repr(processor) == f"{post_processor.__name__}(vocab_size={len(mock_vocab)})"


@pytest.mark.parametrize(
    "arch_name",
    [
        "crnn_vgg16_bn",
        "crnn_mobilenet_v3_small",
        "crnn_mobilenet_v3_large",
        "sar_resnet31",
        "master",
        "vitstr_small",
        "vitstr_base",
    ],
)
def test_recognition_zoo(arch_name):
    batch_size = 2
    # Model
    predictor = recognition.zoo.recognition_predictor(arch_name, pretrained=False)
    predictor.model.eval()
    # object check
    assert isinstance(predictor, RecognitionPredictor)
    input_tensor = torch.rand((batch_size, 3, 128, 128))
    if torch.cuda.is_available():
        predictor.model.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) for word, conf in out)


@pytest.mark.skipif(os.getenv("SLOW", "0") == "0", reason="slow test")
@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (3, 32, 128)],
        ["crnn_mobilenet_v3_small", (3, 32, 128)],
        ["crnn_mobilenet_v3_large", (3, 32, 128)],
        ["sar_resnet31", (3, 32, 128)],
        ["master", (3, 32, 128)],
        ["vitstr_small", (3, 32, 128)],  # testing one vitstr version is enough
    ],
)
def test_models_onnx_export(arch_name, input_shape):
    # Model
    batch_size = 2
    model = recognition.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path = export_model_to_onnx(model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input)
        assert os.path.exists(model_path)
        # Inference
        ort_session = onnxruntime.InferenceSession(
            os.path.join(tmpdir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        ort_outs = ort_session.run(["logits"], {"input": dummy_input.numpy()})
        assert isinstance(ort_outs, list) and len(ort_outs) == 1
        assert ort_outs[0].shape[0] == batch_size
