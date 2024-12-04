import os
import tempfile

import numpy as np
import onnxruntime
import psutil
import pytest
import torch

from doctr.io import DocumentFile
from doctr.models import recognition
from doctr.models.recognition.crnn.pytorch import CTCPostProcessor
from doctr.models.recognition.master.pytorch import MASTERPostProcessor
from doctr.models.recognition.parseq.pytorch import PARSeqPostProcessor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.sar.pytorch import SARPostProcessor
from doctr.models.recognition.vitstr.pytorch import ViTSTRPostProcessor
from doctr.models.utils import _CompiledModule, export_model_to_onnx

system_available_memory = int(psutil.virtual_memory().available / 1024**3)


@pytest.mark.parametrize("train_mode", [True, False])
@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (3, 32, 128)],
        ["crnn_mobilenet_v3_small", (3, 32, 128)],
        ["crnn_mobilenet_v3_large", (3, 32, 128)],
        ["sar_resnet31", (3, 32, 128)],
        ["master", (3, 32, 128)],
        ["vitstr_small", (3, 32, 128)],
        ["vitstr_base", (3, 32, 128)],
        ["parseq", (3, 32, 128)],
    ],
)
def test_recognition_models(arch_name, input_shape, train_mode, mock_vocab):
    batch_size = 4
    model = recognition.__dict__[arch_name](vocab=mock_vocab, pretrained=True, input_shape=input_shape)
    model = model.train() if train_mode else model.eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = ["i", "am", "a", "jedi"]

    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor, target, return_model_output=True, return_preds=not train_mode)
    assert isinstance(out, dict)
    assert len(out) == 3 if not train_mode else len(out) == 2
    if not train_mode:
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
        [PARSeqPostProcessor, [2, 119, 30]],
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
        "parseq",
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


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (3, 32, 128)],
        ["crnn_mobilenet_v3_small", (3, 32, 128)],
        ["crnn_mobilenet_v3_large", (3, 32, 128)],
        pytest.param(
            "sar_resnet31",
            (3, 32, 128),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "master", (3, 32, 128), marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory")
        ),
        ["vitstr_small", (3, 32, 128)],  # testing one vitstr version is enough
        ["parseq", (3, 32, 128)],
    ],
)
def test_models_onnx_export(arch_name, input_shape):
    # Model
    batch_size = 2
    model = recognition.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    pt_logits = model(dummy_input)["logits"].detach().cpu().numpy()
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
    assert ort_outs[0].shape == pt_logits.shape
    # Check that the output is close to the PyTorch output - only warn if not close
    try:
        assert np.allclose(pt_logits, ort_outs[0], atol=1e-4)
    except AssertionError:
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {np.max(np.abs(pt_logits - ort_outs[0]))}")


@pytest.mark.parametrize(
    "arch_name",
    [
        "crnn_vgg16_bn",
        "crnn_mobilenet_v3_small",
        "crnn_mobilenet_v3_large",
        "sar_resnet31",
        # "master",  NOTE: MASTER model isn't 100% safe compilable yet (pytorch v2.5.1) - sometimes it fails to compile.
        "vitstr_small",
        "vitstr_base",
        "parseq",
    ],
)
def test_torch_compiled_models(arch_name, mock_text_box):
    doc = DocumentFile.from_images([mock_text_box])
    predictor = recognition.zoo.recognition_predictor(arch_name, pretrained=True)
    assert isinstance(predictor, RecognitionPredictor)
    out = predictor(doc)

    # Compile the model
    compiled_model = torch.compile(recognition.__dict__[arch_name](pretrained=True).eval())
    assert isinstance(compiled_model, _CompiledModule)
    compiled_predictor = recognition.zoo.recognition_predictor(compiled_model)
    compiled_out = compiled_predictor(doc)

    # Compare
    assert out[0][0] == compiled_out[0][0]
    assert np.allclose(out[0][1], compiled_out[0][1], atol=1e-4)
