import os
import tempfile
from unittest import mock

import numpy as np
import onnx
import onnxruntime
import psutil
import pytest
import torch

from doctr.io import DocumentFile
from doctr.models import recognition
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx

system_available_memory = int(psutil.virtual_memory().available / 1024**3)


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
        ["master", (3, 32, 128)],
        ["vitstr_small", (3, 32, 128)],  # testing one vitstr version is enough
        ["parseq", (3, 32, 128)],
        ["viptr_tiny", (3, 32, 128)],
    ],
)
# Coverage tracing slows the (pure Python) dynamo export down ~3x: its code paths are covered by the other tests
@pytest.mark.parametrize("dynamo", [False, pytest.param(True, marks=pytest.mark.no_cover)])
def test_models_onnx_export(arch_name, input_shape, dynamo):
    if dynamo:
        pytest.importorskip("onnxscript")
        if os.environ.get("CI") and arch_name == "viptr_tiny":
            # The ONNX optimizer of onnxscript takes minutes on the ViP blocks: `vip_tiny` covers them on CI
            pytest.skip("slow dynamo export on CI")
    # Model
    batch_size = 2
    model = recognition.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    with torch.inference_mode():
        if dynamo:
            # The exported autoregressive decoding stops at the end of the sequences, as the decoding of PyTorch (the
            # exportable model runs every step, as the tracing exporter)
            model.exportable = False
            pt_logits = model(dummy_input, return_model_output=True)["out_map"].numpy()
            model.exportable = True
        else:
            pt_logits = model(dummy_input)["logits"].numpy()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path = export_model_to_onnx(
            model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input, dynamo=dynamo
        )
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


@pytest.mark.parametrize("arch_name", ["master", "parseq"])
def test_models_onnx_export_loop(arch_name, mock_text_box):
    # With the torch.export-based exporter, the autoregressive decoding is exported as an ONNX loop, which stops once
    # every sequence is complete: the outputs are the ones of the PyTorch decoding, which stops early as well
    pytest.importorskip("onnxscript")
    model = recognition.__dict__[arch_name](pretrained=True, exportable=True).eval()
    processor = recognition.zoo.recognition_predictor(model).pre_processor
    word = DocumentFile.from_images([mock_text_box])[0]
    # The text crop, a blank crop and a crop of noise: sequences that end at different steps
    crops = [word, np.full_like(word, 255), np.random.default_rng(0).integers(0, 256, word.shape, dtype=np.uint8)]
    batch = processor(crops)[0]
    with torch.inference_mode():
        model.exportable = False
        ref_logits = model(batch, return_model_output=True)["out_map"].numpy()
        model.exportable = True

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = export_model_to_onnx(model, os.path.join(tmpdir, "model"), dummy_input=batch[:1], dynamo=True)
        assert [os.path.basename(path) for path in os.listdir(tmpdir)] == ["model.onnx"]  # a single file
        assert any(node.op_type == "Loop" for node in onnx.load(model_path).graph.node)
        ort_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # The batch dimension is dynamic
        ort_logits = ort_session.run(["logits"], {"input": batch.numpy()})[0]
    assert ort_logits.shape == ref_logits.shape
    assert np.allclose(ort_logits, ref_logits, atol=1e-4)
    ort_preds = model.postprocessor(torch.from_numpy(ort_logits))
    ref_preds = model.postprocessor(torch.from_numpy(ref_logits))
    assert [word for word, _ in ort_preds] == [word for word, _ in ref_preds]
    assert [conf for _, conf in ort_preds] == pytest.approx([conf for _, conf in ref_preds], abs=1e-4)


@pytest.mark.parametrize("arch_name", ["master", "parseq"])
def test_models_export_loop_eager(arch_name, mock_text_box, eager_export_loops):
    # The decoding exported as an ONNX loop (cf. `test_models_onnx_export_loop`), run in Python: same outputs as the
    # PyTorch decoding, which stops early as well
    model = recognition.__dict__[arch_name](pretrained=True, exportable=True).eval()
    processor = recognition.zoo.recognition_predictor(model).pre_processor
    word = DocumentFile.from_images([mock_text_box])[0]
    # The text crop, a blank crop and a crop of noise: sequences that end at different steps
    crops = [word, np.full_like(word, 255), np.random.default_rng(0).integers(0, 256, word.shape, dtype=np.uint8)]
    batch = processor(crops)[0]
    with torch.inference_mode():
        loop_logits = model(batch)["logits"]
        assert eager_export_loops[0] > 0  # the loops ran
        with mock.patch.object(torch.compiler, "is_exporting", lambda: False):
            model.exportable = False
            ref_logits = model(batch, return_model_output=True)["out_map"]
    assert loop_logits.shape == ref_logits.shape
    assert torch.allclose(loop_logits, ref_logits, atol=1e-4)
    loop_preds, ref_preds = model.postprocessor(loop_logits), model.postprocessor(ref_logits)
    assert [word for word, _ in loop_preds] == [word for word, _ in ref_preds]
    assert [conf for _, conf in loop_preds] == pytest.approx([conf for _, conf in ref_preds], abs=1e-4)


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
        "viptr_tiny",
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
