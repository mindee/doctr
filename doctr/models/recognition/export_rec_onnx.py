import time

import numpy as np
import torch.onnx

from doctr.models import ocr_predictor

model = ocr_predictor(reco_arch = "crnn_efficientnetv2_mV2", pretrained=True)
model.reco_predictor.model = model.reco_predictor.model.eval()

input = torch.randn(1, 3, 32, 128)
input2 = torch.randn(49, 3, 32, 128)
model = model.to("cpu")
start = time.time()
pred = model.reco_predictor.model(input)
print("pytorch time", time.time() - start)
torch.onnx.export(model.reco_predictor.model,
                  input,
                  "crnn_effnetv2_mV2.onnx",
                  export_params = True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ["input"],
                  output_names = ["output"],
                  dynamic_axes = {"input":{0:"batch_size"},
                                  "output":{0:"batch_size"}})

import onnxruntime

ort_session = onnxruntime.InferenceSession("crnn_effnetv2_mV2.onnx", providers = ['CPUExecutionProvider'])

ort_inputs = {"input":input.numpy()}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
print("onnx time", time.time() - start)
print(np.testing.assert_allclose(pred.detach().cpu().numpy(), ort_outs[0], rtol=1e-3, atol=1e-5))

ort_inputs = {"input":input2.numpy()}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
print("onnx time", time.time() - start)
start = time.time()
pred = model.reco_predictor.model(input2)
print("pytorch time", time.time() - start)
print(np.testing.assert_allclose(pred.detach().cpu().numpy(), ort_outs[0], rtol=1e-3, atol=1e-5))


from openvino.runtime import Core

ie = Core()
model_onnx = ie.read_model(model="crnn_effnetv2_mV2.onnx")
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

output_layer_onnx = compiled_model_onnx.output(0)

start = time.time()
# Run inference on the input image.
res_onnx = compiled_model_onnx([input2.numpy()])[output_layer_onnx]
print(res_onnx.shape)
print("openvino time", time.time() - start)
print(np.testing.assert_allclose(pred.detach().cpu().numpy(), res_onnx, rtol=1e-3, atol=1e-5))
