import torch.onnx
from doctr.models import ocr_predictor
import numpy as np
import time
import cv2
model = ocr_predictor(pretrained=True)
model.det_predictor.model = model.det_predictor.model.eval()

input = torch.randn(1, 3, 1024, 1024)
input2 = torch.randn(1, 3, 1536, 1536)
start = time.time()
pred = model.det_predictor.model(input)
print("pytorch time", time.time() - start)
torch.onnx.export(model.det_predictor.model,
                  input,
                  "det.onnx",
                  export_params = True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ["input"],
                  output_names = ["output"],
                  dynamic_axes = {"input":{0:"batch_size", 2:"x_axis", 3:"y_axis"},
                                  "output":{0:"batch_size", 2:"x_axis", 3:"y_axis"}})

import onnxruntime

ort_session = onnxruntime.InferenceSession("det.onnx")

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
pred = model.det_predictor.model(input2)
print("pytorch time", time.time() - start)
print(np.testing.assert_allclose(pred.detach().cpu().numpy(), ort_outs[0], rtol=1e-3, atol=1e-5))


from openvino.runtime import Core
ie = Core()
model_onnx = ie.read_model(model="det.onnx")
start = time.time()
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

output_layer_onnx = compiled_model_onnx.output(0)
print("model compilation time", time.time() - start)
start = time.time()
# Run inference on the input image.
print(input2.numpy().shape, input2.dtype)
res_onnx = compiled_model_onnx([input2.numpy()])[output_layer_onnx]
print(res_onnx.shape)
print("openvino time", time.time() - start)
print(np.testing.assert_allclose(pred.detach().cpu().numpy(), res_onnx, rtol=1e-3, atol=1e-5))
