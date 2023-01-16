# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.models import kie_predictor, ocr_predictor

predictor = ocr_predictor(pretrained=True)
det_predictor = predictor.det_predictor
reco_predictor = predictor.reco_predictor
kie_predictor = kie_predictor(pretrained=True)
