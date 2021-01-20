# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf



model =  tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
)

out_1 = model.get_layer('conv2_block3_out')
out_2 = model.get_layer('conv3_block3_out')
out_3 = model.get_layer('conv4_block3_out')
out_4 = model.get_layer('conv5_block3_out')

print(out_1)