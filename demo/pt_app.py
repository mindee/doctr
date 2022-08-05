
# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

# Set environment variables to load models using either Tensorflow or PyTorch
# Note: we need to set env variable before importing anything from doctr!
os.environ["USE_TORCH"] = "YES"

from common import main

DET_ARCHS = ["db_resnet50", "db_mobilenet_v3_large", "linknet_resnet50_rotation"]
RECO_ARCHS = ["crnn_vgg16_bn", "crnn_mobilenet_v3_small", "master", "sar_resnet31"]

if __name__ == '__main__':
    main(DET_ARCHS, RECO_ARCHS)
