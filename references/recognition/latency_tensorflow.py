# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Text recognition latency benchmark"""

import argparse
import os
import time

from doctr.file_utils import ensure_keras_v2

ensure_keras_v2()

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from doctr.models import recognition


def main(args):
    if args.gpu:
        gpu_devices = tf.config.list_physical_devices("GPU")
        if any(gpu_devices):
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        else:
            raise AssertionError("TensorFlow cannot access your GPU. Please investigate!")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    spatial_shape = (args.size, 4 * args.size)
    # Pretrained imagenet model
    model = recognition.__dict__[args.arch](
        pretrained=args.pretrained,
        pretrained_backbone=False,
        input_shape=(*spatial_shape, 3),
    )

    # Input
    img_tensor = tf.random.uniform(shape=[args.batch_size, *spatial_shape, 3], maxval=1, dtype=tf.float32)

    # Warmup
    for _ in range(10):
        _ = model(img_tensor, training=False)

    timings = []

    # Evaluation runs
    for _ in range(args.it):
        start_ts = time.perf_counter()
        _ = model(img_tensor, training=False)
        timings.append(time.perf_counter() - start_ts)

    _timings = np.array(timings)
    print(f"{args.arch} ({args.it} runs on {spatial_shape} inputs in batches of {args.batch_size})")
    print(f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="docTR latency benchmark for text recognition (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("arch", type=str, help="Architecture to use")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="The batch_size")
    parser.add_argument("--size", type=int, default=32, help="The image input size")
    parser.add_argument("--gpu", dest="gpu", help="Should the benchmark be performed on GPU", action="store_true")
    parser.add_argument("--it", type=int, default=100, help="Number of iterations to run")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    args = parser.parse_args()

    main(args)
