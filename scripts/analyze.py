# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Enable GPU growth if using TF
if is_tf_available():
    import tensorflow as tf

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if any(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def main(args):
    model = ocr_predictor(args.detection, args.recognition, pretrained=True)

    if args.path.lower().endswith(".pdf"):
        doc = DocumentFile.from_pdf(args.path)
    else:
        doc = DocumentFile.from_images(args.path)

    out = model(doc)

    for page, img in zip(out.pages, doc):
        page.show(img, block=not args.noblock, interactive=not args.static)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR end-to-end analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("path", type=str, help="Path to the input document (PDF or image)")
    parser.add_argument("--detection", type=str, default="db_resnet50", help="Text detection model to use for analysis")
    parser.add_argument(
        "--recognition", type=str, default="crnn_vgg16_bn", help="Text recognition model to use for analysis"
    )
    parser.add_argument(
        "--noblock", dest="noblock", help="Disables blocking visualization. Used only for CI.", action="store_true"
    )
    parser.add_argument("--static", dest="static", help="Switches to static visualization", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
