# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from doctr.file_utils import is_tf_available

# Enable GPU growth if using TF
if is_tf_available():
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if any(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    import torch


def main(args):

    model = ocr_predictor(args.detection, args.recognition, pretrained=True)

    if not is_tf_available():
        model.det_predictor.pre_processor = model.det_predictor.pre_processor.eval()
        model.det_predictor.model = model.det_predictor.model.eval()
        model.reco_predictor.pre_processor = model.reco_predictor.pre_processor.eval()
        model.reco_predictor.model = model.reco_predictor.model.eval()

    if args.path.endswith(".pdf"):
        doc = DocumentFile.from_pdf(args.path).as_images()
    else:
        doc = DocumentFile.from_images(args.path)

    if is_tf_available():
        out = model(doc, training=False)
    else:
        with torch.no_grad():
            out = model(doc)

    for page, img in zip(out.pages, doc):
        page.show(img, block=not args.noblock, interactive=not args.static)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', type=str, help='Path to the input document (PDF or image)')
    parser.add_argument('--detection', type=str, default='db_resnet50',
                        help='Text detection model to use for analysis')
    parser.add_argument('--recognition', type=str, default='crnn_vgg16_bn',
                        help='Text recognition model to use for analysis')
    parser.add_argument("--noblock", dest="noblock", help="Disables blocking visualization", action="store_true")
    parser.add_argument("--static", dest="static", help="Switches to static visualization", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
