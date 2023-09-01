# git clone https://github.com/mindee/doctr.git
# pip install -e doctr/.[tf]
# conda install -y -c conda-forge weasyprint

import json
import os

import tensorflow as tf

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def main(args):
    # Load docTR model
    model = ocr_predictor(det_arch=args.arch_detection, reco_arch=args.arch_recognition, pretrained=True)

    # load image input file
    single_img_doc = DocumentFile.from_images(args.input_file)

    # inference
    output = model(single_img_doc)

    with open(args.output_file, "w") as f:
        json.dump(output.export(), f)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR inference image script(TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--arch_recognition", type=str, help="text-detection model")
    parser.add_argument("--arch_detection", type=str, help="text-recognition model")
    parser.add_argument("--input_file", type=str, help="path of image file")
    parser.add_argument("--output_file", type=str, help="path of output file")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
