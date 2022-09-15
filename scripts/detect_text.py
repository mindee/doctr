# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

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

IMAGE_FILE_EXTENSIONS = [".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"]
OTHER_EXTENSIONS = [".pdf"]


def _process_file(model, file_path: Path, out_format: str) -> None:

    if out_format not in ["txt", "json", "xml"]:
        raise ValueError(f"Unsupported output format: {out_format}")

    if os.path.splitext(file_path)[1] in IMAGE_FILE_EXTENSIONS:
        doc = DocumentFile.from_images([file_path])
    elif os.path.splitext(file_path)[1] in OTHER_EXTENSIONS:
        doc = DocumentFile.from_pdf(file_path)
    else:
        print(f"Skip unsupported file type: {file_path}")

    out = model(doc)

    if out_format == "json":
        output = json.dumps(out.export(), indent=2)
    elif out_format == "txt":
        output = out.render()
    elif out_format == "xml":
        output = out.export_as_xml()

    path = Path("output").joinpath(file_path.stem + "." + out_format)
    if out_format == "xml":
        for i, (xml_bytes, xml_tree) in enumerate(output):
            path = Path("output").joinpath(file_path.stem + f"_{i}." + out_format)
            xml_tree.write(path, encoding="utf-8", xml_declaration=True)
    else:
        with open(path, "w") as f:
            f.write(output)


def main(args):
    model = ocr_predictor(args.detection, args.recognition, pretrained=True)
    path = Path(args.path)

    os.makedirs(name="output", exist_ok=True)

    if path.is_dir():
        to_process = [
            f for f in path.iterdir() if str(f).lower().endswith(tuple(IMAGE_FILE_EXTENSIONS + OTHER_EXTENSIONS))
        ]
        for file_path in tqdm(to_process):
            _process_file(model, file_path, args.format)
    else:
        _process_file(model, path, args.format)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DocTR text detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Path to process: PDF, image, directory")
    parser.add_argument("--detection", type=str, default="db_resnet50", help="Text detection model to use for analysis")
    parser.add_argument(
        "--recognition", type=str, default="crnn_vgg16_bn", help="Text recognition model to use for analysis"
    )
    parser.add_argument("-f", "--format", choices=["txt", "json", "xml"], default="txt", help="Output format")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
