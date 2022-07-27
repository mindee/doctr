# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

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


_OUTPUT_CHOICE_JSON = "json"
_OUTPUT_CHOICE_TEXT = "txt"


def _process_file(model, file_path: Path, out_format: str) -> str:
    if str(file_path).lower().endswith(".pdf"):
        doc = DocumentFile.from_pdf(file_path)
    else:
        doc = DocumentFile.from_images(file_path)

    out = model(doc)
    export = out.export()

    if out_format == _OUTPUT_CHOICE_JSON:
        out_txt = json.dumps(export, indent=2)
    elif out_format == _OUTPUT_CHOICE_TEXT:
        out_txt = ""
        for page in export["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        out_txt += word["value"] + " "
                    out_txt += "\n"
            out_txt += "\n\n"
    else:
        out_txt = ""
    return out_txt


def main(args):
    model = ocr_predictor(args.detection, args.recognition, pretrained=True)
    path = Path(args.path)
    if path.is_dir():
        allowed = (".pdf", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp")
        to_process = [f for f in path.iterdir() if str(f).lower().endswith(allowed)]
        for filename in tqdm(to_process):
            out_path = path.joinpath(f"{filename}.{args.format}")
            if out_path.exists():
                continue
            in_path = path.joinpath(filename)
            # print(in_path)
            out_str = _process_file(model, in_path, args.format)
            with open(out_path, "w") as fh:
                fh.write(out_str)
    else:
        out_str = _process_file(model, path, args.format)
        print(out_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DocTR text detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to process: PDF, image, directory",
    )
    parser.add_argument(
        "--detection",
        type=str,
        default="db_resnet50",
        help="Text detection model to use for analysis",
    )
    parser.add_argument(
        "--recognition",
        type=str,
        default="crnn_vgg16_bn",
        help="Text recognition model to use for analysis",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=[
            _OUTPUT_CHOICE_JSON,
            _OUTPUT_CHOICE_TEXT,
        ],
        default=_OUTPUT_CHOICE_TEXT,
        help="Output format",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        main(parsed_args)
    except KeyboardInterrupt:
        print("Cancelled")
        pass
