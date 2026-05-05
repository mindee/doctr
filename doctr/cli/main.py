# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import json
import logging
import sys

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def main(argv=None):
    """Main function for the docTR CLI tool"""
    # parse command-line arguments and set up the model
    args = _parse_args(argv)
    model = ocr_predictor(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True,
        pretrained_backbone=True,
        assume_straight_pages=args.assume_straight_pages,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
        symmetric_pad=args.symmetric_pad,
        detect_orientation=args.detect_orientation,
        straighten_pages=args.straighten_pages,
        detect_language=args.detect_language,
        det_bs=args.det_bs,
        reco_bs=args.reco_bs,
    )

    # load the document
    try:
        if args.input_path.lower().endswith(".pdf"):
            doc = DocumentFile.from_pdf(args.input_path)
        else:
            doc = DocumentFile.from_images(args.input_path)
        logging.info(f"Document loaded successfully from {args.input_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {args.input_path}")
        sys.exit(1)
    except ValueError:
        logging.error(f"File could not be read as a valid image or PDF: {args.input_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error occurred while loading the document: {e}")
        sys.exit(1)

    # perform OCR
    logging.info("Performing OCR...")
    result = model(doc)

    # save results to JSON file
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.export(), f, indent=4, ensure_ascii=False)
        logging.info(f"Results saved to {args.output}")
    except FileNotFoundError:
        logging.error(f"Could not write output file at given path: {args.output}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Results could not be saved: {e}")
        sys.exit(1)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="docTR CLI tool for OCR prediction on images and PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required input path
    parser.add_argument("--input_path", type=str, required=True, help="path to input image or PDF file")

    # architecture selection
    parser.add_argument(
        "--det_arch",
        type=str,
        default="db_resnet50",
        help="name of the detection architecture or the model itself to use",
    )
    parser.add_argument(
        "--reco_arch",
        type=str,
        default="crnn_vgg16_bn",
        help="name of the recognition architecture or the model itself to use",
    )

    # processing options
    parser.add_argument(
        "--assume_straight_pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="assume only straight pages without rotated textual elements",
    )
    parser.add_argument(
        "--straighten_pages", action="store_true", help="attempt to straighten skewed pages before analysis"
    )
    parser.add_argument(
        "--preserve_aspect_ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preserve aspect ratio when resizing pages",
    )
    parser.add_argument("--symmetric_pad", action="store_true", help="apply symmetric padding")
    parser.add_argument("--det_bs", type=int, default=2, help="batch size for detection")
    parser.add_argument("--reco_bs", type=int, default=128, help="batch size for recognition")
    parser.add_argument("--detect_orientation", action="store_true", help="automatically detect page orientation")
    parser.add_argument("--detect_language", action="store_true", help="detect language of the text")

    # output options
    parser.add_argument("--output", type=str, default="results.json", help="path to output results in JSON format")

    return parser.parse_args(argv)
