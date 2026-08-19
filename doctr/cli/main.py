# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.version import __version__

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Canonical export formats and their aliases, mirroring `Document.export_as`
FORMAT_ALIASES: dict[str, str] = {
    "json": "json",
    "dict": "json",
    "txt": "txt",
    "text": "txt",
    "md": "md",
    "markdown": "md",
    "adoc": "adoc",
    "asciidoc": "adoc",
    "html": "html",
    "xml": "xml",
    "hocr": "xml",
}

# Used to infer the export format from the `--output` extension when `--format` is not given
SUFFIX_ALIASES: dict[str, str] = {
    ".json": "json",
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
    ".adoc": "adoc",
    ".asciidoc": "adoc",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".hocr": "xml",
}

READING_DIRECTIONS = ["auto", "ltr", "rtl", "ttb-rtl", "ttb-ltr"]


def _resolve_format(args: argparse.Namespace) -> str:
    """Resolve the canonical export format, inferring it from the output extension if needed

    Args:
        args: the parsed command-line arguments

    Returns:
        one of 'json', 'txt', 'md', 'adoc', 'html', 'xml'
    """
    if args.format is not None:
        return FORMAT_ALIASES[args.format.strip().lower()]
    return SUFFIX_ALIASES.get(Path(args.output).suffix.lower(), "json")


def _load_document(args: argparse.Namespace) -> list[np.ndarray]:
    """Load every input into a single list of pages

    Args:
        args: the parsed command-line arguments

    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x 3
    """
    pdf_kwargs: dict[str, Any] = {"scale": args.pdf_scale, "password": args.pdf_password}

    pages: list[np.ndarray] = []
    for input_path in args.input_path:
        try:
            if input_path.lower().startswith(("http://", "https://")):
                pages.extend(DocumentFile.from_url(input_path, **pdf_kwargs))
            elif input_path.lower().endswith(".pdf"):
                pages.extend(DocumentFile.from_pdf(input_path, **pdf_kwargs))
            else:
                pages.extend(DocumentFile.from_images(input_path))
            logging.info(f"Document loaded successfully from {input_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {input_path}")
            sys.exit(1)
        except ValueError:
            logging.error(f"File could not be read as a valid image or PDF: {input_path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error occurred while loading the document: {e}")
            sys.exit(1)

    return pages


def _resolve_device(device: str | None) -> torch.device:
    """Resolve the device the models should be loaded on

    Args:
        device: the `--device` value

    Returns:
        the resolved torch device
    """
    if device is None or device.strip().lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device)
    except (RuntimeError, ValueError) as e:
        logging.error(f"Invalid device '{device}': {e}")
        sys.exit(1)


def _set_thresholds(model: Any, args: argparse.Namespace) -> None:
    """Override the post-processing thresholds of the detection model

    Args:
        model: the predictor
        args: the parsed command-line arguments
    """
    if args.bin_thresh is None and args.box_thresh is None:
        return

    det_model = getattr(getattr(model, "det_predictor", None), "model", None)
    postprocessor = getattr(det_model, "postprocessor", None)
    if postprocessor is None:
        logging.warning("The detection model exposes no postprocessor: --bin_thresh & --box_thresh are ignored")
        return

    if args.bin_thresh is not None:
        postprocessor.bin_thresh = args.bin_thresh
    if args.box_thresh is not None:
        postprocessor.box_thresh = args.box_thresh


def _to_device(model: Any, args: argparse.Namespace) -> Any:
    """Load the predictor on the requested device

    Args:
        model: the predictor
        args: the parsed command-line arguments

    Returns:
        the predictor, loaded on the requested device
    """
    device = _resolve_device(args.device)
    try:
        model = model.to(device)
        if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0):
            model = model.bfloat16()
            logging.info(f"Model loaded on {device} with bfloat16 precision")
    except (RuntimeError, AssertionError, ValueError) as e:
        logging.error(f"Could not load the model on device '{device}': {e}")
        sys.exit(1)
    logging.info(f"Model loaded on {device}")
    return model


def _build_predictor(args: argparse.Namespace) -> Any:
    """Instantiate the OCR predictor matching the requested options

    Args:
        args: the parsed command-line arguments

    Returns:
        the predictor
    """
    # Region masking is resolved by the layout model, so it has to be enabled along with `ignore_regions`
    detect_layout = args.detect_layout or bool(args.ignore_regions)
    if detect_layout and not args.detect_layout:
        logging.info("--ignore_regions requires the layout model: layout detection has been enabled")

    kwargs: dict[str, Any] = {
        "det_arch": args.det_arch,
        "reco_arch": args.reco_arch,
        "pretrained": True,
        "assume_straight_pages": args.assume_straight_pages,
        "preserve_aspect_ratio": args.preserve_aspect_ratio,
        "symmetric_pad": args.symmetric_pad,
        "export_as_straight_boxes": args.export_as_straight_boxes,
        "detect_orientation": args.detect_orientation,
        "straighten_pages": args.straighten_pages,
        "detect_language": args.detect_language,
        "detect_layout": detect_layout,
        "layout_arch": args.layout_arch,
        "ignore_regions": args.ignore_regions or None,
        "det_bs": args.det_bs,
        "reco_bs": args.reco_bs,
        # `_OCRPredictor` keyword args
        "disable_page_orientation": args.disable_page_orientation,
        "disable_crop_orientation": args.disable_crop_orientation,
        "preserve_original_coords": args.preserve_original_coords,
        # `DocumentBuilder` keyword args
        "resolve_lines": args.resolve_lines,
        "resolve_blocks": args.resolve_blocks,
        "paragraph_break": args.paragraph_break,
        "keep_reading_order": args.keep_reading_order,
    }

    model = ocr_predictor(detect_tables=args.detect_tables, **kwargs)

    _set_thresholds(model, args)
    return _to_device(model, args)


def _export_kwargs(fmt: str, args: argparse.Namespace) -> dict[str, Any]:
    """Build the keyword arguments accepted by the exporter of the requested format

    Args:
        fmt: the canonical export format
        args: the parsed command-line arguments

    Returns:
        the keyword arguments to pass to `Document.export_as`
    """
    kwargs: dict[str, Any] = {}
    if fmt in {"json", "xml"}:
        kwargs["reading_order"] = args.reading_order
    if fmt == "xml":
        kwargs["file_title"] = args.file_title
    if fmt in {"txt", "md", "adoc", "html", "xml"}:
        kwargs["direction"] = args.direction
    if fmt in {"md", "adoc"}:
        kwargs["escape"] = args.escape
    if fmt in {"txt", "md", "adoc", "html"}:
        kwargs["include_furniture"] = args.include_furniture
    return kwargs


def _xml_paths(output: str, num_pages: int) -> list[Path]:
    """Build one output path per page, since hOCR describes a single page per file

    Args:
        output: the `--output` value
        num_pages: the number of exported pages

    Returns:
        the list of paths to write to
    """
    path = Path(output)
    if num_pages <= 1:
        return [path]
    return [path.with_name(f"{path.stem}_page_{idx + 1}{path.suffix}") for idx in range(num_pages)]


def _save_results(result: Any, fmt: str, args: argparse.Namespace) -> None:
    """Export the predictions and write them to disk

    Args:
        result: the document returned by the predictor
        fmt: the canonical export format
        args: the parsed command-line arguments
    """
    try:
        exported = result.export_as(fmt, **_export_kwargs(fmt, args))
    except Exception as e:
        logging.error(f"Results could not be exported as '{fmt}': {e}")
        sys.exit(1)

    try:
        if fmt == "xml":
            # `export_as_xml` returns one (bytes, ElementTree) tuple per page
            paths = _xml_paths(args.output, len(exported))
            for path, (xml_bytes, _) in zip(paths, exported):
                path.write_bytes(xml_bytes)
            logging.info(f"Results saved to {', '.join(str(path) for path in paths)}")
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                if fmt == "json":
                    json.dump(exported, f, indent=args.indent, ensure_ascii=False)
                else:
                    f.write(exported)
            logging.info(f"Results saved to {args.output}")
    except FileNotFoundError:
        logging.error(f"Could not write output file at given path: {args.output}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Results could not be saved: {e}")
        sys.exit(1)


def main(argv=None):
    """Main function for the docTR CLI tool"""
    # parse command-line arguments and set up the model
    args = _parse_args(argv)
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    fmt = _resolve_format(args)
    model = _build_predictor(args)

    # load the document
    doc = _load_document(args)

    # perform OCR
    logging.info("Performing OCR...")
    result = model(doc)

    # save results to the requested format
    _save_results(result, fmt, args)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="docTR CLI tool for OCR prediction on images and PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"docTR {__version__}")

    # required input path(s)
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        required=True,
        help="path to one or more input image / PDF files, or URLs of web pages to render as PDF",
    )
    parser.add_argument("--pdf_password", type=str, default=None, help="password to unlock encrypted PDF files")
    parser.add_argument("--pdf_scale", type=int, default=2, help="PDF rendering scale (1 corresponds to 72dpi)")

    # architecture selection
    parser.add_argument(
        "--det_arch",
        type=str,
        default="fast_base",
        help="name of the detection architecture or the model itself to use",
    )
    parser.add_argument(
        "--reco_arch",
        type=str,
        default="crnn_vgg16_bn",
        help="name of the recognition architecture or the model itself to use",
    )
    parser.add_argument("--layout_arch", type=str, default="lw_detr_s", help="name of the layout architecture to use")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to load the models on, e.g. 'cpu', 'cuda', 'cuda:1' or 'mps' ('auto' picks a GPU if available)",
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
    parser.add_argument(
        "--symmetric_pad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply symmetric padding",
    )
    parser.add_argument(
        "--bin_thresh",
        type=float,
        default=None,
        help="binarization threshold of the detection segmentation map (defaults to the architecture value)",
    )
    parser.add_argument(
        "--box_thresh",
        type=float,
        default=None,
        help="minimal objectness score to consider a detected box (defaults to the architecture value)",
    )
    parser.add_argument("--det_bs", type=int, default=2, help="batch size for detection")
    parser.add_argument("--reco_bs", type=int, default=128, help="batch size for recognition")
    parser.add_argument("--detect_orientation", action="store_true", help="automatically detect page orientation")
    parser.add_argument("--detect_language", action="store_true", help="detect language of the text")
    parser.add_argument("--detect_layout", action="store_true", help="attach the detected layout regions to each page")
    parser.add_argument(
        "--detect_tables",
        action="store_true",
        help="regroup the words of the detected tables into structured tables (enables the layout model)",
    )
    parser.add_argument(
        "--ignore_regions",
        type=str,
        nargs="+",
        default=None,
        metavar="CLASS",
        help="layout class names to mask out before detection & recognition (enables the layout model), "
        "e.g. Picture Table",
    )
    parser.add_argument(
        "--export_as_straight_boxes",
        action="store_true",
        help="export rotated predictions as straight boxes (only with --no-assume_straight_pages)",
    )
    parser.add_argument(
        "--preserve_original_coords",
        action="store_true",
        help="map the boxes back to the original page coordinates (only with --straighten_pages)",
    )
    parser.add_argument("--disable_page_orientation", action="store_true", help="disable the page orientation model")
    parser.add_argument("--disable_crop_orientation", action="store_true", help="disable the crop orientation model")

    # document assembling options
    parser.add_argument(
        "--resolve_lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically group words into lines",
    )
    parser.add_argument("--resolve_blocks", action="store_true", help="automatically group lines into blocks")
    parser.add_argument(
        "--paragraph_break",
        type=float,
        default=0.035,
        help="relative length of the minimum space separating paragraphs",
    )
    parser.add_argument(
        "--keep_reading_order", action="store_true", help="arrange the content of every page in reading order"
    )

    # output options
    parser.add_argument("--output", type=str, default="results.json", help="path to output the results")
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=sorted(FORMAT_ALIASES),
        help="export format (inferred from the --output extension, or JSON, when not set)",
    )
    parser.add_argument(
        "--direction", type=str, default="auto", choices=READING_DIRECTIONS, help="reading direction of the document"
    )
    parser.add_argument(
        "--reading_order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="linearize the content in reading order (JSON & hOCR exports)",
    )
    parser.add_argument(
        "--escape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="escape the characters carrying a structural meaning (Markdown & AsciiDoc exports)",
    )
    parser.add_argument(
        "--include_furniture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include page headers, page footers and footnotes (text-like exports)",
    )
    parser.add_argument(
        "--file_title", type=str, default="docTR - XML export (hOCR)", help="title of the exported hOCR files"
    )
    parser.add_argument("--indent", type=int, default=4, help="indentation of the JSON export")
    parser.add_argument("--quiet", action="store_true", help="only log errors")

    args = parser.parse_args(argv)

    for name in ("bin_thresh", "box_thresh"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be between 0 and 1")

    return args


if __name__ == "__main__":
    main()
