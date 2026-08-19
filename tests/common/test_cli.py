import ast
import inspect
import json
import logging
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import doctr.cli.main as cli
from doctr import __version__
from doctr.datasets.generator.base import synthesize_text_img
from doctr.io import DocumentFile
from doctr.io.exporters import DocumentExportsMixin
from doctr.models import ocr_predictor
from doctr.models.builder import DocumentBuilder
from doctr.models.predictor.base import _OCRPredictor

# Lighter architectures, used wherever the point of the test is the CLI rather than the accuracy
LIGHT_ARCHS = ["--det_arch", "db_mobilenet_v3_large", "--reco_arch", "crnn_mobilenet_v3_small"]


# Fixtures


@pytest.fixture(scope="module")
def text_images(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cli_images")
    paths = []
    for idx, sentence in enumerate(["I am a jedi", "No I am your father"]):
        text_img = synthesize_text_img(sentence, background_color=(255, 255, 255), text_color=(0, 0, 0))
        path = folder / f"page_{idx}.png"
        text_img.save(str(path))
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def predictor():
    return ocr_predictor(pretrained=True)


@pytest.fixture
def det_postprocessor(predictor):
    """The detection post-processor of the shared predictor, with its thresholds restored afterwards"""
    postprocessor = predictor.det_predictor.model.postprocessor
    defaults = (postprocessor.bin_thresh, postprocessor.box_thresh)
    yield postprocessor
    postprocessor.bin_thresh, postprocessor.box_thresh = defaults


@pytest.fixture(scope="module")
def ocr_result(predictor, text_images):
    return predictor(DocumentFile.from_images(text_images))


@pytest.fixture
def recorded_predictor(monkeypatch):
    """Record the kwargs `_build_predictor` passes to the zoo, and return the predictor it really builds"""
    recorder = {}

    def _record(**kwargs):
        recorder.update(kwargs)
        return ocr_predictor(**kwargs)

    monkeypatch.setattr(cli, "ocr_predictor", _record)
    return recorder


def _args(*argv):
    return cli._parse_args(["--input_path", "sample.pdf", *argv])


# Argument parsing


def test_parse_args_defaults():
    args = _args()

    assert args.input_path == ["sample.pdf"]
    assert args.pdf_password is None
    assert args.pdf_scale == 2
    assert args.det_arch == "fast_base"
    assert args.reco_arch == "crnn_vgg16_bn"
    assert args.layout_arch == "lw_detr_s"
    assert args.device == "auto"
    assert args.bin_thresh is None
    assert args.box_thresh is None
    assert args.assume_straight_pages is True
    assert args.straighten_pages is False
    assert args.preserve_aspect_ratio is True
    assert args.symmetric_pad is True
    assert args.det_bs == 2
    assert args.reco_bs == 128
    assert args.detect_orientation is False
    assert args.detect_language is False
    assert args.detect_layout is False
    assert args.detect_tables is False
    assert args.ignore_regions is None
    assert args.export_as_straight_boxes is False
    assert args.preserve_original_coords is False
    assert args.disable_page_orientation is False
    assert args.disable_crop_orientation is False
    assert args.resolve_lines is True
    assert args.resolve_blocks is False
    assert args.paragraph_break == 0.035
    assert args.keep_reading_order is False
    assert args.output == "results.json"
    assert args.format is None
    assert args.direction == "auto"
    assert args.reading_order is True
    assert args.escape is True
    assert args.include_furniture is True
    assert args.file_title == "docTR - XML export (hOCR)"
    assert args.indent == 4
    assert args.quiet is False


def test_parse_args_boolean_optional_flags():
    args = _args(
        "--no-assume_straight_pages",
        "--no-preserve_aspect_ratio",
        "--no-symmetric_pad",
        "--no-resolve_lines",
        "--no-reading_order",
        "--no-escape",
        "--no-include_furniture",
    )

    assert args.assume_straight_pages is False
    assert args.preserve_aspect_ratio is False
    assert args.symmetric_pad is False
    assert args.resolve_lines is False
    assert args.reading_order is False
    assert args.escape is False
    assert args.include_furniture is False


def test_parse_args_requires_input_path():
    with pytest.raises(SystemExit):
        cli._parse_args([])


def test_parse_args_custom_values():
    args = _args(
        "--det_arch",
        "custom_det",
        "--reco_arch",
        "custom_reco",
        "--layout_arch",
        "lw_detr_m",
        "--detect_orientation",
        "--detect_language",
        "--output",
        "output.json",
    )

    assert args.det_arch == "custom_det"
    assert args.reco_arch == "custom_reco"
    assert args.layout_arch == "lw_detr_m"
    assert args.detect_orientation is True
    assert args.detect_language is True
    assert args.output == "output.json"


def test_parse_args_multiple_input_paths():
    args = cli._parse_args(["--input_path", "a.png", "b.jpg", "c.pdf"])
    assert args.input_path == ["a.png", "b.jpg", "c.pdf"]


def test_parse_args_advanced_options():
    args = _args(
        "--pdf_password",
        "secret",
        "--pdf_scale",
        "3",
        "--detect_layout",
        "--detect_tables",
        "--ignore_regions",
        "Picture",
        "Table",
        "--export_as_straight_boxes",
        "--preserve_original_coords",
        "--disable_page_orientation",
        "--disable_crop_orientation",
        "--resolve_blocks",
        "--paragraph_break",
        "0.1",
        "--keep_reading_order",
        "--direction",
        "rtl",
        "--indent",
        "2",
        "--quiet",
    )

    assert args.pdf_password == "secret"
    assert args.pdf_scale == 3
    assert args.detect_layout is True
    assert args.detect_tables is True
    assert args.ignore_regions == ["Picture", "Table"]
    assert args.export_as_straight_boxes is True
    assert args.preserve_original_coords is True
    assert args.disable_page_orientation is True
    assert args.disable_crop_orientation is True
    assert args.resolve_blocks is True
    assert args.paragraph_break == 0.1
    assert args.keep_reading_order is True
    assert args.direction == "rtl"
    assert args.indent == 2
    assert args.quiet is True


def test_parse_args_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli._parse_args(["--version"])

    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--format", "yaml"],
        ["--direction", "diagonal"],
        ["--pdf_scale", "not_an_int"],
        ["--bin_thresh", "1.5"],
        ["--box_thresh", "-0.1"],
        ["--bin_thresh", "high"],
    ],
)
def test_parse_args_invalid_values(argv):
    with pytest.raises(SystemExit):
        _args(*argv)


# Export format resolution


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("results.json", "json"),
        ("results.TXT", "txt"),
        ("results.md", "md"),
        ("results.markdown", "md"),
        ("results.adoc", "adoc"),
        ("results.html", "html"),
        ("results.htm", "html"),
        ("results.xml", "xml"),
        ("results.hocr", "xml"),
        ("results", "json"),
        ("results.unknown", "json"),
    ],
)
def test_resolve_format_from_output(output, expected):
    assert cli._resolve_format(_args("--output", output)) == expected


@pytest.mark.parametrize(("alias", "expected"), sorted(cli.FORMAT_ALIASES.items()))
def test_resolve_format_explicit_overrides_output(alias, expected):
    assert cli._resolve_format(_args("--output", "results.json", "--format", alias)) == expected


@pytest.mark.parametrize(
    ("fmt", "expected_keys"),
    [
        ("json", {"reading_order"}),
        ("xml", {"reading_order", "direction", "file_title"}),
        ("txt", {"direction", "include_furniture"}),
        ("md", {"direction", "escape", "include_furniture"}),
        ("adoc", {"direction", "escape", "include_furniture"}),
        ("html", {"direction", "include_furniture"}),
    ],
)
def test_export_kwargs(fmt, expected_keys):
    kwargs = cli._export_kwargs(fmt, _args("--direction", "ltr", "--no-escape"))

    assert set(kwargs) == expected_keys
    assert kwargs.get("direction", "ltr") == "ltr"
    assert kwargs.get("escape", False) is False


@pytest.mark.parametrize(
    ("num_pages", "expected"),
    [
        (1, ["out.xml"]),
        (3, ["out_page_1.xml", "out_page_2.xml", "out_page_3.xml"]),
    ],
)
def test_xml_paths(num_pages, expected):
    assert cli._xml_paths("out.xml", num_pages) == [Path(path) for path in expected]


# Parity between the CLI and the library it drives


CLI_OPTIONS = set(vars(cli._parse_args(["--input_path", "sample.pdf"])))

# Parameters that are deliberately not exposed by the CLI
NOT_EXPOSED = {
    "self",
    "pretrained",  # the CLI always runs inference on pretrained weights
    "pretrained_backbone",  # the CLI always runs inference on pretrained weights
    "det_predictor",
    "reco_predictor",
    "layout_predictor",
    "table_predictor",  # built internally by the zoo
}


def _signature_params(func):
    return {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
    }


def test_cli_exposes_every_predictor_option():
    """Every knob of the OCR stack should be reachable from the command line"""
    expected = (
        _signature_params(ocr_predictor)
        | _signature_params(_OCRPredictor.__init__)
        | _signature_params(DocumentBuilder.__init__)
    ) - NOT_EXPOSED

    assert expected <= CLI_OPTIONS, f"options missing from the CLI: {sorted(expected - CLI_OPTIONS)}"


def test_cli_supports_every_export_format():
    """`--format` should accept every format handled by `Document.export_as`"""
    source = textwrap.dedent(inspect.getsource(DocumentExportsMixin.export_as))
    supported = {
        key.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert set(cli.FORMAT_ALIASES) == supported
    assert set(cli.FORMAT_ALIASES.values()) == set(cli.SUFFIX_ALIASES.values())


# Predictor building


def test_build_predictor_forwards_every_option(recorded_predictor):
    args = _args(
        *LIGHT_ARCHS,
        "--layout_arch",
        "lw_detr_s",
        "--no-assume_straight_pages",
        "--straighten_pages",
        "--no-preserve_aspect_ratio",
        "--no-symmetric_pad",
        "--det_bs",
        "4",
        "--reco_bs",
        "64",
        "--detect_orientation",
        "--detect_language",
        "--detect_layout",
        "--detect_tables",
        "--export_as_straight_boxes",
        "--preserve_original_coords",
        "--disable_page_orientation",
        "--disable_crop_orientation",
        "--no-resolve_lines",
        "--resolve_blocks",
        "--paragraph_break",
        "0.1",
        "--keep_reading_order",
    )
    model = cli._build_predictor(args)

    # every flag reaches the zoo
    assert recorded_predictor == {
        "det_arch": "db_mobilenet_v3_large",
        "reco_arch": "crnn_mobilenet_v3_small",
        "pretrained": True,
        "assume_straight_pages": False,
        "preserve_aspect_ratio": False,
        "symmetric_pad": False,
        "export_as_straight_boxes": True,
        "detect_orientation": True,
        "straighten_pages": True,
        "detect_language": True,
        "detect_layout": True,
        "layout_arch": "lw_detr_s",
        "ignore_regions": None,
        "detect_tables": True,
        "det_bs": 4,
        "reco_bs": 64,
        "disable_page_orientation": True,
        "disable_crop_orientation": True,
        "preserve_original_coords": True,
        "resolve_lines": False,
        "resolve_blocks": True,
        "paragraph_break": 0.1,
        "keep_reading_order": True,
    }
    assert (model.assume_straight_pages, model.straighten_pages, model.preserve_original_coords) == (False, True, True)
    assert (model.preserve_aspect_ratio, model.symmetric_pad) == (False, False)
    assert (model._page_orientation_disabled, model._crop_orientation_disabled) == (True, True)
    assert model.layout_predictor is not None
    assert model.table_predictor is not None
    assert model.det_predictor.pre_processor.batch_size == 4
    assert model.reco_predictor.pre_processor.batch_size == 64

    builder = model.doc_builder
    assert (builder.resolve_lines, builder.resolve_blocks, builder.paragraph_break) == (False, True, 0.1)
    assert (builder.export_as_straight_boxes, builder.keep_reading_order) == (True, True)


def test_build_predictor_ignore_regions_enables_layout(recorded_predictor):
    model = cli._build_predictor(_args(*LIGHT_ARCHS, "--ignore_regions", "Picture"))

    # the layout model is required to locate the regions to mask out
    assert recorded_predictor["detect_layout"] is True
    assert model.ignore_regions == ["Picture"]
    assert model.layout_predictor is not None


def test_build_predictor_applies_thresholds_and_device():
    model = cli._build_predictor(_args(*LIGHT_ARCHS, "--bin_thresh", "0.4", "--box_thresh", "0.25", "--device", "cpu"))

    postprocessor = model.det_predictor.model.postprocessor
    assert (postprocessor.bin_thresh, postprocessor.box_thresh) == (0.4, 0.25)
    assert next(model.det_predictor.model.parameters()).device == torch.device("cpu")


# Device & detection thresholds


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "meta"])
def test_resolve_device_explicit(device):
    assert cli._resolve_device(device) == torch.device(device)


@pytest.mark.parametrize("value", [None, "auto", "AUTO", " auto "])
@pytest.mark.parametrize("cuda_available", [False, True])
def test_resolve_device_auto(monkeypatch, value, cuda_available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert cli._resolve_device(value) == torch.device("cuda:0" if cuda_available else "cpu")


def test_resolve_device_invalid():
    with pytest.raises(SystemExit):
        cli._resolve_device("not_a_device")


def test_to_device_failure(predictor, caplog):
    args = _args("--device", "cuda:99")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        cli._to_device(predictor, args)

    assert "cuda:99" in caplog.text


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], (None, None)),  # untouched: the thresholds of the architecture are kept
        (["--bin_thresh", "0.6"], (0.6, None)),
        (["--box_thresh", "0.0"], (None, 0.0)),
        (["--bin_thresh", "0.6", "--box_thresh", "1.0"], (0.6, 1.0)),
    ],
)
def test_set_thresholds(predictor, det_postprocessor, argv, expected):
    defaults = (det_postprocessor.bin_thresh, det_postprocessor.box_thresh)
    cli._set_thresholds(predictor, _args(*argv))

    expected = tuple(default if value is None else value for value, default in zip(expected, defaults))
    assert (det_postprocessor.bin_thresh, det_postprocessor.box_thresh) == expected


def test_set_thresholds_without_postprocessor(monkeypatch, predictor, caplog):
    monkeypatch.delattr(predictor.det_predictor.model, "postprocessor")

    with caplog.at_level(logging.WARNING):
        cli._set_thresholds(predictor, _args("--bin_thresh", "0.6"))

    assert "postprocessor" in caplog.text


# Document loading


def test_load_document_concatenates_inputs(text_images, mock_pdf):
    pages = cli._load_document(cli._parse_args(["--input_path", *text_images, mock_pdf]))

    # 2 images + the 2 pages of the mock PDF
    assert len(pages) == 4
    assert all(isinstance(page, np.ndarray) and page.ndim == 3 for page in pages)


def test_load_document_forwards_pdf_options(monkeypatch, mock_pdf):
    recorder = {}

    def _record_from_pdf(file, **kwargs):
        recorder.update(file=file, **kwargs)
        return [np.zeros((10, 10, 3), dtype=np.uint8)]

    monkeypatch.setattr(DocumentFile, "from_pdf", _record_from_pdf)

    args = cli._parse_args(["--input_path", mock_pdf, "--pdf_scale", "4", "--pdf_password", "secret"])
    assert len(cli._load_document(args)) == 1
    assert recorder == {"file": mock_pdf, "scale": 4, "password": "secret"}


def test_load_document_from_url(monkeypatch):
    recorder = {}

    def _record_from_url(url, **kwargs):
        recorder.update(url=url, **kwargs)
        return [np.zeros((10, 10, 3), dtype=np.uint8)]

    monkeypatch.setattr(DocumentFile, "from_url", _record_from_url)

    args = cli._parse_args(["--input_path", "https://www.mindee.com"])
    assert len(cli._load_document(args)) == 1
    assert recorder["url"] == "https://www.mindee.com"


# Exporting the predictions


@pytest.mark.parametrize(
    ("fmt", "filename"),
    [
        (None, "results.json"),
        (None, "results.txt"),
        (None, "results.md"),
        (None, "results.adoc"),
        (None, "results.html"),
        ("json", "results.out"),
        ("dict", "results.out"),
        ("text", "results.out"),
        ("txt", "results.out"),
        ("markdown", "results.out"),
        ("md", "results.out"),
        ("asciidoc", "results.out"),
        ("adoc", "results.out"),
        ("html", "results.out"),
    ],
)
def test_save_results_formats(ocr_result, tmp_path, fmt, filename):
    output_path = tmp_path / filename
    argv = ["--output", str(output_path)] + (["--format", fmt] if fmt is not None else [])
    args = _args(*argv)
    resolved = cli._resolve_format(args)

    cli._save_results(ocr_result, resolved, args)

    written = output_path.read_text(encoding="utf-8")
    expected = ocr_result.export_as(resolved, **cli._export_kwargs(resolved, args))
    if resolved == "json":
        # the export is dumped as-is, so it round-trips through JSON
        assert json.loads(written) == json.loads(json.dumps(expected))
        assert len(json.loads(written)["pages"]) == 2
    else:
        assert written == expected


def test_save_results_xml_one_file_per_page(ocr_result, tmp_path):
    output_path = tmp_path / "results.xml"
    args = _args("--output", str(output_path))

    cli._save_results(ocr_result, "xml", args)

    assert not output_path.exists()
    for idx, (expected, _) in enumerate(ocr_result.export_as_xml(**cli._export_kwargs("xml", args))):
        assert (tmp_path / f"results_page_{idx + 1}.xml").read_bytes() == expected


def test_save_results_json_indent(ocr_result, tmp_path):
    compact, indented = tmp_path / "compact.json", tmp_path / "indented.json"
    cli._save_results(ocr_result, "json", _args("--output", str(compact), "--indent", "0"))
    cli._save_results(ocr_result, "json", _args("--output", str(indented), "--indent", "4"))

    assert json.loads(compact.read_text()) == json.loads(indented.read_text())
    assert compact.stat().st_size < indented.stat().st_size


def test_save_results_unsupported_format(ocr_result, tmp_path):
    # `--format` guards the CLI, but the export itself must still fail loudly
    with pytest.raises(SystemExit):
        cli._save_results(ocr_result, "yaml", _args("--output", str(tmp_path / "results.yaml")))


def test_save_results_output_path_not_a_file(ocr_result, tmp_path):
    with pytest.raises(SystemExit):
        cli._save_results(ocr_result, "json", _args("--output", str(tmp_path)))


def test_save_results_output_path_invalid_directory(ocr_result, tmp_path):
    with pytest.raises(SystemExit):
        cli._save_results(ocr_result, "json", _args("--output", str(tmp_path / "missing" / "results.json")))


def test_save_results_xml_output_path_invalid_directory(ocr_result, tmp_path):
    with pytest.raises(SystemExit):
        cli._save_results(ocr_result, "xml", _args("--output", str(tmp_path / "missing" / "results.xml")))


# End-to-end runs


def test_main_with_image(text_images, tmp_path):
    output_path = tmp_path / "results.json"
    cli.main(["--input_path", text_images[0], "--output", str(output_path)])

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(exported["pages"]) == 1
    assert exported["pages"][0]["dimensions"]


def test_main_with_pdf(mock_pdf, tmp_path):
    output_path = tmp_path / "results.json"
    cli.main(["--input_path", mock_pdf, "--output", str(output_path)])

    assert len(json.loads(output_path.read_text(encoding="utf-8"))["pages"]) == 2


def test_main_with_multiple_inputs_as_markdown(text_images, mock_pdf, tmp_path):
    output_path = tmp_path / "results.md"
    cli.main(["--input_path", *text_images, mock_pdf, "--output", str(output_path), *LIGHT_ARCHS])

    assert output_path.exists()


def test_main_with_hocr(text_images, tmp_path):
    output_path = tmp_path / "results.xml"
    cli.main(["--input_path", *text_images, "--output", str(output_path), "--format", "hocr", *LIGHT_ARCHS])

    assert not output_path.exists()
    for idx in range(len(text_images)):
        # hOCR is an XHTML document, one page per file
        page = (tmp_path / f"results_page_{idx + 1}.xml").read_bytes()
        assert page.startswith(b"<html")
        assert page.count(b'class="ocr_page"') == 1


def test_main_format_flag_overrides_extension(text_images, tmp_path):
    output_path = tmp_path / "results.json"
    cli.main(["--input_path", text_images[0], "--output", str(output_path), "--format", "text", *LIGHT_ARCHS])

    # the explicit format wins over the extension, so this is not JSON despite the `.json` name
    with pytest.raises(json.JSONDecodeError):
        json.loads(output_path.read_text(encoding="utf-8"))


def test_main_quiet(text_images, tmp_path, caplog):
    logger = logging.getLogger()
    previous_level = logger.level
    try:
        cli.main(["--input_path", text_images[0], "--output", str(tmp_path / "results.json"), "--quiet", *LIGHT_ARCHS])
    finally:
        logger.setLevel(previous_level)

    assert all(record.levelno >= logging.ERROR for record in caplog.records)


# Error handling


def test_main_no_input_path():
    with pytest.raises(SystemExit):
        cli.main([])


def test_main_invalid_input_path():
    with pytest.raises(SystemExit):
        cli.main(["--input_path", "non_existent_file.pdf", "--output", "results.json"])


def test_main_unsupported_input_file_format(tmp_path):
    unsupported_file = tmp_path / "unsupported.txt"
    unsupported_file.write_text("This is not a valid image or PDF file.")

    with pytest.raises(SystemExit):
        cli.main(["--input_path", str(unsupported_file), "--output", "results.json"])


def test_main_corrupted_input_file(tmp_path):
    corrupted_pdf = tmp_path / "corrupted.pdf"
    corrupted_pdf.write_text("not a real pdf")

    with pytest.raises(SystemExit):
        cli.main(["--input_path", str(corrupted_pdf), "--output", "results.json"])


def test_main_one_invalid_input_among_several(text_images, tmp_path):
    with pytest.raises(SystemExit):
        cli.main([
            "--input_path",
            text_images[0],
            "non_existent_file.pdf",
            "--output",
            str(tmp_path / "results.json"),
        ])
