from pathlib import Path

import pytest

import doctr.cli.main as cli


def test_parse_args_defaults():
    args = cli._parse_args(["--input_path", "sample.pdf"])

    assert args.input_path == "sample.pdf"
    assert args.det_arch == "db_resnet50"
    assert args.reco_arch == "crnn_vgg16_bn"
    assert args.assume_straight_pages is True
    assert args.preserve_aspect_ratio is True
    assert args.symmetric_pad is False
    assert args.det_bs == 2
    assert args.reco_bs == 128
    assert args.detect_orientation is False
    assert args.detect_language is False


def test_parse_args_boolean_optional_flags():
    args = cli._parse_args([
        "--input_path",
        "sample.pdf",
        "--no-assume_straight_pages",
        "--no-preserve_aspect_ratio",
    ])

    assert args.assume_straight_pages is False
    assert args.preserve_aspect_ratio is False


def test_parse_args_requires_input_path():
    with pytest.raises(SystemExit):
        cli._parse_args([])


def test_parse_args_custom_values():
    args = cli._parse_args([
        "--input_path",
        "sample.pdf",
        "--det_arch",
        "custom_det",
        "--reco_arch",
        "custom_reco",
        "--symmetric_pad",
        "--detect_orientation",
        "--detect_language",
        "--output",
        "output.json",
    ])

    assert args.input_path == "sample.pdf"
    assert args.det_arch == "custom_det"
    assert args.reco_arch == "custom_reco"
    assert args.symmetric_pad is True
    assert args.detect_orientation is True
    assert args.detect_language is True
    assert args.output == "output.json"


def test_main_with_image(mock_image_path):
    output_path = "results.json"
    cli.main(["--input_path", mock_image_path, "--output", output_path])

    assert Path(output_path).exists()


def test_main_with_pdf(mock_pdf):
    output_path = "results.json"
    cli.main(["--input_path", mock_pdf, "--output", output_path])

    assert Path(output_path).exists()


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


def test_main_output_path_not_a_file(mock_image_path):
    with pytest.raises(SystemExit):
        cli.main(["--input_path", mock_image_path, "--output", "."])


def test_main_output_path_invalid_directory(mock_image_path):
    with pytest.raises(SystemExit):
        cli.main(["--input_path", mock_image_path, "--output", "non_existent_dir/results.json"])
