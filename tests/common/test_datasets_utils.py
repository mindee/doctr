import numpy as np
import pytest

from doctr.datasets import utils


@pytest.mark.parametrize(
    "input_str, vocab, output_str",
    [
        ["f orêt", "latin", "foret"],
        ["f or êt", "french", "forêt"],
        ["¢¾©téØßřůž", "french", "¢■■té■■ruz"],
        ["Ûæëð", "french", "Û■ë■"],
        ["Ûæë<àð", "latin", "U■e<a■"],
        ["Ûm@læ5€ëð", "currency", "■■■■■■€■■"],
        ["Ûtë3p2ð", "digits", "■■■3■2■"],
    ],
)
def test_translate(input_str, vocab, output_str):
    out = utils.translate(input_str, vocab, unknown_char="■")
    assert out == output_str


def test_translate_unknown_vocab():
    with pytest.raises(KeyError):
        utils.translate("test", "unknown_vocab")


@pytest.mark.parametrize(
    "input_str",
    [
        "frtvorêt",
        "for98€t",
        "uéîUY",
        "ÛAZ$£ë",
    ],
)
def test_encode_decode(input_str):
    mapping = """3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|
        za1ù8,OG€P-kçHëÀÂ2É/ûIJ\'j(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l"""
    encoded = utils.encode_string(input_str, mapping)
    decoded = utils.decode_sequence(encoded, mapping)
    assert decoded == input_str


def test_encode_string_unknown_char():
    with pytest.raises(ValueError):
        utils.encode_string("abc", "xyz")


def test_decode_sequence():
    mapping = "abcdef"
    with pytest.raises(TypeError):
        utils.decode_sequence(123, mapping)
    with pytest.raises(AssertionError):
        utils.decode_sequence(np.array([2, 10]), mapping)
    with pytest.raises(AssertionError):
        utils.decode_sequence(np.array([2, 4.5]), mapping)

    assert utils.decode_sequence([3, 4, 3, 4], mapping) == "dede"


@pytest.mark.parametrize(
    "sequences, vocab, target_size, sos, eos, pad, dynamic_len, error, out_shape, gts",
    [
        [["cba"], "abcdef", None, None, 1, None, False, True, (1, 3), [[2, 1, 0]]],  # eos in vocab
        [["cba"], "abcdef", None, 1, -1, None, False, True, (1, 3), [[2, 1, 0]]],  # sos in vocab
        [["cba"], "abcdef", None, None, -1, 1, False, True, (1, 3), [[2, 1, 0]]],  # pad in vocab
        [["cba", "a"], "abcdef", None, None, -1, None, False, False, (2, 4), [[2, 1, 0, -1], [0, -1, -1, -1]]],
        [["cba", "a"], "abcdef", None, None, 6, None, False, False, (2, 4), [[2, 1, 0, 6], [0, 6, 6, 6]]],
        [["cba", "a"], "abcdef", 2, None, -1, None, False, False, (2, 2), [[2, 1], [0, -1]]],
        [["cba", "a"], "abcdef", 4, None, -1, None, False, False, (2, 4), [[2, 1, 0, -1], [0, -1, -1, -1]]],
        [["cba", "a"], "abcdef", 5, 7, -1, None, False, False, (2, 5), [[7, 2, 1, 0, -1], [7, 0, -1, -1, -1]]],
        [["cba", "a"], "abcdef", 6, 7, -1, None, True, False, (2, 5), [[7, 2, 1, 0, -1], [7, 0, -1, -1, -1]]],
        [["cba", "a"], "abcdef", None, 7, -1, 9, False, False, (2, 6), [[7, 2, 1, 0, -1, 9], [7, 0, -1, 9, 9, 9]]],
    ],
)
def test_encode_sequences(sequences, vocab, target_size, sos, eos, pad, dynamic_len, error, out_shape, gts):
    if error:
        with pytest.raises(ValueError):
            utils.encode_sequences(sequences, vocab, target_size, eos, sos, pad, dynamic_len)
    else:
        out = utils.encode_sequences(sequences, vocab, target_size, eos, sos, pad, dynamic_len)
        assert isinstance(out, np.ndarray)
        assert out.shape == out_shape
        assert np.all(out == np.asarray(gts)), print(out, gts)


# NOTE: main test in test_utils_geometry.py
@pytest.mark.parametrize(
    "target",
    [
        # Boxes
        {"boxes": np.random.rand(3, 4), "labels": ["a", "b", "c"]},
        # Polygons
        {"boxes": np.random.rand(3, 4, 2), "labels": ["a", "b", "c"]},
    ],
)
def test_convert_target_to_relative(target, mock_image_stream):
    img = np.array([[3, 32, 128]])  # ImageTensor
    back_img, target = utils.convert_target_to_relative(img, target)
    assert img.all() == back_img.all()
    assert (target["boxes"].all() >= 0) & (target["boxes"].all() <= 1)


# NOTE: main test in test_utils_geometry.py (extract_rcrops, extract_crops)
@pytest.mark.parametrize(
    "geoms",
    [
        # Boxes
        np.random.randint(low=1, high=20, size=(3, 4)),
        # Polygons
        np.random.randint(low=1, high=20, size=(3, 4, 2)),
    ],
)
def test_crop_bboxes_from_image(geoms, mock_image_path):
    num_crops = 3

    with pytest.raises(ValueError):
        utils.crop_bboxes_from_image(mock_image_path, geoms=np.zeros((3, 1)))

    with pytest.raises(FileNotFoundError):
        utils.crop_bboxes_from_image("123", geoms=np.zeros((2, 4)))

    cropped_imgs = utils.crop_bboxes_from_image(mock_image_path, geoms=geoms)
    # Number of crops
    assert len(cropped_imgs) == num_crops
    # Data type and shape
    assert all(isinstance(crop, np.ndarray) for crop in cropped_imgs)
    assert all(crop.ndim == 3 for crop in cropped_imgs)
