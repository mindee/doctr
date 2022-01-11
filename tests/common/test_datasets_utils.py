import numpy as np
import pytest

from doctr.datasets import utils


@pytest.mark.parametrize(
    "input_str, vocab, output_str",
    [
        ['f orêt', 'latin', 'foret'],
        ['f or êt', 'french', 'forêt'],
        ['¢¾©téØßřůž', 'french', '¢■■té■■ruz'],
        ['Ûæëð', 'french', 'Û■ë■'],
        ['Ûæë<àð', 'latin', 'U■e<a■'],
        ['Ûm@læ5€ëð', 'currency', '■■■■■■€■■'],
        ['Ûtë3p2ð', 'digits', '■■■3■2■'],
    ],
)
def test_translate(input_str, vocab, output_str):
    out = utils.translate(input_str, vocab, unknown_char='■')
    assert out == output_str


@pytest.mark.parametrize(
    "input_str",
    [
        'frtvorêt',
        'for98€t',
        'uéîUY',
        'ÛAZ$£ë',
    ],
)
def test_encode_decode(input_str):
    mapping = """3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|
        za1ù8,OG€P-kçHëÀÂ2É/ûIJ\'j(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l"""
    encoded = utils.encode_string(input_str, mapping)
    decoded = utils.decode_sequence(encoded, mapping)
    assert decoded == input_str


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
        [['cba'], 'abcdef', None, None, 1, None, False, True, (1, 3), [[2, 1, 0]]],  # eos in vocab
        [['cba', 'a'], 'abcdef', None, None, -1, None, False, False, (2, 4), [[2, 1, 0, -1], [0, -1, -1, -1]]],
        [['cba', 'a'], 'abcdef', None, None, 6, None, False, False, (2, 4), [[2, 1, 0, 6], [0, 6, 6, 6]]],
        [['cba', 'a'], 'abcdef', 2, None, -1, None, False, False, (2, 2), [[2, 1], [0, -1]]],
        [['cba', 'a'], 'abcdef', 4, None, -1, None, False, False, (2, 4), [[2, 1, 0, -1], [0, -1, -1, -1]]],
        [['cba', 'a'], 'abcdef', 5, 7, -1, None, False, False, (2, 5), [[7, 2, 1, 0, -1], [7, 0, -1, -1, -1]]],
        [['cba', 'a'], 'abcdef', 6, 7, -1, None, True, False, (2, 5), [[7, 2, 1, 0, -1], [7, 0, -1, -1, -1]]],
        [['cba', 'a'], 'abcdef', None, 7, -1, 9, False, False, (2, 6), [[7, 2, 1, 0, -1, 9], [7, 0, -1, 9, 9, 9]]],
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
