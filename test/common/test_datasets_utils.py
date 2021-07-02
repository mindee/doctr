import pytest
import numpy as np
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
    encoded = utils.encode_sequence(input_str, mapping)
    decoded = utils.decode_sequence(np.array(encoded), mapping)
    assert decoded == input_str


@pytest.mark.parametrize(
    "sequences, vocab, target_size, eos, sos, pad, error, out_shape, gts",
    [
        [['cba'], 'abcdef', None, 1, None, None, True, (1, 3), [[2, 1, 0]]],
        [['cba', 'a'], 'abcdef', None, -1, None, None, False, (2, 3), [[2, 1, 0], [0, -1, -1]]],
        [['cba', 'a'], 'abcdef', None, 6, None, None, False, (2, 3), [[2, 1, 0], [0, 6, 6]]],
        [['cba', 'a'], 'abcdef', 2, -1, None, None, False, (2, 2), [[2, 1], [0, -1]]],
        [['cba', 'a'], 'abcdef', 4, -1, None, None, False, (2, 4), [[2, 1, 0, -1], [0, -1, -1, -1]]],
        [['cba', 'a'], 'abcdef', 5, -1, 7, None, False, (2, 5), [[7, 2, 1, 0, -1], [7, 0, -1, -1, -1]]],
        [['cba', 'a'], 'abcdef', None, -1, 7, 9, False, (2, 5), [[7, 2, 1, 0, -1], [7, 0, -1, 9, 9]]],
    ],
)
def test_encode_sequences(sequences, vocab, target_size, eos, sos, pad, error, out_shape, gts):
    if error:
        with pytest.raises(ValueError):
            _ = utils.encode_sequences(sequences, vocab, target_size, eos, sos, pad)
    else:
        out = utils.encode_sequences(sequences, vocab, target_size, eos, sos, pad)
        assert isinstance(out, np.ndarray)
        assert out.shape == out_shape
        assert np.all(out == np.asarray(gts)), print(out, gts)
