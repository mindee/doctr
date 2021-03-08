import pytest
from doctr import datasets


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
    out = datasets.translate(input_str, vocab)
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
    encoded = datasets.encode_sequence(input_str, mapping)
    decoded = datasets.decode_sequence(encoded, mapping)
    assert decoded == input_str
