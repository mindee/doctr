from doctr.datasets.vocabs import LANGUAGES, VOCABS


def test_vocab_languages():
    for language in LANGUAGES:
        assert language in VOCABS.keys()
