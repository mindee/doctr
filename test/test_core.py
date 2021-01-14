import doctr


def test_version():
    assert len(doctr.__version__.split('.')) == 3
