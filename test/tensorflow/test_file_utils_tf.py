from doctr.file_utils import is_tf_available


def test_file_utils():
    assert is_tf_available()
