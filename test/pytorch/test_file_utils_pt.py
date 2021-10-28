from doctr.file_utils import is_torch_available


def test_file_utils():
    assert is_torch_available()
