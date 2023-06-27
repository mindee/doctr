import pytest

import doctr


def test_version():
    assert len(doctr.__version__.split(".")) == 3


@pytest.mark.skipif(doctr.is_torch_available() and doctr.is_tf_available(), reason="torch and tf are available")
def test_is_tf_available():
    assert doctr.is_tf_available()


@pytest.mark.skipif(doctr.is_torch_available() and doctr.is_tf_available(), reason="torch and tf are available")
def test_is_torch_available():
    assert not doctr.is_torch_available()
