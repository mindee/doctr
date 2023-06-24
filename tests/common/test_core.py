import pytest

import doctr


def test_version():
    assert len(doctr.__version__.split(".")) == 3


@pytest.mark.xfail
def test_is_tf_available():
    assert doctr.is_tf_available()


@pytest.mark.xfail
def test_is_torch_available():
    assert not doctr.is_torch_available()
