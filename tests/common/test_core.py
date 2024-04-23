import pytest

import doctr
from doctr.file_utils import requires_package


def test_version():
    assert len(doctr.__version__.split(".")) == 3


@pytest.mark.skipif(doctr.is_torch_available() and doctr.is_tf_available(), reason="torch and tf are available")
def test_is_tf_available():
    assert doctr.is_tf_available()


@pytest.mark.skipif(doctr.is_torch_available() and doctr.is_tf_available(), reason="torch and tf are available")
def test_is_torch_available():
    assert not doctr.is_torch_available()


def test_requires_package():
    requires_package("numpy")  # availbable
    with pytest.raises(ImportError):  # not available
        requires_package("non_existent_package")
