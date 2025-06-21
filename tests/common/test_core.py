import pytest

import doctr
from doctr.file_utils import requires_package


def test_version():
    assert len(doctr.__version__.split(".")) == 3


def test_requires_package():
    requires_package("numpy")  # available
    with pytest.raises(ImportError):  # not available
        requires_package("non_existent_package")
