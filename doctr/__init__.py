from .file_utils import is_tf_available, is_torch_available
from .version import __version__  # noqa: F401
from . import documents, models, transforms, utils

if is_tf_available():
    from . import datasets
