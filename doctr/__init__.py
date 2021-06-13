from .file_utils import *
from .version import __version__  # noqa: F401

if is_tf_available():
    from . import datasets, documents, models, transforms, utils
elif is_torch_available():
    from . import documents, models, utils
