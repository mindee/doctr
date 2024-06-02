from doctr.file_utils import is_tf_available
from .base import *

if is_tf_available():
    from .tensorflow import *
else:
    from .pytorch import *
