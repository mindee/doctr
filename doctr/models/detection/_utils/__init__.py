from doctr.file_utils import is_tf_available

if is_tf_available():
    from .tensorflow import *
else:
    from .pytorch import *
