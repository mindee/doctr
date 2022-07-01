from doctr.file_utils import is_tf_available, is_torch_available

if not is_tf_available() and is_torch_available():
    from .pytorch import *
