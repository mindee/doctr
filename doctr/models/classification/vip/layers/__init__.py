from doctr.file_utils import is_torch_available

if is_torch_available():
    from .pytorch import *
