from doctr.file_utils import is_tf_available, is_triton_available
from .base import *

if is_tf_available():
    from .tensorflow import *
    from .cv2_fallback import *
else:
    from .pytorch import *
    import torch
    if is_triton_available:
        from .pytorch_compile import *
    else:
        from .cv2_fallback import *
    del torch