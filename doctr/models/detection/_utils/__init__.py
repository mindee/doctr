from doctr.file_utils import is_tf_available, is_pytorch_backend_available
from .base import *

if is_tf_available():
    from .tensorflow import *
    from .cv2_fallback import *
else:
    from .pytorch import *
    if is_pytorch_backend_available():
        from .pytorch_compile import *
    else:
        from .cv2_fallback import *