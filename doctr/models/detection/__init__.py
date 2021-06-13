from doctr.file_utils import *

if is_tf_available():
    from .core import *
    from .differentiable_binarization import *
    from .linknet import *
    from .zoo import *
elif is_torch_available():
    from .differentiable_binarization_pt import *
