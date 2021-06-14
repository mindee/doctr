from .core import *
from .differentiable_binarization import *
from .linknet import *

from doctr.file_utils import is_tf_available, is_torch_available

if is_tf_available():
    from .zoo import *

del linknet
del differentiable_binarization
