from .differentiable_binarization import *
from .linknet import *
from .zoo import *

from doctr import is_tf_available

if is_tf_available():
    from .core import *