from .preprocessor import *
from .detection import *

from doctr.file_utils import is_tf_available

if is_tf_available():
    from .backbones import *
    from .recognition import *
    from . import utils
    from ._utils import *
    from .core import *
    from .export import *
    from .zoo import *
