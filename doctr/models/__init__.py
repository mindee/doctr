from doctr.file_utils import is_tf_available, is_torch_available

if is_tf_available():
    from .preprocessor import *
    from .backbones import *
    from .detection import *
    from .recognition import *
    from . import utils
    from ._utils import *
    from .core import *
    from .export import *
    from .zoo import *
elif is_torch_available():
    from detection import *
