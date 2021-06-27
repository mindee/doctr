from .preprocessor import *
from .detection import *
from . import artefacts
from . import utils
from ._utils import *
from .backbones import *
from .recognition import *


from doctr.file_utils import is_tf_available

if is_tf_available():
    from .core import *
    from .export import *
    from .zoo import *
