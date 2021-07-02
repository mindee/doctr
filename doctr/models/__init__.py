from .preprocessor import *
from .core import *
from . import artefacts
from . import utils
from ._utils import *
from .backbones import *
from .detection import *
from .recognition import *
from .zoo import *

from doctr.file_utils import is_tf_available

if is_tf_available():
    from .export import *
