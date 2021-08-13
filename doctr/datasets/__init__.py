from .utils import *
from .vocabs import *
from .funsd import *
from .cord import *
from .detection import *
from .recognition import *
from .ocr import *
from .sroie import *
from .classification import *

from doctr.file_utils import is_tf_available

if is_tf_available():
    from .loader import *
