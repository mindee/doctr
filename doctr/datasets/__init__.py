from doctr.file_utils import is_tf_available

from .classification import *
from .cord import *
from .detection import *
from .doc_artefacts import *
from .funsd import *
from .ic03 import *
from .iiit5k import *
from .ocr import *
from .recognition import *
from .sroie import *
from .svt import *
from .synthtext import *
from .utils import *
from .vocabs import *

if is_tf_available():
    from .loader import *
