from doctr.file_utils import is_tf_available

from .generator import *
from .cord import *
from .detection import *
from .doc_artefacts import *
from .funsd import *
from .ic03 import *
from .ic13 import *
from .iiit5k import *
from .iiithws import *
from .imgur5k import *
from .mjsynth import *
from .ocr import *
from .recognition import *
from .sroie import *
from .svhn import *
from .svt import *
from .synthtext import *
from .utils import *
from .vocabs import *
from .wildreceipt import *

if is_tf_available():
    from .loader import *
