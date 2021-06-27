from .core import *
from .crnn import *
from .sar import *

from doctr.file_utils import is_tf_available, is_torch_available

if is_tf_available():
    from .zoo import *
    from .master import *
