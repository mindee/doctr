from .core import *
from .crnn import *
from .master import *

from doctr.file_utils import is_tf_available, is_torch_available

if is_tf_available():
    from .sar import *
    from .zoo import *
