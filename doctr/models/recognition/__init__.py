from .core import *
from .crnn import *

from doctr.file_utils import is_tf_available, is_torch_available

if is_tf_available():
    from .sar import *
    from .zoo import *
    from .transformer import *
    from .master import *
