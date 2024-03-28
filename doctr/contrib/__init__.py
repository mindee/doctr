import logging

try:
    from .artefacts import ArtefactDetector
except ImportError:
    logging.warning("onnxruntime is not installed, some features may not work")
    pass
