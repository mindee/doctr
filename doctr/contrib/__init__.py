import logging

try:
    from .artefacts import ArtefactDetector
except ImportError:  # pragma: no cover
    logging.warning("onnxruntime is not installed, some features may not work")
