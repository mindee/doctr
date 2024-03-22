try:
    import onnxruntime
except ImportError:
    raise ImportError("onnxruntime is not installed. You must install onnxruntime to use this feature.")

from .artefacts import ArtefactDetector
