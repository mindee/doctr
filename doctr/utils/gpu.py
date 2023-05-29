import logging
import os
from typing import Tuple
import torch

log = logging.getLogger(__name__)


def select_gpu_device() -> Tuple[str, str]:
    """tries to find either cuda or arm mps gpu accelerator and choses the most appropriate one,
    honoring the environment variables (CUDA_VISIBLE_DEVICES), if any have been set.

    returns tuple(best_detected_device, selected_device)
    best_detected_device reflects capabilities of the system
    selected_device is the device that should be used (might be cpu even if best_detected_device is eg cuda)
    """
    if torch.cuda.is_available():
        detected_gpu_device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        detected_gpu_device = 'mps'
    else:
        detected_gpu_device = 'cpu'

    selected_gpu_device = detected_gpu_device
    match detected_gpu_device:  # various exceptions to the above
        case 'cuda':
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
                selected_gpu_device = 'cpu'
        case 'mps':
            # FIXME detected mps selects cpu here because of the many bugs present in the mps implementation of
            #   torch'es 1.13 LSTM.  As of 5/29/2023, they appear to be actively fixing them.  I did try with torch
            #   2.0.1 and while the bugs look different it's still broken.  Revisit when later versions of torch
            #   are available.
            # pass
            selected_gpu_device = 'cpu'
        case 'cpu':
            pass

    log.info(f"{detected_gpu_device=} {selected_gpu_device=}")
    return detected_gpu_device, selected_gpu_device
