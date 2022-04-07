# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import json
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from doctr.models import obj_detection

__all__ = ['from_hub']


def from_hub(repo_id: str, **kwargs: Any) -> torch.nn.Module:
    """Instantiate & load a pretrained model from HF hub.

    >>> from doctr.models.obj_detection import from_hub
    >>> model = from_hub("mindee/fasterrcnn_mobilenet_v3_large_fpn").eval()
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        repo_id: HuggingFace model hub repo
        kwargs: kwargs of `hf_hub_download`

    Returns:
        Model loaded with the checkpoint
    """

    # Get the config
    with open(hf_hub_download(repo_id, filename='config.json', **kwargs), 'rb') as f:
        cfg = json.load(f)

    model = obj_detection.__dict__[cfg['arch']](
        pretrained=False,
        image_mean=cfg['mean'],
        image_std=cfg['std'],
        max_size=cfg['input_shape'][-1],
        num_classes=len(cfg['classes']),
    )

    # Load the checkpoint
    state_dict = torch.load(hf_hub_download(repo_id, filename='pytorch_model.bin', **kwargs), map_location='cpu')
    model.load_state_dict(state_dict)
    model.cfg = cfg

    return model
