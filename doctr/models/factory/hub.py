# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Inspired by: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hub.py

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import HfApi, HfFolder, Repository

from doctr.file_utils import is_tf_available, is_torch_available

__all__ = ['login_to_hub', 'push_to_hf_hub']


def login_to_hub() -> None:
    """Login to huggingface hub
    """

    access_token = HfFolder.get_token()
    if access_token is not None and HfApi()._is_valid_token(access_token):
        print("Huggingface Hub token found and valid")
        HfApi().set_access_token(access_token)
    else:
        subprocess.call(['huggingface-cli', 'login'])
        HfApi().set_access_token(HfFolder().get_token())
    # check if git lfs is installed
    try:
        subprocess.call(['git', 'lfs', 'version'])
    except FileNotFoundError:
        raise OSError('Looks like you do not have git-lfs installed, please install. \
                      You can install from https://git-lfs.github.com/. \
                      Then run `git lfs install` (you only have to do this once).')


def _save_model_and_config_for_hf_hub(model: Any, save_dir: str, model_config: Dict[str, str]) -> None:
    """Save model and config to disk for pushing to huggingface hub

    Args:
        model: TF or PyTorch model to be saved
        save_dir: directory to save model and config
        model_config: model configuration
    """

    save_directory = Path(save_dir)

    if is_torch_available():
        import torch
        weights_path = save_directory / 'pytorch_model.bin'
        torch.save(model.state_dict(), weights_path)
    elif is_tf_available():
        import tensorflow as tf  # noqa: F401
        weights_path = save_directory / 'tf_model' / 'weights'
        model.save_weights(str(weights_path))
    else:
        raise RuntimeError("Neither PyTorch nor TensorFlow is available.")

    config_path = save_directory / 'config.json'

    with config_path.open('w') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)


def push_to_hf_hub(model: Any, model_name: str, tag: str, run_config: Any) -> None:
    """Save model and its configuration on HF hub

    Args:
        model: TF or PyTorch model to be saved
        model_name: name of the model which is also the repository name
        tag: task name
        run_config: run configuration
    """

    # default readme
    readme = f"""
---
language: en
---

<p align="center">
  <img src="https://github.com/mindee/doctr/releases/download/v0.3.1/Logo_doctr.gif" width="60%">
</p>

**Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 & PyTorch**

## Task: {tag}

https://github.com/mindee/doctr

### Example usage:

```python
>>> from doctr.io import DocumentFile
>>> from doctr.models import ocr_predictor
>>> from doctr.models.<task> import from_hub
>>>
>>> img = DocumentFile.from_images(['<image_path>'])
>>> # Load your model from the hub
>>> model = from_hub('mindee/my-recognition-model').eval()
>>> # Pass it to the predictor
>>> predictor = ocr_predictor(det_arch='db_mobilenet_v3_large',
>>>                           reco_arch=model,
>>>                           pretrained=True)
>>> # Get your predictions
>>> res = predictor(img)
```

### Run Configuration
{json.dumps(vars(run_config))}
"""

    # add model configuration
    model_config = model.cfg
    model_config['arch'] = run_config.arch

    commit_message = f'Add {model_name} model'

    local_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'doctr', 'huggingface_cache', model_name)
    repo_url = HfApi().create_repo(model_name, token=HfFolder.get_token(), exist_ok=False)
    repo = Repository(local_dir=local_cache_dir, clone_from=repo_url, use_auth_token=True)

    with repo.commit(commit_message):

        _save_model_and_config_for_hf_hub(model, repo.local_dir, model_config=model_config)
        readme_path = Path(repo.local_dir) / 'README.md'
        readme_path.write_text(readme)

    repo.git_push()
