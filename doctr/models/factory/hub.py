# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Inspired by: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hub.py

import json
import logging
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, HfFolder, Repository

from doctr.file_utils import is_tf_available, is_torch_available

from ..detection import zoo as det_zoo
from ..recognition import zoo as reco_zoo

if is_torch_available():
    import torch

__all__ = ['login_to_hub', 'push_to_hf_hub', '_save_model_and_config_for_hf_hub']


AVAILABLE_ARCHS = {
    'detection': det_zoo.ARCHS + det_zoo.ROT_ARCHS,
    'recognition': reco_zoo.ARCHS,
    'obj_detection': ['fasterrcnn_mobilenet_v3_large_fpn'] if is_torch_available() else None
}


def login_to_hub() -> None:
    """Login to huggingface hub
    """
    access_token = HfFolder.get_token()
    if access_token is not None and HfApi()._is_valid_token(access_token):
        logging.info("Huggingface Hub token found and valid")
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


def _save_model_and_config_for_hf_hub(model: Any, save_dir: str, arch: str, task: str) -> None:
    """Save model and config to disk for pushing to huggingface hub

    Args:
        model: TF or PyTorch model to be saved
        save_dir: directory to save model and config
        arch: architecture name
        task: task name
    """
    save_directory = Path(save_dir)

    if is_torch_available():
        weights_path = save_directory / 'pytorch_model.bin'
        torch.save(model.state_dict(), weights_path)
    elif is_tf_available():
        weights_path = save_directory / 'tf_model' / 'weights'
        model.save_weights(str(weights_path))

    config_path = save_directory / 'config.json'

    # add model configuration
    model_config = model.cfg
    model_config['arch'] = arch
    model_config['task'] = task

    with config_path.open('w') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)


def push_to_hf_hub(model: Any, model_name: str, task: str, **kwargs) -> None:
    """Save model and its configuration on HF hub

    >>> from doctr.models import login_to_hub, push_to_hf_hub
    >>> from doctr.models.recognition import crnn_mobilenet_v3_small
    >>> login_to_hub()
    >>> model = crnn_mobilenet_v3_small(pretrained=True)
    >>> push_to_hf_hub(model, 'my-model', 'recognition', arch='crnn_mobilenet_v3_small')

    Args:
        model: TF or PyTorch model to be saved
        model_name: name of the model which is also the repository name
        task: task name
        **kwargs: keyword arguments for push_to_hf_hub
    """
    run_config = kwargs.get('run_config', None)
    arch = kwargs.get('arch', None)

    if run_config is None and arch is None:
        raise ValueError('run_config or arch must be specified')
    if task not in ['classification', 'detection', 'recognition', 'obj_detection']:
        raise ValueError('task must be one of classification, detection, recognition, obj_detection')

    # default readme
    readme = textwrap.dedent(f"""
    ---
    language: en
    ---

    <p align="center">
    <img src="https://github.com/mindee/doctr/releases/download/v0.3.1/Logo_doctr.gif" width="60%">
    </p>

    **Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 & PyTorch**

    ## Task: {task}

    https://github.com/mindee/doctr

    ### Example usage:

    ```python
    >>> from doctr.io import DocumentFile
    >>> from doctr.models import ocr_predictor
    >>> from doctr.models.<task> import from_hub

    >>> img = DocumentFile.from_images(['<image_path>'])
    >>> # Load your model from the hub
    >>> model = from_hub('mindee/my-model').eval()

    >>> # Pass it to the predictor
    >>> # If your model is a recognition model:
    >>> predictor = ocr_predictor(det_arch='db_mobilenet_v3_large',
    >>>                           reco_arch=model,
    >>>                           pretrained=True)

    >>> # If your model is a detection model:
    >>> predictor = ocr_predictor(det_arch=model,
    >>>                           reco_arch='crnn_mobilenet_v3_small',
    >>>                           pretrained=True)

    >>> # Get your predictions
    >>> res = predictor(img)
    ```
    """)

    # add run configuration to readme if available
    if run_config is not None:
        arch = run_config.arch
        readme += textwrap.dedent(f"""### Run Configuration
                                  \n{json.dumps(vars(run_config), indent=2, ensure_ascii=False)}""")

    if arch not in AVAILABLE_ARCHS[task]:  # type: ignore
        raise ValueError(f'Architecture: {arch} for task: {task} not found.\
                         \nAvailable architectures: {AVAILABLE_ARCHS}')

    commit_message = f'Add {model_name} model'

    local_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub', model_name)
    repo_url = HfApi().create_repo(model_name, token=HfFolder.get_token(), exist_ok=False)
    repo = Repository(local_dir=local_cache_dir, clone_from=repo_url, use_auth_token=True)

    with repo.commit(commit_message):

        _save_model_and_config_for_hf_hub(model, repo.local_dir, arch=arch, task=task)
        readme_path = Path(repo.local_dir) / 'README.md'
        readme_path.write_text(readme)

    repo.git_push()
