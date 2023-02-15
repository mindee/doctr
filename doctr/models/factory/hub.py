# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Inspired by: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hub.py

import json
import logging
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from huggingface_hub import (  # type: ignore[attr-defined]
    HfApi,
    HfFolder,
    Repository,
    hf_hub_download,
    snapshot_download,
)

from doctr import models
from doctr.file_utils import is_tf_available, is_torch_available

if is_torch_available():
    import torch

__all__ = ["login_to_hub", "push_to_hf_hub", "from_hub", "_save_model_and_config_for_hf_hub"]


AVAILABLE_ARCHS = {
    "classification": models.classification.zoo.ARCHS,
    "detection": models.detection.zoo.ARCHS + models.detection.zoo.ROT_ARCHS,
    "recognition": models.recognition.zoo.ARCHS,
    "obj_detection": ["fasterrcnn_mobilenet_v3_large_fpn"] if is_torch_available() else None,
}


def login_to_hub() -> None:
    """Login to huggingface hub"""
    access_token = HfFolder.get_token()
    if access_token is not None and HfApi()._is_valid_token(access_token):
        logging.info("Huggingface Hub token found and valid")
        HfApi().set_access_token(access_token)
    else:
        subprocess.call(["huggingface-cli", "login"])
        HfApi().set_access_token(HfFolder().get_token())
    # check if git lfs is installed
    try:
        subprocess.call(["git", "lfs", "version"])
    except FileNotFoundError:
        raise OSError(
            "Looks like you do not have git-lfs installed, please install. \
                      You can install from https://git-lfs.github.com/. \
                      Then run `git lfs install` (you only have to do this once)."
        )


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
        weights_path = save_directory / "pytorch_model.bin"
        torch.save(model.state_dict(), weights_path)
    elif is_tf_available():
        weights_path = save_directory / "tf_model" / "weights"
        model.save_weights(str(weights_path))

    config_path = save_directory / "config.json"

    # add model configuration
    model_config = model.cfg
    model_config["arch"] = arch
    model_config["task"] = task

    with config_path.open("w") as f:
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
    run_config = kwargs.get("run_config", None)
    arch = kwargs.get("arch", None)

    if run_config is None and arch is None:
        raise ValueError("run_config or arch must be specified")
    if task not in ["classification", "detection", "recognition", "obj_detection"]:
        raise ValueError("task must be one of classification, detection, recognition, obj_detection")

    # default readme
    readme = textwrap.dedent(
        f"""
    ---
    language: en
    ---

    <p align="center">
    <img src="https://doctr-static.mindee.com/models?id=v0.3.1/Logo_doctr.gif&src=0" width="60%">
    </p>

    **Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 & PyTorch**

    ## Task: {task}

    https://github.com/mindee/doctr

    ### Example usage:

    ```python
    >>> from doctr.io import DocumentFile
    >>> from doctr.models import ocr_predictor, from_hub

    >>> img = DocumentFile.from_images(['<image_path>'])
    >>> # Load your model from the hub
    >>> model = from_hub('mindee/my-model')

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
    """
    )

    # add run configuration to readme if available
    if run_config is not None:
        arch = run_config.arch
        readme += textwrap.dedent(
            f"""### Run Configuration
                                  \n{json.dumps(vars(run_config), indent=2, ensure_ascii=False)}"""
        )

    if arch not in AVAILABLE_ARCHS[task]:  # type: ignore
        raise ValueError(
            f"Architecture: {arch} for task: {task} not found.\
                         \nAvailable architectures: {AVAILABLE_ARCHS}"
        )

    commit_message = f"Add {model_name} model"

    local_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", model_name)
    repo_url = HfApi().create_repo(model_name, token=HfFolder.get_token(), exist_ok=False)
    repo = Repository(local_dir=local_cache_dir, clone_from=repo_url, token=True)

    with repo.commit(commit_message):
        _save_model_and_config_for_hf_hub(model, repo.local_dir, arch=arch, task=task)
        readme_path = Path(repo.local_dir) / "README.md"
        readme_path.write_text(readme)

    repo.git_push()


def from_hub(repo_id: str, **kwargs: Any):
    """Instantiate & load a pretrained model from HF hub.

    >>> from doctr.models import from_hub
    >>> model = from_hub("mindee/fasterrcnn_mobilenet_v3_large_fpn")

    Args:
        repo_id: HuggingFace model hub repo
        kwargs: kwargs of `hf_hub_download` or `snapshot_download`

    Returns:
        Model loaded with the checkpoint
    """

    # Get the config
    with open(hf_hub_download(repo_id, filename="config.json", **kwargs), "rb") as f:
        cfg = json.load(f)

    arch = cfg["arch"]
    task = cfg["task"]
    cfg.pop("arch")
    cfg.pop("task")

    if task == "classification":
        model = models.classification.__dict__[arch](
            pretrained=False, classes=cfg["classes"], num_classes=cfg["num_classes"]
        )
    elif task == "detection":
        model = models.detection.__dict__[arch](pretrained=False)
    elif task == "recognition":
        model = models.recognition.__dict__[arch](pretrained=False, input_shape=cfg["input_shape"], vocab=cfg["vocab"])
    elif task == "obj_detection" and is_torch_available():
        model = models.obj_detection.__dict__[arch](
            pretrained=False,
            image_mean=cfg["mean"],
            image_std=cfg["std"],
            max_size=cfg["input_shape"][-1],
            num_classes=len(cfg["classes"]),
        )

    # update model cfg
    model.cfg = cfg

    # Load checkpoint
    if is_torch_available():
        state_dict = torch.load(hf_hub_download(repo_id, filename="pytorch_model.bin", **kwargs), map_location="cpu")
        model.load_state_dict(state_dict)
    else:  # tf
        repo_path = snapshot_download(repo_id, **kwargs)
        model.load_weights(os.path.join(repo_path, "tf_model", "weights"))

    return model
