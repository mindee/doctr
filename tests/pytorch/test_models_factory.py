import json
import os
import tempfile

import pytest

from doctr import models
from doctr.models.factory import _save_model_and_config_for_hf_hub, from_hub, push_to_hf_hub


def test_push_to_hf_hub():
    model = models.classification.resnet18(pretrained=False)
    with pytest.raises(ValueError):
        # run_config and/or arch must be specified
        push_to_hf_hub(model, model_name="test", task="classification")
    with pytest.raises(ValueError):
        # task must be one of classification, detection, recognition, obj_detection
        push_to_hf_hub(model, model_name="test", task="invalid_task", arch="mobilenet_v3_small")
    with pytest.raises(ValueError):
        # arch not in available architectures for task
        push_to_hf_hub(model, model_name="test", task="detection", arch="crnn_mobilenet_v3_large")


@pytest.mark.parametrize(
    "arch_name, task_name, dummy_model_id",
    [
        ["vgg16_bn_r", "classification", "Felix92/doctr-dummy-torch-vgg16-bn-r"],
        ["resnet18", "classification", "Felix92/doctr-dummy-torch-resnet18"],
        ["resnet31", "classification", "Felix92/doctr-dummy-torch-resnet31"],
        ["resnet34", "classification", "Felix92/doctr-dummy-torch-resnet34"],
        ["resnet34_wide", "classification", "Felix92/doctr-dummy-torch-resnet34-wide"],
        ["resnet50", "classification", "Felix92/doctr-dummy-torch-resnet50"],
        ["magc_resnet31", "classification", "Felix92/doctr-dummy-torch-magc-resnet31"],
        ["mobilenet_v3_small", "classification", "Felix92/doctr-dummy-torch-mobilenet-v3-small"],
        ["mobilenet_v3_large", "classification", "Felix92/doctr-dummy-torch-mobilenet-v3-large"],
        ["vit_b", "classification", "Felix92/doctr-dummy-torch-vit-b"],
        ["db_resnet34", "detection", "Felix92/doctr-dummy-torch-db-resnet34"],
        ["db_resnet50", "detection", "Felix92/doctr-dummy-torch-db-resnet50"],
        ["db_mobilenet_v3_large", "detection", "Felix92/doctr-dummy-torch-db-mobilenet-v3-large"],
        ["db_resnet50_rotation", "detection", "Felix92/doctr-dummy-torch-db-resnet50-rotation"],
        ["linknet_resnet18", "detection", "Felix92/doctr-dummy-torch-linknet-resnet18"],
        ["linknet_resnet34", "detection", "Felix92/doctr-dummy-torch-linknet-resnet34"],
        ["linknet_resnet50", "detection", "Felix92/doctr-dummy-torch-linknet-resnet50"],
        ["crnn_vgg16_bn", "recognition", "Felix92/doctr-dummy-torch-crnn-vgg16-bn"],
        ["crnn_mobilenet_v3_small", "recognition", "Felix92/doctr-dummy-torch-crnn-mobilenet-v3-small"],
        ["crnn_mobilenet_v3_large", "recognition", "Felix92/doctr-dummy-torch-crnn-mobilenet-v3-large"],
        ["sar_resnet31", "recognition", "Felix92/doctr-dummy-torch-sar-resnet31"],
        ["master", "recognition", "Felix92/doctr-dummy-torch-master"],
        [
            "fasterrcnn_mobilenet_v3_large_fpn",
            "obj_detection",
            "Felix92/doctr-dummy-torch-fasterrcnn-mobilenet-v3-large-fpn",
        ],
    ],
)
def test_models_huggingface_hub(arch_name, task_name, dummy_model_id, tmpdir):
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = models.__dict__[task_name].__dict__[arch_name](pretrained=True).eval()

        _save_model_and_config_for_hf_hub(model, arch=arch_name, task=task_name, save_dir=tmp_dir)

        assert hasattr(model, "cfg")
        assert len(os.listdir(tmp_dir)) == 2
        assert os.path.exists(tmp_dir + "/pytorch_model.bin")
        assert os.path.exists(tmp_dir + "/config.json")
        tmp_config = json.load(open(tmp_dir + "/config.json"))
        assert arch_name == tmp_config["arch"]
        assert task_name == tmp_config["task"]
        assert all(key in model.cfg.keys() for key in tmp_config.keys())

        # test from hub
        hub_model = from_hub(repo_id=dummy_model_id)
        assert isinstance(hub_model, type(model))
