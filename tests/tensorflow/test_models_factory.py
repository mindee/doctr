import json
import os
import tempfile

import pytest
import tensorflow as tf

from doctr.models import classification, detection, obj_detection, recognition
from doctr.models.factory import _save_model_and_config_for_hf_hub, push_to_hf_hub


def test_push_to_hf_hub():
    model = classification.resnet18(pretrained=False)
    with pytest.raises(ValueError):
        # run_config and/or arch must be specified
        push_to_hf_hub(model, model_name='test', task='classification')
    with pytest.raises(ValueError):
        # task must be one of classification, detection, recognition, obj_detection
        push_to_hf_hub(model, model_name='test', task='invalid_task', arch='mobilenet_v3_small')


@pytest.mark.parametrize(
    "arch_name, task_name",
    [
        ["vgg16_bn_r", "classification"],
        ["resnet18", "classification"],
        ["resnet31", "classification"],
        ["resnet34", "classification"],
        ["resnet34_wide", "classification"],
        ["resnet50", "classification"],
        ["magc_resnet31", "classification"],
        ["mobilenet_v3_small", "classification"],
        ["mobilenet_v3_large", "classification"],
        ["db_resnet50", "detection"],
        ["db_mobilenet_v3_large", "detection"],
        ["linknet_resnet18", "detection"],
        ["linknet_resnet18_rotation", "detection"],
        ["linknet_resnet34", "detection"],
        ["linknet_resnet50", "detection"],
        ["crnn_vgg16_bn", "recognition"],
        ["crnn_mobilenet_v3_small", "recognition"],
        ["crnn_mobilenet_v3_large", "recognition"],
        ["sar_resnet31", "recognition"],
        ["master", "recognition"],
    ],
)
def test_models_for_hub(arch_name, task_name, tmpdir):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf.keras.backend.clear_session()
        if task_name == "classification":
            model = classification.__dict__[arch_name](pretrained=True)
        elif task_name == "detection":
            model = detection.__dict__[arch_name](pretrained=True)
        elif task_name == "recognition":
            model = recognition.__dict__[arch_name](pretrained=True)
        elif task_name == "obj_detection":
            model = obj_detection.__dict__[arch_name](pretrained=True)

        _save_model_and_config_for_hf_hub(model, arch=arch_name, task=task_name, save_dir=tmp_dir)

        assert hasattr(model, "cfg")
        assert len(os.listdir(tmp_dir)) == 2
        assert os.path.exists(tmp_dir + "/tf_model")
        assert len(os.listdir(tmp_dir + "/tf_model")) == 3
        assert os.path.exists(tmp_dir + "/config.json")
        tmp_config = json.load(open(tmp_dir + "/config.json"))
        assert arch_name == tmp_config['arch']
        assert task_name == tmp_config['task']
        assert all(key in model.cfg.keys() for key in tmp_config.keys())
