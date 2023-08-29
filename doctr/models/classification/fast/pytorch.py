# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch.nn as nn

from doctr.datasets import VOCABS

from ...utils import 	load_pretrained_params

from doctr.models.utils.pytorch import conv_sequence_pt

__all__ = ["textnet_tiny", "textnet_small", "textnet_base"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "textnet_tiny": {
        #"mean": (0.694, 0.695, 0.693),
        #"std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "textnet_small": {
        #"mean": (0.694, 0.695, 0.693),
        #"std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "textnet_base": {
        #"mean": (0.694, 0.695, 0.693),
        #"std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
}


class TextNet(nn.Module):
    """Implements a TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
     <https://arxiv.org/abs/2111.02394>>`_.

    Args:
        num_blocks: number of resnet block in each stage
        output_channels: number of channels in each stage
        stage_conv: whether to add a conv_sequence after each stage
        stage_pooling: pooling to add after each stage (if None, no pooling)
        origin_stem: whether to use the orginal ResNet stem or ResNet-31's
        stem_channels: number of output channels of the stem convolutions
        attn_module: attention module to use in each stage
        include_top: whether the classifier head should be instantiated
        num_classes: number of output classes
    """

    def __init__(
        self,
        stage1: Dict[Any],
        stage2: Dict[Any],
        stage3: Dict[Any],
        stage4: Dict[Any],
        include_top: bool = True,
        num_classes: int = 1000,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
    
        super(TextNet, self).__init__()
        
        _layers: List[nn.Module]        
        self.first_conv = nn.ModuleList[conv_sequence(in_channels, out_channels, True, True, kernel_size=kernel_size, stride=stride)]

        _layers.extend([self.first_conv ])
        for stage in [stage1, stage2, stage3, stage4]:
	        stage_ = nn.ModuleList([RepConvLayer(in_channels, out_channels, kernel_size, stride) for in_channels,out_channels,kernel_size,stride in stage])
	        _layers.extend([stage_])
        
        if include_top:
            _layers.extend(
                [
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(output_channels[-1], num_classes, bias=True),
                ]
            )

        super().__init__(*_layers)
        self.cfg = cfg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def _textnet(
    arch: str,
    pretrained: bool,
    arch_fn,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> TextNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = arch_fn(**kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def textnet_tiny(pretrained: bool = False, **kwargs: Any) -> TVResNet:
    """TextNet architecture as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation",
    <https://arxiv.org/abs/2111.02394>`_.

    >>> import torch
    >>> from doctr.models import textnet_tiny
    >>> model = textnet_tiny(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNet model
    """

    return _textnet(
        "textnet_tiny",
        pretrained,
        TextNet,
        stage1 = [ {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2,
                   {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1}],
                   
        stage2 = [ {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1}],
                   
        stage3 = [ {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},],
                   
        stage4 = [ {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 3], "stride": 1}],
                   
        ignore_keys=["fc.weight", "fc.bias"],
        **kwargs,
    )
    
def textnet_small(pretrained: bool = False, **kwargs: Any) -> TVResNet:
    """TextNet architecture as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation",
    <https://arxiv.org/abs/2111.02394>`_.

    >>> import torch
    >>> from doctr.models import textnet_small
    >>> model = textnet_small(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNet model
    """

    return _textnet(
        "textnet_small",
        pretrained,
        TextNet,
        stage1 = [ {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2}],
                   
        stage2 = [ {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},],
                   
        stage3 = [ {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},],
                   
        stage4 = [ {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},],
        ignore_keys=["fc.weight", "fc.bias"],
        **kwargs,
    )
    
def textnet_base(pretrained: bool = False, **kwargs: Any) -> TVResNet:
    """TextNet architecture as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation",
    <https://arxiv.org/abs/2111.02394>`_.

    >>> import torch
    >>> from doctr.models import textnet_base
    >>> model = textnet_base(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNet model
    """

    return _textnet(
        "textnet_base",
        pretrained,
        TextNet,
        stage1 = [ {"kernel_size": [3, 3], "stride": 1},
                   {"kernel_size": [3, 3], "stride": 2},
                   {"kernel_size": [3, 1], "stride": 1},
                   {"kernel_size": [3, 3], "stride": 1},
                   {"kernel_size": [3, 1], "stride": 1},
                   {"kernel_size": [3, 3], "stride": 1},
                   {"kernel_size": [3, 3], "stride": 1},
                   {"kernel_size": [1, 3], "stride": 1},
                   {"kernel_size": [3, 3], "stride": 1},
                   {"kernel_size": [3, 3], "stride": 1}],

        stage2 = [ {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},],

        stage3 = [ {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
                   {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},],
                   
        stage4 = [ {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
                   {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},]
        ignore_keys=["fc.weight", "fc.bias"],
        **kwargs,
    )
