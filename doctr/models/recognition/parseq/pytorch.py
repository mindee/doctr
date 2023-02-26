# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
import torch
from ..core import RecognitionModel

__all__ = ['PARSEQ', 'parseq_large']

default_cfgs = {
    "parseq": {
        "mean": (0.5, 0.5, 0.5),
        "std":(0.5, 0.5, 0.5),
        "input_shape": (3, 32, 128),
    },
}

class PARSEQ(RecognitionModel, torch.nn.Module):
    """Implements parseq model with model from torch hub with decoding
    """
    def __init__(
        self,
        model,
        cfg = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model.eval()
        
    def postprocessor(self, logits):
        preds = logits.softmax(-1)
        preds, confidences = self.model.tokenizer.decode(preds)
        outputs = []
        for pred, conf in zip(preds, confidences):
            outputs.append((pred, conf.cpu().numpy()[:-1].mean()))
        return outputs
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(x)
        return logits
    
def _parseq(
    arch: str,
    pretrained: bool,
    **kwargs,
) -> PARSEQ:
    device = torch.cuda.is_available()
    model = torch.hub.load('baudm/parseq', arch, pretrained=pretrained).eval()
    if device:
        model = model.cuda()

    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])

    _cfg = default_cfgs[arch]
    _cfg["input_shape"] = kwargs["input_shape"]

    # Build the model
    model = PARSEQ(model, cfg=_cfg, **kwargs)
    return model

def parseq_large(pretrained= True):
    """parseq model

    >>> import torch
    >>> from doctr.models import parseq
    >>> model = parseq(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    model = _parseq(arch = "parseq", pretrained=pretrained)
    
    return model
