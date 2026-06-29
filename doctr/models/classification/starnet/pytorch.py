# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: architecture adapted from "Rewrite the Stars" (https://github.com/ma-xu/Rewrite-the-Stars) as
# used by TableCenterNet (https://github.com/dreamy-xay/TableCenterNet).

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from doctr.datasets import VOCABS

from ...modules.layers import DropPath
from ...utils import load_pretrained_params

__all__ = ["StarNet", "starnet_s3"]


default_cfgs: dict[str, dict[str, Any]] = {
    "starnet_s3": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://github.com/mindee/doctr/releases/download/v1.0.1/starnet_s3-a413415c.pt",
    },
}


class ConvBN(nn.Sequential):
    """A convolution optionally followed by batch-norm (the conv keeps its bias, as in the reference)."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        with_bn: bool = True,
    ):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            bn = nn.BatchNorm2d(out_planes, momentum=0.1)
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
            self.add_module("bn", bn)


class Block(nn.Module):
    """StarNet block: depth-wise conv, the element-wise "star" multiplication of two linear branches,
    a projection and a residual connection."""

    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return identity + self.drop_path(x)


class StarNet(nn.Sequential):
    """Implements the StarNet architecture from `"Rewrite the Stars" <https://arxiv.org/abs/2403.19967>`_.

    Args:
        base_dim: channel width of the first stage (doubles every stage)
        depths: number of blocks in each of the four stages
        mlp_ratio: expansion ratio of the star-multiplication branches
        drop_path_rate: maximum stochastic-depth rate (linearly scaled across blocks)
        stem_dim: number of channels produced by the stem
        num_classes: number of output classes of the classification head
        include_top: whether to add the classification head
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        base_dim: int = 32,
        depths: tuple[int, int, int, int] = (2, 2, 8, 4),
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        stem_dim: int = 32,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        stem = nn.Sequential(ConvBN(3, stem_dim, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        channels = [stem_dim]
        stages: list[nn.Module] = []
        in_channel = stem_dim
        cur = 0
        for i_layer, depth in enumerate(depths):
            embed_dim = base_dim * 2**i_layer
            channels.append(embed_dim)
            down_sampler = ConvBN(in_channel, embed_dim, 3, 2, 1)
            in_channel = embed_dim
            blocks = [Block(in_channel, mlp_ratio, dpr[cur + i]) for i in range(depth)]
            cur += depth
            stages.append(nn.Sequential(down_sampler, *blocks))

        _layers: list[nn.Module] = [stem, *stages]
        if include_top:
            _layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channel, momentum=0.1),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(in_channel, num_classes),
                )
            )

        super().__init__(*_layers)
        self.channels = channels  # [stem_dim, base_dim, base_dim*2, base_dim*4, base_dim*8]
        self.cfg = cfg

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)


def _starnet(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> StarNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = StarNet(**kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def starnet_s3(pretrained: bool = False, **kwargs: Any) -> StarNet:
    """StarNet-S3 from `"Rewrite the Stars" <https://arxiv.org/abs/2403.19967>`_.

    >>> import torch
    >>> from doctr.models import starnet_s3
    >>> model = starnet_s3(pretrained=False)
    >>> out = model(torch.rand((1, 3, 32, 32), dtype=torch.float32))

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the StarNet architecture

    Returns:
        A StarNet-S3 model
    """
    return _starnet("starnet_s3", pretrained, ignore_keys=["5.3.weight", "5.3.bias"], **kwargs)
