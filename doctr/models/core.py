# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Any

from doctr.utils.repr import NestedObject

__all__ = ["BaseModel"]


class BaseModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
