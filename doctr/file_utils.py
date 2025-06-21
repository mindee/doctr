# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import importlib.metadata
import logging

__all__ = ["requires_package", "CLASS_NAME"]

CLASS_NAME: str = "words"
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def requires_package(name: str, extra_message: str | None = None) -> None:  # pragma: no cover
    """
    package requirement helper

    Args:
        name: name of the package
        extra_message: additional message to display if the package is not found
    """
    try:
        _pkg_version = importlib.metadata.version(name)
        logging.info(f"{name} version {_pkg_version} available.")
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            f"\n\n{extra_message if extra_message is not None else ''} "
            f"\nPlease install it with the following command: pip install {name}\n"
        )
