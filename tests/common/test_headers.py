"""Test for python files copyright headers."""

from datetime import datetime
from pathlib import Path


def test_copyright_header():
    copyright_header = "".join(
        [
            f"# Copyright (C) {2021}-{datetime.now().year}, Mindee.\n",
            "# This program is licensed under the Apache License version 2.\n",
            "# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.\n",
        ]
    )
    excluded_files = ["__init__.py"]
    invalid_files = []

    for source_path in Path(__file__).parent.parent.joinpath("src").rglob("*.py"):
        if source_path.name not in excluded_files:
            source_path_content = source_path.read_text()
            if copyright_header not in source_path_content:
                invalid_files.append(source_path)

    assert (
        len(invalid_files) == 0
    ), f"Invalid copyright header in the following files: {invalid_files}"
