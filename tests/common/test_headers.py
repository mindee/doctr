"""Test for python files copyright headers."""

from datetime import datetime
from pathlib import Path


def test_copyright_header():
    copyright_header = "".join(
        [
            f"# Copyright (C) {2021}-{datetime.now().year}, Mindee.\n\n",
            "# This program is licensed under the Apache License 2.0.\n",
            "# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.\n",
        ]
    )
    excluded_files = ["__init__.py", "version.py"]
    invalid_files = []
    locations = [".github", "api/app", "demo", "docs", "doctr", "references", "scripts"]

    for location in locations:
        for source_path in Path(__file__).parent.parent.parent.joinpath(location).rglob("*.py"):
            if source_path.name not in excluded_files:
                source_path_content = source_path.read_text()
                if copyright_header not in source_path_content:
                    invalid_files.append(source_path)
    assert len(invalid_files) == 0, f"Invalid copyright header in the following files: {invalid_files}"
