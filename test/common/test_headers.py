from datetime import datetime
from pathlib import Path


def test_headers():

    shebang = ["#!usr/bin/python\n"]
    blank_line = "\n"

    _copyright_str = f"-{datetime.now().year}" if datetime.now().year > 2021 else ""
    copyright_notice = [f"# Copyright (C) 2021{_copyright_str}, Mindee.\n"]
    license_notice = [
        "# This program is licensed under the Apache License version 2.\n",
        "# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.\n"
    ]

    # Define all header options
    headers = [
        shebang + [blank_line] + copyright_notice + [blank_line] + license_notice,
        copyright_notice + [blank_line] + license_notice
    ]

    excluded_files = ["version.py", "__init__.py"]
    invalid_files = []

    # For every python file in the repository
    for source_path in Path(__file__).parent.parent.parent.joinpath('doctr').rglob('*.py'):
        if source_path.name not in excluded_files:
            # Parse header
            header_length = max(len(option) for option in headers)
            current_header = []
            with open(source_path) as f:
                for idx, line in enumerate(f):
                    current_header.append(line)
                    if idx == header_length - 1:
                        break
            # Validate it
            if not any(
                "".join(current_header[:min(len(option), len(current_header))]) == "".join(option)
                for option in headers
            ):
                invalid_files.append(source_path)

    assert len(invalid_files) == 0, f"Invalid header in the following files: {invalid_files}"
