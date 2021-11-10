from pathlib import Path

import requirements
from requirements.requirement import Requirement


def test_deps_consistency():

    IGNORE = ["flake8", "isort", "mypy", "importlib_metadata", "tensorflow-cpu"]
    # Collect the deps from all requirements.txt
    REQ_FILES = ["requirements.txt", "requirements-pt.txt", "tests/requirements.txt", "docs/requirements.txt"]
    folder = Path(__file__).parent.parent.parent.absolute()
    req_deps = {}
    for file in REQ_FILES:
        with open(folder.joinpath(file), 'r') as f:
            _deps = [(req.name, req.specs) for req in requirements.parse(f)]

        for _dep in _deps:
            lib, specs = _dep
            assert req_deps.get(lib, specs) == specs, f"conflicting deps for {lib}"
            req_deps[lib] = specs

    # Collect the one from setup.py
    setup_deps = {}
    with open(folder.joinpath("setup.py"), 'r') as f:
        setup = f.readlines()
    lines = setup[setup.index("_deps = [\n") + 1:]
    lines = [_dep.strip() for _dep in lines[:lines.index("]\n")]]
    lines = [_dep.split('"')[1] for _dep in lines if _dep.startswith('"')]
    _reqs = [Requirement.parse(_line) for _line in lines]
    _deps = [(req.name, req.specs) for req in _reqs]
    for _dep in _deps:
        lib, specs = _dep
        assert setup_deps.get(lib) is None, f"conflicting deps for {lib}"
        setup_deps[lib] = specs

    # Remove ignores
    for k in IGNORE:
        if isinstance(req_deps.get(k), list):
            del req_deps[k]
        if isinstance(setup_deps.get(k), list):
            del setup_deps[k]

    # Compare them
    assert len(req_deps) == len(setup_deps)
    for k, v in setup_deps.items():
        assert isinstance(req_deps.get(k), list)
        assert req_deps[k] == v, f"Mismatch on dependency {k}: {v} from setup.py, {req_deps[k]} from requirements.txt"
