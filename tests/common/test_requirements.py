import re
from pathlib import Path


def test_deps_consistency():

    IGNORE = ["flake8", "isort", "mypy"]
    # Collect the deps from all requirements.txt
    REQ_FILES = ["requirements.txt", "requirements-pt.txt", "tests/requirements.txt", "docs/requirements.txt"]
    folder = Path(__file__).parent.parent.parent.absolute()
    req_deps = {}
    for file in REQ_FILES:
        with open(folder.joinpath(file), 'r') as f:
            _deps = [dep.strip('\n').strip() for dep in f.readlines()]

        for _dep in _deps:
            _split = re.split("[<>=]", _dep)
            lib = _split[0]
            version_constraint = _split[-1] if len(_split) > 1 else ""
            assert req_deps.get(lib, version_constraint) == version_constraint, f"conflicting deps for {lib}"
            req_deps[lib] = version_constraint

    # Collect the one from setup.py
    setup_deps = {}
    with open(folder.joinpath("setup.py"), 'r') as f:
        setup = f.readlines()
    _deps = setup[setup.index("_deps = [\n") + 1:]
    _deps = [_dep.strip() for _dep in _deps[:_deps.index("]\n")]]
    _deps = [_dep.split('"')[1] for _dep in _deps if _dep.startswith('"')]
    for _dep in _deps:
        _split = re.split("[<>=]", _dep)
        lib = _split[0]
        version_constraint = _split[-1] if len(_split) > 1 else ""
        assert setup_deps.get(lib) is None, f"conflicting deps for {lib}"
        setup_deps[lib] = version_constraint

    # Remove ignores
    for k in IGNORE:
        if isinstance(req_deps.get(k), str):
            del req_deps[k]
        if isinstance(setup_deps.get(k), str):
            del setup_deps[k]

    # Compare them
    assert len(req_deps) == len(setup_deps)
    for k, v in setup_deps.items():
        assert isinstance(req_deps.get(k), str)
        assert req_deps[k] == v, f"Mismatch on dependency {k}: {v} from setup.py, {req_deps[k]} from requirements.txt"
