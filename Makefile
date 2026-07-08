.PHONY: quality style test test-common test-tf test-torch docs-single-version docs
# this target runs checks on all files
quality:
	ruff check .
	mypy doctr/

# this target runs checks on all files and potentially modifies some of them
style:
	ruff format .
	ruff check --fix .

# Run tests for the library
test:
	pytest tests/common/ -rs --cov
	pytest tests/pytorch/ -rs --cov --cov-append
	coverage report --fail-under=80 --show-missing

test-common:
	pytest tests/common/ -rs --cov

test-torch:
	pytest tests/pytorch/ -rs --cov

# Check that docs can build
docs-single-version:
	sphinx-build docs/source docs/_build -a

# Check that docs can build
docs:
	cd docs && bash build.sh
