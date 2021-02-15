# Contributing to DocTR

Everything you need to know to contribute efficiently to the project.



## Codebase structure

- [doctr](https://github.com/mindee/doctr/blob/main/doctr) - The actual doctr library
- [test](https://github.com/mindee/doctr/blob/main/test) - Python unit tests
- [docs](https://github.com/mindee/doctr/blob/main/docs) - Library documentation building


## Continuous Integration

This project uses the following integrations to ensure proper codebase maintenance:

- [Github Worklow](https://help.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow) / [CircleCI](https://circleci.com/) - run jobs for package build and coverage
- [Codecov](https://codecov.io/) - reports back coverage results

As a contributor, you will only have to ensure coverage of your code by adding appropriate unit testing of your code.



## Issues

Use Github [issues](https://github.com/mindee/doctr/issues) for feature requests, or bug reporting. When doing so, use issue templates whenever possible and provide enough information for other contributors to jump in.



## Developing doctr


### Commits

- **Code**: ensure to provide docstrings to your Python code. In doing so, please follow [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) so it can ease the process of documentation later.
- **Commit message**: please follow [Udacity guide](http://udacity.github.io/git-styleguide/)

### Running CI verifications locally

#### Unit tests

In order to run the same unit tests as the CI workflows, you can run unittests locally:

```shell
pytest test/
```

#### Lint verification

To ensure that your incoming PR complies with the lint settings, you need to install [flake8](https://flake8.pycqa.org/en/latest/) and run the following command from the repository's root folder:

```shell
flake8 ./
```
This will read the `.flake8` setting file and let you know whether your commits need some adjustments.

#### Annotation typing

Additionally, to catch type-related issues and have a cleaner codebase, annotation typing are expected. After installing [mypy](https://github.com/python/mypy), you can run the verifications as follows:

```shell
mypy --config-file mypy.ini
```
The `mypy.ini` file will be read to check your typing.


### Modifying the documentation

In order to check locally your modifications to the documentation:
- install doctr
- install dependencies specific to documentation with:
```shell
pip install -r docs/requirements.txt
```
- build the documentation
```shell
sphinx-build docs/source docs/_build -a
```
- you can now open your local version of the documentation located at `docs/_build/index.html` in your browser
