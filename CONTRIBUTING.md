# Contributing to docTR

Everything you need to know to contribute efficiently to the project.



## Codebase structure

- [doctr](https://github.com/mindee/doctr/blob/main/doctr) - The package codebase
- [tests](https://github.com/mindee/doctr/blob/main/tests) - Python unit tests
- [docs](https://github.com/mindee/doctr/blob/main/docs) - Library documentation building
- [scripts](https://github.com/mindee/doctr/blob/main/scripts) - Example scripts
- [references](https://github.com/mindee/doctr/blob/main/references) - Reference training scripts
- [demo](https://github.com/mindee/doctr/blob/main/demo) - Small demo app to showcase docTR capabilities 
- [api](https://github.com/mindee/doctr/blob/main/api) - A minimal template to deploy a REST API with docTR


## Continuous Integration

This project uses the following integrations to ensure proper codebase maintenance:

- [Github Worklow](https://help.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow) - run jobs for package build and coverage
- [Codecov](https://codecov.io/) - reports back coverage results

As a contributor, you will only have to ensure coverage of your code by adding appropriate unit testing of your code.



## Feedback

### Feature requests & bug report

Whether you encountered a problem, or you have a feature suggestion, your input has value and can be used by contributors to reference it in their developments. For this purpose, we advise you to use Github [issues](https://github.com/mindee/doctr/issues). 

First, check whether the topic wasn't already covered in an open / closed issue. If not, feel free to open a new one! When doing so, use issue templates whenever possible and provide enough information for other contributors to jump in.

### Questions

If you are wondering how to do something with docTR, or a more general question, you should consider checking out Github [discussions](https://github.com/mindee/doctr/discussions). See it as a Q&A forum, or the docTR-specific StackOverflow!


## Developing docTR

### Developer mode installation

Install all additional dependencies with the following command:

```shell
pip install -e .[dev]
```

### Commits

- **Code**: ensure to provide docstrings to your Python code. In doing so, please follow [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) so it can ease the process of documentation later.
- **Commit message**: please follow [Udacity guide](http://udacity.github.io/git-styleguide/)


### Unit tests

In order to run the same unit tests as the CI workflows, you can run unittests locally:

```shell
make test
```

### Code quality

To run all quality checks together

```shell
make quality
```

#### Lint verification

To ensure that your incoming PR complies with the lint settings, you need to install [flake8](https://flake8.pycqa.org/en/latest/) and run the following command from the repository's root folder:

```shell
flake8 ./
```
This will read the `.flake8` setting file and let you know whether your commits need some adjustments.

#### Import order

In order to ensure there is a common import order convention, run [isort](https://github.com/PyCQA/isort) as follows:

```shell
isort **/*.py
```
This will reorder the imports of your local files.

#### Annotation typing

Additionally, to catch type-related issues and have a cleaner codebase, annotation typing are expected. After installing [mypy](https://github.com/python/mypy), you can run the verifications as follows:

```shell
mypy --config-file mypy.ini doctr/
```
The `mypy.ini` file will be read to check your typing.

#### Docstring format

To keep a sane docstring structure, if you install [pydocstyle](https://github.com/PyCQA/pydocstyle), you can verify your docstrings as follows:

```shell
pydocstyle doctr/
```
The `.pydocstyle` file will be read to configure this operation.


### Modifying the documentation

In order to check locally your modifications to the documentation:
```shell
make docs-single-version
```
You can now open your local version of the documentation located at `docs/_build/index.html` in your browser


## Let's connect

Should you wish to connect somewhere else than on GitHub, feel free to join us on [Slack](https://join.slack.com/t/mindee-community/shared_invite/zt-uzgmljfl-MotFVfH~IdEZxjp~0zldww), where you will find a `#doctr` channel!
