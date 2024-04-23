# Contribute to Documentation

Please have a look at our [contribution guide](../CONTRIBUTING.md) to see how to install
the development environment and how to generate the documentation.

To install only the `docs` environment, you can do:

```bash
# Make sure you are at the root of the repository before executing these commands
python -m pip install --upgrade pip
pip install -e .[tf,viz,html]  # or .[torch,viz,html]
pip install -e .[docs]
```
