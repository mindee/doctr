# Contribute to Documentation

## Local Setup
The current documentation is built using Continuous Integration (CI). You can also build the documentation locally following these steps:

```bash
# Make sure you are at the root of the repository before executing these commands
python -m pip install --upgrade pip
pip install -e .[tf]
pip install -e .[docs]
```

Now you can build the documentation by running:
```bash
make html
```

Afterwards, you can view the documentation by opening `_build/html/index.html` in your web browser.

## Modifying Documentation

The DocTR documentation is built using `sphinx`. If you make changes to a file, you need to run `make html` again. Please note that files that have not been modified will not be rebuilt. If you want to force a complete rebuild, you can delete the `_build` directory. Additionally, you may need to clear your web browser's cache to see the modifications.
