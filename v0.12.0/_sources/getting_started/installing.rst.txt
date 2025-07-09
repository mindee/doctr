
************
Installation
************

This library requires `Python <https://www.python.org/downloads/>`_ 3.10 or higher.


Prerequisites
=============

Whichever OS you are running, you will need to install at least TensorFlow or PyTorch. You can refer to their corresponding installation pages to do so:

* `TensorFlow 2 <https://www.tensorflow.org/install/>`_
* `PyTorch <https://pytorch.org/get-started/locally/#start-locally>`_

For MacBooks with M1 chip, you will need some additional packages or specific versions:

* `TensorFlow 2 Metal Plugin <https://developer.apple.com/metal/tensorflow-plugin/>`_
* `PyTorch >= 2.0.0 <https://pytorch.org/get-started/locally/#start-locally>`_

.. warning::

   **TensorFlow Backend Deprecation Notice**

   Using docTR with TensorFlow as a backend is **deprecated** and will be removed in the next major release (v1.0.0).
   We **recommend switching to the PyTorch backend**, which is more actively maintained and supports the latest features and models.
   Alternatively, you can use `OnnxTR <https://github.com/felixdittrich92/OnnxTR>`_, which does **not** require TensorFlow or PyTorch.

   This decision was made based on several considerations:

   - Allows better focus on improving the core library
   - Frees up resources to develop new features faster
   - Enables more targeted optimizations with PyTorch

Via Python Package
==================

Install the last stable release of the package using `pip <https://pip.pypa.io/en/stable/installation/>`_:

.. code:: bash

    pip install python-doctr


We strive towards reducing framework-specific dependencies to a minimum, but some necessary features are developed by third-parties for specific frameworks. To avoid missing some dependencies for a specific framework, you can install specific builds as follows:

.. tabs::

    .. tab:: PyTorch

        .. code:: bash

            pip install "python-doctr[torch]"
            # or with preinstalled packages for visualization & html & contrib module support
            pip install "python-doctr[torch,viz,html,contrib]"

    .. tab:: TensorFlow

        .. code:: bash

            pip install "python-doctr[tf]"
            # or with preinstalled packages for visualization & html & contrib module support
            pip install "python-doctr[tf,viz,html,contib]"

Via Conda (Only for Linux)
==========================

Install the last stable release of the package using `conda <https://docs.conda.io/en/latest/>`_:

.. code:: bash

    conda config --set channel_priority strict
    conda install -c techMindee -c pypdfium2-team -c bblanchon -c defaults -c conda-forge python-doctr


Via Git
=======

Install the library in developer mode:

.. tabs::

    .. tab:: PyTorch

        .. code:: bash

            git clone https://github.com/mindee/doctr.git
            pip install -e doctr/.[torch]

    .. tab:: TensorFlow

        .. code:: bash

            git clone https://github.com/mindee/doctr.git
            pip install -e doctr/.[tf]
