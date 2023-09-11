
************
Installation
************

This library requires `Python <https://www.python.org/downloads/>`_ 3.6 or higher.


Prerequisites
=============

Whichever OS you are running, you will need to install at least TensorFlow or PyTorch. You can refer to their corresponding installation pages to do so:

* `TensorFlow 2 <https://www.tensorflow.org/install/>`_
* `PyTorch <https://pytorch.org/get-started/locally/#start-locally>`_

For MacBooks with M1 chip, you will need some additional packages or specific versions:

* `TensorFlow 2 Metal Plugin <https://developer.apple.com/metal/tensorflow-plugin/>`_
* `PyTorch >= 1.12.0 <https://pytorch.org/get-started/locally/#start-locally>`_

If you are running another OS than Linux, you will need a few extra dependencies.

For MacOS users, you can install them using `Homebrew <https://brew.sh/>`_ as follows:

.. code:: shell

    brew install cairo pango gdk-pixbuf libffi

For Windows users, those dependencies are included in GTK. You can find the latest installer over `here <https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases>`_.


Via Python Package
==================

Install the last stable release of the package using `pip <https://pip.pypa.io/en/stable/installation/>`_:

.. code:: bash

    pip install python-doctr


We strive towards reducing framework-specific dependencies to a minimum, but some necessary features are developed by third-parties for specific frameworks. To avoid missing some dependencies for a specific framework, you can install specific builds as follows:

.. tabs::

    .. tab:: TensorFlow

        .. code:: bash

            pip install "python-doctr[tf]"

    .. tab:: PyTorch

        .. code:: bash

            pip install "python-doctr[torch]"


Via Git
=======

Install the library in developper mode:

.. tabs::

    .. tab:: TensorFlow

        .. code:: bash

            git clone https://github.com/mindee/doctr.git
            pip install -e doctr/.[tf]

    .. tab:: PyTorch

        .. code:: bash

            git clone https://github.com/mindee/doctr.git
            pip install -e doctr/.[torch]
