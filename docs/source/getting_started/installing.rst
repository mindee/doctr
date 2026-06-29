
************
Installation
************

This library requires `Python <https://www.python.org/downloads/>`_ 3.10 or higher.


Via Python Package
==================

Install the last stable release of the package using `pip <https://pip.pypa.io/en/stable/installation/>`_:

.. code:: bash

    pip install python-doctr

We strive towards reducing framework-specific dependencies to a minimum, but some necessary features are developed by third-parties for specific frameworks. To avoid missing some dependencies for a specific framework, you can install specific builds as follows:

.. code:: bash

    pip install python-doctr
    # or with preinstalled packages for visualization & html & contrib module support
    pip install "python-doctr[viz,html,contrib]"

Available optional extras:

* ``viz``: installs `matplotlib` and `mplcursors` for result visualisation (e.g. ``Page.show()``)
* ``html``: installs `weasyprint` for reading HTML documents via :func:`~doctr.io.read_html`
* ``contrib``: installs `onnxruntime` for the :class:`~doctr.contrib.ArtefactDetector` contrib module


Via Git
=======

Install the library in developer mode:


.. code:: bash

    git clone https://github.com/mindee/doctr.git
    pip install -e doctr/.


Via Docker
==========

Official Docker images are available on the `GitHub Container Registry <https://github.com/mindee/doctr/pkgs/container/doctr>`_.

.. code:: bash

    docker run -it ghcr.io/mindee/doctr:latest bash
