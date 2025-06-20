
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


Via Git
=======

Install the library in developer mode:


.. code:: bash

    git clone https://github.com/mindee/doctr.git
    pip install -e doctr/.
