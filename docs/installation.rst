.. _installation-guide:

Installation guide
==================

.. _requirements:

Requirements
------------

* `Python <https://www.python.org/>`_ (version 3.6 or higher)
* Python package manager `pip <https://pip.pypa.io/en/stable/>`_

.. _installation:

Installation
------------

Pre-build package available at PyPi:

.. code-block:: none

    pip install highdicom

The library relies on the underlying ``pydicom`` package for decoding of pixel data, which internally delegates the task to either the ``pillow`` or the ``pylibjpeg`` packages.
Since the ``pillow`` is a dependency of *highdicom* and will automatically be installed, some transfer syntax can thus be readily decoded and encoded (baseline JPEG, JPEG-2000, JPEG-LS).
Support for additional transfer syntaxes (e.g., lossless JPEG) requires installation of the ``pylibjpeg`` package as well as the ``pylibjpeg-libjpeg`` and ``pylibjpeg-openjpeg`` packages.
Since ``pylibjpeg-libjpeg`` is licensed under a copyleft GPL v3 license, it is not installed by default when you install *highdicom*. To install the ``pylibjpeg`` packages along with *highdicom*, use

.. code-block:: none

    pip install highdicom[libjpeg]

Install directly from source code (available on Github):

.. code-block:: none

    git clone https://github.com/herrmannlab/highdicom ~/highdicom
    pip install ~/highdicom

