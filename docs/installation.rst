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

Like the underlying ``pydicom`` package, highdicom relies on functionality
implemented in the ``pylibjpeg-libjpeg``
`package <https://pypi.org/project/pylibjpeg-libjpeg/>`_ for the decoding of
DICOM images with certain transfer syntaxes. Since ``pylibjpeg-libjpeg`` is
licensed under a copyleft GPL v3 license, it is not installed by default when
you install highdicom. To install ``pylibjpeg-libjpeg`` along with highdicom,
use

.. code-block:: none

    pip install highdicom[libjpeg]

Install directly from source code (available on Github):

.. code-block:: none

    git clone https://github.com/herrmannlab/highdicom ~/highdicom
    pip install ~/highdicom

