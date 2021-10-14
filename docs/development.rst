.. _developer-guide:

Developer guide
===============

Source code is available at Github and can be cloned via git:

.. code-block:: none

    git clone https://github.com/mghcomputationalpathology/highdicom ~/highdicom

The :mod:`highdicom` package can be installed in *develop* mode for local development:

.. code-block:: none

    pip install -e ~/highdicom


.. _pull-requests:

Pull requests
-------------

Don't commit code changes to the ``master`` branch. New features should be implemented in a separate branch called ``feature/*`` and bug fixes should be applied in separate branch called ``bugfix/*``.

Before creating a pull request on Github, read the coding style guideline, run the tests and check PEP8 compliance.

.. _coding-style:

Coding style
------------

Code must comply with `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.
The `flake8 <http://flake8.pycqa.org/en/latest/>`_ package is used to enforce compliance.

The project uses `numpydoc <https://github.com/numpy/numpydoc/>`_ for documenting code according to `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_ docstring conventions.
Further information and examples for the NumPy style can be found at the `NumPy Github repository <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ and the website of the `Napoleon sphinx extension <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_.

All API classes, functions and modules must be documented (including "private" functions and methods).
Each docstring must describe input parameters and return values.
Types must be specified using type hints as specified by `PEP 484 <https://www.python.org/dev/peps/pep-0484/>`_ (see `typing <https://docs.python.org/3/library/typing.html>`_ module) in both the function definition as well as the docstring.


.. _running-tests:

Running tests
-------------

The project uses `pytest <http://doc.pytest.org/en/latest/>`_ to write and runs unit tests.
Tests should be placed in a separate ``tests`` folder within the package root folder.
Files containing actual test code should follow the pattern ``test_*.py``.

Install requirements:

.. code-block:: none

    pip install -r ~/highdicom/requirements_test.txt

Run tests (including checks for PEP8 compliance):

.. code-block:: none

    cd ~/highdicom
    pytest --flake8

.. _building-documentation:

Building documentation
----------------------

Install requirements:

.. code-block:: none

    pip install -r ~/highdicom/requirements_docs.txt

Build documentation in *HTML* format:

.. code-block:: none

    cd ~/highdicom
    sphinx-build -b html docs/ docs/build/

The built ``index.html`` file will be located in ``docs/build``.

Design principles
-----------------

**Interoperability with Pydicom** - Highdicom is built on the pydicom library.
Highdicom types are typically derived from the ``pydicom.dataset.Dataset`` or
``pydicom.sequence.Sequence`` classes and should remain interoperable with them
as far as possible such that experienced users can use the lower-level pydicom
API to inspect or change the object if needed.

**Standard DICOM Terminology** - Where possible, highdicom types, functions,
parameters, enums, etc map onto concepts within the DICOM standard and should
follow the same terminology to ensure that the meaning is unambiguous. Where
the terminology used in the standard may not be easily understood by those
unfamiliar with it, this should be addressed via documentation rather than
using alternative terminology.

**Standard Compliance on Encoding** - Highdicom should not allow users to
create DICOM objects that are not in compliance with the standard. The library
should validate all parameters passed to it and should raise an exception if
they would result in the creation of an invalid object, and give a clear
explanation to the user why the parameters passed are invalid. Furthermore,
highdicom objects should always exist in a state of standards compliance,
without any intermediate invalid states. Once a constructor has completed, the
user should be confident that they have a valid object.

**Standard Compliance on Decoding** - Unfortunately, many DICOM objects found
in the real world have minor deviations from the standard. When decoding DICOM
objects, highdicom should tolerate minor deviations as far as they do not
interfere with its functionality. When highdicom needs to assume that objects
are standard compliant in order to function, it should check this assumption
first and raise an exception explaining the issue to the user if it finds an
error. Unless there are exceptional circumstances, highdicom should not attempt
to work around issues in non-compliant files produced by other implementations.

**The Decoding API** - Highdicom classes implement functionality for
conveniently accessing information contained within the relevant dataset. To
use this functionality with existing pydicom dataset, such as those read in
from file or received over network, the dataset must first be converted to the
relevant highdicom type.  This is implemented by the alternative
``from_dataset()`` or ``from_sequence()`` constructors on highdicom types.
These methods should perform "eager" type conversion of the dataset and all
datasets contained within it into the relevant highdicom types, where they
exist. This way, objects created from scratch by users and those converted from
pydicom datasets using ``from_dataset()`` or ``from_sequence()`` should appear
identical to users and developers as far as possible.
