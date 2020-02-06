.. _developer-guide:

Developer guide
===============

Source code is available at Github and can be cloned via git:

.. code-block:: none

    git clone https://github.com/mghcomputationalpathology/highdicom ~/highdicom

The :mod:`dicomweb_client` package can be installed in *develop* mode for local development:

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
