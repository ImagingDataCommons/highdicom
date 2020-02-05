.. _user-guide:

User guide
==========

Creation of derived DICOM objects using the :mod:`highdicom` package.

.. _seg:

Segmentation (SEG) images
-------------------------

.. code-block:: python

    from highdicom.seg.sop import Segmentation


.. _sr:

Structured Reports (SR) documents
---------------------------------

.. code-block:: python

    from highdicom.sr.sop import Comprehensive3DSR


.. _legacy:

Legacy Converted Enhanced Images
--------------------------------

.. code-block:: python

    from highdicom.legacy.sop import LegacyConvertedEnhancedCTImage
