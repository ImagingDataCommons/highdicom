.. _remote:

Reading from Remote Filesystems
===============================

Functions like `dcmread` from pydicom and :meth:`highdicom.imread`,
:meth:`highdicom.seg.segread`, :meth:`highdicom.sr.srread`, and
:meth:`highdicom.ann.annread` from highdicom can read from any object that
exposes a "file-like" interface. Many alternative and remote filesystems have
python clients that expose such an interface, and therefore can be read from
directly.

One such example is blobs on Google Cloud Storage buckets when accessed using
the official Python SDK. This is particularly relevant since this is the
storage mechanism underlying the `Imaging Data Commons <IDC>`_ (IDC), a large
repository of public DICOM images.

Coupling this with :ref:`"lazy" frame retrieval <lazy>` option is especially
powerful, allowing frames to be retrieved from the remote filesystem only as
and when they are needed.

In this first example, we use lazy frame retrieval to load only a specific
spatial patch from a large whole slide image from the IDC.

As a further example, we use lazy frame retrieval to load only a specific set
of segments from a large multi-organ segmentation of a CT image in the IDC.

Note that you will need to install the Google Cloud libraries separately to run
these examples. If you can provide examples for reading from storage provided
by other cloud providers, please consider constributing them to this
documentation.

.. _IDC: https://portal.imaging.datacommons.cancer.gov/
