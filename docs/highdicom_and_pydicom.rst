.. _pydicom-and-highdicom:

Highdicom and Pydicom
=====================

The ``highdicom`` library is built on top of ``pydicom``. This page summarizes
the relationship between the two.

Pydicom
-------

`pydicom`_ is a widely-used Python package for working with DICOM files in
Python. It uses a fundamental class, ``pydicom.Dataset``, to represent DICOM
objects within Python programs. It handles operations such as:

- Reading and writing ``Dataset`` objects to/from files.
- Accessing and setting individual attributes on ``Dataset`` objects.
- Decoding pixel data within DICOM files to NumPy arrays, and encoding pixel
  data from NumPy to raw bytes to store within ``Dataset`` objects.


Highdicom
---------

There is a wide variety of DICOM objects defined in the standard, covering many
types of images (X-Ray, CT, MRI, Microscopy, Ophthalmic images) as well as
various types of image-derived information, such as Structured Reports,
Annotations, Presentation States, Segmentations, and Parametric Maps. Formally,
these "types" are known as Information Object Definitions (IODs). Each IOD in
the standard requires different combinations of attributes. For example, the
"Echo Time" attribute exists with the *MRImage* IOD but not within the
*CTImage* IOD. ``pydicom`` represents all of these objects using the same
general ``Dataset`` class, which implements behavior that is common to all
DICOM objects However, it does not attempt to specialize its representation to
implement IOD-specific behavior, leaving this up to the user.

The purpose of ``highdicom`` is to build upon ``pydicom`` to implement specific
behaviors for various IODs to make it easier to correctly create and work with
**specific** types of DICOM object. ``highdicom`` defines sub-classes of
``pydicom.Dataset`` that implement particular IODs, with a specific focus on
IODs that store information derived from other images. For example:

- :class:`highdicom.Image` (this actually covers many IODs)
- :class:`highdicom.seg.Segmentation`
- :class:`highdicom.sr.EnhancedSR`
- :class:`highdicom.sr.ComprehensiveSR`
- :class:`highdicom.sr.Comprehensive3DSR`
- :class:`highdicom.pm.ParametricMap`
- :class:`highdicom.pr.GrayscaleSoftcopyPresentationState`
- :class:`highdicom.pr.PseudoColorSoftcopyPresentationState`
- :class:`highdicom.pr.AdvancedBlendingPresentationState`
- :class:`highdicom.ko.KeyObjectSelectionDocument`
- :class:`highdicom.ann.MicroscopyBulkSimpleAnnotations`
- :class:`highdicom.sc.SCImage`

Since each of these objects are sub-classes of ``pydicom.Dataset``, they all
inherit its behaviors for accessing and setting individual attributes and
writing to file. They should also be interoperable with ``pydicom.Dataset``,
such that they can be used anywhere a ``pydicom.Dataset`` is expected. However
they also have:

- A constructor that dramatically simplifies the creation of the objects while
  ensuring correctness. The constructors guide you through which attributes are
  required and enforce inter-relationships between them required by the
  standard.
- Further methods that allow you to search, filter, and access the information
  within them more easily.

However, some classes within ``highdicom`` are not DICOM objects and as such
are not sub-classes of ``pydicom.Dataset``. Notable examples include
:class:`highdicom.Volume`,
:class:`highdicom.spatial.ImageToReferenceTransformer` (and other similar
objects), :class:`highdicom.io.ImageFileReader`.

.. _`pydicom`: https://pydicom.github.io/pydicom/stable/index.html
