.. _ann:

Microscopy Bulk Simple Annotations
==================================

The Microscopy Bulk Simple Annotation IOD is an IOD designed specifically to
store large numbers of similar annotations and measurements from microscopy
images. Annotations of microscopy images typically refer to very large numbers
of cells or cellular structures. Storing these in a Structured Report Document,
with its highly nested structure, would be very inefficient in storage space
and unnecessarily complex and slow to parse. Microscopy Bulk Simple Annotations
("bulk annotations") solve this problem by allowing you to store large number
of similar annotations or measurements in efficient arrays without duplication
of the descriptive metadata.

Each bulk annotation object contains one or more Annotation Groups, each of
which contains a set of graphical annotations, and optionally one or more
numerical measurements relating to those graphical annotations.

Constructing Annotation Groups
------------------------------

An Annotation Group is a set of multiple similar annotations from a microscopy
image. For example, a single annotation group may contain all annotations of
cell nuclei, lymphocytes, or regions of necrosis in the image. In *highdicom*,
an annotation group is represented by a :class:`highdicom.ann.AnnotationGroup`.

Each annotation group contains some required metadata that describes the
contents of the group, as well as some further optional metadata that may
contain further details about the group or the derivation of the annotations it
contains. The required metadata elements include:

* A ``number`` (``int``), an integer number for the group.
* A ``label`` (``str``) giving a human-readable label for the group.
* A ``uid`` (``str`` or :class:`highdicom.UID`) uniquely identifying the group.
* An ``annotated_property_category`` and ``annotated_property_type``
  (:class:`highdicom.sr.CodedConcept`) coded values (see :ref:`coding`)
  describing the category and specific structure that has been annotated.
* A ``graphic_type`` (:class:`highdicom.ann.GraphicTypeValues`) indicating the
  "form" of the annotations. Permissible values are ``"ELLIPSE"``, ``"POINT"``,
  ``"POLYGON"``, ``"RECTANGLE"``, and ``"POLYLINE"``.
* The ``algorithm_type``
  (:class:`highdicom.ann.AnnotationGroupGenerationTypeValues`), the type of the
  algorithm used to generate the annotations (``"MANUAL"``,
  ``"SEMIAUTOMATIC"``, or ``"AUTOMATIC"``).

Further optional metadata may optionally be provided, see the API documentation
for more information.

The actual annotation data is passed to the group as a list of
``numpy.ndarray`` objects, each of shape (*N* x *D*). *N* is the number of
coordinates required for each individual annotation and is determined by the
graphic type (see :class:`highdicom.ann.GraphicType`). *D* is either 2 -- meaning
that the coordinates are expressed as a (Column,Row) pair in image coordinates
-- or 3 -- meaning that the coordinates are expressed as a (X,Y,Z) triple in 3D
frame of reference coordinates.

Here is a simple example of constructing an annotation group:

.. code-block:: python

    from pydicom.sr.codedict import codes
    from pydicom.sr.coding import Code
    import highdicom as hd
    import numpy as np

    # Graphic data containing two nuclei, each represented by a single point
    # expressed in 2D image coordinates
    graphic_data = [
        np.array([[1234.6, 4088.4], [1239.5, 4088.4]]),
        np.array([[1248.7, 4054.9], [1252.4, 4054.9]]),
    ]

    # Nuclei annotations produced by a manual algorithm
    nuclei_group = hd.ann.AnnotationGroup(
        number=1,
        uid=hd.UID(),
        label='nuclei',
        annotated_property_category=codes.SCT.AnatomicalStructure,
        annotated_property_type=Code("84640000", "SCT", "Nucleus"),
        algorithm_type=hd.ann.AnnotationGroupGenerationTypeValues.MANUAL,
        graphic_type=hd.ann.GraphicTypeValues.POINT,
        graphic_data=graphic_data,
    )

Note that including two nuclei would be very unusual in practice: annotations
often number in the thousands or even millions within large whole slide image.

Including Measurements
----------------------

In addition to the coordinates of the annotations themselves, it is also
possible to attach one or more *measurements* corresponding to those
annotations. The measurements are passed as a
:class:`highdicom.ann.Measurements` object, which contains the *name* of the
measurement (as a coded value), the *unit* of the measurement (also as a coded
value) and an array of the measurements themselves (as a ``numpy.ndarray``).

The length of the measurement array for any measurements attached to an
annotation group must match exactly the number of annotations in the group.
Value *i* in the array therefore represents the measurement of annotation *i*.

Here is the above example with an area measurement included:

.. code-block:: python

    from pydicom.sr.codedict import codes
    from pydicom.sr.coding import Code
    import highdicom as hd
    import numpy as np

    # Graphic data containing two nuclei, each represented by a circle
    # A circle is representing by two point: the center then a point on the
    # circumference
    graphic_data = [
        np.array([[1234.6, 4088.4], [1239.5, 4088.4]]),
        np.array([[1248.7, 4054.9], [1252.4, 4054.9]]),
    ]

    # Measurement object representing the areas of each of the two nuclei
    area_measurement = hd.ann.Measurements(
        name=codes.SCT.Area,
        unit=codes.UCUM.SquareMicrometer,
        values=np.array([20.4, 43.8]),
    )

    # Nuclei annotations produced by a manual algorithm
    nuclei_group = hd.ann.AnnotationGroup(
        number=1,
        uid=hd.UID(),
        label='nuclei',
        annotated_property_category=codes.SCT.AnatomicalStructure,
        annotated_property_type=Code("84640000", "SCT", "Nucleus"),
        algorithm_type=hd.ann.AnnotationGroupGenerationTypeValues.MANUAL,
        graphic_type=hd.ann.GraphicTypeValues.POINT,
        graphic_data=graphic_data,
        measurements=[area_measurement],
    )

Constructing MicroscopyBulkSimpleAnnotation Objects
---------------------------------------------------

When you have constructed the annotation groups, you can include them into
a bulk annotation object along with a bit more metadata using the
:class:`highdicom.ann.MicroscopyBulkSimpleAnnotations` constructor. You also
need to pass the image from which the annotations were derived so that
`highdicom` can copy all the patient, study and slide-level metadata:

.. code-block:: python

    from pydicom import dcmread
    import highdicom as hd

    # Load a slide microscopy image from the highdicom test data (if you have
    # cloned the highdicom git repo)
    sm_image = pydicom.dcmread('data/test_files/sm_image.dcm')

    bulk_annotations = hd.ann.MicroscopyBulkSimpleAnnotations(
        source_images=[sm_image],
        annotation_coordinate_type=hd.ann.AnnotationCoordinateTypeValues.SCOORD,
        annotation_groups=[nuclei_group],
        series_instance_uid=hd.UID(),
        series_number=10,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="MGH Pathology",
        manufacturer_model_name="MGH Pathology Manual Annotations",
        software_versions="0.0.1",
        device_serial_number="1234",
        content_description="Nuclei Annotations",
    )

    bulk_annotations.save_as("nuclei_annotations.dcm")

The result is a complete DICOM object that can be written out as a DICOM file,
transmitted over network, etc.

Reading Existing Bulk Annotation Objects
----------------------------------------

Accessing Annotation Groups
---------------------------

Extracting Information From Annotation Groups
---------------------------------------------
