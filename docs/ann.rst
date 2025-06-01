.. _ann:

Microscopy Bulk Simple Annotation (ANN) Objects
===============================================

The Microscopy Bulk Simple Annotation IOD is an IOD designed specifically to
store large numbers of similar annotations and measurements from microscopy
images. Annotations of microscopy images typically refer to very large numbers
of cells or cellular structures. Storing these in a Structured Report Document,
with its highly nested structure, would be very inefficient in storage space
and unnecessarily complex and slow to parse. Microscopy Bulk Simple Annotation
objects ("bulk annotations") solve this problem by allowing you to store large
number of similar annotations or measurements in efficient arrays without
duplication of the descriptive metadata.

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
  Usually, you will want to generate a UID for this.
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

When considering which type of coordinate to use, bear in mind that the 2D image
coordinates refer only to one image in a image pyramid, whereas 3D frame of
reference coordinates are more easily used with any image in the pyramid.
Also note that although you can include multiple annotation groups in a single
bulk annotation object, they must all use the same coordinate type.

Here is a simple example of constructing an annotation group:

.. code-block:: python

    from pydicom.sr.codedict import codes
    from pydicom.sr.coding import Code
    import highdicom as hd
    import numpy as np

    # Graphic data containing two nuclei, each represented by a single point
    # expressed in 2D image coordinates
    graphic_data = [
        np.array([[34.6, 18.4]]),
        np.array([[28.7, 34.9]]),
    ]

    # Nuclei annotations produced by a manual algorithm
    nuclei_group = hd.ann.AnnotationGroup(
        number=1,
        uid=hd.UID(),
        label='nuclei',
        annotated_property_category=codes.SCT.AnatomicalStructure,
        annotated_property_type=Code('84640000', 'SCT', 'Nucleus'),
        algorithm_type=hd.ann.AnnotationGroupGenerationTypeValues.MANUAL,
        graphic_type=hd.ann.GraphicTypeValues.POINT,
        graphic_data=graphic_data,
    )

Note that including two nuclei would be very unusual in practice: annotations
often number in the thousands or even millions within a large whole slide image.

Including Measurements
----------------------

In addition to the coordinates of the annotations themselves, it is also
possible to attach one or more continuous-valued numeric *measurements*
corresponding to those annotations. The measurements are passed as a
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

    # Graphic data containing two nuclei, each represented by a single point
    # expressed in 2D image coordinates
    graphic_data = [
        np.array([[34.6, 18.4]]),
        np.array([[28.7, 34.9]]),
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
        annotated_property_type=Code('84640000', 'SCT', 'Nucleus'),
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
    sm_image = dcmread('data/test_files/sm_image.dcm')

    bulk_annotations = hd.ann.MicroscopyBulkSimpleAnnotations(
        source_images=[sm_image],
        annotation_coordinate_type=hd.ann.AnnotationCoordinateTypeValues.SCOORD,
        annotation_groups=[nuclei_group],
        series_instance_uid=hd.UID(),
        series_number=10,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='MGH Pathology',
        manufacturer_model_name='MGH Pathology Manual Annotations',
        software_versions='0.0.1',
        device_serial_number='1234',
        content_description='Nuclei Annotations',
        series_description='Example Microscopy Annotations',
    )

    bulk_annotations.save_as('nuclei_annotations.dcm')

The result is a complete DICOM object that can be written out as a DICOM file,
transmitted over network, etc.

Reading Existing Bulk Annotation Objects
----------------------------------------

You can read an existing bulk annotation object from file using the
:func:`highdicom.ann.annread()` function:

.. code-block:: python

    from pydicom import dcmread
    import highdicom as hd

    ann = hd.ann.annread('data/test_files/sm_annotations.dcm')

    assert isinstance(ann, hd.ann.MicroscopyBulkSimpleAnnotations)

Alternatively you can converting an existing ``pydicom.Dataset`` representing a
bulk annotation object to the `highdicom` object like this:

.. code-block:: python

    from pydicom import dcmread
    import highdicom as hd

    ann_dcm = dcmread('data/test_files/sm_annotations.dcm')

    ann = hd.ann.MicroscopyBulkSimpleAnnotations.from_dataset(ann_dcm)

    assert isinstance(ann, hd.ann.MicroscopyBulkSimpleAnnotations)

Note that these examples (and the following examples) uses an example file that
you can access from the test data in the `highdicom` repository. It was created
using exactly the code in the construction example above.

Accessing Annotation Groups
---------------------------

Usually the next step when working with bulk annotation objects is to find the
relevant annotation groups. You have two ways to do this.

If you know either the number or the UID of the group, you can access the group
directly (since either of these should uniquely identify a group). The
:meth:`highdicom.ann.MicroscopyBulkSimpleAnnotations.get_annotation_group()`
method is used for this purpose:

.. code-block:: python

    # Access a group by its number
    group = ann.get_annotation_group(number=1)
    assert isinstance(group, hd.ann.AnnotationGroup)

    # Access a group by its UID
    group = ann.get_annotation_group(
        uid='1.2.826.0.1.3680043.10.511.3.40670836327971302375623613533993686'
    )
    assert isinstance(group, hd.ann.AnnotationGroup)

Alternatively, you can search for groups that match certain filters such as
the annotation property type or category, label, or graphic type. The
:meth:`highdicom.ann.MicroscopyBulkSimpleAnnotations.get_annotation_groups()`
method (note groups instead of group) is used for this. It returns a list
of matching groups, since the filters may match multiple groups.

.. code-block:: python

    from pydicom.sr.coding import Code

    # Search for groups by annotated property type
    groups = ann.get_annotation_groups(
        annotated_property_type=Code('84640000', 'SCT', 'Nucleus'),
    )
    assert len(groups) == 1 and isinstance(groups[0], hd.ann.AnnotationGroup)

    # If there are no matches, an empty list is returned
    groups = ann.get_annotation_groups(
        annotated_property_type=Code('53982002', "SCT", "Cell membrane"),
    )
    assert len(groups) == 0

    # Search for groups by label
    groups = ann.get_annotation_groups(label='nuclei')
    assert len(groups) == 1 and isinstance(groups[0], hd.ann.AnnotationGroup)

    # Search for groups by label and graphic type together (results must match
    # *all* provided filters)
    groups = ann.get_annotation_groups(
        label='nuclei',
        graphic_type=hd.ann.GraphicTypeValues.POINT,
    )
    assert len(groups) == 1 and isinstance(groups[0], hd.ann.AnnotationGroup)


Extracting Information From Annotation Groups
---------------------------------------------

When you have found a relevant group, you can use the Python properties on
the object to conveniently access metadata and the graphic data of the
annotations. For example (see :class:`highdicom.ann.AnnotationGroup` for a full
list):

.. code-block:: python

    # Access the label
    assert group.label == 'nuclei'

    # Access the number
    assert group.number == 1

    # Access the UID
    assert group.uid == '1.2.826.0.1.3680043.10.511.3.40670836327971302375623613533993686'

    # Access the annotated property type (returns a CodedConcept)
    assert group.annotated_property_type == Code('84640000', 'SCT', 'Nucleus')

    # Access the graphic type, describing the "form" of each annotation
    assert group.graphic_type == hd.ann.GraphicTypeValues.POINT


You can access the entire array of annotations at once using
:meth:`highdicom.ann.AnnotationGroup.get_graphic_data()`. You need to pass the
annotation coordinate type from the parent bulk annotation object to the group
so that it knows how to interpret the coordinate data. This method returns a
list of 2D numpy arrays of shape (*N* x *D*), mirroring how you would have
passed the data in to create the annotation with `highdicom`.

.. code-block:: python

    import numpy as np

    graphic_data = group.get_graphic_data(
        coordinate_type=ann.annotation_coordinate_type,
    )
    assert len(graphic_data) == 2 and isinstance(graphic_data[0], np.ndarray)

Alternatively, you can access the coordinate array for a specific annotation
using its (one-based) index in the annotation list:

.. code-block:: python

    # Get the number of annotations
    assert group.number_of_annotations == 2

    # Access an annotation using 1-based index
    first_annotation = group.get_coordinates(
        annotation_number=1,
        coordinate_type=ann.AnnotationCoordinateType,
    )
    assert np.array_equal(first_annotation, np.array([[34.6, 18.4]]))

Extracting Measurements From Annotation Groups
----------------------------------------------

You can use the :meth:`highdicom.ann.AnnotationGroup.get_measurements()` method
to access any measurements included in the group. By default, this will return
all measurements in the group, but you can also filter for measurements matching
a certain name.

Measurements are returned as a tuple of ``(names, values, units)``, where
``names`` is a list of names as :class:`highdicom.sr.CodedConcept` objects,
``units`` is a list of units also as :class:`highdicom.sr.CodedConcept`
objects, and the values is a ``numpy.ndarray`` of values of shape (*N* by *M*)
where *N* is the number of annotations and *M* is the number of measurements.
This return format is intended to facilitate the loading of measurements into
tables or dataframes for further analysis.


.. code-block:: python

    from pydicom.sr.codedict import codes

    names, values, units = group.get_measurements()
    assert names[0] == codes.SCT.Area
    assert units[0] == codes.UCUM.SquareMicrometer
    assert values.shape == (2, 1)
