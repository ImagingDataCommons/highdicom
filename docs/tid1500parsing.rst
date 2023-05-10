Parsing Measurement Reports
===========================

In addition to the ability to create TID 1500 Structured Reports, *highdicom*
also includes functionality to help you find and extract information from
existing SR documents in this format.

First you must get the SR dataset into the format of a highdicom class. You
can do this using the ``srread()`` function:
highdicom SR object.

.. code-block:: python

    import highdicom as hd

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document.dcm")

Alternatively, if you already have a ``pydicom.Dataset`` in memory, you can use
the relevant ``from_dataset`` method like this:

.. code-block:: python

    import pydicom
    import highdicom as hd

    sr_dataset = pydicom.dcmread("data/test_files/sr_document.dcm")

    # Use the appropriate class depending on the specific IOD, here it is a
    # Comprehensive3DSR
    sr = hd.sr.Comprehensive3DSR.from_dataset(sr_dataset)

If the Structured Report conforms to the TID 1500 measurement report template,
when you access the ``content`` property, a
:class:`highdicom.sr.MeasurementReport` object will be returned. Otherwise,
a general :class:`highdicom.sr.ContentSequence` object is returned.

The resulting :class:`highdicom.sr.MeasurementReport` object has methods that
allow you to find and access the content of the report conveniently.

Searching For Measurement Groups
--------------------------------

To search for measurement groups, the :class:`highdicom.sr.MeasurementReport`
class has
:meth:`highdicom.sr.MeasurementReport.get_image_measurement_groups`,
:meth:`highdicom.sr.MeasurementReport.get_planar_roi_measurement_groups`, and
:meth:`highdicom.sr.MeasurementReport.get_volumetric_roi_measurement_groups`
methods, each of which returns a list of the measurement groups of the three
different types from the structured SR. You can additionally provide filters
to return only those measurement groups that meet certain criteria.

The available search criteria include: tracking UID, finding type, finding
site, referenced SOP instance UID, and referenced SOP class UID. If you provide
multiple criteria, the methods return those groups that meet *all* the
specified criteria.

The returned objects are of type
:class:`highdicom.sr.MeasurementsAndQualitativeEvaluations`,
:class:`highdicom.sr.PlanarROIMeasurementsAndQualitativeEvaluations`, or
:class:`highdicom.sr.VolumetricROIMeasurementsAndQualitativeEvaluations`,
respectively, representing the entire sub-template in the SR content tree.

Here are just some examples of using these methods to find
measurement groups of interest within a measurement report. As an example
SR document, we use the SR document created on the previous page (see
:ref:`tid1500_full_example` for the relevant snippet).

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # Get a list of all image measurement groups referencing an image with a
    # particular SOP Instance UID
    groups = sr.content.get_image_measurement_groups(
        referenced_sop_instance_uid="1.3.6.1.4.1.5962.1.1.1.1.1.20040119072730.12322",
    )
    assert len(groups) == 1

    # Get a list of all image measurement groups with a particular tracking UID
    groups = sr.content.get_image_measurement_groups(
        tracking_uid="1.2.826.0.1.3680043.10.511.3.77718622501224431322963356892468048",
    )
    assert len(groups) == 1

    # Get a list of all planar ROI measurement groups with finding type "Nodule"
    # AND finding site "Lung"
    groups = sr.content.get_planar_roi_measurement_groups(
        finding_type=codes.SCT.Nodule,
        finding_site=codes.SCT.Lung,
    )
    assert len(groups) == 1

    # Get a list of all volumetric ROI measurement groups (with no filters)
    groups = sr.content.get_volumetric_roi_measurement_groups()
    assert len(groups) == 1

Additionally for
:meth:`highdicom.sr.MeasurementReport.get_planar_roi_measurement_groups`, and
:meth:`highdicom.sr.MeasurementReport.get_volumetric_roi_measurement_groups` it
is possible to filter by graphic type and reference type (how the ROI is
specified in the measurement group).

To search by graphic type, pass an instance of either the
:class:`highdicom.sr.GraphicTypeValues` or
:class:`highdicom.sr.GraphicTypeValues3D` enums:

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # Get a list of all planar ROI measurement groups with graphic type CIRCLE
    groups = sr.content.get_planar_roi_measurement_groups(
        graphic_type=hd.sr.GraphicTypeValues.CIRCLE,
    )
    assert len(groups) == 1

For reference type, you should provide one of the following values (which
reflect how the SR document stores the information internally):

- ``CodedConcept(value="111030", meaning="Image Region", scheme_designator="DCM")``
  aka ``pydicom.sr.codedict.codes.DCM.ImageRegion`` for ROIs defined in the SR
  as image regions (vector coordinates for planar regions defined within the
  SR document).
- ``CodedConcept(value="121231", meaning="Volume Surface", scheme_designator="DCM")``
  aka ``pydicom.sr.codedict.codes.DCM.VolumeSurface`` for ROIs defined in the
  SR as a volume surface (vector coordinates for a volumetric region defined
  within the SR document).
- ``CodedConcept(value="121191", meaning="Referenced Segment", scheme_designator="DCM")``
  aka ``pydicom.sr.codedict.codes.DCM.ReferencedSegment`` for ROIs defined in the
  SR indirectly by referencing a segment stored in a DICOM Segmentation Image.
- ``CodedConcept(value="121191", meaning="Region In Space", scheme_designator="DCM")``
  For ROIs defined in the SR indirectly by referencing a region stored in a
  DICOM RT Struct object (this is not currently supported by the highdicom
  constructor, but is an option in the standard). Unfortunately this code is
  not including in ``pydicom.sr.codedict.codes`` at this time.

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # Get a list of all planar ROI measurement groups stored as regions
    groups = sr.content.get_planar_roi_measurement_groups(
        reference_type=codes.DCM.ImageRegion,
    )
    assert len(groups) == 2

    # Get a list of all volumetric ROI measurement groups stored as volume
    # surfaces
    groups = sr.content.get_volumetric_roi_measurement_groups(
        reference_type=codes.DCM.VolumeSurface,
    )
    assert len(groups) == 1


Accessing Data in Measurement Groups
------------------------------------

Once you have found measurement groups, there are various properties on the
returned object that allow you to access the information that you may need.
These may be in the form of basic Python within the measurement group's content
items, or highdicom classes representing full sub-templates that in turn have
methods and properties defined on them. These classes are the same classes that
you use to construct the objects.

The following example demonstrates some examples, see the API documentation
of the relevant class for a full list.

.. code-block:: python

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # Use the first (only) image measurement group as an example
    group = sr.content.get_image_measurement_groups()[0]

    # tracking_identifier returns a Python str
    assert group.tracking_identifier == "Image0001"

    # tracking_uid returns a hd.UID, a subclass of str
    assert group.tracking_uid == "1.2.826.0.1.3680043.10.511.3.77718622501224431322963356892468048"

    # source_images returns a list of hd.sr.SourceImageForMeasurementGroup, which
    # in turn have some properties to access data
    assert isinstance(group.source_images[0], hd.sr.SourceImageForMeasurementGroup)
    assert group.source_images[0].referenced_sop_instance_uid == "1.3.6.1.4.1.5962.1.1.1.1.1.20040119072730.12322" 

    # for the various optional pieces of information in a measurement, accessing
    # the relevant property returns None if the information is not present
    assert group.finding_type is None

    # Now use the first planar ROI group as a second example
    group = sr.content.get_planar_roi_measurement_groups()[0]

    # finding_type returns a CodedConcept
    assert group.finding_type == codes.SCT.Nodule

    # finding_sites returns a list of hd.sr.FindingSite objects (a sub-template)
    assert isinstance(group.finding_sites[0], hd.sr.FindingSite)
    # the value of a finding site is a CodedConcept
    assert group.finding_sites[0].value == codes.SCT.Lung

    # reference_type returns a CodedConcept (the same values used above for
    # filtering)
    assert group.reference_type == codes.DCM.ImageRegion

    # since this has reference type ImageRegion, we can access the referenced roi
    # using 'roi', which will return an hd.sr.ImageRegion object
    assert isinstance(group.roi, hd.sr.ImageRegion)

    # the graphic type and actual ROI coordinates (as a numpy array) can be
    # accessed with the graphic_type and value properties of the roi
    assert group.roi.graphic_type == hd.sr.GraphicTypeValues.CIRCLE
    assert isinstance(group.roi.value, np.ndarray)
    assert group.roi.value.shape == (2, 2)

A volumetric group returns a :class:`highdicom.sr.VolumeSurface` or list of
:class:`highdicom.sr.ImageRegion` objects, depending on the reference type. If
instead, a planar/volumetric measurement group uses the ``ReferencedSegment``
reference type, the referenced segment can be accessed by the
``group.referenced_segmention_frame`` property (for planar groups) or
``group.referenced_segment`` property (for volumetric groups), which return
objects of type :class:`highdicom.sr.ReferencedSegmentationFrame` and
:class:`highdicom.sr.ReferencedSegment` respectively.

Searching for Measurements
--------------------------

Each measurement group may optionally contain any number of "measurements",
represented by the TID300 "Measurement" template and the
:class:`highdicom.sr.Measurement` class that implements it in *highdicom*.
A measurement contains a numerical measurement derived from the image, along
with the physical unit of the measurement and various other optional
descriptive metadata 

You can search for measurements within a measurements group using the
``get_measurements()`` method on the relevant measurement group class. You can
optionally provide a ``name`` parameter, which should be a coded value that
allows you to find measurements with a particular name.

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # Use the same example file in the highdicom test data
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # Use the first planar measurement group as an example
    group = sr.content.get_planar_roi_measurement_groups()[0]

    # Get a list of all measurements
    measurements = group.get_measurements()

    # Get a list of measurements for diameter
    measurements = group.get_measurements(name=codes.SCT.Diameter)


Note that although there will usually be only a single measurement with a given
name within a measurement group, this is not disallowed by the standard.
Consequently, the ``get_measurements()`` method returns a list containing 0
or more measurements.

Accessing Data in Measurements
------------------------------

You can access the name of a measurement with the `name` property (returns a
:class:`highdicom.sr.CodedConcept`, its numerical value with the `value`
property (returns a `float`), and the unit with the `unit` property.

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # Use the same example file in the highdicom test data
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # Use the first planar measurement group as an example
    group = sr.content.get_planar_roi_measurement_groups()[0]

    # Get the diameter measurement in this group
    measurement = group.get_measurements(name=codes.SCT.Diameter)[0]

    # Access the measurement's name
    assert measurement.name == codes.SCT.Diameter

    # Access the measurement's value
    assert measurement.value == 10.0

    # Access the measurement's unit
    assert measurement.unit == codes.UCUM.mm

Additionally, the properties `method`, `finding_sites`, `qualifier`,
`referenced_images`, and `derivation` allow you to access further optional
metadata that may be present in the stored measurement.

Searching for Evaluations
-------------------------

In addition to numerical measurements, measurement groups may also contain
"qualitative evaluations". These contain an evaluation of the image represented
using a coded concept.

Similar to measurements, you can search for evaluations with the
`get_qualitative_evaluations()` method. You can optionally filter by name with
the ``name`` parameter. You can access the name and value of the returned
evaluations with the ``name`` and ``value`` properties.

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # Use the same example file in the highdicom test data
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # Use the first planar measurement group as an example
    group = sr.content.get_planar_roi_measurement_groups()[0]

    # Get the diameter measurement in this group
    evaluation = group.get_qualitative_evaluations(
        name=codes.DCM.LevelOfSignificance
    )[0]

    # Access the measurement's name
    assert evaluation.name == codes.DCM.LevelOfSignificance

    # Access the measurement's value
    assert evaluation.value == codes.SCT.NotSignificant
