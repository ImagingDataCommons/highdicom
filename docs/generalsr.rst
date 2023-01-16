.. _generalsr:

Structured Reports: General Overview
====================================

Structured report documents are DICOM files that contain information derived
from a medical image in a structured and computer-readable way. `Highdicom`
supports structured reports through the :mod:`highdicom.sr` sub-package.

SRs are highly complex, and this page attempts to give a basic introduction
while also describing the implementation within `highdicom`. A more thorough
explanation my be found in:

Content Items
-------------

At their core, structured reports are collections of "content items". Each
content item is a collection of DICOM attributes (a DICOM dataset) that are
intended to convey a single piece of information. Each content item consists of
a "name", which is always a `coded concept <coding.html>`_ describing what
information is being conveyed, and a "value", which actually contains the
information of interest. In a loose analogy, you can think of this as similar
to other sorts of key-value mappings such as Python dictionaries and JSON
documents. There are multiple different types of values (known as "value
types"), and accordingly, there are a number of different types of content
item. The classes representing these content items in `highdicom` are:

- :class:`highdicom.sr.CodeContentItem`: The value is a coded concept.
- :class:`highdicom.sr.CompositeContentItem`: The value is a reference to
  another (composite) DICOM object (for example an image or segmentation
  image).
- :class:`highdicom.sr.ContainerContentItem`: The value is a template container
  containing other content items (more on this later).
- :class:`highdicom.sr.DateContentItem`: The value is a date.
- :class:`highdicom.sr.DateTimeContentItem`: The value is a date and a
  time.
- :class:`highdicom.sr.NumContentItem`: The value is a decimal number.
- :class:`highdicom.sr.PnameContentItem`: The value is a person's name.
- :class:`highdicom.sr.ScoordContentItem`: The value is a (2D) spatial
  coordinate in the image coordinate system.
- :class:`highdicom.sr.Scoord3DContentItem`: The value is a 3D spatial
  coordinate in the frame of reference coordinate system.
- :class:`highdicom.sr.TcoordContentItem`: The value is a temporal coordinate
  defined relative to some start point.
- :class:`highdicom.sr.TextContentItem`: The value is a general string.
- :class:`highdicom.sr.TimeContentItem`: The value is a time.
- :class:`highdicom.sr.WaveformContentItem`: The value is a reference to a
  waveform stored within another DICOM object.
- :class:`highdicom.sr.UIDRefContentItem`: The value is a UID (unique
  identifier).

These classes are all subclasses pf ``pydicom.Dataset`` and you can view and
interact with their attributes as you can with any pydicom dataset.

You can look at the API for each class to see how to construct content items of
each type. Here are some simple examples for the more common types:

.. code-block:: python

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes

    # A code content item expressing that the severity is mild
    mild_item = hd.sr.CodeContentItem(
       name=codes.SCT.Severity,
       value=codes.SCT.Mild,
    )

    # A num content item expressing that the depth is 3.4cm
    depth_item = hd.sr.NumContentItem(
       name=codes.DCM.Depth,
       value=3.4,
       unit=codes.UCUM.cm,
    )

    # A scoord content item expressing a point in 3D space of a particular
    # frame of reference
    region_item = hd.sr.Scoord3DContentItem(
       name=codes.DCM.ImageRegion,
       graphic_type=hd.sr.GraphicTypeValues3D.POINT,
       graphic_data=np.array([[10.6, 2.3, -9.6]]),
       frame_of_reference_uid="1.2.826.0.1.3680043.10.511.3.88131829333631241913772141475338566",
    )

    # A composite content item referencing another image as the source for a
    # segmentation
    source_item = hd.sr.CompositeContentItem(
       name=codes.DCM.SourceImageForSegmentation,
       referenced_sop_class_uid="1.2.840.10008.5.1.4.1.1.2",
       referenced_sop_instance_uid="1.2.826.0.1.3680043.10.511.3.21429265101044966075687084803549517",
    )

Graphic Data Content Items (SCOORD and SCOORD3D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two types of Content Item that are worth discussing in greater detail are the
:class:`highdicom.sr.ScoordContentItem` and
:class:`highdicom.sr.Scoord3DContentItem`. These two types both encode "graphic
data" in the form of points/lines/polygons to allow describing locations of an
image in the report.

Scoord (spatial coordinate) Content Items describe locations in 2D image
coordinates. Image coordinates are decimal numbers with sub-pixel accuracy that
are defined in a coordinate system from (0.0, 0.0) at the top left corner of
the top left pixel of the image and (rows, colums) at the bottom right corner
of the bottom right pixel of the image. I.e. the center of the top left pixel
is at location (0.5, 0.5).

Scoord3D (3D spatial coordinate) Content Items describe locations in the 3D
frame of reference that the corresponding image (or images) are defined within.
The points are expressed in millimeters relative to the origin of the
coordinate system (which is not generally the same as the origin of any
particular image, which is given by the "ImagePositionPatient" or
"ImagePositionSlide" attribute of the image). Points expressed in this way
do not change if the underlying image is resampled.

See the :mod:`highdicom.spatial` module for useful utilities for moving
between these two coordinate systems.

Each of these has a distinct but similar list of graphical objects that can be
represented, defined by the enumerations
:class:`highdicom.sr.GraphicTypeValues` (for Scoord Content Items) and
:class:`highdicom.sr.GraphicTypeValues3D`. These types are:


Graphic Type Values (Scoord):

- ``CIRCLE``
- ``ELLIPSE``
- ``MULTIPOINT``
- ``POINT``
- ``POLYLINE``

Graphic Type 3D Values (Scoord3D):

- ``ELLIPSE``
- ``ELLIPSOID``
- ``MULTIPOINT``
- ``POINT``
- ``POLYLINE``
- ``POLYGON``

`highdicom` uses NumPy NdArrays to pass data into the constructors of the
content items. These arrays should have dimensions (*N*, 2) for Scoord Content
Items and (*N*, 3) for Scoord3D Content Items, where *N* is the number of
points. The permissible number of points depends upon the graphic type. For
example, a ``POINT`` is described by exactly one point, a ``CIRCLE`` is
described by exactly 2 points, and a ``POLYLINE`` may contain 2 or more points.
See the documentation of the relevant enumeration class for specific details on
all graphic types.

Furthermore, `highdicom` will reconstruct the graphic data stored into a
content item into a NumPy array of the correct shape if you use the
`value` property of the content item.

Here are some examples of creating Scoord and Scoord3D Content Items and
accessing their graphic data:

.. code-block:: python

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes

    circle_data = np.array(
        [
            [10.0, 10.0],
            [11.0, 11.0],
        ]
    )
    circle_item = hd.sr.ScoordContentItem(
        name=codes.DCM.ImageRegion,
        graphic_type=hd.sr.GraphicTypeValues.CIRCLE,
        graphic_data=circle_data,
    )
    assert np.array_equal(circle_data, circle_item.value)

    multipoint_data = np.array(
        [
            [100.0, 110.0, -90.0],
            [130.0, 70.0, -80.0],
            [-10.0, 400.0, 80.0],
        ]
    )
    multipoint_item = hd.sr.Scoord3DContentItem(
        name=codes.DCM.ImageRegion,
        graphic_type=hd.sr.GraphicTypeValues3D.MULTIPOINT,
        graphic_data=multipoint_data,
       frame_of_reference_uid="1.2.826.0.1.3680043.10.511.3.88131829333631241913772141475338566",
    )
    assert np.array_equal(multipoint_data, multipoint_item.value)

Nesting of Content Items and Sequences
--------------------------------------

Each content item in an SR document may additionally have an attribute named
"ContentSequence", which is a sequence of other Content Items that are the
children of that Content Item. `Highdicom` has the class
:class:`highdicom.sr.ContentSequence` to encapsulate this behavior.

Using ContentSequences containing further Content Items, whose sequences may in
turn contain further items, and so on, it is possible to build highly nested
structures of content items in a "tree" structure.

When this is done, it is necessary to include a "relationship type" attribute
in each child content item (i.e. all Content Items except the one at the root
of the tree) that encodes the relationship that the child item has with the
parent (the Content Item whose Content Sequence the parent belongs to).

The possible relationship types are defined with the enumeration
:class:`highdicom.sr.RelationshipTypeValues` (see the documentation of that
class for more detail):

- ``CONTAINS``
- ``HAS_ACQ_CONTEXT``
- ``HAS_CONCEPT_MOD``
- ``HAS_OBS_CONTEXT``
- ``HAS_PROPERTIES``
- ``INFERRED_FROM``
- ``SELECTED_FROM``

If you construct Content Items with the relationship type, you can nest
Content Items like this:

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # A measurement derived from an image
    depth_item = hd.sr.NumContentItem(
       name=codes.DCM.Depth,
       value=3.4,
       unit=codes.UCUM.cm,
    )

    # The source image from which the measurement was inferred
    source_item = hd.sr.CompositeContentItem(
       name=codes.DCM.SourceImage,
       referenced_sop_class_uid="1.2.840.10008.5.1.4.1.1.2",
       referenced_sop_instance_uid="1.3.6.1.4.1.5962.1.1.1.1.1.20040119072730.12322",
       relationship_type=hd.sr.RelationshipTypeValues.INFERRED_FROM,
    )

    # A tracking identifier identifying the measurment
    tracking_item = hd.sr.UIDRefContentItem(
       name=codes.DCM.TrackingIdentifier,
       value=hd.UID(),  # a newly generated UID
       relationship_type=hd.sr.RelationshipTypeValues.HAS_OBS_CONTEXT,
    )

    # Nest the source item below the depth item
    depth_item.ContentSequence = [source_item, tracking_item]

Structured Reporting IODs
-------------------------

By nesting Content Items and Content Sequences in this way, you can create a
Structured Report DICOM object. There are many IODs (Information Object
Definitions) for Structured Reports, and `highdicom` currently implements three
of them:

- :class:`highdicom.sr.EnhancedSR` -- ??? It does not support Scoord 3D Content Items,
- :class:`highdicom.sr.ComprehensiveSR` -- ??? It does not support Scoord 3D Content Items.
- :class:`highdicom.sr.Comprehensive3DSR` -- This is the most general form of
  SR, but is relatively new and may not be supported by all systems. It does
  support Scoord 3D Content Items.

The constructors for these classes take a number of parameters specifying the
content of the structured report, the evidence from which it was derived in the
form of a list of ``pydicom.Datasets``, as well as various metadata assocaited
with the report.

The content is provided as the ``content`` parameter, which should be a single
content item representing the "root" of the (potentially) nested structure
containing all Content Items in the report.

Using the depth item constructed above as the root Content Item, we can
create a Structured Report like this (here we use an example dataset from
the highdicom test data):

.. code-block:: python

    # Path to single-frame CT image instance stored as PS3.10 file
    image_dataset = pydicom.dcmread("data/test_files/ct_image.dcm")

    # Create the Structured Report instance
    sr_dataset = hd.sr.Comprehensive3DSR(
        evidence=[image_dataset],
        content=depth_item,
        series_number=1,
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Manufacturer'
    )

Note that this is just a toy example and we do **not** recommend producing SRs
like this in practice. Instead of this arbitrary structure of Content Items, it
is far better to follow an existing **template** that encapsulates a
standardized structure of Content Items.

Structured Reporting Templates
------------------------------

The DICOM standard defines a large number of Structured Reporting templates,
which are essentially sets of constraints on the pattern of Content Items
within a report. Each template is intended for a particular purpose.

*Highdicom* currently implements only the TID1500 "Measurement Report" template
and its many sub-templates. This template is highly flexible and provides a
standardized way to store general measurements and evaluations from one or more
images or image regions (expressed in image or frame of reference coordinates).

The following page gives a detailed overview of how to use the Measurement
Report template within *highdicom*.
