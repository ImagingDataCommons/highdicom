.. _seg:

DICOM Segmentation Images
=========================

DICOM Segmentation Images (often abbreviated DICOM Seg) are one of the primary
IODs (information objects definitions) implemented in the *highdicom* library.
Seg images store
`segmentations <https://en.wikipedia.org/wiki/Image_segmentation>`_ of other
DICOM images of other modalities, such as magnetic resonance (MR), computed
tomography (CT), slide microscopy (SM) and many others.
A segmentation is a partitioning of an image into different regions. In medical
imaging these regions may commonly represent different organs or tissue types,
or region of abnormality (e.g. tumor or infarct) identified within an image.

The crucial difference between Segs and other IODs that allow for storing image
regions is that Segs store the segmented regions in *raster* format as pixel
arrays as opposed to the *vector* descriptions of the region's boundary used by
structured reports (SRs) and RT structures. This makes them a more natural
choice for many automatic image processing algorithms such as convolutional
neural networks.

The DICOM standard provides a highly flexible object definition for Segmentation
images that is able to cover a large variety of possible use cases.
Unfortunately, this flexibility comes with complexity that can make Segmentation
images difficult to understand and work with.

Segments
--------

Each distinct region of an image represented in a DICOM Seg is known as a
*segment*. For example a single segment could represent an organ (liver, lung,
kideny), tissue (fat, muscle, bone), or abnormality (tumor, infarct).
Elsewhere the same concept is known by other names such as *class* or *label*.

A single DICOM Seg image can represent one or more segments contained within
the same file.

Each segment in a DICOM Seg image is represented by a separate 2D *frame* (or
set of *frames*) within the Segmentation image. One important ramification of
this is that segments need not be *mutually exclusive*, i.e. a given pixel can
belong to at most one segment. In other words, the segments may *overlap*.
There is an optional attribute called "Segments Overlap" (0062, 0013) that, if
present, will indicate whether the segments overlap in a given Seg image.

Segment Descriptions
--------------------

Within a DICOM Seg image, segments are identified by a Segment Number. Segments
are numbered with consecutive segment numbers starting at 1 (i.e., 1, 2, 3,
...).  Additionally, each segment present is accompanied by information
describing what the segment represents. This information is placed in the
"SegmentsSequence" (0062, 0002) attribute of the segmentation file. In
*highdcom*, we use the :class:`highdicom.seg.SegmentDescription` class to hold
this information. When you construct a DICOM Seg image using *highdicom*, you
must construct a single SegmentDescription object for each segment, and provide
the following information:

- **Segment Label**: A human-readable name for the segment (e.g.  ``"Left
  Kidney"``).
- **Segmented Property Category**: A coded value describing the
  category of the segmented region. For example this could specify that the
  segment represents an anatomical structure, a tissue type, or an abnormality.
- **Segmented Property Type**: Another coded concept that more specifically
  describes the segmented region, as for example a kidney or tumor.
- **Algorithm Type**: Whether the segment was produced by an automatic,
  semi-automatic or manual algorithm.
- **Anatomic Regions**: (Optional) The anatomic region which the segment is
  found. For example, if the segmented property type is "tumor", this can be
  used to convey that the tumor is found in the kidney.
- **Tracking ID and UID**: (Optional) This allows you to provide a ID and unique
  ID to a specific segment. This can be used to uniquely identify particular
  lesions over multiple imaging studies, for example.

Notice that the segment description makes use of coded concepts to ensure that
the way a particular anatomical structure is described is standardized and
unambiguous (if standard nomenclatures are used).

Here is an example of constructing a few different segment descriptions using
*highdicom*:

.. code-block:: python

    from pydicom.sr.codedict import codes

    import highdicom as hd


    # Liver segment produced by a manual algorithm
    liver_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='liver',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Liver,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )

    # For the next segment, we will describe the specific algorithm used to
    # create it
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='Auto-Tumor',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Kidney tumor segment produced by the above algorithm
    tumor_description = hd.seg.SegmentDescription(
        segment_number=2,
        segment_label='kidney tumor',
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Tumor,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        anatomic_regions=[codes.SCT.Kidney]
    )

When working with existing Seg images you can use highdicom to search for
segments whose descriptions meet certain criteria. For example:

.. code-block:: python

    from pydicom.sr.codedict import codes

    import highdicom as hd


    # This is a test file in the highdicom git repository
    seg = hd.seg.segread('data/test_files/seg_image_ct_binary_overlap.dcm')

    # Print the number of segments
    print(seg.number_of_segments)  # '2'

    # Print the range of segment numbers
    print(seg.segment_numbers)  # 'range(1, 3)'

    # Search for segments by label (returns segment numbers of all matching
    # segments)
    print(seg.get_segment_numbers(segment_label='first segment'))  # '[1]'
    print(seg.get_segment_numbers(segment_label='second segment'))  # '[2]'

    # Search for segments by segmented property type (returns segment numbers
    # of all matching segments)
    print(seg.get_segment_numbers(segmented_property_type=codes.SCT.Bone))  # '[1]'
    print(seg.get_segment_numbers(segmented_property_type=codes.SCT.Spine))  # '[2]'

    # Search for segments by tracking UID (returns segment numbers of all
    # matching segments)
    print(seg.get_segment_numbers(tracking_uid='1.2.826.0.1.3680043.10.511.3.83271046815894549094043330632275067'))  # '[1]'
    print(seg.get_segment_numbers(tracking_uid='1.2.826.0.1.3680043.10.511.3.10042414969629429693880339016394772'))  # '[2]'

    # You can also get the full description for a given segment, and access
    # the information in it via properties
    segment_1_description = seg.get_segment_description(1)
    print(segment_1_description.segment_label) #  'first segment'
    print(segment_1_description.tracking_uid)  # '1.2.826.0.1.3680043.10.511.3.83271046815894549094043330632275067'


Binary and Fractional Segs
--------------------------

One particularly important characteristic of segmentation images is its
"Segmentation Type" (0062,0001), which may take the value of either ``"BINARY"``
or ``"FRACTIONAL"`` and describes the values that a given segment may take.
Segments in a ``"BINARY"`` segmentation image may only take values 0 or 1, i.e.
each pixel either belongs to the segment or does not.

By contrast, pixels in a ``"FRACTIONAL"`` segmentation image lie in the range 0
to 1. A second attribute, "Segmentation Fractional Type" (0062,0010) specifies
whether these values should be interpreted as ``"PROBABILITY"`` (i.e. the
probability that a pixel belongs to the segmentation) or ``"OCCUPANCY"`` i.e.
the fraction of the volume of the pixel's (or voxel's) area (or volume) that
belongs to the segment.

A potential source of confusion is that having a Segmentations Type of
``"BINARY"`` only limits the range of values *within a given segment*. It is
perfectly valid for a ``"BINARY"`` segmentation to have multiple segments. It
is therefore not the same as the sense of *binary* that distinguishes *binary*
from *multiclass* segmentations.

*Highdicom* provides the Python enumerations
:class:`highdicom.seg.SegmentationTypeValues` and
:class:`highdicom.seg.SegmentationFractionalTypeValues` for the valid values of
the "Segmentation Type" and "Segmentation Fractional Type" attributes,
respectively.

Constructing Basic Binary Seg Images
------------------------------------

We have now covered enough to construct a basic binary segmentation image. We
use the :class:`highdicom.seg.Segmentation` and provide a description of each
segment, a pixel array as a numpy array with an unsigned integer data type, and
some other basic information.

.. code-block:: python

    import numpy as np

    from pydicom.sr.codedict import codes

    import highdicom as hd


    # Description of liver segment produced by a manual algorithm
    liver_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='liver',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Liver,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )

    # Pixel array is an unsigned integer array with 0 and 1 values
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    # Construct the Segmentation Image
    seg = hd.seg.Segmentation(
        source_images=[],  # Todo
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions: Sequence[SegmentDescription],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Liver Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )


Constructing Binary Seg Images with Multiple Frames
---------------------------------------------------

Constructing Binary Seg Images with Multiple Segments
-----------------------------------------------------



Representation of Fractional Segs
---------------------------------

Although the pixel values of ``"FRACTIONAL"`` segmentation images can be
considered to lie within a continuous range between 0 and 1, they are in fact
not stored this way. Instead they are quantized and scaled so that they may be
stored as unsigned 8-bit integers between 0 and the value of the "Maximum
Fractional Value" (0062,000E) attribute. Thus, assuming a "Maximum Fractional
Value" of 255, a pixel value of *x* should be interpreted as a probability or
occupancy value of *x*/255.

When constructing ``"FRACTIONAL"`` segmentation images, you pass a
floating-point valued pixel array and *highdicom* handles this
quantization for you. If you wish, you may change the "Maximum Fractional Value"
from the default of 255 (which gives the maximum possible level of precision).

Similarly, *highdicom* will rescale stored values back down to the range 0-1 by
default in its methods for retrieving pixel arrays (more on this below).

Compression
-----------

The type of compression available in segmentation images depends on the
segmentation type. Pixels in a ``"BINARY"`` segmentation image are "bit-packed"
such that 8 pixels are grouped into 1 byte in the stored array. If a given frame
contains a number of pixels that is not divisible by 8 exactly, a single byte 
will straddle a frame boundary into the next frame if there is one, or the byte
will be padded with zeroes of there are no further frames. This means that
retrieving individual frames from segmentation images in which each frame
size is not divisible by 8 becomes problematic. No further compression may be
applied to frames of ``"BINARY"`` segmentation images.

Pixels in ``"FRACTIONAL"`` segmentation images may be compressed in the same
manner as other DICOM images. However, since lossy compression methods such as
standard JPEG are not designed to work with these sorts of images, we strongly
advise using only lossless compression methods with Segmentation images.
Currently *highdicom* supports the following compressed transfer syntaxes when
creating segmentation images: ``"RLELossless"`` (lossless),
``"JPEG2000Lossless"`` (lossless), ``"JPEGBaseline8Bit"`` (lossy, not
recommended).

Note that there may be advantages to using ``"FRACTIONAL"`` segmentations to
store segmentation images that are binary in nature (i.e. only taking values 0
and 1):

- If the segmentation is very simple or sparse, the lossless compression methods
  available in ``"FRACTIONAL"`` images may be more efficient than the
  "bit-packing" method required by ``"BINARY"`` segmentations.
- The clear frame boundaries make retrieving individual frames from
  ``"FRACTIONAL"`` image files possible.

Geometry of Seg Images
----------------------

Organization of Frames in Segs
------------------------------

Constructing DICOM Seg Images
-----------------------------

Reconstructing Segmentation Masks From DICOM Segs
-------------------------------------------------

fractional scaling

Viewing DICOM Seg Images
------------------------

