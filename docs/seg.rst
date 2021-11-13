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

In many applications, it is assumed that segments are *mutually exclusive*,
i.e. a given pixel can belong to at most one segment. However DICOM Seg images
do not have this limitation: a single pixel can belong to any number of
different segments. In other words, the segments may *overlap*. There is an
optional attribute called "Segments Overlap" (0062, 0013) that, if present,
will indicate whether the segments overlap in a given Seg image.

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

- Multiple image regions (known as *segments*) that may be mutually exclusive
  or overlapping (non mutually exclusive).
- Binary segmentations (in which
  each pixel unambiguously either belongs to a region or does not belong to a
  region) or *fractional* segmentations, in which the membership of pixel to a
  region is expressed as a number between 0 and 1.



Segmentation Frames and Source Frames
-------------------------------------

Viewing DICOM Seg Images
------------------------

Reconstructing Segmentation Masks From DICOM Segs
-------------------------------------------------

