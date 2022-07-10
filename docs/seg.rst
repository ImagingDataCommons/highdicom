.. _seg:

DICOM Segmentation Images
=========================

DICOM Segmentation Images (often abbreviated DICOM SEG) are one of the primary
IODs (information objects definitions) implemented in the *highdicom* library.
SEG images store `segmentations
<https://en.wikipedia.org/wiki/Image_segmentation>`_ of other DICOM images
(which we will refer to as *source images*) of other modalities, such as
magnetic resonance (MR), computed tomography (CT), slide microscopy (SM) and
many others.  A segmentation is a partitioning of the source image into
different regions. In medical imaging these regions may commonly represent
different organs or tissue types, or regions of abnormality (e.g. tumor or
infarct) identified within an image.

The crucial difference between SEGs and other IODs that allow for storing image
regions is that SEGs store the segmented regions in *raster* format as pixel
arrays as opposed to the *vector* descriptions of the region's boundary used by
structured reports (SRs), presentation states, and RT structures. This makes
them a more natural choice for many automatic image processing algorithms such
as convolutional neural networks.

The DICOM standard provides a highly flexible object definition for Segmentation
images that is able to cover a large variety of possible use cases.
Unfortunately, this flexibility comes with complexity that can make Segmentation
images difficult to understand and work with.

Segments
--------

A SEG image encodes one or more distinct image regions of an image, which are
known as *segments*. A single segment could represent, for example, a
particular organ (liver, lung, kidney), tissue (fat, muscle, bone), or
abnormality (tumor, infarct).  Elsewhere the same concept is known by other
names such as *class* or *label*.

Each segment in a DICOM SEG image is represented by a separate 2D *frame* (or
set of *frames*) within the Segmentation image. One important ramification of
this is that segments need not be *mutually exclusive*, i.e. a given pixel or
spatial location within the source image can belong to multiple segments. In
other words, the segments within a SEG image may *overlap*.  There is an
optional attribute called "Segments Overlap" (0062, 0013) that, if present,
will indicate whether the segments overlap in a given SEG image.

Segment Descriptions
--------------------

Within a DICOM SEG image, segments are identified by a Segment Number. Segments
are numbered with consecutive segment numbers starting at 1 (i.e., 1, 2, 3,
...).  Additionally, each segment present is accompanied by information
describing what the segment represents. This information is placed in the
"SegmentsSequence" (0062, 0002) attribute of the segmentation file. In
*highdcom*, we use the :class:`highdicom.seg.SegmentDescription` class to hold
this information. When you construct a DICOM SEG image using *highdicom*, you
must construct a single :class:`highdicom.seg.SegmentDescription` object for
each segment, and provide the following information:

- **Segment Label**: A human-readable name for the segment (e.g. ``"Left
  Kidney"``). This can be any string.
- **Segmented Property Category**: A coded value describing the
  category of the segmented region. For example this could specify that the
  segment represents an anatomical structure, a tissue type, or an abnormality.
  This is passed as a coded value as either a
  :class:`highdicom.sr.CodedConcept`, or a :class:`pydicom.sr.coding.Code`
  object.
- **Segmented Property Type**: Another coded concept that more specifically
  describes the segmented region, as for example a kidney or tumor.  This is
  passed as a coded value as either a :class:`highdicom.sr.CodedConcept`, or a
  :class:`pydicom.sr.coding.Code` object.
- **Algorithm Type**: Whether the segment was produced by an automatic,
  semi-automatic or manual algorithm. The valid values are contained within the
  enum :class:`highdicom.seg.SegmentAlgorithmTypeValues`.
- **Anatomic Regions**: (Optional) A coded value describing the anatomic region
  in which the segment is found. For example, if the segmented property type is
  "tumor", this can be used to convey that the tumor is found in the kidney.
  This is passed as a sequence of coded values as either
  :class:`highdicom.sr.CodedConcept`, or :class:`pydicom.sr.coding.Code`
  objects.
- **Tracking ID and UID**: (Optional) This allows you to provide a ID and
  unique ID to a specific segment. This can be used to uniquely identify
  particular lesions over multiple imaging studies, for example. These are
  passed as strings.

Notice that the segment description makes use of coded concepts to ensure that
the way a particular anatomical structure is described is standardized and
unambiguous (if standard nomenclatures are used).

Here is an example of constructing a simple segment description for a segment
representing a liver that has been manually segmented.

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

In this second example, we describe a segment representing a tumor that has
been automatically segmented by an artificial intelligence algorithm. For this,
we must first provide more information about the algorithm used in an
:class:`hd.AlgorithmIdentificationSequence`.

.. code-block:: python

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

When working with existing SEG images you can use highdicom to search for
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


Binary and Fractional SEGs
--------------------------

One particularly important characteristic of a segmentation image is its
"Segmentation Type" (0062,0001), which may take the value of either ``"BINARY"``
or ``"FRACTIONAL"`` and describes the values that a given segment may take.
Segments in a ``"BINARY"`` segmentation image may only take values 0 or 1, i.e.
each pixel either belongs to the segment or does not.

By contrast, pixels in a ``"FRACTIONAL"`` segmentation image lie in the range 0
to 1. A second attribute, "Segmentation Fractional Type" (0062,0010) specifies
whether these values should be interpreted as ``"PROBABILITY"`` (i.e. the
number between 0 and 1 respresents a probability that a pixel belongs to the
segment) or ``"OCCUPANCY"`` i.e. the number represents the fraction of the
volume of the pixel's (or voxel's) area (or volume) that belongs to the
segment.

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

Constructing Basic Binary SEG Images
------------------------------------

We have now covered enough to construct a basic binary segmentation image. We
use the :class:`highdicom.seg.Segmentation` class and provide a description of
each segment, a pixel array of the segmentation mask as a numpy array with an
unsigned integer data type, the `pydicom.Datasets` of the source images for the
segmentation, and some other basic information.

.. code-block:: python

    import numpy as np

    from pydicom import dcmread
    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load a CT image
    source_image = dcmread(get_testdata_file('CT_small.dcm'))

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
        source_images=[source_image],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[liver_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Liver Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )

Constructing Binary SEG Images with Multiple Frames
---------------------------------------------------

DICOM SEGs are multiframe objects, which means that they may contain more than
one frame within the same object. For example, a single SEG image may contain
the segmentations for an entire series of CT images. In this case you can pass
a 3D numpy array as the ``pixel_array`` parameter of the constructor. The
segmentation masks of each of the input images are stacked down axis 0 of the
numpy array.  The order of segmentation masks is assumed to match the order of
the frames within the ``source_images`` parameter, i.e. ``pixel_array[i, ...]``
is the segmentation of ``source_images[i]``. Note that highdicom makes no
attempt to sort the input source images in any way. It is the responsibility of
the user to ensure that they pass the source images in a meaningful order, and
the source images and segmentation frames at the same index correspond.


.. code-block:: python

    import numpy as np

    from pydicom import dcmread
    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_files

    import highdicom as hd

    # Load a series of CT images as a list of pydicom.Datasets
    source_images = [
        dcmread(f) for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames by instance number (note that this is illustrative
    # only, sorting by instance number is not generally recommended as this
    # attribute is not guaranteed to be present in all types of source image)
    source_images = sorted(source_images, key=lambda x: x.InstanceNumber)

    # Create a segmentation by thresholding the CT image at 1000 HU
    thresholded = [
        im.pixel_array * im.RescaleSlope + im.RescaleIntercept > 1000
        for im in source_images
    ]

    # Stack segmentations of each frame down axis zero. Now we have an array
    # with shape (frames x height x width)
    mask = np.stack(thresholded, axis=0)

    # Description of liver segment produced by a manual algorithm
    # Note that now there are multiple frames but still only a single segment
    liver_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='liver',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Liver,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )

    # Construct the Segmentation Image
    seg = hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[liver_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Liver Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )

Note that the example of the previous section with a 2D pixel array is simply
a convenient shorthand for the special case where there is only a single source
frame and a single segment. It is equivalent in every way to passing a 3D
with a single frame down axis 0.

Alternatively, we could create a segmentation of a source image that is itself
a multiframe image (such as an Enhanced CT or MR image, or a Whole Slide
Microscopy image). In this case, we just pass the single source image object,
and the ``pixel_array`` input with one segmentation frame in axis 0 for each
frame of the source file, listed in ascending order by frame number. I.e.
``pixel_array[i, ...]`` is the segmentation of frame ``i`` of the single
source image.

.. code-block:: python

    import numpy as np

    from pydicom import dcmread
    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load an enhanced (multiframe) CT image
    source_dcm = dcmread(get_testdata_file('eCT_Supplemental.dcm'))

    # Apply some basic processing to correctly scale the source images
    pixel_xform_seq = source_dcm.SharedFunctionalGroupsSequence[0]\
        .PixelValueTransformationSequence[0]
    slope = pixel_xform_seq.RescaleSlope
    intercept = pixel_xform_seq.RescaleIntercept
    image_array = source_dcm.pixel_array * slope + intercept

    # Create a segmentation by thresholding the CT image at 0 HU
    mask = image_array > 0

    # Description of liver segment produced by a manual algorithm
    # Note that now there are multiple frames but still only a single segment
    liver_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='liver',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Liver,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )

    # Construct the Segmentation Image
    seg = hd.seg.Segmentation(
        source_images=[source_dcm],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[liver_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Liver Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )

Constructing Binary SEG Images with Multiple Segments
-----------------------------------------------------

To further generalize our initial example, we can include multiple segments
representing, for example, multiple organs. The first change is to include
the descriptions of all segments in the ``segment_descriptions`` parameter.
Note that the ``segment_descriptions`` list must contain segment descriptions
ordered consecutively by their ``segment_number``, starting with
``segment_number=1``.

The second change is to include the segmentation mask of each segment within
the `pixel_array` passed to the constructor. There are two methods of doing
this.  The first is to stack the masks for the multiple segments down axis 3
(the fourth axis) of the `pixel_array`. The shape of the resulting
`pixel_array` with *F* source frames of height *H* and width *W*, with *S*
segments, is then (*F* x *H* x *W* *S*). The segmentation mask for the segment
with ``segment_number=i`` should be found at ``pixel_array[:, :, :, i - 1]``
(the offset is because segments are numbered starting at 1 but numpy array
indexing starts at 0).


Note that when multiple segments are used, the first dimension (*F*) must
always be present even if there is a single source frame.

.. code-block:: python

    # Load a series of CT images as a list of pydicom.Datasets
    source_images = [
        dcmread(f) for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames by instance number
    source_images = sorted(source_images, key=lambda x: x.InstanceNumber)
    image_array = np.stack([
        im.pixel_array * im.RescaleSlope + im.RescaleIntercept
        for im in source_images
    ], axis=0)

    # Create a segmentation by thresholding the CT image at 1000 HU
    thresholded_0 = image_array > 1000

    # ...and a second below 500 HU
    thresholded_1 = image_array < 500

    # Stack the two segments down axis 3
    mask = np.stack([thresholded_0, thresholded_1], axis=3)

    # Description of bone segment produced by a manual algorithm
    bone_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='bone',
        segmented_property_category=codes.SCT.Tissue,
        segmented_property_type=codes.SCT.Bone,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )
    # Description of liver segment produced by a manual algorithm
    liver_description = hd.seg.SegmentDescription(
        segment_number=2,
        segment_label='liver',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Liver,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )
    segment_descriptions = [bone_description, liver_description]

    # Construct the Segmentation Image
    seg = hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Multi-Organ Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )

The second way to pass segmentation masks for multiple labels is as a "label
map". A label map is a 3D array (or 2D in the case of a single frame) in which
each pixel's value determines which segment it belongs to, i.e. a pixel with
value 1 belongs to segment 1 (which is the first item in the
``segment_descriptions``). A pixel with value 0 belongs to no segments. The
label map form is more convenient to work with in many applications, however it
is limited to representing segmentations that do not overlap (i.e. those in
which a single pixel can belong to at most one segment). The more general form
does not have this limitation: a given pixel may belong to any number of
segments.

The following snippet produces an equivalent SEG image to the previous snippet,
but passes the mask as a label map rather than as a stack of segments.

.. code-block:: python

    # Load a CT image
    source_images = [
        dcmread(f) for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames by instance number
    source_images = sorted(source_images, key=lambda x: x.InstanceNumber)
    image_array = np.stack([
        im.pixel_array * im.RescaleSlope + im.RescaleIntercept
        for im in source_images
    ], axis=0)

    # Create the same two segments as above as a label map
    mask = np.zeros_like(image_array, np.uint8)
    mask[image_array > 1000] = 1
    mask[image_array < 500] = 2

    # Construct the Segmentation Image
    seg = hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Multi-Organ Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )


Representation of Fractional SEGs
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
Note that this does entail a loss of precision.

Similarly, *highdicom* will rescale stored values back down to the range 0-1 by
default in its methods for retrieving pixel arrays (more on this below).

Otherwise, constructing ``"FRACTIONAL"`` segs is identical to constructing
binary ones ``"BINARY"``, with the caveat that fractional SEGs may not use the
"label map" method to pass multiple segments but must instead stack them along
axis 3.

The example below shows a simple example of construction a fractional seg
representing a probabilistic segmentation of the liver.

.. code-block:: python

    import numpy as np

    from pydicom import dcmread
    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load a CT image
    source_image = dcmread(get_testdata_file('CT_small.dcm'))

    # Description of liver segment produced by a manual algorithm
    liver_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='liver',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Liver,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )

    # Pixel array is an float array with values between 0 and 1
    mask = np.zeros((128, 128), dtype=float)
    mask[10:20, 10:20] = 0.5
    mask[30:40, 30:40] = 0.75

    # Construct the Segmentation Image
    seg = hd.seg.Segmentation(
        source_images=[source_image],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
        fractional_type=hd.seg.SegmentationFractionalTypeValues.PROBABILITY,
        segment_descriptions=[liver_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Liver Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )

Compression
-----------

The type of pixel compression available in segmentation images depends on the
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

Geometry of SEG Images
----------------------

In the simple cases we have seen so far, the geometry of the segmentation
``pixel_array`` has matched that of the source images, i.e. there is a spatial
correspondence between a given pixel in the ``pixel_array`` and the
corresponding pixel in the relevant source frame. While this covers most use
cases, DICOM SEGs actually allow for more general segmentations in which there
is a more complicated relationship between the source frames and the
segmentation masks. This could arise when a source image is resampled or
transformed before the segmentation method is applied, such that there is no
longer a simple correspondence between pixels in the segmentation mask and
pixels in the source image.

Highdicom supports this case by allowing you to manually specify the plane
positions of the each frame in the segmentation mask, and further the
orientations and pixel spacings of these planes if they do not match that in the
source images. In this case, the correspondence between the items of the
``source_images`` list and axis 0 of the segmentation ``pixel_array`` is broken
and the number of frames in each may differ.

.. code-block:: python

    import numpy as np

    from pydicom import dcmread
    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_files

    import highdicom as hd

    # Load a CT image
    source_images = [
        dcmread(f) for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames by instance number
    source_images = sorted(source_images, key=lambda x: x.InstanceNumber)

    # Now the shape and size of the mask does not have to match the source
    # images
    mask = np.zeros((2, 100, 100), np.uint8)
    mask[0, 50:60, 50:60] = 1

    # Define custom positions for each frame
    positions = [
        hd.PlanePositionSequence(
            hd.CoordinateSystemNames.PATIENT,
            [100.0, 50.0, -50.0]
        ),
        hd.PlanePositionSequence(
            hd.CoordinateSystemNames.PATIENT,
            [100.0, 50.0, -48.0]
        ),
    ]

    # Define a custom orientation and spacing for the segmentation mask
    orientation = hd.PlaneOrientationSequence(
        hd.CoordinateSystemNames.PATIENT,
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
    )
    spacings = hd.PixelMeasuresSequence(
        slice_thickness=2.0,
        pixel_spacing=[2.0, 2.0]
    )

    # Description of liver segment produced by a manual algorithm
    # Note that now there are multiple frames but still only a single segment
    liver_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='liver',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Liver,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )

    # Construct the Segmentation Image
    seg = hd.seg.Segmentation(
        source_images=source_images,
        pixel_array=mask,
        plane_positions=positions,
        plane_orientation=orientation,
        pixel_measures=spacings,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[liver_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Liver Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )

Organization of Frames in SEGs
------------------------------

After construction, there may be many 2D frames within an SEG image, each
referring to the segmentation of a certain 2D source image or frame (or a
resampled plane defined by its plane position and orientation) for a certain
segment. Note that this may mean that there are multiple frames of the SEG
image that are derived from each frame of the input image or series. These
frames are stored within the SEG as an array indexed by a frame number
(consecutive integers starting at 1). The DICOM standard gives the creator of a
SEG a lot of freedom about how to organize the resulting frames within the 1D
list within the SEG. To complicate matters further, frames in the segmentation
image that would otherwise be "empty" (contain only 0s) may be omitted from the
SEG image entirely (this is highdicom's default behavior).

Every `pydicom.Dataset` has the `.pixel_array` property, which, in the case of
a multiframe image, returns the full list of frames in the image as an array of
shape (frames x rows x colums), with frames organized. A
:class:`highdicom.seg.Segmentation` is a sub-class of `pydicom.Dataset`, and
therefore also has the `.pixel_array` property. However, given the complexities
outlined above, *it is not recommended* to use to the `.pixel_array` property
with SEG images since the meaning of the resulting array is unclear without
referring to other metadata within the object in all but the most trivial cases
(single segment and/or single source frame with no empty frames). This may be
particularly confusing and perhaps offputting to those working with SEG images
for the first time.

The order in which the creator of a SEG image has chosen to organize the frames
of the SEG image is described by the `"DimensionIndexSequence"
<https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.html#table_C.7.6.17-1>`_
attribute (0020, 9222) of the SEG object. Referring to this, and the
information held about a given frame within the item of the
`"PerFrameFunctionalGroupsSequence"
<https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.html#table_C.7.6.16-1>`_
attribute (5200, 9230) with the matching frame number, it is possible to
determine the meaning of a certain segmentation frame. We will not describe the
full details of this mechanism here.

Instead, highdicom provides a family of methods to help users reconstruct
segmentation masks from SEG objects in a predictable and more intuitive way. We
recommend using these methods over the basic `.pixel_array` in nearly all
circumstances.

Reconstructing Segmentation Masks From DICOM SEGs
-------------------------------------------------

TODO

Viewing DICOM SEG Images
------------------------

Unfortunately, DICOM SEG images are not widely supported by DICOM
viewers. Viewers that do support SEG include:

- The `OHIF Viewer <https://github.com/OHIF/Viewers>`_, an open-source
  web-based viewer.
- `3D Slicer <https://www.slicer.org/>`_, an open-source desktop application
  for 3D medical image computing. It supports both display and creation of
  DICOM SEG files via the "Quantitative Reporting" plugin.
