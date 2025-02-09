.. _seg:

Segmentation (SEG) Images
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
Unfortunately, this flexibility comes with complexity that may make Segmentation
images difficult to understand and work with at first.

Segments
--------

A SEG image encodes one or more distinct regions of an image, which are known
as *segments*. A single segment could represent, for example, a particular
organ or structure (liver, lung, kidney, cell nucleus), tissue (fat, muscle,
bone), or abnormality (tumor, infarct).  Elsewhere the same concept is known by
other names such as *class* or *label*.

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
each segment. The segment description includes the following information:

- **Segment Label**: A human-readable name for the segment (e.g. ``"Left
  Kidney"``). This can be any string.
- **Segmented Property Category**: A coded value describing the
  category of the segmented region. For example this could specify that the
  segment represents an anatomical structure, a tissue type, or an abnormality.
  This is passed as either a
  :class:`highdicom.sr.CodedConcept`, or a :class:`pydicom.sr.coding.Code`
  object.
- **Segmented Property Type**: Another coded value that more specifically
  describes the segmented region, as for example a kidney or tumor.  This is
  passed as either a :class:`highdicom.sr.CodedConcept`, or a
  :class:`pydicom.sr.coding.Code` object.
- **Algorithm Type**: Whether the segment was produced by an automatic,
  semi-automatic, or manual algorithm. The valid values are contained within the
  enum :class:`highdicom.seg.SegmentAlgorithmTypeValues`.
- **Anatomic Regions**: (Optional) A coded value describing the anatomic region
  in which the segment is found. For example, if the segmented property type is
  "tumor", this can be used to convey that the tumor is found in the kidney.
  This is passed as a sequence of coded values as either
  :class:`highdicom.sr.CodedConcept`, or :class:`pydicom.sr.coding.Code`
  objects.
- **Tracking ID and UID**: (Optional) These allow you to provide, respectively,
  a human readable ID and unique ID to a specific segment. This can be used,
  for example, to uniquely identify particular lesions over multiple imaging
  studies. These are passed as strings.

Notice that the segment description makes use of coded concepts to ensure that
the way a particular anatomical structure is described is standardized and
unambiguous (if standard nomenclatures are used). See :ref:`coding` for more
information.

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
:class:`highdicom.AlgorithmIdentificationSequence`.

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

Segmentation Type (Binary, Fractional and Labelmap)
---------------------------------------------------

One particularly important characteristic of a segmentation image is its
"Segmentation Type" (0062,0001). There are three options here, contained within
the highdicom enum :class:`highdicom.seg.SegmentationTypeValues`:

- ``"BINARY"`` segmentations stores each segment in a separate set of frames.
  Within each segment, pixels can only take the value 0 (meaning that the pixel
  does not belong to the segment) or 1 (meaning that the pixel does belong to
  the segment). Note that although the name may suggest that only one segment
  is present, in fact there is no limit on the number of segments (i.e.
  "binary" refers to the possible values of pixels within a segment, not the
  number of segments). Because each segment is stored using a separate set of
  frames, segments are not constrained to be mutually exclusive: a single pixel
  can belong to any number of segments. In other words, segments may overlap
  with each other.

  ``"BINARY"`` segmentations are the most widely supported and used. However
  they also have some important downsides. Storing each segment separately
  means that ``"BINARY"`` segmentations can have a very large number of frames
  if there are a large number of segments. Furthermore, because they are stored
  as single bit images, the options for compression are very limited. As a
  result, the segmentation objects can get very large and unweildy. Lastly,
  having separate frames is simply not a convenient form to work with for many
  applications.

- ``"FRACTIONAL"`` segmentations also store each segment as a separate set of
  frames, but within each segment pixel values lie in the range 0 to 1. A
  second attribute, "Segmentation Fractional Type" (0062,0010) specifies how
  these values should be interpreted. There are two options, represented by the
  enumerated type :class:`highdicom.seg.SegmentationFractionalTypeValues`:

  - ``"PROBABILITY"``, i.e. the number between 0 and 1 represents a probability
    that a pixel belongs to the segment
  - ``"OCCUPANCY"`` i.e. the number represents the fraction of the volume of the
    pixel's (or voxel's) area (or volume) that belongs to the segment

- ``"LABELMAP"`` segmentations are a new type of segmentation introduced to the
  standard in 2024 and supported in highdicom since version 0.24.0. They are
  designed to address the shortcomings of ``"BINARY"`` segmentations described
  above by combining all segments into a single set of frames. The pixel values
  are unsigned 8 or 16 bit integers that encode the segment membership of the
  pixel. This means that a single pixel cannot belong to multiple segments
  (segments must be mutually exclusive and cannot overlap). This represents a
  limitation of the ``"LABELMAP"`` representation, relative to ``"BINARY"``,
  but in practice non-overlapping segments are a very common case.

  Since they use 8 or 16 bit pixels, there are also many more options for
  compression of ``"LABELMAP"`` segmentations than ``"BINARY"`` ones. The
  combination of fewer frames and better compression of each frame can lead to
  ``"LABELMAP"`` segmentations being smaller than the equivalent ``"BINARY"``
  segmentation by 2 orders of magnitude in extreme cases.

  Unfortunately, support for ``"LABELMAP"`` segmentations is currently very
  limited (highdicom is the first software to support them to our knowledge).
  We hope that more applications and libraries will begin to support them in
  the near future. We encourage you to request that other tools you use add
  support for them.

  Advanced users should note that labelmap segmentations are actually encoded
  using a different SOP class (but the same IOD) to fractional or binary
  segmentations. Highdicom handles this for you and uses the
  :class:`highdicom.seg.Segmentation` class for both SOP classes such that this
  distinction should not matter in most situations.

Constructing Basic Binary and Labelmap SEG Images
-------------------------------------------------

We have now covered enough to construct a basic binary or labelmap segmentation
image. We use the :class:`highdicom.seg.Segmentation` class and provide a
description of each segment, a pixel array of the segmentation mask, the source
images as a list of ``pydicom.Dataset`` objects, and some other basic
information. The segmentation pixel array is provided as a numpy array with a
boolean or unsigned integer data type containing only the values 0 and 1.

In this example, we encode this segmentation using the ``"BINARY"`` segmentation
type, however we could straightforwardly change this to ``"LABELMAP"`` and keep
everything else the same.

.. code-block:: python

    import numpy as np

    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load a CT image
    source_image = hd.imread(get_testdata_file('CT_small.dcm'))

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

Constructing Binary/Labelmap SEG Images with Multiple Frames
------------------------------------------------------------

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
that the source images and segmentation frames at the same index correspond.

Again, we could straightforwardly swap ``"BINARY"`` for ``"LABELMAP"`` in this
example to use the labelmap segmentation type.


.. code-block:: python

    import numpy as np

    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_files

    import highdicom as hd

    # Load a series of CT images as a list of pydicom.Datasets
    source_images = [
        hd.imread(f) for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames spatially
    source_images = hd.spatial.sort_datasets(source_images)

    # Create a segmentation by thresholding the CT image at 1000 HU
    thresholded = [
        im.get_frame(1) > 1000
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
frame and a single segment. It is equivalent in every way to passing a 3D array
with a single frame down axis 0.

Constructing Binary/Labelmap SEG Images of Multiframe Source Images
-------------------------------------------------------------------

Alternatively, we could create a segmentation of a source image that is itself
a multiframe image (such as an Enhanced CT, Enhanced MR image, or a Whole Slide
Microscopy image). In this case, we just pass the single source image object,
and the ``pixel_array`` input with one segmentation frame in axis 0 for each
frame of the source file, listed in ascending order by frame number. I.e.
``pixel_array[i, ...]`` is the segmentation of frame ``i + 1`` of the single
source image (the offset of +1 is because numpy indexing starts at 0 whereas
DICOM frame indices start at 1). This is also valid for the ``"LABELMAP"``
segmentation type.

.. code-block:: python

    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load an enhanced (multiframe) CT image
    source_image = hd.imread(get_testdata_file('eCT_Supplemental.dcm'))

    # Get a stack of all the frames of the image
    image_array = source_image.get_frames()

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

Constructing Binary SEG Images with Multiple Segments
-----------------------------------------------------

To further generalize our initial example, we can include multiple segments
representing, for example, multiple organs. The first change is to include
the descriptions of all segments in the ``segment_descriptions`` parameter.
Note that the ``segment_descriptions`` list must contain segment descriptions
ordered consecutively by their ``segment_number``, starting with
``segment_number=1``.

The second change is to include the segmentation mask of each segment within
the ``pixel_array`` passed to the constructor. There are two methods of doing
this.  The first (the "stacked segments" form) is to stack the masks for the
multiple segments down axis 3 (the fourth axis) of the ``pixel_array``. The
shape of the resulting ``pixel_array`` with *F* source frames of height *H* and
width *W*, with *S* segments, is then (*F* x *H* x *W* x *S*). The segmentation
mask for the segment with ``segment_number=i`` should be found at
``pixel_array[:, :, :, i - 1]`` (the offset of -1 is because segments are
numbered starting at 1 but numpy array indexing starts at 0).

Note that when multiple segments are used, the first dimension (*F*) must
always be present even if there is a single source frame.

.. code-block:: python

   import numpy as np

   from pydicom.sr.codedict import codes
   from pydicom.data import get_testdata_files

   import highdicom as hd

    # Load a series of CT images as a list of pydicom.Datasets
    source_images = [
        hd.imread(f) for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames spatially
    source_images = hd.spatial.sort_datasets(source_images)
    image_array = np.stack(
        [
            im.get_frame(1) for im in source_images
        ]
    )

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
``segment_descriptions``). A pixel with value 0 belongs to no segments.

Therefore, the following snippet produces an equivalent SEG image to the
previous snippet, but passes the mask as a label map rather than as a stack of
segments.

.. code-block:: python

   import numpy as np

   from pydicom.sr.codedict import codes
   from pydicom.data import get_testdata_files

   import highdicom as hd

    # Load a series of CT images as a list of pydicom.Datasets
    source_images = [
        hd.imread(f) for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames spatially
    source_images = hd.spatial.sort_datasets(source_images)
    image_array = np.stack(
        [
            im.get_frame(1) for im in source_images
        ]
    )

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

These two forms for the ``pixel_array`` argument ("stacked segments" and "label
map") correspond to the way pixels are stored in segmentations with
``"BINARY"`` and ``"LABELMAP"`` segmentation types, respectively. However, it
is important to understand that the form you provide to the constructor and the
segmentation type used to store it are separate (and largely decoupled)
considerations. *Highdicom* infers which of the two forms you are passing to it
according to the number of dimensions in the ``pixel_array`` (4 dimensions for
"stacked segment" form and 3 for "label map" form), and stores the segmentation
according to the value you provide for ``segmentation_type``. If the two do not
match, *highdicom* will transparently convert between the two forms for you. As
a result, you can provide an array in "label map" form and request that it be
stored as a ``"BINARY"`` segmentation with separate segments. Similarly, you
can provide an array in "stacked segments" form and request that it be stored
with segmentation type ``"LABELMAP"``. However in this latter case the segments
must not overlap or an error will be raised. Similarly, segmentations that do
contain overlapping segments can only be passed in "stacked segment" form, and
can only be stored using the ``"BINARY"`` segmentation type.

Spatially Non-aligned SEG Images
--------------------------------

In the simple cases we have seen so far, the geometry of the segmentation
``pixel_array`` has matched that of the source images, i.e. there is a spatial
correspondence between a given pixel in the ``pixel_array`` and the
corresponding pixel in the relevant source frame. While this covers most use
cases, DICOM SEGs actually allow for more general segmentations in which there
is a more complicated geometrical relationship between the source frames and
the segmentation masks. This could arise when a source image is resampled or
transformed before the segmentation method is applied, such that there is no
longer a simple correspondence between pixels in the segmentation mask and
pixels in the original source DICOM image.

`Highdicom` supports such cases in two ways. The first, possible when the
segmentation array is defined on a regularly-sampled 3D grid, is to pass the
segmentation array as an instance of the :class:`highdicom.Volume` class
instead of a plain NumPy array. (see :doc:`volume` for an overview of volumes).
Since the :class:`highdicom.Volume` class specifies its position within the
frame of reference coordinate system, the position of each plane can be
inferred automatically. The volume can have an arbitrary size, spacing, and
orientation and these properties do not need to match those of the source
images. Just like a standard NumPy array, the volume can be either in the
"label map" form, where each pixel values specifies segment membership, or
"stacked segment" form, with a further array dimension along which binary
segments are stacked. In the "label map" case the volume must have no channel
dimensions. In the "stacked segments" case, the volume must have exactly one
channel dimension with the descriptor being the "SegmentNumber" tag.

.. code-block:: python

    import numpy as np

    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_files

    import highdicom as hd

    # Load a series of CT images
    source_images = [
        hd.imread(f)
        for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Now the shape and size of the mask does not have to match the source
    # images
    mask = np.zeros((2, 100, 100), np.uint8)
    mask[0, 50:60, 50:60] = 1

    volume = hd.Volume.from_components(
        array=mask,
        spacing=[1.25, 1.25, 5.0],
        direction=np.eye(3),
        position=[10.0, 20.0, 30.0],
        frame_of_reference_uid=source_images[0].FrameOfReferenceUID,
        coordinate_system=hd.CoordinateSystemNames.PATIENT,
    )

    # Description of liver segment produced by a manual algorithm
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
        pixel_array=volume,
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

The second way to specify a segmentation that does not align spatially with the
source images is by manually specifying the plane positions of the each frame
in the segmentation mask, and also the orientations and pixel spacings of these
planes if they do not match that in the source images. This is more flexible
but less convenient than using the ``Volumes`` class. In this case, the
correspondence between the items of the ``source_images`` list and axis 0 of
the segmentation ``pixel_array`` is broken and the number of frames in each may
differ.

.. code-block:: python

    import numpy as np

    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_files

    import highdicom as hd

    # Load a CT image
    source_images = [
        hd.imread(f)
        for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]

    # Sort source frames spatially
    source_images = hd.spatial.sort_datasets(source_images)

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

Constructing SEG Images from a Total Pixel Matrix
-------------------------------------------------

Some digital pathology images are represented as "tiled" images,
in which the full image (known as the "total pixel matrix") is divided up
into smaller rectangular regions in the row and column dimensions and each
region ("tile") is stored as a frame in a multiframe DICOM image.

Segmentations of such images are stored as a tiled image in the same manner.
There are a two options in `highdicom` for doing this. You can either pass each
tile/frame individually stacked as a 1D list down the first dimension of the
``pixel_array`` as we have already seen (with the location of each frame either
matching that of the corresponding frame in the source image or explicitly
specified in the ``plane_positions`` argument), or you can pass the 2D total
pixel matrix of the segmentation and have `highdicom` automatically create the
tiles for you.

To enable this latter option, pass the ``pixel_array`` as a single frame (i.e.
a 2D labelmap array, a 3D labelmap array with a single frame stacked down the
first axis, or a 4D array with a single frame stacked down the first dimension
and any number of segments stacked down the last dimension) and set the
``tile_pixel_array`` argument to ``True``. You can optionally choose the size
(in pixels) of each tile using the ``tile_size`` argument, or, by default, the
tile size of the source image will be used (regardless of whether the
segmentation is represented at the same resolution as the source image).

If you need to specify the plane positions of the image explicitly, you should
pass a single item to the ``plane_positions`` argument giving the location of
the top left corner of the full total pixel matrix. Otherwise, all the usual
options are available to you.

.. code-block:: python

    # Use an example slide microscopy image from the highdicom test data
    # directory
    sm_image = hd.imread('data/test_files/sm_image.dcm')

    # The source image has multiple frames/tiles, but here we create a mask
    # corresponding to the entire total pixel matrix
    mask = np.zeros(
        (
            sm_image.TotalPixelMatrixRows,
            sm_image.TotalPixelMatrixColumns
        ),
        dtype=np.uint8,
    )
    mask[38:43, 5:41] = 1

    property_category = hd.sr.CodedConcept("91723000", "SCT", "Anatomical Structure")
    property_type = hd.sr.CodedConcept("84640000", "SCT", "Nucleus")
    segment_descriptions = [
        hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Segment #1',
            segmented_property_category=property_category,
            segmented_property_type=property_type,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        ),
    ]

    seg = hd.seg.Segmentation(
        source_images=[sm_image],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Slide Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        tile_pixel_array=True,
    )

    # The result stores the mask as a set of 10 tiles of the non-empty region of
    # the total pixel matrix, each of size (10, 10), matching # the tile size of
    # the source image
    assert seg.NumberOfFrames == 10
    assert seg.pixel_array.shape == (10, 10, 10)

``"TILED_FULL"`` and ``"TILED_SPARSE"``
---------------------------------------

When the segmentation is stored as a tiled image, there are two ways in which
the locations of each frame/tile may be specified in the resulting object.
These are defined by the value of the
`"DimensionOrganizationType"
<https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.html#table_C.7.6.17-1>`_
attribute:

- ``"TILED_SPARSE"``: The position of each tile is explicitly defined in the
  `"PerFrameFunctionalGroupsSequence"
  <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.html#table_C.7.6.16-1>`_
  of the object. This requires a potentially very long sequence to store all
  the per-frame metadata, but does allow for the omission of empty frames from
  the segmentation and other irregular tiling strategies.
- ``"TILED_FULL"``: The position of each tile is implicitly defined using a
  predetermined order of the frames. This saves the need to store the pre-frame
  metadata but does not allow for the omission of empty frames of the
  segmentation and is generally less flexible. It may also be simpler for a
  receiving application to process, since the tiles are guaranteed to be
  regularly and consistently ordered.

You can control this behavior by specifying the
``dimension_organization_type`` parameter and passing a value of the
:class:`highdicom.DimensionOrganizationTypeValues` enum. The default value is
``"TILED_SPARSE"``. Generally, the ``"TILED_FULL"`` option will be used in
combination with ``tile_pixel_array`` argument.


.. code-block:: python

    # Using the same example as above, this time as TILED_FULL
    seg = hd.seg.Segmentation(
        source_images=[sm_image],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Slide Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        tile_pixel_array=True,
        omit_empty_frames=False,
        dimension_organization_type=hd.DimensionOrganizationTypeValues.TILED_FULL,
    )

    # The result stores the mask as a set of 25 tiles of the entire region of
    # the total pixel matrix, each of size (10, 10), matching the tile size of
    # the source image
    assert seg.NumberOfFrames == 25
    assert seg.pixel_array.shape == (25, 10, 10)

Multi-resolution Pyramids
-------------------------

Whole slide digital pathology images can often be very large and as such it
is common to represent them as *multi-resolution pyramids* of images, i.e.
to store multiple versions of the same image at different resolutions. This
helps viewers render the image at different zoom levels.

Within DICOM, this can also extend to segmentations derived from whole slide
images. Multiple different SEG images may be stored, each representing the
same segmentation at a different resolution, as different instances within a
DICOM series.

*highdicom* provides the :func:`highdicom.seg.create_segmentation_pyramid`
function to assist with this process. This function handles multiple related
scenarios:

* Constructing a segmentation of a source image pyramid given a
  segmentation pixel array of the highest resolution source image.
  Highdicom performs the downsampling automatically to match the
  resolution of the other source images. For this case, pass multiple
  ``source_images`` and a single item in ``pixel_arrays``.
* Constructing a segmentation of a source image pyramid given user-provided
  segmentation pixel arrays for each level in the source pyramid. For this
  case, pass multiple ``source_images`` and a matching number of
  ``pixel_arrays``.
* Constructing a segmentation of a single source image given multiple
  user-provided downsampled segmentation pixel arrays. For this case, pass
  a single item in ``source_images``, and multiple items in
  ``pixel_arrays``).
* Constructing a segmentation of a single source image and a single
  segmentation pixel array by downsampling by a given list of
  ``downsample_factors``. For this case, pass a single item in
  ``source_images``, a single item in ``pixel_arrays``, and a list of one
  or more desired ``downsample_factors``.

Here is a simple of example of specifying a single source image and segmentation
array, and having *highdicom* create a multi-resolution pyramid segmentation
series at user-specified downsample factors.

.. code-block:: python

    import highdicom as hd
    import numpy as np


    # Use an example slide microscopy image from the highdicom test data
    # directory
    sm_image = hd.imread('data/test_files/sm_image.dcm')

    # The source image has multiple frames/tiles, but here we create a mask
    # corresponding to the entire total pixel matrix
    mask = np.zeros(
        (
            sm_image.TotalPixelMatrixRows,
            sm_image.TotalPixelMatrixColumns
        ),
        dtype=np.uint8,
    )
    mask[38:43, 5:41] = 1

    property_category = hd.sr.CodedConcept("91723000", "SCT", "Anatomical Structure")
    property_type = hd.sr.CodedConcept("84640000", "SCT", "Nucleus")
    segment_descriptions = [
        hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Segment #1',
            segmented_property_category=property_category,
            segmented_property_type=property_type,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        ),
    ]

    # This will create a segmentation series of three images: one at the
    # original source image resolution (implicit), one at half the size, and
    # another at a quarter of the original size.
    seg_pyramid = hd.seg.create_segmentation_pyramid(
        source_images=[sm_image],
        pixel_arrays=[mask],
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Slide Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        downsample_factors=[2.0, 4.0]
    )

Note that the :func:`highdicom.seg.create_segmentation_pyramid` function always
behaves as if the ``tile_pixel_array`` input is ``True`` within the segmentation
constructor, i.e. it assumes that the input segmentation masks represent total
pixel matrices.

Representation of Fractional SEGs
---------------------------------

Although the pixel values of ``"FRACTIONAL"`` segmentation images can be
considered to lie within a continuous range between 0 and 1, they are in fact
not stored this way. Instead they are quantized and scaled so that they may be
stored as unsigned 8-bit integers between 0 and the value of the "Maximum
Fractional Value" (0062,000E) attribute. Thus, assuming a "Maximum Fractional
Value" of 255, a pixel value of *x* should be interpreted as a probability or
occupancy value of *x*/255. You can control the "Maximum Fractional Value" by
passing the ``max_fractional_value`` parameter. 255 is used as the default.

When constructing ``"FRACTIONAL"`` segmentation images, you pass a
floating-point valued pixel array and *highdicom* handles this
quantization for you. If you wish, you may change the "Maximum Fractional Value"
from the default of 255 (which gives the maximum possible level of precision).
Note that this does entail a loss of precision.

Similarly, *highdicom* will rescale stored values back down to the range 0-1 by
default in its methods for retrieving pixel arrays (more on this below).

Otherwise, constructing ``"FRACTIONAL"`` segs is identical to constructing
``"BINARY"``/``"LABELMAP"`` ones, with the limitation that fractional SEGs may
not use the "label map" form to pass multiple segments but must instead stack
them along axis 3.

The example below shows a simple example of constructing a fractional seg
representing a probabilistic segmentation of the liver.

.. code-block:: python

    import numpy as np

    from pydicom.sr.codedict import codes
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load a CT image
    source_image = hd.imread(get_testdata_file('CT_small.dcm'))

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

Implicit Conversion to Fractional
---------------------------------

Note that any segmentation pixel array that `highdicom` allows you to store as
a ``"BINARY"``/``"LABELMAP"`` SEG (i.e. a binary segmentation array with
segments stacked down axis 3, or a label-map style segmentation array) may also
be stored as a ``"FRACTIONAL"`` SEG. You just pass the integer array, specify
the ``segmentaton_type`` as ``"FRACTIONAL"`` and `highdicom` does the
conversion for you. Input pixels with value 1 will be automatically stored with
value ``max_fractional_value``. We recommend that if you do this, you specify
``max_fractional_value=1`` to clearly communicate that the segmentation is
inherently binary in nature.

Why would you want to make this seemingly rather strange choice? Well,
``"FRACTIONAL"`` SEGs tend to compress much better than ``"BINARY"`` ones (see
next section) and be more widely supported than ``"LABELMAP"`` ones. Note
however, that this is a misuse of the intent of the standard, so this is
strongly discouraged in all but controlled internal research/development
settings. Since the introduction of ``"LABELMAP"`` segmentations in highdicom
0.24.0, they should be always be preferred unless the segmentations are truly
fractional in nature.

Also note that while this used to be a more serious issue it is less serious
now that ``"JPEG2000Lossless"`` compression is now supported for ``"BINARY"``
segmentations as of highdicom v0.23.0.

Compression
-----------

The types of pixel compression available in segmentation images depends on the
segmentation type.

Pixels in an uncompressed ``"BINARY"`` segmentation image are "bit-packed" such
that 8 pixels are grouped into 1 byte in the stored array. If a given frame
contains a number of pixels that is not divisible by 8 exactly, a single byte
will straddle a frame boundary into the next frame if there is one, or the byte
will be padded with zeroes of there are no further frames. This means that
retrieving individual frames from segmentation images in which each frame size
is not divisible by 8 becomes problematic. For this reason, as well as for
space efficiency (sparse segmentations tend to compress very well), we
recommend favoring ``"LABELMAP"`` segmentations where this is possible, or
using ``"JPEG2000Lossless"`` compression with ``"BINARY"`` segmentations if it
is not. This is the only compression method currently supported for
``"BINARY"`` segmentations. However, beware that reading these single-bit JPEG
2000 images may not be supported by all other tools and viewers.

Pixels in ``"LABELMAP"``/``"FRACTIONAL"`` segmentation images may be compressed using one of
the lossless compression methods available within DICOM. Currently *highdicom*
supports the following compressed transfer syntaxes when creating
``"LABELMAP"``/``"FRACTIONAL"``  segmentation images: ``"RLELossless"``,
``"JPEG2000Lossless"``, and ``"JPEGLSLossless"``.

In our experience, ``"JPEG2000Lossless"`` offers excellent compression and is
well supported by other tools, but is also somewhat slow. ``"JPEGLSLossless"``
gives much better compression and decomression times for similar or slightly
worse compression rates, but is also less widely supported.

Multiprocessing
---------------

When creating large, multiframe segmentations using a
compressed transfer syntax, the time taken to compress the frames can become
large and dominate the time taken to create the segmentation. By default,
frames are compressed in series using the main process, however the ``workers``
parameter allows you to specify a number of additional worker processes that
will be used to compress frames in parallel. Setting ``workers`` to a negative
number uses all available processes on your machine. Note that while this is
likely to result in significantly lower creations times for segmentations with
a very large number of frames, for segmentations with only a few frames the
additional overhead of spawning processes may in fact slow the entire
segmentation creation process down.

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
SEG image entirely (this is `highdicom`'s default behavior but can be turned
off if you prefer by specifying ``omit_empty_frames=False`` in the constructor).

Every ``pydicom.Dataset`` has the ``.pixel_array`` property, which, in the case
of a multiframe image, returns the full list of frames in the image as an array
of shape (frames x rows x columns), with frames organized in whatever manner
they were organized in by the creator of the object. A
:class:`highdicom.seg.Segmentation` is a sub-class of ``pydicom.Dataset``, and
therefore also has the ``.pixel_array`` property. However, given the
complexities outlined above, *it is not recommended* to use to the
``.pixel_array`` property with SEG images since the meaning of the resulting
array is unclear without referring to other metadata within the object in all
but the most trivial cases (single segment and/or single source frame with no
empty frames). This may be particularly confusing and perhaps offputting to
those working with SEG images for the first time.

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

Instead, `highdicom` provides a family of methods to help users reconstruct
segmentation masks from SEG objects in a predictable and more intuitive way. We
recommend using these methods over the basic ``.pixel_array`` in nearly all
circumstances.

Reading Existing Segmentation Images
------------------------------------

Since a segmentation is a DICOM object just like any other image, you can read
it in from a file using ``pydicom`` to give you a ``pydicom.Dataset``. However,
if you read the file in using the :func:`highdicom.seg.segread` function, the
segmentation will have type :class:`highdicom.seg.Segmentation`. This adds
several extra methods that make it easier to work with the segmentation.

.. code-block:: python

    import highdicom as hd

    seg = hd.seg.segread('data/test_files/seg_image_ct_binary.dcm')
    assert isinstance(seg, hd.seg.Segmentation)

Alternatively, you can convert an existing ``pydicom.Dataset`` into a
:class:`highdicom.seg.Segmentation` using the
:meth:`highdicom.seg.Segmentation.from_dataset()` method. This is useful if
you receive the object over network rather than reading from file.

.. code-block:: python

    import highdicom as hd
    import pydicom

    dcm = pydicom.dcmread('data/test_files/seg_image_ct_binary.dcm')

    # Convert to highdicom Segmentation object
    seg = hd.Segmentation.from_dataset(dcm)

    assert isinstance(seg, hd.seg.Segmentation)

By default this operation copies the underlying dataset, which may be slow for
large objects. You can use ``copy=False`` to change the type of the object
without copying the data.

Since :class:`highdicom.seg.Segmentation` is a subclass of ``pydicom.Dataset``,
you can still perform `pydicom` operations on it, such as access DICOM
attributes by their keyword, in the usual way.

.. code-block:: python

    import highdicom as hd
    import pydicom

    seg = hd.seg.segread('data/test_files/seg_image_ct_binary.dcm')
    assert isinstance(seg, pydicom.Dataset)

    # Accessing DICOM attributes as usual in pydicom
    seg.PatientName
    # 'Doe^Archibald'

Searching For Segments
----------------------

When working with existing SEG images you can use the method
:meth:`highdicom.seg.Segmentation.get_segment_numbers()` to search for segments
whose descriptions meet certain criteria. For example:

.. code-block:: python

    from pydicom.sr.codedict import codes

    import highdicom as hd


    # This is a test file in the highdicom git repository
    seg = hd.seg.segread('data/test_files/seg_image_ct_binary_overlap.dcm')

    # Check the number of segments
    assert seg.number_of_segments == 2

    # Check the range of segment numbers
    assert seg.segment_numbers == range(1, 3)

    # Search for segments by label (returns segment numbers of all matching
    # segments)
    assert seg.get_segment_numbers(segment_label='first segment')) == [1]
    assert seg.get_segment_numbers(segment_label='second segment')) == [2]

    # Search for segments by segmented property type (returns segment numbers
    # of all matching segments)
    assert seg.get_segment_numbers(segmented_property_type=codes.SCT.Bone)) == [1]
    assert seg.get_segment_numbers(segmented_property_type=codes.SCT.Spine)) == [2]

    # Search for segments by tracking UID (returns segment numbers of all
    # matching segments)
    assert seg.get_segment_numbers(tracking_uid='1.2.826.0.1.3680043.10.511.3.83271046815894549094043330632275067')) == [1]
    assert seg.get_segment_numbers(tracking_uid='1.2.826.0.1.3680043.10.511.3.10042414969629429693880339016394772')) == [2]

    # You can also get the full description for a given segment, and access
    # the information in it via properties
    segment_1_description = seg.get_segment_description(1)
    assert segment_1_description.segment_label) == 'first segment'
    assert segment_1_description.tracking_uid) == '1.2.826.0.1.3680043.10.511.3.83271046815894549094043330632275067'


Reconstructing Segmentation Masks From DICOM SEGs
-------------------------------------------------

`Highdicom` provides the
:meth:`highdicom.seg.Segmentation.get_pixels_by_source_instance()` and
:meth:`highdicom.seg.Segmentation.get_pixels_by_source_frame()` methods to
handle reconstruction of segmentation masks from SEG objects in which each
frame in the SEG object is derived from a single source frame. The only
difference between the two methods is that the
:meth:`highdicom.seg.Segmentation.get_pixels_by_source_instance()` is used when
the segmentation is derived from a source series consisting of multiple
single-frame instances, while
:meth:`highdicom.seg.Segmentation.get_pixels_by_source_frame()` is used when
the segmentation is derived from a single multiframe source instance.

When reconstructing a segmentation mask using
:meth:`highdicom.seg.Segmentation.get_pixels_by_source_instance()`, the user
must provide a list of SOP Instance UIDs of the source images for which the
segmentation mask should be constructed. Whatever order is chosen here will be
used to order the frames of the output segmentation mask, so it is up to the
user to sort them according to their needs. The default behavior is that the
output pixel array is of shape (*F* x *H* x *W* x *S*), where *F* is the number
of source instance UIDs, *H* and *W* are the height and width of the frames,
and *S* is the number of segments included in the segmentation. In this way,
the output of this method matches the "stacked segments" format `pixel_array`
to the constructor that would create the SEG object if it were created with
`highdicom`. This behavior is consistent for segmentations stored with
``"BINARY"`` and ``"LABELMAP"`` segmentation types, even though the underlying
format in which the arrays are stored differs. In the case of ``"LABELMAP"``
segmentations, this means that *highdicom* actually splits apart the different
segments from the stored labelmap.

The following example (and those in later sections) use DICOM files from the
`highdicom` test data, which may be found in the
`highdicom repository <https://github.com/ImagingDataCommons/highdicom/tree/master/data/test_files>`_
on GitHub.

.. code-block:: python

    import numpy as np
    import highdicom as hd

    seg = hd.seg.segread('data/test_files/seg_image_ct_binary.dcm')

    # List the source images for this segmentation:
    for study_uid, series_uid, sop_uid in seg.get_source_image_uids():
        print(sop_uid)
    # 1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93
    # 1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.94
    # 1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.95
    # 1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.96

    # Get the segmentation array for a subset of these images:
    pixels = seg.get_pixels_by_source_instance(
        source_sop_instance_uids=[
            '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93',
            '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.94'
        ]
    )
    assert pixels.shape == (2, 16, 16, 1)
    assert np.unique(pixels).tolist() == [0, 1]

This second example demonstrates reconstructing segmentation masks from a
segmentation derived from a multiframe image, in this case a whole slide
microscopy image, and also demonstrates an example with multiple, in
this case 20, segments:

.. code-block:: python

    import highdicom as hd

    # Read in the segmentation using highdicom
    seg = hd.seg.segread('data/test_files/seg_image_sm_numbers.dcm')

    assert seg.number_of_segments == 20

    # SOP Instance UID of the single multiframe image from which the
    # segmentation was derived
    _, _, source_sop_instance_uid = seg.get_source_image_uids()[0]

    # Get the segmentation array for a subset of these images:
    pixels = seg.get_pixels_by_source_frame(
        source_sop_instance_uid=source_sop_instance_uid,
        source_frame_numbers=range(1, 26),
    )

    # Source frames are stacked down the first dimension, segments are stacked
    # down the fourth dimension
    assert pixels.shape == (25, 10, 10, 20)

    # Each segment is still binary
    assert np.unique(pixels).tolist() == [0, 1]

Note that these two methods may only be used when the segmentation's metadata
indicates that each segmentation frame is derived from exactly one source
instance or frame of a source instance. If this is not the case, a
``RuntimeError`` is raised.

In the general case, the
:meth:`highdicom.seg.Segmentation.get_pixels_by_dimension_index_values()` method
is available to query directly by the underlying dimension index values. We
will not cover this advanced topic.

Reconstructing Specific Segments
--------------------------------

A further optional parameter, ``segment_numbers``, allows the user to request
only a subset of the segments available within the SEG object by providing a
list of segment numbers. In this case, the output array will have a dimension
equal to the number of segments requested, with the segments stacked in the
order they were requested (which may not be ascending by segment number).

.. code-block:: python

    import highdicom as hd

    # Read in the segmentation using highdicom
    seg = hd.seg.segread('data/test_files/seg_image_sm_numbers.dcm')

    assert seg.number_of_segments == 20

    # SOP Instance UID of the single multiframe image from which the
    # segmentation was derived
    _, _, source_sop_instance_uid = seg.get_source_image_uids()[0]

    # Get the segmentation array for a subset of these images:
    pixels = seg.get_pixels_by_source_frame(
        source_sop_instance_uid=source_sop_instance_uid,
        source_frame_numbers=range(1, 26),
        assert_missing_frames_are_empty=True,
        segment_numbers=[10, 9, 8]
    )

    # Source frames are stacked down the first dimension, segments are stacked
    # down the fourth dimension
    assert pixels.shape == (25, 10, 10, 3)

After this, the array ``pixels[:, :, :, 0]`` contains the pixels for segment
number 10, ``pixels[:, :, :, 1]`` contains the pixels for segment number 9, and
``pixels[:, :, :, 2]`` contains the pixels for segment number 8.

Reconstructing Segmentation Masks as "Label Maps"
-------------------------------------------------

If the segments do not overlap, it is possible to combine the multiple segments
into a simple "label map" style mask, as described above. This can be achieved
by specifying the ``combine_segments`` parameter as ``True``. In this case, the
output will have shape (*F* x *H* x *W*), and a pixel value of *i > 0*
indicates that the pixel belongs to segment *i* or a pixel value of 0
represents that the pixel belongs to none of the requested segments. Again,
this mirrors the way you would have passed this segmentation mask to the
constructor to create the object if you had used a label mask. If the segments
overlap, `highdicom` will raise a ``RuntimeError``. Alternatively, if you
specify the ``skip_overlap_checks`` parameter as ``True``, no error will be
raised and each pixel will be given the value of the highest segment number of
those present in the pixel (or the highest segment value after relabelling has
been applied if you pass ``relabel=True``, see below).

Note that combining segments is only possible when either:

* The segmentation type is ``"LABELMAP"`` or ``"BINARY"``
* The segmentation type is ``"FRACTIONAL"`` but the only two values are actually
  present in the image.

For ``"LABELMAP"`` segmentations, using the ``combine_segments`` option is
actually just returning the stored frames, and will therefore be more efficient
that the default behavior.

Here, we repeat the above example but request the output as a label map:

.. code-block:: python

    import highdicom as hd

    # Read in the segmentation using highdicom
    seg = hd.seg.segread('data/test_files/seg_image_sm_numbers.dcm')

    # SOP Instance UID of the single multiframe image from which the
    # segmentation was derived
    _, _, source_sop_instance_uid = seg.get_source_image_uids()[0]

    # Get the segmentation array for a subset of these images:
    pixels = seg.get_pixels_by_source_frame(
        source_sop_instance_uid=source_sop_instance_uid,
        source_frame_numbers=range(1, 26),
        assert_missing_frames_are_empty=True,
        segment_numbers=[10, 9, 8],
        combine_segments=True,
    )

    # Source frames are stacked down the first dimension, now there is no
    # fourth dimension
    assert pixels.shape == (25, 10, 10)

    assert np.unique(pixels).tolist() == [0, 8, 9, 10]

In the default behavior, the pixel values of the output label map correspond to
the original segment numbers to which those pixels belong. Therefore we see
that the output array contains values 8, 9, and 10, corresponding to the three
segments that we requested (in addition to 0, meaning no segment). However,
when you are specifying a subset of segments, you may wish to "relabel" these
segments such that in the output array the first segment you specify (10 in the
above example) is indicated by pixel value 1, the second segment (9 in the
example) is indicated by pixel value 2, and so on. This is achieved using
the ``relabel`` parameter.

.. code-block:: python

    import highdicom as hd

    # Read in the segmentation using highdicom
    seg = hd.seg.segread('data/test_files/seg_image_sm_numbers.dcm')

    # SOP Instance UID of the single multiframe image from which the
    # segmentation was derived
    _, _, source_sop_instance_uid = seg.get_source_image_uids()[0]

    # Get the segmentation array for a subset of these images:
    pixels = seg.get_pixels_by_source_frame(
        source_sop_instance_uid=source_sop_instance_uid,
        source_frame_numbers=range(1, 26),
        assert_missing_frames_are_empty=True,
        segment_numbers=[10, 9, 8],
        combine_segments=True,
        relabel=True,
    )

    # Source frames are stacked down the first dimension, now there is no
    # fourth dimension
    assert pixels.shape == (25, 10, 10)

    # Now the output segments have been relabelled to 1, 2, 3
    assert np.unique(pixels).tolist() == [0, 1, 2, 3]

Reconstructing Fractional Segmentations
---------------------------------------

For ``"FRACTIONAL"`` SEG objects, `highdicom` will rescale the pixel values in
the segmentation masks from the integer values as which they are stored back
down to the range `0.0` to `1.0` as floating point values by scaling by the
"MaximumFractionalValue" attribute. If desired, this behavior can be disabled
by specifying ``rescale_fractional=False``, in which case the raw integer array
as stored in the SEG will be returned.

.. code-block:: python

    import numpy as np
    import highdicom as hd

    # Read in the segmentation using highdicom
    seg = hd.seg.segread('data/test_files/seg_image_ct_true_fractional.dcm')

    assert seg.segmentation_type == hd.seg.SegmentationTypeValues.FRACTIONAL

    # List the source images for this segmentation:
    sop_uids = [uids[2] for uids in seg.get_source_image_uids()]

    # Get the segmentation array for a subset of these images:
    pixels = seg.get_pixels_by_source_instance(
        source_sop_instance_uids=sop_uids,
    )

    # Each segment values are now floating point
    assert pixels.dtype == np.float32

    print(np.unique(pixels))
    # [0.        0.2509804 0.5019608]

Reconstructing Volumes
----------------------

If the segmentation is defined on a regularly-sampled 3D grid (possibly with
omittted frames, tiled frames, and/or multiple segments), the
:meth:`highdicom.seg.Segmentation.get_volume()` method may be used to create a
:class:`highdicom.Volume` from its frames. The options we have already seen
(``segment_numbers``, combine_segments``, ``relabel``, ``rescale_fractional``)
are also available here.

.. code-block:: python

    import highdicom as hd


    # This is a test file in the highdicom git repository
    seg = hd.seg.segread('data/test_files/seg_image_ct_binary.dcm')

    vol = seg.get_volume(combine_segments=True)
    print(vol.spatial_shape)
    # (3, 16, 16)

    print(vol.affine)
    # [[   0.          0.          0.488281 -125.      ]
    # [   0.          0.488281    0.       -128.100006]
    # [  -1.25        0.          0.        105.519997]
    # [   0.          0.          0.          1.      ]]


Reconstructing Total Pixel Matrices from Tiled Segmentations
------------------------------------------------------------

For segmentations of digital pathology images that are stored as tiled images,
the :meth:`highdicom.seg.Segmentation.get_pixels_by_source_frame()` method will
return the segmentation mask as a set of frames stacked down the first
dimension of the array. However, for such images, you typically want to work
with the large 2D total pixel matrix that is formed by correctly arranging the
tiles into a 2D array. `highdicom` provides the
:meth:`highdicom.seg.Segmentation.get_total_pixel_matrix()` method for this
purpose.

Called without any parameters, it returns a 3D array containing the full total
pixel matrix. The first two dimensions are the spatial dimensions, and the
third is the segments dimension. Behind the scenes highdicom has stitched
together the required frames stored in the original file for you. Like with the
other methods described above, setting ``combine_segments`` to ``True``
combines all the segments into, in this case, a 2D array.

.. code-block:: python

    import highdicom as hd

    # Read in the segmentation using highdicom
    seg = hd.seg.segread('data/test_files/seg_image_sm_control.dcm')

    # Get the full total pixel matrix
    mask = seg.get_total_pixel_matrix()

    expected_shape = (
        seg.TotalPixelMatrixRows,
        seg.TotalPixelMatrixColumns,
        seg.number_of_segments,
    )
    assert mask.shape == expected_shape

    # Combine the segments into a single array
    mask = seg.get_total_pixel_matrix(combine_segments=True)

    assert mask.shape == (seg.TotalPixelMatrixRows, seg.TotalPixelMatrixColumns)

Furthermore, you can request a sub-region of the full total pixel matrix by
specifying the start and/or stop indices for the rows and/or columns within the
total pixel matrix. Note that this method follows DICOM 1-based convention for
indexing rows and columns, i.e. the first row and column of the total pixel
matrix are indexed by the number 1 (not 0 as is common within Python). Negative
indices are also supported to index relative to the last row or column, with -1
being the index of the last row or column. Like for standard Python indexing,
the stop indices are specified as one beyond the final row/column in the
returned array. Note that the requested region does not have to start or stop
at the edges of the underlying frames: `highdicom` stitches together only the
relevant parts of the frames to create the requested image for you.

.. code-block:: python

    import highdicom as hd

    # Read in the segmentation using highdicom
    seg = hd.seg.segread('data/test_files/seg_image_sm_control.dcm')

    # Get a region of the total pixel matrix
    mask = seg.get_total_pixel_matrix(
        combine_segments=True,
        row_start=20,
        row_end=40,
        column_start=10,
        column_end=20,
    )

    assert mask.shape == (20, 10)

    # A further example using negative indices. Since row_end is not provided,
    # the default behavior is to include the last row in the total pixel matrix.
    mask = seg.get_total_pixel_matrix(
        combine_segments=True,
        row_start=21,
        column_start=-30,
        column_end=-25,
    )

    assert mask.shape == (30, 5)

Viewing DICOM SEG Images
------------------------

Unfortunately, DICOM SEG images are not widely supported by DICOM
viewers. Viewers that do support SEG include:

- The `OHIF Viewer <https://github.com/OHIF/Viewers>`_, an open-source
  web-based viewer.
- `3D Slicer <https://www.slicer.org/>`_, an open-source desktop application
  for 3D medical image computing. It supports both display and creation of
  DICOM SEG files via the "Quantitative Reporting" plugin.

Note that these viewers may not support all features of segmentation images
that `highdicom` is able to encode.
