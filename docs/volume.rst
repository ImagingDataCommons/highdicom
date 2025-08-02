.. _volume:

Volumes
=======

Another fundamental class in highdicom is :class:`highdicom.Volume`. Unlike
many other highdicom classes, a volume is not a DICOM object at all. Instead it
provides a means to work with regularly-sampled volumetric arrays conveniently.
Not all DICOM images represent regularly-sampled volumetric data, but a
significant proportion do, particularly many CT, MRI, and PET images. The
:class:`highdicom.Volume` class provides a means to work with these arrays,
derive results (such as segmentations or parametric maps) from them, and store
those results in new DICOM files with the correct spatial metadata without
having to write code to handle spatial metadata.

A Volume object has two core components: an array of voxels with three or more
dimensions, and an affine matrix that describes how that array is positioned in
3D space within the relevant frame-of-reference coordinate system (either the
patient or slide coordinate system).

Creating a Basic Volume
-----------------------

To create a basic volume we need to provide the affine matrix and the array, as
well as specifying the coordinate system being used.

.. code-block:: python

    import numpy as np
    import highdicom as hd


    array = np.zeros((32, 256, 256), dtype=np.uint8)
    affine = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 0.4, 0.0, 20.0],
            [0.0, 0.0, 0.4, -5.0],
            [0.0, 0.0, 0.0,  1.0],
        ]
    )

    vol = hd.Volume(
        array=array,
        affine=affine,
        coordinate_system="PATIENT",
    )

    # Once created, you can access the array and the affine via properties
    assert np.array_equal(vol.array, array)
    assert np.array_equal(vol.affine, affine)

    # The datatype of the array, its shape, and the spatial shape (not
    # including any channels, see below), may be accessed via
    # properties
    print(vol.dtype)
    # uint8

    print(vol.shape)
    # (32, 256, 256)

    print(vol.spatial_shape)
    # (32, 256, 256)

Affine Matrices
---------------

The affine matrix is a 4x4 numpy array that can be used to map (zero-based)
array indices into the position of the center of the indexed voxel in the
frame-of-reference coordinate system.

Within highdicom, the affine matrices used within volumes are always stored
using the "LPS" convention for describing the frame of reference, since this is
the convention used within DICOM itself. Affine matrices defined in other
conventions need to be converted before being used to describe a volume.
Another common convention in the "RAS" convention used within NIfTI files. If
you have an affine matrix generated elsewhere using a different convention, you
can specify the ``from_reference_convention`` parameter to specify the
convention used by the source affine, and highdicom will handle the conversion
to "LPS" convention for you.

The top-left 3x3 matrix of the affine matrix consists of three orthogonal
column vectors that give the vector travelled when incrementing the voxel
indices in each o the three dimensions. The top three elements of the last
column give the position of the voxel at index ``(0, 0, 0)``. The bottom row is
always ``[0., 0., 0., 1.]``. The :meth:`highdicom.Volume.from_components()`
method allows you to construct a volume by specifying these components of the
affine matrix. Furthermore, once constructed you can use properties and methods
of the object to access these components individually.

This code snippet produces the same volume as in the previous that specified
the full affine matrix.

.. code-block:: python

    import numpy as np
    import highdicom as hd


    array = np.zeros((32, 256, 256), dtype=np.uint8)
    direction = np.eye(3)
    spacing = np.array([1.0, 0.4, 0.4])
    position = np.array([10.0, 20.0, -5.0])

    vol = hd.Volume.from_components(
        array=array,
        direction=direction,
        spacing=spacing,
        position=position,
        coordinate_system="PATIENT",
    )

    print(vol.direction)
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]]

    print(vol.spacing)
    # (1.0, 0.4, 0.4)

    print(vol.position)
    # (10.0, 20.0, -5.0)

    print(vol.center_position)
    # (25.5, 71.0, 46.0)

    print(vol.unit_vectors())
    # (array([1., 0., 0.]), array([0., 1., 0.]), array([0., 0., 1.]))

Volume Geometries
-----------------

Sometimes it is useful to work with the affine matrix of a volume without the
full voxel array. The :class:`highdicom.VolumeGeometry` fills this role. It has
an API that is compatible with :class:`highdicom.Volume`, except for any
operations that require access to the pixel data.

.. code-block:: python

    import numpy as np
    import highdicom as hd


    direction = np.eye(3)
    spacing = np.array([1.0, 0.4, 0.4])
    position = np.array([10.0, 20.0, -5.0])

    geometry = hd.VolumeGeometry.from_components(
        spatial_shape=(32, 256, 256),
        direction=direction,
        spacing=spacing,
        position=position,
        coordinate_system="PATIENT",
    )

    print(geometry.direction)
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]]

    print(geometry.spacing)
    # (1.0, 0.4, 0.4)

    print(geometry.position)
    # (10.0, 20.0, -5.0)

    print(geometry.center_position)
    # (25.5, 71.0, 46.0)

    print(geometry.unit_vectors())
    # (array([1., 0., 0.]), array([0., 1., 0.]), array([0., 0., 1.]))

Volumes From Images
-------------------

Volumes are loaded from existing images more often than constructed
directly. To load a volume from a single DICOM image (single frame or
multi-frame), use the :meth:`highdicom.Image.get_volume()` method (see
:doc:`image`).

.. code-block:: python

    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load an enhanced (multiframe) CT image
    im = hd.imread(get_testdata_file('eCT_Supplemental.dcm'))

    geometry = im.get_volume_geometry()

    assert geometry is not None

    vol = im.get_volume()
    print(vol.spatial_shape)
    # (2, 512, 512)

    print(vol.affine)
    # [[   0.          0.         -0.388672   99.5     ]
    #  [  -0.          0.388672    0.       -301.5     ]
    #  [  10.          0.          0.       -159.      ]
    #  [   0.          0.          0.          1.      ]]

Even if the image consists of a single plane, the resulting Volume will have
three spatial dimensions and the singleton dimension is placed first.

Volumes From Image Series
-------------------------

In the case where the frames that make up a volume are stored across multiple,
single-frame files from a series, the
:func:`highdicom.get_volume_from_series()` function may be used to create a
volume.

.. code-block:: python

    import pydicom
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Three test files from pydicom that form a volume
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136'),
        get_testdata_file('dicomdirtests/77654033/CT2/17196'),
        get_testdata_file('dicomdirtests/77654033/CT2/17166'),
    ]
    ct_series = [pydicom.dcmread(f) for f in ct_files]

    vol = get_volume_from_series(ct_series)

Array Manipulation
------------------

Since the volume's array is just a NumPy array, it can be manipulated just like
any other numpy array to process the image. However, any operation that changes
the array's shape is not allowed because changing the shape requires changing
the affine matrix.

.. code-block:: python

    import numpy as np
    import highdicom as hd


    vol = hd.Volume.from_components(
        array=np.random.randint(0, 100, size=(32, 256, 256), dtype=np.uint8),
        direction=np.eye(3),
        spacing=[1., 0.4, 0.4],
        position=[10., 20., 30.],
        coordinate_system="PATIENT",
    )

    # OK
    vol.array = vol.array + 10.0

    # OK
    vol.array /= 100

    # OK
    vol.array = np.exp(vol.array / 1000)

    # Disallowed, changes shape
    vol.array = vol.array[:10]

The above operations edit the volume in-place. If you want to create a new
volume with a new array but the same geometry as an existing volume, use the
:meth:`highdicom.Volume.with_array()` method.

Indexing
--------

Volumes can be indexed along their spatial dimensions using square brackets in
a largely similar way to any NumPy array. This operation crops the array and
also updates the affine matrix to reflect the effect of the crop. However,
there is one important change: spatial dimensions can be reduced to size one
but never removed by indexing (volumes always have three spatial dimensions).

.. code-block:: python

    import numpy as np
    import highdicom as hd


    vol = hd.Volume.from_components(
        array=np.random.randint(0, 100, size=(32, 256, 256), dtype=np.uint8),
        direction=np.eye(3),
        spacing=[1., 0.4, 0.4],
        position=[10., 20., 30.],
        coordinate_system="PATIENT",
    )

    cropped = vol[:10]
    print(cropped.shape)
    # (10, 256, 256)

    cropped = vol[10]
    print(cropped.shape)
    # (1, 256, 256)

    cropped = vol[:, 20:100, -80:]
    print(cropped.shape)
    # (32, 80, 80)

    cropped = vol[:, :, 200:120:-1]
    print(cropped.shape)
    # (32, 256, 80)

Spatial Operations
------------------

The :class:`highdicom.Volume` class provides a number of spatial operations
that manipulate the array and correctly update the affine matrix to reflect the
change. Currently these only include operations that do not require resampling
of the array:

* :meth:`highdicom.Volume.crop_to_spatial_shape()`, center-crops to a given
  spatial shape.
* :meth:`highdicom.Volume.flip_spatial()`, flips along certain axes.
* :meth:`highdicom.Volume.match_geometry()`, given a second volume (or volume
  geometry) manipulate the volume by axis permutations, flips, crops and/or
  pads (but no resampling) to match the geometry of the first volume to that of
  the second volume.
* :meth:`highdicom.Volume.pad()`, pads the array along spatial dimensions.
* :meth:`highdicom.Volume.pad_to_spatial_shape()`, pad to a given spatial
  shape.
* :meth:`highdicom.Volume.pad_or_crop_to_spatial_shape()`, ensures a given
  spatial shape via padding and/or center cropping.
* :meth:`highdicom.Volume.permute_spatial_axes()`, permute (transpose) the
  array dimensions.
* :meth:`highdicom.Volume.random_flip_spatial()`, randomly flip one or more
  spatial axes.
* :meth:`highdicom.Volume.random_permute_spatial_axes()`, randomly permute
  (transpose) the array dimensions.
* :meth:`highdicom.Volume.random_spatial_crop()`, randomly generate a crop of a
  given size.

Patient Orientation
-------------------

For volumes in the patient frame-of-reference coordinate system, the "patient
orientation" describes how the axes of the volume align with the axes of the
patient coordinate system, which are defined from left-to-right,
anterior-to-posterior, and foot-to-head, in that order. The axes of the volume
do not need to be exactly aligned with the frame-of-reference axes to be
described using a patient orientation; if they are not the closest match is
used. For example, the patient orientation "FPL" means that the first axis of
the volume is most closely aligned with the head-to-foot direction, the second
axis of the volume is most closely aligned with the anterior-to-posterior
direction, and the third axis is most closely aligned with the right-to-left
direction.

Patient orientations may be used to describe a volume, and the
:meth:`highdicom.Volume.to_patient_orientation()` is used to manipulate
a volume to align with the given patient orientation as well as possible via
permutations and flips.

Patient orientations may be represented as strings or as tuples of the
:class:`highdicom.PatientOrientationValuesBiped` class.

.. code-block:: python

    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load an enhanced (multiframe) CT image
    im = hd.imread(get_testdata_file('eCT_Supplemental.dcm'))

    vol = im.get_volume()

    print(vol.get_closest_patient_orientation())
    # (<PatientOrientationValuesBiped.H: 'H'>, <PatientOrientationValuesBiped.P: 'P'>, <PatientOrientationValuesBiped.R: 'R'>)

    vol = vol.to_patient_orientation("LAF")

    print(vol.get_closest_patient_orientation())
    # (<PatientOrientationValuesBiped.L: 'L'>, <PatientOrientationValuesBiped.A: 'A'>, <PatientOrientationValuesBiped.F: 'F'>)

Channels
--------

In addition to the three spatial dimensions, a volume may have further
non-spatial dimensions that are referred to as "channels". Channel dimensions
are stacked after the spatial dimensions in the volume's pixel array. The
meaning of each channel is explicitly described in the volume. Common uses for
channels include RGB channels in color images, optical paths in microscopy
images, or contrast phases in radiology images.

The :class:`highdicom.ChannelDescriptor` class is used to describe the meaning
of a single channel dimension. Where possible, it is recommended to use DICOM
attributes to describe channels. A DICOM keyword or the corresponding tag value
may be passed to the :class:`highdicom.ChannelDescriptor` constructor.

When using a DICOM attribute, each channel of the volume is associated with a
particular value for that attribute. For example, if the descriptor uses the
"OpticalPathIdentifier" attribute, each channel will be associated with a
string. Alternatively if an integer-valued attribute like "SegmentNumber" is
used, each channel will be associated with an integer. We refer to this type as
the descriptor's "value type".

This code snippet creates channel descriptors using some DICOM attribute, and
checks the corresponding value types:

.. code-block:: python

    import highdicom as hd


    # Channel descriptor using the "OpticalPathIdentifier"
    optical_path_descriptor = hd.ChannelDescriptor('OpticalPathIdentifier')

    # Using the hexcode for the attribute is equivalent
    optical_path_descriptor = hd.ChannelDescriptor(0x0048_0106)

    # Channel descriptor using the "DiffusionBValue"
    bvalue_descriptor = hd.ChannelDescriptor('DiffusionBValue')

    # Check that the value types are as expected
    print(optical_path_descriptor.value_type)
    # <class 'str'>

    print(bvalue_descriptor.value_type)
    # <class 'float'>

Alternatively, it is possible to define custom identifiers that do not use a
DICOM attribute. In this case, you must specify the value type yourself. The
value type must be either ``int``, ``str``, or ``float`` (or a sub-type of one
of these types), or an enumerated type derived from the Python standard library
``enum.Enum``.

.. code-block:: python

   from enum import Enum
   import highdicom as hd

   # A custom descriptor using integer values
   custom_int_descriptor = hd.ChannelDescriptor(
       'my_int_descriptor',
       is_custom=True,
       value_type=int,
   )

   # A custom descriptor using an enumerated type
   class MyEnum(Enum):
       VALUE1 = "VALUE1"
       VALUE2 = "VALUE2"

   custom_enum_descriptor = hd.ChannelDescriptor(
       'my_enum_descriptor',
       is_custom=True,
       value_type=MyEnum,
   )

One very common channel descriptor that does not correspond to a DICOM
attribute is RGB color channels. The enum :class:`highdicom.RGBColorChannels`
is used as the value type for volumes with color channels, and the descriptor
for this channel is provided as a constant in
``highdicom.RGB_COLOR_CHANNEL_DESCRIPTOR``.

To create a volume with channels, you must provide a dictionary that contains,
for each channel dimension, the channel descriptor and the values of each
channel along that dimension:

.. code-block:: python

    import numpy as np
    import highdicom as hd

    # Array with three spatial dimensions plus 3 color channels and 4 optical
    # paths
    array = np.random.randint(0, 10, size=(1, 50, 50, 3, 4))

    # Names of the 4 optical paths
    path_names = ['path1', 'path2', 'path3', 'path4']

    vol = hd.Volume.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[2.0, 0.5, 0.5],
        coordinate_system="SLIDE",
        array=array,
        channels={
            hd.RGB_COLOR_CHANNEL_DESCRIPTOR: ['R', 'G', 'B'],
            'OpticalPathIdentifier': path_names
        },
    )

    # The total shape of the volume includes the channel dimensions
    assert vol.shape == (1, 50, 50, 3, 4)

    # But the spatial shape excludes them
    assert vol.spatial_shape == (1, 50, 50)

    # The channel shape includes only the channel dimensions, not the spatial
    # dimensions
    assert vol.channel_shape == (3, 4)
    assert vol.number_of_channel_dimensions == 2

    # You can access the descriptors like this
    assert vol.channel_descriptors == (
        hd.RGB_COLOR_CHANNEL_DESCRIPTOR,
        hd.ChannelDescriptor('OpticalPathIdentifier'),
    )

The order of the items in the dictionary is significant and must match the
order of the channel dimensions in the array.

For most purposes, a volume with channels can be treated just like one without.
All spatial operations (including indexing) only alter the array along the
spatial dimensions and leave the channel dimensions unchanged. A separate set
of methods are used to alter the channel dimensions:

* :meth:`highdicom.Volume.get_channel()`: Get a new volume containing just one
  channel of the original volume for a given channel value.
* :meth:`highdicom.Volume.get_channel_values()`: Get the channel values for a
  given channel dimension.
* :meth:`highdicom.Volume.permute_channel_axes()`: Permute the channels
  dimensions to a given order specified by the descriptors.
* :meth:`highdicom.Volume.permute_channel_axes_by_index()`: Permute the channel
  dimensions to a given order specified by the channel dimension index.

This snippet, using the same volume as above, demonstrates how to use these
methods:

.. code-block:: python

    import numpy as np
    import highdicom as hd

    # Array with three spatial dimensions plus 3 color channels and 4 optical
    # paths
    array = np.random.randint(0, 10, size=(1, 50, 50, 3, 4))

    # Names of the 4 optical paths
    path_names = ['path1', 'path2', 'path3', 'path4']

    vol = hd.Volume.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[2.0, 0.5, 0.5],
        coordinate_system="SLIDE",
        array=array,
        channels={
            hd.RGB_COLOR_CHANNEL_DESCRIPTOR: ['R', 'G', 'B'],
            'OpticalPathIdentifier': path_names
        },
    )

    assert (
        vol.get_channel_values('OpticalPathIdentifier') ==
        path_names
    )

    # Get a new volume containing just optical path 'path2'
    path_2_vol = vol.get_channel(OpticalPathIdentifier='path2')

    # Swap the two channel axes by descriptor
    permuted_vol = vol.permute_channel_axes(
        ['OpticalPathIdentifier', 'RGBColorChannel']
    )

    # Swap the two channel axes by index
    permuted_vol = vol.permute_channel_axes_by_index([1, 0])

Full Example
------------

This full example presents a typical workflow of how volumes are used within
highdicom. First, a volume is extracted from an existing image. Then it is
manipulated to prepare it for some automated analysis tool (a simple example
segmentation in this case). The tool's output is placed back into a volume,
which is then passed to the constructor of a highdicom class to ensure that the
spatial metadata in the output object is correct.

.. code-block:: python

    import numpy as np

    from pydicom.sr.codedict import codes
    from pydicom import pixel_array
    from pydicom.data import get_testdata_file
    from pydicom.uid import JPEGLSLossless

    import highdicom as hd


    def complex_segmentation_tool(arr: np.ndarray) -> np.ndarray:
        """This is a stand-in for a generic segmentation tool.

        We assume that the tool has certain requirements on the input array, in
        this case that it has patient orientation "FLP" and a shape of (2, 400,
        400).

        Further, we assume that the tool takes in a numpy array and returns a
        binary segmentation that is pixel-for-pixel aligned with its input array
        (i.e. the tool itself does not do any further spatial manipulation.

        """
        # Basic thresholding as a simple example
        return arr > 0

    # Load an enhanced (multiframe) CT image
    im = hd.imread(get_testdata_file('eCT_Supplemental.dcm'))

    # Load the input volume
    original_volume = im.get_volume()

    # Manipulate the original volume to give a suitable input for the tool
    input_volume = (
        original_volume
        .to_patient_orientation("FLP")
        .crop_to_spatial_shape((2, 400, 400))
    )

    # Run the "complex segmentation tool"
    seg_array = complex_segmentation_tool(input_volume.array)

    # Since the seg array shares its geometry with the inupt array, we can combine
    # the two to create a volume of the segmentation array
    seg_volume = input_volume.with_array(seg_array)

    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='Complex Segmentation Tool',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # metadata needed for a segmentation
    brain_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='brain',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Brain,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
    )

    # Use the segmentation volume as input to create a DICOM Segmentation
    seg_dataset = hd.seg.Segmentation(
        pixel_array=seg_volume,
        source_images=[im],
        segmentation_type=hd.seg.SegmentationTypeValues.LABELMAP,
        segment_descriptions=[brain_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Complex Segmentations Plc.',
        manufacturer_model_name='Complex Segmentation Tool',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        transfer_syntax_uid=JPEGLSLossless,
        series_description='Example Segmentation of CT',
    )

    seg_dataset.save_as('segmentation.dcm')

    # Alternatively, it may be desirable to match the geometry of the output
    # segmentation image to that of the input image. This will "undo" the
    # cropping and axis permutation operations done to the image volume above.
    seg_volume_matched = seg_volume.match_geometry(original_volume)

    # Use the segmentation volume as input to create a DICOM Segmentation
    seg_dataset_matched = hd.seg.Segmentation(
        pixel_array=seg_volume_matched,
        source_images=[im],
        segmentation_type=hd.seg.SegmentationTypeValues.LABELMAP,
        segment_descriptions=[brain_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Complex Segmentations Plc.',
        manufacturer_model_name='Complex Segmentation Tool',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        transfer_syntax_uid=JPEGLSLossless,
        series_description='Example Segmentation of CT',
    )

    seg_dataset_matched.save_as('segmentation_matched.dcm')

Volumes To/From NIfTI Files
---------------------------

`NIfTI`_ is a file format used to store volumetric imaging data. It arose from
neuro-imaging but is now used in other areas of radiology and beyond. When
converting between highdicom Volumes and NIfTI files, it is critical to
remember to account for the difference in convention used to specify the
frame-of-reference coordinate system: highdicom (and DICOM) uses "LPS"
convention, NIfTI uses "RAS" convention.

We plan to add tools to handle this conversion in the near future, but for now
these snippets should correctly handle simple situations converting to and from
NIfTI using the `nibabel`_ package.

Reading a volume from a NIfTI:

.. code-block:: python

   import nibabel as nib
   import highdicom as hd


   nifti_path = '/path/to/nifti.nii'  # or .nii.gz
   nifti = nib.load(nifti_path)

   vol = hd.Volume(
       array=nifti.get_fdata(),
       affine=nifti.affine,
       coordinate_system="PATIENT",
       from_reference_convention='RAS',
   )

Writing a volume to a NIfTI file:

.. code-block:: python

    import nibabel
    import highdicom as hd


    vol = hd.Volume(...)

    nifti = nib.Nifti1Image(
        vol.array,
        vol.get_affine('RAS'),
    )

    nifti_path = '/path/to/nifti.nii'  # or .nii.gz
    nib.save(nifti, nifti_path)

Volumes To/From ITK Images
--------------------------

`ITK`_ is a widely-used library for volumetric image processing. Its ``Image``
class shares many similarities with our :class:`highdicom.Volume` class. Like
highdicom, ITK uses the "LPS" convention. However, when converting to and from
NumPy arrays, ITK reverses the order of dimensions. It is important to account
for this when performing conversions.

We plan to add tools to handle this conversion in the near future, but for now
these snippets should correctly handle simple situations converting to and from
ITK Images.

Creating a volume from an ITK Image:

.. code-block:: python

    import itk
    import numpy as np
    import highdicom as hd


    im = itk.image(...)

    # Reverse array dimension order
    array = itk.array_from_image(im).transpose([2, 1, 0])

    vol2 = hd.Volume.from_components(
        array=array,
        direction=np.asarray(im.GetDirection()),
        spacing=np.asarray(im.GetSpacing()),
        position=np.asarray(im.GetOrigin()),
        coordinate_system="PATIENT"
    )

Creating an ITK Image from a Volume:

.. code-block:: python

    import itk
    import highdicom as hd


    vol = hd.Volume(...)

    # Reverse array dimension order
    array = vol.array.transpose([2, 1, 0])

    im = itk.image_from_array(array)
    im.SetOrigin(vol.position)
    im.SetDirection(vol.direction)
    im.SetSpacing(vol.spacing)


.. _`NIfTI`: https://nifti.nimh.nih.gov/
.. _`ITK`: https://itk.org/
.. _`nibabel`: https://nipy.org/nibabel/
