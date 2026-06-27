.. _pm:

Parametric Maps
===============

DICOM Parametric Maps store pixel arrays of measurements derived from other
images. Example applications of Parametric Maps include storing maps of
parameters (e.g. K-trans) derived from multiple MRI sequences via a
mathematical model and storing saliency maps produced by a neural network.

An important advantage of Parametric Metric maps over all other DICOM objects
that store pixel data is that they can store pixels in floating point format
using either 32-bit (single precision) or 64-bit (double precision) floats.
Signed and unsigned integer pixel values are also supported. Parametric Maps
always use a single sample per pixel (i.e. they are grayscale rather than color
images), but a palette color LUTs may be included to display the image as
color. They also include mechanisms that describe the physical quantity the
image represents.

Like Segmentations (see :ref:`seg`), Parametric Maps are multi-frame DICOM
objects. The multiple frames in a parametric map can store slices of a
regularly-sampled 3D volume (common in Parametric Maps derived from CT, PET, or
MRI) or tiles of a very large 2D plane (common in digital pathology
applications), among other arrangements.

Parametric Maps are always scalar (i.e. monochome not color) images. However
it is possible to specify a palette color LUT transformation in order to
have the scalar arrays render in color. Further multiple "channels" of
measurements can be included in one Parametric Map objects.

Highdicom uses the :class:`highdicom.pm.ParametricMap` to represent Parametric
Maps.

Real World Value Mappings
-------------------------

Each parametric map requires one (or more) *real-world value mappings*. These
both specify how pixel values stored in the Parametric Map should be mapped to
"real world"" values and also specify the semantics of those real world values
after mapping.

In highdicom, *RealWorldValueMappings* are represented by the
:class:`highdicom.pm.RealWorldValueMapping`. To specify the mapping use the
following parameters:

- ``value_range`` (``tuple[int, int]`` or ``tuple[float, float]``): Tuple
  giving the lower and upper limits of the range of stored values that are the
  input to the mapping.
- ``slope`` and ``intercept`` (``int`` or ``float``): These parameters specify
  a linear relationship between the stored values and the real-world values of
  the form ``real_world_value = stored_value * slope + intercept``.
- ``lut_data`` (``Sequence[float]``): Sequence of values to serve as a lookup
  table for mapping stored values into real-world values in case of a
  non-linear relationship. The sequence should contain an entry for each value
  in the specified ``value_range`` such that
  ``len(lut_data) == value_range[1] value_range[0] + 1``. For example, in case
  of a value range of ``(0,
  255)``, the sequence shall have ``256`` entries - one for each value in the
  given range.

If ``lut_data`` is not specified, a linear relationship is assumed with a
default intercept of 0 and slope of 1, (i.e. an identity mapping).

Additionally, the following parameters are used to describe this mapping and
the real-world values it generates:

- ``lut_label`` (``str``):  A short label for this mapping.
- ``lut_explanation`` (``str``):  A longer free-text explanation of the meaning
  of the mapping.
- ``unit`` (:class:`highdicom.sr.CodedConcept` or ``pydicom.sr.coding.Code``).
  Code (see :ref:`coding`) giving the unit of measurement for the pixels. The
  UCUM system of measurement is typically used for this purpose.
- ``quantity_definition`` (:class:`highdicom.sr.CodedConcept` or
  ``pydicom.sr.coding.Code``), optional. Another code representing the physical
  quantity that the parametric map measures.

The following snippet constructs a basic Real World Value Mappings that use a
linear relationship between stored values and real-world values:

.. code-block:: python

  import highdicom as hd
  from pydicom.sr.codedict import codes


  # Apparent diffusion coefficient using a scaling factor and measured in
  # mm^2/s
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="ADC",
      lut_explanation="Apparent diffusion coefficient in mm^2/s",
      value_range=(0, 1000.0),
      slope=0.001,
      quantity_definition=codes.DCM.ApparentDiffusionCoefficient,
      unit=codes.UCUM.SquareMillimeterPerSecond,
  )

  # A simple identity mapping that represents a saliency map from a neural
  # network as a dimensionless quantity
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="Class Activation",
      lut_explanation=(
         "Class activation of a neural network for "
         "pneumonia detection"
      ),
      value_range=(0.0, 100.0),
      quantity_definition=codes.DCM.ClassActivation,
      unit=codes.UCUM.NoUnits,
  )

Pixel Data
----------

The array to be encoded in the parametric map is passed to the constructor of
the :class:`highdicom.pm.ParametricMap` as a ``np.ndarray`` via the ``pixel_array``
parameter. There are several possible variations here.

Firstly, the data type of the array will control the way the pixels are stored
in the resulting DICOM object. It should be either:

- 8- or 16-bit unsigned integer (``np.uint8`` or ``np.uint16``)
- 32- or 64-bit floating point (``np.float32`` or ``np.float64``)

The dimensionality of the array should be either:

- A 2D array, giving either:

  - A single 2D frame giving a parametric map of a single 2D input image, or
  - If ``tile_pixel_array=True`` a 2D total pixel matrix that will be encoded
    in a multi-frame tiled arrangement within the resulting DICOM object
- A 3D array, giving multiple 2D parametric map frames of either a series of
  input image or of a single multiframe image. Frames are stacked down the
  first dimension (index 0)
- A 4D array which is interpreted the same way as the 3D, but with an
  additional final dimension that corresponds to multiple real world value
  mappings passed to the ``real_world_value_mappings`` parameter.
- A :class:`highdicom.Volume` with either no channels, or a single channel
  dimension

We will see examples of most of these below.

Constructing a Single-Frame Parametric Map
------------------------------------------

To construct a basic Parametric Map derived from one single-frame DICOM image,
we pass the source image, the parametric map pixel array, the real-world value
mapping, and some other pieces of basic information, as follows:


.. code-block:: python

  import numpy as np
  import highdicom as hd
  from pydicom.sr.codedict import codes


  # Read in the source image. Here we use a highdicom test file
  # from the highdicom repo
  im = hd.imread("data/test_files/ct_image.dcm")

  # Describe the mapping
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="Class Activation",
      lut_explanation=(
          "Class activation of a neural network for "
          "pneumonia detection"
      ),
      value_range=(0.0, 100.0),
      quantity_definition=codes.DCM.ClassActivation,
      unit=codes.UCUM.NoUnits,
  )

  # Toy example with random pixel array of floats with the same dimension as
  # the input image
  pixel_array = np.random.uniform(
      0.0,
      100.0,
      size=(im.Rows, im.Columns),
  )

  # Construct the parametric map
  pm = hd.pm.ParametricMap(
      source_images=[im],
      pixel_array=pixel_array,
      series_instance_uid=hd.UID(),
      series_number=1,
      sop_instance_uid=hd.UID(),
      instance_number=1,
      manufacturer="manufacturer",
      manufacturer_model_name="model name",
      software_versions="1",
      device_serial_number="123",
      contains_recognizable_visual_features=False,
      real_world_value_mappings=[mapping],
      voi_lut_transformations=[
          hd.VOILUTTransformation(
              window_center=50,
              window_width=100,
          )
      ],
      series_description="Single frame parametric map",
  )

  # The resulting object is a sub-class of pydicom.Dataset
  # and can therefore be saved to the filesystem, etc
  pm.save_as("parametric_map_example.dcm")

This assumes that there is pixel-for-pixel correspondence between the pixels in
the ``pixel_array`` and the pixels of the source image.

There are several other parameters available here that we will not cover in
detail. Please consult the documentation of :class:`highdicom.pm.ParametricMap`
for more information.

Constructing a Multi-frame Parametric Map
-----------------------------------------

To create a parametric map derived from multiple input frames, stack the frames
down the first axis (index zero) of a three-dimensional ``pixel_array``. There
are two options here:

- Each frame of the parametric map is derived from a single instance from a
  series of single frame instances. In this case, pass the list of single frame
  instances to the ``source_images`` parameter, and ensure that the order of
  frames down the first dimension of the pixel array matches the order of
  source images in the list, i.e. the parametric map frame at
  ``pixel_array[i, :, :]`` is derived from ``source_image[i]`` and has pixel-for-pixel
  correspondence with it.
- Each frame of the parametric map is derived from a single frame from one
  multi-frame image instance. In this case, pass the single multi-frame
  instance to the ``source_images`` parameter, and ensure that the order of
  frames down the first dimension of the pixel array matches the order the
  frames are stored in the source image, i.e. the parametric map frame at
  ``pixel_array[i, :, :]`` is derived from ``source_image[0].pixel_array[i, :,
  :]`` and has pixel-for-pixel correspondence with it.

In this first example, we derive a parametric map from a single-frame series:

.. code-block:: python

  import numpy as np
  import highdicom as hd
  from pydicom.data import get_testdata_files
  from pydicom.sr.codedict import codes


  # Read in the source images. Here we use a series of CT images from the
  # pydicom test data
  ct_series = [
      hd.imread(f)
      for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
  ]
  n_frames = len(ct_series)
  n_rows = ct_series[0].Rows
  n_columns = ct_series[0].Columns

  # Describe the mapping
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="Class Activation",
      lut_explanation=(
          "Class activation of a neural network for "
          "pneumonia detection"
      ),
      value_range=(0.0, 100.0),
      quantity_definition=codes.DCM.ClassActivation,
      unit=codes.UCUM.NoUnits,
  )

  # Toy example with random pixel array of floats with the same dimension as the
  # input image. Frames are stacked down the first dimension
  pixel_array = np.random.uniform(
      0.0,
      100.0,
      size=(n_frames, n_rows, n_columns)
  )

  # Construct the parametric map
  pm = hd.pm.ParametricMap(
      source_images=ct_series,  # the full input series
      pixel_array=pixel_array,
      series_instance_uid=hd.UID(),
      series_number=1,
      sop_instance_uid=hd.UID(),
      instance_number=1,
      manufacturer="manufacturer",
      manufacturer_model_name="model name",
      software_versions="1",
      device_serial_number="123",
      contains_recognizable_visual_features=False,
      real_world_value_mappings=[mapping],
      voi_lut_transformations=[
          hd.VOILUTTransformation(
              window_center=50,
              window_width=100,
          )
      ],
      series_description="Multi-frame parametric map",
  )

In this second example, we derive a parametric map from one multi-frame image:

.. code-block:: python

  import numpy as np
  import highdicom as hd
  from pydicom.data import get_testdata_file
  from pydicom.sr.codedict import codes


  # Read in the source image. Here we use a multiframe CT from the pydicom 
  # test data
  multiframe_image = hd.imread(get_testdata_file("eCT_Supplemental.dcm"))
  n_frames = multiframe_image.NumberOfFrames
  n_rows = multiframe_image.Rows
  n_columns = multiframe_image.Columns

  # Describe the mapping
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="Class Activation",
      lut_explanation=(
          "Class activation of a neural network for "
          "pneumonia detection"
      ),
      value_range=(0.0, 100.0),
      quantity_definition=codes.DCM.ClassActivation,
      unit=codes.UCUM.NoUnits,
  )

  # Toy example with random pixel array of floats with the same dimension as the
  # input image. Frames are stacked down the first dimension
  pixel_array = np.random.uniform(
      0.0,
      100.0,
      size=(n_frames, n_rows, n_columns)
  )

  # Construct the parametric map
  pm = hd.pm.ParametricMap(
      source_images=[multiframe_image],  # pass the multiframe image
      pixel_array=pixel_array,
      series_instance_uid=hd.UID(),
      series_number=1,
      sop_instance_uid=hd.UID(),
      instance_number=1,
      manufacturer="manufacturer",
      manufacturer_model_name="model name",
      software_versions="1",
      device_serial_number="123",
      contains_recognizable_visual_features=False,
      real_world_value_mappings=[mapping],
      voi_lut_transformations=[
          hd.VOILUTTransformation(
              window_center=50,
              window_width=100,
          )
      ],
      series_description="Multi-frame parametric map",
  )

This approach places some burden on the calling code to ensure the order of
frames correspond between the pixel array and source images. This can be
difficult in complex analysis pipelines. If you are working with
reguarly-sampled 3D data, we recommend passing a Volume object instead (see
:ref:`pm_from_volume`).

.. _pm_from_volume:

Constructing a Parametric Map from a Volume
-------------------------------------------

If your parametric map pixel array forms a regularly-sampled 3D volume, you can
pass a :class:`highdicom.Volume` object instead of a basic NumPy array (see
:ref:`volume` for an overview of Volumes). This has two advantages:

- It removes the requirement from you to ensure correspondence between frames
  of the ``source_images`` and ``pixel_array``. The constructor will compare
  the spatial information in the Volume to that in the input image to determine
  which frame of the input image/series each frame of the volume was derived
  from (if there is a pixel-for-pixel correspondence).
- It removes the requirement that there be any pixel-for-pixel correspondence
  between the parametric map volume and the source images. This may be useful
  in situations where the parametric map is derived from the source image(s)
  after some sort of spatial transformation (cropping, padding, resampling,
  rotation, etc). In this situation, the spatial information in the output
  Parametric Map object is derived from the Volume rather than from the source
  image(s). (This is also possible by manually specifying the
  ``plane_positions``, but using Volumes is recommended where possible.)

Here is a full worked example for deriving a parametric map from a source
multiframe image where the parametric map has undergone a spatial transformation
with respect to the source image.

.. code-block:: python

  import numpy as np
  import highdicom as hd
  from pydicom.data import get_testdata_file


  # Read in the source image. Here we use a multiframe CT from the pydicom
  # test data
  multiframe_image = hd.imread(get_testdata_file("eCT_Supplemental.dcm"))

  # Get the input image pixels as a volume for convenient processing
  input_volume = multiframe_image.get_volume()

  # Perform spatial operations to the volume (here cropping and flipping)
  # to prepare for analysis
  processed_volume = (
      input_volume
      .crop_to_spatial_shape((2, 400, 400))
      .flip_spatial(0)
  )

  # Example calculation to derive a parameter map (NB this is not a
  # meaningful calculation)
  derived_volume = processed_volume.with_array(
      processed_volume.array ** 2
  )

  # Describe the mapping
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="HU2",
      lut_explanation="Square Hounsfield units",
      value_range=(0.0, 1e7),
      unit=hd.sr.CodedConcept(
          value="[hnsf'U]2",
          scheme_designator="UCUM",
          meaning="Square Hounsfield units",
      ),
  )

  # Construct the parametric map
  pm = hd.pm.ParametricMap(
      source_images=[multiframe_image],
      pixel_array=derived_volume,
      series_instance_uid=hd.UID(),
      series_number=1,
      sop_instance_uid=hd.UID(),
      instance_number=1,
      manufacturer="manufacturer",
      manufacturer_model_name="model name",
      software_versions="1",
      device_serial_number="123",
      contains_recognizable_visual_features=False,
      real_world_value_mappings=[mapping],
      voi_lut_transformations=[
          hd.VOILUTTransformation(
              window_center=50,
              window_width=100,
          )
      ],
      series_description="Volumetric parametric map",
  )

There is more information in the :ref:`seg-from-volume` section about what
effect passing a volume does and does not have on the DICOM object created.

Constructing Tiled Parametric Maps
----------------------------------

If your paramtric map array is derived from a digital pathology image and is in
the form of a total pixel matrix that you wish to store as a tiled image, you
can pass the total pixel matrix to the ``pixel_array`` parameter, specify
``tile_pixel_array=True``, and optionally specify the ``tile_size``.


.. code-block:: python

  import numpy as np
  import highdicom as hd
  from pydicom.sr.codedict import codes


  # Read in a tiled source image. Here we use a test file from the
  # highdicom test data
  tiled_image = hd.imread("data/test_files/sm_image_control.dcm")

  # Toy example with random pixel array of floats with the same dimension as
  # the input image's total pixel matrix
  pixel_array = np.random.uniform(
      0.0,
      100.0,
      size=(
          tiled_image.TotalPixelMatrixRows,
          tiled_image.TotalPixelMatrixColumns
      ),
  )

  # Describe the mapping
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="Class Activation",
      lut_explanation=(
          "Class activation of a neural network for "
          "tumor detection"
      ),
      value_range=(0.0, 100.0),
      quantity_definition=codes.DCM.ClassActivation,
      unit=codes.UCUM.NoUnits,
  )

  # Construct the parametric map
  pm = hd.pm.ParametricMap(
      source_images=[tiled_image],
      pixel_array=pixel_array,
      series_instance_uid=hd.UID(),
      series_number=1,
      sop_instance_uid=hd.UID(),
      instance_number=1,
      manufacturer="manufacturer",
      manufacturer_model_name="model name",
      software_versions="1",
      device_serial_number="123",
      contains_recognizable_visual_features=False,
      real_world_value_mappings=[mapping],
      voi_lut_transformations=[
          hd.VOILUTTransformation(
              window_center=50,
              window_width=100,
          )
      ],
      tile_pixel_array=True,  # specify input should be tiled
      dimension_organization_type="TILED_FULL",  # use "tiled full"
  )

This assumes that the input array is pixel-for-pixel aligned with the total
pixel matrix of the source image. If the geometry of the pixel array does not
match the source image, you may pass a :class:`highdicom.Volume` with a shape
of 1 down the first dimension as the pixel array and the spatial metadata will
be copied from it.

Both "tiled-full" and "tiled-sparse" dimension organizations are supported (see
:ref:`tiled-dimension-organization` for Segmentations). Generally "TILED_FULL"
is recommended in nearly all situations.

Multi-Resolution Pyramids
-------------------------

If the input pixel matrix is very large, you may wish to generate a series of
parametric maps representing the same image at multiple resolutions, known as a
multi-resolution pyramid. The
:func:`highdicom.pm.create_parametric_map_pyramid` function does this for you.
There are multiple ways to specify the levels of the pyramid, including by
specifying downsampling factors, using the same levels as the source image
pyramid (if any), and manually passing downsampled images (see
:ref:`multi-resolution-pyramids` in Segmentations for a detailed breakdown).

Here, we give a simple example manually specifying a downsampling factor to
create a pyramid.

.. code-block:: python

  import numpy as np
  import highdicom as hd
  from pydicom.sr.codedict import codes

  # Read in a tiled source image. Here we use a test file from the
  # highdicom test data
  tiled_image = hd.imread("data/test_files/sm_image_control.dcm")

  # Toy example with random pixel array of floats with the same dimension as
  # the input image's total pixel matrix
  pixel_array = np.random.uniform(
      0.0,
      100.0,
      size=(
          tiled_image.TotalPixelMatrixRows,
          tiled_image.TotalPixelMatrixColumns
      ),
  )

  # Describe the mapping
  mapping = hd.pm.RealWorldValueMapping(
      lut_label="Class Activation",
      lut_explanation=(
          "Class activation of a neural network for "
          "tumor detection"
      ),
      value_range=(0.0, 100.0),
      quantity_definition=codes.DCM.ClassActivation,
      unit=codes.UCUM.NoUnits,
  )

  # Construct the parametric map
  pm_series = hd.pm.create_parametric_map_pyramid(
      source_images=[tiled_image],
      pixel_arrays=[pixel_array],
      downsample_factors=[5.0],
      series_instance_uid=hd.UID(),
      series_number=1,
      manufacturer="manufacturer",
      manufacturer_model_name="model name",
      software_versions="1",
      device_serial_number="123",
      contains_recognizable_visual_features=False,
      real_world_value_mappings=[mapping],
      voi_lut_transformations=[
          hd.VOILUTTransformation(
              window_center=50,
              window_width=100,
          )
      ],
      dimension_organization_type="TILED_FULL",
      interpolator=hd.InterpolationMethods.LINEAR,
  )

The return value is a list of :class:`highdicom.pm.ParametricMap` objects.

Compression
-----------

When the pixel array is passed as an unsigned integer data type, it is possible
to specify a ``transfer_syntax_uid`` to use to compress the pixels. Options
include ``JPEGBaseline8Bit``,  ``JPEG2000``, ``JPEG2000Lossless``,
``JPEGLSLossless``, ``JPEGLSNearLossless``, and ``RLELossless``.

When the image has a large number of frames, compression can become slow. The
``workers`` argument allows you to specify a number of sub-process to use to
parallelize the process of encoding frames.

Compression is not available when using floating point pixels.

Parametric Maps with Multiple Mappings
--------------------------------------

In the examples above, we have always passed a single item to the
``real_world_value_mappings`` parameter, and this will be by far the most
common case. However, it is possible to pass multiple items here. In this case,
the multiple mappings represent alternatives. For example, you may wish to
provide linear and logarithmic alternative mappings to use in different
situations.

Parametric Maps with Multiple "Channels"
----------------------------------------

Highdicom has the ability to include multiple sets of frames with different
mappings in a single Parametric Map object. These could represent different
sets of measurements, or different components of a vector measurment. We will
refer to such sets of measurements here as "channels", but this is not a term
used in DICOM itself.

To do this, pass a 4D array to ``pixel_array``, with the multiple "channels"
stacked down the last axis (index 3). This implies that the first (frames)
access must be present even if it has length (and could therefore otherwise be
omitted). Additionally, pass a nested list of mappings (where previously a
single-level list was passed). The outer list contains the mappings for each
channel (and therefore its length must match the number of channels down the
last axis of the ``pixel_array``). The inner list contains (potentially)
multiple mappings for the given channel (see previous section).

If ``pixel_array`` is a Volume, it should have a single channel dimension
with ``"LUTLabel"`` as the channel descriptor.

Here is an example of a two channel image, representing two directions of a
gradient vector:

.. code-block:: python

  import numpy as np
  import highdicom as hd
  from pydicom.data import get_testdata_files


  # Read in the source images. Here we use a series of CT images from the
  # pydicom test data
  ct_series = [
      hd.imread(f)
      for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
  ]
  n_frames = len(ct_series)
  n_rows = ct_series[0].Rows
  n_columns = ct_series[0].Columns
  n_channels = 2

  # Describe the mappings
  mapping_x = hd.pm.RealWorldValueMapping(
      lut_label="X component",
      lut_explanation="Gradient in the x direction",
      value_range=(0.0, 100.0),
      unit=hd.sr.CodedConcept("/mm", "UCUM", "Per millimeter"),
  )
  mapping_y = hd.pm.RealWorldValueMapping(
      lut_label="Y component",
      lut_explanation="Gradient in the y direction",
      value_range=(0.0, 100.0),
      unit=hd.sr.CodedConcept("/mm", "UCUM", "Per millimeter"),
  )

  # Toy example with random pixel array of floats with the same dimension as the
  # input image with frames are stacked down the first dimension and the
  # additional channel dimension at the end
  pixel_array = np.random.uniform(
      0.0,
      100.0,
      size=(n_frames, n_rows, n_columns, n_channels)
  )

  # Construct the parametric map
  pm = hd.pm.ParametricMap(
      source_images=ct_series,  # the full input series
      pixel_array=pixel_array,
      series_instance_uid=hd.UID(),
      series_number=1,
      sop_instance_uid=hd.UID(),
      instance_number=1,
      manufacturer="manufacturer",
      manufacturer_model_name="model name",
      software_versions="1",
      device_serial_number="123",
      contains_recognizable_visual_features=False,
      real_world_value_mappings=[[mapping_x], [mapping_y]],
      voi_lut_transformations=[
          hd.VOILUTTransformation(
              window_center=50,
              window_width=100,
          )
      ],
      series_description="Multi-channel parametric map",
  )

Although highdicom supports this "multi-channel" configuration, it is more of
an experimental feature and we recommend using it only in limited
circumstances. Though valid within the (highly flexible) standard, this
arrangement is not common in other tools and therefore may not be correctly
understood by them. Furthermore, highdicom itself does not currently have
methods to conveniently work with objects following this pattern (though these
are planned for a future release).

Reading Existing Parametric Maps
--------------------------------

Existing parametric maps can be read from a file using the
:func:`highdicom.pm.pmread`. Since :class:`highdicom.pm.ParametricMap` is a
sub-class of :class:`highdicom.Image`, you can use methods such as
:meth:`highdicom.pm.ParametricMap.get_volume`,
:meth:`highdicom.pm.ParametricMap.get_total_pixel_matrix`. The
:func:`highdicom.pm.pmread` also accepts a ``lazy_frame_retrieval`` parameter,
allowing frames to be retrieved and decoded only as required. See :ref:`image`
for further information.
