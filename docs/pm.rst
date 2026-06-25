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

Real World Value Mappings
-------------------------

Each parametric map requires one (or more) *RealWorldValueMappings*. These
both specify how pixel values stored in the Parametric Map should be mapped
to "real world"" values and also specify the semantics of those real world
values after mapping.

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
- ``unit`` (:class:`highdicom.sr.CodedConcept`` or ``pydicom.sr.coding.Code``).
  Code (see :ref:`coding`) giving the unit of measurement for the pixels. The
  UCUM system of measurement is typically used for this purpose.
- ``quantity_definition`` (:class:`highdicom.sr.CodedConcept`` or
  ``pydicom.sr.coding.Code``), optional. Another code representing the physical
  quantity that the parametric map measures.

The following snippet constructs a basic Real World Value Mappings that use a
linear relationship between stored values and real-world values:

.. code-block:: python

    import highdicom as hd
    from pydicom.sr.codedict import codes


    # Apparent diffusion coefficient using a scaling factor and measured in
    # mm^2/s
    mapping = RealWorldValueMapping(
        lut_label="ADC",
        lut_explanation="Apparent diffusion coefficient in mm^2/s",
        value_range=(0, 1000.0),
        slope=0.001,
        quantity_definition=codes.DCM.ApparentDiffusionCoefficient,
        unit=codes.UCUM.SquareMillimeterPerSecond,
    )

    # A simple identity mapping that represents a saliency map from a neural
    # network as a dimensionless quantity
    mapping = RealWorldValueMapping(
        lut_label="Class Activation",
        lut_explanation=(
            "Class activation of a neural network for the prediction "
            "of pneumonia"
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
    in a tiled arrangement within the resulting DICOM object
- A 3D array, giving multiple 2D parametric map frames of either a series of
  input image or of a single multiframe image. Frames are stacked down the
  first dimension (index 0)
- A 4D array which is interpreted the same way as the 3D, but with an
  additional final dimension that corresponds to multiple real world value
  mappings passed to the ``real_world_value_mappings`` parameter.
- A :class:`highdicom.Volume` with either no channels, or a single channel
  dimension

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

Reading Existing Parametric Maps
--------------------------------

Existing parametic maps can be read from a file using the
:func:`highdicom.pm.pmread`. Since :class:`highdicom.pm.ParametricMap` is a
sub-class of :class:`highdicom.Image`, you can use methods such as
:meth:`highdicom.pm.ParametricMap.get_volume`,
:meth:`highdicom.pm.ParametricMap.get_total_pixel_matrix`. The
:func:`highdicom.pm.pmread` also accepts a `lazy_frame_retrieval` parameter,
allowing frames to be retrieved and decoded only as required. See :ref:`images`
for further information.
