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

Compression
-----------


