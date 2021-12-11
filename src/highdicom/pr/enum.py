"""Enumerated values specific to Presentation State IODs."""

from enum import Enum


class AnnotationUnitsValues(Enum):

    """

    Enumerated values for annotation units, describing how the stored values
    relate to the image position.

    """

    PIXEL = 'PIXEL'
    """Image position in pixel units.

    Image relative position specified with sub-pixel resolution such that the
    origin, which is at the Top Left Hand Corner (TLHC) of the TLHC pixel is
    (0.0, 0.0), the Bottom Right Hand Corner (BRHC) of the TLHC pixel is (1.0,
    1.0), and the BRHC of the BRHC pixel is Columns, Rows.
    The values must be within the range 0, 0 to (Columns, Rows).

    """

    DISPLAY = 'DISPLAY'
    """Fraction of the displayed area.

    Image position relative to the displayed area, where (0.0, 0.0) is the top
    left hand corner of the displayed area and (1.0, 1.0) is the bottom right
    hand corner of the displayed area. Values are between 0.0 and 1.0.
    """

    MATRIX = 'MATRIX'
    """Position relative to the total pixel matrix in whole slide images.

    Image relative position specified with sub-pixel resolution such that the
    origin, which is at the Top Left Hand Corner (TLHC) of the TLHC pixel of
    the Total Pixel Matrix, is (0.0, 0.0), the Bottom Right Hand Corner (BRHC)
    of the TLHC pixel is (1.0, 1.0), and the BRHC of the BRHC pixel of the
    Total Pixel Matrix is (Total Pixel Matrix Columns,Total Pixel Matrix Rows).
    The values must be within the range (0.0, 0.0) to (Total Pixel Matrix
    Columns, Total Pixel Matrix Rows). MATRIX may be used only if the value of
    Referenced SOP Class UID (0008,1150) within Referenced Image Sequence
    (0008,1140) is 1.2.840.10008.5.1.4.1.1.77.1.6 (VL Whole Slide Microscopy
    Image).

    """


class TextJustificationValues(Enum):

    """

    Enumerated values for the BoundingBoxTextHorizontalJustification attribute.

    """

    LEFT = 'LEFT'
    CENTER = 'CENTER'
    RIGHT = 'RIGHT'


class GraphicTypeValues(Enum):

    """Enumerated values for attribute Graphic Type.

    See :dcm:`C.10.5.2 <part03/sect_C.10.5.html#sect_C.10.5.1.2>`.

    """

    CIRCLE = 'CIRCLE'
    """A circle defined by two (column,row) pairs.

    The first pair is the central point and
    the second pair is a point on the perimeter of the circle.

    """

    ELLIPSE = 'ELLIPSE'
    """An ellipse defined by four pixel (column,row) pairs.

    The first two pairs specify the endpoints of the major axis and
    the second two pairs specify the endpoints of the minor axis.

    """

    INTERPOLATED = 'INTERPOLATED'
    """List of end points between which a line is to be interpolated.

    The exact nature of the interpolation is an implementation detail of
    the software rendering the object.

    Each point is represented by a (column,row) pair.

    """

    POINT = 'POINT'
    """A single point defined by two values (column,row)."""

    POLYLINE = 'POLYLINE'
    """List of end points between which straight lines are to be drawn.

    Each point is represented by a (column,row) pair.

    """


class RescaleTypeValues(Enum):

    """Enumerated values for attribute Rescale Type.

    This specifies the units of the result of the rescale operation.
    Other values may be used, but they are not defined by the DICOM standard.

    """

    OD = 'OD'
    """The number in the LUT represents thousands of optical density.

    That is, a value of 2140 represents an optical density of 2.140.

    """

    HU = 'HU'
    """Hounsfield Units (CT)."""

    US = 'US'
    """Unspecified."""

    MGML = 'MGML'
    """Milligrams per milliliter."""

    Z_EFF = 'Z_EFF'
    """Effective Atomic Number (i.e., Effective-Z)."""

    ED = 'ED'
    """Electron density in 1023 electrons/ml."""

    EDW = 'EDW'
    """Electron density normalized to water.

    Units are N/Nw where N is number of electrons per unit volume, and Nw is
    number of electrons in the same unit of water at standard temperature and
    pressure.

    """

    HU_MOD = 'HU_MOD'
    """Modified Hounsfield Unit."""

    PCT = 'PCT'
    """Percentage (%)"""
