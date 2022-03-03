"""Enumerated values specific to Presentation State IODs."""

from enum import Enum


class AnnotationUnitsValues(Enum):

    """

    Enumerated values for annotation units, describing how the stored values
    relate to the image position.

    """

    PIXEL = 'PIXEL'
    """Image coordinates within the image (or image frame/tile).

    Image coordinates in pixel unit specified with sub-pixel resolution such
    that the origin, which is at the Top Left Hand Corner (TLHC) of the TLHC
    pixel is (0.0, 0.0), the Bottom Right Hand Corner (BRHC) of the TLHC pixel
    is (1.0, 1.0), and the BRHC of the BRHC pixel is Columns, Rows.  The values
    must be within the range 0, 0 to (Columns, Rows).

    """

    DISPLAY = 'DISPLAY'
    """Display coordinates.

    Display coordinates in pixel unit specified with sub-pixel resolution,
    where (0.0, 0.0) is the top left hand corner of the displayed area and
    (1.0, 1.0) is the bottom right hand corner of the displayed area. Values
    are between 0.0 and 1.0.

    """

    MATRIX = 'MATRIX'
    """Image coordinates relative to the total pixel matrix of a tiled image.

    Image coordinates in pixel unit specified with sub-pixel resolution such
    that the origin, which is at the Top Left Hand Corner (TLHC) of the TLHC
    pixel of the Total Pixel Matrix, is (0.0, 0.0), the Bottom Right Hand
    Corner (BRHC) of the TLHC pixel is (1.0, 1.0), and the BRHC of the BRHC
    pixel of the Total Pixel Matrix is (Total Pixel Matrix Columns,Total Pixel
    Matrix Rows).  The values must be within the range (0.0, 0.0) to (Total
    Pixel Matrix Columns, Total Pixel Matrix Rows). MATRIX may be used only if
    the referenced image is tiled (i.e. has attributes Total Pixel Matrix Rows
    and Total Pixel Matrix Columns).

    """


class TextJustificationValues(Enum):

    """Enumerated values for attribute Bounding Box Text Horizontal
    Justification.

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


class PresentationLUTShapeValues(Enum):

    """Enumerated values for the Presentation LUT Shape attribute."""

    IDENTITY = 'IDENTITY'
    """No further translation of values is performed."""

    INVERSE = 'INVERSE'
    """

    A value of INVERSE shall mean the same as a value of IDENTITY, except that
    the minimum output value shall convey the meaning of the maximum available
    luminance, and the maximum value shall convey the minimum available
    luminance.

    """
