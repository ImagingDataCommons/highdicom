"""Enumerate values specific to Annotation IODs."""
from enum import Enum


class AnnotationCoordinateTypeValues(Enum):

    """Enumerated values for attribute Annotation Coordinate Type."""

    SCOORD = '2D'
    """Two-dimensional spatial coordinates denoted by (Column,Row) pairs.

    The coordinate system is the pixel matrix of an image and individual
    coordinates are defined relative to center of the (1,1) pixel of either
    the total pixel matrix of the entire image or of the pixel matrix of an
    individual frame, depending on the value of Pixel Origin Interpretation.

    Coordinates have pixel unit.

    """

    SCOORD3D = '3D'
    """Three-dimensional spatial coordinates denoted by (X,Y,Z) triplets.

    The coordinate system is the Frame of Reference (slide or patient) and the
    coordinates are defined relative to origin of the Frame of Reference.

    Coordinates have millimeter unit.

    """


class AnnotationGroupGenerationTypeValues(Enum):

    """Enumerated values for attribute Annotation Group Generation Type."""

    AUTOMATIC = 'AUTOMATIC'
    SEMIAUTOMATIC = 'SEMIAUTOMATIC'
    MANUAL = 'MANUAL'


class GraphicTypeValues(Enum):

    """Enumerated values for attribute Graphic Type.

    Note
    ----
    Coordinates may be either (Column,Row) pairs defined in the 2-dimensional
    Total Pixel Matrix or (X,Y,Z) triplets defined in the 3-dimensional
    Frame of Reference (patient or slide coordinate system).

    Warning
    -------
    Despite having the same names, the definition of values for the Graphic
    Type attribute of the ANN modality may differ from those of the SR
    modality (SCOORD or SCOORD3D value types).

    """

    POINT = 'POINT'
    """An individual piont defined by a single coordinate."""

    POLYLINE = 'POLYLINE'
    """Connected line segments defined by two or more ordered coordinates.

    The coordinates shall be coplanar.

    """

    POLYGON = 'POLYGON'
    """Connected line segments defined by three or more ordered coordinates.

    The coordinates shall be coplanar and form a closed polygon.

    Warning
    -------
    In contrast to the corresponding SR Graphic Type for content items of
    SCOORD3D value type, the first and last points shall NOT be the same.

    """

    ELLIPSE = 'ELLIPSE'
    """An ellipse defined by four coordinates.

    The first two coordinates specify the endpoints of the major axis and
    the second two coordinates specify the endpoints of the minor axis.

    """

    RECTANGLE = 'RECTANGLE'
    """Connected line segments defined by three or more ordered coordinates.

    The coordinates shall be coplanar and form a closed, rectangular polygon.
    The first coordinate is the top left hand corner, the second coordinate is
    the top right hand corner, the third coordinate is the bottom right hand
    corner, and the forth coordinate is the bottom left hand corner.

    The edges of the rectangle need not be aligned with the axes of the
    coordinate system.

    """


class PixelOriginInterpretationValues(Enum):

    """Enumerated values for attribute Pixel Origin Interpretation."""

    FRAME = 'FRAME'
    """Relative to an individual image frame.

    Coordinates have been defined and need to be interpreted relative to the
    (1,1) pixel of an individual image frame.

    """

    VOLUME = 'VOLUME'
    """Relative to the Total Pixel Matrix of a VOLUME image.

    Coordinates have been defined and need to be interpreted relative to the
    (1,1) pixel of the Total Pixel Matrix of the entire image.

    """
