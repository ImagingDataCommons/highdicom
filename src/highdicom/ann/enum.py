"""Enumerate values specific to Annotation IODs."""
from enum import Enum


class AnnotationCoordinateTypeValues(Enum):

    """Enumerated values for attribute Annotation Coordinate Type."""

    SCOORD = '2D'
    SCOORD3D = '3D'


class AnnotationGroupGenerationTypeValues(Enum):

    """Enumerated values for attribute Annotation Group Generation Type."""

    AUTOMATIC = 'AUTOMATIC'
    SEMIAUTOMATIC = 'SEMIAUTOMATIC'
    MANUAL = 'MANUAL'


class GraphicTypeValues(Enum):

    """Enumerated values for attribute Graphic Type.

    Note
    ----
    Coordinates may be either (column,row) pairs defined in the 2-dimensional
    Total Pixel Matrix or (X,Y,Z) triplets defined in the 3-dimensional
    Frame of Reference (patient or slide coordinate system).

    Warning
    -------
    Despite having the same names, the definition of values for the Graphic
    Type attribute of the ANN modality may differ from those of the SR
    modality (SCOORD or SCOORD3D value types).

    """

    POINT = 'POINT'
    POLYLINE = 'POLYLINE'
    POLYGON = 'POLYGON'
    ELLIPSE = 'ELLIPSE'
    RECTANGLE = 'RECTANGLE'


class PixelOriginInterpretationValues(Enum):

    """Enumerated values for attribute Pixel Origin Interpretation."""

    FRAME = 'FRAME'
    VOLUME = 'VOLUME'
