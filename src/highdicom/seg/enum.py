"""Enumerate values specific to the Segmentation IOD."""
from enum import Enum


class SegmentAlgorithmTypes(Enum):

    """Enumerated values for attribute Segment Algorithm Type."""

    AUTOMATIC = 'AUTOMATIC'
    SEMIAUTOMATIC = 'SEMIAUTOMATIC'
    MANUAL = 'MANUAL'


class SegmentationTypes(Enum):

    """Enumerated values for attribute Segmentation Type."""

    BINARY = 'BINARY'
    FRACTIONAL = 'FRACTIONAL'


class SegmentationFractionalTypes(Enum):

    """Enumerated values for attribute Segmentation Fractional Type."""

    PROBABILITY = 'PROBABILITY'
    OCCUPANCY = 'OCCUPANCY'


class SpatialLocationsPreserved(Enum):

    """Enumerated values for attribute Spatial Locations Preserved."""

    YES = 'YES'
    NO = 'NO'
    REORIENTED_ONLY = 'REORIENTED_ONLY'


class SegmentsOverlap(Enum):

    """Enumerated values for attribute Segments Overlap Attribute."""

    YES = 'YES'
    UNDEFINED = 'UNDEFINED'
    NO = 'NO'
