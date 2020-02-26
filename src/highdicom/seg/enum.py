"""Enumerate values specific to Segmentation IODs."""
from enum import Enum


class SegmentAlgorithmTypeValues(Enum):

    """Enumerated values for attribute Segment Algorithm Type."""

    AUTOMATIC = 'AUTOMATIC'
    SEMIAUTOMATIC = 'SEMIAUTOMATIC'
    MANUAL = 'MANUAL'


class SegmentationTypeValues(Enum):

    """Enumerated values for attribute Segmentation Type."""

    BINARY = 'BINARY'
    FRACTIONAL = 'FRACTIONAL'


class SegmentationFractionalTypeValues(Enum):

    """Enumerated values for attribute Segmentation Fractional Type."""

    PROBABILITY = 'PROBABILITY'
    OCCUPANCY = 'OCCUPANCY'


class SpatialLocationsPreservedValues(Enum):

    """Enumerated values for attribute Spatial Locations Preserved."""

    YES = 'YES'
    NO = 'NO'
    REORIENTED_ONLY = 'REORIENTED_ONLY'


class SegmentsOverlapValues(Enum):

    """Enumerated values for attribute Segments Overlap."""

    YES = 'YES'
    UNDEFINED = 'UNDEFINED'
    NO = 'NO'
