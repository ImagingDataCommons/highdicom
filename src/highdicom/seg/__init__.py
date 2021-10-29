"""Package for creation of Segmentation (SEG) instances."""
from highdicom.seg.sop import Segmentation, segread
from highdicom.seg.enum import (
    SegmentAlgorithmTypeValues,
    SegmentationTypeValues,
    SegmentationFractionalTypeValues,
    SpatialLocationsPreservedValues,
    SegmentsOverlapValues,
)
from highdicom.seg.content import (
    SegmentDescription,
    DimensionIndexSequence,
)
from highdicom.seg import utils

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.66.4',  # Segmentation
}

__all__ = [
    'DimensionIndexSequence',
    'Segmentation',
    'segread',
    'SegmentAlgorithmTypeValues',
    'SegmentationFractionalTypeValues',
    'SegmentationTypeValues',
    'SegmentDescription',
    'SegmentsOverlapValues',
    'SpatialLocationsPreservedValues',
    'utils',
]
