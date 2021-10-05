"""Package for creation of Annotation (ANN) instances."""
from highdicom.ann.content import Measurements, AnnotationGroup
from highdicom.ann.enum import (
    AnnotationCoordinateTypeValues,
    AnnotationGroupGenerationTypeValues,
    GraphicTypeValues,
    PixelOriginInterpretationValues,
)
from highdicom.ann.sop import MicroscopyBulkSimpleAnnotations

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.91.1',  # Microscopy Bulk Simple Annotations
}

__all__ = [
    'AnnotationCoordinateTypeValues',
    'AnnotationGroup',
    'AnnotationGroupGenerationTypeValues',
    'GraphicTypeValues',
    'Measurements',
    'MicroscopyBulkSimpleAnnotations',
    'PixelOriginInterpretationValues',
]
