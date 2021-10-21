"""Package for creation of Presentation State instances."""

from highdicom.pr.sop import GrayscaleSoftcopyPresentationState
from highdicom.pr.enum import (
    AnnotationUnitsValues,
    GraphicTypeValues
)
from highdicom.pr.content import GraphicObject, TextObject


SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.11.1',  # Grayscale Softcopy Presentation State
}


__all__ = [
    'GrayscaleSoftcopyPresentationState',
    'GraphicTypeValues',
    'AnnotationUnitsValues',
    'GraphicObject',
    'TextObject',
]
