"""Package for creation of Presentation State instances."""

from highdicom.pr.sop import (
    GrayscaleSoftcopyPresentationState,
    PseudoColorSoftcopyPresentationState,
    ColorSoftcopyPresentationState,
)
from highdicom.pr.enum import (
    AnnotationUnitsValues,
    BlendingModeValues,
    GraphicTypeValues,
    PresentationLUTShapeValues,
    TextJustificationValues,
)
from highdicom.pr.content import (
    AdvancedBlending,
    BlendingDisplay,
    BlendingDisplayInput,
    GraphicAnnotation,
    GraphicGroup,
    GraphicLayer,
    GraphicObject,
    SoftcopyVOILUT,
    TextObject
)


SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.11.1',  # Grayscale Softcopy Presentation State
    '1.2.840.10008.5.1.4.1.1.11.2',  # Color Softcopy Presentation State
    '1.2.840.10008.5.1.4.1.1.11.3',  # Pseudo Color Softcopy Presentation State
}


__all__ = [
    'AdvancedBlending',
    'AnnotationUnitsValues',
    'BlendingModeValues',
    'BlendingDisplay',
    'BlendingDisplayInput',
    'ColorSoftcopyPresentationState',
    'GraphicAnnotation',
    'GraphicGroup',
    'GraphicLayer',
    'GraphicObject',
    'GraphicTypeValues',
    'GrayscaleSoftcopyPresentationState',
    'ModalityLUT',
    'PresentationLUTShapeValues',
    'PseudoColorSoftcopyPresentationState',
    'SoftcopyVOILUT',
    'TextJustificationValues',
    'TextObject',
]
