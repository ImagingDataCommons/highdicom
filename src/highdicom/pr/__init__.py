"""Package for creation of Presentation State instances."""

from highdicom.pr.sop import (
    AdvancedBlendingPresentationState,
    ColorSoftcopyPresentationState,
    GrayscaleSoftcopyPresentationState,
    PseudoColorSoftcopyPresentationState,
)
from highdicom.pr.enum import (
    AnnotationUnitsValues,
    BlendingModeValues,
    GraphicTypeValues,
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
    SoftcopyVOILUTTransformation,
    TextObject
)


SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.11.1',  # Grayscale Softcopy Presentation State
    '1.2.840.10008.5.1.4.1.1.11.2',  # Color Softcopy Presentation State
    '1.2.840.10008.5.1.4.1.1.11.3',  # Pseudo Color Softcopy Presentation State
    '1.2.840.10008.5.1.4.1.1.11.8',  # Advanced Blending Presentation State
}


__all__ = [
    'AdvancedBlending',
    'AdvancedBlendingPresentationState',
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
    'PseudoColorSoftcopyPresentationState',
    'SoftcopyVOILUTTransformation',
    'TextJustificationValues',
    'TextObject',
]
