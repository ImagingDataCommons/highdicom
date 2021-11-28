"""Enumerate values specific to Parametric Map IODs."""
from enum import Enum


class ImageFlavorValues(Enum):

    """Enumerated values for value 3 of attribute Image Type or Frame Type."""

    ANGIO = 'ANGIO'
    CARDIAC = 'CARDIAC'
    CARDIAC_GATED = 'CARDIAC_GATED'
    CARDRESP_GATED = 'CARDRESP_GATED'
    DYNAMIC = 'DYNAMIC'
    FLUOROSCOPY = 'FLUOROSCOPY'
    LOCALIZER = 'LOCALIZER'
    MOTION = 'MOTION'
    PERFUSION = 'PERFUSION'
    PRE_CONTRAST = 'PRE_CONTRAST'
    POST_CONTRAST = 'POST_CONTRAST'
    RESP_GATED = 'RESP_GATED'
    REST = 'REST'
    STATIC = 'STATIC'
    STRESS = 'STRESS'
    VOLUME = 'VOLUME'
    NON_PARALLEL = 'NON_PARALLEL'
    PARALLEL = 'PARALLEL'
    WHOLE_BODY = 'WHOLE_BODY'


class DerivedPixelContrastValues(Enum):

    """Enumerated values for value 4 of attribute Image Type or Frame Type."""

    ADDITION = 'ADDITION'
    DIVISION = 'DIVISION'
    MASKED = 'MASKED'
    MAXIMUM = 'MAXIMUM'
    MEAN = 'MEAN'
    MINIMUM = 'MINIMUM'
    MULTIPLICATION = 'MULTIPLICATION'
    NONE = 'NONE'
    RESAMPLED = 'RESAMPLED'
    STD_DEVIATION = 'STD_DEVIATION'
    SUBTRACTION = 'SUBTRACTION'
    QUANTITY = 'QUANTITY'
