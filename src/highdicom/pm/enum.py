"""Enumerate values specific to the Parametric Map IOD."""
from enum import Enum


class ContentLabelValues(Enum):

    """Enumerated values for attribute Content Label.

    Note
    ----
    These values are not defined by the standard and other user-defined values
    may be used instead.

    """

    ACTIVATION_MAP = 'ACTIVATION_MAP'
    ATTENTION_MAP = 'ATTENTION_MAP'
    FEATURE_MAP = 'FEATURE_MAP'
    SALIENCY_MAP = 'SALIENCY_MAP'
