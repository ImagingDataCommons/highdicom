"""Enumerate values specific to the Parametric Map IOD."""
from enum import Enum


class ContentLabelValues(Enum):

    """Enumerated values for attribute Content Label."""

    ACTIVATION_MAP = 'ACTIVATION_MAP'
    ATTENTION_MAP = 'ATTENTION_MAP'
    FEATURE_MAP = 'FEATURE_MAP'
    SALIENCY_MAP = 'SALIENCY_MAP'
