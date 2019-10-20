"""Enumerated values."""
from enum import Enum


class CoordinateSystemNames(Enum):

    """Enumerated values for coordinate system names."""

    PATIENT = 'PATIENT'
    SLIDE = 'SLIDE'


class ContentQualifications(Enum):

    """Enumerated values for Content Qualification attribute."""

    PRODUCT = 'PRODUCT'
    RESEARCH = 'RESEARCH'
    SERVICE = 'SERVICE'
