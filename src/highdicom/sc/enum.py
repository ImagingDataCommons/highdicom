"""Enumerate values specific to Secondary Capture (SC) Image IODs."""
from enum import Enum


class ConversionTypeValues(Enum):

    """Enumerated values for attribute Conversion Type."""

    DV = 'DV'
    """Digitized Video"""

    DI = 'DI'
    """Digital Interface"""

    DF = 'DF'
    """Digitized Film"""

    WSD = 'WSD'
    """Workstation"""

    SD = 'SD'
    """Scanned Document"""

    SI = 'SI'
    """Scanned Image"""

    DRW = 'DRW'
    """Drawing"""

    SYN = 'SYN'
    """Synthetic Image"""
