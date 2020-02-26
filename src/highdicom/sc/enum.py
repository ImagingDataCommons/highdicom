"""Enumerate values specific to Secondary Capture (SC) Image IODs."""
from enum import Enum


class ConversionTypeValues(Enum):

    """Enumerated values for attribute Conversion Type."""

    DV = 'DV'
    DI = 'DI'
    DF = 'DF'
    WSD = 'WSD'
    SD = 'SD'
    SI = 'SI'
    DRW = 'DRW'
    SYN = 'SYN'
