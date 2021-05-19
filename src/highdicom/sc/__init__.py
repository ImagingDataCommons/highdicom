"""Package for creation of Secondary Capture (SC) Image instances."""
from highdicom.sc.sop import SCImage
from highdicom.sc.enum import ConversionTypeValues

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.7',  # SC Image
}

__all__ = [
    'ConversionTypeValues',
    'SCImage',
]
