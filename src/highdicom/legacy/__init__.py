"""Package for creation of Legacy Converted Enhanced CT, MR or PET Image
instances.
"""
from highdicom.legacy.sop import (
    LegacyConvertedEnhancedCTImage,
    LegacyConvertedEnhancedMRImage,
    LegacyConvertedEnhancedPETImage,
)

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.4.4',    # Legacy Converted Enhanced MR Image
    '1.2.840.10008.5.1.4.1.1.2.2',    # Legacy Converted Enhanced CT Image
    '1.2.840.10008.5.1.4.1.1.128.1',  # Legacy Converted Enhanced PET Image
}

__all__ = [
    'LegacyConvertedEnhancedCTImage',
    'LegacyConvertedEnhancedMRImage',
    'LegacyConvertedEnhancedPETImage',
]
