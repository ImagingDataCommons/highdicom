"""Package for creation of Key Object Selection instances."""

from highdicom.ko.sop import KeyObjectSelectionDocument
from highdicom.ko.content import KeyObjectSelection

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.88.59',  # Key Object Selection Document
}

__all__ = [
    'KeyObjectSelection',
    'KeyObjectSelectionDocument',
]
