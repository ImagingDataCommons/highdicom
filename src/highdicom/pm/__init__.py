"""Package for creation of Parametric Map instances."""

from highdicom.pm.content import DimensionIndexSequence, RealWorldValueMapping
from highdicom.pm.sop import ParametricMap

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.30',  # Parametric Map
}

__all__ = [
    'DimensionIndexSequence',
    'ParametricMap',
    'RealWorldValueMapping',
]
