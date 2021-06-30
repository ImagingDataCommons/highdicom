import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import highdicom
import pydicom.sr.coding
from highdicom.enum import CoordinateSystemNames, UniversalEntityIDTypeValues
from highdicom.sr.coding import CodedConcept
from highdicom.sr.value_types import (
    CodeContentItem,
    ContentSequence,
    DateTimeContentItem,
    NumContentItem,
    TextContentItem,
)
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

# FIXME: Confirm that this is the correct interpretation of real world value
# mapping sequence. The std says that the sequence in the PFFG can hold one or
# more items, so I think I am supposed to make a "RealWorldValueMapping"
# class and then have the parametric map instance hold them in a data element
# sequence?

# Maybe add a note that you can specify codes.UCUM.NoUnits?

# For our purposes we will be passing in something like "resnet_bottleneck_0"
# or whatever for the label and "Resnet feature 0" or whatever for the
# explanation and NoUnits. What data goes into the lookup table?
# There will be one of these for each frame in the map, I think I will just
# add the LUTData as a parameter and maybe

# FIXME: Does this make sense to instantiate outside of the context of a map?
#       it has requirements based


class RealWorldValueMapping(Dataset):
    """Class representing the Real World Value Mapping Item Macro.

    Parameters
    ----------
    Dataset : [type]
        [description]
    """

    def __init__(
        self,
        lut_label: str,
        lut_explanation: str,
        measurement_unit: Union[
            highdicom.sr.CodedConcept, pydicom.sr.coding.Code
        ],
        *args,
        first_value_mapped: Optional[Union[int, float]] = None,
        last_value_mapped: Optional[Union[int, float]] = None,
        intercept: Optional[Union[int, float]] = None,
        slope: Optional[Union[int, float]] = None,
        lut_data: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Check 1C conditions, e.g. PixelData, when RWVM is added to an image's
        # functional groups sequence

        # Manage conditional requirements
        # FIXME: I got the conditional requirements wrong, gotta recheck them
        if lut_data is not None:
            # Raise valueErrors instead
            assert slope is None
            assert intercept is None
            assert first_value_mapped is not None
            assert last_value_mapped is not None
        else:
            if type(intercept) != type(slope):
                raise ValueError("Hell no")

        self.LUTExplanation = lut_explanation
        self.LUTLabel = lut_label

        if not isinstance(measurement_unit, (CodedConcept, Code)):
            raise TypeError(
                'Argument "unit" must have type CodedConcept or Code.'
            )
        if isinstance(measurement_unit, Code):
            unit = CodedConcept(*measurement_unit)

        self.MeasurementUnitsCodeSequence = [unit]

        # TODO: What the hell do I put here?
        """
When the Real World Value LUT Data (0040,9212) Attribute is supplied,
Real World Values are obtained via a lookup operation. The stored pixel value
of the first value mapped is mapped to the first entry in the LUT Data.
Subsequent stored pixel values are mapped to the subsequent entries in the
LUT Data up to a stored pixel value equal to the last value mapped.

The number of entries in the LUT data is given by:

Number of entries = Real World Value Last Value Mapped- Real World Value First Value Mapped + 1
        """
        self.RealWorldValueLUTData: List[float] = []

        # Set slope to 1 and intercept to 0
