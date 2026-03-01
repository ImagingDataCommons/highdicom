from collections.abc import Sequence
import warnings

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code

from highdicom.enum import CoordinateSystemNames
from highdicom.pixels import apply_lut
from highdicom.sr.coding import CodedConcept
from highdicom.sr.value_types import CodeContentItem
from highdicom.image import DimensionIndexSequence as BaseDimensionIndexSequence
from highdicom.valuerep import (
    _check_long_string,
    _check_short_string,
)


class RealWorldValueMapping(Dataset):
    """Class representing the Real World Value Mapping Item Macro. """

    def __init__(
        self,
        lut_label: str,
        lut_explanation: str,
        unit: CodedConcept | Code,
        value_range: tuple[int, int] | tuple[float, float],
        slope: int | float | None = None,
        intercept: int | float | None = None,
        lut_data: Sequence[float] | None = None,
        quantity_definition: CodedConcept | Code | None = None
    ) -> None:
        """
        Parameters
        ----------
        lut_label: str
            Label (identifier) used to identify transformation. Must be less
            than or equal to 16 characters.
        lut_explanation: str
            Explanation (short description) of the meaning of the transformation
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Unit of the real world values. This may be not applicable, because
            the values may not have a (known) unit. In this case, use
            ``pydicom.sr.codedict.codes.UCUM.NoUnits``.
        value_range: Union[Tuple[int, int], Tuple[float, float]]
            Upper and lower value of range of stored values to which the mapping
            should be restricted. For example, values may be stored as
            floating-point values with double precision, but limited to the
            range ``(-1.0, 1.0)`` or ``(0.0, 1.0)`` or stored as 16-bit
            unsigned integer values but limited to range ``(0, 4094)``.
            Note that the type of the values in ``value_range`` is significant
            and is used to determine whether values are stored as integers or
            floating-point values. Therefore, use ``(0.0, 1.0)`` instead of
            ``(0, 1)`` to specify a range of floating-point values.
        slope: Union[int, float, None], optional
            Slope of the linear mapping function applied to values in
            ``value_range``.
        intercept: Union[int, float, None], optional
            Intercept of the linear mapping function applied to values in
            ``value_range``.
        lut_data: Union[Sequence[int], Sequence[float], None], optional
            Sequence of values to serve as a lookup table for mapping stored
            values into real-world values in case of a non-linear relationship.
            The sequence should contain an entry for each value in the specified
            ``value_range`` such that
            ``len(sequence) == value_range[1] - value_range[0] + 1``.
            For example, in case of a value range of ``(0, 255)``, the sequence
            shall have ``256`` entries - one for each value in the given range.
        quantity_definition: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Description of the quantity represented by real world values
            (see :dcm:`CID 7180 <part16/sect_CID_7180.html>`
            "Abstract Multi-dimensional Image Model Component Semantics")

        Note
        ----
        Either ``slope`` and ``intercept`` or ``lut_data`` must be specified.
        Specify ``slope` and ``intercept``` if the mapping can be described by a
        linear function. Specify ``lut_data`` if the relationship between stored
        and real-world values is non-linear. Note, however, that a non-linear
        relationship can only be described for values that are stored as
        integers. Values stored as floating-point numbers must map linearly to
        real-world values.

        """  # noqa: E501
        super().__init__()

        _check_long_string(lut_explanation)
        self.LUTExplanation = str(lut_explanation)
        _check_short_string(lut_label)
        self.LUTLabel = str(lut_label)

        is_floating_point = any(isinstance(v, float) for v in value_range)
        if lut_data is not None:
            if slope is not None or intercept is not None:
                raise TypeError(
                    'Slope and intercept must be provided if LUT data is not '
                    'provided.'
                )
            if is_floating_point:
                raise ValueError(
                    'Only linear mapping is supported for floating-point '
                    'values. The range of values indicates that values are '
                    'as floating-point rather than integer values.'
                )
            n_actual = len(lut_data)
            n_expected = (int(value_range[1]) - int(value_range[0]) + 1)
            if n_actual != n_expected:
                raise ValueError(
                    'The LUT data sequence contains wrong number of entries: '
                    f'expected n={n_expected}, actual n={n_actual}.'
                )
            self.RealWorldValueLUTData = [float(v) for v in lut_data]
        else:
            if slope is None or intercept is None:
                raise TypeError(
                    'Slope and intercept must not be provided if LUT data is '
                    'provided.'
                )
            self.RealWorldValueSlope = float(slope)
            self.RealWorldValueIntercept = float(intercept)

        if is_floating_point:
            self.DoubleFloatRealWorldValueFirstValueMapped = float(
                value_range[0]
            )
            self.DoubleFloatRealWorldValueLastValueMapped = float(
                value_range[1]
            )
        else:
            self.RealWorldValueFirstValueMapped = int(value_range[0])
            self.RealWorldValueLastValueMapped = int(value_range[1])

        if not isinstance(unit, (CodedConcept, Code)):
            raise TypeError(
                'Argument "unit" must have type CodedConcept or Code.'
            )
        if isinstance(unit, Code):
            unit = CodedConcept(*unit)
        self.MeasurementUnitsCodeSequence = [unit]

        if quantity_definition is not None:
            quantity_item = CodeContentItem(
                name=CodedConcept(
                    value='246205007',
                    scheme_designator='SCT',
                    meaning='Quantity'
                ),
                value=quantity_definition
            )
            self.QuantityDefinitionSequence = [quantity_item]

    def has_lut(self) -> bool:
        """Determine whether the mapping contains a non-linear lookup table.

        Returns
        -------
        bool:
            True if the mapping contains a look-up table. False otherwise, when
            the mapping is represented by a slope and intercept defining a
            linear relationship.

        """
        return 'RealWorldValueLUTData' in self

    @property
    def lut_data(self) -> np.ndarray | None:
        """Union[numpy.ndarray, None] LUT data, if present."""
        if self.has_lut():
            return np.array(self.RealWorldValueLUTData)
        return None

    def is_floating_point(self) -> bool:
        """bool: Whether the value range is defined with floating point
        values."""
        return 'DoubleFloatRealWorldValueFirstValueMapped' in self

    @property
    def value_range(self) -> tuple[float, float]:
        """Tuple[float, float]: Range of valid input values."""
        if self.is_floating_point():
            return (
                self.DoubleFloatRealWorldValueFirstValueMapped,
                self.DoubleFloatRealWorldValueLastValueMapped,
            )
        return (
            float(self.RealWorldValueFirstValueMapped),
            float(self.RealWorldValueLastValueMapped),
        )

    def apply(
        self,
        array: np.ndarray,
    ) -> np.ndarray:
        """Apply the mapping to a pixel array.

        Parameters
        ----------
        apply: numpy.ndarray
            Pixel array to which the transform should be applied. Can be of any
            shape but must have an integer datatype if the mapping uses a LUT.

        Returns
        -------
        numpy.ndarray
            Array with LUT applied, will have data type ``numpy.float64``.

        """
        lut_data = self.lut_data
        if lut_data is not None:
            if array.dtype.kind not in ('u', 'i'):
                raise ValueError(
                    'Array must have an integer data type if the mapping '
                    'contains a LUT.'
                )
            first = self.RealWorldValueFirstValueMapped
            last = self.RealWorldValueLastValueMapped
            if len(lut_data) != last + 1 - first:
                raise RuntimeError(
                    "LUT data is stored with the incorrect number of elements."
                )

            return apply_lut(
                array=array,
                lut_data=lut_data,
                first_mapped_value=first,
                clip=False,  # values outside the range are undefined
            )
        else:
            slope = self.RealWorldValueSlope
            intercept = self.RealWorldValueIntercept

            first, last = self.value_range

            if array.min() < first or array.max() > last:
                raise ValueError(
                    'Array contains value outside the valid range.'
                )

            return array * slope + intercept


class DimensionIndexSequence(BaseDimensionIndexSequence):

    """Sequence of data elements describing dimension indices for the patient
    or slide coordinate system based on the Dimension Index functional
    group macro.

    Note
    ----
    The order of indices is fixed.

    Note
    ----
    This class is deprecated and will be removed in a future version of
    highdicom. User code should generally avoid this class, and if necessary,
    the more general :class:`highdicom.DimensionIndexSequence` should be used
    instead.

    """

    def __init__(
        self,
        coordinate_system: str | CoordinateSystemNames | None,
    ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.CoordinateSystemNames, None]
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
            imaging. If None, the imaging does not belong within a frame of
            reference.
        include_segment_number: bool
            Include the segment number as a dimension index.

        """
        warnings.warn(
            "The highdicom.pm.DimensionIndexSequence class is deprecated and "
            "will be removed in a future version of the library. User code "
            "should typically avoid this class, or, if required, use the more "
            "general highdicom.DimensionIndexSequence instead.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            coordinate_system=coordinate_system,
            functional_groups_module=(
                'segmentation-multi-frame-functional-groups'
            ),
        )
