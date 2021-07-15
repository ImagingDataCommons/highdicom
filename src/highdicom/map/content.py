from typing import Optional, Sequence, Tuple, Union

from highdicom.sr.coding import CodedConcept
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code


class RealWorldValueMapping(Dataset):
    """Class representing the Real World Value Mapping Item Macro. """

    def __init__(
        self,
        lut_label: str,
        lut_explanation: str,
        unit: Union[CodedConcept, Code],
        value_range: Union[Tuple[int, int], Tuple[float, float]],
        slope: Optional[Union[int, float]] = None,
        intercept: Optional[Union[int, float]] = None,
        lut_data: Optional[Sequence[float]] = None,
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
            unsigned integer values but limited to range ``(0, 4094).
            Note that the type of the values in `value_range` is significant
            and is used to determine whether values are stored as integers or
            floating-point values. Therefore, use ``(0.0, 1.0)`` instead of
            ``(0, 1)`` to specify a range of floating-point values.
        slope: Union[int, float, None], optional
            Slope of the linear mapping function applied to values in
            `value_range`.
        intercept: Union[int, float, None], optional
            Intercept of the linear mapping function applied to values in
            `value_range`.
        lut_data: Union[Sequence[int], Sequence[float] None], optional
            Sequence of values to serve as a lookup table for mapping stored
            values into real-world values in case of a non-linear relationship.
            The sequence should contain an entry for each value in the specified
            `value_range` such that
            ``len(sequence) == value_range[1] - value_range[0] + 1``.
            For example, in case of a value range of ``(0, 255)``, the sequence
            shall have ``256`` entries - one for each value in the given range.

        Note
        ----
        Either `slope` and `intercept` or `lut_data` must be specified.
        Specify `slope` and `intercept` if the mapping can be described by a
        linear function. Specify `lut_data` if the relationship between stored
        and real-world values is non-linear. Note, however, that a non-linear
        relationship can only be described for values that are stored as
        integers. Values stored as floating-point numbers must map linearly to
        real-world values.

        """
        super().__init__()

        if len(lut_label) > 16:
            raise ValueError(
                'lut_label must be less than or equal to 16 characters, '
                f'given {len(lut_label)}.'
            )

        self.LUTExplanation = str(lut_explanation)
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
