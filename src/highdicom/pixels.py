"""Functional interface for pixel transformations."""
import numpy as np

from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.sequence import Sequence as pydicom_sequence
from pydicom.multival import MultiValue
from highdicom.enum import VOILUTFunctionValues
from highdicom.sr.coding import CodedConcept


def _parse_palette_color_lut_attributes(dataset: Dataset) -> tuple[
    bool,
    tuple[int, int, int],
    tuple[bytes, bytes, bytes],
]:
    """Extract information about palette color lookup table from a dataset.

    Performs various checks that the information retrieved is valid.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset containing Palette Color LUT information. Note that any
        number of other attributes may be included and will be ignored (for
        example allowing an entire image with Palette Color LUT information
        at the top level to be passed).

    Returns
    -------
    is_segmented: bool
        True if the LUT is segmented. False otherwise.
    descriptor: Tuple[int, int, int]
        Lookup table descriptor containing in this order the number of
        entries, first mapped value, and bits per entry. These values are
        shared between the three color LUTs.
    lut_data: Tuple[bytes, bytes, bytes]
        Raw bytes data for the red, green and blue LUTs.

    """
    is_segmented = 'SegmentedRedPaletteColorLookupTableData' in dataset

    if not is_segmented:
        if 'RedPaletteColorLookupTableData' not in dataset:
            raise AttributeError(
                'Dataset does not contain palette color lookup table '
                'attributes.'
            )

    descriptor = dataset.RedPaletteColorLookupTableDescriptor
    if len(descriptor) != 3:
        raise RuntimeError(
            'Invalid Palette Color LUT Descriptor'
        )
    number_of_entries, _, bits_per_entry = descriptor

    if number_of_entries == 0:
        number_of_entries = 2 ** 16

    strip_final_byte = False
    if bits_per_entry == 8:
        expected_num_bytes = number_of_entries
        if expected_num_bytes % 2 == 1:
            # Account for padding byte
            expected_num_bytes += 1
            strip_final_byte = True
    elif bits_per_entry == 16:
        expected_num_bytes = number_of_entries * 2
    else:
        raise RuntimeError(
            'Invalid number of bits per entry found in Palette Color '
            'LUT Descriptor.'
        )

    lut_data = []
    for color in ['Red', 'Green', 'Blue']:
        desc_kw = f'{color}PaletteColorLookupTableDescriptor'
        if desc_kw not in dataset:
            raise AttributeError(
                f"Dataset has no attribute '{desc_kw}'."
            )

        color_descriptor = getattr(dataset, desc_kw)
        if color_descriptor != descriptor:
            # Descriptors must match between all three colors
            raise RuntimeError(
                'Dataset has no mismatched palette color LUT '
                'descriptors.'
            )

        segmented_kw = f'Segmented{color}PaletteColorLookupTableData'
        standard_kw = f'{color}PaletteColorLookupTableData'
        if is_segmented:
            data_kw = segmented_kw
            wrong_data_kw = standard_kw
        else:
            data_kw = standard_kw
            wrong_data_kw = segmented_kw

        if data_kw not in dataset:
            raise AttributeError(
                f"Dataset has no attribute '{desc_kw}'."
            )
        if wrong_data_kw in dataset:
            raise AttributeError(
                "Mismatch of segmented LUT and standard LUT found."
            )

        lut_bytes = getattr(dataset, data_kw)
        if len(lut_bytes) != expected_num_bytes:
            raise RuntimeError(
                "LUT data has incorrect length"
            )
        if strip_final_byte:
            lut_bytes = lut_bytes[:-1]
        lut_data.append(lut_bytes)

    return (
        is_segmented,
        tuple(descriptor),
        tuple(lut_data)
    )


def _get_combined_palette_color_lut(
    dataset: Dataset,
) -> tuple[int, np.ndarray]:
    """Get a LUT array with three color channels from a dataset.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset containing Palette Color LUT information. Note that any number
        of other attributes may be included and will be ignored (for example
        allowing an entire image dataset with Palette Color LUT information at
        the top level to be passed).

    Returns
    -------
    first_mapped_value: int
        The first input value included in the LUT.
    lut_data: numpy.ndarray
        An NumPy array of shape (number_of_entries, 3) containing the red,
        green and blue lut data stacked along the final dimension of the
        array. Data type with be 8 or 16 bit unsigned integer depending on
        the number of bits per entry in the LUT.

    """
    (
        is_segmented,
        (number_of_entries, first_mapped_value, bits_per_entry),
        lut_data,
    ) = _parse_palette_color_lut_attributes(dataset)

    if is_segmented:
        raise RuntimeError(
            'Combined LUT data is not supported for segmented LUTs'
        )

    if bits_per_entry == 8:
        dtype = np.uint8
    else:
        dtype = np.uint16

    combined_array = np.stack(
        [np.frombuffer(buf, dtype=dtype) for buf in lut_data],
        axis=-1
    )

    # Account for padding byte
    if combined_array.shape[0] == number_of_entries + 1:
        combined_array = combined_array[:-1]

    return first_mapped_value, combined_array


def _check_rescale_dtype(
    input_dtype: np.dtype,
    output_dtype: np.dtype,
    intercept: float,
    slope: float,
    input_range: tuple[float, float] | None = None,
) -> None:
    """Checks whether it is appropriate to apply a given rescale to an array
    with a given dtype.

    Raises an error if not compatible.

    Parameters
    ----------
    input_dtype: numpy.dtype
        Datatype of the input array of the rescale operation.
    output_dtype: numpy.dtype
        Datatype of the output array of the rescale operation.
    intercept: float
        Intercept of the rescale operation.
    slope: float
        Slope of the rescale operation.
    input_range: Optional[Tuple[float, float]], optional
        Known limit of values for the input array. This could for example be
        deduced by the number of bits stored in an image. If not specified, the
        full range of values of the ``input_dtype`` is assumed.

    """
    slope_np = np.float64(slope)
    intercept_np = np.float64(intercept)

    # Check dtype is suitable
    if output_dtype.kind not in ('u', 'i', 'f'):
        raise ValueError(
            f'Data type "{output_dtype}" is not suitable.'
        )
    if output_dtype.kind in ('u', 'i'):
        if not (slope.is_integer() and intercept.is_integer()):
            raise ValueError(
                'An integer data type cannot be used if the slope '
                'or intercept is a non-integer value.'
            )
        if input_dtype.kind not in ('u', 'i'):
            raise ValueError(
                'An integer data type cannot be used if the input '
                'array is floating point.'
            )

        if output_dtype.kind == 'u' and intercept < 0.0:
            raise ValueError(
                'An unsigned integer data type cannot be used if the '
                'intercept is negative.'
            )

        if input_range is not None:
            input_min, input_max = input_range
        else:
            input_min = np.iinfo(input_dtype).min
            input_max = np.iinfo(input_dtype).max

        output_max = input_max * slope_np + intercept_np
        output_min = input_min * slope_np + intercept_np
        output_type_max = np.iinfo(output_dtype).max
        output_type_min = np.iinfo(output_dtype).min

        if output_max > output_type_max or output_min < output_type_min:
            raise ValueError(
                f'Datatype {output_dtype} does not have capacity for values '
                f'with slope {slope:.2f} and intercept {intercept:.2f}.'
            )


def _select_voi_window_center_width(
    dataset: Dataset,
    selector: int | str,
) -> tuple[float, float] | None:
    """Get a specific window center and width from a VOI LUT dataset.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to search for window center and width information. This must
        contain at a minimum the 'WindowCenter' and 'WindowWidth' attributes.
        Note that the dataset is not search recursively, only window
        information at the top level of the dataset is searched.
    selector: int | str
        Specification of the window to select. May either be an int or a str.
        If an int, it is interpreted as a (zero-based) index of the list of
        windows to apply. A negative integer may be used to index from the end
        of the list following standard Python indexing convention. If a str,
        the string that will be used to match the Window Center Width
        Explanation to choose from multiple voi windows. Note that such
        explanations are optional according to the standard and therefore may
        not be present.

    Returns
    -------
    tuple[float, float] | None:
        If the specified window is found in the dataset, it is returned as a
        tuple of (window center, window width). If it is not found, ``None`` is
        returned.

    """
    voi_center = dataset.WindowCenter
    voi_width = dataset.WindowWidth

    if isinstance(selector, str):
        explanations = dataset.get(
            'WindowCenterWidthExplanation'
        )
        if explanations is None:
            return None

        if isinstance(explanations, str):
            explanations = [explanations]

        try:
            selector = explanations.index(selector)
        except ValueError:
            return None

    if isinstance(voi_width, MultiValue):
        try:
            voi_width = voi_width[selector]
        except IndexError:
            return None
    elif selector not in (0, -1):
        return None

    if isinstance(voi_center, MultiValue):
        try:
            voi_center = voi_center[selector]
        except IndexError:
            return None
    elif selector not in (0, -1):
        return None

    return float(voi_center), float(voi_width)


def _select_voi_lut(
    dataset: Dataset,
    selector: int | str
) -> Dataset | None:
    """Get a specific VOI LUT dataset from dataset.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to search for VOI LUT information. This must contain the
        'VOILUTSequence'. Note that the dataset is not search recursively, only
        information at the top level of the dataset is searched.
    selector: int | str
        Specification of the LUT to select. May either be an int or a str. If
        an int, it is interpreted as a (zero-based) index of the sequence of
        LUTs to apply. A negative integer may be used to index from the end of
        the list following standard Python indexing convention. If a str, the
        string that will be used to match the LUT Explanation to choose from
        multiple voi LUTs. Note that such explanations are optional according
        to the standard and therefore may not be present.

    Returns
    -------
    pydicom.Dataset | None:
        If the LUT is found in the dataset, it is returned as a
        ``pydicom.Dataset``. If it is not found, ``None`` is returned.

    """
    if isinstance(selector, str):
        explanations = [
            ds.get('LUTExplanation') for ds in dataset.VOILUTSequence
        ]

        try:
            selector = explanations.index(selector)
        except ValueError:
            return None

    try:
        voi_lut_ds = dataset.VOILUTSequence[selector]
    except IndexError:
        return None

    return voi_lut_ds


def apply_voi_window(
    array: np.ndarray,
    window_center: float,
    window_width: float,
    voi_lut_function: (
        str |
        VOILUTFunctionValues
    ) = VOILUTFunctionValues.LINEAR,
    output_range: tuple[float, float] = (0.0, 1.0),
    dtype: type | str | np.dtype | None = np.float64,
    invert: bool = False,
) -> np.ndarray:
    """DICOM VOI windowing function.

    This function applies a "value-of-interest" window, defined by a window
    center and width, to a pixel array. Values within the window are rescaled
    to the output window, while values outside the range are clipped to the
    upper or lower value of the output range.

    Parameters
    ----------
    apply: numpy.ndarray
        Pixel array to which the transformation should be applied. Can be
        of any shape but must have an integer datatype if the
        transformation uses a LUT.
    window_center: float
        Center of the window.
    window_width: float
        Width of the window.
    voi_lut_function: Union[str, highdicom.VOILUTFunctionValues], optional
        Type of VOI LUT function.
    output_range: Tuple[float, float], optional
        Range of output values to which the VOI range is mapped.
    dtype: Union[type, str, numpy.dtype, None], optional
        Data type the output array. Should be a floating point data type.
    invert: bool, optional
        Invert the returned array such that the lowest original value in
        the LUT or input window is mapped to the upper limit and the
        highest original value is mapped to the lower limit. This may be
        used to efficiently combined a VOI LUT transformation with a
        presentation transform that inverts the range.

    Returns
    -------
    numpy.ndarray:
        Array with the VOI window function applied.

    """
    voi_lut_function = VOILUTFunctionValues(voi_lut_function)
    output_min, output_max = output_range
    if output_min >= output_max:
        raise ValueError(
            "Second value of 'output_range' must be higher than the first."
        )

    if dtype is None:
        dtype = np.dtype(np.float64)
    else:
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise ValueError(
                'dtype must be a floating point data type.'
            )

    window_width = dtype.type(window_width)
    window_center = dtype.type(window_center)
    if array.dtype != dtype:
        array = array.astype(dtype)

    if voi_lut_function in (
        VOILUTFunctionValues.LINEAR,
        VOILUTFunctionValues.LINEAR_EXACT,
    ):
        output_scale = (
            output_max - output_min
        )
        if voi_lut_function == VOILUTFunctionValues.LINEAR:
            # LINEAR uses the range
            # from c - 0.5w to c + 0.5w - 1
            scale_factor = (
                output_scale / (window_width - 1)
            )
        else:
            # LINEAR_EXACT uses the full range
            # from c - 0.5w to c + 0.5w
            scale_factor = output_scale / window_width

        window_min = window_center - window_width / 2.0
        if invert:
            array = (
                (window_min - array) * scale_factor +
                output_max
            )
        else:
            array = (
                (array - window_min) * scale_factor +
                output_min
            )

        array = np.clip(array, output_min, output_max)

    elif voi_lut_function == VOILUTFunctionValues.SIGMOID:
        if invert:
            offset_array = window_center - array
        else:
            offset_array = array - window_center
        exp_term = np.exp(
            -4.0 * offset_array /
            window_width
        )
        array = (
            (output_max - output_min) /
            (1.0 + exp_term)
        ) + output_min

    return array


def apply_lut(
    array: np.ndarray,
    lut_data: np.ndarray,
    first_mapped_value: int,
    dtype: type | str | np.dtype | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Apply a LUT to a pixel array.

    Parameters
    ----------
    apply: numpy.ndarray
        Pixel array to which the LUT should be applied. Can be of any shape
        but must have an integer datatype.
    lut_data: numpy.ndarray
        Lookup table data. The items in the LUT will be indexed down axis 0,
        but additional dimensions may optionally be included.
    first_mapped_value: int
        Input value that should be mapped to the first item in the LUT.
    dtype: Union[type, str, numpy.dtype, None], optional
        Datatype of the output array. If ``None``, the output data type will
        match that of the input ``lut_data``. Only safe casts are permitted.
    clip: bool
        If True, values in ``array`` outside the range of the LUT (i.e. those
        below ``first_mapped_value`` or those above ``first_mapped_value +
        len(lut_data) - 1``) are clipped to lie within the range before
        applying the LUT, meaning that after the LUT is applied they will take
        the first or last value in the LUT. If False, values outside the range
        of the LUT will raise an error.

    Returns
    -------
    numpy.ndarray
        Array with LUT applied.

    """
    if array.dtype.kind not in ('i', 'u'):
        raise ValueError(
            "Array must have an integer datatype."
        )

    if dtype is None:
        dtype = lut_data.dtype
    dtype = np.dtype(dtype)

    # Check dtype is suitable
    if dtype.kind not in ('u', 'i', 'f'):
        raise ValueError(
            f'Data type "{dtype}" is not suitable.'
        )

    if dtype != lut_data.dtype:
        # This is probably more efficient on average than applying the LUT and
        # then casting(?)
        lut_data = lut_data.astype(dtype, casting='safe')

    last_mapped_value = first_mapped_value + len(lut_data) - 1

    if clip:
        # Clip because values outside the range should be mapped to the
        # first/last value
        array = np.clip(array, first_mapped_value, last_mapped_value)
    else:
        if array.min() < first_mapped_value or array.max() > last_mapped_value:
            raise ValueError(
                'Array contains values outside the range of the LUT.'
            )

    if first_mapped_value != 0:
        # This is a common case and the subtraction may be slow, so avoid it if
        # not needed
        array = array - first_mapped_value

    return lut_data[array, ...]


def _select_real_world_value_map(
    sequence: pydicom_sequence,
    selector: int | str | CodedConcept | Code,
) -> Dataset | None:
    """Select a real world value map from a sequence.

    Parameters
    ----------
    sequence: pydicom.sequence.Sequence
        Sequence representing a Real World Value Mapping Sequence.
    selector: int | str | highdicom.sr.coding.CodedConcept | pydicom.sr.coding.Code
        Selector specifying an item in the sequence. If an integer, it is used
        as a index to the sequence in the usual way. If a string, the
        ``"LUTLabel"`` attribute of the items will be searched for a value that
        exactly matches the selector. If a code, the
        ``"MeasurementUnitsCodeSequence"`` will be searched for a value that
        matches the selector.

    Returns
    -------
    pydicom.Dataset | None:
        Either an item of the input sequence that matches the selector, or
        ``None`` if no such item is found.

    """  # noqa: E501
    if isinstance(selector, int):
        try:
            item = sequence[selector]
        except IndexError:
            return None

        return item

    elif isinstance(selector, str):
        labels = [item.LUTLabel for item in sequence]

        try:
            index = labels.index(selector)
        except ValueError:
            return None

        return sequence[index]

    elif isinstance(selector, (CodedConcept, Code)):
        units = [
            CodedConcept.from_dataset(item.MeasurementUnitsCodeSequence[0])
            for item in sequence
        ]
        try:
            index = units.index(selector)
        except ValueError:
            return None

        return sequence[index]
