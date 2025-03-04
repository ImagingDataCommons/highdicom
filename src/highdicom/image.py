"""Tools for working with general DICOM images."""
from collections import Counter
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
import logging
from os import PathLike
import sqlite3
from typing import (
    Any,
    BinaryIO,
    cast,
)
from collections.abc import Iterable, Iterator, Generator, Sequence
from typing_extensions import Self

import numpy as np
from pydicom import Dataset
from pydicom.encaps import get_frame
from pydicom.tag import BaseTag
from pydicom.datadict import (
    get_entry,
    tag_for_keyword,
)
from pydicom.filebase import DicomIO, DicomBytesIO
from pydicom.multival import MultiValue
from pydicom.sr.coding import Code
from pydicom.uid import ParametricMapStorage

from highdicom._module_utils import (
    does_iod_have_pixel_data,
    is_multiframe_image,
)
from highdicom.base import SOPClass, _check_little_endian
from highdicom.color import ColorManager
from highdicom.content import LUT, VOILUTTransformation
from highdicom.enum import (
    CoordinateSystemNames,
)
from highdicom.frame import decode_frame
from highdicom.io import ImageFileReader, _wrapped_dcmread
from highdicom.pixels import (
    _check_rescale_dtype,
    _get_combined_palette_color_lut,
    _select_real_world_value_map,
    _select_voi_lut,
    _select_voi_window_center_width,
    apply_lut,
    apply_voi_window,
)
from highdicom.seg.enum import SpatialLocationsPreservedValues
from highdicom.spatial import (
    get_image_coordinate_system,
    get_series_volume_positions,
    get_volume_positions,
    is_tiled_image,
)
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID as UID
from highdicom.utils import (
    iter_tiled_full_frame_data,
)
from highdicom.volume import (
    _DCM_PYTHON_TYPE_MAP,
    VolumeGeometry,
    Volume,
    RGB_COLOR_CHANNEL_DESCRIPTOR,
)


logger = logging.getLogger(__name__)


# Dictionary mapping DCM VRs to appropriate SQLite types
_DCM_SQL_TYPE_MAP = {
    'CS': 'VARCHAR',
    'DS': 'REAL',
    'FD': 'REAL',
    'FL': 'REAL',
    'IS': 'INTEGER',
    'LO': 'TEXT',
    'LT': 'TEXT',
    'PN': 'TEXT',
    'SH': 'TEXT',
    'SL': 'INTEGER',
    'SS': 'INTEGER',
    'ST': 'TEXT',
    'UI': 'TEXT',
    'UL': 'INTEGER',
    'UR': 'TEXT',
    'US or SS': 'INTEGER',
    'US': 'INTEGER',
    'UT': 'TEXT',
}


class _ImageColorType(Enum):
    """Internal enum describing color arrangement of an image."""
    MONOCHROME = 'MONOCHROME'
    COLOR = 'COLOR'
    PALETTE_COLOR = 'PALETTE_COLOR'


def _deduce_color_type(image: Dataset):
    """Deduce the color type for an image.

    Parameters
    ----------
    image: pydicom.Dataset
        Image dataset.

    Returns
    -------
    _ImageColorType:
        Color type of the image.

    """
    photometric_interpretation = image.PhotometricInterpretation

    if photometric_interpretation in (
        'MONOCHROME1',
        'MONOCHROME2',
    ):
        return _ImageColorType.MONOCHROME
    elif photometric_interpretation == 'PALETTE COLOR':
        return _ImageColorType.PALETTE_COLOR
    return _ImageColorType.COLOR


class _CombinedPixelTransform:

    """Class representing a combined pixel transform.

    DICOM images contain multi-stage transforms to apply to the raw stored
    pixel values. This class is intended to provide a single class that
    configurably and efficiently applies the net effect of the selected
    transforms to stored pixel data.

    Depending on the parameters, it may perform operations related to the
    following:

    For monochrome images:
    * Real world value maps, which map stored pixels to real-world values and
    is independent of all other transforms
    * Modality LUT transform, which transforms stored pixel values to
    modality-specific values
    * Value-of-interest (VOI) LUT transform, which transforms the output
    of the Modality LUT transform to output values in order to focus on a
    particular region of intensities values of particular interest (such as a
    windowing operation).
    * Presentation LUT transform, which inverts the range of values for
    display.

    For pseudo-color images (stored as monochrome images but displayed as color
    images):
    * The Palette Color LUT transform, which maps stored single-sample
    pixel values to 3-samples-per-pixel RGB color images.

    For color images and pseudo-color images:
    * The ICCProfile, which performs color correction.

    """

    def __init__(
        self,
        image: Dataset,
        frame_index: int = 0,
        *,
        output_dtype: type | str | np.dtype = np.float64,
        apply_real_world_transform: bool | None = None,
        real_world_value_map_selector: int | str | Code | CodedConcept = 0,
        apply_modality_transform: bool | None = None,
        apply_voi_transform: bool | None = False,
        voi_transform_selector: int | str | VOILUTTransformation = 0,
        voi_output_range: tuple[float, float] = (0.0, 1.0),
        apply_presentation_lut: bool = True,
        apply_palette_color_lut: bool | None = None,
        remove_palette_color_values: Sequence[int] | None = None,
        palette_color_background_index: int = 0,
        apply_icc_profile: bool | None = None,
    ):
        """

        Parameters
        ----------
        image: pydicom.Dataset
            Image (single frame or multiframe) for which the pixel
            transform should be represented.
        frame_index: int
            Zero-based index (one less than the frame number).
        output_dtype: Union[type, str, numpy.dtype], optional
            Data type of the output array.
        apply_real_world_transform: bool | None, optional
            Whether to apply a real-world value map to the frame.
            A real-world value maps converts stored pixel values to output
            values with a real-world meaning, either using a LUT or a linear
            slope and intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if present but no error will be raised if
            it is not present.

            Note that if the dataset contains both a modality LUT and a real
            world value map, the real world value map will be applied
            preferentially. This also implies that specifying both
            ``apply_real_world_transform`` and ``apply_modality_transform`` to
            True is not permitted.
        real_world_value_map_selector: int | str | pydicom.sr.coding.Code | highdicom.sr.coding.CodedConcept, optional
            Specification of the real world value map to use (multiple may be
            present in the dataset). If an int, it is used to index the list of
            available maps. A negative integer may be used to index from the
            end of the list following standard Python indexing convention. If a
            str, the string will be used to match the ``"LUTLabel"`` attribute
            to select the map. If a ``pydicom.sr.coding.Code`` or
            ``highdicom.sr.coding.CodedConcept``, this will be used to match
            the units (contained in the ``"MeasurementUnitsCodeSequence"``
            attribute).
        apply_modality_transform: bool | None, optional
            Whether to apply the modality transform (if present in the
            dataset) to the frame. The modality transform maps stored pixel
            values to output values, either using a LUT or rescale slope and
            intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        apply_voi_transform: bool | None, optional
            Apply the value-of-interest (VOI) transform (if present in the
            dataset), which limits the range of pixel values to a particular
            range of interest using either a windowing operation or a LUT.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        voi_transform_selector: int | str | highdicom.VOILUTTransformation, optional
            Specification of the VOI transform to select (multiple may be
            present). May either be an int or a str. If an int, it is
            interpreted as a (zero-based) index of the list of VOI transforms
            to apply. A negative integer may be used to index from the end of
            the list following standard Python indexing convention. If a str,
            the string that will be used to match the
            ``"WindowCenterWidthExplanation"`` or the ``"LUTExplanation"``
            attributes to choose from multiple VOI transforms. Note that such
            explanations are optional according to the standard and therefore
            may not be present. Ignored if ``apply_voi_transform`` is ``False``
            or no VOI transform is included in the datasets.

            Alternatively, a user-defined
            :class:`highdicom.VOILUTTransformation` may be supplied.
            This will override any such transform specified in the dataset.
        voi_output_range: Tuple[float, float], optional
            Range of output values to which the VOI range is mapped. Only
            relevant if ``apply_voi_transform`` is True and a VOI transform is
            present.
        apply_palette_color_lut: bool | None, optional
            Apply the palette color LUT, if present in the dataset. The palette
            color LUT maps a single sample for each pixel stored in the dataset
            to a 3 sample-per-pixel color image.
        apply_presentation_lut: bool, optional
            Apply the presentation LUT transform to invert the pixel values. If
            the PresentationLUTShape is present with the value ``'INVERSE'``,
            or the PresentationLUTShape is not present but the Photometric
            Interpretation is MONOCHROME1, convert the range of the output
            pixels corresponds to MONOCHROME2 (in which high values are
            represent white and low values represent black). Ignored if
            PhotometricInterpretation is not MONOCHROME1 and the
            PresentationLUTShape is not present, or if a real world value
            transform is applied.
        remove_palette_color_values: Sequence[int] | None, optional
            Remove values from the palette color LUT (if any) by altering the
            LUT so that these values map to the RGB value at position
            ``palette_color_background_index`` instead of their original value.
            This is intended to remove segments from a palette color labelmap
            segmentation.
        palette_color_background_index: int, optional
            The index (i.e. input) of the palette color LUT that corresponds to
            background. Relevant only if ``remove_palette_color_values`` is
            provided.
        apply_icc_profile: bool | None, optional
            Whether colors should be corrected by applying an ICC
            transform. Will only be performed if metadata contain an
            ICC Profile.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present, but no error will be
            raised if it is not present.

        """  # noqa: E501
        if not does_iod_have_pixel_data(image.SOPClassUID):
            raise ValueError(
                'Input dataset does not represent an image.'
            )

        if not isinstance(
            voi_transform_selector,
            (int, str, VOILUTTransformation),
        ):
            raise TypeError(
                "Parameter 'voi_transform_selector' must have type 'int', "
                "'str', or 'highdicom.VOILUTTransformation'."
            )

        self._color_type = _deduce_color_type(image)

        if apply_real_world_transform is None:
            use_rwvm = True
            require_rwvm = False
        else:
            use_rwvm = bool(apply_real_world_transform)
            require_rwvm = use_rwvm

        if apply_modality_transform is None:
            use_modality = True
            require_modality = False
        else:
            use_modality = bool(apply_modality_transform)
            require_modality = use_modality

        if require_modality and require_rwvm:
            raise ValueError(
                "Setting both 'apply_real_world_transform' and "
                "'apply_modality_transform' to True is not "
                "permitted."
            )
        if require_rwvm:
            # No need to search for modality or VOI since they won't be used
            use_modality = False
            use_voi = False
        if require_modality:
            # No need to search for rwvm since it won't be used
            use_rwvm = False

        if apply_voi_transform is None:
            use_voi = True
            require_voi = False
        else:
            use_voi = bool(apply_voi_transform)
            require_voi = use_voi

        if use_voi and not use_modality:
            # The transform is dependent on first applying the modality
            # transform
            raise ValueError(
                "If 'apply_voi_transform' is True or None, "
                "'apply_modality_transform' cannot be False."
            )

        if require_rwvm and self._color_type != _ImageColorType.MONOCHROME:
            raise ValueError(
                'Real-world value map is required but the image is not '
                'a monochrome image.'
            )
        if require_modality and self._color_type != _ImageColorType.MONOCHROME:
            raise ValueError(
                'Modality transform is required but the image is not '
                'a monochrome image.'
            )
        if require_voi and self._color_type != _ImageColorType.MONOCHROME:
            raise ValueError(
                'VOI transform is required but the image is not '
                'a monochrome image.'
            )

        if apply_palette_color_lut is None:
            use_palette_color = True
            require_palette_color = False
        else:
            use_palette_color = bool(apply_palette_color_lut)
            require_palette_color = use_palette_color

        if (
            require_palette_color and self._color_type !=
            _ImageColorType.PALETTE_COLOR
        ):
            raise ValueError(
                'Palette color transform is required but the image is not '
                'a palette color image.'
            )

        if apply_icc_profile is None:
            use_icc = True
            require_icc = False
        else:
            use_icc = bool(apply_icc_profile)
            require_icc = use_icc

        if use_icc and not use_palette_color:
            # The transform is dependent on first applying the icc
            # transform
            raise ValueError(
                "If 'apply_icc_transform' is True or None, "
                "'apply_palette_color_lut' cannot be False."
            )

        if require_icc and self._color_type == _ImageColorType.MONOCHROME:
            raise ValueError(
                'ICC profile is required but the image is not '
                'a color or palette color image.'
            )

        if not isinstance(apply_presentation_lut, bool):
            raise TypeError(
                "Parameter 'apply_presentation_lut' must have type bool."
            )

        output_min, output_max = voi_output_range
        if output_min >= output_max:
            raise ValueError(
                "Second value of 'voi_output_range' must be higher than "
                "the first."
            )

        self.output_dtype = np.dtype(output_dtype)
        self.applies_to_all_frames = True
        self._input_range_check: tuple[int, int] | None = None
        self._voi_output_range = voi_output_range
        self._effective_lut_data: np.ndarray | None = None
        self._effective_lut_first_mapped_value = 0
        self._effective_window_center_width: tuple[float, float] | None = None
        self._effective_voi_function = None
        self._effective_slope_intercept: tuple[float, float] | None = None
        self._invert = False
        self._clip = True

        # Determine input type and range of values
        input_range = None
        if (
            image.SOPClassUID == ParametricMapStorage and
            image.BitsAllocated > 16
        ):
            # Parametric Maps are the only SOP Class (currently) that allows
            # floating point pixels
            if image.BitsAllocated == 32:
                self.input_dtype = np.dtype(np.float32)
            elif image.BitsAllocated == 64:
                self.input_dtype = np.dtype(np.float64)
        else:
            if image.PixelRepresentation == 1:
                if image.BitsAllocated == 8:
                    self.input_dtype = np.dtype(np.int8)
                elif image.BitsAllocated == 16:
                    self.input_dtype = np.dtype(np.int16)
                elif image.BitsAllocated == 32:
                    self.input_dtype = np.dtype(np.int32)

                # 2's complement to define the range
                half_range = 2 ** (image.BitsStored - 1)
                input_range = (-half_range, half_range - 1)
            else:
                if image.BitsAllocated == 1:
                    self.input_dtype = np.dtype(np.uint8)
                elif image.BitsAllocated == 8:
                    self.input_dtype = np.dtype(np.uint8)
                elif image.BitsAllocated == 16:
                    self.input_dtype = np.dtype(np.uint16)
                elif image.BitsAllocated == 32:
                    self.input_dtype = np.dtype(np.uint32)
                input_range = (0, 2 ** image.BitsStored - 1)

        if self._color_type == _ImageColorType.PALETTE_COLOR:
            # Note that some monochrome images have optional palette color
            # LUTs. Currently such LUTs will never be applied
            if use_palette_color:
                if 'SegmentedRedPaletteColorLookupTableData' in image:
                    # TODO
                    raise RuntimeError("Segmented LUTs are not implemented.")

                (
                    self._effective_lut_first_mapped_value,
                    self._effective_lut_data
                ) = _get_combined_palette_color_lut(image)

                # Zero out certain indices if requested
                if (
                    remove_palette_color_values is not None and
                    len(remove_palette_color_values) > 0
                ):
                    to_remove = np.array(
                        remove_palette_color_values
                    ) - self._effective_lut_first_mapped_value
                    target = (
                        palette_color_background_index -
                        self._effective_lut_first_mapped_value
                    )
                    self._effective_lut_data[
                        to_remove, :
                    ] = self._effective_lut_data[target, :]

        elif self._color_type == _ImageColorType.MONOCHROME:
            # Create a list of all datasets to check for transforms for
            # this frame, and whether they are shared by all frames
            datasets = [(image, True)]

            if 'SharedFunctionalGroupsSequence' in image:
                datasets.append(
                    (image.SharedFunctionalGroupsSequence[0], True)
                )

            if 'PerFrameFunctionalGroupsSequence' in image:
                datasets.append(
                    (
                        image.PerFrameFunctionalGroupsSequence[frame_index],
                        False,
                    )
                )

            modality_lut: LUT | None = None
            modality_slope_intercept: tuple[float, float] | None = None

            voi_lut: LUT | None = None
            voi_scaled_lut_data: np.ndarray | None = None
            voi_center_width: tuple[float, float] | None = None
            voi_function = 'LINEAR'
            invert = False
            has_rwvm = False

            if apply_presentation_lut:
                if 'PresentationLUTShape' in image:
                    invert = image.PresentationLUTShape == 'INVERSE'
                elif image.PhotometricInterpretation == 'MONOCHROME1':
                    invert = True

            if use_rwvm:
                for ds, is_shared in datasets:
                    rwvm_seq = ds.get('RealWorldValueMappingSequence')
                    if rwvm_seq is not None:

                        rwvm = _select_real_world_value_map(
                            rwvm_seq,
                            real_world_value_map_selector,
                        )

                        if rwvm is None:
                            raise IndexError(
                                "Requested 'real_world_value_map_selector' is "
                                "not present."
                            )

                        if 'RealWorldValueLUTData' in rwvm:
                            self._effective_lut_data = np.array(
                                rwvm.RealWorldValueLUTData
                            )
                            self._effective_lut_first_mapped_value = int(
                                rwvm.RealWorldValueFirstValueMapped
                            )
                            self._clip = False
                        else:
                            self._effective_slope_intercept = (
                                rwvm.RealWorldValueSlope,
                                rwvm.RealWorldValueIntercept,
                            )
                            if (
                                'DoubleFloatRealWorldValueFirstValueMapped'
                                in rwvm
                            ):
                                self._input_range_check = (
                                    rwvm.
                                    DoubleFloatRealWorldValueFirstValueMapped,
                                    rwvm.
                                    DoubleFloatRealWorldValueLastValueMapped
                                )
                            else:
                                self._input_range_check = (
                                    rwvm.RealWorldValueFirstValueMapped,
                                    rwvm.RealWorldValueLastValueMapped
                                )
                        self.applies_to_all_frames = (
                            self.applies_to_all_frames and is_shared
                        )
                        has_rwvm = True
                        break

            if require_rwvm and not has_rwvm:
                raise RuntimeError(
                    'A real-world value map is required but not found in the '
                    'image.'
                )

            if not has_rwvm and use_modality:

                if 'ModalityLUTSequence' in image:
                    modality_lut = LUT.from_dataset(
                        image.ModalityLUTSequence[0]
                    )
                else:
                    for ds, is_shared in datasets:
                        if 'PixelValueTransformationSequence' in ds:
                            sub_ds = ds.PixelValueTransformationSequence[0]
                        else:
                            sub_ds = ds

                        if (
                            'RescaleSlope' in sub_ds or
                            'RescaleIntercept' in sub_ds
                        ):
                            modality_slope_intercept = (
                                float(sub_ds.get('RescaleSlope', 1.0)),
                                float(sub_ds.get('RescaleIntercept', 0.0))
                            )
                            self.applies_to_all_frames = (
                                self.applies_to_all_frames and is_shared
                            )
                            break

            if (
                require_modality and
                modality_lut is None and
                modality_slope_intercept is None
            ):
                raise RuntimeError(
                    'A modality LUT transform is required but not found in '
                    'the image.'
                )

            if not has_rwvm and use_voi:

                if isinstance(voi_transform_selector, VOILUTTransformation):

                    if voi_transform_selector.has_lut():
                        if len(voi_transform_selector.VOILUTSequence) > 1:
                            raise ValueError(
                                "If providing a VOILUTTransformation as the "
                                "'voi_transform_selector', it must contain "
                                "a single transform."
                            )
                        voi_lut = voi_transform_selector.VOILUTSequence[0]
                    else:
                        voi_center = voi_transform_selector.WindowCenter
                        voi_width = voi_transform_selector.WindowWidth
                        if (
                            isinstance(voi_width, MultiValue) or
                            isinstance(voi_center, MultiValue)
                        ):
                            raise ValueError(
                                "If providing a VOILUTTransformation as the "
                                "'voi_transform_selector', it must contain "
                                "a single transform."
                            )
                        voi_center_width = (float(voi_center), float(voi_width))
                else:
                    # Need to find existing VOI LUT information
                    if 'VOILUTSequence' in image:

                        voi_lut_ds = _select_voi_lut(
                            image,
                            voi_transform_selector
                        )

                        if voi_lut_ds is None:
                            raise IndexError(
                                "Requested 'voi_transform_selector' is "
                                "not present."
                            )

                        voi_lut = LUT.from_dataset(voi_lut_ds)
                    else:
                        for ds, is_shared in datasets:
                            if 'FrameVOILUTSequence' in ds:
                                sub_ds = ds.FrameVOILUTSequence[0]
                            else:
                                sub_ds = ds

                            if (
                                'WindowCenter' in sub_ds or
                                'WindowWidth' in sub_ds
                            ):
                                voi_function = str(
                                    sub_ds.get('VOILUTFunction', 'LINEAR')
                                )

                                voi_center_width = (
                                    _select_voi_window_center_width(
                                        sub_ds,
                                        voi_transform_selector,
                                    )
                                )
                                if voi_center_width is None:
                                    raise IndexError(
                                        "Requested 'voi_transform_selector' is "
                                        'not present.'
                                    )
                                self.applies_to_all_frames = (
                                    self.applies_to_all_frames and is_shared
                                )
                                break

            if (
                require_voi and
                voi_center_width is None and
                voi_lut is None
            ):
                if has_rwvm:
                    raise RuntimeError(
                        'A VOI transform is required but is superseded by '
                        'a real world value transform.'
                    )
                else:
                    raise RuntimeError(
                        'A VOI transform is required but not found in '
                        'the image.'
                    )

            # Determine how to combine modality, voi and presentation
            # transforms
            if modality_lut is not None and not has_rwvm:
                if voi_center_width is not None:
                    # Apply the window function to the modality LUT
                    self._effective_lut_data = apply_voi_window(
                        array=modality_lut.lut_data,
                        window_center=voi_center_width[0],
                        window_width=voi_center_width[1],
                        output_range=voi_output_range,
                        dtype=output_dtype,
                        invert=invert,
                        voi_lut_function=voi_function,
                    )
                    self._effective_lut_first_mapped_value = (
                        modality_lut.first_mapped_value
                    )

                elif voi_lut is not None:
                    # "Compose" the two LUTs together by applying the
                    # second to the first
                    self._effective_lut_data = voi_lut.apply(
                        modality_lut.lut_data
                    )
                    self._effective_lut_first_mapped_value = (
                        modality_lut.first_mapped_value
                    )
                else:
                    # No VOI LUT transform so the modality lut operates alone
                    if invert:
                        self._effective_lut_data = (
                            modality_lut.get_inverted_lut_data()
                        )
                    else:
                        self._effective_lut_data = modality_lut.lut_data
                    self._effective_lut_first_mapped_value = (
                        modality_lut.first_mapped_value
                    )

            elif not has_rwvm:
                # modality LUT either doesn't exist or is a rescale/slope
                if modality_slope_intercept is not None:
                    slope, intercept = modality_slope_intercept
                else:
                    # No rescale slope found in dataset, so treat them as the
                    # 'identity' values
                    slope, intercept = (1.0, 0.0)

                if voi_center_width is not None:
                    # Shift and scale the window to account for the scaling
                    # and intercept
                    center, width = voi_center_width
                    self._effective_window_center_width = (
                        (center - intercept) / slope,
                        width / slope
                    )
                    self._effective_voi_function = voi_function
                    self._invert = invert

                elif voi_lut is not None:
                    # Shift and "scale" the LUT to account for the rescale
                    if not intercept.is_integer() and slope.is_integer():
                        raise ValueError(
                            "Cannot apply a VOI LUT when rescale intercept "
                            "or slope have non-integer values."
                        )
                    intercept = int(intercept)
                    slope = int(slope)
                    voi_scaled_lut_data = voi_lut.get_scaled_lut_data(
                        output_range=voi_output_range,
                        dtype=output_dtype,
                        invert=invert,
                    )
                    if slope != 1:
                        self._effective_lut_data = voi_scaled_lut_data[::slope]
                    else:
                        self._effective_lut_data = voi_scaled_lut_data
                    adjusted_first_value = (
                        (voi_lut.first_mapped_value - intercept) / slope
                    )
                    if not adjusted_first_value.is_integer():
                        raise ValueError(
                            "Cannot apply a VOI LUT when rescale intercept "
                            "or slope have non-integer values."
                        )
                    self._effective_lut_first_mapped_value = int(
                        adjusted_first_value
                    )
                else:
                    # No VOI LUT transform, so the modality rescale
                    # operates alone
                    if invert:
                        # Adjust the parameters to invert the intensities
                        # within the scaled and offset range
                        eff_slope = -slope
                        if input_range is None:
                            # This situation will be unusual: float valued
                            # pixels with a rescale transform that needs to
                            # be inverted. For simplicity, just invert
                            # the pixel values
                            eff_intercept = -intercept
                        else:
                            imin, imax = input_range
                            eff_intercept = (
                                slope * (imin + imax) + intercept
                            )
                        self._effective_slope_intercept = (
                            eff_slope, eff_intercept
                        )
                    else:
                        self._effective_slope_intercept = (
                            modality_slope_intercept
                        )

        self._color_manager = None
        if use_icc:
            if 'ICCProfile' in image:
                # ICC is normally at the top level of the dataset
                self._color_manager = ColorManager(image.ICCProfile)
            elif 'OpticalPathSequence' in image:
                # In certain microscopy images, ICC is in the optical paths
                # sequence
                if len(image.OpticalPathSequence) == 1:
                    optical_path_item = image.OpticalPathSequence[0]
                else:
                    # Multiple optical paths, need to find the identifier for
                    # this frame
                    identifier = None
                    if 'SharedFunctionalGroupsSequence' in image:
                        sfgs = image.SharedFunctionalGroupsSequence[0]
                        if 'OpticalPathIdentificationSequence' in sfgs:
                            identifier = (
                                sfgs
                                .OpticalPathIdentificationSequence[0]
                                .OpticalPathIdentifier
                            )

                    if 'PerFrameFunctionalGroupsSequence' in image:
                        pffg = image.PerFrameFunctionalGroupsSequence[
                            frame_index
                        ]
                        if 'OpticalPathIdentificationSequence' in pffg:
                            identifier = (
                                pffg
                                .OpticalPathIdentificationSequence[0]
                                .OpticalPathIdentifier
                            )
                            self.applies_to_all_frames = False

                    if identifier is None:
                        raise ValueError(
                            'Could not determine optical path identifier.'
                        )

                    for optical_path_item in image.OpticalPathSequence:
                        if (
                            optical_path_item.OpticalPathIdentifier ==
                            identifier
                        ):
                            break
                    else:
                        raise ValueError(
                            'No information on optical path found.'
                        )

                if 'ICCProfile' in optical_path_item:
                    self._color_manager = ColorManager(
                        optical_path_item.ICCProfile
                    )

        if require_icc and self._color_manager is None:
            raise RuntimeError(
                'An ICC profile is required but not found in '
                'the image.'
            )

        if self._effective_lut_data is not None:
            if self._color_manager is None:
                # If using palette color LUT, need to keep pixels as integers
                # to pass into color manager, otherwise eagerly converted the
                # LUT data to the requested output type
                if self._effective_lut_data.dtype != output_dtype:
                    self._effective_lut_data = (
                        self._effective_lut_data.astype(
                            output_dtype,
                            casting='safe',
                        )
                    )

            if self.input_dtype.kind == 'f':
                raise ValueError(
                    'Images with floating point data may not contain LUTs.'
                )

        # Slope/intercept of 1/0 is just a no-op
        if self._effective_slope_intercept is not None:
            if self._effective_slope_intercept == (1.0, 0.0):
                self._effective_slope_intercept = None

        if self._effective_slope_intercept is not None:
            slope, intercept = self._effective_slope_intercept
            _check_rescale_dtype(
                slope=slope,
                intercept=intercept,
                output_dtype=self.output_dtype,
                input_dtype=self.input_dtype,
                input_range=input_range,
            )
            self._effective_slope_intercept = (
                np.float64(slope).astype(self.output_dtype),
                np.float64(intercept).astype(self.output_dtype),
            )

        if self._effective_window_center_width is not None:
            if self.output_dtype.kind != 'f':
                raise ValueError(
                    'The VOI transform requires a floating point data '
                    'type.'
                )

        self.color_output = (
            self._color_type == _ImageColorType.COLOR or
            (
                self._color_type == _ImageColorType.PALETTE_COLOR and
                self._effective_lut_data is not None
            )
        )

        self.transfer_syntax_uid = image.file_meta.TransferSyntaxUID
        self.rows = image.Rows
        self.columns = image.Columns
        self.samples_per_pixel = image.SamplesPerPixel
        self.bits_allocated = image.BitsAllocated
        self.bits_stored = image.get('BitsAllocated', image.BitsAllocated)
        self.photometric_interpretation = image.PhotometricInterpretation
        self.pixel_representation = image.PixelRepresentation
        self.planar_configuration = image.get('PlanarConfiguration')

    def __call__(
        self,
        frame: np.ndarray | bytes,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Apply the composed transform.

        Parameters
        ----------
        frame: numpy.ndarray | bytes
            Input frame for the transform. Either a raw bytes array or the
            numpy array of the stored values.
        frame_index: int, optional
            Frame index. This is only required if frame is a raw bytes array,
            the number of bits allocated is 1 and the number of pixels per
            frame is not a multiple of 8. In this case, the frame index is
            required to extract the frame from the bytes array.

        Returns
        -------
        numpy.ndarray:
            Output frame after the transform is applied.

        """
        if isinstance(frame, bytes):
            frame_out = decode_frame(
                value=frame,
                transfer_syntax_uid=self.transfer_syntax_uid,
                rows=self.rows,
                columns=self.columns,
                samples_per_pixel=self.samples_per_pixel,
                bits_allocated=self.bits_allocated,
                bits_stored=self.bits_stored,
                photometric_interpretation=self.photometric_interpretation,
                pixel_representation=self.pixel_representation,
                planar_configuration=self.planar_configuration,
                index=frame_index,
            )
        elif isinstance(frame, np.ndarray):
            frame_out = frame
        else:
            raise TypeError(
                "Argument 'frame' must be either bytes or a numpy ndarray."
            )

        if self._color_type == _ImageColorType.COLOR:
            if frame_out.ndim != 3 or frame_out.shape[2] != 3:
                raise ValueError(
                    "Expected an image of shape (R, C, 3)."
                )

        else:
            if frame_out.ndim != 2:
                raise ValueError(
                    "Expected an image of shape (R, C)."
                )

        if self._input_range_check is not None:
            first, last = self._input_range_check
            if frame_out.min() < first or frame_out.max() > last:
                raise ValueError(
                    'Array contains value outside the valid range.'
                )

        if self._effective_lut_data is not None:
            frame_out = apply_lut(
                frame_out,
                self._effective_lut_data,
                self._effective_lut_first_mapped_value,
                clip=self._clip,
            )

        elif self._effective_slope_intercept is not None:
            slope, intercept = self._effective_slope_intercept

            # Avoid unnecessary array operations for efficiency
            if slope != 1.0:
                frame_out = frame_out * slope
            if intercept != 0.0:
                frame_out = frame_out + intercept

        elif self._effective_window_center_width is not None:
            frame_out = apply_voi_window(
                frame_out,
                window_center=self._effective_window_center_width[0],
                window_width=self._effective_window_center_width[1],
                dtype=self.output_dtype,
                invert=self._invert,
                output_range=self._voi_output_range,
                voi_lut_function=self._effective_voi_function or 'LINEAR',
            )

        if self._color_manager is not None:
            frame_out = self._color_manager.transform_frame(frame_out)

        if frame_out.dtype != self.output_dtype:
            frame_out = frame_out.astype(self.output_dtype)

        return frame_out


class _SQLTableDefinition:

    """Utility class holding the specification of a single SQL table."""

    def __init__(
        self,
        table_name: str,
        column_defs: Sequence[str],
        column_data: Iterable[Sequence[Any]],
    ):
        """

        Parameters
        ----------
        table_name: str,
            Name of the temporary table.
        column_data: Iterable[Sequence[Any]]
            Column data to place into the table.
        column_defs: Sequence[str]
            SQL syntax strings defining each column in the temporary table, one
            string per column.

        """
        self.table_name = table_name
        self.column_defs = list(column_defs)

        # It is important to convert numpy arrays to lists, otherwise the
        # values get inserted into the table as binary values and this leads to
        # very difficult to detect failures
        if isinstance(column_data, np.ndarray):
            column_data = column_data.tolist()

        sanitized_column_data = []

        for col in column_data:
            if isinstance(col, np.ndarray):
                col = col.tolist()

            sanitized_column_data.append(col)

        self.column_data = sanitized_column_data


class _Image(SOPClass):

    """Base class representing a general DICOM image.

    An "image" is any object representing an Image Information Entity.

    This class serves as a base class for specific image types, including
    Segmentations and Parametric Maps, as well as the general Image base class.

    """

    _coordinate_system: CoordinateSystemNames | None
    _is_tiled_full: bool
    _single_source_frame_per_frame: bool
    _dim_ind_pointers: list[BaseTag]
    # Mapping of tag value to (index column name, val column name(s))
    _dim_ind_col_names: dict[int, tuple[str, str | tuple[str, ...] | None]]
    _locations_preserved: SpatialLocationsPreservedValues | None
    _db_con: sqlite3.Connection
    _file_reader: ImageFileReader | None

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Create an Image from an existing pydicom Dataset.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset representing an image.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        Self:
            Image object from the input dataset.

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        _check_little_endian(dataset)

        # Checks on integrity of input dataset
        if copy:
            im = deepcopy(dataset)
        else:
            im = dataset
        im.__class__ = cls
        im = cast(Self, im)

        im._build_luts()
        return im

    @property
    def number_of_frames(self) -> int:
        """int: Number of frames in the image."""
        return self.get('NumberOfFrames', 1)

    def _get_color_type(self) -> _ImageColorType:
        """_ImageColorType: Color type of the image."""
        return _deduce_color_type(self)

    @property
    def is_tiled(self) -> bool:
        """bool: Whether the image is a tiled multi-frame image."""
        return is_tiled_image(self)

    @property
    def coordinate_system(self) -> CoordinateSystemNames | None:
        """highdicom.CoordinateSystemNames | None: Frame-of-reference
        coordinate system, if any, within which the image exists.

        """
        return self._coordinate_system

    def _standardize_frame_index(
        self,
        frame_number: int,
        as_index: bool,
    ) -> int:
        """Standardize frame index from different conventions.

        Also ensures that the frame index is valid in this image.

        Parameters
        ----------
        frame_number: int
            Number of the frame to retrieve. This is interpreted either as a
            1-based frame number (i.e. the first frame is numbered 1) or a
            0-based index instead (as is more common in Python), depending on
            value of the `as_index` parameter.
        as_index: bool
            Interpret the input `frame_number` as a 0-based index, instead of
            the default 1-based index.

        Returns
        -------
        int:
            Zero-based frame index, validated for use for this image.

        """
        if as_index:
            if frame_number < 0 or frame_number >= self.number_of_frames:
                raise IndexError(
                    f"Invalid frame index '{frame_number}' for image with "
                    f"{self.number_of_frames} frames. Note that frame numbers "
                    "use a 0-based index when 'as_index' is True."
                )

            return frame_number
        else:
            if frame_number < 1 or frame_number > self.number_of_frames:
                raise IndexError(
                    f"Invalid frame number '{frame_number}' for image with "
                    f"{self.number_of_frames} frame. Note that frame numbers "
                    "use a 1-based index."
                )

            return frame_number - 1

    def get_raw_frame(
        self,
        frame_number: int,
        as_index: bool = False,
    ) -> bytes:
        """Get the raw data for an encoded frame as bytes.

        Parameters
        ----------
        frame_number: int
            Number of the frame to retrieve. Under the default behavior, this
            is interpreted as a 1-based frame number (i.e. the first frame is
            numbered 1). This matches the convention used within DICOM when
            referring to frames within an image. To use a 0-based index instead
            (as is more common in Python), use the `as_index` parameter.
        as_index: bool
            Interpret the input `frame_number` as a 0-based index, instead of
            the default 1-based index.

        Returns
        -------
        bytes:
            Raw encoded data relating to the requested frame.

        Note
        ----
        In some situations, where the number of bits allocated is 1, the
        transfer syntax is not encapsulated (i.e. is native), and the number of
        pixels per frame is not a multiple of 8, frame boundaries are not
        aligned with byte boundaries in the raw bytes. In this situation, the
        returned bytes will contain the minimum range of bytes required to
        entirely contain the requested frame, however some bits may need
        stripping from the start and/or end to get the bits related to the
        requested frame.

        """
        frame_index = self._standardize_frame_index(frame_number, as_index)

        if self._file_reader is not None:
            with self._file_reader:
                return self._file_reader.read_frame_raw(frame_index)

        if UID(self.file_meta.TransferSyntaxUID).is_encapsulated:
            return get_frame(
                self.PixelData,
                index=frame_index,
                number_of_frames=self.number_of_frames,
            )
        else:
            if self.PhotometricInterpretation == 'YBR_FULL_422':
                # Account for subsampling of CB and CR when calculating
                # expected number of samples
                # See https://dicom.nema.org/medical/dicom/current/output/chtml
                # /part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2
                n_pixels = self.Rows * self.Columns * 2
            else:
                n_pixels = self.Rows * self.Columns * self.SamplesPerPixel

            frame_length_bits = self.BitsAllocated * n_pixels
            if self.BitsAllocated == 1 and (n_pixels % 8 != 0):
                start = (frame_index * frame_length_bits) // 8
                end = ((frame_index + 1) * frame_length_bits + 7) // 8
            else:
                frame_length = frame_length_bits // 8
                start = frame_index * frame_length
                end = start + frame_length

            return self.PixelData[start:end]

    def get_stored_frame(
        self,
        frame_number: int,
        as_index: bool = False,
    ) -> np.ndarray:
        """Get a single frame of stored values.

        Stored values are the pixel values stored within the dataset. They have
        been decompressed from the raw bytes (if necessary), interpreted as the
        correct pixel datatype (according to the pixel representation and
        planar configuration) and reshaped into a 2D (grayscale image) or 3D
        (color) NumPy array. However, no further pixel transform, such as
        the modality transform, VOI transforms, palette color LUTs, or ICC
        profile, has been applied.

        To get frames with pixel transforms applied (as is appropriate for
        most applications), use :func:`highdicom.Image.get_frame`
        instead.

        Parameters
        ----------
        frame_number: int
            Number of the frame to retrieve. Under the default behavior, this
            is interpreted as a 1-based frame number (i.e. the first frame is
            numbered 1). This matches the convention used within DICOM when
            referring to frames within an image. To use a 0-based index instead
            (as is more common in Python), use the `as_index` parameter.
        as_index: bool
            Interpret the input `frame_number` as a 0-based index, instead of
            the default 1-based index.

        Returns
        -------
        numpy.ndarray
            Numpy array of stored values. This will have shape (Rows, Columns)
            for a grayscale image, or (Rows, Columns, 3) for a color image. The
            data type will depend on how the pixels are stored in the file, and
            may be signed or unsigned integers or float.

        """
        frame_index = self._standardize_frame_index(frame_number, as_index)

        if self._pixel_array is None:
            raw_frame = self.get_raw_frame(frame_number, as_index=as_index)
            frame = decode_frame(
                value=raw_frame,
                transfer_syntax_uid=self.transfer_syntax_uid,
                rows=self.Rows,
                columns=self.Columns,
                samples_per_pixel=self.SamplesPerPixel,
                bits_allocated=self.BitsAllocated,
                bits_stored=self.get('BitsAllocated', self.BitsAllocated),
                photometric_interpretation=self.PhotometricInterpretation,
                pixel_representation=self.PixelRepresentation,
                planar_configuration=self.get('PlanarConfiguration'),
                index=frame_index,
            )
        else:
            if self.number_of_frames == 1:
                frame = self.pixel_array
            else:
                frame = self.pixel_array[frame_index]

        return frame

    def get_stored_frames(
        self,
        frame_numbers: Iterable[int] | None = None,
        as_indices: bool = False,
    ):
        """Get a stack of frames of stored values.

        Parameters
        ----------
        frame_numbers: Iterable[int] | None
            Iterable yielding the frame numbers. The returned array will have
            the specified frames stacked down the first dimension. Under the
            default behavior, the frame numbers are interpreted as a 1-based
            frame numbers (i.e. the first frame is numbered 1). This matches
            the convention used within DICOM when referring to frames within an
            image. To use 0-based indices instead (as is more common in
            Python), use the `as_indices` parameter. If ``None``, all frames
            are retrieved in the order they are stored in the image.
        as_indices: bool
            Interpret each item in the input `frame_numbers` as a 0-based
            index, instead of the default behavior of interpreting them as
            1-based frame numbers.

        Returns
        -------
        numpy.ndarray
            Numpy array of stored values. This will have shape (N, Rows,
            Columns) for a grayscale image, or (N, Rows, Columns, 3) for a
            color image, where ``N`` is the length of the input
            ``frame_numbers`` (or the number of frames in the image if
            ``frame_numbers`` is ``None``). The data type will depend on how
            the pixels are stored in the file, and may be signed or unsigned
            integers or float.

        """  # noqa; E501
        if frame_numbers is None:
            if as_indices:
                frame_numbers = range(0, self.number_of_frames)
            else:
                frame_numbers = range(1, self.number_of_frames + 1)

        context_manager = (
            self._file_reader
            if self._file_reader is not None
            else nullcontext()
        )

        output_frames = []
        with context_manager:

            # loop through output frames
            for frame_number in frame_numbers:

                frame_index = self._standardize_frame_index(
                    frame_number,
                    as_indices,
                )
                if self._pixel_array is None:
                    raw_frame = self.get_raw_frame(
                        frame_number,
                        as_index=as_indices
                    )
                    frame = decode_frame(
                        value=raw_frame,
                        transfer_syntax_uid=self.transfer_syntax_uid,
                        rows=self.Rows,
                        columns=self.Columns,
                        samples_per_pixel=self.SamplesPerPixel,
                        bits_allocated=self.BitsAllocated,
                        bits_stored=self.get(
                            'BitsAllocated',
                            self.BitsAllocated
                        ),
                        photometric_interpretation=(
                            self.PhotometricInterpretation
                        ),
                        pixel_representation=self.PixelRepresentation,
                        planar_configuration=self.get('PlanarConfiguration'),
                        index=frame_index,
                    )
                else:
                    if self.number_of_frames == 1:
                        frame = self.pixel_array
                    else:
                        frame = self.pixel_array[frame_index]

                output_frames.append(frame)

        return np.stack(output_frames)

    def get_frame(
        self,
        frame_number: int,
        as_index: bool = False,
        *,
        dtype: type | str | np.dtype = np.float64,
        apply_real_world_transform: bool | None = None,
        real_world_value_map_selector: int | str | Code | CodedConcept = 0,
        apply_modality_transform: bool | None = None,
        apply_voi_transform: bool | None = False,
        voi_transform_selector: int | str | VOILUTTransformation = 0,
        voi_output_range: tuple[float, float] = (0.0, 1.0),
        apply_presentation_lut: bool = True,
        apply_palette_color_lut: bool | None = None,
        apply_icc_profile: bool | None = None,
    ) -> np.ndarray:
        """Get a single frame of pixels, with transforms applied.

        This method retrieves a frame of stored values and applies various
        intensity transforms specified within the dataset to them, depending on
        the options provided.

        Parameters
        ----------
        frame_number: int
            Number of the frame to retrieve. Under the default behavior, this
            is interpreted as a 1-based frame number (i.e. the first frame is
            numbered 1). This matches the convention used within DICOM when
            referring to frames within an image. To use a 0-based index instead
            (as is more common in Python), use the `as_index` parameter.
        as_index: bool
            Interpret the input `frame_number` as a 0-based index, instead of
            the default behavior of interpreting it as a 1-based frame number.
        dtype: Union[type, str, numpy.dtype],
            Data type of the output array.
        apply_real_world_transform: bool | None, optional
            Whether to apply a real-world value map to the frame.
            A real-world value maps converts stored pixel values to output
            values with a real-world meaning, either using a LUT or a linear
            slope and intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if present but no error will be raised if
            it is not present.

            Note that if the dataset contains both a modality LUT and a real
            world value map, the real world value map will be applied
            preferentially. This also implies that specifying both
            ``apply_real_world_transform`` and ``apply_modality_transform`` to
            True is not permitted.
        real_world_value_map_selector: int | str | pydicom.sr.coding.Code | highdicom.sr.coding.CodedConcept, optional
            Specification of the real world value map to use (multiple may be
            present in the dataset). If an int, it is used to index the list of
            available maps. A negative integer may be used to index from the
            end of the list following standard Python indexing convention. If a
            str, the string will be used to match the ``"LUTLabel"`` attribute
            to select the map. If a ``pydicom.sr.coding.Code`` or
            ``highdicom.sr.coding.CodedConcept``, this will be used to match
            the units (contained in the ``"MeasurementUnitsCodeSequence"``
            attribute).
        apply_modality_transform: bool | None, optional
            Whether to apply the modality transform (if present in the
            dataset) to the frame. The modality transform maps stored pixel
            values to output values, either using a LUT or rescale slope and
            intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        apply_voi_transform: bool | None, optional
            Apply the value-of-interest (VOI) transform (if present in the
            dataset) which limits the range of pixel values to a particular
            range of interest, using either a windowing operation or a LUT.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        voi_transform_selector: int | str | highdicom.VOILUTTransformation, optional
            Specification of the VOI transform to select (multiple may be
            present). May either be an int or a str. If an int, it is
            interpreted as a (zero-based) index of the list of VOI transforms
            to apply. A negative integer may be used to index from the end of
            the list following standard Python indexing convention. If a str,
            the string that will be used to match the
            ``"WindowCenterWidthExplanation"`` or the ``"LUTExplanation"``
            attributes to choose from multiple VOI transforms. Note that such
            explanations are optional according to the standard and therefore
            may not be present. Ignored if ``apply_voi_transform`` is ``False``
            or no VOI transform is included in the datasets.

            Alternatively, a user-defined
            :class:`highdicom.VOILUTTransformation` may be supplied.
            This will override any such transform specified in the dataset.
        voi_output_range: Tuple[float, float], optional
            Range of output values to which the VOI range is mapped. Only
            relevant if ``apply_voi_transform`` is True and a VOI transform is
            present.
        apply_palette_color_lut: bool | None, optional
            Apply the palette color LUT, if present in the dataset. The palette
            color LUT maps a single sample for each pixel stored in the dataset
            to a 3 sample-per-pixel color image.
        apply_presentation_lut: bool, optional
            Apply the presentation LUT transform to invert the pixel values. If
            the PresentationLUTShape is present with the value ``'INVERSE'``,
            or the PresentationLUTShape is not present but the Photometric
            Interpretation is MONOCHROME1, convert the range of the output
            pixels corresponds to MONOCHROME2 (in which high values are
            represent white and low values represent black). Ignored if
            PhotometricInterpretation is not MONOCHROME1 and the
            PresentationLUTShape is not present, or if a real world value
            transform is applied.
        apply_icc_profile: bool | None, optional
            Whether colors should be corrected by applying an ICC
            transform. Will only be performed if metadata contain an
            ICC Profile.

        Returns
        -------
        numpy.ndarray
            Numpy array of frame with pixel transforms applied. This will have
            shape (Rows, Columns) for a grayscale image, or (Rows, Columns, 3)
            for a color image. The data type is controlled by the ``dtype``
            parameter.

        """  # noqa; E501
        frame_index = self._standardize_frame_index(frame_number, as_index)

        frame = self.get_stored_frame(frame_number, as_index=as_index)

        frame_transform = _CombinedPixelTransform(
            self,
            frame_index=frame_index,
            output_dtype=dtype,
            apply_real_world_transform=apply_real_world_transform,
            real_world_value_map_selector=real_world_value_map_selector,
            apply_modality_transform=apply_modality_transform,
            apply_voi_transform=apply_voi_transform,
            voi_transform_selector=voi_transform_selector,
            voi_output_range=voi_output_range,
            apply_presentation_lut=apply_presentation_lut,
            apply_palette_color_lut=apply_palette_color_lut,
            apply_icc_profile=apply_icc_profile,
        )

        return frame_transform(frame)

    def get_frames(
        self,
        frame_numbers: Iterable[int] | None = None,
        as_indices: bool = False,
        *,
        dtype: type | str | np.dtype = np.float64,
        apply_real_world_transform: bool | None = None,
        real_world_value_map_selector: int | str | Code | CodedConcept = 0,
        apply_modality_transform: bool | None = None,
        apply_voi_transform: bool | None = False,
        voi_transform_selector: int | str | VOILUTTransformation = 0,
        voi_output_range: tuple[float, float] = (0.0, 1.0),
        apply_presentation_lut: bool = True,
        apply_palette_color_lut: bool | None = None,
        apply_icc_profile: bool | None = None,
    ):
        """Get a stack of frames, with transforms applied.

        This method retrieves frames of stored values and applies various
        intensity transforms specified within the dataset to them, depending on
        the options provided.

        Parameters
        ----------
        frame_numbers: Iterable[int] | None
            Iterable yielding the frame numbers. The returned array will have
            the specified frames stacked down the first dimension. Under the
            default behavior, the frame numbers are interpreted as a 1-based
            frame numbers (i.e. the first frame is numbered 1). This matches
            the convention used within DICOM when referring to frames within an
            image. To use 0-based indices instead (as is more common in
            Python), use the `as_indices` parameter. If ``None``, all frames
            are retrieved in the order they are stored in the image.
        as_indices: bool
            Interpret each item in the input `frame_numbers` as a 0-based
            index, instead of the default behavior of interpreting them as
            1-based frame numbers.
        dtype: Union[type, str, numpy.dtype],
            Data type of the output array.
        apply_real_world_transform: bool | None, optional
            Whether to apply a real-world value map to the frame.
            A real-world value maps converts stored pixel values to output
            values with a real-world meaning, either using a LUT or a linear
            slope and intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if present but no error will be raised if
            it is not present.

            Note that if the dataset contains both a modality LUT and a real
            world value map, the real world value map will be applied
            preferentially. This also implies that specifying both
            ``apply_real_world_transform`` and ``apply_modality_transform`` to
            True is not permitted.
        real_world_value_map_selector: int | str | pydicom.sr.coding.Code | highdicom.sr.coding.CodedConcept, optional
            Specification of the real world value map to use (multiple may be
            present in the dataset). If an int, it is used to index the list of
            available maps. A negative integer may be used to index from the
            end of the list following standard Python indexing convention. If a
            str, the string will be used to match the ``"LUTLabel"`` attribute
            to select the map. If a ``pydicom.sr.coding.Code`` or
            ``highdicom.sr.coding.CodedConcept``, this will be used to match
            the units (contained in the ``"MeasurementUnitsCodeSequence"``
            attribute).
        apply_modality_transform: bool | None, optional
            Whether to apply the modality transform (if present in the
            dataset) to the frame. The modality transform maps stored pixel
            values to output values, either using a LUT or rescale slope and
            intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        apply_voi_transform: bool | None, optional
            Apply the value-of-interest (VOI) transform (if present in the
            dataset) which limits the range of pixel values to a particular
            range of interest, using either a windowing operation or a LUT.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        voi_transform_selector: int | str | highdicom.VOILUTTransformation, optional
            Specification of the VOI transform to select (multiple may be
            present). May either be an int or a str. If an int, it is
            interpreted as a (zero-based) index of the list of VOI transforms
            to apply. A negative integer may be used to index from the end of
            the list following standard Python indexing convention. If a str,
            the string that will be used to match the
            ``"WindowCenterWidthExplanation"`` or the ``"LUTExplanation"``
            attributes to choose from multiple VOI transforms. Note that such
            explanations are optional according to the standard and therefore
            may not be present. Ignored if ``apply_voi_transform`` is ``False``
            or no VOI transform is included in the datasets.

            Alternatively, a user-defined
            :class:`highdicom.VOILUTTransformation` may be supplied.
            This will override any such transform specified in the dataset.
        voi_output_range: Tuple[float, float], optional
            Range of output values to which the VOI range is mapped. Only
            relevant if ``apply_voi_transform`` is True and a VOI transform is
            present.
        apply_palette_color_lut: bool | None, optional
            Apply the palette color LUT, if present in the dataset. The palette
            color LUT maps a single sample for each pixel stored in the dataset
            to a 3 sample-per-pixel color image.
        apply_presentation_lut: bool, optional
            Apply the presentation LUT transform to invert the pixel values. If
            the PresentationLUTShape is present with the value ``'INVERSE'``,
            or the PresentationLUTShape is not present but the Photometric
            Interpretation is MONOCHROME1, convert the range of the output
            pixels corresponds to MONOCHROME2 (in which high values are
            represent white and low values represent black). Ignored if
            PhotometricInterpretation is not MONOCHROME1 and the
            PresentationLUTShape is not present, or if a real world value
            transform is applied.
        apply_icc_profile: bool | None, optional
            Whether colors should be corrected by applying an ICC
            transform. Will only be performed if metadata contain an
            ICC Profile.

        Returns
        -------
        numpy.ndarray
            Numpy array of frames with pixel transforms applied. This will have
            shape (N, Rows, Columns) for a grayscale image, or (N, Rows,
            Columns, 3) for a color image, where ``N`` is the length of the
            input ``frame_numbers`` (or the number of frames in the image if
            ``frame_numbers`` is ``None``). The data type is controlled by the
            ``dtype`` parameter.

        """  # noqa; E501
        if frame_numbers is None:
            if as_indices:
                frame_numbers = range(0, self.number_of_frames)
            else:
                frame_numbers = range(1, self.number_of_frames + 1)

        shared_frame_transform = _CombinedPixelTransform(
            self,
            apply_real_world_transform=apply_real_world_transform,
            real_world_value_map_selector=real_world_value_map_selector,
            apply_modality_transform=apply_modality_transform,
            apply_voi_transform=apply_voi_transform,
            voi_transform_selector=voi_transform_selector,
            voi_output_range=voi_output_range,
            apply_presentation_lut=apply_presentation_lut,
            apply_palette_color_lut=apply_palette_color_lut,
            apply_icc_profile=apply_icc_profile,
            output_dtype=dtype,
        )

        context_manager = (
            self._file_reader
            if self._file_reader is not None
            else nullcontext()
        )

        output_frames = []
        with context_manager:

            # loop through output frames
            for frame_number in frame_numbers:

                frame_index = self._standardize_frame_index(
                    frame_number,
                    as_indices,
                )

                if shared_frame_transform.applies_to_all_frames:
                    frame_transform = shared_frame_transform
                else:
                    frame_transform = _CombinedPixelTransform(
                        self,
                        frame_index=frame_index,
                        apply_real_world_transform=apply_real_world_transform,
                        real_world_value_map_selector=real_world_value_map_selector,  # noqa: E501
                        apply_modality_transform=apply_modality_transform,
                        apply_voi_transform=apply_voi_transform,
                        voi_transform_selector=voi_transform_selector,
                        voi_output_range=voi_output_range,
                        apply_presentation_lut=apply_presentation_lut,
                        apply_palette_color_lut=apply_palette_color_lut,
                        apply_icc_profile=apply_icc_profile,
                        output_dtype=dtype,
                    )

                if self._pixel_array is None:
                    if self._file_reader is not None:
                        frame_bytes = self._file_reader.read_frame_raw(
                            frame_index
                        )
                    else:
                        frame_bytes = self.get_raw_frame(frame_index + 1)
                    frame = frame_transform(frame_bytes, frame_index)
                else:
                    if self.pixel_array.ndim == 2:
                        if frame_index == 0:
                            frame = self.pixel_array
                        else:
                            raise IndexError(
                                f'Index {frame_index} is out of bounds for '
                                'an image with a single frame.'
                            )
                    else:
                        frame = self.pixel_array[frame_index]
                    frame = frame_transform(frame, frame_index)

                output_frames.append(frame)

        return np.stack(output_frames)

    @property
    def pixel_array(self):
        """Get the full pixel array of stored values for all frames.

        This method is consistent with the behavior of the pydicom Dataset
        class, but additionally functions correctly when lazy frame retrieval
        is used.

        Returns
        -------
        numpy.ndarray:
            Full pixel array of stored values, ordered by frame number. Shape
            is (frames, rows, columns, samples). The frame dimension is omitted
            if it is equal to 1. The samples dimension is omitted for grayscale
            images, and 3 for color images.

        """
        if self._file_reader is not None:
            if self._pixel_array is None:
                # Need to override pydicom's behavior here to perform lazy
                # reading
                if self.number_of_frames == 1:
                    pixel_array = self.get_stored_frame(1)
                else:
                    pixel_array = self.get_stored_frames()
                self._pixel_array = pixel_array
            else:
                # pydicom will complain about missing PixelData even if
                # self._pixel_array is already cached
                return self._pixel_array

        # Defer to pydicom
        return super().pixel_array

    def __getstate__(self) -> dict[str, Any]:
        """Get the state for pickling.

        This is required to work around the fact that a sqlite3
        Connection object cannot be pickled.

        Returns
        -------
        Dict[str, Any]:
            State of the object.

        """
        state = super().__dict__.copy()

        db_data = self._serialize_db()

        del state['_db_con']
        state['db_data'] = db_data

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object.

        This is required to work around the fact that a sqlite3
        Connection object cannot be pickled.

        Parameters
        ----------
        state: Dict[str, Any]
            State of the object.

        """
        self._db_con = sqlite3.connect(':memory:')
        with self._db_con:
            self._db_con.executescript(state['db_data'].decode('utf-8'))

        del state['db_data']

        self.__dict__.update(state)

    def _serialize_db(self) -> bytes:
        """Get a serialized copy of the internal database.

        Returns
        -------
        bytes:
            Serialized copy of the internal database.

        """
        return b''.join(
            [
                line.encode('utf-8')
                for line in self._db_con.iterdump()
            ]
        )

    def _build_luts(self) -> None:
        """Build lookup tables for efficient querying.

        Two lookup tables are currently constructed. The first maps the
        SOPInstanceUIDs of all datasets referenced in the image to a
        tuple containing the StudyInstanceUID, SeriesInstanceUID and
        SOPInstanceUID.

        The second look-up table contains information about each frame of the
        segmentation, including the segment it contains, the instance and frame
        from which it was derived (if these are unique), and its dimension
        index values.

        """
        self._file_reader = None
        self._coordinate_system = get_image_coordinate_system(
            self
        )

        if is_multiframe_image(self):
            self._build_luts_multiframe()
        else:
            self._build_luts_single_frame()

    def _build_luts_single_frame(self) -> None:
        """Populates LUT information for a single frame image."""
        self._is_tiled_full = False
        self._dim_ind_pointers = []
        self._dim_ind_col_names = {}
        self._single_source_frame_per_frame = False
        self._locations_preserved = None
        self._missing_reference_instances = []
        referenced_uids = self._get_ref_instance_uids()
        all_referenced_sops = {uids[2] for uids in referenced_uids}

        col_defs = []
        col_defs.append('FrameNumber INTEGER PRIMARY KEY')
        col_data = [[1]]

        if self._coordinate_system == CoordinateSystemNames.PATIENT:
            for t in [
                0x0020_0032,  # ImagePositionPatient
                0x0020_0037,  # ImageOrientionPatient
                0x0028_0030,  # PixelSpacing
                0x0018_0088,  # SpacingBetweenSlices
            ]:
                vr, vm_str, _, _, kw = get_entry(t)

                vm = int(vm_str)
                sql_type = _DCM_SQL_TYPE_MAP[vr]

                if kw in self:
                    v = self.get(kw)
                    if vm > 1:
                        for i, v in enumerate(v):
                            col_defs.append(f'{kw}_{i} {sql_type} NOT NULL')
                            col_data.append([v])
                    else:
                        col_defs.append(f'{kw} {sql_type} NOT NULL')
                        col_data.append([v])

        if 'SourceImageSequence' in self:
            self._single_source_frame_per_frame = (
                len(self.SourceImageSequence) == 1
            )
            locations_preserved = [
                item.get('SpatialLocationsPreserved')
                for item in self.SourceImageSequence
            ]
            if all(
                v is not None and v == "YES" for v in locations_preserved
            ):
                self._locations_preserved = (
                    SpatialLocationsPreservedValues.YES
                )
            elif all(
                v is not None and v == "NO" for v in locations_preserved
            ):
                self._locations_preserved = (
                    SpatialLocationsPreservedValues.NO
                )
            if self._single_source_frame_per_frame:
                ref_frame = self.SourceImageSequence[0].get(
                    'ReferencedFrameNumber'
                )
                ref_uid = self.SourceImageSequence[0].ReferencedSOPInstanceUID
                if ref_uid not in all_referenced_sops:
                    self._missing_reference_instances.append(ref_uid)
                col_defs.append('ReferencedFrameNumber INTEGER')
                col_defs.append('ReferencedSOPInstanceUID VARCHAR NOT NULL')
                col_defs.append(
                    'FOREIGN KEY(ReferencedSOPInstanceUID) '
                    'REFERENCES InstanceUIDs(SOPInstanceUID)'
                )
                col_data += [
                    [ref_frame],
                    [ref_uid],
                ]

        referenced_uids = self._get_ref_instance_uids()
        self._db_con = sqlite3.connect(":memory:")
        self._create_ref_instance_table(referenced_uids)

        self._create_frame_lut(col_defs, col_data)

    def _build_luts_multiframe(self) -> None:
        """Build lookup tables for efficient querying.

        Two lookup tables are currently constructed. The first maps the
        SOPInstanceUIDs of all datasets referenced in the image to a
        tuple containing the StudyInstanceUID, SeriesInstanceUID and
        SOPInstanceUID.

        The second look-up table contains information about each frame of the
        segmentation, including the segment it contains, the instance and frame
        from which it was derived (if these are unique), and its dimension
        index values.

        """
        referenced_uids = self._get_ref_instance_uids()
        all_referenced_sops = {uids[2] for uids in referenced_uids}

        self._is_tiled_full = (
            hasattr(self, 'DimensionOrganizationType') and
            self.DimensionOrganizationType == 'TILED_FULL'
        )

        self._dim_ind_pointers = []
        func_grp_pointers = {}
        dim_ind_positions = {}
        if 'DimensionIndexSequence' in self:
            self._dim_ind_pointers = [
                dim_ind.DimensionIndexPointer
                for dim_ind in self.DimensionIndexSequence
            ]
            for dim_ind in self.DimensionIndexSequence:
                ptr = dim_ind.DimensionIndexPointer
                if ptr in self._dim_ind_pointers:
                    grp_ptr = getattr(dim_ind, "FunctionalGroupPointer", None)
                    func_grp_pointers[ptr] = grp_ptr
            dim_ind_positions = {
                dim_ind.DimensionIndexPointer: i
                for i, dim_ind in enumerate(self.DimensionIndexSequence)
            }

        sfgs = None
        first_pffgs = None
        if 'SharedFunctionalGroupsSequence' in self:
            sfgs = self.SharedFunctionalGroupsSequence[0]
        if 'PerFrameFunctionalGroupsSequence' in self:
            first_pffgs = self.PerFrameFunctionalGroupsSequence[0]

        dim_indices: dict[int, list[int]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }
        dim_values: dict[int, list[Any]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }

        # Additional information that is not one of the indices
        extra_collection_pointers = []
        extra_collection_func_pointers = {}
        extra_collection_values: dict[int, list[Any]] = {}

        for grp_ptr, ptr in [
            # PlanePositionSequence/ImagePositionPatient
            (0x0020_9113, 0x0020_0032),
            # PlaneOrientationSequence/ImageOrientationPatient
            (0x0020_9116, 0x0020_0037),
            # PixelMeasuresSequence/PixelSpacing
            (0x0028_9110, 0x0028_0030),
            # PixelMeasuresSequence/SpacingBetweenSlices
            (0x0028_9110, 0x0018_0088),
            # FrameContentSequence/StackID
            (0x0020_9111, 0x0020_9056),
            # FrameContentSequence/InStackPositionNumber
            (0x0020_9111, 0x0020_9057),
        ]:
            if ptr in self._dim_ind_pointers:
                # Skip if this attribute is already indexed due to being a
                # dimension index pointer
                continue

            found = False
            dim_val = None

            # Check whether the attribute is in the shared functional groups
            if sfgs is not None and grp_ptr in sfgs:
                grp_seq = None

                if grp_ptr is not None:
                    if grp_ptr in sfgs:
                        grp_seq = sfgs[grp_ptr].value[0]
                else:
                    grp_seq = sfgs

                if grp_seq is not None and ptr in grp_seq:
                    found = True

                    # Get the shared value
                    dim_val = grp_seq[ptr].value

            # Check whether the attribute is in the first per-frame functional
            # group. If so, assume that it is there for all per-frame functional
            # groups
            if first_pffgs is not None and grp_ptr in first_pffgs:
                grp_seq = None

                if grp_ptr is not None:
                    grp_seq = first_pffgs[grp_ptr].value[0]
                else:
                    grp_seq = first_pffgs

                if grp_seq is not None and ptr in grp_seq:
                    found = True

            if found:
                extra_collection_pointers.append(ptr)
                extra_collection_func_pointers[ptr] = grp_ptr
                if dim_val is not None:
                    # Use the shared value for all frames
                    extra_collection_values[ptr] = (
                        [dim_val] * self.number_of_frames
                    )
                else:
                    # Values will be collected later in loop through per-frame
                    # functional groups
                    extra_collection_values[ptr] = []

        slice_spacing_hint = None
        shared_pixel_spacing: list[float] | None = None

        # Get the shared orientation
        shared_image_orientation: list[float] | None = None
        if hasattr(self, 'ImageOrientationSlide'):
            shared_image_orientation = self.ImageOrientationSlide

        self._single_source_frame_per_frame = True
        self._missing_reference_instances = []

        if self._is_tiled_full:
            # With TILED_FULL, there is no PerFrameFunctionalGroupsSequence,
            # so we have to deduce the per-frame information
            row_tag = tag_for_keyword('RowPositionInTotalImagePixelMatrix')
            col_tag = tag_for_keyword('ColumnPositionInTotalImagePixelMatrix')
            x_tag = tag_for_keyword('XOffsetInSlideCoordinateSystem')
            y_tag = tag_for_keyword('YOffsetInSlideCoordinateSystem')
            z_tag = tag_for_keyword('ZOffsetInSlideCoordinateSystem')
            tiled_full_dim_indices = {row_tag, col_tag}
            if len(tiled_full_dim_indices - set(dim_indices.keys())) > 0:
                raise RuntimeError(
                    'Expected images with '
                    '"DimensionOrganizationType" of "TILED_FULL" '
                    'to have the following dimension index pointers: '
                    'RowPositionInTotalImagePixelMatrix, '
                    'ColumnPositionInTotalImagePixelMatrix.'
                )
            self._single_source_frame_per_frame = False
            (
                channel_numbers,
                _,
                dim_values[col_tag],
                dim_values[row_tag],
                dim_values[x_tag],
                dim_values[y_tag],
                dim_values[z_tag],
            ) = zip(*iter_tiled_full_frame_data(self))

            if (
                hasattr(self, 'SegmentSequence') and
                self.SegmentationType != 'LABELMAP'
            ):
                segment_tag = tag_for_keyword('ReferencedSegmentNumber')
                dim_values[segment_tag] = channel_numbers
            elif hasattr(self, 'OpticalPathSequence'):
                op_tag = tag_for_keyword('OpticalPathIdentifier')
                dim_values[op_tag] = channel_numbers

            # Create indices for each of the dimensions
            for ptr, vals in dim_values.items():
                _, indices = np.unique(vals, return_inverse=True)
                dim_indices[ptr] = (indices + 1).tolist()

            # There is no way to deduce whether the spatial locations are
            # preserved in the tiled full case
            self._locations_preserved = None

            referenced_instances = None
            referenced_frames = None
        else:
            referenced_instances: list[str] | None = []
            referenced_frames: list[int] | None = []

            # Create a list of source images and check for spatial locations
            # preserved
            locations_preserved: list[
                SpatialLocationsPreservedValues | None
            ] = []

            # Some of the indexed pointers may be in the shared
            # functional groups
            if 'SharedFunctionalGroupsSequence' in self:
                sfgs = self.SharedFunctionalGroupsSequence[0]
                for ptr in extra_collection_pointers:
                    grp_ptr = extra_collection_func_pointers[ptr]

            for frame_item in self.get('PerFrameFunctionalGroupsSequence', []):
                # Get dimension indices for this frame
                if len(self._dim_ind_pointers) > 0:
                    content_seq = frame_item.FrameContentSequence[0]
                    indices = content_seq.DimensionIndexValues
                    if not isinstance(indices, (MultiValue, list)):
                        # In case there is a single dimension index
                        indices = [indices]
                else:
                    indices = []
                if len(indices) != len(self._dim_ind_pointers):
                    raise RuntimeError(
                        'Unexpected mismatch between dimension index values in '
                        'per-frames functional groups sequence and items in '
                        'the dimension index sequence.'
                    )
                for ptr in self._dim_ind_pointers:
                    dim_indices[ptr].append(indices[dim_ind_positions[ptr]])
                    grp_ptr = func_grp_pointers[ptr]
                    if grp_ptr is not None:
                        dim_val = frame_item[grp_ptr][0][ptr].value
                    else:
                        dim_val = frame_item[ptr].value
                    dim_values[ptr].append(dim_val)
                for ptr in extra_collection_pointers:
                    # Check this wasn't already found in the shared functional
                    # groups
                    if (
                        len(extra_collection_values[ptr]) ==
                        self.number_of_frames
                    ):
                        continue

                    grp_ptr = extra_collection_func_pointers[ptr]
                    if grp_ptr is not None:
                        dim_val = frame_item[grp_ptr][0][ptr].value
                    else:
                        dim_val = frame_item[ptr].value
                    extra_collection_values[ptr].append(dim_val)

                frame_source_instances = []
                frame_source_frames = []
                for der_im in getattr(
                    frame_item,
                    'DerivationImageSequence',
                    []
                ):
                    for src_im in getattr(
                        der_im,
                        'SourceImageSequence',
                        []
                    ):
                        frame_source_instances.append(
                            src_im.ReferencedSOPInstanceUID
                        )
                        if hasattr(src_im, 'SpatialLocationsPreserved'):
                            locations_preserved.append(
                                SpatialLocationsPreservedValues(
                                    src_im.SpatialLocationsPreserved
                                )
                            )
                        else:
                            locations_preserved.append(
                                None
                            )

                        if hasattr(src_im, 'ReferencedFrameNumber'):
                            if isinstance(
                                src_im.ReferencedFrameNumber,
                                MultiValue
                            ):
                                frame_source_frames.extend(
                                    [
                                        int(f)
                                        for f in src_im.ReferencedFrameNumber
                                    ]
                                )
                            else:
                                frame_source_frames.append(
                                    int(src_im.ReferencedFrameNumber)
                                )
                        else:
                            frame_source_frames.append(None)

                if (
                    len(set(frame_source_instances)) != 1 or
                    len(set(frame_source_frames)) != 1
                ):
                    self._single_source_frame_per_frame = False
                else:
                    ref_instance_uid = frame_source_instances[0]
                    if ref_instance_uid not in all_referenced_sops:
                        self._missing_reference_instances.append(
                            ref_instance_uid
                        )
                    referenced_instances.append(ref_instance_uid)
                    referenced_frames.append(frame_source_frames[0])

                # Check that this doesn't have a conflicting orientation
                if shared_image_orientation is not None:
                    if hasattr(frame_item, 'PlaneOrientationSequence'):
                        iop = (
                            frame_item
                            .PlaneOrientationSequence[0]
                            .ImageOrientationPatient
                        )
                        if iop != shared_image_orientation:
                            shared_image_orientation = None

                    if hasattr(frame_item, 'PixelMeasuresSequence'):
                        measures = frame_item.PixelMeasuresSequence[0]

                        fm_slice_spacing = measures.get(
                            'SpacingBetweenSlices'
                        )
                        if (
                            slice_spacing_hint is not None and
                            fm_slice_spacing != slice_spacing_hint
                        ):
                            slice_spacing_hint = None

                        fm_pixel_spacing = measures.get('PixelSpacing')
                        if (
                            shared_pixel_spacing is not None and
                            fm_pixel_spacing != shared_pixel_spacing
                        ):
                            shared_pixel_spacing = None

            # Summarise
            if any(
                isinstance(v, SpatialLocationsPreservedValues) and
                v == SpatialLocationsPreservedValues.NO
                for v in locations_preserved
            ):

                self._locations_preserved = SpatialLocationsPreservedValues.NO
            elif all(
                isinstance(v, SpatialLocationsPreservedValues) and
                v == SpatialLocationsPreservedValues.YES
                for v in locations_preserved
            ):
                self._locations_preserved = SpatialLocationsPreservedValues.YES
            else:
                self._locations_preserved = None

            if not self._single_source_frame_per_frame:
                referenced_instances = None
                referenced_frames = None

        self._db_con = sqlite3.connect(":memory:")

        self._create_ref_instance_table(referenced_uids)

        # Construct the columns and values to put into a frame look-up table
        # table within sqlite. There will be one row per frame in the
        # image
        col_defs = []  # SQL column definitions
        col_data = []  # lists of column data

        # Frame number column
        col_defs.append('FrameNumber INTEGER PRIMARY KEY')
        col_data.append(list(range(1, self.number_of_frames + 1)))

        self._dim_ind_col_names = {}
        for i, t in enumerate(dim_indices.keys()):
            vr, vm_str, _, _, kw = get_entry(t)
            if kw == '':
                kw = f'UnknownDimensionIndex{i}'
            ind_col_name = kw + '_DimensionIndexValues'

            # Add column for dimension index
            col_defs.append(f'{ind_col_name} INTEGER NOT NULL')
            col_data.append(dim_indices[t])

            # Add column for dimension value
            # For this to be possible, must have a fixed VM
            # and a VR that we can map to a sqlite type
            # Otherwise, we just omit the data from the db
            if kw == 'ReferencedSegmentNumber':
                # Special case since this tag technically has VM 1-n
                vm = 1
            else:
                try:
                    vm = int(vm_str)
                except ValueError:
                    self._dim_ind_col_names[t] = (ind_col_name, None)
                    continue
            try:
                sql_type = _DCM_SQL_TYPE_MAP[vr]
            except KeyError:
                self._dim_ind_col_names[t] = (ind_col_name, None)
                continue

            if vm > 1:
                val_col_names = []
                for d in range(vm):
                    data = [el[d] for el in dim_values[t]]
                    col_name = f'{kw}_{d}'
                    col_defs.append(f'{col_name} {sql_type} NOT NULL')
                    col_data.append(data)
                    val_col_names.append(col_name)

                self._dim_ind_col_names[t] = (
                    ind_col_name,
                    tuple(val_col_names)
                )
            else:
                # Single column
                col_defs.append(f'{kw} {sql_type} NOT NULL')
                col_data.append(dim_values[t])
                self._dim_ind_col_names[t] = (ind_col_name, kw)

        for i, t in enumerate(extra_collection_pointers):
            vr, vm_str, _, _, kw = get_entry(t)

            # Add column for dimension value
            # For this to be possible, must have a fixed VM
            # and a VR that we can map to a sqlite type
            # Otherwise, we just omit the data from the db
            vm = int(vm_str)
            sql_type = _DCM_SQL_TYPE_MAP[vr]

            if vm > 1:
                for d in range(vm):
                    data = [el[d] for el in extra_collection_values[t]]
                    col_name = f'{kw}_{d}'
                    col_defs.append(f'{col_name} {sql_type} NOT NULL')
                    col_data.append(data)
            else:
                # Single column
                col_defs.append(f'{kw} {sql_type} NOT NULL')
                col_data.append(extra_collection_values[t])

        # Columns related to source frames, if they are usable for indexing
        if (referenced_frames is None) != (referenced_instances is None):
            raise TypeError(
                "'referenced_frames' and 'referenced_instances' should be "
                "provided together or not at all."
            )
        if referenced_instances is not None:
            col_defs.append('ReferencedFrameNumber INTEGER')
            col_defs.append('ReferencedSOPInstanceUID VARCHAR NOT NULL')
            col_defs.append(
                'FOREIGN KEY(ReferencedSOPInstanceUID) '
                'REFERENCES InstanceUIDs(SOPInstanceUID)'
            )
            col_data += [
                referenced_frames,
                referenced_instances,
            ]

        self._create_frame_lut(col_defs, col_data)

    def _get_frame_lut_col_type(self, column_name: str) -> str:
        """Get the SQL type of a column in the FrameLUT table.

        Parameters
        ----------
        column_name: str
            Name of a colume in the FrameLUT whose type is requested.

        Returns
        -------
        str:
            String representation of the SQL type used for this column.

        """
        query = (
            "SELECT type FROM pragma_table_info('FrameLUT') "
            f"WHERE name = '{column_name}'"
        )
        result = list(self._db_con.execute(query))
        if len(result) == 0:
            raise ValueError(
                f'No such colume found in frame LUT: {column_name}'
            )
        return result[0][0]

    def _get_shared_frame_value(
        self,
        kw: str,
        vm: int = 1,
        filter: str | None = None,
        none_if_missing: bool = False
    ) -> Any:
        """Find the value of an attribute shared across frames.

        First checks whether the requested attribute is shared across all
        frames and raises an error if it is not.

        Parameters
        ----------
        kw: str
            Keyword of an attribute indexed in the LUT.
        vm: int
            Value multiplicity of the attribute.
        filter: str | None, optional
            SQL-syntax string of a filter to apply to frames.
        none_if_missing: bool
            Return ``None`` without raising an error if the attribute is not
            indexed.

        Returns
        -------
        Any:
            Value shared between all filtered frames.

        """
        if vm == 1:
            columns = [kw]
        else:
            columns = [f'{kw}_{n}' for n in range(vm)]

        for c in columns:
            # First check whether the column actually exists
            try:
                self._get_frame_lut_col_type(c)
            except ValueError:
                if none_if_missing:
                    return None
                raise RuntimeError(
                    f'Requested attribute is not in the Frame LUT: {c}'
                )

        if filter is None:
            filter = ''

        all_columns = ','.join(columns)
        cur = self._db_con.execute(
            f'SELECT DISTINCT {all_columns} FROM FrameLUT {filter}'
        )
        vals = list(cur)
        if none_if_missing and len(vals) == 0:
            return None

        if len(vals) != 1:
            raise RuntimeError(
                f'Frames do not have a consistent {kw}.'
            )
        if vm == 1:
            return vals[0][0]
        else:
            return vals[0]

    def _create_frame_lut(
        self,
        column_defs: list[str],
        column_data: list[list[Any]]
    ) -> None:
        """Create a SQL table containing frame information.

        Parameters
        ----------
        column_defs: list[str]
            String for each column containing SQL-syntax column definitions.
        column_data: list[list[Any]]
            Column data. Outer list contains columns, inner list contains
            values within that column.

        """
        # Build LUT from columns
        all_defs = ", ".join(column_defs)
        cmd = f'CREATE TABLE FrameLUT({all_defs})'
        placeholders = ', '.join(['?'] * len(column_data))
        with self._db_con:
            self._db_con.execute(cmd)
            self._db_con.executemany(
                f'INSERT INTO FrameLUT VALUES({placeholders})',
                zip(*column_data),
            )

    def _get_ref_instance_uids(self) -> list[tuple[str, str, str]]:
        """List all instances referenced in the image.

        Returns
        -------
        List[Tuple[str, str, str]]
            List of all instances referenced in the image in the format
            (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID).

        """
        instance_data = []

        def _include_sequence(seq):
            for ds in seq:
                if hasattr(ds, 'ReferencedSeriesSequence'):
                    for ref_series in ds.ReferencedSeriesSequence:

                        # Two different sequences are used here, depending on
                        # which particular top sequence level sequence we are
                        # in
                        if 'ReferencedSOPSequence' in ref_series:
                            instance_sequence = (
                                ref_series.ReferencedSOPSequence
                            )
                        else:
                            instance_sequence = (
                                ref_series.ReferencedInstanceSequence
                            )

                        for ref_ins in instance_sequence:
                            instance_data.append(
                                (
                                    ds.StudyInstanceUID,
                                    ref_series.SeriesInstanceUID,
                                    ref_ins.ReferencedSOPInstanceUID
                                )
                            )

        # Include the "main" referenced series sequence
        _include_sequence([self])
        for kw in [
            'StudiesContainingOtherReferencedInstancesSequence',
            'SourceImageEvidenceSequence'
        ]:
            if hasattr(self, kw):
                _include_sequence(getattr(self, kw))

        # There shouldn't be duplicates here, but there's no explicit rule
        # preventing it.
        # Since dictionary ordering is preserved, this trick deduplicates
        # the list without changing the order
        unique_instance_data = list(dict.fromkeys(instance_data))
        if len(unique_instance_data) != len(instance_data):
            counts = Counter(instance_data)
            duplicate_sop_uids = [
                f"'{key[2]}'" for key, value in counts.items() if value > 1
            ]
            display_str = ', '.join(duplicate_sop_uids)
            logger.warning(
                'Duplicate entries found in the ReferencedSeriesSequence. '
                f"SOP Instance UID: '{self.SOPInstanceUID}', "
                f'duplicated referenced SOP Instance UID items: {display_str}.'
            )

        return unique_instance_data

    def _check_indexing_with_source_frames(
        self,
        ignore_spatial_locations: bool = False
    ) -> None:
        """Check if indexing by source frames is possible.

        Raise exceptions with useful messages otherwise.

        Possible problems include:
            * Spatial locations are not preserved.
            * The dataset does not specify that spatial locations are preserved
              and the user has not asserted that they are.
            * At least one frame in the image lists multiple
              source frames.

        Parameters
        ----------
        ignore_spatial_locations: bool
            Allows the user to ignore whether spatial locations are preserved
            in the frames.

        """
        # Checks that it is possible to index using source frames in this
        # dataset
        if self._is_tiled_full:
            raise RuntimeError(
                'Indexing via source frames is not possible when an '
                'image is stored using the DimensionOrganizationType '
                '"TILED_FULL".'
            )
        elif self._locations_preserved is None:
            if not ignore_spatial_locations:
                raise RuntimeError(
                    'Indexing via source frames is not permissible since this '
                    'image does not specify that spatial locations are '
                    'preserved in the course of deriving the image '
                    'from the source image. If you are confident that spatial '
                    'locations are preserved, or do not require that spatial '
                    'locations are preserved, you may override this behavior '
                    "with the 'ignore_spatial_locations' parameter."
                )
        elif self._locations_preserved == SpatialLocationsPreservedValues.NO:
            if not ignore_spatial_locations:
                raise RuntimeError(
                    'Indexing via source frames is not permissible since this '
                    'image specifies that spatial locations are not preserved '
                    'in the course of deriving the image from the '
                    'source image. If you do not require that spatial '
                    ' locations are preserved you may override this behavior '
                    "with the 'ignore_spatial_locations' parameter."
                )
        if not self._single_source_frame_per_frame:
            raise RuntimeError(
                'Indexing via source frames is not permissible since some '
                'frames in the image specify multiple source frames.'
            )

    @property
    def dimension_index_pointers(self) -> list[BaseTag]:
        """List[pydicom.tag.BaseTag]:
            List of tags used as dimension indices.
        """
        return [BaseTag(t) for t in self._dim_ind_pointers]

    def _create_ref_instance_table(
        self,
        referenced_uids: list[tuple[str, str, str]],
    ) -> None:
        """Create a table of referenced instances.

        The resulting table (called InstanceUIDs) contains Study, Series and
        SOP instance UIDs for each instance referenced by the image.

        Parameters
        ----------
        referenced_uids: List[Tuple[str, str, str]]
            List of UIDs for each instance referenced in the image.
            Each tuple should be in the format
            (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID).

        """
        with self._db_con:
            self._db_con.execute(
                "CREATE TABLE InstanceUIDs("
                "StudyInstanceUID VARCHAR NOT NULL, "
                "SeriesInstanceUID VARCHAR NOT NULL, "
                "SOPInstanceUID VARCHAR PRIMARY KEY"
                ")"
            )
            self._db_con.executemany(
                "INSERT INTO InstanceUIDs "
                "(StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID) "
                "VALUES(?, ?, ?)",
                referenced_uids,
            )

    def _do_columns_identify_unique_frames(
        self,
        column_names: Sequence[str],
        filter: str | None = None,
    ) -> bool:
        """Check if a list of columns uniquely identifies frames.

        For a given list of columns, check whether every combination of values
        for these column identifies a unique image frame. This is a
        pre-requisite for indexing frames using this list of columns.

        Parameters
        ----------
        column_names: Sequence[str]
            Column names.
        filter: str
            A SQL-syntax filter to apply. If provided, determines whether the
            given columns are sufficient to identify frames within the filtered
            subset of frames only.

        Returns
        -------
        bool
            True if combination of columns is sufficient to identify unique
            frames.

        """
        col_str = ", ".join(column_names)
        cur = self._db_con.cursor()

        if filter is not None and filter != '':
            total = cur.execute(
                f"SELECT COUNT(*) FROM FrameLUT {filter}"
            ).fetchone()[0]
        else:
            total = self.number_of_frames
            filter = ''

        n_unique_combos = cur.execute(
            "SELECT COUNT(*) FROM "
            f"(SELECT 1 FROM FrameLUT {filter} GROUP BY {col_str})"
        ).fetchone()[0]
        return n_unique_combos == total

    def are_dimension_indices_unique(
        self,
        dimension_index_pointers: Sequence[int | BaseTag | str],
    ) -> bool:
        """Check if a list of index pointers uniquely identifies frames.

        For a given list of dimension index pointers, check whether every
        combination of index values for these pointers identifies a unique
        image frame. This is a pre-requisite for indexing using this list of
        dimension index pointers.

        Parameters
        ----------
        Sequence[Union[int, pydicom.tag.BaseTag, str]]:
            Sequence of tags serving as dimension index pointers. If strings,
            the items are interpreted as keywords.

        Returns
        -------
        bool
            True if dimension indices are unique.

        """
        column_names = []
        for ptr in dimension_index_pointers:
            if isinstance(ptr, str):
                t = tag_for_keyword(ptr)
                if t is None:
                    raise ValueError(
                        f"Keyword '{ptr}' is not a valid DICOM keyword."
                    )
                ptr = t
            column_names.append(self._dim_ind_col_names[ptr][0])
        return self._do_columns_identify_unique_frames(column_names)

    def get_source_image_uids(self) -> list[tuple[UID, UID, UID]]:
        """Get UIDs of source image instances referenced in the image.

        Returns
        -------
        List[Tuple[highdicom.UID, highdicom.UID, highdicom.UID]]
            (Study Instance UID, Series Instance UID, SOP Instance UID) triplet
            for every image instance referenced in the image.

        """
        for ref_instance_uid in self._missing_reference_instances:
            logger.warning(
                f'SOP instances {ref_instance_uid} referenced in the source '
                'image sequence is not included in the Referenced Series '
                'Sequence, Source Image Evidence Sequence, or Studies '
                'Containing Other Referenced Instances Sequence. This is an '
                'error with the integrity of the object. This instance will '
                'be omitted from the returned list. '
            )
        cur = self._db_con.cursor()
        res = cur.execute(
            'SELECT StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID '
            'FROM InstanceUIDs'
        )

        return [
            (UID(a), UID(b), UID(c)) for a, b, c in res.fetchall()
        ]

    def _get_unique_referenced_sop_instance_uids(self) -> set[str]:
        """Get set of unique Referenced SOP Instance UIDs.

        Returns
        -------
        Set[str]
            Set of unique Referenced SOP Instance UIDs.

        """
        cur = self._db_con.cursor()
        return {
            r[0] for r in
            cur.execute(
                'SELECT DISTINCT(SOPInstanceUID) from InstanceUIDs'
            )
        }

    def _get_max_referenced_frame_number(self) -> int:
        """Get highest frame number of any referenced frame.

        Absent access to the referenced dataset itself, being less than this
        value is a sufficient condition for the existence of a frame number
        in the source image.

        Returns
        -------
        int
            Highest frame number referenced in the image.

        """
        cur = self._db_con.cursor()
        return cur.execute(
            'SELECT MAX(ReferencedFrameNumber) FROM FrameLUT'
        ).fetchone()[0]

    def is_indexable_as_total_pixel_matrix(self) -> bool:
        """Whether the image can be indexed as a total pixel matrix.

        Returns
        -------
        bool:
            True if the image may be indexed using row and column
            positions in the total pixel matrix. False otherwise.

        """
        row_pos_tag = tag_for_keyword('RowPositionInTotalImagePixelMatrix')
        col_pos_tag = tag_for_keyword('ColumnPositionInTotalImagePixelMatrix')
        return (
            row_pos_tag in self._dim_ind_col_names and
            col_pos_tag in self._dim_ind_col_names
        )

    def _get_unique_dim_index_values(
        self,
        dimension_index_pointers: Sequence[int],
    ) -> set[tuple[int, ...]]:
        """Get set of unique dimension index value combinations.

        Parameters
        ----------
        dimension_index_pointers: Sequence[int]
            List of dimension index pointers for which to find unique
            combinations of values.

        Returns
        -------
        Set[Tuple[int, ...]]:
            Set of unique dimension index value combinations for the given
            input dimension index pointers.

        """
        cols = [self._dim_ind_col_names[p][0] for p in dimension_index_pointers]
        cols_str = ', '.join(cols)
        cur = self._db_con.cursor()
        return {
            r for r in
            cur.execute(
                f'SELECT DISTINCT {cols_str} FROM FrameLUT'
            )
        }

    def _get_stacked_volume_geometry(
        self,
        rtol: float | None = None,
        atol: float | None = None,
        allow_missing_positions: bool = False,
        allow_duplicate_positions: bool = True,
        filter: str | None = None,
        slice_start: int | None = None,
        slice_end: int | None = None,
        as_indices: bool = False,
    ) -> tuple[VolumeGeometry, list[tuple[int, int]]]:
        """Get geometry of a volume created by stacking frames.

        For this to succeed the image (or the filtered subset of its frames)
        must consist of a set of frames that are parallel and regularly spaced
        along their normal vectors within the frame-of-reference coordinate
        system.

        Parameters
        ----------
        rtol: float | None, optional
            Relative tolerance for determining spacing regularity. If slice
            spacings vary by less that this proportion of the average spacing,
            they are considered to be regular. If neither ``rtol`` or ``atol``
            are provided, a default relative tolerance of 0.01 is used.
        atol: float | None, optional
            Absolute tolerance for determining spacing regularity. If slice
            spacings vary by less that this value (in mm), they are considered
            to be regular. Incompatible with ``rtol``.
        allow_missing_positions: bool, optional
            Allow volume positions for which no frame exists in the image.
        allow_duplicate_positions: bool, optional
            Allow multiple slices to occupy the same position within the
            volume. If False, duplicated image positions will result in
            failure.
        filter: str | None, optional
            SQL-syntax string of a filter to apply before inferring the
            geometry.
        slice_start: int | none, optional
            zero-based index of the "volume position" of the first slice of the
            returned volume. the "volume position" refers to the position of
            slices after sorting spatially, and may correspond to any frame in
            the segmentation file, depending on its construction. may be
            negative, in which case standard python indexing behavior is
            followed (-1 corresponds to the last volume position, etc).
        slice_end: union[int, none], optional
            zero-based index of the "volume position" one beyond the last slice
            of the returned volume. the "volume position" refers to the
            position of slices after sorting spatially, and may correspond to
            any frame in the segmentation file, depending on its construction.
            may be negative, in which case standard python indexing behavior is
            followed (-1 corresponds to the last volume position, etc). if
            none, the last volume position is included as the last output
            slice.
        as_indices: bool, optional
            If true, interpret slice numbering parameters (``slice_start`` and
            ``slice_end``) as zero-based indices as opposed to the default
            one-based numbers used within dicom.

        Returns
        -------
        highdicom.VolumeGeometry:
            Resulting volume geometry created by stacking (filtered) frames.
        list[tuple[int, int]]:
            List of (1-based) frame numbers and the corresponding zero-based
            volume positions.

        """
        shared_image_orientation = self._get_shared_frame_value(
            'ImageOrientationPatient',
            vm=6,
            filter=filter,
        )
        shared_pixel_spacing = self._get_shared_frame_value(
            'PixelSpacing',
            vm=2,
            filter=filter,
        )
        slice_spacing_hint = self._get_shared_frame_value(
            'SpacingBetweenSlices',
            none_if_missing=True,
            filter=filter,
        )

        if filter is None:
            filter = ''

        query = f"""
            SELECT
                FrameNumber,
                ImagePositionPatient_0,
                ImagePositionPatient_1,
                ImagePositionPatient_2
            FROM FrameLUT {filter}
        """
        results = list(self._db_con.execute(query))

        image_positions = [r[1:] for r in results]
        frame_numbers = [r[0] for r in results]

        volume_spacing, volume_positions = get_volume_positions(
            image_positions=image_positions,
            image_orientation=shared_image_orientation,
            allow_missing_positions=allow_missing_positions,
            allow_duplicate_positions=allow_duplicate_positions,
            spacing_hint=slice_spacing_hint,
            rtol=rtol,
            atol=atol,
        )
        if volume_positions is None:
            raise RuntimeError(
                'Frame positions do not form a regularly-spaced '
                'volume.'
            )
        initial_number_of_slices = max(volume_positions) + 1

        slice_start, slice_end = self._standardize_slice_indices(
            slice_start=slice_start,
            slice_end=slice_end,
            as_indices=as_indices,
            n_vol_positions=initial_number_of_slices,
        )
        origin_slice_index = volume_positions.index(0)

        geometry = VolumeGeometry.from_attributes(
            image_position=image_positions[origin_slice_index],
            image_orientation=shared_image_orientation,
            rows=self.Rows,
            columns=self.Columns,
            pixel_spacing=shared_pixel_spacing,
            number_of_frames=initial_number_of_slices,
            spacing_between_slices=volume_spacing,
            coordinate_system=self._coordinate_system,
        )
        geometry = geometry[slice_start:slice_end]

        frame_positions = []

        # Filter to give volume positions within the requested range
        for f, vol_pos in zip(frame_numbers, volume_positions):
            if vol_pos >= slice_start and vol_pos < slice_end:
                frame_positions.append(
                    (
                        f,
                        vol_pos - slice_start,
                    )
                )

        return geometry, frame_positions

    def _prepare_volume_positions_table(
        self,
        rtol: float | None = None,
        atol: float | None = None,
        allow_missing_positions: bool = False,
        filter: str | None = None,
        slice_start: int | None = None,
        slice_end: int | None = None,
        as_indices: bool = False,
    ) -> tuple[_SQLTableDefinition, VolumeGeometry]:
        """Get geometry of a volume created by stacking frames and prepare a
        SQL table mapping frame number to the resulting position in the volume
        stack.

        Parameters
        ----------
        rtol: float | None, optional
            Relative tolerance for determining spacing regularity. If slice
            spacings vary by less that this proportion of the average spacing,
            they are considered to be regular. If neither ``rtol`` or ``atol``
            are provided, a default relative tolerance of 0.01 is used.
        atol: float | None, optional
            Absolute tolerance for determining spacing regularity. If slice
            spacings vary by less that this value (in mm), they are considered
            to be regular. Incompatible with ``rtol``.
        allow_missing_positions: bool, optional
            Allow volume positions for which no frame exists in the image.
        filter: str | None, optional
            SQL-syntax string of a filter to apply before inferring the
            geometry.
        slice_start: int | none, optional
            zero-based index of the "volume position" of the first slice of the
            returned volume. the "volume position" refers to the position of
            slices after sorting spatially, and may correspond to any frame in
            the segmentation file, depending on its construction. may be
            negative, in which case standard python indexing behavior is
            followed (-1 corresponds to the last volume position, etc).
        slice_end: union[int, none], optional
            zero-based index of the "volume position" one beyond the last slice
            of the returned volume. the "volume position" refers to the
            position of slices after sorting spatially, and may correspond to
            any frame in the segmentation file, depending on its construction.
            may be negative, in which case standard python indexing behavior is
            followed (-1 corresponds to the last volume position, etc). if
            none, the last volume position is included as the last output
            slice.
        as_indices: bool, optional
            If true, interpret slice numbering parameters (``slice_start`` and
            ``slice_end``) as zero-based indices as opposed to the default
            one-based numbers used within dicom.

        Returns
        -------
        _SQLTableDefinition:
            Table definition that may be used to join with the main FrameLUT to
            provide the OutputFrameIndex corresponding to each FrameNumber such
            that the frames are stacked by geometry.
        highdicom.VolumeGeometry:
            Resulting volume geometry created by stacking (filtered) frames.

        """
        geometry, frame_positions = self._get_stacked_volume_geometry(
            rtol=rtol,
            atol=atol,
            allow_missing_positions=allow_missing_positions,
            filter=filter,
            slice_start=slice_start,
            slice_end=slice_end,
            as_indices=as_indices,
        )

        col_defs = [
            'FrameNumber INTEGER PRIMARY KEY',
            'OutputFrameIndex INTEGER NOT NULL',
        ]

        table_def = _SQLTableDefinition(
            table_name='TemporaryVolumePositionTable',
            column_defs=col_defs,
            column_data=frame_positions,
        )

        return table_def, geometry

    def get_volume_geometry(
        self,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        allow_missing_positions: bool = False,
        allow_duplicate_positions: bool = True,
    ) -> VolumeGeometry | None:
        """Get geometry of the image in 3D space.

        This will succeed in two situations. Either the image is a consists of
        a set of frames that are stacked together to give a regularly-spaced 3D
        volume array (typical of CT, MRI, and PET) or the image is a tiled
        image consisting of a set of 2D tiles that are placed together in the
        same plane to form a total pixel matrix.

        A single frame image has a volume geometry if it provides any
        information about its position and orientation within a
        frame-of-reference coordinate system.

        Parameters
        ----------
        rtol: float | None, optional
            Relative tolerance for determining spacing regularity. If slice
            spacings vary by less that this proportion of the average spacing,
            they are considered to be regular. If neither ``rtol`` or ``atol``
            are provided, a default relative tolerance of 0.01 is used.
        atol: float | None, optional
            Absolute tolerance for determining spacing regularity. If slice
            spacings vary by less that this value (in mm), they are considered
            to be regular. Incompatible with ``rtol``.
        allow_missing_positions: bool, optional
            Allow volume positions for which no frame exists in the image.
        allow_duplicate_positions: bool, optional
            Allow multiple slices to occupy the same position within the
            volume. If False, duplicated image positions will result in
            failure.

        Returns
        -------
        highdicom.VolumeGeometry | None:
            Geometry of the volume if the image represents a regularly-spaced
            3D volume or tiled total pixel matrix. ``None`` otherwise.

        """
        try:
            return self._get_volume_geometry(
                atol=atol,
                rtol=rtol,
                allow_missing_positions=allow_missing_positions,
                allow_duplicate_positions=allow_duplicate_positions,
            )
        except RuntimeError:
            return None

    def _get_volume_geometry(
        self,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        allow_missing_positions: bool = False,
        allow_duplicate_positions: bool = True,
    ) -> VolumeGeometry:
        """Get geometry of the image in 3D space.

        Parameters
        ----------
        rtol: float | None, optional
            Relative tolerance for determining spacing regularity. If slice
            spacings vary by less that this proportion of the average spacing,
            they are considered to be regular. If neither ``rtol`` or ``atol``
            are provided, a default relative tolerance of 0.01 is used.
        atol: float | None, optional
            Absolute tolerance for determining spacing regularity. If slice
            spacings vary by less that this value (in mm), they are considered
            to be regular. Incompatible with ``rtol``.
        allow_missing_positions: bool, optional
            Allow volume positions for which no frame exists in the image.
        allow_duplicate_positions: bool, optional
            Allow multiple slices to occupy the same position within the
            volume. If False, duplicated image positions will result in
            failure.

        Returns
        -------
        highdicom.VolumeGeometry:
            Geometry of the volume.

        """
        if self._coordinate_system is None:
            raise RuntimeError(
                "Image does not exist within a frame-of-reference "
                "coordinate system."
            )

        if is_multiframe_image(self):
            if (
                self.is_tiled and
                self._coordinate_system == CoordinateSystemNames.SLIDE
            ):
                pixel_spacing = self._get_shared_frame_value(
                    'PixelSpacing',
                    vm=2
                )
                slice_spacing = self._get_shared_frame_value(
                    'SpacingBetweenSlices',
                    none_if_missing=True,
                )
                if slice_spacing is None:
                    slice_spacing = 1.0

                origin_seq = self.TotalPixelMatrixOriginSequence[0]

                origin_position = [
                    origin_seq.XOffsetInSlideCoordinateSystem,
                    origin_seq.YOffsetInSlideCoordinateSystem,
                    origin_seq.get('ZOffsetInSlideCoordinateSystem', 0.0),
                ]
                shared_image_orientation = self.ImageOrientationSlide

                return VolumeGeometry.from_attributes(
                    image_position=origin_position,
                    image_orientation=shared_image_orientation,
                    rows=self.TotalPixelMatrixRows,
                    columns=self.TotalPixelMatrixColumns,
                    pixel_spacing=pixel_spacing,
                    number_of_frames=1,
                    spacing_between_slices=slice_spacing,
                    coordinate_system=self._coordinate_system,
                )

            if (self._coordinate_system == CoordinateSystemNames.PATIENT):
                geometry, _ = self._get_stacked_volume_geometry(
                    rtol=rtol,
                    atol=atol,
                    allow_missing_positions=allow_missing_positions,
                    allow_duplicate_positions=allow_duplicate_positions,
                )
                return geometry
        else:
            # Single frame image, only supports patient coordinate system
            # currently
            if (
                self._coordinate_system is not None and
                self._coordinate_system == CoordinateSystemNames.PATIENT and
                'ImagePositionPatient' in self
            ):
                position = self.ImagePositionPatient
                orientation = self.ImageOrientationPatient

                return VolumeGeometry.from_attributes(
                    image_position=position,
                    image_orientation=orientation,
                    rows=self.Rows,
                    columns=self.Columns,
                    pixel_spacing=self.PixelSpacing,
                    number_of_frames=1,
                    spacing_between_slices=self.get(
                        'SpacingBetweenSlices',
                        1.0
                    ),
                    coordinate_system=self._coordinate_system,
                )

        raise RuntimeError(
            'Image does not represent a regularly-spaced volume.'
        )

    @contextmanager
    def _generate_temp_tables(
        self,
        table_defs: Sequence[_SQLTableDefinition],
    ) -> Generator[None, None, None]:
        """Context manager that handles multiple temporary table.

        The temporary tables are created with the specified information. Control
        flow then returns to code within the "with" block. After the "with"
        block has completed, the cleanup of the tables is automatically handled.

        Parameters
        ----------
        table_defs: Sequence[_SQLTableDefinition]
            Specifications of each table to create.

        Yields
        ------
        None:
            Yields control to the "with" block, with the temporary tables
            created.

        """
        for tdef in table_defs:
            # First check whether the table already exists and remove it if it
            # does. This shouldn't happen usually as the context manager should
            # ensure that the temporary tables are always cleared up. However
            # it does seem to happen when interactive REPLs such as ipython
            # handle errors within the context manager
            query = (
                "SELECT COUNT(*) FROM sqlite_master "
                f"WHERE type = 'table' AND name = '{tdef.table_name}'"
            )
            result = next(self._db_con.execute(query))[0]
            if result > 0:
                with self._db_con:
                    self._db_con.execute(f"DROP TABLE {tdef.table_name}")

            defs_str = ', '.join(tdef.column_defs)
            create_cmd = (f'CREATE TABLE {tdef.table_name}({defs_str})')
            placeholders = ', '.join(['?'] * len(tdef.column_defs))

            with self._db_con:
                self._db_con.execute(create_cmd)
                self._db_con.executemany(
                    f'INSERT INTO {tdef.table_name} VALUES({placeholders})',
                    tdef.column_data
                )

        # Return control flow to "with" block
        yield

        for tdef in table_defs:
            # Clean up the tables
            cmd = (f'DROP TABLE {tdef.table_name}')
            with self._db_con:
                self._db_con.execute(cmd)

    def _get_pixels_by_frame(
        self,
        spatial_shape: int | tuple[int, int],
        indices_iterator: Iterator[
            tuple[
                int,
                tuple[slice | int, ...],
                tuple[slice | int, ...],
                tuple[int, ...],
            ]
        ],
        *,
        dtype: type | str | np.dtype = np.float64,
        channel_shape: tuple[int, ...] = (),
        apply_real_world_transform: bool | None = None,
        real_world_value_map_selector: int | str | Code | CodedConcept = 0,
        apply_modality_transform: bool | None = None,
        apply_voi_transform: bool | None = False,
        voi_transform_selector: int | str | VOILUTTransformation = 0,
        voi_output_range: tuple[float, float] = (0.0, 1.0),
        apply_presentation_lut: bool = True,
        apply_palette_color_lut: bool | None = None,
        remove_palette_color_values: Sequence[int] | None = None,
        palette_color_background_index: int = 0,
        apply_icc_profile: bool | None = None,
    ) -> np.ndarray:
        """Construct a pixel array given a sequence of frame numbers.

        The output array has 3 dimensions (frame, rows, columns), followed by 1
        for RGB color channels, if applicable, followed by one for each
        additional channel specified.

        Parameters
        ----------
        spatial_shape: Union[int, Tuple[int, int]]
            Spatial shape of the output array. If an integer, this is the
            number of frames in the output array and the number of rows and
            columns are taken to match those of each frame. If a tuple of
            integers, it contains the number of (rows, columns) in the output
            array and there is no frame dimension (this is the tiled case).
            Note in either case, the channel dimensions (if relevant) are
            omitted.
        indices_iterator: Iterator[Tuple[int, Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], Tuple[int, ...]]]
            An iterable object that yields tuples of (frame_index,
            input_indexer, spatial_indexer, channel_indexer) that describes how
            to construct the desired output pixel array from the multiframe
            image's pixel array. 'frame_index' specifies the zero-based index
            of the input frame and 'input_indexer' is a tuple that may be used
            directly to index a region of that frame. 'spatial_indexer' is a
            tuple that may be used directly to index the output array to place
            a single frame's pixels into the output array (excluding the
            channel dimensions). The 'channel_indexer' indexes a channel of the
            output array into which the result should be placed. Note that in
            both cases the indexers access the frame, row and column dimensions
            of the relevant array, but not the channel dimension (if relevant).
        channel_shape: tuple[int, ...], optional
            Channel shape of the output array. The use of channels depends
            on image type, for example it may be segments in a segmentation,
            optical paths in a microscopy image, or B-values in an MRI.
        dtype: Union[type, str, numpy.dtype], optional
            Data type of the returned array.
        apply_real_world_transform: bool | None, optional
            Whether to apply a real-world value map to the frame.
            A real-world value maps converts stored pixel values to output
            values with a real-world meaning, either using a LUT or a linear
            slope and intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if present but no error will be raised if
            it is not present.

            Note that if the dataset contains both a modality LUT and a real
            world value map, the real world value map will be applied
            preferentially. This also implies that specifying both
            ``apply_real_world_transform`` and ``apply_modality_transform`` to
            True is not permitted.
        real_world_value_map_selector: int | str | pydicom.sr.coding.Code | highdicom.sr.coding.CodedConcept, optional
            Specification of the real world value map to use (multiple may be
            present in the dataset). If an int, it is used to index the list of
            available maps. A negative integer may be used to index from the
            end of the list following standard Python indexing convention. If a
            str, the string will be used to match the ``"LUTLabel"`` attribute
            to select the map. If a ``pydicom.sr.coding.Code`` or
            ``highdicom.sr.coding.CodedConcept``, this will be used to match
            the units (contained in the ``"MeasurementUnitsCodeSequence"``
            attribute).
        apply_modality_transform: bool | None, optional
            Whether to apply the modality transform (if present in the
            dataset) to the frame. The modality transform maps stored pixel
            values to output values, either using a LUT or rescale slope and
            intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        apply_voi_transform: bool | None, optional
            Apply the value-of-interest (VOI) transform (if present in the
            dataset), which limits the range of pixel values to a particular
            range of interest using either a windowing operation or a LUT.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        voi_transform_selector: int | str | highdicom.VOILUTTransformation, optional
            Specification of the VOI transform to select (multiple may be
            present). May either be an int or a str. If an int, it is
            interpreted as a (zero-based) index of the list of VOI transforms
            to apply. A negative integer may be used to index from the end of
            the list following standard Python indexing convention. If a str,
            the string that will be used to match the
            ``"WindowCenterWidthExplanation"`` or the ``"LUTExplanation"``
            attributes to choose from multiple VOI transforms. Note that such
            explanations are optional according to the standard and therefore
            may not be present. Ignored if ``apply_voi_transform`` is ``False``
            or no VOI transform is included in the datasets.

            Alternatively, a user-defined
            :class:`highdicom.VOILUTTransformation` may be supplied.
            This will override any such transform specified in the dataset.
        voi_output_range: Tuple[float, float], optional
            Range of output values to which the VOI range is mapped. Only
            relevant if ``apply_voi_transform`` is True and a VOI transform is
            present.
        apply_palette_color_lut: bool | None, optional
            Apply the palette color LUT, if present in the dataset. The palette
            color LUT maps a single sample for each pixel stored in the dataset
            to a 3 sample-per-pixel color image.
        apply_presentation_lut: bool, optional
            Apply the presentation LUT transform to invert the pixel values. If
            the PresentationLUTShape is present with the value ``'INVERSE'``,
            or the PresentationLUTShape is not present but the Photometric
            Interpretation is MONOCHROME1, convert the range of the output
            pixels corresponds to MONOCHROME2 (in which high values are
            represent white and low values represent black). Ignored if
            PhotometricInterpretation is not MONOCHROME1 and the
            PresentationLUTShape is not present, or if a real world value
            transform is applied.
        remove_palette_color_values: Sequence[int] | None, optional
            Remove values from the palette color LUT (if any) by altering the
            LUT so that these values map to the RGB value at position
            ``palette_color_background_index`` instead of their original value.
            This is intended to remove segments from a palette color labelmap
            segmentation.
        palette_color_background_index: int, optional
            The index (i.e. input) of the palette color LUT that corresponds to
            background. Relevant only if ``remove_palette_color_values`` is
            provided.
        apply_icc_profile: bool | None, optional
            Whether colors should be corrected by applying an ICC
            transform. Will only be performed if metadata contain an
            ICC Profile.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present, but no error will be
            raised if it is not present.

        Returns
        -------
        pixel_array: numpy.ndarray
            Pixel array

        """  # noqa: E501
        shared_frame_transform = _CombinedPixelTransform(
            self,
            apply_real_world_transform=apply_real_world_transform,
            real_world_value_map_selector=real_world_value_map_selector,
            apply_modality_transform=apply_modality_transform,
            apply_voi_transform=apply_voi_transform,
            voi_transform_selector=voi_transform_selector,
            voi_output_range=voi_output_range,
            apply_presentation_lut=apply_presentation_lut,
            apply_palette_color_lut=apply_palette_color_lut,
            remove_palette_color_values=remove_palette_color_values,
            palette_color_background_index=palette_color_background_index,
            apply_icc_profile=apply_icc_profile,
            output_dtype=dtype,
        )

        # Initialize empty pixel array
        if isinstance(spatial_shape, tuple):
            initial_shape = spatial_shape
        else:
            initial_shape = (spatial_shape, self.Rows, self.Columns)
        color_frames = shared_frame_transform.color_output
        samples_shape = (3, ) if color_frames else ()

        full_output_shape = (
            *initial_shape,
            *samples_shape,
            *channel_shape,
        )

        out_array = np.zeros(
            full_output_shape,
            dtype=dtype
        )

        context_manager = (
            self._file_reader
            if self._file_reader is not None
            else nullcontext()
        )

        with context_manager:

            # loop through output frames
            for (
                frame_index,
                input_indexer,
                spatial_indexer,
                channel_indexer
            ) in indices_iterator:

                if shared_frame_transform.applies_to_all_frames:
                    frame_transform = shared_frame_transform
                else:
                    frame_transform = _CombinedPixelTransform(
                        self,
                        frame_index=frame_index,
                        apply_real_world_transform=apply_real_world_transform,
                        real_world_value_map_selector=real_world_value_map_selector,  # noqa: E501
                        apply_modality_transform=apply_modality_transform,
                        apply_voi_transform=apply_voi_transform,
                        voi_transform_selector=voi_transform_selector,
                        voi_output_range=voi_output_range,
                        apply_presentation_lut=apply_presentation_lut,
                        apply_palette_color_lut=apply_palette_color_lut,
                        apply_icc_profile=apply_icc_profile,
                        output_dtype=dtype,
                    )

                if color_frames:
                    # Include the sample dimension for color images
                    output_indexer = (
                        *spatial_indexer,
                        slice(None),
                        *channel_indexer
                    )
                else:
                    output_indexer = (*spatial_indexer, *channel_indexer)

                if self._pixel_array is None:
                    if self._file_reader is not None:
                        frame_bytes = self._file_reader.read_frame_raw(
                            frame_index
                        )
                    else:
                        frame_bytes = self.get_raw_frame(frame_index + 1)
                    frame = frame_transform(frame_bytes, frame_index)
                else:
                    if self.pixel_array.ndim == 2:
                        if frame_index == 0:
                            frame = self.pixel_array
                        else:
                            raise IndexError(
                                f'Index {frame_index} is out of bounds for '
                                'an image with a single frame.'
                            )
                    else:
                        frame = self.pixel_array[frame_index]
                    frame = frame_transform(frame, frame_index)

                out_array[output_indexer] = frame[input_indexer]

        return out_array

    def _normalize_dimension_queries(
        self,
        queries: dict[int | str, Any],
        use_indices: bool,
        multiple_values: bool,
        allow_missing_values: bool = False,
    ) -> dict[str, Any]:
        """Check and standardize queries used to specify dimensions.

        Parameters
        ----------
        queries: Dict[Union[int, str], Any]
            Dictionary defining a filter or index along a dimension. The keys
            define the dimensions used. They may be either the tags or keywords
            of attributes in the image's dimension index, or the special
            values, 'ReferencedSOPInstanceUID', and 'ReferencedFrameNumber'.
            The values of the dictionary give sequences of values of
            corresponding dimension that define each slice of the output array
            or a single value that is used to filter the frames. Note that
            multiple dimensions may be used, in which case a frame must match
            the values of all provided dimensions to be placed in the output
            array.
        use_indices: bool
            Whether indexing is done using (integer) dimension index values, as
            opposed to values themselves.
        multiple_values: bool
            Whether multiple values are expected for each dimension. If True,
            defines an index, if False, defines a filter.
        allow_missing_values: bool, optional
            Do not raise an error if some query values are not found in the
            dataset.

        Returns
        -------
        dict[str, Any]:
            Queries after validation, with keys matching the column names used
            in the Frame LUT.

        """
        normalized_queries: dict[str, Any] = {}
        tag: BaseTag | None = None

        if len(queries) == 0:
            raise ValueError("Query definitions must not be empty.")

        if multiple_values:
            n_values = len(list(queries.values())[0])

        for p, value in queries.items():
            if isinstance(p, int):  # also covers BaseTag
                tag = BaseTag(p)

            elif isinstance(p, str):
                # Special cases
                if p == 'ReferencedSOPInstanceUID':
                    col_name = 'ReferencedSOPInstanceUID'
                    python_type = str
                elif p == 'ReferencedFrameNumber':
                    col_name = 'ReferencedFrameNumber'
                    python_type = int
                else:
                    t = tag_for_keyword(p)

                    if t is None:
                        raise ValueError(
                            f'No attribute found with name {p}.'
                        )

                    tag = BaseTag(t)

            else:
                raise TypeError(
                    "Every item in 'stack_dimension_pointers' must be an "
                    'int, str, or pydicom.tag.BaseTag.'
                )

            if tag is None:
                if use_indices:
                    raise ValueError(
                        f'Cannot query by index value for column {p}.'
                    )
            else:
                vr, _, _, _, kw = get_entry(tag)
                if kw == '':
                    kw = '<unknown attribute>'

                try:
                    ind_col_name, val_col_name = self._dim_ind_col_names[tag]
                except KeyError as e:
                    msg = (
                        f'The tag {BaseTag(tag)} ({kw}) is not used as '
                        'a dimension index for this image.'
                    )
                    raise KeyError(msg) from e

                if use_indices:
                    col_name = ind_col_name
                    python_type = int
                else:
                    col_name = val_col_name
                    python_type = _DCM_PYTHON_TYPE_MAP[vr]
                    if col_name is None:
                        raise RuntimeError(
                            f'Cannot query attribute with tag {BaseTag(p)} '
                            'by value. Try querying by index value instead. '
                            'If you think this should be possible, please '
                            'report an issue to the highdicom maintainers.'
                        )
                    elif isinstance(col_name, tuple):
                        raise ValueError(
                            f'Cannot query attribute with tag {BaseTag(p)} '
                            'by value because it is a multi-valued attribute. '
                            'Try querying by index value instead. '
                        )

            allowed_values = {
                v[0] for v in self._db_con.execute(
                    f"SELECT DISTINCT({col_name}) FROM FrameLUT;"
                )
            }

            if multiple_values:
                if isinstance(value, np.ndarray):
                    if value.ndim != 1:
                        raise ValueError('Numpy array of invalid shape.')
                    value = value.tolist()
                else:
                    value = [
                        v.item() if isinstance(v, np.generic) else v
                        for v in value
                    ]

                if len(value) != n_values:
                    raise ValueError(
                        'Number of values along all dimensions must match.'
                    )
                for v in value:
                    if isinstance(v, np.generic):
                        # Change numpy to python type
                        v = v.item()
                    if not isinstance(v, python_type):
                        raise TypeError(
                            f'For dimension {p}, expected all values to be of '
                            f'type {python_type}.'
                        )

                if not allow_missing_values:
                    if not set(value) <= allowed_values:
                        raise ValueError(
                            f"One or more values for dimension {p} are not "
                            "present in the image."
                        )

            else:
                if isinstance(value, np.generic):
                    # Change numpy to python type
                    value = value.item()
                if not isinstance(value, python_type):
                    raise TypeError(
                        f'For dimension {p}, expected value to be of type '
                        f'{python_type}.'
                    )

                if not allow_missing_values:
                    if value not in allowed_values:
                        raise ValueError(
                            f"Value {value} for dimension {p} are not "
                            "present in the image."
                        )

            if col_name in normalized_queries:
                raise ValueError(
                    'All dimensions must be unique.'
                )
            normalized_queries[col_name] = value

        return normalized_queries

    def _prepare_channel_tables(
        self,
        norm_channel_indices_list: list[dict[str, Any]],
        remap_channel_indices: Sequence[int] | None = None,
    ) -> tuple[list[str], list[str], list[_SQLTableDefinition]]:
        """Prepare query elements for a query involving output channels.

        This is common code shared between multiple query types that involve
        channels.

        Parameters
        ----------
        norm_channel_indices_list: list[dict[str, Any]]
            List of dictionaries defining the channel dimensions. The first
            item in the list corresponds to axis 3 of the output array, if any,
            the next to axis 4 and so so. Each dictionary has a format
            identical to that of ``stack_indices``, however the dimensions used
            must be distinct. Note that each item in the list may contain
            multiple items, provided that the number of items in each value
            matches within a single dictionary.
        remap_channel_indices: Sequence[int] | None, optional
            Use these values to remap the channel indices returned in the
            output iterator. The ith item applies to output channel i, and
            within that list index ``j`` is mapped to
            ``remap_channel_indices[i][j]``. Ignored if ``channel_indices`` is
            ``None``. If ``None``, or ``remap_channel_indices[i]`` is ``None``
            no mapping is performed for output channel ``i``.

        Returns
        -------
        list[str]:
            Channel selection strings for the query.
        list[str]:
            Channel join strings for the query.
        list[_SQLTableDefinition]:
            Temporary table definitions for the query.

        """
        selection_lines = []
        join_lines = []
        table_defs = []

        for i, channel_indices_dict in enumerate(
            norm_channel_indices_list
        ):
            channel_table_name = f'TemporaryChannelTable{i}'
            channel_column_defs = (
                ['OutputChannelIndex INTEGER UNIQUE NOT NULL'] +
                [
                    f'{c} {self._get_frame_lut_col_type(c)} NOT NULL'
                    for c in channel_indices_dict.keys()
                ]
            )

            selection_lines.append(
                f'{channel_table_name}.OutputChannelIndex'
            )

            num_channels = len(list(channel_indices_dict.values())[0])
            if (
                remap_channel_indices is not None and
                remap_channel_indices[i] is not None
            ):
                if isinstance(remap_channel_indices[i], np.ndarray):
                    # Need to call tolist to ensure elements end up as standard
                    # python types
                    output_channel_indices = remap_channel_indices[i].tolist()
                else:
                    output_channel_indices = remap_channel_indices[i]
            else:
                output_channel_indices = range(num_channels)

            channel_column_data = list(
                zip(
                    output_channel_indices,
                    *channel_indices_dict.values()
                )
            )

            table_defs.append(
                _SQLTableDefinition(
                    table_name=channel_table_name,
                    column_defs=channel_column_defs,
                    column_data=channel_column_data,
                )
            )

            channel_join_condition = ' AND '.join(
                f'L.{col} = {channel_table_name}.{col}'
                for col in channel_indices_dict.keys()
            )
            join_lines.append(
                f'INNER JOIN {channel_table_name} ON {channel_join_condition}'
            )

        return selection_lines, join_lines, table_defs

    def _prepare_filter_string(
        self,
        filters: dict[int | str, Any] | None = None,
        filters_use_indices: bool = False,
        allow_missing_values: bool = False,
    ) -> str:
        """Get a SQL-syntax filter string from filter definitions.

        Parameters
        ----------
        filters: Union[Dict[Union[int, str], Any], None], optional
            Filters to use to limit frames. Keys are the attributes used to
            define the filters, and values are the values that those attributes
            must match in order to pass the filter.
        filters_use_indices: bool, optional
            Whether the filters used dimension indices instead of the value
            itself.
        allow_missing_values: bool, optional
            Allow queries to include values that are not present in the dataset.
            Relevant parts of the output volume will be left blank.

        Returns
        -------
        str:
            SQL string implementing the requested filters.

        """
        if filters is not None:
            norm_filters = self._normalize_dimension_queries(
                filters,
                filters_use_indices,
                False,
                allow_missing_values=allow_missing_values,
            )

            filter_comparisons = []
            for c, v in norm_filters.items():
                if isinstance(v, str):
                    v = f"'{v}'"
                filter_comparisons.append(f'L.{c} = {v}')

            filter_str = 'WHERE ' + ' AND '.join(filter_comparisons)

            return filter_str

        return ''

    @contextmanager
    def _iterate_indices_for_stack(
        self,
        stack_indices: dict[int | str, Sequence[Any]] | None = None,
        stack_dimension_use_indices: bool = False,
        stack_table_def: _SQLTableDefinition | None = None,
        channel_indices: list[dict[int | str, Sequence[Any]]] | None = None,
        channel_dimension_use_indices: bool = False,
        remap_channel_indices: Sequence[int] | None = None,
        filters: dict[int | str, Any] | None = None,
        filters_use_indices: bool = False,
        allow_missing_values: bool = False,
        allow_missing_combinations: bool = False,
    ) -> Generator[
            Iterator[
                tuple[
                    int,  # frame index
                    tuple[slice, slice],  # input indexer
                    tuple[int, slice, slice],  # output indexer
                    tuple[int, ...],  # channel indexer
                ]
            ],
            None,
            None,
        ]:
        """Get indices required to reconstruct pixels into a stack of frames.

        The frames will be stacked down dimension 0 of the returned array.
        There may optionally be a channel dimension at dimension 3.

        Parameters
        ----------
        stack_indices: Dict[Union[int, str], Sequence[Any]]
            Dictionary defining the stack dimension (axis 0 of the output
            array). The keys define the dimensions used. They may be either the
            tags or keywords of attributes in the image's dimension index, or
            the special values, 'ReferencedSOPInstanceUID',
            and 'ReferencedFrameNumber'. The values of the dictionary give
            sequences of values of corresponding dimension that define each
            slice of the output array. Note that multiple dimensions may be
            used, in which case a frame must match the values of all provided
            dimensions to be placed in the output array. Required unless
            stack_table_def is not None.
        stack_dimension_use_indices: bool, optional
            If True, the values in ``stack_indices`` are integer-valued
            dimension *index* values. If False the dimension values themselves
            are used, whose type depends on the choice of dimension.
        stack_table_def: _SQLTableDefinition, optional
            Use this pre-computed table for the definition of the stack axis.
            Allows incorporation of computed data (such as volume positions)
            into the definition of the stack axis. Must be a table containing
            an integer column OutputFrameIndex, and a second integer column
            FrameNumber, where the latter is used to join to the main LUT.
            Incompatible with stack_indices.
        channel_indices: Union[List[Dict[Union[int, str], Sequence[Any]], None]], optional
            List of dictionaries defining the channel dimensions. The first
            item in the list corresponds to axis 3 of the output array, if any,
            the next to axis 4 and so so. Each dictionary has a format
            identical to that of ``stack_indices``, however the dimensions used
            must be distinct. Note that each item in the list may contain
            multiple items, provided that the number of items in each value
            matches within a single dictionary.
        channel_dimension_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the channel axis.
        remap_channel_indices: Union[Sequence[Union[Sequence[int], None], None], optional
            Use these values to remap the channel indices returned in the
            output iterator. The ith item applies to output channel i, and
            within that list index ``j`` is mapped to
            ``remap_channel_indices[i][j]``. Ignored if ``channel_indices`` is
            ``None``. If ``None``, or ``remap_channel_indices[i]`` is ``None``
            no mapping is performed for output channel ``i``.
        filters: Union[Dict[Union[int, str], Any], None], optional
            Additional filters to use to limit frames. Definition is similar to
            ``stack_indices`` except that the dictionary's values are single
            values rather than lists.
        filters_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the filters.
        allow_missing_values: bool, optional
            Allow queries to include values that are not present in the dataset.
            Relevant parts of the output volume will be left blank.
        allow_missing_combinations: bool, optional
            Allow queries to contain combinations of values along multiple
            dimensions that are not found in the dataset and place blank frames
            into the output at these locations. Ignored if
            `allow_missing_values` is True.

        Yields
        ------
        Iterator[ Tuple[int, Tuple[slice, slice], Tuple[int, slice, slice], Tuple[int, ...]]]:
            Indices required to construct the requested mask. Each triplet
            denotes the (frame_index, input indexer, spatial indexer, channel
            indexer) representing a list of "instructions" to create the
            requested output array by copying frames from the image dataset and
            inserting them into the output array.

        """  # noqa: E501
        if allow_missing_values:
            allow_missing_combinations = True

        all_columns = []
        if stack_indices is not None:
            norm_stack_indices = self._normalize_dimension_queries(
                stack_indices,
                stack_dimension_use_indices,
                True,
                allow_missing_values=allow_missing_values,
            )
            all_columns.extend(list(norm_stack_indices.keys()))
        elif stack_table_def is None:
            raise TypeError(
                "Either 'stack_indices' or 'stack_table_def' must "
                "be provided."
            )

        if channel_indices is not None:
            norm_channel_indices_list = [
                self._normalize_dimension_queries(
                    indices_dict,
                    channel_dimension_use_indices,
                    True,
                    allow_missing_values=allow_missing_values,
                ) for indices_dict in channel_indices
            ]
            for indices_dict in norm_channel_indices_list:
                all_columns.extend(list(indices_dict.keys()))
        else:
            norm_channel_indices_list = []

        all_dimensions = [
            c.replace('_DimensionIndexValues', '')
            for c in all_columns
        ]
        if len(set(all_dimensions)) != len(all_dimensions):
            raise ValueError(
                'Dimensions used for stack, channel, and filter must all be '
                'distinct.'
            )

        filter_str = self._prepare_filter_string(
            filters=filters,
            filters_use_indices=filters_use_indices,
            allow_missing_values=allow_missing_values,
        )

        if stack_table_def is None:
            # Check for uniqueness
            if not self._do_columns_identify_unique_frames(
                all_columns,
                filter=filter_str,
            ):
                raise RuntimeError(
                    'The chosen dimensions do not uniquely identify frames of '
                    'the image. You may need to provide further dimensions or '
                    'a filter to disambiguate.'
                )

            # Create temporary table of desired dimension indices
            stack_table_name = 'TemporaryStackTable'

            stack_column_defs = (
                ['OutputFrameIndex INTEGER UNIQUE NOT NULL'] +
                [
                    f'{c} {self._get_frame_lut_col_type(c)} NOT NULL'
                    for c in norm_stack_indices.keys()
                ]
            )
            stack_column_data = list(
                (i, *row)
                for i, row in enumerate(zip(*norm_stack_indices.values()))
            )
            stack_join_str = ' AND '.join(
                f'F.{col} = L.{col}' for col in norm_stack_indices.keys()
            )

            stack_table_def = _SQLTableDefinition(
                table_name=stack_table_name,
                column_defs=stack_column_defs,
                column_data=stack_column_data,
            )
        else:
            stack_join_str = 'F.FrameNumber = L.FrameNumber'

        selection_lines = [
            'F.OutputFrameIndex',  # frame index of the output array
            'L.FrameNumber - 1',  # frame *index* of the file
        ]

        (
            channel_selection_lines,
            channel_join_lines,
            channel_table_defs,
        ) = self._prepare_channel_tables(
            norm_channel_indices_list,
            remap_channel_indices,
        )

        selection_str = ', '.join(selection_lines + channel_selection_lines)

        # Construct the query. The ORDER BY is not logically necessary but
        # seems to improve performance of the downstream numpy operations,
        # presumably as it is more cache efficient
        query_template = (
            'SELECT {selection_str} '
            f'FROM {stack_table_def.table_name} F '
            f'INNER JOIN FrameLUT L ON {stack_join_str} '
            f'{" ".join(channel_join_lines)} '
            f'{filter_str} '
            '{order_str}'
        )

        with self._generate_temp_tables([stack_table_def] + channel_table_defs):

            if not allow_missing_combinations:
                counting_query = query_template.format(
                    selection_str='COUNT(*)',
                    order_str='',
                )

                # Calculate the number of output frames
                number_of_output_frames = len(stack_table_def.column_data)
                for tdef in channel_table_defs:
                    number_of_output_frames *= len(tdef.column_data)

                # Use a query to find the number of input frames
                found_number = next(self._db_con.execute(counting_query))[0]

                # If these two numbers are not the same, there are missing
                # frames
                if found_number != number_of_output_frames:
                    raise RuntimeError(
                        'The requested set of frames includes frames that '
                        'are missing from the image. You may need to allow '
                        'missing combinations or add additional filters.'
                    )

            full_query = query_template.format(
                selection_str=selection_str,
                order_str='ORDER BY F.OutputFrameIndex',
            )

            yield (
                (
                    fi,
                    (slice(None), slice(None)),
                    (fo, slice(None), slice(None)),
                    tuple(channel),
                )
                for (fo, fi, *channel) in self._db_con.execute(full_query)
            )

    @staticmethod
    def _standardize_slice_indices(
        slice_start: int | None,
        slice_end: int | None,
        n_vol_positions: int,
        as_indices: bool = False,
    ) -> tuple[int, int]:
        """Standardize format of slice indices as given by the user.

        This includes interpretation of None values, and negatives.

        Parameters
        ----------
        slice_start: Optional[int]
            First slice.
        slice_end: Optional[int]
            One beyond last slice.
        n_vol_positions: int
            Total number of volume positions in this volume.
        as_indices: bool
            Interpret start and end parameters as zero-based indices (if True)
            or one-based numbers (if False).

        Returns
        -------
        slice_start: int
            First row as non-negative zero-based index.
        slice_end: int
            One beyond last slice First as non-negative zero-based index.

        """
        # Standardize on zero-based slice indices
        original_slice_end = slice_end
        if not as_indices:
            if slice_start is not None:
                if slice_start == 0:
                    raise ValueError(
                        "Value of 'slice_start' cannot be 0. Did you mean to "
                        "pass 'as_indices=True'?"
                    )
                elif slice_start > 0:
                    slice_start = slice_start - 1

            if slice_end is not None:
                if slice_start == 0:
                    raise ValueError()
                elif slice_end > 0:
                    slice_end = slice_end - 1

        if slice_start is None:
            slice_start = 0
        if slice_start < 0:
            slice_start = n_vol_positions + slice_start

        if slice_end is None:
            slice_end = n_vol_positions
        elif slice_end > n_vol_positions:
            raise IndexError(
                f"Value of {original_slice_end} is not valid for image with "
                f"{n_vol_positions} volume positions."
            )
        elif slice_end < 0:
            if slice_end < (- n_vol_positions):
                raise IndexError(
                    f"Value of {original_slice_end} is not valid for image "
                    f"with {n_vol_positions} volume positions."
                )
            slice_end = n_vol_positions + slice_end

        number_of_slices = cast(int, slice_end) - slice_start

        if number_of_slices < 1:
            raise ValueError(
                "The combination of 'slice_start' and 'slice_end' gives an "
                "empty volume."
            )

        return slice_start, slice_end

    @staticmethod
    def _standardize_row_column_indices(
        row_start: int | None,
        row_end: int | None,
        column_start: int | None,
        column_end: int | None,
        rows: int,
        columns: int,
        as_indices: bool = False,
        outputs_as_indices: bool = False,
    ) -> tuple[int, int, int, int]:
        """Standardize format of row/column indices as given by user.

        This includes interpretation of None values, and negatives.

        Parameters
        ----------
        row_start: Optional[int]
            First row.
        row_end: Optional[int]
            One beyond last row.
        column_start: Optional[int]
            First column.
        column_end: Optional[int]
            One beyond last column.
        rows: int
            Number of rows in image.
        columns: int
            Number of columns in image.
        as_indices: bool
            Interpret start and end parameters as zero-based indices (if True)
            or one-based numbers (if False).
        outputs_as_indices: bool
            Return start and end outputs as zero-based indices (if True) or
            one-based numbers (if False).

        Returns
        -------
        row_start: int
            First row as non-negative integer.
        row_end: int
            One beyond last row as non-negative integer.
        column_start: Optional[int]
            First column as non-negative integer.
        column_end: Optional[int]
            One beyond last column as non-negative integer.

        """
        # Store the passed values for use in error messages
        original_row_start = row_start
        original_row_end = row_end
        original_column_start = column_start
        original_column_end = column_end

        # Standardize on 1 based indices to use internally
        if as_indices:
            if row_start is not None and row_start >= 0:
                row_start = row_start + 1
            if row_end is not None and row_end >= 0:
                row_end = row_end + 1
            if column_start is not None and column_start >= 0:
                column_start = column_start + 1
            if column_end is not None and column_end >= 0:
                column_end = column_end + 1

        if row_start is None:
            row_start = 1
        if row_end is None:
            row_end = rows + 1
        if column_start is None:
            column_start = 1
        if column_end is None:
            column_end = columns + 1

        if column_start == 0 or row_start == 0:
            raise ValueError(
                'Arguments "row_start" and "column_start" may not be 0. '
                "Perhaps you meant to pass 'as_indices=True'?"
            )

        if row_start > rows:
            raise ValueError(
                f'Value of {original_row_start} for "row_start" is out '
                f'of range for image with {rows} '
                'rows in the total pixel matrix.'
            )
        elif row_start < 0:
            row_start = rows + row_start + 1
        if row_end > rows + 1:
            raise ValueError(
                f'Value of {original_row_end} for "row_end" is out '
                f'of range for image with {rows} '
                'rows in the total pixel matrix.'
            )
        elif row_end < 0:
            row_end = rows + row_end + 1

        if column_start > columns:
            raise ValueError(
                f'Value of {original_column_start} for "column_start" is out '
                f'of range for image with {columns} '
                'columns in the total pixel matrix.'
            )
        elif column_start < 0:
            column_start = columns + column_start + 1
        if column_end > columns + 1:
            raise ValueError(
                f'Value of {original_column_end} for "column_end" is out '
                f'of range for image with {columns} '
                'columns in the total pixel matrix.'
            )
        elif column_end < 0:
            column_end = columns + column_end + 1

        if outputs_as_indices:
            return (
                row_start - 1,
                row_end - 1,
                column_start - 1,
                column_end - 1
            )
        else:
            return row_start, row_end, column_start, column_end

    @contextmanager
    def _iterate_indices_for_tiled_region(
        self,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
        as_indices: bool = False,
        channel_indices: list[dict[int | str, Sequence[Any]]] | None = None,
        channel_dimension_use_indices: bool = False,
        remap_channel_indices: Sequence[int] | None = None,
        filters: dict[int | str, Any] | None = None,
        filters_use_indices: bool = False,
        allow_missing_values: bool = False,
        allow_missing_combinations: bool = False,
    ) -> tuple[
            Generator[
                Iterator[
                    tuple[
                        int,
                        tuple[slice, slice],
                        tuple[slice, slice],
                        tuple[int, ...]
                    ]
                ],
                None,
                None,
            ],
            tuple[int, int]
        ]:
        """Iterate over segmentation frame indices for a given region of the
        image's total pixel matrix.

        This is intended for the case of an image that is stored as a tiled
        representation of total pixel matrix.

        This yields an iterator to the underlying database result that iterates
        over information on the steps required to construct the requested
        image from the stored frame.

        This method is intended to be used as a context manager that yields the
        requested iterator. The iterator is only valid while the context
        manager is active.

        Parameters
        ----------
        row_start: int, optional
            1-based row number in the total pixel matrix of the first row to
            include in the output array. Alternatively a zero-based row index
            if ``as_indices`` is True. May be negative, in which case the last
            row is considered index -1. If ``None``, the first row of the
            output is the first row of the total pixel matrix (regardless of
            the value of ``as_indices``).
        row_end: Union[int, None], optional
            1-based row index in the total pixel matrix of the first row beyond
            the last row to include in the output array. A ``row_end`` value of
            ``n`` will include rows ``n - 1`` and below, similar to standard
            Python indexing. If ``None``, rows up until the final row of the
            total pixel matrix are included. May be negative, in which case the
            last row is considered index -1.
        column_start: int, optional
            1-based column number in the total pixel matrix of the first column
            to include in the output array. Alternatively a zero-based column
            index if ``as_indices`` is True.May be negative, in which case the
            last column is considered index -1.
        column_end: Union[int, None], optional
            1-based column index in the total pixel matrix of the first column
            beyond the last column to include in the output array. A
            ``column_end`` value of ``n`` will include columns ``n - 1`` and
            below, similar to standard Python indexing. If ``None``, columns up
            until the final column of the total pixel matrix are included. May
            be negative, in which case the last column is considered index -1.
        as_indices: bool, optional
            If True, interpret all row/column numbering parameters
            (``row_start``, ``row_end``, ``column_start``, and ``column_end``)
            as zero-based indices as opposed to the default one-based numbers
            used within DICOM.
        channel_indices: Union[List[Dict[Union[int, str], Sequence[Any]], None]], optional
            List of dictionaries defining the channel dimensions. Within each
            dictionary, The keys define the dimensions used. They may be either
            the tags or keywords of attributes in the image's dimension index,
            or the special values 'ReferencedSOPInstanceUID',
            and 'ReferencedFrameNumber'. The values of the dictionary give
            sequences of values of corresponding dimension that define each
            slice of the output array. Note that multiple dimensions may be
            used, in which case a frame must match the values of all provided
            dimensions to be placed in the output array.The first item in the
            list corresponds to axis 3 of the output array, if any, the next to
            axis 4 and so so. Note that each item in the list may contain
            multiple items, provided that the number of items in each value
            matches within a single dictionary.
        channel_dimension_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the channel axis.
        remap_channel_indices: Union[Sequence[int], None], optional
            Use these values to remap the channel indices returned in the
            output iterator. The ith item applies to output channel i, and
            within that list index ``j`` is mapped to
            ``remap_channel_indices[i][j]``. Ignored if ``channel_indices`` is
            ``None``. If ``None``, or ``remap_channel_indices[i]`` is ``None``
            no mapping is performed for output channel ``i``.
        filters: Union[Dict[Union[int, str], Any], None], optional
            Additional filters to use to limit frames. Definition is similar to
            ``stack_indices`` except that the dictionary's values are single
            values rather than lists.
        filters_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the filters.
        allow_missing_values: bool, optional
            Allow queries to include values that are not present in the dataset.
            Relevant parts of the output volume will be left blank.
        allow_missing_combinations: bool, optional
            Allow queries to contain combinations of values along multiple
            dimensions that are not found in the dataset and place blank frames
            into the output at these locations. Ignored if
            `allow_missing_values` is True.

        Yields
        ------
        Iterator[Tuple[int, Tuple[slice, slice], Tuple[slice, slice], Tuple[int, ...]]]:
            Indices required to construct the requested mask. Each triplet
            denotes the (frame_index, input indexer, spatial indexer, channel
            indexer) representing a list of "instructions" to create the
            requested output array by copying frames from the image dataset and
            inserting them into the output array.
        tuple[int, int]:
            Output shape.

        """  # noqa: E501
        if allow_missing_values:
            allow_missing_combinations = True

        all_columns = [
            'RowPositionInTotalImagePixelMatrix',
            'ColumnPositionInTotalImagePixelMatrix',
        ]
        if channel_indices is not None:
            norm_channel_indices_list = [
                self._normalize_dimension_queries(
                    indices_dict,
                    channel_dimension_use_indices,
                    True,
                    allow_missing_values=allow_missing_values,
                ) for indices_dict in channel_indices
            ]
            for indices_dict in norm_channel_indices_list:
                all_columns.extend(list(indices_dict.keys()))
        else:
            norm_channel_indices_list = []

        filter_str = self._prepare_filter_string(
            filters=filters,
            filters_use_indices=filters_use_indices,
            allow_missing_values=allow_missing_values,
        )

        all_dimensions = [
            c.replace('_DimensionIndexValues', '')
            for c in all_columns
        ]
        if len(all_dimensions) != len(all_dimensions):
            raise ValueError(
                'Dimensions used for tile position, channel, and filter '
                'must all be distinct.'
            )

        # Check for uniqueness
        if not self._do_columns_identify_unique_frames(
            all_columns,
            filter=filter_str
        ):
            raise RuntimeError(
                'The chosen dimensions do not uniquely identify frames of'
                'the image. You may need to provide further dimensions or '
                'a filter to disambiguate.'
            )

        (
            row_start, row_end, column_start, column_end,
        ) = self._standardize_row_column_indices(
            row_start,
            row_end,
            column_start,
            column_end,
            rows=self.TotalPixelMatrixRows,
            columns=self.TotalPixelMatrixColumns,
            as_indices=as_indices,
            outputs_as_indices=False,
        )

        output_shape = (
            row_end - row_start,
            column_end - column_start,
        )

        th, tw = self.Rows, self.Columns

        oh = row_end - row_start
        ow = column_end - column_start

        row_offset_start = row_start - th + 1
        column_offset_start = column_start - tw + 1

        selection_lines = [
            'L.RowPositionInTotalImagePixelMatrix',
            'L.ColumnPositionInTotalImagePixelMatrix',
            'L.FrameNumber - 1',
        ]

        (
            channel_selection_lines,
            channel_join_lines,
            channel_table_defs,
        ) = self._prepare_channel_tables(
            norm_channel_indices_list,
            remap_channel_indices,
        )

        selection_str = ', '.join(selection_lines + channel_selection_lines)

        # Construct the query The ORDER BY is not logically necessary
        # but seems to improve performance of the downstream numpy
        # operations, presumably as it is more cache efficient
        # Create temporary table of channel indices
        query_template = (
            'SELECT {selection_str} '
            'FROM FrameLUT L '
            f'{" ".join(channel_join_lines)} '
            'WHERE ('
            '    L.RowPositionInTotalImagePixelMatrix >= '
            f'        {row_offset_start}'
            f'    AND L.RowPositionInTotalImagePixelMatrix < {row_end}'
            '    AND L.ColumnPositionInTotalImagePixelMatrix >= '
            f'        {column_offset_start}'
            f'    AND L.ColumnPositionInTotalImagePixelMatrix < {column_end}'
            f'    {filter_str.replace("WHERE", "AND")} '
            ') '
            '{order_str}'
        )

        order_str = (
            'ORDER BY '
            '     L.RowPositionInTotalImagePixelMatrix,'
            '     L.ColumnPositionInTotalImagePixelMatrix'
        )

        with self._generate_temp_tables(channel_table_defs):

            if (
                not allow_missing_combinations and
                self.get('DimensionOrganizationType', '') != "TILED_FULL"
            ):
                counting_query = query_template.format(
                    selection_str='COUNT(*)',
                    order_str='',
                )

                # Calculate the number of output frames
                v_frames = ((row_end - 2) // th) - ((row_start - 1) // th) + 1
                h_frames = (
                    ((column_end - 2) // tw) - ((column_start - 1) // tw) + 1
                )
                number_of_output_frames = v_frames * h_frames
                for tdef in channel_table_defs:
                    number_of_output_frames *= len(tdef.column_data)

                # Use a query to find the number of input frames
                found_number = next(self._db_con.execute(counting_query))[0]

                # If these two numbers are not the same, there are missing
                # frames
                if found_number != number_of_output_frames:
                    raise RuntimeError(
                        'The requested set of frames includes frames that '
                        'are missing from the image. You may need to allow '
                        'missing frames or add additional filters.'
                    )

            full_query = query_template.format(
                selection_str=selection_str,
                order_str=order_str,
            )

            yield (
                (
                    fi,
                    (
                        slice(
                            max(row_start - rp, 0),
                            min(row_end - rp, th)
                        ),
                        slice(
                            max(column_start - cp, 0),
                            min(column_end - cp, tw)
                        ),
                    ),
                    (
                        slice(
                            max(rp - row_start, 0),
                            min(rp + th - row_start, oh)
                        ),
                        slice(
                            max(cp - column_start, 0),
                            min(cp + tw - column_start, ow)
                        ),
                    ),
                    tuple(channel),
                )
                for (rp, cp, fi, *channel) in self._db_con.execute(full_query)
            ), output_shape

    @classmethod
    def from_file(
        cls,
        fp: str | bytes | PathLike | BinaryIO,
        lazy_frame_retrieval: bool = False
    ) -> Self:
        """Read an image stored in DICOM File Format.

        Parameters
        ----------
        fp: Union[str, bytes, os.PathLike]
            Any file-like object representing a DICOM file containing an
            image.
        lazy_frame_retrieval: bool
            If True, the returned image will retrieve frames from the file as
            requested, rather than loading in the entire object to memory
            initially. This may be a good idea if file reading is slow and you
            are likely to need only a subset of the frames in the image.

        Returns
        -------
        Self:
            Image read from the file.

        """
        if lazy_frame_retrieval:
            if isinstance(fp, bytes):
                fp = DicomBytesIO(fp)
            elif not isinstance(fp, (str, PathLike, DicomIO)):
                # General BinaryIO object, wrap in DicomIO
                fp = DicomIO(fp)

            reader = ImageFileReader(fp)
            metadata = reader._change_metadata_ownership()
            image = cls.from_dataset(metadata, copy=False)
            image._file_reader = reader
        else:
            image = cls.from_dataset(_wrapped_dcmread(fp), copy=False)

        return image


class Image(_Image):

    """Class representing a general DICOM image.

    An "image" is any object representing an Image Information Entity.

    Note that this does not correspond to a particular SOP class in DICOM, but
    instead captures behavior that is common to a number of SOP classes. It
    provides various methods to access the frames in the image, apply
    transforms specified in the dataset to the pixels, and arrange them
    spatially.

    The class may not be instantiated directly, but should be created from an
    existing dataset.

    See :doc:`image` for an introduction to using this class.

    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'Instances of this class should not be directly instantiated. Use '
            'the from_dataset method or the imread function instead.'
        )

    def get_volume(
        self,
        *,
        slice_start: int | None = None,
        slice_end: int | None = None,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
        as_indices: bool = False,
        dtype: type | str | np.dtype = np.float64,
        apply_real_world_transform: bool | None = None,
        real_world_value_map_selector: int | str | Code | CodedConcept = 0,
        apply_modality_transform: bool | None = None,
        apply_voi_transform: bool | None = False,
        voi_transform_selector: int | str | VOILUTTransformation = 0,
        voi_output_range: tuple[float, float] = (0.0, 1.0),
        apply_presentation_lut: bool = True,
        apply_palette_color_lut: bool | None = None,
        apply_icc_profile: bool | None = None,
        allow_missing_positions: bool = False,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> Volume:
        """Create a :class:`highdicom.Volume` from the image.

        This is only possible in two situations: either the image represents a
        regularly-spaced 3D volume, or a tiled 2D total pixel matrix.

        Parameters
        ----------
        slice_start: int | none, optional
            1-based index of the "volume position" of the first slice of the
            returned volume. the "volume position" refers to the position of
            slices after sorting spatially, and may correspond to any frame in
            the segmentation file, depending on its construction. may be
            negative, in which case standard python indexing behavior is
            followed (-1 corresponds to the last volume position, etc).
        slice_end: union[int, none], optional
            1-based index of the "volume position" one beyond the last slice
            of the returned volume. the "volume position" refers to the
            position of slices after sorting spatially, and may correspond to
            any frame in the segmentation file, depending on its construction.
            may be negative, in which case standard python indexing behavior is
            followed (-1 corresponds to the last volume position, etc). if
            none, the last volume position is included as the last output
            slice.
        row_start: int, optional
            1-based row number in the total pixel matrix of the first row to
            include in the output array. alternatively a zero-based row index
            if ``as_indices`` is true. may be negative, in which case the last
            row is considered index -1. if ``none``, the first row of the
            output is the first row of the total pixel matrix (regardless of
            the value of ``as_indices``).
        row_end: union[int, none], optional
            1-based row index in the total pixel matrix of the first row beyond
            the last row to include in the output array. a ``row_end`` value of
            ``n`` will include rows ``n - 1`` and below, similar to standard
            python indexing. if ``none``, rows up until the final row of the
            total pixel matrix are included. may be negative, in which case the
            last row is considered index -1.
        column_start: int, optional
            1-based column number in the total pixel matrix of the first column
            to include in the output array. alternatively a zero-based column
            index if ``as_indices`` is true.may be negative, in which case the
            last column is considered index -1.
        column_end: union[int, none], optional
            1-based column index in the total pixel matrix of the first column
            beyond the last column to include in the output array. a
            ``column_end`` value of ``n`` will include columns ``n - 1`` and
            below, similar to standard python indexing. if ``none``, columns up
            until the final column of the total pixel matrix are included. may
            be negative, in which case the last column is considered index -1.
        as_indices: bool, optional
            if true, interpret all slice/row/column numbering parameters
            (``row_start``, ``row_end``, ``column_start``, and ``column_end``)
            as zero-based indices as opposed to the default one-based numbers
            used within dicom.
        dtype: Union[type, str, numpy.dtype], optional
            Data type of the returned array.
        apply_real_world_transform: bool | None, optional
            Whether to apply a real-world value map to the frame.
            A real-world value maps converts stored pixel values to output
            values with a real-world meaning, either using a LUT or a linear
            slope and intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if present but no error will be raised if
            it is not present.

            Note that if the dataset contains both a modality LUT and a real
            world value map, the real world value map will be applied
            preferentially. This also implies that specifying both
            ``apply_real_world_transform`` and ``apply_modality_transform`` to
            True is not permitted.
        real_world_value_map_selector: int | str | pydicom.sr.coding.Code | highdicom.sr.coding.CodedConcept, optional
            Specification of the real world value map to use (multiple may be
            present in the dataset). If an int, it is used to index the list of
            available maps. A negative integer may be used to index from the
            end of the list following standard Python indexing convention. If a
            str, the string will be used to match the ``"LUTLabel"`` attribute
            to select the map. If a ``pydicom.sr.coding.Code`` or
            ``highdicom.sr.coding.CodedConcept``, this will be used to match
            the units (contained in the ``"MeasurementUnitsCodeSequence"``
            attribute).
        apply_modality_transform: bool | None, optional
            Whether to apply the modality transform (if present in the
            dataset) to the frame. The modality transform maps stored pixel
            values to output values, either using a LUT or rescale slope and
            intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        apply_voi_transform: bool | None, optional
            Apply the value-of-interest (VOI) transform (if present in the
            dataset), which limits the range of pixel values to a particular
            range of interest using either a windowing operation or a LUT.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        voi_transform_selector: int | str | highdicom.VOILUTTransformation, optional
            Specification of the VOI transform to select (multiple may be
            present). May either be an int or a str. If an int, it is
            interpreted as a (zero-based) index of the list of VOI transforms
            to apply. A negative integer may be used to index from the end of
            the list following standard Python indexing convention. If a str,
            the string that will be used to match the
            ``"WindowCenterWidthExplanation"`` or the ``"LUTExplanation"``
            attributes to choose from multiple VOI transforms. Note that such
            explanations are optional according to the standard and therefore
            may not be present. Ignored if ``apply_voi_transform`` is ``False``
            or no VOI transform is included in the dataset.

            Alternatively, a user-defined
            :class:`highdicom.VOILUTTransformation` may be supplied.
            This will override any such transform specified in the dataset.
        voi_output_range: Tuple[float, float], optional
            Range of output values to which the VOI range is mapped. Only
            relevant if ``apply_voi_transform`` is True and a VOI transform is
            present.
        apply_palette_color_lut: bool | None, optional
            Apply the palette color LUT, if present in the dataset. The palette
            color LUT maps a single sample for each pixel stored in the dataset
            to a 3 sample-per-pixel color image.
        apply_presentation_lut: bool, optional
            Apply the presentation LUT transform to invert the pixel values. If
            the PresentationLUTShape is present with the value ``'INVERSE'``,
            or the PresentationLUTShape is not present but the Photometric
            Interpretation is MONOCHROME1, convert the range of the output
            pixels corresponds to MONOCHROME2 (in which high values are
            represent white and low values represent black). Ignored if
            PhotometricInterpretation is not MONOCHROME1 and the
            PresentationLUTShape is not present, or if a real world value
            transform is applied.
        apply_icc_profile: bool | None, optional
            Whether colors should be corrected by applying an ICC
            transform. Will only be performed if metadata contain an
            ICC Profile.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present, but no error will be
            raised if it is not present.
        allow_missing_positions: bool, optional
            Allow spatial positions the output array to be blank because these
            frames are omitted from the image. If False and missing positions
            are found, an error is raised.
        rtol: float | None, optional
            Relative tolerance for determining spacing regularity. If slice
            spacings vary by less that this proportion of the average spacing,
            they are considered to be regular. If neither ``rtol`` or ``atol``
            are provided, a default relative tolerance of 0.01 is used.
        atol: float | None, optional
            Absolute tolerance for determining spacing regularity. If slice
            spacings vary by less that this value (in mm), they are considered
            to be regular. Incompatible with ``rtol``.

        Returns
        -------
        highdicom.Volume:
            Volume formed from frames of the image.

        Note
        ----
        By default, this method uses 1-based indexing of rows and columns in
        order to match the conventions used in the DICOM standard. The first
        row of the total pixel matrix is row 1, and the last is
        ``self.TotalPixelMatrixRows``. This is is unlike standard Python and
        NumPy indexing which is 0-based. For negative indices, the two are
        equivalent with the final row/column having index -1. To switch to
        standard Python behavior, specify ``as_indices=True``.

        Note
        ----
        The parameters ``row_start``, ``row_end``, ``column_start`` and
        ``column_end`` are provided primarily for the case where the volume is
        formed from frames tiled into a total pixel matrix. In other scenarios,
        it will behave as expected, but will not reduce the number of frames
        that have to be decoded and transformed.

        """  # noqa: E501
        if self._coordinate_system is None:
            raise RuntimeError(
                "Image does not exist within a frame-of-reference "
                "coordinate system."
            )
        if self.is_tiled:
            total_rows = self.TotalPixelMatrixRows
            total_columns = self.TotalPixelMatrixColumns
        else:
            total_rows = self.Rows
            total_columns = self.Columns

        (
            row_start, row_end, column_start, column_end,
        ) = self._standardize_row_column_indices(
            row_start,
            row_end,
            column_start,
            column_end,
            rows=total_rows,
            columns=total_columns,
            as_indices=as_indices,
            outputs_as_indices=True,
        )

        channel_spec = None

        color_type = self._get_color_type()
        if (
            color_type == _ImageColorType.COLOR or
            (
                color_type == _ImageColorType.PALETTE_COLOR and
                (apply_palette_color_lut or apply_palette_color_lut is None)
            )
        ):
            channel_spec = {RGB_COLOR_CHANNEL_DESCRIPTOR: ['R', 'G', 'B']}

        if self.is_tiled:
            volume_geometry = self._get_volume_geometry()

            slice_start, slice_end = self._standardize_slice_indices(
                slice_start=slice_start,
                slice_end=slice_end,
                as_indices=as_indices,
                n_vol_positions=volume_geometry.spatial_shape[0]
            )

            array = self.get_total_pixel_matrix(
                row_start=row_start,
                row_end=row_end,
                column_start=column_start,
                column_end=column_end,
                apply_real_world_transform=apply_real_world_transform,
                real_world_value_map_selector=real_world_value_map_selector,
                apply_modality_transform=apply_modality_transform,
                apply_voi_transform=apply_voi_transform,
                voi_transform_selector=voi_transform_selector,
                voi_output_range=voi_output_range,
                apply_presentation_lut=apply_presentation_lut,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
                as_indices=True,  # standardized earlier as indices
                dtype=dtype,
            )[None]

            affine = volume_geometry[
                :,
                row_start:,
                column_start:,
            ].affine
        else:
            # Check that the combination of frame numbers uniquely identify
            # frames
            columns = [
                'ImagePositionPatient_0',
                'ImagePositionPatient_1',
                'ImagePositionPatient_2'
            ]
            if not self._do_columns_identify_unique_frames(columns):
                raise RuntimeError(
                    'Volume positions and do not '
                    'uniquely identify frames of the image.'
                )

            (
                stack_table_def,
                volume_geometry,
            ) = self._prepare_volume_positions_table(
                rtol=rtol,
                atol=atol,
                allow_missing_positions=allow_missing_positions,
                slice_start=slice_start,
                slice_end=slice_end,
                as_indices=as_indices,
            )

            with self._iterate_indices_for_stack(
                stack_table_def=stack_table_def,
            ) as indices:

                array = self._get_pixels_by_frame(
                    spatial_shape=volume_geometry.spatial_shape[0],
                    indices_iterator=indices,
                    apply_real_world_transform=apply_real_world_transform,
                    real_world_value_map_selector=real_world_value_map_selector,
                    apply_modality_transform=apply_modality_transform,
                    apply_voi_transform=apply_voi_transform,
                    voi_transform_selector=voi_transform_selector,
                    voi_output_range=voi_output_range,
                    apply_presentation_lut=apply_presentation_lut,
                    apply_palette_color_lut=apply_palette_color_lut,
                    apply_icc_profile=apply_icc_profile,
                    dtype=dtype,
                )

            array = array[:, row_start:row_end, column_start:column_end]
            affine = volume_geometry[
                :,
                row_start:row_end,
                column_start:column_end,
            ].affine

        return Volume(
            array=array,
            affine=affine,
            coordinate_system=self._coordinate_system,
            frame_of_reference_uid=self.FrameOfReferenceUID,
            channels=channel_spec,
        )

    def get_total_pixel_matrix(
        self,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
        dtype: type | str | np.dtype = np.float64,
        apply_real_world_transform: bool | None = None,
        real_world_value_map_selector: int | str | Code | CodedConcept = 0,
        apply_modality_transform: bool | None = None,
        apply_voi_transform: bool | None = False,
        voi_transform_selector: int | str | VOILUTTransformation = 0,
        voi_output_range: tuple[float, float] = (0.0, 1.0),
        apply_presentation_lut: bool = True,
        apply_palette_color_lut: bool | None = None,
        apply_icc_profile: bool | None = None,
        as_indices: bool = False,
    ):
        """Get the pixel array as a (region of) the total pixel matrix.

        This is only possible for tiled images, which are images in which the
        frames are arranged over a 2D plane (like tiles over a floor) and
        typically occur in microscopy. This method is not relevant for other
        types of image.

        Parameters
        ----------
        row_start: int, optional
            1-based row number in the total pixel matrix of the first row to
            include in the output array. Alternatively a zero-based row index
            if ``as_indices`` is True. May be negative, in which case the last
            row is considered index -1. If ``None``, the first row of the
            output is the first row of the total pixel matrix (regardless of
            the value of ``as_indices``).
        row_end: Union[int, None], optional
            1-based row index in the total pixel matrix of the first row beyond
            the last row to include in the output array. A ``row_end`` value of
            ``n`` will include rows ``n - 1`` and below, similar to standard
            Python indexing. If ``None``, rows up until the final row of the
            total pixel matrix are included. May be negative, in which case the
            last row is considered index -1.
        column_start: int, optional
            1-based column number in the total pixel matrix of the first column
            to include in the output array. Alternatively a zero-based column
            index if ``as_indices`` is True.May be negative, in which case the
            last column is considered index -1.
        column_end: Union[int, None], optional
            1-based column index in the total pixel matrix of the first column
            beyond the last column to include in the output array. A
            ``column_end`` value of ``n`` will include columns ``n - 1`` and
            below, similar to standard Python indexing. If ``None``, columns up
            until the final column of the total pixel matrix are included. May
            be negative, in which case the last column is considered index -1.
        dtype: Union[type, str, numpy.dtype], optional
            Data type of the returned array.
        apply_real_world_transform: bool | None, optional
            Whether to apply a real-world value map to the frame.
            A real-world value maps converts stored pixel values to output
            values with a real-world meaning, either using a LUT or a linear
            slope and intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if present but no error will be raised if
            it is not present.

            Note that if the dataset contains both a modality LUT and a real
            world value map, the real world value map will be applied
            preferentially. This also implies that specifying both
            ``apply_real_world_transform`` and ``apply_modality_transform`` to
            True is not permitted.
        real_world_value_map_selector: int | str | pydicom.sr.coding.Code | highdicom.sr.coding.CodedConcept, optional
            Specification of the real world value map to use (multiple may be
            present in the dataset). If an int, it is used to index the list of
            available maps. A negative integer may be used to index from the
            end of the list following standard Python indexing convention. If a
            str, the string will be used to match the ``"LUTLabel"`` attribute
            to select the map. If a ``pydicom.sr.coding.Code`` or
            ``highdicom.sr.coding.CodedConcept``, this will be used to match
            the units (contained in the ``"MeasurementUnitsCodeSequence"``
            attribute).
        apply_modality_transform: bool | None, optional
            Whether to apply the modality transform (if present in the
            dataset) to the frame. The modality transform maps stored pixel
            values to output values, either using a LUT or rescale slope and
            intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        apply_voi_transform: bool | None, optional
            Apply the value-of-interest (VOI) transform (if present in the
            dataset), which limits the range of pixel values to a particular
            range of interest using either a windowing operation or a LUT.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        voi_transform_selector: int | str | highdicom.VOILUTTransformation, optional
            Specification of the VOI transform to select (multiple may be
            present). May either be an int or a str. If an int, it is
            interpreted as a (zero-based) index of the list of VOI transforms
            to apply. A negative integer may be used to index from the end of
            the list following standard Python indexing convention. If a str,
            the string that will be used to match the
            ``"WindowCenterWidthExplanation"`` or the ``"LUTExplanation"``
            attributes to choose from multiple VOI transforms. Note that such
            explanations are optional according to the standard and therefore
            may not be present. Ignored if ``apply_voi_transform`` is ``False``
            or no VOI transform is included in the datasets.

            Alternatively, a user-defined
            :class:`highdicom.VOILUTTransformation` may be supplied.
            This will override any such transform specified in the dataset.
        voi_output_range: Tuple[float, float], optional
            Range of output values to which the VOI range is mapped. Only
            relevant if ``apply_voi_transform`` is True and a VOI transform is
            present.
        apply_palette_color_lut: bool | None, optional
            Apply the palette color LUT, if present in the dataset. The palette
            color LUT maps a single sample for each pixel stored in the dataset
            to a 3 sample-per-pixel color image.
        apply_presentation_lut: bool, optional
            Apply the presentation LUT transform to invert the pixel values. If
            the PresentationLUTShape is present with the value ``'INVERSE'``,
            or the PresentationLUTShape is not present but the Photometric
            Interpretation is MONOCHROME1, convert the range of the output
            pixels corresponds to MONOCHROME2 (in which high values are
            represent white and low values represent black). Ignored if
            PhotometricInterpretation is not MONOCHROME1 and the
            PresentationLUTShape is not present, or if a real world value
            transform is applied.
        apply_icc_profile: bool | None, optional
            Whether colors should be corrected by applying an ICC
            transform. Will only be performed if metadata contain an
            ICC Profile.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present, but no error will be
            raised if it is not present.
        as_indices: bool, optional
            If True, interpret all row/column numbering parameters
            (``row_start``, ``row_end``, ``column_start``, and ``column_end``)
            as zero-based indices as opposed to the default one-based numbers
            used within DICOM.

        Returns
        -------
        pixel_array: numpy.ndarray
            Pixel array representing the image's total pixel matrix.

        Note
        ----
        By default, this method uses 1-based indexing of rows and columns in
        order to match the conventions used in the DICOM standard. The first
        row of the total pixel matrix is row 1, and the last is
        ``self.TotalPixelMatrixRows``. This is is unlike standard Python and
        NumPy indexing which is 0-based. For negative indices, the two are
        equivalent with the final row/column having index -1. To switch to
        standard Python behavior, specify ``as_indices=True``.

        """  # noqa: E501
        # Check whether this segmentation is appropriate for tile-based indexing
        if not self.is_tiled:
            raise RuntimeError("Image is not a tiled image.")
        if not self.is_indexable_as_total_pixel_matrix():
            raise RuntimeError(
                "Image does not have appropriate dimension indices "
                "to be indexed as a total pixel matrix."
            )

        with self._iterate_indices_for_tiled_region(
            row_start=row_start,
            row_end=row_end,
            column_start=column_start,
            column_end=column_end,
            as_indices=as_indices,
        ) as (indices, output_shape):

            return self._get_pixels_by_frame(
                spatial_shape=output_shape,
                indices_iterator=indices,
                apply_real_world_transform=apply_real_world_transform,
                real_world_value_map_selector=real_world_value_map_selector,
                apply_modality_transform=apply_modality_transform,
                apply_voi_transform=apply_voi_transform,
                voi_transform_selector=voi_transform_selector,
                voi_output_range=voi_output_range,
                apply_presentation_lut=apply_presentation_lut,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
                dtype=dtype,
            )


def imread(
    fp: str | bytes | PathLike | BinaryIO,
    lazy_frame_retrieval: bool = False
) -> Image:
    """Read an image stored in DICOM File Format.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing an
        image.
    lazy_frame_retrieval: bool
        If True, the returned image will retrieve frames from the file as
        requested, rather than loading in the entire object to memory
        initially. This may be a good idea if file reading is slow and you are
        likely to need only a subset of the frames in the image.

    Returns
    -------
    highdicom.Image:
        Image read from the file.

    """
    # This is essentially a convenience alias for the classmethod (which is
    # used so that it is inherited correctly by subclasses). It is used
    # because it follows the format of other similar functions around the
    # library
    return Image.from_file(fp, lazy_frame_retrieval=lazy_frame_retrieval)


def get_volume_from_series(
    series_datasets: Sequence[Dataset],
    *,
    dtype: type | str | np.dtype = np.float64,
    apply_real_world_transform: bool | None = None,
    real_world_value_map_selector: int | str | Code | CodedConcept = 0,
    apply_modality_transform: bool | None = None,
    apply_voi_transform: bool | None = False,
    voi_transform_selector: int | str | VOILUTTransformation = 0,
    voi_output_range: tuple[float, float] = (0.0, 1.0),
    apply_presentation_lut: bool = True,
    apply_palette_color_lut: bool | None = None,
    apply_icc_profile: bool | None = None,
    atol: float | None = None,
    rtol: float | None = None,
) -> Volume:
    """Create volume from a series of single frame images.

    Parameters
    ----------
    series_datasets: Sequence[pydicom.Dataset]
        Series of single frame datasets. There is no requirement on the
        sorting of the datasets.
    dtype: Union[type, str, numpy.dtype], optional
        Data type of the returned array.
    apply_real_world_transform: bool | None, optional
        Whether to apply a real-world value map to the frame.
        A real-world value maps converts stored pixel values to output
        values with a real-world meaning, either using a LUT or a linear
        slope and intercept.

        If True, the transform is applied if present, and if not
        present an error will be raised. If False, the transform will not
        be applied, regardless of whether it is present. If ``None``, the
        transform will be applied if present but no error will be raised if
        it is not present.

        Note that if the dataset contains both a modality LUT and a real
        world value map, the real world value map will be applied
        preferentially. This also implies that specifying both
        ``apply_real_world_transform`` and ``apply_modality_transform`` to
        True is not permitted.
    real_world_value_map_selector: int | str | pydicom.sr.coding.Code | highdicom.sr.coding.CodedConcept, optional
        Specification of the real world value map to use (multiple may be
        present in the dataset). If an int, it is used to index the list of
        available maps. A negative integer may be used to index from the
        end of the list following standard Python indexing convention. If a
        str, the string will be used to match the ``"LUTLabel"`` attribute
        to select the map. If a ``pydicom.sr.coding.Code`` or
        ``highdicom.sr.coding.CodedConcept``, this will be used to match
        the units (contained in the ``"MeasurementUnitsCodeSequence"``
        attribute).
    apply_modality_transform: bool | None, optional
        Whether to apply the modality transform (if present in the dataset) to
        the frame. The modality transform maps stored pixel values to
        output values, either using a LUT or rescale slope and intercept.

        If True, the transform is applied if present, and if not
        present an error will be raised. If False, the transform will not
        be applied, regardless of whether it is present. If ``None``, the
        transform will be applied if it is present and no real world value
        map takes precedence, but no error will be raised if it is not
        present.
    apply_voi_transform: bool | None, optional
        Apply the value-of-interest (VOI) transform (if present in the
        dataset), which limits the range of pixel values to a particular
        range of interest using either a windowing operation or a LUT.

        If True, the transform is applied if present, and if not
        present an error will be raised. If False, the transform will not
        be applied, regardless of whether it is present. If ``None``, the
        transform will be applied if it is present and no real world value
        map takes precedence, but no error will be raised if it is not
        present.
    voi_transform_selector: int | str | highdicom.VOILUTTransformation, optional
        Specification of the VOI transform to select (multiple may be
        present). May either be an int or a str. If an int, it is
        interpreted as a (zero-based) index of the list of VOI transforms
        to apply. A negative integer may be used to index from the end of
        the list following standard Python indexing convention. If a str,
        the string that will be used to match the
        ``"WindowCenterWidthExplanation"`` or the ``"LUTExplanation"``
        attributes to choose from multiple VOI transforms. Note that such
        explanations are optional according to the standard and therefore
        may not be present. Ignored if ``apply_voi_transform`` is ``False``
        or no VOI transform is included in the datasets.
    voi_output_range: Tuple[float, float], optional
        Range of output values to which the VOI range is mapped. Only
        relevant if ``apply_voi_transform`` is True and a VOI transform is
        present.
    apply_palette_color_lut: bool | None, optional
        Apply the palette color LUT, if present in the dataset. The palette
        color LUT maps a single sample for each pixel stored in the dataset
        to a 3 sample-per-pixel color image.
    apply_presentation_lut: bool, optional
        Apply the presentation LUT transform to invert the pixel values. If
        the PresentationLUTShape is present with the value ``'INVERSE'``,
        or the PresentationLUTShape is not present but the Photometric
        Interpretation is MONOCHROME1, convert the range of the output
        pixels corresponds to MONOCHROME2 (in which high values are
        represent white and low values represent black). Ignored if
        PhotometricInterpretation is not MONOCHROME1 and the
        PresentationLUTShape is not present, or if a real world value
        transform is applied.
    apply_icc_profile: bool | None, optional
        Whether colors should be corrected by applying an ICC
        transform. Will only be performed if metadata contain an
        ICC Profile.

        If True, the transform is applied if present, and if not
        present an error will be raised. If False, the transform will not
        be applied, regardless of whether it is present. If ``None``, the
        transform will be applied if it is present, but no error will be
        raised if it is not present.
    rtol: float | None, optional
        Relative tolerance for determining spacing regularity. If slice
        spacings vary by less that this proportion of the average spacing, they
        are considered to be regular. If neither ``rtol`` or ``atol`` are
        provided, a default relative tolerance of 0.01 is used.
    atol: float | None, optional
        Absolute tolerance for determining spacing regularity. If slice
        spacings vary by less that this value (in mm), they
        are considered to be regular. Incompatible with ``rtol``.

    Returns
    -------
    highdicom.Volume:
        Volume created from the series.

    """  # noqa: E501
    coordinate_system = get_image_coordinate_system(series_datasets[0])
    if (
        coordinate_system is None or
        coordinate_system != CoordinateSystemNames.PATIENT
    ):
        raise ValueError(
            "Dataset should exist in the patient "
            "coordinate_system."
        )

    image_orientation = series_datasets[0].ImageOrientationPatient
    pixel_spacing = series_datasets[0].PixelSpacing

    frame_of_reference_uid = series_datasets[0].FrameOfReferenceUID
    series_instance_uid = series_datasets[0].SeriesInstanceUID
    if not all(
        ds.SeriesInstanceUID == series_instance_uid
        for ds in series_datasets
    ):
        raise ValueError('Images do not belong to the same series.')

    if not all(
        ds.FrameOfReferenceUID == frame_of_reference_uid
        for ds in series_datasets
    ):
        raise ValueError('Images do not share a frame of reference.')

    if not all(
        ds.ImageOrientationPatient == image_orientation
        for ds in series_datasets
    ):
        raise ValueError('Images do not have the same orientation.')

    if not all(
        ds.PixelSpacing == pixel_spacing
        for ds in series_datasets
    ):
        raise ValueError('Images do not have the same spacing.')

    if len(series_datasets) == 1:
        slice_spacing = series_datasets[0].get('SpacingBetweenSlices', 1.0)
        sorted_datasets = series_datasets
    else:
        slice_spacing, vol_positions = get_series_volume_positions(
            series_datasets,
            atol=atol,
            rtol=rtol,
        )
        if slice_spacing is None:
            raise ValueError('Series is not a regularly-spaced volume.')

        sorted_datasets = [
            series_datasets[vol_positions.index(i)]
            for i in range(len(series_datasets))
        ]

    frames = []
    for ds in sorted_datasets:
        frame = ds.pixel_array
        transf = _CombinedPixelTransform(
            ds,
            output_dtype=dtype,
            apply_real_world_transform=apply_real_world_transform,
            real_world_value_map_selector=real_world_value_map_selector,
            apply_modality_transform=apply_modality_transform,
            apply_voi_transform=apply_voi_transform,
            voi_transform_selector=voi_transform_selector,
            voi_output_range=voi_output_range,
            apply_presentation_lut=apply_presentation_lut,
            apply_palette_color_lut=apply_palette_color_lut,
            apply_icc_profile=apply_icc_profile,
        )

        frame = transf(frame)
        frames.append(frame)

    array = np.stack(frames)

    channels = None
    if array.ndim == 4:
        channels = {RGB_COLOR_CHANNEL_DESCRIPTOR: ['R', 'G', 'B']}

    coordinate_system = get_image_coordinate_system(series_datasets[0])

    first_ds = sorted_datasets[0]

    return Volume.from_attributes(
        array=array,
        frame_of_reference_uid=frame_of_reference_uid,
        image_position=first_ds.ImagePositionPatient,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=slice_spacing,
        channels=channels,
        coordinate_system=coordinate_system,
    )
