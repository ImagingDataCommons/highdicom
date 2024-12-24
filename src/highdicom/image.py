"""Tools for working with general DICOM images."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
import logging
from os import readlink
import sqlite3
from typing import (
    Any,
    Iterable,
    Iterator,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
    cast,
)
from typing_extensions import Self

import numpy as np
from pydicom import Dataset
from pydicom.tag import BaseTag
from pydicom.datadict import (
    get_entry,
    tag_for_keyword,
)
from pydicom.multival import MultiValue
from pydicom.uid import ParametricMapStorage

from highdicom._module_utils import is_multiframe_image
from highdicom.base import SOPClass, _check_little_endian
from highdicom.color import ColorManager
from highdicom.content import LUT
from highdicom.enum import (
    CoordinateSystemNames,
)
from highdicom.pixel_transforms import (
    _check_rescale_dtype,
    _get_combined_palette_color_lut,
    apply_lut,
    voi_window,
)
from highdicom.seg.enum import SpatialLocationsPreservedValues
from highdicom.spatial import (
    get_image_coordinate_system,
    get_volume_positions,
)
from highdicom.uid import UID as hd_UID
from highdicom.utils import (
    iter_tiled_full_frame_data,
)
from highdicom.volume import VolumeGeometry


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
_DCM_PYTHON_TYPE_MAP = {
    'CS': str,
    'DS': float,
    'FD': float,
    'FL': float,
    'IS': int,
    'LO': str,
    'LT': str,
    'PN': str,
    'SH': str,
    'SL': int,
    'SS': int,
    'ST': str,
    'UI': str,
    'UL': int,
    'UR': str,
    'US or SS': int,
    'US': int,
    'UT': str,
}
_NO_FRAME_REF_VALUE = -1


logger = logging.getLogger(__name__)


class _ImageColorType(Enum):
    """Internal enum describing color arrangement of an image."""
    MONOCHROME = 'MONOCHROME'
    COLOR = 'COLOR'
    PALETTE_COLOR = 'PALETTE_COLOR'


class _CombinedPixelTransformation:

    """Class representing a combined pixel transformation.

    DICOM images contain multi-stage transformations to apply to the raw stored
    pixel values. This class is intended to provdie a single class that
    configurably and efficiently applies the net effect of the selected
    transforms to stored pixel data.

    Depending on the parameters, it may perform operations related to the
    following:

    For monochrome images:
    * Real world value maps, which map stored pixels to real-world values and
    is independent of all other transforms
    * Modality LUT transformation, which transforms stored pixel values to
    modality-specific values
    * Value-of-interest (VOI) LUT transformation, which transforms the output
    of the Modality LUT transform to output values in order to focus on a
    particular region of intensities values of particular interest (such as a
    windowing operation).
    * Presentation LUT transformation, which inverts the range of values for
    display.

    For pseudo-color images (stored as monochrome images but displayed as color
    images):
    * The Palette Color LUT transformation, which maps stored single-sample
    pixel values to 3-samples-per-pixel RGB color images.

    For color images and pseudo-color images:
    * The ICCProfile, which performs color correction.

    """

    def __init__(
        self,
        image: Dataset,
        frame_index: int = 0,
        *,
        output_dtype: Union[type, str, np.dtype, None] = np.float64,
        apply_real_world_transform: bool | None = None,
        real_world_value_map_index: int = 0,
        apply_modality_transform: bool | None = None,
        apply_voi_transform: bool | None = False,
        voi_transform_index: int = 0,
        voi_output_range: Tuple[float, float] = (0.0, 1.0),
        apply_presentation_lut: bool = True,
        apply_palette_color_lut: bool | None = None,
        apply_icc_profile: bool | None = None,
    ):
        """

        Parameters
        ----------
        image: pydicom.Dataset
            Image (single frame or multiframe) for which the pixel
            transformation should be represented.
        frame_index: int
            Zero-based index (one less than the frame number).
        output_dtype: Union[type, str, np.dtype, None], optional
            Data type of the output array.
        apply_real_world_transform: bool | None, optional
            Whether to apply to apply the real-world value map to the frame.
            The real world value map converts stored pixel values to output
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
        real_world_value_map_index: int, optional
            Index of the real world value map to use (multiple may be stored
            within the dataset).
        apply_modality_transform: bool | None, optional
            Whether to apply to the modality transform (if present in the
            dataset) the frame. The modality transformation maps stored pixel
            values to output values, either using a LUT or rescale slope and
            intercept.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        apply_voi_transform: bool | None, optional
            Apply the value-of-interest (VOI) transformation (if present in the
            dataset), which limits the range of pixel values to a particular
            range of interest, using either a windowing operation or a LUT.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present and no real world value
            map takes precedence, but no error will be raised if it is not
            present.
        voi_transform_index: int, optional
            Index (zero-based) of the VOI transform to apply if multiple are
            included in the datasets. Ignored if ``apply_voi_transform`` is
            ``False`` or no VOI transform is included in the datasets. May be a
            negative integer, following standard Python indexing convention.
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
            the PresentationLUTShape is present with the value ``'INVERSE''``,
            or the PresentationLUTShape is not present but the Photometric
            Interpretation is MONOCHROME1, convert the range of the output
            pixels corresponds to MONOCHROME2 (in which high values are
            represent white and low values represent black). Ignored if
            PhotometricInterpretation is not MONOCHROME1 and the
            PresentationLUTShape is not present, or if a real world value
            transform is applied.
        apply_icc_profile: bool | None, optional
            Whether colors should be corrected by applying an ICC
            transformation. Will only be performed if metadata contain an
            ICC Profile.

            If True, the transform is applied if present, and if not
            present an error will be raised. If False, the transform will not
            be applied, regardless of whether it is present. If ``None``, the
            transform will be applied if it is present, but no error will be
            raised if it is not present.

        """
        # TODO: choose VOI by explanation?
        # TODO: how to combine with multiframe?
        photometric_interpretation = image.PhotometricInterpretation

        if photometric_interpretation in (
            'MONOCHROME1',
            'MONOCHROME2',
        ):
            self._color_type = _ImageColorType.MONOCHROME
        elif photometric_interpretation == 'PALETTE_COLOR':
            self._color_type = _ImageColorType.PALETTE_COLOR
        else:
            self._color_type = _ImageColorType.COLOR

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
        self._input_range_check: Optional[Tuple[int, int]] = None
        self._voi_output_range = voi_output_range
        self._effective_lut_data: Optional[np.ndarray] = None
        self._effective_lut_first_mapped_value = 0
        self._effective_window_center_width: Optional[Tuple[float, float]] = None
        self._effective_slope_intercept: Optional[Tuple[float, float]] = None
        self._invert = False
        self._clip = True

        # Determine input type and range of values
        input_range = None
        if (
            image.SOPClassUID == ParametricMapStorage
            and image.BitsAllocated > 16
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
            if use_palette_color:
                if 'SegmentedRedPaletteColorLookupTableData' in image:
                    # TODO
                    raise RuntimeError("Segmented LUTs are not implemented.")

                self._first_mapped_value, self._effective_lut = (
                    _get_combined_palette_color_lut(image)
                )

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

            modality_lut: Optional[LUT] = None
            modality_slope_intercept: Optional[Tuple[float, float]] = None

            voi_lut: Optional[LUT] = None
            voi_scaled_lut_data: Optional[np.ndarray] = None
            voi_center_width: Optional[Tuple[float, float]] = None
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
                        try:
                            rwvm_item = rwvm_seq[real_world_value_map_index]
                        except IndexError as e:
                            raise IndexError(
                                "Requested 'real_world_value_map_index' is "
                                "not present."
                            ) from e
                        if 'RealWorldValueLUTData' in rwvm_item:
                            self._effective_lut_data = np.array(
                                rwvm_item.RealWorldValueLUTData
                            )
                            self._effective_lut_first_mapped_value = int(
                                rwvm_item.RealWorldValueFirstValueMapped
                            )
                            self._clip = False
                        else:
                            self._effective_slope_intercept = (
                                rwvm_item.RealWorldValueSlope,
                                rwvm_item.RealWorldValueIntercept,
                            )
                            if 'DoubleFloatRealWorldValueFirstValueMapped' in rwvm_item:
                                self._input_range_check = (
                                    rwvm_item.DoubleFloatRealWorldValueFirstValueMapped,
                                    rwvm_item.DoubleFloatRealWorldValueLastValueMapped
                                )
                            else:
                                self._input_range_check = (
                                    rwvm_item.RealWorldValueFirstValueMapped,
                                    rwvm_item.RealWorldValueLastValueMapped
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

                if 'VOILUTSequence' in image:
                    voi_lut = LUT.from_dataset(
                        image.VOILUTSequence[0]
                    )
                    voi_scaled_lut_data = voi_lut.get_scaled_lut_data(
                        output_range=voi_output_range,
                        dtype=output_dtype,
                        invert=invert,
                    )
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
                            voi_center = sub_ds.WindowCenter
                            voi_width = sub_ds.WindowWidth

                            if 'VOILUTFunction' in sub_ds:
                                voi_function = sub_ds.VOILUTFunction

                            if isinstance(voi_width, list):
                                voi_width = voi_width[
                                    voi_transform_index
                                ]
                            elif voi_transform_index not in (0, -1):
                                raise IndexError(
                                    "Requested 'voi_transform_index' is "
                                    "not present."
                                )

                            if isinstance(voi_center, list):
                                voi_center = voi_center[
                                    voi_transform_index
                                ]
                            elif voi_transform_index not in (0, -1):
                                raise IndexError(
                                    "Requested 'voi_transform_index' is "
                                    "not present."
                                )
                            self.applies_to_all_frames = (
                                self.applies_to_all_frames and is_shared
                            )
                            voi_center_width = (voi_center, voi_width)
                            break

            if (
                require_voi and
                voi_center_width is None and
                voi_lut is None
            ):
                raise RuntimeError(
                    'A VOI transform is required but not found in '
                    'the image.'
                )

            # Determine how to combine modality, voi and presentation
            # transforms
            if modality_lut is not None and not has_rwvm:
                if voi_center_width is not None:
                    # Apply the window function to the modality LUT
                    self._effective_lut_data = voi_window(
                        array=modality_lut.lut_data,
                        window_center=voi_center_width[0],
                        window_width=voi_center_width[1],
                        output_range=voi_output_range,
                        dtype=output_dtype,
                        invert=invert,
                    )
                    self._effective_lut_first_mapped_value = (
                        modality_lut.first_mapped_value
                    )

                elif voi_lut is not None and voi_scaled_lut_data is not None:
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

                elif voi_lut is not None and voi_scaled_lut_data is not None:
                    # Shift and "scale" the LUT to account for the rescale
                    if not intercept.is_integer() and slope.is_integer():
                        raise ValueError(
                            "Cannot apply a VOI LUT when rescale intercept "
                            "or slope have non-integer values."
                        )
                    intercept = int(intercept)
                    slope = int(slope)
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

        if self._effective_lut_data is not None:
            if self._effective_lut_data.dtype != output_dtype:
                self._effective_lut_data = (
                    self._effective_lut_data.astype(output_dtype)
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
                    'The VOI transformation requires a floating point data '
                    'type.'
                )

        # We don't use the color_correct_frame() function here, since we cache
        # the ICC transform on the instance for improved performance.
        if use_icc and 'ICCProfile' in image:
            self._color_manager = ColorManager(image.ICCProfile)
        else:
            self._color_manager = None
            if require_icc:
                raise RuntimeError(
                    'An ICC profile is required but not found in '
                    'the image.'
                )

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply the composed loss.

        Parameters
        ----------
        frame: numpy.ndarray
            Input frame for the transformation.

        Returns
        -------
        numpy.ndarray:
            Output frame after the transformation is applied.

        """
        if self._color_type == _ImageColorType.COLOR:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    "Expected an image of shape (R, C, 3)."
                )

        else:
            if frame.ndim != 2:
                raise ValueError(
                    "Expected an image of shape (R, C)."
                )

        if self._input_range_check is not None:
            first, last = self._input_range_check
            if frame.min() < first or frame.max() > last:
                raise ValueError(
                    'Array contains value outside the valid range.'
                )

        if self._effective_lut_data is not None:
            frame = apply_lut(
                frame,
                self._effective_lut_data,
                self._effective_lut_first_mapped_value,
                clip=self._clip,
            )

        elif self._effective_slope_intercept is not None:
            slope, intercept = self._effective_slope_intercept

            # Avoid unnecessary array operations for efficiency
            if slope != 1.0:
                frame = frame * slope
            if intercept != 0.0:
                frame = frame + intercept

        elif self._effective_window_center_width is not None:
            frame = voi_window(
                frame,
                window_center=self._effective_window_center_width[0],
                window_width=self._effective_window_center_width[1],
                dtype=self.output_dtype,
                invert=self._invert,
                output_range=self._voi_output_range,
            )

        if self._color_manager is not None:
            return self._color_manager.transform_frame(frame)

        if frame.dtype != self.output_dtype:
            frame = frame.astype(self.output_dtype)

        return frame


class MultiFrameImage(SOPClass):

    """Database manager for frame information in a multiframe image."""

    _coordinate_system: CoordinateSystemNames
    _is_tiled_full: bool
    _single_source_frame_per_frame: bool
    _dim_ind_pointers: List[BaseTag]
    # Mapping of tag value to (index column name, val column name(s))
    _dim_ind_col_names: Dict[int, Tuple[str, Union[str, Tuple[str, ...], None]]]
    _locations_preserved: Optional[SpatialLocationsPreservedValues]
    _db_con: sqlite3.Connection
    _volume_geometry: Optional[VolumeGeometry]

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Create a MultiFrameImage from an existing pydicom Dataset.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset of a multi-frame image.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        _check_little_endian(dataset)

        # Checks on integrity of input dataset
        if not is_multiframe_image(dataset):
            raise ValueError('Dataset is not a multiframe image.')
        if copy:
            im = deepcopy(dataset)
        else:
            im = dataset
        im.__class__ = cls
        im = cast(cls, im)

        im._build_luts()
        return im

    def __getstate__(self) -> Dict[str, Any]:
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

    def __setstate__(self, state: Dict[str, Any]) -> None:
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
        self._coordinate_system = get_image_coordinate_system(
            self
        )
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

        # We may want to gather additional information that is not one of the
        # indices
        extra_collection_pointers = []
        extra_collection_func_pointers = {}
        slice_spacing_hint = None
        image_position_tag = tag_for_keyword('ImagePositionPatient')
        shared_pixel_spacing: Optional[List[float]] = None
        if self._coordinate_system == CoordinateSystemNames.PATIENT:
            plane_pos_seq_tag = tag_for_keyword('PlanePositionSequence')
            # Include the image position if it is not an index
            if image_position_tag not in self._dim_ind_pointers:
                extra_collection_pointers.append(image_position_tag)
                extra_collection_func_pointers[
                    image_position_tag
                ] = plane_pos_seq_tag

            if hasattr(self, 'SharedFunctionalGroupsSequence'):
                sfgs = self.SharedFunctionalGroupsSequence[0]
                if hasattr(sfgs, 'PixelMeasuresSequence'):
                    measures = sfgs.PixelMeasuresSequence[0]
                    slice_spacing_hint = measures.get('SpacingBetweenSlices')
                    shared_pixel_spacing = measures.get('PixelSpacing')
            if slice_spacing_hint is None or shared_pixel_spacing is None:
                # Get the orientation of the first frame, and in the later loop
                # check whether it is shared.
                if hasattr(self, 'PerFrameFunctionalGroupsSequence'):
                    pfg1 = self.PerFrameFunctionalGroupsSequence[0]
                    if hasattr(pfg1, 'PixelMeasuresSequence'):
                        measures = pfg1.PixelMeasuresSequence[0]
                        slice_spacing_hint = measures.get(
                            'SpacingBetweenSlices'
                        )
                        shared_pixel_spacing = measures.get('PixelSpacing')

        dim_indices: Dict[int, List[int]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }
        dim_values: Dict[int, List[Any]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }

        extra_collection_values: Dict[int, List[Any]] = {
            ptr: [] for ptr in extra_collection_pointers
        }

        # Get the shared orientation
        shared_image_orientation: Optional[List[float]] = None
        if hasattr(self, 'ImageOrientationSlide'):
            shared_image_orientation = self.ImageOrientationSlide
        if hasattr(self, 'SharedFunctionalGroupsSequence'):
            sfgs = self.SharedFunctionalGroupsSequence[0]
            if hasattr(sfgs, 'PlaneOrientationSequence'):
                shared_image_orientation = (
                    sfgs.PlaneOrientationSequence[0].ImageOrientationPatient
                )
        if shared_image_orientation is None:
            # Get the orientation of the first frame, and in the later loop
            # check whether it is shared.
            if hasattr(self, 'PerFrameFunctionalGroupsSequence'):
                pfg1 = self.PerFrameFunctionalGroupsSequence[0]
                if hasattr(pfg1, 'PlaneOrientationSequence'):
                    shared_image_orientation = (
                        pfg1
                        .PlaneOrientationSequence[0]
                        .ImageOrientationPatient
                    )

        self._single_source_frame_per_frame = True

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
            referenced_instances: Optional[List[str]] = []
            referenced_frames: Optional[List[int]] = []

            # Create a list of source images and check for spatial locations
            # preserved
            locations_list_type = List[
                Optional[SpatialLocationsPreservedValues]
            ]
            locations_preserved: locations_list_type = []

            for frame_item in self.PerFrameFunctionalGroupsSequence:
                # Get dimension indices for this frame
                if 'FrameContentSequence' in frame_item:
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
                            frame_source_frames.append(_NO_FRAME_REF_VALUE)

                if (
                    len(set(frame_source_instances)) != 1 or
                    len(set(frame_source_frames)) != 1
                ):
                    self._single_source_frame_per_frame = False
                else:
                    ref_instance_uid = frame_source_instances[0]
                    if ref_instance_uid not in all_referenced_sops:
                        raise AttributeError(
                            f'SOP instance {ref_instance_uid} referenced in '
                            'the source image sequence is not included in the '
                            'Referenced Series Sequence or Studies Containing '
                            'Other Referenced Instances Sequence. This is an '
                            'error with the integrity of the '
                            'object.'
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
        self._col_types = {}  # dictionary from column name to SQL type

        # Frame number column
        col_defs.append('FrameNumber INTEGER PRIMARY KEY')
        self._col_types['FrameNumber'] = 'INTEGER'
        col_data.append(list(range(1, self.NumberOfFrames + 1)))

        self._dim_ind_col_names = {}
        for i, t in enumerate(dim_indices.keys()):
            vr, vm_str, _, _, kw = get_entry(t)
            if kw == '':
                kw = f'UnknownDimensionIndex{i}'
            ind_col_name = kw + '_DimensionIndexValues'

            # Add column for dimension index
            col_defs.append(f'{ind_col_name} INTEGER NOT NULL')
            self._col_types[ind_col_name] = 'INTEGER'
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
                    self._col_types[col_name] = sql_type
                    col_data.append(data)
                    val_col_names.append(col_name)

                self._dim_ind_col_names[t] = (ind_col_name, tuple(val_col_names))
            else:
                # Single column
                col_defs.append(f'{kw} {sql_type} NOT NULL')
                self._col_types[kw] = sql_type
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
                    self._col_types[col_name] = sql_type
                    col_data.append(data)
            else:
                # Single column
                col_defs.append(f'{kw} {sql_type} NOT NULL')
                self._col_types[kw] = sql_type
                col_data.append(dim_values[t])

        # Volume related information
        self._volume_geometry = None
        if (
            self._coordinate_system == CoordinateSystemNames.PATIENT
            and shared_image_orientation is not None
        ):
            if shared_image_orientation is not None:
                if image_position_tag in self._dim_ind_pointers:
                    image_positions = dim_values[image_position_tag]
                else:
                    image_positions = extra_collection_values[
                        image_position_tag
                    ]
                volume_spacing, volume_positions = get_volume_positions(
                    image_positions=image_positions,
                    image_orientation=shared_image_orientation,
                    allow_missing=True,
                    allow_duplicates=True,
                    spacing_hint=slice_spacing_hint,
                )
                if volume_positions is not None:
                    origin_slice_index = volume_positions.index(0)
                    number_of_slices = max(volume_positions) + 1
                    self._volume_geometry = VolumeGeometry.from_attributes(
                        image_position=image_positions[origin_slice_index],
                        image_orientation=shared_image_orientation,
                        rows=self.Rows,
                        columns=self.Columns,
                        pixel_spacing=shared_pixel_spacing,
                        number_of_frames=number_of_slices,
                        spacing_between_slices=volume_spacing,
                    )
                    col_defs.append('VolumePosition INTEGER NOT NULL')
                    self._col_types['VolumePosition'] = 'INTEGER'
                    col_data.append(volume_positions)

        # Columns related to source frames, if they are usable for indexing
        if (referenced_frames is None) != (referenced_instances is None):
            raise TypeError(
                "'referenced_frames' and 'referenced_instances' should be "
                "provided together or not at all."
            )
        if referenced_instances is not None:
            col_defs.append('ReferencedFrameNumber INTEGER')
            self._col_types['ReferencedFrameNumber'] = 'INTEGER'
            col_defs.append('ReferencedSOPInstanceUID VARCHAR NOT NULL')
            self._col_types['ReferencedSOPInstanceUID'] = 'VARCHAR'
            col_defs.append(
                'FOREIGN KEY(ReferencedSOPInstanceUID) '
                'REFERENCES InstanceUIDs(SOPInstanceUID)'
            )
            col_data += [
                referenced_frames,
                referenced_instances,
            ]

        # Build LUT from columns
        all_defs = ", ".join(col_defs)
        cmd = f'CREATE TABLE FrameLUT({all_defs})'
        placeholders = ', '.join(['?'] * len(col_data))
        with self._db_con:
            self._db_con.execute(cmd)
            self._db_con.executemany(
                f'INSERT INTO FrameLUT VALUES({placeholders})',
                zip(*col_data),
            )

    def _get_ref_instance_uids(self) -> List[Tuple[str, str, str]]:
        """List all instances referenced in the image.

        Returns
        -------
        List[Tuple[str, str, str]]
            List of all instances referenced in the image in the format
            (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID).

        """
        instance_data = []
        if hasattr(self, 'ReferencedSeriesSequence'):
            for ref_series in self.ReferencedSeriesSequence:
                for ref_ins in ref_series.ReferencedInstanceSequence:
                    instance_data.append(
                        (
                            self.StudyInstanceUID,
                            ref_series.SeriesInstanceUID,
                            ref_ins.ReferencedSOPInstanceUID
                        )
                    )
        other_studies_kw = 'StudiesContainingOtherReferencedInstancesSequence'
        if hasattr(self, other_studies_kw):
            for ref_study in getattr(self, other_studies_kw):
                for ref_series in ref_study.ReferencedSeriesSequence:
                    for ref_ins in ref_series.ReferencedInstanceSequence:
                        instance_data.append(
                            (
                                ref_study.StudyInstanceUID,
                                ref_series.SeriesInstanceUID,
                                ref_ins.ReferencedSOPInstanceUID,
                            )
                        )

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
    def dimension_index_pointers(self) -> List[BaseTag]:
        """List[pydicom.tag.BaseTag]:
            List of tags used as dimension indices.
        """
        return [BaseTag(t) for t in self._dim_ind_pointers]

    def _create_ref_instance_table(
        self,
        referenced_uids: List[Tuple[str, str, str]],
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

    def _are_columns_unique(
        self,
        column_names: Sequence[str],
    ) -> bool:
        """Check if a list of columns uniquely identifies frames.

        For a given list of columns, check whether every combination of values
        for these column identifies a unique image frame. This is a
        pre-requisite for indexing frames using this list of columns.

        Parameters
        ----------
        column_names: Sequence[str]
            Column names.

        Returns
        -------
        bool
            True if combination of columns is sufficient to identify unique
            frames.

        """
        col_str = ", ".join(column_names)
        cur = self._db_con.cursor()
        n_unique_combos = cur.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM FrameLUT GROUP BY {col_str})"
        ).fetchone()[0]
        return n_unique_combos == self.NumberOfFrames

    def are_dimension_indices_unique(
        self,
        dimension_index_pointers: Sequence[Union[int, BaseTag]],
    ) -> bool:
        """Check if a list of index pointers uniquely identifies frames.

        For a given list of dimension index pointers, check whether every
        combination of index values for these pointers identifies a unique
        image frame. This is a pre-requisite for indexing using this list of
        dimension index pointers.

        Parameters
        ----------
        dimension_index_pointers: Sequence[Union[int, pydicom.tag.BaseTag]]
            Sequence of tags serving as dimension index pointers.

        Returns
        -------
        bool
            True if dimension indices are unique.

        """
        column_names = []
        for ptr in dimension_index_pointers:
            column_names.append(self._dim_ind_col_names[ptr][0])
        return self._are_columns_unique(column_names)

    def get_source_image_uids(self) -> List[Tuple[hd_UID, hd_UID, hd_UID]]:
        """Get UIDs of source image instances referenced in the image.

        Returns
        -------
        List[Tuple[highdicom.UID, highdicom.UID, highdicom.UID]]
            (Study Instance UID, Series Instance UID, SOP Instance UID) triplet
            for every image instance referenced in the image.

        """
        cur = self._db_con.cursor()
        res = cur.execute(
            'SELECT StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID '
            'FROM InstanceUIDs'
        )

        return [
            (hd_UID(a), hd_UID(b), hd_UID(c)) for a, b, c in res.fetchall()
        ]

    def _get_unique_referenced_sop_instance_uids(self) -> Set[str]:
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
    ) -> Set[Tuple[int, ...]]:
        """Get set of unique dimension index value combinations.

        Parameters
        ----------
        dimension_index_pointers: Sequence[int]
            List of dimension index pointers for which to find unique
            combinations of values.

        Returns
        -------
        Set[Tuple[int, ...]]
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

    @property
    def volume_geometry(self) -> Optional[VolumeGeometry]:
        """Union[highdicom.VolumeGeometry, None]: Geometry of the volume if the
        image represents a regularly-spaced 3D volume. ``None``
        otherwise.

        """
        return self._volume_geometry

    @contextmanager
    def _generate_temp_table(
        self,
        table_name: str,
        column_defs: Sequence[str],
        column_data: Iterable[Sequence[Any]],
    ) -> Generator[None, None, None]:
        """Context manager that handles a temporary table.

        The temporary table is created with the specified information. Control
        flow then returns to code within the "with" block. After the "with"
        block has completed, the cleanup of the table is automatically handled.

        Parameters
        ----------
        table_name: str
            Name of the temporary table.
        column_defs: Sequence[str]
            SQL syntax strings defining each column in the temporary table, one
            string per column.
        column_data: Iterable[Sequence[Any]]
            Column data to place into the table.

        Yields
        ------
        None:
            Yields control to the "with" block, with the temporary table
            created.

        """
        defs_str = ', '.join(column_defs)
        create_cmd = (f'CREATE TABLE {table_name}({defs_str})')
        placeholders = ', '.join(['?'] * len(column_defs))

        with self._db_con:
            self._db_con.execute(create_cmd)
            self._db_con.executemany(
                f'INSERT INTO {table_name} VALUES({placeholders})',
                column_data
            )

        # Return control flow to "with" block
        yield

        # Clean up the table
        cmd = (f'DROP TABLE {table_name}')
        with self._db_con:
            self._db_con.execute(cmd)

    def _get_pixels_by_frame(
        self,
        output_shape: Union[int, Tuple[int, int]],
        indices_iterator: Iterator[
            Tuple[
                Tuple[Union[slice, int], ...],
                Tuple[Union[slice, int], ...],
                int
            ]
        ],
        num_channels: int = 0,
        dtype: Union[type, str, np.dtype, None] = None,
    ) -> np.ndarray:
        """Construct a pixel array given an array of frame numbers.

        The output array is either 4D (``num_channels=0``) or 3D
        (``num_channels>0``), where dimensions are frames x rows x columns x
        channels.

        Parameters
        ----------
        output_shape: Union[int, Tuple[int, int]]
            Shape of the output array. If an integer, this is the number of
            frames in the output array and the number of rows and columns are
            taken to match those of each frame. If a tuple of integers, it
            contains the number of (rows, columns) in the output array and
            there is no frame dimension (this is the tiled case). Note in
            either case, the channels dimension (if relevant) is omitted.
        indices_iterator: Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int ]]
            An iterable object that yields tuples of (output_indexer,
            frame_indexer, channel_number) that describes how to construct the
            desired output pixel array from the multiframe image's pixel array.
            'output_indexer' is a tuple that may be used directly to index the
            output array to place a single frame's pixels into the output
            array. Similarly 'frame_indexer' is a tuple that may be used
            directly to index the image's pixel array to retrieve the pixels to
            place into the output array. with channel number 'channel_number'.
            The channel indexer may be ``None`` if the output array has no
            channels. Note that in both cases the indexers access the frame,
            row and column dimensions of the relevant array, but not the
            channel dimension (if relevant).
        num_channels: int
            Number of channels in the output array. The use of channels depends
            on image type, for example it may be segments in a segmentation,
            optical paths in a microscopy image, or B-values in an MRI.
        dtype: Union[type, str, np.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.

        Returns
        -------
        pixel_array: np.ndarray
            Segmentation pixel array

        """  # noqa: E501
        # TODO multiple samples per pixel
        if dtype is None:
            if self.BitsAllocated == 1:
                dtype = np.uint8
            else:
                if hasattr(self, 'FloatPixelData'):
                    dtype = np.float32
                elif hasattr(self, 'DoubleFloatPixelData'):
                    dtype = np.float64
                else:
                    dtype = np.dtype(f"uint{self.BitsAllocated}")
        dtype = np.dtype(dtype)

        # Check dtype is suitable
        if dtype.kind not in ('u', 'i', 'f', 'b'):
            raise ValueError(
                f'Data type "{dtype}" is not suitable.'
            )

        if self.pixel_array.ndim == 2:
            h, w = self.pixel_array.shape
        else:
            _, h, w = self.pixel_array.shape

        # Initialize empty pixel array
        spatial_shape = (
            output_shape
            if isinstance(output_shape, tuple)
            else (output_shape, h, w)
        )
        if num_channels > 0:
            full_output_shape = (*spatial_shape, num_channels)
        else:
            full_output_shape = spatial_shape

        out_array = np.zeros(
            full_output_shape,
            dtype=dtype
        )

        # loop through output frames
        for (output_indexer, input_indexer, channel) in indices_iterator:

            # Output indexer needs segment index
            if channel is not None:
                output_indexer = (*output_indexer, channel)

            # Copy data to to output array
            if self.pixel_array.ndim == 2:
                # Special case vith a single frame
                out_array[output_indexer] = self.pixel_array[input_indexer[1:]]
            else:
                out_array[output_indexer] = self.pixel_array[input_indexer]

        return out_array

    def _normalize_dimension_queries(
        self,
        queries: Dict[Union[int, str], Any],
        use_indices: bool,
        multiple_values: bool,
    ) -> Dict[str, Any]:
        normalized_queries: Dict[str, Any] = {}
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
                if p == 'VolumePosition':
                    col_name = 'VolumePosition'
                    python_type = int
                elif p == 'ReferencedSOPInstanceUID':
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

            if multiple_values:
                if len(value) != n_values:
                    raise ValueError(
                        f'Number of values along all dimensions must match.'
                    )
                for v in value:
                    if not isinstance(v, python_type):
                        raise TypeError(
                            f'For dimension {p}, expected all values to be of type '
                            f'{python_type}.'
                        )
            else:
                if not isinstance(value, python_type):
                    raise TypeError(
                        f'For dimension {p}, expected value to be of type '
                        f'{python_type}.'
                    )

            if col_name in normalized_queries:
                raise ValueError(
                    'All dimensions must be unique.'
                )
            normalized_queries[col_name] = value

        return normalized_queries


    @contextmanager
    def _iterate_indices_for_stack(
        self,
        stack_indices: Dict[Union[int, str], Sequence[Any]],
        stack_dimension_use_indices: bool = False,
        channel_indices: Optional[Dict[Union[int, str], Sequence[Any]]] = None,
        channel_dimension_use_indices: bool = False,
        remap_channel_indices: Optional[Sequence[int]] = None,
        filters: Optional[Dict[Union[int, str], Any]] = None,
        filters_use_indices: bool = False,
    ) -> Generator[
            Iterator[
                Tuple[
                    Tuple[Union[slice, int], ...],
                    Tuple[Union[slice, int], ...],
                    Optional[int],
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
            the special values 'VolumePosition', 'ReferencedSOPInstanceUID',
            and 'ReferencedFrameNumber'. The values of the dictionary give
            sequences of values of corresponding dimension that define each
            slice of the output array. Note that multiple dimensions may be
            used, in which case a frame must match the values of all provided
            dimensions to be placed in the output array.
        stack_dimension_use_indices: bool, optional
            If True, the values in ``stack_indices`` are integer-valued
            dimension *index* values. If False the dimension values themselves
            are used, whose type depends on the choice of dimension.
        channel_indices: Union[Dict[Union[int, str], Sequence[Any]], None], optional
            Dictionary defining the channel dimension at axis 3 of the output
            array, if any. Definition is identical to that of
            ``stack_indices``, however the dimensions used must be distinct.
        channel_dimension_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the channel axis.
        remap_channel_indices: Union[Sequence[int], None], optional
            Use these values to remap the channel indices returned in the
            output iterator. Index ``i`` is mapped to
            ``remap_channel_indices[i]``. Ignored if ``channel_indices`` is
            ``None``. If ``None`` no mapping is performed.
        filters: Union[Dict[Union[int, str], Any], None], optional
            Additional filters to use to limit frames. Definition is similar to
            ``stack_indices`` except that the dictionary's values are single
            values rather than lists.
        filters_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the filters.

        Yields
        ------
        Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int]]:
            Indices required to construct the requested mask. Each triplet
            denotes the (output indexer, input indexer, output channel number)
            representing a list of "instructions" to create the requested
            output array by copying frames from the image dataset and inserting
            them into the output array. Output indexer and input indexer are
            tuples that can be used to index the output array and image numpy
            arrays directly. Output channel number will be `None`` if
            ``channel_indices`` is ``None``.

        """
        norm_stack_indices = self._normalize_dimension_queries(
            stack_indices,
            stack_dimension_use_indices,
            True,
        )
        all_columns = list(norm_stack_indices.keys())

        if channel_indices is not None:
            norm_channel_indices = self._normalize_dimension_queries(
                channel_indices,
                channel_dimension_use_indices,
                True,
            )
            all_columns.extend(list(norm_channel_indices.keys()))
        else:
            norm_channel_indices = None

        if filters is not None:
            norm_filters = self._normalize_dimension_queries(
                filters,
                filters_use_indices,
                False,
            )
            all_columns.extend(list(norm_filters.keys()))
        else:
            norm_filters = None

        all_dimensions = [
            c.replace('_DimensionIndexValues', '')
            for c in all_columns
        ]
        if len(set(all_dimensions)) != len(all_dimensions):
            raise ValueError(
                'Dimensions used for stack, channel, and filter must all be '
                'distinct.'
            )

        # Check for uniqueness
        if not self._are_columns_unique(all_columns):
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
                f'{c} {self._col_types[c]} NOT NULL'
                for c in norm_stack_indices.keys()
            ]
        )
        stack_column_data = (
            (i, *row)
            for i, row in enumerate(zip(*norm_stack_indices.values()))
        )
        stack_join_str = ' AND '.join(
            f'F.{col} = L.{col}' for col in norm_stack_indices.keys()
        )

        # Filters
        if norm_filters is not None:
            filter_comparisons = []
            for c, v in norm_filters.items():
                if isinstance(v, str):
                    v = f"'{v}'"
                filter_comparisons.append(f'L.{c} = {v}')
            filter_str = 'WHERE ' + ' AND '.join(filter_comparisons)
        else:
            filter_str = ''

        if norm_channel_indices is None:

            # Construct the query. The ORDER BY is not logically necessary but
            # seems to improve performance of the downstream numpy operations,
            # presumably as it is more cache efficient
            query = (
                'SELECT '
                '    F.OutputFrameIndex,'  # frame index of the output array
                '    L.FrameNumber - 1 '  # frame *index* of segmentation image
                f'FROM {stack_table_name} F '
                'INNER JOIN FrameLUT L'
                f'   ON {stack_join_str} '
                f'{filter_str} '
                'ORDER BY F.OutputFrameIndex'
            )

            with self._generate_temp_table(
                table_name=stack_table_name,
                column_defs=stack_column_defs,
                column_data=stack_column_data,
            ):
                yield (
                    (
                        (fo, slice(None), slice(None)),
                        (fi, slice(None), slice(None)),
                        None
                    )
                    for (fo, fi) in self._db_con.execute(query)
                )
        else:
            # Create temporary table of channel indices
            channel_table_name = 'TemporaryChannelTable'

            channel_column_defs = (
                ['OutputChannelIndex INTEGER UNIQUE NOT NULL'] +
                [
                    f'{c} {self._col_types[c]} NOT NULL'
                    for c in norm_channel_indices.keys()
                ]
            )

            num_channels = len(list(norm_channel_indices.values())[0])
            if remap_channel_indices is not None:
                output_channel_indices = remap_channel_indices
            else:
                output_channel_indices = range(num_channels)

            channel_column_data = zip(
                output_channel_indices,
                *norm_channel_indices.values()
            )
            channel_join_str = ' AND '.join(
                f'L.{col} = C.{col}' for col in norm_channel_indices.keys()
            )

            # Construct the query. The ORDER BY is not logically necessary but
            # seems to improve performance of the downstream numpy operations,
            # presumably as it is more cache efficient
            query = (
                'SELECT '
                '    F.OutputFrameIndex,'  # frame index of the output array
                '    L.FrameNumber - 1,'  # frame *index* of segmentation image
                '    C.OutputChannelIndex '  # channel index of the output array
                f'FROM {stack_table_name} F '
                'INNER JOIN FrameLUT L'
                f'   ON {stack_join_str} '
                f'INNER JOIN {channel_table_name} C'
                f'   ON {channel_join_str} '
                f'{filter_str} '
                'ORDER BY F.OutputFrameIndex'
            )

            with self._generate_temp_table(
                table_name=stack_table_name,
                column_defs=stack_column_defs,
                column_data=stack_column_data,
            ):
                with self._generate_temp_table(
                    table_name=channel_table_name,
                    column_defs=channel_column_defs,
                    column_data=channel_column_data,
                ):
                    yield (
                        (
                            (fo, slice(None), slice(None)),
                            (fi, slice(None), slice(None)),
                            channel
                        )
                        for (fo, fi, channel) in self._db_con.execute(query)
                    )

    @contextmanager
    def _iterate_indices_for_tiled_region(
        self,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
        tile_shape: Tuple[int, int],
        channel_indices: Optional[Dict[Union[int, str], Sequence[Any]]] = None,
        channel_dimension_use_indices: bool = False,
        remap_channel_indices: Optional[Sequence[int]] = None,
        filters: Optional[Dict[Union[int, str], Any]] = None,
        filters_use_indices: bool = False,
    ) -> Generator[
            Iterator[
                Tuple[
                    Tuple[Union[slice, int], ...],
                    Tuple[Union[slice, int], ...],
                    Optional[int],
                ]
            ],
            None,
            None,
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
        row_start: int
            Row index (1-based) in the total pixel matrix of the first row of
            the output array. May be negative (last row is -1).
        row_end: int
            Row index (1-based) in the total pixel matrix one beyond the last
            row of the output array. May be negative (last row is -1).
        column_start: int
            Column index (1-based) in the total pixel matrix of the first
            column of the output array. May be negative (last column is -1).
        column_end: int
            Column index (1-based) in the total pixel matrix one beyond the last
            column of the output array. May be negative (last column is -1).
        tile_shape: Tuple[int, int]
            Shape of each tile (rows, columns).
        channel_indices: Union[Dict[Union[int, str], Sequence[Any]], None], optional
            Dictionary defining the channel dimension at axis 2 of the output
            array, if any. Definition is identical to that of
            ``stack_indices``, however the dimensions used must be distinct.
        channel_dimension_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the channel axis.
        remap_channel_indices: Union[Sequence[int], None], optional
            Use these values to remap the channel indices returned in the
            output iterator. Index ``i`` is mapped to
            ``remap_channel_indices[i]``. Ignored if ``channel_indices`` is
            ``None``. If ``None`` no mapping is performed.
        filters: Union[Dict[Union[int, str], Any], None], optional
            Additional filters to use to limit frames. Definition is similar to
            ``stack_indices`` except that the dictionary's values are single
            values rather than lists.
        filters_use_indices: bool, optional
            As ``stack_dimension_use_indices`` but for the filters.

        Yields
        ------
        Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int]]:
            Indices required to construct the requested mask. Each triplet
            denotes the (output indexer, input indexer, output channel number)
            representing a list of "instructions" to create the requested
            output array by copying frames from the image dataset and inserting
            them into the output array. Output indexer and input indexer are
            tuples that can be used to index the output array and image numpy
            arrays directly. Output channel number will be `None`` if
            ``channel_indices`` is ``None``.

        """  # noqa: E501
        all_columns = [
            'RowPositionInTotalImagePixelMatrix',
            'ColumnPositionInTotalImagePixelMatrix',
        ]
        if channel_indices is not None:
            norm_channel_indices = self._normalize_dimension_queries(
                channel_indices,
                channel_dimension_use_indices,
                True,
            )
            all_columns.extend(list(norm_channel_indices.keys()))
        else:
            norm_channel_indices = None

        if filters is not None:
            norm_filters = self._normalize_dimension_queries(
                filters,
                filters_use_indices,
                False,
            )
            all_columns.extend(list(norm_filters.keys()))
        else:
            norm_filters = None

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
        if not self._are_columns_unique(all_columns):
            raise RuntimeError(
                'The chosen dimensions do not uniquely identify frames of'
                'the image. You may need to provide further dimensions or '
                'a filter to disambiguate.'
            )

        # Filters
        if norm_filters is not None:
            filter_comparisons = []
            for c, v in norm_filters:
                if isinstance(v, str):
                    v = f"'{v}'"
                filter_comparisons.append(f'L.{c} = {v}')
            filter_str = ' AND ' + ' AND '.join(filter_comparisons)
        else:
            filter_str = ''

        th, tw = tile_shape

        oh = row_end - row_start
        ow = column_end - column_start

        row_offset_start = row_start - th + 1
        column_offset_start = column_start - tw + 1

        # Construct the query The ORDER BY is not logically necessary
        # but seems to improve performance of the downstream numpy
        # operations, presumably as it is more cache efficient
        if norm_channel_indices is None:
            query = (
                'SELECT '
                '    L.RowPositionInTotalImagePixelMatrix,'
                '    L.ColumnPositionInTotalImagePixelMatrix,'
                '    L.FrameNumber - 1 '
                'FROM FrameLUT L '
                'WHERE ('
                '    L.RowPositionInTotalImagePixelMatrix >= '
                f'        {row_offset_start}'
                f'    AND L.RowPositionInTotalImagePixelMatrix < {row_end}'
                '    AND L.ColumnPositionInTotalImagePixelMatrix >= '
                f'        {column_offset_start}'
                f'    AND L.ColumnPositionInTotalImagePixelMatrix < {column_end}'
                f'    {filter_str} '
                ')'
                'ORDER BY '
                '     L.RowPositionInTotalImagePixelMatrix,'
                '     L.ColumnPositionInTotalImagePixelMatrix'
            )

            yield (
                (
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
                    (
                        fi,
                        slice(
                            max(row_start - rp, 0),
                            min(row_end - rp, th)
                        ),
                        slice(
                            max(column_start - cp, 0),
                            min(column_end - cp, tw)
                        ),
                    ),
                    None,
                )
                for (rp, cp, fi) in self._db_con.execute(query)
            )

        else:
            # Create temporary table of channel indices
            channel_table_name = 'TemporaryChannelTable'

            channel_column_defs = (
                ['OutputChannelIndex INTEGER UNIQUE NOT NULL'] +
                [
                    f'{c} {self._col_types[c]} NOT NULL'
                    for c in norm_channel_indices.keys()
                ]
            )

            num_channels = len(list(norm_channel_indices.values())[0])
            if remap_channel_indices is not None:
                output_channel_indices = remap_channel_indices
            else:
                output_channel_indices = range(num_channels)

            channel_column_data = zip(
                output_channel_indices,
                *norm_channel_indices.values()
            )
            channel_join_str = ' AND '.join(
                f'L.{col} = C.{col}' for col in norm_channel_indices.keys()
            )

            query = (
                'SELECT '
                '    L.RowPositionInTotalImagePixelMatrix,'
                '    L.ColumnPositionInTotalImagePixelMatrix,'
                '    L.FrameNumber - 1,'
                '    C.OutputChannelIndex '
                'FROM FrameLUT L '
                f'INNER JOIN {channel_table_name} C'
                f'   ON {channel_join_str} '
                'WHERE ('
                '    L.RowPositionInTotalImagePixelMatrix >= '
                f'        {row_offset_start}'
                f'    AND L.RowPositionInTotalImagePixelMatrix < {row_end}'
                '    AND L.ColumnPositionInTotalImagePixelMatrix >= '
                f'        {column_offset_start}'
                f'    AND L.ColumnPositionInTotalImagePixelMatrix < {column_end}'
                f'    {filter_str} '
                ')'
                'ORDER BY '
                '     L.RowPositionInTotalImagePixelMatrix,'
                '     L.ColumnPositionInTotalImagePixelMatrix'
            )

            with self._generate_temp_table(
                table_name=channel_table_name,
                column_defs=channel_column_defs,
                column_data=channel_column_data,
            ):
                yield (
                    (
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
                        (
                            fi,
                            slice(
                                max(row_start - rp, 0),
                                min(row_end - rp, th)
                            ),
                            slice(
                                max(column_start - cp, 0),
                                min(column_end - cp, tw)
                            ),
                        ),
                        channel
                    )
                    for (rp, cp, fi, channel) in self._db_con.execute(query)
                )
