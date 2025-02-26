"""Module for SOP classes of the SEG modality."""
import logging
from collections import defaultdict
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from copy import deepcopy
from itertools import chain
from os import PathLike
import pkgutil
from typing import (
    Any,
    BinaryIO,
    cast,
)
from collections.abc import Iterator, Sequence
from typing_extensions import Self
import warnings

import numpy as np
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.encaps import encapsulate
from pydicom.pixels.utils import pack_bits
from pydicom.tag import BaseTag, Tag
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
    RLELossless,
    UID,
)
from pydicom.sr.codedict import codes
from pydicom.valuerep import PersonName, format_number_as_ds
from pydicom.sr.coding import Code

from highdicom._module_utils import (
    ModuleUsageValues,
    does_iod_have_pixel_data,
    get_module_usage,
    is_multiframe_image,
)
from highdicom.image import _Image
from highdicom.base import _check_little_endian
from highdicom.color import CIELabColor
from highdicom.content import (
    ContentCreatorIdentificationCodeSequence,
    PaletteColorLUTTransformation,
    PlaneOrientationSequence,
    PlanePositionSequence,
    PixelMeasuresSequence
)
from highdicom.enum import (
    CoordinateSystemNames,
    DimensionOrganizationTypeValues,
)
from highdicom.frame import encode_frame
from highdicom.pr.content import (
    _add_icc_profile_attributes,
    _add_palette_color_lookup_table_attributes,
)
from highdicom.utils import (
    are_plane_positions_tiled_full,
)
from highdicom.seg.content import (
    DimensionIndexSequence,
    SegmentDescription,
)
from highdicom.seg.enum import (
    SegmentationFractionalTypeValues,
    SegmentationTypeValues,
    SegmentsOverlapValues,
    SegmentAlgorithmTypeValues,
)
from highdicom.seg.utils import iter_segments
from highdicom.spatial import (
    ImageToReferenceTransformer,
    compute_tile_positions_per_frame,
    get_image_coordinate_system,
    get_volume_positions,
    get_tile_array,
)
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import (
    check_person_name,
    _check_code_string,
    _check_long_string,
)
from highdicom.volume import (
    ChannelDescriptor,
    Volume,
    VolumeGeometry,
    RGB_COLOR_CHANNEL_DESCRIPTOR,
    VOLUME_INDEX_CONVENTION,
)


logger = logging.getLogger(__name__)


# These codes are needed many times in loops so we precompute them
_DERIVATION_CODE = CodedConcept.from_code(
    codes.cid7203.SegmentationImageDerivation
)
_PURPOSE_CODE = CodedConcept.from_code(
    codes.cid7202.SourceImageForImageProcessingOperation
)


def _get_unsigned_dtype(max_val: int | np.integer) -> type:
    """Get the smallest unsigned NumPy datatype to accommodate a value.

    Parameters
    ----------
    max_val: int
        The largest non-negative integer that must be accommodated.

    Returns
    -------
    numpy.dtype:
        The selected NumPy datatype.

    """
    if max_val < 256:
        dtype = np.dtype(np.uint8)
    elif max_val < 65536:
        dtype = np.dtype(np.uint16)
    else:
        dtype = np.dtype(np.uint32)  # should be extremely unlikely
    return dtype


def _check_numpy_value_representation(
    max_val: int,
    dtype: np.dtype | str | type
) -> None:
    """Check whether a given maximum value can be represented by a given dtype.

    Parameters
    ----------
    max_val: int
        The largest non-negative integer that must be accommodated.
    dtype: Union[numpy.dtype, str, type]
        Data type of the array to be checked

    Raises
    ------
    ValueError
        If the given maximum value is too large to be represented by dtype.

    """
    dtype = np.dtype(dtype)
    raise_error = False
    if dtype.kind == 'f':
        if max_val > np.finfo(dtype).max:
            raise_error = True
    elif dtype.kind in ('i', 'u'):
        if max_val > np.iinfo(dtype).max:
            raise_error = True
    elif dtype.kind == 'b':
        if max_val > 1:
            raise_error = True
    if raise_error:
        raise ValueError(
            "The maximum output value of the segmentation array is "
            f"{max_val}, which is too large be represented using dtype "
            f"{dtype}."
        )


class Segmentation(_Image):

    """SOP class for the Segmentation IOD.

    See :doc:`seg` for an overview of working with Segmentations.

    """

    def __init__(
        self,
        source_images: Sequence[Dataset],
        pixel_array: np.ndarray | Volume,
        segmentation_type: str | SegmentationTypeValues,
        segment_descriptions: Sequence[SegmentDescription],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: str | tuple[str],
        device_serial_number: str,
        fractional_type: None | (
            str | SegmentationFractionalTypeValues
        ) = SegmentationFractionalTypeValues.PROBABILITY,
        max_fractional_value: int = 255,
        content_description: str | None = None,
        content_creator_name: str | PersonName | None = None,
        transfer_syntax_uid: str | UID = ExplicitVRLittleEndian,
        pixel_measures: PixelMeasuresSequence | None = None,
        plane_orientation: PlaneOrientationSequence | None = None,
        plane_positions: Sequence[PlanePositionSequence] | None = None,
        omit_empty_frames: bool = True,
        content_label: str | None = None,
        content_creator_identification: None | (
            ContentCreatorIdentificationCodeSequence
        ) = None,
        workers: int | Executor = 0,
        dimension_organization_type: (
            DimensionOrganizationTypeValues |
            str |
            None
        ) = None,
        tile_pixel_array: bool = False,
        tile_size: Sequence[int] | None = None,
        pyramid_uid: str | None = None,
        pyramid_label: str | None = None,
        further_source_images: Sequence[Dataset] | None = None,
        palette_color_lut_transformation: None | (
            PaletteColorLUTTransformation
        ) = None,
        icc_profile: bytes | None = None,
        **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        source_images: Sequence[Dataset]
            One or more single- or multi-frame images (or metadata of images)
            from which the segmentation was derived
        pixel_array: numpy.ndarray
            Array of segmentation pixel data of boolean, unsigned integer or
            floating point data type representing a mask image. The array may
            be a 2D, 3D or 4D numpy array.

            If it is a 2D numpy array, it represents the segmentation of a
            single frame image, such as a planar x-ray or single instance from
            a CT or MR series.

            If it is a 3D array, it represents the segmentation of either a
            series of source images (such as a series of CT or MR images) a
            single 3D multi-frame image (such as a multi-frame CT/MR image), or
            a single 2D tiled image (such as a slide microscopy image).

            If ``pixel_array`` represents the segmentation of a 3D image, the
            first dimension represents individual 2D planes. Unless the
            ``plane_positions`` parameter is provided, the frame in
            ``pixel_array[i, ...]`` should correspond to either
            ``source_images[i]`` (if ``source_images`` is a list of single
            frame instances) or ``source_images[0].pixel_array[i, ...]`` if
            ``source_images`` is a single multiframe instance.

            Similarly, if ``pixel_array`` is a 3D array representing the
            segmentation of a tiled 2D image, the first dimension represents
            individual 2D tiles (for one channel and z-stack) and these tiles
            correspond to the frames in the source image dataset.

            If ``pixel_array`` is an unsigned integer or boolean array with
            binary data (containing only the values ``True`` and ``False`` or
            ``0`` and ``1``) or a floating-point array, it represents a single
            segment. In the case of a floating-point array, values must be in
            the range 0.0 to 1.0.

            Otherwise, if ``pixel_array`` is a 2D or 3D array containing multiple
            unsigned integer values, each value is treated as a different
            segment whose segment number is that integer value. This is
            referred to as a *label map* style segmentation.  In this case, all
            segments from 1 through ``pixel_array.max()`` (inclusive) must be
            described in `segment_descriptions`, regardless of whether they are
            present in the image.  Note that this is valid for segmentations
            encoded using the ``"BINARY"``, ``"LABELMAP"`` or ``"FRACTIONAL"``
            methods.

            Note that that a 2D numpy array and a 3D numpy array with a
            single frame along the first dimension may be used interchangeably
            as segmentations of a single frame, regardless of their data type.

            If ``pixel_array`` is a 4D numpy array, the first three dimensions
            are used in the same way as the 3D case and the fourth dimension
            represents multiple segments. In this case
            ``pixel_array[:, :, :, i]`` represents segment number ``i + 1``
            (since numpy indexing is 0-based but segment numbering is 1-based),
            and all segments from 1 through ``pixel_array.shape[-1] + 1`` must
            be described in ``segment_descriptions``.

            Furthermore, a 4D array with unsigned integer data type must
            contain only binary data (``True`` and ``False`` or ``0`` and
            ``1``). In other words, a 4D array is incompatible with the *label
            map* style encoding of the segmentation.

            Where there are multiple segments that are mutually exclusive (do
            not overlap) and binary, they may be passed using either a *label
            map* style array or a 4D array. A 4D array is required if either
            there are multiple segments and they are not mutually exclusive
            (i.e. they overlap) or there are multiple segments and the
            segmentation is fractional.

            Note that if the segmentation of a single source image with
            multiple stacked segments is required, it is necessary to include
            the singleton first dimension in order to give a 4D array.

            For ``"FRACTIONAL"`` segmentations, values either encode the
            probability of a given pixel belonging to a segment
            (if `fractional_type` is ``"PROBABILITY"``)
            or the extent to which a segment occupies the pixel
            (if `fractional_type` is ``"OCCUPANCY"``).

            Alternatively, ``pixel_array`` may be an instance of a
            :class:`highdicom.Volume`. In this case, behavior is the
            same as if the underlying numpy array is passed, and additionally,
            the ``pixel_measures``, ``plane_positions`` and
            ``plane_orientation`` will be computed from the volume, and
            therefore should not be passed as parameters.

        segmentation_type: Union[str, highdicom.seg.SegmentationTypeValues]
            Type of segmentation, either ``"BINARY"``, ``"FRACTIONAL"``, or
            ``"LABELMAP"``.
        segment_descriptions: Sequence[highdicom.seg.SegmentDescription]
            Description of each segment encoded in `pixel_array`. In the case
            of pixel arrays with multiple integer values, the segment
            description with the corresponding segment number is used to
            describe each segment. No description should be provided for pixels
            with value 0, which are considered background pixels.
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the output segmentation series.
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        manufacturer: str
            Name of the manufacturer of the device (developer of the software)
            that creates the instance
        manufacturer_model_name: str
            Name of the device model (name of the software library or
            application) that creates the instance
        software_versions: Union[str, Tuple[str]]
            Version(s) of the software that creates the instance
        device_serial_number: str
            Manufacturer's serial number of the device
        fractional_type: Union[str, highdicom.seg.SegmentationFractionalTypeValues, None], optional
            Type of fractional segmentation that indicates how pixel data
            should be interpreted
        max_fractional_value: int, optional
            Maximum value that indicates probability or occupancy of 1 that
            a pixel represents a given segment
        content_description: Union[str, None], optional
            Description of the segmentation
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the creator of the segmentation (if created manually)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported for encapsulated format encoding in case of
            FRACTIONAL segmentation type:
            RLE Lossless (``"1.2.840.10008.1.2.5"``),
            JPEG 2000 Lossless (``"1.2.840.10008.1.2.4.90"``), and
            JPEG LS Lossless (``"1.2.840.10008.1.2.4.00"``).
        pixel_measures: Union[highdicom.PixelMeasures, None], optional
            Physical spacing of image pixels in `pixel_array`. If ``None``, it
            will be assumed that the segmentation image has the same pixel
            measures as the source image(s). If ``pixel_array`` is an instance
            of :class:`highdicom.Volume`, the pixel measures will be
            computed from it and therefore this parameter should be left an
            ``None``.
        plane_orientation: Union[highdicom.PlaneOrientationSequence, None], optional
            Orientation of planes in `pixel_array` relative to axes of
            three-dimensional patient or slide coordinate space. If ``None``,
            it will be assumed that the segmentation image as the same plane
            orientation as the source image(s). If ``pixel_array`` is an
            instance of :class:`highdicom.Volume`, the plane orientation
            will be computed from it and therefore this parameter should be
            left an ``None``.
        plane_positions: Union[Sequence[highdicom.PlanePositionSequence], None], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space. If ``None``, it will be assumed
            that the segmentation image has the same plane position as the
            source image(s). However, this will only work when the first
            dimension of `pixel_array` matches the number of frames in
            `source_images` (in case of multi-frame source images) or the
            number of `source_images` (in case of single-frame source images).
            If ``pixel_array`` is an instance of
            :class:`highdicom.Volume`, the plane positions will be
            computed from it and therefore this parameter should be left an
            ``None``.
        omit_empty_frames: bool, optional
            If True (default), frames with no non-zero pixels are omitted from
            the segmentation image. If False, all frames are included.
        content_label: Union[str, None], optional
            Content label
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this segmentation.
        workers: Union[int, concurrent.futures.Executor], optional
            Number of worker processes to use for frame compression. If 0, no
            workers are used and compression is performed in the main process
            (this is the default behavior). If negative, as many processes are
            created as the machine has processors.

            Alternatively, you may directly pass an instance of a class derived
            from ``concurrent.futures.Executor`` (most likely an instance of
            ``concurrent.futures.ProcessPoolExecutor``) for highdicom to use.
            You may wish to do this either to have greater control over the
            setup of the executor, or to avoid the setup cost of spawning new
            processes each time this ``__init__`` method is called if your
            application creates a large number of Segmentations.

            Note that if you use worker processes, you must ensure that your
            main process uses the ``if __name__ == "__main__"`` idiom to guard
            against spawned child processes creating further workers.
        dimension_organization_type: Union[highdicom.enum.DimensionOrganizationTypeValues, str, None], optional
            Dimension organization type to use for the output image.
        tile_pixel_array: bool, optional
            If True, `highdicom` will automatically convert an input total
            pixel matrix into a sequence of frames representing tiles of the
            segmentation. This is valid only when the source image supports
            tiling (e.g. VL Whole Slide Microscopy images).

            If True, the input pixel array must consist of a single "frame",
            i.e. must be either a 2D numpy array, a 3D numpy array with a size
            of 1 down the first dimension (axis zero), or a 4D numpy array also
            with a size of 1 down the first dimension. The input pixel array is
            treated as the total pixel matrix of the segmentation, and this is
            tiled along the row and column dimension to create an output image
            with multiple, smaller frames.

            If no ``pixel_measures``, ``plane_positions``,
            ``plane_orientation`` are supplied, the total pixel matrix of the
            segmentation is assumed to correspond to the total pixel matrix of
            the (single) source image. If ``plane_positions`` is supplied, the
            sequence should contain a single item representing the plane
            position of the entire total pixel matrix. Plane positions of
            the newly created tiles will derived automatically from this.

            If False, the pixel array is already considered to consist of one
            or more existing frames, as described above.
        tile_size: Union[Sequence[int], None], optional
            Tile size to use when tiling the input pixel array. If ``None``
            (the default), the tile size is copied from the source image.
            Otherwise the tile size is specified explicitly as (number of rows,
            number of columns). This value is ignored if ``tile_pixel_array``
            is False.
        pyramid_uid: Optional[str], optional
            Unique identifier for the pyramid containing this segmentation.
            Should only be used if this segmentation is part of a
            multi-resolution pyramid.
        pyramid_label: Optional[str], optional
            Human readable label for the pyramid containing this segmentation.
            Should only be used if this segmentation is part of a
            multi-resolution pyramid.
        further_source_images: Optional[Sequence[pydicom.Dataset]], optional
            Additional images to record as source images in the segmentation.
            Unlike the main ``source_images`` parameter, these images will
            *not* be used to infer the position and orientation of the
            ``pixel_array`` in the case that no plane positions are supplied.
            Images from multiple series may be passed, however they must all
            belong to the same study.
        palette_color_lut_transformation: Union[highdicom.PaletteColorLUTTransformation, None], optional
            A palette color lookup table transformation to apply to the pixels
            for display. This is only permitted if segmentation_type is "LABELMAP".
        icc_profile: Union[bytes, None] = None
            An ICC profile to display the segmentation. This is only permitted
            when palette_color_lut_transformation is provided.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Raises
        ------
        ValueError
            When

                * Length of `source_images` is zero.
                * Items of `source_images` are not all part of the same study
                  and series.
                * Items of `source_images` have different number of rows and
                  columns.
                * Length of `plane_positions` does not match number of segments
                  encoded in `pixel_array`.
                * Length of `plane_positions` does not match number of 2D planes
                  in `pixel_array` (size of first array dimension).

        Note
        ----
        The assumption is made that segments in `pixel_array` are defined in
        the same frame of reference as `source_images`.

        """  # noqa: E501
        if len(source_images) == 0:
            raise ValueError('At least one source image is required.')

        uniqueness_criteria = {
            (
                image.StudyInstanceUID,
                image.SeriesInstanceUID,
                image.Rows,
                image.Columns,
                getattr(image, 'FrameOfReferenceUID', None),
            )
            for image in source_images
        }
        if len(uniqueness_criteria) > 1:
            raise ValueError(
                'Source images must all be part of the same series and must '
                'have the same image dimensions (number of rows/columns).'
            )

        src_img = source_images[0]
        is_multiframe = is_multiframe_image(src_img)
        if is_multiframe and len(source_images) > 1:
            raise ValueError(
                'Only one source image should be provided in case images '
                'are multi-frame images.'
            )
        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            JPEG2000Lossless,
            JPEGLSLossless,
            RLELossless,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported.'
            )

        segmentation_type = SegmentationTypeValues(segmentation_type)
        sop_class_uid = (
            '1.2.840.10008.5.1.4.1.1.66.7'
            if segmentation_type == SegmentationTypeValues.LABELMAP
            else '1.2.840.10008.5.1.4.1.1.66.4'
        )
        super().__init__(
            study_instance_uid=src_img.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid=sop_class_uid,
            manufacturer=manufacturer,
            modality='SEG',
            transfer_syntax_uid=transfer_syntax_uid,
            patient_id=src_img.PatientID,
            patient_name=src_img.PatientName,
            patient_birth_date=src_img.PatientBirthDate,
            patient_sex=src_img.PatientSex,
            accession_number=src_img.AccessionNumber,
            study_id=src_img.StudyID,
            study_date=src_img.StudyDate,
            study_time=src_img.StudyTime,
            referring_physician_name=getattr(
                src_img, 'ReferringPhysicianName', None
            ),
            manufacturer_model_name=manufacturer_model_name,
            device_serial_number=device_serial_number,
            software_versions=software_versions,
            **kwargs
        )

        # Frame of Reference
        has_ref_frame_uid = hasattr(src_img, 'FrameOfReferenceUID')
        if has_ref_frame_uid:
            self.FrameOfReferenceUID = src_img.FrameOfReferenceUID
            self.PositionReferenceIndicator = getattr(
                src_img,
                'PositionReferenceIndicator',
                None
            )
        else:
            # Only allow missing FrameOfReferenceUID if it is not required
            # for this IOD
            usage = get_module_usage('frame-of-reference', src_img.SOPClassUID)
            if usage == ModuleUsageValues.MANDATORY:
                raise ValueError(
                    "Source images have no Frame Of Reference UID, but it is "
                    "required by the IOD."
                )

        self._coordinate_system = get_image_coordinate_system(src_img)

        if self._coordinate_system is None:
            # It may be possible to generalize this, but for now only a single
            # source frame is permitted when there is no coordinate system
            if (
                len(source_images) > 1 or
                (is_multiframe and src_img.NumberOfFrames > 1)
            ):
                raise ValueError(
                    "Only a single frame is supported when the source "
                    "image has no Frame of Reference UID."
                )
            if plane_positions is not None:
                raise TypeError(
                    "If source images have no Frame Of Reference UID, the "
                    'argument "plane_positions" may not be specified since the '
                    "segmentation pixel array must be spatially aligned with "
                    "the source images."
                )
            if plane_orientation is not None:
                raise TypeError(
                    "If source images have no Frame Of Reference UID, the "
                    'argument "plane_orientation" may not be specified since '
                    "the segmentation pixel array must be spatially aligned "
                    "with the source images."
                )

        # Check segment numbers
        described_segment_numbers = np.array([
            int(item.SegmentNumber)
            for item in segment_descriptions
        ])
        self._check_segment_numbers(
            described_segment_numbers,
            segmentation_type,
        )

        from_volume = isinstance(pixel_array, Volume)
        if from_volume:
            if self._coordinate_system is None:
                raise ValueError(
                    "A volume should not be passed if the source image(s) "
                    "has/have no FrameOfReferenceUID."
                )
            if pixel_array.frame_of_reference_uid is not None:
                if (
                    pixel_array.frame_of_reference_uid !=
                    src_img.FrameOfReferenceUID
                ):
                    raise ValueError(
                        "The volume passed as the pixel array has a "
                        "different frame of reference from the source "
                        "image."
                    )
            if pixel_measures is not None:
                raise TypeError(
                    "Argument 'pixel_measures' should not be provided if "
                    "'pixel_array' is a highdicom.Volume."
                )
            if plane_orientation is not None:
                raise TypeError(
                    "Argument 'plane_orientation' should not be provided if "
                    "'pixel_array' is a highdicom.Volume."
                )
            if plane_positions is not None:
                raise TypeError(
                    "Argument 'plane_positions' should not be provided if "
                    "'pixel_array' is a highdicom.Volume."
                )
            if pixel_array.number_of_channel_dimensions == 1:
                if pixel_array.channel_descriptors != (
                    ChannelDescriptor('SegmentNumber'),
                ):
                    raise ValueError(
                        "Input volume should have no channels other than "
                        "'SegmentNumber'."
                    )
                vol_seg_nums = pixel_array.get_channel_values('SegmentNumber')
                if not np.array_equal(
                    np.array(vol_seg_nums), described_segment_numbers
                ):
                    raise ValueError(
                        "Segment numbers in the input volume do not match "
                        "the described segments."
                    )
            elif pixel_array.number_of_channel_dimensions != 0:
                raise ValueError(
                    "If 'pixel_array' is a highdicom.Volume, it should have "
                    "0 or 1 channel dimensions."
                )

            plane_positions = pixel_array.get_plane_positions()
            plane_orientation = pixel_array.get_plane_orientation()
            pixel_measures = pixel_array.get_pixel_measures()
            pixel_array = pixel_array.array

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        if pixel_array.ndim not in [3, 4]:
            raise ValueError('Pixel array must be a 2D, 3D, or 4D array.')

        is_tiled = hasattr(src_img, 'TotalPixelMatrixRows')
        if tile_pixel_array and not is_tiled:
            raise ValueError(
                'When argument "tile_pixel_array" is True, the source image '
                'must be a tiled image.'
            )
        if tile_pixel_array and pixel_array.shape[0] != 1:
            raise ValueError(
                'When argument "tile_pixel_array" is True, the input pixel '
                'array must contain only one "frame" representing the '
                'entire pixel matrix.'
            )

        # Remember whether these values were provided by the user, or inferred
        # from the source image. If inferred, we can skip some checks
        user_provided_orientation = plane_orientation is not None
        user_provided_measures = pixel_measures is not None

        # General Reference

        if further_source_images is not None:
            # We make no requirement here that images should be from the same
            # series etc, but they should belong to the same study and be image
            # objects
            for s_img in further_source_images:
                if not isinstance(s_img, Dataset):
                    raise TypeError(
                        "All items in 'further_source_images' should be "
                        "of type 'pydicom.Dataset'."
                    )
                if s_img.StudyInstanceUID != self.StudyInstanceUID:
                    raise ValueError(
                        "All items in 'further_source_images' should belong "
                        "to the same study as 'source_images'."
                    )
                if not does_iod_have_pixel_data(s_img.SOPClassUID):
                    raise ValueError(
                        "All items in 'further_source_images' should be "
                        "image objects."
                    )
        else:
            further_source_images = []

        # Note that appending directly to the SourceImageSequence is typically
        # slow so it's more efficient to build as a Python list then convert
        # later. We save conversion for after the main loop
        source_image_seq: list[Dataset] = []
        referenced_series: dict[str, list[Dataset]] = defaultdict(list)
        for s_img in chain(source_images, further_source_images):
            ref = Dataset()
            ref.ReferencedSOPClassUID = s_img.SOPClassUID
            ref.ReferencedSOPInstanceUID = s_img.SOPInstanceUID
            source_image_seq.append(ref)
            referenced_series[s_img.SeriesInstanceUID].append(ref)
        self.SourceImageSequence = source_image_seq

        # Common Instance Reference
        ref_image_seq: list[Dataset] = []
        for series_instance_uid, referenced_images in referenced_series.items():
            ref = Dataset()
            ref.SeriesInstanceUID = series_instance_uid
            ref.ReferencedInstanceSequence = referenced_images
            ref_image_seq.append(ref)
        self.ReferencedSeriesSequence = ref_image_seq

        # Image Pixel
        if tile_pixel_array:
            # By default use the same tile size as the source image (even if
            # they are not spatially aligned)
            tile_size = tile_size or (src_img.Rows, src_img.Columns)
            self.Rows, self.Columns = (tile_size)
        else:
            self.Rows = pixel_array.shape[1]
            self.Columns = pixel_array.shape[2]

        # Segmentation Image
        self.ImageType = ['DERIVED', 'PRIMARY']
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = 'MONOCHROME2'
        self.PixelRepresentation = 0
        self.SegmentationType = segmentation_type.value

        if content_label is not None:
            _check_code_string(content_label)
            self.ContentLabel = content_label
        else:
            self.ContentLabel = f'{src_img.Modality}_SEG'
        self.ContentDescription = content_description
        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name
        if content_creator_identification is not None:
            if not isinstance(
                content_creator_identification,
                ContentCreatorIdentificationCodeSequence
            ):
                raise TypeError(
                    'Argument "content_creator_identification" must be of type '
                    'ContentCreatorIdentificationCodeSequence.'
                )
            self.ContentCreatorIdentificationCodeSequence = \
                content_creator_identification

        if segmentation_type == SegmentationTypeValues.BINARY:
            dtype = np.uint8
            self.BitsAllocated = 1
            self.HighBit = 0
            if (
                self.file_meta.TransferSyntaxUID != JPEG2000Lossless and
                self.file_meta.TransferSyntaxUID.is_encapsulated
            ):
                raise ValueError(
                    'The chosen transfer syntax '
                    f'{self.file_meta.TransferSyntaxUID} '
                    'is not compatible with the BINARY segmentation type'
                )
        elif segmentation_type == SegmentationTypeValues.FRACTIONAL:
            dtype = np.uint8
            self.BitsAllocated = 8
            self.HighBit = 7
            segmentation_fractional_type = SegmentationFractionalTypeValues(
                fractional_type
            )
            self.SegmentationFractionalType = segmentation_fractional_type.value
            if max_fractional_value > 2**8:
                raise ValueError(
                    'Maximum fractional value must not exceed image bit depth.'
                )
            self.MaximumFractionalValue = max_fractional_value
        elif segmentation_type == SegmentationTypeValues.LABELMAP:
            # Decide on the output datatype and update the image metadata
            # accordingly. Use the smallest possible type
            dtype = _get_unsigned_dtype(described_segment_numbers.max())
            if dtype == np.uint32:
                raise ValueError(
                    "Too many segments to represent with a 16 bit integer."
                )
            self.BitsAllocated = np.iinfo(dtype).bits
            self.HighBit = self.BitsAllocated - 1
            self.BitsStored = self.BitsAllocated
            self.PixelPaddingValue = 0

        self.BitsStored = self.BitsAllocated
        self.LossyImageCompression = getattr(
            src_img,
            'LossyImageCompression',
            '00'
        )
        if self.LossyImageCompression == '01':
            if 'LossyImageCompressionRatio' in src_img:
                self.LossyImageCompressionRatio = \
                    src_img.LossyImageCompressionRatio
            if 'LossyImageCompressionMethod' in src_img:
                self.LossyImageCompressionMethod = \
                    src_img.LossyImageCompressionMethod

        # Use PALETTE COLOR photometric interpretation in the case
        # of a labelmap segmentation with a provided LUT, MONOCHROME2
        # otherwise
        if segmentation_type == SegmentationTypeValues.LABELMAP:
            if palette_color_lut_transformation is None:
                self.PhotometricInterpretation = 'MONOCHROME2'
                if icc_profile is not None:
                    raise TypeError(
                        "Argument 'icc_profile' should "
                        "not be provided if is "
                        "'palette_color_lut_transformation' "
                        "is not specified."
                    )
            else:
                # Using photometric interpretation "PALETTE COLOR"
                # need to specify the LUT in this case
                self.PhotometricInterpretation = 'PALETTE COLOR'

                # Checks on the validity of the LUT
                if not isinstance(
                    palette_color_lut_transformation,
                    PaletteColorLUTTransformation
                ):
                    raise TypeError(
                        'Argument "palette_color_lut_transformation" must be '
                        'of type highdicom.PaletteColorLUTTransformation.'
                    )

                if palette_color_lut_transformation.is_segmented:
                    raise ValueError(
                        'Palette Color LUT Transformations must not be '
                        'segmented when included in a Segmentation.'
                    )

                lut = palette_color_lut_transformation.red_lut
                lut_entries = lut.number_of_entries
                lut_start = lut.first_mapped_value
                lut_end = lut_start + lut_entries

                if (
                    (lut_start > 0) or
                    lut_end <= described_segment_numbers.max()
                ):
                    raise ValueError(
                        'The labelmap provided does not have entries '
                        'to covering all segments and background.'
                    )

                for desc in segment_descriptions:
                    if hasattr(desc, 'RecommendedDisplayCIELabValue'):
                        raise ValueError(
                            'Segment descriptions should not specify a display '
                            'color when using a palette color LUT.'
                        )

                # Add the LUT to this instance
                _add_palette_color_lookup_table_attributes(
                    self,
                    palette_color_lut_transformation,
                )

                if icc_profile is None:
                    # Use default sRGB profile
                    icc_profile = pkgutil.get_data(
                        'highdicom',
                        '_icc_profiles/sRGB_v4_ICC_preference.icc'
                    )
                _add_icc_profile_attributes(
                    self,
                    icc_profile=icc_profile
                )

        else:
            self.PhotometricInterpretation = 'MONOCHROME2'
            if palette_color_lut_transformation is not None:
                raise TypeError(
                    "Argument 'palette_color_lut_transformation' should "
                    "not be provided when 'segmentation_type' is "
                    f"'{segmentation_type.value}'."
                )
            if icc_profile is not None:
                raise TypeError(
                    "Argument 'icc_profile' should "
                    "not be provided when 'segmentation_type' is "
                    f"'{segmentation_type.value}'."
                )

        # Multi-Resolution Pyramid
        if pyramid_uid is not None:
            if not is_tiled:
                raise TypeError(
                    'Argument "pyramid_uid" should only be specified '
                    'for tiled images.'
                )
            if (
                self._coordinate_system is None or
                self._coordinate_system != CoordinateSystemNames.SLIDE
            ):
                raise TypeError(
                    'Argument "pyramid_uid" should only be specified for '
                    'segmentations in the SLIDE coordinate system.'
                )
            self.PyramidUID = pyramid_uid

            if pyramid_label is not None:
                _check_long_string(pyramid_label)
                self.PyramidLabel = pyramid_label

        elif pyramid_label is not None:
            raise TypeError(
                'Argument "pyramid_label" should not be specified if '
                '"pyramid_uid" is not specified.'
            )

        # Multi-Frame Functional Groups and Multi-Frame Dimensions
        source_pixel_measures = self._get_pixel_measures_sequence(
            source_image=src_img,
            is_multiframe=is_multiframe,
            coordinate_system=self._coordinate_system,
        )
        if pixel_measures is None:
            pixel_measures = source_pixel_measures

        if self._coordinate_system is not None:
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                source_plane_orientation = PlaneOrientationSequence(
                    coordinate_system=self._coordinate_system,
                    image_orientation=src_img.ImageOrientationSlide
                )
            else:
                if is_multiframe:
                    src_sfg = src_img.SharedFunctionalGroupsSequence[0]

                    if 'PlaneOrientationSequence' not in src_sfg:
                        raise ValueError(
                            'Source images must have a shared '
                            'orientation.'
                        )

                    source_plane_orientation = (
                        PlaneOrientationSequence.from_sequence(
                            src_sfg.PlaneOrientationSequence
                        )
                    )
                else:
                    iop = src_img.ImageOrientationPatient

                    for image in source_images:
                        if image.ImageOrientationPatient != iop:
                            raise ValueError(
                                'Source images must have a shared '
                                'orientation.'
                            )

                    source_plane_orientation = PlaneOrientationSequence(
                        coordinate_system=self._coordinate_system,
                        image_orientation=src_img.ImageOrientationPatient
                    )

            if plane_orientation is None:
                plane_orientation = source_plane_orientation

        include_segment_number = (
            segmentation_type != SegmentationTypeValues.LABELMAP
        )
        self.DimensionIndexSequence = DimensionIndexSequence(
            coordinate_system=self._coordinate_system,
            include_segment_number=include_segment_number,
        )
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = \
            self.DimensionIndexSequence[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence = [dimension_organization]

        self._add_segment_descriptions(
            segment_descriptions,
            segmentation_type,
        )

        # Checks on pixels and overlap
        pixel_array, segments_overlap = self._check_and_cast_pixel_array(
            pixel_array,
            described_segment_numbers,
            segmentation_type,
            dtype=dtype,
        )
        self.SegmentsOverlap = segments_overlap.value

        if self._coordinate_system is not None:
            if tile_pixel_array:

                src_origin_seq = src_img.TotalPixelMatrixOriginSequence[0]
                src_x_offset = src_origin_seq.XOffsetInSlideCoordinateSystem
                src_y_offset = src_origin_seq.YOffsetInSlideCoordinateSystem
                src_z_offset = src_origin_seq.get(
                    'ZOffsetInSlideCoordinateSystem',
                    0.0,
                )

                if plane_positions is None:
                    # Use the origin of the source image
                    x_offset = src_x_offset
                    y_offset = src_y_offset
                    z_offset = src_z_offset
                    origin_preserved = True
                else:
                    if len(plane_positions) != 1:
                        raise ValueError(
                            "If specifying plane_positions when the "
                            '"tile_pixel_array" argument is True, a '
                            "single plane position should be provided "
                            "representing the position of the top  "
                            "left corner of the total pixel matrix."
                        )
                    # Use the provided image origin
                    pp = plane_positions[0][0]
                    rp = pp.RowPositionInTotalImagePixelMatrix
                    cp = pp.ColumnPositionInTotalImagePixelMatrix
                    if rp != 1 or cp != 1:
                        raise ValueError(
                            "When specifying a single plane position when "
                            'the "tile_pixel_array" argument is True, the '
                            "plane position must be at the top left corner "
                            "of the total pixel matrix. I.e. it must have "
                            "RowPositionInTotalImagePixelMatrix and "
                            "ColumnPositionInTotalImagePixelMatrix equal to 1."
                        )
                    x_offset = pp.XOffsetInSlideCoordinateSystem
                    y_offset = pp.YOffsetInSlideCoordinateSystem
                    z_offset = pp.get(
                        'ZOffsetInSlideCoordinateSystem',
                        0.0,
                    )
                    origin_preserved = (
                        x_offset == src_x_offset and
                        y_offset == src_y_offset and
                        z_offset == src_z_offset
                    )

                orientation = plane_orientation[0].ImageOrientationSlide
                image_position = [x_offset, y_offset, z_offset]

                are_total_pixel_matrix_locations_preserved = (
                    origin_preserved and
                    (
                        not user_provided_orientation or
                        plane_orientation == source_plane_orientation
                    ) and
                    (
                        not user_provided_measures or
                        (
                            pixel_measures[0].PixelSpacing ==
                            source_pixel_measures[0].PixelSpacing
                        )
                    )
                )

                if are_total_pixel_matrix_locations_preserved:
                    if (
                        pixel_array.shape[1:3] !=
                        (
                            src_img.TotalPixelMatrixRows,
                            src_img.TotalPixelMatrixColumns
                        )
                    ):
                        raise ValueError(
                            "Shape of input pixel_array does not match shape "
                            "of the total pixel matrix of the source image."
                        )

                    # The overall total pixel matrix can match the source
                    # image's but if the image is tiled differently, spatial
                    # locations within each frame are not preserved
                    are_spatial_locations_preserved = (
                        tile_size == (src_img.Rows, src_img.Columns)
                    )
                else:
                    are_spatial_locations_preserved = False

                raw_plane_positions = compute_tile_positions_per_frame(
                    rows=self.Rows,
                    columns=self.Columns,
                    total_pixel_matrix_rows=pixel_array.shape[1],
                    total_pixel_matrix_columns=pixel_array.shape[2],
                    total_pixel_matrix_image_position=image_position,
                    image_orientation=orientation,
                    pixel_spacing=pixel_measures[0].PixelSpacing,
                )
                plane_sort_index = np.arange(len(raw_plane_positions))

                # Only need to create the plane position DICOM objects if
                # they will be placed into the object. Otherwise skip this
                # as it is really inefficient
                if (
                    dimension_organization_type !=
                    DimensionOrganizationTypeValues.TILED_FULL
                ):
                    plane_positions = [
                        PlanePositionSequence(
                            CoordinateSystemNames.SLIDE,
                            image_position=coords,
                            pixel_matrix_position=offsets,
                        )
                        for offsets, coords in raw_plane_positions
                    ]
                else:
                    # Unneeded
                    plane_positions = [None]

                # Match the format used elsewhere
                plane_position_values = np.array(
                    [
                        [*offsets, *coords]
                        for offsets, coords in raw_plane_positions
                    ]
                )

                # compute_tile_positions_per_frame returns
                # (c, r, x, y, z) but the dimension index sequence
                # requires (r, c, x, y z). Swap here to correct for
                # this
                plane_position_values = plane_position_values[
                    :, [1, 0, 2, 3, 4]
                ]

            else:
                are_measures_and_orientation_preserved = (
                    (
                        not user_provided_orientation or
                        plane_orientation == source_plane_orientation
                    ) and
                    (
                        not user_provided_measures or
                        (
                            pixel_measures[0].PixelSpacing ==
                            source_pixel_measures[0].PixelSpacing
                        )
                    )
                )

                if (
                    plane_positions is None or
                    are_measures_and_orientation_preserved
                ):
                    # Calculating source positions can be slow, so avoid unless
                    # necessary
                    dim_ind = self.DimensionIndexSequence
                    if is_multiframe:
                        source_plane_positions = \
                            dim_ind.get_plane_positions_of_image(
                                src_img
                            )
                    else:
                        source_plane_positions = \
                            dim_ind.get_plane_positions_of_series(
                                source_images
                            )

                if plane_positions is None:
                    if pixel_array.shape[0] != len(source_plane_positions):
                        raise ValueError(
                            'Number of plane positions in source image(s) does '
                            'not match size of first dimension of '
                            '"pixel_array" argument.'
                        )
                    plane_positions = source_plane_positions
                    are_spatial_locations_preserved = \
                        are_measures_and_orientation_preserved
                else:
                    if pixel_array.shape[0] != len(plane_positions):
                        raise ValueError(
                            'Number of PlanePositionSequence items provided '
                            'via "plane_positions" argument does not match '
                            'size of first dimension of "pixel_array" argument.'
                        )
                    if are_measures_and_orientation_preserved:
                        are_spatial_locations_preserved = all(
                            plane_positions[i] == source_plane_positions[i]
                            for i in range(len(plane_positions))
                        )
                    else:
                        are_spatial_locations_preserved = False

                # plane_position_values is an array giving, for each plane of
                # the input array, the raw values of all attributes that
                # describe its position. The first dimension is sorted the same
                # way as the input pixel array and the second is sorted the
                # same way as the dimension index sequence (without segment
                # number) plane_sort_index is a list of indices into the input
                # planes giving the order in which they should be arranged to
                # correctly sort them for inclusion into the segmentation
                sort_orientation = (
                    plane_orientation[0].ImageOrientationPatient
                    if self._coordinate_system == CoordinateSystemNames.PATIENT
                    else None
                )
                plane_position_values, plane_sort_index = \
                    self.DimensionIndexSequence.get_index_values(
                        plane_positions,
                        image_orientation=sort_orientation,
                        index_convention=VOLUME_INDEX_CONVENTION,
                    )

        else:
            # Only one spatial location supported
            plane_positions = [None]
            plane_position_values = [None]
            plane_sort_index = np.array([0])
            are_spatial_locations_preserved = True

        # Shared functional groops
        sffg_item = Dataset()
        if (
            self._coordinate_system is not None and
            self._coordinate_system == CoordinateSystemNames.PATIENT
        ):
            sffg_item.PlaneOrientationSequence = plane_orientation

            # Automatically populate the spacing between slices in the
            # pixel measures if it was not provided. This is done on the
            # initial plane positions, before any removals, to give the
            # receiver more information about how to reconstruct a volume
            # from the frames in the case that slices are omitted
            if 'SpacingBetweenSlices' not in pixel_measures[0]:
                ori = plane_orientation[0].ImageOrientationPatient
                slice_spacing, _ = get_volume_positions(
                    image_positions=plane_position_values[:, 0, :],
                    image_orientation=ori,
                )
                if slice_spacing is not None:
                    pixel_measures[0].SpacingBetweenSlices = (
                        format_number_as_ds(slice_spacing)
                    )

        if pixel_measures is not None:
            sffg_item.PixelMeasuresSequence = pixel_measures
        self.SharedFunctionalGroupsSequence = [sffg_item]

        if are_spatial_locations_preserved and not tile_pixel_array:
            if pixel_array.shape[1:3] != (src_img.Rows, src_img.Columns):
                raise ValueError(
                    "Shape of input pixel_array does not match shape of "
                    "the source image."
                )

        # Find indices such that empty planes are removed
        if omit_empty_frames:
            if tile_pixel_array:
                included_plane_indices, is_empty = \
                    self._get_nonempty_tile_indices(
                        pixel_array,
                        plane_positions=plane_positions,
                        rows=self.Rows,
                        columns=self.Columns,
                    )
            else:
                included_plane_indices, is_empty = \
                    self._get_nonempty_plane_indices(pixel_array)
            if is_empty:
                # Cannot omit empty frames when all frames are empty
                omit_empty_frames = False
                included_plane_indices = list(range(len(plane_positions)))
            else:
                # Remove all empty plane positions from the list of sorted
                # plane position indices
                included_plane_indices_set = set(included_plane_indices)
                plane_sort_index = [
                    ind for ind in plane_sort_index
                    if ind in included_plane_indices_set
                ]
        else:
            included_plane_indices = list(range(len(plane_positions)))

        # Dimension Organization Type
        dimension_organization_type = self._check_tiled_dimension_organization(
            dimension_organization_type=dimension_organization_type,
            is_tiled=is_tiled,
            omit_empty_frames=omit_empty_frames,
            plane_positions=plane_positions,
            tile_pixel_array=tile_pixel_array,
            rows=self.Rows,
            columns=self.Columns,
        )

        if (
            self._coordinate_system is not None and
            dimension_organization_type !=
            DimensionOrganizationTypeValues.TILED_FULL
        ):
            # Get unique values of attributes in the Plane Position Sequence or
            # Plane Position Slide Sequence, which define the position of the
            # plane with respect to the three dimensional patient or slide
            # coordinate system, respectively. These can subsequently be used
            # to look up the relative position of a plane relative to the
            # indexed dimension.
            unique_dimension_values = [
                np.unique(
                    plane_position_values[included_plane_indices, index],
                    axis=0
                )
                for index in range(plane_position_values.shape[1])
            ]
        else:
            unique_dimension_values = [None]

        if self._coordinate_system == CoordinateSystemNames.PATIENT:
            inferred_dim_org_type = None

            # To be considered "3D", a segmentation should have frames that are
            # differentiated only by location. This rules out any non-labelmap
            # segmentations with more than a single segment.
            # Further, only segmentation with multiple spatial positions in the
            # final segmentation should be considered to have 3D dimension
            # organization type
            if (
                len(included_plane_indices) > 1 and
                (
                    segmentation_type == SegmentationTypeValues.LABELMAP or
                    len(described_segment_numbers) == 1
                )
            ):
                # Calculate the spacing using only the included planes, and
                # enforce ordering
                ori = plane_orientation[0].ImageOrientationPatient
                spacing, _ = get_volume_positions(
                    image_positions=plane_position_values[
                        included_plane_indices, 0, :
                    ],
                    image_orientation=ori,
                    sort=False,
                )
                if spacing is not None and spacing > 0.0:
                    inferred_dim_org_type = (
                        DimensionOrganizationTypeValues.THREE_DIMENSIONAL
                    )

            if (
                dimension_organization_type ==
                DimensionOrganizationTypeValues.THREE_DIMENSIONAL
            ) and inferred_dim_org_type is None:
                raise ValueError(
                    'Dimension organization "3D" has been specified, '
                    'but the source image is not a regularly-spaced 3D '
                    'volume.'
                )
            dimension_organization_type = inferred_dim_org_type

        if dimension_organization_type is not None:
            self.DimensionOrganizationType = dimension_organization_type.value

        if (
            self._coordinate_system is not None and
            self._coordinate_system == CoordinateSystemNames.SLIDE
        ):
            total_pixel_matrix_size = (
                pixel_array.shape[1:3] if tile_pixel_array else None
            )
            self._add_slide_coordinate_metadata(
                source_image=src_img,
                plane_orientation=plane_orientation,
                plane_position_values=plane_position_values,
                pixel_measures=pixel_measures,
                are_spatial_locations_preserved=are_spatial_locations_preserved,
                is_tiled=is_tiled,
                total_pixel_matrix_size=total_pixel_matrix_size,
            )

            plane_position_names = (
                self.DimensionIndexSequence.get_index_keywords()
            )
            row_dim_index = plane_position_names.index(
                'RowPositionInTotalImagePixelMatrix'
            )
            col_dim_index = plane_position_names.index(
                'ColumnPositionInTotalImagePixelMatrix'
            )

        is_encaps = self.file_meta.TransferSyntaxUID.is_encapsulated
        process_pool: Executor | None = None

        if not isinstance(workers, (int, Executor)):
            raise TypeError(
                'Argument "workers" must be of type int or '
                'concurrent.futures.Executor (or a derived class).'
            )
        using_multiprocessing = (
            isinstance(workers, Executor) or workers != 0
        )

        # List of frames. In the case of native transfer syntaxes, we will
        # collect a list of frames as flattened NumPy arrays for bitpacking at
        # the end. In the case of encapsulated transfer syntaxes with no
        # workers, we will accumulate a list of encoded frames to encapsulate
        # at the end
        frames: list[bytes] | list[np.ndarray] = []

        # In the case of native encoding when the number pixels in a frame is
        # not a multiple of 8. This array carries "leftover" pixels that
        # couldn't be encoded in previous iterations, to future iterations. This
        # saves having to keep the entire un-endoded array in memory, which can
        # get extremely heavy on memory in the case of very large arrays
        remainder_pixels = np.empty((0, ), dtype=np.uint8)

        if is_encaps:
            if using_multiprocessing:
                # In the case of encapsulated transfer syntaxes with multiple
                # workers, we will accumulate a list of encoded frames to
                # encapsulate at the end
                frame_futures: list[Future] = []

                # Use the existing executor or create one
                if isinstance(workers, Executor):
                    process_pool = workers
                else:
                    # If workers is negative, pass None to use all processors
                    process_pool = ProcessPoolExecutor(
                        workers if workers > 0 else None
                    )

            # Parameters to use when calling the encode_frame function in
            # either of the above two cases
            encode_frame_kwargs = dict(
                transfer_syntax_uid=self.file_meta.TransferSyntaxUID,
                bits_allocated=self.BitsAllocated,
                bits_stored=self.BitsStored,
                photometric_interpretation=self.PhotometricInterpretation,
                pixel_representation=self.PixelRepresentation
            )
        else:
            if using_multiprocessing:
                warnings.warn(
                    "Setting workers != 0 or passing an instance of "
                    "concurrent.futures.Executor when using a non-encapsulated "
                    "transfer syntax has no effect.",
                    UserWarning,
                    stacklevel=2,
                )
                using_multiprocessing = False

        # Information about individual frames is placed into the
        # PerFrameFunctionalGroupsSequence. Note that a *very* significant
        # efficiency gain is observed when building this as a Python list
        # rather than a pydicom sequence, and then converting to a pydicom
        # sequence at the end
        pffg_sequence: list[Dataset] = []

        # We want the larger loop to work in the labelmap cases (where segments
        # are dealt with together) and the other cases (where segments are
        # dealt with separately). So we define a suitable iterable here for
        # each case
        segments_iterable = (
            [None] if segmentation_type == SegmentationTypeValues.LABELMAP
            else described_segment_numbers
        )

        for segment_number in segments_iterable:

            for plane_dim_ind, plane_index in enumerate(plane_sort_index, 1):

                if tile_pixel_array:
                    if (
                        dimension_organization_type ==
                        DimensionOrganizationTypeValues.TILED_FULL
                    ):
                        row_offset = int(
                            plane_position_values[plane_index, row_dim_index]
                        )
                        column_offset = int(
                            plane_position_values[plane_index, col_dim_index]
                        )
                    else:
                        pos = plane_positions[plane_index][0]
                        row_offset = pos.RowPositionInTotalImagePixelMatrix
                        column_offset = (
                            pos.ColumnPositionInTotalImagePixelMatrix
                        )

                    plane_array = get_tile_array(
                        pixel_array[0],
                        row_offset=row_offset,
                        column_offset=column_offset,
                        tile_rows=self.Rows,
                        tile_columns=self.Columns,
                    )
                else:
                    # Select the relevant existing frame
                    plane_array = pixel_array[plane_index]

                if segment_number is None:
                    # Deal with all segments at once
                    segment_array = plane_array
                else:
                    # Pixel array for just this segment and this position
                    segment_array = self._get_segment_pixel_array(
                        plane_array,
                        segment_number=segment_number,
                        described_segment_numbers=described_segment_numbers,
                        segmentation_type=segmentation_type,
                        max_fractional_value=max_fractional_value,
                        dtype=dtype,
                    )

                # Even though completely empty planes were removed earlier,
                # there may still be planes in which this specific segment is
                # absent. Such frames should be removed
                if segment_number is not None:
                    if omit_empty_frames and not np.any(segment_array):
                        logger.debug(
                            f'skip empty plane {plane_index} of segment '
                            f'#{segment_number}'
                        )
                        continue

                # Log a debug message
                if segment_number is None:
                    msg = f'add plane #{plane_index}'
                else:
                    msg = (
                        f'add plane #{plane_index} for segment '
                        f'#{segment_number}'
                    )
                logger.debug(msg)

                if (
                    dimension_organization_type !=
                    DimensionOrganizationTypeValues.TILED_FULL
                ):
                    # No per-frame functional group for TILED FULL

                    # Get the item of the PerFrameFunctionalGroupsSequence for
                    # this segmentation frame
                    if self._coordinate_system is not None:
                        plane_pos_val = plane_position_values[plane_index]
                        if (
                            self._coordinate_system ==
                            CoordinateSystemNames.SLIDE
                        ):
                            try:
                                dimension_index_values = [
                                    int(
                                        np.where(
                                            unique_dimension_values[idx] == pos
                                        )[0][0] + 1
                                    )
                                    for idx, pos in enumerate(plane_pos_val)
                                ]
                            except IndexError as error:
                                raise IndexError(
                                    'Could not determine position of plane '
                                    f'#{plane_index} in three dimensional '
                                    'coordinate system based on dimension '
                                    f'index values: {error}'
                                ) from error
                        else:
                            dimension_index_values = [plane_dim_ind]
                    else:
                        if segmentation_type == SegmentationTypeValues.LABELMAP:
                            # Here we have to use the "Frame Label" dimension
                            # value (which is used just to have one index since
                            # Referenced Segment cannot be used)
                            dimension_index_values = [1]
                        else:
                            dimension_index_values = []

                    pffg_item = self._get_pffg_item(
                        segment_number=segment_number,
                        dimension_index_values=dimension_index_values,
                        plane_position=plane_positions[plane_index],
                        source_images=source_images,
                        source_image_index=plane_index,
                        are_spatial_locations_preserved=are_spatial_locations_preserved,  # noqa: E501
                        coordinate_system=self._coordinate_system,
                        is_multiframe=is_multiframe,
                    )
                    pffg_sequence.append(pffg_item)

                # Add the segmentation pixel array for this frame to the list
                if is_encaps:
                    if process_pool is None:
                        # Encode this frame and add resulting bytes to the list
                        # for encapsulation at the end
                        frames.append(
                            encode_frame(
                                segment_array,
                                **encode_frame_kwargs,
                            )
                        )
                    else:
                        # Submit this frame for encoding this frame and add the
                        # future to the list for encapsulation at the end
                        future = process_pool.submit(
                            encode_frame,
                            array=segment_array,
                            **encode_frame_kwargs,
                        )
                        frame_futures.append(future)
                else:
                    flat_array = segment_array.flatten()
                    if (
                        self.SegmentationType ==
                        SegmentationTypeValues.BINARY.value and
                        (self.Rows * self.Columns) // 8 != 0
                    ):
                        # Need to encode a multiple of 8 pixels at a time
                        full_array = np.concatenate(
                            [remainder_pixels, flat_array]
                        )
                        # Round down to closest multiple of 8
                        n_pixels_to_take = 8 * (len(full_array) // 8)
                        to_encode = full_array[:n_pixels_to_take]
                        remainder_pixels = full_array[n_pixels_to_take:]
                    else:
                        # Simple - each frame can be individually encoded
                        to_encode = flat_array

                    frames.append(self._encode_pixels_native(to_encode))

        if (
            dimension_organization_type !=
            DimensionOrganizationTypeValues.TILED_FULL
        ):
            self.PerFrameFunctionalGroupsSequence = pffg_sequence

        if is_encaps:
            if process_pool is not None:
                frames = [
                    fut.result() for fut in frame_futures
                ]

                # Shutdown the pool if we created it, otherwise it is the
                # caller's responsibility
                if process_pool is not workers:
                    process_pool.shutdown()

            # Encapsulate all pre-compressed frames
            self.NumberOfFrames = len(frames)
            self.PixelData = encapsulate(frames)
        else:
            self.NumberOfFrames = len(frames)

            # May need to add in a final set of pixels
            if len(remainder_pixels) > 0:
                frames.append(self._encode_pixels_native(remainder_pixels))

            self.PixelData = b''.join(frames)

        # Add a null trailing byte if required
        if len(self.PixelData) % 2 == 1:
            self.PixelData += b'0'

        self.copy_specimen_information(src_img)
        self.copy_patient_and_study_information(src_img)

        # Build lookup tables for efficient decoding
        self._build_luts()

    def add_segments(
        self,
        pixel_array: np.ndarray,
        segment_descriptions: Sequence[SegmentDescription],
        plane_positions: Sequence[PlanePositionSequence] | None = None,
        omit_empty_frames: bool = True,
    ) -> None:
        """To ensure correctness of segmentation images, this
        method was deprecated in highdicom 0.8.0. For more information
        and migration instructions see :ref:`here <add-segments-deprecation>`.

        """  # noqa: E510
        raise AttributeError(
            'To ensure correctness of segmentation images, the add_segments '
            'method was deprecated in highdicom 0.8.0. For more information '
            'and migration instructions visit '
            'https://highdicom.readthedocs.io/en/latest/release_notes.html'
            '#deprecation-of-add-segments-method'
        )

    @staticmethod
    def _check_segment_numbers(
        segment_numbers: np.ndarray,
        segmentation_type: SegmentationTypeValues,
    ):
        """Checks on segment numbers for a new segmentation.

        For BINARY and FRACTIONAL segmentations, segment numbers should start
        at 1 and increase by 1. Strictly there is no requirement on the
        ordering of these items within the segment sequence, however we enforce
        such an order anyway on segmentations created by highdicom. I.e. our
        conditions are stricter than the standard requires.

        For LABELMAP segmentations, there are no such limitations.

        This method checks this and raises an appropriate exception for the
        user if the segment numbers are incorrect.

        Parameters
        ----------
        segment_numbers: numpy.ndarray
            The segment numbers from the segment descriptions, in the order
            they were passed. 1D array of integers.
        segmentation_type: highdicom.seg.SegmentationTypeValues
            Type of segmentation being created.

        Raises
        ------
        ValueError
            If the ``described_segment_numbers`` do not have the required values

        """
        if segmentation_type == SegmentationTypeValues.LABELMAP:
            # Segment numbers must lie between 1 and 65536 to be represented by
            # an unsigned short, but need not be consecutive for LABELMAP segs.
            # 0 is technically allowed by the standard, but at the moment it is
            # always reserved by highdicom to use for the background segment
            min_seg_no = segment_numbers.min()
            max_seg_no = segment_numbers.max()

            if min_seg_no < 0 or max_seg_no > 65535:
                raise ValueError(
                    'Segmentation numbers must be positive integers below '
                    '65536.'
                )
            if min_seg_no == 0:
                raise ValueError(
                    'The segmentation number 0 is reserved by highdicom for '
                    'the background class.'
                )

            # Segment numbers must be unique within an instance
            if len(np.unique(segment_numbers)) < len(segment_numbers):
                raise ValueError(
                    'Segments descriptions must have unique segment numbers.'
                )

            # We additionally impose an ordering constraint (not required by
            # the standard)
            if not np.all(segment_numbers[:-1] <= segment_numbers[1:]):
                raise ValueError(
                    'Segments descriptions must be in ascending order by '
                    'segment number.'
                )
        else:
            # Check segment numbers in the segment descriptions are
            # monotonically increasing by 1
            if not (np.diff(segment_numbers) == 1).all():
                raise ValueError(
                    'Segment descriptions must be sorted by segment number '
                    'and monotonically increasing by 1.'
                )
            if segment_numbers[0] != 1:
                raise ValueError(
                    'Segment descriptions should be numbered starting '
                    f'from 1. Found {segment_numbers[0]}.'
                )

    @staticmethod
    def _get_pixel_measures_sequence(
        source_image: Dataset,
        is_multiframe: bool,
        coordinate_system: CoordinateSystemNames | None,
    ) -> PixelMeasuresSequence | None:
        """Get a Pixel Measures Sequence from the source image.

        This is a helper method used in the constructor.

        Parameters
        ----------
        source_image: pydicom.Dataset
            The first source image.
        is_multiframe: bool
            Whether the source image is multiframe.
        coordinate_system: highdicom.CoordinateSystemNames | None
            The coordinate system of the source image.

        Returns
        -------
        Union[highdicom.PixelMeasuresSequence, None]
            A PixelMeasuresSequence derived from the source image, if this is
            possible. Otherwise None.

        """
        if is_multiframe:
            src_shared_fg = source_image.SharedFunctionalGroupsSequence[0]
            pixel_measures = src_shared_fg.PixelMeasuresSequence
        else:
            if coordinate_system is not None:
                pixel_measures = PixelMeasuresSequence(
                    pixel_spacing=source_image.PixelSpacing,
                    slice_thickness=source_image.SliceThickness,
                    spacing_between_slices=source_image.get(
                        'SpacingBetweenSlices',
                        None
                    )
                )
            else:
                pixel_spacing = getattr(source_image, 'PixelSpacing', None)
                if pixel_spacing is not None:
                    pixel_measures = PixelMeasuresSequence(
                        pixel_spacing=pixel_spacing,
                        slice_thickness=source_image.get(
                            'SliceThickness',
                            None
                        ),
                        spacing_between_slices=source_image.get(
                            'SpacingBetweenSlices',
                            None
                        )
                    )
                else:
                    pixel_measures = None

        return pixel_measures

    def _add_segment_descriptions(
        self,
        segment_descriptions: Sequence[SegmentDescription],
        segmentation_type: SegmentationTypeValues,
    ) -> None:
        """Utility method for constructor that adds segment descriptions.

        Parameters
        ----------
        segment_descriptions: Sequence[highdicom.seg.SegmentDescription]
            User-provided descriptions for each non-background segment.
        segmentation_type: highdicom.seg.SegmentationTypeValues
            Type of segmentation being created.

        """
        if segmentation_type == SegmentationTypeValues.LABELMAP:
            # Need to add a background description in the case of labelmap

            # Set the display color if other segments do
            if any(
                hasattr(desc, 'RecommendedDisplayCIELabValue')
                for desc in segment_descriptions
            ):
                bg_color = CIELabColor(0.0, 0.0, 0.0)  # black
            else:
                bg_color = None

            bg_algo_id = segment_descriptions[0].get(
                'SegmentationAlgorithmIdentificationSequence'
            )

            bg_description = SegmentDescription(
                segment_number=1,
                segment_label='Background',
                segmented_property_category=codes.DCM.Background,
                segmented_property_type=codes.DCM.Background,
                algorithm_type=segment_descriptions[0].SegmentAlgorithmType,
                algorithm_identification=bg_algo_id,
                display_color=bg_color,
            )
            # Override this such that the check on user-constructed segment
            # descriptions having a positive value can remain in place.
            bg_description.SegmentNumber = 0

            self.SegmentSequence = [
                bg_description,
                *segment_descriptions
            ]
        else:
            self.SegmentSequence = segment_descriptions

    def _add_slide_coordinate_metadata(
        self,
        source_image: Dataset,
        plane_orientation: PlaneOrientationSequence,
        plane_position_values: np.ndarray,
        pixel_measures: PixelMeasuresSequence,
        are_spatial_locations_preserved: bool,
        is_tiled: bool,
        total_pixel_matrix_size: tuple[int, int] | None = None,
    ) -> None:
        """Add metadata related to the slide coordinate system.

        This is a helper method used in the constructor.

        Parameters
        ----------
        source_image: pydicom.Dataset
            The source image (assumed to be a single source image).
        plane_orientation: highdicom.PlaneOrientationSequence
            Plane orientation sequence for the segmentation.
        plane_position_values: numpy.ndarray
            Plane positions of each plane.
        pixel_measures: highdicom.PixelMeasuresSequence
            PixelMeasuresSequence for the segmentation.
        are_spatial_locations_preserved: bool
            Whether spatial locations are preserved between the source image
            and the segmentation.
        is_tiled: bool
            Whether the source image is a tiled image.
        total_pixel_matrix_size: Optional[Tuple[int, int]]
            Size (rows, columns) of the total pixel matrix, if known. If None,
            this will be deduced from the specified plane position values.
            Explicitly providing the total pixel matrix size is required if the
            total pixel matrix is smaller than the total area covered by the
            provided tiles (i.e. the provided plane positions are padded).

        """
        plane_position_names = self.DimensionIndexSequence.get_index_keywords()

        self.ImageOrientationSlide = deepcopy(
            plane_orientation[0].ImageOrientationSlide
        )
        if are_spatial_locations_preserved and is_tiled:
            self.TotalPixelMatrixOriginSequence = deepcopy(
                source_image.TotalPixelMatrixOriginSequence
            )
            self.TotalPixelMatrixRows = source_image.TotalPixelMatrixRows
            self.TotalPixelMatrixColumns = source_image.TotalPixelMatrixColumns
            self.TotalPixelMatrixFocalPlanes = 1
        elif are_spatial_locations_preserved and not is_tiled:
            self.ImageCenterPointCoordinatesSequence = deepcopy(
                source_image.ImageCenterPointCoordinatesSequence
            )
        else:
            row_index = plane_position_names.index(
                'RowPositionInTotalImagePixelMatrix'
            )
            row_offsets = plane_position_values[:, row_index]
            col_index = plane_position_names.index(
                'ColumnPositionInTotalImagePixelMatrix'
            )
            col_offsets = plane_position_values[:, col_index]
            frame_indices = np.lexsort([row_offsets, col_offsets])
            first_frame_index = frame_indices[0]
            last_frame_index = frame_indices[-1]
            x_index = plane_position_names.index(
                'XOffsetInSlideCoordinateSystem'
            )
            x_origin = plane_position_values[first_frame_index, x_index]
            y_index = plane_position_names.index(
                'YOffsetInSlideCoordinateSystem'
            )
            y_origin = plane_position_values[first_frame_index, y_index]
            z_index = plane_position_names.index(
                'ZOffsetInSlideCoordinateSystem'
            )
            z_origin = plane_position_values[first_frame_index, z_index]

            if is_tiled:
                origin_item = Dataset()
                origin_item.XOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(x_origin)
                origin_item.YOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(y_origin)
                origin_item.ZOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(z_origin)
                self.TotalPixelMatrixOriginSequence = [origin_item]
                self.TotalPixelMatrixFocalPlanes = 1
                if total_pixel_matrix_size is None:
                    self.TotalPixelMatrixRows = int(
                        plane_position_values[last_frame_index, row_index] +
                        self.Rows - 1
                    )
                    self.TotalPixelMatrixColumns = int(
                        plane_position_values[last_frame_index, col_index] +
                        self.Columns - 1
                    )
                else:
                    self.TotalPixelMatrixRows = total_pixel_matrix_size[0]
                    self.TotalPixelMatrixColumns = total_pixel_matrix_size[1]
            else:
                transform = ImageToReferenceTransformer(
                    image_position=(x_origin, y_origin, z_origin),
                    image_orientation=(
                        plane_orientation[0].ImageOrientationSlide
                    ),
                    pixel_spacing=pixel_measures[0].PixelSpacing
                )
                center_image_coordinates = np.array(
                    [[self.Columns / 2, self.Rows / 2]],
                    dtype=float
                )
                center_reference_coordinates = transform(
                    center_image_coordinates
                )
                x_center = center_reference_coordinates[0, 0]
                y_center = center_reference_coordinates[0, 1]
                z_center = center_reference_coordinates[0, 2]
                center_item = Dataset()
                center_item.XOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(x_center)
                center_item.YOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(y_center)
                center_item.ZOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(z_center)
                self.ImageCenterPointCoordinatesSequence = [center_item]

    @staticmethod
    def _check_tiled_dimension_organization(
        dimension_organization_type: (
            DimensionOrganizationTypeValues |
            str |
            None
        ),
        is_tiled: bool,
        omit_empty_frames: bool,
        plane_positions: Sequence[PlanePositionSequence],
        tile_pixel_array: bool,
        rows: int,
        columns: int,
    ) -> DimensionOrganizationTypeValues | None:
        """Checks that the specified Dimension Organization Type is valid.

        Parameters
        ----------
        dimension_organization_type: Union[highdicom.enum.DimensionOrganizationTypeValues, str, None]
           The specified DimensionOrganizationType for the output Segmentation.
        is_tiled: bool
            Whether the source image is a tiled image.
        omit_empty_frames: bool
            Whether it was specified to omit empty frames.
        tile_pixel_array: bool
            Whether the total pixel matrix was passed.
        plane_positions: Sequence[highdicom.PlanePositionSequence]
            Plane positions of all frames.
        rows: int
            Number of rows in each frame of the segmentation image.
        columns: int
            Number of columns in each frame of the segmentation image.

        Returns
        -------
        Optional[highdicom.enum.DimensionOrganizationTypeValues]:
            DimensionOrganizationType to use for the output Segmentation.

        """  # noqa: E501
        if (
            dimension_organization_type ==
            DimensionOrganizationTypeValues.THREE_DIMENSIONAL_TEMPORAL
        ):
            raise ValueError(
                "Value of 'THREE_DIMENSIONAL_TEMPORAL' for "
                "parameter 'dimension_organization_type' is not supported."
            )
        if is_tiled and dimension_organization_type is None:
            dimension_organization_type = \
                DimensionOrganizationTypeValues.TILED_SPARSE

        if dimension_organization_type is not None:
            dimension_organization_type = DimensionOrganizationTypeValues(
                dimension_organization_type
            )
            tiled_dimension_organization_types = [
                DimensionOrganizationTypeValues.TILED_SPARSE,
                DimensionOrganizationTypeValues.TILED_FULL
            ]

            if (
                dimension_organization_type in
                tiled_dimension_organization_types
            ):
                if not is_tiled:
                    raise ValueError(
                        f"A value of {dimension_organization_type.value} "
                        'for parameter "dimension_organization_type" is '
                        'only valid if the source images are tiled.'
                    )

            if (
                dimension_organization_type ==
                DimensionOrganizationTypeValues.TILED_FULL
            ):
                # Need to check positions if they were not generated by us
                # when using tile_pixel_array
                if (
                    not tile_pixel_array and
                    not are_plane_positions_tiled_full(
                        plane_positions,
                        rows,
                        columns,
                    )
                ):
                    raise ValueError(
                        'A value of "TILED_FULL" for parameter '
                        '"dimension_organization_type" is not permitted '
                        'because the "plane_positions" of the segmentation '
                        'do not follow the relevant requirements. See '
                        'https://dicom.nema.org/medical/dicom/current/output/'
                        'chtml/part03/sect_C.7.6.17.3.html#sect_C.7.6.17.3 .'
                    )
                if omit_empty_frames:
                    raise ValueError(
                        'Parameter "omit_empty_frames" should be False if '
                        'using "dimension_organization_type" of "TILED_FULL".'
                    )

        return dimension_organization_type

    @classmethod
    def _check_and_cast_pixel_array(
        cls,
        pixel_array: np.ndarray,
        segment_numbers: np.ndarray,
        segmentation_type: SegmentationTypeValues,
        dtype: type,
    ) -> tuple[np.ndarray, SegmentsOverlapValues]:
        """Checks on the shape and data type of the pixel array.

        Also checks for overlapping segments and returns the result.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            The segmentation pixel array.
        segment_numbers: numpy.ndarray
            The segment numbers from the segment descriptions, in the order
            they were passed. 1D array of integers.
        segmentation_type: highdicom.seg.SegmentationTypeValues
            The segmentation_type parameter.
        dtype: type
            Pixel type of the output array.

        Returns
        -------
        pixel_array: numpyp.ndarray
            Input pixel array with the data type simplified if possible.
        segments_overlap: highdicom.seg.SegmentationOverlaps
            The value for the SegmentationOverlaps attribute, inferred from the
            pixel array.

        """
        # Note that for large array (common in pathology) the checks in this
        # method can take up a significant amount of the overall creation time.
        # As a result, this method is optimized for runtime efficiency at the
        # expense of simplicity. In particular, there are several common
        # special cases that have optimized implementations, and intermediate
        # results are reused wherever possible
        number_of_segments = len(segment_numbers)

        if pixel_array.ndim == 4:
            # Check that the number of segments in the array matches
            if pixel_array.shape[-1] != number_of_segments:
                raise ValueError(
                    'The number of segments in last dimension of the pixel '
                    f'array ({pixel_array.shape[-1]}) does not match the '
                    'number of described segments '
                    f'({number_of_segments}).'
                )

        if pixel_array.dtype in (np.bool_, np.uint8, np.uint16):

            if pixel_array.ndim == 3:
                # A label-map style array where pixel values represent
                # segment associations

                # The pixel values in the pixel array must all belong to
                # a described segment
                if len(
                    np.setxor1d(
                        np.arange(1, number_of_segments + 1),
                        segment_numbers,
                    )
                ) == 0:
                    # This is a common special case where segment numbers are
                    # consecutive and start at 1 (as is required for FRACTIONAL
                    # and BINARY segmentations). In this case it is sufficient
                    # to check the max pixel value, which is MUCH more
                    # efficient than calculating the set of unique values
                    has_undescribed_segments = (
                        pixel_array.max() > number_of_segments
                    )
                else:
                    # The general case, much slower
                    numbers_with_bg = np.concatenate(
                        [np.array([0]), segment_numbers]
                    )
                    has_undescribed_segments = len(
                        np.setdiff1d(pixel_array, numbers_with_bg)
                    ) != 0

                if has_undescribed_segments:
                    raise ValueError(
                        'Pixel array contains segments that lack '
                        'descriptions.'
                    )

                # By construction of the pixel array, we know that the segments
                # cannot overlap
                segments_overlap = SegmentsOverlapValues.NO
            else:
                max_pixel = pixel_array.max()

                # Pixel array is 4D where each segment is stacked down
                # the last dimension
                # In this case, each segment of the pixel array should be binary
                if max_pixel > 1:
                    raise ValueError(
                        'When passing a 4D stack of segments with an integer '
                        'pixel type, the pixel array must be binary.'
                    )

                # Need to check whether or not segments overlap
                if max_pixel == 0:
                    # Empty segments can't overlap (this skips an unnecessary
                    # further test)
                    segments_overlap = SegmentsOverlapValues.NO
                elif pixel_array.shape[-1] == 1:
                    # A single segment does not overlap
                    segments_overlap = SegmentsOverlapValues.NO
                else:
                    sum_over_segments = pixel_array.sum(axis=-1)
                    if np.any(sum_over_segments > 1):
                        segments_overlap = SegmentsOverlapValues.YES
                    else:
                        segments_overlap = SegmentsOverlapValues.NO

        elif pixel_array.dtype in (np.float32, np.float64):
            unique_values = np.unique(pixel_array)
            if np.min(unique_values) < 0.0 or np.max(unique_values) > 1.0:
                raise ValueError(
                    'Floating point pixel array values must be in the '
                    'range [0, 1].'
                )
            if segmentation_type in (
                SegmentationTypeValues.BINARY,
                SegmentationTypeValues.LABELMAP,
            ):
                non_boolean_values = np.logical_and(
                    unique_values > 0.0,
                    unique_values < 1.0
                )
                if np.any(non_boolean_values):
                    raise ValueError(
                        'Floating point pixel array values must be either '
                        '0.0 or 1.0 in case of BINARY or LABELMAP segmentation '
                        'type.'
                    )
                pixel_array = pixel_array.astype(dtype)

                # Need to check whether or not segments overlap
                if len(unique_values) == 1 and unique_values[0] == 0.0:
                    # All pixels are zero: there can be no overlap
                    segments_overlap = SegmentsOverlapValues.NO
                elif pixel_array.ndim == 3 or pixel_array.shape[-1] == 1:
                    # A single segment does not overlap
                    segments_overlap = SegmentsOverlapValues.NO
                elif pixel_array.sum(axis=-1).max() > 1:
                    segments_overlap = SegmentsOverlapValues.YES
                else:
                    segments_overlap = SegmentsOverlapValues.NO
            else:
                if (pixel_array.ndim == 3) or (pixel_array.shape[-1] == 1):
                    # A single segment does not overlap
                    segments_overlap = SegmentsOverlapValues.NO
                else:
                    # A truly fractional segmentation with multiple segments.
                    # Unclear how overlap should be interpreted in this case
                    segments_overlap = SegmentsOverlapValues.UNDEFINED
        else:
            raise TypeError('Pixel array has an invalid data type.')

        # Combine segments to create a labelmap image if needed
        if segmentation_type == SegmentationTypeValues.LABELMAP:
            if segments_overlap == SegmentsOverlapValues.YES:
                raise ValueError(
                    'It is not possible to store a Segmentation with '
                    'SegmentationType "LABELMAP" if segments overlap.'
                )

            if pixel_array.ndim == 4:
                pixel_array = cls._combine_segments(
                    pixel_array,
                    labelmap_dtype=dtype
                )
            else:
                pixel_array = pixel_array.astype(dtype)

        return pixel_array, segments_overlap

    @staticmethod
    def _get_nonempty_plane_indices(
        pixel_array: np.ndarray
    ) -> tuple[list[int], bool]:
        """Get a list of all indices of original planes that are non-empty.

        Empty planes (without any positive pixels in any of the segments) do
        not need to be included in the segmentation image. This method finds a
        list of indices of the input frames that are non-empty, and therefore
        should be included in the segmentation image.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Segmentation pixel array

        Returns
        -------
        included_plane_indices : List[int]
            List giving for each plane position in the resulting segmentation
            image the index of the corresponding frame in the original pixel
            array.
        is_empty: bool
            Whether the entire image is empty. If so, empty frames should not
            be omitted.

        """
        # This list tracks which source image each non-empty frame came from
        source_image_indices = [
            i for i, frm in enumerate(pixel_array)
            if np.any(frm)
        ]

        if len(source_image_indices) == 0:
            logger.warning(
                'Encoding an empty segmentation with "omit_empty_frames" '
                'set to True. Reverting to encoding all frames since omitting '
                'all frames is not possible.'
            )
            return (list(range(pixel_array.shape[0])), True)

        return (source_image_indices, False)

    @staticmethod
    def _combine_segments(
        pixel_array: np.ndarray,
        labelmap_dtype: type,
    ):
        """Combine multiple segments into a labelmap.
        Parameters
        ----------
        pixel_array: np.ndarray
            Segmentation pixel array with segments stacked along dimension 3.
            Should consist of only values 0 and 1.
        labelmap_dtype: type
            Numpy data type to use for the output array and intermediate
            calculations.
        Returns
        -------
        pixel_array: np.ndarray
            A 3D output array with consisting of the original segments combined
            into a labelmap.
        """
        if pixel_array.shape[3] == 1:
            # Optimization in case of one class
            return pixel_array[:, :, :, 0].astype(labelmap_dtype)

        # Take the indices along axis 3. However this does not
        # distinguish between pixels that are empty and pixels that
        # have class 1. Therefore need to multiply this by the max
        # value
        # Carefully control the dtype here to avoid creating huge
        # interemdiate arrays
        indices = np.zeros(pixel_array.shape[:3], dtype=labelmap_dtype)
        indices = pixel_array.argmax(axis=3, out=indices) + 1
        is_non_empty = np.zeros(pixel_array.shape[:3], dtype=labelmap_dtype)
        is_non_empty = pixel_array.max(axis=3, out=is_non_empty)
        pixel_array = indices * is_non_empty

        return pixel_array

    @staticmethod
    def _get_nonempty_tile_indices(
        pixel_array: np.ndarray,
        plane_positions: Sequence[PlanePositionSequence],
        rows: int,
        columns: int,
    ) -> tuple[list[int], bool]:
        """Get a list of all indices of tile locations that are non-empty.

        This is similar to _get_nonempty_plane_indices, but works on a total
        pixel matrix rather than a set of frames. Empty planes (without any
        positive pixels in any of the segments) do not need to be included in
        the segmentation image. This method finds a list of indices of the
        input frames that are non-empty, and therefore should be included in
        the segmentation image.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Segmentation pixel array
        plane_positions: Sequence[highdicom.PlanePositionSequence]
            Plane positions of each tile.
        rows: int
            Number of rows in each tile.
        columns: int
            Number of columns in each tile.

        Returns
        -------
        included_plane_indices : List[int]
            List giving for each plane position in the resulting segmentation
            image the index of the corresponding frame in the original pixel
            array.
        is_empty: bool
            Whether the entire image is empty. If so, empty frames should not
            be omitted.

        """
        # This list tracks which source image each non-empty frame came from
        source_image_indices = [
            i for i, pos in enumerate(plane_positions)
            if np.any(
                get_tile_array(
                    pixel_array[0],
                    row_offset=pos[0].RowPositionInTotalImagePixelMatrix,
                    column_offset=pos[0].ColumnPositionInTotalImagePixelMatrix,
                    tile_rows=rows,
                    tile_columns=columns,
                )
            )
        ]

        if len(source_image_indices) == 0:
            logger.warning(
                'Encoding an empty segmentation with "omit_empty_frames" '
                'set to True. Reverting to encoding all frames since omitting '
                'all frames is not possible.'
            )
            return (list(range(len(plane_positions))), True)

        return (source_image_indices, False)

    @staticmethod
    def _get_segment_pixel_array(
        pixel_array: np.ndarray,
        segment_number: int,
        described_segment_numbers: np.ndarray,
        segmentation_type: SegmentationTypeValues,
        max_fractional_value: int,
        dtype: type,
    ) -> np.ndarray:
        """Get pixel data array for a specific segment and plane.

        This is a helper method used during the constructor. Note that the
        pixel array is expected to have been processed using the
        ``_check_and_cast_pixel_array`` method before being passed to this
        method.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Segmentation pixel array containing all segments for a single plane.
            Array is therefore either (Rows x Columns x Segments) or (Rows x
            Columns) in case of a "label map" style array.
        segment_number: int
            The segment of interest.
        described_segment_numbers: np.ndarray
            Array of all segment numbers in the segmentation.
        segmentation_type: highdicom.seg.SegmentationTypeValues
            Desired output segmentation type.
        max_fractional_value: int
            Value for scaling FRACTIONAL segmentations.
        dtype: type
            Data type of the returned pixel array.

        Returns
        -------
        numpy.ndarray:
            Pixel data array consisting of pixel data for a single segment for
            a single plane. Output array has the specified dtype and binary
            values (0 or 1).

        """
        if pixel_array.dtype in (np.float32, np.float64):
            # Based on the previous checks and casting, if we get here the
            # output is a FRACTIONAL segmentation. Floating-point numbers must
            # be mapped to 8-bit integers in the range [0,
            # max_fractional_value].
            if pixel_array.ndim == 3:
                segment_array = pixel_array[:, :, segment_number - 1]
            else:
                segment_array = pixel_array
            segment_array = np.around(
                segment_array * float(max_fractional_value)
            )
            segment_array = segment_array.astype(dtype)
        else:
            if pixel_array.ndim == 2:
                # "Label maps" that must be converted to binary masks.
                if np.array_equal(described_segment_numbers, np.array([1])):
                    # We wish to avoid unnecessary comparison or casting
                    # operations here, for efficiency reasons. If there is only
                    # a single segment with value 1, the label map pixel array
                    # is already correct
                    if pixel_array.dtype != dtype:
                        segment_array = pixel_array.astype(dtype)
                    else:
                        segment_array = pixel_array
                else:
                    segment_array = (
                        pixel_array == segment_number
                    ).astype(dtype)
            else:
                segment_array = pixel_array[:, :, segment_number - 1]
                if segment_array.dtype != dtype:
                    segment_array = segment_array.astype(dtype)

            # It may happen that a binary valued array is passed that should be
            # stored as a fractional segmentation. In this case, we also need
            # to stretch pixel values to 8-bit unsigned integer range by
            # multiplying with the maximum fractional value.
            if segmentation_type == SegmentationTypeValues.FRACTIONAL:
                # Avoid an unnecessary multiplication operation if max
                # fractional value is 1
                if int(max_fractional_value) != 1:
                    segment_array *= int(max_fractional_value)

        return segment_array

    @staticmethod
    def _get_pffg_item(
        segment_number: int | None,
        dimension_index_values: list[int],
        plane_position: PlanePositionSequence,
        source_images: list[Dataset],
        source_image_index: int,
        are_spatial_locations_preserved: bool,
        coordinate_system: CoordinateSystemNames | None,
        is_multiframe: bool,
    ) -> Dataset:
        """Get a single item of the Per Frame Functional Groups Sequence.

        This is a helper method used in the constructor.

        Parameters
        ----------
        segment_number: Optional[int]
            Segment number of this segmentation frame. If None, this is a
            LABELMAP segmentation in which each frame has no segment number.
        dimension_index_values: List[int]
            Dimension index values (except segment number) for this frame.
        plane_position: highdicom.seg.PlanePositionSequence
            Plane position of this frame.
        source_images: List[Dataset]
            Full list of source images.
        source_image_index: int
            Index of this frame in the original list of source images.
        are_spatial_locations_preserved: bool
            Whether spatial locations are preserved between the segmentation
            and the source images.
        coordinate_system: Optional[highdicom.CoordinateSystemNames]
            Coordinate system used, if any.
        is_multiframe: bool
            Whether source images are multiframe.

        Returns
        -------
        pydicom.Dataset
            Dataset representing the item of the
            Per Frame Functional Groups Sequence for this segmentation frame.

        """
        # NB this function is called many times in a loop when there are a
        # large number of frames, and has been observed to dominate the
        # creation time of some segmentations. Therefore we use low-level
        # pydicom primitives to improve performance as much as possible
        pffg_item = Dataset()
        frame_content_item = Dataset()

        if segment_number is None:
            all_index_values = dimension_index_values
        else:
            all_index_values = [int(segment_number)] + dimension_index_values

        frame_content_item.add(
            DataElement(
                0x00209157,  # DimensionIndexValues
                'UL',
                all_index_values,
            )
        )

        if segment_number is None and coordinate_system is None:
            # If this is an labelmap segmentation of an image that has no frame
            # of reference, we need to create a dummy frame label to be pointed
            # to as a dimension index because there is nothing else appropriate
            # to use
            frame_content_item.add(
                DataElement(
                    0x00209453,  # FrameLabel
                    'LO',
                    "Segmentation Frame",
                )
            )

        pffg_item.add(
            DataElement(
                0x00209111,  # FrameContentSequence
                'SQ',
                [frame_content_item]
            )
        )

        if coordinate_system is not None:
            if coordinate_system == CoordinateSystemNames.SLIDE:
                pffg_item.add(
                    DataElement(
                        0x0048021a,  # PlanePositionSlideSequence
                        'SQ',
                        plane_position
                    )
                )
            else:
                pffg_item.add(
                    DataElement(
                        0x00209113,  # PlanePositionSequence
                        'SQ',
                        plane_position
                    )
                )

        if are_spatial_locations_preserved:
            derivation_image_item = Dataset()
            derivation_image_item.add(
                DataElement(
                    0x00089215,  # DerivationCodeSequence
                    'SQ',
                    [_DERIVATION_CODE]
                )
            )

            derivation_src_img_item = Dataset()
            if is_multiframe:
                # A single multi-frame source image
                src_img_item = source_images[0]
                # Frame numbers are one-based
                derivation_src_img_item.add(
                    DataElement(
                        0x00081160,  # ReferencedFrameNumber
                        'IS',
                        source_image_index + 1
                    )
                )
            else:
                # Multiple single-frame source images
                src_img_item = source_images[source_image_index]
            derivation_src_img_item.add(
                DataElement(
                    0x00081150,  # ReferencedSOPClassUID
                    'UI',
                    src_img_item[0x00080016].value  # SOPClassUID
                )
            )
            derivation_src_img_item.add(
                DataElement(
                    0x00081155,  # ReferencedSOPInstanceUID
                    'UI',
                    src_img_item[0x00080018].value  # SOPInstanceUID
                )
            )
            derivation_src_img_item.add(
                DataElement(
                    0x0040a170,  # PurposeOfReferenceCodeSequence
                    'SQ',
                    [_PURPOSE_CODE]
                )
            )
            derivation_src_img_item.add(
                DataElement(
                    0x0028135a,  # SpatialLocationsPreserved
                    'CS',
                    'YES'
                )
            )
            derivation_image_item.add(
                DataElement(
                    0x00082112,  # SourceImageSequence
                    'SQ',
                    [derivation_src_img_item]
                )
            )
            pffg_item.add(
                DataElement(
                    0x00089124,  # DerivationImageSequence
                    'SQ',
                    [derivation_image_item]
                )
            )
        else:
            # Determining the source images that map to the frame is not
            # always trivial. Since DerivationImageSequence is a type 2
            # attribute, we leave its value empty.
            pffg_item.add(
                DataElement(
                    0x00089124,  # DerivationImageSequence
                    'SQ',
                    []
                )
            )
            logger.debug('spatial locations not preserved')

        if segment_number is not None:
            identification = Dataset()
            identification.add(
                DataElement(
                    0x0062000b,  # ReferencedSegmentNumber
                    'US',
                    int(segment_number)
                )
            )
            pffg_item.add(
                DataElement(
                    0x0062000a,  # SegmentIdentificationSequence
                    'SQ',
                    [identification]
                )
            )

        return pffg_item

    def _encode_pixels_native(self, planes: np.ndarray) -> bytes:
        """Encode pixel planes using a native transfer syntax.

        Parameters
        ----------
        planes: numpy.ndarray
            Array representing one or more segmentation image planes. If
            multiple image planes, planes stacked down the first dimension
            (index 0).

        Returns
        -------
        bytes
            Encoded pixels

        """
        if self.SegmentationType == SegmentationTypeValues.BINARY.value:
            return pack_bits(planes, pad=False)
        else:
            return planes.tobytes()

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Create instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Segmentation image.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.seg.Segmentation
            Representation of the supplied dataset as a highdicom
            Segmentation.

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        _check_little_endian(dataset)
        # Checks on integrity of input dataset
        if dataset.SOPClassUID not in (
            '1.2.840.10008.5.1.4.1.1.66.4',
            '1.2.840.10008.5.1.4.1.1.66.7',
        ):
            raise ValueError('Dataset is not a Segmentation.')
        if copy:
            seg = deepcopy(dataset)
        else:
            seg = dataset
        seg.__class__ = cls

        sf_groups = seg.SharedFunctionalGroupsSequence[0]
        if hasattr(seg, 'PlaneOrientationSequence'):
            plane_ori_seq = sf_groups.PlaneOrientationSequence[0]
            if hasattr(plane_ori_seq, 'ImageOrientationSlide'):
                seg._coordinate_system = CoordinateSystemNames.SLIDE
            elif hasattr(plane_ori_seq, 'ImageOrientationPatient'):
                seg._coordinate_system = CoordinateSystemNames.PATIENT
            else:
                seg._coordinate_system = None
        else:
            seg._coordinate_system = None

        # Convert contained items to highdicom types
        # Segment descriptions
        seg.SegmentSequence = [
            SegmentDescription.from_dataset(ds, copy=False)
            for ds in seg.SegmentSequence
        ]

        # Shared functional group elements
        if hasattr(sf_groups, 'PlanePositionSequence'):
            plane_pos = PlanePositionSequence.from_sequence(
                sf_groups.PlanePositionSequence,
                copy=False,
            )
            sf_groups.PlanePositionSequence = plane_pos
        if hasattr(sf_groups, 'PlaneOrientationSequence'):
            plane_ori = PlaneOrientationSequence.from_sequence(
                sf_groups.PlaneOrientationSequence,
                copy=False,
            )
            sf_groups.PlaneOrientationSequence = plane_ori
        if hasattr(sf_groups, 'PixelMeasuresSequence'):
            pixel_measures = PixelMeasuresSequence.from_sequence(
                sf_groups.PixelMeasuresSequence,
                copy=False,
            )
            sf_groups.PixelMeasuresSequence = pixel_measures

        # Per-frame functional group items
        if hasattr(seg, 'PerFrameFunctionalGroupsSequence'):
            for pffg_item in seg.PerFrameFunctionalGroupsSequence:
                if hasattr(pffg_item, 'PlanePositionSequence'):
                    plane_pos = PlanePositionSequence.from_sequence(
                        pffg_item.PlanePositionSequence,
                        copy=False
                    )
                    pffg_item.PlanePositionSequence = plane_pos
                if hasattr(pffg_item, 'PlaneOrientationSequence'):
                    plane_ori = PlaneOrientationSequence.from_sequence(
                        pffg_item.PlaneOrientationSequence,
                        copy=False,
                    )
                    pffg_item.PlaneOrientationSequence = plane_ori
                if hasattr(pffg_item, 'PixelMeasuresSequence'):
                    pixel_measures = PixelMeasuresSequence.from_sequence(
                        pffg_item.PixelMeasuresSequence,
                        copy=False,
                    )
                    pffg_item.PixelMeasuresSequence = pixel_measures

        seg = super().from_dataset(seg, copy=False)

        return cast(Self, seg)

    @property
    def segmentation_type(self) -> SegmentationTypeValues:
        """highdicom.seg.SegmentationTypeValues: Segmentation type."""
        return SegmentationTypeValues(self.SegmentationType)

    @property
    def segmentation_fractional_type(
        self
    ) -> SegmentationFractionalTypeValues | None:
        """
        highdicom.seg.SegmentationFractionalTypeValues:
            Segmentation fractional type.

        """
        if not hasattr(self, 'SegmentationFractionalType'):
            return None
        return SegmentationFractionalTypeValues(
            self.SegmentationFractionalType
        )

    def iter_segments(self):
        """Iterates over segments in this segmentation image.

        Returns
        -------
        Iterator[Tuple[numpy.ndarray, Tuple[pydicom.dataset.Dataset, ...], pydicom.dataset.Dataset]]
            For each segment in the Segmentation image instance, provides the
            Pixel Data frames representing the segment, items of the Per-Frame
            Functional Groups Sequence describing the individual frames, and
            the item of the Segment Sequence describing the segment

        """  # noqa
        return iter_segments(self)

    @property
    def number_of_segments(self) -> int:
        """int: The number of non-background segments in this SEG image."""
        if hasattr(self, 'PixelPaddingValue'):
            return len(self.segment_numbers) - 1
        return len(self.SegmentSequence)

    @property
    def segment_numbers(self) -> list[int]:
        """List[int]: The segment numbers of non-background segments present
        in the SEG image."""
        if hasattr(self, 'PixelPaddingValue'):
            return [
                desc.SegmentNumber for desc in self.SegmentSequence
                if desc.SegmentNumber != self.PixelPaddingValue
            ]
        else:
            return [
                desc.SegmentNumber for desc in self.SegmentSequence
            ]

    def get_segment_description(
        self,
        segment_number: int
    ) -> SegmentDescription:
        """Get segment description for a segment.

        Parameters
        ----------
        segment_number: int
            Segment number for the segment.

        Returns
        -------
        highdicom.seg.SegmentDescription
            Description of the given segment.

        """
        for desc in self.SegmentSequence:
            if desc.segment_number == segment_number:
                return desc

        raise IndexError(
            f'{segment_number} is an invalid segment number for this '
            'dataset.'
        )

    def get_segment_numbers(
        self,
        segment_label: str | None = None,
        segmented_property_category: Code | CodedConcept | None = None,
        segmented_property_type: Code | CodedConcept | None = None,
        algorithm_type: SegmentAlgorithmTypeValues | str | None = None,
        tracking_uid: str | None = None,
        tracking_id: str | None = None,
    ) -> list[int]:
        """Get a list of non-background segment numbers with given criteria.

        Any number of optional filters may be provided. A segment must match
        all provided filters to be included in the returned list.

        Parameters
        ----------
        segment_label: Union[str, None], optional
            Segment label filter to apply.
        segmented_property_category: Union[Code, CodedConcept, None], optional
            Segmented property category filter to apply.
        segmented_property_type: Union[Code, CodedConcept, None], optional
            Segmented property type filter to apply.
        algorithm_type: Union[SegmentAlgorithmTypeValues, str, None], optional
            Segmented property type filter to apply.
        tracking_uid: Union[str, None], optional
            Tracking unique identifier filter to apply.
        tracking_id: Union[str, None], optional
            Tracking identifier filter to apply.

        Returns
        -------
        List[int]
            List of all non-background segment numbers matching the provided
            criteria.

        Examples
        --------

        Get segment numbers of all segments that both represent tumors and were
        generated by an automatic algorithm from a segmentation object ``seg``:

        >>> from pydicom.sr.codedict import codes
        >>> from highdicom.seg import SegmentAlgorithmTypeValues, Segmentation
        >>> from pydicom import dcmread
        >>> ds = dcmread('data/test_files/seg_image_sm_control.dcm')
        >>> seg = Segmentation.from_dataset(ds)
        >>> segment_numbers = seg.get_segment_numbers(
        ...     segmented_property_type=codes.SCT.ConnectiveTissue,
        ...     algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC
        ... )
        >>> segment_numbers
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        Get segment numbers of all segments identified by a given
        institution-specific tracking ID:

        >>> segment_numbers = seg.get_segment_numbers(
        ...     tracking_id='Segment #4'
        ... )
        >>> segment_numbers
        [4]

        Get segment numbers of all segments identified a globally unique
        tracking UID:

        >>> uid = '1.2.826.0.1.3680043.8.498.42540123542017542395135803252098380233'
        >>> segment_numbers = seg.get_segment_numbers(tracking_uid=uid)
        >>> segment_numbers
        [13]

        """  # noqa: E501
        filter_funcs = []
        if segment_label is not None:
            filter_funcs.append(
                lambda desc: desc.segment_label == segment_label
            )
        if segmented_property_category is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_category == segmented_property_category
            )
        if segmented_property_type is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_type == segmented_property_type
            )
        if algorithm_type is not None:
            algo_type = SegmentAlgorithmTypeValues(algorithm_type)
            filter_funcs.append(
                lambda desc:
                SegmentAlgorithmTypeValues(desc.algorithm_type) == algo_type
            )
        if tracking_uid is not None:
            filter_funcs.append(
                lambda desc: desc.tracking_uid == tracking_uid
            )
        if tracking_id is not None:
            filter_funcs.append(
                lambda desc: desc.tracking_id == tracking_id
            )
        if hasattr(self, 'PixelPaddingValue'):
            filter_funcs.append(
                lambda desc: desc.segment_number != self.PixelPaddingValue
            )

        return [
            desc.segment_number
            for desc in self.SegmentSequence
            if all(f(desc) for f in filter_funcs)
        ]

    def get_tracking_ids(
        self,
        segmented_property_category: Code | CodedConcept | None = None,
        segmented_property_type: Code | CodedConcept | None = None,
        algorithm_type: SegmentAlgorithmTypeValues | str | None = None
    ) -> list[tuple[str, UID]]:
        """Get all unique tracking identifiers in this SEG image.

        Any number of optional filters may be provided. A segment must match
        all provided filters to be included in the returned list.

        The tracking IDs and the accompanying tracking UIDs are returned
        in a list of tuples.

        Note that the order of the returned list is not significant and will
        not in general match the order of segments.

        Parameters
        ----------
        segmented_property_category: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Segmented property category filter to apply.
        segmented_property_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Segmented property type filter to apply.
        algorithm_type: Union[highdicom.seg.SegmentAlgorithmTypeValues, str, None], optional
            Segmented property type filter to apply.

        Returns
        -------
        List[Tuple[str, pydicom.uid.UID]]
            List of all unique (Tracking Identifier, Unique Tracking Identifier)
            tuples that are referenced in segment descriptions in this
            Segmentation image that match all provided filters.

        Examples
        --------

        Read in an example segmentation image in the highdicom test data:

        >>> import highdicom as hd
        >>> from pydicom.sr.codedict import codes
        >>>
        >>> seg = hd.seg.segread('data/test_files/seg_image_ct_binary_overlap.dcm')

        List the tracking IDs and UIDs present in the segmentation image:

        >>> sorted(seg.get_tracking_ids(), reverse=True)  # otherwise its a random order
        [('Spine', '1.2.826.0.1.3680043.10.511.3.10042414969629429693880339016394772'), ('Bone', '1.2.826.0.1.3680043.10.511.3.83271046815894549094043330632275067')]

        >>> for seg_num in seg.segment_numbers:
        ...     desc = seg.get_segment_description(seg_num)
        ...     print(desc.segmented_property_type.meaning)
        Bone
        Spine

        List tracking IDs only for those segments with a segmented property
        category of 'Spine':

        >>> seg.get_tracking_ids(segmented_property_type=codes.SCT.Spine)
        [('Spine', '1.2.826.0.1.3680043.10.511.3.10042414969629429693880339016394772')]

        """  # noqa: E501
        filter_funcs = []
        if segmented_property_category is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_category == segmented_property_category
            )
        if segmented_property_type is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_type == segmented_property_type
            )
        if algorithm_type is not None:
            algo_type = SegmentAlgorithmTypeValues(algorithm_type)
            filter_funcs.append(
                lambda desc:
                SegmentAlgorithmTypeValues(desc.algorithm_type) == algo_type
            )

        return list({
            (desc.tracking_id, UID(desc.tracking_uid))
            for desc in self.SegmentSequence
            if desc.tracking_id is not None and
            desc.tracking_uid is not None and
            all(f(desc) for f in filter_funcs)
        })

    @property
    def segmented_property_categories(self) -> list[CodedConcept]:
        """Get all unique non-background segmented property categories.

        Returns
        -------
        List[CodedConcept]
            All unique non-background segmented property categories referenced
            in segment descriptions in this SEG image.

        """
        categories = []
        for desc in self.SegmentSequence:
            if (
                'PixelPaddingValue' in self and
                desc.segment_number == self.PixelPaddingValue
            ):
                # Skip background segment
                continue

            if desc.segmented_property_category not in categories:
                categories.append(desc.segmented_property_category)

        return categories

    @property
    def segmented_property_types(self) -> list[CodedConcept]:
        """Get all unique non-background segmented property types.

        Returns
        -------
        List[CodedConcept]
            All unique non-background segmented property types referenced in
            segment descriptions in this SEG image.

        """
        types = []
        for desc in self.SegmentSequence:
            if (
                'PixelPaddingValue' in self and
                desc.segment_number == self.PixelPaddingValue
            ):
                # Skip background segment
                continue

            if desc.segmented_property_type not in types:
                types.append(desc.segmented_property_type)

        return types

    def _get_pixels_by_seg_frame(
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
        segment_numbers: np.ndarray,
        combine_segments: bool = False,
        relabel: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: type | str | np.dtype | None = None,
        apply_palette_color_lut: bool = False,
        apply_icc_profile: bool | None = None,
    ) -> np.ndarray:
        """Construct a segmentation array given an array of frame numbers.

        The output array is either 4D (combine_segments=False) or 3D
        (combine_segments=True), where dimensions are frames x rows x columns x
        segments.

        Parameters
        ----------
        output_shape: Union[int, Tuple[int, int]]
            Shape of the output array. If an integer, this is the
            number of frames in the output array and the number of rows and
            columns are taken to match those of each segmentation frame. If a
            tuple of integers, it contains the number of (rows, columns) in the
            output array and there is no frame dimension (this is the tiled
            case). Note in either case, the segments dimension (if relevant) is
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
        segment_numbers: numpy.ndarray
            One dimensional numpy array containing segment numbers
            corresponding to the columns of the seg frames matrix.
        combine_segments: bool
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, np.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.
        apply_palette_color_lut: bool, optional
            If True, apply the palette color LUT to give RGB output values.
            This is only valid for ``"LABELMAP"`` segmentations that contain
            palette color LUT information, and only when ``combine_segments``
            is ``True`` and ``relabel`` is ``False``.
        apply_icc_profile: bool, optional
            If True apply an ICC profile to the output and require it to be
            present. If None, apply an ICC profile if found but do not require
            it to be present. If False, never apply an ICC profile. Only
            possible when ``apply_palette_color_lut`` is True.

        Returns
        -------
        pixel_array: np.ndarray
            Segmentation pixel array

        """  # noqa: E501
        if not np.all(np.isin(segment_numbers, self.segment_numbers)):
            raise ValueError(
                'Segment numbers array contains invalid values.'
            )

        # Determine output type
        if combine_segments:
            max_output_val = (
                segment_numbers.shape[0] if relabel else segment_numbers.max()
            )
        else:
            max_output_val = 1

        will_be_rescaled = (
            rescale_fractional and
            self.segmentation_type == SegmentationTypeValues.FRACTIONAL and
            not combine_segments
        )
        if dtype is None:
            if will_be_rescaled:
                dtype = np.float32
            else:
                dtype = _get_unsigned_dtype(max_output_val)
        dtype = np.dtype(dtype)

        # Check dtype is suitable
        if dtype.kind not in ('u', 'i', 'f', 'b'):
            raise ValueError(
                f'Data type "{dtype}" is not suitable.'
            )

        _check_numpy_value_representation(max_output_val, dtype)
        num_output_segments = len(segment_numbers)

        if not isinstance(apply_palette_color_lut, bool):
            raise ValueError(
                "'apply_palette_color_lut' must have type bool"
            )
        if apply_palette_color_lut:
            if not combine_segments or relabel:
                raise ValueError(
                    "'apply_palette_color_lut' requires that "
                    "'combine_segments' is True and relabel is False."
                )
        else:
            apply_icc_profile = False
        if apply_icc_profile and not apply_palette_color_lut:
            raise ValueError(
                "'apply_icc_profile' requires that 'apply_palette_color_lut' "
                "is True."
            )

        if self.segmentation_type == SegmentationTypeValues.LABELMAP:

            if apply_palette_color_lut:
                # Remap is handled by the frame transform
                need_remap = False
                # Any segment not requested is mapped to zero
                # note that this assumes the background is RGB(0, 0, 0)
                remove_palette_color_values = [
                    s for s in self.segment_numbers
                    if s not in segment_numbers
                ]
            else:
                if not combine_segments or relabel:
                    # If combining segments (i.e. if expanding segments),
                    # always remap the segments for the one-hot later on.
                    output_segment_numbers = np.arange(1, len(segment_numbers))
                    need_remap = not np.array_equal(
                        segment_numbers,
                        output_segment_numbers
                    )
                else:
                    # Combining segments without relabelling. Need to remap if
                    # any existing segments are not requested, but order is not
                    # important
                    need_remap = len(
                        np.setxor1d(
                            segment_numbers,
                            self.segment_numbers
                        )
                    ) > 1
                remove_palette_color_values = None

            intermediate_dtype = (
                _get_unsigned_dtype(self.BitsStored)
                if need_remap else dtype
            )

            out_array = self._get_pixels_by_frame(
                spatial_shape=spatial_shape,
                indices_iterator=indices_iterator,
                apply_real_world_transform=False,
                apply_modality_transform=False,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
                remove_palette_color_values=remove_palette_color_values,
                palette_color_background_index=self.get(
                    'PixelPaddingValue',
                    0
                ),
                dtype=intermediate_dtype,
            )

            if need_remap:
                num_input_segments = max(self.segment_numbers) + 1
                stored_bg_val = self.get('PixelPaddingValue', 0)
                num_input_segments = max(stored_bg_val + 1, num_input_segments)
                remap_dtype = (
                    dtype if combine_segments else intermediate_dtype
                )
                remapping = np.zeros(num_input_segments + 1, dtype=remap_dtype)
                if combine_segments and not relabel:
                    # A remapping that just sets unused segments to the
                    # background value
                    for s in range(num_input_segments):
                        remapping[s] = (
                            s if s in segment_numbers
                            else stored_bg_val
                        )
                else:
                    # A remapping that applies relabelling logic
                    output_bg_val = 0  # relabel changes background value
                    for s in range(num_input_segments + 1):
                        remapping[s] = (
                            np.nonzero(segment_numbers == s)[0][0] + 1
                            if s in segment_numbers
                            else output_bg_val
                        )

                out_array = remapping[out_array]

            if not combine_segments:
                # Obscure trick to calculate one-hot. By this point, whatever
                # segments were requested will have been remapped to the
                # numbers 1, 2, ... in the order expected in the output
                # channels
                shape = out_array.shape
                flat_array = out_array.flatten()
                out_array = np.eye(
                    num_output_segments + 1,
                    dtype=dtype,
                )[flat_array]

                out_shape = (*shape, num_output_segments)

                # Remove the background segment (channel 0)
                out_array = out_array[:, 1:].reshape(out_shape)

            return out_array

        if will_be_rescaled:
            intermediate_dtype = np.uint8
            if dtype.kind != 'f':
                raise ValueError(
                    'If rescaling a fractional segmentation, the output dtype '
                    'must be a floating-point type.'
                )
        else:
            intermediate_dtype = dtype

        if combine_segments:
            # Check whether segmentation is binary, or fractional with only
            # binary values
            if self.segmentation_type == SegmentationTypeValues.FRACTIONAL:
                if not rescale_fractional:
                    raise ValueError(
                        'In order to combine segments of a FRACTIONAL '
                        'segmentation image, argument "rescale_fractional" '
                        'must be set to True.'
                    )

            # Initialize empty pixel array
            full_output_shape = (
                spatial_shape
                if isinstance(spatial_shape, tuple)
                else (spatial_shape, self.Rows, self.Columns)
            )
            out_array = np.zeros(
                full_output_shape,
                dtype=intermediate_dtype
            )

            # Loop over the supplied iterable
            for (
                frame_index,
                input_indexer,
                output_indexer,
                seg_n
            ) in indices_iterator:
                pix_value = intermediate_dtype.type(seg_n[0])

                pixel_array = self.get_stored_frame(frame_index + 1)
                pixel_array = pixel_array[input_indexer]

                if self.segmentation_type == SegmentationTypeValues.FRACTIONAL:
                    # Combining fractional segs is only possible if there are
                    # two unique values in the array: 0 and
                    # MaximumFractionalValue
                    is_binary = np.isin(
                        np.unique(pixel_array),
                        np.array([0, self.MaximumFractionalValue]),
                        assume_unique=True
                    ).all()
                    if not is_binary:
                        raise ValueError(
                            'Combining segments of a FRACTIONAL segmentation '
                            'is only possible if the pixel array contains only '
                            'zeros and the specified MaximumFractionalValue '
                            f'({self.MaximumFractionalValue}).'
                        )
                    pixel_array = pixel_array // self.MaximumFractionalValue
                    if pixel_array.dtype != np.uint8:
                        pixel_array = pixel_array.astype(np.uint8)

                if not skip_overlap_checks:
                    if np.any(
                        np.logical_and(
                            pixel_array > 0,
                            out_array[output_indexer] > 0
                        )
                    ):
                        raise RuntimeError(
                            "Cannot combine segments because segments "
                            "overlap."
                        )
                out_array[output_indexer] = np.maximum(
                    pixel_array * pix_value,
                    out_array[output_indexer]
                )

        else:
            out_array = self._get_pixels_by_frame(
                spatial_shape=spatial_shape,
                indices_iterator=indices_iterator,
                channel_shape=(num_output_segments, ),
                apply_real_world_transform=False,
                apply_modality_transform=False,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
                dtype=intermediate_dtype,
            )

            if rescale_fractional:
                if self.segmentation_type == SegmentationTypeValues.FRACTIONAL:
                    if out_array.max() > self.MaximumFractionalValue:
                        raise RuntimeError(
                            'Segmentation image contains values greater than '
                            'the MaximumFractionalValue recorded in the '
                            'dataset.'
                        )
                    max_val = self.MaximumFractionalValue
                    out_array = out_array.astype(dtype) / max_val

        return out_array

    def get_default_dimension_index_pointers(
        self
    ) -> list[BaseTag]:
        """Get the default list of tags used to index frames.

        The list of tags used to index dimensions depends upon how the
        segmentation image was constructed, and is stored in the
        DimensionIndexPointer attribute within the DimensionIndexSequence. The
        list returned by this method matches the order of items in the
        DimensionIndexSequence, but omits the ReferencedSegmentNumber
        attribute, since this is handled differently to other tags when
        indexing frames in highdicom.

        Returns
        -------
        List[pydicom.tag.BaseTag]
            List of tags used as the default dimension index pointers.

        """
        referenced_segment_number = tag_for_keyword('ReferencedSegmentNumber')
        return [
            t for t in self.dimension_index_pointers
            if t != referenced_segment_number
        ]

    def are_dimension_indices_unique(
        self,
        dimension_index_pointers: Sequence[int | BaseTag]
    ) -> bool:
        """Check if a list of index pointers uniquely identifies frames.

        For a given list of dimension index pointers, check whether every
        combination of index values for these pointers identifies a unique
        frame per segment in the segmentation image. This is a pre-requisite
        for indexing using this list of dimension index pointers in the
        :meth:`Segmentation.get_pixels_by_dimension_index_values()` method.

        Parameters
        ----------
        dimension_index_pointers: Sequence[Union[int, pydicom.tag.BaseTag]]
            Sequence of tags serving as dimension index pointers.

        Returns
        -------
        bool
            True if the specified list of dimension index pointers uniquely
            identifies frames in the segmentation image. False otherwise.

        Raises
        ------
        KeyError
            If any of the elements of the ``dimension_index_pointers`` are not
            valid dimension index pointers in this segmentation image.

        """
        if len(dimension_index_pointers) == 0:
            raise ValueError(
                'Argument "dimension_index_pointers" may not be empty.'
            )
        dimension_index_pointers = list(dimension_index_pointers)
        for ptr in dimension_index_pointers:
            if ptr not in self.dimension_index_pointers:
                kw = keyword_for_tag(ptr)
                if kw == '':
                    kw = '<no keyword>'
                raise KeyError(
                    f'Tag {ptr} ({kw}) is not used as a dimension index '
                    'in this image.'
                )

        if self.segmentation_type != SegmentationTypeValues.LABELMAP:
            dimension_index_pointers.append(
                tag_for_keyword('ReferencedSegmentNumber')
            )
        return super().are_dimension_indices_unique(
            dimension_index_pointers
        )

    def _get_segment_remap_values(
        self,
        segment_numbers: Sequence[int],
        combine_segments: bool,
        relabel: bool,
    ):
        """Get output segment numbers for retrieving pixels.

        Parameters
        ----------
        segment_numbers: Union[Sequence[int], None]
            Sequence containing segment numbers to include.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False, segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.

        Returns
        -------
        Optional[Sequence[int]]:
            Sequence of output segments for each item of the input segment
            numbers, or None if no remapping is required.

        """
        if combine_segments:
            if relabel:
                return range(1, len(segment_numbers) + 1)
            else:
                return segment_numbers
        return None

    def get_pixels_by_source_instance(
        self,
        source_sop_instance_uids: Sequence[str],
        segment_numbers: Sequence[int] | None = None,
        combine_segments: bool = False,
        relabel: bool = False,
        ignore_spatial_locations: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: type | str | np.dtype | None = None,
        apply_palette_color_lut: bool = False,
        apply_icc_profile: bool | None = None,
    ) -> np.ndarray:
        """Get a pixel array for a list of source instances.

        This is intended for retrieving segmentation masks derived from
        (series of) single frame source images.

        The output array will have 4 dimensions under the default behavior, and
        3 dimensions if ``combine_segments`` is set to ``True``.  The first
        dimension represents the source instances. ``pixel_array[i, ...]``
        represents the segmentation of ``source_sop_instance_uids[i]``.  The
        next two dimensions are the rows and columns of the frames,
        respectively.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (4th) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        With ``"LABELMAP"`` segmentations that use the ``"PALETTE COLOR"``
        photometric interpretation, the ``apply_palette_color_lut`` parameter
        may be used to produce a color image in which each segment is given an
        RGB defined in a palette color LUT within the segmentation object.
        The three color channels (RGB) will be stacked down the final (4th)
        dimension of the pixel array.

        Parameters
        ----------
        source_sop_instance_uids: str
            SOP Instance UID of the source instances to for which segmentations
            are requested.
        segment_numbers: Union[Sequence[int], None], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        ignore_spatial_locations: bool, optional
           Ignore whether or not spatial locations were preserved in the
           derivation of the segmentation frames from the source frames. In
           some segmentation images, the pixel locations in the segmentation
           frames may not correspond to pixel locations in the frames of the
           source image from which they were derived. The segmentation image
           may or may not specify whether or not spatial locations are
           preserved in this way through use of the optional (0028,135A)
           SpatialLocationsPreserved attribute. If this attribute specifies
           that spatial locations are not preserved, or is absent from the
           segmentation image, highdicom's default behavior is to disallow
           indexing by source frames. To override this behavior and retrieve
           segmentation pixels regardless of the presence or value of the
           spatial locations preserved attribute, set this parameter to True.
        assert_missing_frames_are_empty: bool, optional
            Assert that requested source frame numbers that are not referenced
            by the segmentation image contain no segments. If a source frame
            number is not referenced by the segmentation image, highdicom is
            unable to check that the frame number is valid in the source image.
            By default, highdicom will raise an error if any of the requested
            source frames are not referenced in the source image. To override
            this behavior and return a segmentation frame of all zeros for such
            frames, set this parameter to True.
        rescale_fractional: bool, optional
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.
        apply_palette_color_lut: bool, optional
            If True, apply the palette color LUT to give RGB output values.
            This is only valid for ``"LABELMAP"`` segmentations that contain
            palette color LUT information, and only when ``combine_segments``
            is ``True`` and ``relabel`` is ``False``.
        apply_icc_profile: bool, optional
            If True apply an ICC profile to the output and require it to be
            present. If None, apply an ICC profile if found but do not require
            it to be present. If False, never apply an ICC profile. Only
            possible when ``apply_palette_color_lut`` is True.

        Returns
        -------
        pixel_array: numpy.ndarray
            Pixel array representing the segmentation. See notes for full
            explanation.

        Examples
        --------

        Read in an example from the highdicom test data:

        >>> import highdicom as hd
        >>>
        >>> seg = hd.seg.segread('data/test_files/seg_image_ct_binary.dcm')

        List the source images for this segmentation:

        >>> for study_uid, series_uid, sop_uid in seg.get_source_image_uids():
        ...     print(sop_uid)
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.94
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.95
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.96

        Get the segmentation array for a subset of these images:

        >>> pixels = seg.get_pixels_by_source_instance(
        ...     source_sop_instance_uids=[
        ...         '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93',
        ...         '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.94'
        ...     ]
        ... )
        >>> pixels.shape
        (2, 16, 16, 1)

        """
        # Check that indexing in this way is possible
        self._check_indexing_with_source_frames(
            ignore_spatial_locations
        )

        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = self.segment_numbers
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )
        if isinstance(source_sop_instance_uids, str):
            raise TypeError(
                'source_sop_instance_uids should be a sequence of UIDs, not a '
                'single UID'
            )
        if len(source_sop_instance_uids) == 0:
            raise ValueError(
                'Source SOP instance UIDs may not be empty.'
            )

        # Check that the combination of source instances and segment numbers
        # uniquely identify segmentation frames
        columns = ['ReferencedSOPInstanceUID']
        if self.segmentation_type != SegmentationTypeValues.LABELMAP:
            columns.append('ReferencedSegmentNumber')
        if not self._do_columns_identify_unique_frames(columns):
            raise RuntimeError(
                'Source SOP instance UIDs and segment numbers do not '
                'uniquely identify frames of the segmentation image.'
            )

        # Check that all frame numbers requested actually exist
        if not assert_missing_frames_are_empty:
            unique_uids = (
                self._get_unique_referenced_sop_instance_uids()
            )
            missing_uids = set(source_sop_instance_uids) - unique_uids
            if len(missing_uids) > 0:
                msg = (
                    f'SOP Instance UID(s) {list(missing_uids)} do not match '
                    'any referenced source instances. To return an empty '
                    'segmentation mask in this situation, use the '
                    '"assert_missing_frames_are_empty" parameter.'
                )
                raise KeyError(msg)

        remap_channel_indices = self._get_segment_remap_values(
            segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel
        )

        if self.segmentation_type == SegmentationTypeValues.LABELMAP:
            channel_indices = None
        else:
            channel_indices = [{'ReferencedSegmentNumber': segment_numbers}]

        with self._iterate_indices_for_stack(
            stack_indices={
                'ReferencedSOPInstanceUID': source_sop_instance_uids
            },
            channel_indices=channel_indices,
            remap_channel_indices=[remap_channel_indices],
            allow_missing_values=True,
            allow_missing_combinations=True,
        ) as indices:

            return self._get_pixels_by_seg_frame(
                spatial_shape=len(source_sop_instance_uids),
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
            )

    def get_pixels_by_source_frame(
        self,
        source_sop_instance_uid: str,
        source_frame_numbers: Sequence[int],
        segment_numbers: Sequence[int] | None = None,
        combine_segments: bool = False,
        relabel: bool = False,
        ignore_spatial_locations: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: type | str | np.dtype | None = None,
        apply_palette_color_lut: bool = False,
        apply_icc_profile: bool | None = None,
    ):
        """Get a pixel array for a list of frames within a source instance.

        This is intended for retrieving segmentation masks derived from
        multi-frame (enhanced) source images. All source frames for
        which segmentations are requested must belong within the same
        SOP Instance UID.

        The output array will have 4 dimensions under the default behavior, and
        3 dimensions if ``combine_segments`` is set to ``True``.  The first
        dimension represents the source frames. ``pixel_array[i, ...]``
        represents the segmentation of ``source_frame_numbers[i]``.  The
        next two dimensions are the rows and columns of the frames,
        respectively.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (4th) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        With ``"LABELMAP"`` segmentations that use the ``"PALETTE COLOR"``
        photometric interpretation, the ``apply_palette_color_lut`` parameter
        may be used to produce a color image in which each segment is given an
        RGB defined in a palette color LUT within the segmentation object.
        The three color channels (RGB) will be stacked down the final (4th)
        dimension of the pixel array.

        Parameters
        ----------
        source_sop_instance_uid: str
            SOP Instance UID of the source instance that contains the source
            frames.
        source_frame_numbers: Sequence[int]
            A sequence of frame numbers (1-based) within the source instance
            for which segmentations are requested.
        segment_numbers: Optional[Sequence[int]], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        ignore_spatial_locations: bool, optional
            Ignore whether or not spatial locations were preserved in the
            derivation of the segmentation frames from the source frames. In
            some segmentation images, the pixel locations in the segmentation
            frames may not correspond to pixel locations in the frames of the
            source image from which they were derived. The segmentation image
            may or may not specify whether or not spatial locations are
            preserved in this way through use of the optional (0028,135A)
            SpatialLocationsPreserved attribute. If this attribute specifies
            that spatial locations are not preserved, or is absent from the
            segmentation image, highdicom's default behavior is to disallow
            indexing by source frames. To override this behavior and retrieve
            segmentation pixels regardless of the presence or value of the
            spatial locations preserved attribute, set this parameter to True.
        assert_missing_frames_are_empty: bool, optional
            Assert that requested source frame numbers that are not referenced
            by the segmentation image contain no segments. If a source frame
            number is not referenced by the segmentation image and is larger
            than the frame number of the highest referenced frame, highdicom is
            unable to check that the frame number is valid in the source image.
            By default, highdicom will raise an error in this situation. To
            override this behavior and return a segmentation frame of all zeros
            for such frames, set this parameter to True.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.
        apply_palette_color_lut: bool, optional
            If True, apply the palette color LUT to give RGB output values.
            This is only valid for ``"LABELMAP"`` segmentations that contain
            palette color LUT information, and only when ``combine_segments``
            is ``True`` and ``relabel`` is ``False``.
        apply_icc_profile: bool, optional
            If True apply an ICC profile to the output and require it to be
            present. If None, apply an ICC profile if found but do not require
            it to be present. If False, never apply an ICC profile. Only
            possible when ``apply_palette_color_lut`` is True.

        Returns
        -------
        pixel_array: numpy.ndarray
            Pixel array representing the segmentation. See notes for full
            explanation.

        Examples
        --------

        Read in an example from the highdicom test data derived from a
        multiframe slide microscopy image:

        >>> import highdicom as hd
        >>>
        >>> seg = hd.seg.segread('data/test_files/seg_image_sm_control.dcm')

        List the source image SOP instance UID for this segmentation:

        >>> sop_uid = seg.get_source_image_uids()[0][2]
        >>> sop_uid
        '1.2.826.0.1.3680043.9.7433.3.12857516184849951143044513877282227'

        Get the segmentation array for 3 of the frames in the multiframe source
        image.  The resulting segmentation array has 3 10 x 10 frames, one for
        each source frame. The final dimension contains the 20 different
        segments present in this segmentation.

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6]
        ... )
        >>> pixels.shape
        (3, 10, 10, 20)

        This time, select only 4 of the 20 segments:

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6],
        ...     segment_numbers=[10, 11, 12, 13]
        ... )
        >>> pixels.shape
        (3, 10, 10, 4)

        Instead create a multiclass label map for each source frame. Note
        that segments 6, 8, and 10 are present in the three chosen frames.

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6],
        ...     combine_segments=True
        ... )
        >>> pixels.shape, np.unique(pixels)
        ((3, 10, 10), array([ 0,  6,  8, 10], dtype=uint8))

        Now relabel the segments to give a pixel map with values between 0
        and 3 (inclusive):

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6],
        ...     segment_numbers=[6, 8, 10],
        ...     combine_segments=True,
        ...     relabel=True
        ... )
        >>> pixels.shape, np.unique(pixels)
        ((3, 10, 10), array([0, 1, 2, 3], dtype=uint8))

        """
        # Check that indexing in this way is possible
        self._check_indexing_with_source_frames(
            ignore_spatial_locations
        )

        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )

        if len(source_frame_numbers) == 0:
            raise ValueError(
                'Source frame numbers should not be empty.'
            )
        if not all(f > 0 for f in source_frame_numbers):
            raise ValueError(
                'Frame numbers are 1-based indices and must be > 0.'
            )

        # Check that the combination of frame numbers and segment numbers
        # uniquely identify segmentation frames
        columns = ['ReferencedFrameNumber']
        if self.segmentation_type != SegmentationTypeValues.LABELMAP:
            columns.append('ReferencedSegmentNumber')
        if not self._do_columns_identify_unique_frames(columns):
            raise RuntimeError(
                'Source frame numbers and segment numbers do not '
                'uniquely identify frames of the segmentation image.'
            )

        # Check that all frame numbers requested actually exist
        if not assert_missing_frames_are_empty:
            max_frame_number = (
                self._get_max_referenced_frame_number()
            )
            for f in source_frame_numbers:
                if f > max_frame_number:
                    msg = (
                        f'Source frame number {f} is larger than any '
                        'referenced source frame, so highdicom cannot be '
                        'certain that it is valid. To return an empty '
                        'segmentation mask in this situation, use the '
                        "'assert_missing_frames_are_empty' parameter."
                    )
                    raise ValueError(msg)

        if self.segmentation_type == SegmentationTypeValues.LABELMAP:
            channel_indices = None
        else:
            channel_indices = [
                {'ReferencedSegmentNumber': list(segment_numbers)}
            ]

        remap_channel_indices = self._get_segment_remap_values(
            segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel
        )

        with self._iterate_indices_for_stack(
            stack_indices={'ReferencedFrameNumber': list(source_frame_numbers)},
            channel_indices=channel_indices,
            remap_channel_indices=[remap_channel_indices],
            allow_missing_values=True,
            allow_missing_combinations=True,
        ) as indices:

            return self._get_pixels_by_seg_frame(
                spatial_shape=len(source_frame_numbers),
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
            )

    def get_volume_geometry(
        self,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        allow_missing_positions: bool = True,
        allow_duplicate_positions: bool = True,
    ) -> VolumeGeometry | None:
        """Get geometry of the image in 3D space.

        Note that this differs from the method of the same name on the
        :class:`highdicom.Image` base class only by a change of default value
        of the ``allow_missing_positions`` parameter to to ``True``. This
        reflects the fact that empty frames are often omitted from segmentation
        images.

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
            3D volume. ``None`` otherwise.

        """
        return super().get_volume_geometry(
            rtol=rtol,
            atol=atol,
            allow_missing_positions=allow_missing_positions,
            allow_duplicate_positions=allow_duplicate_positions,
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
        dtype: type | str | np.dtype | None = None,
        segment_numbers: Sequence[int] | None = None,
        combine_segments: bool = False,
        relabel: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        apply_palette_color_lut: bool = False,
        apply_icc_profile: bool | None = None,
        allow_missing_positions: bool = True,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> Volume:
        """Create a :class:`highdicom.Volume` from the segmentation.

        This is only possible if the segmentation represents a regularly-spaced
        3D volume.

        Parameters
        ----------
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
        segment_numbers: Optional[Sequence[int]], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.
        apply_palette_color_lut: bool, optional
            If True, apply the palette color LUT to give RGB output values.
            This is only valid for ``"LABELMAP"`` segmentations that contain
            palette color LUT information, and only when ``combine_segments``
            is ``True`` and ``relabel`` is ``False``.
        apply_icc_profile: bool, optional
            If True apply an ICC profile to the output and require it to be
            present. If None, apply an ICC profile if found but do not require
            it to be present. If False, never apply an ICC profile. Only
            possible when ``apply_palette_color_lut`` is True.
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
            spacings vary by less that this value (in mm), they
            are considered to be regular. Incompatible with ``rtol``.

        """
        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
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

        remap_channel_indices = self._get_segment_remap_values(
            segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel
        )

        columns = [
            'ImagePositionPatient_0',
            'ImagePositionPatient_1',
            'ImagePositionPatient_2'
        ]
        if self.segmentation_type == SegmentationTypeValues.LABELMAP:
            channel_indices = None
        else:
            columns.append('ReferencedSegmentNumber')
            channel_indices = [{'ReferencedSegmentNumber': segment_numbers}]

        channel_spec = None
        if not combine_segments:
            channel_spec = {'ReferencedSegmentNumber': segment_numbers}
        if apply_palette_color_lut:
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
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
                as_indices=True,  # due to earlier standardization
                dtype=dtype,
            )[None]

            affine = volume_geometry[
                :,
                row_start - 1:,
                column_start - 1:,
            ].affine
        else:
            # Check that the combination of frame numbers and segment numbers
            # uniquely identify segmentation frames
            if not self._do_columns_identify_unique_frames(columns):
                raise RuntimeError(
                    'Volume positions and segment numbers do not '
                    'uniquely identify frames of the segmentation image.'
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
                channel_indices=channel_indices,
                remap_channel_indices=[remap_channel_indices],
                allow_missing_values=True,
                allow_missing_combinations=True,
            ) as indices:

                array = self._get_pixels_by_seg_frame(
                    spatial_shape=volume_geometry.spatial_shape[0],
                    indices_iterator=indices,
                    segment_numbers=np.array(segment_numbers),
                    combine_segments=combine_segments,
                    relabel=relabel,
                    rescale_fractional=rescale_fractional,
                    skip_overlap_checks=skip_overlap_checks,
                    dtype=dtype,
                    apply_palette_color_lut=apply_palette_color_lut,
                    apply_icc_profile=apply_icc_profile,
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

    def get_pixels_by_dimension_index_values(
        self,
        dimension_index_values: Sequence[Sequence[int]],
        dimension_index_pointers: Sequence[int] | None = None,
        segment_numbers: Sequence[int] | None = None,
        combine_segments: bool = False,
        relabel: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: type | str | np.dtype | None = None,
        apply_palette_color_lut: bool = False,
        apply_icc_profile: bool | None = None,
    ):
        """Get a pixel array for a list of dimension index values.

        This is intended for retrieving segmentation masks using the index
        values within the segmentation object, without referring to the
        source images from which the segmentation was derived.

        The output array will have 4 dimensions under the default behavior, and
        3 dimensions if ``combine_segments`` is set to ``True``.  The first
        dimension represents the source frames. ``pixel_array[i, ...]``
        represents the segmentation frame with index
        ``dimension_index_values[i]``.  The next two dimensions are the rows
        and columns of the frames, respectively.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (4th) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        With ``"LABELMAP"`` segmentations that use the ``"PALETTE COLOR"``
        photometric interpretation, the ``apply_palette_color_lut`` parameter
        may be used to produce a color image in which each segment is given an
        RGB defined in a palette color LUT within the segmentation object.
        The three color channels (RGB) will be stacked down the final (4th)
        dimension of the pixel array.

        Parameters
        ----------
        dimension_index_values: Sequence[Sequence[int]]
            Dimension index values for the requested frames. Each element of
            the sequence is a sequence of 1-based index values representing the
            dimension index values for a single frame of the output
            segmentation. The order of the index values within the inner
            sequence is determined by the ``dimension_index_pointers``
            parameter, and as such the length of each inner sequence must
            match the length of ``dimension_index_pointers`` parameter.
        dimension_index_pointers: Union[Sequence[Union[int, pydicom.tag.BaseTag]], None], optional
            The data element tags that identify the indices used in the
            ``dimension_index_values`` parameter. Each element identifies a
            data element tag by which frames are ordered in the segmentation
            image dataset. If this parameter is set to ``None`` (the default),
            the value of
            :meth:`Segmentation.get_default_dimension_index_pointers()` is
            used. Valid values of this parameter are are determined by
            the construction of the segmentation image and include any
            permutation of any subset of elements in the
            :meth:`Segmentation.get_default_dimension_index_pointers()` list.
        segment_numbers: Union[Sequence[int], None], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        assert_missing_frames_are_empty: bool, optional
            Assert that requested source frame numbers that are not referenced
            by the segmentation image contain no segments. If a source frame
            number is not referenced by the segmentation image, highdicom is
            unable to check that the frame number is valid in the source image.
            By default, highdicom will raise an error if any of the requested
            source frames are not referenced in the source image. To override
            this behavior and return a segmentation frame of all zeros for such
            frames, set this parameter to True.
        rescale_fractional: bool, optional
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the smallest
            unsigned integer type that accommodates all of the output values
            will be chosen.
        apply_palette_color_lut: bool, optional
            If True, apply the palette color LUT to give RGB output values.
            This is only valid for ``"LABELMAP"`` segmentations that contain
            palette color LUT information, and only when ``combine_segments``
            is ``True`` and ``relabel`` is ``False``.
        apply_icc_profile: bool, optional
            If True apply an ICC profile to the output and require it to be
            present. If None, apply an ICC profile if found but do not require
            it to be present. If False, never apply an ICC profile. Only
            possible when ``apply_palette_color_lut`` is True.

        Returns
        -------
        pixel_array: numpy.ndarray
            Pixel array representing the segmentation. See notes for full
            explanation.

        Examples
        --------

        Read a test image of a segmentation of a slide microscopy image

        >>> import highdicom as hd
        >>> from pydicom.datadict import keyword_for_tag, tag_for_keyword
        >>> from pydicom import dcmread
        >>>
        >>> ds = dcmread('data/test_files/seg_image_sm_control.dcm')
        >>> seg = hd.seg.Segmentation.from_dataset(ds)

        Get the default list of dimension index values

        >>> for tag in seg.get_default_dimension_index_pointers():
        ...     print(keyword_for_tag(tag))
        ColumnPositionInTotalImagePixelMatrix
        RowPositionInTotalImagePixelMatrix
        XOffsetInSlideCoordinateSystem
        YOffsetInSlideCoordinateSystem
        ZOffsetInSlideCoordinateSystem


        Use a subset of these index pointers to index the image

        >>> tags = [
        ...     tag_for_keyword('ColumnPositionInTotalImagePixelMatrix'),
        ...     tag_for_keyword('RowPositionInTotalImagePixelMatrix')
        ... ]
        >>> assert seg.are_dimension_indices_unique(tags)  # True

        It is therefore possible to index using just this subset of
        dimension indices

        >>> pixels = seg.get_pixels_by_dimension_index_values(
        ...     dimension_index_pointers=tags,
        ...     dimension_index_values=[[1, 1], [1, 2]]
        ... )
        >>> pixels.shape
        (2, 10, 10, 20)

        """  # noqa: E501
        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )

        referenced_segment_number_tag = tag_for_keyword(
            'ReferencedSegmentNumber'
        )
        if dimension_index_pointers is None:
            dimension_index_pointers = [
                t for t in self.dimension_index_pointers
                if t != referenced_segment_number_tag
            ]
        else:
            if len(dimension_index_pointers) == 0:
                raise ValueError(
                    'Argument "dimension_index_pointers" must not be empty.'
                )
            for ptr in dimension_index_pointers:
                if ptr == referenced_segment_number_tag:
                    raise ValueError(
                        "Do not include the ReferencedSegmentNumber in the "
                        "argument 'dimension_index_pointers'."
                    )
                if ptr not in self.dimension_index_pointers:
                    kw = keyword_for_tag(ptr)
                    if kw == '':
                        kw = '<no keyword>'
                    raise KeyError(
                        f'Tag {Tag(ptr)} ({kw}) is not used as a dimension '
                        'index in this image.'
                    )

        if len(dimension_index_values) == 0:
            raise ValueError(
                'Argument "dimension_index_values" must not be empty.'
            )
        for row in dimension_index_values:
            if len(row) != len(dimension_index_pointers):
                raise ValueError(
                    'Dimension index values must be a sequence of sequences of '
                    'integers, with each inner sequence having a single value '
                    'per dimension index pointer specified.'
                )

        # Check that all frame numbers requested actually exist
        if not assert_missing_frames_are_empty:
            unique_dim_ind_vals = self._get_unique_dim_index_values(
                dimension_index_pointers
            )
            queried_dim_inds = {tuple(r) for r in dimension_index_values}
            missing_dim_inds = queried_dim_inds - unique_dim_ind_vals
            if len(missing_dim_inds) > 0:
                msg = (
                    f'Dimension index values {list(missing_dim_inds)} do not '
                    'match any segmentation frame. To return '
                    'an empty segmentation mask in this situation, '
                    "use the 'assert_missing_frames_are_empty' "
                    'parameter.'
                )
                raise ValueError(msg)

        if self.segmentation_type == SegmentationTypeValues.LABELMAP:
            channel_indices = None
        else:
            channel_indices = [
                {'ReferencedSegmentNumber': list(segment_numbers)}
            ]

        remap_channel_indices = self._get_segment_remap_values(
            segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel,
        )

        stack_indices = {
            ptr: vals
            for ptr, vals in zip(
                dimension_index_pointers,
                zip(*dimension_index_values),
            )
        }

        with self._iterate_indices_for_stack(
            stack_indices=stack_indices,
            stack_dimension_use_indices=True,
            channel_indices=channel_indices,
            remap_channel_indices=[remap_channel_indices],
            allow_missing_values=True,
            allow_missing_combinations=True,
        ) as indices:

            return self._get_pixels_by_seg_frame(
                spatial_shape=len(dimension_index_values),
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
            )

    def get_total_pixel_matrix(
        self,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
        segment_numbers: Sequence[int] | None = None,
        combine_segments: bool = False,
        relabel: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: type | str | np.dtype | None = None,
        apply_palette_color_lut: bool = False,
        apply_icc_profile: bool | None = None,
        as_indices: bool = False,
    ):
        """Get the pixel array as a (region of) the total pixel matrix.

        This is intended for retrieving segmentation masks derived from
        multi-frame (enhanced) source images that are tiled. The method
        returns (a region of) the 2D total pixel matrix implied by the
        frames within the segmentation.

        The output array will have 3 dimensions under the default behavior, and
        2 dimensions if ``combine_segments`` is set to ``True``. The first two
        dimensions are the rows and columns of the total pixel matrix,
        respectively. By default, the full total pixel matrix is returned,
        however a smaller region may be requested using the ``row_start``,
        ``row_end``, ``column_start`` and ``column_end`` parameters as 1-based
        indices into the total pixel matrix.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (3rd) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        With ``"LABELMAP"`` segmentations that use the ``"PALETTE COLOR"``
        photometric interpretation, the ``apply_palette_color_lut`` parameter
        may be used to produce a color image in which each segment is given an
        RGB defined in a palette color LUT within the segmentation object.
        The three color channels (RGB) will be stacked down the final (3rd)
        dimension of the pixel array.

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
        segment_numbers: Optional[Sequence[int]], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.
        apply_palette_color_lut: bool, optional
            If True, apply the palette color LUT to give RGB output values.
            This is only valid for ``"LABELMAP"`` segmentations that contain
            palette color LUT information, and only when ``combine_segments``
            is ``True`` and ``relabel`` is ``False``.
        apply_icc_profile: bool | None, optional
            If True apply an ICC profile to the output and require it to be
            present. If None, apply an ICC profile if found but do not require
            it to be present. If False, never apply an ICC profile. Only
            possible when ``apply_palette_color_lut`` is True.
        as_indices: bool, optional
            If True, interpret all row/column numbering parameters
            (``row_start``, ``row_end``, ``column_start``, and ``column_end``)
            as zero-based indices as opposed to the default one-based numbers
            used within DICOM.

        Returns
        -------
        pixel_array: numpy.ndarray
            Pixel array representing the segmentation's total pixel matrix.

        Note
        ----
        By default, this method uses 1-based indexing of rows and columns in
        order to match the conventions used in the DICOM standard. The first
        row of the total pixel matrix is row 1, and the last is
        ``self.TotalPixelMatrixRows``. This is is unlike standard Python and
        NumPy indexing which is 0-based. For negative indices, the two are
        equivalent with the final row/column having index -1. To switch to
        standard Python behavior, specify ``as_indices=True``.

        """
        # Check whether this segmentation is appropriate for tile-based indexing
        if not self.is_tiled:
            raise RuntimeError("Segmentation is not a tiled image.")
        if not self.is_indexable_as_total_pixel_matrix():
            raise RuntimeError(
                "Segmentation does not have appropriate dimension indices "
                "to be indexed as a total pixel matrix."
            )

        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )

        if self.segmentation_type == SegmentationTypeValues.LABELMAP:
            channel_indices = None
        else:
            channel_indices = [
                {'ReferencedSegmentNumber': segment_numbers}
            ]

        remap_channel_indices = self._get_segment_remap_values(
            segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel,
        )

        with self._iterate_indices_for_tiled_region(
            row_start=row_start,
            row_end=row_end,
            column_start=column_start,
            column_end=column_end,
            channel_indices=channel_indices,
            remap_channel_indices=[remap_channel_indices],
            as_indices=as_indices,
            allow_missing_values=True,
            allow_missing_combinations=True,
        ) as (indices, output_shape):

            return self._get_pixels_by_seg_frame(
                spatial_shape=output_shape,
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
                apply_palette_color_lut=apply_palette_color_lut,
                apply_icc_profile=apply_icc_profile,
            )


def segread(
    fp: str | bytes | PathLike | BinaryIO,
    lazy_frame_retrieval: bool = False,
) -> Segmentation:
    """Read a segmentation image stored in DICOM File Format.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing a
        Segmentation image.
    lazy_frame_retrieval: bool
        If True, the returned segmentation will retrieve frames from the file as
        requested, rather than loading in the entire object to memory
        initially. This may be a good idea if file reading is slow and you are
        likely to need only a subset of the frames in the segmentation.

    Returns
    -------
    highdicom.seg.Segmentation
        Segmentation image read from the file.

    """
    # This is essentially a convenience alias for the classmethod (which is
    # used so that it is inherited correctly by subclasses). It is used
    # because it follows the format of other similar functions around the
    # library
    return Segmentation.from_file(
        fp,
        lazy_frame_retrieval=lazy_frame_retrieval,
    )
