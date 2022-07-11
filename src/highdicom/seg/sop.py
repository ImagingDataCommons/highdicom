"""Module for SOP classes of the SEG modality."""
import logging
from collections import defaultdict, OrderedDict
from copy import deepcopy
from os import PathLike
from typing import (
    Any, cast, Dict, List, Optional, Sequence, Union, Tuple, BinaryIO
)

import numpy as np
from pydicom.dataset import Dataset
from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.encaps import encapsulate
from pydicom.multival import MultiValue
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
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
from pydicom.filereader import dcmread

from highdicom._module_utils import ModuleUsageValues, get_module_usage
from highdicom.base import SOPClass, _check_little_endian
from highdicom.content import (
    ContentCreatorIdentificationCodeSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
    PixelMeasuresSequence
)
from highdicom.enum import CoordinateSystemNames
from highdicom.frame import encode_frame
from highdicom.seg.content import (
    DimensionIndexSequence,
    SegmentDescription,
)
from highdicom.seg.enum import (
    SegmentationFractionalTypeValues,
    SegmentationTypeValues,
    SegmentsOverlapValues,
    SpatialLocationsPreservedValues,
    SegmentAlgorithmTypeValues,
)
from highdicom.seg.utils import iter_segments
from highdicom.spatial import ImageToReferenceTransformer
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import check_person_name, _check_code_string
from highdicom.uid import UID as hd_UID


logger = logging.getLogger(__name__)


_NO_FRAME_REF_VALUE = -1


class Segmentation(SOPClass):

    """SOP class for the Segmentation IOD."""

    def __init__(
        self,
        source_images: Sequence[Dataset],
        pixel_array: np.ndarray,
        segmentation_type: Union[str, SegmentationTypeValues],
        segment_descriptions: Sequence[SegmentDescription],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: Union[str, Tuple[str]],
        device_serial_number: str,
        fractional_type: Optional[
            Union[str, SegmentationFractionalTypeValues]
        ] = SegmentationFractionalTypeValues.PROBABILITY,
        max_fractional_value: int = 255,
        content_description: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        pixel_measures: Optional[PixelMeasuresSequence] = None,
        plane_orientation: Optional[PlaneOrientationSequence] = None,
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
        omit_empty_frames: bool = True,
        content_label: Optional[str] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
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
            frame instances) or source_images[0].pixel_array[i, ...] if
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
            encoded using the ``"BINARY"`` or ``"FRACTIONAL"`` methods.

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

        segmentation_type: Union[str, highdicom.seg.SegmentationTypeValues]
            Type of segmentation, either ``"BINARY"`` or ``"FRACTIONAL"``
        segment_descriptions: Sequence[highdicom.seg.SegmentDescription]
            Description of each segment encoded in `pixel_array`. In the case of
            pixel arrays with multiple integer values, the segment description
            with the corresponding segment number is used to describe each segment.
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the series within the study
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
            RLE Lossless (``"1.2.840.10008.1.2.5"``) and
            JPEG 2000 Lossless (``"1.2.840.10008.1.2.4.90"``).
        pixel_measures: Union[highdicom.PixelMeasures, None], optional
            Physical spacing of image pixels in `pixel_array`.
            If ``None``, it will be assumed that the segmentation image has the
            same pixel measures as the source image(s).
        plane_orientation: Union[highdicom.PlaneOrientationSequence, None], optional
            Orientation of planes in `pixel_array` relative to axes of
            three-dimensional patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image as the
            same plane orientation as the source image(s).
        plane_positions: Union[Sequence[highdicom.PlanePositionSequence], None], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image has the
            same plane position as the source image(s). However, this will only
            work when the first dimension of `pixel_array` matches the number
            of frames in `source_images` (in case of multi-frame source images)
            or the number of `source_images` (in case of single-frame source
            images).
        omit_empty_frames: bool, optional
            If True (default), frames with no non-zero pixels are omitted from
            the segmentation image. If False, all frames are included.
        content_label: Union[str, None], optional
            Content label
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this segmentation.
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

        uniqueness_criteria = set(
            (
                image.StudyInstanceUID,
                image.SeriesInstanceUID,
                image.Rows,
                image.Columns,
                getattr(image, 'FrameOfReferenceUID', None),
            )
            for image in source_images
        )
        if len(uniqueness_criteria) > 1:
            raise ValueError(
                'Source images must all be part of the same series and must '
                'have the same image dimensions (number of rows/columns).'
            )

        src_img = source_images[0]
        is_multiframe = hasattr(src_img, 'NumberOfFrames')
        if is_multiframe and len(source_images) > 1:
            raise ValueError(
                'Only one source image should be provided in case images '
                'are multi-frame images.'
            )
        is_tiled = hasattr(src_img, 'TotalPixelMatrixRows')
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

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        super().__init__(
            study_instance_uid=src_img.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid='1.2.840.10008.5.1.4.1.1.66.4',
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
            # Using Container Type Code Sequence attribute would be more
            # elegant, but unfortunately it is a type 2 attribute.
            if (hasattr(src_img, 'ImageOrientationSlide') or
                    hasattr(src_img, 'ImageCenterPointCoordinatesSequence')):
                self._coordinate_system: Optional[CoordinateSystemNames] = \
                    CoordinateSystemNames.SLIDE
            else:
                self._coordinate_system = CoordinateSystemNames.PATIENT
        else:
            # Only allow missing FrameOfReferenceUID if it is not required
            # for this IOD
            usage = get_module_usage('frame-of-reference', src_img.SOPClassUID)
            if usage == ModuleUsageValues.MANDATORY:
                raise ValueError(
                    "Source images have no Frame Of Reference UID, but it is "
                    "required by the IOD."
                )

            # It may be possible to generalize this, but for now only a single
            # source frame is permitted when no frame of reference exists
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
            self._coordinate_system = None

        # General Reference
        self.SourceImageSequence: List[Dataset] = []
        referenced_series: Dict[str, List[Dataset]] = defaultdict(list)
        for s_img in source_images:
            ref = Dataset()
            ref.ReferencedSOPClassUID = s_img.SOPClassUID
            ref.ReferencedSOPInstanceUID = s_img.SOPInstanceUID
            self.SourceImageSequence.append(ref)
            referenced_series[s_img.SeriesInstanceUID].append(ref)

        # Common Instance Reference
        self.ReferencedSeriesSequence: List[Dataset] = []
        for series_instance_uid, referenced_images in referenced_series.items():
            ref = Dataset()
            ref.SeriesInstanceUID = series_instance_uid
            ref.ReferencedInstanceSequence = referenced_images
            self.ReferencedSeriesSequence.append(ref)

        # Image Pixel
        self.Rows = pixel_array.shape[1]
        self.Columns = pixel_array.shape[2]

        # Segmentation Image
        self.ImageType = ['DERIVED', 'PRIMARY']
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = 'MONOCHROME2'
        self.PixelRepresentation = 0

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

        segmentation_type = SegmentationTypeValues(segmentation_type)
        self.SegmentationType = segmentation_type.value
        if self.SegmentationType == SegmentationTypeValues.BINARY.value:
            self.BitsAllocated = 1
            self.HighBit = 0
            if self.file_meta.TransferSyntaxUID.is_encapsulated:
                raise ValueError(
                    'The chosen transfer syntax '
                    f'{self.file_meta.TransferSyntaxUID} '
                    'is not compatible with the BINARY segmentation type'
                )
        elif self.SegmentationType == SegmentationTypeValues.FRACTIONAL.value:
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
        else:
            raise ValueError(
                'Unknown segmentation type "{}"'.format(segmentation_type)
            )

        self.BitsStored = self.BitsAllocated
        self.LossyImageCompression = getattr(
            src_img,
            'LossyImageCompression',
            '00'
        )
        if self.LossyImageCompression == '01':
            self.LossyImageCompressionRatio = \
                src_img.LossyImageCompressionRatio
            self.LossyImageCompressionMethod = \
                src_img.LossyImageCompressionMethod

        self.SegmentSequence: List[SegmentDescription] = []

        # Multi-Frame Functional Groups and Multi-Frame Dimensions
        sffg_item = Dataset()
        if pixel_measures is None:
            if is_multiframe:
                src_shared_fg = src_img.SharedFunctionalGroupsSequence[0]
                pixel_measures = src_shared_fg.PixelMeasuresSequence
            else:
                if has_ref_frame_uid:
                    pixel_measures = PixelMeasuresSequence(
                        pixel_spacing=src_img.PixelSpacing,
                        slice_thickness=src_img.SliceThickness,
                        spacing_between_slices=src_img.get(
                            'SpacingBetweenSlices',
                            None
                        )
                    )
                else:
                    pixel_spacing = getattr(src_img, 'PixelSpacing', None)
                    if pixel_spacing is not None:
                        pixel_measures = PixelMeasuresSequence(
                            pixel_spacing=pixel_spacing,
                            slice_thickness=src_img.get(
                                'SliceThickness',
                                None
                            ),
                            spacing_between_slices=src_img.get(
                                'SpacingBetweenSlices',
                                None
                            )
                        )
                    else:
                        pixel_measures = None

        if has_ref_frame_uid:
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                source_plane_orientation = PlaneOrientationSequence(
                    coordinate_system=self._coordinate_system,
                    image_orientation=src_img.ImageOrientationSlide
                )
            else:
                if is_multiframe:
                    src_sfg = src_img.SharedFunctionalGroupsSequence[0]
                    source_plane_orientation = deepcopy(
                        src_sfg.PlaneOrientationSequence
                    )
                else:
                    source_plane_orientation = PlaneOrientationSequence(
                        coordinate_system=self._coordinate_system,
                        image_orientation=src_img.ImageOrientationPatient
                    )

            if plane_orientation is None:
                plane_orientation = source_plane_orientation

        self.DimensionIndexSequence = DimensionIndexSequence(
            coordinate_system=self._coordinate_system
        )
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = \
            self.DimensionIndexSequence[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence = [dimension_organization]

        if has_ref_frame_uid:
            if is_multiframe:
                source_plane_positions = \
                    self.DimensionIndexSequence.get_plane_positions_of_image(
                        src_img
                    )
            else:
                source_plane_positions = \
                    self.DimensionIndexSequence.get_plane_positions_of_series(
                        source_images
                    )

        if pixel_measures is not None:
            sffg_item.PixelMeasuresSequence = pixel_measures
        if plane_orientation is not None:
            sffg_item.PlaneOrientationSequence = plane_orientation
        self.SharedFunctionalGroupsSequence = [sffg_item]

        # Information about individual frames will be updated below
        self.NumberOfFrames = 0
        self.PerFrameFunctionalGroupsSequence: List[Dataset] = []

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        if pixel_array.ndim not in [3, 4]:
            raise ValueError('Pixel array must be a 2D, 3D, or 4D array.')

        if pixel_array.shape[1:3] != (self.Rows, self.Columns):
            raise ValueError(
                'Pixel array representing segments has the wrong number of '
                'rows and columns.'
            )

        # Check segment numbers
        described_segment_numbers = np.array([
            int(item.SegmentNumber)
            for item in segment_descriptions
        ])
        self._check_segment_numbers(described_segment_numbers)

        # Checks on pixels and overlap
        pixel_array, segments_overlap = self._check_and_cast_pixel_array(
            pixel_array,
            described_segment_numbers,
            segmentation_type
        )
        self.SegmentsOverlap = segments_overlap.value

        if has_ref_frame_uid:
            if plane_positions is None:
                if pixel_array.shape[0] != len(source_plane_positions):
                    raise ValueError(
                        'Number of plane positions in source image(s) does not '
                        'match size of first dimension of "pixel_array" '
                        'argument.'
                    )
                plane_positions = source_plane_positions
            else:
                if pixel_array.shape[0] != len(plane_positions):
                    raise ValueError(
                        'Number of PlanePositionSequence items provided via '
                        '"plane_positions" argument does not match size of '
                        'first dimension of "pixel_array" argument.'
                    )

            are_spatial_locations_preserved = (
                all(
                    plane_positions[i] == source_plane_positions[i]
                    for i in range(len(plane_positions))
                ) and
                plane_orientation == source_plane_orientation
            )

            plane_position_values, plane_sort_index = \
                self.DimensionIndexSequence.get_index_values(plane_positions)
        else:
            # Only one spatial location supported
            plane_positions = [None]
            plane_position_values = [None]
            plane_sort_index = np.array([0])
            are_spatial_locations_preserved = True

        plane_position_names = self.DimensionIndexSequence.get_index_keywords()

        if (
            has_ref_frame_uid and
            self._coordinate_system == CoordinateSystemNames.SLIDE
        ):
            self.ImageOrientationSlide = deepcopy(
                plane_orientation[0].ImageOrientationSlide
            )
            if are_spatial_locations_preserved and is_tiled:
                self.TotalPixelMatrixOriginSequence = deepcopy(
                    src_img.TotalPixelMatrixOriginSequence
                )
                self.TotalPixelMatrixRows = src_img.TotalPixelMatrixRows
                self.TotalPixelMatrixColumns = src_img.TotalPixelMatrixColumns
            elif are_spatial_locations_preserved and not is_tiled:
                self.ImageCenterPointCoordinatesSequence = deepcopy(
                    src_img.ImageCenterPointCoordinatesSequence
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
                    self.TotalPixelMatrixOriginSequence = [origin_item]
                    self.TotalPixelMatrixRows = int(
                        plane_position_values[last_frame_index, row_index] +
                        self.Rows
                    )
                    self.TotalPixelMatrixColumns = int(
                        plane_position_values[last_frame_index, col_index] +
                        self.Columns
                    )
                else:
                    transform = ImageToReferenceTransformer(
                        image_position=(x_origin, y_origin, z_origin),
                        image_orientation=plane_orientation,
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

        # Remove empty slices
        if omit_empty_frames:
            pixel_array, plane_positions, source_image_indices = \
                self._omit_empty_frames(pixel_array, plane_positions)
        else:
            source_image_indices = list(range(pixel_array.shape[0]))

        if has_ref_frame_uid:
            plane_position_values = plane_position_values[source_image_indices]
            _, plane_sort_index = np.unique(
                plane_position_values,
                axis=0,
                return_index=True
            )

            # Get unique values of attributes in the Plane Position Sequence or
            # Plane Position Slide Sequence, which define the position of the
            # plane with respect to the three dimensional patient or slide
            # coordinate system, respectively. These can subsequently be used
            # to look up the relative position of a plane relative to the
            # indexed dimension.
            dimension_position_values = [
                np.unique(plane_position_values[:, index], axis=0)
                for index in range(plane_position_values.shape[1])
            ]
        else:
            dimension_position_values = [None]

        is_encaps = self.file_meta.TransferSyntaxUID.is_encapsulated
        if is_encaps:
            # In the case of encapsulated transfer syntaxes, we will accumulate
            # a list of encoded frames to encapsulate at the end
            full_frames_list = []
        else:
            # In the case of non-encapsulated (uncompressed) transfer syntaxes
            # we will accumulate a 1D array of pixels from all frames for
            # bitpacking at the end
            full_pixel_array = np.array([], np.bool_)

        for i, segment_number in enumerate(described_segment_numbers):
            # Pixel array for just this segment
            if pixel_array.dtype in (np.float_, np.float32, np.float64):
                # Floating-point numbers must be mapped to 8-bit integers in
                # the range [0, max_fractional_value].
                if pixel_array.ndim == 4:
                    segment_array = pixel_array[:, :, :, segment_number - 1]
                else:
                    segment_array = pixel_array
                planes = np.around(
                    segment_array * float(self.MaximumFractionalValue)
                )
                planes = planes.astype(np.uint8)
            elif pixel_array.dtype in (np.uint8, np.uint16):
                # Note that integer arrays with segments stacked down the last
                # dimension will already have been converted to bool, leaving
                # only "label maps" here, which must be converted to binary
                # masks.
                planes = np.zeros(pixel_array.shape, dtype=np.uint8)
                planes[pixel_array == segment_number] = 1
            elif pixel_array.dtype == np.bool_:
                if pixel_array.ndim == 4:
                    planes = pixel_array[:, :, :, segment_number - 1]
                else:
                    planes = pixel_array
                planes = planes.astype(np.uint8)
                # It may happen that a boolean array is passed that should be
                # interpreted as fractional segmentation type. In this case, we
                # also need to stretch pixel valeus to 8-bit unsigned integer
                # range by multiplying with the maximum fractional value.
                if segmentation_type == SegmentationTypeValues.FRACTIONAL:
                    planes *= int(self.MaximumFractionalValue)
            else:
                raise TypeError('Pixel array has an invalid data type.')

            contained_plane_index = []
            for j in plane_sort_index:
                # Index of this frame in the original list of source indices
                source_image_index = source_image_indices[j]

                # Even though completely empty slices were removed earlier,
                # there may still be slices in which this specific segment is
                # absent. Such frames should be removed
                if omit_empty_frames and np.sum(planes[j]) == 0:
                    logger.info(
                        'skip empty plane {} of segment #{}'.format(
                            j, segment_number
                        )
                    )
                    continue
                contained_plane_index.append(j)
                logger.info(
                    'add plane #{} for segment #{}'.format(
                        j, segment_number
                    )
                )

                pffp_item = Dataset()
                frame_content_item = Dataset()

                if not has_ref_frame_uid:
                    index_values = []
                else:
                    # Look up the position of the plane relative to the indexed
                    # dimension.
                    try:
                        if (
                            self._coordinate_system ==
                            CoordinateSystemNames.SLIDE
                        ):
                            index_values = [
                                np.where(
                                    (dimension_position_values[idx] == pos)
                                )[0][0] + 1
                                for idx, pos in enumerate(
                                    plane_position_values[j]
                                )
                            ]
                        else:
                            # In case of the patient coordinate system, the
                            # value of the attribute the Dimension Index
                            # Sequence points to (Image Position Patient) has a
                            # value multiplicity greater than one.
                            index_values = [
                                np.where(
                                    (dimension_position_values[idx] == pos).all(
                                        axis=1
                                    )
                                )[0][0] + 1
                                for idx, pos in enumerate(
                                    plane_position_values[j]
                                )
                            ]
                    except IndexError as error:
                        raise IndexError(
                            'Could not determine position of plane #{} in '
                            'three dimensional coordinate system based on '
                            'dimension index values: {}'.format(j, error)
                        )
                frame_content_item.DimensionIndexValues = (
                    [segment_number] + index_values
                )
                pffp_item.FrameContentSequence = [frame_content_item]
                if has_ref_frame_uid:
                    pos = plane_positions[j]
                    if self._coordinate_system == CoordinateSystemNames.SLIDE:
                        pffp_item.PlanePositionSlideSequence = pos
                    else:
                        pffp_item.PlanePositionSequence = pos

                # Determining the source images that map to the frame is not
                # always trivial. Since DerivationImageSequence is a type 2
                # attribute, we leave its value empty.
                pffp_item.DerivationImageSequence = []

                if are_spatial_locations_preserved:
                    derivation_image_item = Dataset()
                    derivation_code = codes.cid7203.Segmentation
                    derivation_image_item.DerivationCodeSequence = [
                        CodedConcept.from_code(derivation_code)
                    ]

                    derivation_src_img_item = Dataset()
                    if hasattr(source_images[0], 'NumberOfFrames'):
                        # A single multi-frame source image
                        src_img_item = self.SourceImageSequence[0]
                        # Frame numbers are one-based
                        derivation_src_img_item.ReferencedFrameNumber = (
                            source_image_index + 1
                        )
                    else:
                        # Multiple single-frame source images
                        src_img_item = self.SourceImageSequence[
                            source_image_index
                        ]
                    derivation_src_img_item.ReferencedSOPClassUID = \
                        src_img_item.ReferencedSOPClassUID
                    derivation_src_img_item.ReferencedSOPInstanceUID = \
                        src_img_item.ReferencedSOPInstanceUID
                    purpose_code = \
                        codes.cid7202.SourceImageForImageProcessingOperation
                    derivation_src_img_item.PurposeOfReferenceCodeSequence = [
                        CodedConcept.from_code(purpose_code)
                    ]
                    derivation_src_img_item.SpatialLocationsPreserved = 'YES'
                    derivation_image_item.SourceImageSequence = [
                        derivation_src_img_item,
                    ]
                    pffp_item.DerivationImageSequence.append(
                        derivation_image_item
                    )
                else:
                    logger.warning('spatial locations not preserved')

                identification = Dataset()
                identification.ReferencedSegmentNumber = segment_number
                pffp_item.SegmentIdentificationSequence = [
                    identification,
                ]
                self.PerFrameFunctionalGroupsSequence.append(pffp_item)
                self.NumberOfFrames += 1

            if is_encaps:
                # Encode this frame and add to the list for encapsulation
                # at the end
                for f in contained_plane_index:
                    full_frames_list.append(self._encode_pixels(planes[f]))
            else:
                # Concatenate the 1D array for re-encoding at the end
                full_pixel_array = np.concatenate([
                    full_pixel_array,
                    planes[contained_plane_index].flatten()
                ])

            self.SegmentSequence.append(segment_descriptions[i])

        if is_encaps:
            # Encapsulate all pre-compressed frames
            self.PixelData = encapsulate(full_frames_list)
        else:
            # Encode the whole pixel array at once
            # This allows for correct bit-packing in cases where
            # number of pixels per frame is not a multiple of 8
            self.PixelData = self._encode_pixels(full_pixel_array)

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
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
        omit_empty_frames: bool = True,
    ) -> None:
        """To ensure correctness of segmentation images, this
        method was deprecated in highdicom 0.8.0. For more information
        and migration instructions see :ref:`here <add-segments-deprecation>`.

        """  # noqa E510
        raise AttributeError(
            'To ensure correctness of segmentation images, the add_segments '
            'method was deprecated in highdicom 0.8.0. For more information '
            'and migration instructions visit '
            'https://highdicom.readthedocs.io/en/latest/release_notes.html'
            '#deprecation-of-add-segments-method'
        )

    @staticmethod
    def _check_segment_numbers(described_segment_numbers: np.ndarray):
        """Checks on segment numbers extracted from the segment descriptions.

        Segment numbers should start at 1 and increase by 1. This method checks
        this and raises an appropriate exception for the user if the segment
        numbers are incorrect.

        Parameters
        ----------
        described_segment_numbers: np.ndarray
            The segment numbers from the segment descriptions, in the order
            they were passed. 1D array of integers.

        Raises
        ------
        ValueError
            If the ``described_segment_numbers`` do not have the required values

        """
        # Check segment numbers in the segment descriptions are
        # monotonically increasing by 1
        if not (np.diff(described_segment_numbers) == 1).all():
            raise ValueError(
                'Segment descriptions must be sorted by segment number '
                'and monotonically increasing by 1.'
            )
        if described_segment_numbers[0] != 1:
            raise ValueError(
                'Segment descriptions should be numbered starting '
                f'from 1. Found {described_segment_numbers[0]}. '
            )

    @staticmethod
    def _check_and_cast_pixel_array(
        pixel_array: np.ndarray,
        described_segment_numbers: np.ndarray,
        segmentation_type: SegmentationTypeValues
    ) -> Tuple[np.ndarray, SegmentsOverlapValues]:
        """Checks on the shape and data type of the pixel array.

        Also checks for overlapping segments and returns the result.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            The segmentation pixel array.
        described_segment_numbers: numpy.ndarray
            The segment numbers from the segment descriptions, in the order
            they were passed. 1D array of integers.
        segmentation_type: highdicom.seg.SegmentationTypeValues
            The segmentation_type parameter.

        Returns
        -------
        pixel_array: numpyp.ndarray
            Input pixel array with the data type simplified if possible.
        segments_overlap: highdicom.seg.SegmentationOverlaps
            The value for the SegmentationOverlaps attribute, inferred from the
            pixel array.

        """
        if pixel_array.ndim == 4:
            # Check that the number of segments in the array matches
            if pixel_array.shape[-1] != len(described_segment_numbers):
                raise ValueError(
                    'The number of segments in last dimension of the pixel '
                    f'array ({pixel_array.shape[-1]}) does not match the '
                    'number of described segments '
                    f'({len(described_segment_numbers)}).'
                )

        if pixel_array.dtype in (np.bool_, np.uint8, np.uint16):
            if pixel_array.ndim == 3:
                # A label-map style array where pixel values represent
                # segment associations
                segments_present = np.unique(
                    pixel_array[pixel_array > 0].astype(np.uint16)
                )

                # The pixel values in the pixel array must all belong to
                # a described segment
                if not np.all(
                        np.in1d(segments_present, described_segment_numbers)
                    ):
                    raise ValueError(
                        'Pixel array contains segments that lack '
                        'descriptions.'
                    )

                # By construction of the pixel array, we know that the segments
                # cannot overlap
                segments_overlap = SegmentsOverlapValues.NO
            else:
                # Pixel array is 4D where each segment is stacked down
                # the last dimension
                # In this case, each segment of the pixel array should be binary
                if pixel_array.max() > 1:
                    raise ValueError(
                        'When passing a 4D stack of segments with an integer '
                        'pixel type, the pixel array must be binary.'
                    )
                pixel_array = pixel_array.astype(np.bool_)

                # Need to check whether or not segments overlap
                if pixel_array.shape[-1] == 1:
                    # A single segment does not overlap
                    segments_overlap = SegmentsOverlapValues.NO
                elif pixel_array.sum(axis=-1).max() > 1:
                    segments_overlap = SegmentsOverlapValues.YES
                else:
                    segments_overlap = SegmentsOverlapValues.NO

        elif (pixel_array.dtype in (np.float_, np.float32, np.float64)):
            unique_values = np.unique(pixel_array)
            if np.min(unique_values) < 0.0 or np.max(unique_values) > 1.0:
                raise ValueError(
                    'Floating point pixel array values must be in the '
                    'range [0, 1].'
                )
            if segmentation_type == SegmentationTypeValues.BINARY:
                non_boolean_values = np.logical_and(
                    unique_values > 0.0,
                    unique_values < 1.0
                )
                if np.any(non_boolean_values):
                    raise ValueError(
                        'Floating point pixel array values must be either '
                        '0.0 or 1.0 in case of BINARY segmentation type.'
                    )
                pixel_array = pixel_array.astype(np.bool_)

                # Need to check whether or not segments overlap
                if pixel_array.shape[-1] == 1:
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

        return pixel_array, segments_overlap

    @staticmethod
    def _omit_empty_frames(
        pixel_array: np.ndarray,
        plane_positions: Sequence[Optional[PlanePositionSequence]]
    ) -> Tuple[np.ndarray, List[Optional[PlanePositionSequence]], List[int]]:
        """Remove empty frames from the pixel array.

        Empty frames (without any positive pixels) do not need to be included
        in the segmentation image. This method removes the relevant frames
        and updates the plane positions accordingly.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Segmentation pixel array
        plane_positions: Sequence[Optional[highdicom.PlanePositionSequence]]
            Plane positions for each of the frames

        Returns
        -------
        pixel_array: numpy.ndarray
            Pixel array with empty frames removed
        plane_positions: List[Optional[highdicom.PlanePositionSequence]]
            Plane positions with entries corresponding to empty frames removed.
        source_image_indices: List[int]
            List giving for each frame in the output pixel array the index of
            the corresponding frame in the original pixel array

        """
        non_empty_frames = []
        non_empty_plane_positions = []

        # This list tracks which source image each non-empty frame came from
        source_image_indices = []
        for i, (frm, pos) in enumerate(zip(pixel_array, plane_positions)):
            if frm.sum() > 0:
                non_empty_frames.append(frm)
                non_empty_plane_positions.append(pos)
                source_image_indices.append(i)
        pixel_array = np.stack(non_empty_frames)
        plane_positions = non_empty_plane_positions

        return (pixel_array, plane_positions, source_image_indices)

    def _encode_pixels(self, planes: np.ndarray) -> bytes:
        """Encodes pixel planes.

        Parameters
        ----------
        planes: numpy.ndarray
            Array representing one or more segmentation image planes.
            For encapsulated transfer syntaxes, only a single frame may be
            processed. For other transfer syntaxes, multiple planes in a 3D
            array may be processed.

        Returns
        -------
        bytes
            Encoded pixels

        Raises
        ------
        ValueError
            If multiple frames are passed when using an encapsulated
            transfer syntax.

        """
        if self.file_meta.TransferSyntaxUID.is_encapsulated:
            # Check that only a single plane was passed
            if planes.ndim == 3:
                if planes.shape[0] == 1:
                    planes = planes[0, ...]
                else:
                    raise ValueError(
                        'Only single frame can be encoded at at time '
                        'in case of encapsulated format encoding.'
                    )
            return encode_frame(
                planes,
                transfer_syntax_uid=self.file_meta.TransferSyntaxUID,
                bits_allocated=self.BitsAllocated,
                bits_stored=self.BitsStored,
                photometric_interpretation=self.PhotometricInterpretation,
                pixel_representation=self.PixelRepresentation
            )
        else:
            # The array may represent more than one frame item.
            if self.SegmentationType == SegmentationTypeValues.BINARY.value:
                return pack_bits(planes.flatten())
            else:
                return planes.flatten().tobytes()

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'Segmentation':
        """Create instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Segmentation image.

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
        if dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.66.4':
            raise ValueError('Dataset is not a Segmentation.')
        seg = deepcopy(dataset)
        seg.__class__ = Segmentation

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

        for i, segment in enumerate(seg.SegmentSequence, 1):
            if segment.SegmentNumber != i:
                raise AttributeError(
                    'Segments are expected to start at 1 and be consecutive '
                    'integers.'
                )

        for i, s in enumerate(seg.SegmentSequence, 1):
            if s.SegmentNumber != i:
                raise ValueError(
                    'Segment numbers in the segmentation image must start at '
                    '1 and increase by 1 with the segments sequence.'
                )

        # Convert contained items to highdicom types
        # Segment descriptions
        seg.SegmentSequence = [
            SegmentDescription.from_dataset(ds) for ds in seg.SegmentSequence
        ]

        # Shared functional group elements
        if hasattr(sf_groups, 'PlanePositionSequence'):
            plane_pos = PlanePositionSequence.from_sequence(
                sf_groups.PlanePositionSequence
            )
            sf_groups.PlanePositionSequence = plane_pos
        if hasattr(sf_groups, 'PlaneOrientationSequence'):
            plane_ori = PlaneOrientationSequence.from_sequence(
                sf_groups.PlaneOrientationSequence
            )
            sf_groups.PlaneOrientationSequence = plane_ori
        if hasattr(sf_groups, 'PixelMeasuresSequence'):
            pixel_measures = PixelMeasuresSequence.from_sequence(
                sf_groups.PixelMeasuresSequence
            )
            sf_groups.PixelMeasuresSequence = pixel_measures

        # Per-frame functional group items
        for pffg_item in seg.PerFrameFunctionalGroupsSequence:
            if hasattr(pffg_item, 'PlanePositionSequence'):
                plane_pos = PlanePositionSequence.from_sequence(
                    pffg_item.PlanePositionSequence
                )
                pffg_item.PlanePositionSequence = plane_pos
            if hasattr(pffg_item, 'PlaneOrientationSequence'):
                plane_ori = PlaneOrientationSequence.from_sequence(
                    pffg_item.PlaneOrientationSequence
                )
                pffg_item.PlaneOrientationSequence = plane_ori
            if hasattr(pffg_item, 'PixelMeasuresSequence'):
                pixel_measures = PixelMeasuresSequence.from_sequence(
                    pffg_item.PixelMeasuresSequence
                )
                pffg_item.PixelMeasuresSequence = pixel_measures

        seg._build_luts()

        return cast(Segmentation, seg)

    def _build_ref_instance_lut(self) -> None:
        """Build lookup table for all instance referenced in the segmentation.

        Builds a lookup table mapping the SOPInstanceUIDs of all datasets
        referenced in the segmentation to a tuple containing the
        StudyInstanceUID, SeriesInstanceUID and SOPInstanceUID.

        """
        # Map sop uid to tuple (study uid, series uid, sop uid)
        self._ref_ins_lut = OrderedDict()
        if hasattr(self, 'ReferencedSeriesSequence'):
            for ref_series in self.ReferencedSeriesSequence:
                for ref_ins in ref_series.ReferencedInstanceSequence:
                    self._ref_ins_lut[ref_ins.ReferencedSOPInstanceUID] = (
                        hd_UID(self.StudyInstanceUID),
                        hd_UID(ref_series.SeriesInstanceUID),
                        hd_UID(ref_ins.ReferencedSOPInstanceUID)
                    )
        other_studies_kw = 'StudiesContainingOtherReferencedInstancesSequence'
        if hasattr(self, other_studies_kw):
            for ref_study in getattr(self, other_studies_kw):
                for ref_series in ref_study.ReferencedSeriesSequence:
                    for ref_ins in ref_series.ReferencedInstanceSequence:
                        self._ref_ins_lut[ref_ins.ReferencedSOPInstanceUID] = (
                            hd_UID(ref_study.StudyInstanceUID),
                            hd_UID(ref_series.SeriesInstanceUID),
                            hd_UID(ref_ins.ReferencedSOPInstanceUID)
                        )

        self._source_sop_instance_uids = np.array(
            list(self._ref_ins_lut.keys())
        )

    def _build_luts(self) -> None:
        """Build lookup tables for efficient querying.

        Two lookup tables are currently constructed. The first maps the
        SOPInstanceUIDs of all datasets referenced in the segmentation to a
        tuple containing the StudyInstanceUID, SeriesInstanceUID and
        SOPInstanceUID.

        The second look-up table contains information about each frame of the
        segmentation, including the segment it contains, the instance and frame
        from which it was derived (if these are unique), and its dimension
        index values.

        """
        self._build_ref_instance_lut()

        segnum_col_data = []
        source_instance_col_data = []
        source_frame_col_data = []

        # Get list of all dimension index pointers, excluding the segment
        # number, since this is treated differently
        seg_num_tag = tag_for_keyword('ReferencedSegmentNumber')
        self._dim_ind_pointers = [
            dim_ind.DimensionIndexPointer
            for dim_ind in self.DimensionIndexSequence
            if dim_ind.DimensionIndexPointer != seg_num_tag
        ]
        dim_ind_positions = {
            dim_ind.DimensionIndexPointer: i
            for i, dim_ind in enumerate(self.DimensionIndexSequence)
            if dim_ind.DimensionIndexPointer != seg_num_tag
        }
        dim_index_col_data: Dict[int, List[int]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }

        # Create a list of source images and check for spatial locations
        # preserved and that there is a single source frame per seg frame
        locations_list_type = List[Optional[SpatialLocationsPreservedValues]]
        locations_preserved: locations_list_type = []
        self._single_source_frame_per_seg_frame = True
        for frame_item in self.PerFrameFunctionalGroupsSequence:
            # Get segment number for this frame
            seg_id_seg = frame_item.SegmentIdentificationSequence[0]
            seg_num = seg_id_seg.ReferencedSegmentNumber
            segnum_col_data.append(int(seg_num))

            # Get dimension indices for this frame
            indices = frame_item.FrameContentSequence[0].DimensionIndexValues
            if not isinstance(indices, (MultiValue, list)):
                # In case there is a single dimension index
                indices = [indices]
            if len(indices) != len(self._dim_ind_pointers) + 1:
                # (+1 because referenced segment number is ignored)
                raise RuntimeError(
                    'Unexpected mismatch between dimension index values in '
                    'per-frames functional groups sequence and items in the '
                    'dimension index sequence.'
                )
            for ptr in self._dim_ind_pointers:
                dim_index_col_data[ptr].append(indices[dim_ind_positions[ptr]])

            frame_source_instances = []
            frame_source_frames = []
            for der_im in frame_item.DerivationImageSequence:
                for src_im in der_im.SourceImageSequence:
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
                self._single_source_frame_per_seg_frame = False
            else:
                ref_instance_uid = frame_source_instances[0]
                if ref_instance_uid not in self._source_sop_instance_uids:
                    raise AttributeError(
                        f'SOP instance {ref_instance_uid} referenced in the '
                        'source image sequence is not included in the '
                        'Referenced Series Sequence or Studies Containing '
                        'Other Referenced Instances Sequence.'
                    )
                src_uid_ind = self._get_src_uid_index(frame_source_instances[0])
                source_instance_col_data.append(src_uid_ind)
                source_frame_col_data.append(frame_source_frames[0])

        # Summarise
        if any(
            isinstance(v, SpatialLocationsPreservedValues) and
            v == SpatialLocationsPreservedValues.NO
            for v in locations_preserved
        ):
            Type = Optional[SpatialLocationsPreservedValues]
            self._locations_preserved: Type = SpatialLocationsPreservedValues.NO
        elif all(
            isinstance(v, SpatialLocationsPreservedValues) and
            v == SpatialLocationsPreservedValues.YES
            for v in locations_preserved
        ):
            self._locations_preserved = SpatialLocationsPreservedValues.YES
        else:
            self._locations_preserved = None

        # Frame LUT is a 2D numpy array. Columns represent different ways to
        # index segmentation frames.  Row i represents frame i of the
        # segmentation dataset Possible columns include the segment number, the
        # source uid index in the source_sop_instance_uids list, the source
        # frame number (if applicable), and the dimension index values of the
        # segmentation frames
        # This allows for fairly efficient querying by any of the three
        # variables: seg frame number, source instance/frame number, segment
        # number
        # Column for segment number
        self._lut_seg_col = 0
        col_data = [segnum_col_data]

        # Columns for other dimension index values
        self._lut_dim_ind_cols = {
            i: ptr for ptr, i in
            enumerate(self._dim_ind_pointers, len(col_data))
        }
        col_data += [
            indices for indices in dim_index_col_data.values()
        ]

        # Columns related to source frames, if they are usable for indexing
        if self._single_source_frame_per_seg_frame:
            self._lut_src_instance_col = len(col_data)
            self._lut_src_frame_col = len(col_data) + 1
            col_data += [
                source_instance_col_data,
                source_frame_col_data
            ]

        # Build LUT from columns
        self._frame_lut = np.array(col_data).T

    @property
    def segmentation_type(self) -> SegmentationTypeValues:
        """highdicom.seg.SegmentationTypeValues: Segmentation type."""
        return SegmentationTypeValues(self.SegmentationType)

    @property
    def segmentation_fractional_type(
        self
    ) -> Union[SegmentationFractionalTypeValues, None]:
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
        """int: The number of segments in this SEG image."""
        return len(self.SegmentSequence)

    @property
    def segment_numbers(self) -> range:
        """range: The segment numbers present in the SEG image as a range."""
        return range(1, self.number_of_segments + 1)

    def get_segment_description(
        self,
        segment_number: int
    ) -> SegmentDescription:
        """Get segment description for a segment.

        Parameters
        ----------
        segment_number: int
            Segment number for the segment, as a 1-based index.

        Returns
        -------
        highdicom.seg.SegmentDescription
            Description of the given segment.

        """
        if segment_number < 1 or segment_number > self.number_of_segments:
            raise IndexError(
                f'{segment_number} is an invalid segment number for this '
                'dataset.'
            )
        return self.SegmentSequence[segment_number - 1]

    def get_segment_numbers(
        self,
        segment_label: Optional[str] = None,
        segmented_property_category: Optional[Union[Code, CodedConcept]] = None,
        segmented_property_type: Optional[Union[Code, CodedConcept]] = None,
        algorithm_type: Optional[Union[SegmentAlgorithmTypeValues, str]] = None,
        tracking_uid: Optional[str] = None,
        tracking_id: Optional[str] = None,
    ) -> List[int]:
        """Get a list of segment numbers matching provided criteria.

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
            List of all segment numbers matching the provided criteria.

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

        return [
            desc.segment_number
            for desc in self.SegmentSequence
            if all(f(desc) for f in filter_funcs)
        ]

    def get_tracking_ids(
        self,
        segmented_property_category: Optional[Union[Code, CodedConcept]] = None,
        segmented_property_type: Optional[Union[Code, CodedConcept]] = None,
        algorithm_type: Optional[Union[SegmentAlgorithmTypeValues, str]] = None
    ) -> List[Tuple[str, UID]]:
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
    def segmented_property_categories(self) -> List[CodedConcept]:
        """Get all unique segmented property categories in this SEG image.

        Returns
        -------
        List[CodedConcept]
            All unique segmented property categories referenced in segment
            descriptions in this SEG image.

        """
        categories = []
        for desc in self.SegmentSequence:
            if desc.segmented_property_category not in categories:
                categories.append(desc.segmented_property_category)

        return categories

    @property
    def segmented_property_types(self) -> List[CodedConcept]:
        """Get all unique segmented property types in this SEG image.

        Returns
        -------
        List[CodedConcept]
            All unique segmented property types referenced in segment
            descriptions in this SEG image.

        """
        types = []
        for desc in self.SegmentSequence:
            if desc.segmented_property_type not in types:
                types.append(desc.segmented_property_type)

        return types

    def _get_src_uid_index(self, sop_instance_uid: str) -> int:
        ind = np.argwhere(self._source_sop_instance_uids == sop_instance_uid)
        if len(ind) == 0:
            raise KeyError(
                f'No such source frame: {sop_instance_uid}'
            )
        return ind.item()

    def _get_pixels_by_seg_frame(
        self,
        seg_frames_matrix: np.ndarray,
        segment_numbers: np.ndarray,
        combine_segments: bool = False,
        relabel: bool = False,
        rescale_fractional: bool = True
    ) -> np.ndarray:
        """Construct a segmentation array given an array of frame numbers.

        The output array is either 4D (combine_segments=False) or 3D
        (combine_segments=True), where dimensions are frames x rows x columes x
        segments.

        Parameters
        ----------
        seg_frames_matrix: np.ndarray
            Two dimensional numpy array containing integers. Each row of the
            array corresponds to a frame (0th dimension) of the output array.
            Each column of the array corresponds to a segment of the output
            array. Elements specify the (1-based) frame numbers of the
            segmentation image that contain the pixel data for that output
            frame and segment. A value of -1 signifies that there is no pixel
            data for that frame/segment combination.
        segment_numbers: np.ndarray
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
            ``len(segment_numbers)`` (inclusive) accoring to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.

        Returns
        -------
        pixel_array: np.ndarray
            Segmentation pixel array

        """
        # Checks on input array
        if seg_frames_matrix.ndim != 2:
            raise ValueError('Seg frames matrix must be a 2D array.')
        if not issubclass(seg_frames_matrix.dtype.type, np.integer):
            raise TypeError(
                'Seg frames matrix must have an integer data type.'
            )

        if seg_frames_matrix.min() < -1:
            raise ValueError(
                'Seg frames matrix may not contain negative values other than '
                '-1.'
            )
        if seg_frames_matrix.max() > self.NumberOfFrames:
            raise ValueError(
                'Seg frames matrix contains values outside the range of '
                'segmentation frames in the image.'
            )

        if segment_numbers.shape != (seg_frames_matrix.shape[1], ):
            raise ValueError(
                'Segment numbers array does not match the shape of the '
                'seg frames matrix.'
            )

        if (
            segment_numbers.min() < 1 or
            segment_numbers.max() > self.number_of_segments
        ):
            raise ValueError(
                'Segment numbers array contains invalid values.'
            )

        # Initialize empty pixel array
        pixel_array = np.zeros(
            (
                seg_frames_matrix.shape[0],
                self.Rows,
                self.Columns,
                seg_frames_matrix.shape[1]
            ),
            self.pixel_array.dtype
        )

        # Loop through output frames
        for out_frm, frm_row in enumerate(seg_frames_matrix):
            # Loop through segmentation frames
            for out_seg, seg_frame_num in enumerate(frm_row):
                if seg_frame_num >= 1:
                    seg_frame_ind = seg_frame_num.item() - 1
                    # Copy data to to output array
                    if self.pixel_array.ndim == 2:
                        # Special case with a single segmentation frame
                        pixel_array[out_frm, :, :, out_seg] = \
                            self.pixel_array
                    else:
                        pixel_array[out_frm, :, :, out_seg] = \
                            self.pixel_array[seg_frame_ind, :, :]

        if rescale_fractional:
            if self.segmentation_type == SegmentationTypeValues.FRACTIONAL:
                if pixel_array.max() > self.MaximumFractionalValue:
                    raise RuntimeError(
                        'Segmentation image contains values greater than the '
                        'MaximumFractionalValue recorded in the dataset.'
                    )
                max_val = self.MaximumFractionalValue
                pixel_array = pixel_array.astype(np.float32) / max_val

        if combine_segments:
            # Check whether segmentation is binary, or fractional with only
            # binary values
            if self.segmentation_type != SegmentationTypeValues.BINARY:
                if not rescale_fractional:
                    raise ValueError(
                        'In order to combine segments of a FRACTIONAL '
                        'segmentation image, rescale_fractional must be '
                        'set to True.'
                    )
                is_binary = np.isin(
                    np.unique(pixel_array),
                    np.array([0.0, 1.0]),
                    assume_unique=True
                )
                if not is_binary.all():
                    raise ValueError(
                        'Cannot combine segments if the segmentation is not'
                        'binary'
                    )

            # Check for overlap by summing over the segments dimension
            if np.any(pixel_array.sum(axis=-1) > 1):
                raise RuntimeError(
                    'Segments cannot be combined because they overlap'
                )

            # Scale the array by the segment number using broadcasting
            if relabel:
                pixel_value_map = np.arange(1, len(segment_numbers) + 1)
            else:
                pixel_value_map = segment_numbers
            scaled_array = pixel_array * pixel_value_map.reshape(1, 1, 1, -1)

            # Combine segments by taking maximum down final dimension
            max_array = scaled_array.max(axis=-1)
            pixel_array = max_array

        return pixel_array

    def get_source_image_uids(self) -> List[Tuple[hd_UID, hd_UID, hd_UID]]:
        """Get UIDs for all source SOP instances referenced in the dataset.

        Returns
        -------
        List[Tuple[highdicom.UID, highdicom.UID, highdicom.UID]]
            List of tuples containing Study Instance UID, Series Instance UID
            and SOP Instance UID for every SOP Instance referenced in the
            dataset.

        """
        return [self._ref_ins_lut[sop_uid] for sop_uid in self._ref_ins_lut]

    def get_default_dimension_index_pointers(
        self
    ) -> List[BaseTag]:
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
        return self._dim_ind_pointers[:]

    def are_dimension_indices_unique(
        self,
        dimension_index_pointers: Sequence[Union[int, BaseTag]]
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
        for ptr in dimension_index_pointers:
            if ptr not in self._dim_ind_pointers:
                kw = keyword_for_tag(ptr)
                if kw == '':
                    kw = '<no keyword>'
                raise KeyError(
                    f'Tag {ptr} ({kw}) is not used as a dimension index '
                    'in this image.'
                )
        # Create the sub-matrix of the look up table that indexes
        # by the dimension index values
        dim_ind_cols = [
            self._lut_dim_ind_cols[ptr]
            for ptr in dimension_index_pointers
        ]
        lut = self._frame_lut[:, dim_ind_cols + [self._lut_seg_col]]

        return np.unique(lut, axis=0).shape[0] == lut.shape[0]

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
            * At least one frame in the segmentation lists multiple
              source frames.

        Parameters
        ----------
        ignore_spatial_locations: bool
            Allows the user to ignore whether spatial locations are preserved
            in the frames.

        """
        # Checks that it is possible to index using source frames in this
        # dataset
        if self._locations_preserved is None:
            if not ignore_spatial_locations:
                raise RuntimeError(
                    'Indexing via source frames is not permissible since this '
                    'image does not specify that spatial locations are '
                    'preserved in the course of deriving the segmentation '
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
                    'in the course of deriving the segmentation from the '
                    'source image. If you do not require that spatial '
                    ' locations are preserved you may override this behavior '
                    "with the 'ignore_spatial_locations' parameter."
                )
        if not self._single_source_frame_per_seg_frame:
            raise RuntimeError(
                'Indexing via source frames is not permissible since some '
                'frames in the segmentation specify multiple source frames.'
            )

    def get_pixels_by_source_instance(
        self,
        source_sop_instance_uids: Sequence[str],
        segment_numbers: Optional[Sequence[int]] = None,
        combine_segments: bool = False,
        relabel: bool = False,
        ignore_spatial_locations: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True
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
            ``len(segment_numbers)`` (inclusive) accoring to the position of
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

        Returns
        -------
        pixel_array: np.ndarray
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
        self._check_indexing_with_source_frames(ignore_spatial_locations)

        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
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

        # Initialize seg frame numbers matrix with value signifying
        # "empty" (-1)
        rows = len(source_sop_instance_uids)
        cols = len(segment_numbers)
        seg_frames = -np.ones(shape=(rows, cols), dtype=np.int32)

        # Sub-matrix of the LUT used for indexing by instance and
        # segment number
        query_cols = [self._lut_src_instance_col, self._lut_seg_col]
        lut = self._frame_lut[:, query_cols]

        # Check for uniqueness
        if np.unique(lut, axis=0).shape[0] != lut.shape[0]:
            raise RuntimeError(
                'Source SOP instance UIDs and segment numbers do not '
                'uniquely identify frames of the segmentation image.'
            )

        # Build the segmentation frame matrix
        for r, sop_uid in enumerate(source_sop_instance_uids):
            # Check whether this source UID exists in the LUT
            try:
                src_uid_ind = self._get_src_uid_index(sop_uid)
            except KeyError as e:
                if assert_missing_frames_are_empty:
                    continue
                else:
                    msg = (
                        f'SOP Instance UID {sop_uid} does not match any '
                        'referenced source frames. To return an empty '
                        'segmentation mask in this situation, use the '
                        "'assert_missing_frames_are_empty' parameter."
                    )
                    raise KeyError(msg) from e

            # Iterate over segment numbers for this source instance
            for c, seg_num in enumerate(segment_numbers):
                # Use LUT to find the segmentation frame containing
                # the source frame and segment number
                qry = np.array([src_uid_ind, seg_num])
                seg_frm_indices = np.where(np.all(lut == qry, axis=1))[0]
                if len(seg_frm_indices) == 1:
                    seg_frames[r, c] = seg_frm_indices[0] + 1
                elif len(seg_frm_indices) > 0:
                    # This should never happen due to earlier checks
                    # on unique rows
                    raise RuntimeError(
                        'An unexpected error was encountered during '
                        'indexing of the segmentation image. Please '
                        'file a bug report with the highdicom repository.'
                    )
                # else no segmentation frame found for this segment number,
                # assume this frame is empty and leave the entry in seg_frames
                # as -1

        return self._get_pixels_by_seg_frame(
            seg_frames_matrix=seg_frames,
            segment_numbers=np.array(segment_numbers),
            combine_segments=combine_segments,
            relabel=relabel,
            rescale_fractional=rescale_fractional
        )

    def get_pixels_by_source_frame(
        self,
        source_sop_instance_uid: str,
        source_frame_numbers: Sequence[int],
        segment_numbers: Optional[Sequence[int]] = None,
        combine_segments: bool = False,
        relabel: bool = False,
        ignore_spatial_locations: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True
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

        Parameters
        ----------
        source_sop_instance_uid: str
            SOP Instance UID of the source instance that contains the source
            frames.
        source_frame_numbers: Sequence[int]
            A sequence of frame numbers (1-based) within the source instance
            for which segmentations are requested.
        segment_numbers: Sequence[int, None], optional
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
            ``len(segment_numbers)`` (inclusive) accoring to the position of
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
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.

        Returns
        -------
        pixel_array: np.ndarray
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
        ((3, 10, 10), array([ 0,  6,  8, 10]))

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
        ((3, 10, 10), array([0, 1, 2, 3]))

        """
        # Check that indexing in this way is possible
        self._check_indexing_with_source_frames(ignore_spatial_locations)

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

        rows = len(source_frame_numbers)
        cols = len(segment_numbers)
        seg_frames = -np.ones(shape=(rows, cols), dtype=np.int32)

        # Create the sub-matrix of the look up table that indexes
        # by frame number and segment number within this
        # instance
        src_uid_ind = self._get_src_uid_index(source_sop_instance_uid)
        col = self._lut_src_instance_col
        lut_instance_mask = self._frame_lut[:, col] == src_uid_ind
        lut = self._frame_lut[lut_instance_mask, :]
        query_cols = [self._lut_src_frame_col, self._lut_seg_col]
        lut = lut[:, query_cols]

        if np.unique(lut, axis=0).shape[0] != lut.shape[0]:
            raise RuntimeError(
                'Source frame numbers and segment numbers do not '
                'uniquely identify frames of the segmentation image.'
            )

        # Build the segmentation frame matrix
        for r, src_frm_num in enumerate(source_frame_numbers):
            # Check whether this source frame exists in the LUT
            if src_frm_num not in lut[:, 0]:
                if assert_missing_frames_are_empty:
                    continue
                else:
                    msg = (
                        f'Source frame number {src_frm_num} does not '
                        'match any referenced source frame. To return '
                        'an empty segmentation mask in this situation, '
                        "use the 'assert_missing_frames_are_empty' "
                        'parameter.'
                    )
                    raise ValueError(msg)

            # Iterate over segment numbers for this source frame
            for c, seg_num in enumerate(segment_numbers):
                # Use LUT to find the segmentation frame containing
                # the source frame and segment number
                qry = np.array([src_frm_num, seg_num])
                seg_frm_indices = np.where(np.all(lut == qry, axis=1))[0]
                if len(seg_frm_indices) == 1:
                    seg_frames[r, c] = seg_frm_indices[0] + 1
                elif len(seg_frm_indices) > 0:
                    # This should never happen due to earlier checks
                    # on unique rows
                    raise RuntimeError(
                        'An unexpected error was encountered during '
                        'indexing of the segmentation image. Please '
                        'file a bug report with the highdicom repository.'
                    )
                # else no segmentation frame found for this segment number,
                # assume this frame is empty and leave the entry in seg_frames
                # as -1

        return self._get_pixels_by_seg_frame(
            seg_frames_matrix=seg_frames,
            segment_numbers=np.array(segment_numbers),
            combine_segments=combine_segments,
            relabel=relabel,
            rescale_fractional=rescale_fractional
        )

    def get_pixels_by_dimension_index_values(
        self,
        dimension_index_values: Sequence[Sequence[int]],
        dimension_index_pointers: Optional[Sequence[int]] = None,
        segment_numbers: Optional[Sequence[int]] = None,
        combine_segments: bool = False,
        relabel: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True
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
            ``len(segment_numbers)`` (inclusive) accoring to the position of
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

        Returns
        -------
        pixel_array: np.ndarray
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

        if dimension_index_pointers is None:
            dimension_index_pointers = self._dim_ind_pointers
        else:
            for ptr in dimension_index_pointers:
                if ptr not in self._dim_ind_pointers:
                    kw = keyword_for_tag(ptr)
                    if kw == '':
                        kw = '<no keyword>'
                    raise KeyError(
                        f'Tag {Tag(ptr)} ({kw}) is not used as a dimension '
                        'index in this image.'
                    )

        if len(dimension_index_values) == 0:
            raise ValueError(
                'Dimension index values should not be empty.'
            )
        if len(dimension_index_pointers) == 0:
            raise ValueError(
                'Dimension index pointers should not be empty.'
            )
        rows = len(dimension_index_values)
        cols = len(segment_numbers)
        seg_frames = -np.ones(shape=(rows, cols), dtype=np.int32)

        # Create the sub-matrix of the look up table that indexes
        # by the dimension index values
        dim_ind_cols = [
            self._lut_dim_ind_cols[ptr]
            for ptr in dimension_index_pointers
        ]
        lut = self._frame_lut[:, dim_ind_cols + [self._lut_seg_col]]

        if np.unique(lut, axis=0).shape[0] != lut.shape[0]:
            raise RuntimeError(
                'The chosen dimension indices do not uniquely identify '
                'frames of the segmentation image. You may need to provide '
                'further indices to disambiguate.'
            )

        # Build the segmentation frame matrix
        for r, ind_vals in enumerate(dimension_index_values):
            if len(ind_vals) != len(dimension_index_pointers):
                raise ValueError(
                    'Number of provided indices does not match the expected '
                    'number.'
                )
            if not all(v > 0 for v in ind_vals):
                raise ValueError(
                    'Indices are 1-based and must be greater than 1.'
                )

            # Check whether this frame exists in the LUT at all, ignoring the
            # segment number column
            qry = np.array(ind_vals)
            seg_frm_indices = np.where(np.all(lut[:, :-1] == qry, axis=1))[0]
            if len(seg_frm_indices) == 0:
                if assert_missing_frames_are_empty:
                    # Nothing more to do
                    continue
                else:
                    raise RuntimeError(
                        f'No frame with dimension index values {ind_vals} '
                        'found in the segmentation image. To return a frame of '
                        'zeros in this situation, set the '
                        "'assert_missing_frames_are_empty' parameter to True."
                    )

            # Iterate over requested segment numbers for this source frame
            for c, seg_num in enumerate(segment_numbers):
                # Use LUT to find the segmentation frame containing
                # the index values and segment number
                qry = np.array(list(ind_vals) + [seg_num])
                seg_frm_indices = np.where(np.all(lut == qry, axis=1))[0]
                if len(seg_frm_indices) == 1:
                    seg_frames[r, c] = seg_frm_indices[0] + 1
                elif len(seg_frm_indices) > 0:
                    # This should never happen due to earlier checks
                    # on unique rows
                    raise RuntimeError(
                        'An unexpected error was encountered during '
                        'indexing of the segmentation image. Please '
                        'file a bug report with the highdicom repository.'
                    )
                # else no segmentation frame found for this segment number,
                # assume this frame is empty and leave the entry in seg_frames
                # as -1

        return self._get_pixels_by_seg_frame(
            seg_frames_matrix=seg_frames,
            segment_numbers=np.array(segment_numbers),
            combine_segments=combine_segments,
            relabel=relabel,
            rescale_fractional=rescale_fractional
        )


def segread(fp: Union[str, bytes, PathLike, BinaryIO]) -> Segmentation:
    """Read a segmentation image stored in DICOM File Format.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing a
        Segmentation image.

    Returns
    -------
    highdicom.seg.Segmentation
        Segmentation image read from the file.

    """
    return Segmentation.from_dataset(dcmread(fp))
