"""Module for the SOP class of the Segmentation IOD."""
import logging
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Sequence, Union, Tuple

from pydicom.dataset import Dataset
from pydicom.encaps import decode_data_sequence, encapsulate
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from pydicom.pixel_data_handlers.util import get_expected_length
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    RLELossless,
    UID,
)
from pydicom.sr.codedict import codes
from pydicom.valuerep import PersonName

from highdicom.base import SOPClass
from highdicom.content import (
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
)
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import check_person_name


logger = logging.getLogger(__name__)


class Segmentation(SOPClass):

    """SOP class for a Segmentation, which represents one or more
    regions of interest (ROIs) as mask images (raster graphics) in
    two-dimensional image space.

    """

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
            transfer_syntax_uid: Union[str, UID] = ImplicitVRLittleEndian,
            pixel_measures: Optional[PixelMeasuresSequence] = None,
            plane_orientation: Optional[PlaneOrientationSequence] = None,
            plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
            omit_empty_frames: bool = True,
            **kwargs: Any
        ) -> None:
        """
        Parameters
        ----------
        source_images: Sequence[pydicom.dataset.Dataset]
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

            If `pixel_array` represents the segmentation of a 3D image, the
            first dimension represents individual 2D planes and these planes
            must be ordered based on their position in the three-dimensional
            patient coordinate system (first along the X axis, second along the
            Y axis, and third along the Z axis).

            If `pixel_array` is a 3D array representing the segmentation of a
            tiled 2D image, the first dimension represents individual 2D tiles
            (for one channel and z-stack) and these tiles must be ordered based
            on their position in the tiled total pixel matrix (first along the
            row dimension and second along the column dimension, which are
            defined in the three-dimensional slide coordinate system by the
            direction cosines encoded by the *Image Orientation (Slide)*
            attribute).

            If `pixel_array` is an unsigned integer or boolean array with
            binary data (containing only the values ``True`` and ``False`` or
            ``0`` and ``1``) or a floating-point array, it represents a single
            segment. In the case of a floating-point array, values must be in
            the range 0.0 to 1.0.

            Otherwise, if `pixel_array` is a 2D or 3D array containing multiple
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

            If `pixel_array` is a 4D numpy array, the first three dimensions
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
        series_number: Union[int, None]
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
        fractional_type: Union[str, highdicom.seg.SegmentationFractionalTypeValues], optional
            Type of fractional segmentation that indicates how pixel data
            should be interpreted
        max_fractional_value: int, optional
            Maximum value that indicates probability or occupancy of 1 that
            a pixel represents a given segment
        content_description: str, optional
            Description of the segmentation
        content_creator_name: Optional[Union[str, PersonName]], optional
            Name of the creator of the segmentation
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported for encapsulated format encoding in case of
            FRACTIONAL segmentation type:
            RLE Lossless (``"1.2.840.10008.1.2.5"``) and
            JPEG 2000 Lossless (``"1.2.840.10008.1.2.4.90"``).
        pixel_measures: PixelMeasures, optional
            Physical spacing of image pixels in `pixel_array`.
            If ``None``, it will be assumed that the segmentation image has the
            same pixel measures as the source image(s).
        plane_orientation: highdicom.PlaneOrientationSequence, optional
            Orientation of planes in `pixel_array` relative to axes of
            three-dimensional patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image as the
            same plane orientation as the source image(s).
        plane_positions: Sequence[highdicom.PlanePositionSequence], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image has the
            same plane position as the source image(s). However, this will only
            work when the first dimension of `pixel_array` matches the number
            of frames in `source_images` (in case of multi-frame source images)
            or the number of `source_images` (in case of single-frame source
            images).
        omit_empty_frames: bool
            If True (default), frames with no non-zero pixels are omitted from
            the segmentation image. If False, all frames are included.
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


        """  # noqa
        if len(source_images) == 0:
            raise ValueError('At least one source image is required.')
        self._source_images = source_images

        uniqueness_criteria = set(
            (
                image.StudyInstanceUID,
                image.SeriesInstanceUID,
                image.Rows,
                image.Columns,
            )
            for image in self._source_images
        )
        if len(uniqueness_criteria) > 1:
            raise ValueError(
                'Source images must all be part of the same series and must '
                'have the same image dimensions (number of rows/columns).'
            )

        src_img = self._source_images[0]
        is_multiframe = hasattr(src_img, 'NumberOfFrames')
        if is_multiframe and len(self._source_images) > 1:
            raise ValueError(
                'Only one source image should be provided in case images '
                'are multi-frame images.'
            )
        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            JPEG2000Lossless,
            RLELossless,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                'Transfer syntax "{}" is not supported'.format(
                    transfer_syntax_uid
                )
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
            **kwargs
        )

        # Using Container Type Code Sequence attribute would be more elegant,
        # but unfortunately it is a type 2 attribute.
        if (hasattr(src_img, 'ImageOrientationSlide') or
                hasattr(src_img, 'ImageCenterPointCoordinatesSequence')):
            self._coordinate_system = CoordinateSystemNames.SLIDE
        else:
            self._coordinate_system = CoordinateSystemNames.PATIENT

        # Frame of Reference
        self.FrameOfReferenceUID = src_img.FrameOfReferenceUID
        self.PositionReferenceIndicator = getattr(
            src_img,
            'PositionReferenceIndicator',
            None
        )

        # (Enhanced) General Equipment
        self.DeviceSerialNumber = device_serial_number
        self.ManufacturerModelName = manufacturer_model_name
        self.SoftwareVersions = software_versions

        # General Reference
        self.SourceImageSequence: List[Dataset] = []
        referenced_series: Dict[str, List[Dataset]] = defaultdict(list)
        for s_img in self._source_images:
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
        self.ContentLabel = 'ISO_IR 192'  # UTF-8
        self.ContentDescription = content_description
        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name

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

        # will be updated by "add_segments()"
        self.SegmentSequence: List[Dataset] = []

        # Multi-Frame Functional Groups and Multi-Frame Dimensions
        shared_func_groups = Dataset()
        if pixel_measures is None:
            if is_multiframe:
                src_shared_fg = src_img.SharedFunctionalGroupsSequence[0]
                pixel_measures = src_shared_fg.PixelMeasuresSequence
            else:
                pixel_measures = PixelMeasuresSequence(
                    pixel_spacing=src_img.PixelSpacing,
                    slice_thickness=src_img.SliceThickness,
                    spacing_between_slices=src_img.get(
                        'SpacingBetweenSlices',
                        None
                    )
                )
            # TODO: ensure derived segmentation image and original image have
            # same physical dimensions
            # seg_row_dim = self.Rows * pixel_measures[0].PixelSpacing[0]
            # seg_col_dim = self.Columns * pixel_measures[0].PixelSpacing[1]
            # src_row_dim = src_img.Rows

        if is_multiframe:
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                source_plane_orientation = PlaneOrientationSequence(
                    coordinate_system=self._coordinate_system,
                    image_orientation=src_img.ImageOrientationSlide
                )
            else:
                src_sfg = src_img.SharedFunctionalGroupsSequence[0]
                source_plane_orientation = src_sfg.PlaneOrientationSequence
        else:
            source_plane_orientation = PlaneOrientationSequence(
                coordinate_system=self._coordinate_system,
                image_orientation=src_img.ImageOrientationPatient
            )
        if plane_orientation is None:
            plane_orientation = source_plane_orientation
        self._plane_orientation = plane_orientation
        self._source_plane_orientation = source_plane_orientation

        self.DimensionIndexSequence = DimensionIndexSequence(
            coordinate_system=self._coordinate_system
        )
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = \
            self.DimensionIndexSequence[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence = [dimension_organization]

        if is_multiframe:
            self._source_plane_positions = \
                self.DimensionIndexSequence.get_plane_positions_of_image(
                    self._source_images[0]
                )
        else:
            self._source_plane_positions = \
                self.DimensionIndexSequence.get_plane_positions_of_series(
                    self._source_images
                )

        shared_func_groups.PixelMeasuresSequence = pixel_measures
        shared_func_groups.PlaneOrientationSequence = plane_orientation
        self.SharedFunctionalGroupsSequence = [shared_func_groups]

        # NOTE: Information about individual frames will be updated below
        self.NumberOfFrames = 0
        self.PerFrameFunctionalGroupsSequence: List[Dataset] = []

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        if pixel_array.ndim not in [3, 4]:
            raise ValueError('Pixel array must be a 2D, 3D,or 4D array.')

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

        if pixel_array.ndim == 4:
            # Check that the number of segments in the array matches
            if pixel_array.shape[-1] != len(segment_descriptions):
                raise ValueError(
                    'The number of segments in last dimension of the pixel '
                    f'array ({pixel_array.shape[-1]}) does not match the '
                    'number of described segments '
                    f'({len(segment_descriptions)}).'
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
                self.SegmentsOverlap = SegmentsOverlapValues.NO.value
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
                    self.SegmentsOverlap = SegmentsOverlapValues.NO.value
                elif pixel_array.sum(axis=-1).max() > 1:
                    self.SegmentsOverlap = SegmentsOverlapValues.YES.value
                else:
                    self.SegmentsOverlap = SegmentsOverlapValues.NO.value

        elif (pixel_array.dtype in (np.float_, np.float32, np.float64)):
            unique_values = np.unique(pixel_array)
            if np.min(unique_values) < 0.0 or np.max(unique_values) > 1.0:
                raise ValueError(
                    'Floating point pixel array values must be in the '
                    'range [0, 1].'
                )
            if self.SegmentationType == SegmentationTypeValues.BINARY.value:
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
                    self.SegmentsOverlap = SegmentsOverlapValues.NO.value
                elif pixel_array.sum(axis=-1).max() > 1:
                    self.SegmentsOverlap = SegmentsOverlapValues.YES.value
                else:
                    self.SegmentsOverlap = SegmentsOverlapValues.NO.value
            else:
                if (pixel_array.ndim == 3) or (pixel_array.shape[-1] == 1):
                    # A single segment does not overlap
                    self.SegmentsOverlap = SegmentsOverlapValues.NO.value
                else:
                    # A truly fractional segmentation with multiple segments.
                    # Unclear how overlap should be interpreted in this case
                    self.SegmentsOverlap = SegmentsOverlapValues.UNDEFINED.value
        else:
            raise TypeError('Pixel array has an invalid data type.')

        if plane_positions is None:
            if pixel_array.shape[0] != len(self._source_plane_positions):
                raise ValueError(
                    'Number of frames in pixel array does not match number '
                    'of source image frames.'
                )
            plane_positions = self._source_plane_positions
        else:
            if pixel_array.shape[0] != len(plane_positions):
                raise ValueError(
                    'Number of pixel array planes does not match number of '
                    'provided plane positions.'
                )

        are_spatial_locations_preserved = (
            all(
                plane_positions[i] == self._source_plane_positions[i]
                for i in range(len(plane_positions))
            ) and
            self._plane_orientation == self._source_plane_orientation
        )

        # Remove empty slices
        if omit_empty_frames:
            non_empty_frames = []
            non_empty_plane_positions = []
            source_image_indices = []
            for i, (frm, pos) in enumerate(zip(pixel_array, plane_positions)):
                if frm.sum() > 0:
                    non_empty_frames.append(frm)
                    non_empty_plane_positions.append(pos)
                    source_image_indices.append(i)
            pixel_array = np.stack(non_empty_frames)
            plane_positions = non_empty_plane_positions
        else:
            source_image_indices = list(range(pixel_array.shape[0]))

        plane_position_values, plane_sort_index = \
            self.DimensionIndexSequence.get_index_values(plane_positions)

        # Get unique values of attributes in the Plane Position Sequence or
        # Plane Position Slide Sequence, which define the position of the plane
        # with respect to the three dimensional patient or slide coordinate
        # system, respectively. These can subsequently be used to look up the
        # relative position of a plane relative to the indexed dimension.
        dimension_position_values = [
            np.unique(plane_position_values[:, index], axis=0)
            for index in range(plane_position_values.shape[1])
        ]

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
                planes = np.zeros(pixel_array.shape, dtype=np.bool_)
                planes[pixel_array == segment_number] = True
            elif pixel_array.dtype == np.bool_:
                if pixel_array.ndim == 4:
                    planes = pixel_array[:, :, :, segment_number - 1]
                else:
                    planes = pixel_array
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
                frame_content_item.DimensionIndexValues = [segment_number]

                # Look up the position of the plane relative to the indexed
                # dimension.
                try:
                    if self._coordinate_system == CoordinateSystemNames.SLIDE:
                        index_values = [
                            np.where(
                                (dimension_position_values[idx] == pos)
                            )[0][0] + 1
                            for idx, pos in enumerate(plane_position_values[j])
                        ]
                    else:
                        # In case of the patient coordinate system, the
                        # value of the attribute the Dimension Index Sequence
                        # points to (Image Position Patient) has a value
                        # multiplicity greater than one.
                        index_values = [
                            np.where(
                                (dimension_position_values[idx] == pos).all(
                                    axis=1
                                )
                            )[0][0] + 1
                            for idx, pos in enumerate(plane_position_values[j])
                        ]
                except IndexError as error:
                    raise IndexError(
                        'Could not determine position of plane #{} in '
                        'three dimensional coordinate system based on '
                        'dimension index values: {}'.format(j, error)
                    )
                frame_content_item.DimensionIndexValues.extend(index_values)
                pffp_item.FrameContentSequence = [frame_content_item]
                if self._coordinate_system == CoordinateSystemNames.SLIDE:
                    pffp_item.PlanePositionSlideSequence = plane_positions[j]
                else:
                    pffp_item.PlanePositionSequence = plane_positions[j]

                # Determining the source images that map to the frame is not
                # always trivial. Since DerivationImageSequence is a type 2
                # attribute, we leave its value empty.
                pffp_item.DerivationImageSequence = []

                if are_spatial_locations_preserved:
                    derivation_image_item = Dataset()
                    derivation_code = codes.cid7203.Segmentation
                    derivation_image_item.DerivationCodeSequence = [
                        CodedConcept(
                            derivation_code.value,
                            derivation_code.scheme_designator,
                            derivation_code.meaning,
                            derivation_code.scheme_version
                        ),
                    ]

                    derivation_src_img_item = Dataset()
                    if hasattr(self._source_images[0], 'NumberOfFrames'):
                        # A single multi-frame source image
                        src_img_item = self.SourceImageSequence[0]
                        # Frame numbers are one-based
                        derivation_src_img_item.ReferencedFrameNumber = (
                            source_image_index + 1
                        )
                    else:
                        # Multiple single-frame source images
                        src_img_item = self.SourceImageSequence[source_image_index]
                    derivation_src_img_item.ReferencedSOPClassUID = \
                        src_img_item.ReferencedSOPClassUID
                    derivation_src_img_item.ReferencedSOPInstanceUID = \
                        src_img_item.ReferencedSOPInstanceUID
                    purpose_code = \
                        codes.cid7202.SourceImageForImageProcessingOperation
                    derivation_src_img_item.PurposeOfReferenceCodeSequence = [
                        CodedConcept(
                            purpose_code.value,
                            purpose_code.scheme_designator,
                            purpose_code.meaning,
                            purpose_code.scheme_version
                        ),
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

    def add_segments(
        self,
        pixel_array: np.ndarray,
        segment_descriptions: Sequence[SegmentDescription],
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
        omit_empty_frames: bool = True,
    ) -> None:
        raise DeprecationWarning(
            'To ensure correctness of segmentation images, the add_segments '
            'method was deprecated in highdicom 0.8.0. For more information '
            'and migration instructions visit '
            'https://highdicom.readthedocs.io/en/latest/???????'  # TODO
        )
