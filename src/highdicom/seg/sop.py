"""Module for the SOP class of the Segmentation IOD."""
import itertools
import logging
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Sequence, Union, Tuple

from pydicom.dataset import Dataset
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from pydicom.pixel_data_handlers.util import get_expected_length
from pydicom.uid import UID
from pydicom.sr.codedict import codes
from pydicom._storage_sopclass_uids import (
    SegmentationStorage,
    VLSlideCoordinatesMicroscopicImageStorage,
    VLWholeSlideMicroscopyImageStorage,
)

from highdicom.base import SOPClass
from highdicom.content import (
    PlaneOrientationSequence,
    PlanePositionSequence,
    PixelMeasuresSequence
)
from highdicom.enum import CoordinateSystemNames
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
from highdicom.utils import compute_plane_positions_tiled_full


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
            content_creator_name: Optional[str] = None,
            transfer_syntax_uid: Union[str, UID] = '1.2.840.10008.1.2',
            pixel_measures: Optional[PixelMeasuresSequence] = None,
            plane_orientation: Optional[PlaneOrientationSequence] = None,
            plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
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
            floating point data type representing a mask image. If `pixel_array`
            is a floating-point array or a binary array (containing only the
            values ``True`` and ``False`` or ``0`` and ``1``), the segment number
            used to encode the segment is taken from segment_descriptions.
            Otherwise, if pixel_array contains multiple integer values, each value
            is treated as a different segment whose segment number is that integer
            value. In this case, all segments found in the array must be described
            in `segment_descriptions`. Note that this is valid for both ``"BINARY"``
            and ``"FRACTIONAL"`` segmentations.
            For ``"FRACTIONAL"`` segmentations, values either encode the probability
            of a given pixel belonging to a segment
            (if `fractional_type` is ``"PROBABILITY"``)
            or the extent to which a segment occupies the pixel
            (if `fractional_type` is ``"OCCUPANCY"``).
            When `pixel_array` has a floating point data type, only one segment can be
            encoded. Additional segments can be subsequently
            added to the `Segmentation` instance using the ``add_segments()``
            method.
            If `pixel_array` represents a 3D image, the first dimension
            represents individual 2D planes and these planes must be ordered
            based on their position in the three-dimensional patient
            coordinate system (first along the X axis, second along the Y axis,
            and third along the Z axis).
            If `pixel_array` represents a tiled 2D image, the first dimension
            represents individual 2D tiles (for one channel and z-stack) and
            these tiles must be ordered based on their position in the tiled
            total pixel matrix (first along the row dimension and second along
            the column dimension, which are defined in the three-dimensional
            slide coordinate system by the direction cosines encoded by the
            *Image Orientation (Slide)* attribute).
        segmentation_type: Union[str, highdicom.seg.enum.SegmentationTypeValues]
            Type of segmentation, either ``"BINARY"`` or ``"FRACTIONAL"``
        segment_descriptions: Sequence[highdicom.seg.content.SegmentDescription]
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
        fractional_type: Union[str, highdicom.seg.enum.SegmentationFractionalTypeValues], optional
            Type of fractional segmentation that indicates how pixel data
            should be interpreted
        max_fractional_value: int, optional
            Maximum value that indicates probability or occupancy of 1 that
            a pixel represents a given segment
        content_description: str, optional
            Description of the segmentation
        content_creator_name: str, optional
            Name of the creator of the segmentation
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported: JPEG2000 (``"1.2.840.10008.1.2.4.90"``) and
            JPEG-LS (``"1.2.840.10008.1.2.4.80"``). Lossy compression is not
            supported.
        pixel_measures: PixelMeasures, optional
            Physical spacing of image pixels in `pixel_array`.
            If ``None``, it will be assumed that the segmentation image has the
            same pixel measures as the source image(s).
        plane_orientation: highdicom.content.PlaneOrientationSequence, optional
            Orientation of planes in `pixel_array` relative to axes of
            three-dimensional patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image as the
            same plane orientation as the source image(s).
        plane_positions: Sequence[highdicom.content.PlanePositionSequence], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image has the
            same plane position as the source image(s). However, this will only
            work when the first dimension of `pixel_array` matches the number
            of frames in `source_images` (in case of multi-frame source images)
            or the number of `source_images` (in case of single-frame source
            images).
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
            '1.2.840.10008.1.2',       # Implicit Little Endian
            '1.2.840.10008.1.2.1',     # Explicit Little Endian
            # '1.2.840.10008.1.2.4.90',  # JPEG2000
            # '1.2.840.10008.1.2.4.80',  # JPEG-LS
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
            sop_class_uid=SegmentationStorage,
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
            referring_physician_name=src_img.ReferringPhysicianName,
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
        self.ContentCreatorName = content_creator_name

        segmentation_type = SegmentationTypeValues(segmentation_type)
        self.SegmentationType = segmentation_type.value
        if self.SegmentationType == SegmentationTypeValues.BINARY.value:
            self.BitsAllocated = 1
            self.HighBit = 0
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
            # Do we need to take ImageOrientationPatient/ImageOrientationPatient
            # into account?

        if is_multiframe:
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                source_plane_orientation = PlaneOrientationSequence(
                    coordinate_system=self._coordinate_system,
                    image_orientation=src_img.ImageOrientationSlide
                )
                if src_img.SOPClassUID == VLWholeSlideMicroscopyImageStorage:
                    self.TotalPixelMatrixRows = src_img.TotalPixelMatrixRows
                    self.TotalPixelMatrixColumns = \
                        src_img.TotalPixelMatrixColumns
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
        self._source_plane_orientation = plane_orientation

        self.DimensionIndexSequence = DimensionIndexSequence(
            coordinate_system=self._coordinate_system
        )
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = \
            self.DimensionIndexSequence[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence = [dimension_organization]

        shared_func_groups.PixelMeasuresSequence = pixel_measures
        shared_func_groups.PlaneOrientationSequence = plane_orientation
        self.SharedFunctionalGroupsSequence = [shared_func_groups]

        # NOTE: Information about individual frames will be updated by the
        # "add_segments()" method upon addition of segmentation bitplanes.
        self.NumberOfFrames = 0
        self.PerFrameFunctionalGroupsSequence: List[Dataset] = []

        self._segment_inventory: Set[int] = set()
        self.PixelData = b''
        self.add_segments(
            pixel_array=pixel_array,
            segment_descriptions=segment_descriptions,
            plane_positions=plane_positions
        )

        self.copy_specimen_information(src_img)
        self.copy_patient_and_study_information(src_img)

    def add_segments(
            self,
            pixel_array: np.ndarray,
            segment_descriptions: Sequence[SegmentDescription],
            plane_positions: Optional[Sequence[PlanePositionSequence]] = None
        ) -> Dataset:
        """Adds one or more segments to the segmentation image.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Array of segmentation pixel data of boolean, unsigned integer or
            floating point data type representing a mask image. If `pixel_array`
            is a floating-point array or a binary array (containing only the
            values ``True`` and ``False`` or ``0`` and ``1``), the segment number
            used to encode the segment is taken from segment_descriptions.
            Otherwise, if pixel_array contains multiple integer values, each value
            is treated as a different segment whose segment number is that integer
            value. In this case, all segments found in the array must be described
            in `segment_descriptions`. Note that this is valid for both ``"BINARY"``
            and ``"FRACTIONAL"`` segmentations.
            For ``"FRACTIONAL"`` segmentations, values either encode the probability
            of a given pixel belonging to a segment
            (if `fractional_type` is ``"PROBABILITY"``)
            or the extent to which a segment occupies the pixel
            (if `fractional_type` is ``"OCCUPANCY"``).
            When `pixel_array` has a floating point data type, only one segment can be
            encoded. Additional segments can be subsequently
            added to the `Segmentation` instance using the ``add_segments()``
            method.
            If `pixel_array` represents a 3D image, the first dimension
            represents individual 2D planes and these planes must be ordered
            based on their position in the three-dimensional patient
            coordinate system (first along the X axis, second along the Y axis,
            and third along the Z axis).
            If `pixel_array` represents a tiled 2D image, the first dimension
            represents individual 2D tiles (for one channel and z-stack) and
            these tiles must be ordered based on their position in the tiled
            total pixel matrix (first along the row dimension and second along
            the column dimension, which are defined in the three-dimensional
            slide coordinate system by the direction cosines encoded by the
            *Image Orientation (Slide)* attribute).
        segment_descriptions: Sequence[highdicom.seg.content.SegmentDescription]
            Description of each segment encoded in `pixel_array`. In the case of
            pixel arrays with multiple integer values, the segment description
            with the corresponding segment number is used to describe each
            segment.
        plane_positions: Sequence[highdicom.content.PlanePositionSequence], optional
            Position of each plane in `pixel_array` relative to the
            three-dimensional patient or slide coordinate system.

        Note
        ----
        Items of `segment_descriptions` must be sorted by segment number in
        ascending order.
        In case `segmentation_type` is ``"BINARY"``, the number of items per
        sequence must match the number of unique positive pixel values in
        `pixel_array`. In case `segmentation_type` is ``"FRACTIONAL"``, only
        one segment can be encoded by `pixel_array` and hence only one item is
        permitted per sequence.

        """  # noqa
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        if pixel_array.ndim != 3:
            raise ValueError('Pixel array must be a 2D or 3D array.')

        if pixel_array.shape[1:3] != (self.Rows, self.Columns):
            raise ValueError(
                'Pixel array representing segments has the wrong number of '
                'rows and columns.'
            )

        described_segment_numbers = np.array([
            int(item.SegmentNumber)
            for item in segment_descriptions
        ])
        # Check that there are no duplicated segment numbers in the segment
        # descriptions
        if not (np.diff(described_segment_numbers) > 0).all():
            raise ValueError(
                'Segment descriptions must be sorted by segment number.'
            )

        if pixel_array.dtype in (np.bool, np.uint8, np.uint16):
            segments_present = np.unique(
                pixel_array[pixel_array > 0].astype(np.uint16)
            )

            # Special case where the mask is binary and there is a single
            # segment description. Allow the mark the positive segment with
            # the correct segment number
            if (np.array_equal(segments_present, np.array([1])) and
                    len(segment_descriptions) == 1):
                pixel_array = pixel_array.astype(np.uint8)
                pixel_array *= described_segment_numbers.item()

            # Otherwise, the pixel values in the pixel array must all belong to
            # a described segment
            else:
                if not np.all(
                        np.in1d(segments_present, described_segment_numbers)
                    ):
                    raise ValueError(
                        'Pixel array contains segments that lack descriptions.'
                    )

        elif (pixel_array.dtype == np.float):
            unique_values = np.unique(pixel_array)
            if np.min(unique_values) < 0.0 or np.max(unique_values) > 1.0:
                raise ValueError(
                    'Floating point pixel array values must be in the '
                    'range [0, 1].'
                )
            if len(segment_descriptions) != 1:
                raise ValueError(
                    'When providing a float-valued pixel array, provide only '
                    'a single segment description'
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
                pixel_array = pixel_array.astype(np.bool)
        else:
            raise TypeError('Pixel array has an invalid data type.')

        # Check that the new segments do not already exist
        if len(set(described_segment_numbers) & self._segment_inventory) > 0:
            raise ValueError('Segment with given segment number already exists')

        # Set the optional tag value SegmentsOverlapValues to NO to indicate
        # that the segments do not overlap. We can know this for sure if it's
        # the first segment (or set of segments) to be added because they are
        # contained within a single pixel array.
        if len(self._segment_inventory) == 0:
            self.SegmentsOverlap = SegmentsOverlapValues.NO.value
        else:
            # If this is not the first set of segments to be added, we cannot
            # be sure whether there is overlap with the existing segments
            self.SegmentsOverlap = SegmentsOverlapValues.UNDEFINED.value

        src_img = self._source_images[0]
        is_multiframe = hasattr(src_img, 'NumberOfFrames')
        if self._coordinate_system == CoordinateSystemNames.SLIDE:
            if hasattr(src_img, 'PerFrameFunctionalGroupsSequence'):
                source_plane_positions = [
                    item.PlanePositionSlideSequence
                    for item in src_img.PerFrameFunctionalGroupsSequence
                ]
            else:
                # If Dimension Organization Type is TILED_FULL, plane
                # positions are implicit and need to be computed.
                image_origin = src_img.TotalPixelMatrixOriginSequence[0]
                orientation = (
                    float(src_img.ImageOrientationSlide[0]),
                    float(src_img.ImageOrientationSlide[1]),
                    float(src_img.ImageOrientationSlide[2]),
                    float(src_img.ImageOrientationSlide[3]),
                    float(src_img.ImageOrientationSlide[4]),
                    float(src_img.ImageOrientationSlide[5]),
                )
                tiles_per_column = int(
                    np.ceil(
                        src_img.TotalPixelMatrixRows /
                        src_img.Rows
                    )
                )
                tiles_per_row = int(
                    np.ceil(
                        src_img.TotalPixelMatrixColumns /
                        src_img.Columns
                    )
                )
                num_focal_planes = getattr(
                    src_img,
                    'NumberOfFocalPlanes',
                    1
                )
                row_range = range(1, tiles_per_column + 1)
                column_range = range(1, tiles_per_row + 1)
                depth_range = range(1, num_focal_planes + 1)

                shared_fg = self.SharedFunctionalGroupsSequence[0]
                pixel_measures = shared_fg.PixelMeasuresSequence[0]
                pixel_spacing = (
                    float(pixel_measures.PixelSpacing[0]),
                    float(pixel_measures.PixelSpacing[1]),
                )
                slice_thickness = getattr(
                    pixel_measures,
                    'SliceThickness',
                    1.0
                )
                spacing_between_slices = getattr(
                    pixel_measures,
                    'SpacingBetweenSlices',
                    1.0
                )
                source_plane_positions = [
                    compute_plane_positions_tiled_full(
                        row_index=r,
                        column_index=c,
                        depth_index=d,
                        x_offset=image_origin.XOffsetInSlideCoordinateSystem,
                        y_offset=image_origin.YOffsetInSlideCoordinateSystem,
                        z_offset=1.0,  # TODO
                        rows=self.Rows,
                        columns=self.Columns,
                        image_orientation=orientation,
                        pixel_spacing=pixel_spacing,
                        slice_thickness=slice_thickness,
                        spacing_between_slices=spacing_between_slices
                    )
                    for r, c, d in itertools.product(
                        row_range,
                        column_range,
                        depth_range
                    )
                ]
        else:
            if is_multiframe:
                source_plane_positions = [
                    item.PlanePositionSequence
                    for item in src_img.PerFrameFunctionalGroupsSequence
                ]
            else:
                source_plane_positions = [
                    PlanePositionSequence(
                        coordinate_system=CoordinateSystemNames.PATIENT,
                        image_position=img.ImagePositionPatient
                    )
                    for img in self._source_images
                ]

        if plane_positions is None:
            if pixel_array.shape[0] != len(source_plane_positions):
                if is_multiframe:
                    raise ValueError(
                        'Number of frames in pixel array does not match number '
                        ' of frames in source image.'
                    )
                else:
                    raise ValueError(
                        'Number of frames in pixel array does not match number '
                        'of source images.'
                    )
            plane_positions = source_plane_positions
        else:
            if pixel_array.shape[0] != len(plane_positions):
                raise ValueError(
                    'Number of pixel array planes does not match number of '
                    'provided plane positions.'
                )

        are_spatial_locations_preserved = (
            all(
                plane_positions[i] == source_plane_positions[i]
                for i in range(len(plane_positions))
            ) and
            self._plane_orientation == self._source_plane_orientation
        )

        # For each dimension other than the Referenced Segment Number,
        # obtain the value of the attribute that the Dimension Index Pointer
        # points to in the element of the Plane Position Sequence or
        # Plane Position Slide Sequence.
        # Per definition, this is the Image Position Patient attribute
        # in case of the patient coordinate system, or the
        # X/Y/Z Offset In Slide Coordinate System and the Column/Row
        # Position in Total Image Pixel Matrix attributes in case of the
        # the slide coordinate system.
        plane_position_values = np.array([
            [
                np.array(p[0][indexer.DimensionIndexPointer].value)
                for indexer in self.DimensionIndexSequence[1:]
            ]
            for p in plane_positions
        ])

        # Planes need to be sorted according to the Dimension Index Value
        # based on the order of the items in the Dimension Index Sequence.
        # Here we construct an index vector that we can subsequently use to
        # sort planes before adding them to the Pixel Data element.
        _, plane_sort_index = np.unique(
            plane_position_values,
            axis=0,
            return_index=True
        )

        # Get unique values of attributes in the Plane Position Sequence or
        # Plane Position Slide Sequence, which define the position of the plane
        # with respect to the three dimensional patient or slide coordinate
        # system, respectively. These can subsequently be used to look up the
        # relative position of a plane relative to the indexed dimension.
        dimension_position_values = [
            np.unique(plane_position_values[:, index], axis=0)
            for index in range(plane_position_values.shape[1])
        ]

        # When using binary segmentation type, the previous frames may have been
        # padded to be a multiple of 8. In this case, we need to decode the
        # pixel data, add the new pixels and then re-encode. This process
        # should be avoided if it is not necessary in order to improve
        # efficiency.
        if (self.SegmentationType == SegmentationTypeValues.BINARY.value and
                ((self.Rows * self.Columns * self.SamplesPerPixel) % 8) > 0):
            re_encode_pixel_data = True
            logger.warning(
                'pixel data needs to be re-encoded for binary bitpacking - '
                'consider using FRACTIONAL instead of BINARY segmentation type'
            )
            # If this is the first segment added, the pixel array is empty
            if hasattr(self, 'PixelData') and len(self.PixelData) > 0:
                full_pixel_array = self.pixel_array.flatten()
            else:
                full_pixel_array = np.array([], np.bool)
        else:
            re_encode_pixel_data = False

            # Before adding new pixel data, remove trailing null padding byte
            if len(self.PixelData) == get_expected_length(self) + 1:
                self.PixelData = self.PixelData[:-1]

        for i, segment_number in enumerate(described_segment_numbers):
            if pixel_array.dtype == np.float:
                # Floating-point numbers must be mapped to 8-bit integers in
                # the range [0, max_fractional_value].
                planes = np.around(
                    pixel_array * float(self.MaximumFractionalValue)
                )
                planes = planes.astype(np.uint8)
            elif pixel_array.dtype in (np.uint8, np.uint16):
                # Labeled masks must be converted to binary masks.
                planes = np.zeros(pixel_array.shape, dtype=np.bool)
                planes[pixel_array == segment_number] = True
            elif pixel_array.dtype == np.bool:
                planes = pixel_array

            contained_plane_index = []
            for j in plane_sort_index:
                if np.sum(planes[j]) == 0:
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
                    if len(plane_sort_index) > len(self._source_images):
                        # A single multi-frame source image
                        src_img_item = self.SourceImageSequence[0]
                        # Frame numbers are one-based
                        derivation_src_img_item.ReferencedFrameNumber = j + 1
                    else:
                        # Multiple single-frame source images
                        src_img_item = self.SourceImageSequence[j]
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

            if re_encode_pixel_data:
                full_pixel_array = np.concatenate([
                    full_pixel_array,
                    planes[contained_plane_index].flatten()
                ])
            else:
                self.PixelData += self._encode_pixels(
                    planes[contained_plane_index]
                )

            # In case of a tiled Total Pixel Matrix pixel data for the same
            # segment may be added.
            if segment_number not in self._segment_inventory:
                self.SegmentSequence.append(segment_descriptions[i])
            self._segment_inventory.add(segment_number)

        # Re-encode the whole pixel array at once if necessary
        if re_encode_pixel_data:
            self.PixelData = self._encode_pixels(full_pixel_array)

        # Add back the null trailing byte if required
        if len(self.PixelData) % 2 == 1:
            self.PixelData += b'0'

    def _encode_pixels(self, planes: np.ndarray) -> bytes:
        """Encodes pixel planes.

        Parameters
        ----------
        planes: numpy.ndarray
            Array representing one or more segmentation image planes

        Returns
        -------
        bytes
            Encoded pixels

        """
        # TODO: compress depending on transfer syntax UID
        if self.SegmentationType == SegmentationTypeValues.BINARY.value:
            return pack_bits(planes.flatten())
        else:
            return planes.flatten().tobytes()
