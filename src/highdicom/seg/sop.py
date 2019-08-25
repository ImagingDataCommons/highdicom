"""Module for the SOP class of the Segmentation IOD."""
import datetime
import itertools
import logging
import numpy as np
import pydicom
from collections import namedtuple, defaultdict
from typing import NamedTuple, Optional, Sequence, Union, Tuple

from pydicom.dataset import Dataset
from pydicom.coding import Code
from pydicom.uid import UID
from pydicom._storage_sopclass_uids import (
    SegmentationStorage,
    VLMicroscopicImageStorage,
    VLSlideCoordinatesMicroscopicImageStorage,
    VLWholeSlideMicroscopyImageStorage,
)

from highdicom.base import SOPClass
from highdicom.enum import CoordinateSystemNames
from highdicom.seg.content import (
    DimensionIndexSequence,
    DerivationImageSequence,
    SegmentDescription,
    PlaneOrientationSequence,
    PlanePositionSequence,
    PlanePositionSlideSequence,
    PixelMeasuresSequence,
)
from highdicom.seg.enum import (
    SegmentationFractionalTypes,
    SegmentationTypes,
)
from highdicom.sr.coding import CodedConcept


logger = logging.getLogger(__name__)


class Segmentation(SOPClass):

    """SOP class for a Segmentation, which respresents one or more
    regions of interst (ROIs) as mask images (raster graphics).
    """

    def __init__(
            self,
            source_images: Sequence[Dataset],
            pixel_array: np.ndarray,
            segmentation_type: Union[str, SegmentationTypes],
            segment_descriptions: Sequence[SegmentDescription],
            segment_derivations: Sequence[DerivationImageSequence],
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            manufacturer: str,
            manufacturer_model_name: str,
            software_versions: Union[str, Tuple[str]],
            device_serial_number: str,
            content_description: str = '',
            content_creator_name: Optional[str] = None,
            segmentation_fractional_type: Union[str, SegmentationFractionalTypes] = SegmentationFractionalTypes.PROBABILITY,
            max_fractional_value: Optional[int] = 255,
            transfer_syntax_uid: Optional[Union[str, UID]] = None,
            frame_of_reference_uid: Optional[Union[str, UID]] = None,
            position_reference_indicator: Optional[str] = None,
            pixel_measures: Optional[PixelMeasuresSequence] = None,
            plane_orientation: Optional[PlaneOrientationSequence] = None,
            plane_positions: Optional[Union[Sequence[PlanePositionSequence], Sequence[PlanePositionSlideSequence]]] = None,
        ) -> None:
        """
        Parameters
        ----------
        source_images: Sequence[pydicom.dataset.Dataset]
            One or more single- or multi-frame images (or metadata of images)
            from which the segmentation was derived
        pixel_array: numpy.ndarray
            Array of segmentation pixel data.
            If `segmentation_type` is ``"BINARY"``, a boolean or unsigned 8-bit
            or 16-bit integer array representing a labeled mask image, where
            positive pixel values encode segment numbers.
            If `segmentation_type` is ``"FRACTIONAL"``, a floating-point pixel
            array representing a probabilistic mask image, where pixel values
            either encode the probability of a given pixel belonging to a
            segment (`segmentation_fractional_type` is ``"PROBABILITY"``)
            or the extent to which a segment occupies the pixel
            (`segmentation_fractional_type` is ``"OCCUPANCY"``).
            In the latter case, only one segment can be encoded by
            `segment_pixels_array`. Additional segments can be subsequently
            added to a Segmentation instance using the ``add_segments()``
            method.
            If `segment_pixels_array` represents a 3D image the first dimension
            represents individual 2D planes and these planes must be ordered
            based on their position in the three-dimensional coordinate system
            identified by `frame_of_reference_uid` (first along the X axis,
            second along the Y, and third along the Z axes).
        segmentation_type: Union[str, SegmentationTypes]
            Type of segmentation
        segment_descriptions: Sequence[highdicom.seg.content.SegmentDescription]
            Description of each segment encoded in `pixel_array`
        segment_derivations: Sequence[highdicom.seg.content.DerivationImageSequence]
            References to the source images (and frames within the source
            images) for each segment encoded in `pixel_array`.
            Sequence may be empty if it is not possible to reference segments
            relative to source images, because spatial locations were not
            preserved upon processing of source images, for example due to
            resampling
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
        content_description: str, optional
            Description of the segmentation
        content_creator_name: str, optional
            Name of the creator of the segmentation
        segmentation_fractional_type: Union[str, highdicom.seg.content.SegmentationFractionalTypes], optional
            Type of fractional segmentation that indicates how pixel data
            should be interpreted (required if `segmentation_type` is
            ``SegmentationTypes.FRACTIONAL``)
        max_fractional_value: int, optional
            Maximum value that indicates probability or occupancy of 1 that
            a pixel represents a given segment
            (required if `segmentation_type` is ``SegmentationTypes.FRACTIONAL``)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported: JPEG2000 (UID: ``"1.2.840.10008.1.2.4.90"``) and
            JPEG-LS (UID: ``"1.2.840.10008.1.2.4.80"``).
            Defaults to Implicit VR Little Endian (UID ``"1.2.840.10008.1.2"``).
        frame_of_reference_uid: str, optional
            UID of the frame of reference for determining the absolute position
            of an image within the coordinate system.
            If ``None``, it will be assumed that the segmentation image has the
            same frame of reference as the source image(s).
        positiion_reference_indicator: str, optional
            Part of the imaging target that is used a reference for
            determining the position of an image,
            e.g., ``"SLIDE_CORNER"``
            If ``None``, it will be assumed that the segmentation image has the
            same position reference indicator as the source image(s).
        pixel_measures: PixelMeasures, optional
            Physical spacing of image pixels in `pixel_array`.
            If ``None``, it will be assumed that the segmentation image has the
            same pixel measures as the source image(s).
        plane_orientation: PlaneOrientationSequence, optional
            Orientation of planes in `pixel_array` relative to axes of
            three-dimensional patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image as the
            same plane orientation as the source image(s).
        plane_positions: Union[Sequence[PlanePositionSequence], Sequence[PlanePositionSlideSequence]], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image has the
            same plane position as the source image(s). However, this will only
            work when the first dimension of `pixel_array` matches the number
            of frames in `source_images` (in case of multi-frame source images)
            or the number of `source_images` (in case of single-frame source
            images).

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
                * Length of `segment_descriptions` or `segment_derivations`
                  does not match number of segments encoded in `pixel_array`
                  (number of unique positive values in `pixel_array`).

        """  # noqa
        if len(source_images) == 0:
            raise ValueError('At least one source image is required.')

        uniqueness_criteria = set(
            (
                image.StudyInstanceUID,
                image.SeriesInstanceUID,
                image.Rows,
                image.Columns,
            )
            for image in source_images
        )
        if len(uniqueness_criteria) > 1:
            raise ValueError(
                'Source images must all be part of the same series and must '
                'have the same frame of reference as well as the same '
                'image dimensions (number of rows/columns).'
            )

        src_image = source_images[0]
        is_multiframe = hasattr(src_image, 'SharedFunctionalGroupsSequence')
        if is_multiframe and len(source_images) > 1:
            raise ValueError(
                'Only one source image should be provided in case images '
                'are multi-frame images.'
            )

        super().__init__(
            study_instance_uid=src_image.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid=SegmentationStorage,
            manufacturer=manufacturer,
            modality='SEG',
            transfer_syntax_uid=transfer_syntax_uid,
            patient_id=src_image.PatientID,
            patient_name=src_image.PatientName,
            patient_birth_date=src_image.PatientBirthDate,
            patient_sex=src_image.PatientSex,
            accession_number=src_image.AccessionNumber,
            study_id=src_image.StudyID,
            study_date=src_image.StudyDate,
            study_time=src_image.StudyTime,
            referring_physician_name=src_image.ReferringPhysicianName
        )

        coordinate_system = CoordinateSystemNames.PATIENT
        # Using Container Type Code Sequence attribute would be more elegant,
        # but unfortunately it is a type 2 attribute.
        if src_image.SOPClassUID in (
                VLMicroscopicImageStorage,
                VLSlideCoordinatesMicroscopicImageStorage,
                VLWholeSlideMicroscopyImageStorage,
            ):
            coordinate_system = CoordinateSystemNames.SLIDE

        # Frame of Reference
        if frame_of_reference_uid is None:
            logger.info(
                'frame of reference UID has not been provided - '
                'assuming segmentation series has same frame of reference '
                'as series of source images'
            )
            self.FrameOfReferenceUID = src_image.FrameOfReferenceUID
        else:
            self.FrameOfReferenceUID = frame_of_reference_uid
        if coordinate_system == CoordinateSystemNames.SLIDE:
            self.PositionReferenceIndicator = 'SLIDE_CORNER'
        else:
            self.PositionReferenceIndicator = positiion_reference_indicator

        # (Enhanced) General Equipment
        self.DeviceSerialNumber = device_serial_number
        self.ManufacturerModelName = manufacturer_model_name
        self.SoftwareVersions = software_versions

        # General Reference
        self.SourceImageSequence = []
        referenced_series = defaultdict(list)
        for src_image in source_images:
            ref = Dataset()
            ref.ReferencedSOPClassUID = src_image.SOPClassUID
            ref.ReferencedSOPInstanceUID = src_image.SOPInstanceUID
            self.SourceImageSequence.append(ref)
            referenced_series[src_image.SeriesInstanceUID].append(ref)

        # Common Instance Reference
        self.ReferencedSeriesSequence = []
        for series_instance_uid, referenced_images in referenced_series.items():
            ref = Dataset()
            ref.SeriesInstanceUID = series_instance_uid
            ref.ReferencedInstanceSequence = referenced_images
            self.ReferencedSeriesSequence.append(ref)


        # Image Pixel
        self.Rows = pixel_array.shape[-2]
        self.Columns = pixel_array.shape[-1]

        # Segmentation Image
        self.ImageType = ['DERIVED', 'PRIMARY']
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = 'MONOCHROME2'
        self.PixelRepresentation = 0
        self.ContentLabel = 'ISO_IR 192'  # UTF-8
        self.ContentDescription = content_description
        self.ContentCreatorName = content_creator_name
        segmentation_type = SegmentationTypes(segmentation_type)
        self.SegmentationType = segmentation_type.value
        if self.SegmentationType == SegmentationTypes.FRACTIONAL.value:
            self.BitsAllocated = 8
            self.HighBit = 7
            segmentation_fractional_type = SegmentationFractionalTypes(
                segmentation_fractional_type
            )
            self.SegmentationFractionalType = segmentation_fractional_type.value
            self.MaximumFractionalValue = max_fractional_value
        elif self.SegmentationType == SegmentationTypes.BINARY.value:
            self.BitsAllocated = 1
            self.HighBit = 0
        else:
            raise ValueError(
                'Unknown segmentation type "{}"'.format(segmentation_type)
            )
        self.BitsStored = self.BitsAllocated
        self.LossyImageCompression = src_image.LossyImageCompression
        # TODO: lossy

        # NOTE: Sequence will be updated by the "add_segments()" method.
        self.SegmentSequence = []

        # Multi-Frame Functional Groups and Multi-Frame Dimensions
        shared_func_groups = Dataset()
        if pixel_measures is None:
            if is_multiframe:
                src_shared_fg = src_image.SharedFunctionalGroupsSequence[0]
                pixel_measures = src_shared_fg.PixelMeasuresSequence
            else:
                pixel_measures = PixelMeasuresSequence(
                    pixel_spacing=src_image.PixelSpacing,
                    slice_thickness=src_image.SliceThickness,
                    spacing_between_slices=src_image.SpacingBetweenSlices
                )
        if plane_orientation is None:
            if is_multiframe:
                if coordinate_system == CoordinateSystemNames.SLIDE:
                    plane_orientation = PlaneOrientationSequence(
                        coordinate_system=coordinate_system,
                        image_orientation=src_image.ImageOrientationSlide
                    )
                else:
                    src_shared_fg = src_image.SharedFunctionalGroupsSequence[0]
                    plane_orientation = src_shared_fg.PlaneOrientationSequence
            else:
                plane_orientation = PlaneOrientationSequence(
                    coordinate_system=coordinate_system,
                    image_orientation=src_image.ImageOrientationPatient
                )
        self.DimensionIndexSequence = DimensionIndexSequence(coordinate_system)
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
        self.PerFrameFunctionalGroupsSequence = []

        if plane_positions is None:
            if coordinate_system == CoordinateSystemNames.SLIDE:
                if hasattr(src_image, 'PerFrameFunctionalGroupsSequence'):
                    plane_positions = [
                        item.PlanePositionSlideSequence
                        for item in src_image.PerFrameFunctionalGroupsSequence
                    ]
                else:
                    # If Dimension Organization Type is TILED_FULL, plane
                    # positions are implicit and need to be computed.
                    image_origin = src_image.TotalPixelMatrixOriginSequence[0]
                    orientation = tuple(
                        float(v) for v in src_image.ImageOrientationSlide
                    )
                    tiles_per_column = int(
                        np.ceil(
                            src_image.TotalPixelMatrixRows /
                            src_image.Rows
                        )
                    )
                    tiles_per_row = int(
                        np.ceil(
                            src_image.TotalPixelMatrixColumns /
                            src_image.Columns
                        )
                    )
                    num_focal_planes = getattr(
                        src_image,
                        'NumberOfFocalPlanes',
                        1
                    )
                    row_range = range(1, tiles_per_column + 1)
                    column_range = range(1, tiles_per_row + 1)
                    depth_range = range(1, num_focal_planes + 1)

                    shared_fg = self.SharedFunctionalGroupsSequence[0]
                    pixel_measures = shared_fg.PixelMeasuresSequence[0]
                    pixel_spacing = tuple(
                        float(v) for v in pixel_measures.PixelSpacing
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
                    plane_positions = [
                        PlanePositionSlideSequence.compute_for_tiled_full(
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
                    plane_positions = [
                        item.PlanePositionSequence
                        for item in src_image.PerFrameFunctionalGroupsSequence
                    ]
                else:
                    plane_positions = [
                        PlanePositionSequence(
                            image_position=src_image.ImagePositionPatient
                        )
                        for src_image in source_images
                    ]

        if pixel_array.shape[0] != len(plane_positions):
            raise ValueError(
                'Number of pixel array planes does not match number of '
                'provided image positions.'
            )

        self._segment_inventory = set()
        self.PixelData = b''
        self.add_segments(
            pixel_array=pixel_array,
            segment_descriptions=segment_descriptions,
            segment_derivations=segment_derivations,
            plane_positions=plane_positions
        )

        self.copy_specimen_information(src_image)
        self.copy_patient_and_study_information(src_image)

    def add_segments(
            self,
            pixel_array: np.ndarray,
            segment_descriptions: Sequence[SegmentDescription],
            segment_derivations: Sequence[DerivationImageSequence],
            plane_positions: Union[Sequence[PlanePositionSequence], Sequence[PlanePositionSlideSequence]]
        ) -> Dataset:
        """Adds one or more segments to the segmentation image.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Array of segmentation pixel data.
            If `segmentation_type` is ``"BINARY"``, a boolean or unsigned 8-bit
            or 16-bit integer array representing a labeled mask image, where
            positive pixel values encode segment numbers.
            If `segmentation_type` is ``"FRACTIONAL"``, a floating-point pixel
            array representing a probabilistic mask image, where pixel values
            either encode the probability of a given pixel belonging to a
            segment (`segmentation_fractional_type` is ``"PROBABILITY"``)
            or the extent to which a segment occupies the pixel
            (`segmentation_fractional_type` is ``"OCCUPANCY"``).
            In the latter case, only one segment can be encoded by
            `pixels_array`.
            If `pixels_array` represents a 3D image the first dimension
            represents individual 2D planes and these planes must be ordered
            based on their position in the three-dimensional coordinate system
            identified by `frame_of_reference_uid` (first along the X axis,
            second along the Y, and third along the Z axes).
        segment_descriptions: Sequence[highdicom.seg.content.SegmentDescription]
            Description of each segment encoded in `pixel_array`
        segment_derivations: Sequence[highdicom.seg.content.DerivationImageSequence]
            References for each segment encoded in `pixel_array`.
            Sequence may be empty if it is not possible to reference segments
            relative to source images, because spatial locations were not
            preserved upon processing of source images, for example due to
            resampling
        plane_positions: Union[Sequence[highdicom.seg.content.PlanePositionSequence], Sequence[highdicom.seg.content.PlanePositionSlideSequence]]
            Position of each plane in `pixel_array` relative to the
            three-dimensional patient or slide coordinate system.

        Note
        ----
        Items of `segment_descriptions` and `segment_derivations` must be sorted
        by segment number in ascending order.
        In case `segmentation_type` is ``"BINARY"``, the number of items per
        sequence must match the number of unique positive pixel values in
        `pixel_array`. In case `segmentation_type` is ``"FRACTIONAL"``, only
        one segment can be encoded by `pixel_array` and hence only one item is
        permitted per sequence.

        """
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        if pixel_array.shape[-2:] != (self.Rows, self.Columns):
            raise ValueError(
                'Pixel array representing segments has the wrong number of '
                'rows and columns.'
            )

        if self.SegmentationType == SegmentationTypes.BINARY.value:
            if pixel_array.dtype not in (np.bool, np.uint8, np.uint16):
                raise TypeError(
                    'Pixel array values must be either boolean, unsigned '
                    '8-bit integers, or unsigned 16-bit integers for '
                    '"BINARY" segmentation type.'
                )
            encoded_segment_numbers = np.unique(
                pixel_array[pixel_array > 0].astype(np.uint16)
            )
            if len(encoded_segment_numbers) != len(segment_descriptions):
                raise ValueError(
                    'Number of encoded segments does not match number of '
                    'provided segment descriptions.'
                )
            if len(encoded_segment_numbers) != len(segment_derivations):
                raise ValueError(
                    'Number of encoded segments does not match number of '
                    'provided segment derivations.'
                )
            described_segment_numbers = np.array([
                int(item.SegmentNumber)
                for item in segment_descriptions
            ])
            are_all_segments_described = np.array_equal(
                encoded_segment_numbers,
                described_segment_numbers
            )
            if not are_all_segments_described:
                raise ValueError(
                    'Described and encoded segment numbers must match.'
                )

        elif self.SegmentationType == SegmentationTypes.FRACTIONAL.value:
            if pixel_array.dtype != np.float:
                raise TypeError(
                    'Array values must be floating points for '
                    '"FRACTIONAL" segmentation type.'
                )
            if np.min(pixel_array) < 0.0 or np.max(pixel_array) > 1.0:
                raise ValueError(
                    'Array values must be in the range [0, 1] for '
                    '"FRACTIONAL" segmentation type.'
                )
            if max_fractional_value > 2**8:
                raise ValueError(
                    'Maximum fractional value must not exceed image bit depth.'
                )
            encoded_segment_numbers = 1
            if len(encoded_segment_numbers) != len(segment_descriptions):
                raise ValueError(
                    'One segment description required for "FRACTIONAL" '
                    'segmentation type.'
                )
            if len(encoded_segment_numbers) != len(segment_derivations):
                raise ValueError(
                    'One segment derivation required for "FRACTIONAL" '
                    'segmentation type.'
                )

        plane_position_values = np.array([
            [
                p[0][indexer.DimensionIndexPointer].value
                for indexer in self.DimensionIndexSequence[1:]
            ]
            for p in plane_positions
        ])
        _, plane_sort_index = np.unique(
            plane_position_values,
            axis=0,
            return_index=True
        )

        dimension_position_values = [
            np.unique(plane_position_values[:, index])
            for index in range(plane_position_values.shape[1])
        ]

        for i, segment_number in enumerate(encoded_segment_numbers):
            if self.SegmentationType == SegmentationTypes.BINARY.value:
                if pixel_array.dtype != np.bool:
                    # Labeled masks must be converted to binary masks.
                    planes = np.zeros(pixel_array.shape, dtype=np.bool)
                    planes[array == segment_number] = True
                else:
                    planes = pixel_array
            elif self.SegmentationType == SegmentationTypes.FRACTIONAL.value:
                # Floating-point numbers must be mapped to 8-bit integers in
                # the range [0, max_fractional_value].
                planes = np.around(pixel_array * float(max_fractional_value))
                planes = planes.dtype(np.uint8)

            for j in plane_sort_index:
                pffp_item = Dataset()
                frame_content_item = Dataset()
                frame_content_item.DimensionIndexValues = [segment_number]
                frame_content_item.DimensionIndexValues.extend([
                    np.where(dimension_position_values[index] == pos)[0][0] + 1
                    for index, pos in enumerate(plane_position_values[j])
                ])
                pffp_item.FrameContentSequence = [frame_content_item]
                pffp_item.PlanePositionSlideSequence = plane_positions[j]
                pffp_item.DerivationImageSequence = segment_derivations[i]
                identification = Dataset()
                identification.ReferencedSegmentNumber = segment_number
                pffp_item.SegmentIdentificationSequence = [
                    identification,
                ]
                self.PerFrameFunctionalGroupsSequence.append(pffp_item)
                self.NumberOfFrames += 1
            self.PixelData += self._encode_pixels(planes[plane_sort_index])

            # In case of a tiled Total Pixel Matrix pixel data for the same
            # segment may be added.
            if segment_number not in self._segment_inventory:
                self.SegmentSequence.append(segment_descriptions[i])
            self._segment_inventory.add(segment_number)

    def _encode_pixels(self, planes: np.ndarray) -> bytes:
        """Encodes pixel planes.

        Parameters
        ----------
        planes: numpy.ndarray
            Array representing one or more segmentation planes

        Returns
        -------
        bytes
            Encoded pixels

        """
        pixels = planes.flatten()
        if self.BitsStored == 1:
            # The number of pixels must be a multiple of 8.
            # Zero pad array if necessary.
            bit_depth = 8
            factor = bit_depth * 2
            total_size = planes.size
            remainder = total_size % factor
            pixels = np.pad(pixels, (0, factor - remainder), mode='constant')

            # Reshape array such that there is one row per byte in the output
            pixels = pixels.reshape((-1, bit_depth))

            # Scale pixel values to 8-bit and sum along the rows to pack values
            multiplier = 2**np.arange(bit_depth).astype(np.uint8)
            multiplier = multiplier[np.newaxis, ...]

            pixels = np.sum(multiplier * pixels, axis=1).astype(np.uint8)

        return pixels.tobytes()
