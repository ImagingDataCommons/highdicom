from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
from enum import Enum

import numpy as np
from pydicom.encaps import decode_data_sequence, encapsulate
from pydicom.pixel_data_handlers.util import get_expected_length
from highdicom.base import SOPClass
from highdicom.content import (
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
)
from highdicom.enum import CoordinateSystemNames
from highdicom.frame import encode_frame
from highdicom.map.content import RealWorldValueMapping
from highdicom.seg.content import DimensionIndexSequence
from highdicom.valuerep import check_person_name
from pydicom import Dataset
from pydicom.uid import (
    UID,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    RLELossless,
)


class _PixelDataType(Enum):
    """Helper enum for tracking the type of the pixel data"""

    SHORT = 1
    USHORT = 2
    SINGLE = 3
    DOUBLE = 4


class ParametricMap(SOPClass):

    """SOP class for a Parametric Map."""

    def __init__(
        self,
        source_images: Sequence[Dataset],
        pixel_array: np.ndarray,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: Union[str, Tuple[str]],
        device_serial_number: str,
        contains_recognizable_visual_features: bool,
        real_world_value_mappings: Sequence[RealWorldValueMapping],
        window_center: Union[int, float],
        window_width: Union[int, float],
        transfer_syntax_uid: Union[str, UID] = ImplicitVRLittleEndian,
        content_description: Optional[str] = None,
        content_creator_name: Optional[str] = None,
        pixel_measures: Optional[PixelMeasuresSequence] = None,
        plane_orientation: Optional[PlaneOrientationSequence] = None,
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        source_images: Sequence[pydicom.Dataset]
            Sequence of one or more `pydicom.Dataset`s that this parametric map
            is derived from
        pixel_array: numpy.ndarray
            Array of parametric map pixel data of unsigned integer or
            floating-point data type representing one or more frames of the
            parametric map pixel data. The values are supposed to represent a
            single "feature", i.e., be the result of one set of image
            transformations such that the same `real_world_value_mappings`
            apply.
        series_instance_uid: str
            UID of the series
        series_number: Union[int, None]
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        manufacturer: str
            Name of the manufacturer (developer) of the device (software)
            that creates the instance
        manufacturer_model_name: str,
            Name of the model of the device (software)
            that creates the instance
        software_versions: Union[str, Tuple[str]]
            Versions of relevant software used to create the data
        device_serial_number: str
            Serial number (or other identifier) of the device (software)
            that creates the instance
        contains_recognizable_visual_features: bool
            Whether the image contains recognizable visible features of the
            patient
        real_world_value_mappings: Sequence[highdicom.map.RealWorldValueMapping]
            Descriptions of how stored values map to real-world values. The
            concept of real-world values is a bit fuzzy and the mapping may be
            difficult to describe (e.g., in case of the feature maps of a deep
            convolutional neural network model).
        window_center: Union[int, float, None], optional
            Window center for rescaling stored values for display purposes by
            applying a linear transformation function. For example, in case of
            floating-point values in the range ``[0.0, 1.0]``, the window
            center would be ``0.5``, in case of floating-point values in the
            range ``[-1.0, 1.0]`` the window center would be ``0.0``, in case
            of unsigned integer values in the range ``[0, 255]`` the window
            center would be ``128``.
        window_width: Union[int, float, None], optional
            Window width for rescaling stored values for display purposes by
            applying a linear transformation function. For example, in case of
            floating-point values in the range ``[0.0, 1.0]``, the window
            width would be ``1.0``, in case of floating-point values in the
            range ``[-1.0, 1.0]`` the window width would be ``2.0``, and in
            case of unsigned integer values in the range ``[0, 255]`` the
            window width would be ``256``. In case of unbounded floating-point
            values, a sensible window width should be chosen to allow for
            stored values to be displayed on 8-bit monitors.
        transfer_syntax_uid: Union[str, None], optional
            UID of transfer syntax that should be used for encoding of
            data elements. Defaults to Implicit VR Little Endian
            (UID ``"1.2.840.10008.1.2"``)
        content_description: Union[str, None], optional
            Brief description of the parametric map image
        content_creator_name: Union[str, None], optional
            Name of the person that created the parametric map image
        pixel_measures: Union[highdicom.PixelMeasuresSequence, None], optional
            Physical spacing of image pixels in `pixel_array`.
            If ``None``, it will be assumed that the parametric map image has
            the same pixel measures as the source image(s).
        plane_orientation: Union[highdicom.PlaneOrientationSequence, None], optional
            Orientation of planes in `pixel_array` relative to axes of
            three-dimensional patient or slide coordinate space.
            If ``None``, it will be assumed that the parametric map image as
            the same plane orientation as the source image(s).
        plane_positions: Union[Sequence[PlanePositionSequence], None], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space.
            If ``None``, it will be assumed that the parametric map image has
            the same plane position as the source image(s). However, this will
            only work when the first dimension of `pixel_array` matches the
            number of frames in `source_images` (in case of multi-frame source
            images) or the number of `source_images` (in case of single-frame
            source images).
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
                * Length of `plane_positions` does not match number of 2D planes
                  in `pixel_array` (size of first array dimension).
                * Transfer Syntax specified by `transfer_syntax_uid` is not
                  supported for data type of `pixel_array`.

        Note
        ----
        The assumption is made that planes in `pixel_array` are defined in
        the same frame of reference as `source_images`.

        """  # noqa
        if len(source_images) == 0:
            raise ValueError('At least one source image is required')
        self._source_images = source_images

        uniqueness_criteria = set(
            (
                image.StudyInstanceUID,
                image.SeriesInstanceUID,  # TODO: Might be overly restrictive
                image.Rows,
                image.Columns,
                image.FrameOfReferenceUID,
            )
            for image in self._source_images
        )
        if len(uniqueness_criteria) > 1:
            raise ValueError(
                'Source images must all be part of the same series and must'
                'have the same image dimensions (number of rows/columns).'
            )

        src_img = self._source_images[0]
        is_multiframe = hasattr(src_img, 'NumberOfFrames')
        # TODO: Revisit, may be overly restrictive
        # Check Source Image Sequence attribute in General Reference module
        if is_multiframe:
            if len(self._source_images) > 1:
                raise ValueError(
                    'Only one source image should be provided in case images '
                    'are multi-frame images.'
                )
            self._src_num_frames = src_img.NumberOfFrames

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
        }
        if pixel_array.dtype.kind in ('u', 'i'):
            # If pixel data has unsigned or signed integer data type, then it
            # can be lossless compressed. The standard does not specify any
            # compression codecs for floating-point data types.
            # In case of signed integer data type, values will be rescaled to
            # a signed integer range prior to compression.
            supported_transfer_syntaxes.update(
                {
                    JPEG2000Lossless,
                    RLELossless,
                }
            )
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported.'
            )

        if window_width <= 0:
            raise ValueError('Window width must be greater than zero.')

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        # There are different DICOM Attributes in the SOP instance depending
        # on what type of data is being saved. This lets us keep track of that
        # a bit easier
        self._pixel_data_type_map = {
            _PixelDataType.SHORT: 'PixelData',
            _PixelDataType.USHORT: 'PixelData',
            _PixelDataType.SINGLE: 'FloatPixelData',
            _PixelDataType.DOUBLE: 'DoubleFloatPixelData',
        }

        super().__init__(
            study_instance_uid=src_img.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid='1.2.840.10008.5.1.4.1.1.30',
            manufacturer=manufacturer,
            modality='OT',
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
            **kwargs,
        )

        if hasattr(src_img, 'ImageOrientationSlide') or hasattr(
            src_img, 'ImageCenterPointCoordinatesSequence'
        ):
            self._coordinate_system = CoordinateSystemNames.SLIDE
        else:
            self._coordinate_system = CoordinateSystemNames.PATIENT

        # Frame of Reference
        self.FrameOfReferenceUID = src_img.FrameOfReferenceUID
        self.PositionReferenceIndicator = getattr(
            src_img, 'PositionReferenceIndicator', None
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
        for (
            series_instance_uid,
            referenced_images,
        ) in referenced_series.items():
            ref = Dataset()
            ref.SeriesInstanceUID = series_instance_uid
            ref.ReferencedInstanceSequence = referenced_images
            self.ReferencedSeriesSequence.append(ref)

        # Image Pixel
        self.Rows = pixel_array.shape[1]
        self.Columns = pixel_array.shape[2]

        # Parametric Map Image
        self.ImageType = ['DERIVED', 'PRIMARY']

        self.LossyImageCompression = getattr(
            src_img, 'LossyImageCompression', '00'
        )
        if self.LossyImageCompression == "01":
            self.LossyImageCompressionRatio = (
                src_img.LossyImageCompressionRatio
            )
            self.LossyImageCompressionMethod = (
                src_img.LossyImageCompressionMethod
            )
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = 'MONOCHROME2'
        self.BurnedInAnnotation = 'NO'
        if contains_recognizable_visual_features:
            self.RecognizableVisualFeatures = 'YES'
        else:
            self.RecognizableVisualFeatures = 'NO'
        self.ContentLabel = 'ISO_IR 192'  # UTF-8
        self.ContentDescription = content_description
        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name
        self.PresentationLUTShape = 'IDENTITY'

        # Physical dimensions of the image should match those of the source

        # Multi-Frame Functional Groups and Multi-Frame Dimensions
        shared_fg_item = Dataset()
        self.SharedFunctionalGroupsSequence = []
        if pixel_measures is None:
            if is_multiframe:
                src_shared_fg = src_img.SharedFunctionalGroupsSequence[0]
                pixel_measures = src_shared_fg.PixelMeasuresSequence
            else:
                pixel_measures = PixelMeasuresSequence(
                    pixel_spacing=[float(v) for v in src_img.PixelSpacing],
                    slice_thickness=float(src_img.SliceThickness),
                    spacing_between_slices=src_img.get(
                        'SpacingBetweenSlices', None
                    ),
                )
        if is_multiframe:
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                source_plane_orientation = PlaneOrientationSequence(
                    coordinate_system=self._coordinate_system,
                    image_orientation=[
                        float(v) for v in src_img.ImageOrientationSlide
                    ],
                )
            else:
                src_sfg = src_img.SharedFunctionalGroupsSequence[0]
                source_plane_orientation = src_sfg.PlaneOrientationSequence
        else:
            source_plane_orientation = PlaneOrientationSequence(
                coordinate_system=self._coordinate_system,
                image_orientation=[
                    float(v) for v in src_img.ImageOrientationPatient
                ],
            )
        if plane_orientation is None:
            plane_orientation = source_plane_orientation
        self._plane_orientation = plane_orientation
        self._source_plane_orientation = source_plane_orientation

        # TODO: Double check correctness of DimensionIndexSequence, see
        # discussion on Segmentation
        # Also TODO: This includes some segmentation-specific stuff. Fix that
        self.DimensionIndexSequence = DimensionIndexSequence(
            coordinate_system=self._coordinate_system
        )
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = (
            self.DimensionIndexSequence[0].DimensionOrganizationUID
        )
        self.DimensionOrganizationSequence = [dimension_organization]

        if is_multiframe:
            self._source_plane_positions = (
                self.DimensionIndexSequence.get_plane_positions_of_image(
                    self._source_images[0]
                )
            )
        else:
            self._source_plane_positions = (
                self.DimensionIndexSequence.get_plane_positions_of_series(
                    self._source_images
                )
            )

        shared_fg_item.PixelMeasuresSequence = pixel_measures
        shared_fg_item.PlaneOrientationSequence = plane_orientation

        # Identity Pixel Value Transformation
        if pixel_array.dtype.kind == 'i':
            # In case of signed integer type we rescale values to unsigned
            # 16-bit integer range.
            transformation_item = Dataset()
            transformation_item.RescaleIntercept = 2 ** 16 / 2
            transformation_item.RescaleSlope = 1
            transformation_item.RescaleType = 'US'
        else:
            transformation_item = Dataset()
            transformation_item.RescaleIntercept = 0
            transformation_item.RescaleSlope = 1
            transformation_item.RescaleType = 'US'
        shared_fg_item.PixelValueTransformationSequence = [transformation_item]

        # Frame VOI LUT With LUT
        voi_lut_item = Dataset()
        voi_lut_item.WindowCenter = window_center
        voi_lut_item.WindowWidth = window_width
        voi_lut_item.VOILUTFunction = 'LINEAR_EXACT'
        shared_fg_item.FrameVOILUTSequence = [voi_lut_item]

        # Parametric Map Frame Type
        frame_type_item = Dataset()
        frame_type_item.FrameType = self.ImageType
        shared_fg_item.ParametricMapFrameTypeSequence = [frame_type_item]

        # Real World Value Mapping Sequence
        # If the input was a single RWVM or a sequence of size 1 then we will
        # assign it to the Shared FG Seq. Otherwise it will be per-frame and
        # will be checked later on
        if len(real_world_value_mappings) == 1:
            shared_fg_item.RealWorldValueMappingSequence = real_world_value_mappings  # noqa: E501
            # Set to None so that when it is passed to the add_values method
            # we can tell that it has already been assigned
            rwvm_seq = None
        else:
            # Otherwise just pass the sequence to add_values
            rwvm_seq = real_world_value_mappings

        self.SharedFunctionalGroupsSequence = [shared_fg_item]

        # Information about individual frames will be updated by the
        # "add_values()" method upon addition of parametric map planes.
        self.NumberOfFrames = 0
        self.PerFrameFunctionalGroupsSequence: List[Dataset] = []

        # Get the correct attribute for this Instance's pixel data
        pixel_data_type, pixel_data_attr = self._get_pixel_data_type_and_attr(
            pixel_array
        )
        # Internal value to avoid string comparisons on each map update
        # Not sure that's actually necessary or good but whatever
        self._pixel_data_type = pixel_data_type

        if (
            self._pixel_data_type == _PixelDataType.SHORT
            or self._pixel_data_type == _PixelDataType.USHORT
        ):
            self.BitsAllocated = 16
            self.BitsStored = self.BitsAllocated
            self.HighBit = self.BitsStored - 1
        elif self._pixel_data_type == _PixelDataType.SINGLE:
            self.BitsAllocated = 32
        elif self._pixel_data_type == _PixelDataType.DOUBLE:
            self.BitsAllocated = 64
        # TODO: Determine whether this attribute can be present in data sets
        # with Float Pixel Data or Double Float Pixel Data attribute.
        # The pydicom library requires its presence for decoding.
        self.PixelRepresentation = 0

        self.copy_specimen_information(src_img)
        self.copy_patient_and_study_information(src_img)
        setattr(self, pixel_data_attr, b"")

        self.add_values(
            pixel_array,
            real_world_value_mappings=rwvm_seq,
            plane_positions=plane_positions,
        )

    def add_values(
        self,
        pixel_array: np.ndarray,
        real_world_value_mappings: Optional[
            Union[Sequence[RealWorldValueMapping],
                  Sequence[Sequence[RealWorldValueMapping]]]],
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
    ) -> None:
        """Add values to the parametric map.

        Parameters
        ----------
        pixel_array: np.ndarray
            Array of parametric map pixel data of unsigned integer or
            floating-point data type representing one or more frames of the
            parametric map pixel data. The values are supposed to represent a
            single "feature", i.e., be the result of one set of image
            transformations such that the same `real_world_value_mappings`
            apply.
        real_world_value_mappings: Union[Sequence[highdicom.map.RealWorldValueMapping],
                                         Sequence[
                                             Sequence[highdicom.map.RealWorldValueMapping]]], optional
            Description of the mapping of values stored in `pixel_array` to
            real-world values. Number of items must match the number of planes
            in `pixel_array`. Multiple Real World Value Mappings can be
            assigned to a single frame/plane.
        plane_positions: Sequence[highdicom.PlanePositionSequence], optional
            Position of each plane in `pixel_array` relative to the
            patient or slide coordinate system.

        Raises
        ------
        ValueError
            When:
                * `pixel_array` is not 2D or 3D
                * `pixel_array` rows and columns do not match those of the
                    object
                * `real_world_value_mappings` is empty
                * The number of pixel array planes does not match the number of
                    planes/frames in the referenced source image
                * The number of pixel array planes does not match the number of
                    provided plane positions

        """  # noqa: E501
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        if pixel_array.ndim != 3:
            raise ValueError('Pixel array must be a 2D or 3D array.')

        if pixel_array.shape[1:3] != (self.Rows, self.Columns):
            raise ValueError(
                'Pixel array has the wrong number of rows or columns.'
            )
        if real_world_value_mappings and len(real_world_value_mappings) == 0:
            raise ValueError(
                'At least one RealWorldValueMapping item must be provided.'
            )

        if plane_positions is None:
            if pixel_array.shape[0] != len(self._source_plane_positions):
                raise ValueError(
                    'Number of pixel array planes does not match number '
                    'of planes (frames) in referenced source image.'
                )
            plane_positions = self._source_plane_positions
        else:
            if pixel_array.shape[0] != len(plane_positions):
                raise ValueError(
                    'Number of pixel array planes does not match number of '
                    'provided plane positions.'
                )

        # We can have either a list of RealWorldValueMappings which maps one
        # to each frame, or we can have a list of lists which maps multiple
        # mappings to each frame.
        if real_world_value_mappings:
            if len(real_world_value_mappings) != pixel_array.shape[0]:
                raise ValueError(
                    'Number of items in the Real World Value Mappings sequence '
                    'does not match the number of pixel array planes. '
                    f'Expected {pixel_array.shape[0]}, got '
                    f'{len(real_world_value_mappings)}'
                )

        for i in range(pixel_array.shape[0]):
            self.NumberOfFrames += 1

            # Per-frame Functional Groups
            pffg_item = Dataset()
            pffg_item.DerivationImageSequence = []

            # Plane Position (Patient/Slide)
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                pffg_item.PlanePositionSlideSequence = plane_positions[i]
            else:
                pffg_item.PlanePositionSequence = plane_positions[i]

            # Frame Content
            frame_content_item = Dataset()
            # FIXME
            frame_content_item.DimensionIndexValues = [i + 1]
            pffg_item.FrameContentSequence = [frame_content_item]

            # Real World Value Mapping
            if real_world_value_mappings:
                rwvm = real_world_value_mappings[i]
                # If the input RWVM is a list of individual mappings we need to
                # assign those to a sequence as we go. Otherwise, if it is a
                # list of sequences of RWVMs, then we can just assign them
                # outright
                if isinstance(rwvm, RealWorldValueMapping):
                    pffg_item.RealWorldValueMappingSequence = [rwvm]
                else:
                    pffg_item.RealWorldValueMappingSequence = rwvm

            self.PerFrameFunctionalGroupsSequence.append(pffg_item)

            self._append_pixel_data(pixel_array[i, ...])

    def _append_pixel_data(self, plane: np.ndarray) -> None:
        """Appends the provided array of pixels to the pixel data element.

        Depending on the data type, the pixel data may be stored in either the
        Pixel Data, Float Pixel Data, or Double Float Pixel Data element.

        Parameters
        ----------
        plane: np.ndarray
            Two dimensional array of pixels of an individual frame (plane)

        Raises
        ------
        ValueError
            When the input pixel array's dtype does not match that of the
            SOP Instance

        """
        pixel_data_type, pixel_data_attr = self._get_pixel_data_type_and_attr(
            plane
        )
        if self._pixel_data_type != pixel_data_type:
            raise ValueError(
                'Data type of input pixel array '
                'does not match that of SOP instance. '
                f'Expected "{self._pixel_data_type}", got "{pixel_data_type}".'
            )

        pixel_data_bytes = getattr(self, pixel_data_attr)

        # Before adding new pixel data, remove trailing null padding byte
        if len(pixel_data_bytes) == get_expected_length(self) + 1:
            pixel_data_bytes = pixel_data_bytes[:-1]

        # Add new pixel data
        if self.file_meta.TransferSyntaxUID.is_encapsulated:
            # To add a new frame item to the encapsulated Pixel Data element
            # we first need to unpack the existing frames, then add the pixel
            # data of the current plane, and finally encapsulate all frames.
            all_frames = decode_data_sequence(pixel_data_bytes)
            all_frames.append(self._encode_pixels(plane))
            pixel_data_bytes = encapsulate(all_frames)
        else:
            pixel_data_bytes += self._encode_pixels(plane)

        # FIXME(cg): This fails with the latest release of PyDICOM because the
        # when using Double Float Pixel Data. The bug is fixed in master but
        # has not been released yet.
        # See https://github.com/pydicom/pydicom/pull/1413 for more details
        setattr(self, pixel_data_attr, pixel_data_bytes)

    def _get_pixel_data_type_and_attr(
        self, pixel_array: np.ndarray
    ) -> Tuple[_PixelDataType, str]:
        """Data type and name of pixel data attribute.

        Parameters
        ----------
        pixel_array : np.ndarray
            The array to check

        Returns
        -------
        Tuple[highdicom.map.sop._PixelDataType, str]
            A tuple where the first element is the enum value and the second
            value is the DICOM pixel data attribute for the given datatype.
            One of (``"PixelData"``, ``"FloatPixelData"``,
            ``"DoubleFloatPixelData"``)

        Raises
        ------
        ValueError
            If values in the input array don't have a supported unsigned
            integer or floating-point type.

        """
        if pixel_array.dtype.kind == 'f':
            # Further check for float32 vs float64
            if pixel_array.dtype.name == 'float32':
                return (
                    _PixelDataType.SINGLE,
                    self._pixel_data_type_map[_PixelDataType.SINGLE],
                )
            elif pixel_array.dtype.name == 'float64':
                return (
                    _PixelDataType.DOUBLE,
                    self._pixel_data_type_map[_PixelDataType.DOUBLE],
                )
            else:
                raise ValueError(
                    'Unsupported floating-point type for pixel data: '
                    '32-bit (single-precision) or 64-bit (double-precision) '
                    'floating-point types are supported.'
                )
        elif pixel_array.dtype.kind == 'u':
            if pixel_array.dtype not in (np.uint8, np.uint16):
                raise ValueError(
                    'Unsupported unsigned integer type for pixel data: '
                    '16-bit unsigned integer types are supported.'
                )
            return (
                _PixelDataType.USHORT,
                self._pixel_data_type_map[_PixelDataType.USHORT],
            )
        elif pixel_array.dtype.kind == "i":
            if pixel_array.dtype not in (np.int8, np.int16):
                raise ValueError(
                    'Unsupported signed integer type for pixel data: '
                    '8-bit or 16-bit signed integer types are supported.'
                )
            return (
                _PixelDataType.SHORT,
                self._pixel_data_type_map[_PixelDataType.SHORT],
            )
        raise ValueError(
            'Unsupported data type for pixel data.'
            'Supported are 8-bit or 16-bit signed and unsigned integer types '
            'as well as 32-bit (single-precision) or 64-bit (double-precision) '
            'floating-point types.'
        )

    def _encode_pixels(self, plane: np.ndarray) -> bytes:
        """Encodes a given pixel array as a bytes object

        Parameters
        ----------
        plane : np.ndarray
            The numpy array to encode

        Returns
        -------
        bytes
            `plane` encoded as a `bytes` object

        Raises
        ------
        ValueError
            If the SOP instance uses an encapsulated transfer syntax and
            `plane` is not exactly 2 dimensional.
        """
        if self.file_meta.TransferSyntaxUID.is_encapsulated:
            # Check that only a single plane was passed
            if plane.ndim != 2:
                raise ValueError(
                    'Only single frame can be encoded at at time '
                    'in case of encapsulated format encoding.'
                )
            return encode_frame(
                plane.astype(np.uint16),
                transfer_syntax_uid=self.file_meta.TransferSyntaxUID,
                bits_allocated=self.BitsAllocated,
                bits_stored=self.BitsStored,
                photometric_interpretation=self.PhotometricInterpretation,
                pixel_representation=self.PixelRepresentation,
            )
        else:
            if plane.dtype == np.uint8:
                return plane.astype(np.uint16).flatten().tobytes()
            elif plane.dtype.kind == 'i':
                plane = plane.astype(np.int16) + 2 ** 16 / 2
                return plane.astype(np.uint16).flatten().tobytes()
            else:
                return plane.flatten().tobytes()
