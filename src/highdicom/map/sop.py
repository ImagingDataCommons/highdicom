from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
from enum import Enum

import numpy as np
from pydicom.pixel_data_handlers.util import get_expected_length
from highdicom.base import SOPClass
from highdicom.content import (
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
)
from highdicom.enum import (
    ContentQualificationValues,
    CoordinateSystemNames,
    RecognizableVisualFeaturesValues,
)
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
from pydicom.sr.codedict import codes


class PixelDataType(Enum):
    """Helper enum for tracking the type of the pixel data"""

    INTEGER = 1
    FLOAT = 2
    DOUBLE = 3


class ParametricMap(SOPClass):

    """SOP class for a Parametric Map"""

    def __init__(
        self,
        source_images: Sequence[Dataset],
        pixel_array: np.ndarray,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        device_serial_number: str,
        manufacturer_model_name: str,
        recognizable_visual_features: Union[
            str, RecognizableVisualFeaturesValues
        ],
        software_versions: Union[str, Tuple[str]],
        transfer_syntax_uid: Union[str, UID] = ImplicitVRLittleEndian,
        content_description: Optional[str] = None,
        content_creator_name: Optional[str] = None,
        pixel_measures: Optional[PixelMeasuresSequence] = None,
        plane_orientation: Optional[PlaneOrientationSequence] = None,
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
        **kwargs,
    ):
        if len(source_images) == 0:
            raise ValueError("At least one source image is required")
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
                "Source images must all be part of the same series and must "
                "have the same image dimensions (number of rows/columns)."
            )

        src_img = self._source_images[0]
        is_multiframe = hasattr(src_img, "NumberOfFrames")
        # TODO: Revisit, may be overly restrictive
        # Check Source Image Sequence attribute in General Reference module
        if is_multiframe:
            if len(self._source_images) > 1:
                raise ValueError(
                    "Only one source image should be provided in case images "
                    "are multi-frame images."
                )
            self._src_num_frames = src_img.NumberOfFrames

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                'Transfer syntax "{}" is not supported'.format(
                    transfer_syntax_uid
                )
            )

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        # There are different DICOM Attributes in the SOP instance depending
        # on what type of data is being saved. This lets us keep track of that
        # a bit easier
        self._pixel_data_type_map = {
            PixelDataType.INTEGER: "PixelData",
            PixelDataType.FLOAT: "FloatPixelData",
            PixelDataType.DOUBLE: "DoubleFloatPixelData",
        }
        bits_to_allocate = {
            PixelDataType.INTEGER: 16,
            PixelDataType.FLOAT: 32,
            PixelDataType.DOUBLE: 64,
        }

        super().__init__(
            study_instance_uid=src_img.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid="1.2.840.10008.5.1.4.1.1.30",
            manufacturer=manufacturer,
            modality="OT",
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
                src_img, "ReferringPhysicianName", None
            ),
            **kwargs,
        )
        if hasattr(src_img, "ImageOrientationSlide") or hasattr(
            src_img, "ImageCenterPointCoordinatesSequence"
        ):
            self._coordinate_system = CoordinateSystemNames.SLIDE
        else:
            self._coordinate_system = CoordinateSystemNames.PATIENT

        # Frame of Reference
        self.FrameOfReferenceUID = src_img.FrameOfReferenceUID
        self.PositionReferenceIndicator = getattr(
            src_img, "PositionReferenceIndicator", None
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
        self.ImageType = ["DERIVED", "PRIMARY"]

        self.LossyImageCompression = getattr(
            src_img, "LossyImageCompression", "00"
        )
        if self.LossyImageCompression == "01":
            self.LossyImageCompressionRatio = (
                src_img.LossyImageCompressionRatio
            )
            self.LossyImageCompressionMethod = (
                src_img.LossyImageCompressionMethod
            )
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = "MONOCHROME2"
        self.BurnedInAnnotation = "NO"
        recognizable_visual_features = RecognizableVisualFeaturesValues(
            recognizable_visual_features
        )
        self.RecognizableVisualFeatures = recognizable_visual_features.value
        self.ContentLabel = "ISO_IR 192"  # UTF-8
        self.ContentDescription = content_description
        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name
        self.PresentationLUTShape = "IDENTITY"

        # Physical dimensions of the image should match those of the source

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
                        "SpacingBetweenSlices", None
                    ),
                )
        if is_multiframe:
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                source_plane_orientation = PlaneOrientationSequence(
                    coordinate_system=self._coordinate_system,
                    image_orientation=src_img.ImageOrientationSlide,
                )
            else:
                src_sfg = src_img.SharedFunctionalGroupsSequence[0]
                source_plane_orientation = src_sfg.PlaneOrientationSequence
        else:
            source_plane_orientation = PlaneOrientationSequence(
                coordinate_system=self._coordinate_system,
                image_orientation=src_img.ImageOrientationPatient,
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

        shared_func_groups.PixelMeasuresSequence = pixel_measures
        shared_func_groups.PlaneOrientationSequence = plane_orientation
        # Identity Pixel Value Transformation
        pixel_value_transformation = Dataset()
        pixel_value_transformation.RescaleIntercept = 0
        pixel_value_transformation.RescaleSlope = 1
        pixel_value_transformation.RescaleType = "US"
        shared_func_groups.PixelValueTransformationSequence = [
            pixel_value_transformation
        ]
        # This maps input greyscale values to output greyscale values. Might
        # need to accept as an input or something.
        frame_voi_lut = Dataset()
        # TODO See C.11.2.1.2.1 Note 4. If I read correctly, this should
        # cause the VOI to just be an identity function.
        frame_voi_lut.WindowWidth = 2 ** 32
        frame_voi_lut.WindowCenter = 2 ** 31
        shared_func_groups.FrameVOILUTSequence = [frame_voi_lut]

        # Parametric Map Frame Type
        frame_type_item = Dataset()
        frame_type_item.FrameType = self.ImageType
        shared_func_groups.ParametricMapFrameTypeSequence = [frame_type_item]

        self.SharedFunctionalGroupsSequence = [shared_func_groups]

        # NOTE: Information about individual frames will be updated by the
        # "add_frame()" method upon addition of parametric map planes.
        self.NumberOfFrames = 0
        self.PerFrameFunctionalGroupsSequence: List[Dataset] = []

        # Get the correct attribute for this Instance's pixel data
        pixel_enum, pixel_data_attr = self._get_array_pixel_data_name(
            pixel_array
        )
        setattr(self, pixel_data_attr, b"")
        # Internal value to avoid string comparisons on each map update
        # Not sure that's actually necessary or good but whatever
        self._pixel_data_type = pixel_enum
        # Attributes based on the type of Pixel Data
        if self._pixel_data_type == PixelDataType.INTEGER:
            self.BitsStored = 16
            self.HighBit = 15
            self.PixelRepresentation = 0x1
        # TODO: Add something here for unsigned integer as well
        self.BitsAllocated = bits_to_allocate[self._pixel_data_type]

        # TODO: Take from input instead
        test = RealWorldValueMapping("test", "Test", codes.UCUM.NoUnits)

        self.add_frame(pixel_array, [test])

        self.copy_specimen_information(src_img)
        self.copy_patient_and_study_information(src_img)

    def add_frame(
        self,
        pixel_array: np.ndarray,
        real_world_value_mappings: Sequence[RealWorldValueMapping],
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
    ):
        """TODO

        This method adds a frame to the parametric map.

        Parameters
        ----------
        pixel_array : np.ndarray
            [description]
        real_world_value_mappings : Sequence[RealWorldValueMappingSequence]
            [description]
        plane_positions : Optional[Sequence[PlanePositionSequence]], optional
            [description], by default None

        Raises
        ------
        ValueError
            [description]
        ValueError
            [description]
        NotImplementedError
            [description]
        """
        # Each feature map may be represented by 1 to n frames
        # For now there is only one output frame per map
        # Input to add_map will be one feature at a time, but it could
        # be tiled so [n, rows, cols], but each of n is still only one feature
        # because you can find the same features at different regions in space

        # Proof of concept only use integer, don't use binary, only need to look
        # into fractional case

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        if pixel_array.ndim != 3:
            raise ValueError("Pixel array must be a 2D or 3D array.")

        if pixel_array.shape[1:3] != (self.Rows, self.Columns):
            raise ValueError(
                "Pixel array representing map has the wrong number of "
                "rows and columns."
            )

        # if plane_positions is None:
        #     if pixel_array.shape[0] != len(self._source_plane_positions):
        #         raise ValueError(
        #             "Number of frames in pixel array does not match number "
        #             "of source image frames."
        #         )
        #     plane_positions = self._source_plane_positions
        # else:
        #     if pixel_array.shape[0] != len(plane_positions):
        #         raise ValueError(
        #             "Number of pixel array planes does not match number of "
        #             "provided plane positions."
        #         )
        frame_number = self.NumberOfFrames + 1

        # Per-frame Functional Groups
        pffgrp_item = Dataset()
        # Type 2 attribute, i.e. allowed to be empty
        pffgrp_item.DerivationImageSequence = []

        # Frame Content
        frame_content_item = Dataset()
        frame_content_item.DimensionIndexValues = [frame_number]
        pffgrp_item.FrameContentSequence = [frame_content_item]

        pffgrp_item.RealWorldValueMappingSequence = real_world_value_mappings

        self.PerFrameFunctionalGroupsSequence.append(pffgrp_item)
        self.NumberOfFrames += 1

        self._append_pixel_data(pixel_array)

    def _append_pixel_data(self, pixel_array: np.ndarray) -> None:
        """Appends the provided pixel array to the end of the SOP Instance's
        Pixel Data, Float Pixel Data, or Double Float Pixel Data attributes
        depending on the data type. This method modifies the Instance object.

        Parameters
        ----------
        pixel_array : np.ndarray
            The pixels to append

        Raises
        ------
        ValueError
            When the input pixel array's dtype does not match that of the
            SOP Instance
        """
        pixel_data_enum, pixel_data_attr = self._get_array_pixel_data_name(
            pixel_array
        )
        if self._pixel_data_type != pixel_data_enum:
            raise ValueError(
                "Data type of input pixel array "
                "does not match that of SOP instance. "
                f"Expected {self._pixel_data_type}, got {pixel_data_enum}"
            )

        # Framewise encoding
        # Before adding new pixel data, remove trailing null padding byte
        pixel_data_bytes = getattr(self, pixel_data_attr)
        if len(pixel_data_bytes) == get_expected_length(self) + 1:
            pixel_data_bytes = pixel_data_bytes[:-1]
            setattr(self, pixel_data_attr, pixel_data_bytes)

        if len(pixel_data_bytes) > 0:
            # PyDICOM property which handles the pixel data conversion.
            full_pixel_array = self.pixel_array.flatten()
        else:
            # If the pixel array is empty create a new one.
            full_pixel_array = np.array([], dtype=pixel_array.dtype)

        full_pixel_array = np.concatenate(
            [full_pixel_array, pixel_array.flatten()]
        )

        encoded_pixels = self._encode_pixels(full_pixel_array)
        # Add the 0 padding if necessary
        if len(encoded_pixels) % 2 == 1:
            encoded_pixels += b"0"

        setattr(self, pixel_data_attr, encoded_pixels)

    def _get_array_pixel_data_name(
        self, pixel_array: np.ndarray
    ) -> Tuple[PixelDataType, str]:
        """Returns the DICOM Attribute string for a given `np.ndarray`'s
        `dtype`.

        Parameters
        ----------
        pixel_array : np.ndarray
            The array to check

        Returns
        -------
        Tuple[PixelDataType, str]
            A tuple where the first element is the enum value and the second
            value is the DICOM pixel data attribute for the given datatype.
            One of (`PixelData`, `FloatPixelData`, `DoubleFloatPixelData`)

        Raises
        ------
        ValueError
            If the input array is not a floating point or integer datatype
        """
        if pixel_array.dtype.kind == "f":
            # Further check for float32 vs float64
            if pixel_array.dtype.name == "float32":
                return (
                    PixelDataType.FLOAT,
                    self._pixel_data_type_map[PixelDataType.FLOAT],
                )
            elif pixel_array.dtype.name == "float64":
                return (
                    PixelDataType.DOUBLE,
                    self._pixel_data_type_map[PixelDataType.DOUBLE],
                )
        elif pixel_array.dtype.kind == "i" or pixel_array.dtype.kind == "u":
            return (
                PixelDataType.INTEGER,
                self._pixel_data_type_map[PixelDataType.INTEGER],
            )
        raise ValueError(
            "Data type of argument `pixel_array` "
            "must be floating point or integer values."
        )

    def _encode_pixels(self, maps: np.ndarray) -> bytes:
        return maps.flatten().tobytes()
