"""Module for SOP Classes of Secondary Capture (SC) Image IODs."""

import logging
import datetime
from typing import Any
from collections.abc import Sequence
from typing_extensions import Self
import warnings

import numpy as np
from pydicom.uid import SecondaryCaptureImageStorage
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.valuerep import DA, DS, PersonName, TM
from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    RLELossless,
    JPEGBaseline8Bit,
    JPEG2000,
    JPEG2000Lossless,
    JPEGLSLossless,
    JPEGLSNearLossless,
)

from highdicom._standard_utils import (
    does_iod_have_pixel_data,
    is_multiframe_image,
)
from highdicom.base import SOPClass
from highdicom.base_content import ContributingEquipment
from highdicom.content import (
    IssuerOfIdentifier,
    SpecimenDescription,
)
from highdicom.enum import (
    AnatomicalOrientationTypeValues,
    CoordinateSystemNames,
    PhotometricInterpretationValues,
    LateralityValues,
    PatientOrientationValuesBiped,
    PatientOrientationValuesQuadruped,
    PatientSexValues,
)
from highdicom.image import Image
from highdicom.frame import encode_frame
from highdicom.sc.enum import ConversionTypeValues
from highdicom.spatial import (
    _check_image_orientation,
    _get_spatial_information,
    create_rotation_matrix,
    get_closest_patient_orientation,
    get_image_coordinate_system,
)
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import (
    check_person_name,
    _check_long_string,
)


logger = logging.getLogger(__name__)


class SCImage(Image):

    """SOP class for a Secondary Capture (SC) Image, which represents a
    single-frame image that was converted from a non-DICOM format.
    """

    def __init__(
        self,
        pixel_array: np.ndarray,
        photometric_interpretation: (
            str |
            PhotometricInterpretationValues
        ),
        coordinate_system: str | CoordinateSystemNames,
        study_instance_uid: str,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        patient_id: str | None = None,
        patient_name: str | PersonName | None = None,
        patient_birth_date: str | None = None,
        patient_sex: str | PatientSexValues | None = None,
        accession_number: str | None = None,
        study_id: str | None = None,
        study_date: str | datetime.date | None = None,
        study_time: str | datetime.time | None = None,
        referring_physician_name: str | PersonName | None = None,
        pixel_spacing: tuple[float, float] | None = None,
        laterality: str | LateralityValues | None = None,
        patient_orientation: (
            tuple[str, str] |
            tuple[
                PatientOrientationValuesBiped,
                PatientOrientationValuesBiped,
            ] |
            tuple[
                PatientOrientationValuesQuadruped,
                PatientOrientationValuesQuadruped,
            ]
        ) | None = None,
        anatomical_orientation_type: None | (
            str | AnatomicalOrientationTypeValues
        ) = None,
        container_identifier: str | None = None,
        issuer_of_container_identifier: IssuerOfIdentifier | None = None,
        specimen_descriptions: None | (
            Sequence[SpecimenDescription]
        ) = None,
        transfer_syntax_uid: str = ExplicitVRLittleEndian,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        *,
        bits_stored: int | None = None,
        bits_allocated: int | None = None,
        image_position: Sequence[float] | None = None,
        image_orientation: Sequence[float] | None = None,
        slice_thickness: float | None = None,
        spacing_between_slices: float | None = None,
        frame_of_reference_uid: str | None = None,
        position_reference_indicator: str | None = None,
        **kwargs: Any,
    ):
        """

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Array of pixel values representing a single-frame image; either a
            2D grayscale image with shape (H, W) or a 3D color image with shape
            (H, W, 3) in the RGB color space. Acceptable datatypes are bool,
            uint8, uint16, int8, or int16 for grayscale images, and uint8 or
            uint16 for color images.
        photometric_interpretation: Union[str, highdicom.PhotometricInterpretationValues]
            Interpretation with which to store pixel data in the dataset;
            either ``"MONOCHROME1"`` or ``"MONOCHROME2"`` for single channel
            grayscale images or ``"RGB"`` or ``"YBR_FULL"`` for 3 channel color
            images. Note that this should match the photometric interpretation
            of the input pixel array, except in the following cases: if
            ``transfer_syntax_uid`` is ``"JPEGBaseline8Bit"``,
            ``photometric_interpretation must be ``"YBR_FULL_422"``, if
            ``transfer_syntax_uid`` is ``"JPEG2000"``,
            ``photometric_interpretation must be ``"YBR_ICT"``, if
            ``transfer_syntax_uid`` is ``"JPEG2000Lossless"``,
            ``photometric_interpretation must be ``"YBR_RCT"``.
        coordinate_system: Union[str, highdicom.CoordinateSystemNames]
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
            imaging
        study_instance_uid: str
            Study Instance UID
        series_instance_uid: str
            Series Instance UID of the SC image series
        series_number: int
            Series Number of the SC image series
        sop_instance_uid: str
            SOP instance UID that should be assigned to the SC image instance
        instance_number: int
            Number that should be assigned to this SC image instance
        manufacturer: str
            Name of the manufacturer of the device that creates the SC image
            instance (in a research setting this is typically the same
            as `institution_name`)
        patient_id: Union[str, None], optional
           ID of the patient (medical record number)
        patient_name: Union[str, PersonName, None], optional
           Name of the patient
        patient_birth_date: Union[str, None], optional
           Patient's birth date
        patient_sex: Union[str, highdicom.PatientSexValues, None], optional
           Patient's sex
        study_id: Union[str, None], optional
           ID of the study
        accession_number: Union[str, None], optional
           Accession number of the study
        study_date: Union[str, datetime.date, None], optional
           Date of study creation
        study_time: Union[str, datetime.time, None], optional
           Time of study creation
        referring_physician_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the referring physician
        pixel_spacing: Union[Tuple[float, float], None], optional
            Physical spacing in millimeter between pixels along the row and
            column dimension
        laterality: Union[str, highdicom.LateralityValues, None], optional
            Laterality of the examined body part
        patient_orientation: Union[Tuple[str, str], Tuple[highdicom.PatientOrientationValuesBiped, highdicom.PatientOrientationValuesBiped], Tuple[highdicom.PatientOrientationValuesQuadruped, highdicom.PatientOrientationValuesQuadruped], None], optional
            Orientation of the patient moving from the first to the last row
            (first item) and from the first column to the last column (second
            item). Required if `coordinate_system` is ``"PATIENT"`` and
            ``image_orientation`` is not provided. If ``image_orientation`` is
            provided, it will be used to infer the correct value for
            ``patient_orientation``.
        anatomical_orientation_type: Union[str, highdicom.AnatomicalOrientationTypeValues, None], optional
            Type of anatomical orientation of patient relative to image (may be
            provide if `coordinate_system` is ``"PATIENT"`` and patient is
            an animal)
        container_identifier: Union[str, None], optional
            Identifier of the container holding the specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        issuer_of_container_identifier: Union[highdicom.IssuerOfIdentifier, None], optional
            Issuer of `container_identifier`
        specimen_descriptions: Union[Sequence[highdicom.SpecimenDescription], None], optional
            Description of each examined specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following compressed transfer syntaxes
            are supported: RLE Lossless (``"1.2.840.10008.1.2.5"``), JPEG
            2000 Lossless (``"1.2.840.10008.1.2.4.90"``), JPEG-LS Lossless
            (``"1.2.840.10008.1.2.4.80"``), and JPEG Baseline
            (``"1.2.840.10008.1.2.4.50"``). Note that JPEG Baseline is a
            lossy compression method that will lead to a loss of detail in
            the image.
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        bits_stored: int | None, optional
            Number of bits that should be stored per pixel value. By default,
            the number of bits per pixel in the input array is assumed. Any
            provided value must be less than or equal to the number of bits per
            pixel in the input array.
        bits_allocated: int | None, optional
            *Deprecated* in highdicom 0.29.0. This was previously the name for
            the ``bits_stored`` parameter, but this name was deprecated because
            it incorrectly described how the attribute was used in the
            resulting DICOM image. For backwards compatibility, passing
            ``bits_allocated`` is equivalent to passing ``bits_stored`` and
            controls the bits stored in the file. This parameter will be
            removed entirely in a future release. To upgrade, pass the value
            to the keyword-only ``bits_stored`` parameter instead.
        image_position: Sequence[float] | None, optional
            Position, in mm, of the center of the top left pixel in the
            frame-of reference-coordinate system. Must contain exactly 3
            values. Pass only if using the "PATIENT" coordinate system.
        image_orientation: Sequence[float] | None, optional
            Direction cosines of the image within the frame-of-reference
            coordinate system. Must contain exactly 6 values. Pass only if
            using the "PATIENT" coordinate system.
        slice_thickness: float | None, optional
            Slice thickness of the image in mm. Pass only if using the
            "PATIENT" coordinate system.
        spacing_between_slices: float | None, optional
            Spacing between slices, in mm. Pass only if using the "PATIENT"
            coordinate system.
        frame_of_reference_uid: str | None = None, optional
            Frame of reference UID, required if 'image_orientation' and
            'image_orientation' are provided.
        position_reference_indicator: str | None = None, optional
            Description of the part of the imaging target used to define the
            frame-of-reference coordinate system. Should only be provided
            if 'frame_of_reference_uid' is provided.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            RLELossless,
            JPEGBaseline8Bit,
            JPEG2000,
            JPEG2000Lossless,
            JPEGLSLossless,
            JPEGLSNearLossless,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported'
            )

        # Check names
        if patient_name is not None:
            check_person_name(patient_name)
        if referring_physician_name is not None:
            check_person_name(referring_physician_name)

        SOPClass.__init__(
            self,
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=SecondaryCaptureImageStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            modality='OT',
            transfer_syntax_uid=transfer_syntax_uid,
            patient_id=patient_id,
            patient_name=patient_name,
            patient_birth_date=patient_birth_date,
            patient_sex=patient_sex,
            accession_number=accession_number,
            study_id=study_id,
            study_date=study_date,
            study_time=study_time,
            referring_physician_name=referring_physician_name,
            **kwargs
        )
        self._add_contributing_equipment(contributing_equipment)

        # If we received an integer for coordinate_system, assume that the user
        # is using the old signature that included the deprecated
        # 'bits_allocated' with positional arguments
        if isinstance(bits_allocated, int):
            raise TypeError(
                "Use of 'bits_allocated' as a positional argument was "
                "deprecated in highdicom 0.29.0. Omit the parameter to use "
                "the number of bits per pixel in the input array as both "
                "the number of bits allocated and bits stored or specify "
                "'bits_stored' as a keyword argument."
            )

        coordinate_system = CoordinateSystemNames(coordinate_system)
        if coordinate_system == CoordinateSystemNames.PATIENT:
            # General Series
            if laterality is not None:
                laterality = LateralityValues(laterality)
                self.Laterality = laterality.value

            # General Image
            if anatomical_orientation_type is not None:
                anatomical_orientation_type = AnatomicalOrientationTypeValues(
                    anatomical_orientation_type
                )
                self.AnatomicalOrientationType = \
                    anatomical_orientation_type.value
            else:
                anatomical_orientation_type = \
                    AnatomicalOrientationTypeValues.BIPED

        elif coordinate_system == CoordinateSystemNames.SLIDE:
            if container_identifier is None:
                raise TypeError(
                    'Container identifier is required if coordinate system '
                    'is "SLIDE".'
                )
            if specimen_descriptions is None:
                raise TypeError(
                    'Specimen descriptions are required if coordinate system '
                    'is "SLIDE".'
                )

            # Specimen
            self.ContainerIdentifier = container_identifier
            self.IssuerOfTheContainerIdentifierSequence: list[Dataset] = []
            if issuer_of_container_identifier is not None:
                self.IssuerOfTheContainerIdentifierSequence.append(
                    issuer_of_container_identifier
                )
            code: Code = codes.SCT.MicroscopeSlide
            container_type_item = CodedConcept(*code)
            self.ContainerTypeCodeSequence = [container_type_item]
            self.SpecimenDescriptionSequence = specimen_descriptions

        # SC Equipment
        self.ConversionType = ConversionTypeValues.DI.value

        # SC Image
        now = datetime.datetime.now()
        self.DateOfSecondaryCapture = DA(now.date())
        self.TimeOfSecondaryCapture = TM(now.time())

        # Image Pixel
        self.ImageType = ['DERIVED', 'SECONDARY', 'OTHER']
        self.Rows = pixel_array.shape[0]
        self.Columns = pixel_array.shape[1]

        photometric_interpretation = PhotometricInterpretationValues(
            photometric_interpretation
        )
        if pixel_array.ndim == 3:
            if pixel_array.shape[-1] != 3:
                raise ValueError(
                    'Pixel array has an unexpected number of color channels.'
                )

            if transfer_syntax_uid == JPEGBaseline8Bit:
                accepted_interpretations = {
                    PhotometricInterpretationValues.YBR_FULL_422.value,
                }
            elif transfer_syntax_uid == JPEG2000:
                accepted_interpretations = {
                    PhotometricInterpretationValues.YBR_ICT.value,
                }
            elif transfer_syntax_uid == JPEG2000Lossless:
                accepted_interpretations = {
                    PhotometricInterpretationValues.YBR_RCT.value,
                }
            else:
                accepted_interpretations = {
                    PhotometricInterpretationValues.RGB.value,
                    PhotometricInterpretationValues.YBR_FULL.value,
                }

            if photometric_interpretation.value not in accepted_interpretations:
                raise ValueError(
                    'Pixel array has an unexpected photometric interpretation.'
                )

            if pixel_array.dtype.kind != 'u':
                raise TypeError(
                    'Pixel array must have an unsigned '
                    'integer datatype for color images. '
                    f'Found {pixel_array.dtype}.'
                )

            self.BitsAllocated = pixel_array.itemsize * 8

            if self.BitsAllocated not in (1, 8, 16, 32):
                raise TypeError(
                    'Pixel array has an unexpected bit depth.'
                )

            self.PhotometricInterpretation = photometric_interpretation.value
            self.SamplesPerPixel = 3
            self.PlanarConfiguration = 0
            self.PixelRepresentation = 0

        elif pixel_array.ndim == 2:
            accepted_interpretations = {
                PhotometricInterpretationValues.MONOCHROME1.value,
                PhotometricInterpretationValues.MONOCHROME2.value,
            }

            if photometric_interpretation.value not in accepted_interpretations:
                raise ValueError(
                    'Pixel array has an unexpected photometric interpretation.'
                )

            if pixel_array.dtype.kind == 'b':
                self.BitsAllocated = 1
                self.PixelRepresentation = 0
            elif pixel_array.dtype.kind == 'u':
                self.PixelRepresentation = 0
                self.BitsAllocated = pixel_array.itemsize * 8
            elif pixel_array.dtype.kind == 'i':
                self.PixelRepresentation = 1
                self.BitsAllocated = pixel_array.itemsize * 8
            else:
                raise TypeError(
                    'Pixel array must have signed or unsigned '
                    'integer datatype for monochrome images. '
                    f'Found {pixel_array.dtype}.'
                )

            if self.BitsAllocated not in (1, 8, 16, 32):
                raise TypeError(
                    'Pixel array has an unexpected bit depth.'
                )

            self.PhotometricInterpretation = photometric_interpretation.value
            self.SamplesPerPixel = 1
        else:
            raise ValueError(
                'Pixel array has an unexpected number of dimensions.'
            )

        if bits_allocated is not None:
            # Handle deprecation of bits_allocated
            if bits_stored is not None:
                raise TypeError(
                    "The 'bits_allocated' parameter is deprecated and must "
                    "not be provided alongside 'bits_stored'."
                )

            warnings.warn(
                "The 'bits_allocated' parameter is deprecated as of highdicom "
                "0.29.0 and will be removed in a future version of the "
                "library. Use 'bits_stored' to set the number of bits stored "
                "or skip the parameter entirely to use the number of bits per "
                "pixel in the input array as the number of bits allocated.",
                UserWarning,
                stacklevel=2,
            )
            # Interpret the bits_allocated as bits_stored in line with the old,
            # deprecated behavior
            bits_stored = bits_allocated

        if bits_stored is not None:
            if not isinstance(bits_stored, int):
                raise TypeError("Parameter 'bits_stored' must be an integer.")

            if bits_stored < 1 or bits_stored > self.BitsAllocated:
                raise ValueError(
                    "'bits_stored' must be an integer no greater than the "
                    "number of bits per pixel in the input array."
                )

            self.BitsStored = bits_stored

            if (
                bits_stored != self.BitsAllocated and
                pixel_array.max() >= 2 ** bits_stored
            ):
                raise ValueError(
                    "Provided 'pixel_array' cannot be represented with the "
                    f"specified number of 'bits_stored' ({bits_stored})."
                )
        else:
            # By default, assume the full range of pixel values in the input
            # array
            self.BitsStored = self.BitsAllocated

        self.HighBit = self.BitsStored - 1

        if patient_orientation is not None:
            if len(patient_orientation) != 2:
                raise ValueError(
                    "Wrong number of items in 'patient_orientation'."
                )
            row_orientation, col_orientation = patient_orientation
            if (
                anatomical_orientation_type ==
                AnatomicalOrientationTypeValues.BIPED
            ):
                patient_orientation = (
                    PatientOrientationValuesBiped(row_orientation),
                    PatientOrientationValuesBiped(col_orientation),
                )
            else:
                patient_orientation = (
                    PatientOrientationValuesQuadruped(row_orientation),
                    PatientOrientationValuesQuadruped(col_orientation),
                )

        if frame_of_reference_uid is not None:
            self.FrameOfReferenceUID = frame_of_reference_uid
            if position_reference_indicator is not None:
                if (
                    coordinate_system == CoordinateSystemNames.SLIDE and
                    position_reference_indicator != "SLIDE_CORNER"
                ):
                    raise ValueError(
                        "'position_reference_indicator' must be "
                        "'SLIDE_CORNER' when using the slide"
                        "coordinate system. Leave this argument "
                        "unspecified to use this default."
                    )

                _check_long_string(position_reference_indicator)

            elif coordinate_system == CoordinateSystemNames.SLIDE:
                # This is the only valid value for slides
                position_reference_indicator = "SLIDE_CORNER"

            # Type 2, set to empty if parameter is None and using PATIENT
            # coordinate system
            self.PositionReferenceIndicator = position_reference_indicator
        elif position_reference_indicator is not None:
            raise TypeError(
                "Argument 'position_reference_indicator' may not be provided "
                "if 'frame_of_reference_uid' is not provided."
            )

        if pixel_spacing is not None:
            # Note that pixel spacing apepars in both the Image Plane module
            # (as type 1) and the SC Image module (as type !C). So it can be
            # included regardless of whether position/orientation are included
            if len(pixel_spacing) != 2:
                raise ValueError(
                    'Argument "pixel_spacing" should have length 2.'
                )
            self.PixelSpacing = [
                DS(ps, auto_format=True) for ps in pixel_spacing
            ]

        # The Image Plane moduie is User defined "U" in the SC Image IOD
        if (image_position is None) != (image_orientation is None):
            # Both position and orientation are required if the Image Plane
            # moduleu is to be included
            raise TypeError(
                "Both 'image_position' and 'image_orientation' must be "
                "provided if either is provided."
            )

        if image_position is not None and image_orientation is not None:

            if coordinate_system != CoordinateSystemNames.PATIENT:
                raise TypeError(
                    "Arguments 'image_orientation' and 'image_position' are "
                    "only valid when using the 'PATIENT' coordinate system."
                )

            # In this case, PixelSpacing is type 1. Setting the attribute
            # occurred above, ir present
            if pixel_spacing is None:
                raise TypeError(
                    "Argument 'pixel_spacing' is required if 'image_position' "
                    "and 'image_orientation' are provided."
                )

            # Frame of Reference Module is required if ImagePosition is present
            if frame_of_reference_uid is None:
                raise TypeError(
                    "Argument 'frame_of_reference_uid' is required if "
                    "'image_position' and 'image_orientation' are provided."
                )

            if len(image_position) != 3:
                raise ValueError(
                    "Argument 'image_position' must have length 3."
                )

            self.ImagePositionPatient = [
                DS(p, auto_format=True) for p in image_position
            ]
            _check_image_orientation(image_orientation)
            self.ImageOrientationPatient = [
                DS(p, auto_format=True) for p in image_orientation
            ]

            if slice_thickness is not None:
                self.SliceThickness = DS(slice_thickness, auto_format=True)
            else:
                # Type 2 in Image Plane module
                self.SliceThickness = None

            if spacing_between_slices is not None:
                # Type 3 - skip if not provided
                self.SpacingBetweenSlices = DS(
                    spacing_between_slices,
                    auto_format=True
                )

            if (
                anatomical_orientation_type ==
                AnatomicalOrientationTypeValues.BIPED
            ):
                # For the BIPED case, we can check the patient orientation
                # given the image orientation. For quadrupeds, this is trickier
                # because it depends where you are in the body
                calculated_patient_orientation = (
                    get_closest_patient_orientation(
                        create_rotation_matrix(image_orientation)
                    )[:-1]
                )

                if (
                    patient_orientation is not None and
                    patient_orientation != calculated_patient_orientation
                ):
                    raise ValueError(
                        "The provided 'patient_orientation' is not consistent "
                        "with the provided value for 'image_orientation'. "
                        "Leave 'patient_orientation' unspecified to infer its "
                        "value from 'image_orientation'."
                    )

                patient_orientation = calculated_patient_orientation

        else:
            if coordinate_system == CoordinateSystemNames.PATIENT:
                # We are using the patient coordinate system, but not including
                # the Image Plane module. Therefore the optional Image Plane
                # attributes should not be provided
                reason = (
                    "in combination with 'image_position' and "
                    "'image_orientation'."
                )

                if patient_orientation is None:
                    raise TypeError(
                        "Patient orientation is required if coordinate system "
                        "is 'PATIENT' amd 'image_orientation' is not provided."
                    )
            else:
                # Trying to include Image Plane module in a coordinate system
                # other than patient is not allowed
                reason = "when using the 'PATIENT' coordinate system."

            if slice_thickness is not None:
                raise TypeError(
                    f"Argugment 'slice_thickness' may only be provided {reason}"
                )

            if spacing_between_slices is not None:
                raise TypeError(
                    f"Argugment 'spacing_between_slices' may only be provided "
                    f"{reason}"
                )

        if patient_orientation is not None:
            self.PatientOrientation = [p.value for p in patient_orientation]

        encoded_frame = encode_frame(
            pixel_array,
            transfer_syntax_uid=self.file_meta.TransferSyntaxUID,
            bits_allocated=self.BitsAllocated,
            bits_stored=self.BitsStored,
            photometric_interpretation=self.PhotometricInterpretation,
            pixel_representation=self.PixelRepresentation,
            planar_configuration=self.get('PlanarConfiguration'),
        )
        if self.file_meta.TransferSyntaxUID.is_encapsulated:
            self.PixelData = encapsulate([encoded_frame])
        else:
            self.PixelData = encoded_frame

        self._build_luts()
        self._file_reader = None

    @classmethod
    def from_ref_dataset(
        cls,
        ref_dataset: Dataset,
        pixel_array: np.ndarray,
        photometric_interpretation: (
            str |
            PhotometricInterpretationValues
        ),
        coordinate_system: str | CoordinateSystemNames,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        pixel_spacing: tuple[float, float] | None = None,
        laterality: str | LateralityValues | None = None,
        patient_orientation: (
            tuple[str, str] |
            tuple[
                PatientOrientationValuesBiped,
                PatientOrientationValuesBiped,
            ] |
            tuple[
                PatientOrientationValuesQuadruped,
                PatientOrientationValuesQuadruped,
            ]
        ) | None = None,
        anatomical_orientation_type: None | (
            str | AnatomicalOrientationTypeValues
        ) = None,
        container_identifier: str | None = None,
        issuer_of_container_identifier: IssuerOfIdentifier | None = None,
        specimen_descriptions: None | (
            Sequence[SpecimenDescription]
        ) = None,
        transfer_syntax_uid: str = ImplicitVRLittleEndian,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        *,
        bits_stored: int | None = None,
        bits_allocated: int | None = None,
        image_position: Sequence[float] | None = None,
        image_orientation: Sequence[float] | None = None,
        slice_thickness: float | None = None,
        spacing_between_slices: float | None = None,
        frame_of_reference_uid: str | None = None,
        position_reference_indicator: str | None = None,
        **kwargs: Any
    ) -> Self:
        """Constructor that copies patient/study data from an existing dataset.

        This provides a more concise way to construct an SCImage when an
        existing reference dataset from the study is available. All patient-
        and study- related attributes required by the main constructor are
        copied from the ``ref_dataset``, if present.

        The ``ref_dataset`` may be any dataset from the study to which the
        resulting SC image should belong, and contain all the relevant patient
        and study metadata.

        Parameters
        ----------
        ref_dataset: pydicom.dataset.Dataset
            An existing dataset from the study to which the SCImage should
            belong. Patient- and study-related metadata will be copied from
            this dataset.
        pixel_array: numpy.ndarray
            Array of unsigned integer pixel values representing a single-frame
            image; either a 2D grayscale image or a 3D color image
            (RGB color space)
        photometric_interpretation: Union[str, highdicom.PhotometricInterpretationValues]
            Interpretation of pixel data; either ``"MONOCHROME1"`` or
            ``"MONOCHROME2"`` for 2D grayscale images or ``"RGB"`` or
            ``"YBR_FULL"`` for 3D color images
        bits_allocated: int
            Number of bits that should be allocated per pixel value
        coordinate_system: Union[str, highdicom.CoordinateSystemNames]
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
            imaging
        series_instance_uid: str
            Series Instance UID of the SC image series
        series_number: int
            Series Number of the SC image series
        sop_instance_uid: str
            SOP instance UID that should be assigned to the SC image instance
        instance_number: int
            Number that should be assigned to this SC image instance
        manufacturer: str
            Name of the manufacturer of the device that creates the SC image
            instance (in a research setting this is typically the same
            as `institution_name`)
        pixel_spacing: Union[Tuple[int, int]], optional
            Physical spacing in millimeter between pixels along the row and
            column dimension
        laterality: Union[str, highdicom.LateralityValues, None], optional
            Laterality of the examined body part
        patient_orientation: Union[Tuple[str, str], Tuple[highdicom.PatientOrientationValuesBiped, highdicom.PatientOrientationValuesBiped], Tuple[highdicom.PatientOrientationValuesQuadruped, highdicom.PatientOrientationValuesQuadruped], None], optional
            Orientation of the patient along the row and column axes of the
            image (required if `coordinate_system` is ``"PATIENT"``)
        anatomical_orientation_type: Union[str, highdicom.AnatomicalOrientationTypeValues, None], optional
            Type of anatomical orientation of patient relative to image (may be
            provide if `coordinate_system` is ``"PATIENT"`` and patient is
            an animal)
        container_identifier: Union[str], optional
            Identifier of the container holding the specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        issuer_of_container_identifier: Union[highdicom.IssuerOfIdentifier, None], optional
            Issuer of `container_identifier`
        specimen_descriptions: Union[Sequence[highdicom.SpecimenDescription], None], optional
            Description of each examined specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported: RLE Lossless (``"1.2.840.10008.1.2.5"``), JPEG 2000
            Lossless (``"1.2.840.10008.1.2.4.90"``).
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        bits_stored: int | None, optional
            Number of bits that should be stored per pixel value. By default,
            the number of bits per pixel in the input array is assumed. Any
            provided value must be less than or equal to the number of bits per
            pixel in the input array.
        bits_allocated: int | None, optional
            *Deprecated* in highdicom 0.29.0. This was previously the name for
            the ``bits_stored`` parameter, but this name was deprecated because
            it incorrectly described how the attribute was used in the
            resulting DICOM image. For backwards compatibility, passing
            ``bits_allocated`` is equivalent to passing ``bits_stored`` and
            controls the bits stored in the file. This parameter will be
            removed entirely in a future release.
        image_position: Sequence[float] | None, optional
            Position, in mm, of the center of the top left pixel in the
            frame-of reference-coordinate system. Must contain exactly 3
            values.
        image_orientation: Sequence[float] | None, optional
            Direction cosines of the image within the frame-of-reference
            coordinate system. Must contain exactly 6 values.
        slice_thickness: float | None, optional
            Slice thickness of the image in mm.
        spacing_between_slices: float | None, optional
            Spacing between slices, in mm.
        frame_of_reference_uid: str | None = None, optional
            Frame of reference UID, required if 'image_orientation' and
            'image_orientation' are provided.
        position_reference_indicator: str | None = None, optional
            Description of the part of the imaging target used to define the
            frame-of-reference coordinate system. Should only be provided
            if 'frame_of_reference_uid' is provided.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Returns
        -------
        SCImage
            Secondary capture image.

        """  # noqa: E501
        try:
            acquisition_equipment = (
                ContributingEquipment.for_image_acquisition(ref_dataset)
            )
        except Exception:
            pass
        else:
            if contributing_equipment is None:
                contributing_equipment = []
            else:
                contributing_equipment = list(contributing_equipment)
            contributing_equipment = [
                acquisition_equipment,
                *list(contributing_equipment),
            ]

        if 'specific_character_set' in kwargs:
            specific_character_set = kwargs.pop('specific_character_set')
        else:
            specific_character_set = ref_dataset.get(
                'SpecificCharacterSet'
            )

        sc = cls(
            pixel_array=pixel_array,
            photometric_interpretation=photometric_interpretation,
            coordinate_system=coordinate_system,
            study_instance_uid=ref_dataset.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            manufacturer=manufacturer,
            pixel_spacing=pixel_spacing,
            laterality=laterality,
            patient_orientation=patient_orientation,
            anatomical_orientation_type=anatomical_orientation_type,
            container_identifier=container_identifier,
            issuer_of_container_identifier=issuer_of_container_identifier,
            specimen_descriptions=specimen_descriptions,
            transfer_syntax_uid=transfer_syntax_uid,
            contributing_equipment=contributing_equipment,
            specific_character_set=specific_character_set,
            bits_stored=bits_stored,
            bits_allocated=bits_allocated,
            image_position=image_position,
            image_orientation=image_orientation,
            slice_thickness=slice_thickness,
            spacing_between_slices=spacing_between_slices,
            frame_of_reference_uid=frame_of_reference_uid,
            position_reference_indicator=position_reference_indicator,
            **kwargs
        )
        sc.copy_patient_and_study_information(ref_dataset)
        sc.copy_specimen_information(ref_dataset)

        return sc

    @classmethod
    def from_aligned_ref_image(
        cls,
        ref_image: Dataset,
        pixel_array: np.ndarray,
        *,
        photometric_interpretation: (
            str |
            PhotometricInterpretationValues
        ),
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        transfer_syntax_uid: str = ImplicitVRLittleEndian,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        bits_stored: int | None = None,
        **kwargs: Any
    ) -> Self:
        """Constructor that copies patient/study and spatial data from an
        existing dataset.

        This is similar to :class:`highdicom.sc.SCImage.from_ref_dataset`, but
        additionally copies spatial information from the reference dataset.
        This is only possible when the reference dataset represents an image in
        the "PATIENT" coordinate system, and only correct if the new secondary
        capture image is spatially aligned pixel-for-pixel with the reference
        image.

        Parameters
        ----------
        ref_image: pydicom.dataset.Dataset
            An existing image, spatially aligned with the new SC image and
            belonging to the study to which the SCImage should belong. Patient-
            and study-related and spatial metadata will be copied from this
            image.
        pixel_array: numpy.ndarray
            Array of unsigned integer pixel values representing a single-frame
            image; either a 2D grayscale image or a 3D color image
            (RGB color space)
        photometric_interpretation: Union[str, highdicom.PhotometricInterpretationValues]
            Interpretation of pixel data; either ``"MONOCHROME1"`` or
            ``"MONOCHROME2"`` for 2D grayscale images or ``"RGB"`` or
            ``"YBR_FULL"`` for 3D color images
        bits_allocated: int
            Number of bits that should be allocated per pixel value
        series_instance_uid: str
            Series Instance UID of the SC image series
        series_number: int
            Series Number of the SC image series
        sop_instance_uid: str
            SOP instance UID that should be assigned to the SC image instance
        instance_number: int
            Number that should be assigned to this SC image instance
        manufacturer: str
            Name of the manufacturer of the device that creates the SC image
            instance (in a research setting this is typically the same
            as `institution_name`)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported: RLE Lossless (``"1.2.840.10008.1.2.5"``), JPEG 2000
            Lossless (``"1.2.840.10008.1.2.4.90"``).
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        bits_stored: int | None, optional
            Number of bits that should be stored per pixel value. By default,
            the number of bits per pixel in the input array is assumed. Any
            provided value must be less than or equal to the number of bits per
            pixel in the input array.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Returns
        -------
        SCImage
            Secondary capture image.

        """  # noqa: E501
        # Attempt to copy spatial information from the reference dataset
        # This is only valid in the patient coordinate system and if the
        # image sizes match
        coordinate_system = get_image_coordinate_system(ref_image)
        if coordinate_system != CoordinateSystemNames.PATIENT:
            raise ValueError(
                "Using 'from_aligned_ref_image' is only possible "
                "when using the 'PATIENT' coordinate system."
            )

        if not does_iod_have_pixel_data(ref_image.SOPClassUID):
            raise ValueError(
                "'ref_image' must be an image when using "
                "'from_aligned_ref_image'."
            )

        if is_multiframe_image(ref_image):
            raise ValueError(
                "'ref_image' must be a single frame image when using "
                "'from_aligned_ref_image'."
            )
        if pixel_array.shape[:2] != (ref_image.Rows, ref_image.Columns):
            raise ValueError(
                "Shape of 'pixel_array' must match rows and columns of "
                "'ref_image' when using 'from_aligned_ref_image'."
            )

        (
            image_position,
            image_orientation,
            pixel_spacing,
            spacing_between_slices,
        ) = _get_spatial_information(ref_image)

        slice_thickness = ref_image.get('SliceThickness')
        laterality = ref_image.get('Laterality')
        frame_of_reference_uid = ref_image.FrameOfReferenceUID
        position_reference_indicator = ref_image.get(
            'PositionReferenceIndicator'
        )

        if laterality not in ('L', 'R'):
            # Do not fail due to invalid laterality values in the reference
            laterality = None

        anatomical_orientation_type = ref_image.get('AnatomicalOrientationType')

        patient_orientation = None
        if (
            anatomical_orientation_type is not None and
            anatomical_orientation_type ==
            AnatomicalOrientationTypeValues.QUADRUPED.value
        ):
            # Otherwise it will be inferred from image orientation in the main
            # constructor
            patient_orientation = ref_image.get('PatientOrientation')

        return cls.from_ref_dataset(
            ref_dataset=ref_image,
            pixel_array=pixel_array,
            photometric_interpretation=photometric_interpretation,
            coordinate_system=coordinate_system,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            manufacturer=manufacturer,
            laterality=laterality,
            anatomical_orientation_type=anatomical_orientation_type,
            transfer_syntax_uid=transfer_syntax_uid,
            contributing_equipment=contributing_equipment,
            bits_stored=bits_stored,
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            spacing_between_slices=spacing_between_slices,
            frame_of_reference_uid=frame_of_reference_uid,
            position_reference_indicator=position_reference_indicator,
            patient_orientation=patient_orientation,
            **kwargs,
        )
