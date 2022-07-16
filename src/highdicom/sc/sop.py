"""Module for SOP Classes of Secondary Capture (SC) Image IODs."""

import logging
import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union

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
    JPEG2000Lossless,
    JPEGLSLossless,
)

from highdicom.base import SOPClass
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
from highdicom.frame import encode_frame
from highdicom.sc.enum import ConversionTypeValues
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import check_person_name


logger = logging.getLogger(__name__)


class SCImage(SOPClass):

    """SOP class for a Secondary Capture (SC) Image, which represents a
    single-frame image that was converted from a non-DICOM format.
    """

    def __init__(
            self,
            pixel_array: np.ndarray,
            photometric_interpretation: Union[
                str,
                PhotometricInterpretationValues
            ],
            bits_allocated: int,
            coordinate_system: Union[str, CoordinateSystemNames],
            study_instance_uid: str,
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            manufacturer: str,
            patient_id: Optional[str] = None,
            patient_name: Optional[Union[str, PersonName]] = None,
            patient_birth_date: Optional[str] = None,
            patient_sex: Union[str, PatientSexValues, None] = None,
            accession_number: Optional[str] = None,
            study_id: str = None,
            study_date: Optional[Union[str, datetime.date]] = None,
            study_time: Optional[Union[str, datetime.time]] = None,
            referring_physician_name: Optional[Union[str, PersonName]] = None,
            pixel_spacing: Optional[Tuple[float, float]] = None,
            laterality: Optional[Union[str, LateralityValues]] = None,
            patient_orientation: Optional[
                Union[
                    Tuple[str, str],
                    Tuple[
                        PatientOrientationValuesBiped,
                        PatientOrientationValuesBiped,
                    ],
                    Tuple[
                        PatientOrientationValuesQuadruped,
                        PatientOrientationValuesQuadruped,
                    ]
                ]
            ] = None,
            anatomical_orientation_type: Optional[
                Union[str, AnatomicalOrientationTypeValues]
            ] = None,
            container_identifier: Optional[str] = None,
            issuer_of_container_identifier: Optional[IssuerOfIdentifier] = None,
            specimen_descriptions: Optional[
                Sequence[SpecimenDescription]
            ] = None,
            transfer_syntax_uid: str = ExplicitVRLittleEndian,
            **kwargs: Any
        ):
        """

        Parameters
        ----------
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
        patient_orientation:
                Union[Tuple[str, str], Tuple[highdicom.PatientOrientationValuesBiped, highdicom.PatientOrientationValuesBiped], Tuple[highdicom.PatientOrientationValuesQuadruped, highdicom.PatientOrientationValuesQuadruped], None], optional
            Orientation of the patient along the row and column axes of the
            image (required if `coordinate_system` is ``"PATIENT"``)
        anatomical_orientation_type: Union[str, highdicom.AnatomicalOrientationTypeValues, None], optional
            Type of anatomical orientation of patient relative to image (may be
            provide if `coordinate_system` is ``"PATIENT"`` and patient is
            an animal)
        container_identifier: Union[str, None], optional
            Identifier of the container holding the specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        issuer_of_container_identifier: Union[highdicom.IssuerOfIdentifier, None], optional
            Issuer of `container_identifier`
        specimen_descriptions: Union[Sequence[highdicom.SpecimenDescriptions], None], optional
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
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            RLELossless,
            JPEGBaseline8Bit,
            JPEG2000Lossless,
            JPEGLSLossless,
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

        super().__init__(
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

        coordinate_system = CoordinateSystemNames(coordinate_system)
        if coordinate_system == CoordinateSystemNames.PATIENT:
            if patient_orientation is None:
                raise TypeError(
                    'Patient orientation is required if coordinate system '
                    'is "PATIENT".'
                )

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

            row_orientation, col_orientation = patient_orientation
            if (anatomical_orientation_type ==
                    AnatomicalOrientationTypeValues.BIPED):
                patient_orientation = (
                    PatientOrientationValuesBiped(row_orientation).value,
                    PatientOrientationValuesBiped(col_orientation).value,
                )
            else:
                patient_orientation = (
                    PatientOrientationValuesQuadruped(row_orientation).value,
                    PatientOrientationValuesQuadruped(col_orientation).value,
                )
            self.PatientOrientation = list(patient_orientation)

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
            self.IssuerOfTheContainerIdentifierSequence: List[Dataset] = []
            if issuer_of_container_identifier is not None:
                self.IssuerOftheContainerIdentifierSequence.append(
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
        allowed_types = [np.bool_, np.uint8, np.uint16]
        if not any(pixel_array.dtype == t for t in allowed_types):
            raise TypeError(
                'Pixel array must be of type np.bool_, np.uint8 or np.uint16. '
                f'Found {pixel_array.dtype}.'
            )
        wrong_bit_depth_assignment = (
            pixel_array.dtype == np.bool_ and bits_allocated != 1,
            pixel_array.dtype == np.uint8 and bits_allocated != 8,
            pixel_array.dtype == np.uint16 and bits_allocated not in (12, 16),
        )
        if any(wrong_bit_depth_assignment):
            raise ValueError('Pixel array has an unexpected bit depth.')
        if bits_allocated not in (1, 8, 12, 16):
            raise ValueError('Unexpected number of bits allocated.')
        if transfer_syntax_uid == RLELossless and bits_allocated % 8 != 0:
            raise ValueError(
                'When using run length encoding, bits allocated must be a '
                'multiple of 8'
            )
        self.BitsAllocated = bits_allocated
        self.HighBit = self.BitsAllocated - 1
        self.BitsStored = self.BitsAllocated
        self.PixelRepresentation = 0
        photometric_interpretation = PhotometricInterpretationValues(
            photometric_interpretation
        )
        if pixel_array.ndim == 3:
            accepted_interpretations = {
                PhotometricInterpretationValues.RGB.value,
                PhotometricInterpretationValues.YBR_FULL.value,
                PhotometricInterpretationValues.YBR_FULL_422.value,
                PhotometricInterpretationValues.YBR_PARTIAL_420.value,
            }
            if photometric_interpretation.value not in accepted_interpretations:
                raise ValueError(
                    'Pixel array has an unexpected photometric interpretation.'
                )
            if pixel_array.shape[-1] != 3:
                raise ValueError(
                    'Pixel array has an unexpected number of color channels.'
                )
            if bits_allocated != 8:
                raise ValueError('Color images must be 8-bit.')
            if pixel_array.dtype != np.uint8:
                raise TypeError(
                    'Pixel array must have 8-bit unsigned integer data type '
                    'in case of a color image.'
                )
            self.PhotometricInterpretation = photometric_interpretation.value
            self.SamplesPerPixel = 3
            self.PlanarConfiguration = 0
        elif pixel_array.ndim == 2:
            accepted_interpretations = {
                PhotometricInterpretationValues.MONOCHROME1.value,
                PhotometricInterpretationValues.MONOCHROME2.value,
            }
            if photometric_interpretation.value not in accepted_interpretations:
                raise ValueError(
                    'Pixel array has an unexpected photometric interpretation.'
                )
            self.PhotometricInterpretation = photometric_interpretation.value
            self.SamplesPerPixel = 1
        else:
            raise ValueError(
                'Pixel array has an unexpected number of dimensions.'
            )
        if pixel_spacing is not None:
            if len(pixel_spacing) != 2:
                raise ValueError(
                    'Argument "pixel_spacing" should have length 2.'
                )
            self.PixelSpacing = [
                DS(ps, auto_format=True) for ps in pixel_spacing
            ]

        encoded_frame = encode_frame(
            pixel_array,
            transfer_syntax_uid=self.file_meta.TransferSyntaxUID,
            bits_allocated=self.BitsAllocated,
            bits_stored=self.BitsStored,
            photometric_interpretation=self.PhotometricInterpretation,
            pixel_representation=self.PixelRepresentation,
            planar_configuration=getattr(self, 'PlanarConfiguration', None)
        )
        if self.file_meta.TransferSyntaxUID.is_encapsulated:
            self.PixelData = encapsulate([encoded_frame])
        else:
            self.PixelData = encoded_frame

    @classmethod
    def from_ref_dataset(
        cls,
        ref_dataset: Dataset,
        pixel_array: np.ndarray,
        photometric_interpretation: Union[
            str,
            PhotometricInterpretationValues
        ],
        bits_allocated: int,
        coordinate_system: Union[str, CoordinateSystemNames],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        pixel_spacing: Optional[Tuple[int, int]] = None,
        laterality: Optional[Union[str, LateralityValues]] = None,
        patient_orientation: Optional[
            Union[
                Tuple[str, str],
                Tuple[
                    PatientOrientationValuesBiped,
                    PatientOrientationValuesBiped,
                ],
                Tuple[
                    PatientOrientationValuesQuadruped,
                    PatientOrientationValuesQuadruped,
                ]
            ]
        ] = None,
        anatomical_orientation_type: Optional[
            Union[str, AnatomicalOrientationTypeValues]
        ] = None,
        container_identifier: Optional[str] = None,
        issuer_of_container_identifier: Optional[IssuerOfIdentifier] = None,
        specimen_descriptions: Optional[
            Sequence[SpecimenDescription]
        ] = None,
        transfer_syntax_uid: str = ImplicitVRLittleEndian,
        **kwargs: Any
    ) -> 'SCImage':
        """Constructor that copies patient and study from an existing dataset.

        This provides a more concise way to construct an SCImage when an
        existing reference dataset from the study is available. All patient-
        and study- related attributes required by the main constructor are
        copied from the ``ref_dataset``, if present.

        The ``ref_dataset`` may be any dataset
        from the study to which the resulting SC image should belong, and
        contain all the relevant patient and study metadata. It does not need to
        be specifically related to the contents of the SCImage.

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
        photometric_interpretation: Union[str, highdicom.enum.PhotometricInterpretationValues]
            Interpretation of pixel data; either ``"MONOCHROME1"`` or
            ``"MONOCHROME2"`` for 2D grayscale images or ``"RGB"`` or
            ``"YBR_FULL"`` for 3D color images
        bits_allocated: int
            Number of bits that should be allocated per pixel value
        coordinate_system: Union[str, highdicom.enum.CoordinateSystemNames]
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
        laterality: Union[str, highdicom.enum.LateralityValues, None], optional
            Laterality of the examined body part
        patient_orientation:
                Union[Tuple[str, str], Tuple[highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped], Tuple[highdicom.enum.PatientOrientationValuesQuadruped, highdicom.enum.PatientOrientationValuesQuadruped], None], optional
            Orientation of the patient along the row and column axes of the
            image (required if `coordinate_system` is ``"PATIENT"``)
        anatomical_orientation_type: Union[str, highdicom.enum.AnatomicalOrientationTypeValues, None], optional
            Type of anatomical orientation of patient relative to image (may be
            provide if `coordinate_system` is ``"PATIENT"`` and patient is
            an animal)
        container_identifier: Union[str], optional
            Identifier of the container holding the specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        issuer_of_container_identifier: Union[highdicom.IssuerOfIdentifier, None], optional
            Issuer of `container_identifier`
        specimen_descriptions: Union[Sequence[highdicom.SpecimenDescriptions], None], optional
            Description of each examined specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported: RLE Lossless (``"1.2.840.10008.1.2.5"``), JPEG 2000
            Lossless (``"1.2.840.10008.1.2.4.90"``).
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Returns
        -------
        SCImage
            Secondary capture image.

        """  # noqa: E501
        return cls(
            pixel_array=pixel_array,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=ref_dataset.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            manufacturer=manufacturer,
            patient_id=getattr(ref_dataset, 'PatientID', None),
            patient_name=getattr(ref_dataset, 'PatientName', None),
            patient_birth_date=getattr(ref_dataset, 'PatientBirthDate', None),
            patient_sex=getattr(ref_dataset, 'PatientSex', None),
            accession_number=getattr(ref_dataset, 'AccessionNumber', None),
            study_id=getattr(ref_dataset, 'StudyID', None),
            study_date=getattr(ref_dataset, 'StudyDate', None),
            study_time=getattr(ref_dataset, 'StudyTime', None),
            referring_physician_name=getattr(
                ref_dataset,
                'ReferringPhysicianName',
                None
            ),
            pixel_spacing=pixel_spacing,
            laterality=laterality,
            patient_orientation=patient_orientation,
            anatomical_orientation_type=anatomical_orientation_type,
            container_identifier=container_identifier,
            issuer_of_container_identifier=issuer_of_container_identifier,
            specimen_descriptions=specimen_descriptions,
            transfer_syntax_uid=transfer_syntax_uid,
            **kwargs
        )
