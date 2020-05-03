"""Module for SOP Classes of Secondary Capture (SC) Image IODs."""

import logging
import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydicom._storage_sopclass_uids import SecondaryCaptureImageStorage
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes
from pydicom.valuerep import DA, TM

from highdicom.base import SOPClass
from highdicom.content import (
    IssuerOfIdentifier,
    SpecimenDescription,
)
from highdicom.enum import (
    AnatomicalOrientationTypeValues,
    CoordinateSystemNames,
    LateralityValues,
    PhotometricInterpretationValues,
    PatientOrientationValuesBiped,
    PatientOrientationValuesQuadruped,
)
from highdicom.sc.enum import ConversionTypeValues
from highdicom.sr.coding import CodedConcept


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
            patient_name: Optional[str] = None,
            patient_birth_date: Optional[str] = None,
            patient_sex: Optional[str] = None,
            accession_number: Optional[str] = None,
            study_id: str = None,
            study_date: Optional[Union[str, datetime.date]] = None,
            study_time: Optional[Union[str, datetime.time]] = None,
            referring_physician_name: Optional[str] = None,
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
            **kwargs: Any
        ):
        """

        Parameters
        ----------
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
        study_instance_uid: str
            Study Instance UID
        series_instance_uid: str
            Series Instance UID of the SC image series
        series_number: Union[int, None]
            Series Number of the SC image series
        sop_instance_uid: str
            SOP instance UID that should be assigned to the SC image instance
        instance_number: int
            Number that should be assigned to this SC image instance
        manufacturer: str
            Name of the manufacturer of the device that creates the SC image
            instance (in a research setting this is typically the same
            as `institution_name`)
        patient_id: str, optional
           ID of the patient (medical record number)
        patient_name: str, optional
           Name of the patient
        patient_birth_date: str, optional
           Patient's birth date
        patient_sex: str, optional
           Patient's sex
        study_id: str, optional
           ID of the study
        accession_number: str, optional
           Accession number of the study
        study_date: Union[str, datetime.date], optional
           Date of study creation
        study_time: Union[str, datetime.time], optional
           Time of study creation
        referring_physician_name: str, optional
            Name of the referring physician
        pixel_spacing: Tuple[int, int], optional
            Physical spacing in millimeter between pixels along the row and
            column dimension
        laterality: Union[str, highdicom.enum.LateralityValues], optional
            Laterality of the examined body part (required if
            `coordinate_system` is ``"PATIENT"``)
        patient_orientation:
                Union[Tuple[str, str], Tuple[highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped], Tuple[highdicom.enum.PatientOrientationValuesQuadruped, highdicom.enum.PatientOrientationValuesQuadruped]], optional
            Orientation of the patient along the row and column axes of the
            image (required if `coordinate_system` is ``"PATIENT"``)
        anatomical_orientation_type: Union[str, highdicom.enum.AnatomicalOrientationTypeValues], optional
            Type of anatomical orientation of patient relative to image (may be
            provide if `coordinate_system` is ``"PATIENT"`` and patient is
            an animal)
        container_identifier: str, optional
            Identifier of the container holding the specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        issuer_of_container_identifier: highdicom.content.IssuerOfIdentifier, optional
            Issuer of `container_identifier`
        specimen_descriptions: Sequence[highdicom.content.SpecimenDescriptions], optional
            Description of each examined specimen (required if
            `coordinate_system` is ``"SLIDE"``)
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa
        super().__init__(
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=SecondaryCaptureImageStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            modality='OT',
            transfer_syntax_uid=None,  # uncompressed!
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
            if laterality is None:
                raise TypeError(
                    'Laterality is required if coordinate system '
                    'is "PATIENT".'
                )
            if patient_orientation is None:
                raise TypeError(
                    'Patient orientation is required if coordinate system '
                    'is "PATIENT".'
                )

            # General Series
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
            container_type_item = CodedConcept(*codes.SCT.MicroscopeSlide)
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
        wrong_bit_depth_assignment = (
            pixel_array.dtype == np.bool and bits_allocated != 1,
            pixel_array.dtype == np.uint8 and bits_allocated != 8,
            pixel_array.dtype == np.uint16 and bits_allocated not in (12, 16),
        )
        if any(wrong_bit_depth_assignment):
            raise ValueError('Pixel array has an unexpected bit depth.')
        if bits_allocated not in (1, 8, 12, 16):
            raise ValueError('Unexpected number of bits allocated.')
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
            self.PixelSpacing = pixel_spacing
        self.PixelData = pixel_array.tobytes()
