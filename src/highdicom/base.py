import logging
import datetime
from io import BytesIO
from typing import List, Optional, Sequence, Union

from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.filewriter import write_file_meta_info
from pydicom.uid import (
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    UID_dictionary,
)
from pydicom.valuerep import DA, DT, TM

from highdicom.sr.coding import CodingSchemeIdentificationItem
from highdicom.enum import ContentQualificationValues
from highdicom.version import __version__
from highdicom._iods import IOD_MODULE_MAP
from highdicom._modules import MODULE_ATTRIBUTE_MAP


logger = logging.getLogger(__name__)


class SOPClass(Dataset):

    """Base class for DICOM SOP Instances."""

    def __init__(
            self,
            study_instance_uid: str,
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            sop_class_uid: str,
            instance_number: int,
            modality: str,
            manufacturer: Optional[str] = None,
            transfer_syntax_uid: Optional[str] = None,
            patient_id: Optional[str] = None,
            patient_name: Optional[str] = None,
            patient_birth_date: Optional[str] = None,
            patient_sex: Optional[str] = None,
            accession_number: Optional[str] = None,
            study_id: str = None,
            study_date: Optional[Union[str, datetime.date]] = None,
            study_time: Optional[Union[str, datetime.time]] = None,
            referring_physician_name: Optional[str] = None,
            content_qualification: Optional[
                Union[str, ContentQualificationValues]
            ] = None,
            coding_schemes: Optional[
                Sequence[CodingSchemeIdentificationItem]
            ] = None,
            series_description: Optional[str] = None
        ):
        """
        Parameters
        ----------
        study_instance_uid: str
            UID of the study
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
        modality: str
            Name of the modality
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. Defaults to Implicit VR Little Endian
            (UID ``"1.2.840.10008.1.2"``)
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
        content_qualification: Union[str, highdicom.enum.ContentQualificationValues], optional
            Indicator of content qualification
        coding_schemes: Sequence[highdicom.sr.coding.CodingSchemeIdentificationItem], optional
            private or public coding schemes that are not part of the
            DICOM standard
        series_description: str, optional
            Human readable description of the series

        Note
        ----
        The constructor only provides attributes that are required by the
        standard (type 1 and 2) as part of the Patient, General Study,
        Patient Study, General Series, General Equipment and SOP Common modules.
        Derived classes are responsible for providing additional attributes
        required by the corresponding Information Object Definition (IOD).
        Additional optional attributes can subsequently be added to the dataset.

        """  # noqa
        super().__init__()
        if transfer_syntax_uid is None:
            transfer_syntax_uid = ImplicitVRLittleEndian
        if transfer_syntax_uid == ExplicitVRBigEndian:
            self.is_little_endian = False
        else:
            self.is_little_endian = True
        if transfer_syntax_uid == ImplicitVRLittleEndian:
            self.is_implicit_VR = True
        else:
            self.is_implicit_VR = False

        # Include all File Meta Information required for writing SOP instance
        # to a file in PS3.10 format.
        self.preamble = b'\x00' * 128
        self.file_meta = Dataset()
        self.file_meta.DICOMPrefix = 'DICM'
        self.file_meta.FilePreamble = self.preamble
        self.file_meta.TransferSyntaxUID = transfer_syntax_uid
        self.file_meta.MediaStorageSOPClassUID = str(sop_class_uid)
        self.file_meta.MediaStorageSOPInstanceUID = str(sop_instance_uid)
        self.file_meta.FileMetaInformationVersion = b'\x00\x01'
        self.file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.9.7433.1.1'
        self.file_meta.ImplementationVersionName = '{} v{}'.format(
            __name__.split('.')[0], __version__
        )
        self.fix_meta_info(enforce_standard=True)
        with BytesIO() as fp:
            write_file_meta_info(fp, self.file_meta, enforce_standard=True)
            self.file_meta.FileMetaInformationGroupLength = len(fp.getvalue())

        # Patient
        self.PatientID = patient_id
        self.PatientName = patient_name
        self.PatientBirthDate = patient_birth_date
        self.PatientSex = patient_sex

        # Study
        self.StudyInstanceUID = str(study_instance_uid)
        self.AccessionNumber = accession_number
        self.StudyID = study_id
        self.StudyDate = DA(study_date) if study_date is not None else None
        self.StudyTime = TM(study_time) if study_time is not None else None
        self.ReferringPhysicianName = referring_physician_name

        # Series
        self.SeriesInstanceUID = str(series_instance_uid)
        self.SeriesNumber = series_number
        self.Modality = modality
        if series_description is not None:
            self.SeriesDescription = series_description

        # Equipment
        self.Manufacturer = manufacturer

        # Instance
        self.SOPInstanceUID = str(sop_instance_uid)
        self.SOPClassUID = str(sop_class_uid)
        self.InstanceNumber = instance_number
        self.ContentDate = DA(datetime.datetime.now().date())
        self.ContentTime = TM(datetime.datetime.now().time())
        if content_qualification is not None:
            content_qualification = ContentQualificationValues(
                content_qualification
            )
            self.ContentQualification = content_qualification.value
        if coding_schemes is not None:
            self.CodingSchemeIdentificationSequence: List[Dataset] = []
            for item in coding_schemes:
                if not isinstance(item, CodingSchemeIdentificationItem):
                    raise TypeError(
                        'Coding scheme identification item must have type '
                        '"CodingSchemeIdentificationItem".'
                    )
                self.CodingSchemeIdentificationSequence.append(item)

    def _copy_attribute(
            self,
            dataset: Dataset,
            keyword: str
    ) -> None:
        """Copies an attribute from `dataset` to `self`.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attribute should be copied
        keyword: str
            Keyword of the attribute

        """
        tag = tag_for_keyword(keyword)
        try:
            data_element = dataset[tag]
            logger.debug('copied attribute "{}"'.format(keyword))
        except KeyError:
            logger.debug('skipped attribute "{}"'.format(keyword))
            return
        self.add(data_element)

    def _copy_root_attributes_of_module(
            self,
            dataset: Dataset,
            ie: str
    ) -> None:
        """Copies all attributes at the root level of a given module from
        `dataset` to `self`.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attribute should be copied
        ie: str
            DICOM Information Entity (IE)

        """
        logger.info(
            'copy {}-related attributes from dataset "{}"'.format(
                ie, dataset.SOPInstanceUID
            )
        )
        try:
            sop_class_name = UID_dictionary[dataset.SOPClassUID][0]
        except KeyError:
            raise ValueError(
                'Could not determine IOD for SOP Class UID "{}".'.format(
                    dataset.SOPClassUID
                )
            )
        iod_key = sop_class_name.replace(' Storage', '').replace(' ', '-')
        iod_key = iod_key.lower()
        for module_item in IOD_MODULE_MAP[iod_key]:
            module_key = module_item['key']
            if module_item['ie'] != ie:
                continue
            logger.info(
                'copy attributes of module "{}"'.format(
                    ' '.join([
                        name.capitalize()
                        for name in module_key.split('-')
                    ])
                )
            )
            for item in MODULE_ATTRIBUTE_MAP[module_key]:
                if len(item['path']) == 0:
                    self._copy_attribute(dataset, str(item['keyword']))

    def copy_patient_and_study_information(self, dataset: Dataset) -> None:
        """Copies patient- and study-related metadata from `dataset` that
        are defined in the following modules: Patient, General Study,
        Patient Study, Clinical Trial Subject and Clinical Trial Study.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attributes should be copied

        """
        self._copy_root_attributes_of_module(dataset, 'Patient')
        self._copy_root_attributes_of_module(dataset, 'Study')

    def copy_specimen_information(self, dataset: Dataset) -> None:
        """Copies specimen-related metadata from `dataset` that
        are defined in the Specimen module.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attributes should be copied

        """
        self._copy_root_attributes_of_module(dataset, 'Specimen')

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'SOPClass':
        try:
            inst = cls(
                study_instance_uid=dataset.StudyInstanceUID,
                series_instance_uid=dataset.SeriesInstanceUID,
                series_number=dataset.SeriesNumber,
                sop_instance_uid=dataset.SOPInstanceUID,
                sop_class_uid=dataset.SOPClassUID,
                instance_number=dataset.InstanceNumber,
                manufacturer=dataset.Manufacturer,
                modality=dataset.Modality,
                transfer_syntax_uid=dataset.file_meta.TransferSyntaxUID,
                patient_id=dataset.PatientID,
                patient_name=dataset.PatientName,
                patient_birth_date=dataset.PatientBirthDate,
                patient_sex=dataset.PatientSex,
                accession_number=dataset.AccessionNumber,
                study_id=dataset.StudyID,
                study_date=dataset.StudyDate,
                study_time=dataset.StudyTime,
                referring_physician_name=dataset.ReferringPhysicianName
            )
        except AttributeError as error:
            raise AttributeError(
                'Required attribute missing: {}'.format(error)
            )
        if inst.SOPClassUID != dataset.SOPClassUID:
            raise AttributeError(
                'Incorrect SOP Class UID for type "{}".'.format(cls.__name__)
            )
        return inst
