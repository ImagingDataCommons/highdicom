import logging
import datetime
from io import BytesIO
from collections.abc import Sequence

from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filewriter import write_file_meta_info
from pydicom.sr.codedict import codes
from pydicom.uid import ImplicitVRLittleEndian, UID
from pydicom.valuerep import DA, PersonName, TM

from highdicom.coding_schemes import CodingSchemeIdentificationItem
from highdicom.base_content import ContributingEquipment
from highdicom.enum import (
    ContentQualificationValues,
    PatientSexValues,
)
from highdicom.valuerep import check_person_name, _check_long_string
from highdicom.version import __version__
from highdicom._module_utils import is_attribute_in_iod


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
        manufacturer: str | None = None,
        transfer_syntax_uid: str | None = None,
        patient_id: str | None = None,
        patient_name: str | PersonName | None = None,
        patient_birth_date: str | None = None,
        patient_sex: str | PatientSexValues | None = None,
        accession_number: str | None = None,
        study_id: str | None = None,
        study_date: str | datetime.date | None = None,
        study_time: str | datetime.time | None = None,
        referring_physician_name: str | PersonName | None = None,
        content_qualification: None | (
            str | ContentQualificationValues
        ) = None,
        coding_schemes: None | (
            Sequence[CodingSchemeIdentificationItem]
        ) = None,
        series_description: str | None = None,
        manufacturer_model_name: str | None = None,
        software_versions: str | tuple[str] | None = None,
        device_serial_number: str | None = None,
        institution_name: str | None = None,
        institutional_department_name: str | None = None,
        content_date: str | datetime.date | None = None,
        content_time: str | datetime.time | None = None,
        series_date: str | datetime.date | None = None,
        series_time: str | datetime.time | None = None,
    ):
        """
        Parameters
        ----------
        study_instance_uid: str
            UID of the study
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        modality: str
            Name of the modality
        manufacturer: Union[str, None], optional
            Name of the manufacturer (developer) of the device (software)
            that creates the instance
        transfer_syntax_uid: Union[str, None], optional
            UID of transfer syntax that should be used for encoding of
            data elements. Defaults to Implicit VR Little Endian
            (UID ``"1.2.840.10008.1.2"``)
        patient_id: Union[str, None], optional
           ID of the patient (medical record number)
        patient_name: Union[str, pydicom.valuerep.PersonName, None], optional
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
        content_qualification: Union[str, highdicom.ContentQualificationValues, None], optional
            Indicator of content qualification
        coding_schemes: Union[Sequence[highdicom.sr.CodingSchemeIdentificationItem], None], optional
            private or public coding schemes that are not part of the
            DICOM standard
        series_description: Union[str, None], optional
            Human readable description of the series
        manufacturer_model_name: Union[str, None], optional
            Name of the device model (name of the software library or
            application) that creates the instance
        software_versions: Union[str, Tuple[str]]
            Version(s) of the software that creates the instance
        device_serial_number: str
            Manufacturer's serial number of the device
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance.
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance.
        content_date: str | datetime.date | None, optional
            Date the content of this instance was created. If not specified,
            the current date will be used.
        content_time: str | datetime.time | None, optional
            Time the content of this instance was created. If not specified,
            the current time will be used.
        series_date: str | datetime.date | None, optional
            Date the series was started. This should be the same for all
            instances in a series.
        series_time: str | datetime.time | None, optional
            Time the series was started. This should be the same for all
            instances in a series.

        Note
        ----
        The constructor only provides attributes that are required by the
        standard (type 1 and 2) as part of the Patient, General Study,
        Patient Study, General Series, General Equipment and SOP Common modules.
        Derived classes are responsible for providing additional attributes
        required by the corresponding Information Object Definition (IOD).
        Additional optional attributes can subsequently be added to the dataset.

        """  # noqa: E501
        super().__init__()
        if transfer_syntax_uid is None:
            transfer_syntax_uid = ImplicitVRLittleEndian
        transfer_syntax_uid = UID(transfer_syntax_uid)
        if not transfer_syntax_uid.is_little_endian:
            raise ValueError(
                "Big Endian transfer syntaxes are retired and no longer "
                "supported by highdicom."
            )

        # Include all File Meta Information required for writing SOP instance
        # to a file in PS3.10 format.
        self.preamble = b'\x00' * 128
        self.file_meta = FileMetaDataset()
        self.file_meta.TransferSyntaxUID = transfer_syntax_uid
        self.file_meta.MediaStorageSOPClassUID = UID(sop_class_uid)
        self.file_meta.MediaStorageSOPInstanceUID = UID(sop_instance_uid)
        self.file_meta.FileMetaInformationVersion = b'\x00\x01'
        self.file_meta.ImplementationClassUID = UID(
            '1.2.826.0.1.3680043.9.7433.1.1'
        )
        self.file_meta.ImplementationVersionName = f'highdicom{__version__}'
        with BytesIO() as fp:
            write_file_meta_info(fp, self.file_meta, enforce_standard=True)
            self.file_meta.FileMetaInformationGroupLength = len(fp.getvalue())

        # Patient
        self.PatientID = patient_id
        if patient_name is not None:
            try:
                check_person_name(patient_name)
            except ValueError:
                logger.warn(
                    'value of argument "patient_name" is potentially invalid: '
                    f'"{patient_name}"'
                )
        self.PatientName = patient_name
        self.PatientBirthDate = DA(patient_birth_date)
        if patient_sex is not None and patient_sex != '':
            patient_sex = PatientSexValues(patient_sex).value
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
        if series_number is None:
            raise TypeError('Argument "series_number" is required.')
        if series_number < 1:
            raise ValueError(
                'Argument "series_number" should be a positive integer.'
            )
        self.SeriesNumber = series_number
        self.Modality = modality
        if series_description is not None:
            _check_long_string(series_description)
            self.SeriesDescription = series_description

        # Equipment
        if manufacturer is not None:
            _check_long_string(manufacturer)
        self.Manufacturer = manufacturer
        if manufacturer_model_name is not None:
            _check_long_string(manufacturer_model_name)
            self.ManufacturerModelName = manufacturer_model_name
        if device_serial_number is not None:
            _check_long_string(device_serial_number)
            self.DeviceSerialNumber = device_serial_number
        if software_versions is not None:
            if not isinstance(software_versions, str):
                for v in software_versions:
                    _check_long_string(v)
            else:
                _check_long_string(software_versions)
            self.SoftwareVersions = software_versions
        if institution_name is not None:
            _check_long_string(institution_name)
            self.InstitutionName = institution_name
            if institutional_department_name is not None:
                _check_long_string(institutional_department_name)
                self.InstitutionalDepartmentName = institutional_department_name

        # Instance
        self.SOPInstanceUID = str(sop_instance_uid)
        self.SOPClassUID = str(sop_class_uid)
        if instance_number is None:
            raise TypeError('Argument "instance_number" is required.')
        if instance_number < 1:
            raise ValueError(
                'Argument "instance_number" should be a positive integer.'
            )
        self.InstanceNumber = instance_number

        now = datetime.datetime.now()
        self.InstanceCreationDate = DA(now.date())
        self.InstanceCreationTime = TM(now.time())

        # Content Date and Content Time are not present in all IODs
        if is_attribute_in_iod('ContentDate', sop_class_uid):
            if content_date is None:
                if content_time is not None:
                    raise TypeError(
                        "'content_time' may not be specified without "
                        "'content_date'."
                    )
                content_date = now.date()
            content_date = DA(content_date)
            self.ContentDate = content_date
        elif content_date is not None:
            raise TypeError(
                f"'content_date' should not be specified for SOP Class UID: "
                f"{sop_class_uid}"
            )
        if is_attribute_in_iod('ContentTime', sop_class_uid):
            if content_time is None:
                content_time = now.time()
            content_time = TM(content_time)
            self.ContentTime = content_time
        elif content_time is not None:
            raise TypeError(
                f"'content_time' should not be specified for SOP Class UID: "
                f"{sop_class_uid}"
            )

        if series_date is not None:
            series_date = DA(series_date)
            if content_date is not None:
                if series_date > content_date:
                    raise ValueError(
                        "'series_date' must not be later than 'content_date'."
                    )
            self.SeriesDate = series_date
        if series_time is not None:
            series_time = TM(series_time)
            if series_date is None:
                raise TypeError(
                    "'series_time' may not be specified without "
                    "'series_date'."
                )
            if content_time is not None:
                if series_time > content_time:
                    raise ValueError(
                        "'series_time' must not be later than content time."
                    )
            self.SeriesTime = series_time

        if content_qualification is not None:
            content_qualification = ContentQualificationValues(
                content_qualification
            )
            self.ContentQualification = content_qualification.value
        if coding_schemes is not None:
            self.CodingSchemeIdentificationSequence: list[Dataset] = []
            for item in coding_schemes:
                if not isinstance(item, CodingSchemeIdentificationItem):
                    raise TypeError(
                        'Coding scheme identification item must have type '
                        '"CodingSchemeIdentificationItem".'
                    )
                self.CodingSchemeIdentificationSequence.append(item)

    @property
    def transfer_syntax_uid(self) -> UID:
        """highdicom.UID: TransferSyntaxUID."""
        return UID(self.file_meta.TransferSyntaxUID)

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
        if tag is None:
            raise ValueError(f'No tag not found for keyword "{keyword}".')
        try:
            data_element = dataset[tag]
            logger.debug(f'copied attribute "{keyword}"')
        except KeyError:
            logger.debug(f'skipped attribute "{keyword}"')
            return
        self.add(data_element)

    def _copy_root_attributes_of_module(
        self,
        dataset: Dataset,
        ie: str,
        module: str | None = None
    ) -> None:
        """Copies all attributes at the root level of a given module from
        `dataset` to `self`.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attribute should be copied
        ie: str
            DICOM Information Entity (e.g., ``"Patient"`` or ``"Study"``)
        module: Union[str, None], optional
            DICOM Module (e.g., ``"General Series"`` or ``"Specimen"``)

        """
        from highdicom._iods import IOD_MODULE_MAP, SOP_CLASS_UID_IOD_KEY_MAP
        from highdicom._modules import MODULE_ATTRIBUTE_MAP
        logger.info(
            f'copy {ie}-related attributes from '
            f'dataset "{dataset.SOPInstanceUID}"'
        )
        iod_key = SOP_CLASS_UID_IOD_KEY_MAP[dataset.SOPClassUID]
        for module_item in IOD_MODULE_MAP[iod_key]:
            module_key = module_item['key']
            if module_item['ie'] != ie:
                continue
            if module is not None:
                module_key = module.replace(' ', '-').lower()
                if module_item['key'] != module_key:
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
        self._copy_root_attributes_of_module(dataset, 'Image', 'Specimen')

    def _add_contributing_equipment(
        self,
        contributing_equipment: Sequence[ContributingEquipment] | None = None,
        source_image: Dataset | None = None,
    ):
        """Add Contributing Equipment to this object.

        Parameters
        ----------
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        source_image: pydicom.Dataset | None, optional
            A source image whose acquisition equipment should be included as
            contributing equipment for this object.

        """  # noqa: E501
        # ContributingEquipment (SOP Common)
        if contributing_equipment is None:
            contributing_equipment = []
        else:
            contributing_equipment = list(contributing_equipment)

        for item in contributing_equipment:
            if not isinstance(item, ContributingEquipment):
                raise TypeError(
                    "Items of argument 'contributing_equipment' must be of "
                    'type highdicom.ContributingEquipment.'
                )

        if source_image is not None:
            try:
                acquisition_equipment = (
                    ContributingEquipment.for_image_acquisition(source_image)
                )
            except Exception:
                pass
            else:
                contributing_equipment = [
                    acquisition_equipment,
                    *list(contributing_equipment),
                ]

        equipment_code = None
        if self.SOPClassUID in (
            '1.2.840.10008.5.1.4.1.1.4.4',    # LegacyConvertedEnhancedMRImage
            '1.2.840.10008.5.1.4.1.1.2.2',    # LegacyConvertedEnhancedCTImage
            '1.2.840.10008.5.1.4.1.1.128.1',  # LegacyConvertedEnhancedPETImage
        ):
            equipment_code = codes.DCM.EnhancedMultiFrameConversionEquipment

        contributing_equipment.append(
            ContributingEquipment.for_highdicom(
                purpose_of_reference=equipment_code
            )
        )

        self.ContributingEquipmentSequence = contributing_equipment


def _check_little_endian(dataset: Dataset) -> None:
    """Assert that a dataset uses a little endian transfer syntax.

    Parameters
    ----------
    dataset: Dataset
        Dataset to check.

    Raises
    ------
    ValueError:
        If the dataset does not use a little endian transfer syntax.

    """
    if not hasattr(dataset, 'file_meta'):
        logger.warning(
            'Transfer syntax cannot be determined from the file metadata. '
            'Little endian encoding of attributes has been assumed.'
        )
        return
    if not dataset.file_meta.TransferSyntaxUID.is_little_endian:
        raise ValueError(
            'Parsing of datasets is only valid for datasets with little endian '
            'transfer syntaxes.'
        )
