import logging
from datetime import datetime

from pydicom.dataset import Dataset
from pydicom.datadict import keyword_for_tag
from pydicom.valuerep import DA, DT, TM


logger = logging.getLogger(__name__)


_PATIENT_ATTRIBUTES_TO_COPY = {
    # Patient
    '00080054', '00080100', '00080102', '00080103', '00080104', '00080105',
    '00080106', '00080107', '0008010B', '0008010D', '0008010F', '00080117',
    '00080118', '00080119', '00080120', '00080121', '00080122', '00081120',
    '00081150', '00081155', '00081160', '00081190', '00081199', '00100010',
    '00100020', '00100021', '00100022', '00100024', '00100026', '00100027',
    '00100028', '00100030', '00100032', '00100033', '00100034', '00100035',
    '00100040', '00100200', '00100212', '00100213', '00100214', '00100215',
    '00100216', '00100217', '00100218', '00100219', '00100221', '00100222',
    '00100223', '00100229', '00101001', '00101002', '00101100', '00102160',
    '00102201', '00102202', '00102292', '00102293', '00102294', '00102295',
    '00102296', '00102297', '00102298', '00102299', '00104000', '00120062',
    '00120063', '00120064', '0020000D', '00400031', '00400032', '00400033',
    '00400035', '00400036', '00400039', '0040003A', '0040E001', '0040E010',
    '0040E020', '0040E021', '0040E022', '0040E023', '0040E024', '0040E025',
    '0040E030', '0040E031', '0062000B', '00880130', '00880140',
    # Clinical Trial Subject
    '00120010', '00120020', '00120021', '00120030', '00120031', '00120040',
    '00120042', '00120081', '00120082',
}


_STUDY_ATTRIBUTES_TO_COPY = {
    # Patient Study
    '00080100', '00080102', '00080103', '00080104', '00080105', '00080106',
    '00080107', '0008010B', '0008010D', '0008010F', '00080117', '00080118',
    '00080119', '00080120', '00080121', '00080122', '00081080', '00081084',
    '00101010', '00101020', '00101021', '00101022', '00101023', '00101024',
    '00101030', '00102000', '00102110', '00102180', '001021A0', '001021B0',
    '001021C0', '001021D0', '00102203', '00380010', '00380014', '00380060',
    '00380062', '00380064', '00380500', '00400031', '00400032', '00400033',
    # General Study
    '00080020', '00080030', '00080050', '00080051', '00080080', '00080081',
    '00080082', '00080090', '00080096', '0008009C', '0008009D', '00080100',
    '00080102', '00080103', '00080104', '00080105', '00080106', '00080107',
    '0008010B', '0008010D', '0008010F', '00080117', '00080118', '00080119',
    '00080120', '00080121', '00080122', '00081030', '00081032', '00081048',
    '00081049', '00081060', '00081062', '00081110', '00081150', '00081155',
    '0020000D', '00200010', '00321034', '00400031', '00400032', '00400033',
    '00401012', '00401101', '00401102', '00401103', '00401104',
    # Clinical Trial Study
    '00120020', '00120050', '00120051', '00120052', '00120053', '00120083',
    '00120084', '00120085',
}


_SPECIMEN_ATTRIBUTES_TO_COPY = {
    '00400512', '00400513', '00400515', '00400518', '0040051A', '00400520',
    '00400560',
}


class SOPClass(Dataset):

    """Base class for a DICOM SOP Instance."""

    def __init__(self,
                 study_instance_uid: str,
                 series_instance_uid: str,
                 series_number: int,
                 sop_instance_uid: str,
                 sop_class_uid: str,
                 instance_number: int,
                 manufacturer: str,
                 modality: str,
                 transfer_syntax_uid: Optional[str] = None,
                 patient_id: Optional[str] = None,
                 patient_name: Optional[str] = None,
                 patient_birth_date: Optional[str] = None,
                 patient_sex: Optional[str] = None,
                 accession_number: Optional[str] = None,
                 study_id: str = None,
                 study_date: Optional[Union[str, datetime.date]] = None,
                 study_time: Optional[Union[str, datetime.time]] = None,
                 referring_physician_name: Optional[str] = None):
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

        Note
        ----
        The constructor only provides attributes that are required by the
        standard (type 1 and 2) as part of the Patient, General Study,
        Patient Study, General Series, General Equipment and SOP Common modules.
        Derived classes are responsible for providing additional attributes
        required by the corresponding Information Object Definition (IOD).
        Additional optional attributes can subsequently be added to the dataset.

        """
        if transfer_syntax_uid is None:
            transfer_syntax_uid = ImplicitVRLittleEndian
        self.is_implicit_VR = False
        self.is_little_endian = True
        self.preamble = b'\x00' * 128
        self.file_meta = Dataset()
        self.file_meta.TransferSyntaxUID = transfer_syntax_uid
        self.file_meta.SOPClassUID = str(sop_class_uid)
        self.file_meta.SOPInstanceUID = str(sop_instance_uid)
        self.file_meta.FileMetaInformationVersion = b'\x00\x01'
        self.fix_meta_info(enforce_standard=True)

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

        # Equipment
        self.Manufacturer = manufacturer

        # Instance
        self.SOPInstanceUID = str(sop_instance_uid)
        self.SOPClassUID = str(sop_class_uid)
        self.InstanceNumber = instance_number
        self.ContentDate = DA(now.date())
        self.ContentTime = TM(now.time())

    def _copy_attribute(self, dataset: Dataset, tag: str):
        """Copies an attribute from `dataset` to `self`.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attribute should be copied
        tag: str
            Tag of the attribute

        """
        keyword = keyword_for_tag(tag)
        try:
            data_element = dataset[tag]
            logger.debug('copied attribute "{}"'.format(keyword))
        except KeyError:
            continue
        self.add(data_element)

    def copy_patient_and_study_information(self, dataset: Dataset):
        """Copies patient- and study-related metadata from `dataset` that
        are defined in the following modules: Patient, General Study,
        Patient Study, Clinical Trial Subject and Clinical Trial Study.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attributes should be copied

        """
        logger.info(
            'copy patient-related attributes from dataset "{}"'.format(
                dataset.SOPInstanceUID
            )
        )
        [self._copy_attribute(tag) for tag in _PATIENT_ATTRIBUTES_TO_COPY]
        logger.info(
            'copy study-related attributes from dataset "{}"'.format(
                dataset.SOPInstanceUID
            )
        )
        [self._copy_attribute(tag) for tag in _STUDY_ATTRIBUTES_TO_COPY]

    def copy_specimen_information(self, dataset: Dataset):
        """Copies specimen-related metadata from `dataset` that
        are defined in the Specimen module.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            DICOM Data Set from which attributes should be copied

        """
        logger.info(
            'copy specimen-related attributes from dataset "{}"'.format(
                dataset.SOPInstanceUID
            )
        )
        [self._copy_attribute(tag) for tag in _SPECIMEN_ATTRIBUTES_TO_COPY]
