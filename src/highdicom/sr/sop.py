"""Module for SOP Classes of Structured Report (SR) IODs."""

import collections
import datetime
import logging
from typing import Optional, Sequence, Union

from pydicom.sr.coding import Code
from pydicom.dataset import Dataset
from pydicom.uid import PYDICOM_IMPLEMENTATION_UID, ExplicitVRLittleEndian
from pydicom.valuerep import DA, DT, TM
from pydicom._storage_sopclass_uids import Comprehensive3DSRStorage

from highdicom.base import SOPClass
from highdicom.sr.coding import CodedConcept


logger = logging.getLogger(__name__)


class Comprehensive3DSR(SOPClass):

    """SOP class for a Comprehensive 3D Structured Report (SR) document, whose
    content may include textual and a variety of coded information, numeric
    measurement values, references to SOP Instances, as well as 2D or 3D
    spatial or temporal regions of interest within such SOP Instances.
    """

    def __init__(
            self,
            evidence: Sequence[Dataset],
            content: Dataset,
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            manufacturer: str,
            is_complete: bool = False,
            is_final: bool = False,
            is_verified: bool = False,
            institution_name: Optional[str] = None,
            institutional_department_name: Optional[str] = None,
            verifying_observer_name: Optional[str] = None,
            verifying_organization: Optional[str] = None,
            performed_procedure_codes: Optional[
                Sequence[Union[Code, CodedConcept]]
            ] = None,
            requested_procedures: Optional[Sequence[Dataset]] = None,
            previous_versions: Optional[Sequence[Dataset]] = None,
            record_evidence: bool = True,
            **kwargs
        ):
        """
        Parameters
        ----------
        evidence: Sequence[pydicom.dataset.Dataset]
            Instances that are referenced in the content tree and from which
            the created SR document instance should inherit patient and study
            information
        content: pydicom.dataset.Dataset
            Root container content items that should be included in the
            SR document
        series_instance_uid: str
            Series Instance UID of the SR document series
        series_number: Union[int, None]
            Series Number of the SR document series
        sop_instance_uid: str
            SOP instance UID that should be assigned to the SR document instance
        instance_number: int
            Number that should be assigned to this SR document instance
        manufacturer: str
            Name of the manufacturer of the device that creates the SR document
            instance (in a research setting this is typically the same
            as `institution_name`)
        is_complete: bool, optional
            Whether the content is complete (default: ``False``)
        is_final: bool, optional
            Whether the report is the definitive means of communicating the
            findings (default: ``False``)
        is_verified: bool, optional
            Whether the report has been verified by an observer accountable
            for its content (default: ``False``)
        institution_name: str, optional
            Name of the institution of the person or device that creates the
            SR document instance
        institutional_department_name: str, optional
            Name of the department of the person or device that creates the
            SR document instance
        verifying_observer_name: Union[str, None], optional
            Name of the person that verfied the SR document
            (required if `is_verified`)
        verifying_organization: str
            Name of the organization that verfied the SR document
            (required if `is_verified`)
        performed_procedure_codes: List[pydicom.sr.coding.CodedConcept]
            Codes of the performed procedures that resulted in the SR document
        requested_procedures: List[pydicom.dataset.Dataset]
            Requested procedures that are being fullfilled by creation of the
            SR document
        previous_versions: List[pydicom.dataset.Dataset]
            Instances representing previous versions of the SR document
        record_evidence: bool, optional
            Whether provided `evidence` should be recorded, i.e. included
            in Current Requested Procedure Evidence Sequence or Pertinent
            Other Evidence Sequence (default: ``True``)
        **kwargs: Dict[str, Any], optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Note
        ----
        Each dataset in `evidence` must be part of the same study.

        """
        super().__init__(
            study_instance_uid=evidence[0].StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=Comprehensive3DSRStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            modality='SR',
            transfer_syntax_uid=None,
            patient_id=evidence[0].PatientID,
            patient_name=evidence[0].PatientName,
            patient_birth_date=evidence[0].PatientBirthDate,
            patient_sex=evidence[0].PatientSex,
            accession_number=evidence[0].AccessionNumber,
            study_id=evidence[0].StudyID,
            study_date=evidence[0].StudyDate,
            study_time=evidence[0].StudyTime,
            referring_physician_name=evidence[0].ReferringPhysicianName,
            **kwargs
        )

        if institution_name is not None:
            self.InstitutionName = institution_name
            if institutional_department_name is not None:
                self.InstitutionalDepartmentName = institutional_department_name

        now = datetime.datetime.now()
        if is_complete:
            self.CompletionFlag = 'COMPLETE'
        else:
            self.CompletionFlag = 'PARTIAL'
        if is_verified:
            if verifying_observer_name is None:
                raise ValueError(
                    'Verifying Observer Name must be specified if SR document '
                    'has been verified.'
                )
            if verifying_organization is None:
                raise ValueError(
                    'Verifying Organization must be specified if SR document '
                    'has been verified.'
                )
            self.VerificationFlag = 'VERIFIED'
            observer_item = Dataset()
            observer_item.VerifyingObserverName = verifying_observer_name
            observer_item.VerifyingOrganization = verifying_organization
            observer_item.VerificationDateTime = DT(now)
            self.VerifyingObserverSequence = [observer_item]
        else:
            self.VerificationFlag = 'UNVERIFIED'
        if is_final:
            self.PreliminaryFlag = 'FINAL'
        else:
            self.PreliminaryFlag = 'PRELIMINARY'

        # Add content to dataset
        for tag, value in content.items():
            self[tag] = value

        evd_collection = collections.defaultdict(list)
        for evd in evidence:
            if evd.StudyInstanceUID != evidence[0].StudyInstanceUID:
                raise ValueError(
                    'Referenced data sets must all belong to the same study.'
                )
            evd_instance_item = Dataset()
            evd_instance_item.ReferencedSOPClassUID = evd.SOPClassUID
            evd_instance_item.ReferencedSOPInstanceUID = evd.SOPInstanceUID
            evd_collection[evd.SeriesInstanceUID].append(
                evd_instance_item
            )
        evd_study_item = Dataset()
        evd_study_item.StudyInstanceUID = evidence[0].StudyInstanceUID
        evd_study_item.ReferencedSeriesSequence = []
        for evd_series_uid, evd_instance_items in evd_collection.items():
            evd_series_item = Dataset()
            evd_series_item.SeriesInstanceUID = evd_series_uid
            evd_series_item.ReferencedSOPSequence = evd_instance_items
            evd_study_item.ReferencedSeriesSequence.append(evd_series_item)
        if requested_procedures is not None:
            self.ReferencedRequestSequence = requested_procedures
            self.CurrentRequestedProcedureEvidenceSequence = [evd_study_item]
        else:
            if record_evidence:
                self.PertinentOtherEvidenceSequence = [evd_study_item]

        if previous_versions is not None:
            pre_collection = collections.defaultdict(list)
            for pre in previous_versions:
                if pre.StudyInstanceUID != evidence[0].StudyInstanceUID:
                    raise ValueError(
                        'Previous version data sets must belong to the '
                        'same study.'
                    )
                pre_instance_item = Dataset()
                pre_instance_item.ReferencedSOPClassUID = pre.SOPClassUID
                pre_instance_item.ReferencedSOPInstanceUID = pre.SOPInstanceUID
                pre_collection[pre.SeriesInstanceUID].append(
                    pre_instance_item
                )
            pre_study_item = Dataset()
            pre_study_item.StudyInstanceUID = pre.StudyInstanceUID
            pre_study_item.ReferencedSeriesSequence = []
            for pre_series_uid, pre_instance_items in pre_collection.items():
                pre_series_item = Dataset()
                pre_series_item.SeriesInstanceUID = pre_series_uid
                pre_series_item.ReferencedSOPSequence = pre_instance_items
                pre_study_item.ReferencedSeriesSequence.append(pre_series_item)
            self.PredecessorDocumentsSequence = [pre_study_item]

        if performed_procedure_codes is not None:
            self.PerformedProcedureCodeSequence = performed_procedure_codes
        else:
            self.PerformedProcedureCodeSequence = []

        # TODO
        self.ReferencedPerformedProcedureStepSequence = []

        self.copy_patient_and_study_information(evidence[0])
