"""Module for SOP Classes of Structured Report (SR) IODs."""
import datetime
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, cast, Mapping, List, Optional, Sequence, Tuple, Union

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    UID,
)
from pydicom.valuerep import DT, PersonName
from pydicom.uid import (
    ComprehensiveSRStorage,
    Comprehensive3DSRStorage,
    EnhancedSRStorage,
)

from highdicom.base import SOPClass, _check_little_endian
from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import ValueTypeValues
from highdicom.sr.templates import MeasurementReport
from highdicom.sr.utils import (
    find_content_items,
    collect_evidence,
    _create_references
)
from highdicom.sr.value_types import ContentItem, ContentSequence
from highdicom.valuerep import check_person_name


logger = logging.getLogger(__name__)


class _SR(SOPClass):

    """Abstract base class for Structured Report (SR) SOP classes."""

    def __init__(
        self,
        evidence: Sequence[Dataset],
        content: Dataset,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        sop_class_uid: str,
        instance_number: int,
        manufacturer: Optional[str] = None,
        is_complete: bool = False,
        is_final: bool = False,
        is_verified: bool = False,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        verifying_observer_name: Optional[Union[str, PersonName]] = None,
        verifying_organization: Optional[str] = None,
        performed_procedure_codes: Optional[
            Sequence[Union[Code, CodedConcept]]
        ] = None,
        requested_procedures: Optional[Sequence[Dataset]] = None,
        previous_versions: Optional[Sequence[Dataset]] = None,
        record_evidence: bool = True,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs: Any
    ) -> None:
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
        series_number: int
            Series Number of the SR document series
        sop_instance_uid: str
            SOP Instance UID that should be assigned to the SR document instance
        sop_class_uid: str
            SOP Class UID for the SR document type
        instance_number: int
            Number that should be assigned to this SR document instance
        manufacturer: str, optional
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
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance
        verifying_observer_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person that verified the SR document
            (required if `is_verified`)
        verifying_organization: Union[str, None], optional
            Name of the organization that verified the SR document
            (required if `is_verified`)
        performed_procedure_codes: Union[List[highdicom.sr.CodedConcept], None], optional
            Codes of the performed procedures that resulted in the SR document
        requested_procedures: Union[List[pydicom.dataset.Dataset], None], optional
            Requested procedures that are being fullfilled by creation of the
            SR document
        previous_versions: Union[List[pydicom.dataset.Dataset], None], optional
            Instances representing previous versions of the SR document
        record_evidence: bool, optional
            Whether provided `evidence` should be recorded (i.e. included
            in Pertinent Other Evidence Sequence) even if not referenced by
            content items in the document tree (default: ``True``)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Raises
        ------
        ValueError
            When no `evidence` is provided

        """  # noqa: E501
        if len(evidence) == 0:
            raise ValueError('No evidence was provided.')

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported.'
            )

        super().__init__(
            study_instance_uid=evidence[0].StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=manufacturer,
            modality='SR',
            transfer_syntax_uid=transfer_syntax_uid,
            patient_id=evidence[0].PatientID,
            patient_name=evidence[0].PatientName,
            patient_birth_date=evidence[0].PatientBirthDate,
            patient_sex=evidence[0].PatientSex,
            accession_number=evidence[0].AccessionNumber,
            study_id=evidence[0].StudyID,
            study_date=evidence[0].StudyDate,
            study_time=evidence[0].StudyTime,
            referring_physician_name=getattr(
                evidence[0], 'ReferringPhysicianName', None
            ),
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
            check_person_name(verifying_observer_name)
            observer_item.VerifyingObserverName = verifying_observer_name
            observer_item.VerifyingOrganization = verifying_organization
            observer_item.VerificationDateTime = DT(now)

            #  Type 2 attribute - we will leave empty
            observer_item.VerifyingObserverIdentificationCodeSequence = []

            self.VerifyingObserverSequence = [observer_item]
        else:
            self.VerificationFlag = 'UNVERIFIED'
        if is_final:
            self.PreliminaryFlag = 'FINAL'
        else:
            self.PreliminaryFlag = 'PRELIMINARY'

        # Add content to dataset
        content_copy = deepcopy(content)
        content_item = ContentItem._from_dataset_derived(content_copy)
        self._content = ContentSequence([content_item], is_root=True)
        for tag, value in content.items():
            self[tag] = value

        ref_items, unref_items = collect_evidence(evidence, content)
        if len(ref_items) > 0:
            self.CurrentRequestedProcedureEvidenceSequence = ref_items
        if len(unref_items) > 0 and record_evidence:
            self.PertinentOtherEvidenceSequence = unref_items

        if requested_procedures is not None:
            self.ReferencedRequestSequence = requested_procedures

        if previous_versions is not None:
            pre_items = self._collect_predecessors(previous_versions)
            self.PredecessorDocumentsSequence = pre_items

        if performed_procedure_codes is not None:
            self.PerformedProcedureCodeSequence = [
                CodedConcept.from_code(c) for c in performed_procedure_codes
            ]
        else:
            self.PerformedProcedureCodeSequence = []

        # TODO: unclear how this would work
        self.ReferencedPerformedProcedureStepSequence: List[Dataset] = []

        self.copy_patient_and_study_information(evidence[0])

    def _collect_predecessors(
        self,
        previous_versions: Sequence[Dataset]
    ) -> List[Dataset]:
        """Collect predecessors of the SR document.

        Parameters
        ----------
        previous_versions: List[pydicom.dataset.Dataset]
            Metadata of instances that represent previous versions of the
            SR document content

        Returns
        -------
        List[pydicom.dataset.Dataset]
            Items of the Predecessor Documents Sequence

        """
        group: Mapping[Tuple[str, str], List[Dataset]] = defaultdict(list)
        for pre in previous_versions:
            pre_instance_item = Dataset()
            pre_instance_item.ReferencedSOPClassUID = pre.SOPClassUID
            pre_instance_item.ReferencedSOPInstanceUID = pre.SOPInstanceUID
            key = (pre.StudyInstanceUID, pre.SeriesInstanceUID)
            group[key].append(pre_instance_item)
        return _create_references(group)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> Dataset:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Comprehensive SR document

        Returns
        -------
        highdicom.sr.sop._SR
            SR document

        """
        if not hasattr(dataset, 'ContentSequence'):
            raise ValueError('Dataset is not an SR document.')
        _check_little_endian(dataset)
        sop_instance = deepcopy(dataset)
        sop_instance.__class__ = cls

        root_item = Dataset()
        root_item.ConceptNameCodeSequence = dataset.ConceptNameCodeSequence
        root_item.ContentSequence = dataset.ContentSequence
        root_item.ValueType = dataset.ValueType
        root_item.ContinuityOfContent = dataset.ContinuityOfContent
        try:
            root_item.ContentTemplateSequence = dataset.ContentTemplateSequence
            tid_item = dataset.ContentTemplateSequence[0]
            if tid_item.TemplateIdentifier == '1500':
                sop_instance._content = MeasurementReport.from_sequence(
                    [root_item]
                )
            else:
                sop_instance._content = ContentSequence.from_sequence(
                    [root_item], is_root=True
                )
        except AttributeError:
            sop_instance._content = ContentSequence.from_sequence(
                [root_item], is_root=True
            )

        return sop_instance

    @property
    def content(self) -> ContentSequence:
        """highdicom.sr.value_types.ContentSequence: SR document content"""
        return self._content


class EnhancedSR(_SR):

    """SOP class for an Enhanced Structured Report (SR) document, whose
    content may include textual and a minimal amount of coded information,
    numeric measurement values, references to SOP Instances (retricted to the
    leaves of the tree), as well as 2D spatial or temporal regions of interest
    within such SOP Instances.
    """

    def __init__(
        self,
        evidence: Sequence[Dataset],
        content: Dataset,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: Optional[str] = None,
        is_complete: bool = False,
        is_final: bool = False,
        is_verified: bool = False,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        verifying_observer_name: Optional[Union[str, PersonName]] = None,
        verifying_organization: Optional[str] = None,
        performed_procedure_codes: Optional[
            Sequence[Union[Code, CodedConcept]]
        ] = None,
        requested_procedures: Optional[Sequence[Dataset]] = None,
        previous_versions: Optional[Sequence[Dataset]] = None,
        record_evidence: bool = True,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs: Any
    ) -> None:
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
        series_number: int
            Series Number of the SR document series
        sop_instance_uid: str
            SOP Instance UID that should be assigned to the SR document instance
        instance_number: int
            Number that should be assigned to this SR document instance
        manufacturer: str, optional
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
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance
        verifying_observer_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person that verified the SR document
            (required if `is_verified`)
        verifying_organization: Union[str, None], optional
            Name of the organization that verified the SR document
            (required if `is_verified`)
        performed_procedure_codes: Union[List[highdicom.sr.CodedConcept], None], optional
            Codes of the performed procedures that resulted in the SR document
        requested_procedures: Union[List[pydicom.dataset.Dataset], None], optional
            Requested procedures that are being fullfilled by creation of the
            SR document
        previous_versions: Union[List[pydicom.dataset.Dataset], None], optional
            Instances representing previous versions of the SR document
        record_evidence: bool, optional
            Whether provided `evidence` should be recorded (i.e. included
            in Pertinent Other Evidence Sequence) even if not referenced by
            content items in the document tree (default: ``True``)
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Note
        ----
        Each dataset in `evidence` must be part of the same study.

        """  # noqa: E501
        super().__init__(
            evidence=evidence,
            content=content,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=EnhancedSRStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            is_complete=is_complete,
            is_final=is_final,
            is_verified=is_verified,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            verifying_observer_name=verifying_observer_name,
            verifying_organization=verifying_organization,
            performed_procedure_codes=performed_procedure_codes,
            requested_procedures=requested_procedures,
            previous_versions=previous_versions,
            record_evidence=record_evidence,
            transfer_syntax_uid=transfer_syntax_uid,
            **kwargs
        )
        unsopported_content = find_content_items(
            content,
            value_type=ValueTypeValues.SCOORD3D,
            recursive=True
        )
        if len(unsopported_content) > 0:
            raise ValueError(
                'Enhanced SR does not support content items with '
                'SCOORD3D value type.'
            )


class ComprehensiveSR(_SR):

    """SOP class for a Comprehensive Structured Report (SR) document, whose
    content may include textual and a variety of coded information, numeric
    measurement values, references to SOP Instances, as well as 2D
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
        manufacturer: Optional[str] = None,
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
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs: Any
    ) -> None:
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
        series_number: int
            Series Number of the SR document series
        sop_instance_uid: str
            SOP Instance UID that should be assigned to the SR document instance
        instance_number: int
            Number that should be assigned to this SR document instance
        manufacturer: str, optional
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
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance
        verifying_observer_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person that verified the SR document
            (required if `is_verified`)
        verifying_organization: Union[str, None], optional
            Name of the organization that verified the SR document
            (required if `is_verified`)
        performed_procedure_codes: Union[List[highdicom.sr.CodedConcept], None], optional
            Codes of the performed procedures that resulted in the SR document
        requested_procedures: Union[List[pydicom.dataset.Dataset], None], optional
            Requested procedures that are being fullfilled by creation of the
            SR document
        previous_versions: Union[List[pydicom.dataset.Dataset], None], optional
            Instances representing previous versions of the SR document
        record_evidence: bool, optional
            Whether provided `evidence` should be recorded (i.e. included
            in Pertinent Other Evidence Sequence) even if not referenced by
            content items in the document tree (default: ``True``)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Note
        ----
        Each dataset in `evidence` must be part of the same study.

        """  # noqa: E501
        super().__init__(
            evidence=evidence,
            content=content,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=ComprehensiveSRStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            is_complete=is_complete,
            is_final=is_final,
            is_verified=is_verified,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            verifying_observer_name=verifying_observer_name,
            verifying_organization=verifying_organization,
            performed_procedure_codes=performed_procedure_codes,
            requested_procedures=requested_procedures,
            previous_versions=previous_versions,
            record_evidence=record_evidence,
            transfer_syntax_uid=transfer_syntax_uid,
            **kwargs
        )
        unsupported_content = find_content_items(
            content,
            value_type=ValueTypeValues.SCOORD3D,
            recursive=True
        )
        if len(unsupported_content) > 0:
            raise ValueError(
                'Comprehensive SR does not support content items with '
                'SCOORD3D value type.'
            )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ComprehensiveSR':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Comprehensive SR document

        Returns
        -------
        highdicom.sr.ComprehensiveSR
            Comprehensive SR document

        """
        if dataset.SOPClassUID != ComprehensiveSRStorage:
            raise ValueError('Dataset is not a Comprehensive SR document.')
        sop_instance = super().from_dataset(dataset)
        sop_instance.__class__ = ComprehensiveSR
        return cast(ComprehensiveSR, sop_instance)


class Comprehensive3DSR(_SR):

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
        manufacturer: Optional[str] = None,
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
        **kwargs: Any
    ) -> None:
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
        series_number: int
            Series Number of the SR document series
        sop_instance_uid: str
            SOP instance UID that should be assigned to the SR document instance
        instance_number: int
            Number that should be assigned to this SR document instance
        manufacturer: str, optional
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
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance
        verifying_observer_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person that verified the SR document
            (required if `is_verified`)
        verifying_organization: Union[str, None], optional
            Name of the organization that verified the SR document
            (required if `is_verified`)
        performed_procedure_codes: Union[List[highdicom.sr.CodedConcept], None], optional
            Codes of the performed procedures that resulted in the SR document
        requested_procedures: Union[List[pydicom.dataset.Dataset], None], optional
            Requested procedures that are being fullfilled by creation of the
            SR document
        previous_versions: Union[List[pydicom.dataset.Dataset], None], optional
            Instances representing previous versions of the SR document
        record_evidence: bool, optional
            Whether provided `evidence` should be recorded (i.e. included
            in Pertinent Other Evidence Sequence) even if not referenced by
            content items in the document tree (default: ``True``)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Note
        ----
        Each dataset in `evidence` must be part of the same study.

        """  # noqa: E501
        super().__init__(
            evidence=evidence,
            content=content,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=Comprehensive3DSRStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            is_complete=is_complete,
            is_final=is_final,
            is_verified=is_verified,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            verifying_observer_name=verifying_observer_name,
            verifying_organization=verifying_organization,
            performed_procedure_codes=performed_procedure_codes,
            requested_procedures=requested_procedures,
            previous_versions=previous_versions,
            record_evidence=record_evidence,
            **kwargs
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'Comprehensive3DSR':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Comprehensive 3D SR document

        Returns
        -------
        highdicom.sr.Comprehensive3DSR
            Comprehensive 3D SR document

        """
        if dataset.SOPClassUID != Comprehensive3DSRStorage:
            raise ValueError('Dataset is not a Comprehensive 3D SR document.')
        sop_instance = super().from_dataset(dataset)
        sop_instance.__class__ = Comprehensive3DSR
        return cast(Comprehensive3DSR, sop_instance)
