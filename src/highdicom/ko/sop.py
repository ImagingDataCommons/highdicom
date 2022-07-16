"""Module for SOP Classes of Key Object (KO) IODs."""
import logging
from typing import Any, cast, List, Optional, Sequence, Tuple, Union
from copy import deepcopy

from pydicom.dataset import Dataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    UID,
)
from pydicom.uid import (
    KeyObjectSelectionDocumentStorage,
)

from highdicom.base import SOPClass, _check_little_endian
from highdicom.sr.utils import collect_evidence
from highdicom.sr.value_types import ContainerContentItem
from highdicom.ko.content import KeyObjectSelection

logger = logging.getLogger(__name__)


class KeyObjectSelectionDocument(SOPClass):

    """Key Object Selection Document SOP class."""

    def __init__(
        self,
        evidence: Sequence[Dataset],
        content: KeyObjectSelection,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: Optional[str] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        requested_procedures: Optional[Sequence[Dataset]] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        evidence: Sequence[pydicom.dataset.Dataset]
            Instances that are referenced in the content tree and from which
            the created KO document instance should inherit patient and study
            information
        content: highdicom.ko.KeyObjectSelection
            Content items that should be included in the document
        series_instance_uid: str
            Series Instance UID of the document series
        series_number: int
            Series Number of the document series
        sop_instance_uid: str
            SOP Instance UID that should be assigned to the document instance
        instance_number: int
            Number that should be assigned to this document instance
        manufacturer: str, optional
            Name of the manufacturer of the device that creates the document
            instance (in a research setting this is typically the same
            as `institution_name`)
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            document instance
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            document instance
        requested_procedures: Union[Sequence[pydicom.dataset.Dataset], None], optional
            Requested procedures that are being fullfilled by creation of the
            document
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
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.59',
            instance_number=instance_number,
            manufacturer=manufacturer,
            modality='KO',
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

        # Add content to dataset
        for tag, value in content[0].items():
            self[tag] = value

        ref_items, unref_items = collect_evidence(evidence, content[0])
        if len(ref_items) > 0:
            self.CurrentRequestedProcedureEvidenceSequence = ref_items
            if len(ref_items) > 1:
                raise ValueError(
                    'Key Object Selection Documents that reference instances '
                    'from multiple studies are not supported.'
                )

        if requested_procedures is not None:
            self.ReferencedRequestSequence = requested_procedures

        self.ReferencedPerformedProcedureStepSequence: List[Dataset] = []

        # Cache copy of the content to facilitate subsequent access
        self._content = KeyObjectSelection.from_sequence(content, is_root=True)

        self._reference_lut = {}
        for study_item in self.CurrentRequestedProcedureEvidenceSequence:
            for series_item in study_item.ReferencedSeriesSequence:
                for instance_item in series_item.ReferencedSOPSequence:
                    sop_instance_uid = instance_item.ReferencedSOPInstanceUID
                    self._reference_lut[sop_instance_uid] = (
                        study_item.StudyInstanceUID,
                        series_item.SeriesInstanceUID,
                        sop_instance_uid
                    )

    @property
    def content(self) -> KeyObjectSelection:
        """highdicom.ko.KeyObjectSelection: document content"""
        return self._content

    def resolve_reference(self, sop_instance_uid: str) -> Tuple[str, str, str]:
        """Resolve reference for an object included in the document content.

        Parameters
        ----------
        sop_instance_uid: str
            SOP Instance UID of a referenced object

        Returns
        -------
        Tuple[str, str, str]
            Study, Series, and SOP Instance UID

        """
        try:
            return self._reference_lut[sop_instance_uid]
        except KeyError:
            raise ValueError(
                'Could not find any evidence for SOP Instance UID '
                f'"{sop_instance_uid}" in KOS document.'
            )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'KeyObjectSelectionDocument':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Key Object Selection Document

        Returns
        -------
        highdicom.ko.KeyObjectSelectionDocument
            Key Object Selection Document

        """
        if dataset.SOPClassUID != KeyObjectSelectionDocumentStorage:
            raise ValueError('Dataset is not a Key Object Selection Document.')
        _check_little_endian(dataset)
        sop_instance = deepcopy(dataset)
        sop_instance.__class__ = cls

        # Cache copy of the content to facilitate subsequent access
        root_item = Dataset()
        root_item.ConceptNameCodeSequence = dataset.ConceptNameCodeSequence
        root_item.ContentSequence = dataset.ContentSequence
        root_item.ContentTemplateSequence = dataset.ContentTemplateSequence
        root_item.ValueType = dataset.ValueType
        root_item.ContinuityOfContent = dataset.ContinuityOfContent
        content_item = ContainerContentItem.from_dataset(root_item)
        sop_instance._content = KeyObjectSelection.from_sequence(
            [content_item],
            is_root=True
        )

        sop_instance._reference_lut = {}
        for study_item in sop_instance.CurrentRequestedProcedureEvidenceSequence:  # noqa: E501
            for series_item in study_item.ReferencedSeriesSequence:
                for instance_item in series_item.ReferencedSOPSequence:
                    sop_instance_uid = instance_item.ReferencedSOPInstanceUID
                    sop_instance._reference_lut[sop_instance_uid] = (
                        study_item.StudyInstanceUID,
                        series_item.SeriesInstanceUID,
                        sop_instance_uid
                    )

        return cast(KeyObjectSelectionDocument, sop_instance)
