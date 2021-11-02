"""Module for SOP Classes of Key Object (KO) IODs."""
import logging
from copy import deepcopy
from typing import Any, Optional, Sequence, Union

from pydicom.dataset import Dataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    UID,
)

from highdicom.base import SOPClass
from highdicom.sr.utils import collect_evidence
from highdicom.sr.value_types import ContentItem, ContentSequence

logger = logging.getLogger(__name__)


class KeyObjectSelectionDocument(SOPClass):

    """Key Object Selection Document SOP class."""

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
        content: pydicom.dataset.Dataset
            Root container content items that should be included in the
            SR document
        series_instance_uid: str
            Series Instance UID of the SR document series
        series_number: Union[int, None]
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
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance
        requested_procedures: Union[List[pydicom.dataset.Dataset], None], optional
            Requested procedures that are being fullfilled by creation of the
            SR document
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
            modality='KOS',
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
        content_copy = deepcopy(content)
        content_item = ContentItem._from_dataset_derived(content_copy)
        self._content = ContentSequence([content_item], is_root=True)
        for tag, value in content.items():
            self[tag] = value

        ref_items, unref_items = collect_evidence(evidence, content)
        if len(ref_items) > 0:
            self.CurrentRequestedProcedureEvidenceSequence = ref_items
            if len(ref_items) > 1:
                raise ValueError(
                    'Key Object Selection Documents that reference instances '
                    'from multiple studies are not supported.'
                )

        if requested_procedures is not None:
            self.ReferencedRequestSequence = requested_procedures
