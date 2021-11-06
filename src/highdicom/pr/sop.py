"""Module for SOP Classes of Presentation State (PR) IODs."""
from typing import Sequence, Optional, Union

from pydicom.sr.coding import Code

from highdicom.base import SOPClass
from highdicom.pr.content import GraphicAnnotation
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import check_person_name


class GrayscaleSoftcopyPresentationState(SOPClass):

    """SOP class for a Grayscale Softcopy Presentation State (GSPS) object.

    A GSPS object includes instructions for the presentation of an image by
    software.

    """

    def __init__(
        referenced_images: Sequence[Dataset],
        graphic_annotations: Optional[Sequence[GraphicAnnotation]] = None,
        content_label: str,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        sop_class_uid: str,
        instance_number: int,
        concept_name_code: Union[Code, CodedConcept, None] = None,
        manufacturer: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        """
        super().__init__(
            study_instance_uid=referenced_images[0].StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=manufacturer,
            modality='PR',
            patient_id=referenced_images[0].PatientID,
            transfer_syntax_uid=transfer_syntax_uid,
            patient_name=referenced_images[0].PatientName,
            patient_birth_date=referenced_images[0].PatientBirthDate,
            patient_sex=referenced_images[0].PatientSex,
            accession_number=referenced_images[0].AccessionNumber,
            study_id=referenced_images[0].StudyID,
            study_date=referenced_images[0].StudyDate,
            study_time=referenced_images[0].StudyTime,
            referring_physician_name=getattr(
                referenced_images[0], 'ReferringPhysicianName', None
            ),
            **kwargs
        )
        self.copy_patient_and_study_information(referenced_images[0])

        # Presentation State Identification

        self.ContentLabel = content_label
        if concept_name_code is not None:
            if not isinstance(concept_name_code, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "concept_name_code" should be of type '
                    'pydicom.sr.coding.Code or '
                    'highdicom.sr.coding.CodedConcept.'
                )
            self.ConceptNameCodeSequence = [concept_name_code]

        if content_creator_name is not None:
            check_person_name(content_creator_name)
            self.ContentCreatorName = content_creator_name

        if len(graphic_annotations) > 0:
            for ann in graphic_annotations:
                if not isinstance(ann, GraphicAnnotation):
                    raise TypeError(
                        'Items of "graphic_annotations" should be of type '
                        'highdicom.pr.GraphicAnnotation.'
                    )
            self.GraphicAnnotationSequence = graphic_annotations
