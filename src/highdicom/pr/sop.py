"""Module for SOP Classes of Presentation State (PR) IODs."""
import datetime
from typing import Sequence, Optional, Tuple, Union

from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import ExplicitVRLittleEndian, UID
from pydicom._storage_sopclass_uids import (
    GrayscaleSoftcopyPresentationStateStorage,
)
from pydicom.valuerep import DA, PersonName, TM

from highdicom.base import SOPClass
from highdicom.pr.content import GraphicAnnotation
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import check_person_name, _check_code_string


class GrayscaleSoftcopyPresentationState(SOPClass):

    """SOP class for a Grayscale Softcopy Presentation State (GSPS) object.

    A GSPS object includes instructions for the presentation of an image by
    software.

    """

    def __init__(
        self,
        referenced_images: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: Union[str, Tuple[str]],
        device_serial_number: str,
        content_label: str,
        content_description: Optional[str] = None,
        graphic_annotations: Optional[Sequence[GraphicAnnotation]] = None,
        concept_name_code: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance
        device_serial_number: Union[str, None]
            Manufacturer's serial number of the device
        software_versions: Union[str, Tuple[str]]
            Version(s) of the software that creates the instance
        """
        # Check referenced images are from the same series and have the same
        # size
        ref_series_uid = referenced_images[0].SeriesInstanceUID
        ref_im_rows = referenced_images[0].Rows
        ref_im_columns = referenced_images[0].Columns
        for ref_im in referenced_images:
            series_uid = ref_im.SeriesInstanceUID
            if series_uid != ref_series_uid:
                raise ValueError(
                    'Images belonging to different series are not supported.'
                )
            if ref_im.Rows != ref_im_rows or ref_im.Columns != ref_im_columns:
                raise ValueError(
                    'Images with different sizes are not supported.'
                )

        super().__init__(
            study_instance_uid=referenced_images[0].StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=GrayscaleSoftcopyPresentationStateStorage,
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

        # General Equipment
        self.Manufacturer = manufacturer
        if institution_name is not None:
            self.InstitutionName = institution_name
            if institutional_department_name is not None:
                self.InstitutionalDepartmentName = institutional_department_name
        self.DeviceSerialNumber = device_serial_number
        self.ManufacturerModelName = manufacturer_model_name
        self.SoftwareVersions = software_versions

        # Presentation State Identification
        _check_code_string(content_label)
        self.ContentLabel = content_label
        if content_description is not None and len(content_description) > 64:
            raise ValueError(
                'Argument "content_description" must not exceed 64 characters.'
            )
        self.ContentDescription = content_description
        self.PresentationCreationDate = DA(datetime.datetime.now().date())
        self.PresentationCreationTime = TM(datetime.datetime.now().time())
        if concept_name_code is not None:
            if not isinstance(concept_name_code, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "concept_name_code" should be of type '
                    'pydicom.sr.coding.Code or '
                    'highdicom.sr.coding.CodedConcept.'
                )
            self.ConceptNameCodeSequence = [concept_name_code]
        # TODO Content Creator Identification Code Sequence
        # TODO Alternative Content Description Sequence???

        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name

        # Presentation State Relationship
        ref_im_seq = []
        for im in referenced_images:
            ref_im_item = Dataset()
            ref_im_item.ReferencedSOPClassUID = im.SOPClassUID
            ref_im_item.ReferencedSOPInstanceUID = im.SOPInstanceUID
            ref_im_seq.append(ref_im_item)
        ref_series_item = Dataset()
        ref_series_item.SeriesInstanceUID = ref_series_uid
        ref_series_item.ReferencedImageSequence = ref_im_seq
        self.ReferencedSeriesSequence = [ref_series_item]

        # Graphic Annotation
        ref_uids = {
            (ds.ReferencedSOPClassUID, ds.ReferencedSOPInstanceUID)
            for ds in ref_im_seq
        }
        if len(graphic_annotations) > 0:
            for ann in graphic_annotations:
                if not isinstance(ann, GraphicAnnotation):
                    raise TypeError(
                        'Items of "graphic_annotations" should be of type '
                        'highdicom.pr.GraphicAnnotation.'
                    )
                for item in ann.ReferencedImageSequence:
                    uids = (
                        item.ReferencedSOPClassUID,
                        item.ReferencedSOPInstanceUID
                    )
                    if uids not in ref_uids:
                        raise ValueError(
                            'Instance with SOP Instance UID {uids[1]} and SOP '
                            'Class UID {uids[0]} is referenced in items of '
                            '"graphic_annotations", but included in '
                            '"referenced_images".'
                        )
            self.GraphicAnnotationSequence = graphic_annotations

        # Displayed Area Selection Sequence
        # This implements the simplest case - all images are unchanged
        # May want to generalize this later
        display_area_item = Dataset()
        display_area_item.ReferencedImageSequence = ref_im_seq
        display_area_item.PixelOriginInterpretation = 'FRAME'
        display_area_item.DisplayedAreaTopLeftHandCorner = [1, 1]
        display_area_item.DisplayedAreaBottomRightHandCorner = [
            ref_im_columns,
            ref_im_rows
        ]
        self.DisplayedAreaSelectionSequence = [display_area_item]
        self.PresentationSizeMode = 'SCALE TO FIT'
        self.PresentationPixelAspectRatio = [1, 1]

        # TODO Graphic Layer

        # TODO Graphic Group

        # Presentation State LUT
        self.PresentationLUTShape = 'IDENTITY'
