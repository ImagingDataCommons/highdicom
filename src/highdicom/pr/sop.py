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
from highdicom.pr.content import (
    GraphicLayer,
    GraphicGroup,
    GraphicAnnotation,
    ContentCreatorIdentificationCodeSequence
)
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
        graphic_layers: Optional[Sequence[GraphicLayer]] = None,
        graphic_groups: Optional[Sequence[GraphicGroup]] = None,
        concept_name_code: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        Parameters
        ----------
        referenced_images: Sequence[Dataset]
            List of images referenced in the GSPS.
        series_instance_uid: str
            UID of the series
        series_number: Union[int, None]
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        manufacturer: str
            Name of the manufacturer of the device (developer of the software)
            that creates the instance
        manufacturer_model_name: str
            Name of the device model (name of the software library or
            application) that creates the instance
        software_versions: Union[str, Tuple[str]]
            Version(s) of the software that creates the instance
        device_serial_number: Union[str, None]
            Manufacturer's serial number of the device
        content_label: str
            A label used to describe the content of this presentation state.
            Must be a valid DICOM code string consisting only of capital
            letters, underscores and spaces.
        content_description: Union[str, None]
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None]
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None]
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]]
            Description of graphic groups used in this presentation state.
        concept_name_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            A coded description of the content of this presentation state.
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance.
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance.
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None]
            Name of the person who created the content of this presentation
            state.
        content_creator_identification: Union[highdicom.pr.ContentCreatorIdentificationCodeSequence, None]
            Identifying information for the person who created the content of
            this presentation state.
        transfer_syntax_uid: Union[str, highdicom.UID]
            Transfer syntax UID of the presentation state.

        """  # noqa: E501
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
                    'highdicom.sr.CodedConcept.'
                )
            self.ConceptNameCodeSequence = [
                CodedConcept(
                    concept_name_code.value,
                    concept_name_code.scheme_designator,
                    concept_name_code.meaning,
                    concept_name_code.scheme_version
                )
            ]

        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name

        if content_creator_identification is not None:
            self.ContentCreatorIdentificationCodeSequence = \
                content_creator_identification

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

        # Graphic Groups
        group_ids = []
        if graphic_groups is not None:
            for grp in graphic_groups:
                if not isinstance(grp, GraphicGroup):
                    raise TypeError(
                        'Items of "graphic_groups" must be of type '
                        'highdicom.pr.GraphicGroup.'
                    )
                group_ids.append(grp.graphic_group_id)
            described_groups_ids = set(group_ids)
            if len(described_groups_ids) != len(group_ids):
                raise ValueError(
                    'Each item in "graphic_groups" must have a unique graphic '
                    'group ID.'
                )
            self.GraphicGroupSequence = graphic_groups
        else:
            described_groups_ids = set()

        # Graphic Annotation and Graphic Layer
        ref_uids = {
            (ds.ReferencedSOPClassUID, ds.ReferencedSOPInstanceUID)
            for ds in ref_im_seq
        }
        if graphic_layers is not None:
            labels = [layer.GraphicLayer for layer in graphic_layers]
            if len(labels) != len(set(labels)):
                raise ValueError(
                    'Labels of graphic layers must be unique.'
                )
            labels_unique = set(labels)
            self.GraphicLayerSequence = graphic_layers

        if graphic_annotations is not None:
            for ann in graphic_annotations:
                if not isinstance(ann, GraphicAnnotation):
                    raise TypeError(
                        'Items of "graphic_annotations" must be of type '
                        'highdicom.pr.GraphicAnnotation.'
                    )
                if ann.GraphicLayer not in labels_unique:
                    raise ValueError(
                        'Graphic layer with name "{ann.GraphicLayer}" is '
                        'referenced in "graphic_annotations" but not '
                        'included "graphic_layers".'
                    )
                for item in ann.ReferencedImageSequence:
                    uids = (
                        item.ReferencedSOPClassUID,
                        item.ReferencedSOPInstanceUID
                    )
                    if uids not in ref_uids:
                        raise ValueError(
                            'Instance with SOP Instance UID {uids[1]} and '
                            'SOP Class UID {uids[0]} is referenced in '
                            'items of "graphic_layers", but not included '
                            'in "referenced_images".'
                        )
                for obj in getattr(ann, 'GraphicObjectSequence', []):
                    grp_id = obj.graphic_group_id
                    if grp_id is not None:
                        if grp_id not in described_groups_ids:
                            raise ValueError(
                                'Found graphic object with graphic group '
                                f'ID "{grp_id}", but no such group is '
                                'described in the "graphic_groups" '
                                'argument.'
                            )
                for obj in getattr(ann, 'TextObjectSequence', []):
                    grp_id = obj.graphic_group_id
                    if grp_id is not None:
                        if grp_id not in described_groups_ids:
                            raise ValueError(
                                'Found text object with graphic group ID '
                                f'"{grp_id}", but no such group is '
                                'described in the "graphic_groups" '
                                'argument.'
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
        display_area_item.PresentationSizeMode = 'SCALE TO FIT'
        display_area_item.PresentationPixelAspectRatio = [1, 1]
        self.DisplayedAreaSelectionSequence = [display_area_item]

        # Presentation State LUT
        self.PresentationLUTShape = 'IDENTITY'
