"""Module for SOP Classes of Presentation State (PR) IODs."""
import logging
import pkgutil
from collections import defaultdict
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import ExplicitVRLittleEndian
from pydicom._storage_sopclass_uids import (
    AdvancedBlendingPresentationStateStorage,
    ColorSoftcopyPresentationStateStorage,
    GrayscaleSoftcopyPresentationStateStorage,
    PseudoColorSoftcopyPresentationStateStorage,
)
from pydicom.valuerep import PersonName

from highdicom.base import SOPClass
from highdicom.content import (
    ContentCreatorIdentificationCodeSequence,
    ModalityLUTTransformation,
    PaletteColorLUTTransformation,
    PresentationLUTTransformation,
)
from highdicom.enum import PresentationLUTShapeValues
from highdicom.pr.content import (
    _add_displayed_area_attributes,
    _add_graphic_group_annotation_layer_attributes,
    _add_icc_profile_attributes,
    _add_modality_lut_attributes,
    _add_palette_color_lookup_table_attributes,
    _add_presentation_state_identification_attributes,
    _add_presentation_state_relationship_attributes,
    _add_softcopy_presentation_lut_attributes,
    _add_softcopy_voi_lut_attributes,
    _get_modality_lut_transformation,
    _get_softcopy_voi_lut_transformations,
    _get_icc_profile,
    AdvancedBlending,
    BlendingDisplay,
    GraphicLayer,
    GraphicGroup,
    GraphicAnnotation,
    SoftcopyVOILUTTransformation,
)
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID


logger = logging.getLogger(__name__)


class GrayscaleSoftcopyPresentationState(SOPClass):

    """SOP class for a Grayscale Softcopy Presentation State (GSPS) object.

    A GSPS object includes instructions for the presentation of a grayscale
    image by software.

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
        concept_name: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        modality_lut_transformation: Optional[
            ModalityLUTTransformation
        ] = None,
        voi_lut_transformations: Optional[
            Sequence[SoftcopyVOILUTTransformation]
        ] = None,
        presentation_lut_transformation: Optional[
            PresentationLUTTransformation
        ] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
            Images that should be referenced
        series_instance_uid: str
            UID of the series
        series_number: int
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
        content_description: Union[str, None], optional
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]], optional
            Description of graphic groups used in this presentation state.
        concept_name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
            A coded description of the content of this presentation state.
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance.
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance.
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person who created the content of this presentation
            state.
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this presentation state.
        modality_lut_transformation: Union[highdicom.ModalityLUTTransformation, None], optional
            Description of the Modality LUT Transformation for transforming modality
            dependent into modality independent pixel values. If no value is provided,
            the modality transformation in the referenced images, if any, will be used.
        voi_lut_transformations: Union[Sequence[highdicom.pr.SoftcopyVOILUTTransformation], None], optional
            Description of the VOI LUT Transformation for transforming
            modality pixel values into pixel values that are of interest to a
            user or an application. If no value is provided, the VOI LUT
            transformation in the referenced images, if any, will be
            used.
        presentation_lut_transformation: Union[highdicom.PresentationLUTTransformation, None], optional
            Description of the Presentation LUT Transformation for transforming
            polarity pixel values into device-independent presentation values
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        if len(referenced_images) == 0:
            raise ValueError(
                'Argument "referenced_images" should not be empty.'
            )
        for ref_im in referenced_images:
            if ref_im.SamplesPerPixel != 1:
                raise ValueError(
                    'For grayscale presentation states, all referenced images '
                    'must have a single sample per pixel.'
                )
        ref_im = referenced_images[0]

        super().__init__(
            study_instance_uid=ref_im.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=GrayscaleSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            modality='PR',
            patient_id=ref_im.PatientID,
            transfer_syntax_uid=transfer_syntax_uid,
            patient_name=ref_im.PatientName,
            patient_birth_date=ref_im.PatientBirthDate,
            patient_sex=ref_im.PatientSex,
            accession_number=ref_im.AccessionNumber,
            study_id=ref_im.StudyID,
            study_date=ref_im.StudyDate,
            study_time=ref_im.StudyTime,
            referring_physician_name=getattr(
                ref_im, 'ReferringPhysicianName', None
            ),
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            **kwargs
        )
        self.copy_patient_and_study_information(ref_im)
        self.copy_specimen_information(ref_im)

        # Presentation State Identification
        _add_presentation_state_identification_attributes(
            self,
            content_label=content_label,
            content_description=content_description,
            concept_name=concept_name,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification
        )

        # Presentation State Relationship
        _add_presentation_state_relationship_attributes(
            self,
            referenced_images=referenced_images
        )

        # Graphic Group, Graphic Annotation, and Graphic Layer
        _add_graphic_group_annotation_layer_attributes(
            self,
            referenced_images=referenced_images,
            graphic_groups=graphic_groups,
            graphic_annotations=graphic_annotations,
            graphic_layers=graphic_layers
        )

        # Displayed Area
        _add_displayed_area_attributes(
            self,
            referenced_images=referenced_images
        )

        # Modality LUT
        if modality_lut_transformation is not None:
            _add_modality_lut_attributes(
                self,
                modality_lut_transformation=modality_lut_transformation
            )
        else:
            modality_lut_transformation = _get_modality_lut_transformation(
                referenced_images
            )
            if modality_lut_transformation is None:
                logger.debug(
                    'no Modality LUT attributes found in referenced images'
                )
            else:
                logger.debug(
                    'use Modality LUT attributes from referenced images'
                )
                _add_modality_lut_attributes(
                    self,
                    modality_lut_transformation=modality_lut_transformation
                )

        # Softcopy VOI LUT
        if voi_lut_transformations is not None:
            if len(voi_lut_transformations) == 0:
                raise ValueError(
                    'Argument "voi_lut_transformations" must not be '
                    'empty.'
                )
            for v in voi_lut_transformations:
                if not isinstance(v, SoftcopyVOILUTTransformation):
                    raise TypeError(
                        'Items of argument "voi_lut_transformations" '
                        'must be of type SoftcopyVOILUTTransformation.'
                    )

            if len(voi_lut_transformations) > 1:
                if not all(
                    hasattr(v, 'ReferencedImageSequence')
                    for v in voi_lut_transformations
                ):
                    raise ValueError(
                        'If argument "voi_lut_transformations" '
                        'contains multiple items, each item must reference the '
                        'images that it applies to.'
                    )
            _add_softcopy_voi_lut_attributes(
                self,
                referenced_images=referenced_images,
                voi_lut_transformations=voi_lut_transformations
            )
        else:
            voi_lut_transformations = _get_softcopy_voi_lut_transformations(
                referenced_images
            )
            if len(voi_lut_transformations) > 0:
                logger.debug(
                    'use VOI LUT attributes from referenced images'
                )
                _add_softcopy_voi_lut_attributes(
                    self,
                    referenced_images=referenced_images,
                    voi_lut_transformations=voi_lut_transformations
                )
            else:
                logger.debug(
                    'no VOI LUT attributes found in referenced images'
                )

        # Softcopy Presentation LUT
        if presentation_lut_transformation is None:
            # Use the identity transformation as default presentation
            # transformation
            presentation_lut_transformation = PresentationLUTTransformation(
                presentation_lut_shape=PresentationLUTShapeValues.IDENTITY
            )
        _add_softcopy_presentation_lut_attributes(
            self,
            presentation_lut_transformation=presentation_lut_transformation
        )


class PseudoColorSoftcopyPresentationState(SOPClass):

    """SOP class for a Pseudo-Color Softcopy Presentation State object.

    A Pseudo-Color Softcopy Presentation State object includes instructions for
    the presentation of a grayscale image as a color image by software.

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
        palette_color_lut_transformation: PaletteColorLUTTransformation,
        content_label: str,
        content_description: Optional[str] = None,
        graphic_annotations: Optional[Sequence[GraphicAnnotation]] = None,
        graphic_layers: Optional[Sequence[GraphicLayer]] = None,
        graphic_groups: Optional[Sequence[GraphicGroup]] = None,
        concept_name: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        modality_lut_transformation: Optional[
            ModalityLUTTransformation
        ] = None,
        voi_lut_transformations: Optional[
            Sequence[SoftcopyVOILUTTransformation]
        ] = None,
        icc_profile: Optional[bytes] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
            Images that should be referenced.
        series_instance_uid: str
            UID of the series
        series_number: int
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
        palette_color_lut_transformation: highdicom.PaletteColorLUTTransformation
            Description of the Palette Color LUT Transformation for tranforming
            grayscale into RGB color pixel values
        content_label: str
            A label used to describe the content of this presentation state.
            Must be a valid DICOM code string consisting only of capital
            letters, underscores and spaces.
        content_description: Union[str, None], optional
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]], optional
            Description of graphic groups used in this presentation state.
        concept_name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
            A coded description of the content of this presentation state.
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance.
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance.
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person who created the content of this presentation
            state.
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this presentation state.
        modality_lut_transformation: Union[highdicom.ModalityLUTTransformation, None], optional
            Description of the Modality LUT Transformation for tranforming modality
            dependent into modality independent pixel values
        voi_lut_transformations: Union[Sequence[highdicom.pr.SoftcopyVOILUTTransformation], None], optional
            Description of the VOI LUT Transformation for tranforming
            modality pixel values into pixel values that are of interest to a
            user or an application
        icc_profile: Union[bytes, None], optional
            ICC color profile to include in the presentation state. If none is
            provided, the profile will be copied from the referenced images.
            The profile must follow the constraints listed in :dcm:`C.11.15
            <part03/sect_C.11.15.html>`.
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        if len(referenced_images) == 0:
            raise ValueError(
                'Argument "referenced_images" should not be empty.'
            )
        for ref_im in referenced_images:
            if ref_im.SamplesPerPixel != 1:
                raise ValueError(
                    'For pseudo-color presentation states, all referenced '
                    'images must have a single sample per pixel.'
                )
        ref_im = referenced_images[0]

        super().__init__(
            study_instance_uid=ref_im.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=PseudoColorSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            modality='PR',
            patient_id=ref_im.PatientID,
            transfer_syntax_uid=transfer_syntax_uid,
            patient_name=ref_im.PatientName,
            patient_birth_date=ref_im.PatientBirthDate,
            patient_sex=ref_im.PatientSex,
            accession_number=ref_im.AccessionNumber,
            study_id=ref_im.StudyID,
            study_date=ref_im.StudyDate,
            study_time=ref_im.StudyTime,
            referring_physician_name=getattr(
                ref_im, 'ReferringPhysicianName', None
            ),
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            **kwargs
        )
        self.copy_patient_and_study_information(ref_im)
        self.copy_specimen_information(ref_im)

        # Presentation State Identification
        _add_presentation_state_identification_attributes(
            self,
            content_label=content_label,
            content_description=content_description,
            concept_name=concept_name,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification
        )

        # Presentation State Relationship
        _add_presentation_state_relationship_attributes(
            self,
            referenced_images=referenced_images
        )

        # Graphic Group, Graphic Annotation, and Graphic Layer
        _add_graphic_group_annotation_layer_attributes(
            self,
            referenced_images=referenced_images,
            graphic_groups=graphic_groups,
            graphic_annotations=graphic_annotations,
            graphic_layers=graphic_layers
        )

        # Displayed Area
        _add_displayed_area_attributes(
            self,
            referenced_images=referenced_images
        )

        # Modality LUT
        if modality_lut_transformation is not None:
            _add_modality_lut_attributes(
                self,
                modality_lut_transformation=modality_lut_transformation
            )
        else:
            modality_lut_transformation = _get_modality_lut_transformation(
                referenced_images
            )
            if modality_lut_transformation is None:
                logger.debug(
                    'no Modality LUT attributes found in referenced images'
                )
            else:
                logger.debug(
                    'use Modality LUT attributes from referenced images'
                )
                _add_modality_lut_attributes(
                    self,
                    modality_lut_transformation=modality_lut_transformation
                )

        # Softcopy VOI LUT
        if voi_lut_transformations is not None:
            if len(voi_lut_transformations) == 0:
                raise ValueError(
                    'Argument "voi_lut_transformations" must not be '
                    'empty.'
                )
            for v in voi_lut_transformations:
                if not isinstance(v, voi_lut_transformations):
                    raise TypeError(
                        'Items of argument "voi_lut_transformations" '
                        'must be of type SoftcopyVOILUTTransformation.'
                    )

            if len(voi_lut_transformations) > 1:
                if not all(
                    hasattr(v, 'ReferencedImageSequence')
                    for v in voi_lut_transformations
                ):
                    raise ValueError(
                        'If argument "voi_lut_transformations" '
                        'contains multiple items, each item must reference the '
                        'images that it applies to.'
                    )
            _add_softcopy_voi_lut_attributes(
                self,
                referenced_images=referenced_images,
                voi_lut_transformations=voi_lut_transformations
            )
        else:
            voi_lut_transformations = _get_softcopy_voi_lut_transformations(
                referenced_images
            )
            if len(voi_lut_transformations) > 0:
                logger.debug(
                    'use VOI LUT attributes from referenced images'
                )
                _add_softcopy_voi_lut_attributes(
                    self,
                    referenced_images=referenced_images,
                    voi_lut_transformations=voi_lut_transformations
                )
            else:
                logger.debug(
                    'no VOI LUT attributes found in referenced images'
                )

        # Palette Color Lookup Table
        _add_palette_color_lookup_table_attributes(
            self,
            palette_color_lut_transformation=palette_color_lut_transformation
        )

        # ICC Profile
        if icc_profile is None:
            # Use default sRGB profile
            icc_profile = pkgutil.get_data(
                'highdicom',
                '_icc_profiles/sRGB_v4_ICC_preference.icc'
            )
        _add_icc_profile_attributes(
            self,
            icc_profile=icc_profile
        )


class ColorSoftcopyPresentationState(SOPClass):

    """SOP class for a Color Softcopy Presentation State object.

    A Color Softcopy Presentation State object includes instructions for
    the presentation of a color image by software.

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
        concept_name: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        icc_profile: Optional[bytes] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
            Images that should be referenced
        series_instance_uid: str
            UID of the series
        series_number: int
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
        content_description: Union[str, None], optional
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]], optional
            Description of graphic groups used in this presentation state.
        concept_name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
            A coded description of the content of this presentation state.
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance.
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance.
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person who created the content of this presentation
            state.
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this presentation state.
        icc_profile: Union[bytes, None], optional
            ICC color profile to include in the presentation state. If none is
            provided, the profile will be copied from the referenced images.
            The profile must follow the constraints listed in :dcm:`C.11.15
            <part03/sect_C.11.15.html>`.
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        if len(referenced_images) == 0:
            raise ValueError(
                'Argument "referenced_images" should not be empty.'
            )
        for ref_im in referenced_images:
            if ref_im.SamplesPerPixel != 3:
                raise ValueError(
                    'For color presentation states, all referenced '
                    'images must have three samples per pixel.'
                )
        ref_im = referenced_images[0]

        super().__init__(
            study_instance_uid=ref_im.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=ColorSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            modality='PR',
            patient_id=ref_im.PatientID,
            transfer_syntax_uid=transfer_syntax_uid,
            patient_name=ref_im.PatientName,
            patient_birth_date=ref_im.PatientBirthDate,
            patient_sex=ref_im.PatientSex,
            accession_number=ref_im.AccessionNumber,
            study_id=ref_im.StudyID,
            study_date=ref_im.StudyDate,
            study_time=ref_im.StudyTime,
            referring_physician_name=getattr(
                ref_im, 'ReferringPhysicianName', None
            ),
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            **kwargs
        )
        self.copy_patient_and_study_information(ref_im)
        self.copy_specimen_information(ref_im)

        # Presentation State Identification
        _add_presentation_state_identification_attributes(
            self,
            content_label=content_label,
            content_description=content_description,
            concept_name=concept_name,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification
        )

        # Presentation State Relationship
        _add_presentation_state_relationship_attributes(
            self,
            referenced_images=referenced_images
        )

        # Graphic Group, Graphic Annotation, and Graphic Layer
        _add_graphic_group_annotation_layer_attributes(
            self,
            referenced_images=referenced_images,
            graphic_groups=graphic_groups,
            graphic_annotations=graphic_annotations,
            graphic_layers=graphic_layers
        )

        # Displayed Area
        _add_displayed_area_attributes(
            self,
            referenced_images=referenced_images
        )

        # ICC Profile
        if icc_profile is not None:
            _add_icc_profile_attributes(
                self,
                icc_profile=icc_profile
            )
        else:
            icc_profile = _get_icc_profile(referenced_images)
            _add_icc_profile_attributes(
                self,
                icc_profile=icc_profile
            )


class AdvancedBlendingPresentationState(SOPClass):

    """SOP class for an Advanced Blending Presentation State object.

    An Advanced Blending Presentation State object includes instructions for
    the blending of one or more pseudo-color or color images by software. If
    the referenced images are grayscale images, they first need to be
    pseudo-colored.

    """

    def __init__(
        self,
        referenced_images: Sequence[Dataset],
        blending: Sequence[AdvancedBlending],
        blending_display: Sequence[BlendingDisplay],
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
        concept_name: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        icc_profile: Optional[bytes] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
            Images that should be referenced. This list should contain all
            images that are referenced across all `blending` items.
        blending: Sequence[highdicom.pr.AdvancedBlending]
            Description of groups of images that should be blended to form a
            pseudo-color image.
        blending_display: Sequence[highdicom.pr.BlendingDisplay]
            Description of the blending operations and the images to be
            used. Each item results in an individual pseudo-color RGB image,
            which may reused in a following step.
        series_instance_uid: str
            UID of the series
        series_number: int
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
        content_description: Union[str, None], optional
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]], optional
            Description of graphic groups used in this presentation state.
        concept_name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
            A coded description of the content of this presentation state.
        institution_name: Union[str, None], optional
            Name of the institution of the person or device that creates the
            SR document instance.
        institutional_department_name: Union[str, None], optional
            Name of the department of the person or device that creates the
            SR document instance.
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the person who created the content of this presentation
            state.
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this presentation state.
        icc_profile: Union[bytes, None], optional
            ICC color profile to include in the presentation state. If none is
            provided, a default profile will be included for the sRGB color
            space. The profile must follow the constraints listed in
            :dcm:`C.11.15 <part03/sect_C.11.15.html>`.
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        ref_im = referenced_images[0]
        for im in referenced_images:
            if im.StudyInstanceUID != ref_im.StudyInstanceUID:
                raise ValueError(
                    'For advanced blending presentation state, all referenced '
                    'images must have the same Study Instance UID.'
                )
            if im.FrameOfReferenceUID != ref_im.FrameOfReferenceUID:
                raise ValueError(
                    'For advanced blending presentation state, all referenced '
                    'images must have the same Frame of Reference UID.'
                )

        super().__init__(
            study_instance_uid=ref_im.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=AdvancedBlendingPresentationStateStorage,
            instance_number=instance_number,
            modality='PR',
            patient_id=ref_im.PatientID,
            transfer_syntax_uid=transfer_syntax_uid,
            patient_name=ref_im.PatientName,
            patient_birth_date=ref_im.PatientBirthDate,
            patient_sex=ref_im.PatientSex,
            accession_number=ref_im.AccessionNumber,
            study_id=ref_im.StudyID,
            study_date=ref_im.StudyDate,
            study_time=ref_im.StudyTime,
            referring_physician_name=getattr(
                ref_im, 'ReferringPhysicianName', None
            ),
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            **kwargs
        )
        self.copy_patient_and_study_information(ref_im)
        self.copy_specimen_information(ref_im)

        # Advanced Blending Presentation State
        blending_input_numbers = np.zeros(
            (len(blending_display), ),
            dtype=int
        )
        ref_item = blending[0]
        if ref_item.StudyInstanceUID != ref_im.StudyInstanceUID:
            raise ValueError(
                'Items of argument "blending" must have the same '
                'Study Instance UID as the referenced images.'
            )

        for i, item in enumerate(blending, 1):
            if not isinstance(item, AdvancedBlending):
                raise TypeError(
                    'Items of argument "blending" must have type '
                    'AdvancedBlending.'
                )
            if item.StudyInstanceUID != ref_item.StudyInstanceUID:
                raise ValueError(
                    'All items of argument "blending" must have the same '
                    'Study Instance UID.'
                )
            if int(item.BlendingInputNumber) != i:
                raise ValueError(
                    'The values of attribute Blending Input Number of items of '
                    'argument "blending" must be ordinal numbers starting from '
                    '1 and monotonically increasing by 1.'
                )
        self.AdvancedBlendingSequence = blending

        # Advanced Blending Presentation State Display
        self.PixelPresentation = 'TRUE_COLOR'

        blending_input_numbers = np.zeros(
            (len(blending_display), ),
            dtype=int
        )
        for i, item in enumerate(blending_display):
            if not isinstance(item, BlendingDisplay):
                raise TypeError(
                    'Items of argument "blending_display" must have type '
                    'BlendingDisplay.'
                )
            # One item shall not have a Blending Input Number
            try:
                blending_input_numbers[i] = int(item.BlendingInputNumber)
            except AttributeError:
                pass
        if np.sum(blending_input_numbers == 0) != 1:
            raise ValueError(
                'All but one item of argument "blending_display" must '
                'specify Blending Input Number.'
            )
        blending_input_numbers = blending_input_numbers[
            blending_input_numbers != 0
        ]
        unique_blending_input_numbers = np.unique(blending_input_numbers)
        if len(unique_blending_input_numbers) != len(blending_input_numbers):
            raise ValueError(
                'The values of attribute Blending Input Number of items of '
                'argument "blending_display" must be unique.'
            )
        self.BlendingDisplaySequence = blending_display

        # Presentation State Identification
        _add_presentation_state_identification_attributes(
            self,
            content_label=content_label,
            content_description=content_description,
            concept_name=concept_name,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification
        )

        # Confusingly, the Advanced Blending Presentation State IOD does not
        # include the Presentation State Relationship module, but instead
        # includes the Common Instance Reference module.
        ref_im_items_mapping = defaultdict(list)
        for im in referenced_images:
            item = Dataset()
            item.ReferencedSOPClassUID = im.SOPClassUID
            item.ReferencedSOPInstanceUID = im.SOPInstanceUID
            ref_im_items_mapping[im.SeriesInstanceUID].append(item)

        self.ReferencedSeriesSequence = []
        for series_instance_uid, ref_images in ref_im_items_mapping.items():
            item = Dataset()
            item.SeriesInstanceUID = series_instance_uid
            item.ReferencedInstanceSequence = ref_images
            self.ReferencedSeriesSequence.append(item)

        # Graphic Group, Graphic Annotation, and Graphic Layer
        _add_graphic_group_annotation_layer_attributes(
            self,
            referenced_images=referenced_images,
            graphic_groups=graphic_groups,
            graphic_annotations=graphic_annotations,
            graphic_layers=graphic_layers
        )

        # Displayed Area
        _add_displayed_area_attributes(
            self,
            referenced_images=referenced_images
        )

        # ICC Profile
        if icc_profile is None:
            # Use default sRGB profile
            icc_profile = pkgutil.get_data(
                'highdicom',
                '_icc_profiles/sRGB_v4_ICC_preference.icc'
            )
        _add_icc_profile_attributes(
            self,
            icc_profile=icc_profile
        )
