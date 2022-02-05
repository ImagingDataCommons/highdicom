"""Module for SOP Classes of Presentation State (PR) IODs."""
from collections import defaultdict
import datetime
from typing import Sequence, Optional, Tuple, Union

from PIL.ImageCms import ImageCmsProfile, createProfile

from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import ExplicitVRLittleEndian
from pydicom._storage_sopclass_uids import (
    GrayscaleSoftcopyPresentationStateStorage,
    PseudoColorSoftcopyPresentationStateStorage,
    ColorSoftcopyPresentationStateStorage
)
from pydicom.valuerep import DA, PersonName, TM, format_number_as_ds

import numpy as np

from highdicom.base import SOPClass
from highdicom.content import (
    ContentCreatorIdentificationCodeSequence,
    ModalityLUT,
    LUT,
)
from highdicom.pr.content import (
    GraphicLayer,
    GraphicGroup,
    GraphicAnnotation,
    SoftcopyVOILUT,
)
from highdicom.pr.enum import PresentationLUTShapeValues
from highdicom.enum import RescaleTypeValues
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID
from highdicom.valuerep import (
    check_person_name,
    _check_code_string,
    _check_long_string
)


class _SoftcopyPresentationState(SOPClass):

    """An abstract base class for various Presentation State objects."""

    def __init__(
        self,
        referenced_images: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        sop_class_uid: str,
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
        rescale_intercept: Union[int, float, None] = None,
        rescale_slope: Union[int, float, None] = None,
        rescale_type: Union[RescaleTypeValues, str, None] = None,
        modality_lut: Optional[ModalityLUT] = None,
        softcopy_voi_luts: Optional[Sequence[SoftcopyVOILUT]] = None,
        icc_profile: Optional[ImageCmsProfile] = None,
        copy_modality_lut: bool = False,
        copy_voi_lut: bool = False,
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
        sop_class_uid: str
            SOP Class UID of the instance.
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
        concept_name_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
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
        rescale_intercept: Union[int, float, None], optional
            Intercept used for rescaling pixel values.
        rescale_slope: Union[int, float, None], optional
            Slope used for rescaling pixel values.
        rescale_type: Union[highdicom.RescaleTypeValues, str, None], optional
            String or enumerated value specifying the units of the output of
            the Modality LUT or rescale operation.
        modality_lut: Union[highdicom.ModalityLUT, None], optional
            Lookup table specifying a pixel rescaling operation to apply to
            the stored values to give modality values.
        copy_modality_lut: bool, optional
            Include elements of the Modality LUT module (including
            RescaleIntercept, RescaleSlope and ModalityLUTSequence), if any,
            in the presentation state with values copied from the source images.
        softcopy_voi_luts: Union[Sequence[highdicom.pr.SoftcopyVOILUT], None], optional
            One or more pixel value-of-interest operations to applied after the
            modality LUT and/or rescale operation.
        copy_voi_lut: bool, optional
            Include elements of the Softcopy VOI LUT module (including
            WindowWidth, WindowCenter, and VOILUTSequence), if any, in the
            presentation state with values copied from the source images.
        icc_profile: Union[PIL.ImageCms.ImageCmsProfile, None], optional
            ICC color profile object to include in the presentation state. If
            none is provided, a standard RGB ("sRGB") profile will be assumed.
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        if sop_class_uid not in (
            GrayscaleSoftcopyPresentationStateStorage,
            PseudoColorSoftcopyPresentationStateStorage,
            ColorSoftcopyPresentationStateStorage
        ):
            raise ValueError(
                'The IOD indicated by the provided sop_class_uid is not '
                'implemented.'
            )

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

            if sop_class_uid == ColorSoftcopyPresentationStateStorage:
                if ref_im.SamplesPerPixel != 3:
                    raise ValueError(
                        'For color presentation states, all referenced '
                        'images must have 3 samples per pixel.'
                    )
            else:
                if ref_im.SamplesPerPixel != 1:
                    raise ValueError(
                        'For grayscale and pseduo-color presentation states, '
                        'all referenced images must have a single sample '
                        'per pixel.'
                    )

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
                        f'Graphic layer with name "{ann.GraphicLayer}" is '
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
                            f'Instance with SOP Instance UID {uids[1]} and '
                            f'SOP Class UID {uids[0]} is referenced in '
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

        # Modality LUT
        if sop_class_uid in (
            GrayscaleSoftcopyPresentationStateStorage,
            PseudoColorSoftcopyPresentationState,
        ):
            if copy_modality_lut:
                if (
                    rescale_intercept is not None or
                    rescale_slope is not None or
                    rescale_type is not None
                ):
                    raise TypeError(
                        'If argument "copy_modality_lut" is  True, arguments '
                        '"rescale_slope", "rescale_intercept", and '
                        '"rescale_type" must all be None.'
                    )
                self._copy_modality_lut(referenced_images)
            else:
                self._add_modality_lut(
                    rescale_intercept=rescale_intercept,
                    rescale_slope=rescale_slope,
                    rescale_type=rescale_type,
                    modality_lut=modality_lut
                )

        # Softcopy VOI LUT
        if copy_voi_lut:
            if softcopy_voi_luts is not None:
                raise TypeError(
                    'If argument "copy_voi_lut" is  True, argument '
                    '"softcopy_voi_luts" must be None.'
                )
            self._copy_voi_lut(referenced_images)
        elif softcopy_voi_luts is not None:
            if len(softcopy_voi_luts) == 0:
                raise ValueError(
                    'Argument "softcopy_voi_luts" must not be empty.'
                )
            for v in softcopy_voi_luts:
                if not isinstance(v, SoftcopyVOILUT):
                    raise TypeError(
                        'Items of "softcopy_voi_luts" must be of type '
                        'highdicom.pr.SoftcopyVOILUT.'
                    )

                # If the softcopy VOI LUT references specific images,
                # check that the references are valid
                if hasattr(v, 'ReferencedImageSequence'):
                    for item in v.ReferencedImageSequence:
                        uids = (
                            item.ReferencedSOPClassUID,
                            item.ReferencedSOPInstanceUID
                        )
                        if uids not in ref_uids:
                            raise ValueError(
                                f'Instance with SOP Instance UID {uids[1]} and '
                                f'SOP Class UID {uids[0]} is referenced in '
                                'items of "softcopy_voi_luts", but not '
                                'included in "referenced_images".'
                            )

            self.SoftcopyVOILUTSequence = softcopy_voi_luts

        # ICC Profile
        self._add_icc_profile(icc_profile)

    def _add_modality_lut(
        self,
        rescale_intercept: Union[int, float, None] = None,
        rescale_slope: Union[int, float, None] = None,
        rescale_type: Union[RescaleTypeValues, str, None] = None,
        modality_lut: Optional[ModalityLUT] = None,
    ) -> None:
        """Add attributes of the Modality LUT module (used by some subclasses).

        The Modality LUT module specifies an operation to transform the stored
        pixel values into output pixel values. This is done either by a linear
        transformation represented by rescale intercept, rescale slope and
        rescale type, or by a full look-up table (LUT).

        Parameters
        ----------
        rescale_intercept: Union[int, float, None]
            Intercept used for rescaling pixel values.
        rescale_slope: Union[int, float, None]
            Slope used for rescaling pixel values.
        rescale_type: Union[highdicom.RescaleTypeValues, str, None]
            String or enumerated value specifying the units of the output of
            the Modality LUT or rescale operation.
        modality_lut: Union[highdicom.ModalityLUT, None]
            Lookup table specifying the rescale operation.

        Note
        ----
        Either modality_lut may be specified or all three of rescale_slope,
        rescale_intercept and rescale_type may be specified. All four
        parameters should not be specified simultaneously. All parameters may
        be None if there is no modality LUT to apply to the image.

        """
        if modality_lut is not None:
            if rescale_intercept is not None:
                raise TypeError(
                    'Argument "rescale_intercept" must not be specified when '
                    '"modality_lut" is specified.'
                )
            if rescale_slope is not None:
                raise TypeError(
                    'Argument "rescale_slope" must not be specified when '
                    '"modality_lut" is specified.'
                )
            if rescale_type is not None:
                raise TypeError(
                    'Argument "rescale_type" must not be specified when '
                    '"modality_lut" is specified.'
                )
            if not isinstance(modality_lut, ModalityLUT):
                raise TypeError(
                    'Argument "modality_lut" must be of type '
                    'highdicom.ModalityLUT.'
                )
            self.ModalityLUTSequence = [modality_lut]
        elif rescale_intercept is not None:
            if rescale_slope is None:
                raise TypeError(
                    'Argument "rescale_slope" must be specified when '
                    '"modality_lut" is not specified.'
                )
            if rescale_type is None:
                raise TypeError(
                    'Argument "rescale_type" must be specified when '
                    '"modality_lut" is not specified.'
                )
            self.RescaleIntercept = format_number_as_ds(rescale_intercept)
            self.RescaleSlope = format_number_as_ds(rescale_slope)
            if isinstance(rescale_type, RescaleTypeValues):
                self.RescaleType = rescale_type.value
            else:
                _check_long_string(rescale_type)
                self.RescaleType = rescale_type
        else:
            # Nothing to do except check arguments
            if rescale_slope is not None:
                raise TypeError(
                    'Argument "rescale_slope" must not be specified unless '
                    '"rescale_intercept" is specified.'
                )
            if rescale_type is not None:
                raise TypeError(
                    'Argument "rescale_type" must not be specified when '
                    '"rescale_intercept" is specified.'
                )

    def _copy_modality_lut(self, referenced_images: Sequence[Dataset]) -> None:
        """Copy elements of the ModalityLUT module from the referenced images.

        Any rescale intercept, rescale slope, and rescale type attributes in
        the referenced images are copied to the presentation state dataset.
        Missing values will cause no errors, and will result in the relevant
        (optional) attributes being omitted from the presentation state object.
        However, inconsistent values between referenced images will result in
        an error.

        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
            The referenced images from which the attributes should be copied.

        Raises
        ------
        ValueError
            In case the presence or value of the RescaleSlope, RescaleIntercept,
            or RescaleType attributes are inconsistent between datasets.

        """
        have_slopes = [
            hasattr(ds, 'RescaleSlope') for ds in referenced_images
        ]
        have_intercepts = [
            hasattr(ds, 'RescaleIntercept') for ds in referenced_images
        ]
        have_type = [
            hasattr(ds, 'RescaleType') for ds in referenced_images
        ]

        if any(have_slopes) and not all(have_slopes):
            raise ValueError(
                'Error while copying Modality LUT attributes: presence of '
                '"RescaleSlope" is inconsistent among referenced images.'
            )
        if any(have_intercepts) and not all(have_intercepts):
            raise ValueError(
                'Error while copying Modality LUT attributes: presence of '
                '"RescaleIntercept" is inconsistent among referenced images.'
            )
        if any(have_type) and not all(have_type):
            raise ValueError(
                'Error while copying Modality LUT attributes: presence of '
                '"RescaleType" is inconsistent among referenced images.'
            )

        if all(have_intercepts) != all(have_slopes):
            raise ValueError(
                'Error while copying Modality LUT attributes: datasets should '
                'have both "RescaleIntercept" and "RescaleSlope", or neither.'
            )

        if all(have_intercepts):
            if any(
                ds.RescaleSlope != referenced_images[0].RescaleSlope
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values of '
                    '"RescaleSlope" are inconsistent among referenced images.'
                )
            if any(
                ds.RescaleIntercept != referenced_images[0].RescaleIntercept
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values of '
                    '"RescaleIntercept" are inconsistent among referenced '
                    'images.'
                )
            slope = referenced_images[0].RescaleSlope
            intercept = referenced_images[0].RescaleIntercept
        else:
            slope = None
            intercept = None

        if all(have_type):
            if any(
                ds.RescaleType != referenced_images[0].RescaleType
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values of '
                    '"RescaleType" are inconsistent among referenced images.'
                )
            rescale_type = referenced_images[0].RescaleType
        else:
            if intercept is None:
                rescale_type = None
            else:
                rescale_type = RescaleTypeValues.HU

        self._add_modality_lut(
            rescale_intercept=intercept,
            rescale_slope=slope,
            rescale_type=rescale_type,
        )

    def _copy_voi_lut(self, referenced_images: Sequence[Dataset]) -> None:
        """Copy elements of the Softcopy VOI LUT module from referenced images.

        Any window center, window width, window explanation, VOI LUT function,
        or VOI LUT Sequence attributes the referenced images are copied to the
        presentation state dataset.  Missing values will cause no errors, and
        will result in the relevant (optional) attributes being omitted from
        the presentation state object.  Inconsistent values between
        referenced images will result in multiple different items of the
        Softcopy VOI LUT Sequence in the presentation state object.

        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
            The referenced images from which the attributes should be copied.

        """
        by_window = defaultdict(list)
        by_lut = defaultdict(list)

        for ref_im in referenced_images:
            has_width = hasattr(ref_im, 'WindowWidth')
            has_center = hasattr(ref_im, 'WindowCenter')
            has_lut = hasattr(ref_im, 'VOILUTSequence')

            if has_width != has_center:
                raise ValueError(
                    'Error while copying VOI LUT attributes: found dataset '
                    'with mismatched WindowWidth and WindowCenter attributes.'
                )

            if has_width and has_lut:
                raise ValueError(
                    'Error while copying VOI LUT attributes: found dataset '
                    'with both window width/center and VOI LUT Sequence '
                    'attributes.'
                )

            if has_width:
                by_window[(
                    ref_im.WindowWidth,
                    ref_im.WindowCenter,
                    getattr(ref_im, 'WindowCenterWidthExplanation', None),
                    getattr(ref_im, 'VOILUTFunction', None),
                )].append(ref_im)
            elif has_lut:
                # Create a unique identifier for this list of LUTs
                lut_info = []
                for voi_lut in ref_im.VOILUTSequence:
                    lut_info.append((
                        voi_lut.LUTDescriptor[1],
                        voi_lut.LUTDescriptor[2],
                        getattr(voi_lut, 'LUTExplanation', None),
                        voi_lut.LUTData
                    ))
                lut_id = tuple(lut_info)
                by_lut[lut_id].append(ref_im)

        # TODO multiple LUTs
        # TODO referenced frames/segments
        # TODO multiframe
        softcopy_voi_luts = []
        for (width, center, exp, func), im_list in by_window.items():
            if len(im_list) == len(referenced_images):
                # All datasets included, no need to include the referenced
                # images explicitly
                refs_to_include = None
            else:
                # Include specific references
                refs_to_include = im_list

            softcopy_voi_luts.append(
                SoftcopyVOILUT(
                    window_center=center,
                    window_width=width,
                    window_explanation=exp,
                    voi_lut_function=func,
                    referenced_images=refs_to_include
                )
            )

        for lut_id, im_list in by_lut.items():
            if len(im_list) == len(referenced_images):
                # All datasets included, no need to include the referenced
                # images explicitly
                refs_to_include = None
            else:
                # Include specific references
                refs_to_include = im_list

            luts = [
                LUT(
                    first_mapped_value=fmv,
                    lut_data=np.frombuffer(
                        data,
                        np.uint8 if ba == 8 else np.uint16
                    ),
                    lut_explanation=exp
                ) for (fmv, ba, exp, data) in lut_id
            ]
            softcopy_voi_luts.append(
                SoftcopyVOILUT(
                    referenced_images=refs_to_include,
                    voi_luts=luts
                )
            )

        if len(softcopy_voi_luts) > 0:
            self.SoftcopyVOILUTSequence = softcopy_voi_luts

    def _add_icc_profile(
        self,
        icc_profile: Optional[ImageCmsProfile] = None
    ) -> None:
        """Add elements of ICC Profile module to the dataset.

        Parameters
        ----------
        icc_profile: Union[PIL.ImageCms.ImageCmsProfile, None], optional
            ICC color profile object to include in the presentation state. If
            none is provided, a standard RGB ("sRGB") profile will be assumed.

        """
        if self.SOPClassUID in (
            ColorSoftcopyPresentationStateStorage,
            PseudoColorSoftcopyPresentationStateStorage
        ):
            if icc_profile is None:
                # Use sRGB as the default profile if none was provided
                icc_profile = ImageCmsProfile(createProfile('sRGB'))

                # Populate this optional tag because this is known to be a
                # "well-known" color space
                self.ColorSpace = 'SRGB'
            else:
                if not isinstance(icc_profile, ImageCmsProfile):
                    raise TypeError(
                        'Argument "icc_profile" must be of type '
                        'PIL.ImageCms.ImageCmsProfile, or None.'
                    )

            self.ICCProfile = icc_profile.tobytes()
        else:
            if icc_profile is not None:
                raise TypeError(
                    'Including an "icc_profile" is only valid for Color- and '
                    'PseudoColor- presentation states.'
                )


class GrayscaleSoftcopyPresentationState(_SoftcopyPresentationState):

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
        concept_name_code: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        rescale_intercept: Union[int, float, None] = None,
        rescale_slope: Union[int, float, None] = None,
        rescale_type: Union[RescaleTypeValues, str, None] = None,
        modality_lut: Optional[ModalityLUT] = None,
        copy_modality_lut: bool = False,
        softcopy_voi_luts: Optional[Sequence[SoftcopyVOILUT]] = None,
        copy_voi_lut: bool = False,
        presentation_lut_shape: Union[
            PresentationLUTShapeValues,
            str,
            None
        ] = None,
        presentation_luts: Optional[Sequence[LUT]] = None,
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
        content_description: Union[str, None], optional
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]], optional
            Description of graphic groups used in this presentation state.
        concept_name_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
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
        rescale_intercept: Union[int, float, None], optional
            Intercept used for rescaling pixel values.
        rescale_slope: Union[int, float, None], optional
            Slope used for rescaling pixel values.
        rescale_type: Union[highdicom.RescaleTypeValues, str, None], optional
            String or enumerated value specifying the units of the output of
            the Modality LUT or rescale operation.
        modality_lut: Union[highdicom.ModalityLUT, None], optional
            Lookup table specifying a pixel rescaling operation to apply to
            the stored values to give modality values.
        copy_modality_lut: bool, optional
            Include elements of the Modality LUT module (including
            RescaleIntercept, RescaleSlope and ModalityLUTSequence), if any,
            in the presentation state with values copied from the source images.
        softcopy_voi_luts: Union[Sequence[highdicom.pr.SoftcopyVOILUT], None], optional
            One or more pixel value-of-interest operations to applied after the
            modality LUT and/or rescale operation.
        copy_voi_lut: bool, optional
            Include elements of the Softcopy VOI LUT module (including
            WindowWidth, WindowCenter, and VOILUTSequence), if any, in the
            presentation state with values copied from the source images.
        presentation_lut_shape: Union[highdicom.pr.PresentationLUTShapeValues, str, None], optional
            Shape of the presentation LUT transform, applied after the softcopy
            VOI transform to create display values.
        presentation_luts: Optional[Sequence[highdicom.LUT]], optional
            LUTs for the presentation LUT transform, applied after the softcopy
            VOI transform to create display values.
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        for kw in [
            'icc_profile',
        ]:
            if kw in kwargs:
                raise TypeError(
                    'GrayscaleSoftcopyPresentationState() got an unexpected '
                    f'keyword argument "{kw}".'
                )
        super().__init__(
            referenced_images=referenced_images,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=GrayscaleSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            content_label=content_label,
            content_description=content_description,
            graphic_annotations=graphic_annotations,
            graphic_layers=graphic_layers,
            graphic_groups=graphic_groups,
            concept_name_code=concept_name_code,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification,
            rescale_intercept=rescale_intercept,
            rescale_slope=rescale_slope,
            rescale_type=rescale_type,
            modality_lut=modality_lut,
            copy_modality_lut=copy_modality_lut,
            softcopy_voi_luts=softcopy_voi_luts,
            copy_voi_lut=copy_voi_lut,
            icc_profile=None,
            transfer_syntax_uid=transfer_syntax_uid,
            **kwargs
        )

        # Softcopy Presentation LUT
        if presentation_luts is not None:
            if presentation_lut_shape is not None:
                raise TypeError(
                    'Only one of "presentation_luts" or '
                    '"presentation_lut_shape" should be provided.'
                )
            if len(presentation_luts) == 0:
                raise ValueError(
                    'Argument "presentation_luts" must not be empty.'
                )
            for v in presentation_luts:
                if not isinstance(v, LUT):
                    raise TypeError(
                        'Items of "presentation_luts" should be of type '
                        'highdicom.LUT.'
                    )
            self.PresentationLUTSequence = presentation_luts
        else:
            presentation_lut_shape = (
                presentation_lut_shape or
                PresentationLUTShapeValues.IDENTITY
            )
            self.PresentationLUTShape = PresentationLUTShapeValues(
                presentation_lut_shape
            ).value


class PseudoColorSoftcopyPresentationState(_SoftcopyPresentationState):

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
        content_label: str,
        red_palette_color_lut_data: np.ndarray,
        green_palette_color_lut_data: np.ndarray,
        blue_palette_color_lut_data: np.ndarray,
        red_first_mapped_value: int,
        green_first_mapped_value: int,
        blue_first_mapped_value: int,
        palette_color_lut_uid: Union[UID, str, None] = None,
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
        rescale_intercept: Union[int, float, None] = None,
        rescale_slope: Union[int, float, None] = None,
        rescale_type: Union[RescaleTypeValues, str, None] = None,
        modality_lut: Optional[ModalityLUT] = None,
        copy_modality_lut: bool = False,
        softcopy_voi_luts: Optional[Sequence[SoftcopyVOILUT]] = None,
        copy_voi_lut: bool = False,
        icc_profile: Optional[ImageCmsProfile] = None,
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
        red_palette_color_lut_data: np.ndarray
            Array of values for the red color lookup table data. Must be a 1D
            array of uint16 values, where the first entry is the red output
            value of the palette color lookup table operation when the input
            pixel is ``"red_first_mapped_value"``, and so on.
        green_palette_color_lut_data: np.ndarray
            Array of values for the green color lookup table data. Otherwise as
            described for ``red_palette_color_lut_data``.
        blue_palette_color_lut_data: np.ndarray
            Array of values for the blue color lookup table data. Otherwise as
            described for ``red_palette_color_lut_data``.
        red_first_mapped_value: int
            Integer representing the first input value mapped by the red palette
            lookup table operation.
        green_first_mapped_value: int
            Integer representing the first input value mapped by the green
            lookup table operation.
        blue_first_mapped_value: int
            Integer representing the first input value mapped by the blue
            palette lookup table operation.
        palette_color_lut_uid: Union[UID, str, None], optional
            Unique identifier for the palette color lookup table.
        content_description: Union[str, None], optional
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]], optional
            Description of graphic groups used in this presentation state.
        concept_name_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
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
        rescale_intercept: Union[int, float, None], optional
            Intercept used for rescaling pixel values.
        rescale_slope: Union[int, float, None], optional
            Slope used for rescaling pixel values.
        rescale_type: Union[highdicom.RescaleTypeValues, str, None], optional
            String or enumerated value specifying the units of the output of
            the Modality LUT or rescale operation.
        modality_lut: Union[highdicom.ModalityLUT, None], optional
            Lookup table specifying a pixel rescaling operation to apply to
            the stored values to give modality values.
        copy_modality_lut: bool, optional
            Include elements of the Modality LUT module (including
            RescaleIntercept, RescaleSlope and ModalityLUTSequence), if any,
            in the presentation state with values copied from the source images.
        softcopy_voi_luts: Union[Sequence[highdicom.pr.SoftcopyVOILUT], None], optional
            One or more pixel value-of-interest operations to applied after the
            modality LUT and/or rescale operation.
        copy_voi_lut: bool, optional
            Include elements of the Softcopy VOI LUT module (including
            WindowWidth, WindowCenter, and VOILUTSequence), if any, in the
            presentation state with values copied from the source images.
        icc_profile: Union[PIL.ImageCms.ImageCmsProfile, None], optional
            ICC color profile object to include in the presentation state. If
            none is provided, a standard RGB ("sRGB") profile will be assumed.
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        super().__init__(
            referenced_images=referenced_images,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=PseudoColorSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            content_label=content_label,
            content_description=content_description,
            graphic_annotations=graphic_annotations,
            graphic_layers=graphic_layers,
            graphic_groups=graphic_groups,
            concept_name_code=concept_name_code,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification,
            rescale_intercept=rescale_intercept,
            rescale_slope=rescale_slope,
            rescale_type=rescale_type,
            modality_lut=modality_lut,
            copy_modality_lut=copy_modality_lut,
            softcopy_voi_luts=softcopy_voi_luts,
            copy_voi_lut=copy_voi_lut,
            icc_profile=icc_profile,
            transfer_syntax_uid=transfer_syntax_uid,
            **kwargs
        )

        # Palette Color Lookup Table
        self._add_palette_color_lookup_table(
            red_palette_color_lut_data=red_palette_color_lut_data,
            green_palette_color_lut_data=green_palette_color_lut_data,
            blue_palette_color_lut_data=blue_palette_color_lut_data,
            red_first_mapped_value=red_first_mapped_value,
            green_first_mapped_value=green_first_mapped_value,
            blue_first_mapped_value=blue_first_mapped_value,
            palette_color_lut_uid=palette_color_lut_uid,
        )

    def _add_palette_color_lookup_table(
        self,
        red_palette_color_lut_data: np.ndarray,
        green_palette_color_lut_data: np.ndarray,
        blue_palette_color_lut_data: np.ndarray,
        red_first_mapped_value: int,
        green_first_mapped_value: int,
        blue_first_mapped_value: int,
        palette_color_lut_uid: Union[UID, str, None] = None
    ) -> None:
        """Add attributes from the Palette Color Lookup Table module.

        Parameters
        ----------
        red_palette_color_lut_data: np.ndarray
            Array of values for the red color lookup table data. Must be a 1D
            array of uint16 values, where the first entry is the red output
            value of the palette color lookup table operation when the input
            pixel is ``"red_first_mapped_value"``, and so on.
        green_palette_color_lut_data: np.ndarray
            Array of values for the green color lookup table data. Otherwise as
            described for ``red_palette_color_lut_data``.
        blue_palette_color_lut_data: np.ndarray
            Array of values for the blue color lookup table data. Otherwise as
            described for ``red_palette_color_lut_data``.
        red_first_mapped_value: int
            Integer representing the first input value mapped by the red palette
            lookup table operation.
        green_first_mapped_value: int
            Integer representing the first input value mapped by the green
            lookup table operation.
        blue_first_mapped_value: int
            Integer representing the first input value mapped by the blue
            palette lookup table operation.
        palette_color_lut_uid: Union[UID, str, None], optional
            Unique identifier for the palette color lookup table.

        """
        colors = ['red', 'green', 'blue']
        all_lut_data = [
            red_palette_color_lut_data,
            green_palette_color_lut_data,
            blue_palette_color_lut_data
        ]
        all_first_values = [
            red_first_mapped_value,
            green_first_mapped_value,
            blue_first_mapped_value
        ]

        if palette_color_lut_uid is not None:
            self.PaletteColorLookupTableUID = palette_color_lut_uid

        for color, lut_data, first_mapped_value in zip(
            colors,
            all_lut_data,
            all_first_values
        ):
            if not isinstance(first_mapped_value, int):
                raise TypeError(
                    f'Argument "{color}_first_mapped_value" must be an integer.'
                )
            if first_mapped_value < 0:
                raise ValueError(
                    'Argument "first_mapped_value" must be non-negative.'
                )
            if first_mapped_value >= 2 ** 16:
                raise ValueError(
                    f'Argument "{color}_first_mapped_value" must be less than '
                    '2^16.'
                )

            if not isinstance(lut_data, np.ndarray):
                raise TypeError(
                    f'Argument "f{color}_palette_color_lut_data" must be of '
                    'type np.ndarray.'
                )
            if lut_data.ndim != 1:
                raise ValueError(
                    f'Argument "f{color}_palette_color_lut_data" '
                    'must have a single dimension.'
                )
            len_data = lut_data.size
            if len_data == 0:
                raise ValueError(
                    f'Argument "f{color}_palette_color_lut_data" '
                    'must not be empty.'
                )
            if len_data > 2**16:
                raise ValueError(
                    f'Length of "f{color}_palette_color_lut_data" must be no '
                    'greater than 2^16 elements.'
                )
            elif len_data == 2**16:
                # Per the standard, this is recorded as 0
                len_data = 0

            if lut_data.dtype.type != np.uint16:
                raise ValueError(
                    f'Argument "f{color}_palette_color_lut_data" must have '
                    'dtype uint16.'
                )

            descriptor = [
                len_data,
                first_mapped_value,
                16  # always 16 as part of Palette Color LUT module
            ]
            setattr(
                self,
                f'{color.title()}PaletteColorLookupTableDescriptor',
                descriptor
            )
            setattr(
                self,
                f'{color.title()}PaletteColorLookupTableData',
                lut_data.tobytes()
            )


class ColorSoftcopyPresentationState(_SoftcopyPresentationState):

    """SOP class for a Pseudo-Color Softcopy Presentation State object.

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
        concept_name_code: Union[Code, CodedConcept, None] = None,
        institution_name: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        icc_profile: Optional[ImageCmsProfile] = None,
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
        content_description: Union[str, None], optional
            Description of the content of this presentation state.
        graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
            Graphic annotations to include in this presentation state.
        graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
            Graphic layers to include in this presentation state. All graphic
            layers referenced in "graphic_annotations" must be included.
        graphic_groups: Optional[Sequence[highdicom.pr.GraphicGroup]], optional
            Description of graphic groups used in this presentation state.
        concept_name_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
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
        icc_profile: Union[PIL.ImageCms.ImageCmsProfile, None], optional
            ICC color profile object to include in the presentation state. If
            none is provided, a standard RGB ("sRGB") profile will be assumed.
        transfer_syntax_uid: Union[str, highdicom.UID], optional
            Transfer syntax UID of the presentation state.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        for kw in [
            'rescale_intercept',
            'rescale_slope',
            'rescale_type',
            'modality_lut',
            'softcopy_voi_luts',
            'copy_modality_lut',
            'copy_voi_lut',
        ]:
            if kw in kwargs:
                raise TypeError(
                    'ColorSoftcopyPresentationState() got an unexpected '
                    f'keyword argument "{kw}".'
                )
        super().__init__(
            referenced_images=referenced_images,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=ColorSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            content_label=content_label,
            content_description=content_description,
            graphic_annotations=graphic_annotations,
            graphic_layers=graphic_layers,
            graphic_groups=graphic_groups,
            concept_name_code=concept_name_code,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification,
            rescale_intercept=None,
            rescale_slope=None,
            rescale_type=None,
            modality_lut=None,
            softcopy_voi_luts=None,
            copy_modality_lut=False,
            copy_voi_lut=False,
            icc_profile=icc_profile,
            transfer_syntax_uid=transfer_syntax_uid,
            **kwargs
        )
