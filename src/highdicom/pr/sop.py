"""Module for SOP Classes of Presentation State (PR) IODs."""
from collections import defaultdict
import datetime
import logging
import pkgutil
from io import BytesIO
from typing import Optional, Sequence, Tuple, Union

from PIL.ImageCms import ImageCmsProfile

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
    LUT,
    ModalityLUT,
    PaletteColorLookupTable,
    ReferencedImageSequence,
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
from highdicom.utils import is_tiled_image
from highdicom.valuerep import (
    check_person_name,
    _check_code_string,
    _check_long_string
)


logger = logging.getLogger(__name__)


def _add_equipment_attributes(
    dataset: Dataset,
    manufacturer: str,
    manufacturer_model_name: str,
    software_versions: Union[str, Tuple[str]],
    device_serial_number: str,
    institution_name: Optional[str] = None,
    institutional_department_name: Optional[str] = None,
) -> None:
    """Add attributes of module General Equipment.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
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
    institution_name: Union[str, None], optional
        Name of the institution of the person or device that creates the
        SR document instance.
    institutional_department_name: Union[str, None], optional
        Name of the department of the person or device that creates the
        SR document instance.

    """
    dataset.Manufacturer = manufacturer
    if institution_name is not None:
        dataset.InstitutionName = institution_name
        if institutional_department_name is not None:
            dataset.InstitutionalDepartmentName = institutional_department_name
    dataset.DeviceSerialNumber = device_serial_number
    dataset.ManufacturerModelName = manufacturer_model_name
    dataset.SoftwareVersions = software_versions

    # Not technically part of PR IODs, but we include anyway
    now = datetime.datetime.now()
    dataset.ContentDate = DA(now.date())
    dataset.ContentTime = TM(now.time())


def _add_presentation_state_identification_attributes(
    dataset: Dataset,
    content_label: str,
    content_description: Optional[str] = None,
    concept_name: Union[Code, CodedConcept, None] = None,
    content_creator_name: Optional[Union[str, PersonName]] = None,
    content_creator_identification: Optional[
        ContentCreatorIdentificationCodeSequence
    ] = None,
) -> None:
    """Add attributes of module Presentation State Identification.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    content_label: str
        A label used to describe the content of this presentation state.
        Must be a valid DICOM code string consisting only of capital
        letters, underscores and spaces.
    content_description: Union[str, None], optional
        Description of the content of this presentation state.
    concept_name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
        A coded description of the content of this presentation state.
    content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
        Name of the person who created the content of this presentation
        state.
    content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
        Identifying information for the person who created the content of
        this presentation state.

    """  # noqa: E501
    _check_code_string(content_label)
    dataset.ContentLabel = content_label
    if content_description is not None and len(content_description) > 64:
        raise ValueError(
            'Argument "content_description" must not exceed 64 characters.'
        )
    dataset.ContentDescription = content_description
    now = datetime.datetime.now()
    dataset.PresentationCreationDate = DA(now.date())
    dataset.PresentationCreationTime = TM(now.time())

    if concept_name is not None:
        if not isinstance(concept_name, (Code, CodedConcept)):
            raise TypeError(
                'Argument "concept_name" should be of type '
                'pydicom.sr.coding.Code or '
                'highdicom.sr.CodedConcept.'
            )
        dataset.ConceptNameCodeSequence = [
            CodedConcept(
                concept_name.value,
                concept_name.scheme_designator,
                concept_name.meaning,
                concept_name.scheme_version
            )
        ]

    if content_creator_name is not None:
        check_person_name(content_creator_name)
    dataset.ContentCreatorName = content_creator_name

    if content_creator_identification is not None:
        if not isinstance(
            content_creator_identification,
            ContentCreatorIdentificationCodeSequence
        ):
            raise TypeError(
                'Argument "content_creator_identification" must be of type '
                'ContentCreatorIdentificationCodeSequence.'
            )
        dataset.ContentCreatorIdentificationCodeSequence = \
            content_creator_identification


def _add_presentation_state_relationship_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset]
) -> None:
    """Add attributes of module Presentation State Relationship.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced

    """
    # Assert referenced images are from the same series and have the same size
    ref_im = referenced_images[0]
    ref_series_uid = ref_im.SeriesInstanceUID
    ref_im_seq = []
    for im in referenced_images:
        series_uid = im.SeriesInstanceUID
        if series_uid != ref_series_uid:
            raise ValueError(
                'All referenced images must belong to the same series.'
            )
        if im.Rows != ref_im.Rows or im.Columns != ref_im.Columns:
            raise ValueError(
                'All referenced images must have the same dimensions.'
            )
        if is_tiled_image(ref_im):
            if (
                im.TotalPixelMatrixRows != ref_im.TotalPixelMatrixRows or
                im.TotalPixelMatrixColumns != ref_im.TotalPixelMatrixColumns
            ):
                raise ValueError(
                    'All referenced images must have the same total pixel '
                    'matrix dimensions.'
                )
        ref_im_item = Dataset()
        ref_im_item.ReferencedSOPClassUID = im.SOPClassUID
        ref_im_item.ReferencedSOPInstanceUID = im.SOPInstanceUID
        ref_im_seq.append(ref_im_item)

    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = ref_series_uid
    ref_series_item.ReferencedImageSequence = ref_im_seq
    dataset.ReferencedSeriesSequence = [ref_series_item]


def _add_graphic_group_annotation_layer_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset],
    graphic_groups: Optional[Sequence[GraphicGroup]] = None,
    graphic_annotations: Optional[Sequence[GraphicAnnotation]] = None,
    graphic_layers: Optional[Sequence[GraphicLayer]] = None
) -> None:
    """Add attributes of modules Graphic Group/Annotation/Layer.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced
    graphic_groups: Union[Sequence[highdicom.pr.GraphicGroup], None], optional
        Description of graphic groups used in this presentation state.
    graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
        Graphic annotations to include in this presentation state.
    graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
        Graphic layers to include in this presentation state. All graphic
        layers referenced in "graphic_annotations" must be included.

    """  # noqa: E501
    # Graphic Group
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
        dataset.GraphicGroupSequence = graphic_groups
    else:
        described_groups_ids = set()

    # Graphic Annotation and Graphic Layer
    ref_images_lut = {
        (ds.SOPClassUID, ds.SOPInstanceUID): ds
        for ds in referenced_images
    }
    if graphic_layers is not None:
        labels = [layer.GraphicLayer for layer in graphic_layers]
        if len(labels) != len(set(labels)):
            raise ValueError(
                'Labels of graphic layers must be unique.'
            )
        labels_unique = set(labels)
        dataset.GraphicLayerSequence = graphic_layers

    if graphic_annotations is not None:
        for i, ann in enumerate(graphic_annotations):
            if not isinstance(ann, GraphicAnnotation):
                raise TypeError(
                    f'Item #{i} of "graphic_annotations" must be of type '
                    'highdicom.pr.GraphicAnnotation.'
                )
            if ann.GraphicLayer not in labels_unique:
                raise ValueError(
                    f'Graphic layer with name "{ann.GraphicLayer}" is '
                    f'referenced in item #{i} of "graphic_annotations", '
                    'but not included "graphic_layers".'
                )
            for item in ann.ReferencedImageSequence:
                uids = (
                    item.ReferencedSOPClassUID,
                    item.ReferencedSOPInstanceUID
                )
                if uids not in ref_images_lut:
                    raise ValueError(
                        f'Instance with SOP Instance UID {uids[1]} and '
                        f'SOP Class UID {uids[0]} is referenced in item #{i} '
                        f'of "graphic_annotations", but not included '
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
        dataset.GraphicAnnotationSequence = graphic_annotations


def _add_displayed_area_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset],
) -> None:
    """Add attributes of module Displayed Area.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced

    """
    # This implements the simplest case - the entire area is selected for
    # display and the selection applies to all referenced images.
    # We may want to generalize this later.
    ref_im = referenced_images[0]
    display_area_item = Dataset()
    display_area_item.PixelOriginInterpretation = 'VOLUME'
    display_area_item.DisplayedAreaTopLeftHandCorner = [1, 1]
    display_area_item.DisplayedAreaBottomRightHandCorner = [
        ref_im.Columns,
        ref_im.Rows,
    ]
    display_area_item.PresentationSizeMode = 'SCALE TO FIT'
    display_area_item.PresentationPixelAspectRatio = [1, 1]
    dataset.DisplayedAreaSelectionSequence = [display_area_item]


def _add_modality_lut_attributes(
    dataset: Dataset,
    rescale_intercept: Union[int, float, None] = None,
    rescale_slope: Union[int, float, None] = None,
    rescale_type: Union[RescaleTypeValues, str, None] = None,
    modality_lut: Optional[ModalityLUT] = None,
) -> None:
    """Add attributes of module Modality LUT.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    rescale_intercept: Union[int, float, None], optional
        Intercept of linear function used for rescaling pixel values.
    rescale_slope: Union[int, float, None], optional
        Slope of linear function used for rescaling pixel values.
    rescale_type: Union[highdicom.RescaleTypeValues, str, None], optional
        String or enumerated value specifying the units of the output of
        the Modality LUT or rescale operation.
    modality_lut: Union[highdicom.ModalityLUT, None], optional
        Lookup table specifying a pixel rescaling operation to apply to
        the stored values to give modality values.

    Note
    ----
    Either `modality_lut` may be specified or all three of `rescale_slope`,
    `rescale_intercept` and `rescale_type` may be specified. All four
    parameters should not be specified simultaneously. All parameters may be
    ``None`` if there is no modality LUT to apply to the images.

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
        dataset.ModalityLUTSequence = [modality_lut]
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
        dataset.RescaleIntercept = format_number_as_ds(rescale_intercept)
        dataset.RescaleSlope = format_number_as_ds(rescale_slope)
        if isinstance(rescale_type, RescaleTypeValues):
            dataset.RescaleType = rescale_type.value
        else:
            _check_long_string(rescale_type)
            dataset.RescaleType = rescale_type
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


def _get_modality_lut_attributes(
    referenced_images: Sequence[Dataset]
) -> Dataset:
    """Get attributes of module Modality LUT from the referenced images.

    Parameters
    ----------
    referenced_images: Sequence[pydicom.Dataset]
        The referenced images from which the attributes should be copied.

    Returns
    -------
    pydicom.Dataset
        Dataset containing attributes of module Modality LUT

    Raises
    ------
    ValueError
        In case the presence or value of the RescaleSlope, RescaleIntercept,
        or RescaleType attributes are inconsistent between referenced images.

    """
    # Multframe images
    dataset = Dataset()
    if any(hasattr(im, 'NumberOfFrames') for im in referenced_images):
        if len(referenced_images) > 1:
            raise ValueError(
                "Attributes of Modality LUT module are not available when "
                "multiple images are passed and any of them are multiframe."
            )

        im = referenced_images[0]

        # Check only the Shared Groups, as PRs require all frames to have
        # the same Modality LUT
        slope = None
        intercept = None
        rescale_type = None
        shared_grps = im.SharedFunctionalGroupsSequence[0]
        if hasattr(shared_grps, 'PixelValueTransformationSequence'):
            trans_seq = shared_grps.PixelValueTransformationSequence[0]
            if hasattr(trans_seq, 'RescaleSlope'):
                slope = trans_seq.RescaleSlope
            if hasattr(trans_seq, 'RescaleIntercept'):
                intercept = trans_seq.RescaleIntercept
            if hasattr(trans_seq, 'RescaleType'):
                rescale_type = trans_seq.RescaleType

        # Modality LUT data in the Per Frame Functional Groups will not
        # be copied, but we should check for it rather than silently
        # failing to copy it
        if hasattr(im, 'PerFrameFunctionalGroupsSequence'):
            perframe_grps = im.PerFrameFunctionalGroupsSequence
            if any(
                hasattr(frm_grps, 'PixelValueTransformationSequence')
                for frm_grps in perframe_grps
            ):
                raise ValueError(
                    'This multiframe image contains modality LUT '
                    'table data in the Per-Frame Functional Groups '
                    'Sequence. This is not compatible with the '
                    'Modality LUT module.'
                )

        dataset.RescaleIntercept = intercept
        dataset.RescaleSlope = slope
        dataset.RescaleType = rescale_type

    else:
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
                '"RescaleIntercept" is inconsistent among referenced '
                'images.'
            )
        if any(have_type) and not all(have_type):
            raise ValueError(
                'Error while copying Modality LUT attributes: presence of '
                '"RescaleType" is inconsistent among referenced images.'
            )

        if all(have_intercepts) != all(have_slopes):
            raise ValueError(
                'Error while copying Modality LUT attributes: datasets '
                'should have both "RescaleIntercept" and "RescaleSlope", '
                'or neither.'
            )

        if all(have_intercepts):
            if any(
                ds.RescaleSlope != referenced_images[0].RescaleSlope
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values '
                    'of "RescaleSlope" are inconsistent among referenced '
                    'images.'
                )
            if any(
                ds.RescaleIntercept != referenced_images[0].RescaleIntercept
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values '
                    'of "RescaleIntercept" are inconsistent among '
                    'referenced images.'
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
                    'Error while copying Modality LUT attributes: values '
                    'of "RescaleType" are inconsistent among referenced '
                    'images.'
                )
            rescale_type = referenced_images[0].RescaleType
        else:
            if intercept is None:
                rescale_type = None
            else:
                rescale_type = RescaleTypeValues.HU.value

        dataset.RescaleIntercept = intercept
        dataset.RescaleSlope = slope
        dataset.RescaleType = rescale_type

    return dataset


def _add_softcopy_voi_lut_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset],
    softcopy_voi_luts: Sequence[SoftcopyVOILUT]
) -> None:
    """Add attributes of module Softcopy VOI LUT.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced
    softcopy_voi_luts: Sequence[highdicom.pr.SoftcopyVOILUT]
        One or more pixel value-of-interest operations to be applied after
        the modality LUT and/or rescale operation. Note that multiple
        items should only be provided if no image, or frame within a
        multi-frame image, is referenced by more than one item.

    """
    if len(softcopy_voi_luts) == 0:
        raise ValueError('Argument "softcopy_voi_luts" must not be empty.')
    for i, v in enumerate(softcopy_voi_luts):
        if not isinstance(v, SoftcopyVOILUT):
            raise TypeError(
                f'Item #{i} of "softcopy_voi_luts" must have type '
                'SoftcopyVOILUT.'
            )

    if len(softcopy_voi_luts) > 1:
        if not all(
            hasattr(v, 'ReferencedImageSequence')
            for v in softcopy_voi_luts
        ):
            raise ValueError(
                'If multiple items are passed in "softcopy_voi_luts", '
                'each must specify the images that it applies to.'
            )

    ref_images_lut = {
        (ds.SOPClassUID, ds.SOPInstanceUID): ds
        for ds in referenced_images
    }
    prev_ref_frames = defaultdict(list)
    prev_ref_segs = defaultdict(list)
    for v in softcopy_voi_luts:
        # If the softcopy VOI LUT references specific images,
        # check that the references are valid
        if hasattr(v, 'ReferencedImageSequence'):
            for item in v.ReferencedImageSequence:
                uids = (
                    item.ReferencedSOPClassUID,
                    item.ReferencedSOPInstanceUID
                )
                if uids not in ref_images_lut:
                    raise ValueError(
                        f'Instance with SOP Instance UID {uids[1]} and '
                        f'SOP Class UID {uids[0]} is referenced in '
                        'items of "softcopy_voi_luts", but not '
                        'included in "referenced_images".'
                    )
                ref_im = ref_images_lut[uids]
                is_multiframe = hasattr(
                    ref_im,
                    'NumberOfFrames',
                )
                if uids in prev_ref_frames and not is_multiframe:
                    raise ValueError(
                        f'Instance with SOP Instance UID {uids[1]} '
                        'is referenced in more than one item of the '
                        '"softcopy_voi_luts".'
                    )
                nframes = getattr(ref_im, 'NumberOfFrames', 1)
                if hasattr(item, 'ReferencedFrameNumber'):
                    ref_frames = item.ReferencedFrameNumber
                    if not isinstance(ref_frames, list):
                        ref_frames = [ref_frames]
                    for f in ref_frames:
                        if f > nframes:
                            raise ValueError(
                                f'Frame {f} in image with SOP Instance '
                                f'UID {uids[1]} is referenced in an '
                                'item of the "softcopy_voi_luts" but '
                                'is not a valid frame number in that '
                                'image.'
                            )
                else:
                    # If ReferencedFrameNumber is not present, the
                    # reference refers to all frames
                    ref_frames = list(range(1, nframes))

                for f in ref_frames:
                    if f in prev_ref_frames[uids]:
                        raise ValueError(
                            f'Frame {f} in image with SOP Instance '
                            f'UID {uids[1]} is referenced in more '
                            'than one item of the '
                            '"softcopy_voi_luts".'
                        )
                    prev_ref_frames[uids].append(f)

                if hasattr(item, 'ReferencedSegmentNumber'):
                    try:
                        nsegments = ref_im.NumberOfSegments
                    except AttributeError:
                        raise ValueError(
                            'An item of "softcopy_voi_luts" references '
                            'segments of the image with SOP Instance '
                            f'UID {uids[1]}, but this image does not '
                            'contain segments.'
                        )

                    ref_segs = item.ReferencedSegmentNumber
                    if not isinstance(ref_segs, list):
                        ref_segs = [ref_segs]
                    for s in ref_segs:
                        if s > nsegments:
                            raise ValueError(
                                f'Segment {s} in image with SOP '
                                f'Instance UID {uids[1]} is referenced '
                                'in an item of the "softcopy_voi_luts" '
                                'but is not a valid segment number in '
                                'that image.'
                            )

                if hasattr(ref_im, 'NumberOfSegments'):
                    if not hasattr(item, 'ReferencedSegmentNumber'):
                        ref_segs = list(range(1, nsegments))
                    if s in prev_ref_segs[uids]:
                        raise ValueError(
                            f'Segment {s} in image with SOP '
                            f'Instance  UID {uids[1]} is '
                            'referenced in more than one item of '
                            'the "softcopy_voi_luts".'
                        )
                    prev_ref_segs[uids].append(s)

    dataset.SoftcopyVOILUTSequence = softcopy_voi_luts


def _get_softcopy_voi_lut_attributes(
    referenced_images: Sequence[Dataset]
) -> Dataset:
    """Get attributes of module Softcopy VOI LUT from referenced images.

    Any Window Center, Window Width, Window Explanation, VOI LUT Function,
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

    Returns
    -------
    pydicom.Dataset
        Dataset containing attributes of module Softcopy VOI LUT

    """
    dataset = Dataset()
    if any(hasattr(im, 'NumberOfFrames') for im in referenced_images):
        if len(referenced_images) > 1:
            raise ValueError(
                "The 'copy_voi_lut' option is not available when "
                "multiple images are passed and any of them are multiframe."
            )

        im = referenced_images[0]
        shared_grps = im.SharedFunctionalGroupsSequence[0]
        perframe_grps = im.PerFrameFunctionalGroupsSequence
        if hasattr(shared_grps, 'FrameVOILUTSequence'):
            # Simple case where VOI information is in the Shared functional
            # groups and therefore are consistent between frames
            voi_seq = shared_grps.FrameVOILUTSequence[0]

            softcopy_voi_lut = SoftcopyVOILUT(
                window_center=voi_seq.WindowCenter,
                window_width=voi_seq.WindowWidth,
                window_explanation=getattr(
                    voi_seq,
                    'WindowCenterWidthExplanation',
                    None
                ),
                voi_lut_function=getattr(voi_seq, 'VOILUTFunction', None),
            )
            dataset.SoftcopyVOILUTSequence = [softcopy_voi_lut]

        else:
            # Check the per-frame functional groups, which may be
            # inconsistent between frames and require multiple entries
            # in the GSPS SoftcopyVOILUTSequence
            by_window = defaultdict(list)
            for frame_number, frm_grp in enumerate(perframe_grps, 1):
                if hasattr(frm_grp, 'FrameVOILUTSequence'):
                    voi_seq = frm_grp.FrameVOILUTSequence[0]
                    # Create unique ID for this VOI lookup as a tuple
                    # of the contents
                    by_window[(
                        voi_seq.WindowWidth,
                        voi_seq.WindowCenter,
                        getattr(
                            voi_seq,
                            'WindowCenterWidthExplanation',
                            None
                        ),
                        getattr(voi_seq, 'VOILUTFunction', None),
                    )].append(frame_number)

            softcopy_voi_luts = []
            for (width, center, exp, func), frame_list in by_window.items():
                if len(frame_list) == im.NumberOfFrames:
                    # All frames included, no need to include the
                    # referenced frames explicitly
                    refs_to_include = None
                else:
                    # Include specific references
                    refs_to_include = ReferencedImageSequence(
                        referenced_images=referenced_images,
                        referenced_frame_number=frame_list,
                    )

                softcopy_voi_luts.append(
                    SoftcopyVOILUT(
                        window_center=center,
                        window_width=width,
                        window_explanation=exp,
                        voi_lut_function=func,
                        referenced_images=refs_to_include
                    )
                )

            dataset.SoftcopyVOILUTSequence = softcopy_voi_luts

    else:  # single frame
        by_window = defaultdict(list)
        by_lut = defaultdict(list)
        for ref_im in referenced_images:
            has_width = hasattr(ref_im, 'WindowWidth')
            has_center = hasattr(ref_im, 'WindowCenter')
            has_lut = hasattr(ref_im, 'VOILUTSequence')

            if has_width != has_center:
                raise ValueError(
                    'Error while copying VOI LUT attributes: found dataset '
                    'with mismatched WindowWidth and WindowCenter '
                    'attributes.'
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

        softcopy_voi_luts = []
        for (width, center, exp, func), im_list in by_window.items():
            if len(im_list) == len(referenced_images):
                # All datasets included, no need to include the referenced
                # images explicitly
                refs_to_include = None
            else:
                # Include specific references
                refs_to_include = ReferencedImageSequence(im_list)

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
                refs_to_include = ReferencedImageSequence(im_list)

            luts = [
                LUT(
                    first_mapped_value=fmv,
                    lut_data=np.frombuffer(
                        data,
                        np.uint8 if ba == 8 else np.uint16
                    ),
                    lut_explanation=exp
                )
                for (fmv, ba, exp, data) in lut_id
            ]
            softcopy_voi_luts.append(
                SoftcopyVOILUT(
                    referenced_images=refs_to_include,
                    luts=luts
                )
            )

        dataset.SoftcopyVOILUTSequence = softcopy_voi_luts

    return dataset


def _get_icc_profile_attributes(
    referenced_images: Sequence[Dataset]
) -> Dataset:
    """Get attributes of module ICC Profile from a referenced image.

    Parameters
    ----------
    referenced_images: Sequence[pydicom.Dataset]
        Image datasets from which to extract an ICC profile

    Returns
    -------
    pydicom.Dataset
        Dataset containing attributes of module ICC Profile

    Raises
    ------
    ValueError:
        When no ICC profile is found in any of the referenced images or if
        more than one unique profile is found.

    """
    icc_profiles = []
    for im in referenced_images:
        if hasattr(referenced_images, 'ICCProfile'):
            icc_profiles.append(im.ICCProfile)
        elif hasattr(im, 'OpticalPathSequence'):
            if len(im.OpticalPathSequence) > 1:
                raise ValueError(
                    'Cannot extract ICC Profile from referenced image. '
                    'Color image is expected to contain only a single optical '
                    'path.'
                )
            icc_profiles.append(im.OpticalPathSequence[0].ICCProfile)

    if len(icc_profiles) == 0:
        raise ValueError(
            'Could not find an ICC Profile in any of the referenced images.'
        )
    if len(set(icc_profiles)) > 1:
        raise ValueError(
            'Found more than one ICC Profile in referenced images.'
        )

    dataset = Dataset()
    dataset.ICCProfile = icc_profiles[0]
    return dataset


def _add_icc_profile_attributes(
    dataset: Dataset,
    icc_profile: bytes
) -> None:
    """Add attributes of module ICC Profile.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    icc_profile: bytes
        ICC color profile to include in the presentation state.
        The profile must follow the constraints listed in :dcm:`C.11.15
        <part03/sect_C.11.15.html>`.

    """
    if icc_profile is None:
        raise TypeError('Argument "icc_profile" is required.')

    cms_profile = ImageCmsProfile(BytesIO(icc_profile))
    device_class = cms_profile.profile.device_class.strip()
    if device_class not in ('scnr', 'spac'):
        raise ValueError(
            'The device class of the ICC Profile must be "scnr" or "spac", '
            f'got "{device_class}".'
        )
    color_space = cms_profile.profile.xcolor_space.strip()
    if color_space != 'RGB':
        raise ValueError(
            'The color space of the ICC Profile must be "RGB", '
            f'got "{color_space}".'
        )
    pcs = cms_profile.profile.connection_space.strip()
    if pcs not in ('Lab', 'XYZ'):
        raise ValueError(
            'The profile connection space of the ICC Profile must '
            f'be "Lab" or "XYZ", got "{pcs}".'
        )

    dataset.ICCProfile = icc_profile


def _add_palette_color_lookup_table_attributes(
    dataset: Dataset,
    palette_color_lut: PaletteColorLookupTable
) -> None:
    """Add attributes from the Palette Color Lookup Table module.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    palette_color_lut: highdicom.content.PaletteColorLookupTable
        A palette color lookup table to apply to the image.

    """
    if not isinstance(palette_color_lut, PaletteColorLookupTable):
        raise TypeError(
            'Argument "palette_color_lut" must be of type '
            'PaletteColorLookupTable.'
        )
    colors = ['Red', 'Green', 'Blue']

    for color in colors:
        desc_kw = f'{color}PaletteColorLookupTableDescriptor'
        data_kw = f'{color}PaletteColorLookupTableData'
        desc = getattr(palette_color_lut, desc_kw)
        lut = getattr(palette_color_lut, data_kw)

        setattr(dataset, desc_kw, desc)
        setattr(dataset, data_kw, lut)

    if hasattr(palette_color_lut, 'PaletteColorLookupTableUID'):
        uid = palette_color_lut.PaletteColorLookupTableUID
        dataset.PaletteColorLookupTableUID = uid


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
        rescale_intercept: Union[int, float, None] = None,
        rescale_slope: Union[int, float, None] = None,
        rescale_type: Union[RescaleTypeValues, str, None] = None,
        modality_lut: Optional[ModalityLUT] = None,
        softcopy_voi_luts: Optional[Sequence[SoftcopyVOILUT]] = None,
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
        referenced_images: Sequence[pydicom.Dataset]
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
        softcopy_voi_luts: Union[Sequence[highdicom.pr.SoftcopyVOILUT], None], optional
            One or more pixel value-of-interest operations to be applied after
            the modality LUT and/or rescale operation. Note that multiple
            items should only be provided if no image, or frame within a
            multi-frame image, is referenced by more than one item.
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
        for ref_im in referenced_images:
            if ref_im.SamplesPerPixel != 1:
                raise ValueError(
                    'For grayscale presentation states, all referenced images '
                    'must have a single sample per pixel.'
                )

        super().__init__(
            study_instance_uid=ref_im.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=GrayscaleSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
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
            **kwargs
        )
        self.copy_patient_and_study_information(ref_im)
        self.copy_specimen_information(ref_im)

        # General Equipment
        _add_equipment_attributes(
            self,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name
        )

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
        if modality_lut is not None:
            _add_modality_lut_attributes(
                self,
                modality_lut=modality_lut
            )
        else:
            were_rescale_attributes_provided = [
                rescale_intercept is not None,
                rescale_slope is not None,
                rescale_type is not None,
            ]
            if all(were_rescale_attributes_provided):
                _add_modality_lut_attributes(
                    self,
                    rescale_intercept=rescale_intercept,
                    rescale_slope=rescale_slope,
                    rescale_type=rescale_type
                )
            elif any(were_rescale_attributes_provided):
                raise TypeError(
                    'Arguments "rescale_intercept", "rescale_slope", and '
                    '"rescale_type" must either all be provided or none of '
                    'them shall be provided.'
                )
            else:
                try:
                    ds = _get_modality_lut_attributes(referenced_images)
                except (ValueError, AttributeError):
                    logger.debug(
                        'no Modality LUT attributes found in referenced images'
                    )
                logger.debug(
                    'use Modality LUT attributes from referenced images'
                )
                _add_modality_lut_attributes(
                    self,
                    rescale_intercept=ds.RescaleIntercept,
                    rescale_slope=ds.RescaleSlope,
                    rescale_type=ds.RescaleType
                )

        # Softcopy VOI LUT
        if softcopy_voi_luts is not None:
            if len(softcopy_voi_luts) == 0:
                raise ValueError(
                    'Argument "softcopy_voi_luts" must not be empty.'
                )
            for v in softcopy_voi_luts:
                if not isinstance(v, SoftcopyVOILUT):
                    raise TypeError(
                        'Items of "softcopy_voi_luts" must be of type '
                        'SoftcopyVOILUT.'
                    )

            if len(softcopy_voi_luts) > 1:
                if not all(
                    hasattr(v, 'ReferencedImageSequence')
                    for v in softcopy_voi_luts
                ):
                    raise ValueError(
                        'If multiple items are passed in "softcopy_voi_luts", '
                        'each must specify the images that it applies to.'
                    )
            _add_softcopy_voi_lut_attributes(
                self,
                referenced_images=referenced_images,
                softcopy_voi_luts=softcopy_voi_luts
            )
        else:
            try:
                ds = _get_softcopy_voi_lut_attributes(referenced_images)
            except (AttributeError, ValueError):
                logger.debug(
                    'no VOI LUT attributes found in referenced images'
                )
            if len(ds.SoftcopyVOILUTSequence) > 0:
                logger.debug(
                    'use VOI LUT attributes from referenced images'
                )
                _add_softcopy_voi_lut_attributes(
                    self,
                    referenced_images=referenced_images,
                    softcopy_voi_luts=ds.SoftcopyVOILUTSequence
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
        content_label: str,
        palette_color_lut: PaletteColorLookupTable,
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
        rescale_intercept: Union[int, float, None] = None,
        rescale_slope: Union[int, float, None] = None,
        rescale_type: Union[RescaleTypeValues, str, None] = None,
        modality_lut: Optional[ModalityLUT] = None,
        softcopy_voi_luts: Optional[Sequence[SoftcopyVOILUT]] = None,
        icc_profile: Optional[bytes] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        **kwargs
    ):
        """
        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
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
        palette_color_lut: highdicom.content.PaletteColorLookupTable
            Palette color lookup table to apply to the image.
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
        softcopy_voi_luts: Union[Sequence[highdicom.pr.SoftcopyVOILUT], None], optional
            One or more pixel value-of-interest operations to be applied after
            the modality LUT and/or rescale operation. Note that multiple
            items should only be provided if no image, or frame within a
            multi-frame image, is referenced by more than one item.
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
        for ref_im in referenced_images:
            if ref_im.SamplesPerPixel != 1:
                raise ValueError(
                    'For pseudo-color presentation states, all referenced '
                    'images must have a single sample per pixel.'
                )
        super().__init__(
            study_instance_uid=ref_im.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=PseudoColorSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
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
            **kwargs
        )
        self.copy_patient_and_study_information(ref_im)
        self.copy_specimen_information(ref_im)

        # General Equipment
        _add_equipment_attributes(
            self,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name
        )

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
        if modality_lut is not None:
            _add_modality_lut_attributes(
                self,
                modality_lut=modality_lut
            )
        else:
            were_rescale_attributes_provided = [
                rescale_intercept is not None,
                rescale_slope is not None,
                rescale_type is not None,
            ]
            if all(were_rescale_attributes_provided):
                _add_modality_lut_attributes(
                    self,
                    rescale_intercept=rescale_intercept,
                    rescale_slope=rescale_slope,
                    rescale_type=rescale_type
                )
            elif any(were_rescale_attributes_provided):
                raise TypeError(
                    'Arguments "rescale_intercept", "rescale_slope", and '
                    '"rescale_type" must either all be provided or none of '
                    'them shall be provided.'
                )
            else:
                try:
                    ds = _get_modality_lut_attributes(referenced_images)
                except (ValueError, AttributeError):
                    logger.debug(
                        'no Modality LUT attributes found in referenced images'
                    )
                logger.debug(
                    'use Modality LUT attributes from referenced images'
                )
                _add_modality_lut_attributes(
                    self,
                    rescale_intercept=ds.RescaleIntercept,
                    rescale_slope=ds.RescaleSlope,
                    rescale_type=ds.RescaleType
                )

        # Softcopy VOI LUT
        if softcopy_voi_luts is not None:
            if len(softcopy_voi_luts) == 0:
                raise ValueError(
                    'Argument "softcopy_voi_luts" must not be empty.'
                )
            for v in softcopy_voi_luts:
                if not isinstance(v, SoftcopyVOILUT):
                    raise TypeError(
                        'Items of "softcopy_voi_luts" must be of type '
                        'SoftcopyVOILUT.'
                    )

            if len(softcopy_voi_luts) > 1:
                if not all(
                    hasattr(v, 'ReferencedImageSequence')
                    for v in softcopy_voi_luts
                ):
                    raise ValueError(
                        'If multiple items are passed in "softcopy_voi_luts", '
                        'each must specify the images that it applies to.'
                    )
            _add_softcopy_voi_lut_attributes(
                self,
                referenced_images=referenced_images,
                softcopy_voi_luts=softcopy_voi_luts
            )
        else:
            try:
                ds = _get_softcopy_voi_lut_attributes(referenced_images)
            except (AttributeError, ValueError):
                logger.debug(
                    'no VOI LUT attributes found in referenced images'
                )
            if len(ds.SoftcopyVOILUTSequence) > 0:
                logger.debug(
                    'use VOI LUT attributes from referenced images'
                )
                _add_softcopy_voi_lut_attributes(
                    self,
                    referenced_images=referenced_images,
                    softcopy_voi_luts=ds.SoftcopyVOILUTSequence
                )

        # Palette Color Lookup Table
        _add_palette_color_lookup_table_attributes(
            self,
            palette_color_lut=palette_color_lut
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
        for kw in [
            'rescale_intercept',
            'rescale_slope',
            'rescale_type',
            'modality_lut',
            'softcopy_voi_luts',
        ]:
            if kw in kwargs:
                raise TypeError(
                    'ColorSoftcopyPresentationState() got an unexpected '
                    f'keyword argument "{kw}".'
                )
        for ref_im in referenced_images:
            if ref_im.SamplesPerPixel != 3:
                raise ValueError(
                    'For color presentation states, all referenced '
                    'images must have three samples per pixel.'
                )
        super().__init__(
            study_instance_uid=ref_im.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=ColorSoftcopyPresentationStateStorage,
            instance_number=instance_number,
            manufacturer=manufacturer,
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
            **kwargs
        )
        self.copy_patient_and_study_information(ref_im)
        self.copy_specimen_information(ref_im)

        # General Equipment
        _add_equipment_attributes(
            self,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name
        )

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
            ds = _get_icc_profile_attributes(referenced_images)
            _add_icc_profile_attributes(
                self,
                icc_profile=ds.ICCProfile
            )
