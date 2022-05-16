"""DICOM structured reporting templates."""
import logging
from copy import deepcopy
from typing import cast, Iterable, List, Optional, Sequence, Tuple, Union

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes

from highdicom.sr.coding import CodedConcept
from highdicom.sr.content import (
    FindingSite,
    LongitudinalTemporalOffsetFromEvent,
    ImageRegion,
    ImageRegion3D,
    VolumeSurface,
    RealWorldValueMap,
    ReferencedSegment,
    ReferencedSegmentationFrame,
    SourceImageForMeasurementGroup,
    SourceImageForMeasurement,
    SourceImageForSegmentation,
    SourceSeriesForSegmentation
)
from highdicom.sr.enum import (
    GraphicTypeValues,
    GraphicTypeValues3D,
    RelationshipTypeValues,
    ValueTypeValues,
)
from highdicom.uid import UID
from highdicom.sr.utils import find_content_items, get_coded_name
from highdicom.sr.value_types import (
    CodeContentItem,
    ContainerContentItem,
    ContentItem,
    ContentSequence,
    ImageContentItem,
    NumContentItem,
    PnameContentItem,
    TextContentItem,
    UIDRefContentItem,
)


# Codes missing from pydicom
DEFAULT_LANGUAGE = CodedConcept(
    value='en-US',
    scheme_designator='RFC5646',
    meaning='English (United States)'
)
_REGION_IN_SPACE = Code('130488', 'DCM', 'Region in Space')
_SOURCE = CodedConcept(
    value='260753009',
    scheme_designator='SCT',
    meaning='Source',
)


logger = logging.getLogger(__name__)


def _count_roi_items(
    group_item: ContainerContentItem
) -> Tuple[int, int, int, int, int]:
    """Find content items in a 'Measurement Group' container content item
    structured according to TID 1410 Planar ROI Measurements and Qualitative
    Evaluations or TID 1411 Volumetric ROI Measurements and Qualitative
    Evaluations.

    Parameters
    ----------
    group_item: highdicom.sr.ContainerContentItem
        Container content item of measurement group that is expected to
        contain ROI items

    Returns
    -------
    int:
        Number of 'Image Region' content items
    int:
        Number of 'Volume Surface' content items
    int:
        Number of 'Referenced Segment' content items
    int:
        Number of 'Referenced Segmentation Frame' content items
    int:
        Number of 'Region in Space' content items

    """
    if group_item.ValueType != ValueTypeValues.CONTAINER.value:
        raise ValueError(
            'SR Content Item does not represent a measurement group '
            'because it does not have value type CONTAINER.'
        )
    if group_item.name == codes.DCM.MeasurementGroup:
        raise ValueError(
            'SR Content Item does not represent a measurement group '
            'because it does not have name "Measurement Group".'
        )
    n_image_region_items = 0
    n_volume_surface_items = 0
    n_referenced_segment_items = 0
    n_referenced_segmentation_frame_items = 0
    n_region_in_space_items = 0
    for item in group_item.ContentSequence:
        if (item.name == codes.DCM.ImageRegion and
                (item.value_type == ValueTypeValues.SCOORD or
                 item.value_type == ValueTypeValues.SCOORD3D)):
            n_image_region_items += 1
        if (item.name == codes.DCM.VolumeSurface and
                item.value_type == ValueTypeValues.SCOORD3D):
            n_volume_surface_items += 1
        if (item.name == codes.DCM.ReferencedSegment and
                item.value_type == ValueTypeValues.IMAGE):
            n_referenced_segment_items += 1
        if (item.name == codes.DCM.ReferencedSegmentationFrame and
                item.value_type == ValueTypeValues.IMAGE):
            n_referenced_segmentation_frame_items += 1
        if (item.name == _REGION_IN_SPACE and
                item.value_type == ValueTypeValues.COMPOSITE):
            n_region_in_space_items += 1
    return (
        n_image_region_items,
        n_volume_surface_items,
        n_referenced_segment_items,
        n_referenced_segmentation_frame_items,
        n_region_in_space_items,
    )


def _contains_planar_rois(group_item: ContainerContentItem) -> bool:
    """Checks whether a measurement group item contains planar ROIs.

    Parameters
    ----------
    group_item: highdicom.sr.ContainerContentItem
        SR Content Item representing a "Measurement Group"

    Returns
    -------
    bool
        Whether the `group_item` contains any content items with value type
        SCOORD, SCOORD3D, IMAGE, or COMPOSITE representing planar ROIs

    """
    n_image_region_items, n_volume_surface_items, n_referenced_segment_items, \
        n_referenced_segmentation_frame_items, n_region_in_space_items = \
        _count_roi_items(group_item)

    if (
            n_image_region_items == 1 or
            n_referenced_segmentation_frame_items > 0 or
            n_region_in_space_items == 1
       ) and (
            n_volume_surface_items == 0 and
            n_referenced_segment_items == 0
       ):
        return True
    return False


def _contains_volumetric_rois(group_item: ContainerContentItem) -> bool:
    """Checks whether a measurement group item contains volumetric ROIs.

    Parameters
    ----------
    group_item: highdicom.sr.ContainerContentItem
        SR Content Item representing a "Measurement Group"

    Returns
    -------
    bool
        Whether the `group_item` contains any content items with value type
        SCOORD, SCOORD3D, IMAGE, or COMPOSITE representing volumetric ROIs

    """
    n_image_region_items, n_volume_surface_items, n_referenced_segment_items, \
        n_referenced_segmentation_frame_items, n_region_in_space_items = \
        _count_roi_items(group_item)

    if (
            n_image_region_items > 1 or
            n_referenced_segment_items > 0 or
            n_volume_surface_items > 0 or
            n_region_in_space_items > 0
       ) and (
            n_referenced_segmentation_frame_items == 0
       ):
        return True
    return False


def _get_planar_roi_reference_item(
    group_item: ContainerContentItem,
) -> Tuple[Code, ContentItem]:
    """Get the content item representing a planar measurement group's ROI.

    Parameters
    ----------
    group_item: highdicom.sr.ContainerContentItem
        SR Content Item representing a "Planar ROI Measurement Group"

    Returns
    -------
    highdicom.sr.CodedConcept
        Coded concept representing the concept name of the ROI reference item.
    highdicom.sr.ContentItem
        Content item that defines the reference to the ROI.

    """
    reference_type, items = _get_roi_reference_items(
        group_item,
        PlanarROIMeasurementsAndQualitativeEvaluations._allowed_roi_reference_types  # noqa: E501
    )
    if len(items) > 1:
        raise RuntimeError(
            'Multiple reference items were found in the planar ROI '
            'measurements group.'
        )
    return reference_type, items[0]


def _get_volumetric_roi_reference_items(
    group_item: ContainerContentItem,
) -> Tuple[Code, List[ContentItem]]:
    """Get the content items representing a volumetric measurement group's ROI.

    Parameters
    ----------
    group_item: highdicom.sr.ContainerContentItem
        SR Content Item representing a "Volumetric ROI Measurement Group"

    Returns
    -------
    highdicom.sr.CodedConcept
        Coded concept representing the concept name of the ROI reference item.
    List[highdicom.sr.ContentItem]
        Content items that defines the reference to the ROI.

    """
    return _get_roi_reference_items(
        group_item,
        VolumetricROIMeasurementsAndQualitativeEvaluations._allowed_roi_reference_types  # noqa: E501
    )


def _get_roi_reference_items(
    group_item: ContainerContentItem,
    allowed_reference_types: Iterable[Code]
) -> Tuple[Code, List[ContentItem]]:
    """Get the content items representing a measurement group's roi reference.

    Parameters
    ----------
    group_item: highdicom.sr.ContainerContentItem
        SR Content Item representing a "Measurement Group"
    allowed_reference_types: Iterable[Code]
        Codes that are allowed as concept names for ROI references.

    Returns
    -------
    highdicom.sr.CodedConcept
        Coded concept representing the concept name of the ROI reference item.
    List[highdicom.sr.ContentItem]
        Content items that defines the reference to the ROI.

    Raises
    ------
    RuntimeError:
        If no content item representing a valid content type is found. If
        multiple valid content items are found with different concept names.

    """
    ref_type_value_type_map = {
        codes.DCM.ImageRegion: [
            ValueTypeValues.SCOORD,
            ValueTypeValues.SCOORD3D
        ],
        codes.DCM.VolumeSurface: [ValueTypeValues.SCOORD3D],
        codes.DCM.ReferencedSegment: [ValueTypeValues.IMAGE],
        codes.DCM.ReferencedSegmentationFrame: [ValueTypeValues.IMAGE],
        _REGION_IN_SPACE: [ValueTypeValues.COMPOSITE],
    }
    returned_items = []
    reference_type = None
    for item in group_item.ContentSequence:
        if item.relationship_type != RelationshipTypeValues.CONTAINS:
            # All ROI reference content items have relationship type CONTAINS
            continue

        if item.name in allowed_reference_types:
            expected_value_types = ref_type_value_type_map[item.name]
            if item.value_type in expected_value_types:
                if reference_type is None:
                    reference_type = item.name
                else:
                    if item.name != reference_type:
                        raise RuntimeError(
                            'Multiple different reference types were found.'
                        )
                    if reference_type not in (
                        codes.DCM.ImageRegion,
                        codes.DCM.VolumeSurface
                    ):
                        raise RuntimeError(
                            'Multiple different reference items of type '
                            f'"{reference_type.meaning}" are not permitted.'
                        )

                returned_items.append(item)

    if len(returned_items) == 0:
        raise RuntimeError(
            'No content item representing a valid ROI reference was found.'
        )

    return reference_type, returned_items


def _contains_code_items(
    parent_item: ContentItem,
    name: Union[Code, CodedConcept],
    value: Optional[Union[Code, CodedConcept]] = None,
    relationship_type: Optional[RelationshipTypeValues] = None
) -> bool:
    """Checks whether an item contains a specific item with value type CODE.

    Parameters
    ----------
    parent_item: highdicom.sr.ContentItem
        Parent SR Content Item
    name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
        Name of the child SR Content Item
    value: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
        Code value of the child SR Content Item
    relationship_type: Union[highdicom.sr.RelationshipTypeValues, None], optional
        Relationship between child and parent SR Content Item

    Returns
    -------
    bool
        Whether any of the SR Content Items contained in `parent_item`
        match the filter criteria

    """  # noqa: E501
    matched_items = find_content_items(
        parent_item,
        name=name,
        value_type=ValueTypeValues.CODE,
        relationship_type=relationship_type
    )
    for item in matched_items:
        if value is not None:
            if item.value == value:
                return True
        else:
            return True
    return False


def _contains_text_items(
    parent_item: ContentItem,
    name: Union[Code, CodedConcept],
    value: Optional[str] = None,
    relationship_type: Optional[RelationshipTypeValues] = None
) -> bool:
    """Checks whether an item contains a specific item with value type TEXT.

    Parameters
    ----------
    parent_item: highdicom.sr.ContentItem
        Parent SR Content Item
    name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
        Name of the child SR Content Item
    value: str, optional
        Text value of the child SR Content Item
    relationship_type: Union[highdicom.sr.RelationshipTypeValues, None], optional
        Relationship between child and parent SR Content Item

    Returns
    -------
    bool
        Whether any of the SR Content Items contained in `parent_item`
        match the filter criteria

    """  # noqa: E501
    matched_items = find_content_items(
        parent_item,
        name=name,
        value_type=ValueTypeValues.TEXT,
        relationship_type=relationship_type
    )
    for item in matched_items:
        if value is not None:
            if item.TextValue == value:
                return True
        else:
            return True
    return False


def _contains_uidref_items(
    parent_item: ContentItem,
    name: Union[Code, CodedConcept],
    value: Optional[str] = None,
    relationship_type: Optional[RelationshipTypeValues] = None
) -> bool:
    """Checks whether an item contains a specific item with value type UIDREF.

    Parameters
    ----------
    parent_item: highdicom.sr.ContentItem
        Parent SR Content Item
    name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
        Name of the child SR Content Item
    value: Union[str, None], optional
        UID value of the child SR Content Item
    relationship_type: Union[highdicom.sr.RelationshipTypeValues, None], optional
        Relationship between child and parent SR Content Item

    Returns
    -------
    bool
        Whether any of the SR Content Items contained in `parent_item`
        match the filter criteria

    """  # noqa: E501
    matched_items = find_content_items(
        parent_item,
        name=name,
        value_type=ValueTypeValues.UIDREF,
        relationship_type=relationship_type
    )
    for item in matched_items:
        if value is not None:
            if item.UID == value:
                return True
        else:
            return True
    return False


def _contains_image_items(
    parent_item: ContentItem,
    name: Union[Code, CodedConcept],
    referenced_sop_class_uid: Union[str, None] = None,
    referenced_sop_instance_uid: Union[str, None] = None,
    relationship_type: Optional[RelationshipTypeValues] = None
) -> bool:
    """Check whether an item contains a specific item with value type IMAGE.

    Parameters
    ----------
    parent_item: highdicom.sr.ContentItem
        Parent SR Content Item
    name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
        Name of the child SR Content Item
    referenced_sop_class_uid: Union[str, None], optional
        SOP Class UID referenced by the content item
    referenced_sop_instance_uid: Union[str, None], optional
        SOP Instance UID referenced by the content item
    relationship_type: Union[highdicom.sr.RelationshipTypeValues, None], optional
        Relationship between child and parent SR Content Item

    Returns
    -------
    bool
        Whether any of the SR Content Items contained in `parent_item`
        match the filter criteria

    """  # noqa: E501
    matched_items = find_content_items(
        parent_item,
        name=name,
        value_type=ValueTypeValues.IMAGE,
        relationship_type=relationship_type
    )
    for item in matched_items:
        if referenced_sop_class_uid is not None:
            if item.referenced_sop_class_uid != referenced_sop_class_uid:
                continue
        if referenced_sop_instance_uid is not None:
            if referenced_sop_instance_uid is not None:
                found_uid = item.referenced_sop_instance_uid
                if found_uid != referenced_sop_instance_uid:
                    continue
        return True
    return False


class Template(ContentSequence):

    """Abstract base class for a DICOM SR template."""

    def __init__(
        self,
        items: Optional[Sequence[ContentItem]] = None,
        is_root: bool = False
    ) -> None:
        """

        Parameters
        ----------
        items: Sequence[ContentItem], optional
            content items
        is_root: bool
            Whether this template exists at the root of the SR document
            content tree.

        """
        super().__init__(items, is_root=is_root)


class AlgorithmIdentification(Template):

    """:dcm:`TID 4019 <part16/sect_TID_4019.html>`
    Algorithm Identification"""

    def __init__(
        self,
        name: str,
        version: str,
        parameters: Optional[Sequence[str]] = None
    ) -> None:
        """

        Parameters
        ----------
        name: str
            name of the algorithm
        version: str
            version of the algorithm
        parameters: Union[Sequence[str], None], optional
            parameters of the algorithm

        """
        super().__init__()
        name_item = TextContentItem(
            name=CodedConcept(
                value='111001',
                meaning='Algorithm Name',
                scheme_designator='DCM'
            ),
            value=name,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        self.append(name_item)
        version_item = TextContentItem(
            name=CodedConcept(
                value='111003',
                meaning='Algorithm Version',
                scheme_designator='DCM'
            ),
            value=version,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        self.append(version_item)
        if parameters is not None:
            for param in parameters:
                parameter_item = TextContentItem(
                    name=CodedConcept(
                        value='111002',
                        meaning='Algorithm Parameters',
                        scheme_designator='DCM'
                    ),
                    value=param,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                self.append(parameter_item)


class TrackingIdentifier(Template):

    """:dcm:`TID 4108 <part16/sect_TID_4108.html>` Tracking Identifier"""

    def __init__(
        self,
        uid: Optional[str] = None,
        identifier: Optional[str] = None
    ):
        """

        Parameters
        ----------
        uid: Union[highdicom.UID, str, None], optional
            globally unique identifier
        identifier: Union[str, None], optional
            human readable identifier

        """
        super().__init__()
        if uid is None:
            uid = UID()
        if identifier is not None:
            tracking_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='112039',
                    meaning='Tracking Identifier',
                    scheme_designator='DCM'
                ),
                value=identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(tracking_identifier_item)
        tracking_uid_item = UIDRefContentItem(
            name=CodedConcept(
                value='112040',
                meaning='Tracking Unique Identifier',
                scheme_designator='DCM'
            ),
            value=uid,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(tracking_uid_item)


class TimePointContext(Template):

    """:dcm:`TID 1502 <part16/chapter_A.html#sect_TID_1502>`
     Time Point Context"""  # noqa: E501

    def __init__(
        self,
        time_point: str,
        time_point_type: Optional[Union[CodedConcept, Code]] = None,
        time_point_order: Optional[int] = None,
        subject_time_point_identifier: Optional[str] = None,
        protocol_time_point_identifier: Optional[str] = None,
        temporal_offset_from_event: Optional[
            LongitudinalTemporalOffsetFromEvent
        ] = None
    ):
        """

        Parameters
        ----------
        time_point: str
            actual value representation of the time point
        time_point_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded type of time point, e.g., "Baseline" or "Posttreatment" (see
            :dcm:`CID 6146 <part16/sect_CID_6146.html>`
            "Time Point Types" for options)
        time_point_order: Union[int, None], optional
            number indicating the order of a time point relative to other
            time points in a time series
        subject_time_point_identifier: Union[str, None], optional
           identifier of a specific time point in a time series, which is
           unique within an appropriate local context and specific to a
           particular subject (patient)
        protocol_time_point_identifier: Union[str, None], optional
           identifier of a specific time point in a time series, which is
           unique within an appropriate local context and specific to a
           particular protocol using the same value for different subjects
        temporal_offset_from_event: Union[highdicom.sr.LongitudinalTemporalOffsetFromEvent, None], optional
            offset in time from a particular event of significance, e.g., the
            baseline of an imaging study or enrollment into a clinical trial

        """  # noqa: E501
        super().__init__()
        time_point_item = TextContentItem(
            name=CodedConcept(
                value='C2348792',
                meaning='Time Point',
                scheme_designator='UMLS'
            ),
            value=time_point,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(time_point_item)
        if time_point_type is not None:
            time_point_type_item = CodeContentItem(
                name=CodedConcept(
                    value='126072',
                    meaning='Time Point Type',
                    scheme_designator='DCM'
                ),
                value=time_point_type,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(time_point_type_item)
        if time_point_order is not None:
            time_point_order_item = NumContentItem(
                name=CodedConcept(
                    value='126073',
                    meaning='Time Point Order',
                    scheme_designator='DCM'
                ),
                value=time_point_order,
                unit=Code('1', 'UCUM', 'no units'),
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(time_point_order_item)
        if subject_time_point_identifier is not None:
            subject_time_point_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='126070',
                    meaning='Subject Time Point Identifier',
                    scheme_designator='DCM'
                ),
                value=subject_time_point_identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(subject_time_point_identifier_item)
        if protocol_time_point_identifier is not None:
            protocol_time_point_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='126071',
                    meaning='Protocol Time Point Identifier',
                    scheme_designator='DCM'
                ),
                value=protocol_time_point_identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(protocol_time_point_identifier_item)
        if temporal_offset_from_event is not None:
            if not isinstance(temporal_offset_from_event,
                              LongitudinalTemporalOffsetFromEvent):
                raise TypeError(
                    'Argument "temporal_offset_from_event" must have type '
                    'LongitudinalTemporalOffsetFromEvent.'
                )
            self.append(temporal_offset_from_event)


class MeasurementStatisticalProperties(Template):

    """:dcm:`TID 311 <part16/chapter_A.html#sect_TID_311>`
    Measurement Statistical Properties"""

    def __init__(
        self,
        values: Sequence[NumContentItem],
        description: Optional[str] = None,
        authority: Optional[str] = None
    ):
        """

        Parameters
        ----------
        values: Sequence[highdicom.sr.NumContentItem]
            reference values of the population of measurements, e.g., its
            mean or standard deviation (see
            :dcm:`CID 226 <part16/sect_CID_226.html>`
            "Population Statistical Descriptors" and
            :dcm:`CID 227 <part16/sect_CID_227.html>`
            "Sample Statistical Descriptors" for options)
        description: Union[str, None], optional
            description of the reference population of measurements
        authority: Union[str, None], optional
            authority for a description of the reference population of
            measurements

        """
        super().__init__()
        if not isinstance(values, (list, tuple)):
            raise TypeError('Argument "values" must be a list.')
        for concept in values:
            if not isinstance(concept, NumContentItem):
                raise ValueError(
                    'Items of argument "values" must have type '
                    'NumContentItem.'
                )
        self.extend(values)
        if description is not None:
            description_item = TextContentItem(
                name=CodedConcept(
                    value='121405',
                    meaning='Population Description',
                    scheme_designator='DCM'
                ),
                value=description,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(description_item)
        if authority is not None:
            authority_item = TextContentItem(
                name=CodedConcept(
                    value='121406',
                    meaning='Reference Authority',
                    scheme_designator='DCM'
                ),
                value=authority,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(authority_item)


class NormalRangeProperties(Template):

    """:dcm:`TID 312 <part16/chapter_A.html#sect_TID_312>`
     Normal Range Properties"""

    def __init__(
        self,
        values: Sequence[NumContentItem],
        description: Optional[str] = None,
        authority: Optional[str] = None
    ):
        """

        Parameters
        ----------
        values: Sequence[highdicom.sr.NumContentItem]
            reference values of the normal range, e.g., its upper and lower
            bound (see :dcm:`CID 223 <part16/sect_CID_223.html>`
            "Normal Range Values" for options)
        description: Union[str, None], optional
            description of the normal range
        authority: Union[str, None], optional
            authority for the description of the normal range

        """  # noqa: E501
        super().__init__()
        if not isinstance(values, (list, tuple)):
            raise TypeError('Argument "values" must be a list.')
        for concept in values:
            if not isinstance(concept, NumContentItem):
                raise ValueError(
                    'Items of argument "values" must have type '
                    'NumContentItem.'
                )
        self.extend(values)
        if description is not None:
            description_item = TextContentItem(
                name=codes.DCM.NormalRangeDescription,
                value=description,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(description_item)
        if authority is not None:
            authority_item = TextContentItem(
                name=codes.DCM.NormalRangeAuthority,
                value=authority,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(authority_item)


class MeasurementProperties(Template):

    """:dcm:`TID 310 <part16/chapter_A.html#sect_TID_310>`
     Measurement Properties"""

    def __init__(
        self,
        normality: Optional[Union[CodedConcept, Code]] = None,
        level_of_significance: Optional[Union[CodedConcept, Code]] = None,
        selection_status: Optional[Union[CodedConcept, Code]] = None,
        measurement_statistical_properties: Optional[
            MeasurementStatisticalProperties
        ] = None,
        normal_range_properties: Optional[NormalRangeProperties] = None,
        upper_measurement_uncertainty: Optional[Union[int, float]] = None,
        lower_measurement_uncertainty: Optional[Union[int, float]] = None
    ):
        """

        Parameters
        ----------
        normality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            the extend to which the measurement is considered normal or abnormal
            (see :dcm:`CID 222 <part16/sect_CID_222.html>` "Normality Codes" for
            options)
        level_of_significance: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            the extend to which the measurement is considered normal or abnormal
            (see :dcm:`CID 220 <part16/sect_CID_220.html>` "Level of
            Significance" for options)
        selection_status: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            how the measurement value was selected or computed from a set of
            available values (see :dcm:`CID 224 <part16/sect_CID_224.html>`
            "Selection Method" for options)
        measurement_statistical_properties: Union[highdicom.sr.MeasurementStatisticalProperties, None], optional
            statistical properties of a reference population for a measurement
            and/or the position of a measurement in such a reference population
        normal_range_properties: Union[highdicom.sr.NormalRangeProperties, None], optional
            statistical properties of a reference population for a measurement
            and/or the position of a measurement in such a reference population
        upper_measurement_uncertainty: Union[int, float, None], optional
            upper range of measurement uncertainty
        lower_measurement_uncertainty: Union[int, float, None], optional
            lower range of measurement uncertainty

        """  # noqa: E501
        super().__init__()
        if normality is not None:
            normality_item = CodeContentItem(
                name=CodedConcept(
                    value='121402',
                    meaning='Normality',
                    scheme_designator='DCM'
                ),
                value=normality,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(normality_item)
        if measurement_statistical_properties is not None:
            if not isinstance(measurement_statistical_properties,
                              MeasurementStatisticalProperties):
                raise TypeError(
                    'Argument "measurement_statistical_properties" must have '
                    'type MeasurementStatisticalProperties.'
                )
            self.extend(measurement_statistical_properties)
        if normal_range_properties is not None:
            if not isinstance(normal_range_properties,
                              NormalRangeProperties):
                raise TypeError(
                    'Argument "normal_range_properties" must have '
                    'type NormalRangeProperties.'
                )
            self.extend(normal_range_properties)
        if level_of_significance is not None:
            level_of_significance_item = CodeContentItem(
                name=CodedConcept(
                    value='121403',
                    meaning='Level of Significance',
                    scheme_designator='DCM'
                ),
                value=level_of_significance,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(level_of_significance_item)
        if selection_status is not None:
            selection_status_item = CodeContentItem(
                name=CodedConcept(
                    value='121404',
                    meaning='Selection Status',
                    scheme_designator='DCM'
                ),
                value=selection_status,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(selection_status_item)
        if upper_measurement_uncertainty is not None:
            upper_measurement_uncertainty_item = NumContentItem(
                name=CodedConcept(
                    value='371886008',
                    meaning='+, range of upper measurement uncertainty',
                    scheme_designator='SCT'
                ),
                value=upper_measurement_uncertainty,
                unit=codes.UCUM.NoUnits,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(upper_measurement_uncertainty_item)
        if lower_measurement_uncertainty is not None:
            lower_measurement_uncertainty_item = NumContentItem(
                name=CodedConcept(
                    value='371885007',
                    meaning='-, range of lower measurement uncertainty',
                    scheme_designator='SCT'
                ),
                value=lower_measurement_uncertainty,
                unit=codes.UCUM.NoUnits,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(lower_measurement_uncertainty_item)


class PersonObserverIdentifyingAttributes(Template):

    """:dcm:`TID 1003 <part16/chapter_A.html#sect_TID_1003>`
     Person Observer Identifying Attributes"""

    def __init__(
        self,
        name: str,
        login_name: Optional[str] = None,
        organization_name: Optional[str] = None,
        role_in_organization: Optional[Union[CodedConcept, Code]] = None,
        role_in_procedure: Optional[Union[CodedConcept, Code]] = None
    ):
        """

        Parameters
        ----------
        name: str
            name of the person
        login_name: Union[str, None], optional
            login name of the person
        organization_name: Union[str, None], optional
            name of the person's organization
        role_in_organization: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            role of the person within the organization
        role_in_procedure: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            role of the person in the reported procedure

        """  # noqa: E501
        super().__init__()
        name_item = PnameContentItem(
            name=CodedConcept(
                value='121008',
                meaning='Person Observer Name',
                scheme_designator='DCM',
            ),
            value=name,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(name_item)
        if login_name is not None:
            login_name_item = TextContentItem(
                name=CodedConcept(
                    value='128774',
                    meaning='Person Observer\'s Login Name',
                    scheme_designator='DCM',
                ),
                value=login_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(login_name_item)
        if organization_name is not None:
            organization_name_item = TextContentItem(
                name=CodedConcept(
                    value='121009',
                    meaning='Person Observer\'s Organization Name',
                    scheme_designator='DCM',
                ),
                value=organization_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(organization_name_item)
        if role_in_organization is not None:
            role_in_organization_item = CodeContentItem(
                name=CodedConcept(
                    value='121010',
                    meaning='Person Observer\'s Role in the Organization',
                    scheme_designator='DCM',
                ),
                value=role_in_organization,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(role_in_organization_item)
        if role_in_procedure is not None:
            role_in_procedure_item = CodeContentItem(
                name=CodedConcept(
                    value='121011',
                    meaning='Person Observer\'s Role in this Procedure',
                    scheme_designator='DCM',
                ),
                value=role_in_procedure,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(role_in_procedure_item)

    @property
    def name(self) -> str:
        """str: name of the person"""
        return self[0].value

    @property
    def login_name(self) -> Union[str, None]:
        """Union[str, None]: login name of the person"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverLoginName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Login Name" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def organization_name(self) -> Union[str, None]:
        """Union[str, None]: name of the person's organization"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverOrganizationName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Organization Name" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def role_in_organization(self) -> Union[str, None]:
        """Union[str, None]: role of the person in the organization"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverRoleInTheOrganization
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Role in Organization" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def role_in_procedure(self) -> Union[str, None]:
        """Union[str, None]: role of the person in the procedure"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverRoleInThisProcedure
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Role in Procedure" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'PersonObserverIdentifyingAttributes':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1003 "Person Observer Identifying Attributes"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.PersonObserverIdentifyingAttributes
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('name', codes.DCM.PersonObserverName),
            ('login_name', codes.DCM.PersonObserverLoginName),
            ('organization_name',
             codes.DCM.PersonObserverOrganizationName),
            ('role_in_organization',
             codes.DCM.PersonObserverRoleInTheOrganization),
            ('role_in_procedure',
             codes.DCM.PersonObserverRoleInThisProcedure),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class DeviceObserverIdentifyingAttributes(Template):

    """:dcm:`TID 1004 <part16/chapter_A.html#sect_TID_1004>`
     Device Observer Identifying Attributes
    """

    def __init__(
        self,
        uid: str,
        name: Optional[str] = None,
        manufacturer_name: Optional[str] = None,
        model_name: Optional[str] = None,
        serial_number: Optional[str] = None,
        physical_location: Optional[str] = None,
        role_in_procedure: Optional[Union[Code, CodedConcept]] = None
    ):
        """

        Parameters
        ----------
        uid: str
            device UID
        name: Union[str, None], optional
            name of device
        manufacturer_name: Union[str, None], optional
            name of device's manufacturer
        model_name: Union[str, None], optional
            name of the device's model
        serial_number: Union[str, None], optional
            serial number of the device
        physical_location: Union[str, None], optional
            physical location of the device during the procedure
        role_in_procedure: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            role of the device in the reported procedure

        """  # noqa: E501
        super().__init__()
        device_observer_item = UIDRefContentItem(
            name=CodedConcept(
                value='121012',
                meaning='Device Observer UID',
                scheme_designator='DCM',
            ),
            value=uid,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(device_observer_item)
        if name is not None:
            name_item = TextContentItem(
                name=CodedConcept(
                    value='121013',
                    meaning='Device Observer Name',
                    scheme_designator='DCM',
                ),
                value=name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(name_item)
        if manufacturer_name is not None:
            manufacturer_name_item = TextContentItem(
                name=CodedConcept(
                    value='121014',
                    meaning='Device Observer Manufacturer',
                    scheme_designator='DCM',
                ),
                value=manufacturer_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(manufacturer_name_item)
        if model_name is not None:
            model_name_item = TextContentItem(
                name=CodedConcept(
                    value='121015',
                    meaning='Device Observer Model Name',
                    scheme_designator='DCM',
                ),
                value=model_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(model_name_item)
        if serial_number is not None:
            serial_number_item = TextContentItem(
                name=CodedConcept(
                    value='121016',
                    meaning='Device Observer Serial Number',
                    scheme_designator='DCM',
                ),
                value=serial_number,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(serial_number_item)
        if physical_location is not None:
            physical_location_item = TextContentItem(
                name=codes.DCM.DeviceObserverPhysicalLocationDuringObservation,
                value=physical_location,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(physical_location_item)
        if role_in_procedure is not None:
            role_in_procedure_item = CodeContentItem(
                name=codes.DCM.DeviceRoleInProcedure,
                value=role_in_procedure,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(role_in_procedure_item)

    @property
    def uid(self) -> UID:
        """highdicom.UID: unique device identifier"""
        return UID(self[0].value)

    @property
    def name(self) -> Union[str, None]:
        """Union[str, None]: name of device"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Name" content item '
                'in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def manufacturer_name(self) -> Union[str, None]:
        """Union[str, None]: name of device manufacturer"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverManufacturer
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Manufacturer" content '
                'name in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def model_name(self) -> Union[str, None]:
        """Union[str, None]: name of device model"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverModelName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Model Name" content '
                'item in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def serial_number(self) -> Union[str, None]:
        """Union[str, None]: device serial number"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverSerialNumber
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Serial Number" content '
                'item in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def physical_location(self) -> Union[str, None]:
        """Union[str, None]: location of device"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverPhysicalLocationDuringObservation  # noqa: E501
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Physical Location '
                'During Observation" content item in "Device Observer '
                'Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'DeviceObserverIdentifyingAttributes':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1004 "Device Observer Identifying Attributes"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.templates.DeviceObserverIdentifyingAttributes
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('name', codes.DCM.DeviceObserverName),
            ('uid', codes.DCM.DeviceObserverUID),
            ('manufacturer_name', codes.DCM.DeviceObserverManufacturer),
            ('model_name', codes.DCM.DeviceObserverModelName),
            ('serial_number', codes.DCM.DeviceObserverSerialNumber),
            ('physical_location',
             codes.DCM.DeviceObserverPhysicalLocationDuringObservation),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class ObserverContext(Template):

    """:dcm:`TID 1002 <part16/chapter_A.html#sect_TID_1002>`
     Observer Context"""

    def __init__(
        self,
        observer_type: CodedConcept,
        observer_identifying_attributes: Union[
            PersonObserverIdentifyingAttributes,
            DeviceObserverIdentifyingAttributes
        ]
    ):
        """

        Parameters
        ----------
        observer_type: highdicom.sr.CodedConcept
            type of observer (see :dcm:`CID 270 <part16/sect_CID_270.html>`
            "Observer Type" for options)
        observer_identifying_attributes: Union[highdicom.sr.PersonObserverIdentifyingAttributes, highdicom.sr.DeviceObserverIdentifyingAttributes]
            observer identifying attributes

        """  # noqa: E501
        super().__init__()
        observer_type_item = CodeContentItem(
            name=CodedConcept(
                value='121005',
                meaning='Observer Type',
                scheme_designator='DCM',
            ),
            value=observer_type,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(observer_type_item)
        if observer_type == codes.cid270.Person:
            if not isinstance(observer_identifying_attributes,
                              PersonObserverIdentifyingAttributes):
                raise TypeError(
                    'Observer identifying attributes must have '
                    'type {} for observer type "{}".'.format(
                        PersonObserverIdentifyingAttributes.__name__,
                        observer_type.meaning
                    )
                )
        elif observer_type == codes.cid270.Device:
            if not isinstance(observer_identifying_attributes,
                              DeviceObserverIdentifyingAttributes):
                raise TypeError(
                    'Observer identifying attributes must have '
                    'type {} for observer type "{}".'.format(
                        DeviceObserverIdentifyingAttributes.__name__,
                        observer_type.meaning,
                    )
                )
        else:
            raise ValueError(
                'Argument "oberver_type" must be either "Person" or "Device".'
            )
        self.extend(observer_identifying_attributes)

    @property
    def observer_type(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: observer type"""
        return self[0].value

    @property
    def observer_identifying_attributes(self) -> Union[
        PersonObserverIdentifyingAttributes,
        DeviceObserverIdentifyingAttributes,
    ]:
        """Union[highdicom.sr.PersonObserverIdentifyingAttributes, highdicom.sr.DeviceObserverIdentifyingAttributes]:
        observer identifying attributes
        """  # noqa: E501
        if self.observer_type == codes.DCM.Device:
            return DeviceObserverIdentifyingAttributes.from_sequence(self)
        elif self.observer_type == codes.DCM.Person:
            return PersonObserverIdentifyingAttributes.from_sequence(self)
        else:
            raise ValueError(
                f'Unexpected observer type "{self.observer_type.meaning}"'
            )


class SubjectContextFetus(Template):

    """:dcm:`TID 1008 <part16/chapter_A.html#sect_TID_1008>`
     Subject Context Fetus"""

    def __init__(self, subject_id: str):
        """

        Parameters
        ----------
        subject_id: str
            identifier of the fetus for longitudinal tracking

        """
        super().__init__()
        subject_id_item = TextContentItem(
            name=CodedConcept(
                value='121030',
                meaning='Subject ID',
                scheme_designator='DCM'
            ),
            value=subject_id,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(subject_id_item)

    @property
    def subject_id(self) -> str:
        """str: subject identifier"""
        return self[0].value

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'SubjectContextFetus':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1008 "Subject Context, Fetus"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.SubjectContextFetus
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('subject_id', codes.DCM.SubjectID),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class SubjectContextSpecimen(Template):

    """:dcm:`TID 1009 <part16/chapter_A.html#sect_TID_1009>`
     Subject Context Specimen"""

    def __init__(
        self,
        uid: str,
        identifier: Optional[str] = None,
        container_identifier: Optional[str] = None,
        specimen_type: Optional[Union[Code, CodedConcept]] = None
    ):
        """

        Parameters
        ----------
        uid: str
            Unique identifier of the observed specimen
        identifier: Union[str, None], optional
            Identifier of the observed specimen (may have limited scope,
            e.g., only relevant with respect to the corresponding container)
        container_identifier: Union[str, None], optional
            Identifier of the container holding the speciment (e.g., a glass
            slide)
        specimen_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Type of the specimen (see
            :dcm:`CID 8103 <part16/sect_CID_8103.html>`
            "Anatomic Pathology Specimen Types" for options)

        """  # noqa: E501
        super().__init__()
        specimen_uid_item = UIDRefContentItem(
            name=CodedConcept(
                value='121039',
                meaning='Specimen UID',
                scheme_designator='DCM'
            ),
            value=uid,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(specimen_uid_item)
        if identifier is not None:
            specimen_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='121041',
                    meaning='Specimen Identifier',
                    scheme_designator='DCM'
                ),
                value=identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(specimen_identifier_item)
        if specimen_type is not None:
            specimen_type_item = CodeContentItem(
                name=CodedConcept(
                    value='371439000',
                    meaning='Specimen Type',
                    scheme_designator='SCT'
                ),
                value=specimen_type,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(specimen_type_item)
        if container_identifier is not None:
            container_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='111700',
                    meaning='Specimen Container Identifier',
                    scheme_designator='DCM'
                ),
                value=container_identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(container_identifier_item)

    @property
    def specimen_uid(self) -> str:
        """str: unique specimen identifier"""
        return self[0].value

    @property
    def specimen_identifier(self) -> Union[str, None]:
        """Union[str, None]: specimen identifier"""
        matches = [
            item for item in self
            if item.name == codes.DCM.SpecimenIdentifier
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Specimen Identifier" content '
                'item in "Subject Context Specimen" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def container_identifier(self) -> Union[str, None]:
        """Union[str, None]: specimen container identifier"""
        matches = [
            item for item in self
            if item.name == codes.DCM.SpecimenContainerIdentifier
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Specimen Container Identifier" content '
                'item in "Subject Context Specimen" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def specimen_type(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: type of specimen"""
        matches = [
            item for item in self
            if item.name == codes.SCT.SpecimenType
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Specimen Type" content '
                'item in "Subject Context Specimen" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'SubjectContextSpecimen':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1009 "Subject Context, Specimen"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.SubjectContextSpecimen
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('uid', codes.DCM.SpecimenUID),
            ('identifier', codes.DCM.SpecimenIdentifier),
            ('container_identifier', codes.DCM.SpecimenContainerIdentifier),
            ('specimen_type', codes.SCT.SpecimenType),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class SubjectContextDevice(Template):

    """:dcm:`TID 1010 <part16/chapter_A.html#sect_TID_1010>`
     Subject Context Device"""

    def __init__(
        self,
        name: str,
        uid: Optional[str] = None,
        manufacturer_name: Optional[str] = None,
        model_name: Optional[str] = None,
        serial_number: Optional[str] = None,
        physical_location: Optional[str] = None
    ):
        """

        Parameters
        ----------
        name: str
            name of the observed device
        uid: Union[str, None], optional
            unique identifier of the observed device
        manufacturer_name: Union[str, None], optional
            name of the observed device's manufacturer
        model_name: Union[str, None], optional
            name of the observed device's model
        serial_number: Union[str, None], optional
            serial number of the observed device
        physical_location: str, optional
            physical location of the observed device during the procedure

        """
        super().__init__()
        device_name_item = TextContentItem(
            name=codes.DCM.DeviceSubjectName,
            value=name,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(device_name_item)
        if uid is not None:
            device_uid_item = UIDRefContentItem(
                name=codes.DCM.DeviceSubjectUID,
                value=uid,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(device_uid_item)
        if manufacturer_name is not None:
            manufacturer_name_item = TextContentItem(
                name=CodedConcept(
                    value='121194',
                    meaning='Device Subject Manufacturer',
                    scheme_designator='DCM',
                ),
                value=manufacturer_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(manufacturer_name_item)
        if model_name is not None:
            model_name_item = TextContentItem(
                name=CodedConcept(
                    value='121195',
                    meaning='Device Subject Model Name',
                    scheme_designator='DCM',
                ),
                value=model_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(model_name_item)
        if serial_number is not None:
            serial_number_item = TextContentItem(
                name=CodedConcept(
                    value='121196',
                    meaning='Device Subject Serial Number',
                    scheme_designator='DCM',
                ),
                value=serial_number,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(serial_number_item)
        if physical_location is not None:
            physical_location_item = TextContentItem(
                name=codes.DCM.DeviceSubjectPhysicalLocationDuringObservation,
                value=physical_location,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(physical_location_item)

    @property
    def device_name(self) -> str:
        """str: name of device"""
        return self[0].value

    @property
    def device_uid(self) -> Union[str, None]:
        """Union[str, None]: unique device identifier"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectUID
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject UID" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_manufacturer_name(self) -> Union[str, None]:
        """Union[str, None]: name of device manufacturer"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectManufacturer
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Manufacturer" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_model_name(self) -> Union[str, None]:
        """Union[str, None]: name of device model"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectModelName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Model Name" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_serial_number(self) -> Union[str, None]:
        """Union[str, None]: device serial number"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectSerialNumber
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Serial Number" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_physical_location(self) -> Union[str, None]:
        """Union[str, None]: location of device"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectPhysicalLocationDuringObservation  # noqa: E501
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Physical Location '
                'During Observation" content item in "Subject Context Device" '
                'template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'SubjectContextDevice':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1010 "Subject Context, Device"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.SubjectContextDevice
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('name', codes.DCM.DeviceSubjectName),
            ('uid', codes.DCM.DeviceSubjectUID),
            ('manufacturer_name', codes.DCM.DeviceSubjectManufacturer),
            ('model_name', codes.DCM.DeviceSubjectModelName),
            ('serial_number', codes.DCM.DeviceSubjectSerialNumber),
            ('physical_location',
             codes.DCM.DeviceSubjectPhysicalLocationDuringObservation),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class SubjectContext(Template):

    """:dcm:`TID 1006 <part16/chapter_A.html#sect_TID_1006>`
     Subject Context"""

    def __init__(
        self,
        subject_class: CodedConcept,
        subject_class_specific_context: Union[
            SubjectContextFetus,
            SubjectContextSpecimen,
            SubjectContextDevice
        ]
    ):
        """

        Parameters
        ----------
        subject_class: highdicom.sr.CodedConcept
            type of subject if the subject of the report is not the patient
            (see :dcm:`CID 271 <part16/sect_CID_271.html>`
            "Observation Subject Class" for options)
        subject_class_specific_context: Union[highdicom.sr.SubjectContextFetus, highdicom.sr.SubjectContextSpecimen, highdicom.sr.SubjectContextDevice], optional
            additional context information specific to `subject_class`

        """  # noqa: E501
        super().__init__()
        subject_class_item = CodeContentItem(
            name=CodedConcept(
                value='121024',
                meaning='Subject Class',
                scheme_designator='DCM'
            ),
            value=subject_class,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(subject_class_item)
        if isinstance(subject_class_specific_context, SubjectContextSpecimen):
            if subject_class != codes.DCM.Specimen:
                raise TypeError(
                    'Subject class specific context doesn\'t match '
                    'subject class "Specimen".'
                )
        elif isinstance(subject_class_specific_context, SubjectContextFetus):
            if subject_class != codes.DCM.Fetus:
                raise TypeError(
                    'Subject class specific context doesn\'t match '
                    'subject class "Fetus".'
                )
        elif isinstance(subject_class_specific_context, SubjectContextDevice):
            if subject_class != codes.DCM.Device:
                raise TypeError(
                    'Subject class specific context doesn\'t match '
                    'subject class "Device".'
                )
        else:
            raise TypeError('Unexpected subject class specific context.')
        self.extend(subject_class_specific_context)

    @property
    def subject_class(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: type of subject"""
        return self[0].value

    @property
    def subject_class_specific_context(self) -> Union[
        SubjectContextFetus,
        SubjectContextSpecimen,
        SubjectContextDevice
    ]:
        """Union[highdicom.sr.SubjectContextFetus, highdicom.sr.SubjectContextSpecimen, highdicom.sr.SubjectContextDevice]:
        subject class specific context
        """  # noqa: E501
        if self.subject_class == codes.DCM.Specimen:
            return SubjectContextSpecimen.from_sequence(sequence=self)
        elif self.subject_class == codes.DCM.Fetus:
            return SubjectContextFetus.from_sequence(sequence=self)
        elif self.subject_class == codes.DCM.Device:
            return SubjectContextDevice.from_sequence(sequence=self)
        else:
            raise ValueError('Unexpected subject class "{item.meaning}".')


class ObservationContext(Template):

    """:dcm:`TID 1001 <part16/chapter_A.html#sect_TID_1001>`
     Observation Context"""

    def __init__(
        self,
        observer_person_context: Optional[ObserverContext] = None,
        observer_device_context: Optional[ObserverContext] = None,
        subject_context: Optional[SubjectContext] = None
    ):
        """

        Parameters
        ----------
        observer_person_context: Union[highdicom.sr.ObserverContext, None], optional
            description of the person that reported the observation
        observer_device_context: Union[highdicom.sr.ObserverContext, None], optional
            description of the device that was involved in reporting the
            observation
        subject_context: Union[highdicom.sr.SubjectContext, None], optional
            description of the imaging subject in case it is not the patient
            for which the report is generated (e.g., a pathology specimen in
            a whole-slide microscopy image, a fetus in an ultrasound image, or
            a pacemaker device in a chest X-ray image)

        """  # noqa: E501
        super().__init__()
        if observer_person_context is not None:
            if not isinstance(observer_person_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_person_context" must '
                    'have type {}'.format(
                        ObserverContext.__name__
                    )
                )
            self.extend(observer_person_context)
        if observer_device_context is not None:
            if not isinstance(observer_device_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_device_context" must '
                    'have type {}'.format(
                        ObserverContext.__name__
                    )
                )
            self.extend(observer_device_context)
        if subject_context is not None:
            if not isinstance(subject_context, SubjectContext):
                raise TypeError(
                    'Argument "subject_context" must have type {}'.format(
                        SubjectContext.__name__
                    )
                )
            self.extend(subject_context)


class LanguageOfContentItemAndDescendants(Template):

    """:dcm:`TID 1204 <part16/chapter_A.html#sect_TID_1204>`
     Language of Content Item and Descendants"""

    def __init__(self, language: CodedConcept):
        """

        Parameters
        ----------
        language: highdicom.sr.CodedConcept
            language used for content items included in report

        """
        super().__init__()
        language_item = CodeContentItem(
            name=CodedConcept(
                value='121049',
                meaning='Language of Content Item and Descendants',
                scheme_designator='DCM',
            ),
            value=language,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        self.append(language_item)


class QualitativeEvaluation(Template):

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[Code, CodedConcept]
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            coded value or an enumerated item representing a coded value

        """  # noqa: E501
        item = CodeContentItem(
            name=name,
            value=value,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        super().__init__([item])

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'QualitativeEvaluation':
        """Construct object from a sequence of content items.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Content Sequence containing one SR CODE Content Item
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.QualitativeEvaluation
            Content Sequence containing one SR CODE Content Item

        """
        if len(sequence) > 1:
            raise ValueError(
                'Qualitative Evaluation shall contain only one content item.'
            )
        item = CodeContentItem.from_dataset(sequence[0])
        return cls(name=item.name, value=item.value)

    @property
    def name(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: name of the qualitative evaluation"""
        return self[0].name

    @property
    def value(self) -> Union[int, float]:
        """Union[int, float]: coded value of the qualitative evaluation"""
        return self[0].value


class Measurement(Template):

    """:dcm:`TID 300 <part16/chapter_A.html#sect_TID_300>`
     Measurement
    """

    def __init__(
        self,
        name: Union[CodedConcept, Code],
        value: Union[int, float],
        unit: Union[CodedConcept, Code],
        qualifier: Optional[Union[CodedConcept, Code]] = None,
        tracking_identifier: Optional[TrackingIdentifier] = None,
        algorithm_id: Optional[AlgorithmIdentification] = None,
        derivation: Optional[Union[CodedConcept, Code]] = None,
        finding_sites: Optional[Sequence[FindingSite]] = None,
        method: Optional[Union[CodedConcept, Code]] = None,
        properties: Optional[MeasurementProperties] = None,
        referenced_images: Optional[Sequence[SourceImageForMeasurement]] = None,
        referenced_real_world_value_map: Optional[RealWorldValueMap] = None
    ):
        """

        Parameters
        ----------
        name: highdicom.sr.CodedConcept
            Name of the measurement (see
            :dcm:`CID 7469 <part16/sect_CID_7469.html>`
            "Generic Intensity and Size Measurements" and
            :dcm:`CID 7468 <part16/sect_CID_7468.html>`
            "Texture Measurements" for options)
        value: Union[int, float]
            Numeric measurement value
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Unit of the numeric measurement value (see
            :dcm:`CID 7181 <part16/sect_CID_7181.html>`
            "Abstract Multi-dimensional Image Model Component
            Units" for options)
        qualifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Qualification of numeric measurement value or as an alternative
            qualitative description
        tracking_identifier: Union[highdicom.sr.TrackingIdentifier, None], optional
            Identifier for tracking measurements
        algorithm_id: Union[highdicom.sr.AlgorithmIdentification, None], optional
            Identification of algorithm used for making measurements
        derivation: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            How the value was computed (see
            :dcm:`CID 7464 <part16/sect_CID_7464.html>`
            "General Region of Interest Measurement Modifiers"
            for options)
        finding_sites: Union[Sequence[highdicom.sr.FindingSite], None], optional
            Coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Measurement method (see
            :dcm:`CID 6147 <part16/sect_CID_6147.html>`
            "Response Criteria" for options)
        properties: Union[highdicom.sr.MeasurementProperties, None], optional
            Measurement properties, including evaluations of its normality
            and/or significance, its relationship to a reference population,
            and an indication of its selection from a set of measurements
        referenced_images: Union[Sequence[highdicom.sr.SourceImageForMeasurement], None], optional
            Referenced images which were used as sources for the measurement
        referenced_real_world_value_map: Union[highdicom.sr.RealWorldValueMap, None], optional
            Referenced real world value map for referenced source images

        """  # noqa: E501
        super().__init__()
        value_item = NumContentItem(
            name=name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if tracking_identifier is not None:
            if not isinstance(tracking_identifier, TrackingIdentifier):
                raise TypeError(
                    'Argument "tracking_identifier" must have type '
                    'TrackingIdentifier.'
                )
            content.extend(tracking_identifier)
        if method is not None:
            method_item = CodeContentItem(
                name=CodedConcept(
                    value='370129005',
                    meaning='Measurement Method',
                    scheme_designator='SCT'
                ),
                value=method,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            content.append(method_item)
        if derivation is not None:
            derivation_item = CodeContentItem(
                name=CodedConcept(
                    value='121401',
                    meaning='Derivation',
                    scheme_designator='DCM'
                ),
                value=derivation,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            content.append(derivation_item)
        if finding_sites is not None:
            if not isinstance(finding_sites, (list, tuple, set)):
                raise TypeError(
                    'Argument "finding_sites" must be a sequence.'

                )
            for site in finding_sites:
                if not isinstance(site, FindingSite):
                    raise TypeError(
                        'Items of argument "finding_sites" must have '
                        'type FindingSite.'
                    )
                content.append(site)
        if properties is not None:
            if not isinstance(properties, MeasurementProperties):
                raise TypeError(
                    'Argument "properties" must have '
                    'type MeasurementProperties.'
                )
            content.extend(properties)
        if referenced_images is not None:
            for image in referenced_images:
                if not isinstance(image, SourceImageForMeasurement):
                    raise TypeError(
                        'Arguments "referenced_images" must have type '
                        'SourceImageForMeasurement.'
                    )
                content.append(image)
        if referenced_real_world_value_map is not None:
            if not isinstance(referenced_real_world_value_map,
                              RealWorldValueMap):
                raise TypeError(
                    'Argument "referenced_real_world_value_map" must have type '
                    'RealWorldValueMap.'
                )
            content.append(referenced_real_world_value_map)
        if algorithm_id is not None:
            if not isinstance(algorithm_id, AlgorithmIdentification):
                raise TypeError(
                    'Argument "algorithm_id" must have type '
                    'AlgorithmIdentification.'
                )
            content.extend(algorithm_id)
        if len(content) > 0:
            value_item.ContentSequence = content
        self.append(value_item)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'Measurement':
        """Construct object from a sequence of content items.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Content Sequence containing one SR NUM Content Items
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.Measurement
            Content Sequence containing one SR NUM Content Items

        """
        if len(sequence) > 1:
            raise ValueError('Measurement shall contain only one content item.')
        item = sequence[0]
        if not isinstance(item, NumContentItem):
            raise TypeError('Measurement shall contain a NUM content item.')
        measurement = cls(
            name=item.name,
            value=item.value,
            unit=item.unit,
            qualifier=item.qualifier
        )
        if 'ContentSequence' in item:
            measurement[0].ContentSequence = ContentSequence.from_sequence(
                item.ContentSequence
            )
        return measurement

    @property
    def name(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded name of the measurement"""
        return self[0].name

    @property
    def value(self) -> Union[int, float]:
        """Union[int, float]: measured value"""
        return self[0].value

    @property
    def unit(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: unit"""
        return self[0].unit

    @property
    def qualifier(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: qualifier"""
        return self[0].qualifier

    @property
    def derivation(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: derivation"""
        if not hasattr(self[0], 'ContentSequence'):
            return None
        matches = find_content_items(
            self[0],
            name=codes.DCM.Derivation,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def method(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: method"""
        if not hasattr(self[0], 'ContentSequence'):
            return None
        matches = find_content_items(
            self[0],
            name=codes.SCT.MeasurementMethod,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def referenced_images(self) -> List[SourceImageForMeasurement]:
        """List[highdicom.sr.SourceImageForMeasurement]: referenced images"""
        if not hasattr(self[0], 'ContentSequence'):
            return []
        matches = find_content_items(
            self[0],
            name=codes.DCM.SourceOfMeasurement,
            value_type=ValueTypeValues.IMAGE
        )
        return [SourceImageForMeasurement.from_dataset(m) for m in matches]

    @property
    def finding_sites(self) -> List[FindingSite]:
        """List[highdicom.sr.FindingSite]: finding sites"""
        if not hasattr(self[0], 'ContentSequence'):
            return []
        matches = find_content_items(
            self[0],
            name=codes.SCT.FindingSite,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 0:
            return [FindingSite.from_dataset(m) for m in matches]
        return []


class _MeasurementsAndQualitativeEvaluations(Template):

    """Abstract base class for Measurements and Qualitative Evaluation
    templates."""

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
        time_point_context: Optional[TimePointContext] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        method: Optional[Union[CodedConcept, Code]] = None,
        algorithm_id: Optional[AlgorithmIdentification] = None,
        finding_sites: Optional[Sequence[FindingSite]] = None,
        session: Optional[str] = None,
        measurements: Sequence[Measurement] = None,
        qualitative_evaluations: Optional[
            Sequence[QualitativeEvaluation]
        ] = None,
        finding_category: Optional[Union[CodedConcept, Code]] = None,
    ):
        """

        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements
        referenced_real_world_value_map: Union[highdicom.sr.RealWorldValueMap, None], optional
            Referenced real world value map for region of interest
        time_point_context: Union[highdicom.sr.TimePointContext, None], optional
            Description of the time point context
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of observed finding
        method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded measurement method (see
            :dcm:`CID 6147 <part16/sect_CID_6147.html>`
            "Response Criteria" for options)
        algorithm_id: Union[highdicom.sr.AlgorithmIdentification, None], optional
            Identification of algorithm used for making measurements
        finding_sites: Sequence[highdicom.sr.FindingSite, None], optional
            Coded description of one or more anatomic locations at which
            finding was observed
        session: Union[str, None], optional
            Description of the session
        measurements: Union[Sequence[highdicom.sr.Measurement], None], optional
            Numeric measurements
        qualitative_evaluations: Union[Sequence[highdicom.sr.QualitativeEvaluation], None], optional
            Coded name-value pairs that describe qualitative evaluations
        finding_category: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Category of observed finding, e.g., anatomic structure or
            morphologically abnormal structure

        """  # noqa: E501
        super().__init__()
        group_item = ContainerContentItem(
            name=CodedConcept(
                value='125007',
                meaning='Measurement Group',
                scheme_designator='DCM'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS,
            template_id='1501'
        )
        content = ContentSequence()
        if not isinstance(tracking_identifier, TrackingIdentifier):
            raise TypeError(
                'Argument "tracking_identifier" must have type '
                'TrackingIdentifier.'
            )
        if len(tracking_identifier) == 1:
            raise ValueError(
                'Argument "tracking_identifier" must include a '
                'human readable tracking identifier and a tracking unique '
                'identifier.'
            )
        content.extend(tracking_identifier)
        if session is not None:
            session_item = TextContentItem(
                name=CodedConcept(
                    value='C67447',
                    meaning='Activity Session',
                    scheme_designator='NCIt'
                ),
                value=session,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            content.append(session_item)
        if finding_category is not None:
            finding_category_item = CodeContentItem(
                name=CodedConcept(
                    value='276214006',
                    meaning='Finding category',
                    scheme_designator='SCT'
                ),
                value=finding_category,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(finding_category_item)
        if finding_type is not None:
            finding_type_item = CodeContentItem(
                name=CodedConcept(
                    value='121071',
                    meaning='Finding',
                    scheme_designator='DCM'
                ),
                value=finding_type,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(finding_type_item)
        if method is not None:
            method_item = CodeContentItem(
                name=CodedConcept(
                    value='370129005',
                    meaning='Measurement Method',
                    scheme_designator='SCT'
                ),
                value=method,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(method_item)
        if finding_sites is not None:
            if not isinstance(finding_sites, (list, tuple, set)):
                raise TypeError(
                    'Argument "finding_sites" must be a sequence.'

                )
            for site in finding_sites:
                if not isinstance(site, FindingSite):
                    raise TypeError(
                        'Items of argument "finding_sites" must have '
                        'type FindingSite.'
                    )
                content.append(site)
        if algorithm_id is not None:
            if not isinstance(algorithm_id, AlgorithmIdentification):
                raise TypeError(
                    'Argument "algorithm_id" must have type '
                    'AlgorithmIdentification.'
                )
            content.extend(algorithm_id)
        if time_point_context is not None:
            if not isinstance(time_point_context, TimePointContext):
                raise TypeError(
                    'Argument "time_point_context" must have type '
                    'TimePointContext.'
                )
            content.extend(time_point_context)
        if referenced_real_world_value_map is not None:
            if not isinstance(referenced_real_world_value_map,
                              RealWorldValueMap):
                raise TypeError(
                    'Argument "referenced_real_world_value_map" must have type '
                    'RealWorldValueMap.'
                )
            content.append(referenced_real_world_value_map)
        if measurements is not None:
            for measurement in measurements:
                if not isinstance(measurement, Measurement):
                    raise TypeError(
                        'Items of argument "measurements" must have '
                        'type Measurement.'
                    )
                content.extend(measurement)
        if qualitative_evaluations is not None:
            for evaluation in qualitative_evaluations:
                if not isinstance(evaluation, QualitativeEvaluation):
                    raise TypeError(
                        'Items of argument "qualitative_evaluations" must '
                        'have type QualitativeEvaluation.'
                    )
                content.extend(evaluation)
        if len(content) > 0:
            group_item.ContentSequence = content
        self.append(group_item)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> '_MeasurementsAndQualitativeEvaluations':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing "Measurement Group" SR Content Items
            of Value Type CONTAINER (sequence shall only contain a single item)
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr._MeasurementsAndQualitativeEvaluations
            Content Sequence containing root CONTAINER SR Content Item

        """
        if len(sequence) > 1:
            raise ValueError(
                'Sequence contains more than one SR Content Item.'
            )
        dataset = sequence[0]
        if dataset.ValueType != ValueTypeValues.CONTAINER.value:
            raise ValueError(
                'Item #1 of sequence is not an appropriate SR Content Item '
                'because it does not have Value Type CONTAINER.'
            )
        if get_coded_name(dataset) != codes.DCM.MeasurementGroup:
            raise ValueError(
                'Item #1 of sequence is not an appropriate SR Content Item '
                'because it does not have name "Measurement Group".'
            )
        instance = ContentSequence.from_sequence(sequence)
        instance.__class__ = cls
        return cast(cls, instance)

    @property
    def method(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: measurement method"""
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=codes.SCT.MeasurementMethod,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Measurement Method" content '
                'item in "Measurements and Qualitative Evaluations" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def tracking_identifier(self) -> Union[str, None]:
        """Union[str, None]: tracking identifier"""
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=codes.DCM.TrackingIdentifier,
            value_type=ValueTypeValues.TEXT
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Tracking Identifier" content '
                'item in "Measurements and Qualitative Evaluations" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def tracking_uid(self) -> Union[UID, None]:
        """Union[highdicom.UID, None]: tracking unique identifier"""
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=codes.DCM.TrackingUniqueIdentifier,
            value_type=ValueTypeValues.UIDREF
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Tracking Unique Identifier" content '
                'item in "Measurements and Qualitative Evaluations" template'
            )
        if len(matches) > 0:
            return UID(matches[0].value)
        return None

    @property
    def finding_category(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: finding category"""
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=Code('276214006', 'SCT', 'Finding category'),
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Finding category" content item '
                'in "Measurements and Qualitative Evaluations" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def finding_type(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: finding type"""
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=codes.DCM.Finding,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Finding" content item '
                'in "Measurements and Qualitative Evaluations" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def finding_sites(self) -> List[FindingSite]:
        """List[highdicom.sr.FindingSite]: finding sites"""
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=codes.SCT.FindingSite,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 0:
            return [FindingSite.from_dataset(m) for m in matches]
        return []

    def get_measurements(
        self,
        name: Optional[Union[Code, CodedConcept]] = None
    ) -> List[Measurement]:
        """Get measurements.

        Parameters
        ----------
        name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Name of measurement

        Returns
        -------
        List[highdicom.sr.Measurement]
            Measurements

        """  # noqa: E501
        root_item = self[0]
        if name is None:
            matches = find_content_items(
                root_item,
                value_type=ValueTypeValues.NUM
            )
        else:
            matches = find_content_items(
                root_item,
                name=name,
                value_type=ValueTypeValues.NUM
            )
        return [
            Measurement.from_sequence([item])
            for item in matches
        ]

    def get_qualitative_evaluations(
        self,
        name: Optional[Union[Code, CodedConcept]] = None
    ) -> List[QualitativeEvaluation]:
        """Get qualitative evaluations.

        Parameters
        ----------
        name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Name of evaluation

        Returns
        -------
        List[highdicom.sr.QualitativeEvaluation]
            Qualitative evaluations

        """  # noqa: E501
        root_item = self[0]
        if name is None:
            matches = find_content_items(
                root_item,
                value_type=ValueTypeValues.CODE
            )
        else:
            matches = find_content_items(
                root_item,
                name=name,
                value_type=ValueTypeValues.CODE
            )
        return [
            QualitativeEvaluation.from_sequence([item])
            for item in matches
            if item.name not in (
                codes.DCM.Finding,
                codes.SCT.FindingSite,
                codes.SCT.MeasurementMethod
            )
        ]


class MeasurementsAndQualitativeEvaluations(
    _MeasurementsAndQualitativeEvaluations
):

    """:dcm:`TID 1501 <part16/chapter_A.html#sect_TID_1501>`
     Measurement and Qualitative Evaluation Group"""

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
        time_point_context: Optional[TimePointContext] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        method: Optional[Union[CodedConcept, Code]] = None,
        algorithm_id: Optional[AlgorithmIdentification] = None,
        finding_sites: Optional[Sequence[FindingSite]] = None,
        session: Optional[str] = None,
        measurements: Sequence[Measurement] = None,
        qualitative_evaluations: Optional[
            Sequence[QualitativeEvaluation]
        ] = None,
        finding_category: Optional[Union[CodedConcept, Code]] = None,
        source_images: Optional[
            Sequence[SourceImageForMeasurementGroup]
        ] = None,
    ):
        """

        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements
        referenced_real_world_value_map: Union[highdicom.sr.RealWorldValueMap, None], optional
            Referenced real world value map for region of interest
        time_point_context: Union[highdicom.sr.TimePointContext, None], optional
            Description of the time point context
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of observed finding
        method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded measurement method (see
            :dcm:`CID 6147 <part16/sect_CID_6147.html>`
            "Response Criteria" for options)
        algorithm_id: Union[highdicom.sr.AlgorithmIdentification, None], optional
            Identification of algorithm used for making measurements
        finding_sites: Sequence[highdicom.sr.FindingSite, None], optional
            Coded description of one or more anatomic locations at which
            finding was observed
        session: Union[str, None], optional
            Description of the session
        measurements: Union[Sequence[highdicom.sr.Measurement], None], optional
            Numeric measurements
        qualitative_evaluations: Union[Sequence[highdicom.sr.QualitativeEvaluation], None], optional
            Coded name-value pairs that describe qualitative evaluations
        finding_category: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Category of observed finding, e.g., anatomic structure or
            morphologically abnormal structure
        source_images: Optional[Sequence[highdicom.sr.SourceImageForMeasurementGroup]], optional
            Images to that were the source of the measurements. If not provided,
            all images that listed in the document tree of the containing SR
            document are assumed to be source images.

        """  # noqa: E501
        super().__init__(
            tracking_identifier=tracking_identifier,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations,
            finding_category=finding_category,
        )
        group_item = self[0]

        if source_images is not None:
            for img in source_images:
                if not isinstance(img, SourceImageForMeasurementGroup):
                    raise TypeError(
                        'Items of argument "source_images" must be of type '
                        'highdicom.sr.SourceImageForMeasurementGroup.'
                    )
            group_item.ContentSequence.extend(source_images)

    @property
    def source_images(self) -> List[SourceImageForMeasurementGroup]:
        """List[highdicom.sr.SourceImageForMeasurementGroup]: source images"""
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=_SOURCE,
            value_type=ValueTypeValues.IMAGE,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        if len(matches) > 0:
            return [
                SourceImageForMeasurementGroup.from_dataset(m) for m in matches
            ]
        return []


class _ROIMeasurementsAndQualitativeEvaluations(
    _MeasurementsAndQualitativeEvaluations
):

    """Abstract base class for ROI Measurements and Qualitative Evaluation
    templates."""

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        referenced_regions: Optional[
            Union[Sequence[ImageRegion], Sequence[ImageRegion3D]]
        ] = None,
        referenced_segment: Optional[
            Union[ReferencedSegment, ReferencedSegmentationFrame]
        ] = None,
        referenced_volume_surface: Optional[VolumeSurface] = None,
        referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
        time_point_context: Optional[TimePointContext] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        method: Optional[Union[CodedConcept, Code]] = None,
        algorithm_id: Optional[AlgorithmIdentification] = None,
        finding_sites: Optional[Sequence[FindingSite]] = None,
        session: Optional[str] = None,
        measurements: Sequence[Measurement] = None,
        qualitative_evaluations: Optional[
            Sequence[QualitativeEvaluation]
        ] = None,
        geometric_purpose: Optional[Union[CodedConcept, Code]] = None,
        finding_category: Optional[Union[CodedConcept, Code]] = None,
    ):
        """

        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements
        referenced_regions: Union[Sequence[highdicom.sr.ImageRegion], Sequence[highdicom.sr.ImageRegion3D], None], optional
            Regions of interest in source image(s)
        referenced_segment: Union[highdicom.sr.ReferencedSegment, highdicom.sr.ReferencedSegmentationFrame, None], optional
            Segmentation for region of interest in source image
        referenced_volume_surface: Union[hidicom.sr.content.VolumeSurface, None], optional
            Surface segmentation for region of interest in source image
        referenced_real_world_value_map: Union[highdicom.sr.RealWorldValueMap, None], optional
            Referenced real world value map for region of interest
        time_point_context: Union[highdicom.sr.TimePointContext, None], optional
            Description of the time point context
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of observed finding
        method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded measurement method (see
            :dcm:`CID 6147 "Response Criteria" <part16/sect_CID_6147.html>`
            for options)
        algorithm_id: Union[highdicom.sr.AlgorithmIdentification, None], optional
            Identification of algorithm used for making measurements
        finding_sites: Union[Sequence[highdicom.sr.FindingSite], None], optional
            Coded description of one or more anatomic locations at which
            finding was observed
        session: Union[str, None], optional
            Description of the session
        measurements: Union[Sequence[highdicom.sr.Measurement], None], optional
            Numeric measurements
        qualitative_evaluations: Union[Sequence[highdicom.sr.QualitativeEvaluation], None], optional
            Coded name-value (question-answer) pairs that describe
            qualitative evaluations
        geometric_purpose: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Geometric interpretation of region of interest (see
            :dcm:`CID 219 <part16/sect_CID_219.html>`
            "Geometry Graphical Representation" for options)
        finding_category: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Category of observed finding, e.g., anatomic structure or
            morphologically abnormal structure

        Note
        ----
        Either a segmentation, a list of regions, or a volume needs to
        referenced together with the corresponding source image(s) or series.
        Derived classes determine which of the above will be allowed.

        """  # noqa: E501
        super().__init__(
            tracking_identifier=tracking_identifier,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations,
            finding_category=finding_category
        )
        group_item = self[0]
        were_references_provided = [
            referenced_regions is not None,
            referenced_volume_surface is not None,
            referenced_segment is not None,
        ]
        if sum(were_references_provided) == 0:
            raise ValueError(
                'One of the following arguments must be provided: '
                '"referenced_regions", "referenced_volume_surface", or '
                '"referenced_segment".'
            )
        elif sum(were_references_provided) > 1:
            raise ValueError(
                'Only one of the following arguments should be provided: '
                '"referenced_regions", "referenced_volume_surface", or '
                '"referenced_segment".'
            )
        if geometric_purpose is not None:
            geometric_purpose_item = CodeContentItem(
                name=CodedConcept(
                    value='130400',
                    meaning='Geometric purpose of region',
                    scheme_designator='DCM',
                ),
                value=geometric_purpose,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            group_item.ContentSequence.append(geometric_purpose_item)
        if referenced_regions is not None:
            if len(referenced_regions) == 0:
                raise ValueError(
                    'Argument "referenced_region" must have non-zero length.'
                )
            for region in referenced_regions:
                if not isinstance(region, (ImageRegion, ImageRegion3D)):
                    raise TypeError(
                        'Items of argument "referenced_regions" must have type '
                        'ImageRegion or ImageRegion3D.'
                    )
                group_item.ContentSequence.append(region)
        elif referenced_volume_surface is not None:
            if not isinstance(referenced_volume_surface, VolumeSurface):
                raise TypeError(
                    'Items of argument "referenced_volume_surface" must have '
                    'type VolumeSurface.'
                )
            group_item.ContentSequence.extend(referenced_volume_surface)
        elif referenced_segment is not None:
            if not isinstance(
                    referenced_segment,
                    (ReferencedSegment, ReferencedSegmentationFrame)
                ):
                raise TypeError(
                    'Argument "referenced_segment" must have type '
                    'ReferencedSegment or '
                    'ReferencedSegmentationFrame.'
                )
            group_item.ContentSequence.extend(referenced_segment)


class PlanarROIMeasurementsAndQualitativeEvaluations(
        _ROIMeasurementsAndQualitativeEvaluations):

    """:dcm:`TID 1410 <part16/chapter_A.html#sect_TID_1410>`
     Planar ROI Measurements and Qualitative Evaluations"""

    _allowed_roi_reference_types = {
        codes.DCM.ImageRegion,
        codes.DCM.ReferencedSegmentationFrame,
        _REGION_IN_SPACE
    }

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        referenced_region: Optional[
            Union[ImageRegion, ImageRegion3D]
        ] = None,
        referenced_segment: Optional[ReferencedSegmentationFrame] = None,
        referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
        time_point_context: Optional[TimePointContext] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        method: Optional[Union[CodedConcept, Code]] = None,
        algorithm_id: Optional[AlgorithmIdentification] = None,
        finding_sites: Optional[Sequence[FindingSite]] = None,
        session: Optional[str] = None,
        measurements: Sequence[Measurement] = None,
        qualitative_evaluations: Optional[
            Sequence[QualitativeEvaluation]
        ] = None,
        geometric_purpose: Optional[Union[CodedConcept, Code]] = None,
        finding_category: Optional[Union[CodedConcept, Code]] = None
    ):
        """

        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements
        referenced_region: Union[highdicom.sr.ImageRegion, highdicom.sr.ImageRegion3D, None], optional
            Region of interest in source image
        referenced_segment: Union[highdicom.sr.ReferencedSegmentationFrame, None], optional
            Segmentation for region of interest in source image
        referenced_real_world_value_map: Union[highdicom.sr.RealWorldValueMap, None], optional
            Referenced real world value map for region of interest
        time_point_context: Union[highdicom.sr.TimePointContext, None], optional
            Description of the time point context
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of object that was measured, e.g., organ or tumor
        method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded measurement method (see
            :dcm:`CID 6147 <part16/sect_CID_6147.html>`
            "Response Criteria" for options)
        algorithm_id: Union[highdicom.sr.AlgorithmIdentification, None], optional
            Identification of algorithm used for making measurements
        finding_sites: Union[Sequence[highdicom.sr.FindingSite], None], optional
            Coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        session: Union[str, None], optional
            Description of the session
        measurements: Union[Sequence[highdicom.sr.Measurement], None], optional
            Measurements for a region of interest
        qualitative_evaluations: Union[Sequence[highdicom.sr.QualitativeEvaluation], None], optional
            Coded name-value (question-answer) pairs that describe
            qualitative evaluations of a region of interest
        geometric_purpose: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Geometric interpretation of region of interest (see
            :dcm:`CID 219 <part16/sect_CID_219.html>`
            "Geometry Graphical Representation" for options)
        finding_category: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Category of observed finding, e.g., anatomic structure or
            morphologically abnormal structure

        Note
        ----
        Either a segmentation or a region needs to referenced
        together with the corresponding source image from which the
        segmentation or region was obtained.

        """  # noqa: E501
        were_references_provided = [
            referenced_region is not None,
            referenced_segment is not None,
        ]
        if sum(were_references_provided) == 0:
            raise ValueError(
                'One of the following arguments must be provided: '
                '"referenced_region", "referenced_segment".'
            )
        elif sum(were_references_provided) > 1:
            raise ValueError(
                'Only one of the following arguments should be provided: '
                '"referenced_region", "referenced_segment".'
            )
        referenced_regions: Optional[
            Union[Sequence[ImageRegion], Sequence[ImageRegion3D]]
        ] = None
        if referenced_region is not None:
            # This is just to satisfy mypy
            if isinstance(referenced_region, ImageRegion):
                referenced_regions = [referenced_region]
            elif isinstance(referenced_region, ImageRegion3D):
                referenced_regions = [referenced_region]
            else:
                raise TypeError(
                    'Argument "referenced_region" must have type '
                    'ImageRegion or ImageRegion3D.'
                )
        if referenced_segment is not None:
            if not isinstance(referenced_segment, ReferencedSegmentationFrame):
                raise TypeError(
                    'Argument "referenced_segment" must have type '
                    'ReferencedSegmentationFrame.'
                )
        super().__init__(
            tracking_identifier=tracking_identifier,
            referenced_regions=referenced_regions,
            referenced_segment=referenced_segment,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations,
            geometric_purpose=geometric_purpose,
            finding_category=finding_category
        )
        self[0].ContentTemplateSequence[0].TemplateIdentifier = '1410'

    @property
    def reference_type(self) -> Code:
        """pydicom.sr.coding.Code:

        The "type" of the ROI reference as a coded concept. This will be one of
        the following coded concepts from the DCM coding scheme:

        - Image Region
        - Referenced Segmentation Frame
        - Region In Space

        """
        for item in self[0].ContentSequence:
            for concept_name in self._allowed_roi_reference_types:
                if item.name == concept_name:
                    return concept_name
        else:
            raise RuntimeError(
                'Could not find any allowed ROI reference type.'
            )

    @property
    def roi(self) -> Union[ImageRegion, ImageRegion3D, None]:
        """Union[highdicom.sr.ImageRegion, highdicom.sr.ImageRegion3D, None]:
        image region defined by spatial coordinates
        """  # noqa: E501
        # Image Region may be defined by either SCOORD or SCOORD3D
        root_item = self[0]
        matches = find_content_items(
            root_item,
            name=codes.DCM.ImageRegion,
            value_type=ValueTypeValues.SCOORD
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Image Region" content item '
                'in "Planar ROI Measurements and Qualitative Evaluations" '
                'template'
            )
        if len(matches) > 0:
            return ImageRegion.from_dataset(matches[0])
        matches = find_content_items(
            root_item,
            name=codes.DCM.ImageRegion,
            value_type=ValueTypeValues.SCOORD3D
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Image Region" content item '
                'in "Planar ROI Measurements and Qualitative Evaluations" '
                'template'
            )
        if len(matches) > 0:
            return ImageRegion3D.from_dataset(matches[0])
        return None

    @property
    def _referenced_segmentation_frame_item(
        self
    ) -> Union[ImageContentItem, None]:
        """Union[highdicom.sr.ImageContentItem, None]:
        image content item for referenced segmentation frame
        """  # noqa: E501
        root_item = self[0]

        # Find the referenced segmentation frame content item
        matches = find_content_items(
            root_item,
            name=codes.DCM.ReferencedSegmentationFrame,
            value_type=ValueTypeValues.IMAGE
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Referenced Segmentation Frame" content '
                'item in "Planar ROI Measurements and Qualitative Evaluations" '
                'template'
            )
        elif len(matches) == 0:
            return None

        return matches[0]

    @property
    def _source_image_for_segmentation_item(
        self
    ) -> Union[SourceImageForSegmentation, None]:
        """Union[highdicom.sr.SourceImageForSegmentation, None]:
        source images used for the referenced segment
        """  # noqa: E501
        root_item = self[0]

        # Find the referenced segmentation frame content item
        matches = find_content_items(
            root_item,
            name=codes.DCM.SourceImageForSegmentation,
            value_type=ValueTypeValues.IMAGE
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Source Image For Segmentation" content '
                'item in "Planar ROI Measurements and Qualitative Evaluations" '
                'template'
            )
        elif len(matches) == 0:
            return None

        return SourceImageForSegmentation.from_dataset(matches[0])

    @property
    def referenced_segmentation_frame(
        self
    ) -> Union[ReferencedSegmentationFrame, None]:
        """Union[highdicom.sr.ImageContentItem, None]:
        segmentation frame referenced by the measurements group
        """  # noqa: E501
        if self.reference_type == codes.DCM.ReferencedSegmentationFrame:
            return ReferencedSegmentationFrame.from_sequence(
                self[0].ContentSequence
            )
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'PlanarROIMeasurementsAndQualitativeEvaluations':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing "Measurement Group" SR Content Items
            of Value Type CONTAINER (sequence shall only contain a single item)
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.PlanarROIMeasurementsAndQualitativeEvaluations
            Content Sequence containing root CONTAINER SR Content Item

        """
        instance = super(
            PlanarROIMeasurementsAndQualitativeEvaluations,
            cls
        ).from_sequence(sequence)
        instance.__class__ = PlanarROIMeasurementsAndQualitativeEvaluations
        return cast(PlanarROIMeasurementsAndQualitativeEvaluations, instance)


class VolumetricROIMeasurementsAndQualitativeEvaluations(
        _ROIMeasurementsAndQualitativeEvaluations):

    """:dcm:`TID 1411 <part16/chapter_A.html#sect_TID_1411>`
     Volumetric ROI Measurements and Qualitative Evaluations"""

    _allowed_roi_reference_types = {
        codes.DCM.ImageRegion,
        codes.DCM.ReferencedSegment,
        codes.DCM.VolumeSurface,
        _REGION_IN_SPACE
    }

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        referenced_regions: Optional[
            Union[Sequence[ImageRegion]]
        ] = None,
        referenced_volume_surface: Optional[VolumeSurface] = None,
        referenced_segment: Optional[ReferencedSegment] = None,
        referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
        time_point_context: Optional[TimePointContext] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        method: Optional[Union[CodedConcept, Code]] = None,
        algorithm_id: Optional[AlgorithmIdentification] = None,
        finding_sites: Optional[Sequence[FindingSite]] = None,
        session: Optional[str] = None,
        measurements: Sequence[Measurement] = None,
        qualitative_evaluations: Optional[
            Sequence[QualitativeEvaluation]
        ] = None,
        geometric_purpose: Optional[Union[CodedConcept, Code]] = None,
        finding_category: Optional[Union[CodedConcept, Code]] = None,
    ):
        """

        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements
        referenced_regions: Union[Sequence[highdicom.sr.ImageRegion], None], optional
            Regions of interest in source image(s)
        referenced_volume_surface: Union[highdicom.sr.VolumeSurface, None], optional
            Volume of interest in source image(s)
        referenced_segment: Union[highdicom.sr.ReferencedSegment, None], optional
            Segmentation for region of interest in source image
        referenced_real_world_value_map: Union[highdicom.sr.RealWorldValueMap, None], optional
            Referenced real world value map for region of interest
        time_point_context: Union[highdicom.sr.TimePointContext, None], optional
            Description of the time point context
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of observed finding
        method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded measurement method (see
            :dcm:`CID 6147 <part16/sect_CID_6147.html>`
            "Response Criteria" for options)
        algorithm_id: Union[highdicom.sr.AlgorithmIdentification, None], optional
            Identification of algorithm used for making measurements
        finding_sites: Union[Sequence[highdicom.sr.FindingSite], None], optional
            Coded description of one or more anatomic locations at which the
            finding was observed
        session: Union[str, None], optional
            Description of the session
        measurements: Union[Sequence[highdicom.sr.Measurement], None], optional
            Measurements for a volume of interest
        qualitative_evaluations: Union[Sequence[highdicom.sr.QualitativeEvaluation], None], optional
            Coded name-value (question-answer) pairs that describe
            qualitative evaluations of a volume of interest
        geometric_purpose: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Geometric interpretation of region of interest (see
            :dcm:`CID 219 <part16/sect_CID_219.html>`
            "Geometry Graphical Representation" for options)
        finding_category: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Category of observed finding, e.g., anatomic structure or
            morphologically abnormal structure

        Note
        ----
        Either a segmentation, a list of regions or volume needs to referenced
        together with the corresponding source image(s) or series.

        """  # noqa: E501
        if referenced_regions is not None and any(
            isinstance(r, ImageRegion3D) for r in referenced_regions
        ):
            raise TypeError(
                'Including items of type ImageRegion3D in "referenced_regions" '
                'is invalid within a volumetric ROI measurement group is '
                'invalid. To specify the referenced region in 3D frame of '
                'reference coordinates, use the "referenced_volume_surface" '
                'argument instead.'
            )
        if referenced_segment is not None:
            if not isinstance(referenced_segment, ReferencedSegment):
                raise TypeError(
                    'Argument "referenced_segment" must have type '
                    'ReferencedSegment.'
                )
        super().__init__(
            measurements=measurements,
            tracking_identifier=tracking_identifier,
            referenced_regions=referenced_regions,
            referenced_volume_surface=referenced_volume_surface,
            referenced_segment=referenced_segment,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            qualitative_evaluations=qualitative_evaluations,
            geometric_purpose=geometric_purpose,
            finding_category=finding_category
        )
        self[0].ContentTemplateSequence[0].TemplateIdentifier = '1411'

    @property
    def reference_type(self) -> Code:
        """pydicom.sr.coding.Code

        The "type" of the ROI reference as a coded concept. This will be one of
        the following coded concepts from the DCM coding scheme:

        - Image Region
        - Referenced Segment
        - Volume Surface
        - Region In Space

        """
        for item in self[0].ContentSequence:
            for concept_name in self._allowed_roi_reference_types:
                if item.name == concept_name:
                    return concept_name
        else:
            raise RuntimeError(
                'Could not find any allowed ROI reference type.'
            )

    @property
    def roi(
        self
    ) -> Union[VolumeSurface, List[ImageRegion], None]:
        """Union[highdicom.sr.VolumeSurface, List[highdicom.sr.ImageRegion], None]:
        volume surface or image regions defined by spatial coordinates
        """  # noqa: E501
        reference_type = self.reference_type
        if reference_type == codes.DCM.ImageRegion:
            root_item = self[0]
            matches = find_content_items(
                root_item,
                name=codes.DCM.ImageRegion,
                value_type=ValueTypeValues.SCOORD
            )
            return [
                ImageRegion.from_dataset(item)
                for item in matches
            ]
        elif reference_type == codes.DCM.VolumeSurface:
            return VolumeSurface.from_sequence(self[0].ContentSequence)

        return None

    @property
    def _referenced_segment_item(
        self
    ) -> Union[ImageContentItem, None]:
        """Union[highdicom.sr.ReferencedSegment, None]:
        segment or segmentation frame referenced by the measurements group
        """
        root_item = self[0]

        # Find the referenced segment content item
        matches = find_content_items(
            root_item,
            name=codes.DCM.ReferencedSegment,
            value_type=ValueTypeValues.IMAGE
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Referenced Segment" content item '
                'in "Volumetric ROI Measurements and Qualitative Evaluations" '
                'template'
            )
        if len(matches) > 0:
            return ImageContentItem.from_dataset(matches[0])

    @property
    def _source_image_for_segmentation_items(
        self
    ) -> List[SourceImageForSegmentation]:
        """List[highdicom.sr.SourceImageForSegmentation]:
        source images used for the referenced segment
        """
        root_item = self[0]

        # Find the referenced segmentation frame content item
        matches = find_content_items(
            root_item,
            name=codes.DCM.SourceImageForSegmentation,
            value_type=ValueTypeValues.IMAGE,
            relationship_type=RelationshipTypeValues.CONTAINS
        )

        return [
            SourceImageForSegmentation.from_dataset(match) for match in matches
        ]

    @property
    def _source_series_for_segmentation_item(
        self
    ) -> Union[SourceSeriesForSegmentation, None]:
        """Union[highdicom.sr.SourceImageForSegmentation, None]:
        source series used for the referenced segment
        """
        root_item = self[0]

        # Find the referenced segmentation frame content item
        matches = find_content_items(
            root_item,
            name=codes.DCM.SourceSeriesForSegmentation,
            value_type=ValueTypeValues.UIDREF
        )
        if len(matches) > 1:
            logger.warning(
                'found more than one "Source Series For Segmentation" content '
                'item in "Planar ROI Measurements and Qualitative Evaluations" '
                'template'
            )
        elif len(matches) == 0:
            return None

        return SourceSeriesForSegmentation.from_dataset(matches[0])

    @property
    def referenced_segment(
        self
    ) -> Union[ReferencedSegment, None]:
        """Union[highdicom.sr.ImageContentItem, None]:
        segmentation frame referenced by the measurements group
        """
        if self.reference_type == codes.DCM.ReferencedSegment:
            return ReferencedSegment.from_sequence(self[0].ContentSequence)
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> 'VolumetricROIMeasurementsAndQualitativeEvaluations':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing "Measurement Group" SR Content Items
            of Value Type CONTAINER (sequence shall only contain a single item)
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.VolumetricROIMeasurementsAndQualitativeEvaluations
            Content Sequence containing root CONTAINER SR Content Item

        """
        instance = super(
            VolumetricROIMeasurementsAndQualitativeEvaluations,
            cls
        ).from_sequence(sequence)
        instance.__class__ = VolumetricROIMeasurementsAndQualitativeEvaluations
        return cast(
            VolumetricROIMeasurementsAndQualitativeEvaluations,
            instance
        )


class ImageLibraryEntryDescriptors(Template):

    """`TID 1602 <http://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_A.html#sect_TID_1602>`_
     Image Library Entry Descriptors"""  # noqa: E501

    def __init__(
        self,
        modality: Union[Code, CodedConcept],
        frame_of_reference_uid: str,
        pixel_data_rows: int,
        pixel_data_columns: int,
        additional_descriptors: Optional[Sequence[ContentItem]] = None
    ) -> None:
        """
        Parameters
        ----------
        modality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Modality
        frame_of_reference_uid: str
            Frame of Reference UID
        pixel_data_rows: int
            Number of rows in pixel data frames
        pixel_data_columns: int
            Number of rows in pixel data frames
        additional_descriptors: Union[Sequence[highdicom.sr.ContentItem], None], optional
            Additional SR Content Items that should be included

        """  # noqa: E501
        super().__init__()
        modality_item = CodeContentItem(
            name=CodedConcept(
                value='121139',
                meaning='Modality',
                scheme_designator='DCM'
            ),
            value=modality,
            relationship_type=RelationshipTypeValues.HAS_ACQ_CONTEXT
        )
        self.append(modality_item)
        frame_of_reference_uid_item = UIDRefContentItem(
            name=CodedConcept(
                value='112227',
                meaning='Frame of Reference UID',
                scheme_designator='DCM'
            ),
            value=frame_of_reference_uid,
            relationship_type=RelationshipTypeValues.HAS_ACQ_CONTEXT
        )
        self.append(frame_of_reference_uid_item)
        pixel_data_rows_item = NumContentItem(
            name=CodedConcept(
                value='110910',
                meaning='Pixel Data Rows',
                scheme_designator='DCM'
            ),
            value=pixel_data_rows,
            relationship_type=RelationshipTypeValues.HAS_ACQ_CONTEXT,
            unit=CodedConcept(
                value='{pixels}',
                meaning='Pixels',
                scheme_designator='UCUM'
            )
        )
        self.append(pixel_data_rows_item)
        pixel_data_cols_item = NumContentItem(
            name=CodedConcept(
                value='110911',
                meaning='Pixel Data Columns',
                scheme_designator='DCM'
            ),
            value=pixel_data_columns,
            relationship_type=RelationshipTypeValues.HAS_ACQ_CONTEXT,
            unit=CodedConcept(
                value='{pixels}',
                meaning='Pixels',
                scheme_designator='UCUM'
            )
        )
        self.append(pixel_data_cols_item)
        if additional_descriptors is not None:
            for item in additional_descriptors:
                if not isinstance(item, ContentItem):
                    raise TypeError(
                        'Image Library Entry Descriptor must have type '
                        'ContentItem.'
                    )
                relationship_type = RelationshipTypeValues.HAS_ACQ_CONTEXT
                item.RelationshipType = relationship_type.value
                self.append(item)


class MeasurementReport(Template):

    """:dcm:`TID 1500 <part16/chapter_A.html#sect_TID_1500>`
    Measurement Report
    """

    def __init__(
        self,
        observation_context: ObservationContext,
        procedure_reported: Union[
            Union[CodedConcept, Code],
            Sequence[Union[CodedConcept, Code]],
        ],
        imaging_measurements: Optional[
            Sequence[
                Union[
                    PlanarROIMeasurementsAndQualitativeEvaluations,
                    VolumetricROIMeasurementsAndQualitativeEvaluations,
                    MeasurementsAndQualitativeEvaluations,
                ]
            ]
        ] = None,
        title: Optional[Union[CodedConcept, Code]] = None,
        language_of_content_item_and_descendants: Optional[
            LanguageOfContentItemAndDescendants
        ] = None,
        image_library_groups: Optional[
            Sequence[ImageLibraryEntryDescriptors]
        ] = None
    ):
        """

        Parameters
        ----------
        observation_context: highdicom.sr.ObservationContext
            description of the observation context
        procedure_reported: Union[Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code], Sequence[Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]]]
            one or more coded description(s) of the procedure (see
            :dcm:`CID 100 <part16/sect_CID_100.html>`
            "Quantitative Diagnostic Imaging Procedures" for options)
        imaging_measurements: Union[Sequence[Union[highdicom.sr.PlanarROIMeasurementsAndQualitativeEvaluations, highdicom.sr.VolumetricROIMeasurementsAndQualitativeEvaluations, highdicom.sr.MeasurementsAndQualitativeEvaluations]]], optional
            measurements and qualitative evaluations of images or regions
            within images
        title: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            title of the report (see :dcm:`CID 7021 <part16/sect_CID_7021.html>`
            "Measurement Report Document Titles" for options)
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            specification of the language of report content items
            (defaults to English)
        image_library_groups: Union[Sequence[highdicom.sr.ImageLibraryEntry], None], optional
            Entry descriptors for each image library group

        """  # noqa: E501
        if title is None:
            title = codes.cid7021.ImagingMeasurementReport
        if not isinstance(title, (CodedConcept, Code, )):
            raise TypeError(
                'Argument "title" must have type CodedConcept or Code.'
            )
        item = ContainerContentItem(
            name=title,
            template_id='1500'
        )
        item.ContentSequence = ContentSequence()
        if language_of_content_item_and_descendants is None:
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(
            language_of_content_item_and_descendants
        )
        item.ContentSequence.extend(observation_context)
        if isinstance(procedure_reported, (CodedConcept, Code, )):
            procedure_reported = [procedure_reported]
        for procedure in procedure_reported:
            procedure_item = CodeContentItem(
                name=CodedConcept(
                    value='121058',
                    meaning='Procedure reported',
                    scheme_designator='DCM',
                ),
                value=procedure,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            item.ContentSequence.append(procedure_item)

        image_library_item = ImageLibrary(image_library_groups)
        item.ContentSequence.extend(image_library_item)

        measurements: Union[
            MeasurementsAndQualitativeEvaluations,
            PlanarROIMeasurementsAndQualitativeEvaluations,
            VolumetricROIMeasurementsAndQualitativeEvaluations,
        ]
        if imaging_measurements is not None:
            measurement_types = (
                PlanarROIMeasurementsAndQualitativeEvaluations,
                VolumetricROIMeasurementsAndQualitativeEvaluations,
                MeasurementsAndQualitativeEvaluations,
            )
            container_item = ContainerContentItem(
                name=CodedConcept(
                    value='126010',
                    meaning='Imaging Measurements',
                    scheme_designator='DCM'
                ),
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            container_item.ContentSequence = ContentSequence()
            for measurements in imaging_measurements:
                if not isinstance(measurements, measurement_types):
                    raise TypeError(
                        'Measurements must have one of the following types: '
                        '"{}"'.format(
                            '", "'.join(
                                [
                                    t.__name__
                                    for t in measurement_types
                                ]
                            )
                        )
                    )
                container_item.ContentSequence.extend(measurements)
        item.ContentSequence.append(container_item)
        super().__init__([item], is_root=True)

    def _find_measurement_groups(self) -> List[ContainerContentItem]:
        root_item = self[0]
        imaging_measurement_items = find_content_items(
            root_item,
            name=codes.DCM.ImagingMeasurements,
            value_type=ValueTypeValues.CONTAINER
        )
        if len(imaging_measurement_items) == 0:
            return []
        items = find_content_items(
            imaging_measurement_items[0],
            name=codes.DCM.MeasurementGroup,
            value_type=ValueTypeValues.CONTAINER
        )
        return cast(List[ContainerContentItem], items)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = True
    ) -> 'MeasurementReport':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing "Measurement Report" SR Content Items
            of Value Type CONTAINER (sequence shall only contain a single item)
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.MeasurementReport
            Content Sequence containing root CONTAINER SR Content Item

        """
        if len(sequence) == 0:
            raise ValueError('Sequence contains no SR Content Items.')
        if len(sequence) > 1:
            raise ValueError(
                'Sequence contains more than one SR Content Item.'
            )
        dataset = sequence[0]
        if dataset.ValueType != ValueTypeValues.CONTAINER.value:
            raise ValueError(
                'Item #1 of sequence is not an appropriate SR Content Item '
                'because it does not have Value Type CONTAINER.'
            )
        if dataset.ContentTemplateSequence[0].TemplateIdentifier != '1500':
            raise ValueError(
                'Item #1 of sequence is not an appropriate SR Content Item '
                'because it does not have Template Identifier "1500".'
            )
        instance = ContentSequence.from_sequence(sequence, is_root=True)
        instance.__class__ = MeasurementReport
        return cast(MeasurementReport, instance)

    def get_observer_contexts(
        self,
        observer_type: Optional[Union[CodedConcept, Code]] = None
    ) -> List[ObserverContext]:
        """Get observer contexts.

        Parameters
        ----------
        observer_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of observer ("Device" or "Person") for which should be filtered

        Returns
        -------
        List[highdicom.sr.ObserverContext]
            Observer contexts

        """  # noqa: E501
        root_item = self[0]
        matches = [
            (i, item) for i, item in enumerate(root_item.ContentSequence, 1)
            if item.name == codes.DCM.ObserverType
        ]
        observer_contexts = []
        attributes: Union[
            DeviceObserverIdentifyingAttributes,
            PersonObserverIdentifyingAttributes,
        ]
        for i, (index, item) in enumerate(matches):
            if observer_type is not None:
                if item.value != observer_type:
                    continue
            try:
                next_index = matches[i + 1][0]
            except IndexError:
                next_index = -1
            if item.value == codes.DCM.Device:
                attributes = DeviceObserverIdentifyingAttributes.from_sequence(
                    sequence=root_item.ContentSequence[index:next_index]
                )
            elif item.value == codes.DCM.Person:
                attributes = PersonObserverIdentifyingAttributes.from_sequence(
                    sequence=root_item.ContentSequence[index:next_index]
                )
            else:
                raise ValueError('Unexpected observer type "{item.meaning}".')
            context = ObserverContext(
                observer_type=item.value,
                observer_identifying_attributes=attributes
            )
            observer_contexts.append(context)
        return observer_contexts

    def get_subject_contexts(
        self,
        subject_class: Optional[CodedConcept] = None
    ) -> List[SubjectContext]:
        """Get subject contexts.

        Parameters
        ----------
        subject_class: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of subject ("Specimen", "Fetus", or "Device") for which should
            be filtered

        Returns
        -------
        List[highdicom.sr.SubjectContext]
           Subject contexts

        """  # noqa: E501
        root_item = self[0]
        matches = [
            (i + 1, item) for i, item in enumerate(root_item.ContentSequence)
            if item.name == codes.DCM.SubjectClass
        ]
        subject_contexts = []
        attributes: Union[
            SubjectContextSpecimen,
            SubjectContextFetus,
            SubjectContextDevice,
        ]
        for i, (index, item) in enumerate(matches):
            if subject_class is not None:
                if item.value != subject_class:
                    continue
            try:
                next_index = matches[i + 1][0]
            except IndexError:
                next_index = -1
            if item.value == codes.DCM.Specimen:
                attributes = SubjectContextSpecimen.from_sequence(
                    sequence=root_item.ContentSequence[index:next_index]
                )
            elif item.value == codes.DCM.Fetus:
                attributes = SubjectContextFetus.from_sequence(
                    sequence=root_item.ContentSequence[index:next_index]
                )
            elif item.value == codes.DCM.Device:
                attributes = SubjectContextDevice.from_sequence(
                    sequence=root_item.ContentSequence[index:next_index]
                )
            else:
                raise ValueError('Unexpected subject class "{item.meaning}".')
            context = SubjectContext(
                subject_class=item.value,
                subject_class_specific_context=attributes
            )
            subject_contexts.append(context)
        return subject_contexts

    def get_planar_roi_measurement_groups(
        self,
        tracking_uid: Optional[str] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        finding_site: Optional[Union[CodedConcept, Code]] = None,
        reference_type: Optional[Union[CodedConcept, Code]] = None,
        graphic_type: Optional[
            Union[GraphicTypeValues, GraphicTypeValues3D]
        ] = None,
        referenced_sop_instance_uid: Optional[str] = None,
        referenced_sop_class_uid: Optional[str] = None
    ) -> List[PlanarROIMeasurementsAndQualitativeEvaluations]:
        """Get imaging measurement groups of planar regions of interest.

        Finds (and optionally filters) content items contained in the
        CONTAINER content item "Measurement group" as specified by TID 1410
        "Planar ROI Measurements and Qualitative Evaluations".

        Parameters
        ----------
        tracking_uid: Union[str, None], optional
            Unique tracking identifier
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Finding
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Finding site
        reference_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of referenced ROI. Valid values are limited to codes
            `ImageRegion`, `ReferencedSegmentationFrame`, and `RegionInSpace`.
        graphic_type: Union[highdicom.sr.GraphicTypeValues, highdicom.sr.GraphicTypeValues3D, None], optional
            Graphic type of image region
        referenced_sop_instance_uid: Union[str, None], optional
            SOP Instance UID of the referenced instance, which may be a
            segmentation image, source image for the region or segmentation, or
            RT struct, depending on `reference_type`
        referenced_sop_class_uid: Union[str, None], optional
            SOP Class UID of the referenced instance, which may be a
            segmentation image, source image for the region or segmentation, or
            RT struct, depending on `reference_type`

        Returns
        -------
        List[highdicom.sr.PlanarROIMeasurementsAndQualitativeEvaluations]
            Sequence of content items for each matched measurement group

        """  # noqa: E501
        if graphic_type is not None:
            if not isinstance(
                graphic_type,
                (GraphicTypeValues, GraphicTypeValues3D)
            ):
                raise TypeError(
                    'Argument "graphic_type" must be of type '
                    'GraphicTypeValues, GraphicTypeValues3D, or None.'
                )
            if isinstance(graphic_type, GraphicTypeValues):
                if graphic_type == GraphicTypeValues.MULTIPOINT:
                    raise ValueError(
                        'Graphic type "MULTIPOINT" is not valid for image '
                        'regions within a planar ROI measurements group.'
                    )
            else:
                if graphic_type in (
                    GraphicTypeValues3D.MULTIPOINT,
                    GraphicTypeValues3D.POLYLINE,
                    GraphicTypeValues3D.ELLIPSOID
                ):
                    raise ValueError(
                        f'Graphic type 3D value "{graphic_type}" is not valid '
                        'for image regions within a planar ROI measurements '
                        'group.'
                    )
                # There is no way to check SCOORD3D for referenced UIDs
                if (
                    (referenced_sop_class_uid is not None) or
                    (referenced_sop_instance_uid is not None)
                ):
                    raise TypeError(
                        'Supplying a referenced_sop_class_uid or '
                        'referenced_sop_instance_uidis not valid'
                        'when graphic_type is an instance of '
                        'GraphicTypeValues3D, since SCOORD3D content items do '
                        'not contain references to specific source image '
                        'instances.'
                    )

        # Check a valid code was passed
        if reference_type is not None:
            allowed_vals = PlanarROIMeasurementsAndQualitativeEvaluations.\
                _allowed_roi_reference_types
            if reference_type not in allowed_vals:
                raise ValueError(
                    f'Concept {reference_type} is not valid as a reference '
                    'type in Planar ROI Measurements and Qualitative '
                    'Evaluations.'
                )

            # Check for input options incompatible with this reference type
            if graphic_type is not None:
                if reference_type != codes.DCM.ImageRegion:
                    raise ValueError(
                        'Specifying a graphic type is invalid when using '
                        f'reference type "{reference_type.meaning}"'
                    )

        measurement_group_items = self._find_measurement_groups()
        sequences = []
        for group_item in measurement_group_items:
            if group_item.template_id is not None:
                if group_item.template_id != '1410':
                    continue
            else:
                if not _contains_planar_rois(group_item):
                    continue

            matches = []
            if finding_type is not None:
                matches_finding = _contains_code_items(
                    group_item,
                    name=codes.DCM.Finding,
                    value=finding_type,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                matches.append(matches_finding)
            if finding_site is not None:
                matches_finding_sites = _contains_code_items(
                    group_item,
                    name=codes.SCT.FindingSite,
                    value=finding_site,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                matches.append(matches_finding_sites)
            if tracking_uid is not None:
                matches_tracking_uid = _contains_uidref_items(
                    group_item,
                    name=codes.DCM.TrackingUniqueIdentifier,
                    value=tracking_uid,
                    relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
                )
                matches.append(matches_tracking_uid)

            # Remaining checks all relate to the single content item that
            # describes the ROI reference
            if (
                (reference_type is not None) or
                (graphic_type is not None) or
                (referenced_sop_class_uid is not None) or
                (referenced_sop_instance_uid is not None)
            ):
                # Find the content item representing the ROI reference
                found_ref_type, ref_item = _get_planar_roi_reference_item(
                    group_item
                )
                ref_value_type = ValueTypeValues(ref_item.ValueType)

                if reference_type is not None:
                    matches.append(found_ref_type == reference_type)

                if graphic_type is not None:
                    found_gt: Union[GraphicTypeValues, GraphicTypeValues3D]
                    if isinstance(graphic_type, GraphicTypeValues):
                        if ref_value_type == ValueTypeValues.SCOORD:
                            found_gt = GraphicTypeValues(ref_item.GraphicType)
                            matches.append(found_gt == graphic_type)
                        else:
                            matches.append(False)
                    else:
                        if ref_value_type == ValueTypeValues.SCOORD3D:
                            found_gt = GraphicTypeValues3D(ref_item.GraphicType)
                            matches.append(found_gt == graphic_type)
                        else:
                            matches.append(False)

                if (
                    (referenced_sop_instance_uid is not None) or
                    (referenced_sop_class_uid is not None)
                ):
                    matches_uids = False

                    # Check the references directly in the content item for
                    # IMAGE or COMPOSITE items
                    if found_ref_type in [
                        codes.DCM.ReferencedSegmentationFrame,
                        _REGION_IN_SPACE
                    ]:
                        sop_seq = ref_item.ReferencedSOPSequence[0]
                        matches_instance_uid = (
                            referenced_sop_instance_uid is None or (
                                sop_seq.ReferencedSOPInstanceUID ==
                                referenced_sop_instance_uid
                            )
                        )
                        matches_class_uid = (
                            referenced_sop_class_uid is None or (
                                sop_seq.ReferencedSOPClassUID ==
                                referenced_sop_class_uid
                            )
                        )
                        if matches_class_uid and matches_instance_uid:
                            matches_uids = True

                    if found_ref_type == codes.DCM.ImageRegion:
                        # If 2D image region, check items in its content
                        # sequence for source images
                        if ref_item.value_type == ValueTypeValues.SCOORD:
                            # (SCOORD3 will not contain direct UID
                            # references)
                            if _contains_image_items(
                                ref_item,
                                name=None,
                                referenced_sop_class_uid=referenced_sop_class_uid,  # noqa: E501
                                referenced_sop_instance_uid=referenced_sop_instance_uid,  # noqa: E501
                                relationship_type=RelationshipTypeValues.SELECTED_FROM  # noqa: E501
                            ):
                                matches_uids = True

                    if found_ref_type == codes.DCM.ReferencedSegmentationFrame:
                        # Check for IMAGE item of SourceImageForSegmentation at
                        # the top level
                        if _contains_image_items(
                            group_item,
                            name=codes.DCM.SourceImageForSegmentation,
                            referenced_sop_class_uid=referenced_sop_class_uid,
                            referenced_sop_instance_uid=referenced_sop_instance_uid,  # noqa: E501
                            relationship_type=RelationshipTypeValues.CONTAINS
                        ):
                            matches_uids = True

                    matches.append(matches_uids)

            if len(matches) == 0 or all(matches):
                seq = PlanarROIMeasurementsAndQualitativeEvaluations.from_sequence(  # noqa: E501
                    [group_item]
                )
                sequences.append(seq)

        return sequences

    def get_volumetric_roi_measurement_groups(
        self,
        tracking_uid: Optional[str] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        finding_site: Optional[Union[CodedConcept, Code]] = None,
        reference_type: Optional[Union[CodedConcept, Code]] = None,
        graphic_type: Optional[GraphicTypeValues3D] = None,
        referenced_sop_instance_uid: Optional[str] = None,
        referenced_sop_class_uid: Optional[str] = None
    ) -> List[VolumetricROIMeasurementsAndQualitativeEvaluations]:
        """Get imaging measurement groups of volumetric regions of interest.

        Finds (and optionally filters) content items contained in the
        CONTAINER content item "Measurement group" as specified by TID 1411
        "Volumetric ROI Measurements and Qualitative Evaluations".

        Parameters
        ----------
        tracking_uid: Union[str, None], optional
            Unique tracking identifier
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Finding
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Finding site
        reference_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of referenced ROI. Valid values are limited to codes
            `ImageRegion`, `ReferencedSegment`, `VolumeSurface` and
            `RegionInSpace`.
        graphic_type: Union[highdicom.sr.GraphicTypeValues, highdicom.sr.GraphicTypeValues3D, None], optional
            Graphic type of image region
        referenced_sop_instance_uid: Union[str, None], optional
            SOP Instance UID of the referenced instance, which may be a
            segmentation image, source image for the region or segmentation, or
            RT struct, depending on `reference_type`
        referenced_sop_class_uid: Union[str, None], optional
            SOP Class UID of the referenced instance, which may be a
            segmentation image, source image for the region or segmentation, or
            RT struct, depending on `reference_type`

        Returns
        -------
        List[highdicom.sr.VolumetricROIMeasurementsAndQualitativeEvaluations]
            Sequence of content items for each matched measurement group

        """  # noqa: E501
        if graphic_type is not None:
            if not isinstance(
                graphic_type,
                (GraphicTypeValues, GraphicTypeValues3D)
            ):
                raise TypeError(
                    'graphic_type must be of type GraphicTypeValues or '
                    'GraphicTypeValues3D, or None.'
                )
            if isinstance(graphic_type, GraphicTypeValues):
                if graphic_type == GraphicTypeValues.MULTIPOINT:
                    raise ValueError(
                        'Graphic type "MULTIPOINT" is not valid for image '
                        'regions within a volumetric ROI measurements group.'
                    )
            else:
                if graphic_type in (
                    GraphicTypeValues3D.MULTIPOINT,
                    GraphicTypeValues3D.POLYLINE,
                ):
                    raise ValueError(
                        f'Graphic type 3D value "{graphic_type}" is not valid '
                        'for image regions within a planar ROI measurements '
                        'group.'
                    )
                # There is no way to check SCOORD3D for referenced UIDs
                if (
                    (referenced_sop_class_uid is not None) or
                    (referenced_sop_instance_uid is not None)
                ):
                    raise TypeError(
                        'Supplying a referenced_sop_class_uid or '
                        'referenced_sop_instance_uidis not valid'
                        'when graphic_type is an instance of '
                        'GraphicTypeValues3D, since SCOORD3D content items do '
                        'not contain references to specific source image '
                        'instances.'
                    )

        # Check a valid code was passed
        if reference_type is not None:
            allowed_vals = VolumetricROIMeasurementsAndQualitativeEvaluations.\
                _allowed_roi_reference_types
            if reference_type not in allowed_vals:
                raise ValueError(
                    f'Concept {reference_type} is not valid as a reference '
                    'type in Volumetric ROI Measurements and Qualitative '
                    'Evaluations.'
                )

            # Check for input options incompatible with this reference type
            if graphic_type is not None:
                ref_types_with_graphics = [
                    codes.DCM.ImageRegion,
                    codes.DCM.VolumeSurface
                ]
                if reference_type not in ref_types_with_graphics:
                    raise ValueError(
                        'Specifying a graphic type is invalid when using '
                        f'a reference type "{reference_type.meaning}"'
                    )

                # Check incompatibility of graphic_type and reference_type
                if reference_type == codes.DCM.ImageRegion:
                    if isinstance(graphic_type, GraphicTypeValues3D):
                        raise TypeError(
                            'When specifying a reference type of '
                            '"Image Region", the "graphic_type" argument must '
                            'be of type GraphicTypeValues.'
                        )
                elif reference_type == codes.DCM.VolumeSurface:
                    if isinstance(graphic_type, GraphicTypeValues):
                        raise TypeError(
                            'When specifying a reference type of '
                            '"Volume Surface", the "graphic_type" argument '
                            'must be of type GraphicTypeValues3D.'
                        )

        sequences = []
        measurement_group_items = self._find_measurement_groups()
        for group_item in measurement_group_items:
            if group_item.template_id is not None:
                if group_item.template_id != '1411':
                    continue
            else:
                if not _contains_volumetric_rois(group_item):
                    continue

            matches = []
            if finding_type is not None:
                matches_finding = _contains_code_items(
                    group_item,
                    name=codes.DCM.Finding,
                    value=finding_type,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                matches.append(matches_finding)
            if finding_site is not None:
                matches_finding_sites = _contains_code_items(
                    group_item,
                    name=codes.SCT.FindingSite,
                    value=finding_site,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                matches.append(matches_finding_sites)
            if tracking_uid is not None:
                matches_tracking_uid = _contains_uidref_items(
                    group_item,
                    name=codes.DCM.TrackingUniqueIdentifier,
                    value=tracking_uid,
                    relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
                )
                matches.append(matches_tracking_uid)

            # Remaining checks all relate to the content items that
            # describes the ROI reference
            if (
                (reference_type is not None) or
                (graphic_type is not None) or
                (referenced_sop_class_uid is not None) or
                (referenced_sop_instance_uid is not None)
            ):
                # Find the contents item representing the ROI reference
                found_ref_type, ref_items = _get_volumetric_roi_reference_items(
                    group_item
                )
                ref_value_type = ValueTypeValues(ref_items[0].ValueType)

                if reference_type is not None:
                    matches.append(found_ref_type == reference_type)

                if graphic_type is not None:
                    found_gt: Union[GraphicTypeValues, GraphicTypeValues3D]
                    if isinstance(graphic_type, GraphicTypeValues):
                        if ref_value_type == ValueTypeValues.SCOORD:
                            found_gt = GraphicTypeValues(
                                ref_items[0].GraphicType
                            )
                            matches.append(found_gt == graphic_type)
                        else:
                            matches.append(False)
                    else:
                        if ref_value_type == ValueTypeValues.SCOORD3D:
                            found_gt = GraphicTypeValues3D(
                                ref_items[0].GraphicType
                            )
                            matches.append(found_gt == graphic_type)
                        else:
                            matches.append(False)

                if (
                    (referenced_sop_instance_uid is not None) or
                    (referenced_sop_class_uid is not None)
                ):
                    matches_uids = False

                    # Check the references directly in the content item for
                    # IMAGE or COMPOSITE items. In these cases there will be a
                    # single item
                    if found_ref_type in [
                        codes.DCM.ReferencedSegment,
                        _REGION_IN_SPACE
                    ]:
                        sop_seq = ref_items[0].ReferencedSOPSequence[0]
                        matches_instance_uid = (
                            referenced_sop_instance_uid is None or (
                                sop_seq.ReferencedSOPInstanceUID ==
                                referenced_sop_instance_uid
                            )
                        )
                        matches_class_uid = (
                            referenced_sop_class_uid is None or (
                                sop_seq.ReferencedSOPClassUID ==
                                referenced_sop_class_uid
                            )
                        )
                        if matches_class_uid and matches_instance_uid:
                            matches_uids = True

                    if found_ref_type == codes.DCM.ImageRegion:
                        # If 2D image region, check items in its content
                        # sequence for source images
                        for ref_item in ref_items:
                            if ref_item.value_type == ValueTypeValues.SCOORD:
                                # (SCOORD3 will not contain direct UID
                                # references)
                                if _contains_image_items(
                                    ref_item,
                                    name=None,
                                    referenced_sop_class_uid=referenced_sop_class_uid,  # noqa: E501
                                    referenced_sop_instance_uid=referenced_sop_instance_uid,  # noqa: E501
                                    relationship_type=RelationshipTypeValues.SELECTED_FROM  # noqa: E501
                                ):
                                    matches_uids = True

                    if found_ref_type == codes.DCM.ReferencedSegment:
                        # Check for IMAGE item of SourceImageForSegmentation at
                        # the top level
                        if _contains_image_items(
                            group_item,
                            name=codes.DCM.SourceImageForSegmentation,
                            referenced_sop_class_uid=referenced_sop_class_uid,
                            referenced_sop_instance_uid=referenced_sop_instance_uid,  # noqa: E501
                            relationship_type=RelationshipTypeValues.CONTAINS
                        ):
                            matches_uids = True

                    matches.append(matches_uids)

            if len(matches) == 0 or all(matches):
                seq = VolumetricROIMeasurementsAndQualitativeEvaluations.from_sequence(  # noqa: E501
                    [group_item]
                )
                sequences.append(seq)

        return sequences

    def get_image_measurement_groups(
        self,
        tracking_uid: Optional[str] = None,
        finding_type: Optional[Union[CodedConcept, Code]] = None,
        finding_site: Optional[Union[CodedConcept, Code]] = None,
        referenced_sop_instance_uid: Optional[str] = None,
        referenced_sop_class_uid: Optional[str] = None
    ) -> List[MeasurementsAndQualitativeEvaluations]:
        """Get imaging measurements of images.

        Finds (and optionally filters) content items contained in the
        CONTAINER content item "Measurement Group" as specified by TID 1501
        "Measurement and Qualitative Evaluation Group".

        Parameters
        ----------
        tracking_uid: Union[str, None], optional
            Unique tracking identifier
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Finding
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Finding site
        referenced_sop_instance_uid: Union[str, None], optional
            SOP Instance UID of the referenced instance.
        referenced_sop_class_uid: Union[str, None], optional
            SOP Class UID of the referenced instance.

        Returns
        -------
        List[highdicom.sr.MeasurementsAndQualitativeEvaluations]
            Sequence of content items for each matched measurement group

        """  # noqa: E501
        measurement_group_items = self._find_measurement_groups()
        sequences = []
        for group_item in measurement_group_items:
            if group_item.template_id is not None:
                if group_item.template_id != '1501':
                    continue
            else:
                contains_rois = _contains_planar_rois(group_item)
                contains_rois |= _contains_volumetric_rois(group_item)
                if contains_rois:
                    continue

            matches = []
            if finding_type is not None:
                matches_finding = _contains_code_items(
                    group_item,
                    name=codes.DCM.Finding,
                    value=finding_type,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                matches.append(matches_finding)
            if finding_site is not None:
                matches_finding_sites = _contains_code_items(
                    group_item,
                    name=codes.SCT.FindingSite,
                    value=finding_site,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                matches.append(matches_finding_sites)
            if tracking_uid is not None:
                matches_tracking_uid = _contains_uidref_items(
                    group_item,
                    name=codes.DCM.TrackingUniqueIdentifier,
                    value=tracking_uid,
                    relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
                )
                matches.append(matches_tracking_uid)

            if (
                (referenced_sop_instance_uid is not None) or
                (referenced_sop_class_uid is not None)
            ):
                matches_uids = _contains_image_items(
                    group_item,
                    name=_SOURCE,
                    referenced_sop_class_uid=referenced_sop_class_uid,
                    referenced_sop_instance_uid=referenced_sop_instance_uid,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                matches.append(matches_uids)

            seq = MeasurementsAndQualitativeEvaluations.from_sequence(
                [group_item]
            )
            if len(matches) == 0:
                sequences.append(seq)
            else:
                if all(matches):
                    sequences.append(seq)

        return sequences


class ImageLibrary(Template):

    """:dcm:`TID 1600 <part16/chapter_A.html#sect_TID_1600>` Image Library"""

    def __init__(
        self,
        groups: Optional[Sequence[ImageLibraryEntryDescriptors]] = None
    ) -> None:
        """
        Parameters
        ----------
        groups: Union[Sequence[Sequence[highdicom.sr.ImageLibraryEntryDescriptors]], None], optional
            Entry descriptors for each image library group

        """  # noqa: E501
        super().__init__()
        library_item = ContainerContentItem(
            name=CodedConcept(
                value='111028',
                meaning='Image Library',
                scheme_designator='DCM'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if groups is not None:
            for descriptor_items in groups:
                group_item = ContainerContentItem(
                    name=CodedConcept(
                        value='126200',
                        meaning='Image Library Group',
                        scheme_designator='DCM'
                    ),
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                group_item.ContentSequence = descriptor_items
                # The Image Library Entry template contains the individual
                # Image Library Entry Descriptors content items.
                if not isinstance(descriptor_items,
                                  ImageLibraryEntryDescriptors):
                    raise TypeError(
                        'Image library group items must have type '
                        '"ImageLibraryEntry".'
                    )
                content.append(group_item)
        if len(content) > 0:
            library_item.ContentSequence = content
        self.append(library_item)
