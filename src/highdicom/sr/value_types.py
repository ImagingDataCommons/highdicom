"""DICOM structured reporting content item value types."""
import datetime
from collections import namedtuple
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
from pydicom.uid import UID
from pydicom.valuerep import DA, TM, DT, PersonName

from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import (
    GraphicTypeValues,
    GraphicTypeValues3D,
    PixelOriginInterpretationValues,
    RelationshipTypeValues,
    TemporalRangeTypeValues,
    ValueTypeValues,
)


class ContentItem(Dataset):

    """Abstract base class for a collection of attributes contained in the
    DICOM SR Document Content Module."""

    def __init__(
            self,
            value_type: Union[str, ValueTypeValues],
            name: Union[Code, CodedConcept],
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        value_type: Union[str, highdicom.sr.enum.ValueTypeValues]
            type of value encoded in a content item
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            coded name or an enumerated item representing a coded name
        relationship_type: Union[str, highdicom.sr.enum.RelationshipTypeValues], optional
            type of relationship with parent content item

        """  # noqa
        super(ContentItem, self).__init__()
        value_type = ValueTypeValues(value_type)
        self.ValueType = value_type.value
        if not isinstance(name, (CodedConcept, Code, )):
            raise TypeError(
                'Argument "name" must have type CodedConcept or Code.'
            )
        if isinstance(name, Code):
            name = CodedConcept(*name)
        self.ConceptNameCodeSequence = [name]
        if relationship_type is not None:
            relationship_type = RelationshipTypeValues(relationship_type)
            self.RelationshipType = relationship_type.value

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'ContentSequence':
            super(ContentItem, self).__setattr__(name, ContentSequence(value))
        else:
            super(ContentItem, self).__setattr__(name, value)

    @property
    def name(self) -> CodedConcept:
        """CodedConcept: coded name of the content item"""
        return self.ConceptNameCodeSequence[0]

    @property
    def value_type(self) -> str:
        """str: type of the content item
        (see `highdicom.sr.value_types.ValueTypeValues`)

        """
        return self.ValueType

    @property
    def relationship_type(self) -> str:
        """str: type of relationship the content item has with its parent
        (see `highdicom.sr.enum.RelationshipTypeValues`)

        """
        return getattr(self, 'RelationshipType', None)


class ContentSequence(DataElementSequence):

    """Sequence of DICOM SR Content Items."""

    def __init__(self, items: Optional[Sequence] = None) -> None:
        if items is not None:
            if not all(isinstance(i, ContentItem) for i in items):
                raise TypeError(
                    'Items of "{}" must have type ContentItem.'.format(
                        self.__class__.__name__
                    )
                )
        super(ContentSequence, self).__init__(items)

    def __setitem__(self, position: int, item: ContentItem) -> None:
        self.insert(position, item)

    def __contains__(self, item: ContentItem) -> bool:
        return any(contained_item == item for contained_item in self)

    def get_nodes(self) -> 'ContentSequence':
        """Gets content items that represent nodes in the content tree, i.e.
        target items that have a `ContentSequence` attribute.

        Returns
        -------
        highdicom.sr.value_types.ContentSequence[highdicom.sr.value_types.ContentItem]
            matched content items

        """
        return self.__class__([
            item for item in self
            if hasattr(item, 'ContentSequence')
        ])

    def append(self, item: ContentItem) -> None:
        """Appends a content item to the sequence.

        Parameters
        ----------
        item: highdicom.sr.value_types.ContentItem
            content item

        """
        if not isinstance(item, ContentItem):
            raise TypeError(
                'Items of "{}" must have type ContentItem.'.format(
                    self.__class__.__name__
                )
            )
        super(ContentSequence, self).append(item)

    def extend(self, items: Sequence[ContentItem]) -> None:
        """Extends multiple content items to the sequence.

        Parameters
        ----------
        items: Sequence[highdicom.sr.value_types.ContentItem]
            content items

        """
        for i in items:
            self.append(i)

    def insert(self, position: int, item: ContentItem) -> None:
        """Inserts a content item into the sequence at a given position.

        Parameters
        ----------
        position: int
            index position
        item: highdicom.sr.value_types.ContentItem
            content item

        """
        if not isinstance(item, ContentItem):
            raise TypeError(
                'Items of "{}" must have type ContentItem.'.format(
                    self.__class__.__name__
                )
            )
        super(ContentSequence, self).insert(position, item)


class CodeContentItem(ContentItem):

    """DICOM SR document content item for value type CODE."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[Code, CodedConcept],
        relationship_type: Optional[
            Union[str, RelationshipTypeValues]
        ] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            coded value or an enumerated item representing a coded value
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(CodeContentItem, self).__init__(
            ValueTypeValues.CODE, name, relationship_type
        )
        if not isinstance(value, (CodedConcept, Code, )):
            raise TypeError(
                'Argument "value" must have type CodedConcept or Code.'
            )
        if isinstance(value, Code):
            value = CodedConcept(*value)
        self.ConceptCodeSequence = [value]


class PnameContentItem(ContentItem):

    """DICOM SR document content item for value type PNAME."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[str, PersonName],
        relationship_type: Optional[
            Union[str, RelationshipTypeValues]
        ] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, pydicom.valuerep.PersonName]
            name of the person
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(PnameContentItem, self).__init__(
            ValueTypeValues.PNAME, name, relationship_type
        )
        self.PersonName = PersonName(value)


class TextContentItem(ContentItem):

    """DICOM SR document content item for value type TEXT."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            value: str,
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: str
            description of the concept in free text
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """ # noqa
        super(TextContentItem, self).__init__(
            ValueTypeValues.TEXT, name, relationship_type
        )
        self.TextValue = str(value)


class TimeContentItem(ContentItem):

    """DICOM SR document content item for value type TIME."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            value: Union[str, datetime.time, TM],
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, datetime.time, pydicom.valuerep.TM]
            time
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(TimeContentItem, self).__init__(
            ValueTypeValues.TIME, name, relationship_type
        )
        self.Time = TM(value)


class DateContentItem(ContentItem):

    """DICOM SR document content item for value type DATE."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            value: Union[str, datetime.date, DA],
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, datetime.date, pydicom.valuerep.DA]
            date
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(DateContentItem, self).__init__(
            ValueTypeValues.DATE, name, relationship_type
        )
        self.Date = DA(value)


class DateTimeContentItem(ContentItem):

    """DICOM SR document content item for value type DATETIME."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            value: Union[str, datetime.datetime, DT],
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, datetime.datetime, pydicom.valuerep.DT]
            datetime
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(DateTimeContentItem, self).__init__(
            ValueTypeValues.DATETIME, name, relationship_type
        )
        self.DateTime = DT(value)


class UIDRefContentItem(ContentItem):

    """DICOM SR document content item for value type UIDREF."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            value: Union[str, UID],
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[pydicom.uid.UID, str]
            unique identifier
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(UIDRefContentItem, self).__init__(
            ValueTypeValues.UIDREF, name, relationship_type
        )
        self.UID = value


class NumContentItem(ContentItem):

    """DICOM SR document content item for value type NUM."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            value: Optional[Union[int, float]] = None,
            unit: Optional[Union[Code, CodedConcept]] = None,
            qualifier: Optional[Union[Code, CodedConcept]] = None,
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[int, float], optional
            numeric value
        unit: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded units of measurement (see CID 7181 "Abstract Multi-dimensional
            Image Model Component Units")
        qualifier: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            qualification of numeric value or as an alternative to
            numeric value, e.g., reason for absence of numeric value
            (see CID 42 "Numeric Value Qualifier" for options)
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        Note
        ----
        Either `value` and `unit` or `qualifier` must be specified.

        """ # noqa
        super(NumContentItem, self).__init__(
            ValueTypeValues.NUM, name, relationship_type
        )
        if value is not None:
            self.MeasuredValueSequence: List[Dataset] = []
            measured_value_sequence_item = Dataset()
            if not isinstance(value, (int, float, )):
                raise TypeError(
                    'Argument "value" must have type "int" or "float".'
                )
            measured_value_sequence_item.NumericValue = value
            if isinstance(value, float):
                measured_value_sequence_item.FloatingPointValue = value
            if not isinstance(unit, (CodedConcept, Code, )):
                raise TypeError(
                    'Argument "unit" must have type CodedConcept or Code.'
                )
            if isinstance(unit, Code):
                unit = CodedConcept(*unit)
            measured_value_sequence_item.MeasurementUnitsCodeSequence = [unit]
            self.MeasuredValueSequence.append(measured_value_sequence_item)
        elif qualifier is not None:
            if not isinstance(qualifier, (CodedConcept, Code, )):
                raise TypeError(
                    'Argument "qualifier" must have type "CodedConcept" or '
                    '"Code".'
                )
            if isinstance(qualifier, Code):
                qualifier = CodedConcept(*qualifier)
            self.NumericValueQualifierCodeSequence = [qualifier]
        else:
            raise ValueError(
                'Either argument "value" or "qualifier" must be specified '
                'upon creation of NumContentItem.'
            )


class ContainerContentItem(ContentItem):

    """DICOM SR document content item for value type CONTAINER."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            is_content_continuous: bool = True,
            template_id: Optional[str] = None,
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        is_content_continous: bool, optional
            whether contained content items are logically linked in a
            continuous manner or separate items (default: ``True``)
        template_id: str, optional
            SR template identifier
        relationship_type: str, optional
            type of relationship with parent content item

        """
        super(ContainerContentItem, self).__init__(
            ValueTypeValues.CONTAINER, name, relationship_type
        )
        if is_content_continuous:
            self.ContinuityOfContent = 'CONTINUOUS'
        else:
            self.ContinuityOfContent = 'SEPARATE'
        if template_id is not None:
            item = Dataset()
            item.MappingResource = 'DCMR'
            item.TemplateIdentifier = str(template_id)
            self.ContentTemplateSequence = [item]


class CompositeContentItem(ContentItem):

    """DICOM SR document content item for value type COMPOSITE."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            referenced_sop_class_uid: Union[str, UID],
            referenced_sop_instance_uid: Union[str, UID],
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ):
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        referenced_sop_class_uid: Union[pydicom.uid.UID, str]
            SOP Class UID of the referenced object
        referenced_sop_instance_uid: Union[pydicom.uid.UID, str]
            SOP Instance UID of the referenced object
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(CompositeContentItem, self).__init__(
            ValueTypeValues.COMPOSITE, name, relationship_type
        )
        item = Dataset()
        item.ReferencedSOPClassUID = str(referenced_sop_class_uid)
        item.ReferencedSOPInstanceUID = str(referenced_sop_instance_uid)
        self.ReferencedSOPSequence = [item]


class ImageContentItem(ContentItem):

    """DICOM SR document content item for value type IMAGE."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            referenced_sop_class_uid: Union[str, UID],
            referenced_sop_instance_uid: Union[str, UID],
            referenced_frame_numbers: Optional[
                Union[int, Sequence[int]]
            ] = None,
            referenced_segment_numbers: Optional[
                Union[int, Sequence[int]]
            ] = None,
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        referenced_sop_class_uid: Union[pydicom.uid.UID, str]
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: Union[pydicom.uid.UID, str]
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[int, Sequence[int]], optional
            number of frame(s) to which the reference applies in case of a
            multi-frame image
        referenced_segment_numbers: Union[int, Sequence[int]], optional
            number of segment(s) to which the refernce applies in case of a
            segmentation image
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(ImageContentItem, self).__init__(
            ValueTypeValues.IMAGE, name, relationship_type
        )
        item = Dataset()
        item.ReferencedSOPClassUID = str(referenced_sop_class_uid)
        item.ReferencedSOPInstanceUID = str(referenced_sop_instance_uid)
        if referenced_frame_numbers is not None:
            item.ReferencedFrameNumber = referenced_frame_numbers
        if referenced_segment_numbers is not None:
            item.ReferencedSegmentNumber = referenced_segment_numbers
        self.ReferencedSOPSequence = [item]


class ScoordContentItem(ContentItem):

    """DICOM SR document content item for value type SCOORD.

    Note
    ----
    Spatial coordinates are defined in image space and have pixel units.

    """

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            graphic_type: Union[str, GraphicTypeValues],
            graphic_data: np.ndarray,
            pixel_origin_interpretation: Union[
                str,
                PixelOriginInterpretationValues
            ],
            fiducial_uid: Optional[Union[str, UID]] = None,
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        graphic_type: Union[highdicom.sr.enum.GraphicTypeValues, str]
            name of the graphic type
        graphic_data: numpy.ndarray[numpy.int]
            array of ordered spatial coordinates, where each row of the array
            represents a (column, row) coordinate pair
        pixel_origin_interpretation: Union[highdicom.sr.enum.PixelOriginInterpretationValues, str]
            whether pixel coordinates specified by `graphic_data` are defined
            relative to the total pixel matrix
            (``highdicom.sr.enum.PixelOriginInterpretationValues.VOLUME``) or
            relative to an individual frame
            (``highdicom.sr.enum.PixelOriginInterpretationValues.FRAME``)
        fiducial_uid: Union[pydicom.uid.UID, str, None], optional
            unique identifier for the content item
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(ScoordContentItem, self).__init__(
            ValueTypeValues.SCOORD, name, relationship_type
        )
        graphic_type = GraphicTypeValues(graphic_type)
        pixel_origin_interpretation = PixelOriginInterpretationValues(
            pixel_origin_interpretation
        )
        self.GraphicType = graphic_type.value

        if graphic_type == GraphicTypeValues.POINT:
            if graphic_data.shape[0] != 1 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a scoord of graphic type "POINT" '
                    'must be a single (column row) pair in two-dimensional '
                    'image coordinate space.'
                )
        elif graphic_type == GraphicTypeValues.CIRCLE:
            if graphic_data.shape[0] != 2 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a scoord of graphic type "CIRCLE" '
                    'must be two (column, row) pairs in two-dimensional '
                    'image coordinate space.'
                )
        elif graphic_type == GraphicTypeValues.ELLIPSE:
            if graphic_data.shape[0] != 4 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a scoord of graphic type "ELLIPSE" '
                    'must be four (column, row) pairs in two-dimensional '
                    'image coordinate space.'
                )
        elif graphic_type == GraphicTypeValues.ELLIPSOID:
            if graphic_data.shape[0] != 6 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a scoord of graphic type "ELLIPSOID" '
                    'must be six (column, row) pairs in two-dimensional '
                    'image coordinate space.'
                )
        else:
            if not graphic_data.shape[0] > 1 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a scoord must be multiple '
                    '(column, row) pairs in two-dimensional image '
                    'coordinate space.'
                )
        # Flatten list of coordinate pairs
        self.GraphicData = graphic_data.flatten().tolist()
        self.PixelOriginInterpretation = pixel_origin_interpretation.value
        if fiducial_uid is not None:
            self.FiducialUID = fiducial_uid


class Scoord3DContentItem(ContentItem):

    """DICOM SR document content item for value type SCOORD3D.

    Note
    ----
    Spatial coordinates are defined in the patient or specimen-based coordinate
    system and have milimeter unit.

    """

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            graphic_type: Union[GraphicTypeValues3D, str],
            graphic_data: np.ndarray,
            frame_of_reference_uid: Union[str, UID],
            fiducial_uid: Optional[Union[str, UID]] = None,
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        graphic_type: Union[highdicom.sr.enum.GraphicTypeValues3D, str]
            name of the graphic type
        graphic_data: numpy.ndarray[numpy.float]
            array of spatial coordinates, where each row of the array
            represents a (x, y, z) coordinate triplet
        frame_of_reference_uid: Union[pydicom.uid.UID, str]
            unique identifier of the frame of reference within which the
            coordinates are defined
        fiducial_uid: str, optional
            unique identifier for the content item
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(Scoord3DContentItem, self).__init__(
            ValueTypeValues.SCOORD3D, name, relationship_type
        )
        graphic_type = GraphicTypeValues3D(graphic_type)
        self.GraphicType = graphic_type.value

        if graphic_type == GraphicTypeValues3D.POINT:
            if graphic_data.shape[0] != 1 or not graphic_data.shape[1] == 3:
                raise ValueError(
                    'Graphic data of a scoord 3D of graphic type "POINT" '
                    'must be a single point in three-dimensional patient or '
                    'slide coordinate space in form of a (x, y, z) triplet.'
                )
        elif graphic_type == GraphicTypeValues3D.ELLIPSE:
            if graphic_data.shape[0] != 4 or not graphic_data.shape[1] == 3:
                raise ValueError(
                    'Graphic data of a 3D scoord of graphic type "ELLIPSE" '
                    'must be four (x, y, z) triplets in three-dimensional '
                    'patient or slide coordinate space.'
                )
        elif graphic_type == GraphicTypeValues3D.ELLIPSOID:
            if graphic_data.shape[0] != 6 or not graphic_data.shape[1] == 3:
                raise ValueError(
                    'Graphic data of a 3D scoord of graphic type '
                    '"ELLIPSOID" must be six (x, y, z) triplets in '
                    'three-dimensional patient or slide coordinate space.'
                )
        else:
            if not graphic_data.shape[0] > 1 or not graphic_data.shape[1] == 3:
                raise ValueError(
                    'Graphic data of a 3D scoord must be multiple '
                    '(x, y, z) triplets in three-dimensional patient or '
                    'slide coordinate space.'
                )
        # Flatten list of coordinate triplets
        self.GraphicData = graphic_data.flatten().tolist()
        self.ReferencedFrameOfReferenceUID = frame_of_reference_uid
        if fiducial_uid is not None:
            self.FiducialUID = fiducial_uid


class TcoordContentItem(ContentItem):

    """DICOM SR document content item for value type TCOORD."""

    def __init__(
            self,
            name: Union[Code, CodedConcept],
            temporal_range_type: Union[str, TemporalRangeTypeValues],
            referenced_sample_positions: Optional[Sequence[int]] = None,
            referenced_time_offsets: Optional[Sequence[float]] = None,
            referenced_date_time: Optional[Sequence[datetime.datetime]] = None,
            relationship_type: Optional[
                Union[str, RelationshipTypeValues]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            concept name
        temporal_range_type: Union[highdicom.sr.enum.TemporalRangeTypeValues, str]
            name of the temporal range type
        referenced_sample_positions: Sequence[int], optional
            one-based relative sample position of acquired time points
            within the time series
        referenced_time_offsets: Sequence[float], optional
            seconds after start of the acquisition of the time series
        referenced_date_time: Sequence[datetime.datetime], optional
            absolute time points
        relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
            type of relationship with parent content item

        """  # noqa
        super(TcoordContentItem, self).__init__(
            ValueTypeValues.TCOORD, name, relationship_type
        )
        temporal_range_type = TemporalRangeTypeValues(temporal_range_type)
        self.TemporalRangeType = temporal_range_type.value
        if referenced_sample_positions is not None:
            self.ReferencedSamplePositions = [
                int(v) for v in referenced_sample_positions
            ]
        elif referenced_time_offsets is not None:
            self.ReferencedTimeOffsets = [
                float(v) for v in referenced_time_offsets
            ]
        elif referenced_date_time is not None:
            self.ReferencedDateTime = [
                DT(v) for v in referenced_date_time
            ]
        else:
            raise ValueError(
                'One of the following arguments is required: "{}"'.format(
                    '", "'.join([
                        'referenced_sample_positions',
                        'referenced_time_offsets',
                        'referenced_date_time'
                    ])
                )
            )
