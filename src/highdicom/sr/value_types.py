"""DICOM structured reporting content item value types."""
import datetime
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union
from typing import Any, List, Optional, Sequence, Union
import warnings

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
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
from highdicom.uid import UID
from highdicom.valuerep import check_person_name


def _assert_value_type(
    dataset: Dataset,
    value_type: ValueTypeValues
) -> None:
    """Check whether dataset contains required attributes for a value type.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset representing an SR Content Item
    value_type: highdicom.sr.enum.ValueTypeValues
        Expected value of Value Type attribute

    Raises
    ------
    AttributeError
        When a required attribute is missing
    ValueError
        When the expected and encountered value of Value Type attribute don't
        match

    """
    if not hasattr(dataset, 'ValueType'):
        raise AttributeError('Dataset is not an SR Content Item:\n{dataset}.')
    if not dataset.ValueType == value_type.value:
        raise ValueError(
            'Dataset is not an SR Content Item with value type '
            f'"{value_type.value}":\n{dataset}'
        )
    required_attrs = {
        ValueTypeValues.CODE: ['ConceptCodeSequence'],
        ValueTypeValues.COMPOSITE: ['ReferencedSOPSequence'],
        ValueTypeValues.CONTAINER: ['ContinuityOfContent'],
        ValueTypeValues.DATE: ['Date'],
        ValueTypeValues.DATETIME: ['DateTime'],
        ValueTypeValues.IMAGE: ['ReferencedSOPSequence'],
        ValueTypeValues.NUM: ['MeasuredValueSequence'],
        ValueTypeValues.PNAME: ['PersonName'],
        ValueTypeValues.SCOORD: ['GraphicType', 'GraphicData'],
        ValueTypeValues.SCOORD3D: ['GraphicType', 'GraphicData'],
        ValueTypeValues.TCOORD: ['TemporalRangeType'],
        ValueTypeValues.TIME: ['Time'],
        ValueTypeValues.TEXT: ['TextValue'],
        ValueTypeValues.UIDREF: ['UID'],
    }
    for attr in required_attrs[value_type]:
        if not hasattr(dataset, attr):
            raise AttributeError(
                'Dataset is not an SR Content Item with value type '
                f'"{value_type.value}" because it lacks required '
                f'attribute "{attr}":\n{dataset}'
            )


def _get_content_item_class(value_type: ValueTypeValues) -> type:
    python_types = {
        ValueTypeValues.CODE: CodeContentItem,
        ValueTypeValues.COMPOSITE: CompositeContentItem,
        ValueTypeValues.CONTAINER: ContainerContentItem,
        ValueTypeValues.DATE: DateContentItem,
        ValueTypeValues.DATETIME: DateTimeContentItem,
        ValueTypeValues.IMAGE: ImageContentItem,
        ValueTypeValues.NUM: NumContentItem,
        ValueTypeValues.PNAME: PnameContentItem,
        ValueTypeValues.SCOORD: ScoordContentItem,
        ValueTypeValues.SCOORD3D: Scoord3DContentItem,
        ValueTypeValues.TCOORD: TcoordContentItem,
        ValueTypeValues.TIME: TimeContentItem,
        ValueTypeValues.TEXT: TextContentItem,
        ValueTypeValues.UIDREF: UIDRefContentItem,
    }
    return python_types[value_type]


class ContentItem(Dataset):

    """Abstract base class for a collection of attributes contained in the
    DICOM SR Document Content Module."""

    def __init__(
        self,
        value_type: Union[str, ValueTypeValues],
        name: Union[Code, CodedConcept],
        relationship_type: Union[str, RelationshipTypeValues, None]
    ) -> None:
        """
        Parameters
        ----------
        value_type: Union[str, highdicom.sr.ValueTypeValues]
            type of value encoded in a content item
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            coded name or an enumerated item representing a coded name
        relationship_type: Union[str, highdicom.sr.RelationshipTypeValues], optional
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

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ContentItem':
        """Construct instance of appropriate subtype from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item

        Returns
        -------
        highdicom.sr.value_types.ContentItem
            Content Item

        """
        value_type = ValueTypeValues(dataset.ValueType)
        content_item_cls = _get_content_item_class(value_type)
        return content_item_cls.from_dataset(dataset)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'ContentItem':
        required_attrs = [
            'ValueType',
            'ConceptNameCodeSequence',
        ]
        for attr in required_attrs:
            if not hasattr(dataset, attr):
                raise AttributeError(
                    'Dataset is not an SR Content Item because it lacks '
                    f'required attribute "{attr}".'
                )
        item = deepcopy(dataset)
        item.__class__ = cls
        if hasattr(dataset, 'ContentSequence'):
            item.ContentSequence = ContentSequence.from_sequence(
                dataset.ContentSequence
            )
        return item

    @property
    def name(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded name of the content item"""
        ds = self.ConceptNameCodeSequence[0]
        return CodedConcept(
            value=ds.CodeValue,
            scheme_designator=ds.CodingSchemeDesignator,
            meaning=ds.CodeMeaning
        )

    @property
    def value_type(self) -> ValueTypeValues:
        """ValueTypeValues: type of the content item
        (see `highdicom.sr.ValueTypeValues`)

        """
        return ValueTypeValues(self.ValueType)

    @property
    def relationship_type(self) -> Optional[RelationshipTypeValues]:
        """RelationshipTypeValues: type of relationship the content item has
        with its parent (see `highdicom.sr.RelationshipTypeValues`)

        """
        if hasattr(self, 'RelationshipType'):
            return RelationshipTypeValues(self.RelationshipType)
        else:
            return None


class ContentSequence(DataElementSequence):

    """Sequence of DICOM SR Content Items."""

    def __init__(
        self,
        items: Optional[Sequence] = None,
        is_root: bool = False
    ) -> None:
        self._is_root = is_root
        if items is not None:
            for i in items:
                if not isinstance(i, ContentItem):
                    raise TypeError(
                        'Items of "{}" must have type ContentItem.'.format(
                            self.__class__.__name__
                        )
                    )
                if is_root:
                    if i.relationship_type is not None:
                        raise AttributeError(
                            'Items at the root of the content tree must '
                            'have no relationship type.'
                        )
                    if not isinstance(i, ContainerContentItem):
                        raise TypeError(
                            'Items at the root of a SR content tree must be '
                            'container content sequences.'
                        )
                else:
                    if i.relationship_type is None:
                        raise AttributeError(
                            'Items to be included in a '
                            f'{self.__class__.__name__} must have an '
                            'established relationship type.'
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
        highdicom.sr.ContentSequence[highdicom.sr.ContentItem]
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
        item: highdicom.sr.ContentItem
            content item

        """
        if not isinstance(item, ContentItem):
            raise TypeError(
                'Items of "{}" must have type ContentItem.'.format(
                    self.__class__.__name__
                )
            )
        if self._is_root:
            if item.relationship_type is not None:
                raise AttributeError(
                    f'Items to be appended to a {self.__class__.__name__} '
                    'that is the root of the SR content tree must not have '
                    'relationship type.'
                )
        else:
            if item.relationship_type is None:
                raise AttributeError(
                    f'Items to be appended to a {self.__class__.__name__} must '
                    'have an established relationship type.'
                )
        super(ContentSequence, self).append(item)

    def extend(self, items: Sequence[ContentItem]) -> None:
        """Extends multiple content items to the sequence.

        Parameters
        ----------
        items: Sequence[highdicom.sr.ContentItem]
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
        item: highdicom.sr.ContentItem
            content item

        """
        if not isinstance(item, ContentItem):
            raise TypeError(
                'Items of "{}" must have type ContentItem.'.format(
                    self.__class__.__name__
                )
            )
        if self._is_root:
            if item.relationship_type is not None:
                raise AttributeError(
                    f'Items to be included in a {self.__class__.__name__} '
                    'that is the root of the SR content tree must not have '
                    'relationship type.'
                )
        else:
            if item.relationship_type is None:
                raise AttributeError(
                    f'Items to be inserted into to a {self.__class__.__name__} '
                    'must have an established relationship type.'
                )
        super(ContentSequence, self).insert(position, item)

    @classmethod
    def from_sequence(cls, sequence: Sequence[Dataset]) -> 'ContentSequence':
        """Construct instance from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items

        Returns
        -------
        highdicom.sr.value_types.ContentSequence
            Content Sequence containing SR Content Items

        """
        content_items = []
        for i, dataset in enumerate(sequence, 1):
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    f'Item #{i} of sequence is not an SR Content Item:\n'
                    f'{dataset}'
                )
            try:
                value_type = ValueTypeValues(dataset.ValueType)
            except TypeError:
                raise ValueError(
                    f'Item #{i} of sequence is not an SR Content Item '
                    f'because it has unknown Value Type "{dataset.ValueType}":'
                    f'\n{dataset}'
                )
            except AttributeError:
                raise AttributeError(
                    f'Item #{i} of sequence is not an SR Content Item:\n'
                    f'{dataset}'
                )
            if not hasattr(dataset, 'RelationshipType'):
                raise AttributeError(
                    'Dataset is not an SR Content Item because it lacks '
                    f'required attribute "RelationshipType":\n{dataset}'
                )
            content_item_cls = _get_content_item_class(value_type)
            content_items.append(content_item_cls.from_dataset(dataset))
        return cls(content_items)

    @property
    def is_root(self) -> bool:
        """bool: whether the sequence is intended for use at the root of the
        SR content tree.

        """
        return self._is_root


class CodeContentItem(ContentItem):

    """DICOM SR document content item for value type CODE."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[Code, CodedConcept],
        relationship_type: Union[str, RelationshipTypeValues, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            coded value or an enumerated item representing a coded value
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
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

    @property
    def value(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded concept"""
        ds = self.ConceptCodeSequence[0]
        return CodedConcept(
            value=ds.CodeValue,
            scheme_designator=ds.CodingSchemeDesignator,
            meaning=ds.CodeMeaning
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'CodeContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CODE

        Returns
        -------
        highdicom.sr.value_types.CodeContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.CODE)
        return super(CodeContentItem, cls)._from_dataset(dataset)


class PnameContentItem(ContentItem):

    """DICOM SR document content item for value type PNAME."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[str, PersonName],
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, pydicom.valuerep.PersonName]
            name of the person
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(PnameContentItem, self).__init__(
            ValueTypeValues.PNAME, name, relationship_type
        )
        check_person_name(value)
        self.PersonName = PersonName(value)

    @property
    def value(self) -> PersonName:
        """pydicom.valuerep.PersonName: person name"""
        return PersonName(self.PersonName)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'PnameContentItem':
        """Construct instance from existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type PNAME

        Returns
        -------
        highdicom.sr.value_types.PnameContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.PNAME)
        return super(PnameContentItem, cls)._from_dataset(dataset)


class TextContentItem(ContentItem):

    """DICOM SR document content item for value type TEXT."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: str,
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: str
            description of the concept in free text
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """ # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(TextContentItem, self).__init__(
            ValueTypeValues.TEXT, name, relationship_type
        )
        self.TextValue = str(value)

    @property
    def value(self) -> str:
        """str: text value"""
        return self.TextValue

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'TextContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TEXT

        Returns
        -------
        highdicom.sr.value_types.TextContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.TEXT)
        return super(TextContentItem, cls)._from_dataset(dataset)


class TimeContentItem(ContentItem):

    """DICOM SR document content item for value type TIME."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[str, datetime.time, TM],
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, datetime.time, pydicom.valuerep.TM]
            time
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(TimeContentItem, self).__init__(
            ValueTypeValues.TIME, name, relationship_type
        )
        self.Time = TM(value)

    @property
    def value(self) -> datetime.time:
        """datetime.time: time"""
        allowed_formats = [
            '%H:%M:%S.%f',
            '%H:%M:%S',
            '%H:%M',
            '%H',
        ]
        for fmt in allowed_formats:
            try:
                dt = datetime.datetime.strptime(self.Time.isoformat(), fmt)
                return dt.time()
            except ValueError:
                continue
        raise ValueError(f'Could not decode time value "{self.Time}"')

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'TimeContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TIME

        Returns
        -------
        highdicom.sr.value_types.TimeContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.TIME)
        return super(TimeContentItem, cls)._from_dataset(dataset)


class DateContentItem(ContentItem):

    """DICOM SR document content item for value type DATE."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[str, datetime.date, DA],
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, datetime.date, pydicom.valuerep.DA]
            date
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(DateContentItem, self).__init__(
            ValueTypeValues.DATE, name, relationship_type
        )
        self.Date = DA(value)

    @property
    def value(self) -> datetime.date:
        """datetime.date: date"""
        fmt = '%Y-%m-%d'
        return datetime.datetime.strptime(self.Date.isoformat(), fmt).date()

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'DateContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATE

        Returns
        -------
        highdicom.sr.value_types.DateContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.DATE)
        return super(DateContentItem, cls)._from_dataset(dataset)


class DateTimeContentItem(ContentItem):

    """DICOM SR document content item for value type DATETIME."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[str, datetime.datetime, DT],
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[str, datetime.datetime, pydicom.valuerep.DT]
            datetime
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(DateTimeContentItem, self).__init__(
            ValueTypeValues.DATETIME, name, relationship_type
        )
        self.DateTime = DT(value)

    @property
    def value(self) -> datetime.datetime:
        """datetime.datetime: datetime"""
        allowed_formats = [
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M',
            '%Y-%m-%dT%H:%M%z',
            '%Y-%m-%dT%H',
            '%Y-%m-%dT%H%z',
            '%Y-%m-%d',
            '%Y-%m',
            '%Y',
        ]
        for fmt in allowed_formats:
            try:
                dt = datetime.datetime.strptime(self.DateTime.isoformat(), fmt)
                return dt
            except ValueError:
                continue
        raise ValueError(f'Could not decode datetime value "{self.DateTime}"')

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'DateTimeContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATETIME

        Returns
        -------
        highdicom.sr.value_types.DateTimeContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.DATETIME)
        return super(DateTimeContentItem, cls)._from_dataset(dataset)


class UIDRefContentItem(ContentItem):

    """DICOM SR document content item for value type UIDREF."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[str, UID],
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[highdicom.UID, str]
            unique identifier
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(UIDRefContentItem, self).__init__(
            ValueTypeValues.UIDREF, name, relationship_type
        )
        self.UID = value

    @property
    def value(self) -> UID:
        """highdicom.UID: UID"""
        return UID(self.UID)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'UIDRefContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type UIDREF

        Returns
        -------
        highdicom.sr.value_types.UIDRefContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.UIDREF)
        return super(UIDRefContentItem, cls)._from_dataset(dataset)


class NumContentItem(ContentItem):

    """DICOM SR document content item for value type NUM."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        value: Union[int, float],
        unit: Union[Code, CodedConcept],
        qualifier: Optional[Union[Code, CodedConcept]] = None,
        relationship_type: Union[str, RelationshipTypeValues, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        value: Union[int, float]
            numeric value
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            coded units of measurement (see `CID 7181 <http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_7181.html>`_
            "Abstract Multi-dimensional Image Model Component Units")
        qualifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code], optional
            qualification of numeric value or as an alternative to
            numeric value, e.g., reason for absence of numeric value
            (see `CID 42 <http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_42.html>`_
            "Numeric Value Qualifier" for options)
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """ # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(NumContentItem, self).__init__(
            ValueTypeValues.NUM, name, relationship_type
        )
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
        if qualifier is not None:
            if not isinstance(qualifier, (CodedConcept, Code, )):
                raise TypeError(
                    'Argument "qualifier" must have type "CodedConcept" or '
                    '"Code".'
                )
            if isinstance(qualifier, Code):
                qualifier = CodedConcept(*qualifier)
            self.NumericValueQualifierCodeSequence = [qualifier]

    @property
    def value(self) -> Union[int, float]:
        """Union[int, float]: measured value"""
        item = self.MeasuredValueSequence[0]
        try:
            return float(item.FloatingPointValue)
        except AttributeError:
            return float(item.NumericValue)

    @property
    def unit(self) -> CodedConcept:
        """highdicom.sr.coding.CodedConcept: unit"""
        item = self.MeasuredValueSequence[0]
        return item.MeasurementUnitsCodeSequence[0]

    @property
    def qualifier(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.coding.CodedConcept, None]: qualifier"""
        try:
            return self.NumericValueQualifierCodeSequence[0]
        except AttributeError:
            return None

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'NumContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type NUM

        Returns
        -------
        highdicom.sr.NumContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.NUM)
        instance = super(NumContentItem, cls)._from_dataset(dataset)
        unit_item = (
            instance
            .MeasuredValueSequence[0]
            .MeasurementUnitsCodeSequence[0]
        )
        unit_item = CodedConcept.from_dataset(unit_item)
        if hasattr(instance, 'NumericValueQualifierCodeSequence'):
            qualifier_item = instance.NumericValueQualifierCodeSequence[0]
            qualifier_item = CodedConcept.from_dataset(qualifier_item)
        return instance


class ContainerContentItem(ContentItem):

    """DICOM SR document content item for value type CONTAINER."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        is_content_continuous: bool = True,
        template_id: Optional[str] = None,
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        is_content_continous: bool, optional
            whether contained content items are logically linked in a
            continuous manner or separate items (default: ``True``)
        template_id: str, optional
            SR template identifier
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            type of relationship with parent content item. 

        """  # noqa E501
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

    @property
    def template_id(self) -> Union[str, None]:
        """Union[str, None]: template identifier"""
        try:
            item = self.ContentTemplateSequence[0]
            return item.TemplateIdentifier
        except (AttributeError, IndexError):
            return None

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ContainerContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CONTAINER

        Returns
        -------
        highdicom.sr.ContainerContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.CONTAINER)
        return super(ContainerContentItem, cls)._from_dataset(dataset)


class CompositeContentItem(ContentItem):

    """DICOM SR document content item for value type COMPOSITE."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        referenced_sop_class_uid: Union[str, UID],
        referenced_sop_instance_uid: Union[str, UID],
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ):
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        referenced_sop_class_uid: Union[highdicom.UID, str]
            SOP Class UID of the referenced object
        referenced_sop_instance_uid: Union[highdicom.UID, str]
            SOP Instance UID of the referenced object
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(CompositeContentItem, self).__init__(
            ValueTypeValues.COMPOSITE, name, relationship_type
        )
        item = Dataset()
        item.ReferencedSOPClassUID = str(referenced_sop_class_uid)
        item.ReferencedSOPInstanceUID = str(referenced_sop_instance_uid)
        self.ReferencedSOPSequence = [item]

    @property
    def value(self) -> Tuple[UID, UID]:
        """Tuple[highdicom.UID, highdicom.UID]:
            referenced SOP Class UID and SOP Instance UID
        """
        item = self.ReferencedSOPSequence[0]
        return (
            UID(item.ReferencedSOPClassUID),
            UID(item.ReferencedSOPInstanceUID),
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'CompositeContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type COMPOSITE

        Returns
        -------
        highdicom.sr.CompositeContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.COMPOSITE)
        return super(CompositeContentItem, cls)._from_dataset(dataset)


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
        relationship_type: Union[str, RelationshipTypeValues, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        referenced_sop_class_uid: Union[highdicom.UID, str]
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: Union[highdicom.UID, str]
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[int, Sequence[int]], optional
            number of frame(s) to which the reference applies in case of a
            multi-frame image
        referenced_segment_numbers: Union[int, Sequence[int]], optional
            number of segment(s) to which the refernce applies in case of a
            segmentation image
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
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

    @property
    def value(self) -> Tuple[UID, UID]:
        """Tuple[highdicom.UID, highdicom.UID]:
            referenced SOP Class UID and SOP Instance UID
        """
        item = self.ReferencedSOPSequence[0]
        return (
            UID(item.ReferencedSOPClassUID),
            UID(item.ReferencedSOPInstanceUID),
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ImageContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type IMAGE

        Returns
        -------
        highdicom.sr.ImageContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.IMAGE)
        return super(ImageContentItem, cls)._from_dataset(dataset)


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
        ] = None,
        fiducial_uid: Optional[Union[str, UID]] = None,
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        graphic_type: Union[highdicom.sr.GraphicTypeValues, str]
            name of the graphic type
        graphic_data: numpy.ndarray[numpy.int]
            array of ordered spatial coordinates, where each row of the array
            represents a (column, row) coordinate pair
        pixel_origin_interpretation: Union[highdicom.sr.PixelOriginInterpretationValues, str, None], optional
            whether pixel coordinates specified by `graphic_data` are defined
            relative to the total pixel matrix
            (``highdicom.sr.PixelOriginInterpretationValues.VOLUME``) or
            relative to an individual frame
            (``highdicom.sr.PixelOriginInterpretationValues.FRAME``)
        fiducial_uid: Union[highdicom.UID, str, None], optional
            unique identifier for the content item
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
        super(ScoordContentItem, self).__init__(
            ValueTypeValues.SCOORD, name, relationship_type
        )
        graphic_type = GraphicTypeValues(graphic_type)
        self.GraphicType = graphic_type.value

        if graphic_type == GraphicTypeValues.POINT:
            if graphic_data.shape[0] != 1 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a scoord of graphic type "POINT" '
                    'must be a single (column, row) pair in two-dimensional '
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
        if pixel_origin_interpretation is not None:
            pixel_origin_interpretation = PixelOriginInterpretationValues(
                pixel_origin_interpretation
            )
            self.PixelOriginInterpretation = pixel_origin_interpretation.value
        if fiducial_uid is not None:
            self.FiducialUID = fiducial_uid

    @property
    def value(self) -> np.ndarray:
        """numpy.ndarray: n x 2 array of 2D spatial coordinates"""
        return np.array(self.GraphicData).reshape(-1, 2)

    @property
    def graphic_type(self) -> GraphicTypeValues:
        """GraphicTypeValues: graphic type"""
        return GraphicTypeValues(self.GraphicType)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ScoordContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.ScoordContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.SCOORD)
        return super(ScoordContentItem, cls)._from_dataset(dataset)


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
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        graphic_type: Union[highdicom.sr.GraphicTypeValues3D, str]
            name of the graphic type
        graphic_data: numpy.ndarray[numpy.float]
            array of spatial coordinates, where each row of the array
            represents a (x, y, z) coordinate triplet
        frame_of_reference_uid: Union[highdicom.UID, str]
            unique identifier of the frame of reference within which the
            coordinates are defined
        fiducial_uid: str, optional
            unique identifier for the content item
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
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

    @property
    def value(self) -> np.ndarray:
        """numpy.ndarray: n x 3 array of 3D spatial coordinates"""
        return np.array(self.GraphicData).reshape(-1, 3)

    @property
    def graphic_type(self) -> GraphicTypeValues3D:
        """GraphicTypeValues3D: graphic type"""
        return GraphicTypeValues3D(self.GraphicType)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'Scoord3DContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD3D

        Returns
        -------
        highdicom.sr.Scoord3DContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.SCOORD3D)
        return super(Scoord3DContentItem, cls)._from_dataset(dataset)


class TcoordContentItem(ContentItem):

    """DICOM SR document content item for value type TCOORD."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        temporal_range_type: Union[str, TemporalRangeTypeValues],
        referenced_sample_positions: Optional[Sequence[int]] = None,
        referenced_time_offsets: Optional[Sequence[float]] = None,
        referenced_date_time: Optional[Sequence[datetime.datetime]] = None,
        relationship_type: Union[str, RelationshipTypeValues, None] = None
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            concept name
        temporal_range_type: Union[highdicom.sr.TemporalRangeTypeValues, str]
            name of the temporal range type
        referenced_sample_positions: Sequence[int], optional
            one-based relative sample position of acquired time points
            within the time series
        referenced_time_offsets: Sequence[float], optional
            seconds after start of the acquisition of the time series
        referenced_date_time: Sequence[datetime.datetime], optional
            absolute time points
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str]
            type of relationship with parent content item

        """  # noqa
        if relationship_type is None:
            warnings.warn(
                'A future release will require that relationship types be '
                f'provided for items of type {self.__class__.__name__}.',
                DeprecationWarning
            )
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

    @property
    def value(self) -> Union[List[int], List[float], List[datetime.datetime]]:
        """Union[List[int], List[float], List[datetime.datetime]]: time points
        """
        try:
            return self.ReferencedSamplePositions
        except AttributeError:
            try:
                return self.ReferencedTimeOffsets
            except AttributeError:
                return self.ReferencedDateTime

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'TcoordContentItem':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TCOORD

        Returns
        -------
        highdicom.sr.TcoordContentItem
            Content Item

        """
        _assert_value_type(dataset, ValueTypeValues.TCOORD)
        return super(TcoordContentItem, cls)._from_dataset(dataset)
