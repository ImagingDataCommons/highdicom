"""DICOM structured reporting content item value types."""
import datetime
from collections import defaultdict
from copy import deepcopy
from typing import (
    cast,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    overload,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
from pydicom.valuerep import DA, DS, TM, DT, PersonName

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

        """  # noqa: E501
        super(ContentItem, self).__init__()
        value_type = ValueTypeValues(value_type)
        self.ValueType = value_type.value
        if not isinstance(name, (CodedConcept, Code, )):
            raise TypeError(
                'Argument "name" must have type CodedConcept or Code.'
            )
        if isinstance(name, Code):
            name = CodedConcept.from_code(name)
        self.ConceptNameCodeSequence = [name]
        if relationship_type is not None:
            relationship_type = RelationshipTypeValues(relationship_type)
            self.RelationshipType = relationship_type.value

    def __setattr__(
        self,
        name: str,
        value: Union[DataElement, DataElementSequence]
    ) -> None:
        if name == 'ContentSequence':
            super(ContentItem, self).__setattr__(name, ContentSequence(value))
        else:
            super(ContentItem, self).__setattr__(name, value)

    @classmethod
    def _from_dataset_derived(cls, dataset: Dataset) -> 'ContentItem':
        """Construct object of derived type from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item

        Returns
        -------
        highdicom.sr.ContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        value_type = ValueTypeValues(dataset.ValueType)
        content_item_cls = _get_content_item_class(value_type)
        return content_item_cls._from_dataset(dataset)  # type: ignore

    @classmethod
    def _from_dataset_base(cls, dataset: Dataset) -> 'ContentItem':
        if not hasattr(dataset, 'ValueType'):
            raise AttributeError(
                'Dataset is not an SR Content Item because it lacks '
                'required attribute "Value Type".'
            )

        value_types_with_optional_name = (
            'CompositeContentItem',
            'ImageContentItem',
            'ScoordContentItem',
            'Scoord3DContentItem',
            'TcoordContentItem',
            'WaveformContentItem',
        )
        if not hasattr(dataset, 'ConceptNameCodeSequence'):
            if cls.__name__ in value_types_with_optional_name:
                default_name = CodedConcept(
                    value='260753009',
                    scheme_designator='SCT',
                    meaning='Source',
                )
                dataset.ConceptNameCodeSequence = [default_name]
            else:
                raise AttributeError(
                    'Dataset is not a SR Content Item because it lacks '
                    'required attribute "Concept Name Content Sequence".'
                )

        item = dataset
        item.__class__ = cls
        if hasattr(item, 'ContentSequence'):
            item.ContentSequence = ContentSequence._from_sequence(
                item.ContentSequence
            )
        item.ConceptNameCodeSequence = [
            CodedConcept.from_dataset(item.ConceptNameCodeSequence[0])
        ]
        return cast(ContentItem, item)

    @property
    def name(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded name of the content item"""
        return self.ConceptNameCodeSequence[0]

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
        items: Optional[
            Union[Sequence[ContentItem], 'ContentSequence']
        ] = None,
        is_root: bool = False,
        is_sr: bool = True
    ) -> None:
        """

        Parameters
        ----------
        items: Union[Sequence[highdicom.sr.ContentItem], highdicom.sr.ContentSequence, None], optional
            SR Content items
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree
        is_sr: bool, optional
            Whether the sequence is use to contain SR Content Items that are
            intended to be added to an SR document as opposed to other types
            of IODs based on an acquisition, protocol or workflow context
            template

        """  # noqa: E501
        self._is_root = is_root
        self._is_sr = is_sr

        if is_root and not is_sr:
            raise ValueError(
                'Argument "is_root" is True but "is_sr" is False. '
                'SR content items that are intended to be included at the '
                'root of a document content tree must also be intended to be '
                'added to a SR document.'
            )

        # The implementation of this class is quite a hack. It is derived from
        # "pydicom.sequence.Sequence", because this is the only type that is
        # accepted for Data Elements with Value Representation "SQ"
        # (see "pydicom.dataset.Dataset.__setattr__").
        # Methods of the "pydicom.sequence.Sequence" class generally operate on
        # items of type "pydicom.dataset.Dataset" and accordingly accept
        # instances of that type as arguments
        # (see for example "pydicom.dataset.Dataset.__setitem__").
        # However, we want/need methods to operate on instances of type
        # "highdicom.sr.ContentItem", and making that possible requires a lot
        # of "type: ignore[override]" comments. The implementation is thereby
        # violating the Liskov substitution principle. That's not elegant
        # (to put it kindly), but we currently don't see a better way without
        # having to change the implementation in the pydicom library.

        self._lut: Dict[
            Union[Code, CodedConcept],
            List[ContentItem]
        ] = defaultdict(list)
        if items is not None:
            super().__init__(items)
            for i in items:
                self._lut[i.name].append(i)
        else:
            super().__init__()

        if items is not None:
            for i in items:
                if not isinstance(i, ContentItem):
                    raise TypeError(
                        f'Items of "{self.__class__.__name__}" must have '
                        'type ContentItem.'
                    )
                if is_root:
                    if i.relationship_type is not None:
                        raise AttributeError(
                            'Item at the root of the content tree must '
                            f'have no Relationship Type:\n{i.name}.'
                        )
                    if not isinstance(i, ContainerContentItem):
                        raise TypeError(
                            'Item at the root of a SR content tree must '
                            f'have type ContainerContentItem:\n{i.name}'
                        )
                elif is_sr:
                    if i.relationship_type is None:
                        raise AttributeError(
                            'Item to be included in a '
                            f'{self.__class__.__name__} must have an '
                            f'established relationship type:\n{i.name}'
                        )
                else:
                    if i.relationship_type is not None:
                        raise AttributeError(
                            'Item of content of acquisition, protocol, or '
                            'workflow context must have no Relationship Type: '
                            f'\n{i.name}.'
                        )

    @overload
    def __setitem__(self, idx: int, val: ContentItem) -> None:
        pass

    @overload
    def __setitem__(self, idx: slice, val: Iterable[ContentItem]) -> None:
        pass

    def __setitem__(
        self,
        idx: Union[slice, int],
        val: Union[Iterable[ContentItem], ContentItem]
    ) -> None:   # type: ignore[override]
        if isinstance(val, Iterable):
            items = val
        else:
            items = [val]
        for i in items:
            if not isinstance(i, ContentItem):
                raise TypeError(
                    'Items of "{}" must have type ContentItem.'.format(
                        self.__class__.__name__
                    )
                )
        super().__setitem__(idx, val)  # type: ignore

    def __delitem__(
        self,
        idx: Union[slice, int]
    ) -> None:   # type: ignore[override]
        if isinstance(idx, slice):
            items = self[idx]
        else:
            items = [self[idx]]
        for i in items:
            i = cast(ContentItem, i)
            index = self._lut[i.name].index(i)
            del self._lut[i.name][index]
        super().__delitem__(idx)

    def __iter__(self) -> Iterator[ContentItem]:  # type: ignore[override]
        return super().__iter__()  # type: ignore

    def __contains__(self, val: ContentItem) -> bool:  # type: ignore[override]
        try:
            self.index(val)
        except ValueError:
            return False
        return True

    def index(self, val: ContentItem) -> int:  # type: ignore[override]
        """Get the index of a given item.

        Parameters
        ----------
        val: highdicom.sr.ContentItem
            SR Content Item

        Returns
        -------
        int: Index of the item in the sequence

        """
        if not isinstance(val, ContentItem):
            raise TypeError(
                'Items of "{}" must have type ContentItem.'.format(
                    self.__class__.__name__
                )
            )
        error_message = f'Item "{val.name}" is not in Sequence.'
        try:
            matches = self._lut[val.name]
        except KeyError:
            raise ValueError(error_message)
        try:
            index = matches.index(val)
        except ValueError:
            raise ValueError(error_message)
        return index

    def find(self, name: Union[Code, CodedConcept]) -> 'ContentSequence':
        """Find contained content items given their name.

        Parameters
        ----------
        name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Name of SR Content Items

        Returns
        -------
        highdicom.sr.ContentSequence
            Matched content items

        """
        return ContentSequence(
            self._lut[name],
            is_root=False,
            is_sr=self._is_sr
        )

    def get_nodes(self) -> 'ContentSequence':
        """Get content items that represent nodes in the content tree.

        A node is hereby defined as a content item that has a `ContentSequence`
        attribute.

        Returns
        -------
        highdicom.sr.ContentSequence[highdicom.sr.ContentItem]
            Matched content items

        """
        return self.__class__([
            item for item in self
            if hasattr(item, 'ContentSequence')
        ])

    def append(self, val: ContentItem) -> None:  # type: ignore[override]
        """Append a content item to the sequence.

        Parameters
        ----------
        item: highdicom.sr.ContentItem
            SR Content Item

        """
        if not isinstance(val, ContentItem):
            raise TypeError(
                'Items of "{}" must have type ContentItem.'.format(
                    self.__class__.__name__
                )
            )
        if self._is_root:
            if self._is_sr and val.relationship_type is not None:
                raise AttributeError(
                    f'Items to be appended to a {self.__class__.__name__} '
                    'that is the root of the SR content tree must not have '
                    'relationship type.'
                )
        else:
            if self._is_sr and val.relationship_type is None:
                raise AttributeError(
                    f'Items to be appended to a {self.__class__.__name__} must '
                    'have an established relationship type.'
                )
        self._lut[val.name].append(val)
        super().append(val)

    def extend(  # type: ignore[override]
        self,
        val: Union[Iterable[ContentItem], 'ContentSequence']
    ) -> None:
        """Extend multiple content items to the sequence.

        Parameters
        ----------
        val: Iterable[highdicom.sr.ContentItem, highdicom.sr.ContentSequence]
            SR Content Items

        """
        for item in val:
            self._lut[item.name].append(item)
            self.append(item)

    def insert(   # type: ignore[override]
        self,
        position: int,
        val: ContentItem
    ) -> None:
        """Insert a content item into the sequence at a given position.

        Parameters
        ----------
        position: int
            Index position
        val: highdicom.sr.ContentItem
            SR Content Item

        """
        if not isinstance(val, ContentItem):
            raise TypeError(
                'Items of "{}" must have type ContentItem.'.format(
                    self.__class__.__name__
                )
            )
        if self._is_root:
            if val.relationship_type is not None:
                raise AttributeError(
                    f'Items to be included in a {self.__class__.__name__} '
                    'that is the root of the SR content tree must not have '
                    'relationship type.'
                )
        else:
            if val.relationship_type is None:
                raise AttributeError(
                    f'Items to be inserted into to a {self.__class__.__name__} '
                    'must have an established relationship type.'
                )
        self._lut[val.name].append(val)
        super().insert(position, val)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False,
        is_sr: bool = True
    ) -> 'ContentSequence':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree
        is_sr: bool, optional
            Whether the sequence is use to contain SR Content Items that are
            intended to be added to an SR document as opposed to other types
            of IODs based on an acquisition, protocol or workflow context
            template

        Returns
        -------
        highdicom.sr.ContentSequence
            Content Sequence containing SR Content Items

        """
        content_items = []
        for i, dataset in enumerate(sequence, 1):
            cls._check_dataset(
                dataset,
                is_root=is_root,
                is_sr=is_sr,
                index=i
            )
            dataset_copy = deepcopy(dataset)
            item = ContentItem._from_dataset_derived(dataset_copy)
            content_items.append(item)
        return ContentSequence(content_items, is_root=is_root, is_sr=is_sr)

    @classmethod
    def _from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False,
        is_sr: bool = True
    ) -> 'ContentSequence':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree
        is_sr: bool, optional
            Whether the sequence is use to contain SR Content Items that are
            intended to be added to an SR document as opposed to other types
            of IODs based on an acquisition, protocol or workflow context
            template

        Returns
        -------
        highdicom.sr.ContentSequence
            Content Sequence containing SR Content Items

        """
        content_items = []
        for i, dataset in enumerate(sequence, 1):
            cls._check_dataset(
                dataset,
                is_root=is_root,
                is_sr=is_sr,
                index=i
            )
            item = ContentItem._from_dataset_derived(dataset)
            content_items.append(item)
        return ContentSequence(content_items, is_root=is_root, is_sr=is_sr)

    @classmethod
    def _check_dataset(
        cls,
        dataset: Dataset,
        is_root: bool,
        is_sr: bool,
        index: int
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f'Item #{index} of sequence is not an SR Content Item:\n'
                f'{dataset}'
            )
        try:
            ValueTypeValues(dataset.ValueType)
        except TypeError:
            raise ValueError(
                f'Item #{index} of sequence is not an SR Content Item '
                f'because it has unknown Value Type "{dataset.ValueType}":'
                f'\n{dataset}'
            )
        except AttributeError:
            raise AttributeError(
                f'Item #{index} of sequence is not an SR Content Item:\n'
                f'{dataset}'
            )
        if not hasattr(dataset, 'RelationshipType') and not is_root and is_sr:
            raise AttributeError(
                f'Item #{index} of sequence is not a value SR Content Item '
                'because it is not a root item and lacks the otherwise '
                f'required attribute "RelationshipType":\n{dataset}'
            )

    @property
    def is_root(self) -> bool:
        """bool: whether the sequence is intended for use at the root of the
        SR content tree.

        """
        return self._is_root

    @property
    def is_sr(self) -> bool:
        """bool: whether the sequence is intended for use in an SR document"""
        return self._is_sr


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
            Concept name
        value: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Coded value or an enumerated item representing a coded value
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
        super(CodeContentItem, self).__init__(
            ValueTypeValues.CODE, name, relationship_type
        )
        if not isinstance(value, (CodedConcept, Code, )):
            raise TypeError(
                'Argument "value" must have type CodedConcept or Code.'
            )
        if isinstance(value, Code):
            value = CodedConcept.from_code(value)
        self.ConceptCodeSequence = [value]

    @property
    def value(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: coded concept"""
        return self.ConceptCodeSequence[0]

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'CodeContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CODE

        Returns
        -------
        highdicom.sr.CodeContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'CodeContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CODE

        Returns
        -------
        highdicom.sr.CodeContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.CODE)
        item = super(CodeContentItem, cls)._from_dataset_base(dataset)
        item.ConceptCodeSequence = DataElementSequence([
            CodedConcept.from_dataset(item.ConceptCodeSequence[0])
        ])
        return cast(CodeContentItem, item)


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
            Concept name
        value: Union[str, pydicom.valuerep.PersonName]
            Name of the person
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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
        """Construct object from existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type PNAME

        Returns
        -------
        highdicom.sr.PnameContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'PnameContentItem':
        """Construct object from existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type PNAME

        Returns
        -------
        highdicom.sr.PnameContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.PNAME)
        item = super(PnameContentItem, cls)._from_dataset_base(dataset)
        return cast(PnameContentItem, item)


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
            Concept name
        value: str
            Description of the concept in free text
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TEXT

        Returns
        -------
        highdicom.sr.TextContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'TextContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TEXT

        Returns
        -------
        highdicom.sr.TextContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.TEXT)
        item = super(TextContentItem, cls)._from_dataset_base(dataset)
        return cast(TextContentItem, item)


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
            Concept name
        value: Union[str, datetime.time, pydicom.valuerep.TM]
            Time
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TIME

        Returns
        -------
        highdicom.sr.TimeContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'TimeContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TIME

        Returns
        -------
        highdicom.sr.TimeContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.TIME)
        item = super(TimeContentItem, cls)._from_dataset_base(dataset)
        return cast(TimeContentItem, item)


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
            Concept name
        value: Union[str, datetime.date, pydicom.valuerep.DA]
            Date
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATE

        Returns
        -------
        highdicom.sr.DateContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'DateContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATE

        Returns
        -------
        highdicom.sr.DateContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.DATE)
        item = super(DateContentItem, cls)._from_dataset_base(dataset)
        return cast(DateContentItem, item)


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
            Concept name
        value: Union[str, datetime.datetime, pydicom.valuerep.DT]
            Datetime
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATETIME

        Returns
        -------
        highdicom.sr.DateTimeContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'DateTimeContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATETIME

        Returns
        -------
        highdicom.sr.DateTimeContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.DATETIME)
        item = super(DateTimeContentItem, cls)._from_dataset_base(dataset)
        return cast(DateTimeContentItem, item)


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
            Concept name
        value: Union[highdicom.UID, str]
            Unique identifier
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type UIDREF

        Returns
        -------
        highdicom.sr.UIDRefContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'UIDRefContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type UIDREF

        Returns
        -------
        highdicom.sr.UIDRefContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.UIDREF)
        item = super(UIDRefContentItem, cls)._from_dataset_base(dataset)
        return cast(UIDRefContentItem, item)


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
            Concept name
        value: Union[int, float]
            Numeric value
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code], optional
            Coded units of measurement (see :dcm:`CID 7181 <part16/sect_CID_7181.html>`
            "Abstract Multi-dimensional Image Model Component Units")
        qualifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Qualification of numeric value or as an alternative to
            numeric value, e.g., reason for absence of numeric value
            (see :dcm:`CID 42 <part16/sect_CID_42.html>`
            "Numeric Value Qualifier" for options)
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
        super(NumContentItem, self).__init__(
            ValueTypeValues.NUM, name, relationship_type
        )
        self.MeasuredValueSequence: List[Dataset] = []
        measured_value_sequence_item = Dataset()
        if not isinstance(value, (int, float, )):
            raise TypeError(
                'Argument "value" must have type "int" or "float".'
            )
        measured_value_sequence_item.NumericValue = DS(
            value,
            auto_format=True
        )
        if isinstance(value, float):
            measured_value_sequence_item.FloatingPointValue = value
        if not isinstance(unit, (CodedConcept, Code, )):
            raise TypeError(
                'Argument "unit" must have type CodedConcept or Code.'
            )
        if isinstance(unit, Code):
            unit = CodedConcept.from_code(unit)
        measured_value_sequence_item.MeasurementUnitsCodeSequence = [unit]
        self.MeasuredValueSequence.append(measured_value_sequence_item)
        if qualifier is not None:
            if not isinstance(qualifier, (CodedConcept, Code, )):
                raise TypeError(
                    'Argument "qualifier" must have type "CodedConcept" or '
                    '"Code".'
                )
            if isinstance(qualifier, Code):
                qualifier = CodedConcept.from_code(qualifier)
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
        """highdicom.sr.CodedConcept: unit"""
        item = self.MeasuredValueSequence[0]
        return item.MeasurementUnitsCodeSequence[0]

    @property
    def qualifier(self) -> Union[CodedConcept, None]:
        """Union[highdicom.sr.CodedConcept, None]: qualifier"""
        try:
            return self.NumericValueQualifierCodeSequence[0]
        except AttributeError:
            return None

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'NumContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type NUM

        Returns
        -------
        highdicom.sr.NumContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'NumContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type NUM

        Returns
        -------
        highdicom.sr.NumContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.NUM)
        item = super(NumContentItem, cls)._from_dataset_base(dataset)
        unit_item = (
            item
            .MeasuredValueSequence[0]
            .MeasurementUnitsCodeSequence[0]
        )
        item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence = [
            CodedConcept.from_dataset(unit_item)
        ]
        if hasattr(item, 'NumericValueQualifierCodeSequence'):
            qualifier_item = item.NumericValueQualifierCodeSequence[0]
            item.NumericValueQualifierCodeSequence = DataElementSequence([
                CodedConcept.from_dataset(qualifier_item)
            ])
        return cast(NumContentItem, item)


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
        template_id: Union[str, None], optional
            SR template identifier
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            type of relationship with parent content item.

        """  # noqa: E501
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
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CONTAINER

        Returns
        -------
        highdicom.sr.ContainerContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'ContainerContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CONTAINER

        Returns
        -------
        highdicom.sr.ContainerContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.CONTAINER)
        item = super(ContainerContentItem, cls)._from_dataset_base(dataset)
        return cast(ContainerContentItem, item)


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
            Concept name
        referenced_sop_class_uid: Union[highdicom.UID, str]
            SOP Class UID of the referenced object
        referenced_sop_instance_uid: Union[highdicom.UID, str]
            SOP Instance UID of the referenced object
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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

    @property
    def referenced_sop_class_uid(self) -> UID:
        """highdicom.UID: referenced SOP Class UID"""
        return UID(self.ReferencedSOPSequence[0].ReferencedSOPClassUID)

    @property
    def referenced_sop_instance_uid(self) -> UID:
        """highdicom.UID: referenced SOP Instance UID"""
        return UID(self.ReferencedSOPSequence[0].ReferencedSOPInstanceUID)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'CompositeContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type COMPOSITE

        Returns
        -------
        highdicom.sr.CompositeContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'CompositeContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type COMPOSITE

        Returns
        -------
        highdicom.sr.CompositeContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.COMPOSITE)
        item = super(CompositeContentItem, cls)._from_dataset_base(dataset)
        return cast(CompositeContentItem, item)


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
            Concept name
        referenced_sop_class_uid: Union[highdicom.UID, str]
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: Union[highdicom.UID, str]
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[int, Sequence[int], None], optional
            Number of frame(s) to which the reference applies in case of a
            multi-frame image
        referenced_segment_numbers: Union[int, Sequence[int], None], optional
            Number of segment(s) to which the refernce applies in case of a
            segmentation image
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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

    @property
    def referenced_sop_class_uid(self) -> UID:
        """highdicom.UID: referenced SOP Class UID"""
        return UID(self.ReferencedSOPSequence[0].ReferencedSOPClassUID)

    @property
    def referenced_sop_instance_uid(self) -> UID:
        """highdicom.UID: referenced SOP Instance UID"""
        return UID(self.ReferencedSOPSequence[0].ReferencedSOPInstanceUID)

    @property
    def referenced_frame_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]: referenced frame numbers"""
        if not hasattr(
            self.ReferencedSOPSequence[0],
            'ReferencedFrameNumber',
        ):
            return None
        val = getattr(
            self.ReferencedSOPSequence[0],
            'ReferencedFrameNumber',
        )
        if isinstance(val, MultiValue):
            return [int(v) for v in val]
        else:
            return [int(val)]

    @property
    def referenced_segment_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]
            referenced segment numbers
        """
        if not hasattr(
            self.ReferencedSOPSequence[0],
            'ReferencedSegmentNumber',
        ):
            return None
        val = getattr(
            self.ReferencedSOPSequence[0],
            'ReferencedSegmentNumber',
        )
        if isinstance(val, MultiValue):
            return [int(v) for v in val]
        else:
            return [int(val)]

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ImageContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type IMAGE

        Returns
        -------
        highdicom.sr.ImageContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'ImageContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type IMAGE

        Returns
        -------
        highdicom.sr.ImageContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.IMAGE)
        item = super(ImageContentItem, cls)._from_dataset_base(dataset)
        return cast(ImageContentItem, item)


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
            Concept name
        graphic_type: Union[highdicom.sr.GraphicTypeValues, str]
            Name of the graphic type
        graphic_data: numpy.ndarray
            Array of ordered spatial coordinates, where each row of the array
            represents a (Column,Row) pair
        pixel_origin_interpretation: Union[highdicom.sr.PixelOriginInterpretationValues, str, None], optional
            Whether pixel coordinates specified by `graphic_data` are defined
            relative to the total pixel matrix
            (``highdicom.sr.PixelOriginInterpretationValues.VOLUME``) or
            relative to an individual frame
            (``highdicom.sr.PixelOriginInterpretationValues.FRAME``)
        fiducial_uid: Union[highdicom.UID, str, None], optional
            Unique identifier for the content item
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
        super(ScoordContentItem, self).__init__(
            ValueTypeValues.SCOORD, name, relationship_type
        )
        graphic_type = GraphicTypeValues(graphic_type)
        self.GraphicType = graphic_type.value

        if graphic_type == GraphicTypeValues.POINT:
            if graphic_data.shape[0] != 1 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a SCOORD of graphic type "POINT" '
                    'must be a single (column, row) pair in two-dimensional '
                    'image coordinate space.'
                )
        elif graphic_type == GraphicTypeValues.CIRCLE:
            if graphic_data.shape[0] != 2 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a SCOORD of graphic type "CIRCLE" '
                    'must be two (column, row) pairs in two-dimensional '
                    'image coordinate space.'
                )
        elif graphic_type == GraphicTypeValues.ELLIPSE:
            if graphic_data.shape[0] != 4 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a SCOORD of graphic type "ELLIPSE" '
                    'must be four (column, row) pairs in two-dimensional '
                    'image coordinate space.'
                )
        else:
            if not graphic_data.shape[0] > 1 or not graphic_data.shape[1] == 2:
                raise ValueError(
                    'Graphic data of a SCOORD must be multiple '
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
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.ScoordContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'ScoordContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.ScoordContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.SCOORD)
        item = super(ScoordContentItem, cls)._from_dataset_base(dataset)
        return cast(ScoordContentItem, item)


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
            Concept name
        graphic_type: Union[highdicom.sr.GraphicTypeValues3D, str]
            Name of the graphic type
        graphic_data: numpy.ndarray[numpy.float]
            Array of spatial coordinates, where each row of the array
            represents a (x, y, z) coordinate triplet
        frame_of_reference_uid: Union[highdicom.UID, str]
            Unique identifier of the frame of reference within which the
            coordinates are defined
        fiducial_uid: Union[str, None], optional
            Unique identifier for the content item
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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

    @property
    def frame_of_reference_uid(self) -> UID:
        """highdicom.UID: frame of reference UID"""
        return UID(self.ReferencedFrameOfReferenceUID)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'Scoord3DContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD3D

        Returns
        -------
        highdicom.sr.Scoord3DContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'Scoord3DContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD3D

        Returns
        -------
        highdicom.sr.Scoord3DContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.SCOORD3D)
        item = super(Scoord3DContentItem, cls)._from_dataset_base(dataset)
        return cast(Scoord3DContentItem, item)


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
            Concept name
        temporal_range_type: Union[highdicom.sr.TemporalRangeTypeValues, str]
            Name of the temporal range type
        referenced_sample_positions: Union[Sequence[int], None], optional
            One-based relative sample position of acquired time points
            within the time series
        referenced_time_offsets: Union[Sequence[float], None], optional
            Seconds after start of the acquisition of the time series
        referenced_date_time: Union[Sequence[datetime.datetime], None], optional
            Absolute time points
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
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
                DS(v, auto_format=True) for v in referenced_time_offsets
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

    @property
    def temporal_range_type(self) -> TemporalRangeTypeValues:
        """highdicom.sr.TemporalRangeTypeValues: temporal range type"""
        return TemporalRangeTypeValues(self.TemporalRangeType)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'TcoordContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TCOORD

        Returns
        -------
        highdicom.sr.TcoordContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'TcoordContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TCOORD

        Returns
        -------
        highdicom.sr.TcoordContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.TCOORD)
        item = super(TcoordContentItem, cls)._from_dataset_base(dataset)
        return cast(TcoordContentItem, item)


class WaveformContentItem(ContentItem):

    """DICOM SR document content item for value type WAVEFORM."""

    def __init__(
        self,
        name: Union[Code, CodedConcept],
        referenced_sop_class_uid: Union[str, UID],
        referenced_sop_instance_uid: Union[str, UID],
        referenced_waveform_channels: Optional[
            Union[int, Sequence[int]]
        ] = None,
        relationship_type: Union[str, RelationshipTypeValues, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Concept name
        referenced_sop_class_uid: Union[highdicom.UID, str]
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: Union[highdicom.UID, str]
            SOP Instance UID of the referenced image object
        referenced_waveform_channels: Union[Sequence[Tuple[int, int]], None], optional
            Pairs of waveform number (number of item in the Waveform Sequence)
            and channel definition number (number of item in the Channel
            Defition Sequence) to which the reference applies in case of a
            multi-channel waveform
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
        super(WaveformContentItem, self).__init__(
            ValueTypeValues.WAVEFORM, name, relationship_type
        )
        item = Dataset()
        item.ReferencedSOPClassUID = str(referenced_sop_class_uid)
        item.ReferencedSOPInstanceUID = str(referenced_sop_instance_uid)
        if referenced_waveform_channels is not None:
            item.ReferencedWaveformChannels = [
                i
                for item in referenced_waveform_channels
                for i in item
            ]
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

    @property
    def referenced_sop_class_uid(self) -> UID:
        """highdicom.UID: referenced SOP Class UID"""
        return UID(self.ReferencedSOPSequence[0].ReferencedSOPClassUID)

    @property
    def referenced_sop_instance_uid(self) -> UID:
        """highdicom.UID: referenced SOP Instance UID"""
        return UID(self.ReferencedSOPSequence[0].ReferencedSOPInstanceUID)

    @property
    def referenced_waveform_channels(self) -> Union[
        List[Tuple[int, int]],
        None
    ]:
        """Union[List[Tuple[int, int]], None]: referenced waveform channels"""
        if not hasattr(
            self.ReferencedSOPSequence[0],
            'ReferencedFrameNumber',
        ):
            return None
        val = getattr(
            self.ReferencedSOPSequence[0],
            'ReferencedFrameNumber',
        )
        return [
            (
                int(val[i]),
                int(val[i + 1]),
            )
            for i in range(0, len(val) - 1, 2)
        ]

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'WaveformContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type WAVEFORM

        Returns
        -------
        highdicom.sr.WaveformContentItem
            Content Item

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'WaveformContentItem':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type WAVEFORM

        Returns
        -------
        highdicom.sr.WaveformContentItem
            Content Item

        Note
        ----
        Does not create a copy, but modifies `dataset`.

        """
        _assert_value_type(dataset, ValueTypeValues.IMAGE)
        item = super(WaveformContentItem, cls)._from_dataset_base(dataset)
        return cast(WaveformContentItem, item)
