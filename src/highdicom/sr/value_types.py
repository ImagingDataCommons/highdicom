"""DICOM structured reporting content item value types."""
import datetime
from collections import defaultdict
from copy import deepcopy
from typing import (
    cast,
    overload,
    Union,
)
from collections.abc import Iterable, Iterator, Sequence
from typing_extensions import Self

import numpy as np
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
from pydicom.valuerep import DA, DS, TM, DT, PersonName

from highdicom.spatial import are_points_coplanar
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
    dataset: pydicom.Dataset
        Dataset representing an SR Content Item
    value_type: highdicom.sr.ValueTypeValues
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
        raise AttributeError(f'Dataset is not an SR Content Item:\n{dataset}.')
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
        value_type: str | ValueTypeValues,
        name: Code | CodedConcept,
        relationship_type: str | RelationshipTypeValues | None
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
        super().__init__()
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
        value: DataElement | DataElementSequence
    ) -> None:
        if name == 'ContentSequence':
            super().__setattr__(name, ContentSequence(value))
        else:
            super().__setattr__(name, value)

    @classmethod
    def _from_dataset_derived(cls, dataset: Dataset) -> Self:
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
        return content_item_cls.from_dataset(
            dataset,
            copy=False
        )  # type: ignore

    @classmethod
    def _from_dataset_base(cls, dataset: Dataset) -> Self:
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
            item.ContentSequence = ContentSequence.from_sequence(
                item.ContentSequence,
                copy=False
            )
        item.ConceptNameCodeSequence = [
            CodedConcept.from_dataset(
                item.ConceptNameCodeSequence[0],
                copy=False
            )
        ]
        return cast(Self, item)

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
    def relationship_type(self) -> RelationshipTypeValues | None:
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
        items: None | (
            Union[Sequence[ContentItem], 'ContentSequence']
        ) = None,
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

        self._lut: dict[
            Code | CodedConcept,
            list[ContentItem]
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
        idx: slice | int,
        val: Iterable[ContentItem] | ContentItem
    ) -> None:   # type: ignore[override]
        if isinstance(val, Iterable):
            items = val
        else:
            items = [val]
        for i in items:
            if not isinstance(i, ContentItem):
                raise TypeError(
                    f'Items of "{self.__class__.__name__}" must '
                    'have type ContentItem.'
                )
        super().__setitem__(idx, val)  # type: ignore

    def __delitem__(
        self,
        idx: slice | int
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
                f'Items of "{self.__class__.__name__}" '
                'must have type ContentItem.'
            )
        error_message = f'Item "{val.name}" is not in Sequence.'
        try:
            matches = self._lut[val.name]
        except KeyError as e:
            raise ValueError(error_message) from e
        try:
            index = matches.index(val)
        except ValueError as e:
            raise ValueError(error_message) from e
        return index

    def find(self, name: Code | CodedConcept) -> Self:
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

    def get_nodes(self) -> Self:
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
                f'Items of "{self.__class__.__name__}" '
                'must have type ContentItem.'
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
                f'Items of "{self.__class__.__name__}" '
                'must have type ContentItem.'
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
        is_sr: bool = True,
        copy: bool = True,
    ) -> Self:
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
        copy: bool
            If True, the underlying sequence is deep-copied such that the
            original sequence remains intact. If False, this operation will
            alter the original sequence in place.

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
            if copy:
                dataset_copy = deepcopy(dataset)
            else:
                dataset_copy = dataset
            item = ContentItem._from_dataset_derived(dataset_copy)
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
        except TypeError as e:
            raise ValueError(
                f'Item #{index} of sequence is not an SR Content Item '
                f'because it has unknown Value Type "{dataset.ValueType}":'
                f'\n{dataset}'
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f'Item #{index} of sequence is not an SR Content Item:\n'
                f'{dataset}'
            ) from e
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
        name: Code | CodedConcept,
        value: Code | CodedConcept,
        relationship_type: str | RelationshipTypeValues | None = None,
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
        super().__init__(
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
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CODE
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.CodeContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.CODE)
        item = super()._from_dataset_base(dataset_copy)
        item.ConceptCodeSequence = DataElementSequence([
            CodedConcept.from_dataset(item.ConceptCodeSequence[0], copy=False)
        ])
        return cast(Self, item)


class PnameContentItem(ContentItem):

    """DICOM SR document content item for value type PNAME."""

    def __init__(
        self,
        name: Code | CodedConcept,
        value: str | PersonName,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
            ValueTypeValues.PNAME, name, relationship_type
        )
        check_person_name(value)
        self.PersonName = PersonName(value)

    @property
    def value(self) -> PersonName:
        """pydicom.valuerep.PersonName: person name"""
        return PersonName(self.PersonName)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type PNAME
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.PnameContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.PNAME)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class TextContentItem(ContentItem):

    """DICOM SR document content item for value type TEXT."""

    def __init__(
        self,
        name: Code | CodedConcept,
        value: str,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
            ValueTypeValues.TEXT, name, relationship_type
        )
        self.TextValue = str(value)

    @property
    def value(self) -> str:
        """str: text value"""
        return self.TextValue

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TEXT
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.TextContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.TEXT)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class TimeContentItem(ContentItem):

    """DICOM SR document content item for value type TIME."""

    def __init__(
        self,
        name: Code | CodedConcept,
        value: str | datetime.time | TM,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
            ValueTypeValues.TIME, name, relationship_type
        )
        self.Time = TM(value)

    @property
    def value(self) -> datetime.time:
        """datetime.time: time"""
        if isinstance(self.Time, TM):
            value = self.Time
        else:
            try:
                value = TM(self.Time)
            except ValueError as exception:
                raise ValueError(
                    f'Could not decode time value "{self.Time}"'
                ) from exception
        return value.replace()

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TIME
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.TimeContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.TIME)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class DateContentItem(ContentItem):

    """DICOM SR document content item for value type DATE."""

    def __init__(
        self,
        name: Code | CodedConcept,
        value: str | datetime.date | DA,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
            ValueTypeValues.DATE, name, relationship_type
        )
        self.Date = DA(value)

    @property
    def value(self) -> datetime.date:
        """datetime.date: date"""
        if isinstance(self.Date, DA):
            value = self.Date
        else:
            try:
                value = DA(self.Date)
            except ValueError as exception:
                raise ValueError(
                    f'Could not decode date value "{self.Date}"'
                ) from exception
        return value.replace()

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATE
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.DateContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.DATE)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class DateTimeContentItem(ContentItem):

    """DICOM SR document content item for value type DATETIME."""

    def __init__(
        self,
        name: Code | CodedConcept,
        value: str | datetime.datetime | DT,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
            ValueTypeValues.DATETIME, name, relationship_type
        )
        self.DateTime = DT(value)

    @property
    def value(self) -> datetime.datetime:
        """datetime.datetime: datetime"""
        if isinstance(self.DateTime, DT):
            value = self.DateTime
        else:
            try:
                value = DT(self.DateTime)
            except ValueError as exception:
                raise ValueError(
                    f'Could not decode datetime value "{self.DateTime}"'
                ) from exception
        return value.replace()

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type DATETIME
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.DateTimeContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.DATETIME)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class UIDRefContentItem(ContentItem):

    """DICOM SR document content item for value type UIDREF."""

    def __init__(
        self,
        name: Code | CodedConcept,
        value: str | UID,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
            ValueTypeValues.UIDREF, name, relationship_type
        )
        self.UID = value

    @property
    def value(self) -> UID:
        """highdicom.UID: UID"""
        return UID(self.UID)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type UIDREF
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.UIDRefContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.UIDREF)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class NumContentItem(ContentItem):

    """DICOM SR document content item for value type NUM."""

    def __init__(
        self,
        name: Code | CodedConcept,
        value: float,
        unit: Code | CodedConcept,
        qualifier: Code | CodedConcept | None = None,
        relationship_type: str | RelationshipTypeValues | None = None,
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
        super().__init__(
            ValueTypeValues.NUM, name, relationship_type
        )
        self.MeasuredValueSequence: list[Dataset] = []
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
    def value(self) -> int | float:
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
    def qualifier(self) -> CodedConcept | None:
        """Union[highdicom.sr.CodedConcept, None]: qualifier"""
        try:
            return self.NumericValueQualifierCodeSequence[0]
        except AttributeError:
            return None

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type NUM
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.NumContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.NUM)
        item = super()._from_dataset_base(dataset_copy)
        unit_item = (
            item
            .MeasuredValueSequence[0]
            .MeasurementUnitsCodeSequence[0]
        )
        item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence = [
            CodedConcept.from_dataset(unit_item, copy=False)
        ]
        if hasattr(item, 'NumericValueQualifierCodeSequence'):
            qualifier_item = item.NumericValueQualifierCodeSequence[0]
            item.NumericValueQualifierCodeSequence = DataElementSequence([
                CodedConcept.from_dataset(qualifier_item, copy=False)
            ])
        return cast(Self, item)


class ContainerContentItem(ContentItem):

    """DICOM SR document content item for value type CONTAINER."""

    def __init__(
        self,
        name: Code | CodedConcept,
        is_content_continuous: bool = True,
        template_id: str | None = None,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
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
    def template_id(self) -> str | None:
        """Union[str, None]: template identifier"""
        try:
            item = self.ContentTemplateSequence[0]
            return item.TemplateIdentifier
        except (AttributeError, IndexError):
            return None

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type CONTAINER
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.ContainerContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.CONTAINER)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class CompositeContentItem(ContentItem):

    """DICOM SR document content item for value type COMPOSITE."""

    def __init__(
        self,
        name: Code | CodedConcept,
        referenced_sop_class_uid: str | UID,
        referenced_sop_instance_uid: str | UID,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
            ValueTypeValues.COMPOSITE, name, relationship_type
        )
        item = Dataset()
        item.ReferencedSOPClassUID = str(referenced_sop_class_uid)
        item.ReferencedSOPInstanceUID = str(referenced_sop_instance_uid)
        self.ReferencedSOPSequence = [item]

    @property
    def value(self) -> tuple[UID, UID]:
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
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type COMPOSITE
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.CompositeContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.COMPOSITE)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class ImageContentItem(ContentItem):

    """DICOM SR document content item for value type IMAGE."""

    def __init__(
        self,
        name: Code | CodedConcept,
        referenced_sop_class_uid: str | UID,
        referenced_sop_instance_uid: str | UID,
        referenced_frame_numbers: None | (
            int | Sequence[int]
        ) = None,
        referenced_segment_numbers: None | (
            int | Sequence[int]
        ) = None,
        relationship_type: str | RelationshipTypeValues | None = None,
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
            Number of segment(s) to which the reference applies in case of a
            segmentation image
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
        super().__init__(
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
    def value(self) -> tuple[UID, UID]:
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
    def referenced_frame_numbers(self) -> list[int] | None:
        """Union[List[int], None]: referenced frame numbers"""
        if not hasattr(
            self.ReferencedSOPSequence[0],
            'ReferencedFrameNumber',
        ):
            return None
        val = self.ReferencedSOPSequence[0].ReferencedFrameNumber
        if isinstance(val, MultiValue):
            return [int(v) for v in val]
        else:
            return [int(val)]

    @property
    def referenced_segment_numbers(self) -> list[int] | None:
        """Union[List[int], None]
            referenced segment numbers
        """
        if not hasattr(
            self.ReferencedSOPSequence[0],
            'ReferencedSegmentNumber',
        ):
            return None
        val = self.ReferencedSOPSequence[0].ReferencedSegmentNumber
        if isinstance(val, MultiValue):
            return [int(v) for v in val]
        else:
            return [int(val)]

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type IMAGE
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.ImageContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.IMAGE)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class ScoordContentItem(ContentItem):

    """DICOM SR document content item for value type SCOORD.

    Note
    ----
    Spatial coordinates are defined in image space and have pixel units.

    """

    def __init__(
        self,
        name: Code | CodedConcept,
        graphic_type: str | GraphicTypeValues,
        graphic_data: np.ndarray,
        pixel_origin_interpretation: (
            str |
            PixelOriginInterpretationValues |
            None
        ) = None,
        fiducial_uid: str | UID | None = None,
        relationship_type: str | RelationshipTypeValues | None = None
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
            (``highdicom.sr.PixelOriginInterpretationValues.FRAME``). This
            distinction is only meaningful when the referenced image is a tiled
            image. In other situations, this should be left unspecified.
        fiducial_uid: Union[highdicom.UID, str, None], optional
            Unique identifier for the content item
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
        super().__init__(
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
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.ScoordContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.SCOORD)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class Scoord3DContentItem(ContentItem):

    """DICOM SR document content item for value type SCOORD3D.

    Note
    ----
    Spatial coordinates are defined in the patient or specimen-based coordinate
    system and have millimeter unit.

    """

    def __init__(
        self,
        name: Code | CodedConcept,
        graphic_type: GraphicTypeValues3D | str,
        graphic_data: np.ndarray,
        frame_of_reference_uid: str | UID,
        fiducial_uid: str | UID | None = None,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
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
            if graphic_type == GraphicTypeValues3D.POLYGON:
                if not np.array_equal(graphic_data[0], graphic_data[-1]):
                    raise ValueError(
                        'Graphic data of a 3D scoord of graphic type "POLYGON" '
                        'must be closed, i.e. the first and last points must '
                        'be equal.'
                    )

        # Check for coplanarity, if required by the graphic type
        if graphic_type in (
            GraphicTypeValues3D.POLYGON,
            GraphicTypeValues3D.ELLIPSE,
        ):
            if not are_points_coplanar(graphic_data):
                raise ValueError(
                    'Graphic data of a 3D scoord of type '
                    f'"{graphic_type.value}" must contain co-planar points.'
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
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD3D
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.Scoord3DContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.SCOORD3D)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class TcoordContentItem(ContentItem):

    """DICOM SR document content item for value type TCOORD."""

    def __init__(
        self,
        name: Code | CodedConcept,
        temporal_range_type: str | TemporalRangeTypeValues,
        referenced_sample_positions: Sequence[int] | None = None,
        referenced_time_offsets: Sequence[float] | None = None,
        referenced_date_time: Sequence[datetime.datetime] | None = None,
        relationship_type: str | RelationshipTypeValues | None = None
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
        super().__init__(
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
                'One of the following arguments is required: '
                '"referenced_sample_positions", '
                '"referenced_time_offsets", '
                '"referenced_date_time"'
            )

    @property
    def value(self) -> list[int] | list[float] | list[datetime.datetime]:
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
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type TCOORD
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.TcoordContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.TCOORD)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)


class WaveformContentItem(ContentItem):

    """DICOM SR document content item for value type WAVEFORM."""

    def __init__(
        self,
        name: Code | CodedConcept,
        referenced_sop_class_uid: str | UID,
        referenced_sop_instance_uid: str | UID,
        referenced_waveform_channels: None | (
            int | Sequence[int]
        ) = None,
        relationship_type: str | RelationshipTypeValues | None = None,
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
            Definition Sequence) to which the reference applies in case of a
            multi-channel waveform
        relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
            Type of relationship with parent content item

        """  # noqa: E501
        super().__init__(
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
    def value(self) -> tuple[UID, UID]:
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
    def referenced_waveform_channels(self) -> (
        list[tuple[int, int]] |
        None
    ):
        """Union[List[Tuple[int, int]], None]: referenced waveform channels"""
        if not hasattr(
            self.ReferencedSOPSequence[0],
            'ReferencedFrameNumber',
        ):
            return None
        val = self.ReferencedSOPSequence[0].ReferencedFrameNumber
        return [
            (
                int(val[i]),
                int(val[i + 1]),
            )
            for i in range(0, len(val) - 1, 2)
        ]

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type WAVEFORM
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.sr.WaveformContentItem
            Content Item

        """
        if copy:
            dataset_copy = deepcopy(dataset)
        else:
            dataset_copy = dataset
        _assert_value_type(dataset_copy, ValueTypeValues.IMAGE)
        item = super()._from_dataset_base(dataset_copy)
        return cast(Self, item)
