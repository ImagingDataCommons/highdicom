"""Utilities for working with SR document instances."""
from typing import List, Optional, Union

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code

from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import ValueTypeValues, RelationshipTypeValues
from highdicom.sr.value_types import ContentItem


def find_content_items(
    dataset: Dataset,
    name: Optional[Union[CodedConcept, Code]] = None,
    value_type: Optional[Union[ValueTypeValues, str]] = None,
    relationship_type: Optional[Union[RelationshipTypeValues, str]] = None,
    recursive: bool = False
) -> List[Dataset]:
    """Finds content items in a Structured Report document that match a given
    query.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        SR document instance
    name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
        Coded name that items should have
    value_type: Union[highdicom.sr.ValueTypeValues, str, None], optional
        Type of value that items should have
        (e.g. ``highdicom.sr.ValueTypeValues.CONTAINER``)
    relationship_type: Union[highdicom.sr.RelationshipTypeValues, str, None], optional
        Type of relationship that items should have with its parent
        (e.g. ``highdicom.sr.RelationshipTypeValues.CONTAINS``)
    recursive: bool, optional
        Whether search should be performed recursively, i.e. whether contained
        child content items should also be queried

    Returns
    -------
    List[pydicom.dataset.Dataset]
        flat list of all content items that matched the query

    Raises
    ------
    AttributeError
        When data set does not contain Content Sequence attribute.

    """  # noqa: E501
    def has_name(
        item: ContentItem,
        name: Optional[Union[Code, CodedConcept]]
    ) -> bool:
        if name is None:
            return True
        return item.name == name

    def has_value_type(
        item: ContentItem,
        value_type: Optional[Union[ValueTypeValues, str]]
    ) -> bool:
        if value_type is None:
            return True
        value_type = ValueTypeValues(value_type)
        return item.value_type == value_type

    def has_relationship_type(
            item: ContentItem,
            relationship_type: Optional[Union[RelationshipTypeValues, str]]
    ) -> bool:
        if relationship_type is None:
            return True
        if getattr(item, 'relationship_type', None) is None:
            return False
        relationship_type = RelationshipTypeValues(relationship_type)
        return item.relationship_type == relationship_type

    if not hasattr(dataset, 'ContentSequence'):
        raise AttributeError(
            'Data set does not contain a Content Sequence attribute.'
        )

    def search_tree(
        node: Dataset,
        name: Optional[Union[CodedConcept, Code]],
        value_type: Optional[Union[ValueTypeValues, str]],
        relationship_type: Optional[Union[RelationshipTypeValues, str]],
        recursive: bool
    ) -> List:
        matched_content_items = []
        for i, content_item in enumerate(node.ContentSequence):
            name_code = content_item.ConceptNameCodeSequence[0]
            item = ContentItem(
                value_type=content_item.ValueType,
                name=CodedConcept(
                    value=name_code.CodeValue,
                    scheme_designator=name_code.CodingSchemeDesignator,
                    meaning=name_code.CodeMeaning
                ),
                relationship_type=content_item.get('RelationshipType', None)
            )
            if (has_name(item, name) and
                    has_value_type(item, value_type) and
                    has_relationship_type(item, relationship_type)):
                matched_content_items.append(content_item)
            if hasattr(content_item, 'ContentSequence') and recursive:
                matched_content_items += search_tree(
                    node=content_item,
                    name=name,
                    value_type=value_type,
                    relationship_type=relationship_type,
                    recursive=recursive
                )
        return matched_content_items

    return search_tree(
        node=dataset,
        name=name,
        value_type=value_type,
        relationship_type=relationship_type,
        recursive=recursive
    )


def get_coded_name(item: Dataset) -> CodedConcept:
    """Gets the concept name of a SR Content Item.

    Parameters
    ----------
    item: pydicom.dataset.Dataset
        Content Item

    Returns
    -------
    highdicom.sr.CodedConcept
        Concept name

    """
    try:
        name = item.ConceptNameCodeSequence[0]
    except AttributeError:
        raise AttributeError(
            'Dataset does not contain attribute "ConceptNameCodeSequence" and '
            'thus doesn\'t represent a SR Content Item.'
        )
    return CodedConcept.from_dataset(name)


def get_coded_value(item: Dataset) -> CodedConcept:
    """Gets the value of a SR Content Item with Value Type CODE.

    Parameters
    ----------
    item: pydicom.dataset.Dataset
        Content Item

    Returns
    -------
    highdicom.sr.CodedConcept
        Value

    """
    try:
        value = item.ConceptCodeSequence[0]
    except AttributeError:
        raise AttributeError(
            'Dataset does not contain attribute "ConceptCodeSequence" and '
            'thus doesn\'t represent a SR Content Item of Value Type CODE.'
        )
    return CodedConcept.from_dataset(value)
