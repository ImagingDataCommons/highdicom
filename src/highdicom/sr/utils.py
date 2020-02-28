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
    name: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
        Coded name that items should have
    value_type: Union[highdicom.sr.value_types.ValueTypeValues, str], optional
        Type of value that items should have
        (e.g. ``highdicom.sr.value_types.ValueTypeValues.CONTAINER``)
    relationship_type: Union[highdicom.sr.enum.RelationshipTypeValues, str], optional
        Type of relationship that items should have with its parent
        (e.g. ``highdicom.sr.enum.RelationshipTypeValues.CONTAINS``)
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

    """  # noqa
    def has_name(item, name):
        if name is None:
            return True
        return item.name == name

    def has_value_type(item, value_type):
        if value_type is None:
            return True
        return item.value_type == value_type.value

    def has_relationship_type(item, relationship_type):
        if relationship_type is None:
            return True
        if item.relationship_type is None:
            return False
        return item.relationship_type == relationship_type.value

    if not hasattr(dataset, 'ContentSequence'):
        raise AttributeError(
            'Data set does not contain a Content Sequence attribute.'
        )

    def search_tree(node, name, value_type, relationship_type, recursive):
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
