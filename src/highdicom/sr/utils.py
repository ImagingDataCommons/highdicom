"""Utilities for working with SR document instances."""
from collections import defaultdict
from typing import List, Mapping, Optional, Sequence, Tuple, Union

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


class _ReferencedSOPInstance(Dataset):

    """Class representing an item of Referenced SOP Sequence."""

    def __init__(
        self,
        sop_class_uid: str,
        sop_instance_uid: str,
    ) -> None:
        """
        Parameters
        ----------
        sop_class_uid: str
            SOP Class UID of the referenced object
        sop_instance_uid: str
            SOP Instance UID of the referenced object

        """
        self.ReferencedSOPClassUID = sop_class_uid
        self.ReferencedSOPInstanceUID = sop_instance_uid


def _create_references(
    collection: Mapping[Tuple[str, str], Sequence[_ReferencedSOPInstance]]
) -> List[Dataset]:
    """Create references.

    Parameters
    ----------
    collection: Mapping[Tuple[str, str], Sequence[pydicom.dataset.Dataset]]
        Mapping of Study and Series Instance UIDs to referenced SOP instances

    Returns
    -------
    List[pydicom.dataset.Dataset]
        Items containing the Study Instance UID and the
        Referenced Series Sequence attributes

    """  # noqa: E501
    study_collection: Mapping[str, List[Dataset]] = defaultdict(list)
    for (study_uid, series_uid), instance_items in collection.items():
        series_item = Dataset()
        series_item.SeriesInstanceUID = series_uid
        series_item.ReferencedSOPSequence = instance_items
        study_collection[study_uid].append(series_item)

    ref_items = []
    for study_uid, series_items in study_collection.items():
        study_item = Dataset()
        study_item.StudyInstanceUID = study_uid
        study_item.ReferencedSeriesSequence = series_items
        ref_items.append(study_item)

    return ref_items


def collect_evidence(
    evidence: Sequence[Dataset],
    content: Dataset
) -> Tuple[List[Dataset], List[Dataset]]:
    """Collect evidence for a SR document.

    Any `evidence` that is referenced in `content` via IMAGE or
    COMPOSITE content items will be grouped together for inclusion in the
    Current Requested Procedure Evidence Sequence and all remaining
    evidence will be grouped for potential inclusion in the Pertinent Other
    Evidence Sequence.

    Parameters
    ----------
    evidence: List[pydicom.dataset.Dataset]
        Metadata of instances that serve as evidence for the SR document content
    content: pydicom.dataset.Dataset
        SR document content

    Returns
    -------
    current_requested_procedure_evidence: List[pydicom.dataset.Dataset]
        Items of the Current Requested Procedure Evidence Sequence
    other_pertinent_evidence: List[pydicom.dataset.Dataset]
        Items of the Pertinent Other Evidence Sequence

    Raises
    ------
    ValueError
        When a SOP instance is referenced in `content` but not provided as
        `evidence`

    """  # noqa: E501
    references = find_content_items(
        content,
        value_type=ValueTypeValues.IMAGE,
        recursive=True
    )
    references += find_content_items(
        content,
        value_type=ValueTypeValues.COMPOSITE,
        recursive=True
    )
    ref_uids = set([
        ref.ReferencedSOPSequence[0].ReferencedSOPInstanceUID
        for ref in references
    ])
    evd_uids = set()
    ref_group: Mapping[Tuple[str, str], List[Dataset]] = defaultdict(list)
    unref_group: Mapping[Tuple[str, str], List[Dataset]] = defaultdict(list)
    for evd in evidence:
        if evd.SOPInstanceUID in evd_uids:
            # Skip potential duplicates
            continue
        evd_item = Dataset()
        evd_item.ReferencedSOPClassUID = evd.SOPClassUID
        evd_item.ReferencedSOPInstanceUID = evd.SOPInstanceUID
        key = (evd.StudyInstanceUID, evd.SeriesInstanceUID)
        if evd.SOPInstanceUID in ref_uids:
            ref_group[key].append(evd_item)
        else:
            unref_group[key].append(evd_item)
        evd_uids.add(evd.SOPInstanceUID)
    if not(ref_uids.issubset(evd_uids)):
        missing_uids = ref_uids.difference(evd_uids)
        raise ValueError(
            'No evidence was provided for the following SOP instances, '
            'which are referenced in the document content: "{}"'.format(
                '", "'.join(missing_uids)
            )
        )

    ref_items = _create_references(ref_group)
    unref_items = _create_references(unref_group)
    return (ref_items, unref_items)
