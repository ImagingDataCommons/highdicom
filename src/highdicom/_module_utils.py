from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from pydicom import Dataset

from highdicom._modules import MODULE_ATTRIBUTE_MAP


# Allowed values for the type of an attribute
class AttributeTypeValues(Enum):

    """Enumerated values for the type of an attribute."""

    REQUIRED = '1'
    CONDITIONALLY_REQUIRED = '1C'
    REQUIRED_EMPTY_IF_UNKNOWN = '2'
    CONDITIONALLY_REQUIRED_EMPTY_IF_UNKNOWN = '2C'
    OPTIONAL = '3'


def check_required_attributes(
    dataset: Dataset,
    module: str,
    base_path: Optional[Sequence[str]] = None,
    recursive: bool = True,
    check_optional_sequences: bool = True
) -> None:
    """Check that a dataset contains a module's required attributes.

    This may be used to check a top-level dataset, or for a dataset
    representing an element of a sequence at an arbitrary level of
    nesting, as specified by passing the path parameter.

    The function does not return anything, but throws an AttributeError
    if the checks fail.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        The dataset to be checked.
    module: str
        Name of the module whose attributes should be checked.
    base_path: Optional[Sequence[str]]
        Path within the module that the dataset is intended to occupy,
        represented as a sequence of strings with each string representing
        the name of a data element sequence.  If omitted, the dataset is
        assumed to represent the top-level dataset.
    recursive: bool
        If True (the default), the attributes within nested data element
        sequences will also be checked. If False, only attributes expected
        in the top level of the passed dataset will be checked.
    check_optional_sequences: bool
        If True, the required attributes of an optional sequence will be
        checked if the optional sequence is present in the dataset or
        any of its nested sequences. This is ignored if recursive is
        False.

    Raises
    ------
    AttributeError
        If any of the required (type 1 or 2) attributes are not present
        in the dataset for the given module.

    Notes
    -----
    This function merely checks for the presence of required attributes.
    It does not check whether the data elements are empty or not, whether
    there are additional, invalid attributes, or whether the values of the
    data elements are allowed. Furthermore, it does not attempt to
    check for conditionally required attributes. Therefore, this check
    represents a necessary but not sufficient condition for a dataset
    to be valid according to the DICOM standard.

    """
    # Only check for type 1 and type 2 attributes
    types_to_check = [
        AttributeTypeValues.REQUIRED,
        AttributeTypeValues.REQUIRED_EMPTY_IF_UNKNOWN
    ]

    # Construct tree once and re-use in all recursive calls
    tree = construct_module_tree(module)

    if base_path is not None:
        for p in base_path:
            try:
                tree = tree['attributes'][p]
            except KeyError:
                raise AttributeError(f"Invalid base path: {base_path}.")

    # Define recursive function
    def check(
        dataset: Dataset,
        subtree: Dict[str, Any],
        path: List[str]
    ) -> None:
        for kw, item in subtree['attributes'].items():
            required = item['type'] in types_to_check
            if required:
                if not hasattr(dataset, kw):
                    if len(path) > 0:
                        msg = (
                            "Dataset does not have required attribute "
                            f"'{kw}' at path {path}"
                        )
                    else:
                        msg = (
                            "Dataset does not have required attribute "
                            f"'{kw}'."
                        )
                    raise AttributeError(
                        msg
                    )
            if recursive:
                sequence_exists = (
                    'attributes' in subtree['attributes'][kw] and
                    hasattr(dataset, kw)
                )
                if required or (sequence_exists and check_optional_sequences):
                    # Recurse down to the next level of the tree, if it exists
                    new_subtree = subtree['attributes'][kw]
                    if 'attributes' in new_subtree:
                        # Need to perform the check on all elements of the
                        # sequence
                        for elem in dataset[kw]:
                            check(
                                elem,
                                subtree=new_subtree,
                                path=path + [kw]
                            )

    # Kick off recursion
    check(dataset, tree, [])


def construct_module_tree(module: str) -> Dict[str, Any]:
    """Return module attributes arranged in a tree structure.

    Parameters
    ----------
    module: str
        Name of the module.

    Returns
    -------
    Dict[str, Any]
        Tree-structured representation of the module attributes in the form
        of nested dictionaries. Each level of the tree consists of a dictionary
        with up to two keys: 'type' and 'attributes'. 'type' maps to a
        :class:AttributeTypeValues that describes the type of the attribute
        (1, 1C, 2, 3), and is present at every level except the root.
        'attributes' is present for any attribute that is a sequence
        containing other attributes, and maps attribute keywords to a
        dictionary that forms an item in the next level of the tree structure.

    """
    if module not in MODULE_ATTRIBUTE_MAP:
        raise AttributeError(f"No such module found: '{module}'.")
    tree: Dict[str, Any] = {'attributes': {}}
    for item in MODULE_ATTRIBUTE_MAP[module]:
        location = tree['attributes']
        for p in item['path']:
            if 'attributes' not in location[p]:
                location[p]['attributes'] = {}
            location = location[p]['attributes']
        location[item['keyword']] = {
            'type': AttributeTypeValues(item['type'])
        }
    return tree
