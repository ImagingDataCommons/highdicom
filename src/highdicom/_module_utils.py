from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from pydicom import Dataset

from highdicom._iods import IOD_MODULE_MAP, SOP_CLASS_UID_IOD_KEY_MAP
from highdicom._modules import MODULE_ATTRIBUTE_MAP
from highdicom._iods import (
    IOD_MODULE_MAP,
    SOP_CLASS_UID_IOD_KEY_MAP
)


# Allowed values for the type of an attribute
class AttributeTypeValues(Enum):

    """Enumerated values for the type of an attribute."""

    REQUIRED = '1'
    CONDITIONALLY_REQUIRED = '1C'
    REQUIRED_EMPTY_IF_UNKNOWN = '2'
    CONDITIONALLY_REQUIRED_EMPTY_IF_UNKNOWN = '2C'
    OPTIONAL = '3'


class ModuleUsageValues(Enum):

    """Enumerated values for the usage of a module."""

    MANDATORY = 'M'
    CONDITIONAL = 'U'
    USER_OPTIONAL = 'C'


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


def get_module_usage(
    module_key: str,
    sop_class_uid: str
) -> Union[ModuleUsageValues, None]:
    """Get the usage (M/C/U) of a module within an IOD.

    Parameters
    ----------
    module_key: str
        Key for the module.
    sop_class_uid: str
        SOP Class UID identifying the IOD.

    Returns
    -------
    Union[ModuleUsageValues, None]:
        Usage of the module within the IOD identified by the SOP Class UID, if
        a module with that name is present within the IOD. None, if the module
        is not present within the IOD.


    """
    try:
        iod_name = SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid]
    except KeyError as e:
        msg = f'No IOD found for SOP Class UID: {sop_class_uid}.'
        raise KeyError(msg) from e

    for mod in IOD_MODULE_MAP[iod_name]:
        if mod['key'] == module_key:
            return ModuleUsageValues(mod['usage'])

    return None


def is_attribute_in_iod(attribute: str, sop_class_uid: str) -> bool:
    """Check whether an attribute is present within an IOD.

    Parameters
    ----------
    attribute: str
        Keyword for the attribute
    sop_class_uid: str
        SOP Class UID identifying the IOD.

    Returns
    -------
    bool:
        True if the attribute is present within any module within the IOD
        specified by the sop_class_uid. False otherwise.

    """
    try:
        iod_name = SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid]
    except KeyError as e:
        msg = f'No IOD found for SOP Class UID: {sop_class_uid}.'
        raise KeyError(msg) from e

    for module in IOD_MODULE_MAP[iod_name]:
        module_attributes = MODULE_ATTRIBUTE_MAP[module['key']]
        for attr in module_attributes:
            if attr['keyword'] == attribute:
                return True

    return False


def does_iod_have_pixel_data(sop_class_uid: str) -> bool:
    """Check whether any pixel data attribute is present within an IOD.

    This may be used to determine whether a particular SOP class represents an
    image.

    Parameters
    ----------
    sop_class_uid: str
        SOP Class UID identifying the IOD.

    Returns
    -------
    bool:
        True if the any pixel data attribute is present within any module
        within the IOD specified by the sop_class_uid. False otherwise. Pixel
        data attributes include ``PixelData``, ``FloatPixelData``, and
        ``DoubleFloatPixelData``.

    """
    pixel_attrs = [
        'PixelData',
        'FloatPixelData',
        'DoubleFloatPixelData',
    ]
    return any(
        is_attribute_in_iod(attr, sop_class_uid) for attr in pixel_attrs
    )
