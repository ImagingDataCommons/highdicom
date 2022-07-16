"""Functions for working with DICOM value representations."""
import re
from typing import Union
import warnings

from pydicom.valuerep import PersonName


def check_person_name(person_name: Union[str, PersonName]) -> None:
    """Check value is valid for the value representation "person name".

    The DICOM Person Name (PN) value representation has a specific format with
    multiple components (family name, given name, middle name, prefix, suffix)
    separated by caret characters ('^'), where any number of components may be
    missing and trailing caret separators may be omitted. Unfortunately it is
    both easy to make a mistake when constructing names with this format, and
    impossible to check for certain whether it has been done correctly.

    This function checks for strings representing person names that have a high
    likelihood of having been encoded incorrectly and raises an exception if
    such a case is found.

    A string is considered to be an invalid person name if it contains no caret
    characters.

    Note
    ----
    A name consisting of only a family name component (e.g. ``'Bono'``) is
    valid according to the standard but will be disallowed by this function.
    However if necessary, such a name can be still be encoded by adding a
    trailing caret character to disambiguate the meaning (e.g. ``'Bono^'``).

    Parameters
    ----------
    person_name: Union[str, pydicom.valuerep.PersonName]
        Name to check.

    Raises
    ------
    ValueError
        If the provided value is highly likely to be an invalid person name.
    TypeError
        If the provided person name has an invalid type.

    """
    if not isinstance(person_name, (str, PersonName)):
        raise TypeError('Invalid type for a person name.')

    name_url = (
        'http://dicom.nema.org/dicom/2013/output/chtml/part05/'
        'sect_6.2.html#sect_6.2.1.2'
    )
    if '^' not in person_name and person_name != '':  # empty string is allowed
        warnings.warn(
            f'The string "{person_name}" is unlikely to represent the '
            'intended person name since it contains only a single component. '
            'Construct a person name according to the format in described '
            f'in {name_url}, or, in pydicom 2.2.0 or later, use the '
            'pydicom.valuerep.PersonName.from_named_components() method '
            'to construct the person name correctly. If a single-component '
            'name is really intended, add a trailing caret character to '
            'disambiguate the name.',
            UserWarning
        )


def _check_code_string(value: str) -> None:
    """Check value is valid for the value representation "code string".

    Parameters
    ----------
    value: str
        Code string

    Raises
    ------
    TypeError
        When `value` is not a string
    ValueError
        When `value` has zero or more than 16 characters or when `value`
        contains characters that are invalid for the value representation

    Note
    ----
    The checks performed by this function are stricter than requirements
    imposed by the standard. For example, it does not allow leading or trailing
    spaces or underscores.
    Therefore, it should only be used to check values for creation of objects
    but not for parsing of existing objects.

    """
    if not isinstance(value, str):
        raise TypeError('Invalid type for a code string.')

    if re.match(r'[A-Z0-9_ ]{1,16}$', value) is None:
        raise ValueError(
            'Code string must contain between 1 and 16 characters that are '
            'either uppercase letters, numbers, spaces, or underscores.'
        )

    if re.match(r'[0-9 _]{1}.*', value) is not None:
        raise ValueError(
            'Code string must not start with a number, space, or underscore.'
        )

    if re.match(r'.*[_ ]$', value) is not None:
        raise ValueError(
            'Code string must not end with a space or underscore.'
        )


def _check_long_string(s: str) -> None:
    """Check that a Python string is valid for use as DICOM Long String.

    Parameters
    ----------
    s: str
        Python string to check.

    Raises
    ------
    ValueError:
        If the string s is not valid as a DICOM Long String due to length or
        the characters it contains.

    """
    if len(s) > 64:
        raise ValueError(
            'Values of DICOM value representation Long String (LO) must not '
            'exceed 64 characters.'
        )
    if '\\' in s:
        raise ValueError(
            'Values of DICOM value representation Long String (LO) must not '
            'contain the backslash character.'
        )


def _check_short_text(s: str) -> None:
    """Check that a Python string is valid for use as DICOM Short Text.

    Parameters
    ----------
    s: str
        Python string to check.

    Raises
    ------
    ValueError:
        If the string s is not valid as a DICOM Short Text due to length or
        the characters it contains.

    """
    if len(s) > 1024:
        raise ValueError(
            'Values of DICOM value representation Short Text (ST) must not '
            'exceed 1024 characters.'
        )
    if '\\' in s:
        raise ValueError(
            'Values of DICOM value representation Short Text (ST) must not '
            'contain the backslash character.'
        )


def _check_long_text(s: str) -> None:
    """Check that a Python string is valid for use as DICOM Long Text.

    Parameters
    ----------
    s: str
        Python string to check.

    Raises
    ------
    ValueError:
        If the string s is not valid as a DICOM Long Text due to length or
        the characters it contains.

    """
    if len(s) > 10240:
        raise ValueError(
            'Values of DICOM value representation Long Text (LT) must not '
            'exceed 10240 characters.'
        )
    if '\\' in s:
        raise ValueError(
            'Values of DICOM value representation Long Text (LT) must not '
            'contain the backslash character.'
        )
