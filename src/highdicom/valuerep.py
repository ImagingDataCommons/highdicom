"""Functions for working with DICOM value representations."""
from collections.abc import Sequence
import re
import warnings

from pydicom.valuerep import PersonName
from highdicom.enum import SpecificCharacterSetValues


def check_person_name(person_name: str | PersonName) -> None:
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
        'https://dicom.nema.org/dicom/2013/output/chtml/part05/'
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
            UserWarning,
            stacklevel=2,
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


def _check_short_string(s: str) -> None:
    """Check that a Python string is valid for use as DICOM Short String.

    Parameters
    ----------
    s: str
        Python string to check.

    Raises
    ------
    ValueError:
        If the string s is not valid as a DICOM Short String due to length or
        the characters it contains.

    """
    if len(s) > 16:
        raise ValueError(
            'Values of DICOM value representation Short String (SH) must not '
            'exceed 16 characters.'
        )
    if '\\' in s:
        raise ValueError(
            'Values of DICOM value representation Short String (SH) must not '
            'contain the backslash character.'
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


def _check_specific_character_set(
    specific_character_set: (
        SpecificCharacterSetValues |
        str |
        Sequence[SpecificCharacterSetValues | str]
    ),
) -> str | Sequence[str]:
    """Check a list of user-supplied encodings.

    Returns
    -------
    str | Sequence[str]:
        Value as it should be placed into the Specific Character Set attribute.

    """
    single_byte_code_extensions = {
        SpecificCharacterSetValues.DEFAULT_REPERTOIRE_CODE_EXTENSIONS,
        SpecificCharacterSetValues.LATIN_ALPHABET_NO_1_CODE_EXTENSIONS,
        SpecificCharacterSetValues.LATIN_ALPHABET_NO_2_CODE_EXTENSIONS,
        SpecificCharacterSetValues.LATIN_ALPHABET_NO_3_CODE_EXTENSIONS,
        SpecificCharacterSetValues.LATIN_ALPHABET_NO_4_CODE_EXTENSIONS,
        SpecificCharacterSetValues.CYRILLIC_CODE_EXTENSIONS,
        SpecificCharacterSetValues.ARABIC_CODE_EXTENSIONS,
        SpecificCharacterSetValues.GREEK_CODE_EXTENSIONS,
        SpecificCharacterSetValues.HEBREW_CODE_EXTENSIONS,
        SpecificCharacterSetValues.LATIN_ALPHABET_NO_5_CODE_EXTENSIONS,
        SpecificCharacterSetValues.LATIN_ALPHABET_NO_9_CODE_EXTENSIONS,
        SpecificCharacterSetValues.JAPANESE_CODE_EXTENSIONS,
        SpecificCharacterSetValues.THAI_CODE_EXTENSIONS,
    }
    multi_byte_code_extensions = {
        SpecificCharacterSetValues.JAPANESE_KANJI_CODE_EXTENSIONS,
        SpecificCharacterSetValues.JAPANESE_KANJI_SUPPLEMENTARY_CODE_EXTENSIONS,
        SpecificCharacterSetValues.KOREAN_CODE_EXTENSIONS,
    }
    all_code_extensions = (
        single_byte_code_extensions | multi_byte_code_extensions
    )

    if isinstance(specific_character_set, str):
        specific_character_set = SpecificCharacterSetValues(
            specific_character_set
        )
    elif isinstance(specific_character_set, SpecificCharacterSetValues):
        pass
    else:
        specific_character_set = [
            SpecificCharacterSetValues(enc) for enc in specific_character_set
        ]
        if len(specific_character_set) == 0:
            raise ValueError(
                "Parameter 'specific_character_set' must not be empty."
            )
        elif len(specific_character_set) == 1:
            specific_character_set = specific_character_set[0]
        else:
            # Have multiple values
            if specific_character_set[0] in multi_byte_code_extensions:
                raise ValueError(
                    "When SpecificCharacterSet is multi-valued, a multi-byte "
                    "encoding may not be used as the first value."
                )

            for enc in specific_character_set:
                if enc not in all_code_extensions:
                    raise ValueError(
                        "When SpecificCharacterSet is multi-valued, all "
                        "encodings must use code extensions."
                    )

            return [enc.value for enc in specific_character_set]

    # Single value
    if specific_character_set in all_code_extensions:
        raise ValueError(
            "Encodings with code extensions should not be used when only a "
            "single encoding is included."
        )

    return specific_character_set.value
