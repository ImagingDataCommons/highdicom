import pytest

from pydicom.valuerep import PersonName
from highdicom.valuerep import check_person_name, _check_code_string


test_names = [
    'Doe^John',
    'Doe^John^^Mr',
    'Doe^',
    '山田^太郎',
    '洪^吉洞'
]

invalid_test_names = [
    'John Doe',
    'Mr John Doe',
    'Doe',
    '山田太郎',
    '洪吉洞'
]

test_code_strings = [
    'FOO_BAR',
    'FOOBAR',
    'FOO01',
    'FOO_BAR_33',
    'ABCDEFGHIJKLMNOP',
]

invalid_code_strings = [
    'foo_bar',
    'FooBar',
    'FOO-01',
    ' FOO',
    '_FOO',
    'BAR_',
    'BAR ',
    '-FOO',
    '1FOO',
    'ABCDEFGHIJKLMNOPQ',
]


@pytest.mark.parametrize('name', test_names)
def test_valid_person_name_strings(name):
    check_person_name(name)


@pytest.mark.parametrize('name', test_names)
def test_valid_person_name_objects(name):
    check_person_name(PersonName(name))


@pytest.mark.parametrize('name', invalid_test_names)
def test_invalid_person_name_strings(name):
    with pytest.warns(UserWarning):
        check_person_name(name)


@pytest.mark.parametrize('name', invalid_test_names)
def test_invalid_person_name_objects(name):
    with pytest.warns(UserWarning):
        check_person_name(PersonName(name))


@pytest.mark.parametrize('code_string', test_code_strings)
def test_valid_code_strings(code_string):
    _check_code_string(code_string)


@pytest.mark.parametrize('code_string', invalid_code_strings)
def test_invalid_code_strings(code_string):
    with pytest.raises(ValueError):
        _check_code_string(code_string)
