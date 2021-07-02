import pytest

from pydicom.valuerep import PersonName
from highdicom.valuerep import check_person_name


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


@pytest.mark.parametrize('name', test_names)
def test_valid_strings(name):
    check_person_name(name)


@pytest.mark.parametrize('name', test_names)
def test_valid_person_names(name):
    check_person_name(PersonName(name))


@pytest.mark.parametrize('name', invalid_test_names)
def test_invalid_strings(name):
    with pytest.raises(ValueError):
        check_person_name(name)


@pytest.mark.parametrize('name', invalid_test_names)
def test_invalid_person_names(name):
    with pytest.raises(ValueError):
        check_person_name(PersonName(name))
