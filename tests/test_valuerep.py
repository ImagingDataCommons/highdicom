import pytest

from highdicom.valuerep import get_ds_string


@pytest.mark.parametrize(
    "float_val,expected_str",
    [
        [1.0, "1.0"],
        [0.0, "0.0"],
        [-0.0, "-0.0"],
        [0.123, "0.123"],
        [-0.321, "-0.321"],
        [0.00001, "1e-05"],
        [3.14159265358979323846, '3.14159265358979'],
        [-3.14159265358979323846, '-3.1415926535898'],
        [5.3859401928763739403e-7, '5.3859401929e-07'],
        [-5.3859401928763739403e-7, '-5.385940193e-07'],
        [1.2342534378125532912998323e10, '12342534378.1255'],
        [6.40708699858767842501238e13, '6.4070869986e+13'],
        [1.7976931348623157e+308, '1.797693135e+308'],
    ]
)
def test_get_ds_string(float_val: float, expected_str: str):
    returned_str = get_ds_string(float_val)
    assert len(returned_str) <= 16
    assert expected_str == returned_str


@pytest.mark.parametrize(
    'float_val',
    [float('nan'), float('inf'), float('-nan'), float('-inf')]
)
def test_get_ds_string_invalid(float_val: float):
    with pytest.raises(ValueError):
        get_ds_string(float_val)
