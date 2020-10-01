import pytest

from highdicom.utils import map_pixel_into_coordinate_system


mappings = [
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            spacing_between_slices=0.0,
        ),
        (0.0, 0.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            spacing_between_slices=1.0,
        ),
        (0.0, 0.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            spacing_between_slices=0.0,
        ),
        (1.0, 1.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(1.0, 1.0, 1.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            spacing_between_slices=0.0,
        ),
        (1.0, 1.0, 1.0),
    ),
    pytest.param(
        dict(
            coordinate=(1.0, 1.0),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            spacing_between_slices=0.0,
        ),
        (2.0, 2.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(1.0, 1.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 0.5),
            spacing_between_slices=0.0,
        ),
        (10.5, 60.5, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
            spacing_between_slices=0.0,
        ),
        (20.0, 61.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
            spacing_between_slices=0.0,
        ),
        (0.0, 59.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(2.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
            spacing_between_slices=0.0,
        ),
        (6.0, 59.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(2.0, 4.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
            spacing_between_slices=0.0,
        ),
        (6.0, 58.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(0.5, 2.0),
            spacing_between_slices=0.0,
        ),
        (11.0, 50.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 4.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(0.5, 2.0),
            spacing_between_slices=0.0,
        ),
        (12.0, 50.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 4.0),
            image_position=(10.0, 60.0, 30.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(0.5, 2.0),
            spacing_between_slices=0.0,
        ),
        (12.0, 50.0, 30.0),
    ),
]


@pytest.mark.parametrize('inputs,expected_output', mappings)
def test_map_pixel_into_coordinate_system(inputs, expected_output):
    output = map_pixel_into_coordinate_system(**inputs)
    assert output == expected_output
