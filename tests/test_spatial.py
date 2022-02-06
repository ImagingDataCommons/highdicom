import pytest

from highdicom.spatial import (
    map_coordinate_into_pixel_matrix,
    map_pixel_into_coordinate_system,
)


params_pixel_to_physical = [
    # Slide
    pytest.param(
        dict(
            index=(0, 0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0.0, 0.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(0, 0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0.0, 0.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(0, 0),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (1.0, 1.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(0, 0),
            image_position=(1.0, 1.0, 1.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (1.0, 1.0, 1.0),
    ),
    pytest.param(
        dict(
            index=(1, 1),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (2.0, 2.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(1, 1),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 0.5),
        ),
        (10.5, 60.5, 0.0),
    ),
    pytest.param(
        dict(
            index=(5, 2),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (14.0, 62.5, 0.0),
    ),
    pytest.param(
        dict(
            index=(5, 2),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (6.0, 57.5, 0.0),
    ),
    pytest.param(
        dict(
            index=(2, 2),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (6.0, 59.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(2, 4),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (2.0, 59.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(5, 2),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (12.5, 56.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(5, 4),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (12.5, 52.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(5, 4),
            image_position=(10.0, 60.0, 30.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (12.5, 52.0, 30.0),
    ),
    # Patient
    pytest.param(
        dict(
            index=(5, 4),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (12.5, 68.0, 0.0),
    ),
    pytest.param(
        dict(
            index=(15, 87),
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
        ),
        (-45.0, 42.5, -111.75),
    ),
]


params_physical_to_pixel = [
    # Slide
    pytest.param(
        dict(
            coordinate=(0.0, 0.0, 0.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0, 0, 0),
    ),
    pytest.param(
        dict(
            coordinate=(0.0, 0.0, 0.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0, 0, 0),
    ),
    pytest.param(
        dict(
            coordinate=(1.0, 1.0, 0.0),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0, 0, 0),
    ),
    pytest.param(
        dict(
            coordinate=(1.0, 1.0, 1.0),
            image_position=(1.0, 1.0, 1.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0, 0, 0),
    ),
    pytest.param(
        dict(
            coordinate=(2.0, 2.0, 0.0),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (1, 1, 0),
    ),
    pytest.param(
        dict(
            coordinate=(10.5, 60.5, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 0.5),
        ),
        (1, 1, 0),
    ),
    pytest.param(
        dict(
            coordinate=(14.0, 62.5, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (5, 2, 0),
    ),
    pytest.param(
        dict(
            coordinate=(6.0, 57.5, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (5, 2, 0),
    ),
    pytest.param(
        dict(
            coordinate=(6.0, 59.0, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (2, 2, 0),
    ),
    pytest.param(
        dict(
            coordinate=(2.0, 59.0, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (2, 4, 0),
    ),
    pytest.param(
        dict(
            coordinate=(12.5, 56.0, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (5, 2, 0),
    ),
    pytest.param(
        dict(
            coordinate=(12.5, 52.0, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (5, 4, 0),
    ),
    pytest.param(
        dict(
            coordinate=(12.5, 52.0, 30.0),
            image_position=(10.0, 60.0, 30.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (5, 4, 0),
    ),
    # Patient
    pytest.param(
        dict(
            coordinate=(12.5, 68.0, 0.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (5, 4, 0),
    ),
    pytest.param(
        dict(
            coordinate=(89.0, 47.0, -10.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        (158, -6, -10),
    ),
    pytest.param(
        dict(
            coordinate=(-45.0, 42.5, -111.75),
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
        ),
        (15, 87, 0),
    ),
    pytest.param(
        dict(
            coordinate=(-35.0, 32.5, -60.5),
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
        ),
        (-5, -118, -10),
    ),
    pytest.param(
        dict(
            coordinate=(-35.0, 32.5, -60.5),
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
            spacing_between_slices=0.25
        ),
        (-5, -118, -40),
    ),
]


@pytest.mark.parametrize('inputs,expected_output', params_pixel_to_physical)
def test_map_pixel_into_coordinate_system(inputs, expected_output):
    output = map_pixel_into_coordinate_system(**inputs)
    assert output == expected_output


@pytest.mark.parametrize('inputs,expected_output', params_physical_to_pixel)
def test_map_coordinate_into_pixel_matrix(inputs, expected_output):
    output = map_coordinate_into_pixel_matrix(**inputs)
    assert output == expected_output
