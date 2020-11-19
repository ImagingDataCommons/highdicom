import pytest

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.utils import (
    map_pixel_into_coordinate_system,
    compute_plane_position_tiled_full,
)


params_1 = [
    # Slide
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0.0, 0.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (0.0, 0.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (1.0, 1.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(0.0, 0.0),
            image_position=(1.0, 1.0, 1.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (1.0, 1.0, 1.0),
    ),
    pytest.param(
        dict(
            coordinate=(1.0, 1.0),
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        (2.0, 2.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(1.0, 1.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 0.5),
        ),
        (10.5, 60.5, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (14.0, 62.5, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (6.0, 57.5, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(2.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (6.0, 59.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(2.0, 4.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (2.0, 59.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 2.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (12.5, 56.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 4.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (12.5, 52.0, 0.0),
    ),
    pytest.param(
        dict(
            coordinate=(5.0, 4.0),
            image_position=(10.0, 60.0, 30.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (12.5, 52.0, 30.0),
    ),
    # Patient
    pytest.param(
        dict(
            coordinate=(5.0, 4.0),
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(0.5, 2.0),
        ),
        (12.5, 68.0, 0.0),
    ),
]


params_2 = [
    pytest.param(
        dict(
            row_index=1,
            column_index=1,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(1.0, 1.0, 0.0),
            pixel_matrix_position=(1, 1)
        ),
    ),
    pytest.param(
        dict(
            row_index=2,
            column_index=2,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(17.0, 9.0, 0.0),
            pixel_matrix_position=(9, 17)
        ),
    ),
    pytest.param(
        dict(
            row_index=4,
            column_index=1,
            x_offset=10.0,
            y_offset=20.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(59.0, 21.0, 0.0),
            pixel_matrix_position=(1, 49)
        ),
    ),
    pytest.param(
        dict(
            row_index=4,
            column_index=1,
            x_offset=10.0,
            y_offset=60.0,
            rows=16,
            columns=8,
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(11.0, 11.0, 0.0),
            pixel_matrix_position=(1, 49)
        ),
    ),
]


@pytest.mark.parametrize('inputs,expected_output', params_1)
def test_map_pixel_into_coordinate_system(inputs, expected_output):
    output = map_pixel_into_coordinate_system(**inputs)
    assert output == expected_output


@pytest.mark.parametrize('inputs,expected_output', params_2)
def test_compute_plane_position_tiled_full(inputs, expected_output):
    output = compute_plane_position_tiled_full(**inputs)
    assert output == expected_output
