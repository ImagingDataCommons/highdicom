import pytest

from highdicom import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.utils import compute_plane_position_tiled_full


params_plane_positions = [
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
            slice_index=2,
            spacing_between_slices=1.0
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(11.0, 11.0, 1.0),
            pixel_matrix_position=(1, 49)
        ),
    ),
]


@pytest.mark.parametrize('inputs,expected_output', params_plane_positions)
def test_compute_plane_position_tiled_full(inputs, expected_output):
    output = compute_plane_position_tiled_full(**inputs)
    assert output == expected_output


def test_should_raise_error_when_3d_param_is_missing():
    with pytest.raises(TypeError):
        compute_plane_position_tiled_full(
            row_index=1,
            column_index=1,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            slice_index=1
        )
    with pytest.raises(TypeError):
        compute_plane_position_tiled_full(
            row_index=1,
            column_index=1,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            spacing_between_slices=1.0
        )
