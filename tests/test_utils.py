from pathlib import Path
import math
import itertools

from pydicom import dcmread
from pydicom.dataset import Dataset

import pytest

from highdicom import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.utils import compute_plane_position_tiled_full, is_tiled_image


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
            image_position=(0.0, 0.0, 0.0),
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
            image_position=(16.0, 8.0, 0.0),
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
            image_position=(58.0, 20.0, 0.0),
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
            image_position=(10.0, 12.0, 0.0),
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
            image_position=(10.0, 12.0, 1.0),
            pixel_matrix_position=(1, 49)
        ),
    ),
]


@pytest.mark.parametrize('inputs,expected_output', params_plane_positions)
def test_compute_plane_position_tiled_full(inputs, expected_output):
    output = compute_plane_position_tiled_full(**inputs)
    assert output == expected_output


def test_compute_plane_position_tiled_full_with_missing_parameters():
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


@pytest.mark.parametrize(
    'filepath,expected_output',
    [
        (
            Path(__file__).parents[1].joinpath('data/test_files/ct_image.dcm'),
            False
        ),
        (
            Path(__file__).parents[1].joinpath('data/test_files/sm_image.dcm'),
            True
        ),
        (
            Path(__file__).parents[1].joinpath(
                'data/test_files/seg_image_ct_binary.dcm'
            ),
            False
        ),
        (
            Path(__file__).parents[1].joinpath(
                'data/test_files/seg_image_sm_control.dcm'
            ),
            True
        ),
    ]
)
def test_is_tiled_image(filepath, expected_output):
    dcm = dcmread(filepath)
    assert is_tiled_image(dcm) == expected_output


def compute_plane_position_slide_per_frame():
    iterator = itertools.product(range(1, 4), range(1, 3))
    for num_optical_paths, num_focal_planes in iterator:
        image = Dataset()
        image.Rows = 4
        image.Columns = 4
        image.TotalPixelMatrixRows = 16
        image.TotalPixelMatrixColumns = 16
        image.TotalPixelMatrixFocalPlanes = num_focal_planes
        image.NumberOfOpticalPaths = num_optical_paths
        image.ImageOrientationSlide = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        shared_fg_item = Dataset()
        pixel_measures_item = Dataset()
        pixel_measures_item.PixelSpacing = [1.0, 1.0]
        pixel_measures_item.SliceThickness = 1.0
        pixel_measures_item.SpacingBetweenSlices = 1.0
        shared_fg_item.PixelMeasuresSequence = [pixel_measures_item]
        image.SharedFunctionalGroupsSequence = [shared_fg_item]

        plane_positions = compute_plane_position_tiled_full(image)

        tiles_per_column = math.ceil(image.TotalPixelMatrixRows / image.Rows)
        tiles_per_row = math.ceil(image.TotalPixelMatrixColumns / image.Columns)
        assert len(plane_positions) == math.prod([
            num_optical_paths,
            num_focal_planes,
            tiles_per_row,
            tiles_per_column
        ])
