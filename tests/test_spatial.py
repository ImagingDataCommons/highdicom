from pathlib import Path
import numpy as np
import pydicom
from pydicom.data import get_testdata_file, get_testdata_files
import pytest

from highdicom.spatial import (
    ImageToImageTransformer,
    ImageToReferenceTransformer,
    PixelToPixelTransformer,
    PixelToReferenceTransformer,
    ReferenceToImageTransformer,
    ReferenceToPixelTransformer,
    _are_images_coplanar,
    _normalize_patient_orientation,
    _transform_affine_matrix,
    create_rotation_matrix,
    get_closest_patient_orientation,
    get_series_volume_positions,
    get_volume_positions,
    is_tiled_image,
    rotation_for_patient_orientation,
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
    dcm = pydicom.dcmread(filepath)
    assert is_tiled_image(dcm) == expected_output


params_pixel_to_reference = [
    # Slide
    pytest.param(
        dict(
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        np.array([
            (0, 0),
            (1, 1),
        ]),
        np.array([
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        np.array([
            (0, 0),
        ]),
        np.array([
            (1.0, 1.0, 0.0),
        ]),
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 0.5),
        ),
        np.array([
            (1, 1),
        ]),
        np.array([
            (10.5, 60.5, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (5, 2),
        ]),
        np.array([
            (14.0, 62.5, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (5, 2),
            (2, 2),
            (2, 4),
        ]),
        np.array([
            (6.0, 57.5, 0.0),
            (6.0, 59.0, 0.0),
            (2.0, 59.0, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (5, 2),
            (5, 4),
        ]),
        np.array([
            (12.5, 56.0, 0.0),
            (12.5, 52.0, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 30.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (5, 4),
        ]),
        np.array([
            (12.5, 52.0, 30.0),
        ])
    ),
    # Patient
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (5, 4),
        ]),
        np.array([
            (12.5, 68.0, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
        ),
        np.array([
            (15, 87),
        ]),
        np.array([
            (-45.0, 42.5, -111.75),
        ])
    ),
]


params_reference_to_pixel = [
    # Slide
    pytest.param(
        dict(
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        np.array([
            (0.0, 0.0, 0.0),
        ]),
        np.array([
            (0, 0, 0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(1.0, 1.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        np.array([
            (1.0, 1.0, 0.0),
            (2.0, 2.0, 0.0),
        ]),
        np.array([
            (0, 0, 0),
            (1, 1, 0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(0.5, 0.5),
        ),
        np.array([
            (10.5, 60.5, 0.0),
        ]),
        np.array([
            (1, 1, 0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (14.0, 62.5, 0.0),
        ]),
        np.array([
            (5, 2, 0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (6.0, 57.5, 0.0),
            (6.0, 59.0, 0.0),
            (2.0, 59.0, 0.0),
        ]),
        np.array([
            (5, 2, 0),
            (2, 2, 0),
            (2, 4, 0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (12.5, 52.0, 0.0),
            (12.5, 56.0, 0.0),
        ]),
        np.array([
            (5, 4, 0),
            (5, 2, 0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 30.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (12.5, 52.0, 30.0),
        ]),
        np.array([
            (5, 4, 0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (89.0, 47.0, -10.0),
        ]),
        np.array([
            (158, -6.5, -10),
        ])
    ),
    # Patient
    pytest.param(
        dict(
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
        ),
        np.array([
            (-45.0, 42.5, -111.75),
            (-35.0, 32.5, -60.5),
        ]),
        np.array([
            (15, 87, 0),
            (-5, -118, -10),
        ])
    ),
    pytest.param(
        dict(
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
            spacing_between_slices=0.25
        ),
        np.array([
            (-35.0, 32.5, -60.5),
        ]),
        np.array([
            (-5, -118, -40),
        ])
    ),
]


params_image_to_reference = [
    pytest.param(
        dict(
            image_position=(56.0, 34.2, 1.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(0.5, 0.5)
        ),
        np.array([
            (0.0, 10.0),
            (5.0, 5.0),
        ]),
        np.array([
            (55.75, 38.95, 1.0),
            (58.25, 36.45, 1.0),
        ]),
    ),
    pytest.param(
        dict(
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        np.array([
            (0.0, 0.0),
            (1.0, 1.0),
        ]),
        np.array([
            (-0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
        ]),
    ),
    pytest.param(
        dict(
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
        ),
        np.array([
            (15.0, 87.0),
            (30.0, 87.0),
        ]),
        np.array([
            (-45.0, 42.25, -111.625),
            (-45.0, 49.75, -111.625),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (5.0, 4.0),
        ]),
        np.array([
            (12.25, 67.0, 0.0),
        ])
    ),
]


params_reference_to_image = [
    pytest.param(
        dict(
            image_position=(56.0, 34.2, 1.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(0.5, 0.5)
        ),
        np.array([
            (56.0, 39.2, 1.0),
            (58.5, 36.7, 1.0),
        ]),
        np.array([
            (0.5, 10.5, 0.0),
            (5.5, 5.5, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        np.array([
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 3.0, 0.0),
        ]),
        np.array([
            (0.5, 0.5, 0.0),
            (1.5, 1.5, 0.0),
            (3.5, 2.5, 0.0),
        ]),
    ),
    pytest.param(
        dict(
            image_position=(-45.0, 35.0, -90.0),
            image_orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            pixel_spacing=(0.25, 0.5),
        ),
        np.array([
            (-45.0, 42.25, -111.625),
            (-45.0, 49.75, -111.625),
        ]),
        np.array([
            (15.0, 87.0, 0.0),
            (30.0, 87.0, 0.0),
        ])
    ),
    pytest.param(
        dict(
            image_position=(10.0, 60.0, 0.0),
            image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing=(2.0, 0.5),
        ),
        np.array([
            (12.25, 67.0, 0.0),
        ]),
        np.array([
            (5.0, 4.0, 0.0),
        ])
    ),
]


params_pixel_to_pixel = [
    # Pure translation
    pytest.param(
        dict(
            image_position_from=(56.0, 34.2, 1.0),
            image_orientation_from=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_from=(1.0, 1.0),
            image_position_to=(66.0, 32.2, 1.0),
            image_orientation_to=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_to=(1.0, 1.0),
        ),
        np.array([
            (0, 0),
            (50, 50),
        ]),
        np.array([
            (-10, 2),
            (40, 52),
        ])
    ),
    # Two images with different spacings (e.g. different pyramid levels)
    pytest.param(
        dict(
            image_position_from=(56.0, 34.2, 1.0),
            image_orientation_from=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_from=(1.0, 1.0),
            image_position_to=(56.0, 34.2, 1.0),
            image_orientation_to=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_to=(0.5, 0.5),
        ),
        np.array([
            (0, 0),
            (50, 50),
        ]),
        np.array([
            (0, 0),
            (100, 100),
        ])
    ),
    # 30 degree rotation, anisotropic scale
    pytest.param(
        dict(
            image_position_from=(56.0, 34.2, 1.0),
            image_orientation_from=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_from=(1.0, 1.0),
            image_position_to=(56.0, 34.2, 1.0),
            image_orientation_to=(
                np.cos(np.radians(30)), -np.sin(np.radians(30)), 0.0,
                np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0,
            ),
            pixel_spacing_to=(0.25, 0.5),
        ),
        np.array([
            (0, 0),
            (50, 50),
        ]),
        np.array([
            (0, 0),
            (
                2.0 * (
                    50 * np.cos(np.radians(30)) -
                    50 * np.sin(np.radians(30))
                ),
                4.0 * (
                    50 * np.sin(np.radians(30)) +
                    50 * np.cos(np.radians(30))
                ),
            ),
        ])
    ),
]


params_image_to_image = [
    # Pure translation
    pytest.param(
        dict(
            image_position_from=(56.0, 34.2, 1.0),
            image_orientation_from=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_from=(1.0, 1.0),
            image_position_to=(66.0, 32.2, 1.0),
            image_orientation_to=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_to=(1.0, 1.0),
        ),
        np.array([
            (0.0, 0.0),
            (50.0, 50.0),
        ]),
        np.array([
            (-10.0, 2.0),
            (40.0, 52.0),
        ])
    ),
    # Two images with different spacings (e.g. different pyramid levels)
    pytest.param(
        dict(
            image_position_from=(56.0, 34.2, 1.0),
            image_orientation_from=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_from=(1.0, 1.0),
            image_position_to=(56.0, 34.2, 1.0),
            image_orientation_to=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_to=(0.5, 0.5),
        ),
        np.array([
            (0.5, 0.5),
            (25, 50),
        ]),
        np.array([  # y = 2x - 0.5
            (0.5, 0.5),
            (49.5, 99.5),
        ])
    ),
    # 30 degree rotation, anisotropic scale
    pytest.param(
        dict(
            image_position_from=(56.0, 34.2, 1.0),
            image_orientation_from=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_from=(1.0, 1.0),
            image_position_to=(56.0, 34.2, 1.0),
            image_orientation_to=(
                np.cos(np.radians(30)), -np.sin(np.radians(30)), 0.0,
                np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0,
            ),
            pixel_spacing_to=(0.25, 0.5),
        ),
        np.array([
            (0, 0),
            (50, 50),
        ]),
        np.array([
            (0.133974596, -2.23205081),
            (36.7365150, 270.973030),
        ])
    ),
]


@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_pixel_to_reference
)
def test_map_pixel_into_coordinate_system(params, inputs, expected_outputs):
    transform = PixelToReferenceTransformer(**params)
    outputs = transform(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)


@pytest.mark.parametrize(
    'round_output',
    [False, True],
)
@pytest.mark.parametrize(
    'drop_slice_index',
    [False, True],
)
@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_reference_to_pixel
)
def test_map_coordinate_into_pixel_matrix(
    params,
    inputs,
    expected_outputs,
    round_output,
    drop_slice_index,
):
    transform = ReferenceToPixelTransformer(
        round_output=round_output,
        drop_slice_index=drop_slice_index,
        **params,
    )
    if round_output:
        expected_outputs = np.around(expected_outputs)
        expected_dtype = np.int64
    else:
        expected_dtype = np.float64
    if drop_slice_index:
        if np.abs(expected_outputs[:, 2]).max() >= 0.5:
            # In these cases, the transform should fail
            # because the dropped slice index is no close
            # to zero
            with pytest.raises(RuntimeError):
                transform(inputs)
            return
        expected_outputs = expected_outputs[:, :2]
    outputs = transform(inputs)
    assert outputs.dtype == expected_dtype
    np.testing.assert_array_almost_equal(outputs, expected_outputs)


@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_image_to_reference
)
def test_map_image_to_reference_coordinate(params, inputs, expected_outputs):
    transform = ImageToReferenceTransformer(**params)
    outputs = transform(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)


@pytest.mark.parametrize(
    'drop_slice_coord',
    [False, True],
)
@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_reference_to_image
)
def test_map_reference_to_image_coordinate(
    params,
    inputs,
    expected_outputs,
    drop_slice_coord,
):
    transform = ReferenceToImageTransformer(
        drop_slice_coord=drop_slice_coord,
        **params,
    )
    if drop_slice_coord:
        if np.abs(expected_outputs[:, 2]).max() >= 0.5:
            # In these cases, the transform should fail
            # because the dropped slice index is no close
            # to zero
            with pytest.raises(RuntimeError):
                transform(inputs)
            return
        expected_outputs = expected_outputs[:, :2]
    outputs = transform(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)


@pytest.mark.parametrize(
    'round_output',
    [False, True],
)
@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_pixel_to_pixel
)
def test_map_indices_between_images(
    params,
    inputs,
    expected_outputs,
    round_output,
):
    transform = PixelToPixelTransformer(
        round_output=round_output,
        **params,
    )
    outputs = transform(inputs)
    if round_output:
        expected_outputs = np.around(expected_outputs)
        assert outputs.dtype == np.int64
    else:
        assert outputs.dtype == np.float64

    np.testing.assert_array_almost_equal(outputs, expected_outputs)


@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_image_to_image
)
def test_map_coordinates_between_images(params, inputs, expected_outputs):
    transform = ImageToImageTransformer(**params)
    outputs = transform(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)


@pytest.mark.parametrize(
    'image_orientation,orientation_str',
    [
        ([1, 0, 0, 0, 1, 0], 'LPH'),
        ([0, 1, 0, 1, 0, 0], 'PLF'),
        ([-1, 0, 0, 0, 1, 0], 'RPF'),
        ([0, 0, -1, 1, 0, 0], 'FLA'),
        (
            [
                np.cos(np.pi / 4),
                -np.sin(np.pi / 4),
                0,
                np.sin(np.pi / 4),
                np.cos(np.pi / 4),
                0
            ],
            'LPH'
        ),
    ]
)
def test_get_closest_patient_orientation(
    image_orientation,
    orientation_str,
):
    codes = _normalize_patient_orientation(orientation_str)
    rotation_matrix = create_rotation_matrix(image_orientation)
    assert get_closest_patient_orientation(
        rotation_matrix
    ) == codes


@pytest.mark.parametrize(
    'orientation_str',
    ['LPH', 'PLF', 'RPF', 'FLA']
)
def test_rotation_from_patient_orientation(
    orientation_str,
):
    codes = _normalize_patient_orientation(orientation_str)
    rotation_matrix = rotation_for_patient_orientation(
        orientation_str
    )
    assert get_closest_patient_orientation(
        rotation_matrix
    ) == codes


def test_rotation_from_patient_orientation_spacing():
    rotation_matrix = rotation_for_patient_orientation(
        ['F', 'P', 'L'],
        spacing=(1.0, 2.0, 2.5)
    )
    expected = np.array(
        [
            [0.0, 0.0, 2.5],
            [0.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(
        rotation_matrix,
        expected,
    )


all_single_image_transformer_classes = [
    ImageToReferenceTransformer,
    PixelToReferenceTransformer,
    ReferenceToPixelTransformer,
    ReferenceToImageTransformer,
]


all_image_pair_transformer_classes = [
    ImageToImageTransformer,
    PixelToPixelTransformer,
]


@pytest.mark.parametrize(
    'transformer_cls',
    all_single_image_transformer_classes
)
def test_transformers_ct_image(transformer_cls):
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data/test_files')
    dcm = pydicom.dcmread(data_dir / 'ct_image.dcm')
    transformer_cls.for_image(dcm)


@pytest.mark.parametrize(
    'transformer_cls',
    all_single_image_transformer_classes
)
def test_transformers_sm_image(transformer_cls):
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data/test_files')
    dcm = pydicom.dcmread(data_dir / 'sm_image.dcm')
    transformer_cls.for_image(dcm, frame_number=3)


@pytest.mark.parametrize(
    'transformer_cls',
    all_single_image_transformer_classes
)
def test_transformers_sm_image_total_pixel_matrix(transformer_cls):
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data/test_files')
    dcm = pydicom.dcmread(data_dir / 'sm_image.dcm')
    transformer_cls.for_image(dcm, for_total_pixel_matrix=True)


@pytest.mark.parametrize(
    'transformer_cls',
    all_single_image_transformer_classes
)
def test_transformers_seg_sm_image_total_pixel_matrix(transformer_cls):
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data/test_files')
    dcm = pydicom.dcmread(data_dir / 'seg_image_sm_dots.dcm')
    transformer_cls.for_image(dcm, for_total_pixel_matrix=True)


@pytest.mark.parametrize(
    'transformer_cls',
    all_single_image_transformer_classes
)
def test_transformers_enhanced_ct_image(transformer_cls):
    dcm = pydicom.dcmread(get_testdata_file('eCT_Supplemental.dcm'))
    transformer_cls.for_image(dcm, frame_number=2)


@pytest.mark.parametrize(
    'transformer_cls',
    all_image_pair_transformer_classes
)
def test_pair_transformers_sm_image_frame_to_frame(transformer_cls):
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data/test_files')
    dcm = pydicom.dcmread(data_dir / 'sm_image.dcm')
    transformer_cls.for_images(
        dataset_from=dcm,
        dataset_to=dcm,
        frame_number_from=1,
        frame_number_to=3,
    )


@pytest.mark.parametrize(
    'transformer_cls',
    all_image_pair_transformer_classes
)
def test_pair_transformers_sm_image_tpm_to_frame(transformer_cls):
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data/test_files')
    dcm = pydicom.dcmread(data_dir / 'sm_image.dcm')
    transformer_cls.for_images(
        dataset_from=dcm,
        dataset_to=dcm,
        for_total_pixel_matrix_from=True,
        frame_number_to=3,
    )


@pytest.mark.parametrize(
    'pos_a,ori_a,pos_b,ori_b,result',
    [
        pytest.param(
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            True,
        ),
        pytest.param(
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # flipped
            True,
        ),
        pytest.param(
            [211.0, 11.0, 1.0],  # shifted in plane
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            True,
        ),
        pytest.param(
            [1.0, 1.0, 11.0],  # shifted out of plane
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            False,
        ),
        pytest.param(
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # different orientation
            False,
        ),
    ]
)
def test_are_images_coplanar(pos_a, ori_a, pos_b, ori_b, result):
    assert _are_images_coplanar(
        image_position_a=pos_a,
        image_orientation_a=ori_a,
        image_position_b=pos_b,
        image_orientation_b=ori_b,
    ) == result


def test_get_series_slice_spacing_irregular():
    # A series of single frame CT images
    ct_series = [
        pydicom.dcmread(f)
        for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
    ]
    spacing, _ = get_series_volume_positions(ct_series)
    assert spacing is None


def test_get_series_slice_spacing_regular():
    # Use a subset of this test series that does have regular spacing
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136'),
        get_testdata_file('dicomdirtests/77654033/CT2/17196'),
        get_testdata_file('dicomdirtests/77654033/CT2/17166'),
    ]
    ct_series = [pydicom.dcmread(f) for f in ct_files]
    spacing, _ = get_series_volume_positions(ct_series)
    assert spacing == 1.25


def test_get_spacing_duplicates():
    # Test ability to determine spacing and volume positions with duplicate
    # positions
    position_indices = np.array(
        [0, 1, 2, 3, 4, 5, 2, 5, 5, 3, 1, 1, 2, 4, 1, 2, 0]
    )
    expected_spacing = 0.2
    positions = [
        [0.0, 0.0, i * expected_spacing] for i in position_indices
    ]
    orientation = [1, 0, 0, 0, -1, 0]

    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_duplicate_positions=False,
    )
    assert spacing is None
    assert volume_positions is None

    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_duplicate_positions=True,
    )
    assert np.isclose(spacing, expected_spacing)
    assert volume_positions == position_indices.tolist()


def test_get_spacing_missing():
    # Test ability to determine spacing and volume positions with missing
    # slices
    position_indices = np.array(
        [1, 3, 0, 9],  # an incomplete list of indices from 0 to 9
    )
    expected_spacing = 0.125
    positions = [
        [0.0, 0.0, i * expected_spacing] for i in position_indices
    ]
    orientation = [1, 0, 0, 0, -1, 0]

    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_missing_positions=True
    )

    assert np.isclose(spacing, expected_spacing)
    assert volume_positions == position_indices.tolist()

    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_missing_positions=False
    )
    assert spacing is None
    assert volume_positions is None


def test_get_spacing_missing_duplicates():
    # Test ability to determine spacing and volume positions with missing
    # slices and duplicate positions
    position_indices = np.array(
        [1, 3, 0, 9, 3],
    )
    expected_spacing = 0.125
    positions = [
        [0.0, 0.0, i * expected_spacing] for i in position_indices
    ]
    orientation = [1, 0, 0, 0, -1, 0]

    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_missing_positions=True,
    )
    assert spacing is None
    assert volume_positions is None

    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_duplicate_positions=True,
    )
    assert spacing is None
    assert volume_positions is None

    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_missing_positions=True,
        allow_duplicate_positions=True,
    )
    assert np.isclose(spacing, expected_spacing)
    assert volume_positions == position_indices.tolist()


def test_get_spacing_missing_duplicates_non_consecutive():
    # Test ability to determine spacing and volume positions with missing
    # slices and duplicate positions, with no two positions from consecutive
    # slices
    position_indices = np.array([7, 3, 0, 9, 3])
    expected_spacing = 0.125
    positions = [
        [0.0, 0.0, i * expected_spacing] for i in position_indices
    ]
    orientation = [1, 0, 0, 0, -1, 0]

    # Without the spacing_hint, the positions do not appear to be a volume
    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_missing_positions=True,
        allow_duplicate_positions=True,
    )
    assert spacing is None
    assert volume_positions is None

    # With the hint, the positions should be correctly calculated
    spacing, volume_positions = get_volume_positions(
        positions,
        orientation,
        allow_missing_positions=True,
        allow_duplicate_positions=True,
        spacing_hint=expected_spacing,
    )
    assert np.isclose(spacing, expected_spacing)
    assert volume_positions == position_indices.tolist()


def test_get_spacing_coplanar():
    # Check that coplanar points are not considered a volume
    positions = [
        [0.0, 0.0, 10.0],
        [1.0, 1.0, 10.0],  # Coplanar position
        [0.0, 0.0, 11.0],
        [0.0, 0.0, 12.0],
    ]
    orientation = [1, 0, 0, 0, -1, 0]

    # Regardless of values of allow_missing_positions and
    # allow_duplicate_positions, this is not a volume
    for allow_duplicate_positions in [False, True]:
        for allow_missing_positions in [False, True]:
            spacing, volume_positions = get_volume_positions(
                positions,
                orientation,
                allow_missing_positions=allow_missing_positions,
                allow_duplicate_positions=allow_duplicate_positions,
            )
            assert spacing is None
            assert volume_positions is None


def test_transform_affine_matrix():
    affine = np.array(
        [
            [np.cos(np.radians(30)), -np.sin(np.radians(30)), 0.0, -34.0],
            [np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0, 45.2],
            [0.0, 0.0, 1.0, -1.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transformed = _transform_affine_matrix(
        affine,
        permute_indices=[1, 2, 0],
        shape=[10, 10, 10],
    )
    expected = np.array(
        [
            [-np.sin(np.radians(30)), 0.0, np.cos(np.radians(30)), -34.0],
            [np.cos(np.radians(30)), 0.0, np.sin(np.radians(30)), 45.2],
            [0.0, 1.0, 0.0, -1.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.array_equal(transformed, expected)

    transformed = _transform_affine_matrix(
        affine,
        permute_reference=[1, 2, 0],
        shape=[10, 10, 10],
    )
    expected = np.array(
        [
            [np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0, 45.2],
            [0.0, 0.0, 1.0, -1.2],
            [np.cos(np.radians(30)), -np.sin(np.radians(30)), 0.0, -34.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.array_equal(transformed, expected)

    transformed = _transform_affine_matrix(
        affine,
        flip_indices=[True, False, True],
        shape=[10, 10, 10],
    )
    expected = np.array(
        [
            [
                -np.cos(np.radians(30)),
                -np.sin(np.radians(30)),
                0.0,
                -26.20577137,
            ],
            [
                -np.sin(np.radians(30)),
                np.cos(np.radians(30)),
                0.0,
                40.7,
            ],
            [0.0, 0.0, -1.0, 7.8],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(transformed, expected)

    transformed = _transform_affine_matrix(
        affine,
        flip_reference=[True, False, True],
        shape=[10, 10, 10],
    )
    expected = np.array(
        [
            [-np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0, 34.0],
            [np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0, 45.2],
            [0.0, 0.0, -1.0, 1.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.array_equal(transformed, expected)
