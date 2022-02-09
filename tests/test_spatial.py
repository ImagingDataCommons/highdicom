import numpy as np
import pytest

from highdicom.spatial import (
    ImageToReferenceTransformer,
    PixelToReferenceTransformer,
    ReferenceToImageTransformer,
    ReferenceToPixelTransformer,
)


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
            (158, -6, -10),
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


@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_pixel_to_reference
)
def test_map_pixel_into_coordinate_system(params, inputs, expected_outputs):
    transform = PixelToReferenceTransformer(**params)
    outputs = transform(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)


@pytest.mark.parametrize(
    'params,inputs,expected_outputs',
    params_reference_to_pixel
)
def test_map_coordinate_into_pixel_matrix(params, inputs, expected_outputs):
    transform = ReferenceToPixelTransformer(**params)
    outputs = transform(inputs)
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
    'params,inputs,expected_outputs',
    params_reference_to_image
)
def test_map_reference_to_image_coordinate(params, inputs, expected_outputs):
    transform = ReferenceToImageTransformer(**params)
    outputs = transform(inputs)
    np.testing.assert_array_almost_equal(outputs, expected_outputs)
