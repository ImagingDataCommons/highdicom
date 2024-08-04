from pathlib import Path
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
import pytest


from highdicom.spatial import _normalize_patient_orientation
from highdicom.volume import Volume, concat_channels, volread
from highdicom import UID


def read_multiframe_ct_volume():
    dcm = pydicom.dcmread(get_testdata_file('eCT_Supplemental.dcm'))
    return Volume.from_image(dcm), dcm


def read_ct_series_volume():
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136'),
        get_testdata_file('dicomdirtests/77654033/CT2/17196'),
        get_testdata_file('dicomdirtests/77654033/CT2/17166'),
    ]
    ct_series = [pydicom.dcmread(f) for f in ct_files]
    return Volume.from_image_series(ct_series), ct_series


def test_transforms():
    array = np.zeros((25, 50, 50))
    volume = Volume.from_attributes(
        array=array,
        image_position=[0.0, 0.0, 0.0],
        image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        pixel_spacing=[1.0, 1.0],
        spacing_between_slices=10.0,
    )
    plane_positions = volume.get_plane_positions()
    for i, pos in enumerate(plane_positions):
        assert np.array_equal(pos[0].ImagePositionPatient, [0.0, 0.0, 10.0 * i])

    indices = np.array([[1, 2, 3]])
    coords = volume.map_indices_to_reference(indices)
    assert np.array_equal(coords, np.array([[3.0, 2.0, 10.0]]))
    round_trip = volume.map_reference_to_indices(coords)
    assert np.array_equal(round_trip, indices)
    index_center = volume.get_center_index()
    assert np.array_equal(index_center, [12.0, 24.5, 24.5])
    index_center = volume.get_center_index(round_output=True)
    assert np.array_equal(index_center, [12, 24, 24])
    coord_center = volume.get_center_coordinate()
    assert np.array_equal(coord_center, [24.5, 24.5, 120])


@pytest.mark.parametrize(
    'image_position,image_orientation,pixel_spacing,spacing_between_slices',
    [
        (
            (67.0, 32.4, -45.2),
            (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            (3.2, 1.6),
            1.25,
        ),
        (
            [67.0, 32.4, -45.2],
            (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            (3.2, 1.6),
            1.25,
        ),
        (
            (-67.0, 132.4, -5.2),
            (0.0, 0.0, -1.0, 1.0, 0.0, 0.0),
            (0.25, 0.25),
            3.5,
        ),
        (
            (-67.0, 132.4, -5.2),
            (
                np.cos(np.radians(30)), -np.sin(np.radians(30)), 0.0,
                np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0,
            ),
            (0.75, 0.25),
            3.5,
        ),
    ],
)
def test_volume_from_attributes(
    image_position,
    image_orientation,
    pixel_spacing,
    spacing_between_slices,
):
    array = np.zeros((10, 10, 10))
    volume = Volume.from_attributes(
        array=array,
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=spacing_between_slices,
    )
    assert volume.position == list(image_position)
    assert volume.direction_cosines == list(image_orientation)
    assert volume.pixel_spacing == list(pixel_spacing)
    assert volume.spacing_between_slices == spacing_between_slices
    assert volume.shape == (10, 10, 10)
    assert volume.spatial_shape == (10, 10, 10)
    assert volume.number_of_channels is None


def test_volume_with_channels():
    array = np.zeros((10, 10, 10, 2))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
    )
    assert volume.shape == (10, 10, 10, 2)
    assert volume.spatial_shape == (10, 10, 10)
    assert volume.number_of_channels == 2


def test_with_array():
    array = np.zeros((10, 10, 10))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
    )
    new_array = np.zeros((10, 10, 10, 2), dtype=np.uint8)
    new_volume = volume.with_array(new_array)
    assert new_volume.number_of_channels == 2
    assert isinstance(new_volume, Volume)
    assert volume.spatial_shape == new_volume.spatial_shape
    assert np.array_equal(volume.affine, new_volume.affine)
    assert volume.affine is not new_volume.affine
    assert new_volume.dtype == np.uint8

    concat_volume = concat_channels([volume, new_volume])
    assert isinstance(concat_volume, Volume)
    assert volume.spatial_shape == concat_volume.spatial_shape
    assert concat_volume.number_of_channels == 3


def test_volume_single_frame():
    volume, ct_series = read_ct_series_volume()
    assert isinstance(volume, Volume)
    rows, columns = ct_series[0].Rows, ct_series[0].Columns
    assert volume.shape == (len(ct_series), rows, columns)
    assert volume.spatial_shape == volume.shape
    assert volume.number_of_channels is None
    orientation = ct_series[0].ImageOrientationPatient
    assert volume.direction_cosines == orientation
    direction = volume.direction
    assert np.array_equal(direction[:, 1], orientation[3:])
    assert np.array_equal(direction[:, 2], orientation[:3])
    # Check third direction is normal to others
    assert direction[:, 0] @ direction[:, 1] == 0.0
    assert direction[:, 0] @ direction[:, 2] == 0.0
    assert (direction[:, 0] ** 2).sum() == 1.0
    assert volume.position == ct_series[0].ImagePositionPatient
    assert volume.pixel_spacing == ct_series[0].PixelSpacing
    slice_spacing = 1.25
    assert volume.spacing == [slice_spacing, *ct_series[0].PixelSpacing[::-1]]
    pixel_spacing = ct_series[0].PixelSpacing
    expected_voxel_volume = (
        pixel_spacing[0] * pixel_spacing[1] * slice_spacing
    )
    expected_volume = expected_voxel_volume * np.prod(volume.spatial_shape)
    assert np.allclose(volume.voxel_volume, expected_voxel_volume)
    assert np.allclose(volume.physical_volume, expected_volume)
    u1, u2, u3 = volume.unit_vectors()
    for u in [u1, u2, u3]:
        assert u.shape == (3, )
        assert np.linalg.norm(u) == 1.0
    assert np.allclose(u3, orientation[:3])
    assert np.allclose(u2, orientation[3:])

    v1, v2, v3 = volume.spacing_vectors()
    for v, spacing in zip([v1, v2, v3], volume.spacing):
        assert v.shape == (3, )
        assert np.linalg.norm(v) == spacing


def test_volume_multiframe():
    volume, dcm = read_multiframe_ct_volume()
    assert isinstance(volume, Volume)
    rows, columns = dcm.Rows, dcm.Columns
    assert volume.shape == (dcm.NumberOfFrames, rows, columns)
    assert volume.spatial_shape == volume.shape
    orientation = (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .PlaneOrientationSequence[0]
        .ImageOrientationPatient
    )
    pixel_spacing = (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )
    assert volume.direction_cosines == orientation
    direction = volume.direction
    assert np.array_equal(direction[:, 1], orientation[3:])
    assert np.array_equal(direction[:, 2], orientation[:3])
    # Check third direction is normal to others
    assert direction[:, 0] @ direction[:, 1] == 0.0
    assert direction[:, 0] @ direction[:, 2] == 0.0
    assert (direction[:, 0] ** 2).sum() == 1.0
    first_frame_pos = (
        dcm
        .PerFrameFunctionalGroupsSequence[1]  # due to ordering
        .PlanePositionSequence[0]
        .ImagePositionPatient
    )
    assert volume.position == first_frame_pos
    assert volume.pixel_spacing == pixel_spacing
    slice_spacing = 10.0
    assert volume.spacing == [slice_spacing, *pixel_spacing[::-1]]
    assert volume.number_of_channels is None
    expected_voxel_volume = (
        pixel_spacing[0] * pixel_spacing[1] * slice_spacing
    )
    expected_volume = expected_voxel_volume * np.prod(volume.spatial_shape)
    assert np.allclose(volume.voxel_volume, expected_voxel_volume)
    assert np.allclose(volume.physical_volume, expected_volume)
    u1, u2, u3 = volume.unit_vectors()
    for u in [u1, u2, u3]:
        assert u.shape == (3, )
        assert np.linalg.norm(u) == 1.0
    assert np.allclose(u3, orientation[:3])
    assert np.allclose(u2, orientation[3:])

    v1, v2, v3 = volume.spacing_vectors()
    for v, spacing in zip([v1, v2, v3], volume.spacing):
        assert v.shape == (3, )
        assert np.linalg.norm(v) == spacing


def test_indexing():
    array = np.random.randint(0, 100, (25, 50, 50))
    volume = Volume.from_attributes(
        array=array,
        image_position=[0.0, 0.0, 0.0],
        image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        pixel_spacing=[1.0, 1.0],
        spacing_between_slices=10.0,
    )

    # Single integer index
    subvolume = volume[3]
    assert subvolume.shape == (1, 50, 50)
    expected_affine = np.array([
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0],
        [10.0,  0.0,  0.0, 30.0],
        [ 0.0,  0.0,  0.0,  1.0],
    ])
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[3:4])

    # With colons
    subvolume = volume[3, :]
    assert subvolume.shape == (1, 50, 50)
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[3:4])
    subvolume = volume[3, :, :]
    assert subvolume.shape == (1, 50, 50)
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[3:4])

    # Single slice index
    subvolume = volume[3:13]
    assert subvolume.shape == (10, 50, 50)
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[3:13])

    # Multiple integer indices
    subvolume = volume[3, 7]
    assert subvolume.shape == (1, 1, 50)
    expected_affine = np.array([
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0,  7.0],
        [10.0,  0.0,  0.0, 30.0],
        [ 0.0,  0.0,  0.0,  1.0],
    ])
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[3:4, 7:8])

    # Multiple integer indices in sequence (should be the same as above)
    subvolume = volume[:, 7][3, :]
    assert subvolume.shape == (1, 1, 50)
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[3:4, 7:8])
    subvolume = volume[3, :][:, 7]
    assert subvolume.shape == (1, 1, 50)
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[3:4, 7:8])

    # Negative index
    subvolume = volume[-4]
    assert subvolume.shape == (1, 50, 50)
    expected_affine = np.array([
        [ 0.0,  0.0,  1.0,   0.0],
        [ 0.0,  1.0,  0.0,   0.0],
        [10.0,  0.0,  0.0, 210.0],
        [ 0.0,  0.0,  0.0,   1.0],
    ])
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[-4:-3])

    # Negative index range
    subvolume = volume[-4:-2, :, :]
    assert subvolume.shape == (2, 50, 50)
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[-4:-2])

    # Non-zero steps
    subvolume = volume[12:16:2, ::-1, :]
    assert subvolume.shape == (2, 50, 50)
    expected_affine = np.array([
        [ 0.0,  0.0,  1.0,   0.0],
        [ 0.0, -1.0,  0.0,  49.0],
        [20.0,  0.0,  0.0, 120.0],
        [ 0.0,  0.0,  0.0,   1.0],
    ])
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[12:16:2, ::-1])


def test_indexing_source_dimension_2():
    array = np.random.randint(0, 100, (50, 50, 25))
    affine = np.array([
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0],
        [10.0,  0.0,  0.0, 30.0],
        [ 0.0,  0.0,  0.0,  1.0],
    ])
    volume = Volume(
        array=array,
        affine=affine,
    )

    subvolume = volume[12:14, :, 12:6:-2]
    assert np.array_equal(subvolume.array, array[12:14, :, 12:6:-2])


def test_array_setter():
    array = np.random.randint(0, 100, (50, 50, 25))
    affine = np.array([
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0],
        [10.0,  0.0,  0.0, 30.0],
        [ 0.0,  0.0,  0.0,  1.0],
    ])

    volume = Volume(
        array=array,
        affine=affine,
    )

    new_array = np.random.randint(0, 100, (50, 50, 25))
    volume.array = new_array
    assert np.array_equal(volume.array, new_array)

    new_array = np.random.randint(0, 100, (25, 50, 50))
    with pytest.raises(ValueError):
        volume.array = new_array


@pytest.mark.parametrize(
    'desired',
    [
        'RAF',
        'RAH',
        'RPF',
        'RPH',
        'LAF',
        'LAH',
        'LPF',
        'LPH',
        'HLP',
        'FPR',
        'HRP',
    ]
)
def test_to_patient_orientation(desired):
    array = np.random.randint(0, 100, (25, 50, 50))
    volume = Volume.from_attributes(
        array=array,
        image_position=[0.0, 0.0, 0.0],
        image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        pixel_spacing=[1.0, 1.0],
        spacing_between_slices=10.0,
    )
    desired_tup = _normalize_patient_orientation(desired)

    flipped = volume.to_patient_orientation(desired)
    assert isinstance(flipped, Volume)
    assert flipped.get_closest_patient_orientation() == desired_tup

    flipped = volume.to_patient_orientation(desired_tup)
    assert isinstance(flipped, Volume)
    assert flipped.get_closest_patient_orientation() == desired_tup


@pytest.mark.parametrize(
    'fp,glob',
    [
        (Path(__file__).parent.parent.joinpath('data/test_files/ct_image.dcm'), None),
        (str(Path(__file__).parent.parent.joinpath('data/test_files/ct_image.dcm')), None),
        ([Path(__file__).parent.parent.joinpath('data/test_files/ct_image.dcm')], None),
        (get_testdata_file('eCT_Supplemental.dcm'), None),
        ([get_testdata_file('eCT_Supplemental.dcm')], None),
        (Path(__file__).parent.parent.joinpath('data/test_files/'), 'ct_image.dcm'),
        (str(Path(__file__).parent.parent.joinpath('data/test_files/')), 'ct_image.dcm'),
        (
            [
                get_testdata_file('dicomdirtests/77654033/CT2/17136'),
                get_testdata_file('dicomdirtests/77654033/CT2/17196'),
                get_testdata_file('dicomdirtests/77654033/CT2/17166'),
            ],
            None,
        ),
        (
            [
                Path(get_testdata_file('dicomdirtests/77654033/CT2/17136')),
                Path(get_testdata_file('dicomdirtests/77654033/CT2/17196')),
                Path(get_testdata_file('dicomdirtests/77654033/CT2/17166')),
            ],
            None,
        ),
    ]
)
def test_volread(fp, glob):
    volume = volread(fp, glob=glob)
    assert isinstance(volume, Volume)
