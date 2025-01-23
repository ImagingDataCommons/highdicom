import numpy as np
import pydicom
from pydicom.data import get_testdata_file
import pytest

from highdicom import (
    ChannelDescriptor,
    Volume,
    VolumeGeometry,
    VolumeToVolumeTransformer,
    imread,
    get_volume_from_series,
)
from highdicom.enum import PatientOrientationValuesBiped
from highdicom.spatial import (
    _normalize_patient_orientation,
    _translate_affine_matrix,
)


def read_multiframe_ct_volume():
    im = imread(get_testdata_file('eCT_Supplemental.dcm'))
    return im.get_volume(), im


def read_ct_series_volume():
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136'),
        get_testdata_file('dicomdirtests/77654033/CT2/17196'),
        get_testdata_file('dicomdirtests/77654033/CT2/17166'),
    ]
    ct_series = [pydicom.dcmread(f) for f in ct_files]
    return get_volume_from_series(ct_series), ct_series


def test_transforms():
    array = np.zeros((25, 50, 50))
    volume = Volume.from_attributes(
        array=array,
        image_position=[0.0, 0.0, 0.0],
        image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        pixel_spacing=[1.0, 1.0],
        spacing_between_slices=10.0,
        coordinate_system="PATIENT",
    )
    plane_positions = volume.get_plane_positions()
    for i, pos in enumerate(plane_positions):
        assert np.array_equal(
            pos[0].ImagePositionPatient,
            [0.0, 0.0, -10.0 * i]
        )

    indices = np.array([[1, 2, 3]])
    coords = volume.map_indices_to_reference(indices)
    assert np.array_equal(coords, np.array([[3.0, 2.0, -10.0]]))
    round_trip = volume.map_reference_to_indices(coords)
    assert np.array_equal(round_trip, indices)
    index_center = volume.center_indices
    assert np.array_equal(index_center, [12.0, 24.5, 24.5])
    index_center = volume.nearest_center_indices
    assert np.array_equal(index_center, [12, 24, 24])
    coord_center = volume.center_position
    assert np.array_equal(coord_center, [24.5, 24.5, -120])


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
        coordinate_system="PATIENT",
    )
    assert volume.position == tuple(image_position)
    assert volume.direction_cosines == tuple(image_orientation)
    assert volume.pixel_spacing == tuple(pixel_spacing)
    assert volume.spacing_between_slices == spacing_between_slices
    assert volume.shape == (10, 10, 10)
    assert volume.spatial_shape == (10, 10, 10)
    assert volume.channel_shape == ()
    assert volume.channel_descriptors == ()


def test_volume_from_components():
    vol = Volume.from_components(
        np.zeros((10, 10, 10)),
        position=[1, 2, 3],
        direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        spacing=[2, 2, 5],
        coordinate_system="SLIDE",
    )
    assert vol.position == (1.0, 2.0, 3.0)
    assert np.array_equal(vol.direction, np.eye(3, dtype=np.float32))
    assert vol.spacing == (2.0, 2.0, 5.0)


def test_volume_from_components_np_arrays():
    vol = Volume.from_components(
        np.zeros((10, 10, 10)),
        position=np.array([1, 2, 3]),
        direction=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        spacing=np.array([2, 2, 5]),
        coordinate_system="SLIDE",
    )
    assert vol.position == (1.0, 2.0, 3.0)
    assert np.array_equal(vol.direction, np.eye(3, dtype=np.float32))
    assert vol.spacing == (2.0, 2.0, 5.0)


def test_volume_from_components_np_arrays_2():
    vol = Volume.from_components(
        np.zeros((10, 10, 10)),
        position=np.array([1, 2, 3]),
        direction=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3),
        spacing=np.array([2, 2, 5]),
        coordinate_system="SLIDE",
    )
    assert vol.position == (1.0, 2.0, 3.0)
    assert np.array_equal(vol.direction, np.eye(3, dtype=np.float32))
    assert vol.spacing == (2.0, 2.0, 5.0)


def test_volume_from_components_patient_orientation():
    vol = Volume.from_components(
        np.zeros((10, 10, 10)),
        position=[1, 2, 3],
        patient_orientation="FPL",
        spacing=[2, 2, 5],
        coordinate_system="PATIENT",
    )
    assert vol.position == (1.0, 2.0, 3.0)
    exptected_direction = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(vol.direction, exptected_direction)
    assert vol.spacing == (2.0, 2.0, 5.0)


def test_volume_from_components_center_position():
    vol = Volume.from_components(
        np.zeros((17, 5, 15)),
        center_position=[7.9, 1.3, -9.6],
        patient_orientation="FPL",
        spacing=[2.8, 2.1, 5.7],
        coordinate_system="PATIENT",
    )
    assert np.allclose(vol.center_position, np.array([7.9, 1.3, -9.6]))


def test_volume_with_channels():
    array = np.zeros((10, 10, 10, 2))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        channels={'OpticalPathIdentifier': ['path1', 'path2']},
        coordinate_system="PATIENT",
    )
    assert volume.shape == (10, 10, 10, 2)
    assert volume.spatial_shape == (10, 10, 10)
    assert volume.channel_shape == (2, )
    assert isinstance(volume.channel_descriptors, tuple)
    assert len(volume.channel_descriptors) == 1
    assert isinstance(volume.channel_descriptors[0], ChannelDescriptor)
    expected = ChannelDescriptor('OpticalPathIdentifier')
    assert volume.channel_descriptors[0] == expected
    assert volume.get_channel_values(expected) == ['path1', 'path2']


def test_with_array():
    array = np.zeros((10, 10, 10))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        coordinate_system="PATIENT",
    )
    assert volume.channel_shape == ()
    new_array = np.zeros((10, 10, 10, 2), dtype=np.uint8)
    new_volume = volume.with_array(
        new_array,
        channels={'OpticalPathIdentifier': ['path1', 'path2']},
    )
    assert new_volume.channel_shape == (2, )
    assert isinstance(new_volume, Volume)
    assert volume.spatial_shape == new_volume.spatial_shape
    assert np.array_equal(volume.affine, new_volume.affine)
    assert volume.affine is not new_volume.affine
    assert new_volume.dtype == np.uint8


def test_volume_single_frame():
    volume, ct_series = read_ct_series_volume()
    assert isinstance(volume, Volume)
    rows, columns = ct_series[0].Rows, ct_series[0].Columns
    assert volume.shape == (len(ct_series), rows, columns)
    assert volume.spatial_shape == volume.shape
    assert volume.channel_shape == ()
    orientation = ct_series[0].ImageOrientationPatient
    assert volume.direction_cosines == tuple(orientation)
    direction = volume.direction
    assert np.array_equal(direction[:, 1], orientation[3:])
    assert np.array_equal(direction[:, 2], orientation[:3])
    # Check third direction is normal to others
    assert direction[:, 0] @ direction[:, 1] == 0.0
    assert direction[:, 0] @ direction[:, 2] == 0.0
    assert (direction[:, 0] ** 2).sum() == 1.0

    assert volume.position == tuple(ct_series[1].ImagePositionPatient)

    assert volume.pixel_spacing == tuple(ct_series[0].PixelSpacing)
    slice_spacing = 1.25
    assert volume.spacing == (slice_spacing, *ct_series[0].PixelSpacing[::-1])
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
    assert volume.direction_cosines == tuple(orientation)
    direction = volume.direction
    assert np.array_equal(direction[:, 1], orientation[3:])
    assert np.array_equal(direction[:, 2], orientation[:3])
    # Check third direction is normal to others
    assert direction[:, 0] @ direction[:, 1] == 0.0
    assert direction[:, 0] @ direction[:, 2] == 0.0
    assert (direction[:, 0] ** 2).sum() == 1.0
    first_frame_pos = (
        dcm
        .PerFrameFunctionalGroupsSequence[0]
        .PlanePositionSequence[0]
        .ImagePositionPatient
    )
    assert volume.position == tuple(first_frame_pos)
    assert volume.pixel_spacing == tuple(pixel_spacing)
    slice_spacing = 10.0
    assert volume.spacing == (slice_spacing, *pixel_spacing[::-1])
    assert volume.channel_shape == ()
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
        coordinate_system="PATIENT",
    )

    # Single integer index
    subvolume = volume[3]
    assert subvolume.shape == (1, 50, 50)
    expected_affine = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0, -30.0],
        [0.0, 0.0, 0.0, 1.0],
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
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 7.0],
        [-10.0, 0.0, 0.0, -30.0],
        [0.0, 0.0, 0.0, 1.0],
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
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0, -210.0],
        [0.0, 0.0, 0.0, 1.0],
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
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 49.0],
        [-20.0, 0.0, 0.0, -120.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert np.array_equal(subvolume.affine, expected_affine)
    assert np.array_equal(subvolume.array, array[12:16:2, ::-1])


def test_indexing_source_dimension_2():
    array = np.random.randint(0, 100, (50, 50, 25))
    affine = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    volume = Volume(
        array=array,
        affine=affine,
        coordinate_system="PATIENT",
    )

    subvolume = volume[12:14, :, 12:6:-2]
    assert np.array_equal(subvolume.array, array[12:14, :, 12:6:-2])


def test_array_setter():
    array = np.random.randint(0, 100, (50, 50, 25))
    affine = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    volume = Volume(
        array=array,
        affine=affine,
        coordinate_system="PATIENT",
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
        coordinate_system="PATIENT",
    )
    desired_tup = _normalize_patient_orientation(desired)

    flipped = volume.to_patient_orientation(desired)
    assert isinstance(flipped, Volume)
    assert flipped.get_closest_patient_orientation() == desired_tup

    flipped = volume.to_patient_orientation(desired_tup)
    assert isinstance(flipped, Volume)
    assert flipped.get_closest_patient_orientation() == desired_tup


def test_geometry_from_attributes():
    ori = (0, 1, 0, -1, 0, 0)
    pos = (8.8, 5.3, 9.1)
    spacing_between_slices = 2.5
    pixel_spacing = (1.5, 2.0)
    rows = 10
    columns = 20
    number_of_frames = 5

    geom = VolumeGeometry.from_attributes(
        image_orientation=ori,
        image_position=pos,
        spacing_between_slices=spacing_between_slices,
        pixel_spacing=pixel_spacing,
        rows=rows,
        columns=columns,
        number_of_frames=number_of_frames,
        coordinate_system="PATIENT",
    )

    assert geom.direction_cosines == ori
    assert geom.position == pos
    assert geom.spacing_between_slices == spacing_between_slices
    assert geom.pixel_spacing == pixel_spacing
    assert geom.spatial_shape == (number_of_frames, rows, columns)


def test_geometry_from_components():

    pos = (8.8, 5.3, 9.1)
    spacing = (1.3, 9.7, 6.5)
    shape = (100, 200, 349)

    geom = VolumeGeometry.from_components(
        patient_orientation="LPH",
        position=pos,
        spacing=spacing,
        coordinate_system="PATIENT",
        spatial_shape=shape,
    )

    assert np.array_equal(geom.direction, np.eye(3))
    assert geom.position == pos
    assert geom.spacing == spacing
    assert geom.spatial_shape == shape


def test_geometry_from_components_2():

    pos = (8.8, 5.3, 9.1)
    spacing = (1.3, 9.7, 6.5)
    shape = (100, 200, 349)

    geom = VolumeGeometry.from_components(
        direction=np.eye(3),
        center_position=pos,
        spacing=spacing,
        coordinate_system="PATIENT",
        spatial_shape=shape,
    )

    assert np.allclose(geom.center_position, pos)
    assert geom.get_closest_patient_orientation() == (
        PatientOrientationValuesBiped.L,
        PatientOrientationValuesBiped.P,
        PatientOrientationValuesBiped.H,
    )
    assert geom.spacing == spacing
    assert geom.spatial_shape == shape


def test_volume_transformer():

    geometry = VolumeGeometry(
        np.eye(4),
        [32, 32, 32],
        coordinate_system="PATIENT",
    )

    indices = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
        ]
    )

    expected = np.array(
        [
            [1, 5, 8],
            [1, 5, 9],
        ]
    )

    geometry2 = geometry[1:11, 5:15, 8:18]

    for round_output in [False, True]:
        for check_bounds in [False, True]:
            transformer = VolumeToVolumeTransformer(
                geometry2,
                geometry,
                check_bounds=check_bounds,
                round_output=round_output,
            )

            outputs = transformer(indices)
            if round_output:
                assert outputs.dtype == np.int64
            else:
                assert outputs.dtype == np.float64
            assert np.array_equal(outputs, expected)

    transformer = VolumeToVolumeTransformer(
        geometry2,
        geometry,
        check_bounds=True,
    )
    out_of_bounds_indices = np.array([[31, 0, 0]])
    with pytest.raises(ValueError):
        transformer(out_of_bounds_indices)

    expected = np.array(
        [
            [-1, -5, -8],
            [-1, -5, -7],
        ]
    )
    for round_output in [False, True]:
        transformer = VolumeToVolumeTransformer(
            geometry,
            geometry2,
            round_output=round_output,
        )

        outputs = transformer(indices)
        if round_output:
            assert outputs.dtype == np.int64
        else:
            assert outputs.dtype == np.float64
        assert np.array_equal(outputs, expected)

    transformer = VolumeToVolumeTransformer(
        geometry,
        geometry2,
        check_bounds=True,
    )
    for oob_indices in [
        [0, 5, 8],
        [0, 0, 1],
        [11, 5, 8],
    ]:
        with pytest.raises(ValueError):
            transformer(np.array([oob_indices]))

    geometry3 = geometry2.permute_spatial_axes([2, 1, 0])
    expected = np.array(
        [
            [1, 5, 8],
            [2, 5, 8],
        ]
    )

    for round_output in [False, True]:
        for check_bounds in [False, True]:
            transformer = VolumeToVolumeTransformer(
                geometry3,
                geometry,
                check_bounds=check_bounds,
                round_output=round_output,
            )

            outputs = transformer(indices)
            if round_output:
                assert outputs.dtype == np.int64
            else:
                assert outputs.dtype == np.float64
            assert np.array_equal(outputs, expected)


@pytest.mark.parametrize(
    'crop,pad,permute,reversible',
    [
        (
            (slice(None), slice(14, None), slice(None, None, -1)),
            ((0, 0), (0, 32), (3, 3)),
            (1, 0, 2),
            True,
        ),
        (
            (1, slice(256, 320), slice(256, 320)),
            ((0, 0), (0, 0), (0, 0)),
            (0, 2, 1),
            True,
        ),
        (
            (slice(None), slice(None, None, -1), slice(None)),
            ((12, 31), (1, 23), (5, 7)),
            (0, 2, 1),
            True,
        ),
        (
            (slice(None, None, -1), slice(None, None, -2), slice(None)),
            ((0, 0), (0, 0), (0, 0)),
            (2, 1, 0),
            False,
        ),
    ],
)
def test_match_geometry(crop, pad, permute, reversible):
    vol, _ = read_multiframe_ct_volume()

    transformed = (
        vol[crop]
        .pad(pad)
        .permute_spatial_axes(permute)
     )

    forward_matched = vol.match_geometry(transformed)
    assert forward_matched.geometry_equal(transformed)
    assert np.array_equal(forward_matched.array, transformed.array)

    if reversible:
        reverse_matched = transformed.match_geometry(vol)
        assert reverse_matched.geometry_equal(vol)

        # Perform the transform again on the recovered image to ensure that we
        # end up with the transformed
        inverted_transformed = (
            vol[crop]
            .pad(pad)
            .permute_spatial_axes(permute)
         )
        assert inverted_transformed.geometry_equal(transformed)
        assert np.array_equal(transformed.array, inverted_transformed.array)


def test_match_geometry_nonintersecting():
    vol, _ = read_multiframe_ct_volume()

    new_affine = _translate_affine_matrix(
        vol.affine,
        [0, -32, 32]
    )

    # This geometry has no overlap with the original volume
    geometry = VolumeGeometry(
        new_affine,
        [2, 16, 16],
        coordinate_system="PATIENT",
    )

    transformed = vol.match_geometry(geometry)

    # Result should be an empty array with the requested geometry
    assert transformed.geometry_equal(geometry)
    assert transformed.array.min() == 0
    assert transformed.array.max() == 0


def test_match_geometry_failure_translation():
    vol, _ = read_multiframe_ct_volume()

    new_affine = _translate_affine_matrix(
        vol.affine,
        [0.0, 0.5, 0.0]
    )
    geometry = VolumeGeometry(
        new_affine,
        vol.shape,
        coordinate_system="PATIENT",
    )

    with pytest.raises(RuntimeError):
        vol.match_geometry(geometry)


def test_match_geometry_failure_spacing():
    vol, _ = read_multiframe_ct_volume()

    new_affine = vol.affine.copy()
    new_affine[:3, 2] *= 0.33
    geometry = VolumeGeometry(
        new_affine,
        vol.shape,
        coordinate_system="PATIENT",
    )

    with pytest.raises(RuntimeError):
        vol.match_geometry(geometry)


def test_match_geometry_failure_rotation():
    vol, _ = read_multiframe_ct_volume()

    # Geometry that is rotated with respect to input volume
    geometry = VolumeGeometry.from_attributes(
        image_orientation=(
            np.cos(np.radians(30)), -np.sin(np.radians(30)), 0.0,
            np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0,
        ),
        image_position=vol.position,
        pixel_spacing=vol.pixel_spacing,
        spacing_between_slices=vol.spacing_between_slices,
        number_of_frames=vol.shape[0],
        columns=vol.shape[2],
        rows=vol.shape[1],
        coordinate_system="PATIENT",
    )

    with pytest.raises(RuntimeError):
        vol.match_geometry(geometry)
