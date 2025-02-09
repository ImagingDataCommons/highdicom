import numpy as np
import pydicom
from pydicom.sr.codedict import codes
from pydicom.data import get_testdata_file
from pydicom.datadict import tag_for_keyword
from pydicom.tag import BaseTag
import pytest

from highdicom import (
    AlgorithmIdentificationSequence,
    ChannelDescriptor,
    PadModes,
    PatientOrientationValuesBiped,
    PlaneOrientationSequence,
    PlanePositionSequence,
    RGBColorChannels,
    RGB_COLOR_CHANNEL_DESCRIPTOR,
    UID,
    Volume,
    VolumeGeometry,
    VolumeToVolumeTransformer,
    get_volume_from_series,
    imread,
)
from highdicom.seg import (
    Segmentation,
    SegmentDescription,
    SegmentAlgorithmTypeValues,
    SegmentationTypeValues,
)
from highdicom.spatial import (
    _normalize_patient_orientation,
    _translate_affine_matrix,
)

from tests.utils import write_and_read_dataset


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
    orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    volume = Volume.from_attributes(
        array=array,
        image_position=[0.0, 0.0, 0.0],
        image_orientation=orientation,
        pixel_spacing=[1.0, 1.0],
        spacing_between_slices=10.0,
        coordinate_system="PATIENT",
    )
    plane_positions = volume.get_plane_positions()
    for i, pos in enumerate(plane_positions):
        assert isinstance(pos, PlanePositionSequence)
        assert np.array_equal(
            pos[0].ImagePositionPatient,
            [0.0, 0.0, -10.0 * i]
        )

        # Same thing but retrieve plane position individually
        pos_2 = volume.get_plane_position(i)
        assert isinstance(pos_2, PlanePositionSequence)
        assert np.array_equal(
            pos_2[0].ImagePositionPatient,
            [0.0, 0.0, -10.0 * i]
        )

    ori = volume.get_plane_orientation()
    assert isinstance(ori, PlaneOrientationSequence)
    assert np.array_equal(
        ori[0].ImageOrientationPatient,
        orientation,
    )

    indices = np.array([[1, 2, 3]])
    coords = volume.map_indices_to_reference(indices)
    assert np.array_equal(coords, np.array([[3.0, 2.0, -10.0]]))
    round_trip = volume.map_reference_to_indices(coords)
    assert np.array_equal(round_trip, indices)
    round_trip = volume.map_reference_to_indices(
        coords,
        check_bounds=True,
        round_output=True,
    )
    assert np.array_equal(round_trip, indices)
    index_center = volume.center_indices
    assert np.array_equal(index_center, [12.0, 24.5, 24.5])
    index_center = volume.nearest_center_indices
    assert np.array_equal(index_center, [12, 24, 24])
    coord_center = volume.center_position
    assert np.array_equal(coord_center, [24.5, 24.5, -120])

    ras_affine = volume.get_affine("RAH")
    expected = np.array(
        [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert np.array_equal(ras_affine, expected)

    geom = volume.get_geometry()
    assert isinstance(geom, VolumeGeometry)


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
    assert volume.physical_extent == tuple(
        [n * s for n, s in zip(volume.spatial_shape, volume.spacing)]
    )


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

    array = np.zeros((number_of_frames, rows, columns))
    vol = geom.with_array(array)
    assert isinstance(vol, Volume)


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


def test_swap_axes():
    vol, _ = read_multiframe_ct_volume()

    swapped = vol.swap_spatial_axes(1, 2)
    assert swapped.spatial_shape == (2, 512, 512)

    u1, u2, u3 = vol.spacing_vectors()
    v1, v2, v3 = swapped.spacing_vectors()

    assert np.array_equal(u1, v1)
    assert np.array_equal(u2, v3)
    assert np.array_equal(u3, v2)


def test_random_operations():
    vol = Volume.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[0.5, 0.5, 2.0],
        coordinate_system="PATIENT",
        array=np.random.randint(0, 10, size=(20, 20, 50)),
    )

    vol_2 = (
        vol
        .random_flip_spatial()
        .random_permute_spatial_axes()
        .random_spatial_crop([10, 10, 10])
    )

    matched = vol.match_geometry(vol_2)
    assert np.allclose(matched.affine, vol_2.affine)
    assert np.array_equal(matched.array, vol_2.array)


def test_random_operations_subset_axes():
    vol = Volume.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[0.5, 0.5, 2.0],
        coordinate_system="PATIENT",
        array=np.random.randint(0, 10, size=(20, 20, 50)),
    )

    vol_2 = (
        vol
        .random_flip_spatial([0, 1])
        .random_permute_spatial_axes([0, 1])
        .random_spatial_crop([10, 10, 10])
    )

    matched = vol.match_geometry(vol_2)
    assert np.allclose(matched.affine, vol_2.affine)
    assert np.array_equal(matched.array, vol_2.array)


def test_random_operations_geometry():
    vol = VolumeGeometry.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[0.5, 0.5, 2.0],
        coordinate_system="PATIENT",
        spatial_shape=[20, 20, 50],
    )

    vol_2 = (
        vol
        .random_flip_spatial()
        .random_permute_spatial_axes()
        .random_spatial_crop([10, 10, 10])
    )

    matched = vol.match_geometry(vol_2)
    assert np.allclose(matched.affine, vol_2.affine)


@pytest.mark.parametrize('mode', list(PadModes))
def test_pad(mode):
    vol = Volume.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[0.5, 0.5, 2.0],
        coordinate_system="PATIENT",
        array=np.random.randint(0, 10, size=(20, 20, 50)),
    )

    padded = vol.pad(10, mode=mode)
    assert padded.spatial_shape == (40, 40, 70)

    padded = vol.pad([5, 10], mode=mode)
    assert padded.spatial_shape == (35, 35, 65)

    padded = vol.pad([[5, 10], [10, 10], [25, 15]], mode=mode)
    assert padded.spatial_shape == (35, 40, 90)


@pytest.mark.parametrize('mode', list(PadModes))
@pytest.mark.parametrize('per_channel', [False, True])
def test_pad_with_channels(mode, per_channel):
    vol = Volume.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[0.5, 0.5, 2.0],
        coordinate_system="PATIENT",
        array=np.random.randint(0, 10, size=(20, 20, 50, 3)),
        channels={RGB_COLOR_CHANNEL_DESCRIPTOR: ['R', 'G', 'B']},
    )

    padded = vol.pad(10, mode=mode, per_channel=per_channel)
    assert padded.spatial_shape == (40, 40, 70)
    assert padded.match_geometry(vol).geometry_equal(vol)

    padded = vol.pad([5, 10], mode=mode, per_channel=per_channel)
    assert padded.spatial_shape == (35, 35, 65)
    assert padded.match_geometry(vol).geometry_equal(vol)

    padded = vol.pad(
        [[5, 10], [10, 10], [25, 15]],
        mode=mode,
        per_channel=per_channel
    )
    assert padded.spatial_shape == (35, 40, 90)
    assert padded.match_geometry(vol).geometry_equal(vol)


def test_pad_to_spatial_shape():
    vol, _ = read_multiframe_ct_volume()

    shape = (10, 600, 600)
    padded = vol.pad_to_spatial_shape(shape)
    assert padded.spatial_shape == shape

    assert padded.match_geometry(vol).geometry_equal(vol)


def test_pad_or_crop_to_spatial_shape():
    vol, _ = read_multiframe_ct_volume()

    shape = (10, 240, 240)
    padded = vol.pad_or_crop_to_spatial_shape(shape)
    assert padded.spatial_shape == shape

    assert padded.match_geometry(vol).geometry_equal(vol)


def test_normalize():
    vol, _ = read_multiframe_ct_volume()

    normed = vol.normalize_mean_std()
    assert np.isclose(normed.array.mean(), 0.0)
    assert np.isclose(np.std(normed.array), 1.0)

    normed = vol.normalize_min_max()
    assert np.isclose(normed.array.min(), 0.0)
    assert np.isclose(normed.array.max(), 1.0)


@pytest.mark.parametrize(
    'kw,pytype',
    [
        ('DiffusionBValue', float),
        ('SegmentNumber', int),
        ('SegmentLabel', str),
    ]
)
def test_channel_descriptors(kw, pytype):
    tag = tag_for_keyword(kw)

    d1 = ChannelDescriptor(kw)
    d2 = ChannelDescriptor(tag)
    d3 = ChannelDescriptor(BaseTag(tag))
    d4 = ChannelDescriptor(d1)

    for d in [d1, d2, d3, d4]:
        assert d.tag == tag
        assert isinstance(d.tag, BaseTag)
        assert d.keyword == kw
        assert not d.is_custom
        assert not d.is_enumerated
        assert d.value_type is pytype
        assert hash(d) == hash(kw)
        assert str(d) == kw
        assert repr(d) == kw


def test_channel_descriptors_custom():
    d = ChannelDescriptor('name', is_custom=True, value_type=int)

    assert d.tag is None
    assert d.keyword == 'name'
    assert d.is_custom
    assert not d.is_enumerated
    assert d.value_type is int


def test_channel_descriptors_enum():
    d = ChannelDescriptor(
        'name',
        is_custom=True,
        value_type=SegmentationTypeValues
    )

    assert d.tag is None
    assert d.keyword == 'name'
    assert d.is_custom
    assert d.is_enumerated
    assert d.value_type is SegmentationTypeValues


def test_multi_channels():

    optical_path_desc = ChannelDescriptor('OpticalPathIdentifier')
    path_names = ['path1', 'path2', 'path3', 'path4']

    vol = Volume.from_components(
        direction=np.eye(3),
        center_position=[98.1, 78.4, 23.1],
        spacing=[0.5, 0.5, 2.0],
        coordinate_system="PATIENT",
        array=np.random.randint(0, 10, size=(20, 20, 50, 3, 4)),
        channels={
            RGB_COLOR_CHANNEL_DESCRIPTOR: ['R', 'G', 'B'],
            optical_path_desc: path_names
        },
    )

    assert vol.shape == (20, 20, 50, 3, 4)
    assert vol.spatial_shape == (20, 20, 50)
    assert vol.channel_shape == (3, 4)
    assert vol.number_of_channel_dimensions == 2
    assert vol.channel_descriptors == (
        RGB_COLOR_CHANNEL_DESCRIPTOR,
        optical_path_desc,
    )
    assert vol.get_channel_values(optical_path_desc) == path_names
    assert vol.get_channel_values('OpticalPathIdentifier') == path_names
    assert vol.get_channel_values(optical_path_desc.tag) == path_names
    rgb = [RGBColorChannels.R, RGBColorChannels.G, RGBColorChannels.B]
    assert vol.get_channel_values(RGB_COLOR_CHANNEL_DESCRIPTOR) == rgb
    assert vol.get_channel_values(
        RGB_COLOR_CHANNEL_DESCRIPTOR.keyword
    ) == rgb

    permuted = vol.permute_channel_axes(
        [optical_path_desc, RGB_COLOR_CHANNEL_DESCRIPTOR]
    )
    assert permuted.shape == (20, 20, 50, 4, 3)
    assert permuted.channel_shape == (4, 3)
    assert permuted.channel_descriptors == (
        optical_path_desc,
        RGB_COLOR_CHANNEL_DESCRIPTOR,
    )

    permuted = vol.permute_channel_axes_by_index([1, 0])
    assert permuted.shape == (20, 20, 50, 4, 3)
    assert permuted.channel_shape == (4, 3)
    assert permuted.channel_descriptors == (
        optical_path_desc,
        RGB_COLOR_CHANNEL_DESCRIPTOR,
    )

    path1 = vol.get_channel(OpticalPathIdentifier='path1')
    assert path1.channel_shape == (3, )
    assert path1.number_of_channel_dimensions == 1
    assert path1.channel_descriptors == (RGB_COLOR_CHANNEL_DESCRIPTOR, )
    assert np.array_equal(path1.array, vol.array[:, :, :, :, 0])

    path2 = vol.get_channel(OpticalPathIdentifier='path2', keepdims=True)
    assert path2.channel_shape == (3, 1)
    assert path2.number_of_channel_dimensions == 2
    assert path2.channel_descriptors == (
        RGB_COLOR_CHANNEL_DESCRIPTOR,
        optical_path_desc
    )
    assert np.array_equal(path2.array, vol.array[:, :, :, :, 1:2])
    squeezed = path2.squeeze_channel()
    assert squeezed.channel_shape == (3, )
    assert squeezed.number_of_channel_dimensions == 1
    assert squeezed.channel_descriptors == (RGB_COLOR_CHANNEL_DESCRIPTOR, )
    assert np.array_equal(squeezed.array, vol.array[:, :, :, :, 1])

    red_channel = vol.get_channel(RGBColorChannel='R')
    assert red_channel.channel_shape == (4, )
    assert red_channel.number_of_channel_dimensions == 1
    assert red_channel.channel_descriptors == (optical_path_desc, )
    assert np.array_equal(red_channel.array, vol.array[:, :, :, 0, :])

    red_channel = vol.get_channel(RGBColorChannel=RGBColorChannels.R)
    assert red_channel.channel_shape == (4, )
    assert red_channel.number_of_channel_dimensions == 1
    assert red_channel.channel_descriptors == (optical_path_desc, )
    assert np.array_equal(red_channel.array, vol.array[:, :, :, 0, :])


def test_match_geometry_segmentation():
    # Test that creates a segmentation from a manipulated volume and ensures
    # that the result can be matched back to the input image

    # Load an enhanced (multiframe) CT image
    im = imread(get_testdata_file('eCT_Supplemental.dcm'))

    # Load the input volume
    original_volume = im.get_volume()

    # Manipulate the original volume
    input_volume = (
        original_volume
        .to_patient_orientation("PRF")
        .crop_to_spatial_shape((400, 400, 2))
    )

    # Form a segmentation from the manpulated array
    seg_array = input_volume.array > 0

    # Since the seg array shares its geometry with the inupt array, we can
    # combine the two to create a volume of the segmentation array
    seg_volume = input_volume.with_array(seg_array)

    algorithm_identification = AlgorithmIdentificationSequence(
        name='Complex Segmentation Tool',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # metadata needed for a segmentation
    brain_description = SegmentDescription(
        segment_number=1,
        segment_label='brain',
        segmented_property_category=codes.SCT.Organ,
        segmented_property_type=codes.SCT.Brain,
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
    )

    # Use the segmentation volume as input to create a DICOM Segmentation
    seg_dataset = Segmentation(
        pixel_array=seg_volume,
        source_images=[im],
        segmentation_type=SegmentationTypeValues.LABELMAP,
        segment_descriptions=[brain_description],
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        manufacturer='Complex Segmentations Plc.',
        manufacturer_model_name='Complex Segmentation Tool',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        omit_empty_frames=False,
    )

    seg_dataset = write_and_read_dataset(seg_dataset)
    seg_dataset = Segmentation.from_dataset(seg_dataset, copy=False)

    # The reconstructed volume should be the same as the input volume, but may
    # have a different handedness
    seg_volume_recon = (
        seg_dataset
        .get_volume(combine_segments=True)
        .ensure_handedness(seg_volume.handedness, flip_axis=0)
    )

    assert np.array_equal(seg_volume_recon.affine, seg_volume.affine)
    assert np.array_equal(seg_volume_recon.array, seg_volume.array)

    # Alternatively, it may be desirable to match the geometry of the output
    # segmentation image to that of the input image
    seg_volume_matched = seg_volume.match_geometry(original_volume)

    assert np.array_equal(original_volume.affine, seg_volume_matched.affine)

    # Use the segmentation volume as input to create a DICOM Segmentation
    seg_dataset_matched = Segmentation(
        pixel_array=seg_volume_matched,
        source_images=[im],
        segmentation_type=SegmentationTypeValues.LABELMAP,
        segment_descriptions=[brain_description],
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        manufacturer='Complex Segmentations Plc.',
        manufacturer_model_name='Complex Segmentation Tool',
        software_versions='0.0.1',
        device_serial_number='1234567890',
    )
    seg_dataset_matched = write_and_read_dataset(seg_dataset_matched)
    seg_dataset_matched = Segmentation.from_dataset(
        seg_dataset_matched,
        copy=False
    )
    seg_vol_from_matched_dataset = seg_dataset_matched.get_volume(
        combine_segments=True
    )

    assert np.array_equal(
        seg_vol_from_matched_dataset.affine,
        original_volume.affine
    )
    assert np.array_equal(
        seg_vol_from_matched_dataset.array,
        seg_volume_matched.array
    )
