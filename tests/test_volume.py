import numpy as np
import pydicom
from pydicom.data import get_testdata_file
import pytest


from highdicom.volume import VolumeArray, concat_channels


def test_transforms():
    array = np.zeros((25, 50, 50))
    volume = VolumeArray.from_attributes(
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
    volume = VolumeArray.from_attributes(
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
    volume = VolumeArray.from_attributes(
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
    volume = VolumeArray.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
    )
    new_array = np.zeros((10, 10, 10, 2), dtype=np.uint8)
    new_volume = volume.with_array(new_array)
    assert new_volume.number_of_channels == 2
    assert isinstance(new_volume, VolumeArray)
    assert volume.spatial_shape == new_volume.spatial_shape
    assert np.array_equal(volume.affine, new_volume.affine)
    assert volume.affine is not new_volume.affine
    assert new_volume.dtype == np.uint8

    concat_volume = concat_channels([volume, new_volume])
    assert isinstance(concat_volume, VolumeArray)
    assert volume.spatial_shape == concat_volume.spatial_shape
    assert concat_volume.number_of_channels == 3


def test_volume_single_frame():
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136'),
        get_testdata_file('dicomdirtests/77654033/CT2/17196'),
        get_testdata_file('dicomdirtests/77654033/CT2/17166'),
    ]
    ct_series = [pydicom.dcmread(f) for f in ct_files]
    volume = VolumeArray.from_image_series(ct_series)
    assert isinstance(volume, VolumeArray)
    rows, columns = ct_series[0].Rows, ct_series[0].Columns
    assert volume.shape == (len(ct_files), rows, columns)
    assert volume.spatial_shape == volume.shape
    assert volume.number_of_channels is None
    assert volume.frame_numbers is None
    sop_instance_uids = [
        ct_series[0].SOPInstanceUID,
        ct_series[2].SOPInstanceUID,
        ct_series[1].SOPInstanceUID,
    ]
    assert volume.sop_instance_uids == sop_instance_uids
    assert volume.get_index_for_sop_instance_uid(
        ct_series[2].SOPInstanceUID
    ) == 1
    with pytest.raises(RuntimeError):
        volume.get_index_for_frame_number(2)
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


def test_volume_multiframe():
    dcm = pydicom.dcmread(get_testdata_file('eCT_Supplemental.dcm'))
    volume = VolumeArray.from_image(dcm)
    assert isinstance(volume, VolumeArray)
    rows, columns = dcm.Rows, dcm.Columns
    assert volume.shape == (dcm.NumberOfFrames, rows, columns)
    assert volume.spatial_shape == volume.shape
    assert volume.frame_numbers == [2, 1]
    assert volume.sop_instance_uids is None
    with pytest.raises(RuntimeError):
        volume.get_index_for_sop_instance_uid(
            dcm.SOPInstanceUID
        )
    assert volume.get_index_for_frame_number(2) == 0
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
    first_frame = volume.frame_numbers[0]
    first_frame_pos = (
        dcm
        .PerFrameFunctionalGroupsSequence[first_frame - 1]
        .PlanePositionSequence[0]
        .ImagePositionPatient
    )
    assert volume.position == first_frame_pos
    assert volume.pixel_spacing == pixel_spacing
    slice_spacing = 10.0
    assert volume.spacing == [slice_spacing, *pixel_spacing[::-1]]
    assert volume.number_of_channels is None