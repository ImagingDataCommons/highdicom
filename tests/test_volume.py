import numpy as np
import pydicom
from pydicom.data import get_testdata_file
import pytest


from highdicom.volume import VolumeGeometry


def test_transforms():
    volume = VolumeGeometry.from_attributes(
        image_position=[0.0, 0.0, 0.0],
        image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        pixel_spacing=[1.0, 1.0],
        rows=50,
        columns=50,
        spacing_between_slices=10.0,
        number_of_frames=25,
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
def test_geometry_from_attributes(
    image_position,
    image_orientation,
    pixel_spacing,
    spacing_between_slices,
):
    geometry = VolumeGeometry.from_attributes(
        rows=10,
        columns=10,
        number_of_frames=10,
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=spacing_between_slices,
    )
    assert geometry.position == list(image_position)
    assert geometry.direction_cosines == list(image_orientation)
    assert geometry.pixel_spacing == list(pixel_spacing)
    assert geometry.spacing_between_slices == spacing_between_slices


def test_volume_geometry_single_frame():
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136'),
        get_testdata_file('dicomdirtests/77654033/CT2/17196'),
        get_testdata_file('dicomdirtests/77654033/CT2/17166'),
    ]
    ct_series = [pydicom.dcmread(f) for f in ct_files]
    geometry = VolumeGeometry.for_image_series(ct_series)
    assert isinstance(geometry, VolumeGeometry)
    rows, columns = ct_series[0].Rows, ct_series[0].Columns
    assert geometry.shape == (len(ct_files), rows, columns)
    assert geometry.frame_numbers is None
    sop_instance_uids = [
        ct_series[0].SOPInstanceUID,
        ct_series[2].SOPInstanceUID,
        ct_series[1].SOPInstanceUID,
    ]
    assert geometry.sop_instance_uids == sop_instance_uids
    assert geometry.get_index_for_sop_instance_uid(
        ct_series[2].SOPInstanceUID
    ) == 1
    with pytest.raises(RuntimeError):
        geometry.get_index_for_frame_number(2)
    orientation = ct_series[0].ImageOrientationPatient
    assert geometry.direction_cosines == orientation
    direction = geometry.direction
    assert np.array_equal(direction[:, 1], orientation[3:])
    assert np.array_equal(direction[:, 2], orientation[:3])
    # Check third direction is normal to others
    assert direction[:, 0] @ direction[:, 1] == 0.0
    assert direction[:, 0] @ direction[:, 2] == 0.0
    assert (direction[:, 0] ** 2).sum() == 1.0
    assert geometry.position == ct_series[0].ImagePositionPatient
    assert geometry.pixel_spacing == ct_series[0].PixelSpacing
    slice_spacing = 1.25
    assert geometry.spacing == [slice_spacing, *ct_series[0].PixelSpacing[::-1]]


def test_volume_geometry_multiframe():
    dcm = pydicom.dcmread(get_testdata_file('eCT_Supplemental.dcm'))
    geometry = VolumeGeometry.for_image(dcm)
    assert isinstance(geometry, VolumeGeometry)
    rows, columns = dcm.Rows, dcm.Columns
    assert geometry.shape == (dcm.NumberOfFrames, rows, columns)
    assert geometry.frame_numbers == [2, 1]
    assert geometry.sop_instance_uids is None
    with pytest.raises(RuntimeError):
        geometry.get_index_for_sop_instance_uid(
            dcm.SOPInstanceUID
        )
    assert geometry.get_index_for_frame_number(2) == 0
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
    assert geometry.direction_cosines == orientation
    direction = geometry.direction
    assert np.array_equal(direction[:, 1], orientation[3:])
    assert np.array_equal(direction[:, 2], orientation[:3])
    # Check third direction is normal to others
    assert direction[:, 0] @ direction[:, 1] == 0.0
    assert direction[:, 0] @ direction[:, 2] == 0.0
    assert (direction[:, 0] ** 2).sum() == 1.0
    first_frame = geometry.frame_numbers[0]
    first_frame_pos = (
        dcm
        .PerFrameFunctionalGroupsSequence[first_frame - 1]
        .PlanePositionSequence[0]
        .ImagePositionPatient
    )
    assert geometry.position == first_frame_pos
    assert geometry.pixel_spacing == pixel_spacing
    slice_spacing = 10.0
    assert geometry.spacing == [slice_spacing, *pixel_spacing[::-1]]
