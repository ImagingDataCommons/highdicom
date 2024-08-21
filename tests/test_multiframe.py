"""Tests for the highdicom._multiframe module."""
import pickle
import numpy as np
from pydicom import dcmread
from pydicom.data import get_testdata_file, get_testdata_files

from highdicom._multiframe import MultiFrameImage


def test_slice_spacing():
    ct_multiframe = dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    image = MultiFrameImage.from_dataset(ct_multiframe)

    expected_affine = np.array(
        [
            [0.0,   0.0, -0.388672, 99.5],
            [0.0,   0.388672, 0.0, -301.5],
            [10.0, 0.0, 0.0, -159],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert image.volume_geometry is not None
    assert image.volume_geometry.spatial_shape[0] == 2
    assert np.array_equal(image.volume_geometry.affine, expected_affine)


def test_slice_spacing_irregular():
    ct_multiframe = dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )

    # Mock some iregular spacings
    ct_multiframe.PerFrameFunctionalGroupsSequence[0].\
        PlanePositionSequence[0].ImagePositionPatient = [1.0, 0.0, 0.0]

    image = MultiFrameImage.from_dataset(ct_multiframe)

    assert image.volume_geometry is None


def test_pickle():
    # Check that the database is successfully serialized and deserialized
    ct_multiframe = dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    image = MultiFrameImage.from_dataset(ct_multiframe)

    ptr = image.dimension_index_pointers[0]

    pickled = pickle.dumps(image)

    # Check that the pickling process has not damaged the db on the existing
    # instance
    # This is just an example operation that requires the db
    assert not image.are_dimension_indices_unique([ptr])

    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, MultiFrameImage)

    # Check that the database has been successfully restored in the
    # deserialization process
    assert not unpickled.are_dimension_indices_unique([ptr])
