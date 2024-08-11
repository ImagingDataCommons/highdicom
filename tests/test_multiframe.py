"""Tests for the highdicom._multiframe module."""
import numpy as np
from pydicom import dcmread
from pydicom.data import get_testdata_file, get_testdata_files

from highdicom._multiframe import MultiFrameDBManager


def test_slice_spacing():
    ct_multiframe = dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    db = MultiFrameDBManager(ct_multiframe)

    expected_affine = np.array(
        [
            [0.0,   0.0, -0.388672, 99.5],
            [0.0,   0.388672, 0.0, -301.5],
            [-10.0, 0.0, 0.0, -149],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert db.volume_geometry.spatial_shape[0] == 2
    assert np.array_equal(db.volume_geometry.affine, expected_affine)


def test_slice_spacing_irregular():
    ct_multiframe = dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )

    # Mock some iregular spacings
    ct_multiframe.PerFrameFunctionalGroupsSequence[0].\
        PlanePositionSequence[0].ImagePositionPatient = [1.0, 0.0, 0.0]

    db = MultiFrameDBManager(ct_multiframe)

    assert db.volume_geometry is None
