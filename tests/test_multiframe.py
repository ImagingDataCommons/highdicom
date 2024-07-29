"""Tests for the highdicom._multiframe module."""
from pydicom import dcmread
from pydicom.data import get_testdata_file, get_testdata_files

from highdicom._multiframe import MultiFrameDBManager


def test_slice_spacing():
    ct_multiframe = dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    db = MultiFrameDBManager(ct_multiframe)

    assert db.number_of_volume_positions == 2
    assert db.spacing_between_slices == 10.0


def test_slice_spacing_irregular():
    ct_multiframe = dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )

    # Mock some iregular spacings
    ct_multiframe.PerFrameFunctionalGroupsSequence[0].\
        PlanePositionSequence[0].ImagePositionPatient = [1.0, 0.0, 0.0]

    db = MultiFrameDBManager(ct_multiframe)

    assert db.number_of_volume_positions is None
    assert db.spacing_between_slices is None
