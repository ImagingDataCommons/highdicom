import pytest
import unittest

from pydicom import dcmread
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
)
from pydicom.data import get_testdata_file
from highdicom import SOPClass, UID
from highdicom.base import _check_little_endian


class TestBase(unittest.TestCase):

    def test_big_endian(self):
        with pytest.raises(ValueError):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                transfer_syntax_uid=ExplicitVRBigEndian,
            )

    def test_explicit_vr(self):
        sop_class = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            transfer_syntax_uid=ExplicitVRLittleEndian,
        )
        assert not sop_class.is_implicit_VR
        assert sop_class.is_little_endian

    def test_implicit_vr(self):
        sop_class = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            transfer_syntax_uid=ImplicitVRLittleEndian,
        )
        assert sop_class.is_implicit_VR
        assert sop_class.is_little_endian


class TestEndianCheck(unittest.TestCase):

    def test_big_endian(self):
        ds = dcmread(get_testdata_file('MR_small_bigendian.dcm'))
        with pytest.raises(ValueError):
            _check_little_endian(ds)
