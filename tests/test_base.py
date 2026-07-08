import datetime
import re
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
from highdicom.enum import SpecificCharacterSetValues


class TestBase(unittest.TestCase):

    def test_type_2_attributes(self):
        # Series Number and Instance Number are type 1 for several IODs.
        # Therefore, we decided that we require them even on the base class.
        with pytest.raises(TypeError):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=None,
                modality='SR',
                manufacturer='highdicom',
                transfer_syntax_uid=ExplicitVRLittleEndian,
            )
        with pytest.raises(TypeError):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=None,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                transfer_syntax_uid=ExplicitVRLittleEndian,
            )

    def test_type_3_attributes(self):
        instance = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            manufacturer_model_name='foo-bar',
            software_versions='v1.0.0',
            transfer_syntax_uid=ExplicitVRLittleEndian,
        )
        assert instance.SoftwareVersions is not None
        assert instance.ManufacturerModelName is not None
        assert hasattr(instance, 'ContentDate')
        assert hasattr(instance, 'ContentTime')
        assert not hasattr(instance, 'SeriesDate')
        assert not hasattr(instance, 'SeriesTime')

    def test_content_time_without_date(self):
        msg = (
            "'content_time' may not be specified without "
            "'content_date'."
        )
        with pytest.raises(TypeError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                manufacturer_model_name='foo-bar',
                software_versions='v1.0.0',
                transfer_syntax_uid=ExplicitVRLittleEndian,
                content_time=datetime.time(12, 34, 56),
            )

    def test_series_datetime(self):
        instance = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            manufacturer_model_name='foo-bar',
            software_versions='v1.0.0',
            transfer_syntax_uid=ExplicitVRLittleEndian,
            content_date=datetime.date(2024, 6, 15),
            content_time=datetime.time(14, 0, 0),
            series_date=datetime.date(2024, 6, 15),
            series_time=datetime.time(12, 34, 56),
        )
        assert hasattr(instance, 'SeriesDate')
        assert hasattr(instance, 'SeriesTime')

    def test_series_datetime_earlier_date(self):
        # series_time > content_time is fine when series_date < content_date
        instance = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            manufacturer_model_name='foo-bar',
            software_versions='v1.0.0',
            transfer_syntax_uid=ExplicitVRLittleEndian,
            content_date=datetime.date(2024, 6, 15),
            content_time=datetime.time(8, 0, 0),
            series_date=datetime.date(2024, 6, 14),
            series_time=datetime.time(23, 59, 59),
        )
        assert hasattr(instance, 'SeriesDate')
        assert hasattr(instance, 'SeriesTime')

    def test_series_time_after_content_same_date(self):
        msg = (
            "'series_time' must not be later than content time."
        )
        with pytest.raises(ValueError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                manufacturer_model_name='foo-bar',
                software_versions='v1.0.0',
                transfer_syntax_uid=ExplicitVRLittleEndian,
                content_date=datetime.date(2024, 6, 15),
                content_time=datetime.time(8, 0, 0),
                series_date=datetime.date(2024, 6, 15),
                series_time=datetime.time(12, 34, 56),
            )

    def test_series_date_without_time(self):
        msg = (
            "'series_time' may not be specified without "
            "'series_date'."
        )
        with pytest.raises(TypeError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                manufacturer_model_name='foo-bar',
                software_versions='v1.0.0',
                transfer_syntax_uid=ExplicitVRLittleEndian,
                series_time=datetime.time(12, 34, 56),
            )

    def test_series_date_after_content(self):
        msg = (
            "'series_date' must not be later than 'content_date'."
        )
        with pytest.raises(ValueError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                manufacturer_model_name='foo-bar',
                software_versions='v1.0.0',
                transfer_syntax_uid=ExplicitVRLittleEndian,
                content_date=datetime.date(2000, 12, 1),
                series_date=datetime.date(2000, 12, 2),
            )

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
        _ = SOPClass(
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

    def test_implicit_vr(self):
        _ = SOPClass(
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

    def test_series_description_too_long(self):
        msg = (
            "Values of DICOM value representation Long "
            "String (LO) must not exceed 64 characters."
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                series_description="abc" * 100,
            )

    def test_specific_character_set(self):
        instance = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            specific_character_set='ISO_IR 126',
        )
        assert instance.SpecificCharacterSet == 'ISO_IR 126'

    def test_specific_character_invalid(self):
        msg = "'ISO_IR 666' is not a valid SpecificCharacterSetValues"
        with pytest.raises(ValueError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                specific_character_set='ISO_IR 666',
            )

    def test_specific_character_set_enum(self):
        instance = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            specific_character_set=SpecificCharacterSetValues.ARABIC,
        )
        assert instance.SpecificCharacterSet == 'ISO_IR 127'

    def test_specific_character_invalid_code_extensions(self):
        msg = (
            "Encodings with code extensions should not be used when only a "
            "single encoding is included."
        )
        with pytest.raises(ValueError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                specific_character_set=(
                    SpecificCharacterSetValues.GREEK_CODE_EXTENSIONS,
                ),
            )

    def test_specific_character_multivalue(self):
        instance = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            specific_character_set=[
                SpecificCharacterSetValues.LATIN_ALPHABET_NO_2_CODE_EXTENSIONS,
                SpecificCharacterSetValues.CYRILLIC_CODE_EXTENSIONS,
            ]
        )
        assert instance.SpecificCharacterSet == [
            'ISO 2022 IR 101',
            'ISO 2022 IR 144',
        ]

    def test_specific_character_multivalue_multibyte(self):
        instance = SOPClass(
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
            instance_number=1,
            modality='SR',
            manufacturer='highdicom',
            specific_character_set=[
                SpecificCharacterSetValues.JAPANESE_CODE_EXTENSIONS,
                SpecificCharacterSetValues.JAPANESE_KANJI_CODE_EXTENSIONS,
            ]
        )
        assert instance.SpecificCharacterSet == [
            'ISO 2022 IR 13',
            'ISO 2022 IR 87',
        ]

    def test_specific_character_invalid_multivalue_multibyte(self):
        msg = (
            "When SpecificCharacterSet is multi-valued, a multi-byte "
            "encoding may not be used as the first value."
        )
        with pytest.raises(ValueError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                specific_character_set=[
                    SpecificCharacterSetValues.JAPANESE_KANJI_CODE_EXTENSIONS,
                    SpecificCharacterSetValues.JAPANESE_CODE_EXTENSIONS,
                ]
            )

    def test_specific_character_invalid_multivalue(self):
        msg = (
            "When SpecificCharacterSet is multi-valued, all encodings "
            "must use code extensions."
        )
        with pytest.raises(ValueError, match=msg):
            SOPClass(
                study_instance_uid=UID(),
                series_instance_uid=UID(),
                series_number=1,
                sop_instance_uid=UID(),
                sop_class_uid='1.2.840.10008.5.1.4.1.1.88.33',
                instance_number=1,
                modality='SR',
                manufacturer='highdicom',
                specific_character_set=[
                    SpecificCharacterSetValues.LATIN_ALPHABET_NO_2,
                    SpecificCharacterSetValues.CYRILLIC_CODE_EXTENSIONS,
                ]
            )


class TestEndianCheck(unittest.TestCase):

    def test_big_endian(self):
        ds = dcmread(get_testdata_file('MR_small_bigendian.dcm'))
        with pytest.raises(ValueError):
            _check_little_endian(ds)
