import pytest
import unittest

from pydicom.dataelem import DataElement
from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.utils import compute_plane_position_tiled_full, _DicomHelper


params_plane_positions = [
    pytest.param(
        dict(
            row_index=1,
            column_index=1,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(1.0, 1.0, 0.0),
            pixel_matrix_position=(1, 1)
        ),
    ),
    pytest.param(
        dict(
            row_index=2,
            column_index=2,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(17.0, 9.0, 0.0),
            pixel_matrix_position=(9, 17)
        ),
    ),
    pytest.param(
        dict(
            row_index=4,
            column_index=1,
            x_offset=10.0,
            y_offset=20.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(59.0, 21.0, 0.0),
            pixel_matrix_position=(1, 49)
        ),
    ),
    pytest.param(
        dict(
            row_index=4,
            column_index=1,
            x_offset=10.0,
            y_offset=60.0,
            rows=16,
            columns=8,
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(1.0, 1.0),
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(11.0, 11.0, 0.0),
            pixel_matrix_position=(1, 49)
        ),
    ),
    pytest.param(
        dict(
            row_index=4,
            column_index=1,
            x_offset=10.0,
            y_offset=60.0,
            rows=16,
            columns=8,
            image_orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            slice_index=2,
            spacing_between_slices=1.0
        ),
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(11.0, 11.0, 1.0),
            pixel_matrix_position=(1, 49)
        ),
    ),
]


@pytest.mark.parametrize('inputs,expected_output', params_plane_positions)
def test_compute_plane_position_tiled_full(inputs, expected_output):
    output = compute_plane_position_tiled_full(**inputs)
    assert output == expected_output


def test_should_raise_error_when_3d_param_is_missing():
    with pytest.raises(TypeError):
        compute_plane_position_tiled_full(
            row_index=1,
            column_index=1,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            slice_index=1
        )
    with pytest.raises(TypeError):
        compute_plane_position_tiled_full(
            row_index=1,
            column_index=1,
            x_offset=0.0,
            y_offset=0.0,
            rows=16,
            columns=8,
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            pixel_spacing=(1.0, 1.0),
            spacing_between_slices=1.0
        )



class TestDicomHelper(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        # Build data element for all value representations:
        # vrs = [
        #     'AE', 'AS', 'AT', 'CS', 'DA', 'DS', 'DT', 'FL', 'FD', 'IS', 'LO',
        #     'LT', 'OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'PN', 'SH', 'SL', 'SQ',
        #     'SS', 'ST', 'SV', 'TM', 'UC', 'UI', 'UL', 'UN', 'UR',
        #     'US', 'UT', 'UV']
        self.data = {
            "UL": [
                # Keyword: (0008, 0000)
                DataElement(524288, "UL", 506),
                DataElement(524288, "UL", 506),
                DataElement(524288, "UL", 6),
            ],
            "CS": [
                # Keyword: (0008, 0005) SpecificCharacterSet
                DataElement(524293, "CS", "ISO_IR 100"),
                DataElement(524293, "CS", "ISO_IR 100"),
                DataElement(524293, "CS", "ISO_IR 00"),
            ],
            "UI": [
                # Keyword: (0008, 0016) SOPClassUID
                DataElement(524310, "UI", "1.2.840.10008.5.1.4.1.1.1"),
                DataElement(524310, "UI", "1.2.840.10008.5.1.4.1.1.1"),
                DataElement(524310, "UI", "1.2.840.10008.5.1.4.1.1."),
            ],
            "DA": [
                # Keyword: (0008, 0020) StudyDate
                DataElement(524320, "DA", "19950809"),
                DataElement(524320, "DA", "19950809"),
                DataElement(524320, "DA", "9950809"),
            ],
            "TM": [
                # Keyword: (0008, 0030) StudyTime
                DataElement(524336, "TM", "100044"),
                DataElement(524336, "TM", "100044"),
                DataElement(524336, "TM", "00044"),
            ],
            "US": [
                # Keyword: (0008, 0040) DataSetType
                DataElement(524352, "US", 0),
                DataElement(524352, "US", 0),
                DataElement(524352, "US", 1),
            ],
            "LO": [
                # Keyword: (0008, 0041) DataSetSubtype
                DataElement(524353, "LO", "IMA NONE"),
                DataElement(524353, "LO", "IMA NONE"),
                DataElement(524353, "LO", "IMA ONE"),
            ],
            "SH": [
                # Keyword: (0008, 0050) AccessionNumber
                DataElement(524368, "SH", "1157687691469610"),
                DataElement(524368, "SH", "1157687691469610"),
                DataElement(524368, "SH", "157687691469610"),
            ],
            "PN": [
                # Keyword: (0008, 0090) ReferringPhysicianName
                DataElement(524432, "PN", "Dr Alpha"),
                DataElement(524432, "PN", "Dr Alpha"),
                DataElement(524432, "PN", "Dr Beta"),
            ],
            "ST": [
                # Keyword: (0008, 2111) DerivationDescription
                DataElement(532753, "ST", "G0.9D#1.60+0.00,R4R0.5,,D2B0.6,,,"),
                DataElement(532753, "ST", "G0.9D#1.60+0.00,R4R0.5,,D2B0.6,,,"),
                DataElement(532753, "ST", "G0.9D#1.60+0.00,R4R0.5,,D2B0.,,,"),
            ],
            "UN": [
                # Keyword: (0013, 0000)
                DataElement(1245184, "UN", b'\x00\x00\x00'),
                DataElement(1245184, "UN", b'\x00\x00\x00'),
                DataElement(1245184, "UN", b'\x00\x00\x01'),
            ],
            "DS": [
                # Keyword: (0018, 0060) KVP
                DataElement(1572960, "DS", 110),
                DataElement(1572960, "DS", 110),
                DataElement(1572960, "DS", 10),
            ],
            "IS": [
                # Keyword: (0018, 1150) ExposureTime
                DataElement(1577296, "IS", 32),
                DataElement(1577296, "IS", 32),
                DataElement(1577296, "IS", 2),
            ],
            "AS": [
                # Keyword: (0010, 1010) PatientAge
                DataElement(1052688, "AS", "075Y"),
                DataElement(1052688, "AS", "075Y"),
                DataElement(1052688, "AS", "75Y"),
            ],
            "OW": [
                # Keyword: (7fe0, 0010) PixelData
                DataElement(2145386512, "OW", b'\x00\x00\x00\x00\x00\x00'),
                DataElement(2145386512, "OW", b'\x00\x00\x00\x00\x00\x00'),
                DataElement(2145386512, "OW", b'\x00\x00\x00\x00\x00\x01'),
            ],
            "SS": [
                # Keyword: (0028, 0106) SmallestImagePixelValue
                DataElement(2621702, "SS", 0),
                DataElement(2621702, "SS", 0),
                DataElement(2621702, "SS", 1),
            ],
            "DT": [
                # Keyword: (0008, 002a) AcquisitionDateTime
                DataElement(524330, "DT", "20030922101033.000000"),
                DataElement(524330, "DT", "20030922101033.000000"),
                DataElement(524330, "DT", "20030922101033.00000"),
            ],
            "LT": [
                # Keyword: (0018, 7006) DetectorDescription
                DataElement(1601542, "LT", "DETECTOR VERSION 1.0 MTFCOMP 1.0"),
                DataElement(1601542, "LT", "DETECTOR VERSION 1.0 MTFCOMP 1.0"),
                DataElement(1601542, "LT", "DETECTOR VERSION 1.0 MTFCOMP 1."),
            ],
            "OB": [
                # Keyword: (0029, 1131)
                DataElement(2691377, "OB", b'4.0.701169981 '),
                DataElement(2691377, "OB", b'4.0.701169981 '),
                DataElement(2691377, "OB", b'4.0.01169981 '),
            ],
            "AT": [
                # Keyword: (0028, 0009) FrameIncrementPointer
                DataElement(2621449, "AT", 5505152),
                DataElement(2621449, "AT", 5505152),
                DataElement(2621449, "AT", 505152),
            ],
        }

    def test_attribute_equality(self) -> None:
        for vr, [v1, v2, v3] in self.data.items():
            assert _DicomHelper.isequal(v1.value, v2.value) is True
            assert _DicomHelper.isequal(v1.value, v3.value) is False