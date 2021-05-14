import unittest
from pydicom import FileDataset, Dataset
from pydicom.dataelem import DataElement
from pydicom.uid import generate_uid
from highdicom.legacy import sop
from datetime import datetime, timedelta
import enum


class Modality(enum.IntEnum):
    CT = 0
    MR = 1
    PT = 2


sop_classes = [('CT', '1.2.840.10008.5.1.4.1.1.2'),
               ('MR', '1.2.840.10008.5.1.4.1.1.4'),
               ('PT', '1.2.840.10008.5.1.4.1.1.128')]


class DicomGenerator:

    def __init__(
            self,
            slice_per_frameset: int = 3,
            slice_thickness: float = 0.1,
            pixel_spacing: float = 0.1,
            row: int = 2,
            col: int = 2,) -> None:
        self._slice_per_frameset = slice_per_frameset
        self._slice_thickness = slice_thickness
        self._pixel_spacing = pixel_spacing
        self._row = row
        self._col = col
        self._study_uid = generate_uid()
        self._z_orientation_mat = [
            1.000000, 0.000000, 0.000000,
            0.000000, 1.000000, 0.000000]
        self._z_position_vec = [0.0, 0.0, 1.0]
        self._y_orientation_mat = [
            0.000000, 0.000000, 1.000000,
            1.000000, 0.000000, 0.000000]
        self._y_position_vec = [0.0, 1.0, 0.0]
        self._x_orientation_mat = [
            0.000000, 1.000000, 0.000000,
            0.000000, 0.000000, 1.000000]
        self._x_position_vec = [1.0, 0.0, 0.0]

    def _generate_frameset(self,
                           system: Modality,
                           orientation_mat: list,
                           position_vec: list,
                           series_uid: str,
                           first_slice_offset: float = 0,
                           frameset_idx: int = 0) -> list:
        output_dataset = []
        slice_pos = first_slice_offset
        slice_thickness = self._slice_thickness
        study_uid = self._study_uid
        frame_of_ref_uid = generate_uid()
        date_ = datetime.now().date()
        age = timedelta(days=45 * 365)
        time_ = datetime.now().time()
        cols = self._col
        rows = self._row
        bytes_per_voxel = 2

        for i in range(0, self._slice_per_frameset):
            file_meta = Dataset()
            pixel_array = b"\0" * cols * rows * bytes_per_voxel
            file_meta.MediaStorageSOPClassUID = sop_classes[system][1]
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = generate_uid()
            tmp_dataset = FileDataset('', {}, file_meta=file_meta,
                                      preamble=pixel_array)
            tmp_dataset.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
            tmp_dataset.SliceLocation = slice_pos + i * slice_thickness
            tmp_dataset.SliceThickness = slice_thickness
            tmp_dataset.WindowCenter = 1
            tmp_dataset.WindowWidth = 2
            tmp_dataset.AcquisitionNumber = 1
            tmp_dataset.InstanceNumber = i
            tmp_dataset.SeriesNumber = 1
            tmp_dataset.ImageOrientationPatient = orientation_mat
            tmp_dataset.ImagePositionPatient = [
                tmp_dataset.SliceLocation * i for i in position_vec]
            if system == Modality.CT:
                tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            elif system == Modality.MR:
                tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'OTHER']
            elif system == Modality.PT:
                tmp_dataset.ImageType = [
                    'ORIGINAL', 'PRIMARY', 'RECON', 'EMISSION']
            tmp_dataset.PixelSpacing = [
                self._pixel_spacing, self._pixel_spacing]
            tmp_dataset.PatientName = 'Doe^John'
            tmp_dataset.FrameOfReferenceUID = frame_of_ref_uid
            tmp_dataset.SOPClassUID = sop_classes[system][1]
            tmp_dataset.SOPInstanceUID = generate_uid()
            tmp_dataset.SeriesInstanceUID = series_uid
            tmp_dataset.StudyInstanceUID = study_uid
            tmp_dataset.BitsAllocated = bytes_per_voxel * 8
            tmp_dataset.BitsStored = bytes_per_voxel * 8
            tmp_dataset.HighBit = (bytes_per_voxel * 8 - 1)
            tmp_dataset.PixelRepresentation = 1
            tmp_dataset.Columns = cols
            tmp_dataset.Rows = rows
            tmp_dataset.SamplesPerPixel = 1
            tmp_dataset.AccessionNumber = '1{:05d}'.format(frameset_idx)
            tmp_dataset.AcquisitionDate = date_
            tmp_dataset.AcquisitionTime = datetime.now().time()
            tmp_dataset.AdditionalPatientHistory = 'UTERINE CA PRE-OP EVAL'
            tmp_dataset.ContentDate = date_
            tmp_dataset.ContentTime = datetime.now().time()
            tmp_dataset.Manufacturer = 'Mnufacturer'
            tmp_dataset.ManufacturerModelName = 'Model'
            tmp_dataset.Modality = sop_classes[system][0]
            tmp_dataset.PatientAge = '064Y'
            tmp_dataset.PatientBirthDate = date_ - age
            tmp_dataset.PatientID = 'ID{:05d}'.format(frameset_idx)
            tmp_dataset.PatientIdentityRemoved = 'YES'
            tmp_dataset.PatientPosition = 'FFS'
            tmp_dataset.PatientSex = 'F'
            tmp_dataset.PhotometricInterpretation = 'MONOCHROME2'
            tmp_dataset.PixelData = pixel_array
            tmp_dataset.PositionReferenceIndicator = 'XY'
            tmp_dataset.ProtocolName = 'some protocole'
            tmp_dataset.ReferringPhysicianName = ''
            tmp_dataset.SeriesDate = date_
            tmp_dataset.SeriesDescription = \
                'test series_frameset{:05d}'.format(frameset_idx)
            tmp_dataset.SeriesTime = time_
            tmp_dataset.SoftwareVersions = '01'
            tmp_dataset.SpecificCharacterSet = 'ISO_IR 100'
            tmp_dataset.StudyDate = date_
            tmp_dataset.StudyDescription = 'test study'
            tmp_dataset.StudyID = ''
            if (system == Modality.CT):
                tmp_dataset.RescaleIntercept = 0
                tmp_dataset.RescaleSlope = 1
            tmp_dataset.StudyTime = time_
            output_dataset.append(tmp_dataset)
        return output_dataset

    def generate_mixed_framesets(
            self, system: Modality,
            frame_set_count: int, parallel: bool = True,
            flatten_output: bool = True) -> list:
        out = []
        orients = [
            self._z_orientation_mat,
            self._y_orientation_mat,
            self._x_orientation_mat, ]
        poses = [
            self._z_position_vec,
            self._y_position_vec,
            self._x_position_vec,
        ]
        se_uid = generate_uid()
        for i in range(frame_set_count):
            if parallel:
                pos = poses[0]
                orient = orients[0]
            else:
                pos = poses[i % len(poses)]
                orient = orients[i % len(orients)]
            if flatten_output:
                out.extend(
                    self._generate_frameset(
                        system, orient, pos, se_uid, i * 50, i)
                )
            else:
                out.append(
                    self._generate_frameset(
                        system, orient, pos, se_uid, i * 50, i)
                )
        return out


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
            assert sop._DicomHelper.isequal(v1.value, v2.value)
            assert not sop._DicomHelper.isequal(v1.value, v3.value)


class TestFrameSetCollection(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_frameset_detection(self) -> None:
        data_generator = DicomGenerator()
        for i in range(1, 10):
            data = data_generator.generate_mixed_framesets(
                Modality.CT, i, True, True)
            fset_collection = sop._FrameSetCollection(data)
            assert len(fset_collection.frame_sets) == i

    def test_frameset_framecount_detection(self) -> None:
        for i in range(1, 10):
            data_generator = DicomGenerator(i)
            data = data_generator.generate_mixed_framesets(
                Modality.CT, 1, True, True)
            fset_collection = sop._FrameSetCollection(data)
            assert len(fset_collection.frame_sets) == 1
            assert len(fset_collection.frame_sets[0].frames) == i


class TestLegacyConvertedEnhanceImage(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._modalities = ('CT', 'MR', 'PET')
        self._dicom_generator = DicomGenerator(slice_per_frameset=5)
        self._ref_dataset_seq_CT = \
            self._dicom_generator.generate_mixed_framesets(Modality.CT, 1)
        self._ref_dataset_seq_MR = \
            self._dicom_generator.generate_mixed_framesets(Modality.MR, 1)
        self._ref_dataset_seq_PET = \
            self._dicom_generator.generate_mixed_framesets(Modality.PT, 1)
        self._output_series_instance_uid = generate_uid()
        self._output_sop_instance_uid = generate_uid()
        self._output_series_number = '1'
        self._output_instance_number = '1'

    def test_conversion(self) -> None:
        for i in range(1, 10):
            for j, m in enumerate(self._modalities):
                with self.subTest(m=m):
                    LegacyConverterClass = getattr(
                        sop,
                        "LegacyConvertedEnhanced{}Image".format(m)
                    )
                    data_generator = DicomGenerator(i)
                    data = data_generator.generate_mixed_framesets(
                        Modality(j), 1, True, True)
                    fset_collection = sop._FrameSetCollection(data)
                    assert len(fset_collection.frame_sets) == 1
                    assert len(fset_collection.frame_sets[0].frames) == i
                    convertor = LegacyConverterClass(
                        data,
                        generate_uid(),
                        555,
                        generate_uid(),
                        111)
                    assert convertor.NumberOfFrames == i
                    assert convertor.SOPClassUID == \
                        sop.LEGACY_ENHANCED_SOP_CLASS_UID_MAP[sop_classes[j][1]]

    def test_output_attributes(self) -> None:
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(
                  self, "_ref_dataset_seq_{}".format(m))
                multiframe_item = LegacyConverterClass(
                    ref_dataset_seq,
                    series_instance_uid=self._output_series_instance_uid,
                    series_number=self._output_instance_number,
                    sop_instance_uid=self._output_sop_instance_uid,
                    instance_number=self._output_instance_number)
                assert multiframe_item.SeriesInstanceUID == \
                    self._output_series_instance_uid
                assert multiframe_item.SOPInstanceUID == \
                    self._output_sop_instance_uid
                assert int(multiframe_item.SeriesNumber) == int(
                    self._output_series_number)
                assert int(multiframe_item.InstanceNumber) == int(
                    self._output_instance_number)

    def test_empty_dataset(self) -> None:
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        [],
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number)

    def test_wrong_modality(self) -> None:

        for j, m in enumerate(self._modalities):
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                next_idx = (j + 1) % len(self._modalities)
                ref_dataset_seq = getattr(
                    self, "_ref_dataset_seq_{}".format(
                        self._modalities[next_idx]))
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        ref_dataset_seq,
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number)

    def test_wrong_sop_class_uid(self) -> None:
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(
                    self, "_ref_dataset_seq_{}".format(m))
                tmp_orig_sop_class_id = ref_dataset_seq[0].SOPClassUID
                for ddss in ref_dataset_seq:
                    ddss.SOPClassUID = '1.2.3.4.5.6.7.8.9'
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        ref_dataset_seq,
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number)
                for ddss in ref_dataset_seq:
                    ddss.SOPClassUID = tmp_orig_sop_class_id

    def test_mixed_studies(self) -> None:
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(
                    self, "_ref_dataset_seq_{}".format(m))
                LegacyConverterClass(
                    legacy_datasets=ref_dataset_seq,
                    series_instance_uid=self._output_series_instance_uid,
                    series_number=self._output_instance_number,
                    sop_instance_uid=self._output_sop_instance_uid,
                    instance_number=self._output_instance_number)
                # second run with defected input
                tmp_orig_study_instance_uid = ref_dataset_seq[
                    0].StudyInstanceUID
                ref_dataset_seq[0].StudyInstanceUID = '1.2.3.4.5.6.7.8.9'
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        legacy_datasets=ref_dataset_seq,
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number)
                ref_dataset_seq[
                    0].StudyInstanceUID = tmp_orig_study_instance_uid

    # def test_mixed_series(self):
    #     for m in self._modalities:
    #         with self.subTest(m=m):
    #             LegacyConverterClass = getattr(
    #                 sop,
    #                 "LegacyConvertedEnhanced{}Image".format(m)
    #             )
    #             ref_dataset_seq = getattr(
    #               self, "_ref_dataset_seq_{}".format(m))
    #             # first run with intact input
    #             LegacyConverterClass(
    #                 legacy_datasets=ref_dataset_seq,
    #                 series_instance_uid=self._output_series_instance_uid,
    #                 series_number=self._output_instance_number,
    #                 sop_instance_uid=self._output_sop_instance_uid,
    #                 instance_number=self._output_instance_number)
    #             # second run with defected input
    #             tmp_series_instance_uid = ref_dataset_seq[0].SeriesInstanceUID
    #             ref_dataset_seq[0].SeriesInstanceUID = '1.2.3.4.5.6.7.8.9'
    #             with self.assertRaises(ValueError):
    #                 LegacyConverterClass(
    #                     legacy_datasets=ref_dataset_seq,
    #                     series_instance_uid=self._output_series_instance_uid,
    #                     series_number=self._output_instance_number,
    #                     sop_instance_uid=self._output_sop_instance_uid,
    #                     instance_number=self._output_instance_number)
    #             ref_dataset_seq[0].SeriesInstanceUID = tmp_series_instance_uid

    # def test_mixed_transfer_syntax(self):
    #     for m in self._modalities:
    #         with self.subTest(m=m):
    #             LegacyConverterClass = getattr(
    #                 sop,
    #                 "LegacyConvertedEnhanced{}Image".format(m)
    #             )
    #             ref_dataset_seq = getattr(
    #               self, "_ref_dataset_seq_{}".format(m))
    #             # first run with intact input
    #             LegacyConverterClass(
    #                 legacy_datasets=ref_dataset_seq,
    #                 series_instance_uid=self._output_series_instance_uid,
    #                 series_number=self._output_instance_number,
    #                 sop_instance_uid=self._output_sop_instance_uid,
    #                 instance_number=self._output_instance_number)
    #             # second run with defected input
    #             tmp_transfer_syntax_uid = ref_dataset_seq[
    #                 0].file_meta.TransferSyntaxUID
    #             ref_dataset_seq[
    #                 0].file_meta.TransferSyntaxUID = '1.2.3.4.5.6.7.8.9'
    #             with self.assertRaises(ValueError):
    #                 LegacyConverterClass(
    #                     legacy_datasets=ref_dataset_seq,
    #                     series_instance_uid=self._output_series_instance_uid,
    #                     series_number=self._output_instance_number,
    #                     sop_instance_uid=self._output_sop_instance_uid,
    #                     instance_number=self._output_instance_number)
    #             ref_dataset_seq[
    #                 0].file_meta.TransferSyntaxUID = tmp_transfer_syntax_uid
