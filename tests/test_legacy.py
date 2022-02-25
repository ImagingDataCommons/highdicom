import unittest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, UID
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


class TestLegacyConvertedEnhancedImage(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._modalities = ('CT', 'MR', 'PET')
        self._ref_dataset_seq_CT = \
            self.generate_common_dicom_dataset_series(3, Modality.CT)
        self._ref_dataset_seq_MR = \
            self.generate_common_dicom_dataset_series(3, Modality.MR)
        self._ref_dataset_seq_PET = \
            self.generate_common_dicom_dataset_series(3, Modality.PT)
        self._output_series_instance_uid = generate_uid()
        self._output_sop_instance_uid = generate_uid()
        self._output_series_number = 1
        self._output_instance_number = 1

    def test_output_attributes(self):
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(self, f"_ref_dataset_seq_{m}")

                multiframe_item = LegacyConverterClass(
                    legacy_datasets=ref_dataset_seq,
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

    def test_empty_dataset(self):
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

    def test_wrong_modality(self):

        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(self, f"_ref_dataset_seq_{m}")
                tmp_orig_modality = ref_dataset_seq[0].Modality
                ref_dataset_seq[0].Modality = ''
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        legacy_datasets=ref_dataset_seq,
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number)
                ref_dataset_seq[0].Modality = tmp_orig_modality

    def test_wrong_sop_class_uid(self):
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(self, f"_ref_dataset_seq_{m}")
                tmp_orig_sop_class_id = ref_dataset_seq[0].SOPClassUID
                ref_dataset_seq[0].SOPClassUID = '1.2.3.4.5.6.7.8.9'
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        legacy_datasets=ref_dataset_seq,
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number)
                ref_dataset_seq[0].SOPClassUID = tmp_orig_sop_class_id

    def test_mixed_studies(self):
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(self, f"_ref_dataset_seq_{m}")
                # first run with intact input

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

    def test_mixed_series(self):
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    "LegacyConvertedEnhanced{}Image".format(m)
                )
                ref_dataset_seq = getattr(self, f"_ref_dataset_seq_{m}")
                # first run with intact input
                LegacyConverterClass(
                    legacy_datasets=ref_dataset_seq,
                    series_instance_uid=self._output_series_instance_uid,
                    series_number=self._output_instance_number,
                    sop_instance_uid=self._output_sop_instance_uid,
                    instance_number=self._output_instance_number)
                # second run with defected input
                tmp_series_instance_uid = ref_dataset_seq[0].SeriesInstanceUID
                ref_dataset_seq[0].SeriesInstanceUID = '1.2.3.4.5.6.7.8.9'
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        legacy_datasets=ref_dataset_seq,
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number)
                ref_dataset_seq[0].SeriesInstanceUID = tmp_series_instance_uid

    def test_mixed_transfer_syntax(self):
        for m in self._modalities:
            with self.subTest(m=m):
                LegacyConverterClass = getattr(
                    sop,
                    f'LegacyConvertedEnhanced{m}Image'
                )
                ref_dataset_seq = getattr(self, f'_ref_dataset_seq_{m}')
                # first run with intact input
                LegacyConverterClass(
                    legacy_datasets=ref_dataset_seq,
                    series_instance_uid=self._output_series_instance_uid,
                    series_number=self._output_instance_number,
                    sop_instance_uid=self._output_sop_instance_uid,
                    instance_number=self._output_instance_number
                )
                # second run with defected input
                ref_item = ref_dataset_seq[0]
                tmp_transfer_syntax_uid = str(
                    ref_item.file_meta.TransferSyntaxUID
                )
                ref_item.file_meta.TransferSyntaxUID = '1.2.3.4.5.6.7.8.9'
                with self.assertRaises(ValueError):
                    LegacyConverterClass(
                        legacy_datasets=ref_dataset_seq,
                        series_instance_uid=self._output_series_instance_uid,
                        series_number=self._output_instance_number,
                        sop_instance_uid=self._output_sop_instance_uid,
                        instance_number=self._output_instance_number
                    )
                ref_item.file_meta.TransferSyntaxUID = tmp_transfer_syntax_uid

    def generate_common_dicom_dataset_series(
        self,
        slice_count: int,
        system: Modality
    ) -> list:
        output_dataset = []
        slice_pos = 0
        slice_thickness = 0
        study_uid = generate_uid()
        series_uid = generate_uid()
        frame_of_ref_uid = generate_uid()
        date_ = datetime.now().date()
        age = timedelta(days=45 * 365)
        time_ = datetime.now().time()
        cols = 2
        rows = 2
        bytes_per_voxel = 2

        for i in range(0, slice_count):
            file_meta = FileMetaDataset()
            pixel_array = b"\0" * cols * rows * bytes_per_voxel
            file_meta.MediaStorageSOPClassUID = UID(sop_classes[system][1])
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = generate_uid()

            tmp_dataset = FileDataset('', {}, file_meta=file_meta,
                                      preamble=pixel_array)
            tmp_dataset.file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.1")
            tmp_dataset.SliceLocation = slice_pos + i * slice_thickness
            tmp_dataset.SliceThickness = slice_thickness
            tmp_dataset.WindowCenter = 1
            tmp_dataset.WindowWidth = 2
            tmp_dataset.AcquisitionNumber = 1
            tmp_dataset.InstanceNumber = i
            tmp_dataset.SeriesNumber = 1
            tmp_dataset.ImageOrientationPatient = [1.000000, 0.000000, 0.000000,
                                                   0.000000, 1.000000, 0.000000]
            tmp_dataset.ImagePositionPatient = [0.0, 0.0,
                                                tmp_dataset.SliceLocation]
            tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            tmp_dataset.PixelSpacing = [1, 1]
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
            tmp_dataset.AccessionNumber = '2'
            tmp_dataset.AcquisitionDate = date_
            tmp_dataset.AcquisitionTime = datetime.now().time()
            tmp_dataset.AdditionalPatientHistory = 'UTERINE CA PRE-OP EVAL'
            tmp_dataset.ContentDate = date_
            tmp_dataset.ContentTime = datetime.now().time()
            tmp_dataset.Manufacturer = 'Manufacturer'
            tmp_dataset.ManufacturerModelName = 'Model'
            tmp_dataset.Modality = sop_classes[system][0]
            tmp_dataset.PatientAge = '064Y'
            tmp_dataset.PatientBirthDate = date_ - age
            tmp_dataset.PatientID = 'ID0001'
            tmp_dataset.PatientIdentityRemoved = 'YES'
            tmp_dataset.PatientPosition = 'FFS'
            tmp_dataset.PatientSex = 'F'
            tmp_dataset.PhotometricInterpretation = 'MONOCHROME2'
            tmp_dataset.PixelData = pixel_array
            tmp_dataset.PositionReferenceIndicator = 'XY'
            tmp_dataset.ProtocolName = 'some protocole'
            tmp_dataset.ReferringPhysicianName = ''
            tmp_dataset.SeriesDate = date_
            tmp_dataset.SeriesDescription = 'test series '
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
