from copy import deepcopy
from datetime import datetime, timedelta
import enum

from pydicom import FileDataset, FileMetaDataset, Dataset
from pydicom.uid import generate_uid, UID
from pydicom.valuerep import DSfloat
from highdicom.legacy import (
    LegacyConvertedEnhancedCTImage,
    LegacyConvertedEnhancedMRImage,
    LegacyConvertedEnhancedPETImage,
)

import pytest


class Modality(enum.IntEnum):
    CT = 0
    MR = 1
    PT = 2


MODALITY_LEGACY_SOP_CLASS_MAP = {
    Modality.CT: '1.2.840.10008.5.1.4.1.1.2',
    Modality.MR: '1.2.840.10008.5.1.4.1.1.4',
    Modality.PT: '1.2.840.10008.5.1.4.1.1.128',
}
MODALITY_ENHANCED_SOP_CLASS_MAP = {
    Modality.CT: '1.2.840.10008.5.1.4.1.1.2.2',
    Modality.MR: '1.2.840.10008.5.1.4.1.1.4.4',
    Modality.PT: '1.2.840.10008.5.1.4.1.1.128.1',
}
MODALITY_CLASS_MAP = {
    Modality.CT: LegacyConvertedEnhancedCTImage,
    Modality.MR: LegacyConvertedEnhancedMRImage,
    Modality.PT: LegacyConvertedEnhancedPETImage,
}


class DicomGenerator:

    def __init__(
        self,
        slice_per_frameset: int = 3,
        slice_thickness: float = 0.1,
        pixel_spacing: float = 0.1,
        row: int = 2,
        col: int = 2,
    ) -> None:
        self._slice_per_frameset = slice_per_frameset
        self._slice_thickness = slice_thickness
        self._pixel_spacing = pixel_spacing
        self._row = row
        self._col = col
        self._study_uid = generate_uid()
        self._z_orientation_mat = [
            1.000000, 0.000000, 0.000000,
            0.000000, 1.000000, 0.000000
        ]
        self._z_position_vec = [0.0, 0.0, 1.0]
        self._y_orientation_mat = [
            0.000000, 0.000000, 1.000000,
            1.000000, 0.000000, 0.000000
        ]
        self._y_position_vec = [0.0, 1.0, 0.0]
        self._x_orientation_mat = [
            0.000000, 1.000000, 0.000000,
            0.000000, 0.000000, 1.000000
        ]
        self._x_position_vec = [1.0, 0.0, 0.0]

    def _generate_frameset(
        self,
        modality: Modality,
        orientation_mat: list,
        position_vec: list,
        series_uid: str,
        first_slice_offset: float = 0,
        frameset_idx: int = 0
    ) -> list:
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

        for i in range(self._slice_per_frameset):
            file_meta = Dataset()
            pixel_array = b"\0" * cols * rows * bytes_per_voxel
            file_meta.MediaStorageSOPClassUID = MODALITY_LEGACY_SOP_CLASS_MAP[
                modality
            ]
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = generate_uid()
            tmp_dataset = FileDataset(
                '', {}, file_meta=file_meta, preamble=pixel_array
            )
            tmp_dataset.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
            tmp_dataset.SliceLocation = DSfloat(
                slice_pos + i * slice_thickness,
                auto_format=True
            )
            tmp_dataset.SliceThickness = DSfloat(
                slice_thickness,
                auto_format=True
            )
            tmp_dataset.WindowCenter = 1
            tmp_dataset.WindowWidth = 2
            tmp_dataset.AcquisitionNumber = 1
            tmp_dataset.InstanceNumber = i
            tmp_dataset.SeriesNumber = 1
            tmp_dataset.ImageOrientationPatient = orientation_mat
            tmp_dataset.ImagePositionPatient = [
                DSfloat(tmp_dataset.SliceLocation * i, auto_format=True)
                for i in position_vec
            ]
            if modality == Modality.CT:
                tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            elif modality == Modality.MR:
                tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'OTHER']
            elif modality == Modality.PT:
                tmp_dataset.ImageType = [
                    'ORIGINAL', 'PRIMARY', 'RECON', 'EMISSION']

            tmp_dataset.PixelSpacing = [
                self._pixel_spacing, self._pixel_spacing
            ]
            tmp_dataset.PatientName = 'Doe^John'
            tmp_dataset.FrameOfReferenceUID = frame_of_ref_uid
            tmp_dataset.SOPClassUID = MODALITY_LEGACY_SOP_CLASS_MAP[modality]
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
            tmp_dataset.Manufacturer = 'Manufacturer'
            tmp_dataset.ManufacturerModelName = 'Model'
            tmp_dataset.Modality = modality.name
            tmp_dataset.PatientAge = '064Y'
            tmp_dataset.PatientBirthDate = date_ - age
            tmp_dataset.PatientID = 'ID{:05d}'.format(frameset_idx)
            tmp_dataset.PatientIdentityRemoved = 'YES'
            tmp_dataset.PatientPosition = 'FFS'
            tmp_dataset.PatientSex = 'F'
            tmp_dataset.PhotometricInterpretation = 'MONOCHROME2'
            tmp_dataset.PixelData = pixel_array
            tmp_dataset.PositionReferenceIndicator = 'XY'
            tmp_dataset.ProtocolName = 'some protocol'
            tmp_dataset.ReferringPhysicianName = ''
            tmp_dataset.SeriesDate = date_
            tmp_dataset.SeriesDescription = (
                f'test series_frameset{frameset_idx:05d}'
            )
            tmp_dataset.SeriesTime = time_
            tmp_dataset.SoftwareVersions = '01'
            tmp_dataset.SpecificCharacterSet = 'ISO_IR 100'
            tmp_dataset.StudyDate = date_
            tmp_dataset.StudyDescription = 'test study'
            tmp_dataset.StudyID = ''
            if (modality == Modality.CT):
                tmp_dataset.RescaleIntercept = 0
                tmp_dataset.RescaleSlope = 1

            tmp_dataset.StudyTime = time_
            output_dataset.append(tmp_dataset)

        return output_dataset

    def generate_mixed_framesets(
        self,
        modality: Modality,
        frame_set_count: int,
        parallel: bool = True,
        flatten_output: bool = True,
    ) -> list:
        out = []
        orients = [
            self._z_orientation_mat,
            self._y_orientation_mat,
            self._x_orientation_mat,
        ]
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
                        modality, orient, pos, se_uid, i * 50, i
                    )
                )
            else:
                out.append(
                    self._generate_frameset(
                        modality, orient, pos, se_uid, i * 50, i
                    )
                )

        return out


class TestLegacyConvertedEnhancedImage:

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self._dicom_generator = DicomGenerator(slice_per_frameset=5)
        self._ref_dataset_seq = {
            m: self._dicom_generator.generate_mixed_framesets(m, 1)
            for m in Modality
        }
        self._output_series_instance_uid = generate_uid()
        self._output_sop_instance_uid = generate_uid()
        self._output_series_number = 1
        self._output_instance_number = 1

    @staticmethod
    @pytest.fixture(params=[Modality.MR, Modality.CT, Modality.PT])
    def modality(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=range(1, 7))
    def number_of_frames(request):
        return request.param

    def test_conversion(
        self,
        modality: Modality,
        number_of_frames: int
    ) -> None:
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        data_generator = DicomGenerator(number_of_frames)
        data = data_generator.generate_mixed_framesets(
            modality, 1, True, True
        )
        converted = LegacyConverterClass(
            data,
            generate_uid(),
            555,
            generate_uid(),
            111
        )
        assert converted.NumberOfFrames == number_of_frames
        assert (
            converted.SOPClassUID == MODALITY_ENHANCED_SOP_CLASS_MAP[modality]
        )
        sfgs = converted.SharedFunctionalGroupsSequence[0]
        assert not hasattr(sfgs, 'FrameAnatomySequence')

    def test_output_attributes(self, modality: Modality) -> None:
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = self._ref_dataset_seq[modality]
        multiframe_item = LegacyConverterClass(
            ref_dataset_seq,
            series_instance_uid=self._output_series_instance_uid,
            series_number=self._output_instance_number,
            sop_instance_uid=self._output_sop_instance_uid,
            instance_number=self._output_instance_number
        )
        assert (
            multiframe_item.SeriesInstanceUID ==
            self._output_series_instance_uid
        )
        assert (
            multiframe_item.SOPInstanceUID ==
            self._output_sop_instance_uid
        )
        assert (
            int(multiframe_item.SeriesNumber) ==
            int(self._output_series_number)
        )
        assert (
            int(multiframe_item.InstanceNumber) ==
            int(self._output_instance_number)
        )

    def test_empty_dataset(self, modality: Modality) -> None:
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]

        with pytest.raises(ValueError):
            LegacyConverterClass(
                [],
                series_instance_uid=self._output_series_instance_uid,
                series_number=self._output_instance_number,
                sop_instance_uid=self._output_sop_instance_uid,
                instance_number=self._output_instance_number
            )

    def test_wrong_modality(self, modality: Modality) -> None:
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = self._ref_dataset_seq[modality]
        ref_dataset_seq[0].Modality = ''
        next_idx = (modality.value + 1) % 3
        ref_dataset_seq = self._ref_dataset_seq[Modality(next_idx)]
        with pytest.raises(ValueError):
            LegacyConverterClass(
                ref_dataset_seq,
                series_instance_uid=self._output_series_instance_uid,
                series_number=self._output_instance_number,
                sop_instance_uid=self._output_sop_instance_uid,
                instance_number=self._output_instance_number
            )

    def test_wrong_sop_class_uid(self, modality: Modality) -> None:
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = self._ref_dataset_seq[modality]
        tmp_orig_sop_class_id = ref_dataset_seq[0].SOPClassUID
        for ddss in ref_dataset_seq:
            ddss.SOPClassUID = '1.2.3.4.5.6.7.8.9'
        with pytest.raises(ValueError):
            LegacyConverterClass(
                ref_dataset_seq,
                series_instance_uid=self._output_series_instance_uid,
                series_number=self._output_instance_number,
                sop_instance_uid=self._output_sop_instance_uid,
                instance_number=self._output_instance_number
            )

        for ddss in ref_dataset_seq:
            ddss.SOPClassUID = tmp_orig_sop_class_id

    def test_mixed_studies(self, modality: Modality) -> None:
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = self._ref_dataset_seq[modality]
        # first run with intact input

        LegacyConverterClass(
            legacy_datasets=ref_dataset_seq,
            series_instance_uid=self._output_series_instance_uid,
            series_number=self._output_instance_number,
            sop_instance_uid=self._output_sop_instance_uid,
            instance_number=self._output_instance_number
        )
        # second run with defected input
        tmp_orig_study_instance_uid = (
            ref_dataset_seq[0]
            .StudyInstanceUID
        )
        ref_dataset_seq[0].StudyInstanceUID = '1.2.3.4.5.6.7.8.9'
        with pytest.raises(ValueError):
            LegacyConverterClass(
                legacy_datasets=ref_dataset_seq,
                series_instance_uid=self._output_series_instance_uid,
                series_number=self._output_instance_number,
                sop_instance_uid=self._output_sop_instance_uid,
                instance_number=self._output_instance_number
            )
        ref_dataset_seq[0].StudyInstanceUID = (
            tmp_orig_study_instance_uid
        )

    def test_mixed_series(self, modality: Modality):
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = self._ref_dataset_seq[modality]
        # first run with intact input
        LegacyConverterClass(
            legacy_datasets=ref_dataset_seq,
            series_instance_uid=self._output_series_instance_uid,
            series_number=self._output_instance_number,
            sop_instance_uid=self._output_sop_instance_uid,
            instance_number=self._output_instance_number
        )

    def test_mixed_transfer_syntax(self, modality: Modality):
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = self._ref_dataset_seq[modality]
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
        with pytest.raises(ValueError):
            LegacyConverterClass(
                legacy_datasets=ref_dataset_seq,
                series_instance_uid=self._output_series_instance_uid,
                series_number=self._output_instance_number,
                sop_instance_uid=self._output_sop_instance_uid,
                instance_number=self._output_instance_number
            )

        ref_item.file_meta.TransferSyntaxUID = tmp_transfer_syntax_uid

    def test_body_part_mapping(self, modality: Modality):
        """Test that BodyPartExamined is correctly mapped to a coded
        AnatomicRegionSequence.

        """
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = deepcopy(self._ref_dataset_seq[modality])

        body_part_examined = 'ANTECUBITALV'
        expected_scheme_designator = 'SCT'
        expected_code_value = '128553008'
        expected_code_meaning = 'Antecubital vein'

        for dcm in ref_dataset_seq:
            dcm.BodyPartExamined = body_part_examined

        converted = LegacyConverterClass(
            legacy_datasets=ref_dataset_seq,
            series_instance_uid=self._output_series_instance_uid,
            series_number=self._output_instance_number,
            sop_instance_uid=self._output_sop_instance_uid,
            instance_number=self._output_instance_number
        )

        sfgs = converted.SharedFunctionalGroupsSequence[0]
        assert hasattr(sfgs, 'FrameAnatomySequence')
        fr_an_seq = sfgs.FrameAnatomySequence[0]
        assert (
            fr_an_seq.AnatomicRegionSequence[0].CodeValue ==
            expected_code_value
        )
        assert (
            fr_an_seq.AnatomicRegionSequence[0].CodeMeaning ==
            expected_code_meaning
        )
        assert (
            fr_an_seq.AnatomicRegionSequence[0].CodingSchemeDesignator ==
            expected_scheme_designator
        )
        assert fr_an_seq.FrameLaterality == 'U'  # unimodal (default)

    def test_laterality_from_region_modifier(self, modality: Modality):
        """Test that the laterality is correctly inferred from the
        AnatomicRegionSequence if present in the source file.

        """
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = deepcopy(self._ref_dataset_seq[modality])

        region_item = Dataset()
        region_item.CodeMeaning = 'Kidney'
        region_item.CodeValue = '64033007'
        region_item.CodingSchemeDesignator = 'SCT'
        modifier_item = Dataset()
        modifier_item.CodeMeaning = 'Left'
        modifier_item.CodeValue = '7771000'
        modifier_item.CodingSchemeDesignator = 'SCT'
        region_item.AnatomicRegionModifierSequence = [modifier_item]

        for dcm in ref_dataset_seq:
            dcm.AnatomicRegionSequence = [deepcopy(region_item)]

        converted = LegacyConverterClass(
            legacy_datasets=ref_dataset_seq,
            series_instance_uid=self._output_series_instance_uid,
            series_number=self._output_instance_number,
            sop_instance_uid=self._output_sop_instance_uid,
            instance_number=self._output_instance_number
        )

        sfgs = converted.SharedFunctionalGroupsSequence[0]
        assert hasattr(sfgs, 'FrameAnatomySequence')
        fr_an_seq = sfgs.FrameAnatomySequence[0]
        assert (
            fr_an_seq.AnatomicRegionSequence[0].CodeValue ==
            '64033007'
        )
        assert (
            fr_an_seq.AnatomicRegionSequence[0].CodeMeaning ==
            'Kidney'
        )
        assert (
            fr_an_seq.AnatomicRegionSequence[0].CodingSchemeDesignator ==
            'SCT'
        )
        assert fr_an_seq.FrameLaterality == 'L'  # inferred from modifier

    def test_laterality_from_structure_modifier(self, modality: Modality):
        """Test that the laterality is correctly inferred from the
        PrimaryAnatomicStructureSequence if present in the source file.

        """
        LegacyConverterClass = MODALITY_CLASS_MAP[modality]
        ref_dataset_seq = deepcopy(self._ref_dataset_seq[modality])

        region_item = Dataset()
        region_item.CodeMeaning = 'Thorax'
        region_item.CodeValue = '816094009'
        region_item.CodingSchemeDesignator = 'SCT'
        structure_item = Dataset()
        structure_item.CodeMeaning = 'Lung'
        structure_item.CodeValue = '39607008'
        structure_item.CodingSchemeDesignator = 'SCT'
        modifier_item = Dataset()
        modifier_item.CodeMeaning = 'Bilateral'
        modifier_item.CodeValue = '51440002'
        modifier_item.CodingSchemeDesignator = 'SCT'
        structure_item.PrimaryAnatomicStructureModifierSequence = [
            modifier_item
        ]

        for dcm in ref_dataset_seq:
            dcm.AnatomicRegionSequence = [deepcopy(region_item)]
            dcm.PrimaryAnatomicStructureSequence = [deepcopy(structure_item)]

        converted = LegacyConverterClass(
            legacy_datasets=ref_dataset_seq,
            series_instance_uid=self._output_series_instance_uid,
            series_number=self._output_instance_number,
            sop_instance_uid=self._output_sop_instance_uid,
            instance_number=self._output_instance_number
        )

        sfgs = converted.SharedFunctionalGroupsSequence[0]
        assert hasattr(sfgs, 'FrameAnatomySequence')
        fr_an_seq = sfgs.FrameAnatomySequence[0]
        assert (
            fr_an_seq.PrimaryAnatomicStructureSequence[0].CodeValue ==
            '39607008'
        )
        assert (
            fr_an_seq.PrimaryAnatomicStructureSequence[0].CodeMeaning ==
            'Lung'
        )
        assert (
            fr_an_seq
            .PrimaryAnatomicStructureSequence[0]
            .CodingSchemeDesignator ==
            'SCT'
        )
        assert fr_an_seq.FrameLaterality == 'B'  # inferred from modifier

    def generate_common_dicom_dataset_series(
        self,
        slice_count: int,
        modality: Modality
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

        for i in range(slice_count):
            file_meta = FileMetaDataset()
            pixel_array = b"\0" * cols * rows * bytes_per_voxel
            file_meta.MediaStorageSOPClassUID = UID(
                MODALITY_LEGACY_SOP_CLASS_MAP[modality][1]
            )
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = generate_uid()

            tmp_dataset = FileDataset(
                '', {}, file_meta=file_meta, preamble=pixel_array
            )
            tmp_dataset.file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.1")
            tmp_dataset.SliceLocation = slice_pos + i * slice_thickness
            tmp_dataset.SliceThickness = slice_thickness
            tmp_dataset.WindowCenter = 1
            tmp_dataset.WindowWidth = 2
            tmp_dataset.AcquisitionNumber = 1
            tmp_dataset.InstanceNumber = i
            tmp_dataset.SeriesNumber = 1
            tmp_dataset.ImageOrientationPatient = [
                1.000000, 0.000000, 0.000000,
                0.000000, 1.000000, 0.000000
            ]
            tmp_dataset.ImagePositionPatient = [
                0.0, 0.0,
                tmp_dataset.SliceLocation
            ]
            tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            tmp_dataset.PixelSpacing = [1, 1]
            tmp_dataset.PatientName = 'Doe^John'
            tmp_dataset.FrameOfReferenceUID = frame_of_ref_uid
            tmp_dataset.SOPClassUID = MODALITY_LEGACY_SOP_CLASS_MAP[modality][1]
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
            tmp_dataset.Modality = MODALITY_LEGACY_SOP_CLASS_MAP[modality][0]
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
            if (modality == Modality.CT):
                tmp_dataset.RescaleIntercept = 0
                tmp_dataset.RescaleSlope = 1

            tmp_dataset.StudyTime = time_
            output_dataset.append(tmp_dataset)

        return output_dataset
