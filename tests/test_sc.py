from pathlib import Path
from io import BytesIO
import unittest

import numpy as np
import pytest
from pydicom.encaps import generate_pixel_data_frame
from pydicom.filereader import dcmread
from pydicom.uid import (
    RLELossless,
    JPEGBaseline8Bit,
    JPEG2000Lossless,
    JPEGLSLossless,
)
from pydicom.valuerep import DA, TM

from highdicom import SpecimenDescription
from highdicom.sc import SCImage
from highdicom import UID
from highdicom.frame import decode_frame


class TestSCImage(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._rgb_pixel_array = np.zeros((10, 10, 3), dtype=np.uint8)
        self._monochrome_pixel_array = np.zeros((10, 10), dtype=np.uint16)
        self._study_instance_uid = UID()
        self._series_instance_uid = UID()
        self._sop_instance_uid = UID()
        self._series_number = int(np.random.choice(100)) + 1
        self._instance_number = int(np.random.choice(100)) + 1
        self._manufacturer = 'ABC'
        self._laterality = 'L'
        self._pixel_spacing = [0.5, 0.5]
        self._patient_orientation = ['A', 'R']
        self._container_identifier = str(np.random.choice(100))
        self._specimen_identifier = str(np.random.choice(100))
        self._specimen_uid = UID()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._ref_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )

    @staticmethod
    def get_array_after_writing(instance):
        # Write DICOM object to buffer, read it again and reconstruct the mask
        with BytesIO() as fp:
            instance.save_as(fp)
            fp.seek(0)
            ds = dcmread(fp)

        decoded_frame_arrays = [
            decode_frame(
                value,
                transfer_syntax_uid=ds.file_meta.TransferSyntaxUID,
                rows=ds.Rows,
                columns=ds.Columns,
                samples_per_pixel=ds.SamplesPerPixel,
                bits_allocated=ds.BitsAllocated,
                bits_stored=ds.BitsStored,
                photometric_interpretation=ds.PhotometricInterpretation,
                pixel_representation=ds.PixelRepresentation,
                planar_configuration=getattr(ds, 'PlanarConfiguration', None)
            )
            for value in generate_pixel_data_frame(instance.PixelData)
        ]
        if len(decoded_frame_arrays) > 1:
            return np.stack(decoded_frame_arrays)
        return decoded_frame_arrays[0]

    def test_construct_rgb_patient(self):
        bits_allocated = 8
        photometric_interpretation = 'RGB'
        coordinate_system = 'PATIENT'
        instance = SCImage(
            pixel_array=self._rgb_pixel_array,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            laterality=self._laterality,
            pixel_spacing=self._pixel_spacing,
        )
        assert instance.BitsAllocated == bits_allocated
        assert instance.SamplesPerPixel == 3
        assert instance.PlanarConfiguration == 0
        assert instance.PhotometricInterpretation == photometric_interpretation
        assert instance.StudyInstanceUID == self._study_instance_uid
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.Laterality == self._laterality
        assert instance.PatientOrientation == self._patient_orientation
        assert instance.AccessionNumber is None
        assert instance.PatientName is None
        assert instance.PatientSex is None
        assert instance.StudyTime is None
        assert instance.StudyTime is None
        assert instance.PixelSpacing == [0.5, 0.5]
        assert instance.PixelData == self._rgb_pixel_array.tobytes()
        with pytest.raises(AttributeError):
            instance.ContainerIdentifier
            instance.SpecimenDescriptionSequence
            instance.ContainerTypeCodeSequence
            instance.IssuerOfTheContainerIdentifierSequence

    def test_construct_rgb_patient_missing_parameter(self):
        with pytest.raises(TypeError):
            bits_allocated = 8
            photometric_interpretation = 'RGB'
            coordinate_system = 'PATIENT'
            SCImage(
                pixel_array=self._rgb_pixel_array,
                photometric_interpretation=photometric_interpretation,
                bits_allocated=bits_allocated,
                coordinate_system=coordinate_system,
                study_instance_uid=self._study_instance_uid,
                series_instance_uid=self._series_instance_uid,
                sop_instance_uid=self._sop_instance_uid,
                series_number=self._series_number,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
            )

    def test_construct_rgb_slide_single_specimen(self):
        bits_allocated = 8
        photometric_interpretation = 'RGB'
        coordinate_system = 'SLIDE'
        specimen_description = SpecimenDescription(
            specimen_id=self._specimen_identifier,
            specimen_uid=self._specimen_uid
        )
        instance = SCImage(
            pixel_array=self._rgb_pixel_array,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            container_identifier=self._container_identifier,
            specimen_descriptions=[specimen_description]
        )
        assert instance.BitsAllocated == bits_allocated
        assert instance.SamplesPerPixel == 3
        assert instance.PlanarConfiguration == 0
        assert instance.PhotometricInterpretation == photometric_interpretation
        assert instance.StudyInstanceUID == self._study_instance_uid
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.ContainerIdentifier == self._container_identifier
        assert len(instance.ContainerTypeCodeSequence) == 1
        assert len(instance.IssuerOfTheContainerIdentifierSequence) == 0
        assert len(instance.SpecimenDescriptionSequence) == 1
        specimen_item = instance.SpecimenDescriptionSequence[0]
        assert specimen_item.SpecimenIdentifier == self._specimen_identifier
        assert specimen_item.SpecimenUID == self._specimen_uid
        assert instance.AccessionNumber is None
        assert instance.PatientName is None
        assert instance.PatientSex is None
        assert instance.StudyTime is None
        assert instance.StudyTime is None
        assert instance.PixelData == self._rgb_pixel_array.tobytes()
        with pytest.raises(AttributeError):
            instance.Laterality
            instance.PatientOrientation

    def test_construct_rgb_slide_single_specimen_missing_parameter(self):
        bits_allocated = 8
        photometric_interpretation = 'RGB'
        coordinate_system = 'SLIDE'
        specimen_description = SpecimenDescription(
            specimen_id=self._specimen_identifier,
            specimen_uid=self._specimen_uid
        )
        with pytest.raises(TypeError):
            SCImage(
                pixel_array=self._rgb_pixel_array,
                photometric_interpretation=photometric_interpretation,
                bits_allocated=bits_allocated,
                coordinate_system=coordinate_system,
                study_instance_uid=self._study_instance_uid,
                series_instance_uid=self._series_instance_uid,
                sop_instance_uid=self._sop_instance_uid,
                series_number=self._series_number,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                specimen_descriptions=[specimen_description]
            )

    def test_construct_rgb_slide_single_specimen_missing_parameter_1(self):
        bits_allocated = 8
        photometric_interpretation = 'RGB'
        coordinate_system = 'SLIDE'
        with pytest.raises(TypeError):
            SCImage(
                pixel_array=self._rgb_pixel_array,
                photometric_interpretation=photometric_interpretation,
                bits_allocated=bits_allocated,
                coordinate_system=coordinate_system,
                study_instance_uid=self._study_instance_uid,
                series_instance_uid=self._series_instance_uid,
                sop_instance_uid=self._sop_instance_uid,
                series_number=self._series_number,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                container_identifier=self._container_identifier,
            )

    def test_construct_monochrome_patient(self):
        bits_allocated = 12
        photometric_interpretation = 'MONOCHROME2'
        coordinate_system = 'PATIENT'
        instance = SCImage(
            pixel_array=self._monochrome_pixel_array,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation
        )
        assert instance.BitsAllocated == bits_allocated
        assert instance.SamplesPerPixel == 1
        assert instance.PhotometricInterpretation == photometric_interpretation
        assert instance.StudyInstanceUID == self._study_instance_uid
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.PatientOrientation == self._patient_orientation
        assert instance.AccessionNumber is None
        assert instance.PatientName is None
        assert instance.PatientSex is None
        assert instance.StudyTime is None
        assert instance.StudyDate is None
        assert instance.PixelData == self._monochrome_pixel_array.tobytes()
        with pytest.raises(AttributeError):
            instance.ContainerIdentifier
            instance.SpecimenDescriptionSequence
            instance.ContainerTypeCodeSequence
            instance.IssuerOfTheContainerIdentifierSequence

    def test_monochrome_rle(self):
        bits_allocated = 8  # RLE requires multiple of 8 bits
        photometric_interpretation = 'MONOCHROME2'
        coordinate_system = 'PATIENT'
        frame = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=RLELossless
        )

        assert instance.file_meta.TransferSyntaxUID == RLELossless

        assert np.array_equal(
            self.get_array_after_writing(instance),
            frame
        )

    def test_rgb_rle(self):
        bits_allocated = 8  # RLE requires multiple of 8 bits
        photometric_interpretation = 'RGB'
        coordinate_system = 'PATIENT'
        frame = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=RLELossless
        )

        assert instance.file_meta.TransferSyntaxUID == RLELossless

        assert np.array_equal(
            self.get_array_after_writing(instance),
            frame
        )

    def test_monochrome_jpeg_baseline(self):
        bits_allocated = 8
        photometric_interpretation = 'MONOCHROME2'
        coordinate_system = 'PATIENT'
        frame = np.zeros((256, 256), dtype=np.uint8)
        frame[25:55, 25:55] = 255
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=JPEGBaseline8Bit
        )

        assert instance.file_meta.TransferSyntaxUID == JPEGBaseline8Bit

        assert np.allclose(
            self.get_array_after_writing(instance),
            frame,
            atol=5  # tolerance for lossy compression
        )

    def test_rgb_jpeg_baseline(self):
        bits_allocated = 8
        photometric_interpretation = 'YBR_FULL_422'
        coordinate_system = 'PATIENT'
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[25:55, 25:55, 0] = 255
        frame[35:65, 35:55, 1] = 255
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=JPEGBaseline8Bit
        )

        assert instance.file_meta.TransferSyntaxUID == JPEGBaseline8Bit

        reread_frame = self.get_array_after_writing(instance)
        np.testing.assert_allclose(frame, reread_frame, rtol=1.2)

    def test_monochrome_jpeg2000(self):
        bits_allocated = 8
        photometric_interpretation = 'MONOCHROME2'
        coordinate_system = 'PATIENT'
        frame = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=JPEG2000Lossless
        )

        assert instance.file_meta.TransferSyntaxUID == JPEG2000Lossless

        assert np.array_equal(
            self.get_array_after_writing(instance),
            frame
        )

    def test_rgb_jpeg2000(self):
        bits_allocated = 8
        photometric_interpretation = 'YBR_FULL'
        coordinate_system = 'PATIENT'
        frame = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=JPEG2000Lossless
        )

        assert instance.file_meta.TransferSyntaxUID == JPEG2000Lossless

        assert np.array_equal(
            self.get_array_after_writing(instance),
            frame
        )

    def test_monochrome_jpegls(self):
        bits_allocated = 16
        photometric_interpretation = 'MONOCHROME2'
        coordinate_system = 'PATIENT'
        frame = np.random.randint(0, 2**16, size=(256, 256), dtype=np.uint16)
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=JPEGLSLossless
        )

        assert instance.file_meta.TransferSyntaxUID == JPEGLSLossless

        assert np.array_equal(
            self.get_array_after_writing(instance),
            frame
        )

    def test_rgb_jpegls(self):
        bits_allocated = 8
        photometric_interpretation = 'YBR_FULL'
        coordinate_system = 'PATIENT'
        frame = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        instance = SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            study_instance_uid=self._study_instance_uid,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            transfer_syntax_uid=JPEGLSLossless
        )

        assert instance.file_meta.TransferSyntaxUID == JPEGLSLossless

        assert np.array_equal(
            self.get_array_after_writing(instance),
            frame
        )

    def test_construct_rgb_from_ref_dataset(self):
        bits_allocated = 8
        photometric_interpretation = 'RGB'
        coordinate_system = 'PATIENT'
        instance = SCImage.from_ref_dataset(
            ref_dataset=self._ref_dataset,
            pixel_array=self._rgb_pixel_array,
            photometric_interpretation=photometric_interpretation,
            bits_allocated=bits_allocated,
            coordinate_system=coordinate_system,
            series_instance_uid=self._series_instance_uid,
            sop_instance_uid=self._sop_instance_uid,
            series_number=self._series_number,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            patient_orientation=self._patient_orientation,
            laterality=self._laterality
        )
        assert instance.BitsAllocated == bits_allocated
        assert instance.SamplesPerPixel == 3
        assert instance.PlanarConfiguration == 0
        assert instance.PhotometricInterpretation == photometric_interpretation
        assert instance.StudyInstanceUID == self._ref_dataset.StudyInstanceUID
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.Laterality == self._laterality
        assert instance.PatientOrientation == self._patient_orientation
        assert instance.AccessionNumber == self._ref_dataset.AccessionNumber
        assert instance.PatientName == self._ref_dataset.PatientName
        assert instance.PatientSex == self._ref_dataset.PatientSex
        assert instance.StudyTime == TM(self._ref_dataset.StudyTime)
        assert instance.StudyDate == DA(self._ref_dataset.StudyDate)
        assert instance.StudyID == self._ref_dataset.StudyID
        assert instance.PixelData == self._rgb_pixel_array.tobytes()
