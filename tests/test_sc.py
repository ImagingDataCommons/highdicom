from io import BytesIO
import unittest

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.uid import generate_uid, RLELossless

from highdicom import SpecimenDescription
from highdicom.sc import SCImage


class TestSCImage(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._rgb_pixel_array = np.zeros((10, 10, 3), dtype=np.uint8)
        self._monochrome_pixel_array = np.zeros((10, 10), dtype=np.uint16)
        self._study_instance_uid = generate_uid()
        self._series_instance_uid = generate_uid()
        self._sop_instance_uid = generate_uid()
        self._series_number = int(np.random.choice(100))
        self._instance_number = int(np.random.choice(100))
        self._manufacturer = 'ABC'
        self._laterality = 'L'
        self._patient_orientation = ['A', 'R']
        self._container_identifier = str(np.random.choice(100))
        self._specimen_identifier = str(np.random.choice(100))
        self._specimen_uid = generate_uid()

    @staticmethod
    def get_array_after_writing(instance):
        # Write DICOM object to buffer, read it again and reconstruct the mask
        with BytesIO() as fp:
            instance.save_as(fp)
            fp.seek(0)
            instance_reread = dcmread(fp)

        return instance_reread.pixel_array

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
            laterality=self._laterality
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
        assert instance.StudyTime is None
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
