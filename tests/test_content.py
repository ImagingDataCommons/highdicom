from unittest import TestCase

import pytest
from pydicom import dcmread
from pydicom.sr.codedict import codes
from pydicom.data import get_testdata_file, get_testdata_files

import numpy as np

from highdicom.sr import CodedConcept
from highdicom import (
    ColorLUT,
    ContentCreatorIdentificationCodeSequence,
    ModalityLUT,
    LUT,
    PaletteColorLookupTable,
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
    ReferencedImageSequence,
    RescaleTypeValues,
    SpecimenCollection,
    SpecimenPreparationStep,
    SpecimenSampling,
    UID,
    SpecimenStaining,
    VOILUT,
    VOILUTFunctionValues,
)


class TestContentCreatorIdentification(TestCase):

    def setUp(self):
        super().setUp()
        self._person_codes = [codes.DCM.Person, codes.DCM.Technologist]
        self._institution_name = 'MGH'
        self._person_address = '1000 Main St.'
        self._person_telephone_numbers = ['123456789']
        self._email = 'example@example.com'
        self._institution_address = '123 Broadway'
        self._institution_code = CodedConcept(
            value='1',
            meaning='MGH',
            scheme_designator='HOSPITAL_NAMES',
        )
        self._department_name = 'Radiology'
        self._department_code = codes.SCT.RadiologyDepartment

    def test_construction_minimal(self):
        creator_id = ContentCreatorIdentificationCodeSequence(
            person_identification_codes=self._person_codes,
            institution_name=self._institution_name,
        )
        assert len(creator_id) == 1
        creator_id_item = creator_id[0]
        assert creator_id_item.InstitutionName == self._institution_name
        for code1, code2 in zip(
            creator_id_item.PersonIdentificationCodeSequence,
            self._person_codes
        ):
            assert code1.CodeValue == code2.value

    def test_construction_full(self):
        creator_id = ContentCreatorIdentificationCodeSequence(
            person_identification_codes=self._person_codes,
            institution_name=self._institution_name,
            person_address=self._person_address,
            person_telephone_numbers=self._person_telephone_numbers,
            person_telecom_information=self._email,
            institution_code=self._institution_code,
            institution_address=self._institution_address,
            institutional_department_name=self._department_name,
            institutional_department_type_code=self._department_code,
        )
        assert len(creator_id) == 1
        creator_id_item = creator_id[0]
        assert creator_id_item.InstitutionName == self._institution_name
        for code1, code2 in zip(
            creator_id_item.PersonIdentificationCodeSequence,
            self._person_codes
        ):
            assert code1.CodeValue == code2.value
        assert creator_id_item.PersonAddress == self._person_address
        assert (
            creator_id_item.PersonTelephoneNumbers ==
            self._person_telephone_numbers[0]
        )
        assert (
            creator_id_item.PersonTelecomInformation ==
            self._email
        )
        assert (
            creator_id_item.InstitutionCodeSequence[0].CodeValue ==
            self._institution_code.value
        )
        assert creator_id_item.InstitutionAddress == self._institution_address
        assert (
            creator_id_item.InstitutionalDepartmentName ==
            self._department_name
        )
        department_code = \
            creator_id_item.InstitutionalDepartmentTypeCodeSequence[0]
        assert (department_code.CodeValue == self._department_code.value)


class TestLUT(TestCase):

    def setUp(self):
        super().setUp()
        self._lut_data = np.arange(10, 100, dtype=np.uint8)
        self._lut_data_16 = np.arange(510, 600, dtype=np.uint16)
        self._explanation = 'My LUT'

    def test_construction(self):
        first_value = 0
        lut = LUT(
            first_mapped_value=first_value,
            lut_data=self._lut_data,
        )
        assert lut.LUTDescriptor == [len(self._lut_data), first_value, 8]
        assert lut.bits_per_entry == 8
        assert lut.first_mapped_value == first_value
        assert np.array_equal(lut.lut_data, self._lut_data)
        assert not hasattr(lut, 'LUTExplanation')

    def test_construction_16bit(self):
        first_value = 0
        lut = LUT(
            first_mapped_value=first_value,
            lut_data=self._lut_data_16
        )
        assert lut.LUTDescriptor == [len(self._lut_data), first_value, 16]
        assert lut.bits_per_entry == 16
        assert lut.first_mapped_value == first_value
        assert np.array_equal(lut.lut_data, self._lut_data_16)
        assert not hasattr(lut, 'LUTExplanation')

    def test_construction_explanation(self):
        first_value = 0
        lut = LUT(
            first_mapped_value=first_value,
            lut_data=self._lut_data,
            lut_explanation=self._explanation
        )
        assert lut.LUTDescriptor == [len(self._lut_data), first_value, 8]
        assert lut.bits_per_entry == 8
        assert lut.first_mapped_value == first_value
        assert np.array_equal(lut.lut_data, self._lut_data)
        assert lut.LUTExplanation == self._explanation


class TestModalityLUT(TestCase):

    def setUp(self):
        super().setUp()
        self._lut_data = np.arange(10, 100, dtype=np.uint8)
        self._lut_data_16 = np.arange(510, 600, dtype=np.uint16)
        self._explanation = 'My LUT'

    def test_construction(self):
        first_value = 0
        lut = ModalityLUT(
            lut_type=RescaleTypeValues.HU,
            lut=LUT(
                first_mapped_value=first_value,
                lut_data=self._lut_data,
            )
        )
        assert lut.ModalityLUTType == RescaleTypeValues.HU.value
        assert lut.LUTDescriptor == [len(self._lut_data), first_value, 8]
        assert lut.bits_per_entry == 8
        assert lut.first_mapped_value == first_value
        assert np.array_equal(lut.lut_data, self._lut_data)
        assert not hasattr(lut, 'LUTExplanation')

    def test_construction_16bit(self):
        first_value = 0
        lut = ModalityLUT(
            lut_type=RescaleTypeValues.HU,
            lut=LUT(
                first_mapped_value=first_value,
                lut_data=self._lut_data_16
            )
        )
        assert lut.ModalityLUTType == RescaleTypeValues.HU.value
        assert lut.LUTDescriptor == [len(self._lut_data), first_value, 16]
        assert lut.bits_per_entry == 16
        assert lut.first_mapped_value == first_value
        assert np.array_equal(lut.lut_data, self._lut_data_16)
        assert not hasattr(lut, 'LUTExplanation')

    def test_construction_string_type(self):
        first_value = 0
        lut_type = 'MY_MAPPING'
        lut = ModalityLUT(
            lut_type=lut_type,
            lut=LUT(
                first_mapped_value=first_value,
                lut_data=self._lut_data
            )
        )
        assert lut.ModalityLUTType == lut_type
        assert lut.LUTDescriptor == [len(self._lut_data), first_value, 8]
        assert np.array_equal(lut.lut_data, self._lut_data)
        assert not hasattr(lut, 'LUTExplanation')

    def test_construction_with_exp(self):
        first_value = 0
        lut = ModalityLUT(
            lut_type=RescaleTypeValues.HU,
            lut=LUT(
                first_mapped_value=first_value,
                lut_data=self._lut_data,
                lut_explanation=self._explanation
            )
        )
        assert lut.ModalityLUTType == RescaleTypeValues.HU.value
        assert lut.LUTDescriptor == [len(self._lut_data), first_value, 8]
        assert np.array_equal(lut.lut_data, self._lut_data)
        assert lut.LUTExplanation == self._explanation

    def test_construction_empty_data(self):
        with pytest.raises(ValueError):
            ModalityLUT(
                lut_type=RescaleTypeValues.HU,
                lut=LUT(
                    first_mapped_value=0,
                    lut_data=np.array([]),  # empty data
                )
            )

    def test_construction_negative_first_value(self):
        with pytest.raises(ValueError):
            ModalityLUT(
                lut_type=RescaleTypeValues.HU,
                lut=LUT(
                    first_mapped_value=-1,  # invalid
                    lut_data=self._lut_data,
                )
            )

    def test_construction_wrong_dtype(self):
        with pytest.raises(ValueError):
            ModalityLUT(
                lut_type=RescaleTypeValues.HU,
                lut=LUT(
                    first_mapped_value=0,  # invalid
                    lut_data=np.array([0, 1, 2], dtype=np.int16),
                )
            )


class TestPlanePositionSequence(TestCase):

    def test_construction_slide(self):
        coordinate_system = 'SLIDE'
        image_position = [0., 0., 0.]
        matrix_position = [0, 0]
        seq = PlanePositionSequence(
            coordinate_system=coordinate_system,
            image_position=image_position,
            pixel_matrix_position=matrix_position
        )
        assert len(seq) == 1
        item = seq[0]
        assert float(item.XOffsetInSlideCoordinateSystem) == image_position[0]
        assert float(item.YOffsetInSlideCoordinateSystem) == image_position[1]
        assert float(item.ZOffsetInSlideCoordinateSystem) == image_position[2]
        assert item.RowPositionInTotalImagePixelMatrix == matrix_position[1]
        assert item.ColumnPositionInTotalImagePixelMatrix == matrix_position[0]

    def test_construction_patient(self):
        coordinate_system = 'PATIENT'
        image_position = [0., 0., 0.]
        seq = PlanePositionSequence(
            coordinate_system=coordinate_system,
            image_position=image_position,
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.ImagePositionPatient[0] == image_position[0]
        assert item.ImagePositionPatient[1] == image_position[1]
        assert item.ImagePositionPatient[2] == image_position[2]


class TestPlaneOrientationSequence(TestCase):

    def test_construction_slide(self):
        coordinate_system = 'SLIDE'
        image_orientation = [0., 1., 0., 1., 0., 0.]
        seq = PlaneOrientationSequence(
            coordinate_system=coordinate_system,
            image_orientation=image_orientation,
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.ImageOrientationSlide[0] == image_orientation[0]
        assert item.ImageOrientationSlide[1] == image_orientation[1]
        assert item.ImageOrientationSlide[2] == image_orientation[2]
        assert item.ImageOrientationSlide[3] == image_orientation[3]
        assert item.ImageOrientationSlide[4] == image_orientation[4]
        assert item.ImageOrientationSlide[5] == image_orientation[5]

    def test_construction_patient(self):
        coordinate_system = 'PATIENT'
        image_orientation = [0., 1., 0., 1., 0., 0.]
        seq = PlaneOrientationSequence(
            coordinate_system=coordinate_system,
            image_orientation=image_orientation,
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.ImageOrientationPatient[0] == image_orientation[0]
        assert item.ImageOrientationPatient[1] == image_orientation[1]
        assert item.ImageOrientationPatient[2] == image_orientation[2]
        assert item.ImageOrientationPatient[3] == image_orientation[3]
        assert item.ImageOrientationPatient[4] == image_orientation[4]
        assert item.ImageOrientationPatient[5] == image_orientation[5]


class TestPixelMeasuresSequence(TestCase):

    def test_construction(self):
        pixel_spacing = [0., 0.]
        slice_thickness = 0.
        seq = PixelMeasuresSequence(
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.PixelSpacing[0] == pixel_spacing[0]
        assert item.PixelSpacing[1] == pixel_spacing[1]
        assert item.SliceThickness == slice_thickness
        with pytest.raises(AttributeError):
            item.SpacingBetweenSlices

    def test_construction_with_spacing_between_slices(self):
        pixel_spacing = [0., 0.]
        slice_thickness = 0.
        spacing_between_slices = 0.
        seq = PixelMeasuresSequence(
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            spacing_between_slices=spacing_between_slices
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.SpacingBetweenSlices == spacing_between_slices


class TestSpecimenCollection(TestCase):

    def test_construction(self):
        procedure = codes.SCT.Biopsy
        seq = SpecimenCollection(procedure=procedure)
        assert len(seq) == 1
        assert seq.is_root
        item = seq[0]
        assert item.name == codes.SCT.SpecimenCollection
        assert item.value == procedure
        assert item.relationship_type is None


class TestSpecimenSampling(TestCase):

    def test_construction(self):
        method = codes.SCT.BlockSectioning
        parent_id = '1'
        parent_type = codes.SCT.GrossSpecimen
        seq = SpecimenSampling(
            method=method,
            parent_specimen_id=parent_id,
            parent_specimen_type=parent_type
        )
        assert len(seq) == 3
        assert seq.is_root
        method_item = seq[0]
        assert method_item.name == codes.DCM.SamplingMethod
        assert method_item.value == method
        assert method_item.relationship_type is None
        parent_id_item = seq[1]
        assert parent_id_item.name == codes.DCM.ParentSpecimenIdentifier
        assert parent_id_item.value == parent_id
        assert parent_id_item.relationship_type is None
        parent_type_item = seq[2]
        assert parent_type_item.name == codes.DCM.ParentSpecimenType
        assert parent_type_item.value == parent_type
        assert parent_type_item.relationship_type is None


class TestSpecimenStaining(TestCase):

    def test_construction(self):
        substances = [
            codes.SCT.HematoxylinStain,
            codes.SCT.WaterSolubleEosinStain
        ]
        seq = SpecimenStaining(substances=substances)
        assert len(seq) == 2
        assert seq.is_root
        hematoxylin_item = seq[0]
        assert hematoxylin_item.name == codes.SCT.UsingSubstance
        assert hematoxylin_item.value == substances[0]
        assert hematoxylin_item.relationship_type is None
        eosin_item = seq[1]
        assert eosin_item.name == codes.SCT.UsingSubstance
        assert eosin_item.value == substances[1]
        assert eosin_item.relationship_type is None


class TestSpecimenPreparationStep(TestCase):

    def test_construction_collection(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SpecimenCollection
        procedure = codes.SCT.Excision
        seq = SpecimenPreparationStep(
            specimen_id=specimen_id,
            processing_procedure=SpecimenCollection(procedure=procedure)
        )
        assert len(seq) == 3
        assert seq.is_root
        specimen_id_item = seq[0]
        assert specimen_id_item.name == codes.DCM.SpecimenIdentifier
        assert specimen_id_item.value == specimen_id
        assert specimen_id_item.relationship_type is None
        processing_type_item = seq[1]
        assert processing_type_item.name == codes.DCM.ProcessingType
        assert processing_type_item.value == processing_type
        assert processing_type_item.relationship_type is None
        procedure_item = seq[2]
        assert procedure_item.name == codes.SCT.SpecimenCollection
        assert procedure_item.value == procedure
        assert procedure_item.relationship_type is None

    def test_construction_sampling(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SamplingOfTissueSpecimen
        method = codes.DCM.DissectionWithRepresentativeSectionsSubmission
        parent_specimen_id = 'parent specimen id'
        parent_specimen_type = codes.SCT.TissueSpecimen
        fixative = codes.SCT.Formalin
        embedding_medium = codes.SCT.ParaffinWax
        seq = SpecimenPreparationStep(
            specimen_id=specimen_id,
            processing_procedure=SpecimenSampling(
                method=method,
                parent_specimen_id=parent_specimen_id,
                parent_specimen_type=parent_specimen_type
            ),
            fixative=fixative,
            embedding_medium=embedding_medium
        )
        assert len(seq) == 7
        assert seq.is_root
        specimen_id_item = seq[0]
        assert specimen_id_item.name == codes.DCM.SpecimenIdentifier
        assert specimen_id_item.value == specimen_id
        assert specimen_id_item.relationship_type is None
        processing_type_item = seq[1]
        assert processing_type_item.name == codes.DCM.ProcessingType
        assert processing_type_item.value == processing_type
        assert processing_type_item.relationship_type is None
        method_item = seq[2]
        assert method_item.name == codes.DCM.SamplingMethod
        assert method_item.value == method
        assert method_item.relationship_type is None
        parent_specimen_id_item = seq[3]
        assert parent_specimen_id_item.name == codes.DCM.ParentSpecimenIdentifier  # noqa E501
        assert parent_specimen_id_item.value == parent_specimen_id
        assert parent_specimen_id_item.relationship_type is None
        parent_specimen_type_item = seq[4]
        assert parent_specimen_type_item.name == codes.DCM.ParentSpecimenType
        assert parent_specimen_type_item.value == parent_specimen_type
        assert parent_specimen_type_item.relationship_type is None
        fixative_item = seq[5]
        assert fixative_item.name == codes.SCT.TissueFixative
        assert fixative_item.value == fixative
        assert fixative_item.relationship_type is None
        embedding_medium_item = seq[6]
        assert embedding_medium_item.name == codes.SCT.TissueEmbeddingMedium
        assert embedding_medium_item.value == embedding_medium
        assert embedding_medium_item.relationship_type is None

    def test_construction_staining(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.Staining
        substance = codes.SCT.HematoxylinStain
        seq = SpecimenPreparationStep(
            specimen_id=specimen_id,
            processing_procedure=SpecimenStaining(substances=[substance])
        )
        assert len(seq) == 3
        assert seq.is_root
        specimen_id_item = seq[0]
        assert specimen_id_item.name == codes.DCM.SpecimenIdentifier
        assert specimen_id_item.value == specimen_id
        assert specimen_id_item.relationship_type is None
        processing_type_item = seq[1]
        assert processing_type_item.name == codes.DCM.ProcessingType
        assert processing_type_item.value == processing_type
        assert processing_type_item.relationship_type is None
        staining_item = seq[2]
        assert staining_item.name == codes.SCT.UsingSubstance
        assert staining_item.value == substance
        assert staining_item.relationship_type is None


class TestVOILUT(TestCase):

    def setUp(self):
        super().setUp()
        self._lut = LUT(
            first_mapped_value=0,
            lut_data=np.array([10, 11, 12], np.uint8)
        )

    def test_construction_basic(self):
        lut = VOILUT(
            window_center=40.0,
            window_width=400.0
        )
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0
        assert not hasattr(lut, 'VOILUTSequence')

    def test_construction_explanation(self):
        lut = VOILUT(
            window_center=40.0,
            window_width=400.0,
            window_explanation='Soft Tissue Window'
        )
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0

    def test_construction_multiple(self):
        lut = VOILUT(
            window_center=[40.0, 600.0],
            window_width=[400.0, 1500.0],
            window_explanation=['Soft Tissue Window', 'Lung Window'],
        )
        assert lut.WindowCenter == [40.0, 600.0]
        assert lut.WindowWidth == [400.0, 1500.0]

    def test_construction_multiple_mismatch1(self):
        with pytest.raises(ValueError):
            VOILUT(
                window_center=40.0,
                window_width=[400.0, 1500.0],
            )

    def test_construction_multiple_mismatch2(self):
        with pytest.raises(TypeError):
            VOILUT(
                window_center=[40.0, 600.0],
                window_width=400.0,
            )

    def test_construction_multiple_mismatch3(self):
        with pytest.raises(ValueError):
            VOILUT(
                window_center=[40.0, 600.0],
                window_width=[400.0, 1500.0, -50.0],
            )

    def test_construction_explanation_mismatch(self):
        with pytest.raises(TypeError):
            VOILUT(
                window_center=[40.0, 600.0],
                window_width=[400.0, 1500.0],
                window_explanation='Lung Window',
            )

    def test_construction_explanation_mismatch2(self):
        with pytest.raises(ValueError):
            VOILUT(
                window_center=40.0,
                window_width=400.0,
                window_explanation=['Soft Tissue Window', 'Lung Window'],
            )

    def test_construction_lut_function(self):
        lut = VOILUT(
            window_center=40.0,
            window_width=400.0,
            voi_lut_function=VOILUTFunctionValues.SIGMOID,
        )
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0
        assert lut.VOILUTFunction == VOILUTFunctionValues.SIGMOID.value

    def test_construction_luts(self):
        lut = VOILUT(luts=[self._lut])
        assert len(lut.VOILUTSequence) == 1
        assert not hasattr(lut, 'WindowWidth')
        assert not hasattr(lut, 'WindowCenter')

    def test_construction_both(self):
        lut = VOILUT(
            window_center=40.0,
            window_width=400.0,
            luts=[self._lut]
        )
        assert len(lut.VOILUTSequence) == 1
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0

    def test_construction_neither(self):
        with pytest.raises(TypeError):
            VOILUT()


class TestReferencedImageSequence(TestCase):

    def setUp(self):
        super().setUp()
        self._ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ct_multiframe = dcmread(get_testdata_file('eCT_Supplemental.dcm'))
        self._seg = dcmread(
            'data/test_files/seg_image_ct_binary_overlap.dcm'
        )

    def test_construction_ref_ims(self):
        ref_ims = ReferencedImageSequence(
            referenced_images=self._ct_series
        )
        assert len(ref_ims) == len(self._ct_series)

    def test_construction_empty(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[]
            )

    def test_construction_frame_number(self):
        ref_ims = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],
            referenced_frame_number=1
        )
        assert len(ref_ims) == 1
        assert ref_ims[0].ReferencedFrameNumber == 1

    def test_construction_multi_frame_numbers(self):
        ref_ims = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],
            referenced_frame_number=[1, 2]
        )
        assert len(ref_ims) == 1
        assert ref_ims[0].ReferencedFrameNumber == [1, 2]

    def test_construction_invalid_frame_number_1(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[self._ct_multiframe],
                referenced_frame_number=0
            )

    def test_construction_invalid_frame_number_2(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[self._ct_multiframe],
                referenced_frame_number=self._ct_multiframe.NumberOfFrames + 1
            )

    def test_construction_invalid_frame_number_3(self):
        nonexistent_frame = self._ct_multiframe.NumberOfFrames + 1
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[self._ct_multiframe],
                referenced_frame_number=[1, nonexistent_frame]
            )

    def test_construction_frame_number_single_frames(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=self._ct_series,
                referenced_frame_number=0
            )

    def test_construction_segment_number(self):
        ref_ims = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=1
        )
        assert len(ref_ims) == 1
        assert ref_ims[0].ReferencedSegmentNumber == 1

    def test_construction_segment_number_non_seg(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=self._ct_series,
                referenced_segment_number=1
            )

    def test_construction_segment_number_multiple(self):
        ref_ims = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=[1, 2]
        )
        assert len(ref_ims) == 1
        assert ref_ims[0].ReferencedSegmentNumber == [1, 2]

    def test_construction_segment_number_and_frames(self):
        ref_ims = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=1,
            referenced_frame_number=[1, 2, 3],
        )
        assert len(ref_ims) == 1
        assert ref_ims[0].ReferencedSegmentNumber == 1
        assert ref_ims[0].ReferencedFrameNumber == [1, 2, 3]

    def test_construction_segment_number_and_frames_mismatch(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[self._seg],
                referenced_segment_number=2,
                referenced_frame_number=[1, 2, 3],  # segment frame mismatch
            )

    def test_construction_invalid_segment_number_1(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[self._seg],
                referenced_segment_number=0
            )

    def test_construction_invalid_segment_number_2(self):
        invalid_segment_number = len(self._seg.SegmentSequence) + 1
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[self._seg],
                referenced_segment_number=invalid_segment_number
            )

    def test_construction_invalid_segment_number_3(self):
        invalid_segment_number = len(self._seg.SegmentSequence) + 1
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=[self._seg],
                referenced_segment_number=[1, invalid_segment_number]
            )

    def test_construction_duplicate(self):
        with pytest.raises(ValueError):
            ReferencedImageSequence(
                referenced_images=self._ct_series * 2,
            )


class TestPaletteColorLookupTable(TestCase):

    def setUp(self):
        super().setUp()

    def test_construction(self):
        dtype = np.uint16
        r_lut_data = np.arange(10, 120, dtype=dtype)
        g_lut_data = np.arange(20, 130, dtype=dtype)
        b_lut_data = np.arange(30, 140, dtype=dtype)
        first_mapped_value = 32
        lut_uid = UID()
        r_lut = ColorLUT(first_mapped_value, r_lut_data, color='red')
        g_lut = ColorLUT(first_mapped_value, g_lut_data, color='green')
        b_lut = ColorLUT(first_mapped_value, b_lut_data, color='blue')
        instance = PaletteColorLookupTable(
            red_lut=r_lut,
            green_lut=g_lut,
            blue_lut=b_lut,
            palette_color_lut_uid=lut_uid,
        )
        assert instance.PaletteColorLookupTableUID == lut_uid
        red_desc = [len(r_lut_data), first_mapped_value, 16]
        r_lut_data_retrieved = np.frombuffer(
            instance.RedPaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(r_lut_data, r_lut_data_retrieved)
        assert instance.RedPaletteColorLookupTableDescriptor == red_desc
        green_desc = [len(g_lut_data), first_mapped_value, 16]
        g_lut_data_retrieved = np.frombuffer(
            instance.GreenPaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(g_lut_data, g_lut_data_retrieved)
        assert instance.GreenPaletteColorLookupTableDescriptor == green_desc
        blue_desc = [len(b_lut_data), first_mapped_value, 16]
        b_lut_data_retrieved = np.frombuffer(
            instance.BluePaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(b_lut_data, b_lut_data_retrieved)
        assert instance.BluePaletteColorLookupTableDescriptor == blue_desc

        data = instance.lut_data
        expected_data = np.stack([r_lut_data, g_lut_data, b_lut_data]).T
        assert np.array_equal(data, expected_data)

    def test_construction_no_uid(self):
        r_lut_data = np.arange(10, 120, dtype=np.uint16)
        g_lut_data = np.arange(20, 130, dtype=np.uint16)
        b_lut_data = np.arange(30, 140, dtype=np.uint16)
        first_mapped_value = 32
        r_lut = ColorLUT(first_mapped_value, r_lut_data, color='red')
        g_lut = ColorLUT(first_mapped_value, g_lut_data, color='green')
        b_lut = ColorLUT(first_mapped_value, b_lut_data, color='blue')
        instance = PaletteColorLookupTable(
            red_lut=r_lut,
            green_lut=g_lut,
            blue_lut=b_lut,
        )
        assert not hasattr(instance, 'PaletteColorLookupTableUID')

    def test_construction_different_lengths(self):
        r_lut_data = np.arange(10, 120, dtype=np.uint16)
        g_lut_data = np.arange(20, 120, dtype=np.uint16)
        b_lut_data = np.arange(30, 120, dtype=np.uint16)
        first_mapped_value = 32
        r_lut = ColorLUT(first_mapped_value, r_lut_data, color='red')
        g_lut = ColorLUT(first_mapped_value, g_lut_data, color='green')
        b_lut = ColorLUT(first_mapped_value, b_lut_data, color='blue')
        with pytest.raises(ValueError):
            PaletteColorLookupTable(
                red_lut=r_lut,
                green_lut=g_lut,
                blue_lut=b_lut,
            )

    def test_construction_different_dtypes(self):
        r_lut_data = np.arange(10, 120, dtype=np.uint8)
        g_lut_data = np.arange(20, 130, dtype=np.uint16)
        b_lut_data = np.arange(30, 140, dtype=np.uint16)
        first_mapped_value = 32
        r_lut = ColorLUT(first_mapped_value, r_lut_data, color='red')
        g_lut = ColorLUT(first_mapped_value, g_lut_data, color='green')
        b_lut = ColorLUT(first_mapped_value, b_lut_data, color='blue')
        with pytest.raises(ValueError):
            PaletteColorLookupTable(
                red_lut=r_lut,
                green_lut=g_lut,
                blue_lut=b_lut,
            )

    def test_construction_different_first_values(self):
        r_lut_data = np.arange(10, 120, dtype=np.uint16)
        g_lut_data = np.arange(20, 130, dtype=np.uint16)
        b_lut_data = np.arange(30, 140, dtype=np.uint16)
        r_first_mapped_value = 32
        g_first_mapped_value = 24
        b_first_mapped_value = 32
        r_lut = ColorLUT(r_first_mapped_value, r_lut_data, color='red')
        g_lut = ColorLUT(g_first_mapped_value, g_lut_data, color='green')
        b_lut = ColorLUT(b_first_mapped_value, b_lut_data, color='blue')
        with pytest.raises(ValueError):
            PaletteColorLookupTable(
                red_lut=r_lut,
                green_lut=g_lut,
                blue_lut=b_lut,
            )
