from unittest import TestCase

import pytest
from pydicom.sr.codedict import codes

from highdicom import (
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
    SpecimenCollection,
    SpecimenPreparationStep,
    SpecimenSampling,
    SpecimenStaining,
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
            processing_type=processing_type,
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
            processing_type=processing_type,
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
            processing_type=processing_type,
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
