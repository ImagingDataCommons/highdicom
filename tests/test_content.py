from unittest import TestCase

import pytest
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes

from highdicom.content import (
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
    SpecimenCollection,
    SpecimenDescription,
    SpecimenPreparationStep,
    SpecimenProcessing,
    SpecimenSampling,
    SpecimenStaining,
)
from highdicom.sr.value_types import CodeContentItem, TextContentItem
from highdicom.uid import UID

from .utils import write_and_read_dataset


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
        assert not seq.is_root
        assert not seq.is_sr
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
        assert not seq.is_root
        assert not seq.is_sr
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
        assert not seq.is_root
        assert not seq.is_sr
        hematoxylin_item = seq[0]
        assert hematoxylin_item.name == codes.SCT.UsingSubstance
        assert hematoxylin_item.value == substances[0]
        assert hematoxylin_item.relationship_type is None
        eosin_item = seq[1]
        assert eosin_item.name == codes.SCT.UsingSubstance
        assert eosin_item.value == substances[1]
        assert eosin_item.relationship_type is None

    def test_construction_missing_substances(self):
        with pytest.raises(ValueError):
            SpecimenStaining(substances=[])


class TestSpecimenPreparationStep(TestCase):

    def test_construction_collection(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SpecimenCollection
        procedure = codes.SCT.Excision
        instance = SpecimenPreparationStep(
            specimen_id=specimen_id,
            processing_procedure=SpecimenCollection(procedure=procedure)
        )
        seq = instance.SpecimenPreparationStepContentItemSequence
        assert len(seq) == 3
        assert not seq.is_root
        assert not seq.is_sr
        assert instance.specimen_id == specimen_id
        specimen_id_item = seq[0]
        assert specimen_id_item.name == codes.DCM.SpecimenIdentifier
        assert specimen_id_item.value == specimen_id
        assert specimen_id_item.relationship_type is None

        assert instance.processing_type == processing_type
        processing_type_item = seq[1]
        assert processing_type_item.name == codes.DCM.ProcessingType
        assert processing_type_item.value == processing_type
        assert processing_type_item.relationship_type is None

        assert isinstance(instance.processing_procedure, SpecimenCollection)
        procedure_item = seq[2]
        assert procedure_item.name == codes.SCT.SpecimenCollection
        assert procedure_item.value == procedure
        assert procedure_item.relationship_type is None

    def test_construction_collection_from_dataset(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SpecimenCollection
        procedure = codes.SCT.Excision
        dataset = Dataset()
        dataset.SpecimenPreparationStepContentItemSequence = [
            TextContentItem(
                name=codes.DCM.SpecimenIdentifier,
                value=specimen_id
            ),
            CodeContentItem(
                name=codes.DCM.ProcessingType,
                value=processing_type
            ),
            CodeContentItem(
                name=codes.SCT.SpecimenCollection,
                value=procedure
            )
        ]
        dataset_reread = write_and_read_dataset(dataset)
        instance = SpecimenPreparationStep.from_dataset(dataset_reread)
        assert isinstance(instance, SpecimenPreparationStep)
        assert len(instance.SpecimenPreparationStepContentItemSequence) == 3
        assert instance.specimen_id == specimen_id
        assert instance.processing_type == processing_type
        assert instance.processing_procedure.procedure == procedure
        assert instance.fixative is None
        assert instance.embedding_medium is None
        processing_procedure = instance.processing_procedure
        assert isinstance(processing_procedure, SpecimenCollection)
        assert processing_procedure.procedure == procedure

    def test_construction_sampling(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SamplingOfTissueSpecimen
        method = codes.DCM.DissectionWithRepresentativeSectionsSubmission
        parent_specimen_id = 'parent specimen id'
        parent_specimen_type = codes.SCT.TissueSpecimen
        fixative = codes.SCT.Formalin
        embedding_medium = codes.SCT.ParaffinWax
        instance = SpecimenPreparationStep(
            specimen_id=specimen_id,
            processing_procedure=SpecimenSampling(
                method=method,
                parent_specimen_id=parent_specimen_id,
                parent_specimen_type=parent_specimen_type
            ),
            fixative=fixative,
            embedding_medium=embedding_medium
        )
        seq = instance.SpecimenPreparationStepContentItemSequence
        assert len(seq) == 7
        assert not seq.is_root
        assert not seq.is_sr

        assert instance.specimen_id == specimen_id
        assert instance.processing_type == processing_type
        assert instance.fixative == fixative
        assert instance.embedding_medium == embedding_medium
        assert isinstance(instance.processing_procedure, SpecimenSampling)

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

    def test_construction_sampling_from_dataset(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SamplingOfTissueSpecimen
        method = codes.DCM.DissectionWithRepresentativeSectionsSubmission
        parent_specimen_id = 'parent specimen id'
        parent_specimen_type = codes.SCT.TissueSpecimen
        fixative = codes.SCT.Formalin
        embedding_medium = codes.SCT.ParaffinWax
        dataset = Dataset()
        dataset.SpecimenPreparationStepContentItemSequence = [
            TextContentItem(
                name=codes.DCM.SpecimenIdentifier,
                value=specimen_id
            ),
            CodeContentItem(
                name=codes.DCM.ProcessingType,
                value=processing_type
            ),
            CodeContentItem(
                name=codes.DCM.SamplingMethod,
                value=method
            ),
            TextContentItem(
                name=codes.DCM.ParentSpecimenIdentifier,
                value=parent_specimen_id
            ),
            CodeContentItem(
                name=codes.DCM.ParentSpecimenType,
                value=parent_specimen_type
            ),
            CodeContentItem(
                name=codes.SCT.TissueFixative,
                value=fixative
            ),
            CodeContentItem(
                name=codes.SCT.TissueEmbeddingMedium,
                value=embedding_medium
            )
        ]
        dataset_reread = write_and_read_dataset(dataset)
        instance = SpecimenPreparationStep.from_dataset(dataset_reread)
        assert isinstance(instance, SpecimenPreparationStep)
        assert instance.specimen_id == specimen_id
        assert instance.processing_type == processing_type
        assert instance.fixative == fixative
        assert instance.embedding_medium == embedding_medium
        processing_procedure = instance.processing_procedure
        assert isinstance(processing_procedure, SpecimenSampling)
        assert processing_procedure.method == method
        assert processing_procedure.parent_specimen_id == parent_specimen_id
        assert processing_procedure.parent_specimen_type == parent_specimen_type

    def test_construction_staining(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.Staining
        substance = codes.SCT.HematoxylinStain
        instance = SpecimenPreparationStep(
            specimen_id=specimen_id,
            processing_procedure=SpecimenStaining(substances=[substance])
        )
        seq = instance.SpecimenPreparationStepContentItemSequence
        assert len(seq) == 3
        assert not seq.is_root
        assert not seq.is_sr

        assert instance.specimen_id == specimen_id
        assert instance.processing_type == processing_type
        assert instance.fixative is None
        assert instance.embedding_medium is None

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

    def test_construction_staining_from_dataset(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.Staining
        substance = codes.SCT.HematoxylinStain
        dataset = Dataset()
        dataset.SpecimenPreparationStepContentItemSequence = [
            TextContentItem(
                name=codes.DCM.SpecimenIdentifier,
                value=specimen_id
            ),
            CodeContentItem(
                name=codes.DCM.ProcessingType,
                value=processing_type
            ),
            CodeContentItem(
                name=codes.SCT.UsingSubstance,
                value=substance
            ),
        ]
        dataset_reread = write_and_read_dataset(dataset)
        instance = SpecimenPreparationStep.from_dataset(dataset_reread)
        assert isinstance(instance, SpecimenPreparationStep)
        assert instance.specimen_id == specimen_id
        assert instance.processing_type == processing_type
        assert instance.fixative is None
        assert instance.embedding_medium is None
        processing_procedure = instance.processing_procedure
        assert isinstance(processing_procedure, SpecimenStaining)
        assert processing_procedure.substances == [substance]

    def test_construction_processing(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SpecimenProcessing
        description = codes.SCT.SpecimenFreezing
        instance = SpecimenPreparationStep(
            specimen_id=specimen_id,
            processing_procedure=SpecimenProcessing(description=description)
        )
        seq = instance.SpecimenPreparationStepContentItemSequence
        assert len(seq) == 3
        assert not seq.is_root
        assert not seq.is_sr

        assert instance.specimen_id == specimen_id
        assert instance.processing_type == processing_type
        assert instance.fixative is None
        assert instance.embedding_medium is None

        specimen_id_item = seq[0]
        assert specimen_id_item.name == codes.DCM.SpecimenIdentifier
        assert specimen_id_item.value == specimen_id
        assert specimen_id_item.relationship_type is None

        processing_type_item = seq[1]
        assert processing_type_item.name == codes.DCM.ProcessingType
        assert processing_type_item.value == processing_type
        assert processing_type_item.relationship_type is None

        staining_item = seq[2]
        assert staining_item.name == codes.DCM.ProcessingStepDescription
        assert staining_item.value == description
        assert staining_item.relationship_type is None

    def test_construction_processing_from_dataset(self):
        specimen_id = 'specimen id'
        processing_type = codes.SCT.SpecimenProcessing
        description = codes.SCT.SpecimenFreezing
        dataset = Dataset()
        dataset.SpecimenPreparationStepContentItemSequence = [
            TextContentItem(
                name=codes.DCM.SpecimenIdentifier,
                value=specimen_id
            ),
            CodeContentItem(
                name=codes.DCM.ProcessingType,
                value=processing_type
            ),
            CodeContentItem(
                name=codes.DCM.ProcessingStepDescription,
                value=description
            ),
        ]
        dataset_reread = write_and_read_dataset(dataset)
        instance = SpecimenPreparationStep.from_dataset(dataset_reread)
        assert isinstance(instance, SpecimenPreparationStep)
        assert instance.specimen_id == specimen_id
        assert instance.processing_type == processing_type
        assert instance.fixative is None
        assert instance.embedding_medium is None
        processing_procedure = instance.processing_procedure
        assert isinstance(processing_procedure, SpecimenProcessing)
        assert processing_procedure.description == description


class TestSpecimenDescription(TestCase):

    def test_construction(self):
        specimen_id = 'specimen 1'
        specimen_uid = UID()
        instance = SpecimenDescription(
            specimen_id=specimen_id,
            specimen_uid=specimen_uid
        )
        assert instance.specimen_id == specimen_id
        assert instance.specimen_uid == specimen_uid
        assert len(instance.specimen_preparation_steps) == 0

    def test_construction_with_preparation_steps(self):
        parent_specimen_id = 'surgical specimen'
        specimen_id = 'section specimen'
        specimen_uid = UID()
        specimen_collection = SpecimenCollection(procedure=codes.SCT.Biopsy)
        specimen_sampling = SpecimenSampling(
            method=codes.SCT.BlockSectioning,
            parent_specimen_id=parent_specimen_id,
            parent_specimen_type=codes.SCT.GrossSpecimen
        )
        specimen_staining = SpecimenStaining(
            substances=[
                codes.SCT.HematoxylinStain,
                codes.SCT.WaterSolubleEosinStain,
            ]
        )
        instance = SpecimenDescription(
            specimen_id=specimen_id,
            specimen_uid=specimen_uid,
            specimen_preparation_steps=[
                SpecimenPreparationStep(
                    specimen_id=parent_specimen_id,
                    processing_procedure=specimen_collection,
                ),
                SpecimenPreparationStep(
                    specimen_id=specimen_id,
                    processing_procedure=specimen_sampling,
                ),
                SpecimenPreparationStep(
                    specimen_id=specimen_id,
                    processing_procedure=specimen_staining,
                ),
            ]
        )
        assert instance.specimen_id == specimen_id
        assert instance.specimen_uid == specimen_uid
        assert len(instance.specimen_preparation_steps) == 3

    def test_construction_from_dataset(self):
        specimen_id = 'specimen 1'
        specimen_uid = UID()
        dataset = Dataset()
        dataset.SpecimenIdentifier = specimen_id
        dataset.SpecimenUID = str(specimen_uid)
        dataset.IssuerOfTheSpecimenIdentifierSequence = []
        dataset.SpecimenPreparationSequence = []
        dataset_reread = write_and_read_dataset(dataset)
        instance = SpecimenDescription.from_dataset(dataset_reread)
        assert instance.specimen_id == specimen_id
        assert instance.specimen_uid == specimen_uid
        assert len(instance.specimen_preparation_steps) == 0
