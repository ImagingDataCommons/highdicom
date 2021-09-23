from unittest import TestCase

import pytest
from pydicom.sr.codedict import codes

from highdicom import (
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
    SpecimenCollection,
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
