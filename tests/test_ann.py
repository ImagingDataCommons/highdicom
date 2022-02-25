import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

from highdicom.ann.content import Measurements, AnnotationGroup
from highdicom.ann.enum import (
    AnnotationGroupGenerationTypeValues,
    GraphicTypeValues,
)
from highdicom.ann.sop import MicroscopyBulkSimpleAnnotations
from highdicom.content import AlgorithmIdentificationSequence
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID


class TestMeasurements(unittest.TestCase):

    def test_construction(self):
        name = codes.SCT.Area
        num_total = 10
        missing_indices = np.array([1, 3, 6, 7])
        values = np.random.random((num_total, ))
        values[missing_indices] = np.nan
        unit = codes.UCUM.SquareMicrometer
        measurements = Measurements(
            name=name,
            values=values,
            unit=unit
        )

        assert measurements.name == name
        assert measurements.unit == unit
        np.testing.assert_allclose(
            measurements.get_values(num_total),
            values
        )

        assert measurements.ConceptNameCodeSequence[0] == name
        assert measurements.MeasurementUnitsCodeSequence[0] == unit
        item = measurements.MeasurementValuesSequence[0]
        num_stored = len(values) - len(missing_indices)
        stored_indices = np.setdiff1d(np.arange(num_total), missing_indices)
        retrieved_values = np.frombuffer(item.FloatingPointValues, np.float32)
        retrieved_indices = np.frombuffer(item.AnnotationIndexList, np.int32)
        assert len(retrieved_values) == num_stored
        assert len(retrieved_indices) == num_stored
        np.testing.assert_array_equal(
            retrieved_indices,
            stored_indices + 1
        )
        np.testing.assert_allclose(
            retrieved_values,
            values[stored_indices]
        )

    def test_construction_missing_name(self):
        with pytest.raises(TypeError):
            Measurements(
                values=np.ones((10, ), dtype=np.float32),
                unit=codes.UCUM.Millimeter
            )

    def test_construction_missing_unit(self):
        with pytest.raises(TypeError):
            Measurements(
                name=codes.SCT.Diameter,
                values=np.ones((10, ), dtype=np.float32),
            )

    def test_construction_missing_values(self):
        with pytest.raises(TypeError):
            Measurements(
                name=codes.SCT.Diameter,
                unit=codes.UCUM.Millimeter
            )

    def test_alternative_construction_from_dataset(self):
        dataset = Dataset()
        name = Dataset()
        name.CodeValue = '81827009'
        name.CodingSchemeDesignator = 'SCT'
        name.CodeMeaning = 'Diameter'
        dataset.ConceptNameCodeSequence = [name]
        unit = Dataset()
        unit.CodeValue = 'mm'
        unit.CodingSchemeDesignator = 'UCUM'
        unit.CodeMeaning = 'Millimeter'
        dataset.MeasurementUnitsCodeSequence = [unit]
        values = np.array([1.4, 1.2, 0.5], np.float32)
        index = np.array([1, 2, 3], np.int32)
        measurement_values = Dataset()
        measurement_values.FloatingPointValues = values.tobytes()
        measurement_values.AnnotationIndexList = index.tobytes()
        dataset.MeasurementValuesSequence = [measurement_values]

        measurements = Measurements.from_dataset(dataset)

        assert measurements.name == CodedConcept.from_dataset(name)
        assert measurements.unit == CodedConcept.from_dataset(unit)
        np.testing.assert_allclose(measurements.get_values(3), values)


class TestAnnotationGroup(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._property_category = Code(
            '91723000',
            'SCT',
            'Anatomical Structure'
        )
        self._property_type = Code(
            '4421005',
            'SCT',
            'Cell'
        )
        self._algorithm_type = AnnotationGroupGenerationTypeValues.AUTOMATIC
        self._algorithm_identification = AlgorithmIdentificationSequence(
            name='test',
            family=codes.DCM.ArtificialIntelligence,
            version='1.0'
        )
        self._anatomic_region = codes.SCT.Thorax
        self._anatomic_structure = codes.SCT.Lung

    def test_construction(self):
        number = 1
        label = 'first'
        uid = UID()
        graphic_type = GraphicTypeValues.POLYGON
        graphic_data = [
            np.array([
                [1.0, 1.0, 0.0],
                [0.5, 3.0, 0.0],
                [1.0, 3.0, 0.0],
            ]),
            np.array([
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
            ]),
        ]

        measurement_values = np.array([[0.5], [1.0]])
        measurement_names = [codes.SCT.Area]
        measurement_units = [codes.UCUM.SquareMicrometer]
        measurements = [
            Measurements(
                name=measurement_names[0],
                unit=measurement_units[0],
                values=measurement_values
            ),
        ]

        group = AnnotationGroup(
            number=number,
            uid=uid,
            label=label,
            annotated_property_category=self._property_category,
            annotated_property_type=self._property_type,
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            algorithm_type=self._algorithm_type,
            algorithm_identification=self._algorithm_identification,
            measurements=measurements,
            description='annotation',
            anatomic_regions=[self._anatomic_region],
            primary_anatomic_structures=[self._anatomic_structure]
        )

        assert group.graphic_type == graphic_type
        assert group.annotated_property_category == self._property_category
        assert group.annotated_property_type == self._property_type
        assert group.algorithm_type == self._algorithm_type
        assert group.algorithm_identification == self._algorithm_identification
        assert group.anatomic_regions[0] == self._anatomic_region
        assert len(group.PrimaryAnatomicStructureSequence) == 1
        assert group.primary_anatomic_structures[0] == self._anatomic_structure

        decoded_graphic_data = group.get_graphic_data(coordinate_type='3D')
        assert len(decoded_graphic_data) == len(graphic_data)
        for i in range(len(decoded_graphic_data)):
            np.testing.assert_allclose(
                decoded_graphic_data[i],
                graphic_data[i]
            )
        np.testing.assert_allclose(
            group.get_coordinates(annotation_number=1, coordinate_type='3D'),
            graphic_data[0]
        )
        np.testing.assert_allclose(
            group.get_coordinates(annotation_number=2, coordinate_type='3D'),
            graphic_data[1]
        )

        names, values, units = group.get_measurements()
        assert len(names) == 1
        assert names[0] == measurement_names[0]
        assert len(units) == 1
        assert units[0] == measurement_units[0]
        assert values.dtype == np.float32
        assert values.shape == (2, 1)
        np.testing.assert_allclose(values, measurement_values)

        names, values, units = group.get_measurements(
            name=measurement_names[0]
        )
        assert len(names) == 1
        assert names[0] == measurement_names[0]
        assert len(units) == 1
        assert units[0] == measurement_units[0]
        assert values.dtype == np.float32
        assert values.shape == (2, 1)
        np.testing.assert_allclose(values, measurement_values)

        names, values, units = group.get_measurements(
            name=codes.SCT.Volume
        )
        assert names == []
        assert units == []
        assert values.size == 0
        assert values.dtype == np.float32
        assert values.shape == (2, 0)

    def test_alternative_construction_from_dataset(self):
        dataset = Dataset()
        dataset.AnnotationGroupNumber = 1
        dataset.AnnotationGroupUID = str(UID())
        dataset.AnnotationGroupLabel = 'one'
        dataset.AnnotationGroupDescription = 'first group'
        dataset.AnnotationGroupGenerationType = 'MANUAL'
        annotated_category = Dataset()
        annotated_category.CodeValue = '91723000'
        annotated_category.CodingSchemeDesignator = 'SCT'
        annotated_category.CodeMeaning = 'Anatomical Structure'
        dataset.AnnotationPropertyCategoryCodeSequence = [annotated_category]
        annotated_type = Dataset()
        annotated_type.CodeValue = '4421005'
        annotated_type.CodingSchemeDesignator = 'SCT'
        annotated_type.CodeMeaning = 'Cell'
        dataset.AnnotationPropertyTypeCodeSequence = [annotated_type]
        dataset.NumberOfAnnotations = 3
        dataset.GraphicType = 'POINT'
        dataset.DoublePointCoordinatesData = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        dataset.AnnotationAppliesToAllZPlanes = 'NO'
        dataset.AnnotationAppliesToAllOpticalPaths = 'YES'

        group = AnnotationGroup.from_dataset(dataset)
        assert isinstance(group, AnnotationGroup)
        assert isinstance(group.graphic_type, GraphicTypeValues)
        assert isinstance(group.algorithm_type,
                          AnnotationGroupGenerationTypeValues)
        assert group.algorithm_identification is None
        assert isinstance(group.annotated_property_category, CodedConcept)
        assert isinstance(group.annotated_property_type, CodedConcept)

        names, values, units = group.get_measurements()
        assert names == []
        assert units == []
        assert values.size == 0
        assert values.dtype == np.float32
        assert values.shape == (3, 0)

    def test_alternative_construction_from_dataset_missing_attributes(self):
        dataset = Dataset()
        with pytest.raises(AttributeError):
            AnnotationGroup.from_dataset(dataset)

    def test_construction_with_wrong_graphic_data_point(self):
        with pytest.raises(ValueError):
            AnnotationGroup(
                number=1,
                uid=UID(),
                label='foo',
                annotated_property_category=self._property_category,
                annotated_property_type=self._property_type,
                graphic_type=GraphicTypeValues.POINT,
                graphic_data=[
                    np.array([
                        [1.0, 1.0, 0.0],
                        [0.5, 3.0, 0.0],
                        [1.0, 3.0, 0.0],
                    ]),
                    np.array([
                        [1.0, 1.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [2.0, 1.0, 0.0],
                    ]),
                ],
                algorithm_type=self._algorithm_type,
                algorithm_identification=self._algorithm_identification,
                description='annotation'
            )

    def test_construction_with_wrong_graphic_data_polygon(self):
        with pytest.raises(ValueError):
            AnnotationGroup(
                number=1,
                uid=UID(),
                label='foo',
                annotated_property_category=self._property_category,
                annotated_property_type=self._property_type,
                graphic_type=GraphicTypeValues.POLYGON,
                graphic_data=[
                    np.array([
                        [1.0, 1.0, 0.0],
                        [0.5, 3.0, 0.0],
                    ]),
                    np.array([
                        [1.0, 1.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [2.0, 1.0, 0.0],
                    ]),
                ],
                algorithm_type=self._algorithm_type,
                algorithm_identification=self._algorithm_identification,
                description='annotation'
            )

    def test_construction_with_wrong_graphic_data_rectangle(self):
        with pytest.raises(ValueError):
            AnnotationGroup(
                number=1,
                uid=UID(),
                label='foo',
                annotated_property_category=self._property_category,
                annotated_property_type=self._property_type,
                graphic_type=GraphicTypeValues.RECTANGLE,
                graphic_data=[
                    np.array([
                        [1.0, 1.0, 0.0],
                        [0.5, 3.0, 0.0],
                        [1.0, 3.0, 0.0],
                    ]),
                    np.array([
                        [1.0, 1.0, 0.0],
                        [1.0, 2.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [2.0, 1.0, 0.0],
                    ]),
                ],
                algorithm_type=self._algorithm_type,
                algorithm_identification=self._algorithm_identification,
                description='annotation'
            )


class TestMicroscopyBulkSimpleAnnotations(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._sm_image = dcmread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )

    def test_construction(self):
        property_category = Code('91723000', 'SCT', 'Anatomical Structure')
        algorithm_type = AnnotationGroupGenerationTypeValues.AUTOMATIC
        algorithm_identification = AlgorithmIdentificationSequence(
            name='test',
            family=codes.DCM.ArtificialIntelligence,
            version='1.0'
        )

        annotation_coordinate_type = '3D'
        first_property_type = Code('4421005', 'SCT', 'Cell')
        first_label = 'cells'
        first_uid = UID()
        first_graphic_type = GraphicTypeValues.POLYGON
        first_graphic_data = [
            np.array([
                [1.0, 1.0, 0.0],
                [0.5, 3.0, 0.0],
                [1.0, 3.0, 0.0],
            ]),
            np.array([
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
            ]),
        ]

        second_property_type = Code('84640000', 'SCT', 'Nucleus')
        second_label = 'nuclei'
        second_uid = UID()
        second_graphic_type = GraphicTypeValues.POINT
        second_graphic_data = [
            np.array([[0.84, 2.34, 0.0]]),
            np.array([[1.5, 1.5, 0.0]]),
        ]

        groups = [
            AnnotationGroup(
                number=1,
                uid=first_uid,
                label=first_label,
                annotated_property_category=property_category,
                annotated_property_type=first_property_type,
                graphic_type=first_graphic_type,
                graphic_data=first_graphic_data,
                algorithm_type=algorithm_type,
                algorithm_identification=algorithm_identification,
                description='first group'
            ),
            AnnotationGroup(
                number=2,
                uid=second_uid,
                label=second_label,
                annotated_property_category=property_category,
                annotated_property_type=second_property_type,
                graphic_type=second_graphic_type,
                graphic_data=second_graphic_data,
                algorithm_type=algorithm_type,
                algorithm_identification=algorithm_identification,
                description='second group'
            ),
        ]
        annotations = MicroscopyBulkSimpleAnnotations(
            source_images=[self._sm_image],
            annotation_coordinate_type=annotation_coordinate_type,
            annotation_groups=groups,
            series_instance_uid=UID(),
            series_number=2,
            sop_instance_uid=UID(),
            instance_number=1,
            manufacturer='highdicom',
            manufacturer_model_name='test',
            software_versions='0.1.0rc',
            device_serial_number='XYZ'
        )

        with BytesIO() as fp:
            annotations.save_as(fp)
            fp.seek(0)
            dataset = dcmread(fp)

        annotations = MicroscopyBulkSimpleAnnotations.from_dataset(dataset)

        retrieved_groups = annotations.get_annotation_groups()
        assert len(retrieved_groups) == 2

        first_retrieved_group = retrieved_groups[0]
        first_retrieved_graphic_data = first_retrieved_group.get_graphic_data(
            coordinate_type=annotation_coordinate_type
        )
        assert len(first_retrieved_graphic_data) == len(first_graphic_data)
        for i in range(len(first_retrieved_graphic_data)):
            np.testing.assert_allclose(
                first_retrieved_graphic_data[i],
                first_graphic_data[i]
            )

        second_retrieved_group = retrieved_groups[1]
        second_retrieved_graphic_data = second_retrieved_group.get_graphic_data(
            coordinate_type=annotation_coordinate_type
        )
        assert len(second_retrieved_graphic_data) == len(second_graphic_data)
        for i in range(len(second_retrieved_graphic_data)):
            np.testing.assert_allclose(
                second_retrieved_graphic_data[i],
                second_graphic_data[i]
            )

        with pytest.raises(ValueError):
            second_retrieved_group.get_graphic_data(coordinate_type='2D')

        retrieved_groups = annotations.get_annotation_groups(
            graphic_type=GraphicTypeValues.POINT
        )
        assert len(retrieved_groups) == 1

        retrieved_groups = annotations.get_annotation_groups(
            graphic_type=GraphicTypeValues.POLYGON
        )
        assert len(retrieved_groups) == 1

        retrieved_groups = annotations.get_annotation_groups(
            graphic_type=second_graphic_type,
            annotated_property_type=first_property_type
        )
        assert len(retrieved_groups) == 0

        retrieved_groups = annotations.get_annotation_groups(
            graphic_type=first_graphic_type,
            annotated_property_type=first_property_type
        )
        assert len(retrieved_groups) == 1

        retrieved_group = annotations.get_annotation_group(number=1)
        assert isinstance(retrieved_group, AnnotationGroup)
        assert retrieved_group.number == 1
        assert retrieved_group.label == first_label

        retrieved_group = annotations.get_annotation_group(uid=first_uid)
        assert isinstance(retrieved_group, AnnotationGroup)
        assert retrieved_group.number == 1
        assert retrieved_group.label == first_label
