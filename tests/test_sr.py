import unittest
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid
from pydicom.valuerep import DA, DS, DT, TM, PersonName
from pydicom.uid import SegmentationStorage

from highdicom.sr import CodedConcept
from highdicom.sr import (
    AlgorithmIdentification,
    CodeContentItem,
    CompositeContentItem,
    Comprehensive3DSR,
    ComprehensiveSR,
    ContainerContentItem,
    ContentSequence,
    DateContentItem,
    DateTimeContentItem,
    DeviceObserverIdentifyingAttributes,
    EnhancedSR,
    FindingSite,
    GraphicTypeValues,
    GraphicTypeValues3D,
    ImageContentItem,
    ImageLibrary,
    ImageLibraryEntryDescriptors,
    ImageRegion,
    ImageRegion3D,
    LongitudinalTemporalOffsetFromEvent,
    Measurement,
    MeasurementProperties,
    MeasurementReport,
    MeasurementStatisticalProperties,
    NumContentItem,
    ObservationContext,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
    PixelOriginInterpretationValues,
    MeasurementsAndQualitativeEvaluations,
    PlanarROIMeasurementsAndQualitativeEvaluations,
    PnameContentItem,
    QualitativeEvaluation,
    RealWorldValueMap,
    ReferencedSegment,
    ReferencedSegmentationFrame,
    RelationshipTypeValues,
    Scoord3DContentItem,
    ScoordContentItem,
    SourceImageForMeasurement,
    SourceImageForMeasurementGroup,
    SourceImageForRegion,
    SourceImageForSegmentation,
    SourceSeriesForSegmentation,
    SubjectContext,
    SubjectContextDevice,
    SubjectContextSpecimen,
    TextContentItem,
    TimeContentItem,
    TimePointContext,
    TrackingIdentifier,
    UIDRefContentItem,
    ValueTypeValues,
    VolumeSurface,
    VolumetricROIMeasurementsAndQualitativeEvaluations,
)
from highdicom.sr.utils import find_content_items
from highdicom import UID


def _build_coded_concept_dataset(code: Code) -> Dataset:
    ds = Dataset()
    ds.CodeValue = code[0]
    ds.CodingSchemeDesignator = code[1]
    ds.CodeMeaning = code[2]
    return ds


class TestImageRegion(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_construction_ct_image(self):
        source_image = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2',
            referenced_sop_instance_uid=generate_uid()
        )
        graphic_type = GraphicTypeValues.POINT
        graphic_data = np.array([[1.0, 1.0]])
        region = ImageRegion(
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            source_image=source_image
        )
        assert region.graphic_type == graphic_type
        assert region.GraphicType == graphic_type.value
        np.testing.assert_array_equal(region.value, graphic_data)
        assert region.GraphicData[0] == graphic_data[0][0]
        assert region.GraphicData[1] == graphic_data[0][1]
        with pytest.raises(AttributeError):
            region.PixelOriginInterpretation

    def test_construction_sm_image_without_pixel_origin_interpretation(self):
        source_image = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.6',
            referenced_sop_instance_uid=generate_uid()
        )
        graphic_type = GraphicTypeValues.POINT
        graphic_data = np.array([[1.0, 1.0]])
        region = ImageRegion(
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            source_image=source_image
        )
        assert region.PixelOriginInterpretation == \
            PixelOriginInterpretationValues.VOLUME.value

    def test_construction_sm_image_with_pixel_origin_interpretation(self):
        source_image = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.6',
            referenced_sop_instance_uid=generate_uid(),
            referenced_frame_numbers=[1, 2]
        )
        graphic_type = GraphicTypeValues.POINT
        graphic_data = np.array([[1.0, 1.0]])
        pixel_origin_interpretation = PixelOriginInterpretationValues.FRAME
        region = ImageRegion(
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            source_image=source_image,
            pixel_origin_interpretation=pixel_origin_interpretation
        )
        region.PixelOriginInterpretation == pixel_origin_interpretation.value

    def test_construction_sm_image_with_wrong_pixel_origin_interpretation(self):
        source_image = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.6',
            referenced_sop_instance_uid=generate_uid(),
        )
        graphic_type = GraphicTypeValues.POINT
        graphic_data = np.array([[1.0, 1.0]])
        pixel_origin_interpretation = PixelOriginInterpretationValues.FRAME
        with pytest.raises(ValueError):
            ImageRegion(
                graphic_type=graphic_type,
                graphic_data=graphic_data,
                source_image=source_image,
                pixel_origin_interpretation=pixel_origin_interpretation
            )


class TestVolumeSurface(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._frame_of_reference_uid = generate_uid()
        self._source_images = [
            SourceImageForSegmentation(
                referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.6',
                referenced_sop_instance_uid=generate_uid(),
            )
        ]
        self._source_series = SourceSeriesForSegmentation(generate_uid())
        delta = np.array([[0.0, 0.0, 1.0]])
        self._point = np.array([[1.0, 2.0, 3.0]])
        self._point_2 = self._point + delta
        self._ellipsoid = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ])
        self._ellipsoid_2 = self._ellipsoid + delta
        self._ellipse = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self._ellipse_2 = self._ellipse + delta
        self._polygon = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        self._polygon_2 = self._polygon + delta

    def test_from_point(self):
        surface = VolumeSurface(
            graphic_type=GraphicTypeValues3D.POINT,
            graphic_data=[self._point],
            frame_of_reference_uid=self._frame_of_reference_uid,
            source_images=self._source_images
        )

        assert len(surface) == 2
        assert surface.graphic_type == GraphicTypeValues3D.POINT
        graphic_data = surface.graphic_data
        assert np.array_equal(graphic_data, self._point)

        assert surface.frame_of_reference_uid == self._frame_of_reference_uid

        assert surface.has_source_images()
        src_img = surface.source_images_for_segmentation
        assert len(src_img) == 1
        assert isinstance(src_img[0], SourceImageForSegmentation)
        assert surface.source_series_for_segmentation is None

    def test_from_point_with_series(self):
        surface = VolumeSurface(
            graphic_type=GraphicTypeValues3D.POINT,
            graphic_data=[self._point],
            frame_of_reference_uid=self._frame_of_reference_uid,
            source_series=self._source_series
        )
        assert surface.graphic_type == GraphicTypeValues3D.POINT
        graphic_data = surface.graphic_data
        assert np.array_equal(graphic_data, self._point)

        assert surface.frame_of_reference_uid == self._frame_of_reference_uid

        assert not surface.has_source_images()
        src_img = surface.source_images_for_segmentation
        assert len(src_img) == 0
        assert isinstance(
            surface.source_series_for_segmentation,
            SourceSeriesForSegmentation
        )

    def test_from_two_points(self):
        # Two points are invalid
        with pytest.raises(ValueError):
            VolumeSurface(
                graphic_type=GraphicTypeValues3D.POINT,
                graphic_data=[self._point, self._point_2],
                frame_of_reference_uid=self._frame_of_reference_uid
            )

    def test_from_ellipsoid(self):
        surface = VolumeSurface(
            graphic_type=GraphicTypeValues3D.ELLIPSOID,
            graphic_data=[self._ellipsoid],
            frame_of_reference_uid=self._frame_of_reference_uid,
            source_images=self._source_images
        )
        assert surface.graphic_type == GraphicTypeValues3D.ELLIPSOID
        graphic_data = surface.graphic_data
        assert np.array_equal(graphic_data, self._ellipsoid)

        assert surface.frame_of_reference_uid == self._frame_of_reference_uid

        assert surface.has_source_images()
        src_img = surface.source_images_for_segmentation
        assert len(src_img) == 1
        assert isinstance(src_img[0], SourceImageForSegmentation)
        assert surface.source_series_for_segmentation is None

    def test_from_two_ellipsoids(self):
        # Two ellipsoids are invalid
        with pytest.raises(ValueError):
            VolumeSurface(
                graphic_type=GraphicTypeValues3D.ELLIPSOID,
                graphic_data=[self._ellipsoid, self._ellipsoid_2],
                frame_of_reference_uid=self._frame_of_reference_uid
            )

    def test_from_ellipses(self):
        arrays = [self._ellipse, self._ellipse_2]
        surface = VolumeSurface(
            graphic_type=GraphicTypeValues3D.ELLIPSE,
            graphic_data=arrays,
            frame_of_reference_uid=self._frame_of_reference_uid,
            source_images=self._source_images
        )
        assert surface.graphic_type == GraphicTypeValues3D.ELLIPSE
        graphic_data = surface.graphic_data
        assert len(graphic_data) == 2
        for item, arr in zip(graphic_data, arrays):
            assert np.array_equal(item, arr)

        assert surface.frame_of_reference_uid == self._frame_of_reference_uid

        assert surface.has_source_images()
        src_img = surface.source_images_for_segmentation
        assert len(src_img) == 1
        assert isinstance(src_img[0], SourceImageForSegmentation)
        assert surface.source_series_for_segmentation is None

    def test_from_one_ellipse(self):
        # One ellipse is invalid
        with pytest.raises(ValueError):
            VolumeSurface(
                graphic_type=GraphicTypeValues3D.ELLIPSE,
                graphic_data=[self._ellipse],
                frame_of_reference_uid=self._frame_of_reference_uid
            )

    def test_from_polygons(self):
        arrays = [self._polygon, self._polygon_2]
        surface = VolumeSurface(
            graphic_type=GraphicTypeValues3D.POLYGON,
            graphic_data=arrays,
            frame_of_reference_uid=self._frame_of_reference_uid,
            source_images=self._source_images
        )
        assert surface.graphic_type == GraphicTypeValues3D.POLYGON
        graphic_data = surface.graphic_data
        assert len(graphic_data) == 2
        for item, arr in zip(graphic_data, arrays):
            assert np.array_equal(item, arr)

        assert surface.frame_of_reference_uid == self._frame_of_reference_uid

        assert surface.has_source_images()
        src_img = surface.source_images_for_segmentation
        assert len(src_img) == 1
        assert isinstance(src_img[0], SourceImageForSegmentation)
        assert surface.source_series_for_segmentation is None

    def test_from_one_polygon(self):
        # One polygon is invalid
        with pytest.raises(ValueError):
            VolumeSurface(
                graphic_type=GraphicTypeValues3D.POLYGON,
                graphic_data=[self._polygon],
                frame_of_reference_uid=self._frame_of_reference_uid
            )

    def test_invalid_graphic_types(self):
        # Polyline and multipoint are invalid
        with pytest.raises(ValueError):
            VolumeSurface(
                graphic_type=GraphicTypeValues3D.POLYLINE,
                graphic_data=[self._polygon],
                frame_of_reference_uid=self._frame_of_reference_uid
            )
        with pytest.raises(ValueError):
            VolumeSurface(
                graphic_type=GraphicTypeValues3D.MULTIPOINT,
                graphic_data=[self._polygon],
                frame_of_reference_uid=self._frame_of_reference_uid
            )


class TestAlgorithmIdentification(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._name = "Foo's Method"
        self._version = '1.0'
        self._parameters = ['spam=True', 'eggs=False']

    def test_construction_basic(self):
        algo_id = AlgorithmIdentification(
            name=self._name,
            version=self._version,
        )
        assert len(algo_id) == 2
        assert algo_id[0].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.AlgorithmName.value
        assert algo_id[0].TextValue == self._name
        assert algo_id[1].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.AlgorithmVersion.value
        assert algo_id[1].TextValue == self._version

    def test_construction_parameters(self):
        algo_id = AlgorithmIdentification(
            name=self._name,
            version=self._version,
            parameters=self._parameters,
        )
        assert len(algo_id) == 2 + len(self._parameters)
        assert algo_id[0].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.AlgorithmName.value
        assert algo_id[0].TextValue == self._name
        assert algo_id[1].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.AlgorithmVersion.value
        assert algo_id[1].TextValue == self._version
        for i, param in enumerate(self._parameters, start=2):
            assert algo_id[i].ConceptNameCodeSequence[0].CodeValue == \
                codes.DCM.AlgorithmParameters.value
            assert algo_id[i].TextValue == param


class TestMeasurementStatisticalProperties(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._value_name = codes.SCT.Volume
        self._value_unit = codes.UCUM.CubicMillimeter
        self._value_number = 0.12345
        self._values = [
            NumContentItem(
                name=self._value_name,
                value=self._value_number,
                unit=self._value_unit,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
        ]
        self._description = 'Population of Foo'
        self._authority = "World Foo Organization"

    def test_construction_basic(self):
        stat_props = MeasurementStatisticalProperties(
            values=self._values,
        )
        assert len(stat_props) == 1
        assert stat_props[0].ConceptNameCodeSequence[0].CodeValue == \
            self._value_name.value
        assert str(stat_props[0].MeasuredValueSequence[0].NumericValue) == \
            str(self._value_number)

    def test_construction_description(self):
        stat_props = MeasurementStatisticalProperties(
            values=self._values,
            description=self._description,
        )
        assert len(stat_props) == 2
        assert stat_props[0].ConceptNameCodeSequence[0].CodeValue == \
            self._value_name.value
        assert str(stat_props[0].MeasuredValueSequence[0].NumericValue) == \
            str(self._value_number)
        assert stat_props[1].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.PopulationDescription.value


class TestCodedConcept(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._value = '373098007'
        self._meaning = 'Mean Value of population'
        self._scheme_designator = 'SCT'

    def test_construction_kwargs(self):
        c = CodedConcept(
            value=self._value,
            scheme_designator=self._scheme_designator,
            meaning=self._meaning
        )
        assert c.value == self._value
        assert c.scheme_designator == self._scheme_designator
        assert c.meaning == self._meaning
        assert c.scheme_version is None
        assert c.CodeValue == self._value
        assert c.CodingSchemeDesignator == self._scheme_designator
        assert c.CodeMeaning == self._meaning
        with pytest.raises(AttributeError):
            assert c.CodingSchemeVersion

    def test_construction_kwargs_optional(self):
        version = 'v1.0'
        c = CodedConcept(
            value=self._value,
            scheme_designator=self._scheme_designator,
            meaning=self._meaning,
            scheme_version=version
        )
        assert c.value == self._value
        assert c.scheme_designator == self._scheme_designator
        assert c.meaning == self._meaning
        assert c.scheme_version == version
        assert c.CodeValue == self._value
        assert c.CodingSchemeDesignator == self._scheme_designator
        assert c.CodeMeaning == self._meaning
        assert c.CodingSchemeVersion == version

    def test_construction_args(self):
        c = CodedConcept(self._value, self._scheme_designator, self._meaning)
        assert c.value == self._value
        assert c.scheme_designator == self._scheme_designator
        assert c.meaning == self._meaning
        assert c.scheme_version is None
        assert c.CodeValue == self._value
        assert c.CodingSchemeDesignator == self._scheme_designator
        assert c.CodeMeaning == self._meaning
        with pytest.raises(AttributeError):
            assert c.CodingSchemeVersion

    def test_construction_args_optional(self):
        version = 'v1.0'
        c = CodedConcept(
            self._value, self._scheme_designator, self._meaning, version
        )
        assert c.value == self._value
        assert c.scheme_designator == self._scheme_designator
        assert c.meaning == self._meaning
        assert c.scheme_version == version
        assert c.CodeValue == self._value
        assert c.CodingSchemeDesignator == self._scheme_designator
        assert c.CodeMeaning == self._meaning
        assert c.CodingSchemeVersion == version

    def test_equal(self):
        c1 = CodedConcept(self._value, self._scheme_designator, self._meaning)
        c2 = CodedConcept(self._value, self._scheme_designator, self._meaning)
        assert c1 == c2

    def test_not_equal(self):
        c1 = CodedConcept(self._value, self._scheme_designator, self._meaning)
        c2 = CodedConcept('373099004', 'SCT', 'Median Value of population')
        assert c1 != c2

    def test_equal_ignore_meaning(self):
        c1 = CodedConcept(self._value, self._scheme_designator, self._meaning)
        c2 = CodedConcept(self._value, self._scheme_designator, 'bla bla bla')
        assert c1 == c2

    def test_equal_equivalent_coding(self):
        c1 = CodedConcept(self._value, self._scheme_designator, self._meaning)
        c2 = CodedConcept('R-00317', 'SRT', self._meaning)
        assert c1 == c2


class TestContentItem(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_code_item_construction(self):
        name = codes.SCT.FindingSite
        value = codes.SCT.Abdomen
        rel_type = RelationshipTypeValues.HAS_PROPERTIES
        i = CodeContentItem(
            name=name,
            value=value,
            relationship_type=rel_type,
        )
        assert i.ValueType == 'CODE'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.ConceptCodeSequence[0] == value
        assert i.name == CodedConcept(*name)
        assert i.value == value
        assert i.RelationshipType == rel_type.value

    def test_text_item_construction(self):
        name = codes.DCM.TrackingIdentifier
        value = '1234'
        rel_type = RelationshipTypeValues.HAS_PROPERTIES
        i = TextContentItem(
            name=name,
            value=value,
            relationship_type=rel_type,
        )
        assert i.ValueType == 'TEXT'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.TextValue == value
        assert i.name == CodedConcept(*name)
        assert i.value == value
        assert i.RelationshipType == rel_type.value

    def test_text_item_from_dataset(self):
        name = codes.DCM.TrackingIdentifier
        text_name_ds = _build_coded_concept_dataset(name)
        dataset = Dataset()
        dataset.ValueType = 'TEXT'
        dataset.ConceptNameCodeSequence = [text_name_ds]
        dataset.TextValue = 'foo'
        dataset.RelationshipType = 'HAS PROPERTIES'

        item = TextContentItem.from_dataset(dataset)
        assert isinstance(item, TextContentItem)
        assert isinstance(item.name, CodedConcept)
        assert item.name == name
        assert item.TextValue == dataset.TextValue

    def test_text_item_from_dataset_with_missing_name(self):
        dataset = Dataset()
        dataset.ValueType = 'TEXT'
        dataset.TextValue = 'foo'
        dataset.RelationshipType = 'HAS PROPERTIES'
        with pytest.raises(AttributeError):
            TextContentItem.from_dataset(dataset)

    def test_text_item_from_dataset_with_missing_value(self):
        text_name_ds = _build_coded_concept_dataset(
            codes.DCM.SpecimenIdentifier
        )
        dataset = Dataset()
        dataset.ValueType = 'TEXT'
        dataset.ConceptNameCodeSequence = [text_name_ds]
        dataset.RelationshipType = RelationshipTypeValues.HAS_PROPERTIES.value
        with pytest.raises(AttributeError):
            TextContentItem.from_dataset(dataset)

    def test_time_item_construction_from_string(self):
        name = codes.DCM.StudyTime
        value = '153000'
        rel_type = RelationshipTypeValues.HAS_OBS_CONTEXT
        i = TimeContentItem(
            name=name,
            value=value,
            relationship_type=rel_type,
        )
        assert i.ValueType == 'TIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Time == TM(value)
        assert i.name == CodedConcept(*name)
        assert i.value == datetime.strptime(value, '%H%M%S').time()
        assert i.RelationshipType == rel_type.value

    def test_time_item_construction_from_string_malformatted(self):
        name = codes.DCM.StudyTime
        value = 'abc'
        with pytest.raises(ValueError):
            TimeContentItem(
                name=name,
                value=value,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT,
            )

    def test_time_item_construction_from_time(self):
        name = codes.DCM.StudyTime
        value = datetime.now().time()
        rel_type = RelationshipTypeValues.HAS_OBS_CONTEXT
        i = TimeContentItem(
            name=name,
            value=value,
            relationship_type=rel_type
        )
        assert i.ValueType == 'TIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Time == TM(value)
        assert i.name == CodedConcept(*name)
        assert i.value == value
        assert i.RelationshipType == rel_type.value

    def test_date_item_construction_from_string(self):
        name = codes.DCM.StudyDate
        value = '20190821'
        rel_type = RelationshipTypeValues.HAS_OBS_CONTEXT
        i = DateContentItem(
            name=name,
            value=value,
            relationship_type=rel_type
        )
        assert i.ValueType == 'DATE'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Date == DA(value)
        assert i.name == CodedConcept(*name)
        assert i.value == datetime.strptime(value, '%Y%m%d').date()
        assert i.RelationshipType == rel_type.value

    def test_date_item_construction_from_string_malformatted(self):
        name = codes.DCM.StudyDate
        value = 'abcd'
        rel_type = RelationshipTypeValues.HAS_OBS_CONTEXT
        with pytest.raises(ValueError):
            DateContentItem(
                name=name,
                value=value,
                relationship_type=rel_type
            )

    def test_date_item_construction_from_time(self):
        name = codes.DCM.StudyTime
        value = datetime.now().date()
        rel_type = RelationshipTypeValues.HAS_OBS_CONTEXT
        i = DateContentItem(
            name=name,
            value=value,
            relationship_type=rel_type
        )
        assert i.ValueType == 'DATE'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Date == DA(value)
        assert i.name == CodedConcept(*name)
        assert i.value == value
        assert i.RelationshipType == rel_type.value

    def test_datetime_item_construction_from_string(self):
        name = codes.DCM.ImagingStartDatetime
        value = '20190821153000'
        rel_type = RelationshipTypeValues.HAS_OBS_CONTEXT
        i = DateTimeContentItem(
            name=name,
            value=value,
            relationship_type=rel_type
        )
        assert i.ValueType == 'DATETIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.DateTime == DT(value)
        assert i.name == CodedConcept(*name)
        assert i.value == datetime.strptime(value, '%Y%m%d%H%M%S')
        assert i.RelationshipType == rel_type.value

    def test_datetime_item_construction_from_string_malformatted(self):
        name = codes.DCM.ImagingStartDatetime
        value = 'abcd'
        with pytest.raises(ValueError):
            DateTimeContentItem(
                name=name,
                value=value,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )

    def test_datetime_item_construction_from_datetime(self):
        name = codes.DCM.ImagingStartDatetime
        value = datetime.now()
        rel_type = RelationshipTypeValues.HAS_OBS_CONTEXT
        i = DateTimeContentItem(
            name=name,
            value=value,
            relationship_type=rel_type
        )
        assert i.ValueType == 'DATETIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.DateTime == DT(value)
        assert i.name == CodedConcept(*name)
        assert i.value == value
        assert i.RelationshipType == rel_type.value

    def test_uidref_item_construction_from_string(self):
        name = codes.DCM.SeriesInstanceUID
        value = '1.2.3.4.5.6'
        rel_type = RelationshipTypeValues.INFERRED_FROM
        i = UIDRefContentItem(
            name=name,
            value=value,
            relationship_type=rel_type
        )
        assert i.ValueType == 'UIDREF'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.UID == UID(value)
        assert i.name == CodedConcept(*name)
        assert i.value == value
        assert i.RelationshipType == rel_type.value

    def test_uidref_item_construction_wrong_value_type(self):
        name = codes.DCM.SeriesInstanceUID
        value = 123456
        with pytest.raises(TypeError):
            UIDRefContentItem(
                name=name,
                value=value,
                relationship_type=RelationshipTypeValues.INFERRED_FROM
            )

    def test_uidref_item_construction_from_uid(self):
        name = codes.DCM.SeriesInstanceUID
        value = UID('1.2.3.4.5.6')
        rel_type = RelationshipTypeValues.INFERRED_FROM
        i = UIDRefContentItem(
            name=name,
            value=value,
            relationship_type=rel_type
        )
        assert i.ValueType == 'UIDREF'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.UID == UID(value)
        assert i.RelationshipType == rel_type.value

    def test_num_item_construction_from_integer(self):
        name = codes.SCT.Area
        value = 100
        unit = Code('um2', 'UCUM', 'Square Micrometer')
        rel_type = RelationshipTypeValues.HAS_PROPERTIES
        i = NumContentItem(
            name=name,
            value=value,
            unit=unit,
            relationship_type=rel_type
        )
        assert i.ValueType == 'NUM'
        assert i.ConceptNameCodeSequence[0] == name
        value_item = i.MeasuredValueSequence[0]
        unit_code_item = value_item.MeasurementUnitsCodeSequence[0]
        assert value_item.NumericValue == value
        assert i.name == CodedConcept(*name)
        assert i.value == value
        assert i.unit == CodedConcept(*unit)
        with pytest.raises(AttributeError):
            assert value_item.FloatingPointValue
        assert unit_code_item.CodeValue == unit.value
        assert unit_code_item.CodingSchemeDesignator == unit.scheme_designator
        assert i.RelationshipType == rel_type.value
        with pytest.raises(AttributeError):
            assert i.NumericValueQualifierCodeSequence

    def test_num_item_construction_from_float(self):
        name = codes.SCT.Area
        value = 100.0
        unit = Code('um2', 'UCUM', 'Square Micrometer')
        rel_type = RelationshipTypeValues.HAS_PROPERTIES
        i = NumContentItem(
            name=name,
            value=value,
            unit=unit,
            relationship_type=rel_type
        )
        assert i.value == value
        assert i.unit == unit
        assert i.qualifier is None
        assert i.ValueType == 'NUM'
        assert i.ConceptNameCodeSequence[0] == name
        value_item = i.MeasuredValueSequence[0]
        unit_code_item = value_item.MeasurementUnitsCodeSequence[0]
        assert value_item.NumericValue == value
        assert value_item.FloatingPointValue == value
        assert unit_code_item.CodeValue == unit.value
        assert unit_code_item.CodingSchemeDesignator == unit.scheme_designator
        assert i.RelationshipType == rel_type.value
        with pytest.raises(AttributeError):
            assert i.NumericValueQualifierCodeSequence

    def test_num_item_construction_from_qualifier_code(self):
        name = codes.SCT.Area
        value = 100.0
        unit = Code('um2', 'UCUM', 'Square Micrometer')
        qualifier = Code('114000', 'SCT', 'Not a number')
        rel_type = RelationshipTypeValues.HAS_PROPERTIES
        i = NumContentItem(
            name=name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            relationship_type=rel_type
        )
        assert i.value == value
        assert i.unit == unit
        assert i.qualifier == qualifier
        assert i.ValueType == 'NUM'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.RelationshipType == rel_type.value
        qualifier_code_item = i.NumericValueQualifierCodeSequence[0]
        assert qualifier_code_item.CodeValue == qualifier.value

    def test_pname_content_item(self):
        name = codes.DCM.PersonObserverName
        value = 'Doe^John'
        i = PnameContentItem(
            name=name,
            value=value,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        assert i.PersonName == 'Doe^John'

    def test_pname_content_item_from_person_name(self):
        name = codes.DCM.PersonObserverName
        value = PersonName('Doe^John')
        i = PnameContentItem(
            name=name,
            value=value,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        assert i.PersonName == 'Doe^John'

    def test_pname_content_item_invalid_name(self):
        name = codes.DCM.PersonObserverName
        value = 'John Doe'  # invalid name format
        with pytest.warns(UserWarning):
            PnameContentItem(
                name=name,
                value=value,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )

    def test_container_item_construction(self):
        name = codes.DCM.ImagingMeasurementReport
        tid = '1500'
        i = ContainerContentItem(
            name=name,
            template_id=tid,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        assert i.ValueType == 'CONTAINER'
        assert i.ConceptNameCodeSequence[0] == name
        template_item = i.ContentTemplateSequence[0]
        assert template_item.TemplateIdentifier == tid
        assert template_item.MappingResource == 'DCMR'
        assert i.ContinuityOfContent == 'CONTINUOUS'
        assert i.name == CodedConcept(*name)
        with pytest.raises(AttributeError):
            assert i.value
        with pytest.raises(AttributeError):
            assert i.ContentSequence

    def test_composite_item_construction(self):
        name = codes.DCM.RealWorldValueMapUsedForMeasurement
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.2'
        sop_instance_uid = '1.2.3.4'
        i = CompositeContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        assert i.ValueType == 'COMPOSITE'
        assert i.ConceptNameCodeSequence[0] == name
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert i.name == CodedConcept(*name)
        assert i.value == (sop_class_uid, sop_instance_uid)

    def test_image_item_construction(self):
        name = codes.DCM.SourceImageForSegmentation
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.2'
        sop_instance_uid = '1.2.3.4'
        i = ImageContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        assert i.ValueType == 'IMAGE'
        assert i.ConceptNameCodeSequence[0] == name
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert i.name == CodedConcept(*name)
        assert i.value == (sop_class_uid, sop_instance_uid)
        assert i.referenced_sop_instance_uid == sop_instance_uid
        assert i.referenced_sop_class_uid == sop_class_uid
        with pytest.raises(AttributeError):
            ref_sop_item.ReferencedFrameNumber
        with pytest.raises(AttributeError):
            ref_sop_item.ReferencedSegmentNumber
        assert i.referenced_frame_numbers is None
        assert i.referenced_segment_numbers is None

    def test_image_item_construction_with_multiple_frame_numbers(self):
        name = codes.DCM.SourceImageForSegmentation
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.2.2'
        sop_instance_uid = '1.2.3.4'
        frame_numbers = [1, 2, 3]
        i = ImageContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_frame_numbers=frame_numbers,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert ref_sop_item.ReferencedFrameNumber == frame_numbers
        assert i.referenced_frame_numbers == frame_numbers
        with pytest.raises(AttributeError):
            ref_sop_item.ReferencedSegmentNumber

    def test_image_item_construction_with_single_frame_number(self):
        name = codes.DCM.SourceImageForSegmentation
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.2.2'
        sop_instance_uid = '1.2.3.4'
        frame_number = 1
        i = ImageContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_frame_numbers=frame_number,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert ref_sop_item.ReferencedFrameNumber == frame_number
        assert i.referenced_frame_numbers == [frame_number]
        with pytest.raises(AttributeError):
            ref_sop_item.ReferencedSegmentNumber

    def test_image_item_construction_single_segment_number(self):
        name = codes.DCM.SourceImageForSegmentation
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.66.4'
        sop_instance_uid = '1.2.3.4'
        segment_number = 1
        i = ImageContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_segment_numbers=segment_number,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert ref_sop_item.ReferencedSegmentNumber == segment_number
        assert i.referenced_segment_numbers == [segment_number]
        with pytest.raises(AttributeError):
            ref_sop_item.ReferencedFrameNumber

    def test_scoord_item_construction_point(self):
        name = codes.DCM.ImageRegion
        graphic_type = GraphicTypeValues.POINT
        graphic_data = np.array([[1.0, 1.0]])
        pixel_origin_interpretation = 'FRAME'
        i = ScoordContentItem(
            name=name,
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            pixel_origin_interpretation=pixel_origin_interpretation,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        assert i.ValueType == 'SCOORD'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.GraphicType == graphic_type.value
        assert i.GraphicData == graphic_data.flatten().tolist()
        assert i.PixelOriginInterpretation == pixel_origin_interpretation
        with pytest.raises(AttributeError):
            i.FiducialUID

    def test_scoord_item_construction_circle(self):
        name = codes.DCM.ImageRegion
        graphic_type = GraphicTypeValues.CIRCLE
        graphic_data = np.array([[1.0, 1.0], [2.0, 2.0]])
        pixel_origin_interpretation = 'VOLUME'
        i = ScoordContentItem(
            name=name,
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            pixel_origin_interpretation=pixel_origin_interpretation,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        assert i.ValueType == 'SCOORD'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.graphic_type == graphic_type
        assert np.all(i.GraphicData[:2] == graphic_data[0, :])
        assert np.all(i.GraphicData[2:4] == graphic_data[1, :])
        assert i.PixelOriginInterpretation == pixel_origin_interpretation
        with pytest.raises(AttributeError):
            i.FiducialUID

    def test_scoord3d_item_construction_point(self):
        name = codes.DCM.ImageRegion
        graphic_type = GraphicTypeValues3D.POINT
        graphic_data = np.array([[1.0, 1.0, 1.0]])
        frame_of_reference_uid = '1.2.3'
        i = Scoord3DContentItem(
            name=name,
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            frame_of_reference_uid=frame_of_reference_uid,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        assert i.ValueType == 'SCOORD3D'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.GraphicType == graphic_type.value
        assert np.all(i.GraphicData == graphic_data[0, :])
        assert i.ReferencedFrameOfReferenceUID == frame_of_reference_uid
        with pytest.raises(AttributeError):
            i.FiducialUID

    def test_scoord3d_item_construction_polygon(self):
        name = codes.DCM.ImageRegion
        graphic_type = GraphicTypeValues3D.POLYGON
        graphic_data = np.array([
            [1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0]
        ])
        frame_of_reference_uid = '1.2.3'
        i = Scoord3DContentItem(
            name=name,
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            frame_of_reference_uid=frame_of_reference_uid,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        assert i.ValueType == 'SCOORD3D'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.graphic_type == graphic_type
        assert np.all(i.GraphicData[:3] == graphic_data[0, :])
        assert np.all(i.GraphicData[3:6] == graphic_data[1, :])
        assert np.all(i.GraphicData[6:9] == graphic_data[2, :])
        assert i.ReferencedFrameOfReferenceUID == frame_of_reference_uid
        with pytest.raises(AttributeError):
            i.FiducialUID

    def test_container_item_from_dataset(self):
        code_name_ds = _build_coded_concept_dataset(codes.DCM.Finding)
        code_value_ds = _build_coded_concept_dataset(codes.SCT.Neoplasm)
        code_ds = Dataset()
        code_ds.ValueType = 'CODE'
        code_ds.ConceptNameCodeSequence = [code_name_ds]
        code_ds.ConceptCodeSequence = [code_value_ds]
        code_ds.RelationshipType = 'CONTAINS'

        num_name_ds = _build_coded_concept_dataset(codes.SCT.Length)
        num_unit_ds = _build_coded_concept_dataset(codes.UCUM.Millimeter)
        num_value_ds = Dataset()
        num_value_ds.NumericValue = 1.
        num_value_ds.MeasurementUnitsCodeSequence = [num_unit_ds]
        num_ds = Dataset()
        num_ds.ValueType = 'NUM'
        num_ds.ConceptNameCodeSequence = [num_name_ds]
        num_ds.MeasuredValueSequence = [num_value_ds]
        num_ds.RelationshipType = 'CONTAINS'

        container_name_ds = _build_coded_concept_dataset(
            codes.DCM.MeasurementGroup
        )
        container_ds = Dataset()
        container_ds.ContinuityOfContent = 'CONTINUOUS'
        container_ds.ValueType = 'CONTAINER'
        container_ds.ConceptNameCodeSequence = [container_name_ds]
        container_ds.RelationshipType = 'CONTAINS'
        container_ds.ContentSequence = [code_ds, num_ds]

        container_item = ContainerContentItem.from_dataset(container_ds)
        assert isinstance(container_item, ContainerContentItem)
        assert isinstance(container_item.name, CodedConcept)
        assert isinstance(container_item.ContentSequence, ContentSequence)
        code_item = container_item.ContentSequence[0]
        assert isinstance(code_item, CodeContentItem)
        assert isinstance(code_item.name, CodedConcept)
        assert isinstance(code_item.value, CodedConcept)


class TestContentSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._item = CodeContentItem(
            name=codes.SCT.FindingSite,
            value=codes.SCT.Abdomen,
            relationship_type=RelationshipTypeValues.HAS_PROPERTIES
        )
        self._item_no_rel = CodeContentItem(
            name=codes.SCT.FindingSite,
            value=codes.SCT.Abdomen
        )
        self._root_item = ContainerContentItem(
            name=codes.DCM.ImagingMeasurementReport,
            template_id='1500'
        )
        self._root_item_with_rel = ContainerContentItem(
            name=codes.DCM.ImagingMeasurementReport,
            template_id='1500',
            relationship_type=RelationshipTypeValues.CONTAINS
        )

    def test_append(self):
        seq = ContentSequence()
        seq.append(self._item)

    def test_extend(self):
        seq = ContentSequence()
        seq.extend([self._item])

    def test_insert(self):
        seq = ContentSequence()
        seq.insert(0, self._item)

    def test_construct(self):
        ContentSequence([self._item])

    def test_append_with_no_relationship(self):
        seq = ContentSequence()
        with pytest.raises(AttributeError):
            seq.append(self._item_no_rel)

    def test_extend_with_no_relationship(self):
        seq = ContentSequence()
        with pytest.raises(AttributeError):
            seq.extend([self._item_no_rel])

    def test_insert_with_no_relationship(self):
        seq = ContentSequence()
        with pytest.raises(AttributeError):
            seq.insert(0, self._item_no_rel)

    def test_construct_with_no_relationship(self):
        with pytest.raises(AttributeError):
            ContentSequence([self._item_no_rel])

    def test_append_root_item(self):
        seq = ContentSequence([], is_root=True)
        seq.append(self._root_item)

    def test_extend_root_item(self):
        seq = ContentSequence([], is_root=True)
        seq.extend([self._root_item])

    def test_insert_root_item(self):
        seq = ContentSequence([], is_root=True)
        seq.insert(0, self._root_item)

    def test_construct_root_item(self):
        ContentSequence([self._root_item], is_root=True)

    def test_construct_root_item_not_sr_iod(self):
        with pytest.raises(ValueError):
            ContentSequence([self._root_item], is_root=True, is_sr=False)

    def test_append_root_item_with_relationship(self):
        seq = ContentSequence([], is_root=True)
        with pytest.raises(AttributeError):
            seq.append(self._root_item_with_rel)

    def test_extend_root_item_with_relationship(self):
        seq = ContentSequence([], is_root=True)
        with pytest.raises(AttributeError):
            seq.extend([self._root_item_with_rel])

    def test_insert_root_item_with_relationship(self):
        seq = ContentSequence([], is_root=True)
        with pytest.raises(AttributeError):
            seq.insert(0, self._root_item_with_rel)

    def test_construct_root_with_relationship(self):
        with pytest.raises(AttributeError):
            ContentSequence([self._root_item_with_rel], is_root=True)

    def test_content_item_setattr(self):
        # Integration test that setting a ContentItem's content sequence
        # should be possible if the items have relationships
        self._root_item.ContentSequence = [self._item]

    def test_content_item_setattr_with_no_relationship(self):
        # Integration test that setting a ContentItem's content sequence
        # triggers the relevant relationship_type checks
        with pytest.raises(AttributeError):
            self._root_item.ContentSequence = [self._item_no_rel]


class TestSubjectContextDevice(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._name = 'Foo Device'
        self._uid = generate_uid()
        self._manufacturer = 'Foomakers Inc.'
        self._model_name = 'Foo Mark II'
        self._serial_number = '987654321'
        self._physical_location = 'Planet Foo'

    def test_construction_basic(self):
        context = SubjectContextDevice(name=self._name)
        assert len(context) == 1
        assert context[0].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.DeviceSubjectName.value
        assert context[0].TextValue == self._name

    def test_construction_all(self):
        context = SubjectContextDevice(
            name=self._name,
            uid=self._uid,
            manufacturer_name=self._manufacturer,
            model_name=self._model_name,
            serial_number=self._serial_number,
            physical_location=self._physical_location
        )
        assert len(context) == 6
        assert context[0].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.DeviceSubjectName.value
        assert context[0].TextValue == self._name
        assert context[1].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.DeviceSubjectUID.value
        assert context[1].UID == self._uid
        assert context[2].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.DeviceSubjectManufacturer.value
        assert context[2].TextValue == self._manufacturer
        assert context[3].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.DeviceSubjectModelName.value
        assert context[3].TextValue == self._model_name
        assert context[4].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.DeviceSubjectSerialNumber.value
        assert context[4].TextValue == self._serial_number
        assert context[5].ConceptNameCodeSequence[0].CodeValue == \
            codes.DCM.DeviceSubjectPhysicalLocationDuringObservation.value
        assert context[5].TextValue == self._physical_location


class TestObservationContext(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._person_name = 'Bar^Foo'
        self._device_uid = generate_uid()
        self._specimen_uid = generate_uid()
        self._observer_person_context = ObserverContext(
            observer_type=codes.cid270.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name=self._person_name
            )
        )
        self._observer_device_context = ObserverContext(
            observer_type=codes.cid270.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=self._device_uid
            )
        )
        self._subject_context = SubjectContext(
            subject_class=codes.cid271.Specimen,
            subject_class_specific_context=SubjectContextSpecimen(
                uid=self._specimen_uid
            )
        )
        self._observation_context = ObservationContext(
            observer_person_context=self._observer_person_context,
            observer_device_context=self._observer_device_context,
            subject_context=self._subject_context
        )

    def test_observer_context(self):
        # person
        assert len(self._observer_person_context) == 2
        item = self._observer_person_context[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '121005'
        assert item.ConceptCodeSequence[0] == codes.cid270.Person
        item = self._observer_person_context[1]
        assert item.ConceptNameCodeSequence[0].CodeValue == '121008'
        assert item.PersonName == self._person_name
        # device
        assert len(self._observer_device_context) == 2
        item = self._observer_device_context[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '121005'
        assert item.ConceptCodeSequence[0] == codes.cid270.Device
        item = self._observer_device_context[1]
        assert item.ConceptNameCodeSequence[0].CodeValue == '121012'
        assert item.UID == self._device_uid

    def test_subject_context(self):
        assert len(self._subject_context) == 2
        item = self._subject_context[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '121024'
        assert item.ConceptCodeSequence[0] == codes.cid271.Specimen
        item = self._subject_context[1]
        assert item.ConceptNameCodeSequence[0].CodeValue == '121039'
        assert item.UID == self._specimen_uid

    def test_content_length(self):
        assert len(self._observation_context) == 6


class TestPersonObserverIdentifyingAttributes(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._person_name = 'Doe^John'
        self._invalid_name = 'John Doe'
        self._login_name = 'jd123'
        self._organization_name = 'The General Hospital'
        self._role_in_organization = codes.DCM.Surgeon
        self._role_in_procedure = codes.DCM.PerformingPhysician

    def test_construction(self):
        observer = PersonObserverIdentifyingAttributes(
            name=self._person_name
        )
        assert observer[0].PersonName == self._person_name

    def test_construction_all(self):
        seq = PersonObserverIdentifyingAttributes(
            name=self._person_name,
            login_name=self._login_name,
            organization_name=self._organization_name,
            role_in_organization=self._role_in_organization,
            role_in_procedure=self._role_in_procedure
        )
        assert len(seq) == 5
        assert seq[0].PersonName == self._person_name
        assert seq[1].TextValue == self._login_name
        assert seq[2].TextValue == self._organization_name
        assert seq[3].ConceptCodeSequence[0] == self._role_in_organization
        assert seq[4].ConceptCodeSequence[0] == self._role_in_procedure

    def test_construction_invalid(self):
        with pytest.warns(UserWarning):
            PersonObserverIdentifyingAttributes(
                name=self._invalid_name
            )


class TestFindingSiteOptional(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._location = codes.cid7151.LobeOfLung
        self._laterality = codes.cid244.Right
        self._modifier = codes.cid2.Apical
        self._finding_site = FindingSite(
            anatomic_location=self._location,
            laterality=self._laterality,
            topographical_modifier=self._modifier
        )

    def test_finding_site(self):
        item = self._finding_site
        assert item.ConceptNameCodeSequence[0].CodeValue == '363698007'
        assert item.ConceptCodeSequence[0] == self._location
        assert len(item.ContentSequence) == 2

    def test_laterality(self):
        item = self._finding_site.ContentSequence[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '272741003'
        assert item.ConceptCodeSequence[0] == self._laterality

    def test_topographical_modifier(self):
        item = self._finding_site.ContentSequence[1]
        assert item.ConceptNameCodeSequence[0].CodeValue == '106233006'
        assert item.ConceptCodeSequence[0] == self._modifier


class TestFindingSite(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._location = \
            codes.cid6300.RightAnteriorMiddlePeripheralZoneOfProstate
        self._finding_site = FindingSite(
            anatomic_location=self._location
        )

    def test_finding_site(self):
        item = self._finding_site
        assert item.ConceptNameCodeSequence[0].CodeValue == '363698007'
        assert item.ConceptCodeSequence[0] == self._location
        assert not hasattr(item, 'ContentSequence')


class TestSourceImageForSegmentation(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._src_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._src_dataset_multiframe = dcmread(
            get_testdata_file('eCT_Supplemental.dcm')
        )
        self._invalid_src_dataset_sr = dcmread(
            get_testdata_file('reportsi.dcm')
        )
        self._invalid_src_dataset_seg = dcmread(
            str(data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm'))
        )
        self._ref_frames = [1, 2]
        self._ref_frames_invalid = [
            self._src_dataset_multiframe.NumberOfFrames + 1
        ]

    def test_construction(self):
        src_image = SourceImageForSegmentation(
            self._src_dataset.SOPClassUID,
            self._src_dataset.SOPInstanceUID
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )

    def test_construction_with_frame_reference(self):
        src_image = SourceImageForSegmentation(
            self._src_dataset_multiframe.SOPClassUID,
            self._src_dataset_multiframe.SOPInstanceUID,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image(self):
        src_image = SourceImageForSegmentation.from_source_image(
            self._src_dataset
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )

    def test_from_source_image_with_referenced_frames(self):
        src_image = SourceImageForSegmentation.from_source_image(
            self._src_dataset_multiframe,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image_with_invalid_referenced_frames(self):
        with pytest.raises(ValueError):
            SourceImageForSegmentation.from_source_image(
                self._src_dataset_multiframe,
                self._ref_frames_invalid
            )

    def test_from_invalid_source_image_sr(self):
        with pytest.raises(ValueError):
            SourceImageForSegmentation.from_source_image(
                self._invalid_src_dataset_sr
            )

    def test_from_invalid_source_image_seg(self):
        with pytest.raises(ValueError):
            SourceImageForSegmentation.from_source_image(
                self._invalid_src_dataset_seg
            )


class TestSourceSeriesForSegmentation(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._src_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._invalid_src_dataset_sr = dcmread(
            get_testdata_file('reportsi.dcm')
        )
        self._invalid_src_dataset_seg = dcmread(
            str(data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm'))
        )

    def test_construction(self):
        src_series = SourceSeriesForSegmentation(
            self._src_dataset.SeriesInstanceUID,
        )
        assert src_series.UID == self._src_dataset.SeriesInstanceUID

    def test_from_source_image(self):
        src_series = SourceSeriesForSegmentation.from_source_image(
            self._src_dataset
        )
        assert src_series.UID == self._src_dataset.SeriesInstanceUID

    def test_from_invalid_source_image_sr(self):
        with pytest.raises(ValueError):
            SourceSeriesForSegmentation.from_source_image(
                self._invalid_src_dataset_sr
            )

    def test_from_invalid_source_image_seg(self):
        with pytest.raises(ValueError):
            SourceSeriesForSegmentation.from_source_image(
                self._invalid_src_dataset_seg
            )


class TestSourceImageForRegion(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._src_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._src_dataset_multiframe = dcmread(
            get_testdata_file('eCT_Supplemental.dcm')
        )
        self._invalid_src_dataset_sr = dcmread(
            get_testdata_file('reportsi.dcm')
        )
        self._invalid_src_dataset_seg = dcmread(
            str(data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm'))
        )
        self._ref_frames = [1, 2]
        self._ref_frames_invalid = [
            self._src_dataset_multiframe.NumberOfFrames + 1
        ]

    def test_construction(self):
        src_image = SourceImageForRegion(
            self._src_dataset.SOPClassUID,
            self._src_dataset.SOPInstanceUID
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )

    def test_construction_with_frame_reference_frames(self):
        src_image = SourceImageForRegion(
            self._src_dataset_multiframe.SOPClassUID,
            self._src_dataset_multiframe.SOPInstanceUID,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image(self):
        src_image = SourceImageForRegion.from_source_image(
            self._src_dataset
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )

    def test_from_source_image_with_referenced_frames(self):
        src_image = SourceImageForRegion.from_source_image(
            self._src_dataset_multiframe,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image_with_invalid_referenced_frames(self):
        with pytest.raises(ValueError):
            SourceImageForRegion.from_source_image(
                self._src_dataset_multiframe,
                self._ref_frames_invalid
            )

    def test_from_invalid_source_image_sr(self):
        with pytest.raises(ValueError):
            SourceImageForRegion.from_source_image(
                self._invalid_src_dataset_sr
            )

    def test_from_invalid_source_image_seg(self):
        with pytest.raises(ValueError):
            SourceImageForRegion.from_source_image(
                self._invalid_src_dataset_seg
            )


class TestSourceImage(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._src_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._src_dataset_multiframe = dcmread(
            get_testdata_file('eCT_Supplemental.dcm')
        )
        self._invalid_src_dataset_sr = dcmread(
            get_testdata_file('reportsi.dcm')
        )
        self._invalid_src_dataset_seg = dcmread(
            str(data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm'))
        )
        self._ref_frames = [1, 2]
        self._ref_frames_invalid = [
            self._src_dataset_multiframe.NumberOfFrames + 1
        ]

    def test_construction(self):
        src_image = SourceImageForMeasurementGroup(
            self._src_dataset.SOPClassUID,
            self._src_dataset.SOPInstanceUID
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )
        assert src_image.value_type == ValueTypeValues.IMAGE
        assert src_image.relationship_type == RelationshipTypeValues.CONTAINS

    def test_construction_with_frame_reference(self):
        src_image = SourceImageForMeasurementGroup(
            self._src_dataset_multiframe.SOPClassUID,
            self._src_dataset_multiframe.SOPInstanceUID,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image(self):
        src_image = SourceImageForMeasurementGroup.from_source_image(
            self._src_dataset
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )

    def test_from_source_image_with_referenced_frames(self):
        src_image = SourceImageForMeasurementGroup.from_source_image(
            self._src_dataset_multiframe,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image_with_invalid_referenced_frames(self):
        with pytest.raises(ValueError):
            SourceImageForMeasurementGroup.from_source_image(
                self._src_dataset_multiframe,
                self._ref_frames_invalid
            )

    def test_from_invalid_source_image_sr(self):
        with pytest.raises(ValueError):
            SourceImageForMeasurementGroup.from_source_image(
                self._invalid_src_dataset_sr
            )

    def test_from_invalid_source_image_seg(self):
        with pytest.raises(ValueError):
            SourceImageForMeasurement.from_source_image(
                self._invalid_src_dataset_seg
            )


class TestSourceImageForMeasurement(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._src_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._src_dataset_multiframe = dcmread(
            get_testdata_file('eCT_Supplemental.dcm')
        )
        self._invalid_src_dataset_sr = dcmread(
            get_testdata_file('reportsi.dcm')
        )
        self._invalid_src_dataset_seg = dcmread(
            str(data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm'))
        )
        self._ref_frames = [1, 2]
        self._ref_frames_invalid = [
            self._src_dataset_multiframe.NumberOfFrames + 1
        ]

    def test_construction(self):
        src_image = SourceImageForMeasurement(
            self._src_dataset.SOPClassUID,
            self._src_dataset.SOPInstanceUID
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )

    def test_construction_with_frame_reference(self):
        src_image = SourceImageForMeasurement(
            self._src_dataset_multiframe.SOPClassUID,
            self._src_dataset_multiframe.SOPInstanceUID,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image(self):
        src_image = SourceImageForMeasurement.from_source_image(
            self._src_dataset
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset.SOPInstanceUID
        )

    def test_from_source_image_with_referenced_frames(self):
        src_image = SourceImageForMeasurement.from_source_image(
            self._src_dataset_multiframe,
            self._ref_frames
        )
        assert len(src_image.ReferencedSOPSequence) == 1
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_dataset_multiframe.SOPClassUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_dataset_multiframe.SOPInstanceUID
        )
        assert (
            src_image.ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frames
        )

    def test_from_source_image_with_invalid_referenced_frames(self):
        with pytest.raises(ValueError):
            SourceImageForMeasurement.from_source_image(
                self._src_dataset_multiframe,
                self._ref_frames_invalid
            )

    def test_from_invalid_source_image_sr(self):
        with pytest.raises(ValueError):
            SourceImageForMeasurement.from_source_image(
                self._invalid_src_dataset_sr
            )

    def test_from_invalid_source_image_seg(self):
        with pytest.raises(ValueError):
            SourceImageForMeasurement.from_source_image(
                self._invalid_src_dataset_seg
            )


class TestReferencedSegment(unittest.TestCase):

    def setUp(self):
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._filepath = str(
            data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm')
        )
        self._seg_dataset = dcmread(self._filepath)
        self._src_sop_class_uid = self._seg_dataset.ReferencedSeriesSequence[0]\
            .ReferencedInstanceSequence[0].ReferencedSOPClassUID
        self._src_sop_ins_uid = self._seg_dataset.ReferencedSeriesSequence[0]\
            .ReferencedInstanceSequence[0].ReferencedSOPInstanceUID
        self._src_series_ins_uid = self._seg_dataset.\
            ReferencedSeriesSequence[0].SeriesInstanceUID
        self._ref_frame_number = 38
        self._wrong_ref_frame_number = 13  # does not match the segment
        self._invalid_ref_frame_number = 0
        self._ref_segment_number = 35
        self._invalid_ref_segment_number = 8  # does not exist in this dataset
        self._src_images = [
            SourceImageForSegmentation(
                self._src_sop_class_uid,
                self._src_sop_ins_uid
            )
        ]
        self._src_series = SourceSeriesForSegmentation(
            self._src_series_ins_uid
        )

    def test_construction(self):
        ref_seg = ReferencedSegment(
            sop_class_uid=self._seg_dataset.SOPClassUID,
            sop_instance_uid=self._seg_dataset.SOPInstanceUID,
            segment_number=self._ref_segment_number,
            source_images=self._src_images
        )
        assert len(ref_seg) == 2
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._seg_dataset.SOPClassUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._seg_dataset.SOPInstanceUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSegmentNumber ==
            self._ref_segment_number
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_construction_with_frame_reference(self):
        ref_seg = ReferencedSegment(
            sop_class_uid=self._seg_dataset.SOPClassUID,
            sop_instance_uid=self._seg_dataset.SOPInstanceUID,
            segment_number=self._ref_segment_number,
            frame_numbers=[self._ref_frame_number],
            source_images=self._src_images
        )
        assert len(ref_seg) == 2
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._seg_dataset.SOPClassUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._seg_dataset.SOPInstanceUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSegmentNumber ==
            self._ref_segment_number
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frame_number
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_construction_series(self):
        ref_seg = ReferencedSegment(
            sop_class_uid=self._seg_dataset.SOPClassUID,
            sop_instance_uid=self._seg_dataset.SOPInstanceUID,
            segment_number=self._ref_segment_number,
            frame_numbers=[self._ref_frame_number],
            source_series=self._src_series
        )
        assert (ref_seg[1].UID == self._src_series_ins_uid)

    def test_from_segmenation(self):
        ref_seg = ReferencedSegment.from_segmentation(
            segmentation=self._seg_dataset,
            segment_number=self._ref_segment_number
        )
        assert len(ref_seg) == 2
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._seg_dataset.SOPClassUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._seg_dataset.SOPInstanceUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSegmentNumber ==
            self._ref_segment_number
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_from_segmentation_with_frames(self):
        ref_seg = ReferencedSegment.from_segmentation(
            segmentation=self._seg_dataset,
            segment_number=self._ref_segment_number,
            frame_numbers=[self._ref_frame_number]
        )
        assert len(ref_seg) == 2
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._seg_dataset.SOPClassUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._seg_dataset.SOPInstanceUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSegmentNumber ==
            self._ref_segment_number
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frame_number
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_from_segmentation_wrong_frame(self):
        # Test with a frame that doesn't match the segment
        with pytest.raises(ValueError):
            ReferencedSegment.from_segmentation(
                segmentation=self._seg_dataset,
                segment_number=self._ref_segment_number,
                frame_numbers=[self._wrong_ref_frame_number]
            )

    def test_from_segmentation_invalid_frame(self):
        # Test with an invalid frame number
        with pytest.raises(ValueError):
            ReferencedSegment.from_segmentation(
                segmentation=self._seg_dataset,
                segment_number=self._ref_segment_number,
                frame_numbers=[self._invalid_ref_frame_number]
            )

    def test_from_segmentation_invalid_segment(self):
        # Test with a non-existent segment
        with pytest.raises(ValueError):
            ReferencedSegment.from_segmentation(
                segmentation=self._seg_dataset,
                segment_number=self._invalid_ref_segment_number,
            )

    def test_from_segmentation_no_derivation_image(self):
        # Delete the derivation image information
        temp_dataset = deepcopy(self._seg_dataset)
        for frame_info in temp_dataset.PerFrameFunctionalGroupsSequence:
            del frame_info.DerivationImageSequence
        ref_seg = ReferencedSegment.from_segmentation(
            segmentation=temp_dataset,
            segment_number=self._ref_segment_number,
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_from_segmentation_no_derivation_image_no_instance_info(self):
        # Delete the derivation image information and the referenced instance
        # information such that the method is forced to look for series level
        # information
        temp_dataset = deepcopy(self._seg_dataset)
        for frame_info in temp_dataset.PerFrameFunctionalGroupsSequence:
            del frame_info.DerivationImageSequence
        del temp_dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        ref_seg = ReferencedSegment.from_segmentation(
            segmentation=temp_dataset,
            segment_number=self._ref_segment_number,
        )
        assert (ref_seg[1].UID == self._src_series_ins_uid)

    def test_from_segmentation_no_referenced_series_uid(self):
        # Delete the derivation image information and the referenced instance
        # and series information. This should give an error
        temp_dataset = deepcopy(self._seg_dataset)
        for frame_info in temp_dataset.PerFrameFunctionalGroupsSequence:
            del frame_info.DerivationImageSequence
        del temp_dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        del temp_dataset.ReferencedSeriesSequence[0].SeriesInstanceUID
        with pytest.raises(AttributeError):
            ReferencedSegment.from_segmentation(
                segmentation=temp_dataset,
                segment_number=self._ref_segment_number,
            )

    def test_from_segmentation_no_referenced_series_sequence(self):
        # Delete the derivation image information and the referenced instance
        # information such that the method is forced to look for series level
        # information
        temp_dataset = deepcopy(self._seg_dataset)
        for frame_info in temp_dataset.PerFrameFunctionalGroupsSequence:
            del frame_info.DerivationImageSequence
        del temp_dataset.ReferencedSeriesSequence
        with pytest.raises(AttributeError):
            ReferencedSegment.from_segmentation(
                segmentation=temp_dataset,
                segment_number=self._ref_segment_number,
            )


class TestReferencedSegmentationFrame(unittest.TestCase):

    def setUp(self):
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._filepath = str(
            data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm')
        )
        self._seg_dataset = dcmread(self._filepath)
        self._src_sop_class_uid = self._seg_dataset.ReferencedSeriesSequence[0]\
            .ReferencedInstanceSequence[0].ReferencedSOPClassUID
        self._src_sop_ins_uid = self._seg_dataset.ReferencedSeriesSequence[0]\
            .ReferencedInstanceSequence[0].ReferencedSOPInstanceUID
        self._src_series_ins_uid = self._seg_dataset.\
            ReferencedSeriesSequence[0].SeriesInstanceUID
        self._ref_segment_number = 35
        self._ref_frame_numbers = [38, 39]
        self._invalid_ref_frame_number = 0
        self._src_image = SourceImageForSegmentation(
            self._src_sop_class_uid,
            self._src_sop_ins_uid
        )

    def test_construction(self):
        ref_seg = ReferencedSegmentationFrame(
            sop_class_uid=self._seg_dataset.SOPClassUID,
            sop_instance_uid=self._seg_dataset.SOPInstanceUID,
            segment_number=self._ref_segment_number,
            frame_number=self._ref_frame_numbers,
            source_image=self._src_image
        )
        assert len(ref_seg) == 2
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._seg_dataset.SOPClassUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._seg_dataset.SOPInstanceUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSegmentNumber ==
            self._ref_segment_number
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frame_numbers
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_from_segmentation_with_frame_number(self):
        ref_seg = ReferencedSegmentationFrame.from_segmentation(
            self._seg_dataset,
            frame_number=self._ref_frame_numbers,
        )
        assert len(ref_seg) == 2
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._seg_dataset.SOPClassUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._seg_dataset.SOPInstanceUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSegmentNumber ==
            self._ref_segment_number
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frame_numbers
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_from_segmentation_with_segment_number(self):
        ref_seg = ReferencedSegmentationFrame.from_segmentation(
            self._seg_dataset,
            segment_number=self._ref_segment_number,
        )
        assert len(ref_seg) == 2
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._seg_dataset.SOPClassUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._seg_dataset.SOPInstanceUID
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedSegmentNumber ==
            self._ref_segment_number
        )
        assert (
            ref_seg[0].ReferencedSOPSequence[0].ReferencedFrameNumber ==
            self._ref_frame_numbers
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_from_segmentation_invalid_frame(self):
        with pytest.raises(ValueError):
            ReferencedSegmentationFrame.from_segmentation(
                self._seg_dataset,
                frame_number=self._invalid_ref_frame_number,
            )

    def test_from_segmentation_no_derivation_image(self):
        # Delete the derivation image information
        temp_dataset = deepcopy(self._seg_dataset)
        for frame_info in temp_dataset.PerFrameFunctionalGroupsSequence:
            del frame_info.DerivationImageSequence
        ref_seg = ReferencedSegmentationFrame.from_segmentation(
            segmentation=temp_dataset,
            frame_number=self._ref_frame_numbers,
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPClassUID ==
            self._src_sop_class_uid
        )
        assert (
            ref_seg[1].ReferencedSOPSequence[0].ReferencedSOPInstanceUID ==
            self._src_sop_ins_uid
        )

    def test_from_segmentation_no_referenced_series_uid(self):
        # Delete the derivation image information and the referenced instance
        # and series information. This should give an error
        temp_dataset = deepcopy(self._seg_dataset)
        for frame_info in temp_dataset.PerFrameFunctionalGroupsSequence:
            del frame_info.DerivationImageSequence
        del temp_dataset.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        del temp_dataset.ReferencedSeriesSequence[0].SeriesInstanceUID
        with pytest.raises(AttributeError):
            ReferencedSegmentationFrame.from_segmentation(
                segmentation=temp_dataset,
                frame_number=self._ref_frame_numbers,
            )

    def test_from_segmentation_no_referenced_series_sequence(self):
        # Delete the derivation image information and the referenced instance
        # information such that the method is forced to look for series level
        # information
        temp_dataset = deepcopy(self._seg_dataset)
        for frame_info in temp_dataset.PerFrameFunctionalGroupsSequence:
            del frame_info.DerivationImageSequence
        del temp_dataset.ReferencedSeriesSequence
        with pytest.raises(AttributeError):
            ReferencedSegmentationFrame.from_segmentation(
                segmentation=temp_dataset,
                frame_number=self._ref_frame_numbers,
            )


class TestTrackingIdentifierOptional(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._uid = generate_uid()
        self._identifier = 'prostate zone size measurements'
        self._tracking_identifier = TrackingIdentifier(
            uid=self._uid,
            identifier=self._identifier
        )

    def test_identifier(self):
        item = self._tracking_identifier[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '112039'
        assert item.TextValue == self._identifier

    def test_uid(self):
        item = self._tracking_identifier[1]
        assert item.ConceptNameCodeSequence[0].CodeValue == '112040'
        assert item.UID == self._uid


class TestTrackingIdentifier(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._uid = generate_uid()
        self._tracking_identifier = TrackingIdentifier(
            uid=self._uid
        )

    def test_uid(self):
        item = self._tracking_identifier[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '112040'
        assert item.UID == self._uid


class TestTrackingIdentifierDefault(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._tracking_identifier = TrackingIdentifier()

    def test_uid(self):
        item = self._tracking_identifier[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '112040'


class TestTimePointContext(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._time_point = 'first'
        self._time_point_context = TimePointContext(
            time_point=self._time_point
        )

    def test_time_point(self):
        item = self._time_point_context[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == 'C2348792'
        assert item.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'UMLS'
        assert item.TextValue == self._time_point


class TestTimePointContextOptional(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._time_point = 'first'
        self._time_point_type = codes.DCM.Posttreatment
        self._time_point_order = 1
        self._subject_time_point_identifier = 'subject'
        self._protocol_time_point_identifier = 'protocol'
        self._temporal_offset_from_event = LongitudinalTemporalOffsetFromEvent(
            value=5,
            unit=Code('d', 'UCUM', 'days'),
            event_type=codes.DCM.Baseline
        )
        self._time_point_context = TimePointContext(
            time_point=self._time_point,
            time_point_type=self._time_point_type,
            time_point_order=self._time_point_order,
            subject_time_point_identifier=self._subject_time_point_identifier,
            protocol_time_point_identifier=self._protocol_time_point_identifier,
            temporal_offset_from_event=self._temporal_offset_from_event
        )

    def test_time_point_type(self):
        item = self._time_point_context[1]
        assert item.ConceptNameCodeSequence[0].CodeValue == '126072'
        assert item.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'
        value = self._time_point_type.value
        assert item.ConceptCodeSequence[0].CodeValue == value

    def test_time_point_order(self):
        item = self._time_point_context[2]
        assert item.ConceptNameCodeSequence[0].CodeValue == '126073'
        assert item.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'
        value = self._time_point_order
        assert item.MeasuredValueSequence[0].NumericValue == value

    def test_subject_time_point_identifier(self):
        item = self._time_point_context[3]
        assert item.ConceptNameCodeSequence[0].CodeValue == '126070'
        assert item.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'
        value = self._subject_time_point_identifier
        assert item.TextValue == value

    def test_protocol_time_point_identifier(self):
        item = self._time_point_context[4]
        assert item.ConceptNameCodeSequence[0].CodeValue == '126071'
        assert item.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'
        value = self._protocol_time_point_identifier
        assert item.TextValue == value

    def test_temporal_offset_from_event(self):
        item = self._time_point_context[5]
        ref_item = self._temporal_offset_from_event
        assert item == ref_item
        assert item.ContentSequence[0] == ref_item.ContentSequence[0]


class TestMeasurement(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._name = codes.cid7469.Area
        self._value = 10.0
        self._unit = codes.cid7181.SquareMillimeter
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='prostate zone size measurement'
        )
        self._derivation = codes.cid7464.Total
        self._method = codes.cid7473.AreaOfClosedIrregularPolygon
        self._location = \
            codes.cid6300.RightAnteriorMiddlePeripheralZoneOfProstate
        self._finding_site = FindingSite(
            anatomic_location=self._location
        )
        self._image = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=generate_uid()
        )
        self._region = ImageRegion(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=np.array([[1.0, 1.0]]),
            source_image=self._image
        )
        self._ref_images = [
            SourceImageForMeasurement(
                referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
                referenced_sop_instance_uid=generate_uid()
            ) for _ in range(3)
        ]

    def test_construction_with_required_parameters(self):
        measurement = Measurement(
            name=self._name,
            value=self._value,
            unit=self._unit,
        )
        item = measurement[0]
        assert item.ConceptNameCodeSequence[0] == self._name
        assert len(item.MeasuredValueSequence) == 1
        assert len(item.MeasuredValueSequence[0]) == 3
        assert item.MeasuredValueSequence[0].NumericValue == DS(self._value)
        value_item = item.MeasuredValueSequence[0]
        unit_item = value_item.MeasurementUnitsCodeSequence[0]
        assert unit_item == self._unit
        with pytest.raises(AttributeError):
            item.NumericValueQualifierCodeSequence
        with pytest.raises(AttributeError):
            item.ContentSequence

        # Direct property access
        assert measurement.name == self._name
        assert measurement.value == self._value
        assert measurement.unit == self._unit
        assert measurement.qualifier is None

    def test_construction_with_missing_required_parameters(self):
        with pytest.raises(TypeError):
            Measurement(
                value=self._value,
                unit=self._unit
            )
        with pytest.raises(TypeError):
            Measurement(
                name=self._name,
                value=self._value
            )
        with pytest.raises(TypeError):
            Measurement(
                name=self._name,
                unit=self._unit
            )

    def test_construction_with_optional_parameters(self):
        measurement = Measurement(
            name=self._name,
            value=self._value,
            unit=self._unit,
            tracking_identifier=self._tracking_identifier,
            method=self._method,
            derivation=self._derivation,
            finding_sites=[self._finding_site, ],
            referenced_images=self._ref_images
        )

        subitem = measurement[0].ContentSequence[0]
        assert subitem.ConceptNameCodeSequence[0].CodeValue == '112039'
        assert subitem == self._tracking_identifier[0]

        subitem = measurement[0].ContentSequence[1]
        assert subitem.ConceptNameCodeSequence[0].CodeValue == '112040'
        assert subitem == self._tracking_identifier[1]

        subitem = measurement[0].ContentSequence[2]
        assert subitem.ConceptNameCodeSequence[0].CodeValue == '370129005'
        assert subitem.ConceptCodeSequence[0] == self._method

        subitem = measurement[0].ContentSequence[3]
        assert subitem.ConceptNameCodeSequence[0].CodeValue == '121401'
        assert subitem.ConceptCodeSequence[0] == self._derivation

        subitem = measurement[0].ContentSequence[4]
        assert subitem.ConceptNameCodeSequence[0].CodeValue == '363698007'
        assert subitem.ConceptCodeSequence[0] == self._location
        # Laterality and topological modifier were not specified
        assert not hasattr(subitem, 'ContentSequence')

        sites = measurement.finding_sites
        assert len(sites) == 1
        assert sites[0] == self._finding_site

        assert measurement.derivation == self._derivation
        assert measurement.method == self._method
        ref_images = measurement.referenced_images
        assert len(ref_images) == len(self._ref_images)
        for retrieved, original in zip(ref_images, self._ref_images):
            assert isinstance(retrieved, SourceImageForMeasurement)
            assert retrieved == original


class TestQualitativeEvaluation(unittest.TestCase):

    def setUp(self):
        self._name = codes.DCM.LevelOfSignificance
        self._value = codes.SCT.HighlySignificant

    def test_construction(self):
        evaluation = QualitativeEvaluation(
            name=self._name,
            value=self._value
        )

        assert len(evaluation) == 1
        assert evaluation.name == self._name
        assert evaluation.value == self._value


class TestMeasurementsAndQualitativeEvaluations(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='planar roi measurements'
        )
        self._measurements = [
            Measurement(
                name=codes.SCT.Area,
                value=5,
                unit=codes.UCUM.SquareCentimeter
            ),
        ]
        self._src_instance_uid = UID()
        self._source_images = [
            SourceImageForMeasurementGroup(
                referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
                referenced_sop_instance_uid=self._src_instance_uid
            )
        ]

    def test_construction(self):
        template = MeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            measurements=self._measurements,
        )
        root_item = template[0]
        assert root_item.ContentTemplateSequence[0].TemplateIdentifier == '1501'
        assert len(template.source_images) == 0

    def test_construction_with_source_images(self):
        template = MeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            measurements=self._measurements,
            source_images=self._source_images,
        )
        root_item = template[0]
        assert root_item.ContentTemplateSequence[0].TemplateIdentifier == '1501'

        source_images = template.source_images
        assert len(source_images) == 1
        src_image = source_images[0]
        assert isinstance(src_image, SourceImageForMeasurementGroup)
        print(src_image)
        sop_class_uid = src_image.referenced_sop_class_uid
        assert sop_class_uid == '1.2.840.10008.5.1.4.1.1.2.2'
        assert src_image.referenced_sop_instance_uid == self._src_instance_uid


class TestPlanarROIMeasurementsAndQualitativeEvaluations(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='planar roi measurements'
        )
        self._src_region_instance_uid = generate_uid()
        self._image_for_region = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=self._src_region_instance_uid
        )
        self._src_seg_instance_uid = generate_uid()
        self._image_for_segment = SourceImageForSegmentation(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=self._src_seg_instance_uid
        )
        self._region = ImageRegion(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=np.array([[1.0, 1.0], [2.0, 2.0]]),
            source_image=self._image_for_region
        )
        self._seg_instance_uid = generate_uid()
        self._region_3d = ImageRegion3D(
            graphic_type=GraphicTypeValues3D.POLYGON,
            graphic_data=np.array([
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
                [1.0, 1.0, 0.0]
            ]),
            frame_of_reference_uid=generate_uid()
        )
        self._segment = ReferencedSegmentationFrame(
            sop_class_uid='1.2.840.10008.5.1.4.1.1.66.4',
            sop_instance_uid=self._seg_instance_uid,
            segment_number=1,
            frame_number=1,
            source_image=self._image_for_segment
        )
        self._real_world_value_map = RealWorldValueMap(
            referenced_sop_instance_uid=generate_uid()
        )
        self._finding_category = codes.SCT.MorphologicallyAbnormalStructure
        self._finding_type = codes.SCT.Nodule
        self._method = codes.DCM.RECIST1Point1
        self._algo_id = AlgorithmIdentification(
            name='Foo Method',
            version='1.0.1',
        )
        self._finding_sites = [
            FindingSite(
                anatomic_location=codes.cid7151.LobeOfLung,
                laterality=codes.cid244.Right,
                topographical_modifier=codes.cid2.Apical
            )
        ]
        self._session = 'Session 1'
        self._geometric_purpose = codes.DCM.Center
        self._measurements = [
            Measurement(
                name=codes.SCT.Area,
                value=5,
                unit=codes.UCUM.SquareCentimeter
            ),
        ]
        self._qualitative_evaluations = [
            QualitativeEvaluation(
                name=CodedConcept(
                    value="RID49502",
                    meaning="clinically significant prostate cancer",
                    scheme_designator="RADLEX"
                ),
                value=codes.SCT.Yes
            )
        ]
        self._evaluations_wrong_rel_type = [
            CodeContentItem(
                name=CodedConcept(
                    value="RID49502",
                    meaning="clinically significant prostate cancer",
                    scheme_designator="RADLEX"
                ),
                value=codes.SCT.Yes,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
        ]

    def test_construction_with_region(self):
        template = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_region=self._region
        )
        root_item = template[0]
        assert root_item.ContentTemplateSequence[0].TemplateIdentifier == '1410'
        assert template.reference_type == codes.DCM.ImageRegion

    def test_construction_with_region_3d(self):
        template = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_region=self._region_3d
        )
        root_item = template[0]
        assert root_item.ContentTemplateSequence[0].TemplateIdentifier == '1410'

    def test_from_sequence_with_region(self):
        name = codes.DCM.MeasurementGroup
        container_name_ds = _build_coded_concept_dataset(name)
        container_ds = Dataset()
        container_ds.ValueType = 'CONTAINER'
        container_ds.ContinuityOfContent = 'CONTINUOUS'
        container_ds.RelationshipType = 'CONTAINS'
        container_ds.ConceptNameCodeSequence = [container_name_ds]
        container_ds.ContentSequence = [self._region]
        seq = PlanarROIMeasurementsAndQualitativeEvaluations.from_sequence(
            [container_ds]
        )
        assert len(seq) == 1
        assert isinstance(seq[0], ContainerContentItem)
        assert seq[0].name == name
        assert seq.referenced_segmentation_frame is None
        assert seq.reference_type == codes.DCM.ImageRegion

    def test_construction_with_segment(self):
        seq = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_segment=self._segment
        )
        assert seq.roi is None
        assert seq.reference_type == codes.DCM.ReferencedSegmentationFrame

        ref_seg = seq.referenced_segmentation_frame
        assert isinstance(ref_seg, ReferencedSegmentationFrame)
        assert ref_seg.referenced_sop_instance_uid == self._seg_instance_uid
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.66.4'
        assert ref_seg.referenced_sop_class_uid == sop_class_uid
        assert ref_seg.referenced_frame_numbers == [1]
        assert ref_seg.referenced_segment_numbers == [1]

        src_image = ref_seg.source_image_for_segmentation
        assert isinstance(src_image, SourceImageForSegmentation)
        assert (
            src_image.referenced_sop_instance_uid == self._src_seg_instance_uid
        )
        assert (
            src_image.referenced_sop_class_uid == '1.2.840.10008.5.1.4.1.1.2.2'
        )

    def test_construction_all_parameters(self):
        group = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_region=self._region,
            referenced_real_world_value_map=self._real_world_value_map,
            finding_type=self._finding_type,
            method=self._method,
            algorithm_id=self._algo_id,
            finding_sites=self._finding_sites,
            session=self._session,
            measurements=self._measurements,
            qualitative_evaluations=self._qualitative_evaluations,
            geometric_purpose=self._geometric_purpose,
            finding_category=self._finding_category
        )
        assert group.finding_type == self._finding_type
        assert group.finding_category == self._finding_category
        assert group.finding_sites == self._finding_sites

    def test_constructed_without_human_readable_tracking_identifier(self):
        tracking_identifier = TrackingIdentifier(
            uid=generate_uid()
        )
        with pytest.raises(ValueError):
            PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=tracking_identifier,
                referenced_region=self._region
            )

    def test_constructed_without_reference(self):
        with pytest.raises(ValueError):
            PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=self._tracking_identifier
            )

    def test_constructed_with_multiple_references(self):
        with pytest.raises(ValueError):
            PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=self._tracking_identifier,
                referenced_region=self._region,
                referenced_segment=self._segment
            )


class TestVolumetricROIMeasurementsAndQualitativeEvaluations(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='volumetric roi measurements'
        )
        self._images_for_region = [
            SourceImageForRegion(
                referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
                referenced_sop_instance_uid=generate_uid()
            )
            for i in range(3)
        ]
        self._src_seg_instance_uid = generate_uid()
        self._images_for_segment = [
            SourceImageForSegmentation(
                referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
                referenced_sop_instance_uid=self._src_seg_instance_uid
            )
        ]
        self._src_seg_series_uid = generate_uid()
        self._series_for_segment = SourceSeriesForSegmentation(
            self._src_seg_series_uid
        )
        self._regions = [
            ImageRegion(
                graphic_type=GraphicTypeValues.POLYLINE,
                graphic_data=np.array([
                    [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0]
                ]),
                source_image=self._images_for_region[i]
            )
            for i in range(3)
        ]
        self._seg_instance_uid = generate_uid()
        self._regions_3d = [
            ImageRegion3D(
                graphic_type=GraphicTypeValues3D.POLYGON,
                graphic_data=np.array([
                    [1.0, 1.0, i],
                    [2.0, 2.0, i],
                    [3.0, 3.0, i],
                    [1.0, 1.0, i]
                ]),
                frame_of_reference_uid=generate_uid()
            )
            for i in range(3)
        ]
        self._segment = ReferencedSegment(
            sop_class_uid='1.2.840.10008.5.1.4.1.1.66.4',
            sop_instance_uid=self._seg_instance_uid,
            segment_number=1,
            source_images=self._images_for_segment
        )
        self._segment_from_series = ReferencedSegment(
            sop_class_uid='1.2.840.10008.5.1.4.1.1.66.4',
            sop_instance_uid=self._seg_instance_uid,
            segment_number=1,
            source_series=self._series_for_segment
        )
        self._real_world_value_map = RealWorldValueMap(
            referenced_sop_instance_uid=generate_uid()
        )
        self._finding_type = codes.SCT.Nodule
        self._method = codes.DCM.RECIST1Point1
        self._algo_id = AlgorithmIdentification(
            name='Foo Method',
            version='1.0.1',
        )
        self._finding_sites = [
            FindingSite(
                anatomic_location=codes.cid7151.LobeOfLung,
                laterality=codes.cid244.Right,
                topographical_modifier=codes.cid2.Apical
            )
        ]
        self._session = 'Session 1'
        self._geometric_purpose = codes.DCM.Center
        self._qualitative_evaluations = [
            QualitativeEvaluation(
                CodedConcept(
                    value="RID49502",
                    meaning="clinically significant prostate cancer",
                    scheme_designator="RADLEX"
                ),
                codes.SCT.Yes
            )
        ]
        self._evaluations_wrong_rel_type = [
            CodeContentItem(
                CodedConcept(
                    value="RID49502",
                    meaning="clinically significant prostate cancer",
                    scheme_designator="RADLEX"
                ),
                codes.SCT.Yes,
                RelationshipTypeValues.HAS_PROPERTIES
            )
        ]

    def test_constructed_with_regions(self):
        template = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_regions=self._regions
        )
        root_item = template[0]
        assert root_item.ContentTemplateSequence[0].TemplateIdentifier == '1411'
        assert all(isinstance(region, ImageRegion) for region in template.roi)
        assert template.referenced_segment is None
        assert template.reference_type == codes.DCM.ImageRegion

    def test_constructed_with_regions_3d(self):
        with pytest.raises(TypeError):
            VolumetricROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=self._tracking_identifier,
                referenced_regions=self._regions_3d
            )

    def test_from_sequence_with_region(self):
        name = codes.DCM.MeasurementGroup
        container_name_ds = _build_coded_concept_dataset(name)
        container_ds = Dataset()
        container_ds.ValueType = 'CONTAINER'
        container_ds.ContinuityOfContent = 'CONTINUOUS'
        container_ds.RelationshipType = 'CONTAINS'
        container_ds.ConceptNameCodeSequence = [container_name_ds]
        container_ds.ContentSequence = self._regions
        seq = VolumetricROIMeasurementsAndQualitativeEvaluations.from_sequence(
            [container_ds]
        )
        assert len(seq) == 1
        assert isinstance(
            seq,
            VolumetricROIMeasurementsAndQualitativeEvaluations
        )
        assert isinstance(seq[0], ContainerContentItem)
        assert seq[0].name == name
        assert seq.reference_type == codes.DCM.ImageRegion

    def test_constructed_with_segment(self):
        template = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_segment=self._segment
        )
        assert template.reference_type == codes.DCM.ReferencedSegment
        ref_seg = template.referenced_segment
        assert isinstance(ref_seg, ReferencedSegment)
        assert ref_seg.has_source_images()
        assert ref_seg.referenced_sop_instance_uid == self._seg_instance_uid
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.66.4'
        assert ref_seg.referenced_sop_class_uid == sop_class_uid
        assert ref_seg.referenced_frame_numbers is None
        assert ref_seg.referenced_segment_numbers == [1]

        src_images = ref_seg.source_images_for_segmentation
        assert len(src_images) == len(self._images_for_segment)
        assert (
            src_images[0].referenced_sop_instance_uid ==
            self._src_seg_instance_uid
        )
        assert ref_seg.source_series_for_segmentation is None

    def test_constructed_with_segment_from_series(self):
        template = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_segment=self._segment_from_series
        )
        assert template.reference_type == codes.DCM.ReferencedSegment
        ref_seg = template.referenced_segment
        assert isinstance(ref_seg, ReferencedSegment)
        assert not ref_seg.has_source_images()
        assert ref_seg.referenced_sop_instance_uid == self._seg_instance_uid
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.66.4'
        assert ref_seg.referenced_sop_class_uid == sop_class_uid
        assert ref_seg.referenced_frame_numbers is None
        assert ref_seg.referenced_segment_numbers == [1]

        assert len(ref_seg.source_images_for_segmentation) == 0
        src_series = ref_seg.source_series_for_segmentation
        assert isinstance(src_series, SourceSeriesForSegmentation)
        assert src_series.value == self._src_seg_series_uid

    def test_construction_all_parameters(self):
        VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_regions=self._regions,
            referenced_real_world_value_map=self._real_world_value_map,
            finding_type=self._finding_type,
            method=self._method,
            algorithm_id=self._algo_id,
            finding_sites=self._finding_sites,
            session=self._session,
            qualitative_evaluations=self._qualitative_evaluations,
            geometric_purpose=self._geometric_purpose
        )

    def test_constructed_with_volume(self):
        image = SourceImageForSegmentation(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=generate_uid()
        )
        volume = VolumeSurface(
            graphic_type=GraphicTypeValues3D.ELLIPSOID,
            graphic_data=np.array([
                [1.0, 2.0, 2.0], [3.0, 2.0, 2.0],
                [2.0, 1.0, 2.0], [2.0, 3.0, 2.0],
                [2.0, 2.0, 1.0], [2.0, 2.0, 3.0],
            ]),
            source_images=[image],
            frame_of_reference_uid=generate_uid()
        )
        measurements = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_volume_surface=volume
        )
        assert len(measurements) == 1
        assert len(measurements[0].ContentSequence) == 4

    def test_constructed_without_reference(self):
        with pytest.raises(ValueError):
            VolumetricROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=self._tracking_identifier
            )

    def test_constructed_with_multiple_references(self):
        with pytest.raises(ValueError):
            VolumetricROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=self._tracking_identifier,
                referenced_regions=self._regions,
                referenced_volume_surface=self._regions
            )


class TestMeasurementReport(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._person_observer_name = 'Bar^Foo'
        self._observer_person_context = ObserverContext(
            observer_type=codes.cid270.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name=self._person_observer_name
            )
        )
        self._device_observer_uid = generate_uid()
        self._observer_device_context = ObserverContext(
            observer_type=codes.cid270.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=self._device_observer_uid
            )
        )
        self._specimen_uid = '1.2.3.4'
        self._specimen_id = 'Specimen-XY'
        self._container_id = 'Container-XY'
        self._specimen_type = codes.SCT.TissueSection
        self._subject_context = SubjectContext(
            subject_class=codes.DCM.Specimen,
            subject_class_specific_context=SubjectContextSpecimen(
                uid=self._specimen_uid,
                identifier=self._specimen_id,
                container_identifier=self._container_id,
                specimen_type=self._specimen_type
            )
        )
        self._observation_context = ObservationContext(
            observer_person_context=self._observer_person_context,
            observer_device_context=self._observer_device_context,
            subject_context=self._subject_context
        )
        self._procedure_reported = codes.cid100.CTPerfusionHeadWithContrastIV
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='planar roi measurements'
        )
        self._source_image_uid = generate_uid()
        self._source_images = [
            SourceImageForMeasurementGroup(
                referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
                referenced_sop_instance_uid=self._source_image_uid
            )
        ]
        self._source_image_region_uid = generate_uid()
        self._image = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=self._source_image_region_uid
        )
        self._region = ImageRegion(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=np.array([[1.0, 1.0], [2.0, 2.0]]),
            source_image=self._image
        )
        self._finding_type = codes.SCT.Neoplasm
        self._finding_site = FindingSite(codes.SCT.Lung)
        self._measurements = [
            Measurement(
                name=codes.cid7469.Area,
                value=10.0,
                unit=codes.cid7181.SquareMillimeter
            ),
            Measurement(
                name=codes.cid7469.Length,
                value=5.0,
                unit=codes.cid7181.Millimeter
            ),
        ]
        self._qualitative_evaluations = [
            QualitativeEvaluation(
                name=codes.SCT.AssociatedMorphology,
                value=Code('35917007', 'SCT', 'Adenocarcinoma')
            ),
            QualitativeEvaluation(
                name=Code('116677004', 'SCT', 'AssociatedTopography'),
                value=codes.SCT.Lung
            ),
        ]
        self._image_group = MeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            finding_type=self._finding_type,
            finding_sites=[self._finding_site],
            measurements=self._measurements,
            qualitative_evaluations=self._qualitative_evaluations,
            source_images=self._source_images,
        )
        self._roi_group = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_region=self._region,
            finding_type=self._finding_type,
            finding_sites=[self._finding_site],
            measurements=self._measurements,
            qualitative_evaluations=self._qualitative_evaluations,
        )
        self._roi_group_3d = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_regions=[self._region],
            finding_type=self._finding_type,
            finding_sites=[self._finding_site],
            measurements=self._measurements,
            qualitative_evaluations=self._qualitative_evaluations,
        )

    def test_construction_image(self):
        measurement_report = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=self._procedure_reported,
            imaging_measurements=[self._image_group]
        )
        item = measurement_report[0]
        assert len(item.ContentSequence) == 13

        template_item = item.ContentTemplateSequence[0]
        assert template_item.TemplateIdentifier == '1500'

        content_item_expectations = [
            # Observation context
            (0, '121049'),
            # Observer context - Person
            (1, '121005'),
            (2, '121008'),
            # Observer context - Device
            (3, '121005'),
            (4, '121012'),
            # Subject context - Specimen
            (5, '121024'),
            (6, '121039'),
            (7, '121041'),
            (8, '371439000'),
            (9, '111700'),
            # Procedure reported
            (10, '121058'),
            # Image library
            (11, '111028'),
            # Imaging measurements
            (12, '126010'),
        ]
        for index, value in content_item_expectations:
            content_item = item.ContentSequence[index]
            assert content_item.ConceptNameCodeSequence[0].CodeValue == value

        matches = measurement_report.get_image_measurement_groups(
            finding_type=self._finding_type
        )
        assert len(matches) == 1

        matches = measurement_report.get_image_measurement_groups(
            finding_type=codes.SCT.Tissue
        )
        assert len(matches) == 0

        matches = measurement_report.get_image_measurement_groups(
            finding_site=self._finding_site.value
        )
        assert len(matches) == 1

        matches = measurement_report.get_image_measurement_groups(
            finding_site=codes.SCT.Colon
        )
        assert len(matches) == 0

        matches = measurement_report.get_image_measurement_groups(
            tracking_uid=self._tracking_identifier[1].value
        )
        assert len(matches) == 1

        matches = measurement_report.get_image_measurement_groups(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
        )
        assert len(matches) == 1

        matches = measurement_report.get_image_measurement_groups(
            referenced_sop_instance_uid=self._source_image_uid,
        )
        assert len(matches) == 1

        matches = measurement_report.get_image_measurement_groups(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=self._source_image_uid,
        )
        assert len(matches) == 1

        matches = measurement_report.get_image_measurement_groups(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=self._source_image_uid,
        )
        assert len(matches) == 1

        matches = measurement_report.get_image_measurement_groups(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.1',
        )
        assert len(matches) == 0

        matches = measurement_report.get_image_measurement_groups(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=UID(),
        )
        assert len(matches) == 0

    def test_construction_planar(self):
        measurement_report = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=self._procedure_reported,
            imaging_measurements=[self._roi_group]
        )
        item = measurement_report[0]
        assert len(item.ContentSequence) == 13

        template_item = item.ContentTemplateSequence[0]
        assert template_item.TemplateIdentifier == '1500'

        content_item_expectations = [
            # Observation context
            (0, '121049'),
            # Observer context - Person
            (1, '121005'),
            (2, '121008'),
            # Observer context - Device
            (3, '121005'),
            (4, '121012'),
            # Subject context - Specimen
            (5, '121024'),
            (6, '121039'),
            (7, '121041'),
            (8, '371439000'),
            (9, '111700'),
            # Procedure reported
            (10, '121058'),
            # Image library
            (11, '111028'),
            # Imaging measurements
            (12, '126010'),
        ]
        for index, value in content_item_expectations:
            content_item = item.ContentSequence[index]
            assert content_item.ConceptNameCodeSequence[0].CodeValue == value

        matches = measurement_report.get_planar_roi_measurement_groups(
            finding_type=self._finding_type
        )
        assert len(matches) == 1

        matches = measurement_report.get_planar_roi_measurement_groups(
            finding_type=codes.SCT.Tissue
        )
        assert len(matches) == 0

        matches = measurement_report.get_planar_roi_measurement_groups(
            finding_site=self._finding_site.value
        )
        assert len(matches) == 1

        matches = measurement_report.get_planar_roi_measurement_groups(
            finding_site=codes.SCT.Colon
        )
        assert len(matches) == 0

        matches = measurement_report.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion
        )
        assert len(matches) == 1

        matches = measurement_report.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ReferencedSegmentationFrame
        )
        assert len(matches) == 0

        matches = measurement_report.get_planar_roi_measurement_groups(
            tracking_uid=self._tracking_identifier[1].value
        )
        assert len(matches) == 1

        matches = measurement_report.get_planar_roi_measurement_groups(
            tracking_uid=generate_uid()
        )
        assert len(matches) == 0

    def test_construction_volumetric(self):
        measurement_report = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=self._procedure_reported,
            imaging_measurements=[self._roi_group_3d]
        )
        item = measurement_report[0]
        assert len(item.ContentSequence) == 13

        template_item = item.ContentTemplateSequence[0]
        assert template_item.TemplateIdentifier == '1500'

        content_item_expectations = [
            # Observation context
            (0, '121049'),
            # Observer context - Person
            (1, '121005'),
            (2, '121008'),
            # Observer context - Device
            (3, '121005'),
            (4, '121012'),
            # Subject context - Specimen
            (5, '121024'),
            (6, '121039'),
            (7, '121041'),
            (8, '371439000'),
            (9, '111700'),
            # Procedure reported
            (10, '121058'),
            # Image library
            (11, '111028'),
            # Imaging measurements
            (12, '126010'),
        ]
        for index, value in content_item_expectations:
            content_item = item.ContentSequence[index]
            assert content_item.ConceptNameCodeSequence[0].CodeValue == value

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            finding_type=self._finding_type
        )
        assert len(matches) == 1

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            finding_type=codes.SCT.Tissue
        )
        assert len(matches) == 0

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            finding_site=self._finding_site.value
        )
        assert len(matches) == 1

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            finding_site=codes.SCT.Colon
        )
        assert len(matches) == 0

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion
        )
        assert len(matches) == 1

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.ReferencedSegment
        )
        assert len(matches) == 0

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            tracking_uid=self._tracking_identifier[1].value
        )
        assert len(matches) == 1

        matches = measurement_report.get_volumetric_roi_measurement_groups(
            tracking_uid=generate_uid()
        )
        assert len(matches) == 0

    def test_from_sequence(self):
        measurement_report = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=self._procedure_reported,
            imaging_measurements=[self._roi_group]
        )
        template = MeasurementReport.from_sequence(measurement_report)
        assert isinstance(template, MeasurementReport)
        assert len(template) == 1
        root_item = template[0]
        assert isinstance(root_item, ContainerContentItem)

        # Observer contexts
        observer_contexts = template.get_observer_contexts()
        assert len(observer_contexts) == 2
        assert isinstance(observer_contexts[0], ObserverContext)
        person_observer_contexts = template.get_observer_contexts(
            observer_type=codes.DCM.Person
        )
        assert len(person_observer_contexts) == 1
        person_observer = person_observer_contexts[0]
        assert isinstance(person_observer, ObserverContext)
        assert person_observer.observer_type == codes.DCM.Person
        person_observer_attrs = person_observer.observer_identifying_attributes
        assert person_observer_attrs.name == self._person_observer_name
        device_observer_contexts = template.get_observer_contexts(
            observer_type=codes.DCM.Device
        )
        assert len(device_observer_contexts) == 1
        device_observer = device_observer_contexts[0]
        assert isinstance(device_observer, ObserverContext)
        assert device_observer.observer_type == codes.DCM.Device
        device_observer_attrs = device_observer.observer_identifying_attributes
        assert device_observer_attrs.uid == self._device_observer_uid

        # Subject contexts
        subject_contexts = template.get_subject_contexts()
        assert len(subject_contexts) == 1
        assert isinstance(subject_contexts[0], SubjectContext)
        specimen_subject_contexts = template.get_subject_contexts(
            subject_class=codes.DCM.Specimen
        )
        assert len(specimen_subject_contexts) == 1
        specimen_subject = specimen_subject_contexts[0]
        assert isinstance(specimen_subject, SubjectContext)
        assert specimen_subject.subject_class == codes.DCM.Specimen
        specimen_attrs = specimen_subject.subject_class_specific_context
        assert specimen_attrs.specimen_uid == self._specimen_uid
        assert specimen_attrs.specimen_identifier == self._specimen_id
        assert specimen_attrs.specimen_type == self._specimen_type
        assert specimen_attrs.container_identifier == self._container_id
        device_subject_contexts = template.get_subject_contexts(
            subject_class=codes.DCM.Device
        )
        assert len(device_subject_contexts) == 0

        # Imaging Measurements
        planar_rois = template.get_planar_roi_measurement_groups()
        assert len(planar_rois) == 1
        group = planar_rois[0]
        assert isinstance(
            group,
            PlanarROIMeasurementsAndQualitativeEvaluations
        )
        assert isinstance(group[0], ContainerContentItem)
        assert group[0].name == codes.DCM.MeasurementGroup
        # Finding Type
        assert isinstance(group.finding_type, CodedConcept)
        assert group.finding_type == self._finding_type
        # Finding Site
        assert len(group.finding_sites) == 1
        assert isinstance(group.finding_sites[0], FindingSite)
        # Tracking Identifier
        assert group.finding_sites[0].value == self._finding_site.value
        assert group.tracking_uid == self._tracking_identifier[1].value
        # Image Region
        assert isinstance(group.roi, ImageRegion)
        assert group.roi.graphic_type == GraphicTypeValues.CIRCLE
        assert isinstance(group.roi.value, np.ndarray)
        assert group.roi.value.shape == (2, 2)
        # Measurements and Qualitative Evaluations
        measurements = group.get_measurements()
        assert len(measurements) == 2
        qualitative_evaluations = group.get_qualitative_evaluations()
        assert len(qualitative_evaluations) == 2
        measurements = group.get_measurements(
            name=codes.SCT.Area
        )
        assert len(measurements) == 1
        assert isinstance(measurements[0], Measurement)
        qualitative_evaluations = group.get_qualitative_evaluations(
            name=codes.SCT.AssociatedMorphology
        )
        assert len(qualitative_evaluations) == 1
        assert isinstance(qualitative_evaluations[0], QualitativeEvaluation)


class TestEnhancedSR(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._ref_dataset = Dataset()
        self._ref_dataset.PatientID = '1'
        self._ref_dataset.PatientName = 'Doe^John'
        self._ref_dataset.PatientBirthDate = '20000101'
        self._ref_dataset.PatientSex = 'O'
        self._ref_dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2.2'
        self._ref_dataset.SOPInstanceUID = generate_uid()
        self._ref_dataset.SeriesInstanceUID = generate_uid()
        self._ref_dataset.StudyInstanceUID = generate_uid()
        self._ref_dataset.AccessionNumber = '2'
        self._ref_dataset.StudyID = '3'
        self._ref_dataset.StudyDate = datetime.now().date()
        self._ref_dataset.StudyTime = datetime.now().time()
        self._ref_dataset.ReferringPhysicianName = 'doctor'

        self._series_instance_uid = generate_uid()
        self._series_number = 3
        self._sop_instance_uid = generate_uid()
        self._instance_number = 4
        self._institution_name = 'institute'
        self._department_name = 'department'
        self._manufacturer = 'manufacturer'
        self._performed_procedures = [codes.LN.CTUnspecifiedBodyRegion]

        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Bar^Foo'
            )
        )
        observer_device_context = ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=generate_uid()
            )
        )
        observation_context = ObservationContext(
            observer_person_context=observer_person_context,
            observer_device_context=observer_device_context,
        )
        referenced_region = ImageRegion(
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=np.array([
                (165.0, 200.0),
                (170.0, 200.0),
                (170.0, 220.0),
                (165.0, 220.0),
                (165.0, 200.0),
            ]),
            source_image=SourceImageForRegion(
                referenced_sop_class_uid=self._ref_dataset.SOPClassUID,
                referenced_sop_instance_uid=self._ref_dataset.SOPInstanceUID
            )
        )
        finding_sites = [
            FindingSite(
                anatomic_location=codes.SCT.CervicoThoracicSpine,
                topographical_modifier=codes.SCT.VertebralForamen
            ),
        ]
        measurements = [
            Measurement(
                name=codes.SCT.AreaOfDefinedRegion,
                value=1.7,
                unit=codes.UCUM.SquareMillimeter,
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
                properties=MeasurementProperties(
                    normality=CodedConcept(
                        value="17621005",
                        meaning="Normal",
                        scheme_designator="SCT"
                    ),
                    level_of_significance=codes.SCT.NotSignificant
                )
            )
        ]
        imaging_measurements = [
            PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=TrackingIdentifier(
                    uid=generate_uid(),
                    identifier='Planar ROI Measurements'
                ),
                referenced_region=referenced_region,
                finding_type=codes.SCT.SpinalCord,
                measurements=measurements,
                finding_sites=finding_sites
            )
        ]
        self._content = MeasurementReport(
            observation_context=observation_context,
            procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
            imaging_measurements=imaging_measurements
        )[0]

    def test_construction(self):
        report = EnhancedSR(
            evidence=[self._ref_dataset],
            content=self._content,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            institution_name=self._institution_name,
            institutional_department_name=self._department_name,
            manufacturer=self._manufacturer,
            performed_procedure_codes=self._performed_procedures
        )
        assert report.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.22'

    def test_evidence_missing(self):
        with pytest.raises(ValueError):
            EnhancedSR(
                evidence=[],
                content=self._content,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                institution_name=self._institution_name,
                institutional_department_name=self._department_name,
                manufacturer=self._manufacturer
            )


class TestComprehensiveSR(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._ref_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )

        self._series_instance_uid = generate_uid()
        self._series_number = 3
        self._sop_instance_uid = generate_uid()
        self._instance_number = 4
        self._institution_name = 'institute'
        self._department_name = 'department'
        self._manufacturer = 'manufacturer'
        self._procedure_reported = codes.LN.CTUnspecifiedBodyRegion
        self._performed_procedures = [codes.LN.CTUnspecifiedBodyRegion]

        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Bar^Foo'
            )
        )
        observer_device_context = ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=generate_uid()
            )
        )
        self._observation_context = ObservationContext(
            observer_person_context=observer_person_context,
            observer_device_context=observer_device_context,
        )

        referenced_region = ImageRegion(
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=np.array([
                (165.0, 200.0),
                (170.0, 200.0),
                (170.0, 220.0),
                (165.0, 220.0),
                (165.0, 200.0),
            ]),
            source_image=SourceImageForRegion(
                referenced_sop_class_uid=self._ref_dataset.SOPClassUID,
                referenced_sop_instance_uid=self._ref_dataset.SOPInstanceUID
            )
        )
        finding_sites = [
            FindingSite(
                anatomic_location=codes.SCT.CervicoThoracicSpine,
                topographical_modifier=codes.SCT.VertebralForamen
            ),
        ]
        measurements = [
            Measurement(
                name=codes.SCT.AreaOfDefinedRegion,
                value=1.7,
                unit=codes.UCUM.SquareMillimeter,
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
                properties=MeasurementProperties(
                    normality=CodedConcept(
                        value="17621005",
                        meaning="Normal",
                        scheme_designator="SCT"
                    ),
                    level_of_significance=codes.SCT.NotSignificant
                )
            )
        ]
        measurement_group = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(
                uid=generate_uid(),
                identifier='Planar ROI Measurements'
            ),
            referenced_region=referenced_region,
            finding_type=codes.SCT.SpinalCord,
            measurements=measurements,
            finding_sites=finding_sites
        )
        self._imaging_measurements = [measurement_group]
        self._content = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=self._procedure_reported,
            imaging_measurements=self._imaging_measurements
        )[0]

    def test_construction(self):
        report = ComprehensiveSR(
            evidence=[self._ref_dataset],
            content=self._content,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            institution_name=self._institution_name,
            institutional_department_name=self._department_name,
            manufacturer=self._manufacturer,
            performed_procedure_codes=self._performed_procedures,
        )
        assert report.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.33'

        ref_evd_items = report.CurrentRequestedProcedureEvidenceSequence
        assert len(ref_evd_items) == 1
        with pytest.raises(AttributeError):
            assert report.PertinentOtherEvidenceSequence

    def test_evidence_duplication(self):
        report = Comprehensive3DSR(
            evidence=[self._ref_dataset, self._ref_dataset],
            content=self._content,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            institution_name=self._institution_name,
            institutional_department_name=self._department_name,
            manufacturer=self._manufacturer
        )
        ref_evd_items = report.CurrentRequestedProcedureEvidenceSequence
        assert len(ref_evd_items) == 1
        with pytest.raises(AttributeError):
            assert report.PertinentOtherEvidenceSequence

    def test_evidence_missing(self):
        ref_dataset = deepcopy(self._ref_dataset)
        ref_dataset.SOPInstanceUID = '1.2.3.4'
        with pytest.raises(ValueError):
            Comprehensive3DSR(
                evidence=[ref_dataset],
                content=self._content,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                institution_name=self._institution_name,
                institutional_department_name=self._department_name,
                manufacturer=self._manufacturer
            )

    def test_evidence_multiple_studies(self):
        ref_dataset = deepcopy(self._ref_dataset)
        ref_dataset.StudyInstanceUID = '1.2.6'
        ref_dataset.SeriesInstanceUID = '1.2.7'
        ref_dataset.SOPInstanceUID = '1.2.9'
        referenced_region = ImageRegion(
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=np.array([
                (65.0, 100.0),
                (70.0, 100.0),
                (70.0, 120.0),
                (65.0, 120.0),
                (65.0, 100.0),
            ]),
            source_image=SourceImageForRegion(
                referenced_sop_class_uid=ref_dataset.SOPClassUID,
                referenced_sop_instance_uid=ref_dataset.SOPInstanceUID
            )
        )
        finding_sites = [
            FindingSite(anatomic_location=codes.SCT.CervicoThoracicSpine),
        ]
        measurements = [
            Measurement(
                name=codes.SCT.AreaOfDefinedRegion,
                value=0.7,
                unit=codes.UCUM.SquareMillimeter,
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
            )
        ]
        measurement_group = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(
                uid=generate_uid(),
                identifier='Planar ROI Measurements'
            ),
            referenced_region=referenced_region,
            finding_type=codes.SCT.SpinalCord,
            measurements=measurements,
            finding_sites=finding_sites
        )
        imaging_measurements = deepcopy(self._imaging_measurements)
        imaging_measurements.append(measurement_group)
        content = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=self._procedure_reported,
            imaging_measurements=imaging_measurements
        )[0]
        report = Comprehensive3DSR(
            evidence=[self._ref_dataset, ref_dataset],
            content=content,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            institution_name=self._institution_name,
            institutional_department_name=self._department_name,
            manufacturer=self._manufacturer
        )
        ref_evd_items = report.CurrentRequestedProcedureEvidenceSequence
        assert len(ref_evd_items) == 2
        with pytest.raises(AttributeError):
            assert report.PertinentOtherEvidenceSequence


class TestComprehensive3DSR(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._ref_dataset = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._sr_document = dcmread(
            str(data_dir.joinpath('test_files', 'sr_document.dcm'))
        )

        self._series_instance_uid = generate_uid()
        self._series_number = 3
        self._sop_instance_uid = generate_uid()
        self._instance_number = 4
        self._institution_name = 'institute'
        self._department_name = 'department'
        self._manufacturer = 'manufacturer'
        self._performed_procedures = [codes.LN.CTUnspecifiedBodyRegion]

        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Bar^Foo'
            )
        )
        observer_device_context = ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=generate_uid()
            )
        )
        observation_context = ObservationContext(
            observer_person_context=observer_person_context,
            observer_device_context=observer_device_context,
        )
        referenced_region = ImageRegion3D(
            graphic_type=GraphicTypeValues3D.POLYGON,
            graphic_data=np.array([
                (165.0, 200.0, 134.0),
                (170.0, 200.0, 134.0),
                (170.0, 220.0, 134.0),
                (165.0, 220.0, 134.0),
                (165.0, 200.0, 134.0),
            ]),
            frame_of_reference_uid=self._ref_dataset.FrameOfReferenceUID
        )
        finding_sites = [
            FindingSite(
                anatomic_location=codes.SCT.CervicoThoracicSpine,
                topographical_modifier=codes.SCT.VertebralForamen
            ),
        ]
        measurements = [
            Measurement(
                name=codes.SCT.AreaOfDefinedRegion,
                value=1.7,
                unit=codes.UCUM.SquareMillimeter,
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
                properties=MeasurementProperties(
                    normality=CodedConcept(
                        value="17621005",
                        meaning="Normal",
                        scheme_designator="SCT"
                    ),
                    level_of_significance=codes.SCT.NotSignificant
                )
            )
        ]
        imaging_measurements = [
            PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=TrackingIdentifier(
                    uid=generate_uid(),
                    identifier='Planar ROI Measurements'
                ),
                referenced_region=referenced_region,
                finding_type=codes.SCT.SpinalCord,
                measurements=measurements,
                finding_sites=finding_sites
            )
        ]
        self._content = MeasurementReport(
            observation_context=observation_context,
            procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
            imaging_measurements=imaging_measurements
        )[0]

    def test_construction(self):
        report = Comprehensive3DSR(
            evidence=[self._ref_dataset],
            content=self._content,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            institution_name=self._institution_name,
            institutional_department_name=self._department_name,
            manufacturer=self._manufacturer,
            performed_procedure_codes=self._performed_procedures,
        )
        assert report.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.34'
        assert report.PatientID == self._ref_dataset.PatientID
        assert report.PatientName == self._ref_dataset.PatientName
        assert report.StudyInstanceUID == self._ref_dataset.StudyInstanceUID
        assert report.AccessionNumber == self._ref_dataset.AccessionNumber
        assert report.SeriesInstanceUID == self._series_instance_uid
        assert report.SeriesNumber == self._series_number
        assert report.SOPInstanceUID == self._sop_instance_uid
        assert report.InstanceNumber == self._instance_number
        assert report.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.34'
        assert report.InstitutionName == self._institution_name
        assert report.Manufacturer == self._manufacturer
        assert report.Modality == 'SR'

        with pytest.raises(AttributeError):
            assert report.CurrentRequestedProcedureEvidenceSequence
        unref_evd_items = report.PertinentOtherEvidenceSequence
        assert len(unref_evd_items) == 1

    def test_from_dataset(self):
        report = Comprehensive3DSR.from_dataset(self._sr_document)
        assert isinstance(report, Comprehensive3DSR)
        assert isinstance(report.content, ContentSequence)
        assert isinstance(report.content, MeasurementReport)

    def test_evidence_duplication(self):
        report = Comprehensive3DSR(
            evidence=[self._ref_dataset, self._ref_dataset],
            content=self._content,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            institution_name=self._institution_name,
            institutional_department_name=self._department_name,
            manufacturer=self._manufacturer
        )
        unref_evd_items = report.PertinentOtherEvidenceSequence
        assert len(unref_evd_items) == 1


class TestSRUtilities(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._sr_document = dcmread(
            str(data_dir.joinpath('test_files', 'sr_document.dcm'))
        )

    def test_find_content_items(self):
        items = find_content_items(self._sr_document)
        assert len(items) == 8

    def test_find_content_items_filtered_by_name(self):
        items = find_content_items(
            self._sr_document,
            name=codes.DCM.ProcedureReported
        )
        assert len(items) == 1
        name_code_value = items[0].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value == codes.DCM.ProcedureReported.value

    def test_find_content_items_filtered_by_relationship_type(self):
        items = find_content_items(
            self._sr_document,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        assert len(items) == 4
        name_code_value_1 = items[0].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value_1 == codes.DCM.ObserverType.value
        name_code_value_2 = items[1].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value_2 == codes.DCM.PersonObserverName.value
        name_code_value_3 = items[2].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value_3 == codes.DCM.ObserverType.value
        name_code_value_4 = items[3].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value_4 == codes.DCM.DeviceObserverUID.value

    def test_find_content_items_filtered_by_value_type(self):
        items = find_content_items(
            self._sr_document,
            value_type=ValueTypeValues.UIDREF
        )
        assert len(items) == 1
        name_code_value = items[0].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value == codes.DCM.DeviceObserverUID.value

    def test_find_content_items_filtered_by_value_type_relationship_type(self):
        items = find_content_items(
            self._sr_document,
            value_type=ValueTypeValues.CODE,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        assert len(items) == 2
        name_code_value_1 = items[0].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value_1 == codes.DCM.ObserverType.value
        name_code_value_2 = items[1].ConceptNameCodeSequence[0].CodeValue
        assert name_code_value_2 == codes.DCM.ObserverType.value

    def test_find_content_items_recursively(self):
        items = find_content_items(self._sr_document, recursive=True)
        assert len(items) == 20

    def test_find_content_items_filter_by_name_recursively(self):
        items = find_content_items(
            self._sr_document,
            name=codes.DCM.TrackingUniqueIdentifier,
            recursive=True
        )
        assert len(items) == 2

    def test_find_content_items_filter_by_value_type_recursively(self):
        items = find_content_items(
            self._sr_document,
            value_type=ValueTypeValues.SCOORD,
            recursive=True
        )
        assert len(items) == 1

    def test_find_content_items_filter_by_value_type_recursively_1(self):
        items = find_content_items(
            self._sr_document,
            value_type=ValueTypeValues.CODE,
            recursive=True
        )
        assert len(items) == 9

    def test_find_content_items_filter_by_relationship_type_recursively(self):
        items = find_content_items(
            self._sr_document,
            relationship_type=RelationshipTypeValues.CONTAINS,
            recursive=True
        )
        assert len(items) == 6


class TestGetPlanarMeasurementGroups(unittest.TestCase):

    """Integration test for SR parsing.

    Constructs an SR with a measurement report containing a variety of planar
    measurement groups, and tests the ability to filter the measurement groups
    by various parameters.

    """

    def setUp(self):
        super().setUp()

        # Read in series of source images
        self._ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ref_seg = dcmread(
            'data/test_files/seg_image_ct_binary_single_frame.dcm'
        )

        # Measurement group with image region of type polyline
        self._polyline_src_sop_uid = self._ct_series[0].SOPInstanceUID
        self._polyline_src_sop_class_uid = self._ct_series[0].SOPClassUID
        self._polyline_src = SourceImageForRegion(
            referenced_sop_class_uid=self._polyline_src_sop_class_uid,
            referenced_sop_instance_uid=self._polyline_src_sop_uid,
        )
        self._polyline = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [1.0, 1.0]
        ])
        self._img_reg_polyline = ImageRegion(
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=self._polyline,
            source_image=self._polyline_src
        )
        self._polyline_uid = UID()
        self._polyline_id = 'polyline'
        polyline_tracker = TrackingIdentifier(
            uid=self._polyline_uid,
            identifier=self._polyline_id
        )
        self._polyline_grp = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=polyline_tracker,
            referenced_region=self._img_reg_polyline,
        )

        # Measurement group with image region of type circle
        self._circle_src_sop_uid = self._ct_series[1].SOPInstanceUID
        self._circle_src_sop_class_uid = self._ct_series[1].SOPClassUID
        self._circle_src = SourceImageForRegion(
            referenced_sop_class_uid=self._circle_src_sop_class_uid,
            referenced_sop_instance_uid=self._circle_src_sop_uid,
        )
        self._circle = np.array([
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        self._img_reg_circle = ImageRegion(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            source_image=self._circle_src
        )
        self._circle_uid = UID()
        self._circle_id = 'circle'
        circle_tracker = TrackingIdentifier(
            uid=self._circle_uid,
            identifier=self._circle_id
        )
        self._circle_grp = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=circle_tracker,
            referenced_region=self._img_reg_circle,
        )

        # Measurement group with image region 3D of type point
        self._point_src_sop_uid = self._ct_series[2].SOPInstanceUID
        self._point_src_sop_class_uid = self._ct_series[2].SOPClassUID
        self._point_src = SourceImageForRegion(
            referenced_sop_class_uid=self._point_src_sop_class_uid,
            referenced_sop_instance_uid=self._point_src_sop_uid,
        )
        self._point = np.array([[1.0, 2.0]])
        self._img_reg_point = ImageRegion(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=self._point,
            source_image=self._point_src
        )
        self._point_uid = UID()
        self._point_id = 'point'
        point_tracker = TrackingIdentifier(
            uid=self._point_uid,
            identifier=self._point_id
        )
        self._point_grp = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=point_tracker,
            referenced_region=self._img_reg_point,
        )

        # Measurement group with image region 3D of type point
        self._point3d_src_sop_uid = self._ct_series[3].SOPInstanceUID
        self._point3d_src_sop_class_uid = self._ct_series[3].SOPClassUID
        self._point3d = np.array([[1.0, 2.0, 3.0]])
        self._img_reg_point3d = ImageRegion3D(
            graphic_type=GraphicTypeValues3D.POINT,
            graphic_data=self._point3d,
            frame_of_reference_uid=self._ct_series[0].FrameOfReferenceUID
        )
        self._point3d_uid = UID()
        self._point3d_id = 'point3d'
        point3d_tracker = TrackingIdentifier(
            uid=self._point3d_uid,
            identifier=self._point3d_id
        )
        self._point3d_grp = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=point3d_tracker,
            referenced_region=self._img_reg_point3d,
        )

        # Measurement group with segmentation frame
        self._seg_frame_src_image = SourceImageForSegmentation(
            referenced_sop_class_uid=self._ct_series[0].SOPClassUID,
            referenced_sop_instance_uid=self._ct_series[0].SOPInstanceUID,
        )
        self._ref_seg_frame = ReferencedSegmentationFrame(
            sop_class_uid=self._ref_seg.SOPClassUID,
            sop_instance_uid=self._ref_seg.SOPInstanceUID,
            frame_number=1,
            segment_number=1,
            source_image=self._seg_frame_src_image
        )
        self._seg_uid = UID()
        self._seg_id = 'seg_frame'
        seg_tracker = TrackingIdentifier(
            uid=self._seg_uid,
            identifier=self._seg_id
        )
        self._seg_grp = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=seg_tracker,
            referenced_segment=self._ref_seg_frame,
        )

        # Save the sr and re-read it
        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Bar^Foo'
            )
        )
        observer_device_context = ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=UID()
            )
        )
        self._observation_context = ObservationContext(
            observer_person_context=observer_person_context,
            observer_device_context=observer_device_context,
        )
        self._all_grps = [
            self._polyline_grp,
            self._circle_grp,
            self._point_grp,
            self._point3d_grp,
            self._seg_grp
        ]
        report = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
            imaging_measurements=self._all_grps,
        )
        sr = Comprehensive3DSR(
            evidence=self._ct_series + [self._ref_seg],
            content=report[0],
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
            record_evidence=False
        )

        # Write out to a buffer and read back in to revert to a standard
        # pydicom dataset and test the conversion functionality sets up
        # everything correctly for parsing.
        with BytesIO() as buf:
            sr.save_as(buf)
            buf.seek(0)
            sr_from_file = dcmread(buf)
        self._sr = Comprehensive3DSR.from_dataset(sr_from_file)
        self._content = self._sr.content

    def test_all_groups(self):
        grps = self._content.get_planar_roi_measurement_groups()
        assert len(grps) == len(self._all_grps)

    def test_get_image_region_groups(self):
        # Find all groups with reference type ImageRegion
        # (includes the polyline, circle and point x 2 groups)
        grps = self._content.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion
        )
        assert len(grps) == 4

    def test_get_polyline_groups(self):
        # Find the polyline group with and without explicitly
        # specifying the reference_type as ImageRegion
        grps = self._content.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion,
            graphic_type=GraphicTypeValues.POLYLINE
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._polyline_id
        assert grps[0].tracking_uid == self._polyline_uid
        grps = self._content.get_planar_roi_measurement_groups(
            graphic_type=GraphicTypeValues.POLYLINE
        )
        assert grps[0].tracking_identifier == self._polyline_id
        assert grps[0].tracking_uid == self._polyline_uid
        assert len(grps) == 1
        assert np.array_equal(grps[0].roi.value, self._polyline)

    def test_get_circle_groups(self):
        # Find the circle group with and without explicitly
        # specifying the reference_type as ImageRegion
        grps = self._content.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion,
            graphic_type=GraphicTypeValues.CIRCLE
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._circle_id
        assert grps[0].tracking_uid == self._circle_uid
        grps = self._content.get_planar_roi_measurement_groups(
            graphic_type=GraphicTypeValues.CIRCLE
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._circle_id
        assert grps[0].tracking_uid == self._circle_uid
        assert np.array_equal(grps[0].roi.value, self._circle)

    def test_get_point_groups(self):
        # Find the point group
        grps = self._content.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion,
            graphic_type=GraphicTypeValues.POINT
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._point_id
        assert grps[0].tracking_uid == self._point_uid
        assert np.array_equal(grps[0].roi.value, self._point)

    def test_get_point3d_groups(self):
        # Find the point group
        grps = self._content.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion,
            graphic_type=GraphicTypeValues3D.POINT
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._point3d_id
        assert grps[0].tracking_uid == self._point3d_uid
        assert np.array_equal(grps[0].roi.value, self._point3d)

    def test_find_seg_groups(self):
        # Find the seg group
        grps = self._content.get_planar_roi_measurement_groups(
            reference_type=codes.DCM.ReferencedSegmentationFrame,
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._seg_id
        assert grps[0].tracking_uid == self._seg_uid
        ref_seg_frame = grps[0].referenced_segmentation_frame
        assert isinstance(ref_seg_frame, ReferencedSegmentationFrame)
        ins_uid = ref_seg_frame.referenced_sop_instance_uid
        class_uid = ref_seg_frame.referenced_sop_class_uid
        assert ins_uid == self._ref_seg.SOPInstanceUID
        assert class_uid == self._ref_seg.SOPClassUID

    def test_get_groups_invalid_reference_types(self):
        with pytest.raises(ValueError):
            # ReferencedSegment is invalid for planar groups
            self._content.get_planar_roi_measurement_groups(
                reference_type=codes.DCM.ReferencedSegment
            )

    def test_get_volumetric_groups(self):
        grps = self._content.get_volumetric_roi_measurement_groups()
        assert len(grps) == 0

    def test_get_groups_by_tracking_id(self):
        grps = self._content.get_planar_roi_measurement_groups(
            tracking_uid=self._polyline_uid
        )
        assert len(grps) == 1
        assert grps[0].tracking_uid == self._polyline_uid

    def test_get_groups_by_ref_uid_1(self):
        # Should match the polyline and seg groups
        grps = self._content.get_planar_roi_measurement_groups(
            referenced_sop_instance_uid=self._ct_series[0].SOPInstanceUID
        )
        assert len(grps) == 2
        found_tracking_uids = {g.tracking_uid for g in grps}
        assert found_tracking_uids == {self._polyline_uid, self._seg_uid}

    def test_get_groups_by_ref_uid_2(self):
        # Should match the seg group
        grps = self._content.get_planar_roi_measurement_groups(
            referenced_sop_class_uid=SegmentationStorage
        )
        assert len(grps) == 1
        assert grps[0].tracking_uid == self._seg_uid

    def test_get_groups_by_ref_uid_3(self):
        # Should match the seg group
        grps = self._content.get_planar_roi_measurement_groups(
            referenced_sop_instance_uid=self._ref_seg.SOPInstanceUID
        )
        assert len(grps) == 1
        assert grps[0].tracking_uid == self._seg_uid

    def test_get_groups_invalid_graphic_type_1(self):
        # Any graphic type is invalid when reference type is not ImageRegion
        with pytest.raises(ValueError):
            self._content.get_planar_roi_measurement_groups(
                reference_type=codes.DCM.ReferencedSegmentationFrame,
                graphic_type=GraphicTypeValues.CIRCLE
            )

    def test_get_groups_invalid_graphic_type_2(self):
        # Multipoint is always invalid
        with pytest.raises(ValueError):
            self._content.get_planar_roi_measurement_groups(
                graphic_type=GraphicTypeValues.MULTIPOINT
            )

    def test_get_groups_invalid_graphic_type_3d(self):
        # Multipoint is always invalid
        with pytest.raises(ValueError):
            self._content.get_planar_roi_measurement_groups(
                graphic_type=GraphicTypeValues3D.MULTIPOINT
            )

    def test_get_groups_with_ref_uid_and_graphic_type_3d(self):
        # Multipoint is always invalid
        with pytest.raises(TypeError):
            self._content.get_planar_roi_measurement_groups(
                graphic_type=GraphicTypeValues3D.POINT,
                referenced_sop_instance_uid=UID()
            )


class TestGetVolumetricMeasurementGroups(unittest.TestCase):

    """Integration test for SR parsing.

    Constructs an SR with a measurement report containing a variety of
    volumetric measurement groups, and tests the ability to filter the
    measurement groups by various parameters.

    """

    def setUp(self):
        super().setUp()

        # Read in series of source images
        self._ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ref_seg = dcmread(
            'data/test_files/seg_image_ct_binary_single_frame.dcm'
        )

        # Measurement group with image region 3D of type polyline
        self._polyline_src_sop_uid = self._ct_series[0].SOPInstanceUID
        self._polyline_src_sop_class_uid = self._ct_series[0].SOPClassUID
        self._polyline_src = SourceImageForRegion(
            referenced_sop_class_uid=self._polyline_src_sop_class_uid,
            referenced_sop_instance_uid=self._polyline_src_sop_uid,
        )
        self._polyline = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [1.0, 1.0]
        ])
        self._img_reg_polyline = ImageRegion(
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=self._polyline,
            source_image=self._polyline_src
        )
        self._polyline_uid = UID()
        self._polyline_id = 'polyline'
        polyline_tracker = TrackingIdentifier(
            uid=self._polyline_uid,
            identifier=self._polyline_id
        )
        self._polyline_grp = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=polyline_tracker,
            referenced_regions=[self._img_reg_polyline],
        )

        # Measurement group with image region 3D of type circle
        self._circle_src_sop_uid = self._ct_series[1].SOPInstanceUID
        self._circle_src_sop_class_uid = self._ct_series[1].SOPClassUID
        self._circle_src = SourceImageForRegion(
            referenced_sop_class_uid=self._circle_src_sop_class_uid,
            referenced_sop_instance_uid=self._circle_src_sop_uid,
        )
        self._circle = np.array([
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        self._img_reg_circle = ImageRegion(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            source_image=self._circle_src
        )
        self._circle_uid = UID()
        self._circle_id = 'circle'
        circle_tracker = TrackingIdentifier(
            uid=self._circle_uid,
            identifier=self._circle_id
        )
        self._circle_grp = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=circle_tracker,
            referenced_regions=[self._img_reg_circle],
        )

        # Measurement group with image region 3D of type point
        self._point_src_sop_uid = self._ct_series[2].SOPInstanceUID
        self._point_src_sop_class_uid = self._ct_series[2].SOPClassUID
        self._point_src = SourceImageForRegion(
            referenced_sop_class_uid=self._point_src_sop_class_uid,
            referenced_sop_instance_uid=self._point_src_sop_uid,
        )
        self._point = np.array([[1.0, 2.0]])
        self._img_reg_point = ImageRegion(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=self._point,
            source_image=self._point_src
        )
        self._point_uid = UID()
        self._point_id = 'point'
        point_tracker = TrackingIdentifier(
            uid=self._point_uid,
            identifier=self._point_id
        )
        self._point_grp = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=point_tracker,
            referenced_regions=[self._img_reg_point],
        )

        # Measurement group with a volume surface of type point
        self._point3d_src_image = SourceImageForSegmentation.from_source_image(
            self._ct_series[3]
        )
        self._point3d = np.array([[1.0, 2.0, 3.0]])
        self._vol_surf_point = VolumeSurface(
            graphic_type=GraphicTypeValues3D.POINT,
            graphic_data=self._point3d,
            source_images=[self._point3d_src_image],
            frame_of_reference_uid=self._ct_series[3].FrameOfReferenceUID
        )
        self._point3d_uid = UID()
        self._point3d_id = 'point3d'
        point3d_tracker = TrackingIdentifier(
            uid=self._point3d_uid,
            identifier=self._point3d_id
        )
        self._point3d_grp = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=point3d_tracker,
            referenced_volume_surface=self._vol_surf_point,
        )

        # Measurement group with a volume surface of type polygon
        self._polygon3d_src_img = SourceImageForSegmentation.from_source_image(
            self._ct_series[3]
        )
        self._polygon3d = [
            np.array([
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
                [3.0, 3.0, 1.0],
                [1.0, 1.0, 1.0]
            ]),
            np.array([
                [1.0, 1.0, 2.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 2.0],
                [1.0, 1.0, 2.0]
            ])
        ]
        self._vol_surf_polygon = VolumeSurface(
            graphic_type=GraphicTypeValues3D.POLYGON,
            graphic_data=self._polygon3d,
            source_images=[self._polygon3d_src_img],
            frame_of_reference_uid=self._ct_series[3].FrameOfReferenceUID
        )
        self._polygon3d_uid = UID()
        self._polygon3d_id = 'polygon3d'
        polygon3d_tracker = TrackingIdentifier(
            uid=self._polygon3d_uid,
            identifier=self._polygon3d_id
        )
        self._polygon3d_grp = VolumetricROIMeasurementsAndQualitativeEvaluations(  # noqa: E501
            tracking_identifier=polygon3d_tracker,
            referenced_volume_surface=self._vol_surf_polygon,
        )

        # Measurement group with segmentation frame
        self._seg_src_image = SourceImageForSegmentation(
            referenced_sop_class_uid=self._ct_series[0].SOPClassUID,
            referenced_sop_instance_uid=self._ct_series[0].SOPInstanceUID,
        )
        self._ref_segment = ReferencedSegment(
            sop_class_uid=self._ref_seg.SOPClassUID,
            sop_instance_uid=self._ref_seg.SOPInstanceUID,
            segment_number=1,
            source_images=[self._seg_src_image]
        )
        self._seg_uid = UID()
        self._seg_id = 'seg'
        seg_tracker = TrackingIdentifier(
            uid=self._seg_uid,
            identifier=self._seg_id
        )
        self._seg_grp = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=seg_tracker,
            referenced_segment=self._ref_segment,
        )

        # Save the sr and re-read it
        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Bar^Foo'
            )
        )
        observer_device_context = ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=UID()
            )
        )
        self._observation_context = ObservationContext(
            observer_person_context=observer_person_context,
            observer_device_context=observer_device_context,
        )
        self._all_grps = [
            self._polyline_grp,
            self._circle_grp,
            self._point_grp,
            self._point3d_grp,
            self._polygon3d_grp,
            self._seg_grp
        ]
        report = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
            imaging_measurements=self._all_grps,
        )
        sr = Comprehensive3DSR(
            evidence=self._ct_series + [self._ref_seg],
            content=report[0],
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
            record_evidence=False
        )

        # Write out to a buffer and read back in to revert to a standard
        # pydicom dataset and test the conversion functionality sets up
        # everything correctly for parsing.
        with BytesIO() as buf:
            sr.save_as(buf)
            buf.seek(0)
            sr_from_file = dcmread(buf)
        self._sr = Comprehensive3DSR.from_dataset(sr_from_file)
        self._content = self._sr.content

    def test_all_groups(self):
        grps = self._content.get_volumetric_roi_measurement_groups()
        assert len(grps) == len(self._all_grps)

    def test_get_image_region_groups(self):
        # Find all groups with reference type ImageRegion
        # (includes the polyline, circle and point groups)
        grps = self._content.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion
        )
        assert len(grps) == 3

    def test_get_polyline_groups(self):
        # Find the polyline group with and without explicitly
        # specifying the reference_type as ImageRegion
        grps = self._content.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion,
            graphic_type=GraphicTypeValues.POLYLINE
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._polyline_id
        assert grps[0].tracking_uid == self._polyline_uid
        grps = self._content.get_volumetric_roi_measurement_groups(
            graphic_type=GraphicTypeValues.POLYLINE
        )
        assert grps[0].tracking_identifier == self._polyline_id
        assert grps[0].tracking_uid == self._polyline_uid
        assert len(grps) == 1

        # Check the graphic data matches
        rois = grps[0].roi
        assert len(rois) == 1  # a single ImageRegion
        assert isinstance(rois[0], ImageRegion)
        assert rois[0].graphic_type == GraphicTypeValues.POLYLINE
        assert np.array_equal(rois[0].value, self._polyline)

    def test_get_circle_groups(self):
        # Find the circle group with and without explicitly
        # specifying the reference_type as ImageRegion
        grps = self._content.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion,
            graphic_type=GraphicTypeValues.CIRCLE
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._circle_id
        assert grps[0].tracking_uid == self._circle_uid
        grps = self._content.get_volumetric_roi_measurement_groups(
            graphic_type=GraphicTypeValues.CIRCLE
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._circle_id
        assert grps[0].tracking_uid == self._circle_uid

        # Check the graphic data matches
        rois = grps[0].roi
        assert len(rois) == 1  # a single ImageRegion
        assert isinstance(rois[0], ImageRegion)
        assert rois[0].graphic_type == GraphicTypeValues.CIRCLE
        assert np.array_equal(rois[0].value, self._circle)

    def test_get_point_groups(self):
        # Find the point group
        grps = self._content.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.ImageRegion,
            graphic_type=GraphicTypeValues.POINT
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._point_id
        assert grps[0].tracking_uid == self._point_uid

        # Check the graphic data matches
        rois = grps[0].roi
        assert len(rois) == 1  # a single ImageRegion
        assert isinstance(rois[0], ImageRegion)
        assert rois[0].graphic_type == GraphicTypeValues.POINT
        assert np.array_equal(rois[0].value, self._point)

    def test_get_point3d_groups(self):
        # Find the point 3D group, with and without specifying the
        # reference type as volume surface
        grps = self._content.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.VolumeSurface,
            graphic_type=GraphicTypeValues3D.POINT
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._point3d_id
        assert grps[0].tracking_uid == self._point3d_uid
        grps = self._content.get_volumetric_roi_measurement_groups(
            graphic_type=GraphicTypeValues3D.POINT
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._point3d_id
        assert grps[0].tracking_uid == self._point3d_uid
        vol = grps[0].roi
        assert isinstance(vol, VolumeSurface)
        assert vol.graphic_type == GraphicTypeValues3D.POINT
        graphic_data = vol.graphic_data
        assert np.array_equal(graphic_data, self._point3d)

    def test_get_polygon3d_groups(self):
        # Find the polygon 3D group
        grps = self._content.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.VolumeSurface,
            graphic_type=GraphicTypeValues3D.POLYGON
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._polygon3d_id
        assert grps[0].tracking_uid == self._polygon3d_uid

        vol = grps[0].roi
        assert isinstance(vol, VolumeSurface)
        assert vol.graphic_type == GraphicTypeValues3D.POLYGON
        items = vol.graphic_data
        assert len(items) == len(self._polygon3d)
        for item, arr in zip(items, self._polygon3d):
            assert np.array_equal(item, arr)

    def test_find_seg_groups(self):
        # Find the seg group
        grps = self._content.get_volumetric_roi_measurement_groups(
            reference_type=codes.DCM.ReferencedSegment,
        )
        assert len(grps) == 1
        assert grps[0].tracking_identifier == self._seg_id
        assert grps[0].tracking_uid == self._seg_uid
        ref_seg = grps[0].referenced_segment
        assert isinstance(ref_seg, ReferencedSegment)
        ins_uid = ref_seg.referenced_sop_instance_uid
        class_uid = ref_seg.referenced_sop_class_uid
        assert ins_uid == self._ref_seg.SOPInstanceUID
        assert class_uid == self._ref_seg.SOPClassUID

    def test_get_groups_invalid_reference_types(self):
        with pytest.raises(ValueError):
            # ReferencedSegment is invalid for planar groups
            self._content.get_volumetric_roi_measurement_groups(
                reference_type=codes.DCM.ReferencedSegmentationFrame
            )

    def test_get_planar_groups(self):
        grps = self._content.get_planar_roi_measurement_groups()
        assert len(grps) == 0

    def test_get_groups_by_tracking_id(self):
        grps = self._content.get_volumetric_roi_measurement_groups(
            tracking_uid=self._polyline_uid
        )
        assert len(grps) == 1
        assert grps[0].tracking_uid == self._polyline_uid

    def test_get_groups_by_ref_uid_1(self):
        # Should match the polyline and seg groups
        grps = self._content.get_volumetric_roi_measurement_groups(
            referenced_sop_instance_uid=self._ct_series[0].SOPInstanceUID
        )
        assert len(grps) == 2
        found_tracking_uids = {g.tracking_uid for g in grps}
        assert found_tracking_uids == {self._polyline_uid, self._seg_uid}

    def test_get_groups_by_ref_uid_2(self):
        # Should match the seg group
        grps = self._content.get_volumetric_roi_measurement_groups(
            referenced_sop_class_uid=SegmentationStorage
        )
        assert len(grps) == 1
        assert grps[0].tracking_uid == self._seg_uid

    def test_get_groups_by_ref_uid_3(self):
        # Should match the seg group
        grps = self._content.get_volumetric_roi_measurement_groups(
            referenced_sop_instance_uid=self._ref_seg.SOPInstanceUID
        )
        assert len(grps) == 1
        assert grps[0].tracking_uid == self._seg_uid

    def test_get_groups_invalid_graphic_type_1(self):
        # Any graphic type is invalid when reference type is not ImageRegion
        # or VolumeSurface
        with pytest.raises(ValueError):
            self._content.get_volumetric_roi_measurement_groups(
                reference_type=codes.DCM.ReferencedSegment,
                graphic_type=GraphicTypeValues.CIRCLE
            )

    def test_get_groups_invalid_graphic_type_2(self):
        # Multipoint is always invalid
        with pytest.raises(ValueError):
            self._content.get_volumetric_roi_measurement_groups(
                graphic_type=GraphicTypeValues.MULTIPOINT
            )

    def test_get_groups_invalid_graphic_type_3(self):
        # Specifying a GraphicTypeValues3D with ImageRegion is not allowed
        # for volumetric measurement groups (unlike planar groups)
        with pytest.raises(TypeError):
            self._content.get_volumetric_roi_measurement_groups(
                reference_type=codes.DCM.ImageRegion,
                graphic_type=GraphicTypeValues3D.ELLIPSE
            )

    def test_get_groups_invalid_graphic_type_3d(self):
        # Multipoint is always invalid
        with pytest.raises(ValueError):
            self._content.get_volumetric_roi_measurement_groups(
                graphic_type=GraphicTypeValues3D.MULTIPOINT
            )

    def test_get_groups_with_ref_uid_and_graphic_type_3d(self):
        # Multipoint is always invalid
        with pytest.raises(TypeError):
            self._content.get_volumetric_roi_measurement_groups(
                graphic_type=GraphicTypeValues3D.POINT,
                referenced_sop_instance_uid=UID()
            )


class TestImageLibraryEntryDescriptors(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_construction(self):
        modality = codes.cid29.SlideMicroscopy
        frame_of_reference_uid = '1.2.3'
        pixel_data_rows = 10
        pixel_data_columns = 20
        content_date = datetime.now().date()
        content_time = datetime.now().time()
        content_date_item = DateContentItem(
            name=codes.DCM.ContentDate,
            value=content_date,
            relationship_type=RelationshipTypeValues.HAS_ACQ_CONTEXT
        )
        content_time_item = TimeContentItem(
            name=codes.DCM.ContentTime,
            value=content_time,
            relationship_type=RelationshipTypeValues.HAS_ACQ_CONTEXT
        )
        group = ImageLibraryEntryDescriptors(
            modality=modality,
            frame_of_reference_uid=frame_of_reference_uid,
            pixel_data_rows=pixel_data_rows,
            pixel_data_columns=pixel_data_columns,
            additional_descriptors=[content_date_item, content_time_item]
        )
        assert len(group) == 6
        assert isinstance(group[0], CodeContentItem)
        assert group[0].name == codes.DCM.Modality
        assert group[0].value == modality
        assert isinstance(group[1], UIDRefContentItem)
        assert group[1].name == codes.DCM.FrameOfReferenceUID
        assert group[1].value == frame_of_reference_uid
        assert isinstance(group[2], NumContentItem)
        assert group[2].name == codes.DCM.PixelDataRows
        assert group[2].value == pixel_data_rows
        assert isinstance(group[3], NumContentItem)
        assert group[3].name == codes.DCM.PixelDataColumns
        assert group[3].value == pixel_data_columns
        assert isinstance(group[4], DateContentItem)
        assert group[4].name == codes.DCM.ContentDate
        assert group[4].value == content_date
        assert isinstance(group[5], TimeContentItem)
        assert group[5].name == codes.DCM.ContentTime
        assert group[5].value == content_time


class TestImageLibrary(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_construction(self):
        modality = codes.cid29.SlideMicroscopy
        frame_of_reference_uid = '1.2.3'
        pixel_data_rows = 10
        pixel_data_columns = 20
        descriptor_items = ImageLibraryEntryDescriptors(
            modality=modality,
            frame_of_reference_uid=frame_of_reference_uid,
            pixel_data_rows=pixel_data_rows,
            pixel_data_columns=pixel_data_columns,
        )
        library_items = ImageLibrary(groups=[descriptor_items])
        assert len(library_items) == 1
        library_group_item = library_items[0].ContentSequence[0]
        assert len(library_group_item.ContentSequence) == len(descriptor_items)
        assert library_group_item.name == codes.DCM.ImageLibraryGroup
