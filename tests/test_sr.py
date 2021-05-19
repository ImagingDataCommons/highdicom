import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid, UID
from pydicom.valuerep import DA, DS, DT, TM

from highdicom.sr import CodedConcept
from highdicom.sr import (
    FindingSite,
    ImageRegion,
    ImageRegion3D,
    LongitudinalTemporalOffsetFromEvent,
    RealWorldValueMap,
    ReferencedSegment,
    ReferencedSegmentationFrame,
    SourceImageForRegion,
    SourceImageForMeasurement,
    SourceImageForSegmentation,
    SourceSeriesForSegmentation,
    VolumeSurface,
)
from highdicom.sr import (
    GraphicTypeValues,
    GraphicTypeValues3D,
    RelationshipTypeValues,
    ValueTypeValues,
)
from highdicom.sr.utils import find_content_items
from highdicom.sr import (
    CodeContentItem,
    ContainerContentItem,
    CompositeContentItem,
    DateContentItem,
    DateTimeContentItem,
    ImageContentItem,
    NumContentItem,
    ScoordContentItem,
    Scoord3DContentItem,
    TextContentItem,
    TimeContentItem,
    UIDRefContentItem,
)
from highdicom.sr import (
    ComprehensiveSR,
    Comprehensive3DSR,
    EnhancedSR,
)
from highdicom.sr import (
    AlgorithmIdentification,
    DeviceObserverIdentifyingAttributes,
    Measurement,
    MeasurementStatisticalProperties,
    MeasurementProperties,
    MeasurementReport,
    ObservationContext,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
    PlanarROIMeasurementsAndQualitativeEvaluations,
    SubjectContext,
    SubjectContextSpecimen,
    SubjectContextDevice,
    TimePointContext,
    TrackingIdentifier,
    VolumetricROIMeasurementsAndQualitativeEvaluations,
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
                unit=self._value_unit
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
        i = CodeContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'CODE'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.ConceptCodeSequence[0] == value
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_text_item_construction(self):
        name = codes.DCM.TrackingIdentifier
        value = '1234'
        i = TextContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'TEXT'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.TextValue == value
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_time_item_construction_from_string(self):
        name = codes.DCM.StudyTime
        value = '1530'
        i = TimeContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'TIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Time == TM(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_time_item_construction_from_string_malformatted(self):
        name = codes.DCM.StudyTime
        value = 'abc'
        with pytest.raises(ValueError):
            TimeContentItem(
                name=name,
                value=value
            )

    def test_time_item_construction_from_time(self):
        name = codes.DCM.StudyTime
        value = datetime.now().time()
        i = TimeContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'TIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Time == TM(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_date_item_construction_from_string(self):
        name = codes.DCM.StudyDate
        value = '20190821'
        i = DateContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'DATE'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Date == DA(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_date_item_construction_from_string_malformatted(self):
        name = codes.DCM.StudyDate
        value = 'abcd'
        with pytest.raises(ValueError):
            DateContentItem(
                name=name,
                value=value
            )

    def test_date_item_construction_from_time(self):
        name = codes.DCM.StudyTime
        value = datetime.now().date()
        i = DateContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'DATE'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.Date == DA(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_datetime_item_construction_from_string(self):
        name = codes.DCM.ImagingStartDatetime
        value = '20190821-15:30'
        i = DateTimeContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'DATETIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.DateTime == DT(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_datetime_item_construction_from_string_malformatted(self):
        name = codes.DCM.ImagingStartDatetime
        value = 'abcd'
        with pytest.raises(ValueError):
            DateTimeContentItem(
                name=name,
                value=value
            )

    def test_datetime_item_construction_from_datetime(self):
        name = codes.DCM.ImagingStartDatetime
        value = datetime.now()
        i = DateTimeContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'DATETIME'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.DateTime == DT(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_uidref_item_construction_from_string(self):
        name = codes.DCM.SeriesInstanceUID
        value = '1.2.3.4.5.6'
        i = UIDRefContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'UIDREF'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.UID == UID(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_uidref_item_construction_wrong_value_type(self):
        name = codes.DCM.SeriesInstanceUID
        value = 123456
        with pytest.raises(TypeError):
            UIDRefContentItem(
                name=name,
                value=value
            )

    def test_uidref_item_construction_from_uid(self):
        name = codes.DCM.SeriesInstanceUID
        value = UID('1.2.3.4.5.6')
        i = UIDRefContentItem(
            name=name,
            value=value
        )
        assert i.ValueType == 'UIDREF'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.UID == UID(value)
        with pytest.raises(AttributeError):
            assert i.RelationshipType

    def test_num_item_construction_from_integer(self):
        name = codes.SCT.Area
        value = 100
        unit = Code('um2', 'UCUM', 'Square Micrometer')
        i = NumContentItem(
            name=name,
            value=value,
            unit=unit
        )
        assert i.ValueType == 'NUM'
        assert i.ConceptNameCodeSequence[0] == name
        value_item = i.MeasuredValueSequence[0]
        unit_code_item = value_item.MeasurementUnitsCodeSequence[0]
        assert value_item.NumericValue == value
        with pytest.raises(AttributeError):
            assert value_item.FloatingPointValue
        assert unit_code_item.CodeValue == unit.value
        assert unit_code_item.CodingSchemeDesignator == unit.scheme_designator
        with pytest.raises(AttributeError):
            assert i.RelationshipType
        with pytest.raises(AttributeError):
            assert i.NumericValueQualifierCodeSequence

    def test_num_item_construction_from_float(self):
        name = codes.SCT.Area
        value = 100.0
        unit = Code('um2', 'UCUM', 'Square Micrometer')
        i = NumContentItem(
            name=name,
            value=value,
            unit=unit
        )
        assert i.ValueType == 'NUM'
        assert i.ConceptNameCodeSequence[0] == name
        value_item = i.MeasuredValueSequence[0]
        unit_code_item = value_item.MeasurementUnitsCodeSequence[0]
        assert value_item.NumericValue == value
        assert value_item.FloatingPointValue == value
        assert unit_code_item.CodeValue == unit.value
        assert unit_code_item.CodingSchemeDesignator == unit.scheme_designator
        with pytest.raises(AttributeError):
            assert i.RelationshipType
        with pytest.raises(AttributeError):
            assert i.NumericValueQualifierCodeSequence

    def test_num_item_construction_from_qualifier_code(self):
        name = codes.SCT.Area
        qualifier = Code('114000', 'SCT', 'Not a number')
        i = NumContentItem(
            name=name,
            qualifier=qualifier
        )
        assert i.ValueType == 'NUM'
        assert i.ConceptNameCodeSequence[0] == name
        with pytest.raises(AttributeError):
            assert i.MeasuredValueSequence
        with pytest.raises(AttributeError):
            assert i.RelationshipType
        qualifier_code_item = i.NumericValueQualifierCodeSequence[0]
        assert qualifier_code_item.CodeValue == qualifier.value

    def test_container_item_construction(self):
        name = codes.DCM.ImagingMeasurementReport
        tid = '1500'
        i = ContainerContentItem(
            name=name,
            template_id=tid
        )
        assert i.ValueType == 'CONTAINER'
        assert i.ConceptNameCodeSequence[0] == name
        template_item = i.ContentTemplateSequence[0]
        assert template_item.TemplateIdentifier == tid
        assert template_item.MappingResource == 'DCMR'
        assert i.ContinuityOfContent == 'CONTINUOUS'

    def test_composite_item_construction(self):
        name = codes.DCM.RealWorldValueMapUsedForMeasurement
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.2'
        sop_instance_uid = '1.2.3.4'
        i = CompositeContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
        )
        assert i.ValueType == 'COMPOSITE'
        assert i.ConceptNameCodeSequence[0] == name
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid

    def test_image_item_construction(self):
        name = codes.DCM.SourceImageForSegmentation
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.2'
        sop_instance_uid = '1.2.3.4'
        i = ImageContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
        )
        assert i.ValueType == 'IMAGE'
        assert i.ConceptNameCodeSequence[0] == name
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        with pytest.raises(AttributeError):
            ref_sop_item.ReferencedFrameNumber
        with pytest.raises(AttributeError):
            ref_sop_item.ReferencedSegmentNumber

    def test_image_item_construction_with_multiple_frame_numbers(self):
        name = codes.DCM.SourceImageForSegmentation
        sop_class_uid = '1.2.840.10008.5.1.4.1.1.2.2'
        sop_instance_uid = '1.2.3.4'
        frame_numbers = [1, 2, 3]
        i = ImageContentItem(
            name=name,
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_frame_numbers=frame_numbers
        )
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert ref_sop_item.ReferencedFrameNumber == frame_numbers
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
            referenced_frame_numbers=frame_number
        )
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert ref_sop_item.ReferencedFrameNumber == frame_number
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
            referenced_segment_numbers=segment_number
        )
        ref_sop_item = i.ReferencedSOPSequence[0]
        assert ref_sop_item.ReferencedSOPClassUID == sop_class_uid
        assert ref_sop_item.ReferencedSOPInstanceUID == sop_instance_uid
        assert ref_sop_item.ReferencedSegmentNumber == segment_number
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
            pixel_origin_interpretation=pixel_origin_interpretation
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
            pixel_origin_interpretation=pixel_origin_interpretation
        )
        assert i.ValueType == 'SCOORD'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.GraphicType == graphic_type.value
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
            frame_of_reference_uid=frame_of_reference_uid
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
            frame_of_reference_uid=frame_of_reference_uid
        )
        assert i.ValueType == 'SCOORD3D'
        assert i.ConceptNameCodeSequence[0] == name
        assert i.GraphicType == graphic_type.value
        assert np.all(i.GraphicData[:3] == graphic_data[0, :])
        assert np.all(i.GraphicData[3:6] == graphic_data[1, :])
        assert np.all(i.GraphicData[6:9] == graphic_data[2, :])
        assert i.ReferencedFrameOfReferenceUID == frame_of_reference_uid
        with pytest.raises(AttributeError):
            i.FiducialUID


class TestContentSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()


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
        self._person_name = 'Foo Bar'
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
        assert item.TextValue == self._person_name
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
            [self._ref_frame_number]
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
            [self._ref_frame_number]
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
        self._ref_frame_number = 38
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
            frame_number=self._ref_frame_number,
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

    def test_from_segmentation(self):
        ref_seg = ReferencedSegmentationFrame.from_segmentation(
            self._seg_dataset,
            frame_number=self._ref_frame_number,
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
            frame_number=self._ref_frame_number,
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
                frame_number=self._ref_frame_number,
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
                frame_number=self._ref_frame_number,
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
        self._value = 10.0
        self._unit = codes.cid7181.SquareMillimeter
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='prostate zone size measurement'
        )
        self._name = codes.cid7469.Area
        self._measurement = Measurement(
            name=self._name,
            value=self._value,
            unit=self._unit,
            tracking_identifier=self._tracking_identifier
        )

    def test_name(self):
        item = self._measurement[0]
        assert item.ConceptNameCodeSequence[0] == self._name

    def test_value(self):
        item = self._measurement[0]
        assert len(item.MeasuredValueSequence) == 1
        assert len(item.MeasuredValueSequence[0]) == 3
        assert item.MeasuredValueSequence[0].NumericValue == DS(self._value)
        value_item = item.MeasuredValueSequence[0]
        unit_item = value_item.MeasurementUnitsCodeSequence[0]
        assert unit_item == self._unit
        with pytest.raises(AttributeError):
            item.NumericValueQualifierCodeSequence

    def test_tracking_identifier(self):
        item = self._measurement[0].ContentSequence[0]
        assert item.ConceptNameCodeSequence[0].CodeValue == '112039'

    def test_tracking_unique_identifier(self):
        item = self._measurement[0].ContentSequence[1]
        assert item.ConceptNameCodeSequence[0].CodeValue == '112040'


class TestMeasurementOptional(unittest.TestCase):

    def setUp(self):
        '''Creates a Measurement for a numeric value in millimiter unit with
        derivation, method and reference to an image region.'''
        super().setUp()
        self._value = 10
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
        self._name = codes.cid7469.Area
        self._measurement = Measurement(
            name=self._name,
            value=self._value,
            unit=self._unit,
            tracking_identifier=self._tracking_identifier,
            method=self._method,
            derivation=self._derivation,
            finding_sites=[self._finding_site, ]
        )

    def test_method(self):
        item = self._measurement[0].ContentSequence[2]
        assert item.ConceptNameCodeSequence[0].CodeValue == '370129005'
        assert item.ConceptCodeSequence[0] == self._method

    def test_derivation(self):
        item = self._measurement[0].ContentSequence[3]
        assert item.ConceptNameCodeSequence[0].CodeValue == '121401'
        assert item.ConceptCodeSequence[0] == self._derivation

    def test_finding_site(self):
        item = self._measurement[0].ContentSequence[4]
        assert item.ConceptNameCodeSequence[0].CodeValue == '363698007'
        assert item.ConceptCodeSequence[0] == self._location
        # Laterality and topological modifier were not specified
        assert not hasattr(item, 'ContentSequence')


class TestImageRegion(unittest.TestCase):

    def setUp(self):
        pass


class TestVolumeSurface(unittest.TestCase):

    def setUp(self):
        pass


class TestPlanarROIMeasurementsAndQualitativeEvaluations(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='planar roi measurements'
        )
        self._image_for_region = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=generate_uid()
        )
        self._image_for_segment = SourceImageForSegmentation(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=generate_uid()
        )
        self._region = ImageRegion(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=np.array([[1.0, 1.0], [2.0, 2.0]]),
            source_image=self._image_for_region
        )
        self._segment = ReferencedSegmentationFrame(
            sop_class_uid='1.2.840.10008.5.1.4.1.1.66.4',
            sop_instance_uid=generate_uid(),
            segment_number=1,
            frame_number=1,
            source_image=self._image_for_segment
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
            CodeContentItem(
                CodedConcept(
                    value="RID49502",
                    meaning="clinically significant prostate cancer",
                    scheme_designator="RADLEX"
                ),
                codes.SCT.Yes, RelationshipTypeValues.CONTAINS
            )
        ]

    def test_construction_with_region(self):
        PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_region=self._region
        )

    def test_construction_with_segment(self):
        PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_segment=self._segment
        )

    def test_construction_all_parameters(self):
        # TODO add time_point_context and measurements
        PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_region=self._region,
            referenced_real_world_value_map=self._real_world_value_map,
            finding_type=self._finding_type,
            method=self._method,
            algorithm_id=self._algo_id,
            finding_sites=self._finding_sites,
            session=self._session,
            qualitative_evaluations=self._qualitative_evaluations,
            geometric_purpose=self._geometric_purpose
        )

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
        self._images_for_segment = [
            SourceImageForSegmentation(
                referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
                referenced_sop_instance_uid=generate_uid()
            )
        ]
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
        self._segment = ReferencedSegment(
            sop_class_uid='1.2.840.10008.5.1.4.1.1.66.4',
            sop_instance_uid=generate_uid(),
            segment_number=1,
            source_images=self._images_for_segment
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
            CodeContentItem(
                CodedConcept(
                    value="RID49502",
                    meaning="clinically significant prostate cancer",
                    scheme_designator="RADLEX"
                ),
                codes.SCT.Yes, RelationshipTypeValues.CONTAINS
            )
        ]

    def test_constructed_with_regions(self):
        self._measurements = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_regions=self._regions
        )

    def test_constructed_with_segment(self):
        self._measurements = VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_segment=self._segment
        )

    def test_construction_all_parameters(self):
        # TODO add time_point_context and measurements
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
        assert len(measurements[0].ContentSequence) == 3

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
        self._observer_person_context = ObserverContext(
            observer_type=codes.cid270.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Foo Bar'
            )
        )
        self._observer_device_context = ObserverContext(
            observer_type=codes.cid270.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=generate_uid()
            )
        )
        self._observation_context = ObservationContext(
            observer_person_context=self._observer_person_context,
            observer_device_context=self._observer_device_context
        )
        self._procedure_reported = codes.cid100.CTPerfusionHeadWithContrastIV
        self._tracking_identifier = TrackingIdentifier(
            uid=generate_uid(),
            identifier='planar roi measurements'
        )
        self._image = SourceImageForRegion(
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2.2',
            referenced_sop_instance_uid=generate_uid()
        )
        self._region = ImageRegion(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=np.array([[1.0, 1.0], [2.0, 2.0]]),
            source_image=self._image
        )
        self._measurements = PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=self._tracking_identifier,
            referenced_region=self._region
        )

    def test_construction(self):
        measurement_report = MeasurementReport(
            observation_context=self._observation_context,
            procedure_reported=self._procedure_reported,
            imaging_measurements=[self._measurements]
        )
        item = measurement_report[0]
        assert len(item.ContentSequence) == 8

        template_item = item.ContentTemplateSequence[0]
        assert template_item.TemplateIdentifier == '1500'

        content_item_expectations = [
            (0, '121049'),
            (1, '121005'),
            (2, '121008'),
            (3, '121005'),
            (4, '121012'),
            # Procedure reported
            (5, '121058'),
            # Image library
            (6, '111028'),
            # Imaging measurements
            (7, '126010'),
        ]
        for index, value in content_item_expectations:
            content_item = item.ContentSequence[index]
            assert content_item.ConceptNameCodeSequence[0].CodeValue == value


class TestEnhancedSR(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._ref_dataset = Dataset()
        self._ref_dataset.PatientID = '1'
        self._ref_dataset.PatientName = 'patient'
        self._ref_dataset.PatientBirthDate = '2000101'
        self._ref_dataset.PatientSex = 'o'
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

        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Foo'
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
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
                value=1.7,
                unit=codes.UCUM.SquareMillimeter,
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
            manufacturer=self._manufacturer
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

        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Foo'
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
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
                value=1.7,
                unit=codes.UCUM.SquareMillimeter,
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
            manufacturer=self._manufacturer
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
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
                value=0.7,
                unit=codes.UCUM.SquareMillimeter,
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


class TestComprehensive3DSR(unittest.TestCase):

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

        observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Foo'
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
                tracking_identifier=TrackingIdentifier(uid=generate_uid()),
                value=1.7,
                unit=codes.UCUM.SquareMillimeter,
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
            manufacturer=self._manufacturer
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
