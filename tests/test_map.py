import unittest
from pathlib import Path

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.data import get_testdata_files
from pydicom.sr.codedict import codes

import highdicom as hd


class TestRealWorldValueMapping(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_failed_construction_missing_or_unnecessary_parameters(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = [0, 255]
        lut_data = [v**2 for v in range(256)]
        intercept = 0
        slope = 1
        with pytest.raises(TypeError):
            hd.map.RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
            )
        with pytest.raises(TypeError):
            hd.map.RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                slope=slope
            )
        with pytest.raises(TypeError):
            hd.map.RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                slope=slope,
                intercept=intercept,
                lut_data=lut_data
            )
        with pytest.raises(TypeError):
            hd.map.RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                slope=slope,
                lut_data=lut_data
            )
        with pytest.raises(TypeError):
            hd.map.RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                intercept=intercept,
                lut_data=lut_data
            )

    def test_construction_integer_linear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = [0, 255]
        intercept = 0
        slope = 1
        m = hd.map.RealWorldValueMapping(
            lut_label=lut_label,
            lut_explanation=lut_explanation,
            unit=unit,
            value_range=value_range,
            intercept=intercept,
            slope=slope
        )
        assert m.LUTLabel == lut_label
        assert m.LUTExplanation == lut_explanation
        assert isinstance(m.RealWorldValueSlope, float)
        assert m.RealWorldValueSlope == float(slope)
        assert isinstance(m.RealWorldValueIntercept, float)
        assert m.RealWorldValueIntercept == float(intercept)
        assert m.MeasurementUnitsCodeSequence[0] == unit
        assert isinstance(m.RealWorldValueFirstValueMapped, int)
        assert m.RealWorldValueFirstValueMapped == value_range[0]
        assert isinstance(m.RealWorldValueLastValueMapped, int)
        assert m.RealWorldValueLastValueMapped == value_range[1]
        with pytest.raises(AttributeError):
            m.DoubleFloatRealWorldValueFirstValueMapped
        with pytest.raises(AttributeError):
            m.DoubleFloatRealWorldValueLastValueMapped
        with pytest.raises(AttributeError):
            m.RealWorldValueLUTData

    def test_construction_integer_nonlinear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = [0, 255]
        lut_data = [v**2 for v in range(256)]
        m = hd.map.RealWorldValueMapping(
            lut_label=lut_label,
            lut_explanation=lut_explanation,
            unit=unit,
            value_range=value_range,
            lut_data=lut_data
        )
        assert m.LUTLabel == lut_label
        assert m.LUTExplanation == lut_explanation
        assert len(m.RealWorldValueLUTData) == len(lut_data)
        assert isinstance(m.RealWorldValueLUTData[0], float)
        assert m.MeasurementUnitsCodeSequence[0] == unit
        assert isinstance(m.RealWorldValueFirstValueMapped, int)
        assert m.RealWorldValueFirstValueMapped == value_range[0]
        assert isinstance(m.RealWorldValueLastValueMapped, int)
        assert m.RealWorldValueLastValueMapped == value_range[1]
        with pytest.raises(AttributeError):
            m.DoubleFloatRealWorldValueFirstValueMapped
        with pytest.raises(AttributeError):
            m.DoubleFloatRealWorldValueLastValueMapped
        with pytest.raises(AttributeError):
            m.RealWorldValueSlope
        with pytest.raises(AttributeError):
            m.RealWorldValueIntercept

    def test_construction_floating_point_linear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = [0.0, 1.0]
        intercept = 0
        slope = 1
        m = hd.map.RealWorldValueMapping(
            lut_label=lut_label,
            lut_explanation=lut_explanation,
            unit=unit,
            value_range=value_range,
            intercept=intercept,
            slope=slope
        )
        assert m.LUTLabel == lut_label
        assert m.LUTExplanation == lut_explanation
        assert isinstance(m.RealWorldValueSlope, float)
        assert m.RealWorldValueSlope == float(slope)
        assert isinstance(m.RealWorldValueIntercept, float)
        assert m.RealWorldValueIntercept == float(intercept)
        assert m.MeasurementUnitsCodeSequence[0] == unit
        assert isinstance(m.DoubleFloatRealWorldValueFirstValueMapped, float)
        assert m.DoubleFloatRealWorldValueFirstValueMapped == value_range[0]
        assert isinstance(m.DoubleFloatRealWorldValueLastValueMapped, float)
        assert m.DoubleFloatRealWorldValueLastValueMapped == value_range[1]
        with pytest.raises(AttributeError):
            m.RealWorldValueFirstValueMapped
        with pytest.raises(AttributeError):
            m.RealWorldValueLastValueMapped
        with pytest.raises(AttributeError):
            m.RealWorldValueLUTData

    def test_failed_construction_floating_point_nonlinear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = [0.0, 1.0]
        lut_data = [
            v**2
            for v in np.arange(value_range[0], value_range[1], 0.1)
        ]
        with pytest.raises(ValueError):
            hd.map.RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                lut_data=lut_data
            )


class TestParametricMap(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')

        self._series_instance_uid = hd.UID()
        self._series_number = 1
        self._sop_instance_uid = hd.UID()
        self._instance_number = 1
        self._manufacturer = 'MyManufacturer'
        self._manufacturer_model_name = 'MyModel'
        self._software_versions = 'v1.0'
        self._device_serial_number = '1-2-3'
        self._content_description = 'Test Parametric Map'
        self._content_creator_name = 'Will^I^Am'

        self._ct_image = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )

        self._sm_image = dcmread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )

        ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ct_series = sorted(
            ct_series,
            key=lambda x: x.ImagePositionPatient[2]
        )

    def test_multi_frame_sm_image_single_native(self):
        pixel_array = np.random.random(self._sm_image.pixel_array.shape[:3])
        pixel_array = pixel_array.astype(np.float32)
        window_center = 0.5
        window_width = 1.0
        real_world_value_mapping = hd.map.RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0.0, 1.0],
            intercept=0,
            slope=1
        )
        pmap = hd.map.ParametricMap(
            [self._sm_image],
            pixel_array,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[real_world_value_mapping],
            window_center=window_center,
            window_width=window_width
        )
        assert pmap.SOPClassUID == '1.2.840.10008.5.1.4.1.1.30'
        assert pmap.SOPInstanceUID == self._sop_instance_uid
        assert pmap.SeriesInstanceUID == self._series_instance_uid
        assert pmap.SeriesNumber == self._series_number
        assert pmap.Manufacturer == self._manufacturer
        assert pmap.ManufacturerModelName == self._manufacturer_model_name
        assert pmap.SoftwareVersions == self._software_versions
        assert pmap.DeviceSerialNumber == self._device_serial_number
        assert pmap.StudyInstanceUID == self._sm_image.StudyInstanceUID
        assert pmap.PatientID == self._sm_image.PatientID
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_multi_frame_sm_image_ushort_native(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256
        real_world_value_mapping = hd.map.RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        pmap = hd.map.ParametricMap(
            [self._sm_image],
            pixel_array,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[real_world_value_mapping],
            window_center=window_center,
            window_width=window_width
        )
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_multi_frame_sm_image_ushort_encapsulated(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256

        real_world_value_mapping = hd.map.RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        pmap = hd.map.ParametricMap(
            [self._sm_image],
            pixel_array,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[real_world_value_mapping],
            window_center=window_center,
            window_width=window_width,
            transfer_syntax_uid='1.2.840.10008.1.2.4.90'
        )
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_single_frame_ct_image_double(self):
        pixel_array = np.random.uniform(-1, 1, self._ct_image.pixel_array.shape)
        window_center = 0.0
        window_width = 2.0
        real_world_value_mapping = hd.map.RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-1, 1],
            intercept=0,
            slope=1
        )
        pmap = hd.map.ParametricMap(
            [self._ct_image],
            pixel_array,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[real_world_value_mapping],
            window_center=window_center,
            window_width=window_width,
        )
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_single_frame_ct_image_ushort_native(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**12,
            size=self._ct_image.pixel_array.shape,
            dtype=np.uint16
        )
        window_center = 2**12 / 2.0
        window_width = 2**12
        real_world_value_mapping = hd.map.RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 4095],
            intercept=0,
            slope=1
        )
        pmap = hd.map.ParametricMap(
            [self._ct_image],
            pixel_array,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[real_world_value_mapping],
            window_center=window_center,
            window_width=window_width,
        )
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_single_frame_ct_image_short(self):
        pixel_array = np.random.randint(
            low=-3200,
            high=24000,
            size=self._ct_image.pixel_array.shape,
            dtype=np.int16
        )
        window_center = 0
        window_width = 2**16
        real_world_value_mapping = hd.map.RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-32768, 32767],
            intercept=0,
            slope=1
        )
        pmap = hd.map.ParametricMap(
            [self._ct_image],
            pixel_array,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[real_world_value_mapping],
            window_center=window_center,
            window_width=window_width,
        )

        retrieved_pixel_array = pmap.pixel_array
        shared_fg = pmap.SharedFunctionalGroupsSequence[0]
        transformation = shared_fg.PixelValueTransformationSequence[0]
        slope = transformation.RescaleSlope
        intercept = transformation.RescaleIntercept
        rescaled_pixel_array = (
            retrieved_pixel_array.astype(float) *
            float(slope) +
            float(intercept)
        ).astype(np.int16)
        assert np.array_equal(rescaled_pixel_array, pixel_array)

    def test_series_single_frame_ct_image_single(self):
        size = (len(self._ct_series), ) + self._ct_series[0].pixel_array.shape
        pixel_array = np.random.uniform(-1, 1, size)
        pixel_array = pixel_array.astype(np.float32)
        window_center = 0.0
        window_width = 2.0
        real_world_value_mapping = hd.map.RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-1, 1],
            intercept=0,
            slope=1
        )
        pmap = hd.map.ParametricMap(
            self._ct_series,
            pixel_array,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[real_world_value_mapping],
            window_center=window_center,
            window_width=window_width,
        )

        assert np.array_equal(pmap.pixel_array, pixel_array)
