import unittest
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.data import get_testdata_files
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.uid import (
    JPEG2000Lossless,
    JPEGLSLossless,
)

from highdicom import (
    PaletteColorLUT,
    PaletteColorLUTTransformation,
    PlanePositionSequence,
    PixelMeasuresSequence,
    PlaneOrientationSequence,
)
from highdicom.enum import ContentQualificationValues, CoordinateSystemNames
from highdicom.pm.content import RealWorldValueMapping
from highdicom.pm.enum import (
    DerivedPixelContrastValues,
    ImageFlavorValues,
)
from highdicom.pm.sop import ParametricMap
from highdicom.uid import UID


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
            RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
            )
        with pytest.raises(TypeError):
            RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                slope=slope
            )
        with pytest.raises(TypeError):
            RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                slope=slope,
                intercept=intercept,
                lut_data=lut_data
            )
        with pytest.raises(TypeError):
            RealWorldValueMapping(
                lut_label=lut_label,
                lut_explanation=lut_explanation,
                unit=unit,
                value_range=value_range,
                slope=slope,
                lut_data=lut_data
            )
        with pytest.raises(TypeError):
            RealWorldValueMapping(
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
        quantity_definition = Code('130402', 'DCM', 'Class activation')
        m = RealWorldValueMapping(
            lut_label=lut_label,
            lut_explanation=lut_explanation,
            unit=unit,
            value_range=value_range,
            intercept=intercept,
            slope=slope,
            quantity_definition=quantity_definition
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
        assert len(m.QuantityDefinitionSequence) == 1
        quantity_item = m.QuantityDefinitionSequence[0]
        assert quantity_item.name == codes.SCT.Quantity
        assert quantity_item.value == quantity_definition

    def test_construction_integer_nonlinear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = [0, 255]
        lut_data = [v**2 for v in range(256)]
        m = RealWorldValueMapping(
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
        m = RealWorldValueMapping(
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
            RealWorldValueMapping(
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

        self._series_instance_uid = UID()
        self._series_number = 1
        self._sop_instance_uid = UID()
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

    @staticmethod
    def check_dimension_index_vals(seg):
        # Function to apply some checks (necessary but not sufficient for
        # correctness) to ensure that the dimension indices are correct
        is_patient_coord_system = hasattr(
            seg.PerFrameFunctionalGroupsSequence[0],
            'PlanePositionSequence'
        )
        if is_patient_coord_system:
            # Build up the mapping from index to value
            index_mapping = defaultdict(list)
            for f in seg.PerFrameFunctionalGroupsSequence:
                posn_index = f.FrameContentSequence[0].DimensionIndexValues[1]
                # This is not general, but all the tests run here use axial
                # images so just check the z coordinate
                posn_val = f.PlanePositionSequence[0].ImagePositionPatient[2]
                index_mapping[posn_index].append(posn_val)

            # Check that each index value found references a unique value
            for values in index_mapping.values():
                assert [v == values[0] for v in values]

            # Check that the indices are monotonically increasing from 1
            expected_keys = range(1, len(index_mapping) + 1)
            assert set(index_mapping.keys()) == set(expected_keys)

            # Check that values are sorted
            old_v = float('-inf')
            for k in expected_keys:
                assert index_mapping[k][0] > old_v
                old_v = index_mapping[k][0]
        else:
            # Build up the mapping from index to value
            for dim_kw, dim_ind in zip([
                'ColumnPositionInTotalImagePixelMatrix',
                'RowPositionInTotalImagePixelMatrix'
            ], [1, 2]):
                index_mapping = defaultdict(list)
                for f in seg.PerFrameFunctionalGroupsSequence:
                    content_item = f.FrameContentSequence[0]
                    posn_index = content_item.DimensionIndexValues[dim_ind]
                    # This is not general, but all the tests run here use axial
                    # images so just check the z coordinate
                    posn_item = f.PlanePositionSlideSequence[0]
                    posn_val = getattr(posn_item, dim_kw)
                    index_mapping[posn_index].append(posn_val)

                # Check that each index value found references a unique value
                for values in index_mapping.values():
                    assert [v == values[0] for v in values]

                # Check that the indices are monotonically increasing from 1
                expected_keys = range(1, len(index_mapping) + 1)
                assert set(index_mapping.keys()) == set(expected_keys)

                # Check that values are sorted
                old_v = float('-inf')
                for k in expected_keys:
                    assert index_mapping[k][0] > old_v
                    old_v = index_mapping[k][0]

    def test_multi_frame_sm_image_single_native(self):
        pixel_array = np.random.random(self._sm_image.pixel_array.shape[:3])
        pixel_array = pixel_array.astype(np.float32)
        window_center = 0.5
        window_width = 1.0
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0.0, 1.0],
            intercept=0,
            slope=1
        )
        content_label = 'MY_MAP'
        pmap = ParametricMap(
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
            content_label=content_label
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
        assert pmap.ContentLabel == content_label
        assert pmap.TotalPixelMatrixRows == \
            self._sm_image.TotalPixelMatrixRows
        assert pmap.TotalPixelMatrixColumns == \
            self._sm_image.TotalPixelMatrixColumns
        assert pmap.TotalPixelMatrixOriginSequence == \
            self._sm_image.TotalPixelMatrixOriginSequence
        assert np.array_equal(pmap.pixel_array, pixel_array)
        assert isinstance(pmap.AcquisitionContextSequence, Sequence)
        assert pmap.ContentQualification == 'RESEARCH'
        assert pmap.ImageType[0] == 'DERIVED'
        assert pmap.ImageType[1] == 'PRIMARY'
        assert pmap.ImageType[2] == 'VOLUME'
        assert pmap.ImageType[3] == 'QUANTITY'
        sffg_item = pmap.SharedFunctionalGroupsSequence[0]
        voi_lut_item = sffg_item.FrameVOILUTSequence[0]
        assert voi_lut_item.WindowCenter == str(window_center)
        assert voi_lut_item.WindowWidth == str(window_width)

    def test_multi_frame_sm_image_ushort_native(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        instance = ParametricMap(
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
            content_qualification=ContentQualificationValues.SERVICE,
            image_flavor=ImageFlavorValues.WHOLE_BODY,
            derived_pixel_contrast=DerivedPixelContrastValues.NONE
        )
        sffg_item = instance.SharedFunctionalGroupsSequence[0]
        assert hasattr(sffg_item, 'RealWorldValueMappingSequence')
        assert len(sffg_item.RealWorldValueMappingSequence) == 1
        pffg_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert not hasattr(pffg_item, 'RealWorldValueMappingSequence')
        assert instance.BitsAllocated == 8
        assert instance.pixel_array.dtype == np.uint8
        assert np.array_equal(instance.pixel_array, pixel_array)
        assert instance.ContentQualification == 'SERVICE'
        assert instance.ImageType[0] == 'DERIVED'
        assert instance.ImageType[1] == 'PRIMARY'
        assert instance.ImageType[2] == 'WHOLE_BODY'
        assert instance.ImageType[3] == 'NONE'
        assert instance.PixelPresentation == 'MONOCHROME'

    def test_multi_frame_palette_lut(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        r_lut_data = np.arange(10, 120, dtype=np.uint16)
        g_lut_data = np.arange(20, 130, dtype=np.uint16)
        b_lut_data = np.arange(30, 140, dtype=np.uint16)
        r_first_mapped_value = 32
        g_first_mapped_value = 32
        b_first_mapped_value = 32
        lut_uid = UID()
        r_lut = PaletteColorLUT(r_first_mapped_value, r_lut_data, color='red')
        g_lut = PaletteColorLUT(g_first_mapped_value, g_lut_data, color='green')
        b_lut = PaletteColorLUT(b_first_mapped_value, b_lut_data, color='blue')
        transformation = PaletteColorLUTTransformation(
            red_lut=r_lut,
            green_lut=g_lut,
            blue_lut=b_lut,
            palette_color_lut_uid=lut_uid,
        )
        instance = ParametricMap(
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
            content_qualification=ContentQualificationValues.SERVICE,
            image_flavor=ImageFlavorValues.WHOLE_BODY,
            derived_pixel_contrast=DerivedPixelContrastValues.NONE,
            palette_color_lut_transformation=transformation,
        )
        assert instance.PixelPresentation == 'COLOR_RANGE'
        assert instance.PaletteColorLookupTableUID == lut_uid
        red_desc = [len(r_lut_data), r_first_mapped_value, 16]
        r_lut_data_retrieved = np.frombuffer(
            instance.RedPaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(r_lut_data, r_lut_data_retrieved)
        assert instance.RedPaletteColorLookupTableDescriptor == red_desc
        green_desc = [len(g_lut_data), g_first_mapped_value, 16]
        g_lut_data_retrieved = np.frombuffer(
            instance.GreenPaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(g_lut_data, g_lut_data_retrieved)
        assert instance.GreenPaletteColorLookupTableDescriptor == green_desc
        blue_desc = [len(b_lut_data), b_first_mapped_value, 16]
        b_lut_data_retrieved = np.frombuffer(
            instance.BluePaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(b_lut_data, b_lut_data_retrieved)
        assert instance.BluePaletteColorLookupTableDescriptor == blue_desc

    def test_multi_frame_sm_image_ushort_palette_lut(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        instance = ParametricMap(
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
            content_qualification=ContentQualificationValues.SERVICE,
            image_flavor=ImageFlavorValues.WHOLE_BODY,
            derived_pixel_contrast=DerivedPixelContrastValues.NONE
        )
        sffg_item = instance.SharedFunctionalGroupsSequence[0]
        assert hasattr(sffg_item, 'RealWorldValueMappingSequence')
        assert len(sffg_item.RealWorldValueMappingSequence) == 1
        pffg_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert not hasattr(pffg_item, 'RealWorldValueMappingSequence')
        assert instance.BitsAllocated == 8
        assert instance.pixel_array.dtype == np.uint8
        assert np.array_equal(instance.pixel_array, pixel_array)
        assert instance.ContentQualification == 'SERVICE'
        assert instance.ImageType[0] == 'DERIVED'
        assert instance.ImageType[1] == 'PRIMARY'
        assert instance.ImageType[2] == 'WHOLE_BODY'
        assert instance.ImageType[3] == 'NONE'
        assert instance.PixelPresentation == 'MONOCHROME'

    def test_multi_frame_sm_image_ushort_encapsulated_jpeg2000(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        pmap = ParametricMap(
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
            transfer_syntax_uid=JPEG2000Lossless
        )
        assert pmap.BitsAllocated == 8
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_multi_frame_sm_image_ushort_encapsulated_jpegls(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint16
        )
        window_center = 128
        window_width = 256

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        pmap = ParametricMap(
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
            transfer_syntax_uid=JPEGLSLossless
        )
        assert pmap.BitsAllocated == 16
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_single_frame_ct_image_double(self):
        pixel_array = np.random.uniform(-1, 1, self._ct_image.pixel_array.shape)
        window_center = 0.0
        window_width = 2.0
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-1, 1],
            intercept=0,
            slope=1
        )
        pmap = ParametricMap(
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
        assert pmap.BitsAllocated == 64
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
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 4095],
            intercept=0,
            slope=1
        )
        pmap = ParametricMap(
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
        assert pmap.BitsAllocated == 16
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_single_frame_ct_image_ushort(self):
        pixel_array = np.random.randint(
            low=120,
            high=24000,
            size=self._ct_image.pixel_array.shape,
            dtype=np.uint16
        )
        window_center = 2**16 / 2
        window_width = 2**16
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 2**16],
            intercept=-1,
            slope=2. / (2**16 - 1)
        )
        pmap = ParametricMap(
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
        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-1, 1],
            intercept=0,
            slope=1
        )
        pmap = ParametricMap(
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

    def test_multi_frame_sm_image_with_spatial_positions_not_preserved(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256

        pixel_spacing = (0.5, 0.5)
        slice_thickness = 0.3
        pixel_measures = PixelMeasuresSequence(
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )
        image_orientation = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        plane_orientation = PlaneOrientationSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_orientation=image_orientation
        )
        plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.SLIDE,
                image_position=(i * 1.0, i * 1.0, 1.0),
                pixel_matrix_position=(i * 1, i * 1)
            )
            for i in range(self._sm_image.pixel_array.shape[0])
        ]

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )

        instance = ParametricMap(
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
            pixel_measures=pixel_measures,
            plane_orientation=plane_orientation,
            plane_positions=plane_positions
        )
        assert instance.pixel_array.dtype == np.uint8
        assert instance.BitsAllocated == 8

        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == list(pixel_spacing)
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationSlide == list(image_orientation)
        self.check_dimension_index_vals(instance)
