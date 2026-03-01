from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from pydicom.data import get_testdata_file, get_testdata_files
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
    Volume,
    imread,
)
from highdicom.content import VOILUTTransformation
from highdicom.enum import (
    ContentQualificationValues,
    CoordinateSystemNames,
    DimensionOrganizationTypeValues,
    InterpolationMethods,
)
from highdicom.pm.content import RealWorldValueMapping
from highdicom.pm.enum import (
    DerivedPixelContrastValues,
    ImageFlavorValues,
)
from highdicom.pm.sop import ParametricMap
from highdicom.pm.pyramid import create_parametric_map_pyramid
from highdicom.spatial import (
    create_affine_matrix_from_attributes,
    sort_datasets,
)
from highdicom.uid import UID

from .utils import write_and_read_dataset


class TestRealWorldValueMapping():

    def test_failed_construction_missing_or_unnecessary_parameters(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = (0, 255)
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
        value_range = (0, 255)
        intercept = 200
        slope = 10
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
            m.DoubleFloatRealWorldValueFirstValueMapped  # noqa: B018
        with pytest.raises(AttributeError):
            m.DoubleFloatRealWorldValueLastValueMapped  # noqa: B018
        with pytest.raises(AttributeError):
            m.RealWorldValueLUTData  # noqa: B018
        assert len(m.QuantityDefinitionSequence) == 1
        quantity_item = m.QuantityDefinitionSequence[0]
        assert quantity_item.name == codes.SCT.Quantity
        assert quantity_item.value == quantity_definition
        assert not m.has_lut()
        assert m.value_range == value_range

        array = np.array(
            [
                [0, 0, 0],
                [5, 5, 5],
                [10, 10, 10],
            ],
        )
        expected = np.array(
            [
                [200, 200, 200],
                [250, 250, 250],
                [300, 300, 300],
            ],
        )

        out = m.apply(array)
        assert np.array_equal(out, expected)
        assert out.dtype == np.float64

    def test_construction_integer_nonlinear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = (0, 255)
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
            m.DoubleFloatRealWorldValueFirstValueMapped  # noqa: B018
        with pytest.raises(AttributeError):
            m.DoubleFloatRealWorldValueLastValueMapped  # noqa: B018
        with pytest.raises(AttributeError):
            m.RealWorldValueSlope  # noqa: B018
        with pytest.raises(AttributeError):
            m.RealWorldValueIntercept  # noqa: B018
        assert m.has_lut()
        assert m.value_range == value_range

        array = np.array(
            [
                [0, 0, 0],
                [5, 5, 5],
                [10, 10, 10],
            ],
        )
        expected = np.array(
            [
                [0, 0, 0],
                [25, 25, 25],
                [100, 100, 100],
            ],
        )

        out = m.apply(array)
        assert np.array_equal(out, expected)
        assert out.dtype == np.float64

    def test_construction_floating_point_linear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = (-1000.0, 1000.0)
        intercept = -23.13
        slope = 5.0
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
            m.RealWorldValueFirstValueMapped  # noqa: B018
        with pytest.raises(AttributeError):
            m.RealWorldValueLastValueMapped  # noqa: B018
        with pytest.raises(AttributeError):
            m.RealWorldValueLUTData  # noqa: B018
        assert not m.has_lut()
        assert m.value_range == value_range

        array = np.array(
            [
                [0, 0, 0],
                [5, 5, 5],
                [10, 10, 10],
            ],
        )
        expected = np.array(
            [
                [-23.13, -23.13, -23.13],
                [1.87, 1.87, 1.87],
                [26.87, 26.87, 26.87],
            ],
        )

        out = m.apply(array)
        assert np.allclose(out, expected)
        assert out.dtype == np.float64

        invalid_array = np.array(
            [
                [1200, 0, 0],
                [5, 5, 5],
                [10, 10, 10],
            ],
        )
        msg = 'Array contains value outside the valid range.'
        with pytest.raises(ValueError, match=msg):
            m.apply(invalid_array)

    def test_failed_construction_floating_point_nonlinear_relationship(self):
        lut_label = '1'
        lut_explanation = 'Feature 1'
        unit = codes.UCUM.NoUnits
        value_range = (0.0, 1.0)
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


class TestParametricMap():

    @pytest.fixture(autouse=True)
    def setUp(self):
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

        self._integer_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        self._float_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-10000.0, 10000.0],
            intercept=0,
            slope=1
        )

        self._voi_transformation = VOILUTTransformation(
            window_width=128,
            window_center=120,
        )

        self._ct_image = imread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )

        self._ct_multiframe_image = imread(
            get_testdata_file('eCT_Supplemental.dcm')
        )

        self._sm_image = imread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )

        ct_series = [
            imread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ct_series = sort_datasets(ct_series)

        self._ct_volume = (
            imread(get_testdata_file('eCT_Supplemental.dcm'))
            .get_volume()
        )

    @staticmethod
    @pytest.fixture(
        params=[
            DimensionOrganizationTypeValues.TILED_FULL,
            DimensionOrganizationTypeValues.TILED_SPARSE
        ]
    )
    def tiled_dimension_organization(request):
        return request.param

    @staticmethod
    def check_dimension_index_vals(pm):
        # Function to apply some checks (necessary but not sufficient for
        # correctness) to ensure that the dimension indices are correct
        is_patient_coord_system = hasattr(
            pm.PerFrameFunctionalGroupsSequence[0],
            'PlanePositionSequence'
        )
        if is_patient_coord_system:
            # Build up the mapping from index to value
            index_mapping = defaultdict(list)
            for f in pm.PerFrameFunctionalGroupsSequence:
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
            for dim_kw in [
                'ColumnPositionInTotalImagePixelMatrix',
                'RowPositionInTotalImagePixelMatrix',
            ]:
                dim_ind = pm.DimensionIndexSequence.get_index_position(dim_kw)
                index_mapping = defaultdict(list)
                for f in pm.PerFrameFunctionalGroupsSequence:
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

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

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
            voi_lut_transformations=voi_transformations,
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
        assert not hasattr(pmap, 'ICCProfile')
        assert not hasattr(pmap, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(pmap, 'RedPaletteColorLookupTableData')
        assert not hasattr(pmap, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(pmap, 'GreenPaletteColorLookupTableData')
        assert not hasattr(pmap, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(pmap, 'BluePaletteColorLookupTableData')

    def test_multi_frame_sm_image_ushort_native(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
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
            voi_lut_transformations=voi_transformations,
            content_qualification=ContentQualificationValues.SERVICE,
            image_flavor=ImageFlavorValues.WHOLE_BODY,
            derived_pixel_contrast=DerivedPixelContrastValues.NONE
        )
        sffg_item = instance.SharedFunctionalGroupsSequence[0]
        assert hasattr(sffg_item, 'RealWorldValueMappingSequence')
        assert len(sffg_item.RealWorldValueMappingSequence) == 1
        assert hasattr(sffg_item, 'PixelValueTransformationSequence')
        assert hasattr(sffg_item, 'FrameVOILUTSequence')
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
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')

    def test_multi_frame_palette_lut(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

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
            voi_lut_transformations=voi_transformations,
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

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
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
            voi_lut_transformations=voi_transformations,
            content_qualification=ContentQualificationValues.SERVICE,
            image_flavor=ImageFlavorValues.WHOLE_BODY,
            derived_pixel_contrast=DerivedPixelContrastValues.NONE
        )
        sffg_item = instance.SharedFunctionalGroupsSequence[0]
        assert hasattr(sffg_item, 'RealWorldValueMappingSequence')
        assert len(sffg_item.RealWorldValueMappingSequence) == 1
        assert hasattr(sffg_item, 'PixelValueTransformationSequence')
        assert hasattr(sffg_item, 'FrameVOILUTSequence')
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
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')

    def test_multi_frame_sm_image_ushort_encapsulated_jpeg2000(self):
        pytest.importorskip("openjpeg")
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._ct_multiframe_image.pixel_array.shape[:3],
            dtype=np.uint8
        )
        window_center = 128
        window_width = 256

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[0, 255],
            intercept=0,
            slope=1
        )
        pmap = ParametricMap(
            [self._ct_multiframe_image],
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
            voi_lut_transformations=voi_transformations,
            transfer_syntax_uid=JPEG2000Lossless
        )
        assert pmap.BitsAllocated == 8
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_multi_frame_sm_image_ushort_encapsulated_jpegls(self):
        pytest.importorskip("libjpeg")
        pixel_array = np.random.randint(
            low=0,
            high=2**8,
            size=self._sm_image.pixel_array.shape[:3],
            dtype=np.uint16
        )
        window_center = 128
        window_width = 256

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

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
            voi_lut_transformations=voi_transformations,
            transfer_syntax_uid=JPEGLSLossless,
            use_extended_offset_table=True
        )
        assert pmap.BitsAllocated == 16
        assert np.array_equal(pmap.pixel_array, pixel_array)
        assert hasattr(pmap, 'ExtendedOffsetTable')
        assert hasattr(pmap, 'ExtendedOffsetTableLengths')

    def test_single_frame_ct_image_double(self):
        pixel_array = np.random.uniform(-1, 1, self._ct_image.pixel_array.shape)
        window_center = 0.0
        window_width = 2.0

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=(-1.0, 1.0),
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
            voi_lut_transformations=voi_transformations,
        )
        assert pmap.BitsAllocated == 64
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_single_frame_ct_image_double_wrong_value_range(self):
        pixel_array = np.random.uniform(-1, 1, self._ct_image.pixel_array.shape)
        window_center = 0.0
        window_width = 2.0

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=(-1, 1),  # should be float
            intercept=0,
            slope=1
        )

        msg = (
            "When using a floating point-valued pixel_array, "
            "all items in 'real_world_value_mappings' must have "
            "their value range specified with floats."
        )
        with pytest.raises(ValueError, match=msg):
            ParametricMap(
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
                voi_lut_transformations=voi_transformations,
            )

    def test_single_frame_ct_image_ushort_native(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**12,
            size=self._ct_image.pixel_array.shape,
            dtype=np.uint16
        )
        window_center = 2**12 / 2.0
        window_width = 2**12

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

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
            voi_lut_transformations=voi_transformations,
        )
        assert pmap.BitsAllocated == 16
        assert np.array_equal(pmap.pixel_array, pixel_array)

    def test_single_frame_ct_image_ushort_wrong_value_range(self):
        pixel_array = np.random.randint(
            low=0,
            high=2**12,
            size=self._ct_image.pixel_array.shape,
            dtype=np.uint16
        )
        window_center = 2**12 / 2.0
        window_width = 2**12

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=(0.0, 4095.0),  # should be int
            intercept=0,
            slope=1
        )

        msg = (
            "When using an integer-valued 'pixel_array', all items "
            "in 'real_world_value_mappings' must have their value "
            "range specified with integers."
        )
        with pytest.raises(ValueError, match=msg):
            ParametricMap(
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
                voi_lut_transformations=voi_transformations,
            )

    def test_single_frame_ct_image_ushort(self):
        pixel_array = np.random.randint(
            low=120,
            high=24000,
            size=self._ct_image.pixel_array.shape,
            dtype=np.uint16
        )
        window_center = 2**16 / 2
        window_width = 2**16

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

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
            voi_lut_transformations=voi_transformations,
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

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

        real_world_value_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=(-1.0, 1.0),
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
            voi_lut_transformations=voi_transformations,
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

        voi_transformations = [
            VOILUTTransformation(
                window_width=window_width,
                window_center=window_center,
            )
        ]

        origin = (134.2, 12.4, -45.4)
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
        affine = create_affine_matrix_from_attributes(
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            image_position=origin,
            index_convention="RD",  # use column, row
        )

        # Arbitrary tile positions
        # use (column, row ) format to match input to PlanePositionSequence
        tile_positions = np.array(
            [
                [639, 109],
                [467, 275],
                [869, 64],
                [221, 134],
                [372, 943],
                [590, 515],
                [617, 823],
                [761, 912],
                [832, 955],
                [29, 421],
                [546, 1002],
                [444, 47],
                [633, 984],
                [3, 14],
                [867, 1022],
                [911, 610],
                [739, 612],
                [991, 805],
                [489, 462],
                [52, 555],
                [224, 486],
                [168, 307],
                [906, 582],
                [584, 564],
                [724, 384],
            ]
        )
        n = tile_positions.shape[0]

        # Subtract 1 (since top left pixel has position 1, 1) and add column of
        # ones
        tile_positions_aug = np.column_stack(
            [tile_positions - 1, np.ones((n, 1))]
        )

        # Ignore the third column of the affine since there is no slice offset
        plane_position_values = (affine[:3, [0, 1, 3]] @ tile_positions_aug.T).T
        plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.SLIDE,
                pixel_matrix_position=tp.tolist(),
                image_position=pp.tolist(),
            )
            for tp, pp in zip(tile_positions, plane_position_values)
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
            voi_lut_transformations=voi_transformations,
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
        assert not hasattr(shared_item, 'PlaneOrientationSequence')
        assert instance.ImageOrientationSlide == list(image_orientation)
        self.check_dimension_index_vals(instance)

        # Check that the correct origin was inferred from the plane positions
        origin_seq = instance.TotalPixelMatrixOriginSequence[0]
        assert origin_seq.XOffsetInSlideCoordinateSystem == origin[0]
        assert origin_seq.YOffsetInSlideCoordinateSystem == origin[1]
        assert origin_seq.ZOffsetInSlideCoordinateSystem == origin[2]

        # Repeat with plane positions that are inconsistent with the tile
        # positions - should raise an error
        bad_plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.SLIDE,
                pixel_matrix_position=tp.tolist(),
                image_position=(1.0, 1.0, 1.0),  # same position for every tile
            )
            for tp in tile_positions
        ]
        msg = (
            "Some plane positions are not consistent with the provided "
            "plane orientation and pixel measures."
        )
        with pytest.raises(ValueError, match=msg):
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
                voi_lut_transformations=voi_transformations,
                pixel_measures=pixel_measures,
                plane_orientation=plane_orientation,
                plane_positions=bad_plane_positions
            )

    def test_from_volume(self):
        # Creating a parametric map from a volume aligned with the source
        # images
        instance = ParametricMap(
            [self._ct_multiframe_image],
            self._ct_volume,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[self._float_mapping],
            voi_lut_transformations=[self._voi_transformation],
        )
        for frame_item in instance.PerFrameFunctionalGroupsSequence:
            assert len(frame_item.DerivationImageSequence) == 1

        instance = ParametricMap.from_dataset(
            write_and_read_dataset(instance)
        )

        vol = instance.get_volume()
        assert vol.geometry_equal(self._ct_volume)
        assert np.array_equal(vol.array, self._ct_volume.array)

    def test_from_volume_multichannel(self):
        # Creating a parametric map from a volume with multiple channels
        # aligned with the source images
        channels = {'LUTLabel': ['LUT1', 'LUT2']}

        channel_2 = 1000.0 - self._ct_volume.array

        array = np.stack([self._ct_volume.array, channel_2], -1)
        volume = self._ct_volume.with_array(array, channels=channels)

        mapping1 = RealWorldValueMapping(
            lut_label='LUT1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-10000.0, 10000.0],
            intercept=0,
            slope=1
        )
        mapping2 = RealWorldValueMapping(
            lut_label='LUT2',
            lut_explanation='feature_002',
            unit=codes.UCUM.NoUnits,
            value_range=[-10000.0, 10000.0],
            intercept=0,
            slope=1
        )

        # Should fail if the LUTLabels are mimatched between the volume and the
        # real_world_value_mappings parameter
        msg = (
            "The LUTLabels of the 'real_world_value_mappings' "
            "must match those within the channel indentifiers "
            "of the 'pixel_array'."
        )
        with pytest.raises(ValueError, match=msg):
            ParametricMap(
                [self._ct_multiframe_image],
                volume,
                self._series_instance_uid,
                self._series_number,
                self._sop_instance_uid,
                self._instance_number,
                self._manufacturer,
                self._manufacturer_model_name,
                self._software_versions,
                self._device_serial_number,
                contains_recognizable_visual_features=False,
                real_world_value_mappings=[[mapping2], [mapping1]],
                voi_lut_transformations=[self._voi_transformation],
            )

        instance = ParametricMap(
            [self._ct_multiframe_image],
            volume,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[[mapping1], [mapping2]],
            voi_lut_transformations=[self._voi_transformation],
        )
        for frame_item in instance.PerFrameFunctionalGroupsSequence:
            assert len(frame_item.DerivationImageSequence) == 1

        instance = ParametricMap.from_dataset(
            write_and_read_dataset(instance)
        )

        vol = instance.get_volume()
        assert vol.geometry_equal(self._ct_volume)
        assert np.array_equal(vol.array, array)

    def test_from_volume_non_aligned(self):
        # Creating a parametric map from a volume that is not aligned with the
        # source images
        volume = Volume(
            array=self._ct_volume.array,
            affine=np.eye(4),
            coordinate_system='PATIENT',
        )

        instance = ParametricMap(
            [self._ct_multiframe_image],
            volume,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[self._float_mapping],
            voi_lut_transformations=[self._voi_transformation],
        )
        for frame_item in instance.PerFrameFunctionalGroupsSequence:
            assert len(frame_item.DerivationImageSequence) == 0

        instance = ParametricMap.from_dataset(
            write_and_read_dataset(instance)
        )

        vol = instance.get_volume()
        assert vol.geometry_equal(volume)
        assert np.array_equal(vol.array, volume.array)

    def test_autotile(self, tiled_dimension_organization):
        # Creating a parametric map from a total pixel matrix
        tpm = self._sm_image.get_total_pixel_matrix().mean(axis=-1)

        instance = ParametricMap(
            [self._sm_image],
            tpm,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[self._float_mapping],
            voi_lut_transformations=[self._voi_transformation],
            tile_pixel_array=True,
            dimension_organization_type=tiled_dimension_organization,
        )

        instance = ParametricMap.from_dataset(
            write_and_read_dataset(instance)
        )

        new_tpm = instance.get_total_pixel_matrix()
        assert np.array_equal(new_tpm, tpm)

        volume = instance.get_volume()
        assert np.array_equal(volume.array[0], tpm)

    def test_invalid_tiled_full(self):
        # Cannot use TILED_FULL when there are multiple channels
        tpm = self._sm_image.get_total_pixel_matrix()[None, :, :, :2]

        mapping1 = RealWorldValueMapping(
            lut_label='LUT1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=[-10000.0, 10000.0],
            intercept=0,
            slope=1
        )
        mapping2 = RealWorldValueMapping(
            lut_label='LUT2',
            lut_explanation='feature_002',
            unit=codes.UCUM.NoUnits,
            value_range=[-10000.0, 10000.0],
            intercept=0,
            slope=1
        )

        msg = (
            'A value of "TILED_FULL" for parameter '
            '"dimension_organization_type" is not permitted '
            'because the image contains multiple channels. See '
            'https://dicom.nema.org/medical/dicom/current/output/'
            'chtml/part03/sect_C.7.6.17.3.html#sect_C.7.6.17.3.'
        )

        with pytest.raises(ValueError, match=msg):
            ParametricMap(
                [self._sm_image],
                tpm,
                self._series_instance_uid,
                self._series_number,
                self._sop_instance_uid,
                self._instance_number,
                self._manufacturer,
                self._manufacturer_model_name,
                self._software_versions,
                self._device_serial_number,
                contains_recognizable_visual_features=False,
                real_world_value_mappings=[[mapping1], [mapping2]],
                voi_lut_transformations=[self._voi_transformation],
                tile_pixel_array=True,
                dimension_organization_type="TILED_FULL",
            )


class TestParametricMapPyramid():

    @pytest.fixture(autouse=True)
    def setUp(self):
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._sm_image = imread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )
        self._pixel_array = (
            self._sm_image
            .get_total_pixel_matrix().mean(axis=-1)
            [None]
        )

        self._series_instance_uid = UID()
        self._series_number = 76
        self._manufacturer = 'MyManufacturer'
        self._manufacturer_model_name = 'MyModel'
        self._software_versions = 'v1.0'
        self._device_serial_number = '1-2-3'
        self._content_description = 'test parametric map'
        self._content_creator_name = 'will^i^am'
        self._pyramid_uid = UID()
        self._pyramid_label = 'Giza001'
        self._series_description = 'A test pyramid'

        self._float_mapping = RealWorldValueMapping(
            lut_label='1',
            lut_explanation='feature_001',
            unit=codes.UCUM.NoUnits,
            value_range=(-10000.0, 10000.0),
            intercept=0,
            slope=1
        )

        self._voi_transformation = VOILUTTransformation(
            window_width=128,
            window_center=120,
        )

    def test_contruct_pm_pyramid(self):
        pmaps = create_parametric_map_pyramid(
            source_images=[self._sm_image],
            pixel_arrays=[self._pixel_array],
            interpolator=InterpolationMethods.LINEAR,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number,
            contains_recognizable_visual_features=False,
            real_world_value_mappings=[self._float_mapping],
            voi_lut_transformations=[self._voi_transformation],
            downsample_factors=[5.0],
            pyramid_uid=self._pyramid_uid,
            pyramid_label=self._pyramid_label,
            series_description=self._series_description,
        )

        assert len(pmaps) == 2

        for pm in pmaps:
            assert isinstance(pm, ParametricMap)
            assert pm.SeriesInstanceUID == self._series_instance_uid
            assert pm.SeriesNumber == self._series_number
            assert pm.Manufacturer == self._manufacturer
            assert (
                pm.ManufacturerModelName ==
                self._manufacturer_model_name
            )
            assert pm.SoftwareVersions == self._software_versions
            assert pm.DeviceSerialNumber == self._device_serial_number
            assert pm.PyramidLabel == self._pyramid_label
            assert pm.PyramidUID == self._pyramid_uid
            assert 'SeriesDate' in pm
            assert 'SeriesTime' in pm

        assert pmaps[1].TotalPixelMatrixRows == 10
        assert pmaps[1].TotalPixelMatrixColumns == 10
