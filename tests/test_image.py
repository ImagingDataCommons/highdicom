"""Tests for the highdicom.image module."""
from io import BytesIO
from pathlib import Path
from pydicom.filebase import DicomBytesIO
from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.sr.codedict import codes
import numpy as np
import pickle
import pkgutil
import pydicom
import pytest
import re

from highdicom import (
    Image,
    Volume,
    imread,
)
from highdicom._module_utils import (
    does_iod_have_pixel_data,
)
from highdicom.content import VOILUTTransformation
from highdicom.image import (
    _CombinedPixelTransform,
)
from highdicom.pixels import (
    apply_voi_window,
)
from highdicom.pr.content import (
    _add_icc_profile_attributes,
)
from highdicom.pm import (
    RealWorldValueMapping,
    ParametricMap,
)
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID
from tests.utils import find_readable_images


def test_slice_spacing():
    ct_multiframe = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    image = Image.from_dataset(ct_multiframe)

    expected_affine = np.array(
        [
            [0.0, 0.0, -0.388672, 99.5],
            [0.0, 0.388672, 0.0, -301.5],
            [10.0, 0.0, 0.0, -159],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    volume_geometry = image.get_volume_geometry()
    assert volume_geometry is not None
    assert volume_geometry.spatial_shape[0] == 2
    assert np.array_equal(volume_geometry.affine, expected_affine)


def test_slice_spacing_irregular():
    ct_multiframe = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )

    # Mock some irregular spacings
    ct_multiframe.PerFrameFunctionalGroupsSequence[0].\
        PlanePositionSequence[0].ImagePositionPatient = [1.0, 0.0, 0.0]

    image = Image.from_dataset(ct_multiframe)

    assert image.get_volume_geometry() is None


def test_pickle():
    # Check that the database is successfully serialized and deserialized
    ct_multiframe = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    image = Image.from_dataset(ct_multiframe)

    ptr = image.dimension_index_pointers[0]

    pickled = pickle.dumps(image)

    # Check that the pickling process has not damaged the db on the existing
    # instance
    # This is just an example operation that requires the db
    assert not image.are_dimension_indices_unique([ptr])

    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, Image)

    # Check that the database has been successfully restored in the
    # deserialization process
    assert not unpickled.are_dimension_indices_unique([ptr])


def test_combined_transform_ect_rwvm():

    dcm = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    rwvm_seq = (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .RealWorldValueMappingSequence[0]
    )
    slope = rwvm_seq.RealWorldValueSlope
    intercept = rwvm_seq.RealWorldValueIntercept
    first = rwvm_seq.RealWorldValueFirstValueMapped
    last = rwvm_seq.RealWorldValueLastValueMapped

    for output_dtype in [
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ]:
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
        )

        assert tf.applies_to_all_frames

        assert tf._effective_slope_intercept == (
            slope, intercept
        )
        assert tf._input_range_check == (
            first, last
        )

        input_arr = np.array([[1, 2], [3, 4]], np.uint16)
        expected = input_arr * slope + intercept

        output_arr = tf(input_arr)

        assert np.array_equal(output_arr, expected)

        assert output_arr.dtype == output_dtype

        full_output_arr = tf(dcm.pixel_array[0])
        assert full_output_arr.dtype == output_dtype

        out_of_range_input = np.array(
            [[last + 1, 1], [1, 1]],
            np.uint16
        )
        msg = 'Array contains value outside the valid range.'
        with pytest.raises(ValueError, match=msg):
            tf(out_of_range_input)

    msg = (
        'An unsigned integer data type cannot be used if the intercept is '
        'negative.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransform(
            dcm,
            output_dtype=np.uint32,
        )

    msg = (
        'Palette color transform is required but the image is not a palette '
        'color image.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_palette_color_lut=True,
        )

    msg = (
        'ICC profile is required but the image is not a color or palette '
        'color image.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_icc_profile=True,
        )

    msg = (
        'Datatype int16 does not have capacity for values '
        'with slope 1.00 and intercept -1024.0.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransform(
            dcm,
            output_dtype=np.int16,
        )

    # Various different indexing methods
    unit_code = CodedConcept('ml/100ml/s', 'UCUM', 'ml/100ml/s', '1.4')
    for selector in [-1, 'RCBF', unit_code]:
        tf = _CombinedPixelTransform(
            dcm,
            real_world_value_map_selector=selector,
        )
        assert tf._effective_slope_intercept == (slope, intercept)

    # Various different incorrect indexing methods
    msg = "Requested 'real_world_value_map_selector' is not present."
    other_unit_code = CodedConcept('m/s', 'UCUM', 'm/s', '1.4')
    for selector in [2, -2, 'ABCD', other_unit_code]:
        with pytest.raises(IndexError, match=msg):
            _CombinedPixelTransform(
                dcm,
                real_world_value_map_selector=selector,
            )

    # Delete the real world value map
    del (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .RealWorldValueMappingSequence
    )
    msg = (
        'A real-world value map is required but not found in the image.'
    )
    with pytest.raises(RuntimeError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_real_world_transform=True,
        )


def test_combined_transform_ect_modality():

    dcm = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    pix_value_seq = (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .PixelValueTransformationSequence[0]
    )
    slope = pix_value_seq.RescaleSlope
    intercept = pix_value_seq.RescaleIntercept

    for output_dtype in [
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ]:
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
            apply_real_world_transform=False,
        )

        assert tf.applies_to_all_frames

        assert tf._effective_slope_intercept == (
            slope, intercept
        )
        assert tf._input_range_check is None

        input_arr = np.array([[1, 2], [3, 4]], np.uint16)
        expected = input_arr * slope + intercept

        output_arr = tf(input_arr)

        assert np.array_equal(output_arr, expected)

        assert output_arr.dtype == output_dtype

        full_output_arr = tf(dcm.pixel_array[0])
        assert full_output_arr.dtype == output_dtype

        # Same thing should work by requiring the modality LUT
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
            apply_modality_transform=True,
        )

        assert tf.applies_to_all_frames

        assert tf._effective_slope_intercept == (
            slope, intercept
        )
        assert tf._input_range_check is None

        full_output_arr = tf(dcm.pixel_array[0])
        assert full_output_arr.dtype == output_dtype

    msg = (
        'An unsigned integer data type cannot be used if the intercept is '
        'negative.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransform(
            dcm,
            output_dtype=np.uint32,
        )

    msg = (
        'Datatype int16 does not have capacity for values '
        'with slope 1.00 and intercept -1024.0.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransform(
            dcm,
            output_dtype=np.int16,
        )

    # Delete the modality transform
    del (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .PixelValueTransformationSequence
    )
    msg = (
        'A modality LUT transform is required but not found in '
        'the image.'
    )
    with pytest.raises(RuntimeError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_modality_transform=True,
        )


def test_combined_transform_ect_with_voi():

    dcm = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    pix_value_seq = (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .PixelValueTransformationSequence[0]
    )
    slope = pix_value_seq.RescaleSlope
    intercept = pix_value_seq.RescaleIntercept
    frame_voi_seq = (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .FrameVOILUTSequence[0]
    )
    center = frame_voi_seq.WindowCenter
    width = frame_voi_seq.WindowWidth

    lower = center - width // 2
    upper = center + width // 2

    for output_dtype in [
        np.float16,
        np.float32,
        np.float64,
    ]:
        for output_range in [
            (0., 1.),
            (-10.0, 10.0),
            (50., 100.0),
        ]:
            tf = _CombinedPixelTransform(
                dcm,
                output_dtype=output_dtype,
                apply_real_world_transform=False,
                apply_voi_transform=None,
                voi_output_range=output_range,
            )

            assert tf.applies_to_all_frames

            assert tf._effective_window_center_width == (
                center - intercept, width / slope
            )
            assert tf._input_range_check is None
            assert tf._effective_slope_intercept is None
            assert tf._color_manager is None
            assert tf._voi_output_range == output_range
            assert tf._effective_voi_function == 'LINEAR'

            input_arr = np.array(
                [
                    [lower - intercept, center - intercept],
                    [upper - intercept - 1, upper - intercept - 1]
                ],
                np.uint16
            )
            expected = np.array([[0.0, 0.5], [1.0, 1.0]])
            output_scale = output_range[1] - output_range[0]
            expected = expected * output_scale + output_range[0]

            output_arr = tf(input_arr)

            assert np.allclose(output_arr, expected, atol=0.5)

            assert output_arr.dtype == output_dtype

            full_output_arr = tf(dcm.pixel_array[0])
            assert full_output_arr.dtype == output_dtype

    msg = (
        'The VOI transform requires a floating point data type.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransform(
            dcm,
            output_dtype=np.int32,
            apply_real_world_transform=False,
            apply_voi_transform=None,
        )

    # Delete the voi transform
    del (
        dcm
        .SharedFunctionalGroupsSequence[0]
        .PixelValueTransformationSequence
    )
    msg = (
        'A modality LUT transform is required but not found in '
        'the image.'
    )
    with pytest.raises(RuntimeError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_modality_transform=True,
        )


def test_combined_transform_modality_lut():
    # A test file that has a modality LUT
    f = get_testdata_file('mlut_18.dcm')
    dcm = pydicom.dcmread(f)
    lut_data = dcm.ModalityLUTSequence[0].LUTData

    input_arr = np.array([[-2048, -2047], [-2046, -2045]], np.int16)
    expected = np.array(
        [
            [lut_data[0], lut_data[1]],
            [lut_data[2], lut_data[3]],
        ],
    )

    for output_dtype in [
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]:
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype
        )
        assert tf._effective_lut_data is not None
        assert tf._effective_window_center_width is None
        assert tf._effective_slope_intercept is None
        assert tf._color_manager is None
        assert tf._input_range_check is None

        output_arr = tf(input_arr)
        assert np.array_equal(output_arr, expected)
        assert output_arr.dtype == output_dtype

        full_output_arr = tf(dcm.pixel_array)
        assert full_output_arr.dtype == output_dtype

    msg = (
        'A VOI transform is required but not found in the image.'
    )
    with pytest.raises(RuntimeError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_voi_transform=True,
        )

    msg = re.escape(
        "Cannot cast array data from dtype('uint16') to "
        "dtype('float16') according to the rule 'safe'"
    )
    with pytest.raises(TypeError, match=msg):
        tf = _CombinedPixelTransform(dcm, output_dtype=np.float16)

    # Add a voi lut
    dcm.WindowCenter = 24
    dcm.WindowWidth = 24

    tf = _CombinedPixelTransform(dcm, apply_voi_transform=None)
    output_arr = tf(input_arr)
    expected = np.array([[0.0, 0.17391304], [0.86956522, 1.0]])
    assert np.allclose(output_arr, expected)


def test_combined_transform_multiple_vois():
    # This test file includes multiple windows
    f = get_testdata_file('examples_overlay.dcm')
    dcm = pydicom.dcmread(f)
    c1, c2 = dcm.WindowCenter
    w1, w2 = dcm.WindowWidth

    tf = _CombinedPixelTransform(dcm, apply_voi_transform=None)
    assert tf._effective_window_center_width == (c1, w1)

    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector=1,
    )
    assert tf._effective_window_center_width == (c2, w2)
    assert tf._effective_voi_function == 'LINEAR'

    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector=-1,
    )
    assert tf._effective_window_center_width == (c2, w2)
    assert tf._effective_voi_function == 'LINEAR'

    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector=-2,
    )
    assert tf._effective_window_center_width == (c1, w1)
    assert tf._effective_voi_function == 'LINEAR'

    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector='WINDOW1',
    )
    assert tf._effective_window_center_width == (c1, w1)
    assert tf._effective_voi_function == 'LINEAR'

    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector='WINDOW2',
    )
    assert tf._effective_window_center_width == (c2, w2)
    assert tf._effective_voi_function == 'LINEAR'

    msg = "Requested 'voi_transform_selector' is not present."
    with pytest.raises(IndexError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_voi_transform=None,
            voi_transform_selector='DOES_NOT_EXIST',
        )
    with pytest.raises(IndexError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_voi_transform=None,
            voi_transform_selector=2,
        )
    with pytest.raises(IndexError, match=msg):
        tf = _CombinedPixelTransform(
            dcm,
            apply_voi_transform=None,
            voi_transform_selector=-3,
        )

    c3, w3 = (40, 400)
    external_voi = VOILUTTransformation(
        window_center=c3,
        window_width=w3,
    )
    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector=external_voi,
    )
    assert tf._effective_window_center_width == (c3, w3)

    # External VOIs should not contain multiple transforms
    invalid_external_voi = VOILUTTransformation(
        window_center=[100, 200],
        window_width=[300, 400],
    )
    msg = (
        "If providing a VOILUTTransformation as the "
        "'voi_transform_selector', it must contain a single transform."
    )
    with pytest.raises(ValueError, match=msg):
        tf = _CombinedPixelTransform(
            dcm,
            apply_voi_transform=None,
            voi_transform_selector=invalid_external_voi,
        )


def test_combined_transform_voi_lut():
    # A test file that has a voi LUT
    f = get_testdata_file('vlut_04.dcm')
    dcm = pydicom.dcmread(f)
    lut_data = dcm.VOILUTSequence[0].LUTData
    first_mapped_value = dcm.VOILUTSequence[0].LUTDescriptor[1]

    for output_dtype in [
        np.float32,
        np.float64,
    ]:
        for output_range in [
            (0., 1.),
            (-10.0, 10.0),
            (50., 100.0),
        ]:
            tf = _CombinedPixelTransform(
                dcm,
                output_dtype=output_dtype,
                apply_voi_transform=None,
                voi_output_range=output_range,
            )
            assert tf._effective_lut_data is not None
            assert tf._effective_window_center_width is None
            assert tf._effective_slope_intercept is None
            assert tf._color_manager is None
            assert tf._input_range_check is None
            assert tf._voi_output_range == output_range

            input_arr = np.array(
                [
                    [first_mapped_value, first_mapped_value + 1],
                    [first_mapped_value + 2, first_mapped_value + 3],
                ]
            )
            output_scale = (
                (max(lut_data) - min(lut_data)) /
                (output_range[1] - output_range[0])
            )
            expected = np.array(
                [
                    [lut_data[0], lut_data[1]],
                    [lut_data[2], lut_data[3]],
                ]
            ) / output_scale + output_range[0]

            output_arr = tf(input_arr)
            assert np.allclose(output_arr, expected, atol=0.1)
            assert output_arr.dtype == output_dtype

            full_output_arr = tf(dcm.pixel_array)
            assert full_output_arr.dtype == output_dtype

    # Create an explanation to use for searching by explanation
    dcm.VOILUTSequence[0].LUTExplanation = 'BONE'

    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector='BONE'
    )
    assert tf._effective_lut_data is not None

    tf = _CombinedPixelTransform(
        dcm,
        apply_voi_transform=None,
        voi_transform_selector=-1,
    )
    assert tf._effective_lut_data is not None

    msg = "Requested 'voi_transform_selector' is not present."
    with pytest.raises(IndexError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_voi_transform=None,
            voi_transform_selector='NOT_BONE',
        )
    with pytest.raises(IndexError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_voi_transform=None,
            voi_transform_selector=1,
        )


def test_combined_transform_monochrome():
    # A test file that has a modality LUT
    f = get_testdata_file('RG1_UNCR.dcm')
    dcm = pydicom.dcmread(f)

    center_width = (dcm.WindowCenter, dcm.WindowWidth)

    max_value = 2 ** dcm.BitsStored - 1

    for output_dtype in [
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]:
        # Default behavior; inversion but no VOI
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
        )
        assert tf._effective_slope_intercept == (-1, max_value)
        assert tf._effective_lut_data is None
        assert tf._effective_window_center_width is None
        assert tf._color_manager is None
        assert tf._input_range_check is None

        output_arr = tf(dcm.pixel_array)

        expected = max_value - dcm.pixel_array
        expected = expected.astype(output_dtype)
        if output_dtype != np.float16:
            # float16 seems to give a lot of precision related errors in this
            # range
            assert np.array_equal(output_arr, expected)
        assert output_arr.dtype == output_dtype

        # No inversion
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
            apply_presentation_lut=False,
        )
        assert tf._effective_slope_intercept is None
        assert tf._effective_lut_data is None
        assert tf._effective_window_center_width is None
        assert tf._color_manager is None
        assert tf._input_range_check is None

        output_arr = tf(dcm.pixel_array)

        expected = dcm.pixel_array
        expected = expected.astype(output_dtype)
        assert np.array_equal(output_arr, expected)
        assert output_arr.dtype == output_dtype

    for output_dtype in [
        np.float16,
        np.float32,
        np.float64,
    ]:
        # VOI and inversion
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
            apply_voi_transform=None,
        )
        assert tf._effective_slope_intercept is None
        assert tf._effective_lut_data is None
        assert tf._effective_window_center_width == center_width
        assert tf._color_manager is None
        assert tf._input_range_check is None
        assert tf._invert

        output_arr = tf(dcm.pixel_array)

        expected = apply_voi_window(
            dcm.pixel_array,
            window_width=dcm.WindowWidth,
            window_center=dcm.WindowCenter,
            dtype=output_dtype,
            invert=True,
        )
        if output_dtype != np.float16:
            # float16 seems to give a lot of precision related errors in this
            # range
            assert np.array_equal(output_arr, expected)
        assert output_arr.dtype == output_dtype

        # VOI and no inversion
        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
            apply_voi_transform=None,
            apply_presentation_lut=False,
        )
        assert tf._effective_slope_intercept is None
        assert tf._effective_lut_data is None
        assert tf._effective_window_center_width == center_width
        assert tf._color_manager is None
        assert tf._input_range_check is None
        assert not tf._invert

        output_arr = tf(dcm.pixel_array)

        expected = apply_voi_window(
            dcm.pixel_array,
            window_width=dcm.WindowWidth,
            window_center=dcm.WindowCenter,
            dtype=output_dtype,
            invert=False,
        )
        if output_dtype != np.float16:
            # float16 seems to give a lot of precision related errors in this
            # range
            assert np.array_equal(output_arr, expected)
        assert output_arr.dtype == output_dtype


def test_combined_transform_color():
    # A simple color image test file, with no ICC profile
    f = get_testdata_file('color-pl.dcm')
    dcm = pydicom.dcmread(f)

    # Not quite sure why this is needed...
    # The original UID is not recognized
    dcm.SOPClassUID = pydicom.uid.UltrasoundImageStorage

    tf = _CombinedPixelTransform(dcm)
    assert tf._effective_slope_intercept is None
    assert tf._effective_lut_data is None
    assert tf._effective_window_center_width is None
    assert tf._color_manager is None
    assert tf._input_range_check is None
    assert not tf._invert

    output_arr = tf(dcm.pixel_array)
    assert np.array_equal(output_arr, dcm.pixel_array)

    msg = "An ICC profile is required but not found in the image."
    with pytest.raises(RuntimeError, match=msg):
        _CombinedPixelTransform(
            dcm,
            apply_icc_profile=True,
        )

    # Add an ICC profile
    # Use default sRGB profile
    icc_profile = pkgutil.get_data(
        'highdicom',
        '_icc_profiles/sRGB_v4_ICC_preference.icc'
    )
    _add_icc_profile_attributes(
        dcm,
        icc_profile=icc_profile,
    )
    tf = _CombinedPixelTransform(dcm)
    assert tf._effective_slope_intercept is None
    assert tf._effective_lut_data is None
    assert tf._effective_window_center_width is None
    assert tf._color_manager is not None
    assert tf._input_range_check is None
    assert not tf._invert

    output_arr = tf(dcm.pixel_array)


def test_combined_transform_labelmap_seg():
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data')
    f = data_dir / 'test_files/seg_image_sm_control_labelmap_palette_color.dcm'

    dcm = pydicom.dcmread(f)

    for output_dtype in [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ]:
        tf = _CombinedPixelTransform(dcm, output_dtype=output_dtype)
        assert tf._effective_slope_intercept is None
        assert tf._effective_lut_data is not None
        assert tf._effective_window_center_width is None
        assert tf._color_manager is not None
        assert tf._input_range_check is None
        assert not tf._invert

        input_arr = dcm.pixel_array[0]
        output_arr = tf(input_arr)
        assert output_arr.shape == (dcm.Rows, dcm.Columns, 3)
        assert output_arr.dtype == output_dtype

        tf = _CombinedPixelTransform(
            dcm,
            output_dtype=output_dtype,
            apply_icc_profile=False,
        )
        assert tf._effective_slope_intercept is None
        assert tf._effective_lut_data is not None
        assert tf._effective_lut_data.dtype == output_dtype
        assert tf._effective_window_center_width is None
        assert tf._color_manager is None
        assert tf._input_range_check is None
        assert not tf._invert

        input_arr = dcm.pixel_array[0]
        output_arr = tf(input_arr)
        assert output_arr.shape == (dcm.Rows, dcm.Columns, 3)
        assert output_arr.dtype == output_dtype


def test_combined_transform_sm_image():
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data')
    f = data_dir / 'test_files/sm_image_control.dcm'

    dcm = pydicom.dcmread(f)

    for output_dtype in [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ]:
        tf = _CombinedPixelTransform(dcm, output_dtype=output_dtype)
        assert tf._effective_slope_intercept is None
        assert tf._effective_lut_data is None
        assert tf._effective_window_center_width is None
        assert tf._color_manager is not None
        assert tf._input_range_check is None
        assert not tf._invert

        input_arr = dcm.pixel_array[0]
        output_arr = tf(input_arr)
        assert output_arr.shape == (dcm.Rows, dcm.Columns, 3)
        assert output_arr.dtype == output_dtype


def test_combined_transform_all_test_files():
    # A simple test that the transform at least does something for the default
    # parameters for all images in the pydicom test suite
    all_files = get_testdata_files()

    for f in all_files:
        try:
            dcm = pydicom.dcmread(f)
        except Exception:
            continue

        if 'SOPClassUID' not in dcm:
            continue
        if not does_iod_have_pixel_data(dcm.SOPClassUID):
            continue

        try:
            pix = dcm.pixel_array
        except Exception:
            continue

        tf = _CombinedPixelTransform(dcm)

        # Crudely decide whether indexing by frame is needed
        expected_dims = 3 if dcm.SamplesPerPixel > 1 else 2
        if pix.ndim > expected_dims:
            pix = pix[0]

        out = tf(pix)
        assert isinstance(out, np.ndarray)


def test_combined_transform_pmap_rwvm_lut():
    # Construct a temporary parametric map with a real world value map lut
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data')
    f = data_dir / 'test_files/ct_image.dcm'
    source_image = pydicom.dcmread(f)

    m = RealWorldValueMapping(
        lut_label='1',
        lut_explanation='Feature 1',
        unit=codes.UCUM.NoUnits,
        value_range=(0, 255),
        lut_data=[v**2 - 0.15 for v in range(256)]
    )

    pixel_array = np.zeros(
        source_image.pixel_array.shape,
        dtype=np.uint16
    )

    pmap = ParametricMap(
        pixel_array=pixel_array,
        source_images=[source_image],
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        manufacturer='manufacturer',
        manufacturer_model_name='manufacturer_model_name',
        software_versions='software_versions',
        device_serial_number='12345',
        real_world_value_mappings=[m],
        contains_recognizable_visual_features=False,
        window_center=0,
        window_width=100,
    )

    output_dtype = np.float64
    tf = _CombinedPixelTransform(pmap, output_dtype=output_dtype)
    assert tf._effective_lut_data is not None
    assert tf._effective_lut_data.dtype == output_dtype
    assert tf._effective_slope_intercept is None
    assert tf._effective_window_center_width is None
    assert tf._color_manager is None
    assert tf._input_range_check is None
    assert not tf._invert

    out = tf(pmap.pixel_array)
    assert out.dtype == output_dtype

    test_arr = np.array([[0, 1], [254, 255]], np.uint16)
    output_arr = tf(test_arr)
    assert output_arr.dtype == output_dtype

    msg = re.escape(
        "Cannot cast array data from dtype('float64') to "
        "dtype('float32') according to the rule 'safe'"
    )
    with pytest.raises(TypeError, match=msg):
        tf = _CombinedPixelTransform(pmap, output_dtype=np.float32)


def test_get_volume_multiframe_ct():
    im = imread(get_testdata_file('eCT_Supplemental.dcm'))
    volume = im.get_volume()

    assert isinstance(volume, Volume)
    assert volume.spatial_shape == (2, 512, 512)
    assert volume.channel_shape == ()

    volume = im.get_volume(
        apply_voi_transform=True,
        apply_real_world_transform=False
    )
    assert volume.array.min() == 0.0
    assert volume.array.max() == 1.0

    for dtype in [
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]:
        volume = im.get_volume(dtype=dtype)
        assert volume.array.dtype == dtype


def test_get_total_pixel_matrix_dtypes():
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data')
    f = data_dir / 'test_files/sm_image_control.dcm'
    im = imread(f)

    for dtype in [
        np.uint8,
        np.uint32,
        np.int64,
        np.float32,
        np.float64,
    ]:
        tpm = im.get_total_pixel_matrix(dtype=dtype)
        assert tpm.shape == (
            im.TotalPixelMatrixRows,
            im.TotalPixelMatrixColumns,
            3
        )
        assert tpm.dtype == dtype


def test_get_total_pixel_matrix_subvolumes():
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data')
    f = data_dir / 'test_files/sm_image_control.dcm'
    im = imread(f)

    tpm = im.get_total_pixel_matrix()
    assert tpm.shape == (
        im.TotalPixelMatrixRows,
        im.TotalPixelMatrixColumns,
        3
    )
    assert tpm.dtype == np.float64
    volume = im.get_volume()
    assert np.array_equal(volume.array, tpm[None])

    origin_seq = im.TotalPixelMatrixOriginSequence[0]
    expected_position = np.array(
        [
            origin_seq.XOffsetInSlideCoordinateSystem,
            origin_seq.YOffsetInSlideCoordinateSystem,
            origin_seq.get('ZOffsetInSlideCoordinateSystem', 0.0),
        ]
    )
    assert np.array_equal(volume.position, expected_position)
    orientation = im.ImageOrientationSlide
    assert np.array_equal(volume.direction_cosines, orientation)

    sub_tpm = im.get_total_pixel_matrix(
        row_start=25,
        row_end=35,
        column_start=15,
        column_end=35,
    )
    assert np.array_equal(sub_tpm, tpm[24:34, 14:34])

    sub_volume = im.get_volume(
        row_start=25,
        row_end=35,
        column_start=15,
        column_end=35,
    )
    assert np.array_equal(sub_volume.array, sub_tpm[None])
    assert np.array_equal(sub_volume.affine, volume[:, 24:34, 14:34].affine)

    sub_tpm = im.get_total_pixel_matrix(
        row_start=25,
        row_end=35,
        column_start=15,
        column_end=35,
        as_indices=True,
    )
    assert np.array_equal(sub_tpm, tpm[25:35, 15:35])

    sub_volume = im.get_volume(
        row_start=25,
        row_end=35,
        column_start=15,
        column_end=35,
        as_indices=True
    )
    assert np.array_equal(sub_volume.array, sub_tpm[None])
    assert np.array_equal(sub_volume.affine, volume[:, 25:35, 15:35].affine)

    sub_tpm = im.get_total_pixel_matrix(
        row_start=25,
        row_end=-16,
        column_start=-36,
        column_end=35,
    )
    assert np.array_equal(sub_tpm, tpm[24:34, 14:34])

    sub_volume = im.get_volume(
        row_start=25,
        row_end=-16,
        column_start=-36,
        column_end=35,
    )
    assert np.array_equal(sub_volume.array, sub_tpm[None])
    assert np.array_equal(sub_volume.affine, volume[:, 24:34, 14:34].affine)

    sub_tpm = im.get_total_pixel_matrix(
        row_start=24,
        row_end=-16,
        column_start=-36,
        column_end=34,
        as_indices=True,
    )
    assert np.array_equal(sub_tpm, tpm[24:34, 14:34])

    sub_volume = im.get_volume(
        row_start=24,
        row_end=-16,
        column_start=-36,
        column_end=34,
        as_indices=True,
    )
    assert np.array_equal(sub_volume.array, sub_tpm[None])
    assert np.array_equal(sub_volume.affine, volume[:, 24:34, 14:34].affine)


def test_get_volume_multiframe_ct_subvolumes():
    # Test various combinations of the parameters to retrieve a sub-volume
    im = imread(get_testdata_file('eCT_Supplemental.dcm'))
    full_volume = im.get_volume()
    sub_volume = im.get_volume(
        slice_start=2,
        row_start=11,
        column_start=21,
    )

    assert sub_volume.spatial_shape == (1, 502, 492)
    assert sub_volume.channel_shape == ()
    assert np.array_equal(sub_volume.array, full_volume[1, 10:, 20:].array)
    assert np.array_equal(sub_volume.affine, full_volume[1, 10:, 20:].affine)

    sub_volume = im.get_volume(
        slice_start=1,
        row_start=10,
        column_start=20,
        as_indices=True,
    )

    assert sub_volume.spatial_shape == (1, 502, 492)
    assert sub_volume.channel_shape == ()
    assert np.array_equal(sub_volume.array, full_volume[1, 10:, 20:].array)
    assert np.array_equal(sub_volume.affine, full_volume[1, 10:, 20:].affine)

    sub_volume = im.get_volume(
        slice_start=-1,
        row_start=257,
        row_end=-10,
        column_start=-48,
        column_end=504,
    )

    assert sub_volume.spatial_shape == (1, 246, 39)
    assert sub_volume.channel_shape == ()
    assert np.array_equal(
        sub_volume.array,
        full_volume[1, 256:-10, -48:503].array
    )
    assert np.array_equal(
        sub_volume.affine,
        full_volume[1, 256:-10, -48:503].affine
    )

    sub_volume = im.get_volume(
        slice_start=-1,
        row_start=256,
        row_end=-10,
        column_start=-48,
        column_end=503,
        as_indices=True,
    )

    assert sub_volume.spatial_shape == (1, 246, 39)
    assert sub_volume.channel_shape == ()
    assert np.array_equal(
        sub_volume.array,
        full_volume[1, 256:-10, -48:503].array
    )
    assert np.array_equal(
        sub_volume.affine,
        full_volume[1, 256:-10, -48:503].affine
    )


def test_instantiation():
    # Instantiation of the Image class is not allowed
    msg = (
        'Instances of this class should not be directly instantiated. Use '
        'the from_dataset method or the imread function instead.'
    )
    with pytest.raises(RuntimeError, match=msg):
        Image()


def test_get_frames():
    f = get_testdata_file('eCT_Supplemental.dcm')
    im = imread(f)
    dcm = pydicom.dcmread(f)

    all_frames = im.get_frames()
    all_frames_reversed = im.get_frames([2, 1])
    assert np.array_equal(all_frames, all_frames_reversed[::-1])

    all_frames_reversed = im.get_frames(
        (i for i in [1, 0]),  # generator rather than list
        as_indices=True
    )
    assert np.array_equal(all_frames, all_frames_reversed[::-1])

    all_stored_frames = im.get_stored_frames()
    assert np.array_equal(all_stored_frames, dcm.pixel_array)
    all_stored_frames_reversed = im.get_stored_frames([2, 1])
    assert np.array_equal(
        all_stored_frames,
        all_stored_frames_reversed[::-1]
    )

    all_stored_frames_reversed = im.get_stored_frames(
        (i for i in [1, 0]),  # generator rather than list
        as_indices=True
    )
    assert np.array_equal(
        all_stored_frames,
        all_stored_frames_reversed[::-1]
    )


@pytest.mark.parametrize(
    'f,dependency',
    find_readable_images(),
)
def test_imread_all_test_files(f, dependency):
    # A simple test that the reads in all images in the pydicom test suite
    # and gets a single frame
    if dependency is not None:
        pytest.importorskip(dependency)

    im = imread(f)
    im_lazy = imread(f, lazy_frame_retrieval=True)

    all_frames = im.get_frames()
    all_frames_lazy = im_lazy.get_frames()
    assert np.array_equal(all_frames, all_frames_lazy)

    # Check first frames match between lazy and normal
    frame = im.get_frame(1)
    frame_lazy = im_lazy.get_frame(1)
    assert np.array_equal(frame, frame_lazy)
    assert np.array_equal(frame, all_frames[0])

    frame_2 = im.get_frames([1])
    assert np.array_equal(frame_2[0], frame)

    # If multiple frames, also check the last frame
    if im.number_of_frames > 1:
        frame = im.get_frame(im.number_of_frames)
        frame_lazy = im_lazy.get_frame(im.number_of_frames)
        assert np.array_equal(frame, frame_lazy)

        frame_2 = im.get_frames([im.number_of_frames])
        assert np.array_equal(frame_2[0], frame)

    # Skip segmentations as Image doesn't know how to handle segments as a
    # dimension
    is_segmentation = 'SegmentSequence' in im

    # If a volume can be formed into a volume, test this also matches
    # between the lazy and normal
    if (
        im.get_volume_geometry(allow_duplicate_positions=False) is not None and
        not is_segmentation
    ):
        # Only test images that be "simply" indexed as a volume, i.e.
        # without having to filter on any other dimension
        # Check that we can retrieve volumes
        vol = im.get_volume()
        vol_lazy = im_lazy.get_volume()

        assert np.array_equal(vol.array, vol_lazy.array)
    elif im.coordinate_system is None:
        msg = (
            'Image does not exist within a frame-of-reference coordinate '
            'system.'
        )
        with pytest.raises(RuntimeError, match=msg):
            im.get_volume()

    assert isinstance(im.pixel_array, np.ndarray)
    assert np.array_equal(im.pixel_array, im_lazy.pixel_array)
    assert im_lazy._pixel_array is not None

    # Perform this check again to check the caching behavior
    assert np.array_equal(im.pixel_array, im_lazy.pixel_array)


def test_imread_from_bytes():
    dcm = pydicom.dcmread(get_testdata_file('eCT_Supplemental.dcm'))

    with BytesIO() as buf:
        dcm.save_as(buf)
        im = imread(buf.getvalue())

        assert isinstance(im, Image)

        # Two reads to ensure opening/closing is handled
        im.get_frame(1)
        im.get_frame(2)


def test_imread_from_bytes_lazy():
    dcm = pydicom.dcmread(get_testdata_file('eCT_Supplemental.dcm'))

    with BytesIO() as buf:
        dcm.save_as(buf)
        im = imread(buf.getvalue(), lazy_frame_retrieval=True)

        assert isinstance(im, Image)

        # Two reads to ensure opening/closing is handled
        im.get_frame(1)
        im.get_frame(2)


def test_imread_from_bytes_io():
    dcm_bytes = open(get_testdata_file('eCT_Supplemental.dcm'), 'rb').read()

    with BytesIO(dcm_bytes) as buf:
        im = imread(buf)

        assert isinstance(im, Image)

        # Two reads to ensure opening/closing is handled
        im.get_frame(1)
        im.get_frame(2)


def test_imread_from_bytes_io_lazy():
    dcm_bytes = open(get_testdata_file('eCT_Supplemental.dcm'), 'rb').read()

    with BytesIO(dcm_bytes) as buf:
        im = imread(buf, lazy_frame_retrieval=True)

        assert isinstance(im, Image)

        # Two reads to ensure opening/closing is handled
        im.get_frame(1)
        im.get_frame(2)


def test_imread_from_dicom_bytes_io():
    dcm_bytes = open(get_testdata_file('eCT_Supplemental.dcm'), 'rb').read()

    with DicomBytesIO(dcm_bytes) as buf:
        im = imread(buf)

        assert isinstance(im, Image)

        # Two reads to ensure opening/closing is handled
        im.get_frame(1)
        im.get_frame(2)


def test_imread_from_dicom_bytes_io_lazy():
    dcm_bytes = open(get_testdata_file('eCT_Supplemental.dcm'), 'rb').read()

    with DicomBytesIO(dcm_bytes) as buf:
        im = imread(buf, lazy_frame_retrieval=True)

        assert isinstance(im, Image)

        # Two reads to ensure opening/closing is handled
        im.get_frame(1)
        im.get_frame(2)
