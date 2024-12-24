"""Tests for the highdicom.image module."""
import pickle
import numpy as np
import pydicom
from pydicom.data import get_testdata_file, get_testdata_files
import pytest

from highdicom.image import (
    _CombinedPixelTransformation,
    MultiFrameImage,
)


def test_slice_spacing():
    ct_multiframe = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    image = MultiFrameImage.from_dataset(ct_multiframe)

    expected_affine = np.array(
        [
            [0.0, 0.0, -0.388672, 99.5],
            [0.0, 0.388672, 0.0, -301.5],
            [10.0, 0.0, 0.0, -159],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert image.volume_geometry is not None
    assert image.volume_geometry.spatial_shape[0] == 2
    assert np.array_equal(image.volume_geometry.affine, expected_affine)


def test_slice_spacing_irregular():
    ct_multiframe = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )

    # Mock some iregular spacings
    ct_multiframe.PerFrameFunctionalGroupsSequence[0].\
        PlanePositionSequence[0].ImagePositionPatient = [1.0, 0.0, 0.0]

    image = MultiFrameImage.from_dataset(ct_multiframe)

    assert image.volume_geometry is None


def test_pickle():
    # Check that the database is successfully serialized and deserialized
    ct_multiframe = pydicom.dcmread(
        get_testdata_file('eCT_Supplemental.dcm')
    )
    image = MultiFrameImage.from_dataset(ct_multiframe)

    ptr = image.dimension_index_pointers[0]

    pickled = pickle.dumps(image)

    # Check that the pickling process has not damaged the db on the existing
    # instance
    # This is just an example operation that requires the db
    assert not image.are_dimension_indices_unique([ptr])

    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, MultiFrameImage)

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
        tf = _CombinedPixelTransformation(
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
        _CombinedPixelTransformation(
            dcm,
            output_dtype=np.uint32,
        )

    msg = (
        'Palette color transform is required but the image is not a palette '
        'color image.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransformation(
            dcm,
            apply_palette_color_lut=True,
        )

    msg = (
        'ICC profile is required but the image is not a color or palette '
        'color image.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransformation(
            dcm,
            apply_icc_profile=True,
        )

    msg = (
        f'Datatype int16 does not have capacity for values '
        f'with slope 1.00 and intercept -1024.0.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransformation(
            dcm,
            output_dtype=np.int16,
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
        _CombinedPixelTransformation(
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
        tf = _CombinedPixelTransformation(
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

        # Same thing should work by requiring the modality LUT
        tf = _CombinedPixelTransformation(
            dcm,
            output_dtype=output_dtype,
            apply_modality_transform=True,
        )

        assert tf.applies_to_all_frames

        assert tf._effective_slope_intercept == (
            slope, intercept
        )
        assert tf._input_range_check is None

    msg = (
        'An unsigned integer data type cannot be used if the intercept is '
        'negative.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransformation(
            dcm,
            output_dtype=np.uint32,
        )

    msg = (
        f'Datatype int16 does not have capacity for values '
        f'with slope 1.00 and intercept -1024.0.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransformation(
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
        _CombinedPixelTransformation(
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
            tf = _CombinedPixelTransformation(
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

    msg = (
        'The VOI transformation requires a floating point data type.'
    )
    with pytest.raises(ValueError, match=msg):
        _CombinedPixelTransformation(
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
        _CombinedPixelTransformation(
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
        np.float16,
        np.float32,
        np.float64,
    ]:
        tf = _CombinedPixelTransformation(
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

    msg = (
        'A VOI transform is required but not found in the image.'
    )
    with pytest.raises(RuntimeError, match=msg):
        _CombinedPixelTransformation(
            dcm,
            apply_voi_transform=True,
        )

    # Add a voi lut
    dcm.WindowCenter = 24
    dcm.WindowWidth = 24

    tf = _CombinedPixelTransformation(dcm, apply_voi_transform=None)
    output_arr = tf(input_arr)
    expected = np.array([[0.0, 0.17391304], [0.86956522, 1.0]])
    assert np.allclose(output_arr, expected)


def test_combined_transform_voi_lut():
    # A test file that has a modality LUT
    f = get_testdata_file('vlut_04.dcm')
    dcm = pydicom.dcmread(f)
    lut_data = dcm.VOILUTSequence[0].LUTData
    first_mapped_value = dcm.VOILUTSequence[0].LUTDescriptor[1]

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
            tf = _CombinedPixelTransformation(
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
