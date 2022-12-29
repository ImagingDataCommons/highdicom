import pkgutil
import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL.ImageCms import ImageCmsProfile
from pydicom import dcmread
from pydicom.sr.codedict import codes
from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.dataset import Dataset

from highdicom import (
    PaletteColorLUT,
    ContentCreatorIdentificationCodeSequence,
    ModalityLUT,
    ModalityLUTTransformation,
    PresentationLUT,
    PresentationLUTShapeValues,
    PresentationLUTTransformation,
    PaletteColorLUTTransformation,
    RescaleTypeValues,
    ReferencedImageSequence,
    SegmentedPaletteColorLUT,
    UID,
    VOILUT,
    VOILUTFunctionValues,
)
from highdicom.color import CIELabColor
from highdicom.pr import (
    AdvancedBlending,
    AnnotationUnitsValues,
    BlendingDisplay,
    BlendingDisplayInput,
    BlendingModeValues,
    ColorSoftcopyPresentationState,
    GraphicAnnotation,
    GraphicGroup,
    GraphicLayer,
    GraphicObject,
    GraphicTypeValues,
    GrayscaleSoftcopyPresentationState,
    PseudoColorSoftcopyPresentationState,
    SoftcopyVOILUTTransformation,
    TextObject,
)


class TestSoftcopyVOILUTTransformation(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._lut = VOILUT(
            first_mapped_value=0,
            lut_data=np.array([10, 11, 12], np.uint16)
        )
        self._ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ct_multiframe = dcmread(get_testdata_file('eCT_Supplemental.dcm'))
        self._seg = dcmread(
            'data/test_files/seg_image_ct_binary_overlap.dcm'
        )

    def test_construction_basic(self):
        lut = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0
        )
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0
        assert not hasattr(lut, 'VOILUTSequence')

    def test_construction_explanation(self):
        lut = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            window_explanation='Soft Tissue Window'
        )
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0

    def test_construction_multiple(self):
        lut = SoftcopyVOILUTTransformation(
            window_center=[40.0, 600.0],
            window_width=[400.0, 1500.0],
            window_explanation=['Soft Tissue Window', 'Lung Window'],
        )
        assert lut.WindowCenter == [40.0, 600.0]
        assert lut.WindowWidth == [400.0, 1500.0]

    def test_construction_multiple_mismatch1(self):
        with pytest.raises(ValueError):
            SoftcopyVOILUTTransformation(
                window_center=40.0,
                window_width=[400.0, 1500.0],
            )

    def test_construction_multiple_mismatch2(self):
        with pytest.raises(TypeError):
            SoftcopyVOILUTTransformation(
                window_center=[40.0, 600.0],
                window_width=400.0,
            )

    def test_construction_multiple_mismatch3(self):
        with pytest.raises(ValueError):
            SoftcopyVOILUTTransformation(
                window_center=[40.0, 600.0],
                window_width=[400.0, 1500.0, -50.0],
            )

    def test_construction_explanation_mismatch(self):
        with pytest.raises(TypeError):
            SoftcopyVOILUTTransformation(
                window_center=[40.0, 600.0],
                window_width=[400.0, 1500.0],
                window_explanation='Lung Window',
            )

    def test_construction_explanation_mismatch2(self):
        with pytest.raises(ValueError):
            SoftcopyVOILUTTransformation(
                window_center=40.0,
                window_width=400.0,
                window_explanation=['Soft Tissue Window', 'Lung Window'],
            )

    def test_construction_lut_function(self):
        lut = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            voi_lut_function=VOILUTFunctionValues.SIGMOID,
        )
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0
        assert lut.VOILUTFunction == VOILUTFunctionValues.SIGMOID.value

    def test_construction_luts(self):
        lut = SoftcopyVOILUTTransformation(voi_luts=[self._lut])
        assert len(lut.VOILUTSequence) == 1
        assert not hasattr(lut, 'WindowWidth')
        assert not hasattr(lut, 'WindowCenter')

    def test_construction_both(self):
        lut = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            voi_luts=[self._lut]
        )
        assert len(lut.VOILUTSequence) == 1
        assert lut.WindowCenter == 40.0
        assert lut.WindowWidth == 400.0

    def test_construction_neither(self):
        with pytest.raises(TypeError):
            SoftcopyVOILUTTransformation()

    def test_construction_ref_ims(self):
        lut = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ReferencedImageSequence(self._ct_series)
        )
        assert len(lut.ReferencedImageSequence) == len(self._ct_series)


class TestGraphicObject(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._point = np.array([[10.0, 10.0]])
        self._circle = np.array([
            [10.0, 10.0],
            [11.0, 10.0]
        ])
        self._polyline = np.array([
            [10.0, 10.0],
            [11.0, 11.0],
            [9.0, 12.0],
            [8.0, 13.0]
        ])
        self._ellipse = np.array([
            [5.0, 10.0],
            [15.0, 10.0],
            [10.0, 12.0],
            [10.0, 8.0]
        ])
        # In "display" coordinates
        self._circle_display = np.array([
            [0.5, 0.5],
            [0.6, 0.5]
        ])

    def test_circle(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            units=AnnotationUnitsValues.PIXEL,
        )
        assert ann.graphic_type == GraphicTypeValues.CIRCLE
        assert np.array_equal(ann.graphic_data, self._circle)
        assert ann.units == AnnotationUnitsValues.PIXEL
        assert ann.GraphicFilled == 'N'

    def test_circle_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.CIRCLE,
                graphic_data=self._point,
                units=AnnotationUnitsValues.PIXEL,
            )

    def test_ellipse(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.ELLIPSE,
            graphic_data=self._ellipse,
            units=AnnotationUnitsValues.PIXEL,
        )
        assert ann.graphic_type == GraphicTypeValues.ELLIPSE
        assert np.array_equal(ann.graphic_data, self._ellipse)
        assert ann.units == AnnotationUnitsValues.PIXEL

    def test_ellipse_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.ELLIPSE,
                graphic_data=self._circle,
                units=AnnotationUnitsValues.PIXEL,
            )

    def test_point(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=self._point,
            units=AnnotationUnitsValues.PIXEL,
        )
        assert ann.graphic_type == GraphicTypeValues.POINT
        assert np.array_equal(ann.graphic_data, self._point)
        assert ann.units == AnnotationUnitsValues.PIXEL

    def test_point_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.POINT,
                graphic_data=self._circle,
                units=AnnotationUnitsValues.PIXEL,
            )

    def test_polyline(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=self._polyline,
            units=AnnotationUnitsValues.PIXEL,
        )
        assert ann.graphic_type == GraphicTypeValues.POLYLINE
        assert np.array_equal(ann.graphic_data, self._polyline)
        assert ann.units == AnnotationUnitsValues.PIXEL

    def test_polyline_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.POLYLINE,
                graphic_data=self._point,
                units=AnnotationUnitsValues.PIXEL,
            )

    def test_interpolated(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.INTERPOLATED,
            graphic_data=self._polyline,
            units=AnnotationUnitsValues.PIXEL,
        )
        assert ann.graphic_type == GraphicTypeValues.INTERPOLATED
        assert np.array_equal(ann.graphic_data, self._polyline)

    def test_tracking_ids(self):
        uid = UID()
        label = 'circle'
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            units=AnnotationUnitsValues.PIXEL,
            tracking_id=label,
            tracking_uid=uid
        )
        assert ann.tracking_id == label
        assert ann.tracking_uid == uid

    def test_tracking_ids_invalid(self):
        # Must provide both tracking id and uid
        uid = UID()
        with pytest.raises(TypeError):
            GraphicObject(
                graphic_type=GraphicTypeValues.CIRCLE,
                graphic_data=self._circle,
                units=AnnotationUnitsValues.PIXEL,
                tracking_uid=uid
            )

    def test_group_id(self):
        group = GraphicGroup(
            7,
            label='Group1',
            description='the first group'
        )
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            units=AnnotationUnitsValues.PIXEL,
            graphic_group=group
        )
        assert int(ann.GraphicGroupID) == 7

    def test_units(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle_display,
            units=AnnotationUnitsValues.DISPLAY
        )
        assert ann.units == AnnotationUnitsValues.DISPLAY

    def test_units_invalid(self):
        # Passing graphic data with values outside the 0 to 1 range
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.CIRCLE,
                graphic_data=self._circle,
                units=AnnotationUnitsValues.DISPLAY
            )

    def test_filled(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            units=AnnotationUnitsValues.PIXEL,
            is_filled=True
        )
        assert ann.GraphicFilled == 'Y'

    def test_filled_invalid(self):
        # Cannot have a filled polyline
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.POLYLINE,
                graphic_data=self._polyline,
                units=AnnotationUnitsValues.PIXEL,
                is_filled=True
            )


class TestTextObject(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._text = 'Look Here!'
        self._bounding_box = (5, 6, 7, 8)
        self._anchor_point = (100, 110)
        self._tracking_id = 'tracking_id_1'
        self._tracking_uid = UID()

    def test_bounding_box(self):
        text_obj = TextObject(
            text_value=self._text,
            units=AnnotationUnitsValues.PIXEL,
            bounding_box=self._bounding_box
        )
        assert text_obj.text_value == self._text
        assert text_obj.anchor_point is None
        assert text_obj.units == AnnotationUnitsValues.PIXEL
        assert text_obj.tracking_id is None
        assert text_obj.tracking_uid is None

    def test_anchor_point(self):
        text_obj = TextObject(
            text_value=self._text,
            units=AnnotationUnitsValues.PIXEL,
            anchor_point=self._anchor_point
        )
        assert text_obj.text_value == self._text
        assert text_obj.bounding_box is None
        assert text_obj.anchor_point == self._anchor_point
        assert text_obj.units == AnnotationUnitsValues.PIXEL
        assert text_obj.tracking_id is None
        assert text_obj.tracking_uid is None

    def test_bounding_box_wrong_number(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                units=AnnotationUnitsValues.PIXEL,
                bounding_box=self._anchor_point  # wrong number of elements
            )

    def test_anchor_point_wrong_number(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                units=AnnotationUnitsValues.PIXEL,
                anchor_point=self._bounding_box  # wrong number of elements
            )

    def test_tracking_uids(self):
        text_obj = TextObject(
            text_value=self._text,
            units=AnnotationUnitsValues.PIXEL,
            bounding_box=self._bounding_box,
            tracking_id=self._tracking_id,
            tracking_uid=self._tracking_uid,
        )
        assert text_obj.text_value == self._text
        assert text_obj.anchor_point is None
        assert text_obj.units == AnnotationUnitsValues.PIXEL
        assert text_obj.tracking_id == self._tracking_id
        assert text_obj.tracking_uid == self._tracking_uid

    def test_tracking_uids_only_one(self):
        with pytest.raises(TypeError):
            TextObject(
                text_value=self._text,
                units=AnnotationUnitsValues.PIXEL,
                bounding_box=self._bounding_box,
                tracking_id=self._tracking_id,
            )

    def test_bounding_box_invalid_values(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                units=AnnotationUnitsValues.PIXEL,
                bounding_box=(-1.0, 3.0, 7.8, 9.0),
            )

    def test_bounding_box_invalid_values_display(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                units=AnnotationUnitsValues.DISPLAY,
                bounding_box=(1.0, 3.0, 7.8, 9.0),
            )

    def test_bounding_box_br_invalid_for_tl(self):
        # The BR pixel is not below and to the right of the TL pixel
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                units=AnnotationUnitsValues.PIXEL,
                bounding_box=(5.0, 5.0, 4.0, 4.0),
            )

    def test_anchor_point_invalid_values(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                units=AnnotationUnitsValues.PIXEL,
                anchor_point=(-1.0, 3.0),
            )

    def test_anchor_point_invalid_values_display(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                anchor_point=(1.0, 3.0),
                units=AnnotationUnitsValues.DISPLAY
            )


class TestGraphicAnnotation(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ct_multiframe = dcmread(get_testdata_file('eCT_Supplemental.dcm'))
        self._sm_image = dcmread(data_dir / 'test_files/sm_image_dots.dcm')
        seg_path = data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm')
        self._segmentation = dcmread(seg_path)
        self._frame_number = 2
        self._frame_numbers = [1, 2]
        self._segment_number = 5
        self._segment_numbers = [7, 12]
        self._text_value = 'Look Here!'
        self._bounding_box = (5, 6, 7, 8)
        self._text_object = TextObject(
            text_value=self._text_value,
            units=AnnotationUnitsValues.PIXEL,
            bounding_box=self._bounding_box
        )
        self._circle = np.array([
            [8.0, 8.0],
            [9.0, 8.0]
        ])
        self._graphic_layer = GraphicLayer(
            layer_name='LAYER1',
            order=1,
        )
        self._graphic_layer_full = GraphicLayer(
            layer_name='LAYER1',
            order=1,
            description='The first layer',
            display_color=CIELabColor(0.0, 127.0, 127.0)
        )
        self._graphic_object = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            units=AnnotationUnitsValues.PIXEL,
        )

    def test_construction_text(self):
        ann = GraphicAnnotation(
            referenced_images=self._ct_series,
            graphic_layer=self._graphic_layer,
            text_objects=[self._text_object]
        )
        assert len(ann.ReferencedImageSequence) == len(self._ct_series)
        assert ann.GraphicLayer == self._graphic_layer.GraphicLayer
        for ref_im, ds in zip(ann.ReferencedImageSequence, self._ct_series):
            assert ref_im.ReferencedSOPClassUID == ds.SOPClassUID
            assert ref_im.ReferencedSOPInstanceUID == ds.SOPInstanceUID
        assert len(ann.TextObjectSequence) == 1
        assert not hasattr(ann, 'GraphicObjectSequence')

    def test_construction_graphic(self):
        ann = GraphicAnnotation(
            referenced_images=self._ct_series,
            graphic_layer=self._graphic_layer,
            graphic_objects=[self._graphic_object]
        )
        assert len(ann.ReferencedImageSequence) == len(self._ct_series)
        for ref_im, ds in zip(ann.ReferencedImageSequence, self._ct_series):
            assert ref_im.ReferencedSOPClassUID == ds.SOPClassUID
            assert ref_im.ReferencedSOPInstanceUID == ds.SOPInstanceUID
        assert len(ann.GraphicObjectSequence) == 1
        assert not hasattr(ann, 'TextObjectSequence')

    def test_construction_both(self):
        ann = GraphicAnnotation(
            referenced_images=self._ct_series,
            graphic_layer=self._graphic_layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object]
        )
        assert len(ann.ReferencedImageSequence) == len(self._ct_series)
        for ref_im, ds in zip(ann.ReferencedImageSequence, self._ct_series):
            assert ref_im.ReferencedSOPClassUID == ds.SOPClassUID
            assert ref_im.ReferencedSOPInstanceUID == ds.SOPInstanceUID
        assert len(ann.TextObjectSequence) == 1
        assert len(ann.GraphicObjectSequence) == 1

    def test_construction_full_graphic_layer(self):
        ann = GraphicAnnotation(
            referenced_images=self._ct_series,
            graphic_layer=self._graphic_layer_full,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object]
        )
        assert len(ann.ReferencedImageSequence) == len(self._ct_series)
        for ref_im, ds in zip(ann.ReferencedImageSequence, self._ct_series):
            assert ref_im.ReferencedSOPClassUID == ds.SOPClassUID
            assert ref_im.ReferencedSOPInstanceUID == ds.SOPInstanceUID
        assert len(ann.TextObjectSequence) == 1
        assert len(ann.GraphicObjectSequence) == 1

    def test_construction_multiframe(self):
        ann = GraphicAnnotation(
            referenced_images=[self._ct_multiframe],
            graphic_layer=self._graphic_layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object]
        )
        assert len(ann.ReferencedImageSequence) == 1
        ref_im = ann.ReferencedImageSequence[0]
        assert ref_im.ReferencedSOPClassUID == self._ct_multiframe.SOPClassUID
        sop_ins_uid = self._ct_multiframe.SOPInstanceUID
        assert ref_im.ReferencedSOPInstanceUID == sop_ins_uid
        assert len(ann.TextObjectSequence) == 1
        assert len(ann.GraphicObjectSequence) == 1

    def test_construction_multiple_multiframe(self):
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=[self._ct_multiframe, self._ct_multiframe],
                graphic_objects=[self._graphic_object],
                graphic_layer=self._graphic_layer,
                text_objects=[self._text_object]
            )

    def test_construction_frame_number(self):
        ann = GraphicAnnotation(
            referenced_images=[self._ct_multiframe],
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object],
            graphic_layer=self._graphic_layer,
            referenced_frame_number=self._frame_number
        )
        assert len(ann.ReferencedImageSequence) == 1
        ref_im = ann.ReferencedImageSequence[0]
        assert ref_im.ReferencedSOPClassUID == self._ct_multiframe.SOPClassUID
        sop_ins_uid = self._ct_multiframe.SOPInstanceUID
        assert ref_im.ReferencedSOPInstanceUID == sop_ins_uid
        assert ref_im.ReferencedFrameNumber == self._frame_number
        assert len(ann.TextObjectSequence) == 1
        assert len(ann.GraphicObjectSequence) == 1

    def test_construction_frame_numbers(self):
        ann = GraphicAnnotation(
            referenced_images=[self._ct_multiframe],
            graphic_layer=self._graphic_layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object],
            referenced_frame_number=self._frame_numbers
        )
        assert len(ann.ReferencedImageSequence) == 1
        ref_im = ann.ReferencedImageSequence[0]
        assert ref_im.ReferencedSOPClassUID == self._ct_multiframe.SOPClassUID
        sop_ins_uid = self._ct_multiframe.SOPInstanceUID
        assert ref_im.ReferencedSOPInstanceUID == sop_ins_uid
        assert ref_im.ReferencedFrameNumber == self._frame_numbers
        assert len(ann.TextObjectSequence) == 1
        assert len(ann.GraphicObjectSequence) == 1

    def test_construction_frame_number_single_frame(self):
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=self._ct_series,
                graphic_layer=self._graphic_layer,
                graphic_objects=[self._graphic_object],
                text_objects=[self._text_object],
                referenced_frame_number=self._frame_number
            )

    def test_construction_frame_number_invalid(self):
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=[self._ct_multiframe],
                graphic_layer=self._graphic_layer,
                graphic_objects=[self._graphic_object],
                text_objects=[self._text_object],
                referenced_frame_number=self._ct_multiframe.NumberOfFrames + 1
            )

    def test_construction_frame_numbers_invalid(self):
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=[self._ct_multiframe],
                graphic_layer=self._graphic_layer,
                graphic_objects=[self._graphic_object],
                text_objects=[self._text_object],
                referenced_frame_number=[
                    1,
                    self._ct_multiframe.NumberOfFrames + 1
                ]
            )

    def test_construction_segment_number(self):
        ann = GraphicAnnotation(
            referenced_images=[self._segmentation],
            graphic_layer=self._graphic_layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object],
            referenced_segment_number=self._segment_number
        )
        assert len(ann.ReferencedImageSequence) == 1
        ref_im = ann.ReferencedImageSequence[0]
        assert ref_im.ReferencedSOPClassUID == self._segmentation.SOPClassUID
        sop_ins_uid = self._segmentation.SOPInstanceUID
        assert ref_im.ReferencedSOPInstanceUID == sop_ins_uid
        assert ref_im.ReferencedSegmentNumber == self._segment_number
        assert len(ann.TextObjectSequence) == 1
        assert len(ann.GraphicObjectSequence) == 1

    def test_construction_segment_numbers(self):
        ann = GraphicAnnotation(
            referenced_images=[self._segmentation],
            graphic_layer=self._graphic_layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object],
            referenced_segment_number=self._segment_numbers
        )
        assert len(ann.ReferencedImageSequence) == 1
        ref_im = ann.ReferencedImageSequence[0]
        assert ref_im.ReferencedSOPClassUID == self._segmentation.SOPClassUID
        sop_ins_uid = self._segmentation.SOPInstanceUID
        assert ref_im.ReferencedSOPInstanceUID == sop_ins_uid
        assert ref_im.ReferencedSegmentNumber == self._segment_numbers
        assert len(ann.TextObjectSequence) == 1
        assert len(ann.GraphicObjectSequence) == 1

    def test_construction_segment_number_single_segment(self):
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=self._ct_series,
                graphic_layer=self._graphic_layer,
                graphic_objects=[self._graphic_object],
                text_objects=[self._text_object],
                referenced_segment_number=self._segment_number
            )

    def test_construction_segment_number_invalid(self):
        seg_num = len(self._segmentation.SegmentSequence) + 1
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=[self._segmentation],
                graphic_layer=self._graphic_layer,
                graphic_objects=[self._graphic_object],
                text_objects=[self._text_object],
                referenced_segment_number=seg_num
            )

    def test_construction_segment_numbers_invalid(self):
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=[self._segmentation],
                graphic_layer=self._graphic_layer,
                graphic_objects=[self._graphic_object],
                text_objects=[self._text_object],
                referenced_segment_number=[
                    1,
                    len(self._segmentation.SegmentSequence) + 1
                ]
            )

    def test_construction_invalid_use_of_matrix(self):
        go = GraphicObject(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=np.array([[1.0, 1.0]]),
            units=AnnotationUnitsValues.MATRIX,
        )
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=self._ct_series,
                graphic_layer=self._graphic_layer,
                graphic_objects=[go]
            )

    def test_construction_out_of_bounds_pixel(self):
        cols = self._ct_series[0].Columns
        go = GraphicObject(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=np.array([[cols + 1.0, 1.0]]),
            units=AnnotationUnitsValues.PIXEL,
        )
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=self._ct_series,
                graphic_layer=self._graphic_layer,
                graphic_objects=[go]
            )

    def test_construction_out_of_bounds_matrix(self):
        cols = self._sm_image.TotalPixelMatrixColumns
        go = GraphicObject(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=np.array([[cols + 1.0, 1.0]]),
            units=AnnotationUnitsValues.MATRIX,
        )
        with pytest.raises(ValueError):
            GraphicAnnotation(
                referenced_images=[self._sm_image],
                graphic_layer=self._graphic_layer,
                graphic_objects=[go]
            )


class TestAdvancedBlending(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._sm_image = dcmread(
            data_dir / 'test_files/sm_image_grayscale.dcm'
        )
        self._sm_image_reversed = dcmread(
            data_dir / 'test_files/sm_image_grayscale_reversed.dcm'
        )

    def test_construction(self):
        ref_images = [self._sm_image, self._sm_image_reversed]
        window_width = 24.
        window_center = 12.

        ds = AdvancedBlending(
            referenced_images=ref_images,
            blending_input_number=1,
            voi_lut_transformations=[
                SoftcopyVOILUTTransformation(
                    window_center=window_center,
                    window_width=window_width
                )
            ],
            palette_color_lut_transformation=PaletteColorLUTTransformation(
                red_lut=SegmentedPaletteColorLUT(
                    first_mapped_value=0,
                    segmented_lut_data=np.array(
                        [
                            0, 1, 0,
                            1, 255, 0,
                        ],
                        dtype=np.uint16
                    ),
                    color='red'
                ),
                green_lut=SegmentedPaletteColorLUT(
                    first_mapped_value=0,
                    segmented_lut_data=np.array(
                        [
                            0, 1, 0,
                            1, 255, 0,
                        ],
                        dtype=np.uint16
                    ),
                    color='green'
                ),
                blue_lut=SegmentedPaletteColorLUT(
                    first_mapped_value=0,
                    segmented_lut_data=np.array(
                        [
                            0, 1, 0,
                            1, 255, 255,
                        ],
                        dtype=np.uint16
                    ),
                    color='blue'
                )
            )
        )

        assert isinstance(ds, Dataset)
        assert ds.StudyInstanceUID == ref_images[0].StudyInstanceUID
        assert ds.SeriesInstanceUID == ref_images[0].SeriesInstanceUID
        assert len(ds.ReferencedImageSequence) == 2
        for i in range(len(ds.ReferencedImageSequence)):
            ref_im = ref_images[i]
            ref_item = ds.ReferencedImageSequence[i]
            assert ref_item.ReferencedSOPInstanceUID == ref_im.SOPInstanceUID
        assert ds.BlendingInputNumber == 1

        assert len(ds.SoftcopyVOILUTSequence) == 1
        item = ds.SoftcopyVOILUTSequence[0]
        assert item.WindowWidth == window_width
        assert item.WindowCenter == window_center
        assert not hasattr(item, 'VOILUTSequence')

        assert len(ds.PaletteColorLookupTableSequence) == 1
        item = ds.PaletteColorLookupTableSequence[0]
        assert len(item.RedPaletteColorLookupTableDescriptor) == 3
        assert item.RedPaletteColorLookupTableDescriptor[0] == 256
        assert item.RedPaletteColorLookupTableDescriptor[1] == 0
        assert item.RedPaletteColorLookupTableDescriptor[2] == 16
        assert len(item.GreenPaletteColorLookupTableDescriptor) == 3
        assert item.GreenPaletteColorLookupTableDescriptor[0] == 256
        assert item.GreenPaletteColorLookupTableDescriptor[1] == 0
        assert item.GreenPaletteColorLookupTableDescriptor[2] == 16
        assert len(item.BluePaletteColorLookupTableDescriptor) == 3
        assert item.BluePaletteColorLookupTableDescriptor[0] == 256
        assert item.BluePaletteColorLookupTableDescriptor[1] == 0
        assert item.BluePaletteColorLookupTableDescriptor[2] == 16
        assert len(item.SegmentedRedPaletteColorLookupTableData) == 12
        assert len(item.SegmentedGreenPaletteColorLookupTableData) == 12
        assert len(item.SegmentedBluePaletteColorLookupTableData) == 12

        assert not hasattr(ds, 'ContentDescription')
        assert not hasattr(ds, 'ConceptNameCodeSequence')
        assert not hasattr(ds, 'ThresholdSequence')
        assert not hasattr(ds, 'RescaleSlope')
        assert not hasattr(ds, 'RescaleIntercept')
        assert not hasattr(ds, 'ModalityLUTSequence')


class TestBlendingDisplay(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._sm_image = dcmread(
            data_dir / 'test_files/sm_image_grayscale.dcm'
        )

    def test_construction_equal(self):
        ds = BlendingDisplay(
            blending_mode=BlendingModeValues.EQUAL,
            blending_display_inputs=[
                BlendingDisplayInput(
                    blending_input_number=1
                )
            ]
        )
        assert isinstance(ds, Dataset)
        assert ds.BlendingMode == 'EQUAL'
        assert not hasattr(ds, 'RelativeOpacity')
        assert len(ds.BlendingDisplayInputSequence) == 1

    def test_construction_forebround(self):
        ds = BlendingDisplay(
            blending_mode=BlendingModeValues.FOREGROUND,
            blending_display_inputs=[
                BlendingDisplayInput(
                    blending_input_number=1
                ),
                BlendingDisplayInput(
                    blending_input_number=2
                )
            ],
            relative_opacity=0.5
        )
        assert isinstance(ds, Dataset)
        assert ds.BlendingMode == 'FOREGROUND'
        assert ds.RelativeOpacity == 0.5
        assert len(ds.BlendingDisplayInputSequence) == 2

    def test_construction_equal_wrong_blending_display_input_type(self):
        with pytest.raises(TypeError):
            BlendingDisplay(
                blending_mode=BlendingModeValues.EQUAL,
                blending_display_inputs=BlendingDisplayInput(
                    blending_input_number=1
                )
            )

    def test_construction_equal_wrong_blending_display_input_value(self):
        with pytest.raises(ValueError):
            BlendingDisplay(
                blending_mode=BlendingModeValues.EQUAL,
                blending_display_inputs=[]
            )

    def test_construction_foreground_wrong_relative_opacity_type(self):
        with pytest.raises(TypeError):
            BlendingDisplay(
                blending_mode=BlendingModeValues.FOREGROUND,
                blending_display_inputs=[
                    BlendingDisplayInput(
                        blending_input_number=1
                    ),
                    BlendingDisplayInput(
                        blending_input_number=2
                    )
                ]
            )

    def test_construction_foreground_wrong_blending_display_input_value(self):
        with pytest.raises(ValueError):
            BlendingDisplay(
                blending_mode=BlendingModeValues.FOREGROUND,
                blending_display_inputs=[
                    BlendingDisplayInput(
                        blending_input_number=1
                    )
                ],
                relative_opacity=1.
            )


class TestXSoftcopyPresentationState(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        self._test_dir = file_path.parent.parent.joinpath(
            'data',
            'test_files'
        )
        self._series_uid = UID()
        self._sop_uid = UID()
        self._ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._ct_multiframe = dcmread(get_testdata_file('eCT_Supplemental.dcm'))
        self._seg = dcmread(
            'data/test_files/seg_image_ct_binary_overlap.dcm'
        )
        self._sm_image = dcmread(self._test_dir / 'sm_image_dots.dcm')
        self._text_value = 'Look Here!'
        self._bounding_box = (5, 6, 7, 8)
        self._group = GraphicGroup(
            1,
            'Group1',
            'Description of Group 1'
        )
        self._other_group = GraphicGroup(
            43,
            'Group43',
            'Description of Group 43'
        )
        self._text_object = TextObject(
            text_value=self._text_value,
            units=AnnotationUnitsValues.PIXEL,
            bounding_box=self._bounding_box
        )
        self._circle = np.array([
            [8.0, 8.0],
            [9.0, 8.0]
        ])
        self._graphic_object = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            units=AnnotationUnitsValues.PIXEL,
        )
        self._display_color = CIELabColor(0.0, 127.0, 127.0)
        self._layer = GraphicLayer(
            layer_name='LAYER1',
            order=1,
            description='Basic layer',
            display_color=self._display_color
        )
        self._other_layer = GraphicLayer(
            layer_name='LAYER2',
            order=3,
            description='Another Basic layer',
            display_color=self._display_color
        )
        self._ann_ct = GraphicAnnotation(
            referenced_images=self._ct_series,
            graphic_layer=self._layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object]
        )
        self._ann_sm = GraphicAnnotation(
            referenced_images=[self._sm_image],
            graphic_layer=self._layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object]
        )

        # Same thing, but with object belonging to groups
        self._text_object_grp = TextObject(
            text_value=self._text_value,
            units=AnnotationUnitsValues.PIXEL,
            bounding_box=self._bounding_box,
            graphic_group=self._group,
        )
        self._graphic_object_grp = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            units=AnnotationUnitsValues.PIXEL,
            graphic_group=self._group,
        )
        self._layer_grp = GraphicLayer(
            layer_name='LAYER1',
            order=1,
            description='Basic layer',
            display_color=self._display_color
        )
        self._ann_grp = GraphicAnnotation(
            referenced_images=self._ct_series,
            graphic_layer=self._layer_grp,
            graphic_objects=[self._graphic_object_grp],
            text_objects=[self._text_object_grp]
        )

        self._creator_id = ContentCreatorIdentificationCodeSequence(
            person_identification_codes=[codes.DCM.Person],
            institution_name='MGH'
        )

        self._modality_lut = ModalityLUT(
            lut_type=RescaleTypeValues.HU,
            first_mapped_value=0,
            lut_data=np.arange(256, dtype=np.uint16)
        )

        self._presentation_lut = PresentationLUT(
            lut_data=np.arange(10, 100, dtype=np.uint16),
            first_mapped_value=0
        )

    def test_construction(self):
        pr = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
        )
        assert pr.SeriesInstanceUID == self._series_uid
        assert pr.SOPInstanceUID == self._sop_uid
        assert pr.SeriesNumber == 123
        assert pr.InstanceNumber == 456
        assert pr.Manufacturer == 'Foo Corp.'
        assert pr.ManufacturerModelName == 'Bar, Mark 2'
        assert pr.SoftwareVersions == '0.0.1'
        assert pr.DeviceSerialNumber == '12345'
        assert pr.ContentLabel == 'DOODLE'
        assert len(pr.ReferencedSeriesSequence) == 1
        assert len(pr.GraphicLayerSequence) == 1
        assert pr.InstitutionName == 'MGH'
        assert pr.InstitutionalDepartmentName == 'Radiology'
        assert pr.ContentCreatorName == 'Doe^John'
        assert pr.ConceptNameCodeSequence[0].CodeValue == 'PR'
        assert not hasattr(pr, 'ModalityLUTSequence')
        assert hasattr(pr, 'RescaleSlope')
        assert hasattr(pr, 'RescaleIntercept')
        assert hasattr(pr, 'RescaleType')
        assert pr.RescaleIntercept == self._ct_series[0].RescaleIntercept
        assert pr.RescaleSlope == self._ct_series[0].RescaleSlope

    def test_construction_with_modality_rescale(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            modality_lut_transformation=ModalityLUTTransformation(
                rescale_intercept=1024.0,
                rescale_slope=2.0,
                rescale_type='HU',
            )
        )
        assert gsps.SeriesInstanceUID == self._series_uid
        assert gsps.SOPInstanceUID == self._sop_uid
        assert gsps.SeriesNumber == 123
        assert gsps.InstanceNumber == 456
        assert gsps.Manufacturer == 'Foo Corp.'
        assert gsps.ManufacturerModelName == 'Bar, Mark 2'
        assert gsps.SoftwareVersions == '0.0.1'
        assert gsps.DeviceSerialNumber == '12345'
        assert gsps.ContentLabel == 'DOODLE'
        assert len(gsps.ReferencedSeriesSequence) == 1
        assert len(gsps.GraphicLayerSequence) == 1
        assert gsps.InstitutionName == 'MGH'
        assert gsps.InstitutionalDepartmentName == 'Radiology'
        assert gsps.ContentCreatorName == 'Doe^John'
        assert gsps.ConceptNameCodeSequence[0].CodeValue == 'PR'
        assert gsps.RescaleSlope == 2
        assert gsps.RescaleIntercept == 1024
        assert gsps.RescaleType == 'HU'
        assert not hasattr(gsps, 'ModalityLUTSequence')

    def test_construction_with_modality_lut(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            modality_lut_transformation=ModalityLUTTransformation(
                modality_lut=self._modality_lut,
            )
        )
        assert gsps.SeriesInstanceUID == self._series_uid
        assert gsps.SOPInstanceUID == self._sop_uid
        assert gsps.SeriesNumber == 123
        assert gsps.InstanceNumber == 456
        assert gsps.Manufacturer == 'Foo Corp.'
        assert gsps.ManufacturerModelName == 'Bar, Mark 2'
        assert gsps.SoftwareVersions == '0.0.1'
        assert gsps.DeviceSerialNumber == '12345'
        assert gsps.ContentLabel == 'DOODLE'
        assert len(gsps.ReferencedSeriesSequence) == 1
        assert len(gsps.GraphicLayerSequence) == 1
        assert gsps.InstitutionName == 'MGH'
        assert gsps.InstitutionalDepartmentName == 'Radiology'
        assert gsps.ContentCreatorName == 'Doe^John'
        assert gsps.ConceptNameCodeSequence[0].CodeValue == 'PR'
        assert not hasattr(gsps, 'RescaleSlope')
        assert not hasattr(gsps, 'RescaleIntercept')
        assert not hasattr(gsps, 'RescaleType')
        assert len(gsps.ModalityLUTSequence) == 1

    def test_construction_with_copy_modality_lut(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert gsps.RescaleIntercept == self._ct_series[0].RescaleIntercept
        assert gsps.RescaleSlope == self._ct_series[0].RescaleSlope
        assert gsps.RescaleType == 'HU'

    def test_construction_with_copy_modality_lut_multiframe(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=[self._ct_multiframe],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        shared_grp = self._ct_multiframe.SharedFunctionalGroupsSequence[0]
        trans_seq = shared_grp.PixelValueTransformationSequence[0]
        assert gsps.RescaleIntercept == trans_seq.RescaleIntercept
        assert gsps.RescaleSlope == trans_seq.RescaleSlope
        assert gsps.RescaleType == trans_seq.RescaleType

    def test_construction_with_voi_lut(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            voi_lut_transformations=[
                SoftcopyVOILUTTransformation(
                    window_center=40.0,
                    window_width=400.0,
                    window_explanation='Soft Tissue Window'
                )
            ]

        )
        assert len(gsps.SoftcopyVOILUTSequence) == 1
        assert hasattr(gsps.SoftcopyVOILUTSequence[0], 'WindowWidth')
        assert hasattr(gsps.SoftcopyVOILUTSequence[0], 'WindowCenter')

    def test_construction_with_copy_voi_lut(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert len(gsps.SoftcopyVOILUTSequence) == 1
        expected_width = self._ct_series[0].WindowWidth
        expected_center = self._ct_series[0].WindowCenter
        assert gsps.SoftcopyVOILUTSequence[0].WindowWidth == expected_width
        assert gsps.SoftcopyVOILUTSequence[0].WindowCenter == expected_center
        assert not hasattr(
            gsps.SoftcopyVOILUTSequence[0],
            'ReferencedImageSequence'
        )

    def test_construction_with_copy_voi_lut_different(self):
        new_dcm = deepcopy(self._ct_series[0])
        new_dcm.WindowCenter = 40.0
        new_dcm.WindowWidth = 400.0
        series = [new_dcm] + self._ct_series[1:]
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert len(gsps.SoftcopyVOILUTSequence) == 2
        ref_ims_0 = gsps.SoftcopyVOILUTSequence[0].ReferencedImageSequence
        assert len(ref_ims_0) == 1
        assert ref_ims_0[0].ReferencedSOPInstanceUID == new_dcm.SOPInstanceUID
        ref_ims_1 = gsps.SoftcopyVOILUTSequence[1].ReferencedImageSequence
        for im, ref_im in zip(self._ct_series[1:], ref_ims_1):
            assert ref_im.ReferencedSOPInstanceUID == im.SOPInstanceUID
        assert len(ref_ims_1) == 3
        assert len(gsps.SoftcopyVOILUTSequence[1].ReferencedImageSequence) == 3
        expected_width = new_dcm.WindowWidth
        expected_center = new_dcm.WindowCenter
        assert gsps.SoftcopyVOILUTSequence[0].WindowWidth == expected_width
        assert gsps.SoftcopyVOILUTSequence[0].WindowCenter == expected_center
        expected_width = self._ct_series[1].WindowWidth
        expected_center = self._ct_series[1].WindowCenter
        assert gsps.SoftcopyVOILUTSequence[1].WindowWidth == expected_width
        assert gsps.SoftcopyVOILUTSequence[1].WindowCenter == expected_center

    def test_construction_with_copy_voi_lut_multival(self):
        new_series = []
        widths = [40.0, 20.0]
        centers = [400.0, 600.0]
        for dcm in self._ct_series:
            new_dcm = deepcopy(dcm)
            new_dcm.WindowCenter = centers
            new_dcm.WindowWidth = widths
            new_series.append(new_dcm)
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=new_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert len(gsps.SoftcopyVOILUTSequence) == 1
        assert gsps.SoftcopyVOILUTSequence[0].WindowWidth == widths
        assert gsps.SoftcopyVOILUTSequence[0].WindowCenter == centers

    def test_construction_with_copy_voi_lut_multival_with_expl(self):
        new_series = []
        widths = [40.0, 20.0]
        centers = [400.0, 600.0]
        exps = ['WINDOW1', 'WINDOW2']
        for dcm in self._ct_series:
            new_dcm = deepcopy(dcm)
            new_dcm.WindowCenter = centers
            new_dcm.WindowWidth = widths
            new_dcm.WindowCenterWidthExplanation = exps
            new_series.append(new_dcm)
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=new_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert len(gsps.SoftcopyVOILUTSequence) == 1
        assert gsps.SoftcopyVOILUTSequence[0].WindowWidth == widths
        assert gsps.SoftcopyVOILUTSequence[0].WindowCenter == centers
        assert (
            gsps.SoftcopyVOILUTSequence[0].WindowCenterWidthExplanation == exps
        )

    def test_construction_with_copy_voi_lut_empty(self):
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        # This file has no VOI LUT info
        ct_im = dcmread(data_dir / 'test_files/ct_image.dcm')
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=[ct_im],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert not hasattr(gsps, 'SoftcopyVOILUTSequence')

    def test_construction_with_copy_voi_lut_data(self):
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        # This file has no VOI LUT info
        ct_im = dcmread(data_dir / 'test_files/ct_image.dcm')
        # Construct test LUT data
        luts = [
            VOILUT(0, np.array([2, 3, 4], np.uint16)),
            VOILUT(0, np.array([5, 6, 7], np.uint16)),
        ]
        ct_im.VOILUTSequence = luts
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=[ct_im],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert len(gsps.SoftcopyVOILUTSequence) == 1
        assert len(gsps.SoftcopyVOILUTSequence[0].VOILUTSequence) == 2

    def test_construction_with_copy_voi_lut_multiframe(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=[self._ct_multiframe],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        shared_grp = self._ct_multiframe.SharedFunctionalGroupsSequence[0]
        voi_seq = shared_grp.FrameVOILUTSequence[0]
        assert len(gsps.SoftcopyVOILUTSequence) == 1
        expected_width = voi_seq.WindowWidth
        expected_center = voi_seq.WindowCenter
        assert gsps.SoftcopyVOILUTSequence[0].WindowWidth == expected_width
        assert gsps.SoftcopyVOILUTSequence[0].WindowCenter == expected_center
        assert not hasattr(
            gsps.SoftcopyVOILUTSequence[0],
            'ReferencedImageSequence'
        )

    def test_construction_with_copy_voi_lut_multiframe_perframe(self):
        # Create a multiframe image with different VOI LUTs per frame
        edited_multiframe = deepcopy(self._ct_multiframe)
        delattr(
            edited_multiframe.SharedFunctionalGroupsSequence[0],
            'FrameVOILUTSequence'
        )
        new_voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
        )
        new_voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
        )
        edited_multiframe.PerFrameFunctionalGroupsSequence[0]. \
            FrameVOILUTSequence = [new_voi_lut_0]
        edited_multiframe.PerFrameFunctionalGroupsSequence[1]. \
            FrameVOILUTSequence = [new_voi_lut_1]

        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=[edited_multiframe],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John'
        )
        assert len(gsps.SoftcopyVOILUTSequence) == 2

        expected_width = new_voi_lut_0.WindowWidth
        expected_center = new_voi_lut_0.WindowCenter
        assert gsps.SoftcopyVOILUTSequence[0].WindowWidth == expected_width
        assert gsps.SoftcopyVOILUTSequence[0].WindowCenter == expected_center
        ref_im = gsps.SoftcopyVOILUTSequence[0].ReferencedImageSequence[0]
        assert ref_im.ReferencedFrameNumber == 1

        expected_width = new_voi_lut_1.WindowWidth
        expected_center = new_voi_lut_1.WindowCenter
        assert gsps.SoftcopyVOILUTSequence[1].WindowWidth == expected_width
        assert gsps.SoftcopyVOILUTSequence[1].WindowCenter == expected_center
        ref_im = gsps.SoftcopyVOILUTSequence[1].ReferencedImageSequence[0]
        assert ref_im.ReferencedFrameNumber == 2

    def test_construction_with_voi_lut_missing_references(self):
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series[:2],  # missing images
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                voi_lut_transformations=[
                    SoftcopyVOILUTTransformation(
                        window_center=40.0,
                        window_width=400.0,
                        window_explanation='Soft Tissue Window',
                        referenced_images=ReferencedImageSequence(
                            self._ct_series
                        )
                    )
                ]
                # references missing images ^
            )

    def test_construction_with_voi_lut_with_ref_ims(self):
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ReferencedImageSequence(self._ct_series[:1])
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ReferencedImageSequence(self._ct_series[1:])
        )
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            voi_lut_transformations=[voi_lut_0, voi_lut_1],
        )
        assert len(gsps.SoftcopyVOILUTSequence) == 2
        assert hasattr(
            gsps.SoftcopyVOILUTSequence[0],
            'ReferencedImageSequence'
        )

    def test_construction_with_voi_lut_two_vois_for_image(self):
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ReferencedImageSequence(self._ct_series)
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ReferencedImageSequence(self._ct_series[2:3])
        )  # overlap with images in previous VOILUT
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series,
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._layer],
                graphic_annotations=[self._ann_ct],
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                voi_lut_transformations=[voi_lut_0, voi_lut_1],
            )

    def test_construction_with_multiple_voi_luts_with_no_ref_ims(self):
        # Two VOI LUT sequences, one of which does not specify images
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ReferencedImageSequence(self._ct_series[2:3])
        )
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series,
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._layer],
                graphic_annotations=[self._ann_ct],
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                voi_lut_transformations=[voi_lut_0, voi_lut_1],
            )

    def test_construction_with_softcopy_voi_lut_with_ref_frames(self):
        ref_ims_0 = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],
            referenced_frame_number=[1]
        )
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ref_ims_0,
        )
        ref_ims_1 = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],
            referenced_frame_number=[2]
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ref_ims_1,
        )
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=[self._ct_multiframe],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            voi_lut_transformations=[voi_lut_0, voi_lut_1],
        )
        ref_im_seq_0 = gsps.SoftcopyVOILUTSequence[0].ReferencedImageSequence
        assert ref_im_seq_0[0].ReferencedFrameNumber == 1
        ref_im_seq_1 = gsps.SoftcopyVOILUTSequence[1].ReferencedImageSequence
        assert ref_im_seq_1[0].ReferencedFrameNumber == 2

    def test_construction_with_softcopy_voi_lut_with_duplicate_frames(self):
        ref_ims_0 = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],
            referenced_frame_number=[1]
        )
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ref_ims_0,
        )
        ref_ims_1 = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],
            referenced_frame_number=[1, 2]  # overlap with previous frames
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ref_ims_1,
        )
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=[self._ct_multiframe],
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                voi_lut_transformations=[voi_lut_0, voi_lut_1],
            )

    def test_construction_with_softcopy_voi_lut_with_duplicate_frames_2(self):
        ref_ims_0 = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],  # applies to all frames
        )
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ref_ims_0,
        )
        ref_ims_1 = ReferencedImageSequence(
            referenced_images=[self._ct_multiframe],
            referenced_frame_number=[1, 2]  # overlap with previous frames
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ref_ims_1,
        )
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=[self._ct_multiframe],
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                voi_lut_transformations=[voi_lut_0, voi_lut_1],
            )

    def test_construction_with_softcopy_voi_lut_with_ref_segments(self):
        ref_ims_0 = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=[1]
        )
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ref_ims_0,
        )
        ref_ims_1 = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=[2]
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ref_ims_1,
        )
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=[self._seg],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            voi_lut_transformations=[voi_lut_0, voi_lut_1],
        )
        ref_im_seq_0 = gsps.SoftcopyVOILUTSequence[0].ReferencedImageSequence
        assert ref_im_seq_0[0].ReferencedSegmentNumber == 1
        ref_im_seq_1 = gsps.SoftcopyVOILUTSequence[1].ReferencedImageSequence
        assert ref_im_seq_1[0].ReferencedSegmentNumber == 2

    def test_construction_with_softcopy_voi_lut_with_duplicate_segments(self):
        ref_ims_0 = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=[1]
        )
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ref_ims_0,
        )
        ref_ims_1 = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=[1, 2]  # overlap with previous
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ref_ims_1,
        )
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=[self._seg],
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                voi_lut_transformations=[voi_lut_0, voi_lut_1],
            )

    def test_construction_with_softcopy_voi_lut_with_duplicate_segments_2(self):
        ref_ims_0 = ReferencedImageSequence(
            referenced_images=[self._seg],  # applies to all segments
        )
        voi_lut_0 = SoftcopyVOILUTTransformation(
            window_center=1600.0,
            window_width=2000.0,
            referenced_images=ref_ims_0,
        )
        ref_ims_1 = ReferencedImageSequence(
            referenced_images=[self._seg],
            referenced_segment_number=[1, 2]  # overlap with previous
        )
        voi_lut_1 = SoftcopyVOILUTTransformation(
            window_center=40.0,
            window_width=400.0,
            referenced_images=ref_ims_1,
        )
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=[self._seg],
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                voi_lut_transformations=[voi_lut_0, voi_lut_1],
            )

    def test_construction_with_presentation_lut_shape(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            presentation_lut_transformation=PresentationLUTTransformation(
                presentation_lut_shape=PresentationLUTShapeValues.INVERSE
            )
        )
        assert gsps.PresentationLUTShape == 'INVERSE'
        assert not hasattr(gsps, 'PresentationLUTSequence')

    def test_construction_with_presentation_lut(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            presentation_lut_transformation=PresentationLUTTransformation(
                presentation_lut=self._presentation_lut
            )
        )
        assert len(gsps.PresentationLUTSequence) == 1
        assert not hasattr(gsps, 'PresentationLUTShape')

    def test_construction_with_presentation_both(self):
        with pytest.raises(TypeError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series,
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._layer],
                graphic_annotations=[self._ann_ct],
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
                presentation_lut_transformation=PresentationLUTTransformation(
                    presentation_lut=self._presentation_lut,
                    presentation_lut_shape=PresentationLUTShapeValues.INVERSE,
                )
            )

    def test_construction_color(self):
        pr = ColorSoftcopyPresentationState(
            referenced_images=[self._sm_image],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_sm],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
        )
        assert pr.SeriesInstanceUID == self._series_uid
        assert pr.SOPInstanceUID == self._sop_uid
        assert pr.SeriesNumber == 123
        assert pr.InstanceNumber == 456
        assert pr.Manufacturer == 'Foo Corp.'
        assert pr.ManufacturerModelName == 'Bar, Mark 2'
        assert pr.SoftwareVersions == '0.0.1'
        assert pr.DeviceSerialNumber == '12345'
        assert pr.ContentLabel == 'DOODLE'
        assert len(pr.ReferencedSeriesSequence) == 1
        assert len(pr.GraphicLayerSequence) == 1
        assert pr.InstitutionName == 'MGH'
        assert pr.InstitutionalDepartmentName == 'Radiology'
        assert pr.ContentCreatorName == 'Doe^John'
        assert pr.ConceptNameCodeSequence[0].CodeValue == 'PR'
        assert not hasattr(pr, 'ModalityLUTSequence')
        assert not hasattr(pr, 'RescaleSlope')
        assert not hasattr(pr, 'RescaleIntercept')
        assert not hasattr(pr, 'RescaleType')

        # Write out dataset and test icc profile works as expected
        with BytesIO() as buf:
            pr.save_as(buf)
            buf.seek(0)
            reread = dcmread(buf)

        # A basic check that the profile was read correctly
        original_profile = self._sm_image.OpticalPathSequence[0].ICCProfile
        assert reread.ICCProfile == original_profile

    def test_construction_gray_with_color_images(self):
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=[self._sm_image],  # color image
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._layer],
                graphic_annotations=[self._ann_sm],
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
            )

    def test_construction_color_with_gray_images(self):
        with pytest.raises(ValueError):
            ColorSoftcopyPresentationState(
                referenced_images=self._ct_series,  # grayscale images
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._layer],
                graphic_annotations=[self._ann_ct],
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
            )

    def test_construction_icc_profile(self):
        icc_profile = pkgutil.get_data(
            'highdicom',
            '_icc_profiles/sRGB_v4_ICC_preference.icc'
        )

        pr = ColorSoftcopyPresentationState(
            referenced_images=[self._sm_image],
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_sm],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            icc_profile=icc_profile
        )
        assert hasattr(pr, 'ICCProfile')

        # Write out dataset and test it works as expected
        with BytesIO() as buf:
            pr.save_as(buf)
            buf.seek(0)
            reread = dcmread(buf)

        # A basic check that the profile was read correctly
        image_profile = ImageCmsProfile(BytesIO(reread.ICCProfile))
        assert (
            image_profile.profile.profile_description ==
            'sRGB v4 ICC preference perceptual intent beta'
        )

    def test_construction_creator_id(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            content_creator_identification=self._creator_id
        )
        assert len(gsps.ContentCreatorIdentificationCodeSequence) == 1

    def test_construction_with_group(self):
        gsps = GrayscaleSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer_grp],
            graphic_groups=[self._group],
            graphic_annotations=[self._ann_grp],
        )
        assert gsps.SeriesInstanceUID == self._series_uid
        assert gsps.SOPInstanceUID == self._sop_uid
        assert gsps.SeriesNumber == 123
        assert gsps.InstanceNumber == 456
        assert gsps.Manufacturer == 'Foo Corp.'
        assert gsps.ManufacturerModelName == 'Bar, Mark 2'
        assert gsps.SoftwareVersions == '0.0.1'
        assert gsps.DeviceSerialNumber == '12345'
        assert gsps.ContentLabel == 'DOODLE'
        assert len(gsps.ReferencedSeriesSequence) == 1
        assert len(gsps.GraphicLayerSequence) == 1

    def test_construction_with_images_missing(self):
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series[1:],  # missing image!
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._layer],
                graphic_annotations=[self._ann_ct],
            )

    def test_construction_with_duplicate_layers(self):
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series,
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._layer, self._layer],  # duplicate
                graphic_annotations=[self._ann_ct],
            )

    def test_construction_with_group_missing(self):
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series,
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_annotations=[self._ann_grp],
                graphic_layers=[self._layer_grp],
                graphic_groups=[self._other_group]  # wrong group for objects!
            )

    def test_construction_with_duplicate_group(self):
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series,
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_annotations=[self._ann_grp],
                graphic_layers=[self._layer_grp],
                graphic_groups=[self._group, self._group]  # duplicates
            )

    def test_construction_with_missing_layer(self):
        with pytest.raises(ValueError):
            GrayscaleSoftcopyPresentationState(
                referenced_images=self._ct_series,
                series_instance_uid=self._series_uid,
                series_number=123,
                sop_instance_uid=self._sop_uid,
                instance_number=456,
                manufacturer='Foo Corp.',
                manufacturer_model_name='Bar, Mark 2',
                software_versions='0.0.1',
                device_serial_number='12345',
                content_label='DOODLE',
                graphic_layers=[self._other_layer],  # wrong layer!
                graphic_annotations=[self._ann_ct],
                concept_name=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
            )

    def test_construction_palette_color_lut_transformation(self):
        r_lut_data = np.arange(10, 120, dtype=np.uint16)
        g_lut_data = np.arange(20, 130, dtype=np.uint16)
        b_lut_data = np.arange(30, 140, dtype=np.uint16)
        r_first_mapped_value = 32
        g_first_mapped_value = 32
        b_first_mapped_value = 32
        r_lut = PaletteColorLUT(r_first_mapped_value, r_lut_data, color='red')
        g_lut = PaletteColorLUT(g_first_mapped_value, g_lut_data, color='green')
        b_lut = PaletteColorLUT(b_first_mapped_value, b_lut_data, color='blue')
        transformation = PaletteColorLUTTransformation(
            red_lut=r_lut,
            green_lut=g_lut,
            blue_lut=b_lut,
            palette_color_lut_uid=UID(),
        )
        pr = PseudoColorSoftcopyPresentationState(
            referenced_images=self._ct_series,
            series_instance_uid=self._series_uid,
            series_number=123,
            sop_instance_uid=self._sop_uid,
            instance_number=456,
            manufacturer='Foo Corp.',
            manufacturer_model_name='Bar, Mark 2',
            software_versions='0.0.1',
            device_serial_number='12345',
            content_label='DOODLE',
            graphic_layers=[self._layer],
            graphic_annotations=[self._ann_ct],
            concept_name=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
            palette_color_lut_transformation=transformation,
        )
        assert pr.SeriesInstanceUID == self._series_uid
        assert pr.SOPInstanceUID == self._sop_uid
        assert pr.SeriesNumber == 123
        assert pr.InstanceNumber == 456
        assert pr.Manufacturer == 'Foo Corp.'
        assert pr.ManufacturerModelName == 'Bar, Mark 2'
        assert pr.SoftwareVersions == '0.0.1'
        assert pr.DeviceSerialNumber == '12345'
        assert pr.ContentLabel == 'DOODLE'
        assert len(pr.ReferencedSeriesSequence) == 1
        assert len(pr.GraphicLayerSequence) == 1
        assert pr.InstitutionName == 'MGH'
        assert pr.InstitutionalDepartmentName == 'Radiology'
        assert pr.ContentCreatorName == 'Doe^John'
        assert pr.ConceptNameCodeSequence[0].CodeValue == 'PR'
        assert not hasattr(pr, 'ModalityLUTSequence')
        assert hasattr(pr, 'RescaleSlope')
        assert hasattr(pr, 'RescaleIntercept')
        assert hasattr(pr, 'RescaleType')
        assert pr.RescaleSlope == self._ct_series[0].RescaleSlope
        assert pr.RescaleIntercept == self._ct_series[0].RescaleIntercept
        red_descriptor = [len(r_lut_data), r_first_mapped_value, 16]
        r_lut_data_retrieved = np.frombuffer(
            pr.RedPaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(r_lut_data, r_lut_data_retrieved)
        assert pr.RedPaletteColorLookupTableDescriptor == red_descriptor
        green_descriptor = [len(g_lut_data), g_first_mapped_value, 16]
        g_lut_data_retrieved = np.frombuffer(
            pr.GreenPaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(g_lut_data, g_lut_data_retrieved)
        assert pr.GreenPaletteColorLookupTableDescriptor == green_descriptor
        blue_descriptor = [len(b_lut_data), b_first_mapped_value, 16]
        b_lut_data_retrieved = np.frombuffer(
            pr.BluePaletteColorLookupTableData,
            dtype=np.uint16
        )
        assert np.array_equal(b_lut_data, b_lut_data_retrieved)
        assert pr.BluePaletteColorLookupTableDescriptor == blue_descriptor
