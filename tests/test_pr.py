from pathlib import Path
import unittest

import numpy as np

import pytest

from pydicom import dcmread
from pydicom.sr.codedict import codes
from pydicom.data import get_testdata_file, get_testdata_files

from highdicom import UID
from highdicom.color import CIELabColor
from highdicom.pr import (
    ContentCreatorIdentificationCodeSequence,
    GraphicAnnotation,
    GraphicGroup,
    GraphicLayer,
    GraphicObject,
    GraphicTypeValues,
    GrayscaleSoftcopyPresentationState,
    AnnotationUnitsValues,
    TextObject
)
from highdicom.sr import CodedConcept


class TestContentCreatorIdentification(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._person_codes = [codes.DCM.Person, codes.DCM.Technologist]
        self._institution_name = 'MGH'
        self._person_address = '1000 Main St.'
        self._person_telephone_numbers = ['123456789']
        self._email = 'example@example.com'
        self._institution_address = '123 Broadway'
        self._institution_code = CodedConcept(
            value='1',
            meaning='MGH',
            scheme_designator='HOSPITAL_NAMES',
        )
        self._department_name = 'Radiology'
        self._department_code = codes.SCT.RadiologyDepartment

    def test_construction_minimal(self):
        creator_id = ContentCreatorIdentificationCodeSequence(
            person_identification_codes=self._person_codes,
            institution_name=self._institution_name,
        )
        assert len(creator_id) == 1
        creator_id_item = creator_id[0]
        print(creator_id)
        assert creator_id_item.InstitutionName == self._institution_name
        for code1, code2 in zip(
            creator_id_item.PersonIdentificationCodeSequence,
            self._person_codes
        ):
            assert code1.CodeValue == code2.value

    def test_construction_full(self):
        creator_id = ContentCreatorIdentificationCodeSequence(
            person_identification_codes=self._person_codes,
            institution_name=self._institution_name,
            person_address=self._person_address,
            person_telephone_numbers=self._person_telephone_numbers,
            person_telecom_information=self._email,
            institution_code=self._institution_code,
            institution_address=self._institution_address,
            institutional_department_name=self._department_name,
            institutional_department_type_code=self._department_code,
        )
        assert len(creator_id) == 1
        creator_id_item = creator_id[0]
        print(creator_id)
        assert creator_id_item.InstitutionName == self._institution_name
        for code1, code2 in zip(
            creator_id_item.PersonIdentificationCodeSequence,
            self._person_codes
        ):
            assert code1.CodeValue == code2.value
        assert creator_id_item.PersonAddress == self._person_address
        assert (
            creator_id_item.PersonTelephoneNumbers ==
            self._person_telephone_numbers
        )
        assert (
            creator_id_item.PersonTelecomInformation ==
            self._email
        )
        assert (
            creator_id_item.InstitutionCodeSequence[0].CodeValue ==
            self._institution_code.value
        )
        assert creator_id_item.InstitutionAddress == self._institution_address
        assert (
            creator_id_item.InstitutionalDepartmentName ==
            self._department_name
        )
        department_code = \
            creator_id_item.InstitutionalDepartmentTypeCodeSequence[0]
        assert (department_code.CodeValue == self._department_code.value)


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
        )
        assert ann.graphic_type == GraphicTypeValues.CIRCLE
        assert np.array_equal(ann.graphic_data, self._circle)
        assert ann.units == AnnotationUnitsValues.PIXEL
        assert ann.GraphicFilled == 'N'

    def test_circle_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.CIRCLE,
                graphic_data=self._point
            )

    def test_ellipse(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.ELLIPSE,
            graphic_data=self._ellipse,
        )
        assert ann.graphic_type == GraphicTypeValues.ELLIPSE
        assert np.array_equal(ann.graphic_data, self._ellipse)
        assert ann.units == AnnotationUnitsValues.PIXEL

    def test_ellipse_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.ELLIPSE,
                graphic_data=self._circle
            )

    def test_point(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.POINT,
            graphic_data=self._point,
        )
        assert ann.graphic_type == GraphicTypeValues.POINT
        assert np.array_equal(ann.graphic_data, self._point)
        assert ann.units == AnnotationUnitsValues.PIXEL

    def test_point_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.POINT,
                graphic_data=self._circle
            )

    def test_polyline(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=self._polyline,
        )
        assert ann.graphic_type == GraphicTypeValues.POLYLINE
        assert np.array_equal(ann.graphic_data, self._polyline)
        assert ann.units == AnnotationUnitsValues.PIXEL

    def test_polyline_wrong_number(self):
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.POLYLINE,
                graphic_data=self._point
            )

    def test_interpolated(self):
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.INTERPOLATED,
            graphic_data=self._polyline,
        )
        assert ann.graphic_type == GraphicTypeValues.INTERPOLATED
        assert np.array_equal(ann.graphic_data, self._polyline)

    def test_tracking_ids(self):
        uid = UID()
        label = 'circle'
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
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
            filled=True
        )
        assert ann.GraphicFilled == 'Y'

    def test_filled_invalid(self):
        # Cannot have a filled polyline
        with pytest.raises(ValueError):
            GraphicObject(
                graphic_type=GraphicTypeValues.POLYLINE,
                graphic_data=self._polyline,
                filled=True
            )


class TestTextObject(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._text = 'Look Here!'
        self._bounding_box = (10, 30, 40, 60)
        self._anchor_point = (100, 110)
        self._tracking_id = 'tracking_id_1'
        self._tracking_uid = UID()

    def test_bounding_box(self):
        text_obj = TextObject(
            text_value=self._text,
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
                bounding_box=self._anchor_point  # wrong number of elements
            )

    def test_anchor_point_wrong_number(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                anchor_point=self._bounding_box  # wrong number of elements
            )

    def test_tracking_uids(self):
        text_obj = TextObject(
            text_value=self._text,
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
                bounding_box=self._bounding_box,
                tracking_id=self._tracking_id,
            )

    def test_bounding_box_invalid_values(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                bounding_box=(-1.0, 3.0, 7.8, 9.0),
            )

    def test_bounding_box_invalid_values_display(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
                bounding_box=(1.0, 3.0, 7.8, 9.0),
                units=AnnotationUnitsValues.DISPLAY
            )

    def test_anchor_point_invalid_values(self):
        with pytest.raises(ValueError):
            TextObject(
                text_value=self._text,
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
        seg_path = data_dir.joinpath('test_files', 'seg_image_sm_dots.dcm')
        self._segmentation = dcmread(seg_path)
        self._frame_number = 2
        self._frame_numbers = [1, 2]
        self._segment_number = 5
        self._segment_numbers = [7, 12]
        self._text_value = 'Look Here!'
        self._bounding_box = (10, 30, 40, 60)
        self._text_object = TextObject(
            text_value=self._text_value,
            bounding_box=self._bounding_box
        )
        self._circle = np.array([
            [10.0, 10.0],
            [11.0, 10.0]
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
        with pytest.raises(TypeError):
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
        with pytest.raises(TypeError):
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


class TestGSPS(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._series_uid = UID()
        self._sop_uid = UID()
        self._ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        self._text_value = 'Look Here!'
        self._bounding_box = (10.0, 30.0, 40.0, 60.0)
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
            bounding_box=self._bounding_box
        )
        self._circle = np.array([
            [10.0, 10.0],
            [11.0, 10.0]
        ])
        self._graphic_object = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
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
        self._ann = GraphicAnnotation(
            referenced_images=self._ct_series,
            graphic_layer=self._layer,
            graphic_objects=[self._graphic_object],
            text_objects=[self._text_object]
        )

        # Same thing, but with object belonging to groups
        self._text_object_grp = TextObject(
            text_value=self._text_value,
            bounding_box=self._bounding_box,
            graphic_group=self._group,
        )
        self._graphic_object_grp = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
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

    def test_construction(self):
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
            graphic_annotations=[self._ann],
            concept_name_code=codes.DCM.PresentationState,
            institution_name='MGH',
            institutional_department_name='Radiology',
            content_creator_name='Doe^John',
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
            graphic_annotations=[self._ann],
            concept_name_code=codes.DCM.PresentationState,
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
                graphic_annotations=[self._ann],
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
                graphic_annotations=[self._ann],
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
                graphic_annotations=[self._ann],
                concept_name_code=codes.DCM.PresentationState,
                institution_name='MGH',
                institutional_department_name='Radiology',
                content_creator_name='Doe^John',
            )
