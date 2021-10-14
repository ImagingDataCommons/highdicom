import unittest

import numpy as np

import pytest

from highdicom import UID
from highdicom.pr import (
    GraphicObject,
    GrayscaleSoftcopyPresentationState,
    GraphicTypeValues,
    AnnotationUnitsValues,
    GraphicObject,
)


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
        group_id = 7
        ann = GraphicObject(
            graphic_type=GraphicTypeValues.CIRCLE,
            graphic_data=self._circle,
            graphic_group_id=group_id
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
