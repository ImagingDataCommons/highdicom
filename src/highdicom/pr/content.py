"""Data Elements that are specific to the Presentation State IODs."""

from typing import Optional, Union

from pydicom.dataset import Dataset

import numpy as np

from highdicom.pr.enum import (
    AnnotationUnitsValues,
    GraphicTypeValues,
)
from highdicom.uid import UID


class GraphicObject(Dataset):

    """Dataset describing a graphic annotation object."""

    def __init__(
        self,
        graphic_type: Union[GraphicTypeValues, str],
        graphic_data: np.ndarray,
        units: Union[AnnotationUnitsValues, str] = AnnotationUnitsValues.PIXEL,
        filled: bool = False,
        tracking_id: Optional[str] = None,
        tracking_uid: Optional[str] = None,
        graphic_group_id: Optional[int] = None,
    ):
        """

        Parameters
        ----------
        graphic_type: Union[highdicom.pr.GraphicTypeValues, str]
            Type of the graphic data.
        graphic_data: np.ndarray
            Graphic data contained in a 2D NumPy array. The shape of the array
            should be (N, 2), where N is the number of 2D points in this
            graphic object.  Each row of the array therefore describes a
            (column, row) value for a single 2D point, and the interpretation
            of the points depends upon the graphic type. See
            :class:`highdicom.pr.enum.GraphicTypeValues` for details.
        units: Union[highdicom.pr.AnnotationUnitsValues, str]
            The units in which each point in graphic data is expressed.
        filled: bool
            Whether the graphic object should be rendered as a solid shape
            (True), or just an outline (False). Using True is only valid
            when the graphic type is ``'CIRCLE'`` or ``'ELLIPSE'``, or the
            graphic type is ``'INTERPOLATED'`` or ``'POLYLINE'`` and the first
            and last points are equal giving a closed shape.
        tracking_id: str
            User defined text identifier for tracking this finding or feature.
            Shall be unique within the domain in which it is used.
        tracking_uid: str
            Unique identifier for tracking this finding or feature.

        """
        super().__init__()

        self.GraphicDimensions = 2
        graphic_type = GraphicTypeValues(graphic_type)
        self.GraphicType = graphic_type.value
        units = AnnotationUnitsValues(units)
        self.GraphicAnnotationUnits = units.value

        if not isinstance(graphic_data, np.ndarray):
            raise TypeError(
                'graphic_data must be a numpy array.'
            )
        if graphic_data.ndim != 2:
            raise ValueError(
                'graphic_data must be a 2D array.'
            )
        if graphic_data.shape[1] != 2:
            raise ValueError(
                'graphic_data must be an array of shape (N, 2).'
            )
        num_points = graphic_data.shape[0]
        self.NumberOfGraphicPoints = num_points

        if graphic_type == GraphicTypeValues.POINT:
            if num_points != 1:
                raise ValueError(
                    'Graphic data of type POINT must be a single (column, row)'
                    'pair.'
                )
            if filled:
                raise ValueError(
                    'Setting "filled" to True is invalid when using a '
                    '"POINT" graphic type.'
                )
        elif graphic_type == GraphicTypeValues.CIRCLE:
            if num_points != 2:
                raise ValueError(
                    'Graphic data of type POINT must be two (column, row)'
                    'pairs.'
                )
        elif graphic_type == GraphicTypeValues.ELLIPSE:
            if num_points != 4:
                raise ValueError(
                    'Graphic data of type POINT must be four (column, row)'
                    'pairs.'
                )
        elif graphic_type in (
            GraphicTypeValues.POLYLINE,
            GraphicTypeValues.INTERPOLATED
        ):
            if num_points < 2:
                raise ValueError(
                    'Graphic data of type POINT must be two or more '
                    '(column, row) pairs.'
                )
            if filled:
                if not np.array_equal(graphic_data[0, :], graphic_data[-1, :]):
                    raise ValueError(
                        'Setting "filled" to True when using a '
                        '"POLYLINE" or "INTERPOLATED" graphic type requires '
                        'that the first and last points are equal.'
                    )
        if units == AnnotationUnitsValues.PIXEL:
            if graphic_data.min() < 0.0:
                raise ValueError('Graphic data must be non-negative.')
        elif units == AnnotationUnitsValues.DISPLAY:
            if graphic_data.min() < 0.0 or graphic_data.max() > 1.0:
                raise ValueError(
                    'Graphic data must be in the range 0.0 to 1.0 when using '
                    '"DISPLAY" units.'
                )
        self.GraphicData = graphic_data.flatten().tolist()
        self.GraphicFilled = 'Y' if filled else 'N'

        if (tracking_id is None) != (tracking_uid is None):
            raise TypeError(
                'If either "tracking_id" or "tracking_uid" is provided, the '
                'other must also be provided.'
            )
        if tracking_id is not None:
            self.TrackingID = tracking_id
            self.TrackingUID = tracking_uid

        if graphic_group_id is not None:
            if not isinstance(graphic_group_id, int):
                raise ValueError('Graphic group ID should be an integer.')
            if graphic_group_id < 1:
                raise ValueError(
                    'Graphic group ID should be a positive integer.'
                )
            self.GraphicGroupID = graphic_group_id

    @property
    def graphic_data(self) -> np.ndarray:
        """numpy.ndarray: n x 2 array of 2D coordinates"""
        return np.array(self.GraphicData).reshape(-1, 2)

    @property
    def graphic_type(self) -> GraphicTypeValues:
        """highdicom.pr.GraphicTypeValues: graphic type"""
        return GraphicTypeValues(self.GraphicType)

    @property
    def units(self) -> AnnotationUnitsValues:
        """highdicom.pr.AnnotationUnitsValues: annotation units"""
        return AnnotationUnitsValues(self.GraphicAnnotationUnits)

    @property
    def tracking_id(self) -> Union[str, None]:
        """Union[str, None]: tracking identifier"""
        return getattr(self, 'TrackingID', None)

    @property
    def tracking_uid(self) -> Union[UID, None]:
        """Union[highdicom.UID, None]: tracking UID"""
        if hasattr(self, 'TrackingUID'):
            return UID(self.TrackingUID)
        return None
