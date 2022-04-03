"""Data Elements that are specific to the Presentation State IODs."""

from typing import Optional, Union, Sequence, Tuple

from pydicom.dataset import Dataset

import numpy as np

from highdicom.color import CIELabColor
from highdicom.content import LUT, ReferencedImageSequence, VOILUT
from highdicom.enum import VOILUTFunctionValues
from highdicom.pr.enum import (
    AnnotationUnitsValues,
    GraphicTypeValues,
    TextJustificationValues,
)
from highdicom.uid import UID
from highdicom.utils import is_tiled_image
from highdicom.valuerep import (
    _check_code_string,
    _check_long_string,
    _check_short_text
)


class GraphicLayer(Dataset):

    """A layer of graphic annotations that should be rendered together."""

    def __init__(
        self,
        layer_name: str,
        order: int,
        description: Optional[str] = None,
        display_color: Optional[CIELabColor] = None
    ):
        """

        Parameters
        ----------
        layer_name: str
            Name for the layer.  Should be a valid DICOM Code String (CS), i.e.
            16 characters or fewer containing only uppercase letters, spaces
            and underscores.
        order: int
            Integer indicating the order in which this layer should be rendered.
            Lower values are rendered first.
        description: Union[str, None], optional
            A description of the contents of this graphic layer.
        display_color: Union[CIELabColor, None], optional
            A default color value for rendering this layer.

        """
        super().__init__()
        _check_code_string(layer_name)
        self.GraphicLayer = layer_name
        if not isinstance(order, int):
            raise TypeError('"order" must be an integer.')
        self.GraphicLayerOrder = order
        if description is not None:
            _check_long_string(description)
            self.GraphicLayerDescription = description
        if display_color is not None:
            if not isinstance(display_color, CIELabColor):
                raise TypeError(
                    '"recommended_display_color" must be of type '
                    'highdicom.color.CIELabColor.'
                )
            self.GraphicLayerRecommendedDisplayCIELabValue = list(
                display_color.value
            )


class GraphicGroup(Dataset):

    """Dataset describing a grouping of annotations.

    Note
    ----
    ``GraphicGroup`` s represent an independent concept from ``GraphicLayer``
    s. Where a ``GraphicLayer`` (:class:`highdicom.pr.GraphicLayer`) specifies
    which annotations are rendered first, a ``GraphicGroup`` specifies which
    annotations belong together and shall be handled together (e.g., rotate,
    move) independent of the ``GraphicLayer`` to which they are assigned.

    Each annotation (:class:`highdicom.pr.GraphicObject` or
    :class:`highdicom.pr.TextObject`) may optionally be assigned to a single
    ``GraphicGroup`` upon construction, whereas assignment to a
    :class:`highdicom.pr.GraphicLayer` is required.

    For example, suppose a presentation state is to include two
    ``GraphicObject`` s, each accompanied by a corresponding ``TextObject`` that
    indicates the meaning of the graphic and should be rendered above the
    ``GraphicObject`` if they overlap. In this situation, it may be useful to
    group each ``TextObject`` with the corresponding ``GraphicObject`` as a
    distinct ``GraphicGroup`` (giving two ``GraphicGroup`` s each containing one
    ``TextObject`` and one ``GraphicObject``) and also place both
    ``GraphicObject`` s in one ``GraphicLayer`` and both ``TextObject`` s in a
    second ``GraphicLayer`` with a higher ``order`` to control rendering.

    """
    def __init__(
        self,
        graphic_group_id: int,
        label: str,
        description: Optional[str] = None
    ):
        """

        Parameters
        ----------
        graphic_group_id: int
            A positive integer that uniquely identifies this graphic group.
        label: str
            Name used to identify the Graphic Group (maximum 64 characters).
        description: Union[str, None], optional
            Description of the group (maxiumum 10240 characters).

        """
        super().__init__()
        if not isinstance(graphic_group_id, int):
            raise TypeError(
                'Argument "graphic_group_id" must be an integer.'
            )
        if graphic_group_id < 1:
            raise ValueError(
                'Argument "graphic_group_id" must be a positive integer.'
            )
        self.GraphicGroupID = graphic_group_id
        _check_long_string(label)
        self.GraphicGroupLabel = label
        if description is not None:
            _check_short_text(description)
            self.GraphicGroupDescription = description

    @property
    def graphic_group_id(self) -> int:
        """int: The ID of the graphic group."""
        return self.GraphicGroupID


class GraphicObject(Dataset):

    """Dataset describing a graphic annotation object."""

    def __init__(
        self,
        graphic_type: Union[GraphicTypeValues, str],
        graphic_data: np.ndarray,
        units: Union[AnnotationUnitsValues, str],
        is_filled: bool = False,
        tracking_id: Optional[str] = None,
        tracking_uid: Optional[str] = None,
        graphic_group: Optional[GraphicGroup] = None,
    ):
        """

        Parameters
        ----------
        graphic_type: Union[highdicom.pr.GraphicTypeValues, str]
            Type of the graphic data.
        graphic_data: numpy.ndarray
            Graphic data contained in a 2D NumPy array. The shape of the array
            should be (N, 2), where N is the number of 2D points in this
            graphic object.  Each row of the array therefore describes a
            (column, row) value for a single 2D point, and the interpretation
            of the points depends upon the graphic type. See
            :class:`highdicom.pr.enum.GraphicTypeValues` for details.
        units: Union[highdicom.pr.AnnotationUnitsValues, str]
            The units in which each point in graphic data is expressed.
        is_filled: bool, optional
            Whether the graphic object should be rendered as a solid shape
            (``True``), or just an outline (``False``). Using ``True`` is only
            valid when the graphic type is ``'CIRCLE'`` or ``'ELLIPSE'``, or
            the graphic type is ``'INTERPOLATED'`` or ``'POLYLINE'`` and the
            first and last points are equal giving a closed shape.
        tracking_id: str, optional
            User defined text identifier for tracking this finding or feature.
            Shall be unique within the domain in which it is used.
        tracking_uid: str, optional
            Unique identifier for tracking this finding or feature.
        graphic_group: Union[highdicom.pr.GraphicGroup, None]
            Graphic group to which this annotation belongs.

        """
        super().__init__()

        self.GraphicDimensions = 2
        graphic_type = GraphicTypeValues(graphic_type)
        self.GraphicType = graphic_type.value
        units = AnnotationUnitsValues(units)
        self.GraphicAnnotationUnits = units.value

        if not isinstance(graphic_data, np.ndarray):
            raise TypeError('Argument "graphic_data" must be a numpy array.')
        if graphic_data.ndim != 2:
            raise ValueError('Argument "graphic_data" must be a 2D array.')
        if graphic_data.shape[1] != 2:
            raise ValueError(
                'Argument "graphic_data" must be an array of shape (N, 2).'
            )
        num_points = graphic_data.shape[0]
        self.NumberOfGraphicPoints = num_points

        if graphic_type == GraphicTypeValues.POINT:
            if num_points != 1:
                raise ValueError(
                    'Graphic data of type "POINT" '
                    'must be a single (column, row)'
                    'pair.'
                )
            if is_filled:
                raise ValueError(
                    'Setting "is_filled" to True is invalid when using a '
                    '"POINT" graphic type.'
                )
        elif graphic_type == GraphicTypeValues.CIRCLE:
            if num_points != 2:
                raise ValueError(
                    'Graphic data of type "CIRCLE" '
                    'must be two (column, row) pairs.'
                )
        elif graphic_type == GraphicTypeValues.ELLIPSE:
            if num_points != 4:
                raise ValueError(
                    'Graphic data of type "ELLIPSE" '
                    'must be four (column, row) pairs.'
                )
        elif graphic_type in (
            GraphicTypeValues.POLYLINE,
            GraphicTypeValues.INTERPOLATED,
        ):
            if num_points < 2:
                raise ValueError(
                    'Graphic data of type "POLYLINE" or "INTERPOLATED" '
                    'must be two or more (column, row) pairs.'
                )
            if is_filled:
                if not np.array_equal(graphic_data[0, :], graphic_data[-1, :]):
                    raise ValueError(
                        'Setting "is_filled" to True when using a '
                        '"POLYLINE" or "INTERPOLATED" graphic type requires '
                        'that the first and last points are equal, '
                        'i.e., that the graphic has a closed contour. '
                    )
        if (
            units == AnnotationUnitsValues.PIXEL or
            units == AnnotationUnitsValues.MATRIX
        ):
            if graphic_data.min() < 0.0:
                raise ValueError('Graphic data must be non-negative.')
        elif units == AnnotationUnitsValues.DISPLAY:
            if graphic_data.min() < 0.0 or graphic_data.max() > 1.0:
                raise ValueError(
                    'Graphic data must be in the range 0.0 to 1.0 when using '
                    '"DISPLAY" units.'
                )
        self.GraphicData = graphic_data.flatten().tolist()
        self.GraphicFilled = 'Y' if is_filled else 'N'

        if (tracking_id is None) != (tracking_uid is None):
            raise TypeError(
                'If either "tracking_id" or "tracking_uid" is provided, the '
                'other must also be provided.'
            )
        if tracking_id is not None:
            self.TrackingID = tracking_id
            self.TrackingUID = tracking_uid

        if graphic_group is not None:
            if not isinstance(graphic_group, GraphicGroup):
                raise TypeError(
                    'Argument "graphic_group" should be of type '
                    'highdicom.pr.GraphicGroup.'
                )
            self.GraphicGroupID = graphic_group.graphic_group_id

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

    @property
    def graphic_group_id(self) -> Union[int, None]:
        """Union[int, None]: The ID of the graphic group, if any."""
        return getattr(self, 'GraphicGroupID', None)


class TextObject(Dataset):

    """Dataset describing a text annotation object."""

    def __init__(
        self,
        text_value: str,
        units: Union[AnnotationUnitsValues, str],
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        anchor_point: Optional[Tuple[float, float]] = None,
        text_justification: Union[
            TextJustificationValues, str
        ] = TextJustificationValues.CENTER,
        anchor_point_visible: bool = True,
        tracking_id: Optional[str] = None,
        tracking_uid: Optional[str] = None,
        graphic_group: Optional[GraphicGroup] = None,
    ):
        """

        Parameters
        ----------
        text_value: str
            The unformatted text value.
        units: Union[highdicom.pr.AnnotationUnitsValues, str]
            The units in which the coordinates of the bounding box and/or
            anchor point are expressed.
        bounding_box: Union[Tuple[float, float, float, float], None], optional
            Coordinates of the bounding box in which the text should be
            displayed, given in the following order [left, top, right, bottom],
            where 'left' and 'right' are the horizontal offsets of the left and
            right sides of the box, respectively, and 'top' and 'bottom' are
            the vertical offsets of the upper and lower sides of the box.
        anchor_point: Union[Tuple[float, float], None], optional
            Location of a point in the image to which the text value is related,
            given as a (Column, Row) pair.
        anchor_point_visible: bool, optional
            Whether the relationship between the anchor point and the text
            should be displayed in the image, for example via a line or arrow.
            This parameter is ignored if the anchor_point is not provided.
        tracking_id: str, optional
            User defined text identifier for tracking this finding or feature.
            Shall be unique within the domain in which it is used.
        tracking_uid: str, optional
            Unique identifier for tracking this finding or feature.
        graphic_group: Union[highdicom.pr.GraphicGroup, None], optional
            Graphic group to which this annotation belongs.

        Note
        ----
        Either the ``anchor_point`` or the ``bounding_box`` parameter (or both)
        must be provided to localize the text in the image.

        """
        super().__init__()
        _check_short_text(text_value)
        self.UnformattedTextValue = text_value

        units = AnnotationUnitsValues(units)

        if bounding_box is None and anchor_point is None:
            raise TypeError(
                'Either an anchor point or a bounding box (or both) must be '
                'specified.'
            )

        if bounding_box is not None:
            if len(bounding_box) != 4:
                raise ValueError('Bounding box must contain four values.')
            if min(bounding_box) < 0.0:
                raise ValueError(
                    'All coordinates in the bounding box must be non-negative.'
                )
            self.BoundingBoxTopLeftHandCorner = list(bounding_box[:2])
            self.BoundingBoxBottomRightHandCorner = list(bounding_box[2:])
            text_justification = TextJustificationValues(text_justification)
            self.BoundingBoxTextHorizontalJustification = \
                text_justification.value
            self.BoundingBoxAnnotationUnits = units.value
            if units == AnnotationUnitsValues.DISPLAY:
                if max(bounding_box) > 1.0:
                    raise ValueError(
                        'All coordinates in the bounding box must be less '
                        'than or equal to 1 when using DISPLAY units.'
                    )

        if anchor_point is not None:
            if len(anchor_point) != 2:
                raise ValueError('Anchor point must contain two values.')
            if min(anchor_point) < 0.0:
                raise ValueError(
                    'All coordinates in the bounding box must be non-negative.'
                )
            self.AnchorPoint = anchor_point
            self.AnchorPointAnnotationUnits = units.value
            self.AnchorPointVisibility = 'Y' if anchor_point_visible else 'N'
            if units == AnnotationUnitsValues.DISPLAY:
                if max(anchor_point) > 1.0:
                    raise ValueError(
                        'All coordinates in the anchor point must be less '
                        'than or equal to 1 when using DISPLAY units.'
                    )

        if (tracking_id is None) != (tracking_uid is None):
            raise TypeError(
                'If either "tracking_id" or "tracking_uid" is provided, the '
                'other must also be provided.'
            )
        if tracking_id is not None:
            self.TrackingID = tracking_id
            self.TrackingUID = tracking_uid

        if graphic_group is not None:
            if not isinstance(graphic_group, GraphicGroup):
                raise TypeError(
                    'Argument "graphic_group" should be of type '
                    'highdicom.pr.GraphicGroup.'
                )
            self.GraphicGroupID = graphic_group.graphic_group_id

    @property
    def text_value(self) -> str:
        """str: unformatted text value"""
        return self.UnformattedTextValue

    @property
    def bounding_box(self) -> Union[Tuple[float, float, float, float], None]:
        """Union[Tuple[float, float, float, float], None]:
        bounding box in the format [left, top, right, bottom]

        """
        if not hasattr(self, 'BoundingBoxTopLeftHandCorner'):
            return None
        return tuple(self.BoundingBoxTopLeftHandCorner) + tuple(
            self.BoundingBoxBottomRightHandCorner
        )

    @property
    def anchor_point(self) -> Union[Tuple[float, float], None]:
        """Union[Tuple[float, float], None]:
        anchor point as a (Row, Column) pair of image coordinates

        """
        if not hasattr(self, 'AnchorPoint'):
            return None
        return tuple(self.AnchorPoint)

    @property
    def units(self) -> AnnotationUnitsValues:
        """highdicom.pr.AnnotationUnitsValues: annotation units"""
        if hasattr(self, 'BoundingBoxAnnotationUnits'):
            return AnnotationUnitsValues(self.BoundingBoxAnnotationUnits)
        return AnnotationUnitsValues(self.AnchorPointAnnotationUnits)

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

    @property
    def graphic_group_id(self) -> Union[int, None]:
        """Union[int, None]: The ID of the graphic group, if any."""
        return getattr(self, 'GraphicGroupID', None)


class GraphicAnnotation(Dataset):

    """Dataset describing related graphic and text objects."""

    def __init__(
        self,
        referenced_images: Sequence[Dataset],
        graphic_layer: GraphicLayer,
        referenced_frame_number: Union[int, Sequence[int], None] = None,
        referenced_segment_number: Union[int, Sequence[int], None] = None,
        graphic_objects: Optional[Sequence[GraphicObject]] = None,
        text_objects: Optional[Sequence[TextObject]] = None,
    ):
        """
        Parameters
        ----------
        referenced_images: Sequence[pydicom.dataset.Dataset]
            Sequence of referenced datasets. Graphic and text objects shall be
            rendered on all images in this list.
        graphic_layer: highdicom.pr.GraphicLayer
            Graphic layer to which this annotation should belong.
        referenced_frame_number: Union[int, Sequence[int], None], optional
            Frame number(s) in a multiframe image upon which annotations shall
            be rendered.
        referenced_segment_number: Union[int, Sequence[int], None], optional
            Frame number(s) in a multi-frame image upon which annotations shall
            be rendered.
        graphic_objects: Union[Sequence[highdicom.pr.GraphicObject], None], optional
            Graphic objects to render over the referenced images.
        text_objects: Union[Sequence[highdicom.pr.TextObject], None], optional
            Text objects to render over the referenced images.

        """  # noqa: E501
        super().__init__()
        if len(referenced_images) == 0:
            raise ValueError('List of referenced images must not be empty.')
        referenced_series_uid = referenced_images[0].SeriesInstanceUID
        if not isinstance(graphic_layer, GraphicLayer):
            raise TypeError(
                'Argument "graphic_layer" should be of type '
                'highdicom.pr.GraphicLayer.'
            )
        self.GraphicLayer = graphic_layer.GraphicLayer

        is_multiframe = hasattr(referenced_images[0], 'NumberOfFrames')
        if is_multiframe and len(referenced_images) > 1:
            raise ValueError(
                'If referenced images are multi-frame, only a single image '
                'should be passed.'
            )
        if is_multiframe:
            if (
                referenced_frame_number is not None and
                referenced_segment_number is not None
            ):
                raise TypeError(
                    'At most one of "referenced_frame_number" or '
                    '"referenced_segment_number" should be provided.'
                )
        for ref_im in referenced_images:
            if ref_im.SeriesInstanceUID != referenced_series_uid:
                raise ValueError(
                    'All referenced images must belong to the same series.'
                )
        self.ReferencedImageSequence = ReferencedImageSequence(
            referenced_images=referenced_images,
            referenced_frame_number=referenced_frame_number,
            referenced_segment_number=referenced_segment_number
        )

        have_graphics = graphic_objects is not None and len(graphic_objects) > 0
        have_text = text_objects is not None and len(text_objects) > 0
        if not have_graphics and not have_text:
            raise TypeError(
                'Either "graphic_objects" or "text_objects" must contain at '
                'least one item.'
            )
        if have_graphics:
            for go in graphic_objects:
                if not isinstance(go, GraphicObject):
                    raise TypeError(
                        'All items in "graphic_objects" must be of type '
                        'highdicom.pr.GraphicObject'
                    )
                if go.units == AnnotationUnitsValues.MATRIX:
                    if not is_tiled_image(referenced_images[0]):
                        raise ValueError(
                            'Graphic Objects may only use MATRIX units if the '
                            'referenced images are tiled images. '
                        )
            self.GraphicObjectSequence = graphic_objects
        if have_text:
            for to in text_objects:
                if not isinstance(to, TextObject):
                    raise TypeError(
                        'All items in text_objects must be of type '
                        'highdicom.pr.TextObject'
                    )
                if to.units == AnnotationUnitsValues.MATRIX:
                    if not is_tiled_image(referenced_images[0]):
                        raise ValueError(
                            'Text Objects may only use MATRIX units if the '
                            'referenced images are tiled images. '
                        )
            self.TextObjectSequence = text_objects

    @staticmethod
    def _check_coords(
        graphic_data: np.ndarray,
        referenced_image: Dataset,
        units: AnnotationUnitsValues,
    ) -> None:
        """Check whether pixel data in PIXEL units is valid for an image.

        Raises an exception if any value is invalid.

        """
        pass


class SoftcopyVOILUT(VOILUT):

    """Dataset describing an item of the Softcopy VOI LUT Sequence."""

    def __init__(
        self,
        window_center: Union[float, Sequence[float], None] = None,
        window_width: Union[float, Sequence[float], None] = None,
        window_explanation: Union[str, Sequence[str], None] = None,
        voi_lut_function: Union[VOILUTFunctionValues, str, None] = None,
        luts: Optional[Sequence[LUT]] = None,
        referenced_images: Optional[ReferencedImageSequence] = None,
    ):
        """

        Parameters
        ----------
        window_center: Union[float, Sequence[float], None], optional
            Center value of the intensity window used for display.
        window_width: Union[float, Sequence[float], None], optional
            Width of the intensity window used for display.
        window_explanation: Union[str, Sequence[str], None], optional
            Free-form explanation of the window center and width.
        voi_lut_function: Union[highdicom.VOILUTFunctionValues, str, None], optional
            Description of the LUT function parametrized by ``window_center``.
            and ``window_width``.
        luts: Union[Sequence[highdicom.LUT], None], optional
            Intensity lookup tables used for display.
        referenced_images: Union[highdicom.ReferencedImageSequence, None], optional
            Images to which the VOI LUT described in this dataset applies. Note
            that if unspecified, the VOI LUT applies to every image referenced
            in the presentation state object that this dataset is included in.

        Note
        ----
        Either ``window_center`` and ``window_width`` should be provided or
        ``luts`` should be provided, or both. ``window_explanation`` should
        only be provided if ``window_center`` is provided.

        """  # noqa: E501
        super().__init__(
            window_center=window_center,
            window_width=window_width,
            window_explanation=window_explanation,
            voi_lut_function=voi_lut_function,
            luts=luts,
        )

        if referenced_images is not None:
            if not isinstance(referenced_images, ReferencedImageSequence):
                raise TypeError(
                    'Argument "referenced_images" must be of type '
                    'highdicom.ReferencedImageSequence.'
                )
            self.ReferencedImageSequence = referenced_images
