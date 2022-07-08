"""Data Elements that are specific to the Presentation State IODs."""
import datetime
import logging
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL.ImageCms import ImageCmsProfile
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.multival import MultiValue
from pydicom.valuerep import DA, PersonName, TM
from typing import Optional, Union, Sequence, Tuple

from highdicom.color import CIELabColor
from highdicom.content import (
    ContentCreatorIdentificationCodeSequence,
    ModalityLUTTransformation,
    PaletteColorLUTTransformation,
    PresentationLUTTransformation,
    ReferencedImageSequence,
    VOILUT,
    VOILUTTransformation,
)
from highdicom.enum import (
    RescaleTypeValues,
    VOILUTFunctionValues,
)
from highdicom.pr.enum import (
    AnnotationUnitsValues,
    BlendingModeValues,
    GraphicTypeValues,
    TextJustificationValues,
)
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID
from highdicom.utils import is_tiled_image
from highdicom.valuerep import (
    check_person_name,
    _check_code_string,
    _check_long_string,
    _check_short_text
)

logger = logging.getLogger(__name__)


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
            if (
                bounding_box[0] >= bounding_box[2] or
                bounding_box[1] >= bounding_box[3]
            ):
                raise ValueError(
                    'The bottom right hand corner of the bounding box must be '
                    'below and to the right of the top left hand corner.'
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
        rows = referenced_images[0].Rows
        columns = referenced_images[0].Columns
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
            if not is_tiled_image(ref_im):
                if ref_im.Columns != columns or ref_im.Rows != rows:
                    raise ValueError(
                        'All referenced images must have the same number '
                        'of rows and columns.'
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
                self._check_coords(
                    go.graphic_data,
                    referenced_images[0],
                    go.units,
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
                if to.bounding_box is not None:
                    graphic_data = np.array(to.bounding_box).reshape((2, 2))
                    self._check_coords(
                        graphic_data,
                        referenced_images[0],
                        to.units,
                    )
                if to.anchor_point is not None:
                    graphic_data = np.array(to.anchor_point).reshape((1, 2))
                    self._check_coords(
                        graphic_data,
                        referenced_images[0],
                        to.units,
                    )
            self.TextObjectSequence = text_objects

    @staticmethod
    def _check_coords(
        graphic_data: np.ndarray,
        referenced_image: Dataset,
        units: AnnotationUnitsValues,
    ) -> None:
        """Check whether graphic data is valid for an image.

        Parameters
        ----------
        graphic_data: np.ndarray
            Graphic data as stored within a GraphicObject.
        referenced_image: pydicom.Dataset
            Image to which the graphic data refers.
        units: highdicom.pr.AnnotationUnitsValues
            Units in which the graphic data are expressed.

        Raises
        ------
        ValueError:
            Raises an exception if any value in graphic_data is outside the
            valid range of coordinates for referenced_image when using the
            units specified by the units parameter.

        """
        min_col = graphic_data[:, 1].min()
        max_col = graphic_data[:, 0].max()
        min_row = graphic_data[:, 1].min()
        max_row = graphic_data[:, 1].max()

        if units == AnnotationUnitsValues.DISPLAY:
            col_limit = 1.0
            row_limit = 1.0
            col_limit_msg = '1.0'
            row_limit_msg = '1.0'
        elif units == AnnotationUnitsValues.PIXEL:
            col_limit = float(referenced_image.Columns)
            row_limit = float(referenced_image.Rows)
            col_limit_msg = 'Columns'
            row_limit_msg = 'Rows'
        elif units == AnnotationUnitsValues.MATRIX:
            col_limit = float(referenced_image.TotalPixelMatrixColumns)
            row_limit = float(referenced_image.TotalPixelMatrixRows)
            col_limit_msg = 'TotalPixelMatrixColumns'
            row_limit_msg = 'TotalPixelMatrixRows'

        if (
            min_col < 0.0 or
            min_row < 0.0 or
            max_col > col_limit or
            max_row > row_limit
        ):
            raise ValueError(
                'Found graphic data outside the valid range within one or '
                'more GraphicObjects or TextObjects. When using units '
                f'of type {units.value}, all column coordinates must lie in '
                f'the range 0.0 to {col_limit_msg} and all row coordinates '
                f'must lie in the range 0.0 to {row_limit_msg}.'
            )


class SoftcopyVOILUTTransformation(VOILUTTransformation):

    """Dataset describing the VOI LUT Transformation as part of the Pixel
    Transformation Sequence to transform the modality pixel values into
    pixel values that are of interest to a user or an application.

    The description is specific to the application of the VOI LUT
    Transformation in the context of a Softcopy Presentation State, where
    potentially only a subset of explicitly referenced images should be
    transformed.

    """

    def __init__(
        self,
        window_center: Union[float, Sequence[float], None] = None,
        window_width: Union[float, Sequence[float], None] = None,
        window_explanation: Union[str, Sequence[str], None] = None,
        voi_lut_function: Union[VOILUTFunctionValues, str, None] = None,
        voi_luts: Optional[Sequence[VOILUT]] = None,
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
        voi_luts: Union[Sequence[highdicom.VOILUT], None], optional
            Intensity lookup tables used for display.
        referenced_images: Union[highdicom.ReferencedImageSequence, None], optional
            Images to which the VOI LUT Transformation described in this
            dataset applies. Note that if unspecified, the VOI LUT
            Transformation applies to every frame of every image referenced in
            the presentation state object that this dataset is included in.

        Note
        ----
        Either ``window_center`` and ``window_width`` should be provided or
        ``voi_luts`` should be provided, or both. ``window_explanation`` should
        only be provided if ``window_center`` is provided.

        """  # noqa: E501
        super().__init__(
            window_center=window_center,
            window_width=window_width,
            window_explanation=window_explanation,
            voi_lut_function=voi_lut_function,
            voi_luts=voi_luts
        )
        if referenced_images is not None:
            if not isinstance(referenced_images, ReferencedImageSequence):
                raise TypeError(
                    'Argument "referenced_images" must be of type '
                    'highdicom.ReferencedImageSequence.'
                )
            self.ReferencedImageSequence = referenced_images


def _add_presentation_state_identification_attributes(
    dataset: Dataset,
    content_label: str,
    content_description: Optional[str] = None,
    concept_name: Union[Code, CodedConcept, None] = None,
    content_creator_name: Optional[Union[str, PersonName]] = None,
    content_creator_identification: Optional[
        ContentCreatorIdentificationCodeSequence
    ] = None,
) -> None:
    """Add attributes of module Presentation State Identification.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    content_label: str
        A label used to describe the content of this presentation state.
        Must be a valid DICOM code string consisting only of capital
        letters, underscores and spaces.
    content_description: Union[str, None], optional
        Description of the content of this presentation state.
    concept_name: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
        A coded description of the content of this presentation state.
    content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
        Name of the person who created the content of this presentation
        state.
    content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
        Identifying information for the person who created the content of
        this presentation state.

    """  # noqa: E501
    _check_code_string(content_label)
    dataset.ContentLabel = content_label
    if content_description is not None:
        if len(content_description) > 64:
            raise ValueError(
                'Argument "content_description" must not exceed 64 characters.'
            )
        dataset.ContentDescription = content_description
    now = datetime.datetime.now()
    dataset.PresentationCreationDate = DA(now.date())
    dataset.PresentationCreationTime = TM(now.time())

    if concept_name is not None:
        if not isinstance(concept_name, (Code, CodedConcept)):
            raise TypeError(
                'Argument "concept_name" should be of type '
                'pydicom.sr.coding.Code or '
                'highdicom.sr.CodedConcept.'
            )
        dataset.ConceptNameCodeSequence = [
            CodedConcept(
                concept_name.value,
                concept_name.scheme_designator,
                concept_name.meaning,
                concept_name.scheme_version
            )
        ]

    if content_creator_name is not None:
        check_person_name(content_creator_name)
    dataset.ContentCreatorName = content_creator_name

    if content_creator_identification is not None:
        if not isinstance(
            content_creator_identification,
            ContentCreatorIdentificationCodeSequence
        ):
            raise TypeError(
                'Argument "content_creator_identification" must be of type '
                'ContentCreatorIdentificationCodeSequence.'
            )
        dataset.ContentCreatorIdentificationCodeSequence = \
            content_creator_identification

    # Not technically part of PR IODs, but we include anyway
    now = datetime.datetime.now()
    dataset.ContentDate = DA(now.date())
    dataset.ContentTime = TM(now.time())


def _add_presentation_state_relationship_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset]
) -> None:
    """Add attributes of module Presentation State Relationship.

    Also perform checks that the referenced images are suitable.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced

    """
    # Assert referenced images are from the same series and have the same size
    ref_im = referenced_images[0]
    ref_im_items_mapping = defaultdict(list)
    for im in referenced_images:
        if im.Rows != ref_im.Rows or im.Columns != ref_im.Columns:
            raise ValueError(
                'All referenced images must have the same dimensions.'
            )
        item = Dataset()
        item.ReferencedSOPClassUID = im.SOPClassUID
        item.ReferencedSOPInstanceUID = im.SOPInstanceUID
        ref_im_items_mapping[im.SeriesInstanceUID].append(item)

    dataset.ReferencedSeriesSequence = []
    for series_instance_uid, ref_images in ref_im_items_mapping.items():
        item = Dataset()
        item.SeriesInstanceUID = series_instance_uid
        item.ReferencedImageSequence = ref_images
        dataset.ReferencedSeriesSequence.append(item)


def _add_graphic_group_annotation_layer_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset],
    graphic_groups: Optional[Sequence[GraphicGroup]] = None,
    graphic_annotations: Optional[Sequence[GraphicAnnotation]] = None,
    graphic_layers: Optional[Sequence[GraphicLayer]] = None
) -> None:
    """Add attributes of modules Graphic Group/Annotation/Layer.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced
    graphic_groups: Union[Sequence[highdicom.pr.GraphicGroup], None], optional
        Description of graphic groups used in this presentation state.
    graphic_annotations: Union[Sequence[highdicom.pr.GraphicAnnotation], None], optional
        Graphic annotations to include in this presentation state.
    graphic_layers: Union[Sequence[highdicom.pr.GraphicLayer], None], optional
        Graphic layers to include in this presentation state. All graphic
        layers referenced in "graphic_annotations" must be included.

    """  # noqa: E501
    # Graphic Group
    group_ids = []
    if graphic_groups is not None:
        for grp in graphic_groups:
            if not isinstance(grp, GraphicGroup):
                raise TypeError(
                    'Items of "graphic_groups" must be of type '
                    'highdicom.pr.GraphicGroup.'
                )
            group_ids.append(grp.graphic_group_id)
        described_groups_ids = set(group_ids)
        if len(described_groups_ids) != len(group_ids):
            raise ValueError(
                'Each item in "graphic_groups" must have a unique graphic '
                'group ID.'
            )
        dataset.GraphicGroupSequence = graphic_groups
    else:
        described_groups_ids = set()

    # Graphic Annotation and Graphic Layer
    ref_images_lut = {
        (ds.SOPClassUID, ds.SOPInstanceUID): ds
        for ds in referenced_images
    }
    if graphic_layers is not None:
        labels = [layer.GraphicLayer for layer in graphic_layers]
        if len(labels) != len(set(labels)):
            raise ValueError(
                'Labels of graphic layers must be unique.'
            )
        labels_unique = set(labels)
        dataset.GraphicLayerSequence = graphic_layers

    if graphic_annotations is not None:
        for i, ann in enumerate(graphic_annotations):
            if not isinstance(ann, GraphicAnnotation):
                raise TypeError(
                    f'Item #{i} of "graphic_annotations" must be of type '
                    'highdicom.pr.GraphicAnnotation.'
                )
            if ann.GraphicLayer not in labels_unique:
                raise ValueError(
                    f'Graphic layer with name "{ann.GraphicLayer}" is '
                    f'referenced in item #{i} of "graphic_annotations", '
                    'but not included "graphic_layers".'
                )
            for item in ann.ReferencedImageSequence:
                uids = (
                    item.ReferencedSOPClassUID,
                    item.ReferencedSOPInstanceUID
                )
                if uids not in ref_images_lut:
                    raise ValueError(
                        f'Instance with SOP Instance UID {uids[1]} and '
                        f'SOP Class UID {uids[0]} is referenced in item #{i} '
                        f'of "graphic_annotations", but not included '
                        'in "referenced_images".'
                    )
            for obj in getattr(ann, 'GraphicObjectSequence', []):
                grp_id = obj.graphic_group_id
                if grp_id is not None:
                    if grp_id not in described_groups_ids:
                        raise ValueError(
                            'Found graphic object with graphic group '
                            f'ID "{grp_id}", but no such group is '
                            'described in the "graphic_groups" '
                            'argument.'
                        )
            for obj in getattr(ann, 'TextObjectSequence', []):
                grp_id = obj.graphic_group_id
                if grp_id is not None:
                    if grp_id not in described_groups_ids:
                        raise ValueError(
                            'Found text object with graphic group ID '
                            f'"{grp_id}", but no such group is '
                            'described in the "graphic_groups" '
                            'argument.'
                        )
        dataset.GraphicAnnotationSequence = graphic_annotations


def _add_displayed_area_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset],
) -> None:
    """Add attributes of module Displayed Area.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced

    """
    # This implements the simplest case - the entire area is selected for
    # display and the selection applies to all referenced images.
    # We may want to generalize this later.
    ref_im = referenced_images[0]
    display_area_item = Dataset()
    display_area_item.PixelOriginInterpretation = 'VOLUME'
    display_area_item.DisplayedAreaTopLeftHandCorner = [1, 1]
    if is_tiled_image(ref_im):
        # In case the images form a multi-resolution pyramid, select the image
        # at lowest resolution (top of the pyramid).
        sorted_images = sorted(
            referenced_images,
            key=lambda im: im.TotalPixelMatrixRows * im.TotalPixelMatrixColumns
        )
        low_res_im = sorted_images[0]
        display_area_item.ReferencedImageSequence = ReferencedImageSequence(
            referenced_images=[low_res_im],
        )
        display_area_item.DisplayedAreaBottomRightHandCorner = [
            low_res_im.TotalPixelMatrixColumns,
            low_res_im.TotalPixelMatrixRows,
        ]
    else:
        display_area_item.DisplayedAreaBottomRightHandCorner = [
            ref_im.Columns,
            ref_im.Rows,
        ]
    display_area_item.PresentationSizeMode = 'SCALE TO FIT'
    display_area_item.PresentationPixelAspectRatio = [1, 1]
    dataset.DisplayedAreaSelectionSequence = [display_area_item]


def _add_modality_lut_attributes(
    dataset: Dataset,
    modality_lut_transformation: ModalityLUTTransformation,
) -> None:
    """Add attributes of module Modality LUT.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    modality_lut_transformation: highdicom.ModalityLUTTransformation
        Description of the Modality LUT Transformation for transforming modality
        dependent into modality independent pixel values

    """
    if not isinstance(modality_lut_transformation, ModalityLUTTransformation):
        raise ValueError(
            'Argument "modality_lut_transformation" must have type '
            'ModalityLUTTransformation.'
        )
    for element in modality_lut_transformation:
        dataset[element.tag] = element


def _get_modality_lut_transformation(
    referenced_images: Sequence[Dataset]
) -> Union[ModalityLUTTransformation, None]:
    """Get Modality LUT Transformation from the referenced images.

    Parameters
    ----------
    referenced_images: Sequence[pydicom.Dataset]
        The referenced images from which the attributes should be copied.

    Returns
    -------
    Union[highdicom.ModalityLUTTransformation, None]
        Description of the Modality LUT Transformation for transforming modality
        dependent into modality independent pixel values. None if no such
        attributes are found in the referenced images.

    Raises
    ------
    ValueError
        In case the presence or value of the RescaleSlope, RescaleIntercept,
        or RescaleType attributes are inconsistent between referenced images.

    """
    # Multframe images
    if any(hasattr(im, 'NumberOfFrames') for im in referenced_images):
        im = referenced_images[0]
        if len(referenced_images) > 1 and not is_tiled_image(im):
            raise ValueError(
                "Attributes of Modality LUT module are not available when "
                "multiple images are passed and any of them are multiframe."
            )

        # Check only the Shared Groups, as PRs require all frames to have
        # the same Modality LUT
        slope = None
        intercept = None
        rescale_type = None
        shared_grps = im.SharedFunctionalGroupsSequence[0]
        if hasattr(shared_grps, 'PixelValueTransformationSequence'):
            trans_seq = shared_grps.PixelValueTransformationSequence[0]
            if hasattr(trans_seq, 'RescaleSlope'):
                slope = trans_seq.RescaleSlope
            if hasattr(trans_seq, 'RescaleIntercept'):
                intercept = trans_seq.RescaleIntercept
            if hasattr(trans_seq, 'RescaleType'):
                rescale_type = trans_seq.RescaleType

        # Modality LUT data in the Per Frame Functional Groups will not
        # be copied, but we should check for it rather than silently
        # failing to copy it
        if hasattr(im, 'PerFrameFunctionalGroupsSequence'):
            perframe_grps = im.PerFrameFunctionalGroupsSequence
            if any(
                hasattr(frm_grps, 'PixelValueTransformationSequence')
                for frm_grps in perframe_grps
            ):
                raise ValueError(
                    'This multiframe image contains modality LUT '
                    'table data in the Per-Frame Functional Groups '
                    'Sequence. This is not compatible with the '
                    'Modality LUT module.'
                )

    else:
        have_slopes = [
            hasattr(ds, 'RescaleSlope') for ds in referenced_images
        ]
        have_intercepts = [
            hasattr(ds, 'RescaleIntercept') for ds in referenced_images
        ]
        have_type = [
            hasattr(ds, 'RescaleType') for ds in referenced_images
        ]

        if any(have_slopes) and not all(have_slopes):
            raise ValueError(
                'Error while copying Modality LUT attributes: presence of '
                '"RescaleSlope" is inconsistent among referenced images.'
            )
        if any(have_intercepts) and not all(have_intercepts):
            raise ValueError(
                'Error while copying Modality LUT attributes: presence of '
                '"RescaleIntercept" is inconsistent among referenced '
                'images.'
            )
        if any(have_type) and not all(have_type):
            raise ValueError(
                'Error while copying Modality LUT attributes: presence of '
                '"RescaleType" is inconsistent among referenced images.'
            )

        if all(have_intercepts) != all(have_slopes):
            raise ValueError(
                'Error while copying Modality LUT attributes: datasets '
                'should have both "RescaleIntercept" and "RescaleSlope", '
                'or neither.'
            )

        if all(have_intercepts):
            if any(
                ds.RescaleSlope != referenced_images[0].RescaleSlope
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values '
                    'of "RescaleSlope" are inconsistent among referenced '
                    'images.'
                )
            if any(
                ds.RescaleIntercept != referenced_images[0].RescaleIntercept
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values '
                    'of "RescaleIntercept" are inconsistent among '
                    'referenced images.'
                )
            slope = referenced_images[0].RescaleSlope
            intercept = referenced_images[0].RescaleIntercept
        else:
            slope = None
            intercept = None

        if all(have_type):
            if any(
                ds.RescaleType != referenced_images[0].RescaleType
                for ds in referenced_images
            ):
                raise ValueError(
                    'Error while copying Modality LUT attributes: values '
                    'of "RescaleType" are inconsistent among referenced '
                    'images.'
                )
            rescale_type = referenced_images[0].RescaleType
        else:
            if intercept is None:
                rescale_type = None
            else:
                rescale_type = RescaleTypeValues.HU.value

    if intercept is None:
        return None

    return ModalityLUTTransformation(
        rescale_intercept=intercept,
        rescale_slope=slope,
        rescale_type=rescale_type
    )


def _add_softcopy_voi_lut_attributes(
    dataset: Dataset,
    referenced_images: Sequence[Dataset],
    voi_lut_transformations: Sequence[SoftcopyVOILUTTransformation]
) -> None:
    """Add attributes of module Softcopy VOI LUT.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    referenced_images: Sequence[pydicom.Dataset]
        Images that should be referenced
    voi_lut_transformations: Sequence[highdicom.pr.SoftcopyVOILUTTransformation]
        Description of the VOI LUT Transformation for transforming modality
        pixel values into pixel values that are of interest to a user or an
        application

    """  # noqa: E501
    if len(voi_lut_transformations) == 0:
        raise ValueError(
            'Argument "voi_lut_transformations" must not be empty.'
        )
    for i, v in enumerate(voi_lut_transformations):
        if not isinstance(v, SoftcopyVOILUTTransformation):
            raise TypeError(
                f'Item #{i} of "voi_lut_transformations" must have '
                'highdicom.pr.SoftcopyVOILUTTransformation.'
            )

    if len(voi_lut_transformations) > 1:
        if not all(
            hasattr(v, 'ReferencedImageSequence')
            for v in voi_lut_transformations
        ):
            raise ValueError(
                'If multiple items of argument '
                '"voi_lut_transformations" are passed, '
                'each must reference the images that it applies to.'
            )

    ref_images_lut = {
        (ds.SOPClassUID, ds.SOPInstanceUID): ds
        for ds in referenced_images
    }
    prev_ref_frames = defaultdict(list)
    prev_ref_segs = defaultdict(list)
    for transformation in voi_lut_transformations:
        # If the softcopy VOI LUT references specific images,
        # check that the references are valid
        if hasattr(transformation, 'ReferencedImageSequence'):
            for item in transformation.ReferencedImageSequence:
                uids = (
                    item.ReferencedSOPClassUID,
                    item.ReferencedSOPInstanceUID
                )
                if uids not in ref_images_lut:
                    raise ValueError(
                        f'Instance with SOP Instance UID {uids[1]} and '
                        f'SOP Class UID {uids[0]} is referenced in '
                        'items of "voi_lut_transformations", but not '
                        'included in "referenced_images".'
                    )
                ref_im = ref_images_lut[uids]
                is_multiframe = hasattr(
                    ref_im,
                    'NumberOfFrames',
                )
                if uids in prev_ref_frames and not is_multiframe:
                    raise ValueError(
                        f'Instance with SOP Instance UID {uids[1]} '
                        'is referenced in more than one item of the '
                        '"softcopy_voi_luts".'
                    )
                nframes = getattr(ref_im, 'NumberOfFrames', 1)
                if hasattr(item, 'ReferencedFrameNumber'):
                    ref_frames = item.ReferencedFrameNumber
                    if not isinstance(ref_frames, MultiValue):
                        ref_frames = [ref_frames]
                else:
                    if hasattr(item, 'ReferencedSegmentNumber'):
                        # Do not check frames if segments are specified
                        ref_frames = []
                    else:
                        # If ReferencedFrameNumber is not present, the
                        # reference refers to all frames
                        ref_frames = list(range(1, nframes + 1))

                for f in ref_frames:
                    if f in prev_ref_frames[uids]:
                        raise ValueError(
                            f'Frame {f} in image with SOP Instance '
                            f'UID {uids[1]} is referenced in more '
                            'than one item of the '
                            '"softcopy_voi_luts".'
                        )
                    prev_ref_frames[uids].append(f)

                if hasattr(item, 'ReferencedSegmentNumber'):
                    ref_segs = item.ReferencedSegmentNumber
                    if not isinstance(ref_segs, MultiValue):
                        ref_segs = [ref_segs]

                if hasattr(ref_im, 'SegmentSequence'):
                    nsegments = len(ref_im.SegmentSequence)
                    if not hasattr(item, 'ReferencedSegmentNumber'):
                        ref_segs = list(range(1, nsegments))
                    for s in ref_segs:
                        if s in prev_ref_segs[uids]:
                            raise ValueError(
                                f'Segment {s} in image with SOP '
                                f'Instance  UID {uids[1]} is '
                                'referenced in more than one item of '
                                'the "softcopy_voi_luts".'
                            )
                        prev_ref_segs[uids].append(s)

    dataset.SoftcopyVOILUTSequence = voi_lut_transformations


def _get_softcopy_voi_lut_transformations(
    referenced_images: Sequence[Dataset]
) -> Sequence[SoftcopyVOILUTTransformation]:
    """Get Softcopy VOI LUT Transformation from referenced images.

    Any Window Center, Window Width, Window Explanation, VOI LUT Function,
    or VOI LUT Sequence attributes the referenced images are copied to the
    new sequence.  Missing values will cause no errors, and
    will result in the relevant (optional) attributes being omitted from
    the presentation state object.  Inconsistent values between
    referenced images will result in multiple different items of the
    Softcopy VOI LUT Sequence in the presentation state object.

    Parameters
    ----------
    referenced_images: Sequence[pydicom.Dataset]
        The referenced images from which the attributes should be copied.

    Returns
    -------
    Sequence[highdicom.SoftcopyVOILUTTransformation]
        Dataset containing attributes of module Softcopy VOI LUT

    """
    transformations = []
    if any(hasattr(im, 'NumberOfFrames') for im in referenced_images):
        if len(referenced_images) > 1:
            raise ValueError(
                "If multiple images are passed and any of them are multiframe, "
                "a 'softcopy_voi_lut_transformation' must be explicitly "
                "provided."
            )

        im = referenced_images[0]
        shared_grps = im.SharedFunctionalGroupsSequence[0]
        perframe_grps = im.PerFrameFunctionalGroupsSequence
        if hasattr(shared_grps, 'FrameVOILUTSequence'):
            # Simple case where VOI information is in the Shared functional
            # groups and therefore are consistent between frames
            voi_seq = shared_grps.FrameVOILUTSequence[0]

            softcopy_voi_lut_transformation = SoftcopyVOILUTTransformation(
                window_center=voi_seq.WindowCenter,
                window_width=voi_seq.WindowWidth,
                window_explanation=getattr(
                    voi_seq,
                    'WindowCenterWidthExplanation',
                    None
                ),
                voi_lut_function=getattr(voi_seq, 'VOILUTFunction', None),
            )
            transformations.append(softcopy_voi_lut_transformation)

        else:
            # Check the per-frame functional groups, which may be
            # inconsistent between frames and require multiple entries
            # in the GSPS SoftcopyVOILUTSequence
            by_window = defaultdict(list)
            for frame_number, frm_grp in enumerate(perframe_grps, 1):
                if hasattr(frm_grp, 'FrameVOILUTSequence'):
                    voi_seq = frm_grp.FrameVOILUTSequence[0]
                    # Create unique ID for this VOI lookup as a tuple
                    # of the contents
                    by_window[(
                        voi_seq.WindowWidth,
                        voi_seq.WindowCenter,
                        getattr(
                            voi_seq,
                            'WindowCenterWidthExplanation',
                            None
                        ),
                        getattr(voi_seq, 'VOILUTFunction', None),
                    )].append(frame_number)

            for (width, center, exp, func), frame_list in by_window.items():
                if len(frame_list) == im.NumberOfFrames:
                    # All frames included, no need to include the
                    # referenced frames explicitly
                    refs_to_include = None
                else:
                    # Include specific references
                    refs_to_include = ReferencedImageSequence(
                        referenced_images=referenced_images,
                        referenced_frame_number=frame_list,
                    )

                transformations.append(
                    SoftcopyVOILUTTransformation(
                        window_center=center,
                        window_width=width,
                        window_explanation=exp,
                        voi_lut_function=func,
                        referenced_images=refs_to_include
                    )
                )

    else:  # single frame
        by_window = defaultdict(list)
        by_lut = defaultdict(list)
        for ref_im in referenced_images:
            has_width = hasattr(ref_im, 'WindowWidth')
            has_center = hasattr(ref_im, 'WindowCenter')
            has_lut = hasattr(ref_im, 'VOILUTSequence')

            if has_width != has_center:
                raise ValueError(
                    'Error while copying VOI LUT attributes: found dataset '
                    'with mismatched WindowWidth and WindowCenter '
                    'attributes.'
                )

            if has_width and has_lut:
                raise ValueError(
                    'Error while copying VOI LUT attributes: found dataset '
                    'with both window width/center and VOI LUT Sequence '
                    'attributes.'
                )

            if has_width:
                by_window[(
                    ref_im.WindowWidth,
                    ref_im.WindowCenter,
                    getattr(ref_im, 'WindowCenterWidthExplanation', None),
                    getattr(ref_im, 'VOILUTFunction', None),
                )].append(ref_im)
            elif has_lut:
                # Create a unique identifier for this list of LUTs
                lut_info = []
                for voi_lut in ref_im.VOILUTSequence:
                    lut_info.append((
                        voi_lut.LUTDescriptor[1],
                        voi_lut.LUTDescriptor[2],
                        getattr(voi_lut, 'LUTExplanation', None),
                        voi_lut.LUTData
                    ))
                lut_id = tuple(lut_info)
                by_lut[lut_id].append(ref_im)

        for (width, center, exp, func), im_list in by_window.items():
            if len(im_list) == len(referenced_images):
                # All datasets included, no need to include the referenced
                # images explicitly
                refs_to_include = None
            else:
                # Include specific references
                refs_to_include = ReferencedImageSequence(im_list)

            transformations.append(
                SoftcopyVOILUTTransformation(
                    window_center=center,
                    window_width=width,
                    window_explanation=exp,
                    voi_lut_function=func,
                    referenced_images=refs_to_include
                )
            )

        for lut_id, im_list in by_lut.items():
            if len(im_list) == len(referenced_images):
                # All datasets included, no need to include the referenced
                # images explicitly
                refs_to_include = None
            else:
                # Include specific references
                refs_to_include = ReferencedImageSequence(im_list)

            luts = [
                VOILUT(
                    first_mapped_value=fmv,
                    lut_data=np.frombuffer(
                        data,
                        np.uint8 if ba == 8 else np.uint16
                    ),
                    lut_explanation=exp
                )
                for (fmv, ba, exp, data) in lut_id
            ]
            transformations.append(
                SoftcopyVOILUTTransformation(
                    referenced_images=refs_to_include,
                    voi_luts=luts
                )
            )

    return transformations


def _get_icc_profile(referenced_images: Sequence[Dataset]) -> bytes:
    """Get ICC Profile from a referenced image.

    Parameters
    ----------
    referenced_images: Sequence[pydicom.Dataset]
        Image datasets from which to extract an ICC profile

    Returns
    -------
    bytes
        ICC Profile

    Raises
    ------
    ValueError:
        When no ICC profile is found in any of the referenced images or if
        more than one unique profile is found.

    """
    icc_profiles = []
    for im in referenced_images:
        if hasattr(referenced_images, 'ICCProfile'):
            icc_profiles.append(im.ICCProfile)
        elif hasattr(im, 'OpticalPathSequence'):
            if len(im.OpticalPathSequence) > 1:
                raise ValueError(
                    'Cannot extract ICC Profile from referenced image. '
                    'Color image is expected to contain only a single optical '
                    'path.'
                )
            icc_profiles.append(im.OpticalPathSequence[0].ICCProfile)

    if len(icc_profiles) == 0:
        raise ValueError(
            'Could not find an ICC Profile in any of the referenced images.'
        )
    if len(set(icc_profiles)) > 1:
        raise ValueError(
            'Found more than one ICC Profile in referenced images.'
        )

    return icc_profiles[0]


def _add_icc_profile_attributes(
    dataset: Dataset,
    icc_profile: bytes
) -> None:
    """Add attributes of module ICC Profile.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    icc_profile: bytes
        ICC color profile to include in the presentation state.
        The profile must follow the constraints listed in :dcm:`C.11.15
        <part03/sect_C.11.15.html>`.

    """
    if icc_profile is None:
        raise TypeError('Argument "icc_profile" is required.')

    cms_profile = ImageCmsProfile(BytesIO(icc_profile))
    device_class = cms_profile.profile.device_class.strip()
    if device_class not in ('scnr', 'spac'):
        raise ValueError(
            'The device class of the ICC Profile must be "scnr" or "spac", '
            f'got "{device_class}".'
        )
    color_space = cms_profile.profile.xcolor_space.strip()
    if color_space != 'RGB':
        raise ValueError(
            'The color space of the ICC Profile must be "RGB", '
            f'got "{color_space}".'
        )
    pcs = cms_profile.profile.connection_space.strip()
    if pcs not in ('Lab', 'XYZ'):
        raise ValueError(
            'The profile connection space of the ICC Profile must '
            f'be "Lab" or "XYZ", got "{pcs}".'
        )

    dataset.ICCProfile = icc_profile


def _add_palette_color_lookup_table_attributes(
    dataset: Dataset,
    palette_color_lut_transformation: PaletteColorLUTTransformation
) -> None:
    """Add attributes from the Palette Color Lookup Table module.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    palette_color_lut_transformation: highdicom.PaletteColorLUTTransformation
        Description of the Palette Color LUT Transformation for transforming
        grayscale into RGB color pixel values

    """  # noqa: E501
    if not isinstance(
        palette_color_lut_transformation,
        PaletteColorLUTTransformation
    ):
        raise TypeError(
            'Argument "palette_color_lut_transformation" must be of type '
            'PaletteColorLUTTransformation.'
        )

    for element in palette_color_lut_transformation:
        dataset[element.tag] = element


def _add_softcopy_presentation_lut_attributes(
    dataset: Dataset,
    presentation_lut_transformation: PresentationLUTTransformation,
) -> None:
    """Add attributes of module Softcopy Presentation LUT.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset to which attributes should be added
    presentation_lut_transformation: highdicom.PresentationLUTTransformation
        Description of the Modality LUT Transformation for transforming modality
        dependent into modality independent pixel values

    """
    if not isinstance(
        presentation_lut_transformation,
        PresentationLUTTransformation
    ):
        raise ValueError(
            'Argument "presenation_lut_transformation" must have type '
            'PresentationLUTTransformation.'
        )
    for element in presentation_lut_transformation:
        dataset[element.tag] = element


class AdvancedBlending(Dataset):

    """Class for an item of the Advanced Blending Sequence."""

    def __init__(
        self,
        referenced_images: Sequence[Dataset],
        blending_input_number: int,
        modality_lut_transformation: Optional[
            ModalityLUTTransformation
        ] = None,
        voi_lut_transformations: Optional[
            Sequence[SoftcopyVOILUTTransformation]
        ] = None,
        palette_color_lut_transformation: Optional[
            PaletteColorLUTTransformation
        ] = None,
    ) -> None:
        """

        Parameters
        ----------
        referenced_images: Sequence[pydicom.Dataset]
            Images that should be referenced
        blending_input_number: int
            Relative one-based index of the item for input into the blending
            operation
        modality_lut_transformation: Union[highdicom.ModalityLUTTransformation, None], optional
            Description of the Modality LUT Transformation for transforming modality
            dependent into modality independent pixel values
        voi_lut_transformations: Union[Sequence[highdicom.pr.SoftcopyVOILUTTransformation], None], optional
            Description of the VOI LUT Transformation for transforming
            modality pixel values into pixel values that are of interest to a
            user or an application
        palette_color_lut_transformation: Union[highdicom.PaletteColorLUTTransformation, None], optional
            Description of the Palette Color LUT Transformation for transforming
            grayscale into RGB color pixel values

        """  # noqa: E501
        super().__init__()
        ref_im = referenced_images[0]
        if ref_im.SamplesPerPixel == 1:
            if palette_color_lut_transformation is None:
                raise ValueError(
                    'For advanced blending presentation, if referenced images '
                    'are grayscale a palette color lookup table must be '
                    'provided to pseudo-color the image prior to blending.'
                )
        for im in referenced_images:
            if im.SamplesPerPixel != ref_im.SamplesPerPixel:
                raise ValueError(
                    'For advanced blending presentation, all referenced '
                    'images of an advanced blending item must have the same '
                    'number of samples per pixel.'
                )
            if im.StudyInstanceUID != ref_im.StudyInstanceUID:
                raise ValueError(
                    'For advanced blending presentation, all referenced '
                    'images of an advanced blending item must be part of the '
                    'same study.'
                )
            if im.SeriesInstanceUID != ref_im.SeriesInstanceUID:
                raise ValueError(
                    'For advanced blending presentation, all referenced '
                    'images of an advanced blending item must be part of the '
                    'same series.'
                )

        self.BlendingInputNumber = blending_input_number

        ref_im = referenced_images[0]
        ref_series_uid = ref_im.SeriesInstanceUID
        ref_im_seq = []
        for im in referenced_images:
            series_uid = im.SeriesInstanceUID
            if series_uid != ref_series_uid:
                raise ValueError(
                    'All referenced images must belong to the same series.'
                )
            if not is_tiled_image(im):
                if im.Rows != ref_im.Rows or im.Columns != ref_im.Columns:
                    raise ValueError(
                        'All referenced images must have the same dimensions.'
                    )
            ref_im_item = Dataset()
            ref_im_item.ReferencedSOPClassUID = im.SOPClassUID
            ref_im_item.ReferencedSOPInstanceUID = im.SOPInstanceUID
            ref_im_seq.append(ref_im_item)
        self.ReferencedImageSequence = ref_im_seq
        self.StudyInstanceUID = ref_im.StudyInstanceUID
        self.SeriesInstanceUID = ref_im.SeriesInstanceUID

        if modality_lut_transformation is not None:
            _add_modality_lut_attributes(
                self,
                modality_lut_transformation=modality_lut_transformation
            )
        else:
            modality_lut_transformation = _get_modality_lut_transformation(
                referenced_images
            )
            if modality_lut_transformation is None:
                logger.debug(
                    'no Modality LUT attributes found in referenced images'
                )
            else:
                logger.debug(
                    'use Modality LUT attributes from referenced images'
                )
                _add_modality_lut_attributes(
                    self,
                    modality_lut_transformation=modality_lut_transformation
                )

        # Softcopy VOI LUT
        if voi_lut_transformations is not None:
            if len(voi_lut_transformations) == 0:
                raise ValueError(
                    'Argument "voi_lut_transformations" must not be '
                    'empty.'
                )
            for v in voi_lut_transformations:
                if not isinstance(v, SoftcopyVOILUTTransformation):
                    raise TypeError(
                        'Items of argument "voi_lut_transformations" '
                        'must be of type SoftcopyVOILUTTransformation.'
                    )

            if len(voi_lut_transformations) > 1:
                if not all(
                    hasattr(v, 'ReferencedImageSequence')
                    for v in voi_lut_transformations
                ):
                    raise ValueError(
                        'If argument "voi_lut_transformations" '
                        'contains multiple items, each item must reference the '
                        'images that it applies to.'
                    )
            _add_softcopy_voi_lut_attributes(
                self,
                referenced_images=referenced_images,
                voi_lut_transformations=voi_lut_transformations
            )
        else:
            voi_lut_transformations = _get_softcopy_voi_lut_transformations(
                referenced_images
            )
            if len(voi_lut_transformations) > 0:
                logger.debug('use VOI LUT attributes from referenced images')
                _add_softcopy_voi_lut_attributes(
                    self,
                    referenced_images=referenced_images,
                    voi_lut_transformations=voi_lut_transformations
                )
            else:
                logger.debug('no VOI LUT attributes found in referenced images')

        # Palette Color Lookup Table
        palette_color_lut_item = Dataset()
        _add_palette_color_lookup_table_attributes(
            palette_color_lut_item,
            palette_color_lut_transformation=palette_color_lut_transformation
        )
        self.PaletteColorLookupTableSequence = [palette_color_lut_item]


class BlendingDisplayInput(Dataset):

    """Class for an item of the Blending Display Input Sequence attribute."""

    def __init__(
        self,
        blending_input_number: int
    ) -> None:
        """

        Parameters
        ----------
        blending_input_number: int
            One-based identification index number of the input series to which
            the blending information should be applied

        """
        super().__init__()
        self.BlendingInputNumber = blending_input_number


class BlendingDisplay(Dataset):

    """Class for an item of the Blending Display Sequence attribute."""

    def __init__(
        self,
        blending_mode: Union[BlendingModeValues, str],
        blending_display_inputs: Sequence[BlendingDisplayInput],
        blending_input_number: Optional[int] = None,
        relative_opacity: Optional[float] = None,
    ) -> None:
        """

        Parameters
        ----------
        blending_mode: Union[str, highdicom.pr.BlendingModeValues]
            Method for weighting the different input images during the blending
            operation using alpha composition with premultiplication
        blending_display_inputs: Sequence[highdicom.pr.BlendingDisplayInput]
            Inputs for the blending operation. The order of items determines
            the order in which images will be blended.
        blending_input_number: Union[int, None], optional
            One-based identification index number of the result.  Required if
            the output of the blending operation should not be directly
            displayed but used as input for a subsequent blending operation.
        relative_opacity: Union[float, None], optional
            Relative opacity (alpha value) that should be premultiplied with
            pixel values of the foreground image. Pixel values of the background
            image will be premultilied with 1 - `relative_opacity`.
            Required if `blending_mode` is ``"FOREGROUND"``. Will be ignored
            otherwise.

        """
        super().__init__()
        blending_mode = BlendingModeValues(blending_mode)
        self.BlendingMode = blending_mode.value

        if not isinstance(blending_display_inputs, Sequence):
            raise TypeError(
                'Argument "blending_display_inputs" must be a sequence.'
            )

        if blending_mode == BlendingModeValues.FOREGROUND:
            if len(blending_display_inputs) != 2:
                raise ValueError(
                    'Argument "blending_display_inputs" must contain exactly '
                    'two items if blending mode is "FOREGROUND".'
                )
            if relative_opacity is None:
                raise TypeError(
                    'Argument "relative_opacity" is required if blending mode '
                    'is "FOREGROUND".'
                )
            self.RelativeOpacity = float(relative_opacity)
        elif blending_mode == BlendingModeValues.EQUAL:
            if len(blending_display_inputs) == 0:
                raise ValueError(
                    'Argument "blending_display_input" must contain one or '
                    'more items if blending mode is "EQUAL".'
                )
        for item in blending_display_inputs:
            if not isinstance(item, BlendingDisplayInput):
                raise TypeError(
                    'Items of argument "blending_display_input" must have '
                    'type BlendingDisplayInput.'
                )
        self.BlendingDisplayInputSequence = blending_display_inputs
