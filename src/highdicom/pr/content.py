"""Data Elements that are specific to the Presentation State IODs."""

from typing import Optional, Union, Sequence, Tuple

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
from pydicom._storage_sopclass_uids import (
    SegmentationStorage,
    VLWholeSlideMicroscopyImageStorage
)

import numpy as np

from highdicom.color import CIELabColor
from highdicom.pr.enum import (
    AnnotationUnitsValues,
    GraphicTypeValues,
    TextJustificationValues,
)
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID
from highdicom.valuerep import (
    _check_code_string,
    _check_long_string,
    _check_long_text,
    _check_short_text
)


class ContentCreatorIdentificationCodeSequence(DataElementSequence):

    """

    Sequence identifying the person who created the content.

    """
    def __init__(
        self,
        person_identification_codes: Sequence[Union[Code, CodedConcept]],
        institution_name: str,
        person_address: Optional[str] = None,
        person_telephone_numbers: Optional[Sequence[str]] = None,
        person_telecom_information: Optional[str] = None,
        institution_code: Union[Code, CodedConcept, None] = None,
        institution_address: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        institutional_department_type_code: Union[
            Code,
            CodedConcept,
            None
        ] = None,
    ):
        """

        Parameters
        ----------
        person_identification_codes: Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None]]
            Coded description(s) identifying the person.
        institution_name: str
            Name of the to which the identified individual is responsible or
            accountable.
        person_address: Union[str, None]
            Mailing address of the person.
        person_telephone_numbers: Union[Sequence[str], None]
            Person's telephone number(s).
        person_telecom_information: Union[str, None]
            The person's telecommunication contact information, including
            email or other addresses.
        institution_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None]
            Coded concept identifying the institution.
        institution_address: Union[str, None]
            Mailing address of the institution.
        institutional_department_name: Union[str, None]
            Name of the department, unit or service within the healthcare
            facility.
        institutional_department_type_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None]
            A coded description of the type of Department or Service.

        """  # noqa: E501
        super().__init__()
        item = Dataset()

        if len(person_identification_codes) < 1:
            raise ValueError(
                'Sequence "person_identification_codes" must not be empty.'
            )
        for name_code in person_identification_codes:
            if not isinstance(name_code, (Code, CodedConcept)):
                raise TypeError(
                    'Items of "person_identification_codes" must be of type '
                    'pydicom.sr.coding.Code or '
                    'highdicom.sr.CodedConcept.'
                )
        item.PersonIdentificationCodeSequence = [
            CodedConcept.from_code(c) for c in person_identification_codes
        ]

        _check_long_string(institution_name)
        item.InstitutionName = institution_name

        if institution_code is not None:
            if not isinstance(institution_code, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "institution_code" must be of type '
                    'pydicom.sr.coding.Code or '
                    'highdicom.sr.CodedConcept.'
                )
            item.InstitutionCodeSequence = [institution_code]

        if person_address is not None:
            _check_short_text(person_address)
            item.PersonAddress = person_address

        if person_telephone_numbers is not None:
            if len(person_telephone_numbers) < 1:
                raise ValueError(
                    'Sequence "person_telephone_numbers" must not be empty.'
                )
            for phone_number in person_telephone_numbers:
                _check_long_string(phone_number)
            item.PersonTelephoneNumbers = person_telephone_numbers

        if person_telecom_information is not None:
            _check_long_text(person_telecom_information)
            item.PersonTelecomInformation = person_telecom_information

        if institution_address is not None:
            _check_short_text(institution_address)
            item.InstitutionAddress = institution_address

        if institutional_department_name is not None:
            _check_long_string(institutional_department_name)
            item.InstitutionalDepartmentName = institutional_department_name

        if institutional_department_type_code is not None:
            if not isinstance(
                institutional_department_type_code,
                (Code, CodedConcept)
            ):
                raise TypeError(
                    'Argument "institutional_department_type_code" must be of '
                    'type pydicom.sr.coding.Code or '
                    'highdicom.sr.CodedConcept.'
                )
            item.InstitutionalDepartmentTypeCodeSequence = [
                CodedConcept.from_code(institutional_department_type_code)
            ]

        self.append(item)


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
        description: Union[str, None]
            A description of the contents of this graphic layer.
        display_color: Union[CIELabColor, None]
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
            self.GraphicLayerRecommendedDisplayCIELabValue = display_color.value


class GraphicGroup(Dataset):

    """Dataset describing a grouping of annotations.

    Note
    ----
    Graphic Groups are an independent concept from Graphic Layers. Where a
    Graphic Layer (:class:`highdicom.pr.GraphicLayer`) specifies which
    annotations are rendered first, a Graphic Group specifies which annotations
    belong together and shall be handled together (e.g., rotate, move)
    independent of the Graphic Layer to which they are assigned.

    Each annotation (:class:`highdicom.pr.GraphicObject` or
    :class:`highdicom.pr.TextObject`) may optionally be assigned to a
    GraphicGroup upon construction (whereas assignment to a
    :class:`highdicom.pr.GraphicLayer` is required.)

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
        description: Union[str, None]
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
        units: Union[AnnotationUnitsValues, str] = AnnotationUnitsValues.PIXEL,
        filled: bool = False,
        tracking_id: Optional[str] = None,
        tracking_uid: Optional[str] = None,
        graphic_group: Optional[GraphicGroup] = None,
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
            raise TypeError('graphic_data must be a numpy array.')
        if graphic_data.ndim != 2:
            raise ValueError('graphic_data must be a 2D array.')
        if graphic_data.shape[1] != 2:
            raise ValueError('graphic_data must be an array of shape (N, 2).')
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
            GraphicTypeValues.INTERPOLATED,
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

        if graphic_group is not None:
            if not isinstance(graphic_group, GraphicGroup):
                raise TypeError(
                    'Argument "graphic_group" ID should be of type '
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
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        anchor_point: Optional[Tuple[float, float]] = None,
        units: Union[AnnotationUnitsValues, str] = AnnotationUnitsValues.PIXEL,
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
        bounding_box: Optional[Tuple[float, float, float, float]]
            Coordinates of the bounding box in which the text should be
            displayed, given in the following order [left, top, right, bottom],
            where 'left' and 'right' are the horizontal offsets of the left and
            right sides of the box, respectively, and 'top' and 'bottom' are
            the vertical offsets of the upper and lower sides of the box.
        anchor_point: Optional[Tuple[float, float]]
            Location of a point in the image to which the text value is related,
            given as a (Column, Row) pair.
        units: Union[highdicom.pr.AnnotationUnitsValues, str]
            The units in which the coordinates of the bounding box and/or
            anchor point are is expressed.
        anchor_point_visible: bool
            Whether the relationship between the anchor point and the text
            should be displayed in the image, for example via a line or arrow.
            This parameter is ignored if the anchor_point is not provided.
        tracking_id: str
            User defined text identifier for tracking this finding or feature.
            Shall be unique within the domain in which it is used.
        tracking_uid: str
            Unique identifier for tracking this finding or feature.
        graphic_group: Union[highdicom.pr.GraphicGroup, None]
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
                    'Argument "graphic_group" ID should be of type '
                    'highdicom.pr.GraphicGroup.'
                )
            self.GraphicGroupID = graphic_group.graphic_group_id

    @property
    def text_value(self) -> Union[str, None]:
        """str: unformatted text value"""
        return self.UnformattedTextValue

    @property
    def bounding_box(self) -> Union[Tuple[float, float, float, float], None]:
        if not hasattr(self, 'BoundingBoxTopLeftHandCorner'):
            return None
        return tuple(self.BoundingBoxTopLeftHandCorner) + tuple(
            self.BoundingBoxBottomRightHandCorner
        )

    @property
    def anchor_point(self) -> Union[Tuple[float, float], None]:
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
        referenced_images: Sequence[Dataset]
            Sequenced of referenced datasets. Graphic and text objects shall be
            rendered on all images in this list.
        graphic_layer: highdicom.pr.GraphicLayer
            Graphic layer to which this annotation should belong.
        referenced_frame_number: Union[int, Sequence[int], None]
            Frame number(s) in a multiframe image upon which annotations shall
            be rendered.
        referenced_segment_number: Union[int, Sequence[int], None]
            Frame number(s) in a multi-frame image upon which annotations shall
            be rendered.
        graphic_objects: Union[Sequence[highdicom.pr.GraphicObject], None]
            Graphic objects to render over the referenced images.
        text_objects: Union[Sequence[highdicom.pr.TextObject], None]
            Text objects to render over the referenced images.

        """
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
                'If datasets are multi-frame, only a single dataset should'
                'be passed.'
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
        else:
            if referenced_frame_number is not None:
                raise TypeError(
                    'Passing a "referenced_frame_number" is not valid with '
                    'single-frame referenced images.'
                )
        if referenced_segment_number is not None:
            if referenced_images[0].SOPClassUID != SegmentationStorage:
                raise TypeError(
                    '"referenced_segment_number" is only valid when the '
                    'referenced image is a segmentation.'
                )
        ref_im_seq = []
        for ref_im in referenced_images:
            if not isinstance(ref_im, Dataset):
                raise TypeError(
                    'Argument "referenced_images" must be a sequence of '
                    'pydicom.Datasets.'
                )
            if ref_im.SeriesInstanceUID != referenced_series_uid:
                raise ValueError(
                    'All referenced images must belong to the same series.'
                )
            ref_im_item = Dataset()
            ref_im_item.ReferencedSOPClassUID = ref_im.SOPClassUID
            ref_im_item.ReferencedSOPInstanceUID = ref_im.SOPInstanceUID

            if referenced_frame_number is not None:

                def check_frame_number(f: int) -> None:
                    if f < 1:
                        raise ValueError(
                            'Frame numbers must be positive integers'
                        )
                    elif f > ref_im.NumberOfFrames:
                        raise ValueError(
                            f'Frame number {f} is invalid for image with '
                            f'{ref_im.NumberOfFrames} frames.'
                        )

                if isinstance(referenced_frame_number, Sequence):
                    for f in referenced_frame_number:
                        check_frame_number(f)
                else:
                    check_frame_number(referenced_frame_number)

                ref_im_item.ReferencedFrameNumber = referenced_frame_number

            if referenced_segment_number is not None:
                n_segments = len(ref_im.SegmentSequence)

                def check_segment_number(f: int) -> None:
                    if f < 1:
                        raise ValueError(
                            'Segment numbers must be positive integers'
                        )
                    elif f > n_segments:
                        raise ValueError(
                            f'Segment number {f} is invalid for image with '
                            f'{n_segments} segments.'
                        )

                if isinstance(referenced_segment_number, Sequence):
                    for f in referenced_segment_number:
                        check_segment_number(f)
                else:
                    check_segment_number(referenced_segment_number)

                ref_im_item.ReferencedSegmentNumber = referenced_segment_number

            ref_im_seq.append(ref_im_item)
        self.ReferencedImageSequence = ref_im_seq

        have_graphics = graphic_objects is not None and len(graphic_objects) > 0
        have_text = text_objects is not None and len(text_objects) > 0
        if not have_graphics and not have_text:
            raise TypeError(
                'Either graphic_objects or text_objects must contain at least '
                'one item.'
            )
        if have_graphics:
            for go in graphic_objects:
                if not isinstance(go, GraphicObject):
                    raise TypeError(
                        'All items in graphic_objects must be of type '
                        'highdicom.pr.GraphicObject'
                    )
                if go.units == AnnotationUnitsValues.MATRIX:
                    sm_uid = VLWholeSlideMicroscopyImageStorage
                    if referenced_images[0].SOPClassUID != sm_uid:
                        raise ValueError(
                            'Graphic Objects may only use MATRIX units if the '
                            'referenced images are VL Whole Slide Microscopy '
                            'images.'
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
                    sm_uid = VLWholeSlideMicroscopyImageStorage
                    if referenced_images[0].SOPClassUID != sm_uid:
                        raise ValueError(
                            'Text Objects may only use MATRIX units if the '
                            'referenced images are VL Whole Slide Microscopy '
                            'images.'
                        )
            self.TextObjectSequence = text_objects
