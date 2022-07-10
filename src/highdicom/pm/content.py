from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.sr.coding import CodedConcept
from highdicom.sr.value_types import CodeContentItem
from highdicom.uid import UID
from highdicom.utils import compute_plane_position_slide_per_frame


class RealWorldValueMapping(Dataset):
    """Class representing the Real World Value Mapping Item Macro. """

    def __init__(
        self,
        lut_label: str,
        lut_explanation: str,
        unit: Union[CodedConcept, Code],
        value_range: Union[Tuple[int, int], Tuple[float, float]],
        slope: Optional[Union[int, float]] = None,
        intercept: Optional[Union[int, float]] = None,
        lut_data: Optional[Sequence[float]] = None,
        quantity_definition: Optional[Union[CodedConcept, Code]] = None
    ) -> None:
        """
        Parameters
        ----------
        lut_label: str
            Label (identifier) used to identify transformation. Must be less
            than or equal to 16 characters.
        lut_explanation: str
            Explanation (short description) of the meaning of the transformation
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Unit of the real world values. This may be not applicable, because
            the values may not have a (known) unit. In this case, use
            ``pydicom.sr.codedict.codes.UCUM.NoUnits``.
        value_range: Union[Tuple[int, int], Tuple[float, float]]
            Upper and lower value of range of stored values to which the mapping
            should be restricted. For example, values may be stored as
            floating-point values with double precision, but limited to the
            range ``(-1.0, 1.0)`` or ``(0.0, 1.0)`` or stored as 16-bit
            unsigned integer values but limited to range ``(0, 4094).
            Note that the type of the values in `value_range` is significant
            and is used to determine whether values are stored as integers or
            floating-point values. Therefore, use ``(0.0, 1.0)`` instead of
            ``(0, 1)`` to specify a range of floating-point values.
        slope: Union[int, float, None], optional
            Slope of the linear mapping function applied to values in
            `value_range`.
        intercept: Union[int, float, None], optional
            Intercept of the linear mapping function applied to values in
            `value_range`.
        lut_data: Union[Sequence[int], Sequence[float], None], optional
            Sequence of values to serve as a lookup table for mapping stored
            values into real-world values in case of a non-linear relationship.
            The sequence should contain an entry for each value in the specified
            `value_range` such that
            ``len(sequence) == value_range[1] - value_range[0] + 1``.
            For example, in case of a value range of ``(0, 255)``, the sequence
            shall have ``256`` entries - one for each value in the given range.
        quantity_definition: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Description of the quantity represented by real world values
            (see :dcm:`CID 7180 <part16/sect_CID_7180.html>`
            "Abstract Multi-dimensional Image Model Component Semantics")

        Note
        ----
        Either `slope` and `intercept` or `lut_data` must be specified.
        Specify `slope` and `intercept` if the mapping can be described by a
        linear function. Specify `lut_data` if the relationship between stored
        and real-world values is non-linear. Note, however, that a non-linear
        relationship can only be described for values that are stored as
        integers. Values stored as floating-point numbers must map linearly to
        real-world values.

        """  # noqa: E501
        super().__init__()

        if len(lut_label) > 16:
            raise ValueError(
                'lut_label must be less than or equal to 16 characters, '
                f'given {len(lut_label)}.'
            )

        self.LUTExplanation = str(lut_explanation)
        self.LUTLabel = str(lut_label)

        is_floating_point = any(isinstance(v, float) for v in value_range)
        if lut_data is not None:
            if slope is not None or intercept is not None:
                raise TypeError(
                    'Slope and intercept must be provided if LUT data is not '
                    'provided.'
                )
            if is_floating_point:
                raise ValueError(
                    'Only linear mapping is supported for floating-point '
                    'values. The range of values indicates that values are '
                    'as floating-point rather than integer values.'
                )
            n_actual = len(lut_data)
            n_expected = (int(value_range[1]) - int(value_range[0]) + 1)
            if n_actual != n_expected:
                raise ValueError(
                    'The LUT data sequence contains wrong number of entries: '
                    f'expected n={n_expected}, actual n={n_actual}.'
                )
            self.RealWorldValueLUTData = [float(v) for v in lut_data]
        else:
            if slope is None or intercept is None:
                raise TypeError(
                    'Slope and intercept must not be provided if LUT data is '
                    'provided.'
                )
            self.RealWorldValueSlope = float(slope)
            self.RealWorldValueIntercept = float(intercept)

        if is_floating_point:
            self.DoubleFloatRealWorldValueFirstValueMapped = float(
                value_range[0]
            )
            self.DoubleFloatRealWorldValueLastValueMapped = float(
                value_range[1]
            )
        else:
            self.RealWorldValueFirstValueMapped = int(value_range[0])
            self.RealWorldValueLastValueMapped = int(value_range[1])

        if not isinstance(unit, (CodedConcept, Code)):
            raise TypeError(
                'Argument "unit" must have type CodedConcept or Code.'
            )
        if isinstance(unit, Code):
            unit = CodedConcept(*unit)
        self.MeasurementUnitsCodeSequence = [unit]

        if quantity_definition is not None:
            quantity_item = CodeContentItem(
                name=CodedConcept(
                    value='246205007',
                    scheme_designator='SCT',
                    meaning='Quantity'
                ),
                value=quantity_definition
            )
            self.QuantityDefinitionSequence = [quantity_item]


class DimensionIndexSequence(DataElementSequence):

    """Sequence of data elements describing dimension indices for the patient
    or slide coordinate system based on the Dimension Index functional
    group macro.
    Note
    ----
    The order of indices is fixed.
    """

    def __init__(
        self,
        coordinate_system: Union[str, CoordinateSystemNames]
    ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.CoordinateSystemNames]
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
            imaging
        """
        super().__init__()
        dim_uid = UID()

        self._coordinate_system = CoordinateSystemNames(coordinate_system)
        if self._coordinate_system == CoordinateSystemNames.SLIDE:
            x_axis_index = Dataset()
            x_axis_index.DimensionIndexPointer = tag_for_keyword(
                'XOffsetInSlideCoordinateSystem'
            )
            x_axis_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            x_axis_index.DimensionOrganizationUID = dim_uid
            x_axis_index.DimensionDescriptionLabel = \
                'X Offset in Slide Coordinate System'

            y_axis_index = Dataset()
            y_axis_index.DimensionIndexPointer = tag_for_keyword(
                'YOffsetInSlideCoordinateSystem'
            )
            y_axis_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            y_axis_index.DimensionOrganizationUID = dim_uid
            y_axis_index.DimensionDescriptionLabel = \
                'Y Offset in Slide Coordinate System'

            z_axis_index = Dataset()
            z_axis_index.DimensionIndexPointer = tag_for_keyword(
                'ZOffsetInSlideCoordinateSystem'
            )
            z_axis_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            z_axis_index.DimensionOrganizationUID = dim_uid
            z_axis_index.DimensionDescriptionLabel = \
                'Z Offset in Slide Coordinate System'

            row_dimension_index = Dataset()
            row_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'ColumnPositionInTotalImagePixelMatrix'
            )
            row_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            row_dimension_index.DimensionOrganizationUID = dim_uid
            row_dimension_index.DimensionDescriptionLabel = \
                'Column Position In Total Image Pixel Matrix'

            column_dimension_index = Dataset()
            column_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'RowPositionInTotalImagePixelMatrix'
            )
            column_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            column_dimension_index.DimensionOrganizationUID = dim_uid
            column_dimension_index.DimensionDescriptionLabel = \
                'Row Position In Total Image Pixel Matrix'

            # Organize frames for each segment similar to TILED_FULL, first
            # along the row dimension (column indices from left to right) and
            # then along the column dimension (row indices from top to bottom)
            # of the Total Pixel Matrix.
            self.extend([
                row_dimension_index,
                column_dimension_index,
                x_axis_index,
                y_axis_index,
                z_axis_index,
            ])

        elif self._coordinate_system == CoordinateSystemNames.PATIENT:
            image_position_index = Dataset()
            image_position_index.DimensionIndexPointer = tag_for_keyword(
                'ImagePositionPatient'
            )
            image_position_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSequence'
            )
            image_position_index.DimensionOrganizationUID = dim_uid
            image_position_index.DimensionDescriptionLabel = \
                'Image Position Patient'

            self.append(image_position_index)

        else:
            raise ValueError(
                f'Unknown coordinate system "{self._coordinate_system}"'
            )

    def get_plane_positions_of_image(
        self,
        image: Dataset
    ) -> List[PlanePositionSequence]:
        """Get plane positions of frames in multi-frame image.

        Parameters
        ----------
        image: Dataset
            Multi-frame image

        Returns
        -------
        List[highdicom.PlanePositionSequence]
            Plane position of each frame in the image

        """
        is_multiframe = hasattr(image, 'NumberOfFrames')
        if not is_multiframe:
            raise ValueError('Argument "image" must be a multi-frame image.')

        if self._coordinate_system == CoordinateSystemNames.SLIDE:
            if hasattr(image, 'PerFrameFunctionalGroupsSequence'):
                plane_positions = [
                    item.PlanePositionSlideSequence
                    for item in image.PerFrameFunctionalGroupsSequence
                ]
            else:
                # If Dimension Organization Type is TILED_FULL, plane
                # positions are implicit and need to be computed.
                plane_positions = compute_plane_position_slide_per_frame(
                    image
                )
        else:
            plane_positions = [
                item.PlanePositionSequence
                for item in image.PerFrameFunctionalGroupsSequence
            ]

        return plane_positions

    def get_plane_positions_of_series(
        self,
        images: Sequence[Dataset]
    ) -> List[PlanePositionSequence]:
        """Gets plane positions for series of single-frame images.

        Parameters
        ----------
        images: Sequence[Dataset]
            Series of single-frame images

        Returns
        -------
        List[highdicom.PlanePositionSequence]
            Plane position of each frame in the image

        """
        is_multiframe = any([hasattr(img, 'NumberOfFrames') for img in images])
        if is_multiframe:
            raise ValueError(
                'Argument "images" must be a series of single-frame images.'
            )

        plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.PATIENT,
                image_position=img.ImagePositionPatient
            )
            for img in images
        ]

        return plane_positions

    def get_index_values(
        self,
        plane_positions: Sequence[PlanePositionSequence]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the values of indexed attributes.

        Parameters
        ----------
        plane_positions: Sequence[highdicom.PlanePositionSequence]
            Plane position of frames in a multi-frame image or in a series of
            single-frame images

        Returns
        -------
        dimension_index_values: numpy.ndarray
            2D array of spatial dimension index values
        plane_indices: numpy.ndarray
            1D array of planes indices for sorting frames according to their
            spatial position specified by the dimension index.

        """
        # For each indexed spatial dimension  obtain the value of the attribute
        # that the Dimension Index Pointer points to in the element of the
        # Plane Position Sequence or Plane Position Slide Sequence.
        # In case of the patient coordinate system, this is the Image Position
        # Patient attribute. In case of the slide coordinate system, these are
        # X/Y/Z Offset In Slide Coordinate System and the Column/Row
        # Position in Total Image Pixel Matrix attributes.
        plane_position_values = np.array([
            [
                np.array(p[0][indexer.DimensionIndexPointer].value)
                for indexer in self
            ]
            for p in plane_positions
        ])

        # Build an array that can be used to sort planes according to the
        # Dimension Index Value based on the order of the items in the
        # Dimension Index Sequence.
        _, plane_sort_indices = np.unique(
            plane_position_values,
            axis=0,
            return_index=True
        )

        return (plane_position_values, plane_sort_indices)

    def get_index_position(self, pointer: str) -> int:
        """Get relative position of a given dimension in the dimension index.

        Parameters
        ----------
        pointer: str
            Name of the dimension (keyword of the attribute),
            e.g., ``"XOffsetInSlideCoordinateSystem"``

        Returns
        -------
        int
            Zero-based relative position

        Examples
        --------
        >>> dimension_index = DimensionIndexSequence("SLIDE")
        >>> i = dimension_index.get_index_position("XOffsetInSlideCoordinateSystem")
        >>> x_offsets = dimension_index[i]

        """  # noqa: E501
        indices = [
            i
            for i, indexer in enumerate(self)
            if indexer.DimensionIndexPointer == tag_for_keyword(pointer)
        ]
        if len(indices) == 0:
            raise ValueError(
                f'Dimension index does not contain a dimension "{pointer}".'
            )
        return indices[0]

    def get_index_keywords(self) -> List[str]:
        """Get keywords of attributes that specify the position of planes.

        Returns
        -------
        List[str]
            Keywords of indexed attributes

        """
        return [
            keyword_for_tag(indexer.DimensionIndexPointer)
            for indexer in self
        ]
