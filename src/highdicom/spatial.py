from typing import List, Optional, Sequence, Tuple

from pydicom import Dataset
import numpy as np

from highdicom.enum import CoordinateSystemNames
from highdicom._module_utils import is_multiframe_image


def get_image_coordinate_system(
    dataset: Dataset
) -> Optional[CoordinateSystemNames]:
    """Get the coordinate system used by an image.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset representing an image.

    Returns
    --------
    Union[highdicom.enum.CoordinateSystemNames, None]
        Coordinate system used by the image, if any.

    """
    if not hasattr(dataset, "FrameOfReferenceUID"):
        return None

    # Using Container Type Code Sequence attribute would be more
    # elegant, but unfortunately it is a type 2 attribute.
    if (
        hasattr(dataset, 'ImageOrientationSlide') or
        hasattr(dataset, 'ImageCenterPointCoordinatesSequence')
    ):
        return CoordinateSystemNames.SLIDE
    else:
        return CoordinateSystemNames.PATIENT


def _get_spatial_information(
    dataset: Dataset,
    frame_number: Optional[int] = None
) -> Tuple[List[float], List[float], List[float]]:
    """Get spatial information from an image dataset.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset representing an image.
    frame_number: Union[int, None], optional
        Specific 1-based frame number. Required if dataset is a multi-frame image.
        Should be None otherwise.

    Returns
    -------
    image_position: List[float]
        Image position (3 elements) of the dataset or frame.
    image_orientation: List[float]
        Image orientation (6 elements) of the dataset or frame.
    pixel_spacing: List[float]
        Pixel spacing (2 elements) of the dataset or frame.

    """
    coordinate_system = get_image_coordinate_system(dataset)

    if coordinate_system is None:
        raise ValueError(
            'The input "dataset" has no spatial information '
            'as it has no frame of reference.'
        )
    if is_multiframe_image(dataset):
        if frame_number is None:
            raise TypeError(
                'Argument "frame_number" must be specified for a multi-frame '
                'image.'
            )
        shared_seq = dataset.SharedFunctionalGroupsSequence[0]
        frame_seq = dataset.PerFrameFunctionalGroupsSequence[
            frame_number - 1
        ]

        # Find spacing in either shared or per-frame sequences (this logic is
        # the same for patient or slide coordinate system)
        if hasattr(shared_seq, 'PixelMeasuresSequence'):
            spacing = shared_seq.PixelMeasuresSequence[0].PixelSpacing
        elif hasattr(frame_seq, 'PixelMeasuresSequence'):
            spacing = frame_seq.PixelMeasuresSequence[0].PixelSpacing
        else:
            raise ValueError('No pixel measures information found.')

        if coordinate_system == CoordinateSystemNames.SLIDE:
            # Find position in either shared or per-frame sequences
            if hasattr(shared_seq, 'PlanePositionSlideSequence'):
                pos_seq = shared_seq.PlanePositionSlideSequence[0]
            elif hasattr(frame_seq, 'PlanePositionSlideSequence'):
                pos_seq = frame_seq.PlanePositionSlideSequence[0]
            else:
                raise ValueError('No frame position information found.')

            position = [
                pos_seq.XOffsetInSlideCoordinateSystem,
                pos_seq.YOffsetInSlideCoordinateSystem,
                pos_seq.ZOffsetInSlideCoordinateSystem,
            ]

            orientation = dataset.ImageOrientationSlide

        else:  # PATIENT coordinate system (multiframe)
            # Find position in either shared or per-frame sequences
            if hasattr(shared_seq, 'PlanePositionSequence'):
                pos_seq = shared_seq.PlanePositionSequence[0]
            elif hasattr(frame_seq, 'PlanePositionSequence'):
                pos_seq = frame_seq.PlanePositionSequence[0]
            else:
                raise ValueError('No frame position information found.')

            position = pos_seq.ImagePositionPatient

            # Find orientation  in either shared or per-frame sequences
            if hasattr(shared_seq, 'PlaneOrientationSequence'):
                pos_seq = shared_seq.PlaneOrientationSequence[0]
            elif hasattr(frame_seq, 'PlaneOrientationSequence'):
                pos_seq = frame_seq.PlaneOrientationSequence[0]
            else:
                raise ValueError('No frame orientation information found.')

            orientation = pos_seq.ImageOrientationPatient

    else:  # Single-frame image
        if frame_number is not None:
            raise TypeError(
                'Argument "frame_number" must be None for a single-frame '
                "image."
            )
        position = dataset.ImagePositionPatient
        orientation = dataset.ImageOrientationPatient
        spacing = dataset.PixelSpacing

    return position, orientation, spacing


def create_rotation_matrix(
    image_orientation: Sequence[float],
) -> np.ndarray:
    """Builds a rotation matrix.

    Parameters
    ----------
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing column index) and the column direction (second triplet:
        vertical, top to bottom, increasing row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        frame of reference

    Returns
    -------
    numpy.ndarray
        3 x 3 rotation matrix

    """
    if len(image_orientation) != 6:
        raise ValueError('Argument "image_orientation" must have length 6.')
    row_cosines = np.array(image_orientation[:3], dtype=float)
    column_cosines = np.array(image_orientation[3:], dtype=float)
    n = np.cross(row_cosines.T, column_cosines.T)
    return np.column_stack([
        row_cosines,
        column_cosines,
        n
    ])


def _create_affine_transformation_matrix(
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
) -> np.ndarray:
    """Create affine matrix for transformation.

    The resulting transformation matrix maps the center of a pixel identified
    by zero-based integer indices into the frame of reference, i.e., an input
    value of (0, 0) represents the center of the top left hand corner pixel.

    See :dcm:`Equation C.7.6.2.1-1 <part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1>`.

    Parameters
    ----------
    image_position: Sequence[float]
        Position of the slice (image or frame) in the frame of reference, i.e.,
        the offset of the top left hand corner pixel in the pixel matrix from
        the origin of the reference coordinate system along the X, Y, and Z
        axis
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to
        right, increasing column index) and the column direction (second
        triplet: vertical, top to bottom, increasing row index) direction
        expressed in the three-dimensional patient or slide coordinate
        system defined by the frame of reference
    pixel_spacing: Sequence[float]
        Spacing between pixels in millimeter unit along the column
        direction (first value: spacing between rows, vertical, top to
        bottom, increasing row index) and the rows direction (second value:
        spacing between columns: horizontal, left to right, increasing
        column index)

    Returns
    -------
    numpy.ndarray
        4 x 4 affine transformation matrix

    """
    if not isinstance(image_position, Sequence):
        raise TypeError('Argument "image_position" must be a sequence.')
    if len(image_position) != 3:
        raise ValueError('Argument "image_position" must have length 3.')
    if not isinstance(image_orientation, Sequence):
        raise TypeError('Argument "image_orientation" must be a sequence.')
    if len(image_orientation) != 6:
        raise ValueError('Argument "image_orientation" must have length 6.')
    if not isinstance(pixel_spacing, Sequence):
        raise TypeError('Argument "pixel_spacing" must be a sequence.')
    if len(pixel_spacing) != 2:
        raise ValueError('Argument "pixel_spacing" must have length 2.')

    x_offset = float(image_position[0])
    y_offset = float(image_position[1])
    z_offset = float(image_position[2])
    translation = np.array([x_offset, y_offset, z_offset], dtype=float)

    rotation = create_rotation_matrix(image_orientation)
    # Column direction (spacing between rows)
    column_spacing = float(pixel_spacing[0])
    # Row direction (spacing between columns)
    row_spacing = float(pixel_spacing[1])
    rotation[:, 0] *= row_spacing
    rotation[:, 1] *= column_spacing

    # 4x4 transformation matrix
    return np.row_stack(
        [
            np.column_stack([
                rotation,
                translation,
            ]),
            [0.0, 0.0, 0.0, 1.0]
        ]
    )


def _create_inv_affine_transformation_matrix(
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    spacing_between_slices: float = 1.0
) -> np.ndarray:
    """Create affine matrix for inverse transformation.

    The resulting transformation matrix maps a frame of reference coordinate to
    pixel indices, where integer pixel index values represent the center of the
    pixel in the image, i.e., an output value of exactly (0.0, 0.0) represents
    the center of the top left hand corner pixel.

    Parameters
    ----------
    image_position: Sequence[float]
        Position of the slice (image or frame) in the frame of reference, i.e.,
        the offset of the top left hand corner pixel in the pixel matrix from
        the origin of the reference coordinate system along the X, Y, and Z
        axis
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to
        right, increasing column index) and the column direction (second
        triplet: vertical, top to bottom, increasing row index) direction
        expressed in the three-dimensional patient or slide coordinate
        system defined by the frame of reference
    pixel_spacing: Sequence[float]
        Spacing between pixels in millimeter unit along the column
        direction (first value: spacing between rows, vertical, top to
        bottom, increasing row index) and the rows direction (second value:
        spacing between columns: horizontal, left to right, increasing
        column index)
    spacing_between_slices: float, optional
        Distance (in the coordinate defined by the frame of reference)
        between neighboring slices. Default: 1

    Raises
    ------
    TypeError
        When `image_position`, `image_orientation`, or `pixel_spacing` is
        not a sequence.
    ValueError
        When `image_position`, `image_orientation`, or `pixel_spacing` has
        an incorrect length.

    """
    if not isinstance(image_position, Sequence):
        raise TypeError('Argument "image_position" must be a sequence.')
    if len(image_position) != 3:
        raise ValueError('Argument "image_position" must have length 3.')
    if not isinstance(image_orientation, Sequence):
        raise TypeError('Argument "image_orientation" must be a sequence.')
    if len(image_orientation) != 6:
        raise ValueError('Argument "image_orientation" must have length 6.')
    if not isinstance(pixel_spacing, Sequence):
        raise TypeError('Argument "pixel_spacing" must be a sequence.')
    if len(pixel_spacing) != 2:
        raise ValueError('Argument "pixel_spacing" must have length 2.')

    x_offset = float(image_position[0])
    y_offset = float(image_position[1])
    z_offset = float(image_position[2])
    translation = np.array([x_offset, y_offset, z_offset])

    rotation = create_rotation_matrix(image_orientation)
    # Column direction (spacing between rows)
    column_spacing = float(pixel_spacing[0])
    # Row direction (spacing between columns)
    row_spacing = float(pixel_spacing[1])
    rotation[:, 0] *= row_spacing
    rotation[:, 1] *= column_spacing
    rotation[:, 2] *= spacing_between_slices
    inv_rotation = np.linalg.inv(rotation)

    # 4x4 transformation matrix
    return np.row_stack(
        [
            np.column_stack([
                inv_rotation,
                -np.dot(inv_rotation, translation)
            ]),
            [0.0, 0.0, 0.0, 1.0]
        ]
    )


class PixelToReferenceTransformer:

    """Class for transforming pixel indices to reference coordinates.

    This class facilitates the mapping of pixel indices to the pixel matrix of
    an image or an image frame (tile or plane) into the patient or slide
    coordinate system defined by the frame of reference.

    Pixel indices are (column, row) pairs of zero-based integer values, where
    the (0, 0) index is located at the **center** of the top left hand corner
    pixel of the pixel matrix.

    Reference coordinates are (x, y, z) triplets of floating-point values,
    where the (0.0, 0.0, 0.0) point is located at the origin of the frame of
    reference.

    Examples
    --------

    >>> import numpy as np
    >>>
    >>> # Create a transformer by specifying the reference space of
    >>> # an image
    >>> transformer = PixelToReferenceTransformer(
    ...     image_position=[56.0, 34.2, 1.0],
    ...     image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ...     pixel_spacing=[0.5, 0.5])
    >>>
    >>> # Use the transformer to convert coordinates
    >>> pixel_indices = np.array([[0, 10], [5, 5]])
    >>> ref_coords = transformer(pixel_indices)
    >>> print(ref_coords)
    [[56.  39.2  1. ]
     [58.5 36.7  1. ]]

    Warning
    -------
    This class shall not be used to map spatial coordinates (SCOORD)
    to 3D spatial coordinates (SCOORD3D). Use the
    :class:`highdicom.spatial.ImageToReferenceTransformer` class instead.

    """

    def __init__(
        self,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float],
    ):
        """Construct transformation object.

        Parameters
        ----------
        image_position: Sequence[float]
            Position of the slice (image or frame) in the frame of reference,
            i.e., the offset of the top left hand corner pixel in the pixel
            matrix from the origin of the reference coordinate system along the
            X, Y, and Z axis
        image_orientation: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing column index) and the column direction (second
            triplet: vertical, top to bottom, increasing row index) direction
            expressed in the three-dimensional patient or slide coordinate
            system defined by the frame of reference
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing row index) and the rows direction (second value:
            spacing between columns: horizontal, left to right, increasing
            column index)

        Raises
        ------
        TypeError
            When any of the arguments is not a sequence.
        ValueError
            When any of the arguments has an incorrect length.

        """
        self._affine = _create_affine_transformation_matrix(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing
        )

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix"""
        return self._affine

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """Transform image pixel indices to frame of reference coordinates.

        Parameters
        ----------
        indices: numpy.ndarray
            Array of (column, row) zero-based pixel indices in the range
            [0, Columns - 1] and [0, Rows - 1], respectively.
            Array of integer values with shape ``(n, 2)``, where *n* is
            the number of indices, the first column represents the `column`
            index and the second column represents the `row` index.
            The ``(0, 0)`` coordinate is located at the **center** of the top
            left pixel in the total pixel matrix.

        Returns
        -------
        numpy.ndarray
            Array of (x, y, z) coordinates in the coordinate system defined by
            the frame of reference. Array has shape ``(n, 3)``, where *n* is
            the number of coordinates, the first column represents the `x`
            offsets, the second column represents the `y` offsets and the third
            column represents the `z` offsets

        Raises
        ------
        ValueError
            When `indices` has incorrect shape.
        TypeError
            When `indices` don't have integer data type.

        """
        if indices.shape[1] != 2:
            raise ValueError(
                'Argument "indices" must be a two-dimensional array '
                'with shape [n, 2].'
            )
        if indices.dtype.kind not in ('u', 'i'):
            raise TypeError(
                'Argument "indices" must be a two-dimensional array '
                'of integers.'
            )
        pixel_matrix_coordinates = np.row_stack([
            indices.T.astype(float),
            np.zeros((indices.shape[0], ), dtype=float),
            np.ones((indices.shape[0], ), dtype=float),
        ])
        reference_coordinates = np.dot(self._affine, pixel_matrix_coordinates)
        return reference_coordinates[:3, :].T

    @classmethod
    def for_image(
        cls,
        dataset: Dataset,
        frame_number: Optional[int] = None,
    ) -> 'PixelToReferenceTransformer':
        """Construct a transformer for a given image or image frame.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset representing an image.
        frame_number: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame for which to get
            the transformer. This should be provided if and only if the dataset
            is a multi-frame image.

        Returns
        -------
        PixelToReferenceTransformer:
            Transformer object for the given image, or image frame.

        """
        position, orientation, spacing = _get_spatial_information(
            dataset,
            frame_number
        )
        return cls(
            image_position=position,
            image_orientation=orientation,
            pixel_spacing=spacing,
        )


class ReferenceToPixelTransformer:

    """Class for transforming reference coordinates to pixel indices.

    This class facilitates the mapping of coordinates in the patient or slide
    coordinate system defined by the frame of reference into the total pixel
    matrix.

    Reference coordinates are (x, y, z) triplets of floating-point values,
    where the (0.0, 0.0, 0.0) point is located at the origin of the frame of
    reference.

    Pixel indices are (column, row) pairs of zero-based integer values, where
    the (0, 0) index is located at the **center** of the top left hand corner
    pixel of the pixel matrix.

    Examples
    --------

    >>> transformer = ReferenceToPixelTransformer(
    ...     image_position=[56.0, 34.2, 1.0],
    ...     image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ...     pixel_spacing=[0.5, 0.5]
    ... )
    >>>
    >>> ref_coords = np.array([[56., 39.2,  1. ], [58.5, 36.7, 1.]])
    >>> pixel_indices = transformer(ref_coords)
    >>> print(pixel_indices)
    [[ 0 10  0]
     [ 5  5  0]]

    Warning
    -------
    This class shall not be used to map 3D spatial coordinates (SCOORD3D)
    to spatial coordinates (SCOORD). Use the
    :class:`highdicom.spatial.ReferenceToImageTransformer` class instead.

    """

    def __init__(
        self,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float],
        spacing_between_slices: float = 1.0
    ):
        """Construct transformation object.

        Builds an inverse of an affine transformation matrix for mapping
        coordinates from the frame of reference into the two
        dimensional pixel matrix.

        Parameters
        ----------
        image_position: Sequence[float]
            Position of the slice (image or frame) in the frame of reference,
            i.e., the offset of the top left hand corner pixel in the pixel
            matrix from the origin of the reference coordinate system along the
            X, Y, and Z axis
        image_orientation: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing column index) and the column direction (second
            triplet: vertical, top to bottom, increasing row index) direction
            expressed in the three-dimensional patient or slide coordinate
            system defined by the frame of reference
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing row index) and the rows direction (second value:
            spacing between columns: horizontal, left to right, increasing
            column index)
        spacing_between_slices: float, optional
            Distance (in the coordinate defined by the frame of reference)
            between neighboring slices. Default: 1

        Raises
        ------
        TypeError
            When `image_position`, `image_orientation` or `pixel_spacing` is
            not a sequence.
        ValueError
            When `image_position`, `image_orientation` or `pixel_spacing` has
            an incorrect length.

        """
        self._affine = _create_inv_affine_transformation_matrix(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=spacing_between_slices
        )

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4 x 4 affine transformation matrix"""
        return self._affine

    def __call__(self, coordinates: np.ndarray) -> np.ndarray:
        """Transform frame of reference coordinates into image pixel indices.

        Parameters
        ----------
        coordinates: numpy.ndarray
            Array of (x, y, z) coordinates in the coordinate system defined by
            the frame of reference. Array has shape ``(n, 3)``, where *n* is
            the number of coordinates, the first column represents the *X*
            offsets, the second column represents the *Y* offsets and the third
            column represents the *Z* offsets

        Returns
        -------
        numpy.ndarray
            Array of (column, row) zero-based indices at pixel resolution.
            Array of integer values with shape ``(n, 2)``, where *n* is
            the number of indices, the first column represents the `column`
            index and the second column represents the `row` index.
            The ``(0, 0)`` coordinate is located at the **center** of the top
            left pixel in the total pixel matrix.

        Note
        ----
        The returned pixel indices may be negative if `coordinates` fall
        outside of the total pixel matrix.

        Raises
        ------
        ValueError
            When `indices` has incorrect shape.

        """
        if coordinates.shape[1] != 3:
            raise ValueError(
                'Argument "coordinates" must be a two-dimensional array '
                'with shape [n, 3].'
            )
        reference_coordinates = np.row_stack([
            coordinates.T.astype(float),
            np.ones((coordinates.shape[0], ), dtype=float)
        ])
        pixel_matrix_coordinates = np.dot(self._affine, reference_coordinates)
        return np.around(pixel_matrix_coordinates[:3, :].T).astype(int)

    @classmethod
    def for_image(
        cls,
        dataset: Dataset,
        frame_number: Optional[int] = None,
    ) -> 'ReferenceToPixelTransformer':
        """Construct a transformer for a given image or image frame.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset representing an image.
        frame_number: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame for which to get
            the transformer. This should be provided if and only if the dataset
            is a multi-frame image.

        Returns
        -------
        ReferenceToPixelTransformer:
            Transformer object for the given image, or image frame.

        """
        position, orientation, spacing = _get_spatial_information(
            dataset,
            frame_number
        )
        return cls(
            image_position=position,
            image_orientation=orientation,
            pixel_spacing=spacing,
        )


class ImageToReferenceTransformer:

    """Class for transforming coordinates from image to reference space.

    This class facilitates the mapping of image coordinates in the pixel matrix
    of an image or an image frame (tile or plane) into the patient or slide
    coordinate system defined by the frame of reference.
    For example, this class may be used to map spatial coordinates (SCOORD)
    to 3D spatial coordinates (SCOORD3D).

    Image coordinates are (column, row) pairs of floating-point values, where
    the (0.0, 0.0) point is located at the top left corner of the top left hand
    corner pixel of the pixel matrix. Image coordinates have pixel units at
    sub-pixel resolution.

    Reference coordinates are (x, y, z) triplets of floating-point values,
    where the (0.0, 0.0, 0.0) point is located at the origin of the frame of
    reference. Reference coordinates have millimeter units.

    Examples
    --------

    >>> transformer = ImageToReferenceTransformer(
    ...     image_position=[56.0, 34.2, 1.0],
    ...     image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ...     pixel_spacing=[0.5, 0.5]
    ... )
    >>>
    >>> image_coords = np.array([[0.0, 10.0], [5.0, 5.0]])
    >>> ref_coords = transformer(image_coords)
    >>> print(ref_coords)
    [[55.75 38.95  1.  ]
     [58.25 36.45  1.  ]]

    Warning
    -------
    This class shall not be used for pixel indices. Use the
    class:`highdicom.spatial.PixelToReferenceTransformer` class instead.

    """

    def __init__(
        self,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float]
    ):
        """Construct transformation object.

        Parameters
        ----------
        image_position: Sequence[float]
            Position of the slice (image or frame) in the frame of reference,
            i.e., the offset of the top left hand corner pixel in the pixel
            matrix from the origin of the reference coordinate system along the
            X, Y, and Z axis
        image_orientation: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing column index) and the column direction (second
            triplet: vertical, top to bottom, increasing row index) direction
            expressed in the three-dimensional patient or slide coordinate
            system defined by the frame of reference
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing row index) and the rows direction (second value:
            spacing between columns: horizontal, left to right, increasing
            column index)

        Raises
        ------
        TypeError
            When any of the arguments is not a sequence.
        ValueError
            When any of the arguments has an incorrect length.

        """
        affine = _create_affine_transformation_matrix(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing
        )
        correction_affine = np.array([
            [1.0, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._affine = np.dot(affine, correction_affine)

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix"""
        return self._affine

    def __call__(self, coordinates: np.ndarray) -> np.ndarray:
        """Transform image coordinates to frame of reference coordinates.

        Parameters
        ----------
        coordinates: numpy.ndarray
            Array of (column, row) coordinates at sub-pixel resolution in the
            range [0, Columns] and [0, Rows], respectively.
            Array of floating-point values with shape ``(n, 2)``, where *n* is
            the number of coordinates, the first column represents the `column`
            values and the second column represents the `row` values.
            The ``(0.0, 0.0)`` coordinate is located at the top left corner
            of the top left hand corner pixel in the total pixel matrix.

        Returns
        -------
        numpy.ndarray
            Array of (x, y, z) coordinates in the coordinate system defined by
            the frame of reference. Array has shape ``(n, 3)``, where *n* is
            the number of coordinates, the first column represents the *X*
            offsets, the second column represents the *Y* offsets and the third
            column represents the *Z* offsets

        Raises
        ------
        ValueError
            When `coordinates` has incorrect shape.

        """
        if coordinates.shape[1] != 2:
            raise ValueError(
                'Argument "coordinates" must be a two-dimensional array '
                'with shape [n, 2].'
            )
        image_coordinates = np.row_stack([
            coordinates.T.astype(float),
            np.zeros((coordinates.shape[0], ), dtype=float),
            np.ones((coordinates.shape[0], ), dtype=float),
        ])
        reference_coordinates = np.dot(self._affine, image_coordinates)
        return reference_coordinates[:3, :].T

    @classmethod
    def for_image(
        cls,
        dataset: Dataset,
        frame_number: Optional[int] = None,
    ) -> 'ImageToReferenceTransformer':
        """Construct a transformer for a given image or image frame.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset representing an image.
        frame_number: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame for which to get
            the transformer. This should be provided if and only if the dataset
            is a multi-frame image.

        Returns
        -------
        ImageToReferenceTransformer:
            Transformer object for the given image, or image frame.

        """
        position, orientation, spacing = _get_spatial_information(
            dataset,
            frame_number
        )
        return cls(
            image_position=position,
            image_orientation=orientation,
            pixel_spacing=spacing,
        )


class ReferenceToImageTransformer:

    """Class for transforming coordinates from reference to image space.

    This class facilitates the mapping of coordinates in the patient or slide
    coordinate system defined by the frame of reference into the total pixel
    matrix.
    For example, this class may be used to map 3D spatial coordinates (SCOORD3D)
    to spatial coordinates (SCOORD).

    Reference coordinates are (x, y, z) triplets of floating-point values,
    where the (0.0, 0.0, 0.0) point is located at the origin of the frame of
    reference. Reference coordinates have millimeter units.

    Image coordinates are (column, row) pairs of floating-point values, where
    the (0.0, 0.0) point is located at the top left corner of the top left hand
    corner pixel of the pixel matrix. Image coordinates have pixel units at
    sub-pixel resolution.

    Examples
    --------

    >>> # Create a transformer by specifying the reference space of
    >>> # an image
    >>> transformer = ReferenceToImageTransformer(
    ...     image_position=[56.0, 34.2, 1.0],
    ...     image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ...     pixel_spacing=[0.5, 0.5]
    ... )
    >>>
    >>> # Use the transformer to convert coordinates
    >>> ref_coords = np.array([[56., 39.2,  1. ], [58.5, 36.7, 1.]])
    >>> image_coords = transformer(ref_coords)
    >>> print(image_coords)
    [[ 0.5 10.5  0. ]
     [ 5.5  5.5  0. ]]

    Warning
    -------
    This class shall not be used for pixel indices. Use the
    :class:`highdicom.spatial.ReferenceToPixelTransformer` class instead.

    """

    def __init__(
        self,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float],
        spacing_between_slices: float = 1.0
    ):
        """Construct transformation object.

        Builds an inverse of an affine transformation matrix for mapping
        coordinates from the frame of reference into the two
        dimensional pixel matrix.

        Parameters
        ----------
        image_position: Sequence[float]
            Position of the slice (image or frame) in the frame of reference,
            i.e., the offset of the top left hand corner pixel in the pixel
            matrix from the origin of the reference coordinate system along the
            X, Y, and Z axis
        image_orientation: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing column index) and the column direction (second
            triplet: vertical, top to bottom, increasing row index) direction
            expressed in the three-dimensional patient or slide coordinate
            system defined by the frame of reference
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing row index) and the rows direction (second value:
            spacing between columns: horizontal, left to right, increasing
            column index)
        spacing_between_slices: float, optional
            Distance (in the coordinate defined by the frame of reference)
            between neighboring slices. Default: 1

        Raises
        ------
        TypeError
            When `image_position`, `image_orientation` or `pixel_spacing` is
            not a sequence.
        ValueError
            When `image_position`, `image_orientation` or `pixel_spacing` has
            an incorrect length.

        """
        # Image coordinates are shifted relative to pixel matrix indices by
        # 0.5 pixels and we thus have to correct for this shift.
        correction_affine = np.array([
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        affine = _create_inv_affine_transformation_matrix(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=spacing_between_slices
        )
        self._affine = np.dot(correction_affine, affine)

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4 x 4 affine transformation matrix"""
        return self._affine

    def __call__(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the inverse of an affine transformation matrix to a batch of
        coordinates in the frame of reference to obtain the corresponding pixel
        matrix indices.

        Parameters
        ----------
        coordinates: numpy.ndarray
            Array of (x, y, z) coordinates in the coordinate system defined by
            the frame of reference. Array should have shape ``(n, 3)``, where
            *n* is the number of coordinates, the first column represents the
            *X* offsets, the second column represents the *Y* offsets and the
            third column represents the *Z* offsets

        Returns
        -------
        numpy.ndarray
            Array of (column, row, slice) indices, where `column` and `row` are
            zero-based indices to the total pixel matrix and the `slice` index
            represents the signed distance of the input coordinate in the
            direction normal to the plane of the total pixel matrix.
            The `row` and `column` indices are constrained by the dimension of
            the total pixel matrix. Note, however, that in general, the
            resulting coordinate may not lie within the imaging plane, and
            consequently the `slice` offset may be non-zero.

        Raises
        ------
        ValueError
            When `coordinates` has incorrect shape.

        """
        if coordinates.shape[1] != 3:
            raise ValueError(
                'Argument "coordinates" must be a two-dimensional array '
                'with shape [n, 3].'
            )
        reference_coordinates = np.row_stack([
            coordinates.T.astype(float),
            np.ones((coordinates.shape[0], ), dtype=float)
        ])
        image_coordinates = np.dot(self._affine, reference_coordinates)
        return image_coordinates[:3, :].T

    @classmethod
    def for_image(
        cls,
        dataset: Dataset,
        frame_number: Optional[int] = None,
    ) -> 'ReferenceToImageTransformer':
        """Construct a transformer for a given image or image frame.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset representing an image.
        frame_number: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame for which to get
            the transformer. This should be provided if and only if the dataset
            is a multi-frame image.

        Returns
        -------
        ReferenceToImageTransformer:
            Transformer object for the given image, or image frame.

        """
        position, orientation, spacing = _get_spatial_information(
            dataset,
            frame_number
        )
        return cls(
            image_position=position,
            image_orientation=orientation,
            pixel_spacing=spacing,
        )


def map_pixel_into_coordinate_system(
    index: Sequence[int],
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
) -> Tuple[float, float, float]:
    """Map an index to the pixel matrix into the reference coordinate system.

    Parameters
    ----------
    index: Sequence[float]
        (column, row) zero-based index at pixel resolution in the range
        [0, Columns - 1] and [0, Rows - 1], respectively.
    image_position: Sequence[float]
        Position of the slice (image or frame) in the frame of reference, i.e.,
        the offset of the center of top left hand corner pixel in the total
        pixel matrix from the origin of the reference coordinate system along
        the X, Y, and Z axis
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing column index) and the column direction (second triplet:
        vertical, top to bottom, increasing row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        frame of reference
    pixel_spacing: Sequence[float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing row index) and the row direction (second value: spacing
        between columns: horizontal, left to right, increasing column index)

    Returns
    -------
    Tuple[float, float, float]
        (x, y, z) coordinate in the coordinate system defined by the
        frame of reference

    Note
    ----
    This function is a convenient wrapper around
    :class:`highdicom.spatial.PixelToReferenceTransformer` for mapping an
    individual coordinate. When mapping a large number of coordinates, consider
    using this class directly for speedup.

    Raises
    ------
    TypeError
        When `image_position`, `image_orientation`, or `pixel_spacing` is not a
        sequence.
    ValueError
        When `image_position`, `image_orientation`, or `pixel_spacing` has an
        incorrect length.

    """
    transformer = PixelToReferenceTransformer(
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing
    )
    transformed_coordinates = transformer(np.array([index], dtype=int))
    reference_coordinates = transformed_coordinates[0, :].tolist()
    return (
        reference_coordinates[0],
        reference_coordinates[1],
        reference_coordinates[2],
    )


def map_coordinate_into_pixel_matrix(
    coordinate: Sequence[float],
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    spacing_between_slices: float = 1.0,
) -> Tuple[int, int, int]:
    """Map a reference coordinate into an index to the total pixel matrix.

    Parameters
    ----------
    coordinate: Sequence[float]
        (x, y, z) coordinate in the coordinate system in millimeter unit.
    image_position: Sequence[float]
        Position of the slice (image or frame) in the frame of reference, i.e.,
        the offset of the center of top left hand corner pixel in the total
        pixel matrix from the origin of the reference coordinate system along
        the X, Y, and Z axis
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing column index) and the column direction (second triplet:
        vertical, top to bottom, increasing row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        frame of reference
    pixel_spacing: Sequence[float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing row index) and the rows direction (second value: spacing
        between columns: horizontal, left to right, increasing column index)
    spacing_between_slices: float, optional
        Distance (in the coordinate defined by the frame of reference) between
        neighboring slices. Default: ``1.0``

    Returns
    -------
    Tuple[int, int, int]
        (column, row, slice) index, where `column` and `row` are pixel indices
        in the total pixel matrix, `slice` represents the signed distance of
        the input coordinate in the direction normal to the plane of the total
        pixel matrix.  If the `slice` offset is ``0``, then the input
        coordinate lies in the imaging plane, otherwise it lies off the plane
        of the total pixel matrix and `column` and `row` indices may be
        interpreted as the projections of the input coordinate onto the imaging
        plane.

    Note
    ----
    This function is a convenient wrapper around
    :class:`highdicom.spatial.ReferenceToPixelTransformer`.
    When mapping a large number of coordinates, consider using these underlying
    functions directly for speedup.

    Raises
    ------
    TypeError
        When `image_position`, `image_orientation`, or `pixel_spacing` is not a
        sequence.
    ValueError
        When `image_position`, `image_orientation`, or `pixel_spacing` has an
        incorrect length.

    """
    transformer = ReferenceToPixelTransformer(
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=spacing_between_slices
    )
    transformed_coordinates = transformer(np.array([coordinate], dtype=float))
    pixel_matrix_coordinates = transformed_coordinates[0, :].tolist()
    return (
        round(pixel_matrix_coordinates[0]),
        round(pixel_matrix_coordinates[1]),
        round(pixel_matrix_coordinates[2]),
    )


def are_points_coplanar(
    points: np.ndarray,
    tol: float = 1e-5,
) -> bool:
    """Check whether a set of 3D points are coplanar (to within a tolerance).

    Parameters
    ----------
    points: np.ndarray
        Numpy array of shape (n x 3) containing 3D points.
    tol: float
        Tolerance on the distance of the furthest point from the plane of best
        fit.

    Returns
    -------
    bool:
        True if the points are coplanar within a tolerance tol, False
        otherwise. Note that if n < 4, points are always coplanar.

    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Array should have shape (n x 3).")

    n = points.shape[0]
    if n < 4:
        # Any set of three or fewer points is coplanar
        return True

    # Center points by subtracting mean
    c = np.mean(points, axis=0, keepdims=True)
    points_centered = points - c

    # Use a SVD to determine the normal of the plane of best fit, then
    # find maximum deviation from it
    u, _, _ = np.linalg.svd(points_centered.T)
    normal = u[:, -1]
    deviations = normal.T @ points_centered.T
    max_dev = np.abs(deviations).max()
    return max_dev <= tol
