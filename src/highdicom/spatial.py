from typing import Sequence, Tuple

import numpy as np


def create_rotation_matrix(
    image_orientation: Sequence[float],
) -> np.ndarray:
    """Builds a rotation matrix.

    Parameters
    ----------
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference

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


class ImageToReferenceTransformer(object):

    """Class for transforming coordinates from image to reference space.

    Builds an affine transformation matrix for mapping two dimensional
    pixel matrix coordinates into the three dimensional frame of reference.

    Examples
    --------

    >>> # Create a transformer by specifying the reference space of
    >>> # an image
    >>> transformer = ImageToReferenceTransformer(
    >>>     image_position=[56.0, 34.2, 1.0],
    >>>     image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    >>>     pixel_spacing=[0.5, 0.5]
    >>> )
    >>> # Use the transformer to convert coordinates
    >>> image_coords = np.array([[0.0, 10.0], [5.0, 5.0]])
    >>> ref_coords = transformer(image_coords)
    >>> print(ref_coords)
    >>> # [[56.  39.2  1. ]
    >>> #  [58.5 36.7  1. ]]

    """

    def __init__(
        self,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float],
    ):
        """Constructs transformation object.

        Parameters
        ----------
        image_position: Sequence[float]
            Position of the slice (image or frame) in the Frame of Reference,
            i.e., the offset of the top left pixel in the pixel matrix from the
            origin of the reference coordinate system along the X, Y, and Z
            axis
        image_orientation: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing Column index) and the column direction (second
            triplet: vertical, top to bottom, increasing Row index) direction
            expressed in the three-dimensional patient or slide coordinate
            system defined by the Frame of Reference
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing Row index) and the rows direction (second value:
            spacing between columns: horizontal, left to right, increasing
            Column index)

        Raises
        ------
        TypeError
            When any of the arguments is not a sequence.
        ValueError
            When any of the arguments has an incorrect length.

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
        self._affine = np.row_stack(
            [
                np.column_stack([
                    rotation,
                    translation,
                ]),
                [0.0, 0.0, 0.0, 1.0]
            ]
        )

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix"""
        return self._affine

    def __call__(self, coordinates: np.ndarray) -> np.ndarray:
        """Transform coordinates from image space to frame of reference.

        Applies the affine transformation matrix to a batch of pixel matrix
        coordinates to obtain the corresponding coordinates in the frame of
        reference.

        Parameters
        ----------
        coordinates: numpy.ndarray
            Array of (Column, Row) coordinates in the Total Pixel Matrix in
            pixel unit at sub-pixel resolution. Array should have shape
            ``(n, 2)``, where *n* is the number of coordinates, the first
            column represents the *Column* values and the second column
            represents the *Row* values.

        Returns
        -------
        numpy.ndarray
            Array of (X, Y, Z) coordinates in the coordinate system defined by
            the Frame of Reference. Array has shape ``(n, 3)``, where *n* is
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
        pixel_matrix_coordinates = np.row_stack([
            coordinates.T,
            np.zeros((coordinates.shape[0], ), dtype=float),
            np.ones((coordinates.shape[0], ), dtype=float),
        ])
        physical_coordinates = np.dot(self._affine, pixel_matrix_coordinates)
        return physical_coordinates[:3, :].T


class ReferenceToImageTransformer(object):

    """Class for transforming coordinates from reference to image space.

    Builds an affine transformation matrix for mapping coordinates in the
    three dimensional frame of reference into two-dimensional pixel matrix
    coordinates.

    Examples
    --------

    >>> # Create a transformer by specifying the reference space of
    >>> # an image
    >>> transformer = ReferenceToImageTransformer(
    >>>     image_position=[56.0, 34.2, 1.0],
    >>>     image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    >>>     pixel_spacing=[0.5, 0.5]
    >>> )
    >>>
    >>> # Use the transformer to convert coordinates
    >>> ref_coords = np.array([[56., 39.2,  1. ], [58.5, 36.7, 1.]])
    >>> image_coords = transformer(ref_coords)
    >>>
    >>> print(image_coords)
    >>> # [[ 0. 10.  0.]
    >>> #  [ 5.  5.  0.]]

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
            Position of the slice (image or frame) in the Frame of Reference,
            i.e., the offset of the top left pixel in the pixel matrix from the
            origin of the reference coordinate system along the X, Y, and Z
            axis
        image_orientation: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing Column index) and the column direction (second
            triplet: vertical, top to bottom, increasing Row index) direction
            expressed in the three-dimensional patient or slide coordinate
            system defined by the Frame of Reference
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing Row index) and the rows direction (second value:
            spacing between columns: horizontal, left to right, increasing
            Column index)
        spacing_between_slices: float, optional
            Distance (in the coordinate defined by the Frame of Reference)
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
        self._affine = np.row_stack(
            [
                np.column_stack([
                    inv_rotation,
                    -np.dot(inv_rotation, translation)
                ]),
                [0.0, 0.0, 0.0, 1.0]
            ]
        )

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4 x 4 affine transformation matrix"""
        return self._affine

    def __call__(self, coordinates: np.ndarray) -> np.ndarray:
        """Applies the inverse of an affine transformation matrix to a batch of
        coordinates in the frame of reference to obtain the corresponding pixel
        matrix coordinates.

        Parameters
        ----------
        coordinates: numpy.ndarray
            Array of (X, Y, Z) coordinates in the coordinate system defined by
            the Frame of Reference. Array should have shape ``(n, 3)``, where
            *n* is the number of coordinates, the first column represents the
            *X* offsets, the second column represents the *Y* offsets and the
            third column represents the *Z* offsets

        Returns
        -------
        numpy.ndarray
            Array of (Column, Row, Slice) coordinates, where the
            `Column` and `Row` offsets relate to the Total Pixel Matrix in pixel
            units at sub-pixel resolution and the `Slice` offset represents the
            signed distance of the input coordinate in the direction normal to
            the plane of the Total Pixel Matrix represented in units of the
            given spacing between slices.
            The `Row` and `Column` offsets are constrained by the dimension of
            the Total Pixel Matrix. Note, however, that in general, the
            resulting coordinate may not lie within the imaging plane, and
            consequently the `Slice` offset may be non-zero.

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
        physical_coordinates = np.row_stack([
            coordinates.T,
            np.ones((coordinates.shape[0], ), dtype=float)
        ])
        pixel_matrix_coordinates = np.dot(self._affine, physical_coordinates)
        return pixel_matrix_coordinates[:3, :].T


def map_pixel_into_coordinate_system(
    coordinate: Sequence[float],
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
) -> Tuple[float, float, float]:
    """Maps a coordinate in the pixel matrix into the physical coordinate
    system (e.g., Slide or Patient) defined by the frame of reference.

    Parameters
    ----------
    coordinate: Sequence[float]
        (Column, Row) coordinate in the Total Pixel Matrix in pixel unit at
        sub-pixel resolution.
    image_position: Sequence[float]
        Position of the slice (image or frame) in the Frame of Reference, i.e.,
        the offset of the top left pixel in the Total Pixel Matrix from the
        origin of the reference coordinate system along the X, Y, and Z axis
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference
    pixel_spacing: Sequence[float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the row direction (second value: spacing
        between columns: horizontal, left to right, increasing Column index)

    Returns
    -------
    Tuple[float, float, float]
        (X, Y, Z) coordinate in the coordinate system defined by the
        Frame of Reference

    Note
    ----
    This function is a convenient wrapper around
    ``highdicom.spatial.ImageToReferenceTransformation`` for mapping an
    individual coordinate. When mapping a large number of coordinates, consider
    using this class directly for speedup.

    Raises
    ------
    TypeError
        When `image_position`, `image_orientation` or `pixel_spacing` is not a
        sequence.
    ValueError
        When `image_position`, `image_orientation` or `pixel_spacing` has an
        incorrect length.

    """
    transformer = ImageToReferenceTransformer(
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing
    )
    transformed_coordinates = transformer(np.array([coordinate], dtype=float))
    physical_coordinates = transformed_coordinates[0, :].tolist()
    return (
        physical_coordinates[0],
        physical_coordinates[1],
        physical_coordinates[2],
    )


def map_coordinate_into_pixel_matrix(
    coordinate: Sequence[float],
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    spacing_between_slices: float = 1.0,
) -> Tuple[float, float, float]:
    """Maps a coordinate in the physical coordinate system (e.g., Slide or
    Patient) defined by the frame of reference into the pixel matrix.

    Parameters
    ----------
    coordinate: Sequence[float]
        (X, Y, Z) coordinate in the coordinate system in millimeter unit.
    image_position: Sequence[float]
        Position of the slice (image or frame) in the Frame of Reference, i.e.,
        the offset of the top left pixel in the Total Pixel matrix from the
        origin of the reference coordinate system along the X, Y, and Z axis
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference
    pixel_spacing: Sequence[float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the rows direction (second value: spacing
        between columns: horizontal, left to right, increasing Column index)
    spacing_between_slices: float, optional
        Distance (in the coordinate defined by the Frame of Reference) between
        neighboring slices. Default: ``1.0``

    Returns
    -------
    Tuple[float, float, float]
        (Column, Row, Slice) coordinate, where `Column` and `Row` are pixel
        coordinates in the Total Pixel Matrix, `Slice` represents the signed
        distance of the input coordinate in the direction normal to the plane
        of the Total Pixel Matrix represented in units of the given spacing
        between slices. If the `Slice` offset is ``0.0``, then the input
        coordinate lies in the imaging plane, otherwise it lies off the plane
        of the Total Pixel Matrix and `Column` and `Row` offsets may be
        interpreted as the projections of the input coordinate onto the
        imaging plane.

    Note
    ----
    This function is a convenient wrapper around
    ``build_ref_to_image_transform()`` and ``apply_ref_to_image_transform()``.
    When mapping a large number of coordinates, consider using these underlying
    functions directly for speedup.

    Raises
    ------
    TypeError
        When `image_position`, `image_orientation` or `pixel_spacing` is not a
        sequence.
    ValueError
        When `image_position`, `image_orientation` or `pixel_spacing` has an
        incorrect length.

    """
    transformer = ReferenceToImageTransformer(
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=spacing_between_slices
    )
    transformed_coordinates = transformer(np.array([coordinate], dtype=float))
    pixel_matrix_coordinates = transformed_coordinates[0, :].tolist()
    return (
        pixel_matrix_coordinates[0],
        pixel_matrix_coordinates[1],
        pixel_matrix_coordinates[2],
    )
