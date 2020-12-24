import itertools
from typing import Iterator, List, Optional, Tuple

import numpy as np
from pydicom.dataset import Dataset

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames


def tile_pixel_matrix(
        total_pixel_matrix_rows: int,
        total_pixel_matrix_columns: int,
        rows: int,
        columns: int,
    ) -> Iterator[Tuple[int, int]]:
    """Tiles an image into smaller frames (rectangular regions).

    Parameters
    ----------
    total_pixel_matrix_rows: int
        Number of rows in the Total Pixel Matrix
    total_pixel_matrix_columns: int
        Number of columns in the Total Pixel Matrix
    rows: int
        Number of rows per Frame (tile)
    columns: int
        Number of columns per Frame (tile)

    Returns
    -------
    Iterator
        One-based Column, Row index of each Frame (tile)

    """
    tiles_per_col = int(np.ceil(total_pixel_matrix_rows / rows))
    tiles_per_row = int(np.ceil(total_pixel_matrix_columns / columns))
    tile_row_indices = iter(range(1, tiles_per_col + 1))
    tile_col_indices = iter(range(1, tiles_per_row + 1))
    return itertools.product(tile_col_indices, tile_row_indices)


def compute_plane_position_tiled_full(
        row_index: int,
        column_index: int,
        x_offset: float,
        y_offset: float,
        rows: int,
        columns: int,
        image_orientation: Tuple[float, float, float, float, float, float],
        pixel_spacing: Tuple[float, float],
        slice_thickness: Optional[float] = None,
        spacing_between_slices: Optional[float] = None,
        slice_index: Optional[float] = None
    ) -> PlanePositionSequence:
    """Computes the absolute position of a Frame (image plane) in the
    Frame of Reference defined by the three-dimensional slide coordinate
    system given their relative position in the Total Pixel Matrix.

    This information is not provided in image instances with Dimension
    Orientation Type TILED_FULL and therefore needs to be computed.

    Parameters
    ----------
    row_index: int
        One-based Row index value for a given frame (tile) along the column
        direction of the tiled Total Pixel Matrix, which is defined by
        the second triplet in `image_orientation` (values should be in the
        range [1, *n*], where *n* is the number of tiles per column)
    column_index: int
        One-based Column index value for a given frame (tile) along the row
        direction of the tiled Total Pixel Matrix, which is defined by
        the first triplet in `image_orientation` (values should be in the
        range [1, *n*], where *n* is the number of tiles per row)
    x_offset: float
        X offset of the Total Pixel Matrix in the slide coordinate system
        in millimeters
    y_offset: float
        Y offset of the Total Pixel Matrix in the slide coordinate system
        in millimeters
    rows: int
        Number of rows per Frame (tile)
    columns: int
        Number of columns per Frame (tile)
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction for X, Y, and
        Z axis of the slide coordinate system defined by the Frame of Reference
    pixel_spacing: Tuple[float, float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the row direction (second value: spacing
        between columns, horizontal, left to right, increasing Column index)
    slice_thickness: float, optional
        Thickness of a focal plane in micrometers
    spacing_between_slices: float, optional
        Distance between neighboring focal planes in micrometers
    slice_index: int, optional
        Relative one-based index of the slice in the array of slices
        within the volume

    Returns
    -------
    highdicom.content.PlanePositionSequence
        Positon of the plane in the slide coordinate system

    Raises
    ------
    TypeError
        When only one of `slice_index` and `spacing_between_slices` is provided

    """
    # Offset values are one-based, i.e., the top left pixel in the Total Pixel
    # Matrix has offset (1, 1) rather than (0, 0)
    row_offset_frame = ((row_index - 1) * rows) + 1
    column_offset_frame = ((column_index - 1) * columns) + 1

    provided_3d_params = (
        slice_index is not None,
        spacing_between_slices is not None,
    )
    if not(sum(provided_3d_params) == 0 or sum(provided_3d_params) == 2):
        raise TypeError(
            'None or both of the following parameters need to be provided: '
            '"slice_index", "spacing_between_slices"'
        )
    # These checks are needed for mypy to be able to determine the correct type
    if (slice_index is not None and spacing_between_slices is not None):
        z_offset = float(slice_index - 1) * spacing_between_slices
    else:
        z_offset = 0.0

    # We should only be dealing with planar rotations.
    x, y, z = map_pixel_into_coordinate_system(
        coordinate=(column_offset_frame, row_offset_frame),
        image_position=(x_offset, y_offset, z_offset),
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
    )

    return PlanePositionSequence(
        coordinate_system=CoordinateSystemNames.SLIDE,
        image_position=(x, y, z),
        pixel_matrix_position=(column_offset_frame, row_offset_frame)
    )


def compute_plane_position_slide_per_frame(
        dataset: Dataset
    ) -> List[PlanePositionSequence]:
    """Computes the plane position for each frame in given dataset with
    respect to the slide coordinate system.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        VL Whole Slide Microscopy Image

    Returns
    -------
    List[highdicom.content.PlanePositionSequence]
        Plane Position Sequence per frame

    Raises
    ------
    ValueError
        When `dataset` does not represent a VL Whole Slide Microscopy Image

    """
    if not dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.77.1.6':
        raise ValueError('Expected a VL Whole Slide Microscopy Image')

    image_origin = dataset.TotalPixelMatrixOriginSequence[0]
    image_orientation = (
        float(dataset.ImageOrientationSlide[0]),
        float(dataset.ImageOrientationSlide[1]),
        float(dataset.ImageOrientationSlide[2]),
        float(dataset.ImageOrientationSlide[3]),
        float(dataset.ImageOrientationSlide[4]),
        float(dataset.ImageOrientationSlide[5]),
    )
    tiles_per_column = int(
        np.ceil(dataset.TotalPixelMatrixRows / dataset.Rows)
    )
    tiles_per_row = int(
        np.ceil(dataset.TotalPixelMatrixColumns / dataset.Columns)
    )
    num_focal_planes = getattr(
        dataset,
        'NumberOfFocalPlanes',
        1
    )

    shared_fg = dataset.SharedFunctionalGroupsSequence[0]
    pixel_measures = shared_fg.PixelMeasuresSequence[0]
    pixel_spacing = (
        float(pixel_measures.PixelSpacing[0]),
        float(pixel_measures.PixelSpacing[1]),
    )
    slice_thickness = getattr(
        pixel_measures,
        'SliceThickness',
        1.0
    )
    spacing_between_slices = getattr(
        pixel_measures,
        'SpacingBetweenSlices',
        1.0
    )

    return [
        compute_plane_position_tiled_full(
            row_index=r,
            column_index=c,
            x_offset=image_origin.XOffsetInSlideCoordinateSystem,
            y_offset=image_origin.YOffsetInSlideCoordinateSystem,
            rows=dataset.Rows,
            columns=dataset.Columns,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            spacing_between_slices=spacing_between_slices,
            slice_index=s,
        )
        for s, r, c in itertools.product(
            range(1, num_focal_planes + 1),
            range(1, tiles_per_column + 1),  # column direction, top to bottom
            range(1, tiles_per_row + 1),  # row direction, left to right
        )
    ]


def create_rotation_matrix(
        image_orientation: Tuple[float, float, float, float, float, float],
    ) -> np.ndarray:
    """Builds a rotation matrix.

    Parameters
    ----------
    image_orientation: Tuple[float, float, float, float, float, float]
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
    row_cosines = np.array(image_orientation[:3])
    column_cosines = np.array(image_orientation[3:])
    n = np.cross(row_cosines.T, column_cosines.T)
    return np.column_stack([
        row_cosines,
        column_cosines,
        n
    ])


def compute_rotation(
        image_orientation: Tuple[float, float, float, float, float, float],
        in_degrees: bool = False
    ) -> float:
    """Computes the rotation of the image with respect to the frame of
    reference (patient or slide coordinate system).

    This is only valid if the two coordinate systems are related via a planar
    rotation. Otherwise, an exception is raised.

    Parameters
    ----------
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference
    in_degrees: bool, optional
        Whether angle should be returned in degrees rather than radians

    Returns
    -------
    float
        Angle (in radians or degrees, depending on whether `in_degrees`
        is ``False`` or ``True``, respectively)

    Raises
    ------
    ValueError
        If the provided image orientation is not related to the frame of
        reference coordinate system by a rotation within the image plane.

    """
    if (image_orientation[2] != 0.0) or (image_orientation[5] != 0.0):
        raise ValueError(
            "The provided image orientation is not related to the frame of "
            "reference coordinate system by a simple planar rotation"
        )
    rotation = create_rotation_matrix(image_orientation)
    if rotation[2, 2] < 0.0:
        raise ValueError(
            "The provided image orientation indicates that the image "
            "coordinate system is flipped relative to the frame of "
            "reference coordinate system"
        )
    angle = np.arctan2(-rotation[0, 1], rotation[0, 0])
    if in_degrees:
        return np.degrees(angle)
    else:
        return angle


def build_transform(
        image_position: Tuple[float, float, float],
        image_orientation: Tuple[float, float, float, float, float, float],
        pixel_spacing: Tuple[float, float],
    ) -> np.ndarray:
    """Builds an affine transformation matrix for mapping coordinates in the
    two dimensional pixel matrix into the three dimensional frame of reference.

    Parameters
    ----------
    image_position: Tuple[float, float, float]
        Position of the slice (image or frame) in the Frame of Reference, i.e.,
        the offset of the top left pixel in the pixel matrix from the
        origin of the reference coordinate system along the X, Y, and Z axis
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference
    pixel_spacing: Tuple[float, float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the rows direction (second value: spacing
        between columns: horizontal, left to right, increasing Column index)

    Returns
    -------
    numpy.ndarray
        4 x 4 affine transformation matrix

    """
    x_offset = float(image_position[0])
    y_offset = float(image_position[1])
    z_offset = float(image_position[2])
    translation = np.array([x_offset, y_offset, z_offset])
    rotation = create_rotation_matrix(image_orientation)
    column_spacing = float(pixel_spacing[0])  # column direction (between rows)
    row_spacing = float(pixel_spacing[1])  # row direction (between columns)
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


def build_inverse_transform(
        image_position: Tuple[float, float, float],
        image_orientation: Tuple[float, float, float, float, float, float],
        pixel_spacing: Tuple[float, float],
        slice_spacing: float = 1.0
    ) -> np.ndarray:
    """Builds an inverse of an affine transformation matrix for mapping
    coordinates from the three dimensional frame of reference into the two
    dimensional pixel matrix.

    Parameters
    ----------
    image_position: Tuple[float, float, float]
        Position of the slice (image or frame) in the Frame of Reference, i.e.,
        the offset of the top left pixel in the pixel matrix from the
        origin of the reference coordinate system along the X, Y, and Z axis
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference
    pixel_spacing: Tuple[float, float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the rows direction (second value: spacing
        between columns: horizontal, left to right, increasing Column index)
    slice_spacing: float
        Distance (in the coordinate defined by the Frame of Reference) between
        neighboring slices. Default: 1

    Returns
    -------
    numpy.ndarray
        4 x 4 affine transformation matrix

    """
    x_offset = float(image_position[0])
    y_offset = float(image_position[1])
    z_offset = float(image_position[2])
    translation = np.array([x_offset, y_offset, z_offset])

    rotation = create_rotation_matrix(image_orientation)
    column_spacing = float(pixel_spacing[0])  # column direction (between rows)
    row_spacing = float(pixel_spacing[1])  # row direction (between columns)
    rotation[:, 0] *= row_spacing
    rotation[:, 1] *= column_spacing
    rotation[:, 2] *= slice_spacing
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


def apply_transform(
        coordinate: Tuple[float, float],
        affine: np.ndarray
    ) -> Tuple[float, float, float]:
    """Applies an affine transformation matrix to a pixel matrix coordinate
    to obtain a coordinate in the three-dimensional frame of reference.

    Parameters
    ----------
    coordinate: Tuple[float, float]
        One-based (Column, Row) index of the Total Pixel Matrix in pixel unit.
        Note that these values are one-based and in column-major order, which
        is different from the way NumPy indexes arrays (zero-based and
        row-major order)
    affine: numpy.ndarray
        4 x 4 affine transformation matrix

    Returns
    -------
    Tuple[float, float, float]
        (X, Y, Z) coordinate in the coordinate system defined by the
        Frame of Reference

    """
    column_offset = float(coordinate[0])
    row_offset = float(coordinate[1])
    pixel_matrix_coordinate = np.array([[column_offset, row_offset, 0.0, 1.0]])
    physical_coordinate = np.dot(affine, pixel_matrix_coordinate.T)
    return tuple(physical_coordinate[:3].flatten().tolist())


def apply_inverse_transform(
        coordinate: Tuple[float, float, float],
        affine: np.ndarray
    ) -> Tuple[float, float, float]:
    """Applies the inverse of an affine transformation matrix to a
    coordinate in the three-dimensional frame of reference to obtain a pixel
    matrix coordinate.

    Parameters
    ----------
    coordinate: Tuple[float, float, float]
        (X, Y, Z) coordinate in the coordinate system defined by the
        Frame of Reference
    affine: numpy.ndarray
        4 x 4 affine transformation matrix

    Returns
    -------
    Tuple[float, float, float]
        One-based (Column, Row, Slice) index of the Total Pixel Matrix in pixel unit.
        Note that these values are one-based and in column-major order, which
        is different from the way NumPy indexes arrays (zero-based and
        row-major order). Note that in general, the resulting coordinate may not
        lie within the imaging plane, and consequently the slice index value may be
        non-zero.

    """
    x = float(coordinate[0])
    y = float(coordinate[1])
    z = float(coordinate[2])
    physical_coordinate = np.array([[x, y, z, 1.0]])
    pixel_matrix_coordinate = np.dot(affine, physical_coordinate.T)
    return tuple(pixel_matrix_coordinate[:3].flatten().tolist())


def map_pixel_into_coordinate_system(
        coordinate: Tuple[float, float],
        image_position: Tuple[float, float, float],
        image_orientation: Tuple[float, float, float, float, float, float],
        pixel_spacing: Tuple[float, float],
    ) -> Tuple[float, float, float]:
    """Maps a coordinate in the pixel matrix into the physical coordinate
    system (e.g., Slide or Patient) defined by a frame of reference.

    Parameters
    ----------
    coordinate: Tuple[float, float]
        One-based (Column, Row) index of the Total Pixel Matrix in pixel unit.
        Note that these values are one-based and in column-major order, which
        is different from the way NumPy indexes arrays (zero-based and
        row-major order)
    image_position: Tuple[float, float, float]
        Position of the slice (image or frame) in the Frame of Reference, i.e.,
        the offset of the top left pixel in the pixel matrix from the
        origin of the reference coordinate system along the X, Y, and Z axis
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference
    pixel_spacing: Tuple[float, float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the rows direction (second value: spacing
        between columns: horizontal, left to right, increasing Column index)

    Returns
    -------
    Tuple[float, float, float]
        (X, Y, Z) coordinate in the coordinate system defined by the
        Frame of Reference

    Note
    ----
    This function is a convenient wrapper around ``build_transform()`` and
    ``apply_transform()``. When mapping a large number of coordinates,
    consider using the underlying functions directly for speedup.

    """
    affine = build_transform(
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing
    )
    return apply_transform(coordinate, affine=affine)


def map_coordinate_into_pixel_matrix(
        coordinate: Tuple[float, float, float],
        image_position: Tuple[float, float, float],
        image_orientation: Tuple[float, float, float, float, float, float],
        pixel_spacing: Tuple[float, float],
        slice_spacing: float = 1.0,
    ) -> Tuple[float, float, float]:
    """Maps a coordinate in the physical coordinate system (e.g., Slide or
    Patient) into the pixel matrix.

    Parameters
    ----------
    coordinate: Tuple[float, float, float]
        (X, Y, Z) coordinate in the coordinate system in millimeter unit.
    image_position: Tuple[float, float, float]
        Position of the slice (image or frame) in the Frame of Reference, i.e.,
        the offset of the top left pixel in the pixel matrix from the
        origin of the reference coordinate system along the X, Y, and Z axis
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        Frame of Reference
    pixel_spacing: Tuple[float, float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the rows direction (second value: spacing
        between columns: horizontal, left to right, increasing Column index)
    slice_spacing: float
        Distance (in the coordinate defined by the Frame of Reference) between
        neighboring slices. Default: 1

    Returns
    -------
    Tuple[float, float, float]
        (Column, Row, Slice) coordinate in the Total Pixel Matrix

    Note
    ----
    This function is a convenient wrapper around ``build_inverse_transform()``
    and ``apply_inverse_transform()``. When mapping a large number of
    coordinates, consider using the underlying functions directly for speedup.

    """
    affine = build_inverse_transform(
        image_position=image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        slice_spacing=slice_spacing
    )
    return apply_inverse_transform(coordinate, affine=affine)
