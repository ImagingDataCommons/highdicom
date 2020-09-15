import itertools
from typing import Iterator, Tuple

import numpy as np

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames


def tile_pixel_matrix(
        total_pixel_matrix_rows: int,
        total_pixel_matrix_columns: int,
        rows: int,
        columns: int,
        image_orientation: Tuple[float, float, float, float, float, float]
    ) -> Iterator[Tuple[int, int]]:
    """Tiles an image into smaller frames given the size of the
    total pixel matrix, the size of each frame and the orientation of the image
    with respect to the three-dimensional slide coordinate system.

    Parameters
    ----------
    total_pixel_matrix_rows: int
        Number of rows in the total pixel matrix
    total_pixel_matrix_columns: int
        Number of columns in the total pixel matrix
    rows: int
        Number of rows per tile
    columns: int
        Number of columns per tile
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of row (first triplet) and column (second triplet) direction
        for x, y and z axis of the slide coordinate system

    Returns
    -------
    Iterator
        One-based row, column coordinates of each image tile

    """
    tiles_per_row = int(np.ceil(total_pixel_matrix_rows / rows))
    tiles_per_col = int(np.ceil(total_pixel_matrix_columns / columns))
    if tuple(image_orientation[:3]) == (0.0, -1.0, 0.0):
        tile_row_indices = reversed(range(1, tiles_per_row + 1))
    else:
        tile_row_indices = iter(range(1, tiles_per_row + 1))
    if tuple(image_orientation[3:]) == (-1.0, 0.0, 0.0):
        tile_col_indices = reversed(range(1, tiles_per_col + 1))
    else:
        tile_col_indices = iter(range(1, tiles_per_col + 1))
    return itertools.product(tile_row_indices, tile_col_indices)


def compute_plane_positions_tiled_full(
        row_index: int,
        column_index: int,
        depth_index: int,
        x_offset: float,
        y_offset: float,
        z_offset: float,
        rows: int,
        columns: int,
        image_orientation: Tuple[float, float, float, float, float, float],
        pixel_spacing: Tuple[float, float],
        slice_thickness: float,
        spacing_between_slices: float
    ) -> PlanePositionSequence:
    """Computes the absolute position of a plane (frame) in the
    Frame of Reference defined by the three-dimensional slide coordinate system
    given their relative position in the Total Pixel Matrix.

    This information is not provided in image instances with Dimension
    Orientation Type TILED_FULL and therefore needs to be computed.

    Parameters
    ----------
    row_index: int
        Relative one-based index value for a given frame along the row
        direction of the the tiled total pixel matrix, which is defined by
        the first triplet in `image_orientation`
    column_index: int
        Relative one-based index value for a given frame along the column
        direction of the the tiled total pixel matrix, which is defined by
        the second triplet in `image_orientation`
    depth_index: int
        Relative one-based index value for a given frame along the depth
        direction from the glass slide to the coverslip (focal plane)
    x_offset: float
        X offset of the total pixel matrix in the slide coordinate system
        in millimeters
    y_offset: float
        Y offset of the total pixel matrix in the slide coordinate system
        in millimeters
    z_offset: float
        Z offset of the total pixel matrix (focal plane) in the slide
        coordinate system in micrometers
    rows: int
        Number of rows per tile
    columns: int
        Number of columns per tile
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of row (first triplet) and column (second triplet) direction
        for x, y and z axis of the slide coordinate system
    pixel_spacing: Tuple[float, float]
        Physical distance between the centers of neighboring pixels along
        the row and column direction
    slice_thickness: float
        Thickness of a focal plane in micrometers
    spacing_between_slices: float
        Distance between neighboring focal planes in micrometers

    Returns
    -------
    highdicom.content.PlanePositionSequence
        Positon of the plane in the slide coordinate system

    """
    row_offset_frame = ((row_index - 1) * rows) + 1
    column_offset_frame = ((column_index - 1) * columns) + 1

    # We should only be dealing with planar rotations.
    x, y, z = map_pixel_to_physical_coordinate(
        pixel_offset=(column_offset_frame, row_offset_frame),
        image_position=(x_offset, y_offset, z_offset),
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=spacing_between_slices
    )

    return PlanePositionSequence(
        coordinate_system=CoordinateSystemNames.SLIDE,
        image_position=(x, y, z),
        pixel_matrix_position=(row_offset_frame, column_offset_frame)
    )


def map_pixel_to_physical_coordinate(
    coordinate: Tuple[float, float],
    image_position: Tuple[float, float, float],
    image_orientation: Tuple[float, float, float, float, float, float],
    pixel_spacing: Tuple[float, float],
    spacing_between_slices: float
) -> Tuple[float, float, float]:
    """Maps a coordinate in the Total Pixel Matrix into the physical coordinate
    system (e.g., Slide or Patient) defined by the Frame of Reference.

    Parameters
    ----------
    coordinate: Tuple[float, float]
        (Column, Row) coordinate of a point relative to the Total Pixel Matrix
        in pixel unit
    image_position: Tuple[float, float, float]
        Position of the Total Pixel Matrix in the Frame of Reference, i.e.,
        the offset of the top left pixel in the Total Pixel Matrix from the
        origin of the reference coordinate system along the X, Y and Z axis
        in the unit of this coordinate system (e.g., millimeter)
    image_orientation: Tuple[float, float, float, float, float, float]
        Orientation of the Total Pixel Matrix relative to the Frame of
        Reference, i.e., the direction cosines of the first row and column of
        the Total Pixel Matrix with respect to the three axes (X, Y, Z) of the
        referenced coordinate system
    pixel_spacing: Tuple[float, float]
        Spacing between neighboring pixels in the Frame of Reference along the
        Column and Row dimension of the Total Pixel Matrix in the unit of the
        referenced coordinate system
    spacing_between_slices: float
        Spacing between two neighboring image slices (planes) in the Frame
        of Reference in the unit of the referenced coordinate system

    Returns
    -------
    Tuple[float, float, float]
        (X, Y, Z) coordinate in the coordinage system defined by the
        Frame of Reference

    """
    # Read the below article for further information about the mapping
    # between coordinates in the Total Pixel Matrix and the Frame of Reference:
    # https://nipy.org/nibabel/dicom/dicom_orientation.html
    x_offset = image_position[0]
    y_offset = image_position[1]
    z_offset = image_position[2]
    image_offset = np.array([x_offset, y_offset, z_offset])
    row_direction_cosines = np.array(image_orientation[:3])
    column_direction_cosines = np.array(image_orientation[3:])

    # TODO: slice number
    n = np.cross(column_direction_cosines.T, row_direction_cosines.T)

    # Affine transformation matrix
    mapping = np.concatenate(
        [
            row_direction_cosines[..., None].T * pixel_spacing[1],
            column_direction_cosines[..., None].T * pixel_spacing[0],
            n[..., None].T * spacing_between_slices,
            image_offset[..., None].T,
        ]
    ).T
    mapping = np.concatenate([mapping, np.array([[0.0, 0.0, 0.0, 1.0]])])

    column_offset = float(coordinate[0])
    row_offset = float(coordinate[1])
    pixel_matrix_coordinate = np.array([[column_offset, row_offset, 0.0, 1.0]])

    physical_coordinate = np.dot(mapping, pixel_matrix_coordinate.T)
    x = physical_coordinate[0][0]
    y = physical_coordinate[1][0]
    z = physical_coordinate[2][0]

    return (x, y, z)
