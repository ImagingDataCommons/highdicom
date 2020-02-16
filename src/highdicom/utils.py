import itertools
from typing import Generator, Tuple

import numpy as np

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames


def tile_pixel_matrix(
        total_pixel_matrix_rows: int,
        total_pixel_matrix_columns: int,
        rows: int,
        columns: int,
        image_orientation: Tuple[float, float, float, float, float, float]
    ) -> Generator:
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
    Generator
        One-based row, column coordinates of each image tile

    """
    tiles_per_row = int(np.ceil(total_pixel_matrix_rows / rows))
    tiles_per_col = int(np.ceil(total_pixel_matrix_columns / columns))
    tile_row_indices = range(1, tiles_per_row + 1)
    if tuple(image_orientation[:3]) == (0.0, -1.0, 0.0):
        tile_row_indices = reversed(tile_row_indices)
    tile_col_indices = range(1, tiles_per_col + 1)
    if tuple(image_orientation[3:]) == (-1.0, 0.0, 0.0):
        tile_col_indices = reversed(tile_col_indices)
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
    three-dimensional slide coordinate system given their relative position
    in the Total Pixel Matrix.
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
    x_offset_image: float
        X offset of the total pixel matrix in the slide coordinate system
    y_offset_image: float
        Y offset of the total pixel matrix in the slide coordinate system
    z_offset_image: float
        Z offset of the total pixel matrix (focal plane) in the slide
        coordinate system
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
        Physical thickness of a focal plane
    spacing_between_slices: float
        Physical distance between neighboring focal planes

    Returns
    -------
    highdicom.content.PlanePositionSequence
        Positon of each plane in the slide coordinate system

    """
    row_offset_frame = ((row_index - 1) * rows) + 1
    column_offset_frame = ((column_index - 1) * columns) + 1
    # We need to take rotation of pixel matrix relative to slide into account.
    # According to the standard, we only have to deal with planar rotations by
    # 180 degrees along the row and/or column direction.
    if tuple([float(v) for v in image_orientation[:3]]) == (0.0, -1.0, 0.0):
        x_func = np.subtract
    else:
        x_func = np.add
    x_offset_frame = x_func(
        x_offset,
        (row_offset_frame * pixel_spacing[1])
    )
    if tuple([float(v) for v in image_orientation[3:]]) == (-1.0, 0.0, 0.0):
        y_func = np.subtract
    else:
        y_func = np.add
    y_offset_frame = y_func(
        y_offset,
        (column_offset_frame * pixel_spacing[0])
    )
    z_offset_frame = np.sum([
        z_offset,
        (float(depth_index - 1) * slice_thickness),
        (float(depth_index - 1) * spacing_between_slices)
    ])
    return PlanePositionSequence(
        coordinate_system=CoordinateSystemNames.SLIDE,
        image_position=(x_offset_frame, y_offset_frame, z_offset_frame),
        pixel_matrix_position=(row_offset_frame, column_offset_frame)
    )
