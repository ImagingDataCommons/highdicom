import itertools
from typing import Generator, Tuple

import numpy as np


def generate_tile_positions(
        total_pixel_matrix_rows: int,
        total_pixel_matrix_columns: int,
        rows: int,
        columns: int,
        image_orientation: Tuple[float, float, float, float, float, float]
    ) -> Generator:
    """Generates the position of planes in a tiled image with dimension
    organization TILED_FULL.

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
        One-based row, column coordinates of tiles

    """
    tiles_per_row = int(np.ceil(total_pixel_matrix_rows / rows))
    tiles_per_col = int(np.ceil(total_pixel_matrix_columns / columns))
    tile_row_indices = range(1, tiles_per_row+1)
    if tuple(image_orientation[:3]) == (0.0, -1.0, 0.0):
        tile_row_indices = reversed(tile_row_indices)
    tile_col_indices = range(1, tiles_per_col+1)
    if tuple(image_orientation[3:]) == (-1.0, 0.0, 0.0):
        tile_col_indices = reversed(tile_col_indices)
    return itertools.product(tile_row_indices, tile_col_indices)


