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
    image_orientation: Tuple[float, float, float, float, float, float]
) -> Iterator[Tuple[int, int]]:
    """Tiles an image into smaller frames (rectangular regions) given the size
    of the total pixel matrix, the size of each frame and the orientation of
    the image with respect to the three-dimensional slide coordinate system.

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
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of row (first triplet: horizontal, left to right) and
        column (second triplet: vertical, top to bottom) direction
        for X, Y, and Z axis of the slide coordinate system

    Returns
    -------
    Iterator
        One-based row, column coordinates of each Frame (tile)

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
        Relative one-based index value for a given frame along the row
        direction of the tiled Total pixel Matrix, which is defined by
        the first triplet in `image_orientation`
    column_index: int
        Relative one-based index value for a given frame along the column
        direction of the the tiled Total Pixel Matrix, which is defined by
        the second triplet in `image_orientation`
    depth_index: int
        Relative one-based index value for a given Frame along the depth
        direction from the glass slide to the coverslip (focal plane)
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
        Cosines of row (first triplet: horizontal, left to right) and
        column (second triplet: vertical, top to bottom) direction
        for X, Y, and Z axis of the slide coordinate system
    pixel_spacing: Tuple[float, float]
        Physical distance between the centers of neighboring pixels along
        the row (horizontal) and column (vertical) direction in millimeters
    slice_thickness: float, optional
        Thickness of a focal plane in micrometers
    spacing_between_slices: float, optional
        Distance between neighboring focal planes in micrometers
    slice_index: int, optional
        Relative zero-based index of the slice in the array of slices
        within the volume

    Returns
    -------
    highdicom.content.PlanePositionSequence
        Positon of the plane in the slide coordinate system

    """
    # Offset values are one-based, i.e., the top left pixel in the Total Pixel
    # Matrix has offset (1, 1) rather than (0, 0)
    row_offset_frame = ((row_index - 1) * rows) + 1
    column_offset_frame = ((column_index - 1) * columns) + 1

    z_offset = (
        slice_index * slice_thickness +
        slice_index * spacing_between_slices
    )
    # We should only be dealing with planar rotations.
    x, y, z = map_pixel_into_coordinate_system(
        coordinate=(column_offset_frame, row_offset_frame),
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
    row_direction = range(1, tiles_per_row + 1)
    column_direction = range(1, tiles_per_column + 1)
    depth_direction = range(1, num_focal_planes + 1)

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
        for c, r, s in itertools.product(
            row_direction,  # left to right
            column_direction,  # top to bottom
            depth_direction
        )
    ]


def map_pixel_into_coordinate_system(
    coordinate: Tuple[float, float],
    image_position: Tuple[float, float, float],
    image_orientation: Tuple[float, float, float, float, float, float],
    pixel_spacing: Tuple[float, float],
    spacing_between_slices: float = 0.0
) -> Tuple[float, float, float]:
    """Maps a coordinate in the pixel matrix into the physical coordinate
    system (e.g., Slide or Patient) defined by a frame of reference.

    Parameters
    ----------
    coordinate: Tuple[float, float]
        (Column, Row) coordinate of a point relative to the Total Pixel Matrix
        in pixel unit
    image_position: Tuple[float, float, float]
        Position of the slice (image or frame) in the Frame of Reference, i.e.,
        the offset of the top left pixel in the pixel matrix from the
        origin of the reference coordinate system along the X, Y, and Z axis
    image_orientation: Tuple[float, float, float, float, float, float]
        Cosines of row (first triplet: horizontal, left to right) and
        column (second triplet: vertical, top to bottom) direction
        for X, Y, and Z axis of the slide coordinate system
    spacing_between_slices: float, optional
        Spacing between two neighboring image slices (planes) in the Frame
        of Reference in the unit of the referenced coordinate system

    Returns
    -------
    Tuple[float, float, float]
        (X, Y, Z) coordinate in the coordinate system defined by the
        Frame of Reference

    Raises
    ------
    ValueError
        When the X, Y or Z coordinate has a negative value

    """
    # Read the below article for further information about the mapping
    # between coordinates in the pixel matrix and the frame of reference:
    # https://nipy.org/nibabel/dicom/dicom_orientation.html
    x_offset = float(image_position[0])
    y_offset = float(image_position[1])
    z_offset = float(image_position[2])
    image_offset = np.array([x_offset, y_offset, z_offset])
    row_cosines = np.array(image_orientation[:3])
    column_cosines = np.array(image_orientation[3:])
    row_spacing = float(pixel_spacing[0])
    column_spacing = float(pixel_spacing[1])

    n = np.cross(column_cosines.T, row_cosines.T)
    k = n * spacing_between_slices

    f_row = row_cosines * row_spacing
    f_col = column_cosines * column_spacing

    # 4x4 affine transformation matrix
    affine = np.concatenate(
        [
            f_row[..., None],
            f_col[..., None],
            k[..., None],
            image_offset[..., None],
        ],
        axis=1
    )
    affine = np.concatenate(
        [
            affine,
            np.array([[0.0, 0.0, 0.0, 1.0]])
        ],
        axis=0
    )

    column_offset = float(coordinate[0])
    row_offset = float(coordinate[1])
    pixel_matrix_coordinate = np.array([
        [row_offset, column_offset, 0.0, 1.0]
    ])

    physical_coordinate = np.dot(affine, pixel_matrix_coordinate.T)

    x = physical_coordinate[0][0]
    if (x < 0.0):
        raise ValueError('X offset in coordinate system cannot be negative.')

    y = physical_coordinate[1][0]
    if (y < 0.0):
        raise ValueError('Y offset in coordinate system cannot be negative.')

    z = np.sum([n[0] * x_offset, n[1] * y_offset, n[2] * z_offset])
    if (z < 0.0):
        raise ValueError('Z offset in coordinate system cannot be negative.')

    return (x, y, z)
