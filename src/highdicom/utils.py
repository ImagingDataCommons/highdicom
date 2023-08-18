import itertools
from typing import Iterator, Generator, List, Optional, Sequence, Tuple

import numpy as np
from pydicom.dataset import Dataset

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.spatial import (
    map_pixel_into_coordinate_system,
    PixelToReferenceTransformer,
)


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
        One-based (Column, Row) index of each Frame (tile)

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
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    slice_thickness: Optional[float] = None,
    spacing_between_slices: Optional[float] = None,
    slice_index: Optional[int] = None
) -> PlanePositionSequence:
    """Compute the position of a frame (image plane) in the frame of reference
    defined by the three-dimensional slide coordinate system.

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
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing Column index) and the column direction (second triplet:
        vertical, top to bottom, increasing Row index) direction for X, Y, and
        Z axis of the slide coordinate system defined by the Frame of Reference
    pixel_spacing: Sequence[float]
        Spacing between pixels in millimeter unit along the column direction
        (first value: spacing between rows, vertical, top to bottom,
        increasing Row index) and the row direction (second value: spacing
        between columns, horizontal, left to right, increasing Column index)
    slice_thickness: Union[float, None], optional
        Thickness of a focal plane in micrometers
    spacing_between_slices: Union[float, None], optional
        Distance between neighboring focal planes in micrometers
    slice_index: Union[int, None], optional
        Relative one-based index of the focal plane in the array of focal
        planes within the imaged volume from the slide to the coverslip

    Returns
    -------
    highdicom.PlanePositionSequence
        Position, of the plane in the slide coordinate system

    Raises
    ------
    TypeError
        When only one of `slice_index` and `spacing_between_slices` is provided

    """
    row_offset_frame = ((row_index - 1) * rows)
    column_offset_frame = ((column_index - 1) * columns)

    provided_3d_params = (
        slice_index is not None,
        spacing_between_slices is not None,
    )
    if not (sum(provided_3d_params) == 0 or sum(provided_3d_params) == 2):
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
        index=(column_offset_frame, row_offset_frame),
        image_position=(x_offset, y_offset, z_offset),
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
    )

    return PlanePositionSequence(
        coordinate_system=CoordinateSystemNames.SLIDE,
        image_position=(x, y, z),
        # Position of plane (tile) in Total Pixel Matrix:
        # First tile has position (1, 1)
        pixel_matrix_position=(column_offset_frame + 1, row_offset_frame + 1)
    )


def iter_tiled_full_frame_data(
    dataset: Dataset,
) -> Generator[Tuple[int, int, int, int, float, float, float], None, None]:
    """Get data on the position of each tile in a TILED_FULL image.

    This works only with images with Dimension Organization Type of
    "TILED_FULL".

    Unlike :func:`highdicom.utils.compute_plane_position_slide_per_frame`,
    this functions returns the data in their basic Python types rather than
    wrapping as :class:`highdicom.PlanePositionSequence`

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        VL Whole Slide Microscopy Image or Segmentation Image using the
        "TILED_FULL" DimensionOrganizationType.

    Returns
    -------
    channel: int
        1-based integer index of the "channel". The meaning of "channel"
        depends on the image type. For segmentation images, the channel is the
        segment number. For other images, it is the optical path number.
    focal_plane_index: int
        1-based integer index of the focal plane.
    column_position: int
        1-based column position of the tile (measured left from the left side
        of the total pixel matrix).
    row_position: int
        1-based row position of the tile (measured down from the top of the
        total pixel matrix).
    x: float
        X coordinate in the frame-of-reference coordinate system in millimeter
        units.
    y: float
        Y coordinate in the frame-of-reference coordinate system in millimeter
        units.
    z: float
        Z coordinate in the frame-of-reference coordinate system in millimeter
        units.

    """
    allowed_sop_class_uids = {
        '1.2.840.10008.5.1.4.1.1.77.1.6',  # VL Whole Slide Microscopy Image
        '1.2.840.10008.5.1.4.1.1.66.4',  # Segmentation Image
    }
    if dataset.SOPClassUID not in allowed_sop_class_uids:
        raise ValueError(
            'Expected a VL Whole Slide Microscopy Image or Segmentation Image.'
        )
    if (
        not hasattr(dataset, "DimensionOrganizationType") or
        dataset.DimensionOrganizationType != "TILED_FULL"
    ):
        raise ValueError(
            'Expected an image with "TILED_FULL" dimension organization type.'
        )

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
        'TotalPixelMatrixFocalPlanes',
        1
    )

    is_segmentation = dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'

    # The "channels" output is either segment for segmentations, or optical
    # path for other images
    if is_segmentation:
        num_channels = len(dataset.SegmentSequence)
    else:
        num_channels = getattr(
            dataset,
            'NumberOfOpticalPaths',
            len(dataset.OpticalPathSequence)
        )

    shared_fg = dataset.SharedFunctionalGroupsSequence[0]
    pixel_measures = shared_fg.PixelMeasuresSequence[0]
    pixel_spacing = (
        float(pixel_measures.PixelSpacing[0]),
        float(pixel_measures.PixelSpacing[1]),
    )
    spacing_between_slices = float(
        getattr(
            pixel_measures,
            'SpacingBetweenSlices',
            1.0
        )
    )
    x_offset = image_origin.XOffsetInSlideCoordinateSystem
    y_offset = image_origin.YOffsetInSlideCoordinateSystem

    # Array of tile indices (col_index, row_index)
    tile_indices = np.array(
        [
            (c, r) for (r, c) in
            itertools.product(
                range(1, tiles_per_column + 1),
                range(1, tiles_per_row + 1)
            )
        ]
    )

    # Pixel offsets of each in the total pixel matrix
    frame_pixel_offsets = (
        (tile_indices - 1) * np.array([dataset.Columns, dataset.Rows])
    )

    for channel in range(1, num_channels + 1):
        for slice_index in range(1, num_focal_planes + 1):
            # These checks are needed for mypy to determine the correct type
            z_offset = float(slice_index - 1) * spacing_between_slices
            transformer = PixelToReferenceTransformer(
                image_position=(x_offset, y_offset, z_offset),
                image_orientation=image_orientation,
                pixel_spacing=pixel_spacing
            )

            reference_coordinates = transformer(frame_pixel_offsets)

            for offsets, coords in zip(
                frame_pixel_offsets,
                reference_coordinates
            ):
                yield (
                    channel,
                    slice_index,
                    int(offsets[0] + 1),
                    int(offsets[1] + 1),
                    float(coords[0]),
                    float(coords[1]),
                    float(coords[2]),
                )


def compute_plane_position_slide_per_frame(
    dataset: Dataset
) -> List[PlanePositionSequence]:
    """Computes the plane position for each frame in given dataset with
    respect to the slide coordinate system for an image using the TILED_FULL
    DimensionOrganizationType.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        VL Whole Slide Microscopy Image or Segmentation Image using the
        "TILED_FULL" DimensionOrganizationType.

    Returns
    -------
    List[highdicom.PlanePositionSequence]
        Plane Position Sequence per frame

    Raises
    ------
    ValueError
        When `dataset` does not represent a VL Whole Slide Microscopy Image or
        Segmentation Image or the image does not use the "TILED_FULL" dimension
        organization type.

    """
    return [
        PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=(x, y, z),
            pixel_matrix_position=(c, r),
        )
        for _, _, c, r, x, y, z in iter_tiled_full_frame_data(dataset)
    ]


def is_tiled_image(dataset: Dataset) -> bool:
    """Determine whether a dataset represents a tiled image.

    Returns
    -------
    bool:
        True if the dataset is a tiled image. False otherwise.

    """
    if (
        hasattr(dataset, 'TotalPixelMatrixRows') and
        hasattr(dataset, 'TotalPixelMatrixColumns') and
        hasattr(dataset, 'NumberOfFrames')
    ):
        return True
    return False
