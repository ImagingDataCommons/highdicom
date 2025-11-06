import itertools
from collections.abc import Sequence
import warnings

import numpy as np
from pydicom.dataset import Dataset

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.spatial import (
    get_image_coordinate_system,
    get_tile_array,
    is_multiframe_image,
    is_tiled_image,
    iter_tiled_full_frame_data,
    map_pixel_into_coordinate_system,
    tile_pixel_matrix,
)


# Several functions that were initially defined in this module were moved to
# highdicom.spatial to consolidate similar functionality and prevent circular
# dependencies. Therefore they are re-exported here for backwards compatibility
__all__ = [
    "tile_pixel_matrix",  # backwards compatibility
    "get_tile_array",  # backwards compatibility
    "iter_tiled_full_frame_data",  # backwards compatibility
    "is_tiled_image",  # backwards compatibility
    "compute_plane_position_slide_per_frame",
    "compute_plane_position_tiled_full",
    "are_plane_positions_tiled_full",
]


def compute_plane_position_tiled_full(
    row_index: int,
    column_index: int,
    x_offset: float,
    y_offset: float,
    rows: int,
    columns: int,
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    slice_thickness: float | None = None,  # unused (deprecated)
    spacing_between_slices: float | None = None,
    slice_index: int | None = None
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
        This parameter is unused and passing anything other than None will
        cause a warning to be issued. Use spacing_between_slices to specify the
        spacing between neighboring slices. This parameter will be removed in a
        future version of the library.
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
    if slice_thickness is not None:
        warnings.warn(
            "Passing a slice_thickness other than None has no effect and "
            "will be deprecated in a future version of the library.",
            UserWarning,
            stacklevel=2,
        )
    if row_index < 1 or column_index < 1:
        raise ValueError("Row and column indices must be positive integers.")
    row_offset_frame = ((row_index - 1) * rows)
    column_offset_frame = ((column_index - 1) * columns)

    provided_3d_params = (
        slice_index is not None,
        spacing_between_slices is not None,
    )
    if sum(provided_3d_params) not in (0, 2):
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


def get_plane_positions_of_image(
    image: Dataset,
) -> Sequence[PlanePositionSequence]:
    """Gets plane positions of frames in multi-frame image.

    Parameters
    ----------
    image: Dataset
        Multi-frame image

    Returns
    -------
    List[highdicom.PlanePositionSequence]
        Plane position of each frame in the image

    """

    is_multiframe = is_multiframe_image(image)
    if not is_multiframe:
        raise ValueError('Argument "image" must be a multi-frame image.')

    coordinate_system = get_image_coordinate_system(image)
    if coordinate_system is None:
        raise ValueError(
            'Cannot calculate plane positions when images do not exist '
            'within a frame of reference.'
        )
    elif coordinate_system == CoordinateSystemNames.SLIDE:
        if hasattr(image, 'PerFrameFunctionalGroupsSequence'):
            plane_positions = [PlanePositionSequence.from_sequence(
                item.PlanePositionSlideSequence
            )
                for item in image.PerFrameFunctionalGroupsSequence
            ]
        else:
            # If Dimension Organization Type is TILED_FULL, plane
            # positions are implicit and need to be computed.
            plane_positions = compute_plane_position_slide_per_frame(image)
    else:
        plane_positions = [
            PlanePositionSequence.from_sequence(item.PlanePositionSequence)
            for item in image.PerFrameFunctionalGroupsSequence
        ]

    return plane_positions


def get_plane_positions_of_series(
    images: Sequence[Dataset],
) -> Sequence[PlanePositionSequence]:
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
    is_multiframe = any([is_multiframe_image(img) for img in images])
    if is_multiframe:
        raise ValueError(
            'Argument "images" must be a series of single-frame images.'
        )

    coordinate_system = get_image_coordinate_system(images[0])
    if coordinate_system is None:
        raise ValueError(
            'Cannot calculate plane positions when images do not exist '
            'within a frame of reference.'
        )
    elif coordinate_system == CoordinateSystemNames.SLIDE:
        plane_positions = []
        for img in images:
            # Unfortunately, the image position is not specified relative to
            # the top left corner but to the center of the image.
            # Therefore, we need to compute the offset and subtract it.
            center_item = img.ImageCenterPointCoordinatesSequence[0]
            x_center = center_item.XOffsetInSlideCoordinateSystem
            y_center = center_item.YOffsetInSlideCoordinateSystem
            z_center = center_item.ZOffsetInSlideCoordinateSystem
            offset_coordinate = map_pixel_into_coordinate_system(
                index=((img.Columns / 2, img.Rows / 2)),
                image_position=(x_center, y_center, z_center),
                image_orientation=img.ImageOrientationSlide,
                pixel_spacing=img.PixelSpacing
            )
            center_coordinate = np.array((0., 0., 0.), dtype=float)
            origin_coordinate = center_coordinate - offset_coordinate
            plane_positions.append(
                PlanePositionSequence(
                    coordinate_system=CoordinateSystemNames.SLIDE,
                    image_position=origin_coordinate,
                    pixel_matrix_position=(1, 1)
                )
            )
    else:
        plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.PATIENT,
                image_position=img.ImagePositionPatient
            )
            for img in images
        ]

    return plane_positions


def compute_plane_position_slide_per_frame(
    dataset: Dataset
) -> list[PlanePositionSequence]:
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


def are_plane_positions_tiled_full(
    plane_positions: Sequence[PlanePositionSequence],
    rows: int,
    columns: int,
) -> bool:
    """Determine whether a list of plane positions matches "TILED_FULL".

    This takes a list of plane positions for each frame and determines whether
    the plane positions satisfy the requirements of "TILED_FULL". Plane
    positions match the TILED_FULL dimension organization type if they are
    non-overlapping, and cover the entire image plane in the order specified in
    the standard.

    The test implemented in this function is necessary and sufficient for the
    use of TILED_FULL in a newly created tiled image (thus allowing the plane
    positions to be omitted from the image and defined implicitly).

    Parameters
    ----------
    plane_positions: Sequence[PlanePositionSequence]
        Plane positions of each frame.
    rows: int
        Number of rows in each frame.
    columns: int
        Number of columns in each frame.

    Returns
    -------
    bool:
        True if the supplied plane positions satisfy the requirements for
        TILED_FULL. False otherwise.

    """
    max_r = -1
    max_c = -1
    for plane_position in plane_positions:
        r = plane_position[0].RowPositionInTotalImagePixelMatrix
        c = plane_position[0].ColumnPositionInTotalImagePixelMatrix
        if r > max_r:
            max_r = r
        if c > max_c:
            max_c = c

    expected_positions = [
        (r, c) for (r, c) in itertools.product(
            range(1, max_r + 1, rows),
            range(1, max_c + 1, columns),
        )
    ]
    if len(expected_positions) != len(plane_positions):
        return False

    for (r_exp, c_exp), plane_position in zip(
        expected_positions,
        plane_positions
    ):
        r = plane_position[0].RowPositionInTotalImagePixelMatrix
        c = plane_position[0].ColumnPositionInTotalImagePixelMatrix
        if r != r_exp or c != c_exp:
            return False

    return True
