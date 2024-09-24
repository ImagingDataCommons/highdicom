import itertools
from typing import Generator, Iterator, List, Optional, Sequence, Tuple

from pydicom import Dataset
import numpy as np

from highdicom.enum import CoordinateSystemNames
from highdicom._module_utils import is_multiframe_image


# Tolerance value used by default in tests for equality
_DEFAULT_TOLERANCE = 1e-5


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


def tile_pixel_matrix(
    total_pixel_matrix_rows: int,
    total_pixel_matrix_columns: int,
    rows: int,
    columns: int,
) -> Iterator[Tuple[int, int]]:
    """Tiles an image into smaller frames (rectangular regions).

    Follows the convention used in image with Dimension Organization Type
    "TILED_FULL" images.

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
    return (
        (c, r) for (r, c) in itertools.product(
            tile_row_indices,
            tile_col_indices
        )
    )


def get_tile_array(
    pixel_array: np.ndarray,
    row_offset: int,
    column_offset: int,
    tile_rows: int,
    tile_columns: int,
    pad: bool = True,
) -> np.ndarray:
    """Extract a tile from a total pixel matrix array.

    Parameters
    ----------
    pixel_array: np.ndarray
        Array representing a total pixel matrix. The first two dimensions
        are treated as the rows and columns, respectively, of the total pixel
        matrix. Any subsequent dimensions are not used but are retained in the
        output array.
    row_offset: int
        Offset of the first row of the requested tile from the top of the total
        pixel matrix (1-based index).
    column_offset: int
        Offset of the first column of the requested tile from the left of the
        total pixel matrix (1-based index).
    tile_rows: int
        Number of rows per tile.
    tile_columns:
        Number of columns per tile.
    pad: bool
        Whether to pad the returned array with zeros at the right and/or bottom
        to ensure that it matches the correct tile size. Otherwise, the returned
        array is not padded and may be smaller than the full tile size.

    Returns
    -------
    np.ndarray:
        Returned pixel array for the requested tile.

    """
    if row_offset < 1 or row_offset > pixel_array.shape[0]:
        raise ValueError(
            "Row offset must be between 1 and the size of dimension 0 of the "
            "pixel array."
        )
    if column_offset < 1 or column_offset > pixel_array.shape[1]:
        raise ValueError(
            "Column offset must be between 1 and the size of dimension 1 of "
            "the pixel array."
        )
    # Move to pythonic 1-based indexing
    row_offset -= 1
    column_offset -= 1
    row_end = row_offset + tile_rows
    if row_end > pixel_array.shape[0]:
        pad_rows = row_end - pixel_array.shape[0]
        row_end = pixel_array.shape[0]
    else:
        pad_rows = 0
    column_end = column_offset + tile_columns
    if column_end > pixel_array.shape[1]:
        pad_columns = column_end - pixel_array.shape[1]
        column_end = pixel_array.shape[1]
    else:
        pad_columns = 0
    # Account for 1-based to 0-based index conversion
    tile_array = pixel_array[row_offset:row_end, column_offset:column_end]
    if pad and (pad_rows > 0 or pad_columns > 0):
        extra_dims = pixel_array.ndim - 2
        padding = [(0, pad_rows), (0, pad_columns)] + [(0, 0)] * extra_dims
        tile_array = np.pad(tile_array, padding)

    return tile_array


def compute_tile_positions_per_frame(
    rows: int,
    columns: int,
    total_pixel_matrix_rows: int,
    total_pixel_matrix_columns: int,
    total_pixel_matrix_image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
) -> List[Tuple[List[int], List[float]]]:
    """Get positions of each tile in a TILED_FULL image.

    A TILED_FULL image is one with DimensionOrganizationType of "TILED_FULL".

    Parameters
    ----------
    rows: int
        Number of rows per tile.
    columns: int
        Number of columns per tile.
    total_pixel_matrix_rows: int
        Number of rows in the total pixel matrix.
    total_pixel_matrix_columns: int
        Number of columns in the total pixel matrix.
    total_pixel_matrix_image_position: Sequence[float]
        Position of the top left pixel of the total pixel matrix in the frame
        of reference. Sequence of length 3.
    image_orientation: Sequence[float]
        Orientation cosines of the total pixel matrix. Sequence of length 6.
    pixel_spacing: Sequence[float]
        Pixel spacing between the (row, columns) in mm. Sequence of length 2.

    Returns
    -------
    List[Tuple[List[int], List[float]]]:
        List with positions for each of the tiles in the tiled image. The
        first tuple contains the (column offset, row offset) values, which
        are one-based offsets of the tile in pixel units from the top left
        of the total pixel matrix. The second tuple contains the image
        position in the frame of reference for the tile.

    """
    if len(total_pixel_matrix_image_position) != 3:
        raise ValueError(
            "Argument 'total_pixel_matrix_image_position' must have length 3."
        )
    if len(image_orientation) != 6:
        raise ValueError(
            "Argument 'image_orientation' must have length 6."
        )
    if len(pixel_spacing) != 2:
        raise ValueError(
            "Argument 'pixel_spacing' must have length 2."
        )

    tiles_per_column = (
        (total_pixel_matrix_columns - 1) // columns + 1
    )
    tiles_per_row = (total_pixel_matrix_rows - 1) // rows + 1

    # N x 2 array of (c, r) tile indices
    tile_indices = np.stack(
        np.meshgrid(
            range(tiles_per_column),
            range(tiles_per_row),
            indexing='xy',
        )
    ).reshape(2, -1).T

    # N x 2 array of (c, r) pixel indices
    pixel_indices = tile_indices * [columns, rows]

    transformer = PixelToReferenceTransformer(
        image_position=total_pixel_matrix_image_position,
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
    )
    image_positions = transformer(pixel_indices)

    # Convert 0-based to 1-based indexing for the output
    pixel_indices += 1

    return list(
        zip(
            pixel_indices.tolist(),
            image_positions.tolist()
        )
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

    for channel in range(1, num_channels + 1):
        for slice_index in range(1, num_focal_planes + 1):
            z_offset = float(slice_index - 1) * spacing_between_slices

            for offsets, coords in compute_tile_positions_per_frame(
                rows=dataset.Rows,
                columns=dataset.Columns,
                total_pixel_matrix_rows=dataset.TotalPixelMatrixRows,
                total_pixel_matrix_columns=dataset.TotalPixelMatrixColumns,
                total_pixel_matrix_image_position=(
                    x_offset, y_offset, z_offset
                ),
                image_orientation=image_orientation,
                pixel_spacing=pixel_spacing
            ):
                yield (
                    channel,
                    slice_index,
                    int(offsets[0]),
                    int(offsets[1]),
                    float(coords[0]),
                    float(coords[1]),
                    float(coords[2]),
                )


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
    frame_number: Optional[int] = None,
    for_total_pixel_matrix: bool = False,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    Optional[float],
]:
    """Get spatial information from an image dataset.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset representing an image.
    frame_number: Union[int, None], optional
        Specific 1-based frame number. Required if dataset is a multi-frame
        image. Should be None otherwise.
    for_total_pixel_matrix: bool, optional
        If True, get spatial information for the total pixel matrix of a tiled
        image. This should only be True if the image is a tiled image and is
        incompatible with specifying a frame number.

    Returns
    -------
    image_position: List[float]
        Image position (3 elements) of the dataset or frame.
    image_orientation: List[float]
        Image orientation (6 elements) of the dataset or frame.
    pixel_spacing: List[float]
        Pixel spacing (2 elements) of the dataset or frame.
    spacing_between_slices: Union[float, None]
        Spacing between adjacent slices, if found in the dataset. Note that
        there is no guarantee in general that slices will be consistently
        spaced or even parallel. This function does not attempt to calculate
        spacing if it is not found in the dataset.

    """
    coordinate_system = get_image_coordinate_system(dataset)

    if coordinate_system is None:
        raise ValueError(
            'The input "dataset" has no spatial information '
            'as it has no frame of reference.'
        )

    if for_total_pixel_matrix:
        if not hasattr(dataset, 'TotalPixelMatrixOriginSequence'):
            raise ValueError('Image is not a tiled image.')
        origin_seq = dataset.TotalPixelMatrixOriginSequence[0]
        position = (
            origin_seq.XOffsetInSlideCoordinateSystem,
            origin_seq.YOffsetInSlideCoordinateSystem,
            getattr(origin_seq, 'ZOffsetInSlideCoordinateSystem', 0.0)
        )
        shared_seq = dataset.SharedFunctionalGroupsSequence[0]
        if hasattr(shared_seq, 'PixelMeasuresSequence'):
            pixel_spacing = shared_seq.PixelMeasuresSequence[0].PixelSpacing
            spacing_between_slices = getattr(
                shared_seq.PixelMeasuresSequence[0],
                "SpacingBetweenSlices",
                None,
            )
        else:
            raise ValueError(
                "PixelMeasuresSequence not found in the "
                "SharedFunctionalGroupsSequence."
            )

        orientation = dataset.ImageOrientationSlide
        return position, orientation, pixel_spacing, spacing_between_slices

    if is_multiframe_image(dataset):
        if frame_number is None:
            raise TypeError(
                'Argument "frame_number" must be specified for a multi-frame '
                'image.'
            )
        shared_seq = dataset.SharedFunctionalGroupsSequence[0]
        is_tiled_full = (
            dataset.get("DimensionOrganizationType", "") == "TILED_FULL"
        )
        if is_tiled_full:
            frame_seq = None
        else:
            frame_seq = dataset.PerFrameFunctionalGroupsSequence[
                frame_number - 1
            ]

        # Find spacing in either shared or per-frame sequences (this logic is
        # the same for patient or slide coordinate system)
        if hasattr(shared_seq, 'PixelMeasuresSequence'):
            pixel_measures = shared_seq.PixelMeasuresSequence[0]
        elif (
            frame_seq is not None and
            hasattr(frame_seq, 'PixelMeasuresSequence')
        ):
            pixel_measures = frame_seq.PixelMeasuresSequence[0]
        else:
            raise ValueError('No pixel measures information found.')
        pixel_spacing = pixel_measures.PixelSpacing
        spacing_between_slices = getattr(
            pixel_measures,
            'SpacingBetweenSlices',
            None,
        )

        if coordinate_system == CoordinateSystemNames.SLIDE:
            if is_tiled_full:
                # TODO this iteration is probably rather inefficient
                _, _, _, _, *position = next(
                    itertools.islice(
                        iter_tiled_full_frame_data(dataset),
                        frame_number - 1,
                        frame_number,
                    )
                )
            else:
                # Find position in either shared or per-frame sequences
                if hasattr(shared_seq, 'PlanePositionSlideSequence'):
                    pos_seq = shared_seq.PlanePositionSlideSequence[0]
                elif (
                    frame_seq is not None and
                    hasattr(frame_seq, 'PlanePositionSlideSequence')
                ):
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
            elif (
                frame_seq is not None and
                hasattr(frame_seq, 'PlanePositionSequence')
            ):
                pos_seq = frame_seq.PlanePositionSequence[0]
            else:
                raise ValueError('No frame position information found.')

            position = pos_seq.ImagePositionPatient

            # Find orientation  in either shared or per-frame sequences
            if hasattr(shared_seq, 'PlaneOrientationSequence'):
                pos_seq = shared_seq.PlaneOrientationSequence[0]
            elif (
                frame_seq is not None and
                hasattr(frame_seq, 'PlaneOrientationSequence')
            ):
                pos_seq = frame_seq.PlaneOrientationSequence[0]
            else:
                raise ValueError('No frame orientation information found.')

            orientation = pos_seq.ImageOrientationPatient

    else:  # Single-frame image
        if frame_number is not None and frame_number != 1:
            raise TypeError(
                'Argument "frame_number" must be None or 1 for a single-frame '
                "image."
            )
        position = dataset.ImagePositionPatient
        orientation = dataset.ImageOrientationPatient
        pixel_spacing = dataset.PixelSpacing
        spacing_between_slices = getattr(
            dataset,
            "SpacingBetweenSlices",
            None,
        )

    return position, orientation, pixel_spacing, spacing_between_slices


def _get_normal_vector(image_orientation: Sequence[float]) -> np.ndarray:
    """Get normal vector given image cosines.

    Parameters
    ----------
    image_orientation: Sequence[float]
        Row and column cosines (6 element list) giving the orientation of the
        image.

    Returns
    -------
    np.ndarray
        Array of shape (3, ) giving the normal vector to the image plane.

    """
    row_cosines = np.array(image_orientation[:3], dtype=float)
    column_cosines = np.array(image_orientation[3:], dtype=float)
    n = np.cross(row_cosines.T, column_cosines.T)
    return n


def _are_images_coplanar(
    image_position_a: Sequence[float],
    image_orientation_a: Sequence[float],
    image_position_b: Sequence[float],
    image_orientation_b: Sequence[float],
    tol: float = _DEFAULT_TOLERANCE,
) -> bool:
    """Determine whether two images or image frames are coplanar.

    Two images are coplanar in the frame of reference coordinate system if and
    only if their vectors have the same (or opposite direction) and the
    shortest distance from the plane to the coordinate system origin is
    the same for both planes.

    Parameters
    ----------
    image_position_a: Sequence[float]
        Image position (3 element list) giving the position of the center of
        the top left pixel of the first image.
    image_orientation_a: Sequence[float]
        Row and column cosines (6 element list) giving the orientation of the
        first image.
    image_position_b: Sequence[float]
        Image position (3 element list) giving the position of the center of
        the top left pixel of the second image.
    image_orientation_b: Sequence[float]
        Row and column cosines (6 element list) giving the orientation of the
        second image.
    tol: float
        Tolerance to use to determine equality.

    Returns
    -------
    bool
        True if the two images are coplanar. False otherwise.

    """
    n_a = _get_normal_vector(image_orientation_a)
    n_b = _get_normal_vector(image_orientation_b)
    if 1.0 - np.abs(n_a @ n_b) > tol:
        return False

    # Find distances of both planes along n_a
    dis_a = np.array(image_position_a, dtype=float) @ n_a
    dis_b = np.array(image_position_b, dtype=float) @ n_a

    return abs(dis_a - dis_b) < tol


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
    return np.vstack(
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
    return np.vstack(
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

    This class facilitates the mapping of pixel indices of the pixel matrix
    of an image or an image frame (tile or plane) into the patient or slide
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
        pixel_matrix_coordinates = np.vstack([
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
        for_total_pixel_matrix: bool = False,
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
        for_total_pixel_matrix: bool, optional
            If True, use the spatial information for the total pixel matrix of
            a tiled image. The result will be a transformer that maps pixel
            indices of the total pixel matrix to frame of reference
            coordinates. This should only be True if the image is a tiled image
            and is incompatible with specifying a frame number.

        Returns
        -------
        highdicom.spatial.PixelToReferenceTransformer:
            Transformer object for the given image, or image frame.

        """
        position, orientation, spacing, _ = _get_spatial_information(
            dataset,
            frame_number=frame_number,
            for_total_pixel_matrix=for_total_pixel_matrix,
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
    pixel of the pixel matrix. The result of the transform also contains a
    third coordinate, giving position along the normal vector of the imaging
    plane.

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
        spacing_between_slices: float = 1.0,
        round_output: bool = True,
        drop_slice_index: bool = False,
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
        round_output: bool, optional
            If True, outputs are rounded to the nearest integer. Otherwise,
            they are returned as float.
        drop_slice_index: bool, optional
            Whether to remove the 3rd column of the output array
            (representing the out-of-plane coordinate) and return a 2D output
            array. If this option is taken, and the resulting coordinates
            do not lie in the range -0.5 to 0.5, a ``RuntimeError`` will be
            triggered.

        Raises
        ------
        TypeError
            When `image_position`, `image_orientation` or `pixel_spacing` is
            not a sequence.
        ValueError
            When `image_position`, `image_orientation` or `pixel_spacing` has
            an incorrect length.

        """
        self._round_output = round_output
        self._drop_slice_index = drop_slice_index
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
            Array of (column, row, slice) zero-based indices at pixel
            resolution. Array of integer or floating point values with shape
            ``(n, 3)``, where *n* is the number of indices, the first column
            represents the `column` index, the second column represents the
            `row` index, and the third column represents the `slice` coordinate
            in the direction normal to the image plane (with scale given by the
            ``spacing_between_slices_to`` parameter). The ``(0, 0, 0)``
            coordinate is located at the **center** of the top left pixel in
            the total pixel matrix. The datatype of the array will be integer
            if ``round_output`` is True (the default), or float if
            ``round_output`` is False.

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
        reference_coordinates = np.vstack([
            coordinates.T.astype(float),
            np.ones((coordinates.shape[0], ), dtype=float)
        ])
        pixel_matrix_coordinates = np.dot(self._affine, reference_coordinates)
        pixel_matrix_coordinates = pixel_matrix_coordinates[:3, :].T
        if self._drop_slice_index:
            if np.abs(pixel_matrix_coordinates[:, 2]).max() > 0.5:
                raise RuntimeError(
                    "Output indices do not lie within the given image "
                    "plane."
                )
            pixel_matrix_coordinates = pixel_matrix_coordinates[:, :2]
        if self._round_output:
            return np.around(pixel_matrix_coordinates).astype(int)
        else:
            return pixel_matrix_coordinates

    @classmethod
    def for_image(
        cls,
        dataset: Dataset,
        frame_number: Optional[int] = None,
        for_total_pixel_matrix: bool = False,
        round_output: bool = True,
        drop_slice_index: bool = False,
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
        for_total_pixel_matrix: bool, optional
            If True, use the spatial information for the total pixel matrix of
            a tiled image. The result will be a transformer that maps frame of
            reference coordinates to indices of the total pixel matrix. This
            should only be True if the image is a tiled image and is
            incompatible with specifying a frame number.
        round_output: bool, optional
            If True, outputs are rounded to the nearest integer. Otherwise,
            they are returned as float.
        drop_slice_index: bool, optional
            Whether to remove the 3rd element of the output array
            (representing the out-of-plane coordinate) and return a 2D output
            array. If this option is taken, and the resulting coordinates
            do not lie in the range -0.5 to 0.5, a ``RuntimeError`` will be
            triggered.

        Returns
        -------
        highdicom.spatial.ReferenceToPixelTransformer:
            Transformer object for the given image, or image frame.

        """
        (
            position,
            orientation,
            spacing,
            slice_spacing,
        ) = _get_spatial_information(
            dataset,
            frame_number=frame_number,
            for_total_pixel_matrix=for_total_pixel_matrix,
        )
        if slice_spacing is None:
            slice_spacing = 1.0
        return cls(
            image_position=position,
            image_orientation=orientation,
            pixel_spacing=spacing,
            spacing_between_slices=slice_spacing,
            round_output=round_output,
            drop_slice_index=drop_slice_index,
        )


class PixelToPixelTransformer:

    """Class for transforming pixel indices between two images.

    This class facilitates the mapping of pixel indices of the pixel matrix of
    an image or an image frame (tile or plane) into those of another image or
    image frame in the same frame of reference. This can include (but is not
    limited) to mapping between different frames of the same image, or
    different images within the same series (e.g. two levels of a spatial
    pyramid). However, it is required that the two images be coplanar
    within the frame-of-reference coordinate system.

    Pixel indices are (column, row) pairs of zero-based integer values, where
    the (0, 0) index is located at the **center** of the top left hand corner
    pixel of the pixel matrix.

    Examples
    --------

    Create a transformer for two images, where the second image has an axis
    flipped relative to the first.

    >>> transformer = PixelToPixelTransformer(
    ...     image_position_from=[0.0, 0.0, 0.0],
    ...     image_orientation_from=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ...     pixel_spacing_from=[1.0, 1.0],
    ...     image_position_to=[0.0, 100.0, 0.0],
    ...     image_orientation_to=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    ...     pixel_spacing_to=[1.0, 1.0],
    ... )

    >>> indices_in = np.array([[0, 0], [50, 50]])
    >>> indices_out = transformer(indices_in)
    >>> print(indices_out)
    [[  0 100]
     [ 50  50]]

    Warning
    -------
    This class shall not be used to map spatial coordinates (SCOORD)
    between images. Use the
    :class:`highdicom.spatial.ImageToImageTransformer` class instead.

    """

    def __init__(
        self,
        image_position_from: Sequence[float],
        image_orientation_from: Sequence[float],
        pixel_spacing_from: Sequence[float],
        image_position_to: Sequence[float],
        image_orientation_to: Sequence[float],
        pixel_spacing_to: Sequence[float],
        round_output: bool = True,
    ):
        """Construct transformation object.

        The resulting object will map pixel indices of the "from" image to
        pixel indices of the "to" image.

        Parameters
        ----------
        image_position_from: Sequence[float]
            Position of the "from" image in the frame of reference,
            i.e., the offset of the top left hand corner pixel in the pixel
            matrix from the origin of the reference coordinate system along the
            X, Y, and Z axis
        image_orientation_from: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing column index) and the column direction (second
            triplet: vertical, top to bottom, increasing row index) direction
            of the "from" image expressed in the three-dimensional patient or
            slide coordinate system defined by the frame of reference
        pixel_spacing_from: Sequence[float]
            Spacing between pixels of the "from" imagem in millimeter unit
            along the column direction (first value: spacing between rows,
            vertical, top to bottom, increasing row index) and the rows
            direction (second value: spacing between columns: horizontal, left
            to right, increasing column index)
        image_position_to: Sequence[float]
            Position of the "to" image using the same definition as the "from"
            image.
        image_orientation_to: Sequence[float]
            Orientation cosines of the "to" image using the same definition as
            the "from" image.
        pixel_spacing_to: Sequence[float]
            Pixel spacing of the "to" image using the same definition as
            the "from" image.
        round_output: bool, optional
            If True, outputs are rounded to the nearest integer. Otherwise,
            they are returned as float.

        Raises
        ------
        TypeError
            When any of the arguments is not a sequence.
        ValueError
            When any of the arguments has an incorrect length, or if the two
            images are not coplanar in the frame of reference coordinate
            system.

        """
        self._round_output = round_output
        if not _are_images_coplanar(
            image_position_a=image_position_from,
            image_orientation_a=image_orientation_from,
            image_position_b=image_position_to,
            image_orientation_b=image_orientation_to,
        ):
            raise ValueError(
                "To two images do not exist within the same plane "
                "in the frame of reference. and therefore pixel-to-pixel "
                "transformation is not possible."
            )
        pix_to_ref = _create_affine_transformation_matrix(
            image_position=image_position_from,
            image_orientation=image_orientation_from,
            pixel_spacing=pixel_spacing_from,
        )
        ref_to_pix = _create_inv_affine_transformation_matrix(
            image_position=image_position_to,
            image_orientation=image_orientation_to,
            pixel_spacing=pixel_spacing_to,
        )
        self._affine = np.dot(ref_to_pix, pix_to_ref)

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix"""
        return self._affine

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """Transform pixel indices between two images.

        Parameters
        ----------
        indices: numpy.ndarray
            Array of (column, row) zero-based pixel indices of the "from" image
            in the range [0, Columns - 1] and [0, Rows - 1], respectively.
            Array of integer values with shape ``(n, 2)``, where *n* is the
            number of indices, the first column represents the `column` index
            and the second column represents the `row` index. The ``(0, 0)``
            coordinate is located at the **center** of the top left pixel in
            the total pixel matrix.

        Returns
        -------
        numpy.ndarray
            Array of (column, row) zero-based pixel indices of the "to" image.

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
        pixel_matrix_coordinates = np.vstack([
            indices.T.astype(float),
            np.zeros((indices.shape[0], ), dtype=float),
            np.ones((indices.shape[0], ), dtype=float),
        ])
        output_coordinates = np.dot(self._affine, pixel_matrix_coordinates)
        output_coordinates = output_coordinates[:2, :].T
        if self._round_output:
            return np.around(output_coordinates).astype(int)
        else:
            return output_coordinates

    @classmethod
    def for_images(
        cls,
        dataset_from: Dataset,
        dataset_to: Dataset,
        frame_number_from: Optional[int] = None,
        frame_number_to: Optional[int] = None,
        for_total_pixel_matrix_from: bool = False,
        for_total_pixel_matrix_to: bool = False,
        round_output: bool = True,
    ) -> 'PixelToPixelTransformer':
        """Construct a transformer for two given images or image frames.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset representing an image.
        frame_number: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame for which to get
            the transformer. This should be provided if and only if the dataset
            is a multi-frame image.
        for_total_pixel_matrix: bool, optional
            If True, use the spatial information for the total pixel matrix of
            a tiled image. The result will be a transformer that maps pixel
            indices of the total pixel matrix to frame of reference
            coordinates. This should only be True if the image is a tiled image
            and is incompatible with specifying a frame number.
        round_output: bool, optional
            If True, outputs are rounded to the nearest integer. Otherwise,
            they are returned as float.

        Returns
        -------
        highdicom.spatial.PixelToPixelTransformer:
            Transformer object for the given image, or image frame.

        """
        if (
            not hasattr(dataset_from, 'FrameOfReferenceUID') or
            not hasattr(dataset_to, 'FrameOfReferenceUID')
        ):
            raise ValueError(
                'Cannot determine spatial relationship because datasets '
                'lack a frame of reference UID.'
            )
        if dataset_from.FrameOfReferenceUID != dataset_to.FrameOfReferenceUID:
            raise ValueError(
                'Datasets do not share a frame of reference, so the spatial '
                'relationship between them is not defined.'
            )

        pos_f, ori_f, spa_f, _ = _get_spatial_information(
            dataset_from,
            frame_number=frame_number_from,
            for_total_pixel_matrix=for_total_pixel_matrix_from,
        )
        pos_t, ori_t, spa_t, _ = _get_spatial_information(
            dataset_to,
            frame_number=frame_number_to,
            for_total_pixel_matrix=for_total_pixel_matrix_to,
        )
        return cls(
            image_position_from=pos_f,
            image_orientation_from=ori_f,
            pixel_spacing_from=spa_f,
            image_position_to=pos_t,
            image_orientation_to=ori_t,
            pixel_spacing_to=spa_t,
            round_output=round_output,
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
        image_coordinates = np.vstack([
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
        for_total_pixel_matrix: bool = False,
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
        for_total_pixel_matrix: bool, optional
            If True, use the spatial information for the total pixel matrix of
            a tiled image. The result will be a transformer that maps image
            coordinates of the total pixel matrix to frame of reference
            coordinates. This should only be True if the image is a tiled image
            and is incompatible with specifying a frame number.

        Returns
        -------
        highdicom.spatial.ImageToReferenceTransformer:
            Transformer object for the given image, or image frame.

        """
        position, orientation, spacing, _ = _get_spatial_information(
            dataset,
            frame_number=frame_number,
            for_total_pixel_matrix=for_total_pixel_matrix,
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
        spacing_between_slices: float = 1.0,
        drop_slice_coord: bool = False,
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
        drop_slice_coord: bool, optional
            Whether to remove the 3rd column of the output array
            (representing the out-of-plane coordinate) and return a 2D output
            array. If this option is taken, and the resulting coordinates
            do not lie in the range -0.5 to 0.5, a ``RuntimeError`` will be
            triggered.

        Raises
        ------
        TypeError
            When `image_position`, `image_orientation` or `pixel_spacing` is
            not a sequence.
        ValueError
            When `image_position`, `image_orientation` or `pixel_spacing` has
            an incorrect length.

        """
        self._drop_slice_coord = drop_slice_coord
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
            zero-based coordinates in the total pixel matrix and the `slice`
            index represents the signed distance of the input coordinate in the
            direction normal to the plane of the total pixel matrix. The `row`
            and `column` coordinates are constrained by the dimension of the
            total pixel matrix. Note, however, that in general, the resulting
            coordinate may not lie within the imaging plane, and consequently
            the `slice` offset may be non-zero.

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
        reference_coordinates = np.vstack([
            coordinates.T.astype(float),
            np.ones((coordinates.shape[0], ), dtype=float)
        ])
        image_coordinates = np.dot(self._affine, reference_coordinates)
        image_coordinates = image_coordinates[:3, :].T
        if self._drop_slice_coord:
            if np.abs(image_coordinates[:, 2]).max() > 0.5:
                raise RuntimeError(
                    "Output coordinates do not lie within the given image "
                    "plane."
                )
            image_coordinates = image_coordinates[:, :2]
        return image_coordinates

    @classmethod
    def for_image(
        cls,
        dataset: Dataset,
        frame_number: Optional[int] = None,
        for_total_pixel_matrix: bool = False,
        drop_slice_coord: bool = False,
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
        for_total_pixel_matrix: bool, optional
            If True, use the spatial information for the total pixel matrix of
            a tiled image. The result will be a transformer that maps frame of
            reference coordinates to indices of the total pixel matrix. This
            should only be True if the image is a tiled image and is
            incompatible with specifying a frame number.
        drop_slice_coord: bool, optional
            Whether to remove the 3rd column of the output array
            (representing the out-of-plane coordinate) and return a 2D output
            array. If this option is taken, and the resulting coordinates
            do not lie in the range -0.5 to 0.5, a ``RuntimeError`` will be
            triggered.

        Returns
        -------
        highdicom.spatial.ReferenceToImageTransformer:
            Transformer object for the given image, or image frame.

        """
        (
            position,
            orientation,
            spacing,
            slice_spacing,
        ) = _get_spatial_information(
            dataset,
            frame_number=frame_number,
            for_total_pixel_matrix=for_total_pixel_matrix,
        )
        if slice_spacing is None:
            slice_spacing = 1.0
        return cls(
            image_position=position,
            image_orientation=orientation,
            pixel_spacing=spacing,
            spacing_between_slices=slice_spacing,
            drop_slice_coord=drop_slice_coord,
        )


class ImageToImageTransformer:

    """Class for transforming image coordinates between two images.

    This class facilitates the mapping of image coordinates of
    an image or an image frame (tile or plane) into those of another image or
    image frame in the same frame of reference. This can include (but is not
    limited) to mapping between different frames of the same image, or
    different images within the same series (e.g. two levels of a spatial
    pyramid). However, it is required that the two images be coplanar
    within the frame-of-reference coordinate system.

    Image coordinates are (column, row) pairs of floating-point values, where
    the (0.0, 0.0) point is located at the top left corner of the top left hand
    corner pixel of the pixel matrix. Image coordinates have pixel units at
    sub-pixel resolution.

    Examples
    --------

    Create a transformer for two images, where the second image has an axis
    flipped relative to the first.

    >>> transformer = ImageToImageTransformer(
    ...     image_position_from=[0.0, 0.0, 0.0],
    ...     image_orientation_from=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ...     pixel_spacing_from=[1.0, 1.0],
    ...     image_position_to=[0.0, 100.0, 0.0],
    ...     image_orientation_to=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    ...     pixel_spacing_to=[1.0, 1.0],
    ... )

    >>> coords_in = np.array([[0, 0], [50, 50]])
    >>> coords_out = transformer(coords_in)
    >>> print(coords_out)
    [[  0. 101.]
     [ 50.  51.]]

    Warning
    -------
    This class shall not be used to pixel indices between images. Use the
    :class:`highdicom.spatial.PixelToPixelTransformer` class instead.

    """

    def __init__(
        self,
        image_position_from: Sequence[float],
        image_orientation_from: Sequence[float],
        pixel_spacing_from: Sequence[float],
        image_position_to: Sequence[float],
        image_orientation_to: Sequence[float],
        pixel_spacing_to: Sequence[float],
    ):
        """Construct transformation object.

        The resulting object will map image coordinates of the "from" image to
        image coordinates of the "to" image.

        Parameters
        ----------
        image_position_from: Sequence[float]
            Position of the "from" image in the frame of reference,
            i.e., the offset of the top left hand corner pixel in the pixel
            matrix from the origin of the reference coordinate system along the
            X, Y, and Z axis
        image_orientation_from: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing column index) and the column direction (second
            triplet: vertical, top to bottom, increasing row index) direction
            of the "from" image expressed in the three-dimensional patient or
            slide coordinate system defined by the frame of reference
        pixel_spacing_from: Sequence[float]
            Spacing between pixels of the "from" imagem in millimeter unit
            along the column direction (first value: spacing between rows,
            vertical, top to bottom, increasing row index) and the rows
            direction (second value: spacing between columns: horizontal, left
            to right, increasing column index)
        image_position_to: Sequence[float]
            Position of the "to" image using the same definition as the "from"
            image.
        image_orientation_to: Sequence[float]
            Orientation cosines of the "to" image using the same definition as
            the "from" image.
        pixel_spacing_to: Sequence[float]
            Pixel spacing of the "to" image using the same definition as
            the "from" image.

        Raises
        ------
        TypeError
            When any of the arguments is not a sequence.
        ValueError
            When any of the arguments has an incorrect length, or if the two
            images are not coplanar in the frame of reference coordinate
            system.

        """
        if not _are_images_coplanar(
            image_position_a=image_position_from,
            image_orientation_a=image_orientation_from,
            image_position_b=image_position_to,
            image_orientation_b=image_orientation_to,
        ):
            raise ValueError(
                "To two images do not exist within the same plane "
                "in the frame of reference. and therefore pixel-to-pixel "
                "transformation is not possible."
            )
        # Image coordinates are shifted relative to pixel matrix indices by
        # 0.5 pixels and we thus have to correct for this shift.
        pix_to_im = np.array([
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        ref_to_pix = _create_inv_affine_transformation_matrix(
            image_position=image_position_to,
            image_orientation=image_orientation_to,
            pixel_spacing=pixel_spacing_to,
        )
        pix_to_ref = _create_affine_transformation_matrix(
            image_position=image_position_from,
            image_orientation=image_orientation_from,
            pixel_spacing=pixel_spacing_from,
        )
        im_to_pix = np.array([
            [1.0, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._affine = pix_to_im @ ref_to_pix @ pix_to_ref @ im_to_pix

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix"""
        return self._affine

    def __call__(self, coordinates: np.ndarray) -> np.ndarray:
        """Transform pixel indices between two images.

        Parameters
        ----------
        indices: numpy.ndarray
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
            Array of (column, row) image coordinates in the "to" image.

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
        image_coordinates = np.vstack([
            coordinates.T.astype(float),
            np.zeros((coordinates.shape[0], ), dtype=float),
            np.ones((coordinates.shape[0], ), dtype=float),
        ])
        output_coordinates = np.dot(self._affine, image_coordinates)
        return output_coordinates[:2, :].T

    @classmethod
    def for_images(
        cls,
        dataset_from: Dataset,
        dataset_to: Dataset,
        frame_number_from: Optional[int] = None,
        frame_number_to: Optional[int] = None,
        for_total_pixel_matrix_from: bool = False,
        for_total_pixel_matrix_to: bool = False,
    ) -> 'ImageToImageTransformer':
        """Construct a transformer for two given images or image frames.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset representing an image.
        frame_number: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame for which to get
            the transformer. This should be provided if and only if the dataset
            is a multi-frame image.
        for_total_pixel_matrix: bool, optional
            If True, use the spatial information for the total pixel matrix of
            a tiled image. The result will be a transformer that maps image
            coordinates of the total pixel matrix to frame of reference
            coordinates. This should only be True if the image is a tiled image
            and is incompatible with specifying a frame number.

        Returns
        -------
        highdicom.spatial.ImageToImageTransformer:
            Transformer object for the given image, or image frame.

        """
        if (
            not hasattr(dataset_from, 'FrameOfReferenceUID') or
            not hasattr(dataset_to, 'FrameOfReferenceUID')
        ):
            raise ValueError(
                'Cannot determine spatial relationship because datasets '
                'lack a frame of reference UID.'
            )
        if dataset_from.FrameOfReferenceUID != dataset_to.FrameOfReferenceUID:
            raise ValueError(
                'Datasets do not share a frame of reference, so the spatial '
                'relationship between them is not defined.'
            )

        pos_f, ori_f, spa_f, _ = _get_spatial_information(
            dataset_from,
            frame_number=frame_number_from,
            for_total_pixel_matrix=for_total_pixel_matrix_from,
        )
        pos_t, ori_t, spa_t, _ = _get_spatial_information(
            dataset_to,
            frame_number=frame_number_to,
            for_total_pixel_matrix=for_total_pixel_matrix_to,
        )
        return cls(
            image_position_from=pos_f,
            image_orientation_from=ori_f,
            pixel_spacing_from=spa_f,
            image_position_to=pos_t,
            image_orientation_to=ori_t,
            pixel_spacing_to=spa_t,
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
