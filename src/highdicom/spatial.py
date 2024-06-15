import itertools
from typing import (
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydicom import Dataset
import numpy as np
import pydicom

from highdicom._module_utils import is_multiframe_image
from highdicom.enum import (
    CoordinateSystemNames,
    PixelIndexDirections,
    PatientFrameOfReferenceDirections,
)


_DEFAULT_SPACING_TOLERANCE = 1e-4
"""Default tolerance for determining whether slices are regularly spaced."""


_DEFAULT_EQUALITY_TOLERANCE = 1e-5
"""Tolerance value used by default in tests for equality"""


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
    tol: float = _DEFAULT_EQUALITY_TOLERANCE,
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


def _normalize_pixel_index_convention(
    c: Union[str, Sequence[Union[str, PixelIndexDirections]]],
) -> Tuple[PixelIndexDirections, PixelIndexDirections, PixelIndexDirections]:
    """Normalize and check a pixel index convention.

    Parameters
    ----------
    c: Union[str, Sequence[Union[str, highdicom.enum.PixelIndexDirections]]]
        Pixel index convention description consisting of three directions,
        either L or R, either U or D, and either I or O, in any order.

    Returns
    -------
    Tuple[highdicom.enum.PixelIndexDirections, highdicom.enum.PixelIndexDirections, highdicom.enum.PixelIndexDirections]:
        Convention description in a canonical form as a tuple of three enum
        instances. Furthermore this is guaranteed to be a valid description.

    """  # noqa: E501
    if len(c) != 3:
        raise ValueError('Length of pixel index convention must be 3.')

    c = tuple(PixelIndexDirections(d) for d in c)

    c_set = {d.value for d in c}

    criteria = [
        ('L' in c_set) != ('R' in c_set),
        ('U' in c_set) != ('D' in c_set),
        ('I' in c_set) != ('O' in c_set),
    ]
    if not all(criteria):
        c_str = [d.value for d in c]
        raise ValueError(f'Invalid combination of pixel directions: {c_str}.')

    return c


def _normalize_reference_direction_convention(
    c: Union[str, Sequence[Union[str, PatientFrameOfReferenceDirections]]],
) -> Tuple[
    PatientFrameOfReferenceDirections,
    PatientFrameOfReferenceDirections,
    PatientFrameOfReferenceDirections,
]:
    """Normalize and check a frame of reference direction convention.

    Parameters
    ----------
    c: Union[str, Sequence[Union[str, highdicom.enum.PatientFrameOfReferenceDirections]]]
        Frame of reference convention description consisting of three directions,
        either L or R, either A or P, and either I or S, in any order.

    Returns
    -------
    Tuple[highdicom.enum.PatientFrameOfReferenceDirections, highdicom.enum.PatientFrameOfReferenceDirections, highdicom.enum.PatientFrameOfReferenceDirections]:
        Convention description in a canonical form as a tuple of three enum
        instances. Furthermore this is guaranteed to be a valid description.

    """  # noqa: E501
    if len(c) != 3:
        raise ValueError('Length of pixel index convention must be 3.')

    c = tuple(PatientFrameOfReferenceDirections(d) for d in c)

    c_set = {d.value for d in c}

    criteria = [
        ('L' in c_set) != ('R' in c_set),
        ('A' in c_set) != ('P' in c_set),
        ('I' in c_set) != ('S' in c_set),
    ]
    if not all(criteria):
        c_str = [d.value for d in c]
        raise ValueError(
            'Invalid combination of frame of reference directions: '
            f'{c_str}.'
        )

    return c


def _is_matrix_orthogonal(
    m: np.ndarray,
    tol: float = _DEFAULT_EQUALITY_TOLERANCE,
) -> bool:
    """Check whether a matrix is orthogonal.

    Note this does not require that the columns have unit norm.

    Parameters
    ----------
    m: numpy.ndarray
        A matrix.
    tol: float
        Tolerance. ``m`` will be deemed orthogonal if the product ``m.T @ m``
        is equal to diagonal matrix of squared column norms within this
        tolerance.

    Returns
    -------
    bool:
        True if the matrix ``m`` is a square orthogonal matrix. False
        otherwise.

    """
    if m.ndim != 2:
        raise ValueError(
            'Argument "m" should be an array with 2 dimensions.'
         )
    if m.shape[0] != m.shape[1]:
        return False
    norm_squared = (m ** 2).sum(axis=0)
    return np.allclose(m.T @ m, np.diag(norm_squared), atol=tol)


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
        frame of reference.

    Returns
    -------
    numpy.ndarray
        3 x 3 rotation matrix. Pre-multiplying a pixel index in format (column
        index, row index, slice index) by this matrix gives the x, y, z
        position in the frame-of-reference coordinate system.

    """
    if len(image_orientation) != 6:
        raise ValueError('Argument "image_orientation" must have length 6.')
    row_cosines = np.array(image_orientation[:3], dtype=float)
    column_cosines = np.array(image_orientation[3:], dtype=float)
    n = np.cross(row_cosines.T, column_cosines.T)

    return np.column_stack([
        row_cosines,
        column_cosines,
        n,
    ])


def _create_affine_transformation_matrix(
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    spacing_between_slices: float = 1.0,
    index_convention: Optional[Sequence[PixelIndexDirections]] = None,
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
    spacing_between_slices: float
        Spacing between consecutive slices.
    index_convention: Union[Sequence[highdicom.enum.PixelIndexDirections], None]
        Desired convention for the pixel index directions. Must consist of only
        D, I, and R.

    Returns
    -------
    numpy.ndarray
        4 x 4 affine transformation matrix. Pre-multiplying a pixel index in
        format (column index, row index, slice index, 1) by this matrix gives
        the (x, y, z, 1) position in the frame-of-reference coordinate system.

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

    rotation = create_rotation_matrix(
        image_orientation,
    )
    # Column direction (spacing between rows)
    spacing_between_rows = float(pixel_spacing[0])
    # Row direction (spacing between columns)
    spacing_between_columns = float(pixel_spacing[1])

    rotation[:, 0] *= spacing_between_columns
    rotation[:, 1] *= spacing_between_rows
    rotation[:, 2] *= spacing_between_slices

    # 4x4 transformation matrix
    affine = np.row_stack(
        [
            np.column_stack([
                rotation,
                translation,
            ]),
            [0.0, 0.0, 0.0, 1.0]
        ]
    )

    if index_convention is not None:
        current_convention = (
            PixelIndexDirections.R,
            PixelIndexDirections.D,
            PixelIndexDirections.I,
        )
        if set(index_convention) != set(current_convention):
            raise ValueError(
                'Index convention must consist of D, I, and R.'
            )
        affine = _transform_affine_to_convention(
            affine=affine,
            shape=(1, 1, 1),  # dummy (not used)
            from_index_convention=current_convention,
            to_index_convention=index_convention,
        )

    return affine


def _create_inv_affine_transformation_matrix(
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    spacing_between_slices: float = 1.0,
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

    Returns
    -------
    numpy.ndarray
        4 x 4 affine transformation matrix. Pre-multiplying a
        frame-of-reference coordinate in the format (x, y, z, 1) by this matrix
        gives the pixel indices in the form (column index, row index, slice
        index, 1).

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
    spacing_between_rows = float(pixel_spacing[0])
    # Row direction (spacing between columns)
    spacing_between_columns = float(pixel_spacing[1])
    rotation[:, 0] *= spacing_between_columns
    rotation[:, 1] *= spacing_between_rows
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


def _transform_affine_matrix(
    affine: np.ndarray,
    shape: Sequence[int],
    flip_indices: Optional[Sequence[bool]] = None,
    flip_reference: Optional[Sequence[bool]] = None,
    permute_indices: Optional[Sequence[int]] = None,
    permute_reference: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Transform an affine matrix between conventions.

    Parameters
    ----------
    affine: np.ndarray
        4 x 4 affine matrix to transform.
    shape: Sequence[int]
        Shape of the array.
    flip_indices: Optional[Sequence[bool]], optional
        Whether to flip each of the pixel index axes to index from the other
        side of the array. Must consist of three boolean values, one for each
        of the index axes (before any permutation is applied).
    flip_reference: Optional[Sequence[bool]], optional
        Whether to flip each of the frame of reference axes to about the
        origin. Must consist of three boolean values, one for each of the frame
        of reference axes (before any permutation is applied).
    permute_indices: Optional[Sequence[int]], optional
        Permutation (if any) to apply to the pixel index axes. Must consist of
        the values [0, 1, 2] in some order.
    permute_reference: Optional[Sequence[int]], optional
        Permutation (if any) to apply to the frame of reference axes. Must
        consist of the values [0, 1, 2] in some order.

    Returns
    -------
    np.ndarray:
        Affine matrix after operations are applied.

    """
    if affine.shape != (4, 4):
        raise ValueError("Affine matrix must have shape (4, 4).")
    if len(shape) != 3:
        raise ValueError("Shape must have shape three elements.")

    transformed = affine.copy()

    if flip_indices is not None and any(flip_indices):
        # Move the origin to the opposite side of the array
        enable = np.array(flip_indices, np.uint8)
        offset = transformed[:3, :3] * (np.array(shape).reshape(3, 1) - 1)
        transformed[:3, 3] += enable @ offset

        # Inverting the columns
        transformed *= np.array(
            [*[-1 if x else 1 for x in flip_indices], 1]
        )

    if flip_reference is not None and any(flip_reference):
        # Flipping the reference means inverting the rows (including the
        # translation)
        row_inv = np.diag(
            [*[-1 if x else 1 for x in flip_reference], 1]
        )
        transformed = row_inv @ transformed

    # Permuting indices is a permutation of the columns
    if permute_indices is not None:
        if len(permute_indices) != 3:
            raise ValueError(
                'Argument "permute_indices" should have 3 elements.'
            )
        if set(permute_indices) != set((0, 1, 2)):
            raise ValueError(
                'Argument "permute_indices" should contain elements 0, 1, '
                "and 3 in some order."
            )
        transformed = transformed[:, [*permute_indices, 3]]

    # Permuting the reference is a permutation of the rows
    if permute_reference is not None:
        if len(permute_reference) != 3:
            raise ValueError(
                'Argument "permute_reference" should have 3 elements.'
            )
        if set(permute_reference) != set((0, 1, 2)):
            raise ValueError(
                'Argument "permute_reference" should contain elements 0, 1, '
                "and 3 in some order."
            )
        transformed = transformed[[*permute_reference, 3], :]

    return transformed


def _transform_affine_to_convention(
    affine: np.ndarray,
    shape: Sequence[int],
    from_index_convention: Union[
        str, Sequence[Union[str, PixelIndexDirections]], None
    ] = None,
    to_index_convention: Union[
        str, Sequence[Union[str, PixelIndexDirections]], None
    ] = None,
    from_reference_convention: Union[
        str, Sequence[Union[str, PatientFrameOfReferenceDirections]], None
    ] = None,
    to_reference_convention: Union[
        str, Sequence[Union[str, PatientFrameOfReferenceDirections]], None
    ] = None,
) -> np.ndarray:
    """Transform an affine matrix between different conventions.

    Parameters
    ----------
    affine: np.ndarray
        Affine matrix to transform.
    shape: Sequence[int]
        Shape of the array.
    from_index_convention: Union[str, Sequence[Union[str, PixelIndexDirections]], None], optional
        Index convention used in the input affine.
    to_index_convention: Union[str, Sequence[Union[str, PixelIndexDirections]], None], optional
        Desired index convention for the output affine.
    from_reference_convention: Union[str, Sequence[Union[str, PatientFrameOfReferenceDirections]], None], optional
        Reference convention used in the input affine.
    to_reference_convention: Union[str, Sequence[Union[str, PatientFrameOfReferenceDirections]], None], optional
        Desired reference convention for the output affine.

    Returns
    -------
    np.ndarray:
        Affine matrix after operations are applied.

    """  # noqa: E501
    indices_opposites = {
        PixelIndexDirections.U: PixelIndexDirections.D,
        PixelIndexDirections.D: PixelIndexDirections.U,
        PixelIndexDirections.L: PixelIndexDirections.R,
        PixelIndexDirections.R: PixelIndexDirections.L,
        PixelIndexDirections.I: PixelIndexDirections.O,
        PixelIndexDirections.O: PixelIndexDirections.I,
    }
    pfrd = PatientFrameOfReferenceDirections  # shorthand
    reference_opposites = {
        pfrd.L: pfrd.R,
        pfrd.R: pfrd.L,
        pfrd.A: pfrd.P,
        pfrd.P: pfrd.A,
        pfrd.I: pfrd.S,
        pfrd.S: pfrd.I,
    }

    if (from_index_convention is None) != (to_index_convention is None):
        raise TypeError(
            'Arguments "from_index_convention" and "to_index_convention" '
            'should either both be None, or neither should be None.'
        )
    if from_index_convention is not None and to_index_convention is not None:
        from_index_normed = _normalize_pixel_index_convention(
            from_index_convention
        )
        to_index_normed = _normalize_pixel_index_convention(
            to_index_convention
        )
        flip_indices = [
            d not in to_index_normed for d in from_index_normed
        ]

        permute_indices = []
        for d, flipped in zip(to_index_normed, flip_indices):
            if flipped:
                d_ = indices_opposites[d]
                permute_indices.append(from_index_normed.index(d_))
            else:
                permute_indices.append(from_index_normed.index(d))
    else:
        flip_indices = None
        permute_indices = None

    if (
        (from_reference_convention is None) != (to_reference_convention is None)
    ):
        raise TypeError(
            'Arguments "from_reference_convention" and "to_reference_convention" '
            'should either both be None, or neither should be None.'
        )
    if (
        from_reference_convention is not None
        and to_reference_convention is not None
    ):
        from_reference_normed = _normalize_reference_direction_convention(
            from_reference_convention
        )
        to_reference_normed = _normalize_reference_direction_convention(
            to_reference_convention
        )

        flip_reference = [
            d not in to_reference_normed for d in from_reference_normed
        ]
        permute_reference = []
        for d, flipped in zip(to_reference_normed, flip_reference):
            if flipped:
                d_ = reference_opposites[d]
                permute_reference.append(from_reference_normed.index(d_))
            else:
                permute_reference.append(from_reference_normed.index(d))
    else:
        flip_reference = None
        permute_reference = None

    return _transform_affine_matrix(
        affine=affine,
        shape=shape,
        permute_indices=permute_indices,
        permute_reference=permute_reference,
        flip_indices=flip_indices,
        flip_reference=flip_reference,
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
    ...     pixel_spacing=[0.5, 0.5],
    ... )
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
        return self._affine.copy()

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
        return self._affine.copy()

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
        reference_coordinates = np.row_stack([
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
        return self._affine.copy()

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
        pixel_matrix_coordinates = np.row_stack([
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
        return self._affine.copy()

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
        return self._affine.copy()

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
        reference_coordinates = np.row_stack([
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
        return self._affine.copy()

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
        image_coordinates = np.row_stack([
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


class VolumeGeometry:

    """Class representing the geomtry of a regularly-spaced 3D array.

    All such geometries exist within DICOM's patient coordinate system.

    Internally this class uses the following conventions to represent the
    geometry, however this can be constructed from or transformed to other
    conventions with appropriate optional parameters to its methods:

    * The pixel indices are ordered (slice index, row index, column index).
    * Pixel indices are zero-based and represent the center of the pixel.
    * Column indices are ordered top to bottom, row indices are ordered left to
      right. The interpretation of the slice indices direction is not defined.
    * The x, y, z coordinates of frame-of-reference coordinate system follow
      the "LPS" convention used in DICOM (see
      :dcm:`Part 3 Section C.7.6.2.1.1 <part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1>`).
      I.e.
      * The first coordinate (``x``) increases from the patient's right to left
      * The second coordinate (``y``) increases from the patient's anterior to
        posterior.
      * The third coordinate (``z``) increases from the patient's caudal
        direction (inferior) to cranial direction (superior).

    Note
    ----
    The ordering of pixel indices used by this class (slice, row, column)
    matches the way pydicom and highdicom represent pixel arrays but differs
    from the (column, row, slice) convention used by the various "transformer"
    classes in the ``highdicom.spatial`` module.

    """
    def __init__(
        self,
        affine: np.ndarray,
        shape: Sequence[int],
        frame_of_reference_uid: Optional[str] = None,
        sop_instance_uids: Optional[Sequence[str]] = None,
        frame_numbers: Optional[Sequence[int]] = None,
    ):
        """

        Parameters
        ----------
        affine: np.ndarray
            4 x 4 affine matrix representing the transformation from pixel
            indices (slice index, row index, column index) to the
            frame-of-reference coordinate system. The top left 3 x 3 matrix
            should be a scaled orthogonal matrix representing the rotation and
            scaling. The top right 3 x 1 vector represents the translation
            component. The last row should have value [0, 0, 0, 1].
        shape: Sequence[int]
            Shape (slices, rows, columns) of the implied volume array.
        frame_of_reference_uid: Optional[str], optional
            Frame of reference UID for the frame of reference, if known.
        sop_instance_uids: Optional[Sequence[str]], optional
            SOP instance UIDs corresponding to each slice (stacked down
            dimension 0) of the implied volume. This is relevant if and only if
            the volume is formed from a series of single frame DICOM images.
        frame_numbers: Optional[Sequence[int]], optional
            Frame numbers of corresponding to each slice (stacked down
            dimension 0) of the implied volume. This is relevant if and only if
            the volume is formed from a set of frames of a single multiframe
            DICOM image.

        """

        if affine.shape != (4, 4):
            raise ValueError("Affine matrix must have shape (4, 4).")
        if not np.array_equal(affine[-1, :], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError(
                "Final row of affine matrix must be [0.0, 0.0, 0.0, 1.0]."
            )
        if not _is_matrix_orthogonal(affine[:3, :3]):
            raise ValueError(
                "Argument 'affine' must be an orthogonal matrix."
            )
        if len(shape) != 3:
            raise ValueError(
                "Argument 'shape' must have three elements."
            )

        self._affine = affine
        if len(shape) != 3:
            raise ValueError("Argument 'shape' must have three items.")
        self._shape = tuple(shape)
        self._frame_of_reference_uid = frame_of_reference_uid
        if frame_numbers is not None:
            if any(not isinstance(f, int) for f in frame_numbers):
                raise TypeError(
                    "Argument 'frame_numbers' should be a sequence of ints."
                )
            if any(f < 1 for f in frame_numbers):
                raise ValueError(
                    "Argument 'frame_numbers' should contain only (strictly) "
                    "positive integers."
                )
            if len(frame_numbers) != shape[0]:
                raise ValueError(
                    "Length of 'frame_numbers' should match first item of "
                    "'shape'."
                )
            self._frame_numbers = list(frame_numbers)
        else:
            self._frame_numbers = None
        if sop_instance_uids is not None:
            if any(not isinstance(u, str) for u in sop_instance_uids):
                raise TypeError(
                    "Argument 'sop_instance_uids' should be a sequence of "
                    "str."
                )
            if len(sop_instance_uids) != shape[0]:
                raise ValueError(
                    "Length of 'sop_instance_uids' should match first item "
                    "of 'shape'."
                )
            self._sop_instance_uids = list(sop_instance_uids)
        else:
            self._sop_instance_uids = None

    @classmethod
    def for_image_series(
        cls,
        series_datasets: Sequence[Dataset],
    ) -> "VolumeGeometry":
        """Get volume geometry for a series of single frame images.

        Parameters
        ----------
        series_datasets: Sequence[pydicom.Dataset]
            Series of single frame datasets. There is no requirement on the
            sorting of the datasets.

        Returns
        -------
        VolumeGeometry:
            Object representing the geometry of the series.

        """
        coordinate_system = get_image_coordinate_system(series_datasets[0])
        if (
            coordinate_system is None or
            coordinate_system != CoordinateSystemNames.PATIENT
        ):
            raise ValueError(
                "Dataset should exist in the patient "
                "coordinate_system."
            )
        frame_of_reference_uid = series_datasets[0].FrameOfReferenceUID
        if not all(
            ds.FrameOfReferenceUID == frame_of_reference_uid
            for ds in series_datasets
        ):
            raise ValueError('Images do not share a frame of reference.')

        series_datasets = sort_datasets(series_datasets)
        sorted_sop_instance_uids = [
            ds.SOPInstanceUID for ds in series_datasets
        ]

        slice_spacing = get_series_slice_spacing(series_datasets)
        if slice_spacing is None:
            raise ValueError('Series is not a regularly spaced volume.')
        ds = series_datasets[0]
        shape = (len(series_datasets), ds.Rows, ds.Columns)
        affine = _create_affine_transformation_matrix(
            image_position=ds.ImagePositionPatient,
            image_orientation=ds.ImageOrientationPatient,
            pixel_spacing=ds.PixelSpacing,
            spacing_between_slices=slice_spacing,
            index_convention=(
                PixelIndexDirections.I,
                PixelIndexDirections.D,
                PixelIndexDirections.R,
            ),
        )

        return cls(
            affine=affine,
            shape=shape,
            frame_of_reference_uid=frame_of_reference_uid,
            sop_instance_uids=sorted_sop_instance_uids,
        )

    @classmethod
    def for_image(
        cls,
        dataset: Dataset,
    ) -> "VolumeGeometry":
        """Get volume geometry for a multiframe image.

        Parameters
        ----------
        dataset: pydicom.Dataset
            A multi-frame image dataset.

        Returns
        -------
        VolumeGeometry:
            Object representing the geometry of the image.

        """
        if not is_multiframe_image(dataset):
            raise ValueError(
                'Dataset should be a multi-frame image.'
            )
        coordinate_system = get_image_coordinate_system(dataset)
        if (
            coordinate_system is None or
            coordinate_system != CoordinateSystemNames.PATIENT
        ):
            raise ValueError(
                "Dataset should exist in the patient "
                "coordinate_system."
            )
        sfgs = dataset.SharedFunctionalGroupsSequence[0]
        if 'PlaneOrientationSequence' not in sfgs:
            raise ValueError('Frames do not share an orientation.')
        image_orientation = (
            sfgs
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        pffgs = dataset.PerFrameFunctionalGroupsSequence
        image_positions = [
            g.PlanePositionSequence[0].ImagePositionPatient
            for g in pffgs
        ]
        sort_index = get_plane_sort_index(
            image_positions,
            image_orientation,
        )
        sorted_positions = [image_positions[i] for i in sort_index]
        sorted_frame_numbers = [f + 1 for f in sort_index]

        if 'PixelMeasuresSequence' not in sfgs:
            raise ValueError('Frames do not share pixel measures.')
        pixel_spacing = sfgs.PixelMeasuresSequence[0].PixelSpacing

        slice_spacing = get_regular_slice_spacing(
            image_positions=image_positions,
            image_orientation=image_orientation,
        )
        if slice_spacing is None:
            raise ValueError(
                'Dataset does not represent a regularly sampled volume.'
            )

        shape = (dataset.NumberOfFrames, dataset.Rows, dataset.Columns)
        affine = _create_affine_transformation_matrix(
            image_position=sorted_positions[0],
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=slice_spacing,
            index_convention=(
                PixelIndexDirections.I,
                PixelIndexDirections.D,
                PixelIndexDirections.R,
            ),
        )

        return cls(
            affine=affine,
            shape=shape,
            frame_of_reference_uid=dataset.FrameOfReferenceUID,
            frame_numbers=sorted_frame_numbers,
        )

    @classmethod
    def from_attributes(
        cls,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float],
        spacing_between_slices: float,
        rows:int,
        columns: int,
        number_of_frames: int,
        frame_of_reference_uid: Optional[str] = None,
        sop_instance_uids: Optional[Sequence[str]] = None,
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> "VolumeGeometry":
        """Create a volume geometry from DICOM attributes.

        Parameters
        ----------
        image_position: Sequence[float]
            Position in the frame of reference space of the center of the top
            left pixel of the image. Corresponds to DICOM attributes
            "ImagePositionPatient". Should be a sequence of length 3.
        image_orientation: Sequence[float]
            Cosines of the row direction (first triplet: horizontal, left to
            right, increasing column index) and the column direction (second
            triplet: vertical, top to bottom, increasing row index) direction
            expressed in the three-dimensional patient or slide coordinate
            system defined by the frame of reference. Corresponds to the DICOM
            attribute "ImageOrientationPatient".
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing row index) and the row direction (second value:
            spacing between columns: horizontal, left to right, increasing
            column index). Corresponds to DICOM attribute "PixelSpacing".
        spacing_between_slices: float
            Spacing between slices in millimeter units in the frame of
            reference coordinate system space. Corresponds to the DICOM
            attribute "SpacingBetweenSlices" (however, this may not be present in
            many images and may need to be inferred from "ImagePositionPatient"
            attributes of consecutive slices).
        rows:int
            Number of rows in the image. Corresponds to the DICOM attribute
            "Rows".
        columns: int
            Number of columns in the image. Corresponds to the DICOM attribute
            "Columns".
        number_of_frames: int
            Number of frames in the image. Corresponds to NumberOfFrames
            attribute, or to the number of images in the case of an image
            series.
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID, if known. Corresponds to DICOM attribute
            FrameOfReferenceUID.
        sop_instance_uids: Union[Sequence[str], None], optional
            Ordered SOP Instance UIDs of each frame, if known, in the situation
            that the volume is formed from a sequence of individual DICOM
            instances.
        frame_numbers: Union[Sequence[int], None], optional
            Ordered frame numbers of each frame, if known, in the situation
            that the volume is formed from a sequence of frames of one
            multi-frame DICOM image.

        """
        affine = _create_affine_transformation_matrix(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=spacing_between_slices,
            index_convention=(
                PixelIndexDirections.I,
                PixelIndexDirections.D,
                PixelIndexDirections.R,
            ),
        )
        shape = (number_of_frames, rows, columns)
        return cls(
            affine=affine,
            shape=shape,
            frame_of_reference_uid=frame_of_reference_uid,
            sop_instance_uids=sop_instance_uids,
            frame_numbers=frame_numbers,
        )

    @classmethod
    def from_components(
        cls,
        position: Sequence[float],
        direction: Sequence[float],
        spacing: Sequence[float],
        shape: Sequence[int],
        frame_of_reference_uid: Optional[str] = None,
        sop_instance_uids: Optional[Sequence[str]] = None,
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> "VolumeGeometry":
        """"""
        if not isinstance(position, Sequence):
            raise TypeError('Argument "position" must be a sequence.')
        if len(position) != 3:
            raise ValueError('Argument "position" must have length 3.')
        if not isinstance(spacing, Sequence):
            raise TypeError('Argument "spacing" must be a sequence.')
        if len(spacing) != 3:
            raise ValueError('Argument "spacing" must have length 3.')
        direction_arr = np.array(direction, dtype=np.float32)
        if direction_arr.shape == (9, ):
            direction_arr = direction_arr.reshape(3, 3)
        elif direction_arr.shape == (3, 3):
            pass
        else:
            raise ValueError(
                "Argument 'direction' must have shape (9, ) or (3, 3)."
            )
        scaled_direction = direction_arr * spacing
        affine = np.row_stack(
            [
                np.column_stack([scaled_direction, position]),
                [0.0, 0.0, 0.0, 1.0]
            ]
        )
        return cls(
            affine=affine,
            shape=shape,
            frame_of_reference_uid=frame_of_reference_uid,
            sop_instance_uids=sop_instance_uids,
            frame_numbers=frame_numbers,
        )

    def get_index_for_frame_number(
        self,
        frame_number: int,
    ) -> int:
        """Get the slice index for a frame number.

        This is intended for volumes representing for multi-frame images.

        Parameters
        ----------
        frame_number: int
            1-based frame number in the original image.

        Returns
        -------
            0-based index of this frame number down the
            slice dimension (axis 0) of the volume.

        """
        if self._frame_numbers is None:
            raise RuntimeError(
                "Frame information is not present."
            )
        return self._frame_numbers.index(frame_number)

    def get_index_for_sop_instance_uid(
        self,
        sop_instance_uid: str,
    ) -> int:
        """Get the slice index for a SOP Instance UID.

        This is intended for volumes representing a series of single-frame
        images.

        Parameters
        ----------
        sop_instance_uid: str
            SOP Instance of a particular image in the series.

        Returns
        -------
            0-based index of the image with the given SOP Instance UID down the
            slice dimension (axis 0) of the volume.

        """
        if self._sop_instance_uids is None:
            raise RuntimeError(
                "SOP Instance UID information is not present."
            )
        return self._sop_instance_uids.index(sop_instance_uid)

    @property
    def frame_of_reference_uid(self) -> Optional[str]:
        """Union[str, None]: Frame of reference UID."""
        return self._frame_of_reference_uid

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix"""
        return self._affine.copy()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Shape of the volume."""
        return self._shape

    @property
    def sop_instance_uids(self) -> Union[List[str], None]:
        """Union[List[str], None]: SOP Instance UID at each index."""
        if self._sop_instance_uids is not None:
            return self._sop_instance_uids.copy()

    @property
    def frame_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]: Frame number at each index."""
        if self._frame_numbers is not None:
            return self._frame_numbers.copy()

    @property
    def direction_cosines(self) -> List[float]:
        vec_along_rows = self._affine[:3, 2].copy()
        vec_along_columns = self._affine[:3, 1].copy()
        vec_along_columns /= np.sqrt((vec_along_columns ** 2).sum())
        vec_along_rows /= np.sqrt((vec_along_rows ** 2).sum())
        return [*vec_along_rows.tolist(), *vec_along_columns.tolist()]

    @property
    def pixel_spacing(self) -> List[float]:
        """List[float]: Within-plane pixel spacing in millimeter units. Two
        values (spacing between rows, spacing between columns)."""
        vec_along_rows = self._affine[:3, 2]
        vec_along_columns = self._affine[:3, 1]
        spacing_between_columns = np.sqrt((vec_along_rows ** 2).sum()).item()
        spacing_between_rows = np.sqrt((vec_along_columns ** 2).sum()).item()
        return [spacing_between_rows, spacing_between_columns]

    @property
    def spacing_between_slices(self) -> float:
        """float: Spacing between consecutive slices in millimeter units."""
        slice_vec = self._affine[:3, 0]
        spacing = np.sqrt((slice_vec ** 2).sum()).item()
        return spacing

    @property
    def spacing(self) -> List[float]:
        dir_mat = self._affine[:3, :3]
        norms = np.sqrt((dir_mat ** 2).sum(axis=0))
        return norms.tolist()

    @property
    def position(self) -> List[float]:
        return self._affine[:3, 3].tolist()

    @property
    def direction(self) -> np.ndarray:
        dir_mat = self._affine[:3, :3]
        norms = np.sqrt((dir_mat ** 2).sum(axis=0))
        return dir_mat / norms


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


def get_series_slice_spacing(
    datasets: Sequence[pydicom.Dataset],
    tol: float = _DEFAULT_SPACING_TOLERANCE,
) -> Optional[float]:
    """Get slice spacing, if any, for a series of single frame images.

    First determines whether the image series represents a 3D volume.
    A 3D volume consists of regularly spaced slices with orthogonal axes, i.e.
    the slices are spaced equally along the direction orthogonal to the
    in-plane image coordinates.

    If the series does represent a volume, returns the absolute value of the
    slice spacing. If the series does not represent a volume, returns None.

    Note that we stipulate that a single image is a 3D volume for the purposes
    of this function. In this case the returned slice spacing will be 0.0.

    Parameters
    ----------
    datasets: Sequence[pydicom.Dataset]
        Set of datasets representing an imaging series.
    tol: float
        Tolerance for determining spacing regularity. If slice spacings vary by
        less that this spacing, they are considered to be regular.

    Returns
    -------
    float:
        Absolute value of the regular slice spacing if the series of images
        meets the definition of a 3D volume, above. None otherwise.

    """
    if len(datasets) == 0:
        raise ValueError("List must not be empty.")
    # We stipluate that a single image does represent a volume with spacing 0.0
    if len(datasets) == 1:
        return 0.0
    for ds in datasets:
        if is_multiframe_image(ds):
            raise ValueError(
                "Datasets should be single-frame images."
            )

    # Check image orientations are consistent
    image_orientation = datasets[0].ImageOrientationPatient
    for ds in datasets[1:]:
        if ds.ImageOrientationPatient != image_orientation:
            return None

    positions = np.array(
        [ds.ImagePositionPatient for ds in datasets]
    )

    return get_regular_slice_spacing(
        image_positions=positions,
        image_orientation=np.array(image_orientation),
        tol=tol,
    )


def get_regular_slice_spacing(
    image_positions: Sequence[Sequence[float]],
    image_orientation: Sequence[float],
    tol: float = _DEFAULT_SPACING_TOLERANCE,
    sort: bool = True,
    enforce_right_handed: bool = False,
) -> Optional[float]:
    """Get the regular spacing between set of image positions, if any.

    A 3D volume consists of regularly spaced slices with orthogonal axes, i.e.
    the slices are spaced equally along the direction orthogonal to the
    in-plane image coordinates.

    Note that we stipulate that a single image is a 3D volume for the purposes
    of this function. In this case the returned slice spacing will be 0.0.

    Parameters
    ----------
    image_positions: Sequence[Sequence[float]]
        Array of image positions for multiple frames. Should be a 2D array of
        shape (N, 3) where N is the number of frames. Either a numpy array or
        anything convertible to it may be passed.
    image_orientation: Sequence[float]
        Image orientation as direction cosine values taken directly from the
        ImageOrientationPatient attribute. 1D array of length 6. Either a numpy
        array or anything convertible to it may be passed.
    tol: float
        Tolerance for determining spacing regularity. If slice spacings vary by
        less that this spacing, they are considered to be regular.
    sort: bool
        Sort the image positions before finding the spacing. If True, this
        makes the function tolerant of unsorted inputs. Set to False to check
        whether the positions represent a 3D volume in the specific order in
        which they are passed.
    enforce_positive: bool
        If True and sort is False, require that the images are not only
        regularly spaced but also that they are ordered correctly to give a
        right-handed coordinate system, i.e. frames are ordered along the
        direction of the increasing normal vector, as opposed to being ordered
        regularly along the direction of the decreasing normal vector. If sort
        is True, this has no effect since positions will be sorted in the
        right-handed direction before finding the spacing.

    Returns
    -------
    Union[float, None]
        If the image positions are regularly spaced, the (absolute value of) the
        slice spacing. If the image positions are not regularly spaced, returns
        None.

    """
    image_positions = np.array(image_positions)

    if image_positions.ndim != 2 or image_positions.shape[1] != 3:
        raise ValueError(
            "Argument 'image_positions' should be an (N, 3) array."
        )
    n = image_positions.shape[0]
    if n == 0:
        raise ValueError(
            "Argument 'image_positions' should contain at least 1 position."
        )
    elif n == 1:
        # Special case, we stipluate that this has spacing 0.0
        return 0.0

    normal_vector = get_normal_vector(image_orientation)

    # Calculate distance of each slice from coordinate system origin along the
    # normal vector
    origin_distances = _get_slice_distances(image_positions, normal_vector)

    if sort:
        sort_index = np.argsort(origin_distances)
        origin_distances = origin_distances[sort_index]
    else:
        sort_index = np.arange(image_positions.shape[0])

    spacings = np.diff(origin_distances)
    avg_spacing = spacings.mean()

    is_regular = np.isclose(
        avg_spacing,
        spacings,
        atol=tol
    ).all()
    if is_regular and enforce_right_handed:
        if avg_spacing < 0.0:
            return None

    # Additionally check that the vector from the first to the last plane lies
    # approximately along the normal vector
    pos1 = image_positions[sort_index[0], :]
    pos2 = image_positions[sort_index[-1], :]
    span = (pos2 - pos1)
    span /= np.linalg.norm(span)

    is_perpendicular = abs(normal_vector.T @ span - 1.0) < tol

    if is_regular and is_perpendicular:
        return abs(avg_spacing)
    else:
        return None


def get_normal_vector(
    image_orientation: Sequence[float],
):
    """Get a vector normal to an imaging plane.

    Parameters
    ----------
    image_orientation: Sequence[float]
        Image orientation in the standard DICOM format used for the
        ImageOrientationPatient and ImageOrientationSlide attributes,
        consisting of 6 numbers representing the direction cosines along the
        rows (first three elements) and columns (second three elements).

    Returns
    -------
    np.ndarray:
        Unit normal vector as a NumPy array with shape (3, ).

    """
    image_orientation = np.array(image_orientation)
    if image_orientation.ndim != 1 or image_orientation.shape[0] != 6:
        raise ValueError(
            "Argument 'image_orientation' should be an array of "
            "length 6."
        )

    # Find normal vector to the imaging plane
    v1 = image_orientation[:3]
    v2 = image_orientation[3:]
    v3 = np.cross(v1, v2)

    return v3


def get_plane_sort_index(
    image_positions: Sequence[Sequence[float]],
    image_orientation: Sequence[float],
) -> List[int]:
    """

    Parameters
    ----------
    image_positions: Sequence[Sequence[float]]
        Array of image positions for multiple frames. Should be a 2D array of
        shape (N, 3) where N is the number of frames. Either a numpy array or
        anything convertible to it may be passed.
    image_orientation: Sequence[float]
        Image orientation as direction cosine values taken directly from the
        ImageOrientationPatient attribute. 1D array of length 6. Either a numpy
        array or anything convertible to it may be passed.

    Returns
    -------
    List[int]
        Sorting index for the input planes. Element i of this list gives the
        index in the original list of the frames such that the output list
        is sorted along the positive direction of the normal vector of the
        imaging plane.

    """
    pos_arr = np.array(image_positions)
    if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
        raise ValueError("Argument 'image_positions' must have shape (N, 3)")
    ori_arr = np.array(image_orientation)
    if ori_arr.ndim != 1 or ori_arr.shape[0] != 6:
        raise ValueError("Argument 'image_orientation' must have shape (6, )")

    normal_vector = get_normal_vector(ori_arr)

    # Calculate distance of each slice from coordinate system origin along the
    # normal vector
    origin_distances = _get_slice_distances(pos_arr, normal_vector)

    sort_index = np.argsort(origin_distances)

    return sort_index.tolist()


def get_dataset_sort_index(datasets: Sequence[Dataset]) -> List[int]:
    """Get index to sort single frame datasets spatially.

    Parameters
    ----------
    datasets: Sequence[pydicom.Dataset]
        Datasets containing single frame images, with a consistent orientation.

    Returns
    -------
    List[int]
        Sorting index for the input datasets. Element i of this list gives the
        index in the original list of datasets such that the output list is
        sorted along the positive direction of the normal vector of the imaging
        plane.

    """
    if is_multiframe_image(datasets[0]):
        raise ValueError('Datasets should be single frame images.')
    if 'ImageOrientationPatient' not in datasets[0]:
        raise AttributeError(
            'Datasets do not have an orientation.'
        )
    image_orientation = datasets[0].ImageOrientationPatient
    if not all(
        np.allclose(ds.ImageOrientationPatient, image_orientation)
        for ds in datasets
    ):
        raise ValueError('Datasets do not have a consistent orientation.')
    positions = [ds.ImagePositionPatient for ds in datasets]
    return get_plane_sort_index(positions, image_orientation)


def sort_datasets(datasets: Sequence[Dataset]) -> List[Dataset]:
    """Sort single frame datasets spatially.

    Parameters
    ----------
    datasets: Sequence[pydicom.Dataset]
        Datasets containing single frame images, with a consistent orientation.

    Returns
    -------
    List[Dataset]
        Sorting index for the input datasets. Element i of this list gives the
        index in the original list of datasets such that the output list is
        sorted along the positive direction of the normal vector of the imaging
        plane.

    """
    sort_index = get_dataset_sort_index(datasets)
    return [datasets[i] for i in sort_index]


def _get_slice_distances(
    image_positions: np.ndarray,
    normal_vector: np.ndarray,
) -> np.ndarray:
    """Get distances of a set of planes from the origin.

    For each plane position, find (signed) distance from origin along the vector normal
    to the imaging plane.

    Parameters
    ----------
    image_positions: np.ndarray
        Image positions array. 2D array of shape (N, 3) where N is the number of
        planes and each row gives the (x, y, z) image position of a plane.
    normal_vector: np.ndarray
        Unit normal vector (perpendicular to the imaging plane).

    Returns
    -------
    np.ndarray:
        1D array of shape (N, ) giving signed distance from the origin of each
        plane position.

    """
    origin_distances = normal_vector[None] @ image_positions.T
    origin_distances = origin_distances.squeeze(0)

    return origin_distances
