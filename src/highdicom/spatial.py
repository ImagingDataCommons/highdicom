import itertools
from collections.abc import Generator, Iterator, Sequence
from typing_extensions import Self

from pydicom import Dataset
import numpy as np
import pydicom

from highdicom._module_utils import is_multiframe_image
from highdicom.enum import (
    AxisHandedness,
    CoordinateSystemNames,
    PixelIndexDirections,
    PatientOrientationValuesBiped,
)


_DEFAULT_SPACING_RELATIVE_TOLERANCE = 1e-2
"""Default tolerance for determining whether slices are regularly spaced."""


_DEFAULT_EQUALITY_TOLERANCE = 1e-5
"""Tolerance value used by default in tests for equality"""

_DOT_PRODUCT_PERPENDICULAR_TOLERANCE = 1e-3
"""Tolerance value used on the dot product to determine perpendicularity."""


PATIENT_ORIENTATION_OPPOSITES = {
    PatientOrientationValuesBiped.L: PatientOrientationValuesBiped.R,
    PatientOrientationValuesBiped.R: PatientOrientationValuesBiped.L,
    PatientOrientationValuesBiped.A: PatientOrientationValuesBiped.P,
    PatientOrientationValuesBiped.P: PatientOrientationValuesBiped.A,
    PatientOrientationValuesBiped.F: PatientOrientationValuesBiped.H,
    PatientOrientationValuesBiped.H: PatientOrientationValuesBiped.F,
}
"""Mapping of each patient orientation value to its opposite."""


VOLUME_INDEX_CONVENTION = (
    PixelIndexDirections.D,
    PixelIndexDirections.R,
)
"""Indexing convention used within volumes."""


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
) -> Iterator[tuple[int, int]]:
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
    numpy.ndarray:
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
) -> list[tuple[list[int], list[float]]]:
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
) -> Generator[
    tuple[int | None, int, int, int, float, float, float],
    None,
    None,
]:
    """Get data on the position of each tile in a TILED_FULL image.

    This works only with images with Dimension Organization Type of
    "TILED_FULL".

    Unlike :func:`highdicom.utils.compute_plane_position_slide_per_frame`,
    this functions returns the data in their basic Python types rather than
    wrapping as instances of :class:`highdicom.PlanePositionSequence`.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        VL Whole Slide Microscopy Image or Segmentation Image using the
        "TILED_FULL" DimensionOrganizationType.

    Returns
    -------
    channel: Union[int, None]
        1-based integer index of the "channel". The meaning of "channel"
        depends on the image type. For segmentation images, the channel is the
        segment number. For other images, it is the optical path number. For
        Segmentations of SegmentationType "LABELMAP", the returned value will
        be None for all frames.
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
        '1.2.840.10008.5.1.4.1.1.66.7',  # Label Map Segmentation Image
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

    is_segmentation = dataset.SOPClassUID in (
        '1.2.840.10008.5.1.4.1.1.66.4',
        '1.2.840.10008.5.1.4.1.1.66.7',
    )

    # The "channels" output is either segment for segmentations, or optical
    # path for other images
    if is_segmentation:
        if dataset.SegmentationType == "LABELMAP":
            # No "channel" in this case -> return None
            channels = [None]
        else:
            channels = range(1, len(dataset.SegmentSequence) + 1)
    else:
        num_optical_paths = getattr(
            dataset,
            'NumberOfOpticalPaths',
            len(dataset.OpticalPathSequence)
        )
        channels = range(1, num_optical_paths + 1)

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

    for channel in channels:
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
) -> CoordinateSystemNames | None:
    """Get the coordinate system used by an image.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Dataset representing an image.

    Returns
    --------
    Union[highdicom.CoordinateSystemNames, None]
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
        # Some images can have a frame of reference UID but no further position
        # or orientation information

        # Single frame images should have image position at the root
        if 'ImagePositionPatient' in dataset:
            return CoordinateSystemNames.PATIENT

        for kw in [
            'SharedFunctionalGroupsSequence',
            'PerFrameFunctionalGroupsSequence',
        ]:
            fgs = dataset.get(kw)
            if fgs is not None:
                if 'PlanePositionSequence' in fgs[0]:
                    pps = fgs[0].PlanePositionSequence[0]
                    if 'ImagePositionPatient' in pps:
                        return CoordinateSystemNames.PATIENT

        # No position information found: infer that there is no coordinate
        # system
        return None


def _get_spatial_information(
    dataset: Dataset,
    frame_number: int | None = None,
    for_total_pixel_matrix: bool = False,
) -> tuple[
    list[float],
    list[float],
    list[float],
    float | None,
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


def _are_images_coplanar(
    image_position_a: Sequence[float],
    image_orientation_a: Sequence[float],
    image_position_b: Sequence[float],
    image_orientation_b: Sequence[float],
    tol: float = _DEFAULT_EQUALITY_TOLERANCE,
) -> bool:
    """Determine whether two images or image frames are coplanar.

    Two images are coplanar in the frame of reference coordinate system if and
    only if their normal vectors have the same (or opposite direction) and the
    shortest distance from the plane to the coordinate system origin is the
    same for both planes.

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
    n_a = get_normal_vector(image_orientation_a)
    n_b = get_normal_vector(image_orientation_b)
    if 1.0 - np.abs(n_a @ n_b) > tol:
        return False

    # Find distances of both planes along n_a
    dis_a = np.array(image_position_a, dtype=float) @ n_a
    dis_b = np.array(image_position_b, dtype=float) @ n_a

    return abs(dis_a - dis_b) < tol


def _normalize_pixel_index_convention(
    c: str | Sequence[str | PixelIndexDirections],
) -> tuple[PixelIndexDirections, PixelIndexDirections]:
    """Normalize and check a pixel index convention.

    Parameters
    ----------
    c: Union[str, Sequence[Union[str, highdicom.enum.PixelIndexDirections]]]
        Pixel index convention description consisting of two directions,
        either L or R, and either U or D.

    Returns
    -------
    Tuple[highdicom.enum.PixelIndexDirections, highdicom.enum.PixelIndexDirections]:
        Convention description in a canonical form as a tuple of two enum
        instances. Furthermore this is guaranteed to be a valid description.

    """  # noqa: E501
    if len(c) != 2:
        raise ValueError('Length of pixel index convention must be 2.')

    c = tuple(PixelIndexDirections(d) for d in c)

    c_set = {d.value for d in c}

    criteria = [
        ('L' in c_set) != ('R' in c_set),
        ('U' in c_set) != ('D' in c_set),
    ]
    if not all(criteria):
        c_str = [d.value for d in c]
        raise ValueError(f'Invalid combination of pixel directions: {c_str}.')

    return c


def _normalize_patient_orientation(
    c: str | Sequence[str | PatientOrientationValuesBiped],
) -> tuple[
    PatientOrientationValuesBiped,
    PatientOrientationValuesBiped,
    PatientOrientationValuesBiped,
]:
    """Normalize and check a patient orientation.

    Parameters
    ----------
    c: Union[str, Sequence[Union[str, highdicom.enum.PatientOrientationValuesBiped]]]
        Patient orientation consisting of three directions, either L or R,
        either A or P, and either F or H, in any order.

    Returns
    -------
    Tuple[highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped]:
        Convention description in a canonical form as a tuple of three enum
        instances. Furthermore this is guaranteed to be a valid description.

    """  # noqa: E501
    if len(c) != 3:
        raise ValueError('Length of pixel index convention must be 3.')

    c = tuple(PatientOrientationValuesBiped(d) for d in c)

    c_set = {d.value for d in c}

    criteria = [
        ('L' in c_set) != ('R' in c_set),
        ('A' in c_set) != ('P' in c_set),
        ('F' in c_set) != ('H' in c_set),
    ]
    if not all(criteria):
        c_str = [d.value for d in c]
        raise ValueError(
            'Invalid combination of frame of reference directions: '
            f'{c_str}.'
        )

    return c


def get_closest_patient_orientation(affine: np.ndarray) -> tuple[
    PatientOrientationValuesBiped,
    PatientOrientationValuesBiped,
    PatientOrientationValuesBiped,
]:
    """Given an affine matrix, find the closest patient orientation.

    Parameters
    ----------
    affine: numpy.ndarray
        Direction matrix (4x4 affine matrices and 3x3 direction matrices are
        acceptable).

    Returns
    -------
    Tuple[PatientOrientationValuesBiped, PatientOrientationValuesBiped, PatientOrientationValuesBiped]:
        Tuple of PatientOrientationValuesBiped values, giving for each of the
        three axes of the volume represented by the affine matrix, the closest
        direction in the patient frame of reference coordinate system.

    """  # noqa: E501
    if (
        affine.ndim != 2 or
        (
            affine.shape != (3, 3) and
            affine.shape != (4, 4)
        )
    ):
        raise ValueError(f"Invalid shape for array: {affine.shape}")

    if not _is_matrix_orthogonal(affine[:3, :3], require_unit=False):
        raise ValueError('Matrix is not orthogonal.')

    # Matrix representing alignment of dot product of rotation vector i with
    # FoR reference j
    alignments = np.eye(3) @ affine[:3, :3]
    sort_indices = np.argsort(-np.abs(alignments), axis=0)

    result = []
    pos_directions = [
        PatientOrientationValuesBiped.L,
        PatientOrientationValuesBiped.P,
        PatientOrientationValuesBiped.H,
    ]
    neg_directions = [
        PatientOrientationValuesBiped.R,
        PatientOrientationValuesBiped.A,
        PatientOrientationValuesBiped.F,
    ]
    for d, sortind in enumerate(sort_indices.T):
        # Check that this axis has not already been used. This can happen if
        # one or more array axis is at 45 deg to some FoR axis. In this case
        # take the next index in the sort list.
        for i in sortind:
            if (
                pos_directions[i] not in result and
                neg_directions[i] not in result
            ):
                break

        if alignments[i, d] > 0:
            result.append(pos_directions[i])
        else:
            result.append(neg_directions[i])

    return tuple(result)


def _is_matrix_orthogonal(
    m: np.ndarray,
    require_unit: bool = True,
    tol: float = _DEFAULT_EQUALITY_TOLERANCE,
) -> bool:
    """Check whether a matrix is orthogonal.

    Parameters
    ----------
    m: numpy.ndarray
        A matrix.
    require_unit: bool, optional
        Whether to require that the row vectors are unit vectors.
    tol: float, optional
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
    if require_unit:
        if not np.allclose(
            norm_squared,
            np.array([1.0, 1.0, 1.0]),
            atol=tol,
        ):
            return False

    return np.allclose(m.T @ m, np.diag(norm_squared), atol=tol)


def get_normal_vector(
    image_orientation: Sequence[float],
    index_convention: str | Sequence[PixelIndexDirections | str] = (
        PixelIndexDirections.R,
        PixelIndexDirections.D,
    ),
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
) -> np.ndarray:
    """Get a vector normal to an imaging plane.

    Parameters
    ----------
    image_orientation: Sequence[float]
        Image orientation in the standard DICOM format used for the
        ImageOrientationPatient and ImageOrientationSlide attributes,
        consisting of 6 numbers representing the direction cosines along the
        rows (first three elements) and columns (second three elements).
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to index pixels. Should be a sequence of two
        :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Choose the positive direction of the resulting normal in order to give
        this handedness in the resulting coordinate system. This assumes that
        the normal vector will be used to define a coordinate system when
        combined with the column cosines (unit vector pointing down the
        columns) and row cosines (unit vector pointing along the rows) in that
        order (for the sake of handedness, it does not matter whether the axis
        defined by the normal vector is placed before or after the column and
        row vectors because the two possibilities are cyclic permutations of
        each other). If used to define a coordinate system with the row cosines
        followed by the column cosines, the handedness of the resulting
        coordinate system will be inverted.

    Returns
    -------
    numpy.ndarray:
        Unit normal vector as a NumPy array with shape (3, ).

    """  # noqa: E501
    image_orientation_arr = np.array(image_orientation, dtype=np.float64)
    if image_orientation_arr.ndim != 1 or image_orientation_arr.shape[0] != 6:
        raise ValueError(
            "Argument 'image_orientation' should be an array of "
            "length 6."
        )
    index_convention_ = _normalize_pixel_index_convention(index_convention)
    handedness_ = AxisHandedness(handedness)

    # Find normal vector to the imaging plane
    row_cosines = image_orientation_arr[:3]
    column_cosines = image_orientation_arr[3:]

    rotation_columns = []
    for d in index_convention_:
        if d == PixelIndexDirections.R:
            rotation_columns.append(row_cosines)
        elif d == PixelIndexDirections.L:
            rotation_columns.append(-row_cosines)
        elif d == PixelIndexDirections.D:
            rotation_columns.append(column_cosines)
        elif d == PixelIndexDirections.U:
            rotation_columns.append(-column_cosines)

    if handedness_ == AxisHandedness.RIGHT_HANDED:
        n = np.cross(rotation_columns[0], rotation_columns[1])
    else:
        n = np.cross(rotation_columns[1], rotation_columns[0])

    return n


def create_rotation_matrix(
    image_orientation: Sequence[float],
    index_convention: str | Sequence[PixelIndexDirections | str] = (
        PixelIndexDirections.R,
        PixelIndexDirections.D,
    ),
    slices_first: bool = False,
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
    pixel_spacing: float | Sequence[float] = 1.0,
    spacing_between_slices: float = 1.0,
) -> np.ndarray:
    """Builds a rotation matrix (with or without scaling).

    Parameters
    ----------
    image_orientation: Sequence[float]
        Cosines of the row direction (first triplet: horizontal, left to right,
        increasing column index) and the column direction (second triplet:
        vertical, top to bottom, increasing row index) direction expressed in
        the three-dimensional patient or slide coordinate system defined by the
        frame of reference.
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to index pixels. Should be a sequence of two
        :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.
    slices_first: bool, optional
        Whether the slice index dimension is placed before the rows and columns
        (``True``) or after them.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Handedness to use to determine the positive direction of the slice
        index. The resulting rotation matrix will have the given handedness.
    pixel_spacing: Union[float, Sequence[float]], optional
        Spacing between pixels in the in-frame dimensions. Either a single
        value to apply in both the row and column dimensions, or a sequence of
        length 2 giving ``[spacing_between_rows, spacing_between_columns]`` in
        the same format as the DICOM "PixelSpacing" attribute.

    Returns
    -------
    numpy.ndarray
        3 x 3 rotation matrix. Pre-multiplying an image coordinate in the
        format (column index, row index, slice index) by this matrix gives the
        x, y, z position in the frame-of-reference coordinate system.

    """  # noqa: E501
    if len(image_orientation) != 6:
        raise ValueError('Argument "image_orientation" must have length 6.')
    index_convention_ = _normalize_pixel_index_convention(index_convention)
    handedness_ = AxisHandedness(handedness)

    row_cosines = np.array(image_orientation[:3], dtype=float)
    column_cosines = np.array(image_orientation[3:], dtype=float)
    if isinstance(pixel_spacing, (Sequence, np.ndarray)):
        if len(pixel_spacing) != 2:
            raise ValueError(
                "A sequence passed to argument 'pixel_spacing' must have "
                "length 2."
            )
        spacing_between_rows = float(pixel_spacing[0])
        spacing_between_columns = float(pixel_spacing[1])
    else:
        spacing_between_rows = pixel_spacing
        spacing_between_columns = pixel_spacing

    if spacing_between_rows <= 0.0 or spacing_between_columns <= 0.0:
        raise ValueError(
            "All values in 'pixel_spacing' must be positive."
        )

    rotation_columns = []
    spacings = []
    for d in index_convention_:
        if d == PixelIndexDirections.R:
            rotation_columns.append(row_cosines)
            spacings.append(spacing_between_columns)
        elif d == PixelIndexDirections.L:
            rotation_columns.append(-row_cosines)
            spacings.append(spacing_between_columns)
        elif d == PixelIndexDirections.D:
            rotation_columns.append(column_cosines)
            spacings.append(spacing_between_rows)
        elif d == PixelIndexDirections.U:
            rotation_columns.append(-column_cosines)
            spacings.append(spacing_between_rows)

    if handedness_ == AxisHandedness.RIGHT_HANDED:
        n = np.cross(rotation_columns[0], rotation_columns[1])
    else:
        n = np.cross(rotation_columns[1], rotation_columns[0])

    if slices_first:
        rotation_columns.insert(0, n)
        spacings.insert(0, spacing_between_slices)
    else:
        rotation_columns.append(n)
        spacings.append(spacing_between_slices)

    rotation_columns = [c * s for c, s in zip(rotation_columns, spacings)]

    return np.column_stack(rotation_columns)


def _stack_affine_matrix(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Create an affine matrix by stacking together.

    Parameters
    ----------
    rotation: numpy.ndarray
        Numpy array of shape ``(3, 3)`` representing a scaled rotation matrix.
    position: numpy.ndarray
        Numpy array with three elements representing a translation.

    Returns
    -------
    numpy.ndarray:
        Affine matrix of shape ``(4, 4)``.

    """
    if rotation.shape != (3, 3):
        raise ValueError(
            "Argument 'rotation' must have shape (3, 3)."
        )
    if translation.size != 3:
        raise ValueError(
            "Argument 'translation' must have 3 elements."
        )

    return np.vstack(
        [
            np.column_stack([rotation, translation.reshape(3, 1)]),
            [0.0, 0.0, 0.0, 1.0]
        ]
    )


def create_affine_matrix_from_attributes(
    image_position: Sequence[float],
    image_orientation: Sequence[float],
    pixel_spacing: float | Sequence[float],
    spacing_between_slices: float = 1.0,
    index_convention: str | Sequence[PixelIndexDirections | str] = (
        PixelIndexDirections.R,
        PixelIndexDirections.D,
    ),
    slices_first: bool = False,
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
) -> np.ndarray:
    """Create affine matrix from attributes found in a DICOM object.

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
        column index). This matches the format of the DICOM "PixelSpacing"
        attribute. Alternatively, a single value that is used along both
        directions.
    spacing_between_slices: float
        Spacing between consecutive slices in the frame of reference coordinate
        system in millimeter units.
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to index pixels. Should be a sequence of two
        :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.
    slices_first: bool, optional
        Whether the slice index dimension is placed before the rows and columns
        (``True``) or after them.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Handedness to use to determine the positive direction of the slice
        index. The resulting rotation matrix will have the given handedness.

    Returns
    -------
    numpy.ndarray
        4 x 4 affine transformation matrix. pre-multiplying a pixel index in
        format (column index, row index, slice index, 1) as a column vector by
        this matrix gives the (x, y, z, 1) position in the frame-of-reference
        coordinate system.

    """  # noqa: E501
    if not isinstance(image_position, (Sequence, np.ndarray)):
        raise TypeError('Argument "image_position" must be a sequence.')
    if len(image_position) != 3:
        raise ValueError('Argument "image_position" must have length 3.')
    if not isinstance(image_orientation, (Sequence, np.ndarray)):
        raise TypeError('Argument "image_orientation" must be a sequence.')
    if len(image_orientation) != 6:
        raise ValueError('Argument "image_orientation" must have length 6.')
    if not isinstance(pixel_spacing, (Sequence, np.ndarray)):
        raise TypeError('Argument "pixel_spacing" must be a sequence.')
    if len(pixel_spacing) != 2:
        raise ValueError('Argument "pixel_spacing" must have length 2.')

    index_convention_ = _normalize_pixel_index_convention(index_convention)
    if (
        PixelIndexDirections.L in index_convention_ or
        PixelIndexDirections.U in index_convention_
    ):
        raise ValueError(
            "Index convention cannot include 'L' or 'U'."
        )
    translation = np.array([float(x) for x in image_position], dtype=float)

    rotation = create_rotation_matrix(
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=spacing_between_slices,
        index_convention=index_convention_,
        handedness=handedness,
        slices_first=slices_first,
    )

    # 4x4 transformation matrix
    affine = _stack_affine_matrix(rotation, translation)

    return affine


def _create_inv_affine_matrix_from_attributes(
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
        frame-of-reference coordinate in the format (x, y, z, 1) as a column
        vector by this matrix gives the pixel indices in the form (column
        index, row index, slice index, 1).

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

    translation = np.array([float(x) for x in image_position], dtype=float)

    rotation = create_rotation_matrix(
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
        spacing_between_slices=spacing_between_slices,
    )

    inv_rotation = np.linalg.inv(rotation)

    # 4x4 transformation matrix
    return _stack_affine_matrix(
        rotation=inv_rotation,
        translation=-np.dot(inv_rotation, translation)
    )


def rotation_for_patient_orientation(
    patient_orientation: (
        str |
        Sequence[str | PatientOrientationValuesBiped]
    ),
    spacing: float | Sequence[float] = 1.0,
) -> np.ndarray:
    """Create a (scaled) rotation matrix for a given patient orientation.

    The result is an axis-aligned rotation matrix.

    Parameters
    ----------
    patient_orientation: Union[str, Sequence[Union[str, highdicom.enum.PatientOrientationValuesBiped]]]
        Desired patient orientation, as either a sequence of three
        highdicom.enum.PatientOrientationValuesBiped values, or a string
        such as ``"FPL"`` using the same characters.
    spacing: Union[float, Sequence[float]], optional
        Spacing between voxels along each of the three dimensions in the frame
        of reference coordinate system in pixel units.

    Returns
    -------
    numpy.ndarray:
        (Scaled) rotation matrix of shape (3 x 3).

    """  # noqa: E501
    norm_orientation = _normalize_patient_orientation(patient_orientation)

    direction_to_vector_mapping = {
        PatientOrientationValuesBiped.L: np.array([1., 0., 0.]),
        PatientOrientationValuesBiped.R: np.array([-1., 0., 0.]),
        PatientOrientationValuesBiped.P: np.array([0., 1., 0.]),
        PatientOrientationValuesBiped.A: np.array([0., -1., 0.]),
        PatientOrientationValuesBiped.H: np.array([0., 0., 1.]),
        PatientOrientationValuesBiped.F: np.array([0., 0., -1.]),
    }

    if isinstance(spacing, float):
        spacing = [spacing] * 3

    return np.column_stack(
        [
            s * direction_to_vector_mapping[d]
            for d, s in zip(norm_orientation, spacing)
        ]
    )


def create_affine_matrix_from_components(
    *,
    spacing: Sequence[float] | float,
    position: Sequence[float] | None = None,
    center_position: Sequence[float] | None = None,
    direction: Sequence[float] | None = None,
    patient_orientation: (
        str |
        Sequence[str | PatientOrientationValuesBiped] |
        None
    ) = None,
    spatial_shape: Sequence[int] | None = None,
) -> np.ndarray:
    """Construct an affine matrix from components.

    The resulting 4 x 4 affine matrix maps 3D image indices to frame of
    reference coordinates.

    Parameters
    ----------
    spacing: Sequence[float]
        Spacing between pixel centers in the the frame of reference
        coordinate system along each of the dimensions of the array. Should
        be either a sequence of length 3 to give the values along the three
        spatial dimensions, or a single float value to be shared by all
        spatial dimensions.
    position: Sequence[float]
        Sequence of three floats giving the position in the frame of
        reference coordinate system of the center of the voxel at location
        (0, 0, 0).
    center_position: Sequence[float]
        Sequence of three floats giving the position in the frame of
        reference coordinate system of the center of the volume. Note that
        the center of the volume will not lie at the center of any
        particular voxel unless the shape of the array is odd along all
        three spatial dimensions. Incompatible with ``position``.
    direction: Sequence[float]
        Direction matrix for the volume. The columns of the direction
        matrix are orthogonal unit vectors that give the direction in the
        frame of reference space of the increasing direction of each axis
        of the array. This matrix may be passed either as a 3x3 matrix or a
        flattened 9 element array (first row, second row, third row).
    patient_orientation: Union[str, Sequence[Union[str, highdicom.PatientOrientationValuesBiped]]]
        Patient orientation used to define an axis-aligned direction
        matrix, as either a sequence of three
        highdicom.PatientOrientationValuesBiped values, or a string such as
        ``"FPL"`` using the same characters. Incompatible with ``direction``.
    spatial_shape: Sequence[int] | None
        Sequence of three integers giving the shape of the volume. Required
        only if ``center_position`` is used, irrelevant otherwise.

    Returns
    -------
    numpy.ndarray
        4 x 4 affine transformation matrix. pre-multiplying a pixel index in
        format (column index, row index, slice index, 1) as a column vector by
        this matrix gives the (x, y, z, 1) position in the frame-of-reference
        coordinate system.

    """  # noqa: E501
    if (direction is None) == (patient_orientation is None):
        raise TypeError(
            "Exactly one of 'direction' or 'patient_orientation' must be "
            'provided.'
        )
    if (position is None) == (center_position is None):
        raise TypeError(
            "Exactly one of 'position' or 'center_position' must be "
            'provided.'
        )

    if isinstance(spacing, (float, int)):
        spacing_arr = [spacing] * 3
    elif not isinstance(spacing, (Sequence, np.ndarray)):
        raise TypeError('Argument "spacing" must be a sequence or float.')
    spacing_arr = np.array(spacing)
    if len(spacing_arr) != 3:
        raise ValueError('Argument "spacing" must have length 3.')
    if spacing_arr.min() <= 0.0:
        raise ValueError('All items of "spacing" must be positive.')

    if direction is not None:
        direction_arr = np.array(direction, dtype=np.float64)
        if direction_arr.shape == (9, ):
            direction_arr = direction_arr.reshape(3, 3)
        elif direction_arr.shape == (3, 3):
            pass
        else:
            raise ValueError(
                "Argument 'direction' must have shape (9, ) or (3, 3)."
            )
        if not _is_matrix_orthogonal(direction_arr, require_unit=True):
            raise ValueError(
                "Argument 'direction' must be an orthogonal matrix of "
                "unit vectors."
            )
    else:
        direction_arr = rotation_for_patient_orientation(
            patient_orientation
        )

    scaled_direction = direction_arr * spacing

    if position is not None:
        if not isinstance(position, (Sequence, np.ndarray)):
            raise TypeError('Argument "position" must be a sequence.')
        position_arr = np.array(position)
        if position_arr.shape != (3, ):
            raise ValueError('Argument "position" must have length 3.')
    else:
        if spatial_shape is None:
            raise TypeError(
                "Argument 'spatial_shape' must be provided if "
                "'center_position' is used."
            )
        shape_arr = np.array(spatial_shape)
        if shape_arr.shape != (3, ):
            raise ValueError(
                "Argument 'spatial_shape' must have length 3."
            )
        if not isinstance(center_position, (Sequence, np.ndarray)):
            raise TypeError('Argument "center_position" must be a sequence.')
        center_position_arr = np.array(center_position)
        if center_position_arr.shape != (3, ):
            raise ValueError(
                'Argument "center_position" must have length 3.'
            )
        center_index = (shape_arr - 1.0) / 2.0
        position_arr = center_position_arr - scaled_direction @ center_index.T

    affine = _stack_affine_matrix(scaled_direction, position_arr)
    return affine


def _transform_affine_matrix(
    affine: np.ndarray,
    shape: Sequence[int],
    flip_indices: Sequence[bool] | None = None,
    flip_reference: Sequence[bool] | None = None,
    permute_indices: Sequence[int] | None = None,
    permute_reference: Sequence[int] | None = None,
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
        if set(permute_indices) != {0, 1, 2}:
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
        if set(permute_reference) != {0, 1, 2}:
            raise ValueError(
                'Argument "permute_reference" should contain elements 0, 1, '
                "and 3 in some order."
            )
        transformed = transformed[[*permute_reference, 3], :]

    return transformed


def _translate_affine_matrix(
    affine: np.ndarray,
    pixel_offset: Sequence[int],
) -> np.ndarray:
    """Translate the origin of an affine matrix by a pixel offset.

    Parameters
    ----------
    affine: numpy.ndarray
        Original affine matrix (4 x 4).
    pixel_offset: Sequence[int]
        Offset, in pixel units.

    Returns
    -------
    numpy.ndarray:
        Translated affine matrix.

    """
    if len(pixel_offset) != 3:
        raise ValueError(
            "Argument 'pixel_spacing' must have three elements."
        )
    offset_arr = np.array(pixel_offset)
    origin = affine[:3, 3]
    direction = affine[:3, :3]
    reference_offset = direction @ offset_arr
    new_origin = origin + reference_offset
    result = affine.copy()
    result[:3, 3] = new_origin
    return result


def _transform_affine_to_convention(
    affine: np.ndarray,
    shape: Sequence[int],
    from_reference_convention: (
        str | Sequence[str | PatientOrientationValuesBiped]
    ),
    to_reference_convention: (
        str | Sequence[str | PatientOrientationValuesBiped]
    )
) -> np.ndarray:
    """Transform an affine matrix between different conventions.

    Parameters
    ----------
    affine: np.ndarray
        Affine matrix to transform.
    shape: Sequence[int]
        Shape of the array.
    from_reference_convention: Union[str, Sequence[Union[str, PatientOrientationValuesBiped]]],
        Reference convention used in the input affine.
    to_reference_convention: Union[str, Sequence[Union[str, PatientOrientationValuesBiped]]],
        Desired reference convention for the output affine.

    Returns
    -------
    np.ndarray:
        Affine matrix after operations are applied.

    """  # noqa: E501
    from_reference_normed = _normalize_patient_orientation(
        from_reference_convention
    )
    to_reference_normed = _normalize_patient_orientation(
        to_reference_convention
    )

    flip_reference = [
        d not in to_reference_normed for d in from_reference_normed
    ]
    permute_reference = []
    for d, flipped in zip(to_reference_normed, flip_reference):
        if flipped:
            d_ = PATIENT_ORIENTATION_OPPOSITES[d]
            permute_reference.append(from_reference_normed.index(d_))
        else:
            permute_reference.append(from_reference_normed.index(d))

    return _transform_affine_matrix(
        affine=affine,
        shape=shape,
        permute_indices=None,
        permute_reference=permute_reference,
        flip_indices=None,
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
        self._affine = create_affine_matrix_from_attributes(
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
        frame_number: int | None = None,
        for_total_pixel_matrix: bool = False,
    ) -> Self:
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
        self._affine = _create_inv_affine_matrix_from_attributes(
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
        frame_number: int | None = None,
        for_total_pixel_matrix: bool = False,
        round_output: bool = True,
        drop_slice_index: bool = False,
    ) -> Self:
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
            Spacing between pixels of the "from" images in millimeter unit
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
        pix_to_ref = create_affine_matrix_from_attributes(
            image_position=image_position_from,
            image_orientation=image_orientation_from,
            pixel_spacing=pixel_spacing_from,
        )
        ref_to_pix = _create_inv_affine_matrix_from_attributes(
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
        frame_number_from: int | None = None,
        frame_number_to: int | None = None,
        for_total_pixel_matrix_from: bool = False,
        for_total_pixel_matrix_to: bool = False,
        round_output: bool = True,
    ) -> Self:
        """Construct a transformer for two given images or image frames.

        Parameters
        ----------
        dataset_from: pydicom.Dataset
            Dataset representing the image whose pixel indices will be the
            input to the transformer.
        dataset_to: pydicom.Dataset
            Dataset representing the image whose pixel indices will be the
            output of the transformer.
        frame_number_from: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame of
            ``dataset_from`` for which to construct the transformer. This
            should be provided if and only if the dataset is a multi-frame
            image.
        frame_number_to: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame of
            ``dataset_to`` for which to construct the transformer. This
            should be provided if and only if the dataset is a multi-frame
            image.
        for_total_pixel_matrix_from: bool, optional
            If True, use the spatial information for the total pixel matrix of
            the "from" image. The result will be a transformer that whose
            inputs are pixel indices of the total pixel matrix of the "from"
            image. This should only be True if the "from" image is a tiled
            image and is incompatible with specifying ``frame_number_from``.
        for_total_pixel_matrix_to: bool, optional
            If True, use the spatial information for the total pixel matrix of
            the "to" image. The result will be a transformer whose outputs are
            pixel indices of the total pixel matrix of the "to" image. This
            should only be True if the "to" image is a tiled image and is
            incompatible with specifying ``frame_number_to``.
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
        affine = create_affine_matrix_from_attributes(
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
        frame_number: int | None = None,
        for_total_pixel_matrix: bool = False,
    ) -> Self:
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
        affine = _create_inv_affine_matrix_from_attributes(
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
        frame_number: int | None = None,
        for_total_pixel_matrix: bool = False,
        drop_slice_coord: bool = False,
    ) -> Self:
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
            Spacing between pixels of the "from" images in millimeter unit
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
        ref_to_pix = _create_inv_affine_matrix_from_attributes(
            image_position=image_position_to,
            image_orientation=image_orientation_to,
            pixel_spacing=pixel_spacing_to,
        )
        pix_to_ref = create_affine_matrix_from_attributes(
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
        frame_number_from: int | None = None,
        frame_number_to: int | None = None,
        for_total_pixel_matrix_from: bool = False,
        for_total_pixel_matrix_to: bool = False,
    ) -> Self:
        """Construct a transformer for two given images or image frames.

        Parameters
        ----------
        dataset_from: pydicom.Dataset
            Dataset representing the image whose image coordinates will be the
            input to the transformer.
        dataset_to: pydicom.Dataset
            Dataset representing the image whose image coordinates will be the
            output of the transformer.
        frame_number_from: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame of
            ``dataset_from`` for which to construct the transformer. This
            should be provided if and only if the dataset is a multi-frame
            image.
        frame_number_to: Union[int, None], optional
            Frame number (using 1-based indexing) of the frame of
            ``dataset_to`` for which to construct the transformer. This
            should be provided if and only if the dataset is a multi-frame
            image.
        for_total_pixel_matrix_from: bool, optional
            If True, use the spatial information for the total pixel matrix of
            the "from" image. The result will be a transformer that whose
            inputs are image coordinates of the total pixel matrix of the
            "from" image. This should only be True if the "from" image is a
            tiled image and is incompatible with specifying
            ``frame_number_from``.
        for_total_pixel_matrix_to: bool, optional
            If True, use the spatial information for the total pixel matrix of
            the "to" image. The result will be a transformer whose outputs are
            image coordinates of the total pixel matrix of the "to" image. This
            should only be True if the "to" image is a tiled image and is
            incompatible with specifying ``frame_number_to``.

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
) -> tuple[float, float, float]:
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
) -> tuple[int, int, int]:
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


def get_series_volume_positions(
    datasets: Sequence[pydicom.Dataset],
    *,
    rtol: float | None = None,
    atol: float | None = None,
    sort: bool = True,
    allow_missing_positions: bool = False,
    allow_duplicate_positions: bool = False,
    index_convention: (
        str |
        Sequence[PixelIndexDirections | str]
    ) = VOLUME_INDEX_CONVENTION,
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
    enforce_handedness: bool = False,
) -> tuple[float | None, list[int] | None]:
    """Get volume positions and spacing for a series of single frame images.

    First determines whether the image series represents a 3D volume.
    A 3D volume consists of regularly spaced slices with orthogonal axes, i.e.
    the slices are spaced equally along the direction orthogonal to the
    in-plane image coordinates.

    If the series does represent a volume, returns the absolute value of the
    slice spacing and the slice indices in the volume for each of the input
    datasets. If the series does not represent a volume, returns None for both
    outputs.

    Note that we stipulate that a single image is a 3D volume for the purposes
    of this function. In this case the returned slice spacing will be 1.0.

    Parameters
    ----------
    datasets: Sequence[pydicom.Dataset]
        Set of datasets representing an imaging series.
    rtol: float | None, optional
        Relative tolerance for determining spacing regularity. If slice
        spacings vary by less that this proportion of the average spacing, they
        are considered to be regular. If neither ``rtol`` or ``atol`` are
        provided, a default relative tolerance of 0.01 is used.
    atol: float | None, optional
        Absolute tolerance for determining spacing regularity. If slice
        spacings vary by less that this value (in mm), they
        are considered to be regular. Incompatible with ``rtol``.
    sort: bool, optional
        Sort the image positions before finding the spacing. If True, this
        makes the function tolerant of unsorted inputs. Set to False to check
        whether the positions represent a 3D volume in the specific order in
        which they are passed.
    allow_missing_postions: bool, optional
        Allow for slices missing from the volume. If True, the smallest
        distance between two consecutive slices is found and returned as the
        slice spacing, provided all other spacings are an integer multiple of
        this value (within tolerance). Alternatively, if a SpacingBetweenSlices
        value is found in the datasets, that value will be used instead of the
        minimum consecutive spacing. If False, any gaps will result in failure.
    allow_duplicate_positions: bool, optional
        Allow multiple slices to occupy the same position within the volume. If
        False, duplicated image positions will result in failure.
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to determine how to order frames. Should be a sequence
        of two :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.

        This is used in combination with the ``handedness`` to determine
        the positive direction used to order frames.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Choose the frame order such that the frame axis creates a
        coordinate system with this handedness in the when combined with
        the within-frame convention given by ``index_convention``.
    enforce_handedness: bool, optional
        If True and sort is False, require that the images are not only
        regularly spaced but also that they are ordered correctly to give a
        coordinate system with the specified handedness, i.e. frames are
        ordered along the direction of the increasing normal vector, as opposed
        to being ordered regularly along the direction of the decreasing normal
        vector. If sort is True, this has no effect since positions will be
        sorted in the correct direction before finding the spacing.

    Returns
    -------
    Union[float, None]:
        If the image positions are regularly spaced, the (absolute value of)
        the slice spacing. If the image positions do not represent a
        regularly-spaced volume, returns None.
    Union[List[int], None]:
        List with the same length as the number of image positions. Each
        element gives the zero-based index of the corresponding input position
        in the volume. If the image positions do not represent a volume,
        returns None.

    """  # noqa: E501
    if len(datasets) == 0:
        raise ValueError("List must not be empty.")
    # We stipluate that a single image does represent a volume with spacing 0.0
    if len(datasets) == 1:
        return 1.0, [0]
    for ds in datasets:
        if is_multiframe_image(ds):
            raise ValueError(
                "Datasets should be single-frame images."
            )

    # Check image orientations are consistent
    image_orientation = datasets[0].ImageOrientationPatient
    for ds in datasets[1:]:
        if ds.ImageOrientationPatient != image_orientation:
            return None, None

    positions = [ds.ImagePositionPatient for ds in datasets]

    spacing_hint = datasets[0].get('SpacingBetweenSlices')

    return get_volume_positions(
        image_positions=positions,
        image_orientation=image_orientation,
        rtol=rtol,
        atol=atol,
        sort=sort,
        allow_duplicate_positions=allow_duplicate_positions,
        allow_missing_positions=allow_missing_positions,
        spacing_hint=spacing_hint,
        index_convention=index_convention,
        handedness=handedness,
        enforce_handedness=enforce_handedness,
    )


def get_volume_positions(
    image_positions: Sequence[Sequence[float]],
    image_orientation: Sequence[float],
    *,
    rtol: float | None = None,
    atol: float | None = None,
    sort: bool = True,
    allow_missing_positions: bool = False,
    allow_duplicate_positions: bool = False,
    spacing_hint: float | None = None,
    index_convention: (
        str |
        Sequence[PixelIndexDirections | str]
    ) = VOLUME_INDEX_CONVENTION,
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
    enforce_handedness: bool = False,
) -> tuple[float | None, list[int] | None]:
    """Get the spacing and positions of images within a 3D volume.

    First determines whether the image positions and orientation represent a 3D
    volume. A 3D volume consists of regularly spaced slices with orthogonal
    axes, i.e. the slices are spaced equally along the direction orthogonal to
    the two axes of the image plane.

    If the positions represent a volume, returns the absolute value of the
    slice spacing and the volume indices for each of the input positions. If
    the positions do not represent a volume, returns None for both outputs.

    Note that we stipulate that a single plane is a 3D volume for the purposes
    of this function. In this case, and if ``spacing_hint`` is not provided,
    the returned slice spacing will be 1.0.

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
    rtol: float, optional
        Relative tolerance for determining spacing regularity. If slice
        spacings vary by less that this proportion of the average spacing, they
        are considered to be regular. If neither ``rtol`` or ``atol`` are
        provided, a default relative tolerance of 0.01 is used.
    atol: float, optional
        Absolute tolerance for determining spacing regularity. If slice
        spacings vary by less that this value (in mm), they
        are considered to be regular. Incompatible with ``rtol``.
    sort: bool, optional
        Sort the image positions before finding the spacing. If True, this
        makes the function tolerant of unsorted inputs. Set to False to check
        whether the positions represent a 3D volume in the specific order in
        which they are passed.
    allow_missing_positions: bool, optional
        Allow for slices missing from the volume. If True, the smallest
        distance between two consecutive slices is found and returned as the
        slice spacing, provided all other spacings are an integer multiple of
        this value (within tolerance). Alternatively, if ``spacing_hint`` is
        used, that value will be used instead of the minimum consecutive
        spacing. If False, any gaps will result in failure.
    allow_duplicate_positions: bool, optional
        Allow multiple slices to occupy the same position within the volume.
        If False, duplicated image positions will result in failure.
    spacing_hint: Union[float, None], optional
        Expected spacing between slices. If the calculated value is not equal
        to this, within tolerance, the outputs will be None. The primary use of
        this option is in combination with ``allow_missing``. If
        ``allow_missing`` is ``True`` and a ``spacing_hint`` is given, the hint
        is used to calculate the index positions instead of the smallest
        consecutive spacing.
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to determine how to order frames. Should be a sequence
        of two :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving, in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.

        This is used in combination with the ``handedness`` to determine
        the positive direction used to order frames.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Choose the frame order in order such that the frame axis creates a
        coordinate system with this handedness when combined with the
        within-frame convention given by ``index_convention``.
    enforce_handedness: bool, optional
        If True and sort is False, require that the images are not only
        regularly spaced but also that they are ordered correctly to give a
        coordinate system with the specified handedness, i.e. frames are
        ordered along the direction of the increasing normal vector, as opposed
        to being ordered regularly along the direction of the decreasing normal
        vector. If sort is True, this has no effect since positions will be
        sorted in the correct direction before finding the spacing.

    Returns
    -------
    Union[float, None]:
        If the image positions are regularly spaced, the (absolute value of)
        the slice spacing. If the image positions do not represent a
        regularly-spaced volume, returns None.
    Union[List[int], None]:
        List with the same length as the number of image positions. Each
        element gives the zero-based index of the corresponding input position
        in the volume. If the image positions do not represent a volume,
        returns None.

    """  # noqa: E501
    if not sort:
        if allow_duplicate_positions:
            raise ValueError(
                "Argument 'allow_duplicates' requires 'sort'."
            )
        if allow_missing_positions:
            raise ValueError(
                "Argument 'allow_missing' requires 'sort'."
            )

    if spacing_hint is not None:
        if spacing_hint < 0.0:
            # There are some edge cases of the standard where this is valid
            spacing_hint = abs(spacing_hint)
        if spacing_hint == 0.0:
            raise ValueError("Argument 'spacing_hint' cannot be 0.")

    if atol is not None and rtol is not None:
        raise TypeError(
            "Arguments 'rtol' and 'atol' may not be provided together."
        )
    elif atol is not None:
        rtol = 0.0
    elif rtol is not None:
        atol = 0.0
    else:
        # Default situation. Just use rtol
        rtol = _DEFAULT_SPACING_RELATIVE_TOLERANCE
        atol = 0.0

    image_positions_arr = np.array(image_positions)

    if image_positions_arr.ndim != 2 or image_positions_arr.shape[1] != 3:
        raise ValueError(
            "Argument 'image_positions' should be an (N, 3) array."
        )
    n = image_positions_arr.shape[0]
    if n == 0:
        raise ValueError(
            "Argument 'image_positions' should contain at least 1 position."
        )
    elif n == 1:
        # Special case, we stipulate that this has spacing 1.0
        # if not otherwise specified
        spacing = 1.0 if spacing_hint is None else spacing_hint
        return spacing, [0]

    normal_vector = get_normal_vector(
        image_orientation,
        index_convention=index_convention,
        handedness=handedness,
    )

    # Unique index specifies, for each position in the input positions
    # array, the position in the unique_positions array of the
    # de-duplicated position
    unique_positions, unique_index = np.unique(
        image_positions_arr,
        axis=0,
        return_inverse=True,
    )
    if not allow_duplicate_positions:
        if unique_positions.shape[0] < image_positions_arr.shape[0]:
            return None, None

    if len(unique_positions) == 1:
        # Special case, we stipulate that this has spacing 1.0
        # if not otherwise specified
        spacing = 1.0 if spacing_hint is None else spacing_hint
        return spacing, [0] * n

    # Calculate distance of each slice from coordinate system origin along the
    # normal vector
    origin_distances = _get_slice_distances(unique_positions, normal_vector)

    if sort:
        # sort_index index gives, for each position in the sorted unique
        # positions, the initial index of the corresponding unique position
        sort_index = np.argsort(origin_distances)
        origin_distances_sorted = origin_distances[sort_index]
        inverse_sort_index = np.argsort(sort_index)
    else:
        sort_index = np.arange(unique_positions.shape[0])
        origin_distances_sorted = origin_distances
        inverse_sort_index = sort_index

    if allow_missing_positions:
        if spacing_hint is not None:
            spacing = spacing_hint
        else:
            spacings = np.diff(origin_distances_sorted)
            spacing = spacings.min()
            # Check here to prevent divide by zero errors. Positions should
            # have been de-duplicated already, if this is allowed, so there
            # should only be zero spacings if some positions are related by
            # in-plane translations
            if np.isclose(spacing, 0.0, atol=_DEFAULT_EQUALITY_TOLERANCE):
                return None, None

        origin_distance_multiples = (
            (origin_distances - origin_distances.min()) / spacing
        )

        is_regular = np.allclose(
            origin_distance_multiples,
            origin_distance_multiples.round(),
            rtol=rtol,
            atol=atol,
        )

        inverse_sort_index = origin_distance_multiples.round().astype(np.int64)

    else:
        spacings = np.diff(origin_distances_sorted)

        spacing = (
            (origin_distances_sorted[-1] - origin_distances_sorted[0]) /
            (len(origin_distances_sorted) - 1)
        )

        if spacing_hint is not None:
            if not np.isclose(
                abs(spacing),
                spacing_hint,
                rtol=rtol,
                atol=atol,
            ):
                raise RuntimeError(
                    f"Inferred spacing ({abs(spacing):.3f}) does not match the "
                    f"given 'spacing_hint' ({spacing_hint})."
                )

        is_regular = np.isclose(
            spacings,
            spacing,
            rtol=rtol,
            atol=atol,
        ).all()

    if is_regular and enforce_handedness:
        if spacing < 0.0:
            return None, None

    # Additionally check that the vector from the first to the last plane lies
    # approximately along the normal vector
    pos1 = unique_positions[sort_index[0], :]
    pos2 = unique_positions[sort_index[-1], :]
    span = (pos2 - pos1)
    span /= np.linalg.norm(span)

    dot_product = normal_vector.T @ span
    is_perpendicular = (
        abs(dot_product - 1.0) < _DOT_PRODUCT_PERPENDICULAR_TOLERANCE or
        abs(dot_product + 1.0) < _DOT_PRODUCT_PERPENDICULAR_TOLERANCE
    )

    if is_regular and is_perpendicular:
        vol_positions = [
            inverse_sort_index[unique_index[i]].item()
            for i in range(len(image_positions_arr))
        ]
        return abs(spacing), vol_positions
    else:
        return None, None


def get_plane_sort_index(
    image_positions: Sequence[Sequence[float]],
    image_orientation: Sequence[float],
    index_convention: (
        str |
        Sequence[PixelIndexDirections | str]
    ) = VOLUME_INDEX_CONVENTION,
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
) -> list[int]:
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
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to determine how to order frames. Should be a sequence
        of two :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.

        This is used in combination with the ``handedness`` to determine
        the positive direction used to order frames.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Choose the frame order in order such that the frame axis creates a
        coordinate system with this handedness in the when combined with
        the within-frame convention given by ``index_convention``.

    Returns
    -------
    List[int]
        Sorting index for the input planes. Element i of this list gives the
        index in the original list of the frames such that the output list
        is sorted along the positive direction of the normal vector of the
        imaging plane.

    """  # noqa: E501
    pos_arr = np.array(image_positions)
    if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
        raise ValueError("Argument 'image_positions' must have shape (N, 3)")
    ori_arr = np.array(image_orientation)
    if ori_arr.ndim != 1 or ori_arr.shape[0] != 6:
        raise ValueError("Argument 'image_orientation' must have shape (6, )")

    normal_vector = get_normal_vector(
        ori_arr,
        index_convention=index_convention,
        handedness=handedness,
    )

    # Calculate distance of each slice from coordinate system origin along the
    # normal vector
    origin_distances = _get_slice_distances(pos_arr, normal_vector)

    sort_index = np.argsort(origin_distances)

    return sort_index.tolist()


def get_dataset_sort_index(
    datasets: Sequence[Dataset],
    index_convention: (
        str |
        Sequence[PixelIndexDirections | str]
    ) = VOLUME_INDEX_CONVENTION,
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
) -> list[int]:
    """Get index to sort single frame datasets spatially.

    Parameters
    ----------
    datasets: Sequence[pydicom.Dataset]
        Datasets containing single frame images, with a consistent orientation.
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to determine how to order frames. Should be a sequence
        of two :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.

        This is used in combination with the ``handedness`` to determine
        the positive direction used to order frames.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Choose the frame order in order such that the frame axis creates a
        coordinate system with this handedness in the when combined with
        the within-frame convention given by ``index_convention``.

    Returns
    -------
    List[int]
        Sorting index for the input datasets. Element i of this list gives the
        index in the original list of datasets such that the output list is
        sorted along the positive direction of the normal vector of the imaging
        plane.

    """  # noqa: E501
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
    return get_plane_sort_index(
        positions,
        image_orientation,
        index_convention=index_convention,
        handedness=handedness,
    )


def sort_datasets(
    datasets: Sequence[Dataset],
    index_convention: (
        str |
        Sequence[PixelIndexDirections | str]
    ) = VOLUME_INDEX_CONVENTION,
    handedness: AxisHandedness | str = AxisHandedness.RIGHT_HANDED,
) -> list[Dataset]:
    """Sort single frame datasets spatially.

    Parameters
    ----------
    datasets: Sequence[pydicom.Dataset]
        Datasets containing single frame images, with a consistent orientation.
    index_convention: Sequence[Union[highdicom.enum.PixelIndexDirections, str]], optional
        Convention used to determine how to order frames. Should be a sequence
        of two :class:`highdicom.enum.PixelIndexDirections` or their string
        representations, giving in order, the indexing conventions used for
        specifying pixel indices. For example ``('R', 'D')`` means that the
        first pixel index indexes the columns from left to right, and the
        second pixel index indexes the rows from top to bottom (this is the
        convention typically used within DICOM). As another example ``('D',
        'R')`` would switch the order of the indices to give the convention
        typically used within NumPy.

        Alternatively, a single shorthand string may be passed that combines
        the string representations of the two directions. So for example,
        passing ``'RD'`` is equivalent to passing ``('R', 'D')``.

        This is used in combination with the ``handedness`` to determine
        the positive direction used to order frames.
    handedness: Union[highdicom.enum.AxisHandedness, str], optional
        Choose the frame order in order such that the frame axis creates a
        coordinate system with this handedness in the when combined with
        the within-frame convention given by ``index_convention``.

    Returns
    -------
    List[pydicom.Dataset]
        List of datasets sorted according to spatial position, using the
        convention specified by the input parameters.

    """  # noqa: E501
    sort_index = get_dataset_sort_index(
        datasets,
        index_convention=index_convention,
        handedness=handedness,
    )
    return [datasets[i] for i in sort_index]


def _get_slice_distances(
    image_positions: np.ndarray,
    normal_vector: np.ndarray,
) -> np.ndarray:
    """Get distances of a set of planes from the origin.

    For each plane position, find (signed) distance from origin along the
    vector normal to the imaging plane.

    Parameters
    ----------
    image_positions: np.ndarray
        Image positions array. 2D array of shape (N, 3) where N is the number
        of planes and each row gives the (x, y, z) image position of a plane.
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
