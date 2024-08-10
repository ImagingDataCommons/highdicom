from abc import ABC, abstractmethod
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import List, Optional, Sequence, Union, Tuple, cast

import numpy as np

from highdicom._module_utils import is_multiframe_image
from highdicom.color import ColorManager
from highdicom.enum import (
    AxisHandedness,
    CoordinateSystemNames,
    PadModes,
    PatientOrientationValuesBiped,
)
from highdicom.spatial import (
    _create_affine_transformation_matrix,
    _is_matrix_orthogonal,
    _normalize_patient_orientation,
    _stack_affine_matrix,
    _transform_affine_matrix,
    _translate_affine_matrix,
    PATIENT_ORIENTATION_OPPOSITES,
    VOLUME_INDEX_CONVENTION,
    get_closest_patient_orientation,
    get_image_coordinate_system,
    get_plane_sort_index,
    get_volume_positions,
    get_series_volume_positions,
    sort_datasets,
)
from highdicom.content import (
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
)

from pydicom import Dataset, dcmread
from pydicom.pixel_data_handlers.util import (
    apply_modality_lut,
    apply_color_lut,
    apply_voi_lut,
    convert_color_space,
)


# TODO add basic arithmetric operations
# TODO add pixel value transformations
# TODO should methods copy arrays?
# TODO random crop, random flip, random permute
# TODO match geometry
# TODO trim non-zero
# TODO support slide coordinate system
# TODO volume to volume transformer
# TODO volread and metadata
# TODO make RIGHT handed the default
# TODO constructors for geometry, do they make sense for volume?
# TODO ordering of frames in seg, applying 3D
# TODO use VolumeGeometry within MultiFrameDB manager


class _VolumeBase(ABC):

    """Base class for object exhibiting volume geometry."""

    def __init__(
        self,
        affine: np.ndarray,
        frame_of_reference_uid: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        affine: numpy.ndarray
            4 x 4 affine matrix representing the transformation from pixel
            indices (slice index, row index, column index) to the
            frame-of-reference coordinate system. The top left 3 x 3 matrix
            should be a scaled orthogonal matrix representing the rotation and
            scaling. The top right 3 x 1 vector represents the translation
            component. The last row should have value [0, 0, 0, 1].
        frame_of_reference_uid: Optional[str], optional
            Frame of reference UID for the frame of reference, if known.

        """
        if affine.shape != (4, 4):
            raise ValueError("Affine matrix must have shape (4, 4).")
        if not np.array_equal(affine[-1, :], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError(
                "Final row of affine matrix must be [0.0, 0.0, 0.0, 1.0]."
            )
        if not _is_matrix_orthogonal(affine[:3, :3], require_unit=False):
            raise ValueError(
                "Argument 'affine' must be an orthogonal matrix."
            )

        self._affine = affine
        self._frame_of_reference_uid = frame_of_reference_uid


    @property
    @abstractmethod
    def spatial_shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Spatial shape of the array.

        Does not include the channel dimension.

        """
        pass

    def get_center_index(self, round_output: bool = False) -> np.ndarray:
        """Get array index of center of the volume.

        Parameters
        ----------
        round_output: bool, optional
            If True, the result is returned rounded down to and with an integer
            datatype. Otherwise it is returned as a floating point datatype
            without rounding, to sub-voxel precision.

        Returns
        -------
        numpy.ndarray:
            Array of shape 3 representing the array indices at the center of
            the volume.

        """
        if round_output:
            center = np.array(
                [(self.spatial_shape[d] - 1) // 2 for d in range(3)],
                dtype=np.uint32,
            )
        else:
            center = np.array(
                [(self.spatial_shape[d] - 1) / 2.0 for d in range(3)]
            )

        return center

    def get_center_coordinate(self) -> np.ndarray:
        """Get frame-of-reference coordinate at the center of the volume.

        Returns
        -------
        numpy.ndarray:
            Array of shape 3 representing the frame-of-reference coordinate at
            the center of the volume.

        """
        center_index = self.get_center_index().reshape((1, 3))
        center_coordinate = self.map_indices_to_reference(center_index)

        return center_coordinate.reshape((3, ))

    def map_indices_to_reference(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Transform image pixel indices to frame of reference coordinates.

        Parameters
        ----------
        indices: numpy.ndarray
            Array of zero-based array indices. Array of integer values with
            shape ``(n, 3)``, where *n* is the number of indices, the first
            column represents the `column` index and the second column
            represents the `row` index.

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

        """
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError(
                'Argument "indices" must be a two-dimensional array '
                'with shape [n, 3].'
            )
        indices_augmented = np.row_stack([
            indices.T.astype(float),
            np.ones((indices.shape[0], ), dtype=float),
        ])
        reference_coordinates = np.dot(self._affine, indices_augmented)
        return reference_coordinates[:3, :].T

    def map_reference_to_indices(
        self,
        coordinates: np.ndarray,
        round_output: bool = False,
        check_bounds: bool = False,
    ) -> np.ndarray:
        """Transform frame of reference coordinates into array indices.

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
            Array of zero-based array indices at pixel resolution. Array of
            integer or floating point values with shape ``(n, 3)``, where *n*
            is the number of indices. The datatype of the array will be integer
            if ``round_output`` is True (the default), or float if
            ``round_output`` is False.
        round_output: bool, optional
            Whether to round the output to the nearest voxel. If True, the
            output will have integer datatype. If False, the returned array
            will have floating point data type and sub-voxel precision.
        check_bounds: bool, optional
            Whether to check that the returned indices lie within the bounds of
            the array. If True, a ``RuntimeError`` will be raised if the
            resulting array indices (before rounding) lie out of the bounds of
            the array.

        Note
        ----
        The returned pixel indices may be negative if `coordinates` fall
        outside of the array.

        Raises
        ------
        ValueError
            When `indices` has incorrect shape.
        RuntimeError
            If `check_bounds` is True and any map coordinate lies outside the
            bounds of the array.

        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                'Argument "coordinates" must be a two-dimensional array '
                'with shape [n, 3].'
            )
        reference_coordinates = np.row_stack([
            coordinates.T.astype(float),
            np.ones((coordinates.shape[0], ), dtype=float)
        ])
        indices = np.dot(self.inverse_affine, reference_coordinates)
        indices = indices[:3, :].T

        if check_bounds:
            out_of_bounds = False
            for d in range(3):
                if indices[:, d].min() < -0.5:
                    out_of_bounds = True
                    break
                if indices[:, d].max() > self.spatial_shape[d] - 0.5:
                    out_of_bounds = True
                    break

            if out_of_bounds:
                raise RuntimeError("Bounds check failed.")

        if round_output:
            return np.around(indices).astype(int)
        else:
            return indices

    def get_plane_position(self, plane_number: int) -> PlanePositionSequence:
        """Get plane position of a given plane.

        Parameters
        ----------
        plane_number: int
            Zero-based plane index (down the first dimension of the array).

        Returns
        -------
        highdicom.content.PlanePositionSequence:
            Plane position of the plane.

        """
        if plane_number < 0 or plane_number >= self.spatial_shape[0]:
            raise ValueError("Invalid plane number for volume.")
        index = np.array([[plane_number, 0, 0]])
        position = self.map_indices_to_reference(index)[0]

        return PlanePositionSequence(
            CoordinateSystemNames.PATIENT,
            position,
        )

    def get_plane_positions(self) -> List[PlanePositionSequence]:
        """Get plane positions of all planes in the volume.

        This assumes that the volume is encoded in a DICOM file with frames
        down axis 0, rows stacked down axis 1, and columns stacked down axis 2.

        Returns
        -------
        List[highdicom.content.PlanePositionSequence]:
            Plane position of the all planes (stacked down axis 0 of the
            volume).

        """
        indices = np.array(
            [
                [p, 0, 0] for p in range(self.spatial_shape[0])
            ]
        )
        positions = self.map_indices_to_reference(indices)

        return [
            PlanePositionSequence(
                CoordinateSystemNames.PATIENT,
                pos,
            )
            for pos in positions
        ]

    def get_plane_orientation(self) -> PlaneOrientationSequence:
        """Get plane orientation sequence for the volume.

        This assumes that the volume is encoded in a DICOM file with frames
        down axis 0, rows stacked down axis 1, and columns stacked down axis 2.

        Returns
        -------
        highdicom.PlaneOrientationSequence:
            Plane orientation sequence 

        """
        return PlaneOrientationSequence(
            CoordinateSystemNames.PATIENT,
            self.direction_cosines,
        )

    def get_pixel_measures(self) -> PixelMeasuresSequence:
        """Get pixel measures sequence for the volume.

        This assumes that the volume is encoded in a DICOM file with frames
        down axis 0, rows stacked down axis 1, and columns stacked down axis 2.

        Returns
        -------
        highdicom.PixelMeasuresSequence:
            Pixel measures sequence for the volume.

        """
        return PixelMeasuresSequence(
            pixel_spacing=self.pixel_spacing,
            slice_thickness=None,
            spacing_between_slices=self.spacing_between_slices,
        )

    @property
    def frame_of_reference_uid(self) -> Optional[str]:
        """Union[str, None]: Frame of reference UID."""
        return self._frame_of_reference_uid

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix

        This matrix maps an index into the array into a position in the LPS
        frame of reference coordinate space.

        """
        return self._affine.copy()

    @property
    def inverse_affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 inverse affine transformation matrix

        Inverse of the affine matrix. This matrix maps a position in the LPS
        frame of reference coordinate space into an index into the array.

        """
        return np.linalg.inv(self._affine)

    @property
    def direction_cosines(self) -> List[float]:
        """List[float]:

        List of 6 floats giving the direction cosines of the
        vector along the rows and the vector along the columns, matching the
        format of the DICOM Image Orientation Patient attribute.

        """
        vec_along_rows = self._affine[:3, 2].copy()
        vec_along_columns = self._affine[:3, 1].copy()
        vec_along_columns /= np.sqrt((vec_along_columns ** 2).sum())
        vec_along_rows /= np.sqrt((vec_along_rows ** 2).sum())
        return [*vec_along_rows.tolist(), *vec_along_columns.tolist()]

    @property
    def pixel_spacing(self) -> List[float]:
        """List[float]:

        Within-plane pixel spacing in millimeter units. Two
        values (spacing between rows, spacing between columns).

        """
        vec_along_rows = self._affine[:3, 2]
        vec_along_columns = self._affine[:3, 1]
        spacing_between_columns = np.sqrt((vec_along_rows ** 2).sum()).item()
        spacing_between_rows = np.sqrt((vec_along_columns ** 2).sum()).item()
        return [spacing_between_rows, spacing_between_columns]

    @property
    def spacing_between_slices(self) -> float:
        """float:

        Spacing between consecutive slices in millimeter units.

        """
        slice_vec = self._affine[:3, 0]
        spacing = np.sqrt((slice_vec ** 2).sum()).item()
        return spacing

    @property
    def spacing(self) -> List[float]:
        """List[float]:

        Pixel spacing in millimeter units for the three spatial directions.
        Three values, one for each spatial dimension.

        """
        dir_mat = self._affine[:3, :3]
        norms = np.sqrt((dir_mat ** 2).sum(axis=0))
        return norms.tolist()

    @property
    def voxel_volume(self) -> float:
        """float: The volume of a single voxel in cubic millimeters."""
        return np.prod(self.spacing).item()

    @property
    def position(self) -> List[float]:
        """List[float]:

        Position in the frame of reference space of the center of voxel at
        indices (0, 0, 0).

        """
        return self._affine[:3, 3].tolist()

    @property
    def physical_extent(self) -> List[float]:
        """List[float]: Side lengths of the volume in millimeters."""
        return [n * d for n, d in zip(self.spatial_shape, self.spacing)]

    @property
    def physical_volume(self) -> float:
        """float: Total volume in cubic millimeter."""
        return self.voxel_volume * np.prod(self.spatial_shape).item()

    @property
    def direction(self) -> np.ndarray:
        """numpy.ndarray:

        Direction matrix for the volume. The columns of the direction
        matrix are orthogonal unit vectors that give the direction in the
        frame of reference space of the increasing direction of each axis
        of the array. This matrix may be passed either as a 3x3 matrix or a
        flattened 9 element array (first row, second row, third row).

        """
        dir_mat = self._affine[:3, :3]
        norms = np.sqrt((dir_mat ** 2).sum(axis=0))
        return dir_mat / norms

    def spacing_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the vectors along the three array dimensions.

        Note that these vectors are not normalized, they have length equal to
        the spacing along the relevant dimension.

        Returns
        -------
        numpy.ndarray:
            Vector between voxel centers along the increasing first axis.
            1D NumPy array.
        numpy.ndarray:
            Vector between voxel centers along the increasing second axis.
            1D NumPy array.
        numpy.ndarray:
            Vector between voxel centers along the increasing third axis.
            1D NumPy array.

        """
        return tuple(self.affine[:3, :3].T)

    def unit_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the normalized vectors along the three array dimensions.

        Returns
        -------
        numpy.ndarray:
            Unit vector along the increasing first axis. 1D NumPy array.
        numpy.ndarray:
            Unit vector along the increasing second axis. 1D NumPy array.
        numpy.ndarray:
            Unit vector along the increasing third axis. 1D NumPy array.

        """
        return tuple(self.direction.T)

    @abstractmethod
    def __getitem__(
        self,
        index: Union[int, slice, Tuple[Union[int, slice]]],
    ) -> '_VolumeBase':
        pass

    def _prepare_getitem_index(
        self,
        index: Union[int, slice, Tuple[Union[int, slice]]],
    ) -> Tuple[Tuple[slice], Tuple[int, int, int], np.ndarray]:

        def _check_int(val: int, dim: int) -> None:
            if (
                val < -self.spatial_shape[dim]
                or val >= self.spatial_shape[dim]
            ):
                raise IndexError(
                    f'Index {val} is out of bounds for axis {dim} with size '
                    f'{self.spatial_shape[dim]}.'
                )

        def _check_slice(val: slice, dim: int) -> None:
            if (
                val.start is not None and
                (
                    val.start < -self.spatial_shape[dim] or
                    val.start >= self.spatial_shape[dim]
                )
            ):
                raise ValueError(
                    f'val {val.start} is out of bounds for axis {dim} with '
                    f'size {self.spatial_shape[dim]}.'
                )
            if (
                val.stop is not None and
                (
                    val.stop < -self.spatial_shape[dim] - 1 or
                    val.stop > self.spatial_shape[dim]
                )
            ):
                raise ValueError(
                    f'val {val.stop} is out of bounds for axis {dim} with '
                    f'size {self.spatial_shape[dim]}.'
                )

        if isinstance(index, int):
            # Change the index to a slice of length one so that all dimensions
            # are retained in the output array. Also make into a tuple of
            # length 1 to standardize format
            _check_int(index, 0)
            tuple_index = (slice(index, index + 1), )
        elif isinstance(index, slice):
            # Make into a tuple of length one to standardize the format
            _check_slice(index, 0)
            tuple_index = (cast(slice, index), )
        elif isinstance(index, tuple):
            index_list: List[slice] = []
            for dim, item in enumerate(index):
                if isinstance(item, int):
                    # Change the index to a slice of length one so that all
                    # dimensions are retained in the output array.
                    _check_int(item, dim)
                    item = slice(item, item + 1)
                    index_list.append(item)
                elif isinstance(item, slice):
                    _check_slice(item, dim)
                    index_list.append(item)
                else:
                    raise TypeError(
                        'Items within "index" must be ints, or slices. Got '
                        f'{type(item)}.'
                    )

            tuple_index = tuple(index_list)

        else:
            raise TypeError(
                'Argument "index" must be an int, slice or tuple. Got '
                f'{type(index)}.'
            )

        new_vectors = []
        origin_indices = []
        new_shape = []
        for d in range(0, 3):
            # The index item along this dimension
            if len(tuple_index) > d:
                index_item = tuple_index[d]
                first, last, step = index_item.indices(self.spatial_shape[d])
                index_range = last - first
                if index_range == 0 or ((index_range < 0) != (step < 0)):
                    raise IndexError('Indexing would result in an empty array.')
                size = (abs(index_range) - 1) // abs(step) + 1
                new_shape.append(size)
            else:
                index_item = None
                first = 0
                step = 1
                new_shape.append(self.spatial_shape[d])

            new_vectors.append(self._affine[:3, d] * step)
            origin_indices.append(first)

        origin_index_arr = np.array([origin_indices])
        new_origin_arr = self.map_indices_to_reference(origin_index_arr).T

        new_rotation = np.column_stack(new_vectors)
        new_affine = _stack_affine_matrix(new_rotation, new_origin_arr)

        return tuple_index, tuple(new_shape), new_affine

    @abstractmethod
    def pad(
        self,
        pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]],
        mode: Union[PadModes, str] = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> '_VolumeBase':
        pass

    def _prepare_pad_width(
        self,
        pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]],
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Pad volume along the three spatial dimensions.

        Parameters
        ----------
        pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]]
            Values to pad the array. Takes the same form as ``numpy.pad()``.
            May be:

            * A single integer value, which results in that many voxels being
              added to the beginning and end of all three spatial dimensions,
              or
            * A sequence of two values in the form ``[before, after]``, which
              results in 'before' voxels being added to the beginning of each
              of the three spatial dimensions, and 'after' voxels being added
              to the end of each of the three spatial dimensions, or
            * A nested sequence of integers of the form ``[[pad1], [pad2],
              [pad3]]``, in which separate padding values are supplied for each
              of the three spatial axes and used to pad before and after along
              those axes, or
            * A nested sequence of integers in the form ``[[before1, after1],
              [before2, after2], [before3, after3]]``, in which separate values
              are supplied for the before and after padding of each of the
              three spatial dimensions.

            In all cases, all integer values must be non-negative.

        Returns
        -------
        numpy.ndarray:
            Affine matrix of the padded array.
        List[List[int]]:
            Padding specification along three spatial dimensions in format
            ``[[before1, after1], [before2, after2], [before3, after3]]``.

        """
        if isinstance(pad_width, int):
            if pad_width < 0:
                raise ValueError(
                    f"Argument 'pad_width' cannot contain negative values."
                )
            origin_offset = [-pad_width] * 3
            full_pad_width: List[List[int]] = [[pad_width, pad_width]] * 3
        elif isinstance(pad_width, Sequence):
            if isinstance(pad_width[0], int):
                if len(pad_width) != 2:
                    raise ValueError("Invalid arrangement in 'pad_width'.")
                if pad_width[0] < 0 or pad_width[1] < 0:
                    raise ValueError(
                        f"Argument 'pad_width' cannot contain negative values."
                    )
                full_pad_width = [list(pad_width)] * 3
            elif isinstance(pad_width[0], Sequence):
                if len(pad_width) != 3:
                    raise ValueError("Invalid arrangement in 'pad_width'.")
                if len(pad_width[0]) == 1:
                    if len(pad_width[1]) != 1 or len(pad_width[2]) != 1:
                        raise ValueError("Invalid arrangement in 'pad_width'.")
                    full_pad_width = [[w[0], w[0]] for w in pad_width]
                elif len(pad_width[0]) == 2:
                    if len(pad_width[1]) != 2 or len(pad_width[2]) != 2:
                        raise ValueError("Invalid arrangement in 'pad_width'.")
                    full_pad_width = [list(w) for w in pad_width]
                else:
                    raise ValueError("Invalid arrangement in 'pad_width'.")
        else:
            raise TypeError("Invalid format for 'pad_width'.")

        origin_offset = [-p[0] for p in full_pad_width]
        new_affine = _translate_affine_matrix(self.affine, origin_offset)

        return new_affine, full_pad_width


    def _permute_affine(self, indices: Sequence[int]) -> np.ndarray:
        """Get affine after permuting spatial axes.

        Parameters
        ----------
        indices: Sequence[int]
            List of three integers containing the values 0, 1 and 2 in some
            order. Note that you may not change the position of the channel
            axis (if present).

        Returns
        -------
        numpy.numpy:
            Affine matrix (4 x 4) spatial axes permuted in the provided order.

        """
        if len(indices) != 3 or set(indices) != {0, 1, 2}:
            raise ValueError(
                f'Argument "indices" must consist of the values 0, 1, and 2 '
                'in some order.'
            )

        return _transform_affine_matrix(
            affine=self._affine,
            shape=self.spatial_shape,
            permute_indices=indices,
        )

    @abstractmethod
    def permute_axes(self, indices: Sequence[int]) -> '_VolumeBase':
        """Create a new volume by permuting the spatial axes.

        Parameters
        ----------
        indices: Sequence[int]
            List of three integers containing the values 0, 1 and 2 in some
            order. Note that you may not change the position of the channel
            axis (if present).

        Returns
        -------
        highdicom._VolumeBase:
            New volume with spatial axes permuted in the provided order.

        """
        pass

    def get_closest_patient_orientation(self) -> Tuple[
        PatientOrientationValuesBiped,
        PatientOrientationValuesBiped,
        PatientOrientationValuesBiped,
    ]:
        """Get patient orientation codes that best represent the affine.

        Returns
        -------
        Tuple[highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped]:
            Tuple giving the closest patient orientation.

        """  # noqa: E501
        return get_closest_patient_orientation(self._affine)

    def to_patient_orientation(
        self,
        patient_orientation: Union[
            str,
            Sequence[Union[str, PatientOrientationValuesBiped]],
        ],
    ) -> '_VolumeBase':
        """Rearrange the array to a given orientation.

        The resulting volume is formed from this volume through a combination
        of axis permutations and flips of the spatial axes. Its patient
        orientation will be as close to the desired orientation as can be
        achieved with these operations alone (and in particular without
        resampling the array).

        Parameters
        ----------
        patient_orientation: Union[str, Sequence[Union[str, highdicom.enum.PatientOrientationValuesBiped]]]
            Desired patient orientation, as either a sequence of three
            highdicom.enum.PatientOrientationValuesBiped values, or a string
            such as ``"FPL"`` using the same characters.

        Returns
        -------
        highdicom.Volume:
            New volume with the requested patient orientation.

        """  # noqa: E501
        desired_orientation = _normalize_patient_orientation(
            patient_orientation
        )

        current_orientation = self.get_closest_patient_orientation()

        permute_indices = []
        flip_axes = []
        for d in desired_orientation:
            if d in current_orientation:
                from_index = current_orientation.index(d)
            else:
                d_inv = PATIENT_ORIENTATION_OPPOSITES[d]
                from_index = current_orientation.index(d_inv)
                flip_axes.append(from_index)
            permute_indices.append(from_index)

        if len(flip_axes) > 0:
            result = self.flip(flip_axes)
        else:
            result = self

        return result.permute_axes(permute_indices)

    def swap_axes(self, axis_1: int, axis_2: int) -> '_VolumeBase':
        """Swap the spatial axes of the array.

        Parameters
        ----------
        axis_1: int
            Spatial axis index (0, 1 or 2) to swap with ``axis_2``.
        axis_2: int
            Spatial axis index (0, 1 or 2) to swap with ``axis_1``.

        Returns
        -------
        highdicom.Volume:
            New volume with spatial axes swapped as requested.

        """
        for a in [axis_1, axis_2]:
            if a not in {0, 1, 2}:
                raise ValueError(
                    'Axis values must be one of 0, 1 or 2.'
                )

        if axis_1 == axis_2:
            raise ValueError(
                "Arguments 'axis_1' and 'axis_2' must be different."
            )

        permutation = [0, 1, 2]
        permutation[axis_1] = axis_2
        permutation[axis_2] = axis_1

        return self.permute_axes(permutation)

    def flip(self, axis: Union[int, Sequence[int]]) -> '_VolumeBase':
        """Flip the spatial axes of the array.

        Note that this flips the array and updates the affine to reflect the
        flip.

        Parameters
        ----------
        axis: Union[int, Sequence[int]]
            Axis or list of axis indices that should be flipped. These should
            include only the spatial axes (0, 1, and/or 2).

        Returns
        -------
        highdicom.Volume:
            New volume with spatial axes flipped as requested.

        """
        if isinstance(axis, int):
            axis = [axis]

        if len(axis) > 3 or len(set(axis) - {0, 1, 2}) > 0:
            raise ValueError(
                'Argument "axis" must contain only values 0, 1, and/or 2.'
            )

        # We will re-use the existing __getitem__ implementation, which has all
        # this logic figured out already
        index = []
        for d in range(3):
            if d in axis:
                index.append(slice(-1, None, -1))
            else:
                index.append(slice(None))

        return self[tuple(index)]

    @property
    def handedness(self) -> AxisHandedness:
        """highdicom.AxisHandedness: Axis handedness of the volume."""
        v1, v2, v3 = self.spacing_vectors()
        if np.cross(v1, v2) @ v3 < 0.0:
            return AxisHandedness.LEFT_HANDED
        return AxisHandedness.RIGHT_HANDED

    def ensure_handedness(
        self,
        handedness: Union[AxisHandedness, str],
        flip_axis: Optional[int] = None,
        swap_axes: Optional[Sequence[int]] = None,
    ) -> '_VolumeBase':
        """Manipulate the volume if necessary to ensure a given handedness.

        If the volume already has the specified handedness, it is returned
        unaltered.

        If the volume does not meet the requirement, the volume is manipulated
        using a user specified operation to meet the requirement. The two
        options are reversing the direction of a single axis ("flipping") or
        swapping the position of two axes.

        Parameters
        ----------
        handedness: highdicom.AxisHandedness
            Handedness to ensure.
        flip_axis: Union[int, None], optional
            Specification of a spatial axis index (0, 1, or 2) to flip if
            required to meet the given handedness requirement.
        swap_axes: Union[int, None], optional
            Specification of a sequence of two spatial axis indices (each being
            0, 1, or 2) to swap if required to meet the given handedness
            requirement.

        Note
        ----
        Either ``flip_axis`` or ``swap_axes`` must be provided (and not both)
        to specify the operation to perform to correct the handedness (if
        required).

        """
        if (flip_axis is None) == (swap_axes is None):
            raise TypeError(
                "Exactly one of either 'flip_axis' or 'swap_axes' "
                "must be specified."
            )
        handedness = AxisHandedness(handedness)
        if handedness == self.handedness:
            return self

        if flip_axis is not None:
            return self.flip(flip_axis)

        if len(swap_axes) != 2:
            raise ValueError(
                "Argument 'swap_axes' must have length 2."
            )

        return self.swap_axes(swap_axes[0], swap_axes[1])

    def pad_to_shape(
        self,
        spatial_shape: Sequence[int],
        mode: PadModes = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> '_VolumeBase':
        """Pad volume to given spatial shape.

        The volume is padded symmetrically, placing the original array at the
        center of the output array, to achieve the given shape. If this
        requires an odd number of elements to be added along a certain
        dimension, one more element is placed at the end of the array than at
        the start.

        Parameters
        ----------
        spatial_shape: Sequence[int]
            Sequence of three integers specifying the spatial shape to pad to.
            This shape must be no smaller than the existing shape along any of
            the three spatial dimensions.
        mode: highdicom.PadModes, optional
            Mode to use to pad the array. See :class:`highdicom.PadModes` for
            options.
        constant_value: Union[float, Sequence[float]], optional
            Value used to pad when mode is ``"CONSTANT"``. If ``per_channel``
            if True, a sequence whose length is equal to the number of channels
            may be passed, and each value will be used for the corresponding
            channel. With other pad modes, this argument is ignored.
        per_channel: bool, optional
            For padding modes that involve calculation of image statistics to
            determine the padding value (i.e. ``MINIMUM``, ``MAXIMUM``,
            ``MEAN``, ``MEDIAN``), pad each channel separately using the value
            calculated using that channel alone (rather than the statistics of
            the entire array). For other padding modes, this argument makes no
            difference. This should not the True if the image does not have a
            channel dimension.

        Returns
        -------
        highdicom.Volume:
            Volume with padding applied.

        """
        if len(spatial_shape) != 3:
            raise ValueError(
                "Argument 'shape' must have length 3."
            )

        pad_width = []
        for insize, outsize in zip(self.spatial_shape, spatial_shape):
            to_pad = outsize - insize
            if to_pad < 0:
                raise ValueError(
                    'Shape is smaller than existing shape along at least '
                    'one axis.'
                )
            pad_front = to_pad // 2
            pad_back = to_pad - pad_front
            pad_width.append((pad_front, pad_back))

        return self.pad(
            pad_width=pad_width,
            mode=mode,
            constant_value=constant_value,
            per_channel=per_channel,
        )

    def crop_to_shape(self, spatial_shape: Sequence[int]) -> '_VolumeBase':
        """Center-crop volume to a given spatial shape.

        Parameters
        ----------
        spatial_shape: Sequence[int]
            Sequence of three integers specifying the spatial shape to crop to.
            This shape must be no larger than the existing shape along any of
            the three spatial dimensions.

        Returns
        -------
        highdicom.Volume:
            Volume with padding applied.

        """
        if len(spatial_shape) != 3:
            raise ValueError(
                "Argument 'shape' must have length 3."
            )

        crop_vals = []
        for insize, outsize in zip(self.spatial_shape, spatial_shape):
            to_crop = insize - outsize
            if to_crop < 0:
                raise ValueError(
                    'Shape is larger than existing shape along at least '
                    'one axis.'
                )
            crop_front = to_crop // 2
            crop_back = to_crop - crop_front
            crop_vals.append((crop_front, insize - crop_back))

        return self[
            crop_vals[0][0]:crop_vals[0][1],
            crop_vals[1][0]:crop_vals[1][1],
            crop_vals[2][0]:crop_vals[2][1],
        ]

    def pad_or_crop_to_shape(
        self,
        spatial_shape: Sequence[int],
        mode: PadModes = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> '_VolumeBase':
        """Pad and/or crop volume to given spatial shape.

        For each dimension where padding is required, the volume is padded
        symmetrically, placing the original array at the center of the output
        array, to achieve the given shape. If this requires an odd number of
        elements to be added along a certain dimension, one more element is
        placed at the end of the array than at the start.

        For each dimension where cropping is required, center cropping is used.

        Parameters
        ----------
        spatial_shape: Sequence[int]
            Sequence of three integers specifying the spatial shape to pad or
            crop to.
        mode: highdicom.PadModes, optional
            Mode to use to pad the array, if padding is required. See
            :class:`highdicom.PadModes` for options.
        constant_value: Union[float, Sequence[float]], optional
            Value used to pad when mode is ``"CONSTANT"``. If ``per_channel``
            if True, a sequence whose length is equal to the number of channels
            may be passed, and each value will be used for the corresponding
            channel. With other pad modes, this argument is ignored.
        per_channel: bool, optional
            For padding modes that involve calculation of image statistics to
            determine the padding value (i.e. ``MINIMUM``, ``MAXIMUM``,
            ``MEAN``, ``MEDIAN``), pad each channel separately using the value
            calculated using that channel alone (rather than the statistics of
            the entire array). For other padding modes, this argument makes no
            difference. This should not the True if the image does not have a
            channel dimension.

        Returns
        -------
        highdicom.Volume:
            Volume with padding and/or cropping applied.

        """
        if len(spatial_shape) != 3:
            raise ValueError(
                "Argument 'shape' must have length 3."
            )

        pad_width = []
        crop_vals = []
        for insize, outsize in zip(self.spatial_shape, spatial_shape):
            diff = outsize - insize
            if diff > 0:
                pad_front = diff // 2
                pad_back = diff - pad_front
                pad_width.append((pad_front, pad_back))
                crop_vals.append((0, outsize))
            elif diff < 0:
                crop_front = (-diff) // 2
                crop_back = (-diff) - crop_front
                crop_vals.append((crop_front, insize - crop_back))
                pad_width.append((0, 0))
            else:
                pad_width.append((0, 0))
                crop_vals.append((0, outsize))

        cropped = self[
            crop_vals[0][0]:crop_vals[0][1],
            crop_vals[1][0]:crop_vals[1][1],
            crop_vals[2][0]:crop_vals[2][1],
        ]
        padded = cropped.pad(
            pad_width=pad_width,
            mode=mode,
            constant_value=constant_value,
            per_channel=per_channel,
        )
        return padded

    def random_crop(self, spatial_shape: Sequence[int]) -> '_VolumeBase':
        """Create a random crop of a certain shape from the volume.

        Parameters
        ----------
        spatial_shape: Sequence[int]
            Sequence of three integers specifying the spatial shape to pad or
            crop to.

        Returns
        -------
        highdicom.Volume:
            New volume formed by cropping the volumes.

        """
        crop_slices = []
        for c, d in zip(spatial_shape, self.spatial_shape):
            max_start = d - c
            if max_start < 0:
                raise ValueError(
                    'Crop shape is larger than volume in at least one '
                    'dimension.'
                )
            start = np.random.randint(0, max_start + 1)
            crop_slices.append(slice(start, start + c))

        return self[tuple(crop_slices)]


class VolumeGeometry(_VolumeBase):

    def __init__(
        self,
        affine: np.ndarray,
        spatial_shape: Sequence[int],
        frame_of_reference_uid: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        affine: numpy.ndarray
            4 x 4 affine matrix representing the transformation from pixel
            indices (slice index, row index, column index) to the
            frame-of-reference coordinate system. The top left 3 x 3 matrix
            should be a scaled orthogonal matrix representing the rotation and
            scaling. The top right 3 x 1 vector represents the translation
            component. The last row should have value [0, 0, 0, 1].
        spatial_shape: Sequence[int]
            Number of voxels in the volume along the three spatial dimensions.
        frame_of_reference_uid: Optional[str], optional
            Frame of reference UID for the frame of reference, if known.

        """
        super().__init__(affine, frame_of_reference_uid)

        if len(spatial_shape) != 3:
            raise ValueError("Argument 'spatial_shape' must have length 3.")
        self._spatial_shape = tuple(spatial_shape)

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Spatial shape of the array.

        Does not include the channel dimension.

        """
        return self._spatial_shape

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple[int, ...]: Shape of the underlying array.

        For objects of type :class:`highdicom.VolumeGeometry`, this is
        equivalent to `.shape`.

        """
        return self.spatial_shape

    def __getitem__(
        self,
        index: Union[int, slice, Tuple[Union[int, slice]]],
    ) -> "VolumeGeometry":
        """Get a sub-volume of this volume as a new volume.

        Parameters
        ----------
        index: Union[int, slice, Tuple[Union[int, slice]]]
            Index values. Most possibilities supported by numpy arrays are
            supported, including negative indices and different step sizes.
            Indexing with lists is not supported.

        Returns
        -------
        highdicom.VolumeGeometry:
            New volume representing a sub-volume of the original volume.

        """
        _, new_shape, new_affine = self._prepare_getitem_index(index)
        self._spatial_shape = new_shape

        return self.__class__(
            affine=new_affine,
            spatial_shape=new_shape,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def pad(
        self,
        pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]],
        mode: Union[PadModes, str] = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> 'VolumeGeometry':
        """Pad volume along the three spatial dimensions.

        Parameters
        ----------
        pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]]
            Values to pad the array. Takes the same form as ``numpy.pad()``.
            May be:

            * A single integer value, which results in that many voxels being
              added to the beginning and end of all three spatial dimensions,
              or
            * A sequence of two values in the form ``[before, after]``, which
              results in 'before' voxels being added to the beginning of each
              of the three spatial dimensions, and 'after' voxels being added
              to the end of each of the three spatial dimensions, or
            * A nested sequence of integers of the form ``[[pad1], [pad2],
              [pad3]]``, in which separate padding values are supplied for each
              of the three spatial axes and used to pad before and after along
              those axes, or
            * A nested sequence of integers in the form ``[[before1, after1],
              [before2, after2], [before3, after3]]``, in which separate values
              are supplied for the before and after padding of each of the
              three spatial dimensions.

            In all cases, all integer values must be non-negative.
        mode: Union[highdicom.PadModes, str], optional
            Ignored for :class:`highdicom.VolumeGeometry`.
        constant_value: Union[float, Sequence[float]], optional
            Ignored for :class:`highdicom.VolumeGeometry`.
        per_channel: bool, optional
            Ignored for :class:`highdicom.VolumeGeometry`.

        Returns
        -------
        highdicom.VolumeGeometry:
            Volume with padding applied.

        """
        new_affine, full_pad_width = self._prepare_pad_width(pad_width)

        new_shape = [
            d + p[0] + p[1] for d, p in zip(self.spatial_shape, full_pad_width)
        ]

        return self.__class__(
            spatial_shape=new_shape,
            affine=new_affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def permute_axes(self, indices: Sequence[int]) -> 'VolumeGeometry':
        """Create a new geometry by permuting the spatial axes.

        Parameters
        ----------
        indices: Sequence[int]
            List of three integers containing the values 0, 1 and 2 in some
            order. Note that you may not change the position of the channel
            axis (if present).

        Returns
        -------
        highdicom.VolumeGeometry:
            New geometry with spatial axes permuted in the provided order.

        """
        new_affine = self._permute_affine(indices)

        new_shape = [self.spatial_shape[i] for i in indices]

        return self.__class__(
            spatial_shape=new_shape,
            affine=new_affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )


class Volume(_VolumeBase):

    """Class representing a 3D array of regularly-spaced frames in 3D space.

    This class combines a 3D NumPy array with an affine matrix describing the
    location of the voxels in the frame of reference coordinate space. A
    Volume is not a DICOM object itself, but represents a volume that may
    be extracted from DICOM image, and/or encoded within a DICOM object,
    potentially following any number of processing steps.

    All such volumes have a geometry that exists within DICOM's patient
    coordinate system.

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
        array: np.ndarray,
        affine: np.ndarray,
        frame_of_reference_uid: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        array: numpy.ndarray
            Array of voxel data. Must be either 3D (three spatial dimensions),
            or 4D (three spatial dimensions followed by a channel dimension).
            Any datatype is permitted.
        affine: numpy.ndarray
            4 x 4 affine matrix representing the transformation from pixel
            indices (slice index, row index, column index) to the
            frame-of-reference coordinate system. The top left 3 x 3 matrix
            should be a scaled orthogonal matrix representing the rotation and
            scaling. The top right 3 x 1 vector represents the translation
            component. The last row should have value [0, 0, 0, 1].
        frame_of_reference_uid: Optional[str], optional
            Frame of reference UID for the frame of reference, if known.

        """
        super().__init__(
            affine=affine,
            frame_of_reference_uid=frame_of_reference_uid,
        )
        if array.ndim not in (3, 4):
            raise ValueError(
                "Argument 'array' must be three or four-dimensional."
            )
        self._array = array

    @classmethod
    def from_image_series(
        cls,
        series_datasets: Sequence[Dataset],
        apply_modality_transform: bool = True,
        apply_voi_transform: bool = False,
        voi_transform_index: int = 0,
        apply_palette_color_lut: bool = True,
        apply_icc_transform: bool = True,
        standardize_color_space: bool = True,
    ) -> "Volume":
        """Create volume from a series of single frame images.

        Parameters
        ----------
        series_datasets: Sequence[pydicom.Dataset]
            Series of single frame datasets. There is no requirement on the
            sorting of the datasets.
        apply_modality_transform: bool, optional
            Whether to apply the modality transform (either a rescale intercept
            and slope or modality LUT) to the pixel values, if present in the
            datasets.
        apply_voi_transform: bool, optional
            Whether to apply the value of interest (VOI) transform (either a
            windowing operation or VOI LUT) to the pixel values, if present in
            the datasets.
        voi_transform_index: int, optional
            Index of the VOI transform to apply if multiple are included in the
            datasets. Ignored if ``apply_voi_transform`` is ``False`` or no VOI
            transform is included in the datasets.
        apply_palette_color_lut: bool, optional
            Whether to apply the palette color LUT if a dataset has photometric
            interpretation ``'PALETTE_COLOR'``.
        apply_icc_transform: bool, optional
            Whether to apply an ICC color profile, if present in the datasets.
        convert_color_space: bool, optional
            Whether to convert the color space to a standardized space. If
            True, images with photometric interpretation ``MONOCHROME1`` are
            inverted to mimic ``MONOCHROME2``, and images with photometric
            interpretation ``YBR_FULL`` or ``YBR_FULL_422`` are converted to
            ``RGB``.

        Returns
        -------
        Volume:
            Volume created from the series.

        """
        if apply_voi_transform and not apply_modality_lut:
            raise ValueError(
                "Argument 'apply_voi_transform' requires 'apply_modality_lut'."
            )
        series_instance_uid = series_datasets[0].SeriesInstanceUID
        if not all(
            ds.SeriesInstanceUID == series_instance_uid
            for ds in series_datasets
        ):
            raise ValueError('Images do not belong to the same series.')

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

        ds = series_datasets[0]

        if len(series_datasets) == 1:
            slice_spacing = ds.get('SpacingBetweenSlices', 1.0)
        else:
            slice_spacing, _ = get_series_volume_positions(series_datasets)
            if slice_spacing is None:
                raise ValueError('Series is not a regularly-spaced volume.')

        affine = _create_affine_transformation_matrix(
            image_position=ds.ImagePositionPatient,
            image_orientation=ds.ImageOrientationPatient,
            pixel_spacing=ds.PixelSpacing,
            spacing_between_slices=slice_spacing,
            index_convention=VOLUME_INDEX_CONVENTION,
            slices_first=True,
        )

        frames = []
        for ds in series_datasets:
            frame = ds.pixel_array
            max_value = 2 ** np.iinfo(ds.pixel_array.dtype).bits
            if apply_modality_transform:
                frame = apply_modality_lut(frame, ds)
            if apply_voi_transform:
                frame = apply_voi_lut(frame, ds, voi_transform_index)
            if (
                apply_palette_color_lut and 
                ds.PhotometricInterpretation == 'PALETTE_COLOR'
            ):
                frame = apply_color_lut(frame, ds)
            if apply_icc_transform and 'ICCProfile' in ds:
                manager = ColorManager(ds.ICCProfile)
                frame = manager.transform_frame(frame)
            if standardize_color_space:
                if ds.PhotometricInterpretation == 'MONOCHROME1':
                    # TODO what if a VOI_LUT has been applied
                    frame = max_value - frame
                elif ds.PhotometricInterpretation in (
                    'YBR_FULL', 'YBR_FULL_422'
                ):
                    frame = convert_color_space(
                        frame,
                        current=ds.PhotometricInterpretation,
                        desired='RGB'
                    )

            frames.append(frame)

        array = np.stack(frames)

        return cls(
            affine=affine,
            array=array,
            frame_of_reference_uid=frame_of_reference_uid,
        )

    @classmethod
    def from_image(
        cls,
        dataset: Dataset,
    ) -> "Volume":
        """Create volume from a multiframe image.

        Parameters
        ----------
        dataset: pydicom.Dataset
            A multi-frame image dataset.

        Returns
        -------
        Volume:
            Volume created from the image.

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

        if 'PixelMeasuresSequence' not in sfgs:
            raise ValueError('Frames do not share pixel measures.')
        pixel_spacing = sfgs.PixelMeasuresSequence[0].PixelSpacing

        slice_spacing, _ = get_volume_positions(
            image_positions=image_positions,
            image_orientation=image_orientation,
        )
        if slice_spacing is None:
            raise ValueError(
                'Dataset does not represent a regularly sampled volume.'
            )

        affine = _create_affine_transformation_matrix(
            image_position=sorted_positions[0],
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=slice_spacing,
            index_convention=VOLUME_INDEX_CONVENTION,
            slices_first=True,
        )

        # TODO apply VOI color modality LUT etc
        array = dataset.pixel_array
        if array.ndim == 2:
            array = array[np.newaxis]
        array = array[sort_index]

        return cls(
            affine=affine,
            array=array,
            frame_of_reference_uid=dataset.FrameOfReferenceUID,
        )

    @classmethod
    def from_attributes(
        cls,
        array: np.ndarray,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float],
        spacing_between_slices: float,
        frame_of_reference_uid: Optional[str] = None,
    ) -> "Volume":
        """Create a volume from DICOM attributes.

        Parameters
        ----------
        array: numpy.ndarray
            Three dimensional array of voxel data. The first dimension indexes
            slices, the second dimension indexes rows, and the final dimension
            indexes columns.
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
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID, if known. Corresponds to DICOM attribute
            FrameOfReferenceUID.

        Returns
        -------
        highdicom.Volume:
            New Volume using the given array and DICOM attributes.

        """
        affine = _create_affine_transformation_matrix(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=spacing_between_slices,
            index_convention=VOLUME_INDEX_CONVENTION,
            slices_first=True,
        )
        return cls(
            affine=affine,
            array=array,
            frame_of_reference_uid=frame_of_reference_uid,
        )

    @classmethod
    def from_components(
        cls,
        array: np.ndarray,
        position: Sequence[float],
        direction: Sequence[float],
        spacing: Sequence[float],
        frame_of_reference_uid: Optional[str] = None,
    ) -> "Volume":
        """Construct a Volume from components.

        Parameters
        ----------
        array: numpy.ndarray
            Three dimensional array of voxel data.
        position: Sequence[float]
            Sequence of three floats giving the position in the frame of
            reference coordinate system of the center of the pixel at location
            (0, 0, 0).
        direction: Sequence[float]
            Direction matrix for the volume. The columns of the direction
            matrix are orthogonal unit vectors that give the direction in the
            frame of reference space of the increasing direction of each axis
            of the array. This matrix may be passed either as a 3x3 matrix or a
            flattened 9 element array (first row, second row, third row).
        spacing: Sequence[float]
            Spacing between pixel centers in the the frame of reference
            coordinate system along each of the dimensions of the array.
        shape: Sequence[int]
            Sequence of three integers giving the shape of the volume.
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID for the frame of reference, if known.

        Returns
        -------
        highdicom.spatial.Volume:
            Volume constructed from the provided components.

        """
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
        if not _is_matrix_orthogonal(direction_arr, require_unit=True):
            raise ValueError(
                "Argument 'direction' must be an orthogonal matrix of "
                "unit vectors."
            )

        scaled_direction = direction_arr * spacing
        affine = _stack_affine_matrix(scaled_direction, np.array(position))
        return cls(
            array=array,
            affine=affine,
            frame_of_reference_uid=frame_of_reference_uid,
        )

    def get_geometry(self) -> VolumeGeometry:
        """Get geometry for this volume.

        Returns
        -------
        hd.VolumeGeometry:
            Geometry object matching this volume.

        """
        return VolumeGeometry(
            affine=self._affine.copy(),
            spatial_shape=self.spatial_shape,
            frame_of_reference_uid=self.frame_of_reference_uid
        )

    @property
    def dtype(self) -> type:
        """type: Datatype of the array."""
        return self._array.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple[int, ...]: Shape of the underlying array.

        May or may not include a fourth channel dimension.

        """
        return tuple(self._array.shape)

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: Spatial shape of the array.

        Does not include the channel dimension.

        """
        return tuple(self._array.shape[:3])

    @property
    def number_of_channels(self) -> Optional[int]:
        """Optional[int]: Number of channels.

        If the array has no channel dimension, returns None.

        """
        if self._array.ndim == 4:
            return self._array.shape[3]
        return None

    @property
    def array(self) -> np.ndarray:
        """numpy.ndarray: Volume array."""
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        """Change the voxel array without changing the affine.

        Parameters
        ----------
        array: np.ndarray
            New 3D or 4D array of voxel data. The spatial shape must match the
            existing array, but the presence and number of channels and/or the
            voxel datatype may differ.

        """
        if value.ndim not in (3, 4):
            raise ValueError(
                "Argument 'array' must be a three or four dimensional array."
            )
        if value.shape[:3] != self.spatial_shape:
            raise ValueError(
                "Array must match the spatial shape of the existing array."
            )
        self._array = value

    def astype(self, dtype: type) -> 'Volume':
        """Get new volume with a new datatype.

        Parameters
        ----------
        dtype: type
            A numpy datatype for the new volume.

        Returns
        -------
        highdicom.Volume:
            New volume with given datatype, and metadata copied from this
            volume.

        """
        new_array = self._array.astype(dtype)

        return self.with_array(new_array)

    def copy(self) -> 'Volume':
        """Get an unaltered copy of the volume.

        Returns
        -------
        highdicom.Volume:
            Copy of the original volume.

        """
        return self.__class__(
            array=self.array,  # TODO should this copy?
            affine=self._affine.copy(),
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def with_array(self, array: np.ndarray) -> 'Volume':
        """Get a new volume using a different array.

        The spatial and other metadata will be copied from this volume.
        The original volume will be unaltered.

        Parameters
        ----------
        array: np.ndarray
            New 3D or 4D array of voxel data. The spatial shape must match the
            existing array, but the presence and number of channels and/or the
            voxel datatype may differ.

        Returns
        -------
        highdicom.Volume:
            New volume using the given array and the metadata of this volume.

        """
        if array.ndim not in (3, 4):
            raise ValueError(
                "Argument 'array' must be a three or four dimensional array."
            )
        if array.shape[:3] != self.spatial_shape:
            raise ValueError(
                "Array must match the spatial shape of the existing array."
            )
        return self.__class__(
            array=array,
            affine=self._affine.copy(),
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def __getitem__(
        self,
        index: Union[int, slice, Tuple[Union[int, slice]]],
    ) -> "Volume":
        """Get a sub-volume of this volume as a new volume.

        Parameters
        ----------
        index: Union[int, slice, Tuple[Union[int, slice]]]
            Index values. Most possibilities supported by numpy arrays are
            supported, including negative indices and different step sizes.
            Indexing with lists is not supported.

        Returns
        -------
        highdicom.Volume:
            New volume representing a sub-volume of the original volume.

        """
        tuple_index, _, new_affine = self._prepare_getitem_index(index)

        new_array = self._array[tuple_index]

        return self.__class__(
            array=new_array,
            affine=new_affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def permute_axes(self, indices: Sequence[int]) -> 'Volume':
        """Create a new volume by permuting the spatial axes.

        Parameters
        ----------
        indices: Sequence[int]
            List of three integers containing the values 0, 1 and 2 in some
            order. Note that you may not change the position of the channel
            axis (if present).

        Returns
        -------
        highdicom.Volume:
            New volume with spatial axes permuted in the provided order.

        """
        new_affine = self._permute_affine(indices)

        if self._array.ndim == 3:
            new_array = np.transpose(self._array, indices)
        else:
            new_array = np.transpose(self._array, [*indices, 3])

        return self.__class__(
            array=new_array,
            affine=new_affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def normalize_mean_std(
        self,
        per_channel: bool = True,
        output_mean: float = 0.0,
        output_std: float = 1.0,
    ) -> 'Volume':
        """Normalize the intensities using the mean and variance.

        The resulting volume has zero mean and unit variance.

        Parameters
        ----------
        per_channel: bool, optional
            If True (the default), each channel is normalized by its own mean
            and variance. If False, all channels are normalized together using
            the overall mean and variance.
        output_mean: float, optional
            The mean value of the output array (or channel), after scaling.
        output_std: float, optional
            The standard deviation of the output array (or channel),
            after scaling.

        Returns
        -------
        highdicom.Volume:
            Volume with normalized intensities. Note that the dtype will
            be promoted to floating point.

        """
        if (
            per_channel and
            self.number_of_channels is not None and
            self.number_of_channels > 1
        ):
            new_array = self.array.astype(np.float64)
            for c in range(self.number_of_channels):
                channel = new_array[:, :, :, c]
                new_array[:, :, :, c] = (
                    (channel - channel.mean()) /
                    (channel.std() / output_std)
                ) + output_mean
        else:
            new_array = (
                (self.array - self.array.mean()) /
                (self.array.std() / output_std)
                + output_mean
            )

        return self.with_array(new_array)

    def normalize_min_max(
        self,
        output_min: float = 0.0,
        output_max: float = 1.0,
        per_channel: bool = False,
    ) -> 'Volume':
        """Normalize by mapping its full intensity range to a fixed range.

        Other pixel values are scaled linearly within this range.

        Parameters
        ----------
        output_min: float, optional
            The value to which the minimum intensity is mapped.
        output_max: float, optional
            The value to which the maximum intensity is mapped.
        per_channel: bool, optional
            If True, each channel is normalized by its own mean and variance.
            If False (the default), all channels are normalized together using
            the overall mean and variance.

        Returns
        -------
        highdicom.Volume:
            Volume with normalized intensities. Note that the dtype will
            be promoted to floating point.

        """
        output_range = output_max - output_min
        if output_range <= 0.0:
            raise ValueError('Output min must be below output max.')

        if (
            per_channel and
            self.number_of_channels is not None and
            self.number_of_channels > 1
        ):
            new_array = self.array.astype(np.float64)
            for c in range(self.number_of_channels):
                channel = new_array[:,:, :, c]
                imin = channel.min()
                imax = channel.max()
                scale_factor = output_range / (imax - imin)
                new_array[:, :, :, c] = (
                    (channel - imin) * scale_factor + output_min
                )
        else:
            imin = self.array.min()
            imax = self.array.max()
            scale_factor = output_range / (imax - imin)
            new_array = (self.array - imin) * scale_factor + output_min

        return self.with_array(new_array)

    def clip(
        self,
        a_min: Optional[float],
        a_max: Optional[float],
    ) -> 'Volume':
        """Clip voxel intensities to lie within a given range.

        Parameters
        ----------
        a_min: Union[float, None]
            Lower value to clip. May be None if no lower clipping is to be
            applied. Voxel intensities below this value are set to this value.
        a_max: Union[float, None]
            Upper value to clip. May be None if no upper clipping is to be
            applied. Voxel intensities above this value are set to this value.

        Returns
        -------
        highdicom.Volume:
            Volume with clipped intensities.

        """
        new_array = np.clip(self.array, a_min, a_max)

        return self.with_array(new_array)

    def apply_window(
        self,
        *,
        window_min: Optional[float] = None,
        window_max: Optional[float]= None,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        output_min: float = 0.0,
        output_max: float = 1.0,
        clip: bool = True,
    ) -> 'Volume':
        """Apply a window (similar to VOI transform) to the volume.

        Parameters
        ----------
        window_min: Union[float, None], optional
            Minimum value of window (mapped to ``output_min``).
        window_max: Union[float, None], optional
            Maximum value of window (mapped to ``output_max``).
        window_center: Union[float, None], optional
            Center value of the window.
        window_width: Union[float, None], optional
            Width of the window.
        output_min: float, optional
            Value to which the lower edge of the window is mapped.
        output_max: float, optional
            Value to which the upper edge of the window is mapped.
        clip: bool, optional
            Whether to clip the values to lie within the output range.

        Note
        ----
        Either ``window_min`` and ``window_max`` or ``window_center`` and
        ``window_width`` should be specified. Other combinations are not valid.

        Returns
        -------
        highdicom.Volume:
            Volume with windowed intensities.

        """
        if window_min is None != window_max is None:
            raise TypeError("Invalid combination of inputs specified.")
        if window_center is None != window_width is None:
            raise TypeError("Invalid combination of inputs specified.")
        if window_center is None == window_min is None:
            raise TypeError("Invalid combination of inputs specified.")

        if window_min is None:
            window_min = window_center - (window_width / 2)
        if window_width is None:
            window_width = window_max - window_min
        output_range = output_max - output_min
        scale_factor = output_range / window_width

        new_array = (self.array - window_min) * scale_factor + output_min

        if clip:
            new_array = np.clip(new_array, output_min, output_max)

        return self.with_array(new_array)

    def squeeze_channel(self) -> 'Volume':
        """Remove a singleton channel axis.

        If the volume has no channels, returns an unaltered copy.

        Returns
        -------
        highdicom.Volume:
            Volume with channel axis removed.

        """
        if self.number_of_channels is None:
            return self.copy()
        if self.number_of_channels == 1:
            return self.with_array(self.array.squeeze(3))
        else:
            raise RuntimeError(
                'Volume with multiple channels cannot be squeezed.'
            )

    def ensure_channel(self) -> 'Volume':
        """Add a singleton channel axis, if needed.

        If the volume has channels already, returns an unaltered copy.

        Returns
        -------
        highdicom.Volume:
            Volume with added channel axis (if required).

        """
        if self.number_of_channels is None:
            return self.with_array(self.array[:, :, :, None])
        return self.copy()

    def pad(
        self,
        pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]],
        mode: Union[PadModes, str] = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> 'Volume':
        """Pad volume along the three spatial dimensions.

        Parameters
        ----------
        pad_width: Union[int, Sequence[int], Sequence[Sequence[int]]]
            Values to pad the array. Takes the same form as ``numpy.pad()``.
            May be:

            * A single integer value, which results in that many voxels being
              added to the beginning and end of all three spatial dimensions,
              or
            * A sequence of two values in the form ``[before, after]``, which
              results in 'before' voxels being added to the beginning of each
              of the three spatial dimensions, and 'after' voxels being added
              to the end of each of the three spatial dimensions, or
            * A nested sequence of integers of the form ``[[pad1], [pad2],
              [pad3]]``, in which separate padding values are supplied for each
              of the three spatial axes and used to pad before and after along
              those axes, or
            * A nested sequence of integers in the form ``[[before1, after1],
              [before2, after2], [before3, after3]]``, in which separate values
              are supplied for the before and after padding of each of the
              three spatial dimensions.

            In all cases, all integer values must be non-negative.
        mode: Union[highdicom.PadModes, str], optional
            Mode to use to pad the array. See :class:`highdicom.PadModes` for
            options.
        constant_value: Union[float, Sequence[float]], optional
            Value used to pad when mode is ``"CONSTANT"``. If ``per_channel``
            if True, a sequence whose length is equal to the number of channels
            may be passed, and each value will be used for the corresponding
            channel. With other pad modes, this argument is ignored.
        per_channel: bool, optional
            For padding modes that involve calculation of image statistics to
            determine the padding value (i.e. ``MINIMUM``, ``MAXIMUM``,
            ``MEAN``, ``MEDIAN``), pad each channel separately using the value
            calculated using that channel alone (rather than the statistics of
            the entire array). For other padding modes, this argument makes no
            difference. This should not the True if the image does not have a
            channel dimension.

        Returns
        -------
        highdicom.Volume:
            Volume with padding applied.

        """
        if isinstance(mode, str):
            mode = mode.upper()
        mode = PadModes(mode)

        if per_channel and self.number_of_channels is None:
            raise ValueError(
                "Argument 'per_channel' may not be True if the image has no "
                "channels."
            )

        if mode in (
            PadModes.MINIMUM,
            PadModes.MAXIMUM,
            PadModes.MEAN,
            PadModes.MEDIAN,
        ):
            used_mode = PadModes.CONSTANT
        elif (
            mode == PadModes.CONSTANT and
            isinstance(constant_value, Sequence)
        ):
            used_mode = mode
            if not per_channel:
                raise TypeError(
                    "Argument 'constant_value' should be a single value if "
                    "'per_channel' is False."
                )
            if len(constant_value) != self.number_of_channels:
                raise ValueError(
                    "Argument 'constant_value' must have length equal to the "
                    'number of channels in the volume.'
                )
        else:
            used_mode = mode
            # per_channel result is same as default result, so just ignore it
            per_channel = False

        if (
            self.number_of_channels is None or
            self.number_of_channels == 1
        ):
            # Only one channel, so can ignore the per_channel logic
            per_channel = False

        new_affine, full_pad_width = self._prepare_pad_width(pad_width)

        if self.number_of_channels is not None and not per_channel:
            full_pad_width.append([0, 0])  # no padding for channel dim

        def pad_array(array: np.ndarray, cval: float) -> float:
            if used_mode == PadModes.CONSTANT:
                if mode == PadModes.MINIMUM:
                    v = array.min()
                elif mode == PadModes.MAXIMUM:
                    v = array.max()
                elif mode == PadModes.MEAN:
                    v = array.mean()
                elif mode == PadModes.MEDIAN:
                    v = np.median(array)
                elif mode == PadModes.CONSTANT:
                    v = cval
                pad_kwargs = {'constant_values': v}
            else:
                pad_kwargs = {}

            return np.pad(
                array,
                pad_width=full_pad_width,
                mode=used_mode.value.lower(),
                **pad_kwargs,
            )

        if per_channel:
            if not isinstance(constant_value, Sequence):
                constant_value = [constant_value] * self.number_of_channels
            padded_channels = []
            for c, v in enumerate(constant_value):
                padded_channels.append(pad_array(self.array[:, :, :, c], v))
            new_array = np.stack(padded_channels, axis=-1)
        else:
            new_array = pad_array(self.array, constant_value)

        return self.__class__(
            array=new_array,
            affine=new_affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )


def concat_channels(volumes: Sequence[Volume]) -> Volume:
    """Form a new volume by concatenating channels of existing volumes.

    Parameters
    ----------
    volumes: Sequence[highdicom.Volume]
        Sequence of one or more volumes to concatenate. Volumes must
        share the same spatial shape and affine matrix, but may differ
        by number and presence of channels.

    Returns
    -------
    highdicom.Volume:
        New volume formed by concatenating the input volumes.

    """
    if len(volumes) < 1:
        raise ValueError("Argument 'volumes' should not be empty.")
    spatial_shape = volumes[0].spatial_shape
    affine = volumes[0].affine.copy()
    frame_of_reference_uids = [
        v.frame_of_reference_uid for v in volumes
        if v.frame_of_reference_uid is not None
    ]
    if len(set(frame_of_reference_uids)) > 1:
        raise ValueError(
            "Volumes have differing frame of reference UIDs."
        )
    if len(frame_of_reference_uids) > 0:
        frame_of_reference_uid = frame_of_reference_uids[0]
    else:
        frame_of_reference_uid = None
    if not all(v.spatial_shape == spatial_shape for v in volumes):
        raise ValueError(
            "All items in 'volumes' should have the same spatial "
            "shape."
        )
    if not all(np.allclose(v.affine, affine) for v in volumes):
        raise ValueError(
            "All items in 'volumes' should have the same affine "
            "matrix."
        )

    arrays = []
    for v in volumes:
        array = v.array
        if array.ndim == 3:
            array = array[:, :, :, None]

        arrays.append(array)

    concat_array = np.concatenate(arrays, axis=3)
    return Volume(
        array=concat_array,
        affine=affine,
        frame_of_reference_uid=frame_of_reference_uid,
    )


def volread(
    fp: Union[str, bytes, PathLike, List[Union[str, PathLike]]],
    glob: str = '*.dcm',
) -> Volume:
    """Read a volume from a file or list of files or file-like objects.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object, directory, list of file-like objects representing
        a DICOM file or set of files.
    glob: str, optional
        Glob pattern used to find files within the direcotry in the case that
        ``fp`` is a string or path that represents a directory. Follows the
        format of the standard library glob ``module``.

    Returns
    -------
    highdicom.Volume
        Volume formed from the specified image file(s).

    """
    if isinstance(fp, (str, PathLike)):
        fp = Path(fp)
    if isinstance(fp, Path) and fp.is_dir():
        fp = list(fp.glob(glob))

    if isinstance(fp, Sequence):
        dcms = [dcmread(f) for f in fp]
    else:
        dcms = [dcmread(fp)]

    if len(dcms) == 1 and is_multiframe_image(dcms[0]):
        return Volume.from_image(dcms[0])

    return Volume.from_image_series(dcms)
