"""Representations of multidimensional arrays with spatial metadata."""
from abc import ABC, abstractmethod
from enum import Enum
import itertools
from typing import cast, Union
from collections.abc import Sequence
from pydicom.tag import BaseTag
from typing_extensions import Self

import numpy as np

from highdicom.enum import (
    AxisHandedness,
    CoordinateSystemNames,
    PadModes,
    PatientOrientationValuesBiped,
    RGBColorChannels,
)
from highdicom.spatial import (
    PATIENT_ORIENTATION_OPPOSITES,
    VOLUME_INDEX_CONVENTION,
    _DEFAULT_EQUALITY_TOLERANCE,
    _is_matrix_orthogonal,
    _normalize_patient_orientation,
    _stack_affine_matrix,
    _transform_affine_matrix,
    _translate_affine_matrix,
    _transform_affine_to_convention,
    create_affine_matrix_from_attributes,
    create_affine_matrix_from_components,
    get_closest_patient_orientation,
)
from highdicom.content import (
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
)
from highdicom.uid import UID

from pydicom.datadict import (
    get_entry,
    tag_for_keyword,
    keyword_for_tag,
)


_DCM_PYTHON_TYPE_MAP = {
    'CS': str,
    'DS': float,
    'FD': float,
    'FL': float,
    'IS': int,
    'LO': str,
    'LT': str,
    'PN': str,
    'SH': str,
    'SL': int,
    'SS': int,
    'ST': str,
    'UI': str,
    'UL': int,
    'UR': str,
    'US or SS': int,
    'US': int,
    'UT': str,
}


class ChannelDescriptor:

    """Descriptor of a channel (non-spatial) dimension within a Volume.

    A channel dimension may be described either using a standard DICOM
    attribute (preferable where possible) or a custom descriptor that defines
    the quantity or characteristic that varies along the dimension.

    """

    def __init__(
        self,
        identifier: str | int | Self,
        is_custom: bool = False,
        value_type: type | None = None,
    ):
        """

        Parameters
        ----------
        identifier: str | int | highdicom.ChannelDescriptor
            Identifier of the attribute. May be a DICOM attribute identified
            either by its keyword or integer tag value. Alternatively, if
            ``is_custom`` is True, an arbitrary string used to identify the
            dimension.
        is_custom: bool
            Whether the identifier is a custom identifier, as opposed to a
            DICOM attribute.
        value_type: type | None
            The python type of the values that vary along the dimension. Should
            be provided if and only if a custom identifier is used. Only ints,
            floats, strs, or enum.Enums, or their sub-classes, are allowed.

        """
        if isinstance(identifier, self.__class__):
            self._keyword = identifier.keyword
            self._tag = identifier.tag
            self._value_type = identifier.value_type
            return

        if is_custom:
            if not isinstance(identifier, str):
                raise TypeError(
                    'Custom identifiers must be specified via a string.'
                )
            if tag_for_keyword(identifier) is not None:
                raise ValueError(
                    f"The string '{identifier}' cannot be used for a "
                    "custom channel identifier because it is a DICOM "
                    "keyword."
                )
            self._keyword = identifier
            self._tag: BaseTag | None = None

            if value_type is None:
                raise TypeError(
                    "Argument 'value_type' must be specified when defining "
                    "a custom channel identifier."
                )

            if not issubclass(value_type, (str, int, float, Enum)):
                raise ValueError(
                    "Argument 'value_type' must be str, int, float, Enum "
                    "or a subclass of these."
                )

            if value_type is Enum:
                raise ValueError(
                    "When using Enums, argument 'value_type' must be a "
                    "specific subclass of Enum."
                )

            self._value_type = value_type
        else:
            if isinstance(identifier, int):  # also covers BaseTag
                self._tag = BaseTag(identifier)
                keyword = keyword_for_tag(identifier)
                if keyword is None:
                    self._keyword = str(self._tag)
                else:
                    self._keyword = keyword

            elif isinstance(identifier, str):
                t = tag_for_keyword(identifier)

                if t is None:
                    raise ValueError(
                        f'No attribute found with keyword {identifier}. '
                        'You may need to specify a custom identifier '
                        "using 'is_custom'."
                    )

                self._tag = BaseTag(t)
                self._keyword = identifier
            else:
                raise TypeError(
                    "Argument 'identifier' must be an int or str."
                )

            if value_type is not None:
                raise TypeError(
                    "Argument 'value_type' should only be specified when "
                    "defining a custom channel identifier."
                )

            vr, _, _, _, _ = get_entry(self._tag)
            self._value_type = _DCM_PYTHON_TYPE_MAP[vr]

    @property
    def value_type(self) -> type:
        """type: Python type of the quantity that varies along the
        dimension.

        """
        return self._value_type

    @property
    def keyword(self) -> str:
        """str: The DICOM keyword or custom string for the descriptor."""
        return self._keyword

    @property
    def tag(self) -> BaseTag | None:
        """str: The DICOM tag for the attribute.

        ``None`` for custom descriptors.

        """
        return self._tag

    @property
    def is_custom(self) -> bool:
        """bool: Whether the descriptor is custom, as opposed to using a DICOM
        attribute.

        """
        return self._tag is None

    @property
    def is_enumerated(self) -> bool:
        """bool: Whether the value type is enumerated.

        """
        return issubclass(self.value_type, Enum)

    def __hash__(self) -> int:
        return hash(self._keyword)

    def __str__(self):
        return self._keyword

    def __repr__(self):
        return self._keyword

    def __eq__(self, other):
        return self._keyword == other._keyword


RGB_COLOR_CHANNEL_DESCRIPTOR = ChannelDescriptor(
    'RGBColorChannel',
    value_type=RGBColorChannels,
    is_custom=True,
)
"""Descriptor used for an RGB color channel dimension."""


class _VolumeBase(ABC):

    """Base class for objects exhibiting volume geometry."""

    def __init__(
        self,
        affine: np.ndarray,
        coordinate_system: CoordinateSystemNames | str,
        frame_of_reference_uid: str | None = None,
    ):
        """

        Parameters
        ----------
        affine: numpy.ndarray
            4 x 4 affine matrix representing the transformation from voxel
            indices to the frame-of-reference coordinate system. The top left 3
            x 3 matrix should be a scaled orthogonal matrix representing the
            rotation and scaling. The top right 3 x 1 vector represents the
            translation component. The last row should have value [0, 0, 0, 1].
        coordinate_system: highdicom.CoordinateSystemNames | str
            Coordinate system (``"PATIENT"`` or ``"SLIDE"``) in which the volume
            is defined).
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

        self._affine = affine.astype(np.float64)
        self._coordinate_system = CoordinateSystemNames(coordinate_system)
        self._frame_of_reference_uid = frame_of_reference_uid

    @property
    @abstractmethod
    def spatial_shape(self) -> tuple[int, int, int]:
        """Tuple[int, int, int]: 3D spatial shape of the array.

        Does not include channel dimensions.

        """
        pass

    @property
    def coordinate_system(self) -> CoordinateSystemNames:
        """highdicom.CoordinateSystemNames | str:
            Coordinate system (``"PATIENT"`` or ``"SLIDE"``) in which the volume
            is defined).
        """
        return self._coordinate_system

    @property
    def nearest_center_indices(self) -> tuple[int, int, int]:
        """Array index of center of the volume, rounded down to the nearest
        integer value.

        Results are discrete zero-based array indices.

        Returns
        -------
        x: int
            First array index of the volume center.
        y: int
            Second array index of the volume center.
        z: int
            Third array index of the volume center.

        """
        return tuple((d - 1) // 2 for d in self.spatial_shape)

    @property
    def center_indices(self) -> tuple[float, float, float]:
        """Array index of center of the volume, as floats with sub-voxel
        precision.

        Results are continuous zero-based array indices.

        Returns
        -------
        x: float
            First array index of the volume center.
        y: float
            Second array index of the volume center.
        z: float
            Third array index of the volume center.

        """
        return tuple((d - 1) / 2.0 for d in self.spatial_shape)

    @property
    def center_position(self) -> tuple[float, float, float]:
        """Get frame-of-reference coordinates of the volume's center.

        Returns
        -------
        x: float
            Frame of reference x coordinate of the volume center.
        y: float
            Frame of reference y coordinate of the volume center.
        z: float
            Frame of reference z coordinate of the volume center.

        """
        center_index = np.array(self.center_indices).reshape((1, 3))
        center_position = self.map_indices_to_reference(center_index)

        return tuple(center_position.flatten().tolist())

    def map_indices_to_reference(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Transform image pixel indices to frame-of-reference coordinates.

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
        indices_augmented = np.vstack([
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
        reference_coordinates = np.vstack([
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

    def get_plane_position(self, plane_index: int) -> PlanePositionSequence:
        """Get plane position of a given plane.

        Parameters
        ----------
        plane_number: int
            Zero-based plane index (down the first dimension of the array).

        Returns
        -------
        highdicom.PlanePositionSequence:
            Plane position of the plane.

        """
        if plane_index < 0 or plane_index >= self.spatial_shape[0]:
            raise ValueError("Invalid plane number for volume.")
        index = np.array([[plane_index, 0, 0]])
        position = self.map_indices_to_reference(index)[0]

        if self.coordinate_system == CoordinateSystemNames.SLIDE:
            matrix_position = (1, 1)
        else:
            matrix_position = None

        return PlanePositionSequence(
            self.coordinate_system,
            position,
            pixel_matrix_position=matrix_position,
        )

    def get_plane_positions(self) -> list[PlanePositionSequence]:
        """Get plane positions of all planes in the volume.

        This assumes that the volume is encoded in a DICOM file with frames
        down axis 0, rows stacked down axis 1, and columns stacked down axis 2.

        Returns
        -------
        List[highdicom.PlanePositionSequence]:
            Plane position of the all planes (stacked down axis 0 of the
            volume).

        """
        indices = np.array(
            [
                [p, 0, 0] for p in range(self.spatial_shape[0])
            ]
        )
        positions = self.map_indices_to_reference(indices)

        if self.coordinate_system == CoordinateSystemNames.SLIDE:
            matrix_position = (1, 1)
        else:
            matrix_position = None

        return [
            PlanePositionSequence(
                self.coordinate_system,
                pos,
                pixel_matrix_position=matrix_position,
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
            Plane orientation sequence.

        """
        return PlaneOrientationSequence(
            self.coordinate_system,
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
            slice_thickness=self.spacing_between_slices,
            spacing_between_slices=self.spacing_between_slices,
        )

    @property
    def frame_of_reference_uid(self) -> UID | None:
        """Union[highdicom.UID, None]: Frame of reference UID."""
        if self._frame_of_reference_uid is None:
            return None
        return UID(self._frame_of_reference_uid)

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix

        This matrix maps an index of the array into a position in the LPS
        frame-of-reference coordinate space.

        """
        return self._affine.copy()

    def get_affine(
        self,
        output_convention: (
            str |
            Sequence[str | PatientOrientationValuesBiped] |
            None
        ),
    ) -> np.ndarray:
        """Get affine matrix in a particular convention.

        Note that DICOM uses the left-posterior-superior ("LPS") convention
        relative to the patient, in which the increasing direction of the first
        moves from the patient's right to left, the increasing direction of the
        second axis moves from the patient's anterior to posterior, and the
        increasing direction of the third axis moves from the patient's
        inferior (foot) to superior (head). In highdicom, this is represented
        by the string ``"LPH"`` (left-posterior-head). Since highdicom volumes
        follow this convention, the affine matrix is stored internally as a
        matrix that maps array indices into coordinates along these three
        axes.

        This method allows you to get the affine matrix that maps the same
        array indices into coordinates in a frame-of-reference that uses a
        different convention. Another convention in widespread use is the
        ``"RAH"`` (aka "RAS") convention used by the Nifti file format and many
        neuro-image analysis tools.

        Parameters
        ----------
        output_convention: str | Sequence[str | highdicom.PatientOrientationValuesBiped] | None
            Description of a convention for defining patient-relative
            frame-of-reference consisting of three directions, either L or R,
            either A or P, and either F or H, in any order. May be passed
            either as a tuple of
            :class:`highdicom.PatientOrientationValuesBiped` values or the
            single-letter codes representing them, or the same characters as a
            single three-character string, such as ``"RAH"``.

        Returns
        -------
        numpy.ndarray:
            4x4 affine transformation matrix mapping augmented voxel indices to
            frame-of-reference coordinates defined by the chosen convention.

        """  # noqa: E501
        affine = self.affine
        if output_convention is not None:
            affine = _transform_affine_to_convention(
                affine,
                self.spatial_shape,
                from_reference_convention=(
                    PatientOrientationValuesBiped.L,
                    PatientOrientationValuesBiped.P,
                    PatientOrientationValuesBiped.H,
                ),
                to_reference_convention=output_convention,
            )
        return affine

    @property
    def inverse_affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 inverse affine transformation matrix

        Inverse of the affine matrix. This matrix maps a position in the LPS
        frame of reference coordinate space into an index into the array.

        """
        return np.linalg.inv(self._affine)

    @property
    def direction_cosines(self) -> tuple[
        float, float, float, float, float, float
    ]:
        """Tuple[float, float, float, float, float float]:

        Tuple of 6 floats giving the direction cosines of the vector along the
        rows and the vector along the columns, matching the format of the DICOM
        Image Orientation Patient and Image Orientation Slide attributes.

        Assumes that frames are stacked down axis 0, rows down axis 1, and
        columns down axis 2 (the convention used to create volumes from
        images).

        """
        vec_along_rows = self._affine[:3, 2].copy()
        vec_along_columns = self._affine[:3, 1].copy()
        vec_along_columns /= np.sqrt((vec_along_columns ** 2).sum())
        vec_along_rows /= np.sqrt((vec_along_rows ** 2).sum())
        return tuple([*vec_along_rows.tolist(), *vec_along_columns.tolist()])

    @property
    def pixel_spacing(self) -> tuple[float, float]:
        """Tuple[float, float]:

        Within-plane pixel spacing in millimeter units. Two values (spacing
        between rows, spacing between columns), matching the format of the
        DICOM PixelSpacing attribute.

        Assumes that frames are stacked down axis 0, rows down axis 1, and
        columns down axis 2 (the convention used to create volumes from
        images).

        """
        vec_along_rows = self._affine[:3, 2]
        vec_along_columns = self._affine[:3, 1]
        spacing_between_columns = np.sqrt((vec_along_rows ** 2).sum()).item()
        spacing_between_rows = np.sqrt((vec_along_columns ** 2).sum()).item()
        return spacing_between_rows, spacing_between_columns

    @property
    def spacing_between_slices(self) -> float:
        """float:

        Spacing between consecutive slices in millimeter units.

        Assumes that frames are stacked down axis 0, rows down axis 1, and
        columns down axis 2 (the convention used to create volumes from
        images).

        """
        slice_vec = self._affine[:3, 0]
        spacing = np.sqrt((slice_vec ** 2).sum()).item()
        return spacing

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Tuple[float, float, float]:

        Pixel spacing in millimeter units for the three spatial directions.
        Three values, one for each spatial dimension.

        """
        dir_mat = self._affine[:3, :3]
        norms = np.sqrt((dir_mat ** 2).sum(axis=0))
        return tuple(norms.tolist())

    @property
    def voxel_volume(self) -> float:
        """float: The volume of a single voxel in cubic millimeters."""
        return np.prod(self.spacing).item()

    @property
    def position(self) -> tuple[float, float, float]:
        """Tuple[float, float, float]:

        Position in the frame of reference space of the center of voxel at
        indices (0, 0, 0).

        """
        return tuple(self._affine[:3, 3].tolist())

    @property
    def physical_extent(self) -> tuple[float, float, float]:
        """tuple[float, float, float]: Side lengths of the volume
        in millimeters.

        """
        return tuple(
            [n * d for n, d in zip(self.spatial_shape, self.spacing)]
        )

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
        of the array.

        """
        dir_mat = self._affine[:3, :3]
        norms = np.sqrt((dir_mat ** 2).sum(axis=0))
        return dir_mat / norms

    def spacing_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def unit_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        index: int | slice | tuple[int | slice],
    ) -> Self:
        pass

    def _prepare_getitem_index(
        self,
        index: int | slice | tuple[int | slice],
    ) -> tuple[tuple[slice], tuple[int, int, int], np.ndarray]:

        def _check_int(val: int, dim: int) -> None:
            if (
                val < -self.spatial_shape[dim] or
                val >= self.spatial_shape[dim]
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
            if index == -1:
                end_index = None
            else:
                end_index = index + 1
            tuple_index = (slice(index, end_index), )
        elif isinstance(index, slice):
            # Make into a tuple of length one to standardize the format
            _check_slice(index, 0)
            tuple_index = (cast(slice, index), )
        elif isinstance(index, tuple):
            index_list: list[slice] = []
            for dim, item in enumerate(index):
                if isinstance(item, int):
                    # Change the index to a slice of length one so that all
                    # dimensions are retained in the output array.
                    _check_int(item, dim)
                    if item == -1:
                        end_index = None
                    else:
                        end_index = item + 1
                    item = slice(item, end_index)
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
        pad_width: int | Sequence[int] | Sequence[Sequence[int]],
        *,
        mode: PadModes | str = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> Self:
        pass

    def _prepare_pad_width(
        self,
        pad_width: int | Sequence[int] | Sequence[Sequence[int]],
    ) -> tuple[np.ndarray, list[list[int]]]:
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
                    "Argument 'pad_width' cannot contain negative values."
                )
            full_pad_width: list[list[int]] = [[pad_width, pad_width]] * 3
        elif isinstance(pad_width, Sequence):
            if isinstance(pad_width[0], int):
                if len(pad_width) != 2:
                    raise ValueError("Invalid arrangement in 'pad_width'.")
                if pad_width[0] < 0 or pad_width[1] < 0:
                    raise ValueError(
                        "Argument 'pad_width' cannot contain negative values."
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
                'Argument "indices" must consist of the values 0, 1, and 2 '
                'in some order.'
            )

        return _transform_affine_matrix(
            affine=self._affine,
            shape=self.spatial_shape,
            permute_indices=indices,
        )

    @abstractmethod
    def copy(self) -> Self:
        """Create a copy of the object.

        Returns
        -------
        Self:
            Copy of the original object.

        """
        pass

    @abstractmethod
    def permute_spatial_axes(self, indices: Sequence[int]) -> Self:
        """Create a new volume by permuting the spatial axes.

        Parameters
        ----------
        indices: Sequence[int]
            List of three integers containing the values 0, 1 and 2 in some
            order. Note that you may not change the position of the channel
            axis (if present).

        Returns
        -------
        Self:
            New volume with spatial axes permuted in the provided order.

        """
        pass

    def random_permute_spatial_axes(
        self,
        axes: Sequence[int] = (0, 1, 2)
    ) -> Self:
        """Create a new geometry by randomly permuting the spatial axes.

        Parameters
        ----------
        axes: Optional[Sequence[int]]
            Sequence of three integers containing the values 0, 1 and 2 in some
            order. The sequence must contain 2 or 3 elements. This subset of
            axes will axes will be included when generating indices for
            permutation. Any axis not in this sequence will remain in its
            original position.

        Returns
        -------
        Self:
            New geometry with spatial axes permuted randomly.

        """
        if len(axes) < 2 or len(axes) > 3:
            raise ValueError(
                "Argument 'axes' must contain 2 or 3 items."
            )

        if len(set(axes)) != len(axes):
            raise ValueError(
                "Argument 'axes' should contain unique values."
            )

        if not set(axes) <= {0, 1, 2}:
            raise ValueError(
                "Argument 'axes' should contain only 0, 1, and 2."
            )

        indices = np.random.permutation(axes).tolist()
        if len(indices) == 2:
            missing_index = list({0, 1, 2} - set(indices))[0]
            indices.insert(missing_index, missing_index)

        return self.permute_spatial_axes(indices)

    def get_closest_patient_orientation(self) -> tuple[
        PatientOrientationValuesBiped,
        PatientOrientationValuesBiped,
        PatientOrientationValuesBiped,
    ]:
        """Get patient orientation codes that best represent the affine.

        Note that this is not valid if the volume is not defined within the
        patient coordinate system.

        Returns
        -------
        Tuple[highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped, highdicom.enum.PatientOrientationValuesBiped]:
            Tuple giving the closest patient orientation.

        """  # noqa: E501
        if self.coordinate_system != CoordinateSystemNames.PATIENT:
            raise RuntimeError(
                'Volume is not defined in the patient coordinate system.'
            )
        return get_closest_patient_orientation(self._affine)

    def to_patient_orientation(
        self,
        patient_orientation: (
            str |
            Sequence[str | PatientOrientationValuesBiped]
        ),
    ) -> Self:
        """Rearrange the array to a given orientation.

        The resulting volume is formed from this volume through a combination
        of axis permutations and flips of the spatial axes. Its patient
        orientation will be as close to the desired orientation as can be
        achieved with these operations alone (and in particular without
        resampling the array).

        Note that this is not valid if the volume is not defined within the
        patient coordinate system.

        Parameters
        ----------
        patient_orientation: Union[str, Sequence[Union[str, highdicom.PatientOrientationValuesBiped]]]
            Desired patient orientation, as either a sequence of three
            highdicom.PatientOrientationValuesBiped values, or a string
            such as ``"FPL"`` using the same characters.

        Returns
        -------
        Self:
            New volume with the requested patient orientation.

        """  # noqa: E501
        if self.coordinate_system != CoordinateSystemNames.PATIENT:
            raise RuntimeError(
                'Volume is not defined in the patient coordinate system.'
            )
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
            result = self.flip_spatial(flip_axes)
        else:
            result = self

        return result.permute_spatial_axes(permute_indices)

    def swap_spatial_axes(self, axis_1: int, axis_2: int) -> Self:
        """Swap two spatial axes of the array.

        Parameters
        ----------
        axis_1: int
            Spatial axis index (0, 1 or 2) to swap with ``axis_2``.
        axis_2: int
            Spatial axis index (0, 1 or 2) to swap with ``axis_1``.

        Returns
        -------
        Self:
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

        return self.permute_spatial_axes(permutation)

    def flip_spatial(self, axes: int | Sequence[int]) -> Self:
        """Flip the spatial axes of the array.

        Note that this flips the array and updates the affine to reflect the
        flip.

        Parameters
        ----------
        axes: Union[int, Sequence[int]]
            Axis or list of axis indices that should be flipped. These should
            include only the spatial axes (0, 1, and/or 2).

        Returns
        -------
        highdicom.Volume:
            New volume with spatial axes flipped as requested.

        """
        if isinstance(axes, int):
            axes = [axes]

        if len(axes) > 3 or len(set(axes) - {0, 1, 2}) > 0:
            raise ValueError(
                'Argument "axis" must contain only values 0, 1, and/or 2.'
            )

        # We will reuse the existing __getitem__ implementation, which has all
        # this logic figured out already
        index = []
        for d in range(3):
            if d in axes:
                index.append(slice(-1, None, -1))
            else:
                index.append(slice(None))

        return self[tuple(index)]

    def random_flip_spatial(self, axes: Sequence[int] = (0, 1, 2)) -> Self:
        """Randomly flip the spatial axes of the array.

        Note that this flips the array and updates the affine to reflect the
        flip.

        Parameters
        ----------
        axes: Union[int, Sequence[int]]
            Axis or list of axis indices that may be flipped. These should
            include only the spatial axes (0, 1, and/or 2). Each axis in this
            list is flipped in the output volume with probability 0.5.

        Returns
        -------
        Self:
            New volume with selected spatial axes randomly flipped.

        """
        if len(axes) < 2 or len(axes) > 3:
            raise ValueError(
                "Argument 'axes' must contain 2 or 3 items."
            )

        if len(set(axes)) != len(axes):
            raise ValueError(
                "Argument 'axes' should contain unique values."
            )

        if not set(axes) <= {0, 1, 2}:
            raise ValueError(
                "Argument 'axes' should contain only 0, 1, and 2."
            )

        slices = []
        for d in range(3):
            if d in axes:
                if np.random.randint(2) == 1:
                    slices.append(slice(None, None, -1))
                else:
                    slices.append(slice(None))
            else:
                slices.append(slice(None))

        return self[tuple(slices)]

    @property
    def handedness(self) -> AxisHandedness:
        """highdicom.AxisHandedness: Axis handedness of the volume.

        This indicates whether the volume's three spatial axes form a
        right-handed or left-handed coordinate system in the frame-of-reference
        space.

        """
        v1, v2, v3 = self.spacing_vectors()
        if np.cross(v1, v2) @ v3 < 0.0:
            return AxisHandedness.LEFT_HANDED
        return AxisHandedness.RIGHT_HANDED

    def ensure_handedness(
        self,
        handedness: AxisHandedness | str,
        *,
        flip_axis: int | None = None,
        swap_axes: Sequence[int] | None = None,
    ) -> Self:
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
        swap_axes: Union[Sequence[int], None], optional
            Specification of a sequence of two spatial axis indices (each being
            0, 1, or 2) to swap if required to meet the given handedness
            requirement.

        Returns
        -------
        Self:
            New volume with corrected handedness.

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
            return self.flip_spatial(flip_axis)

        if len(swap_axes) != 2:
            raise ValueError(
                "Argument 'swap_axes' must have length 2."
            )

        return self.swap_spatial_axes(swap_axes[0], swap_axes[1])

    def pad_to_spatial_shape(
        self,
        spatial_shape: Sequence[int],
        *,
        mode: PadModes = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> Self:
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
        Self:
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

    def crop_to_spatial_shape(self, spatial_shape: Sequence[int]) -> Self:
        """Center-crop volume to a given spatial shape.

        Parameters
        ----------
        spatial_shape: Sequence[int]
            Sequence of three integers specifying the spatial shape to crop to.
            This shape must be no larger than the existing shape along any of
            the three spatial dimensions.

        Returns
        -------
        Self:
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

    def pad_or_crop_to_spatial_shape(
        self,
        spatial_shape: Sequence[int],
        *,
        mode: PadModes = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> Self:
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
        Self:
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
                crop_vals.append((0, insize))
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

    def random_spatial_crop(self, spatial_shape: Sequence[int]) -> Self:
        """Create a random crop of a certain shape from the volume.

        Parameters
        ----------
        spatial_shape: Sequence[int]
            Sequence of three integers specifying the spatial shape to pad or
            crop to.

        Returns
        -------
        Self:
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

    def geometry_equal(
        self,
        other: Union['Volume', 'VolumeGeometry'],
        tol: float | None = _DEFAULT_EQUALITY_TOLERANCE,
    ) -> bool:
        """Determine whether two volumes have the same geometry.

        Parameters
        ----------
        other: Union[highdicom.Volume, highdicom.VolumeGeometry]
            Volume or volume geometry to which this volume should be compared.
        tol: Union[float, None], optional
            Absolute Tolerance used to determine equality of affine matrices.
            If None, affine matrices must match exactly.

        Return
        ------
        bool:
            True if the geometries match (up to the specified tolerance). False
            otherwise.

        """
        if (
            self.frame_of_reference_uid is not None and
            other.frame_of_reference_uid is not None
        ):
            if self.frame_of_reference_uid != self.frame_of_reference_uid:
                return False

        if self.spatial_shape != other.spatial_shape:
            return False

        if self.coordinate_system != other.coordinate_system:
            return False

        if tol is None:
            return np.array_equal(self._affine, other._affine)
        else:
            return np.allclose(
                self._affine,
                other._affine,
                atol=tol,
            )

    def match_geometry(
        self,
        other: Union['Volume', 'VolumeGeometry'],
        *,
        mode: PadModes = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
        tol: float = _DEFAULT_EQUALITY_TOLERANCE,
    ) -> Self:
        """Match the geometry of this volume to another.

        This performs a combination of permuting, padding and cropping, and
        flipping (in that order) such that the geometry of this volume matches
        that of ``other``. Notably, the voxels are not resampled. If the
        geometry cannot be matched using these operations, then a
        ``RuntimeError`` is raised.

        Parameters
        ----------
        other: Union[highdicom.Volume, highdicom.VolumeGeometry]
            Volume or volume geometry to which this volume should be matched.

        Returns
        -------
        Self:
            New volume formed by matching the geometry of this volume to that
            of ``other``.

        Raises
        ------
        RuntimeError:
            If the geometries cannot be matched without resampling the array.

        """
        if (
            self.frame_of_reference_uid is not None and
            other.frame_of_reference_uid is not None
        ):
            if self.frame_of_reference_uid != self.frame_of_reference_uid:
                raise RuntimeError(
                    "Volumes do not have matching frame of reference UIDs."
                )

        if self.coordinate_system != other.coordinate_system:
            raise RuntimeError(
                "Volumes do not exist in the same coordinate system."
            )

        permute_indices = []
        step_sizes = []
        for u, s in zip(other.unit_vectors(), other.spacing):
            for j, (v, t) in enumerate(
                zip(self.unit_vectors(), self.spacing)
            ):
                dot_product = u @ v
                if (
                    np.abs(dot_product - 1.0) < tol or
                    np.abs(dot_product + 1.0) < tol
                ):
                    permute_indices.append(j)

                    scale_factor = s / t
                    step = int(np.round(scale_factor))
                    if abs(scale_factor - step) > tol:
                        raise RuntimeError(
                            "Non-integer scale factor required."
                        )

                    if dot_product < 0.0:
                        step = -step

                    step_sizes.append(step)

                    break
            else:
                raise RuntimeError(
                    "Direction vectors could not be aligned."
                )

        requires_permute = permute_indices != [0, 1, 2]
        if requires_permute:
            new_volume = self.permute_spatial_axes(permute_indices)
        else:
            new_volume = self

        # Now figure out cropping
        origin_offset = (
            np.array(other.position) -
            np.array(new_volume.position)
        )

        crop_slices = []
        pad_values = []
        requires_crop = False
        requires_pad = False

        for v, spacing, step, out_shape, in_shape in zip(
            new_volume.unit_vectors(),
            new_volume.spacing,
            step_sizes,
            other.spatial_shape,
            new_volume.spatial_shape,
        ):
            offset = v @ origin_offset
            start_ind = offset / spacing
            start_pos = int(np.round(start_ind))
            end_pos = start_pos + out_shape * step

            if abs(start_pos - start_ind) > tol:
                raise RuntimeError(
                    "Required translation is non-integer "
                    "multiple of voxel spacing."
                )

            if step > 0:
                pad_before = max(-start_pos, 0)
                pad_after = max(end_pos - in_shape, 0)
                crop_start = start_pos + pad_before
                crop_stop = end_pos + pad_before

                if crop_start > 0 or crop_stop < out_shape:
                    requires_crop = True
            else:
                pad_after = max(start_pos - in_shape + 1, 0)
                pad_before = max(-end_pos - 1, 0)
                crop_start = start_pos + pad_before
                crop_stop = end_pos + pad_before

                # Need the crop operation to flip
                requires_crop = True

                if crop_stop == -1:
                    crop_stop = None

            if pad_before > 0 or pad_after > 0:
                requires_pad = True

            crop_slices.append(
                slice(crop_start, crop_stop, step)
            )
            pad_values.append((pad_before, pad_after))

        if not (
            requires_permute or requires_pad or requires_crop
        ):
            new_volume = new_volume.copy()

        if requires_pad:
            new_volume = new_volume.pad(
                pad_values,
                mode=mode,
                constant_value=constant_value,
                per_channel=per_channel,
            )

        if requires_crop:
            new_volume = new_volume[tuple(crop_slices)]

        return new_volume


class VolumeGeometry(_VolumeBase):

    """Class encapsulating the geometry of a volume.

    Unlike the similar :class:`highdicom.Volume`, items of this class do not
    contain voxel data for the underlying volume, just a description of the
    geometry.

    See :doc:`volume` for an introduction to using volumes and volume
    geometries.

    """

    def __init__(
        self,
        affine: np.ndarray,
        spatial_shape: Sequence[int],
        coordinate_system: CoordinateSystemNames | str,
        frame_of_reference_uid: str | None = None,
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
            Number of voxels in the (implied) volume along the three spatial
            dimensions.
        coordinate_system: highdicom.CoordinateSystemNames | str
            Coordinate system (``"PATIENT"`` or ``"SLIDE"``) in which the volume
            is defined).
        frame_of_reference_uid: Optional[str], optional
            Frame of reference UID for the frame of reference, if known.

        """
        super().__init__(
            affine,
            coordinate_system=coordinate_system,
            frame_of_reference_uid=frame_of_reference_uid,
        )

        if len(spatial_shape) != 3:
            raise ValueError("Argument 'spatial_shape' must have length 3.")
        self._spatial_shape = tuple(spatial_shape)

    @classmethod
    def from_attributes(
        cls,
        *,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        rows: int,
        columns: int,
        pixel_spacing: Sequence[float],
        spacing_between_slices: float,
        number_of_frames: int,
        coordinate_system: CoordinateSystemNames | str,
        frame_of_reference_uid: str | None = None,
    ) -> Self:
        """Create a volume from DICOM attributes.

        The resulting geometry assumes that the frames of the image whose
        attributes are used are stacked down axis 0, the rows down axis 1, and
        the columns down axis 2. Furthermore, frames will be stacked such that
        the resulting geometry forms a right-handed coordinate system in the
        frame-of-reference coordinate system.

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
        rows: int
            Number of rows in each frame.
        columns: int
            Number of columns in each frame.
        pixel_spacing: Sequence[float]
            Spacing between pixels in millimeter unit along the column
            direction (first value: spacing between rows, vertical, top to
            bottom, increasing row index) and the row direction (second value:
            spacing between columns: horizontal, left to right, increasing
            column index). Corresponds to DICOM attribute "PixelSpacing".
        spacing_between_slices: float
            Spacing between slices in millimeter units in the frame of
            reference coordinate system space. Corresponds to the DICOM
            attribute "SpacingBetweenSlices" (however, this may not be present
            in many images and may need to be inferred from
            "ImagePositionPatient" attributes of consecutive slices).
        number_of_frames: int
            Number of frames in the volume.
        coordinate_system: highdicom.CoordinateSystemNames | str
            Coordinate system (``"PATIENT"`` or ``"SLIDE"``) in which the volume
            is defined).
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID, if known. Corresponds to DICOM attribute
            FrameOfReferenceUID.

        Returns
        -------
        highdicom.VolumeGeometry:
            New Volume using the given array and DICOM attributes.

        """
        affine = create_affine_matrix_from_attributes(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=spacing_between_slices,
            index_convention=VOLUME_INDEX_CONVENTION,
            slices_first=True,
        )
        spatial_shape = (number_of_frames, rows, columns)

        return cls(
            affine=affine,
            spatial_shape=spatial_shape,
            coordinate_system=coordinate_system,
            frame_of_reference_uid=frame_of_reference_uid,
        )

    @classmethod
    def from_components(
        cls,
        spatial_shape: Sequence[int],
        *,
        spacing: Sequence[float] | float,
        coordinate_system: CoordinateSystemNames | str,
        position: Sequence[float] | None = None,
        center_position: Sequence[float] | None = None,
        direction: Sequence[float] | None = None,
        patient_orientation: (
            str |
            Sequence[str | PatientOrientationValuesBiped] |
            None
        ) = None,
        frame_of_reference_uid: str | None = None,
    ) -> Self:
        """Construct a VolumeGeometry from components of the affine matrix.

        Parameters
        ----------
        array: numpy.ndarray
            Three dimensional array of voxel data.
        spacing: Sequence[float]
            Spacing between pixel centers in the the frame of reference
            coordinate system along each of the dimensions of the array. Should
            be either a sequence of length 3 to give the values along the three
            spatial dimensions, or a single float value to be shared by all
            spatial dimensions.
        coordinate_system: highdicom.CoordinateSystemNames | str
            Coordinate system (``"PATIENT"`` or ``"SLIDE"`` in which the volume
            is defined).
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
            ``"FPL"`` using the same characters. Incompatible with
            ``direction``.
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID for the frame of reference, if known.
        channels: dict[int | str | ChannelDescriptor, Sequence[int | str | float | Enum]] | None, optional
            Specification of channels of the array. Channels are additional
            dimensions of the array beyond the three spatial dimensions. For
            each such additional dimension (if any), an item in this dictionary
            is required to specify the meaning. The dictionary key specifies
            the meaning of the dimension, which must be either an instance of
            highdicom.ChannelDescriptor, specifying a DICOM tag whose attribute
            describes the channel, a a DICOM keyword describing a DICOM
            attribute, or an integer representing the tag of a DICOM attribute.
            The corresponding item of the dictionary is a sequence giving the
            value of the relevant attribute at each index in the array. The
            insertion order of the dictionary is significant as it is used to
            match items to the corresponding dimensions of the array (the first
            item in the dictionary corresponds to axis 3 of the array and so
            on).

        Returns
        -------
        highdicom.VolumeGeometry:
            Volume constructed from the provided components.

        """  # noqa: E501
        if patient_orientation is not None:
            if (
                CoordinateSystemNames(coordinate_system) !=
                CoordinateSystemNames.PATIENT
            ):
                raise ValueError(
                    "Argument 'patient_orientation' should be provided only "
                    "for volumes in the patient coordinate system."
                )

        affine = create_affine_matrix_from_components(
            spacing=spacing,
            direction=direction,
            patient_orientation=patient_orientation,
            position=position,
            center_position=center_position,
            spatial_shape=spatial_shape,
        )
        return cls(
            spatial_shape=spatial_shape,
            affine=affine,
            coordinate_system=coordinate_system,
            frame_of_reference_uid=frame_of_reference_uid,
        )

    def copy(self) -> Self:
        """Get an unaltered copy of the geometry.

        Returns
        -------
        highdicom.VolumeGeometry:
            Copy of the original geometry.

        """
        return self.__class__(
            affine=self._affine.copy(),
            spatial_shape=self.spatial_shape,
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Tuple[int, int, int]: Spatial shape of the array.

        Does not include the channel dimension.

        """
        return self._spatial_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple[int, ...]: Shape of the underlying array.

        For objects of type :class:`highdicom.VolumeGeometry`, this is
        equivalent to `.shape`.

        """
        return self.spatial_shape

    def __getitem__(
        self,
        index: int | slice | tuple[int | slice],
    ) -> Self:
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

        return self.__class__(
            affine=new_affine,
            spatial_shape=new_shape,
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def pad(
        self,
        pad_width: int | Sequence[int] | Sequence[Sequence[int]],
        *,
        mode: PadModes | str = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> Self:
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
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def permute_spatial_axes(self, indices: Sequence[int]) -> Self:
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
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def with_array(
        self,
        array: np.ndarray,
        channels: dict[
            BaseTag | int | str | ChannelDescriptor,
            Sequence[int | str | float | Enum]
        ] | None = None,
    ) -> 'Volume':
        """Create a volume using this geometry and an array.

        Parameters
        ----------
        array: numpy.ndarray
            Array of voxel data. Must have the same spatial shape as the
            existing volume (i.e. first three elements of the shape match).
            Must additionally have the same shape along the channel dimensions,
            unless the `channels` parameter is provided.
        channels: dict[int | str | ChannelDescriptor, Sequence[int | str | float | Enum]] | None, optional
            Specification of channels of the array. Channels are additional
            dimensions of the array beyond the three spatial dimensions. For
            each such additional dimension (if any), an item in this dictionary
            is required to specify the meaning. The dictionary key specifies
            the meaning of the dimension, which must be either an instance of
            highdicom.ChannelDescriptor, specifying a DICOM tag whose attribute
            describes the channel, a a DICOM keyword describing a DICOM
            attribute, or an integer representing the tag of a DICOM attribute.
            The corresponding item of the dictionary is a sequence giving the
            value of the relevant attribute at each index in the array. The
            insertion order of the dictionary is significant as it is used to
            match items to the corresponding dimensions of the array (the first
            item in the dictionary corresponds to axis 3 of the array and so
            on).

        Returns
        -------
        highdicom.Volume:
            Volume objects using this geometry and the given array.

        """  # noqa: E501
        return Volume(
            array=array,
            affine=self.affine,
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
            channels=channels,
        )


class Volume(_VolumeBase):

    """Class representing an array of regularly-spaced frames in 3D space.

    This class combines a NumPy array with an affine matrix describing the
    location of the voxels in the frame-of-reference coordinate space. A
    Volume is not a DICOM object itself, but represents a volume that may
    be extracted from DICOM image, and/or encoded within a DICOM object,
    potentially following any number of processing steps.

    All such volumes have a geometry that exists either within DICOM's patient
    coordinate system or its slide coordinate system, both of which clearly
    define the meaning of the three spatial axes of the frame of reference
    coordinate system.

    All volume arrays have three spatial dimensions. They may optionally have
    further non-spatial dimensions, known as "channel" dimensions, whose
    meaning is explicitly specified.

    See :doc:`volume` for an introduction to using volumes and volume
    geometries.

    """

    def __init__(
        self,
        array: np.ndarray,
        affine: np.ndarray,
        coordinate_system: CoordinateSystemNames | str,
        frame_of_reference_uid: str | None = None,
        channels: dict[
            BaseTag | int | str | ChannelDescriptor,
            Sequence[int | str | float | Enum]
        ] | None = None,
    ):
        """

        Parameters
        ----------
        array: numpy.ndarray
            Array of voxel data. Must be at least 3D. The first three
            dimensions are the three spatial dimensions, and any subsequent
            dimensions are channel dimensions. Any datatype is permitted.
        affine: numpy.ndarray
            4 x 4 affine matrix representing the transformation from pixel
            indices (slice index, row index, column index) to the
            frame-of-reference coordinate system. The top left 3 x 3 matrix
            should be a scaled orthogonal matrix representing the rotation and
            scaling. The top right 3 x 1 vector represents the translation
            component. The last row should have value [0, 0, 0, 1].
        coordinate_system: highdicom.CoordinateSystemNames | str
            Coordinate system (``"PATIENT"`` or ``"SLIDE"`` in which the volume
            is defined).
        frame_of_reference_uid: Optional[str], optional
            Frame of reference UID for the frame of reference, if known.
        channels: dict[int | str | ChannelDescriptor, Sequence[int | str | float | Enum]] | None, optional
            Specification of channels of the array. Channels are additional
            dimensions of the array beyond the three spatial dimensions. For
            each such additional dimension (if any), an item in this dictionary
            is required to specify the meaning. The dictionary key specifies
            the meaning of the dimension, which must be either an instance of
            highdicom.ChannelDescriptor, specifying a DICOM tag whose attribute
            describes the channel, a a DICOM keyword describing a DICOM
            attribute, or an integer representing the tag of a DICOM attribute.
            The corresponding item of the dictionary is a sequence giving the
            value of the relevant attribute at each index in the array. The
            insertion order of the dictionary is significant as it is used to
            match items to the corresponding dimensions of the array (the first
            item in the dictionary corresponds to axis 3 of the array and so
            on).

        """  # noqa: E501
        super().__init__(
            affine=affine,
            coordinate_system=coordinate_system,
            frame_of_reference_uid=frame_of_reference_uid,
        )
        if array.ndim < 3:
            raise ValueError(
                "Argument 'array' must be at least three dimensional."
            )

        if channels is None:
            channels = {}

        if len(channels) != array.ndim - 3:
            raise ValueError(
                "Number of items in the 'channels' parameter "
                f'({len(channels)}) does not match the number of channel '
                f'dimensions in the array ({array.ndim - 3}).'
            )

        self._channels: dict[
            ChannelDescriptor, list[str | int | float | Enum]
        ] = {}

        # NB insertion order of the dictionary is significant
        for a, (iden, values) in enumerate(channels.items()):

            channel_number = a + 3

            if not isinstance(iden, ChannelDescriptor):
                iden_obj = ChannelDescriptor(iden)
            else:
                iden_obj = iden

            if iden_obj.is_enumerated:
                values = [iden_obj.value_type(v) for v in values]

            expected_length = array.shape[channel_number]
            if len(values) != expected_length:
                raise ValueError(
                    f'Number of values for channel number {channel_number} '
                    f'({len(values)}) does not match the size of the '
                    'corresponding dimension of the array '
                    f'({expected_length}).'
                )

            if not all(isinstance(v, iden_obj.value_type) for v in values):
                raise TypeError(
                    f'Values for channel {iden_obj} '
                    f'do not have the expected type ({iden_obj.value_type}).'
                )

            if iden_obj in self._channels:
                raise ValueError(
                    'Channel identifiers must represent unique attributes.'
                )
            self._channels[iden_obj] = list(values)

        self._array = array

    @classmethod
    def from_attributes(
        cls,
        *,
        array: np.ndarray,
        image_position: Sequence[float],
        image_orientation: Sequence[float],
        pixel_spacing: Sequence[float],
        spacing_between_slices: float,
        coordinate_system: CoordinateSystemNames | str,
        frame_of_reference_uid: str | None = None,
        channels: dict[
            BaseTag | int | str | ChannelDescriptor,
            Sequence[int | str | float | Enum]
        ] | None = None,
    ) -> Self:
        """Create a volume from DICOM attributes.

        The resulting geometry assumes that the frames of the image whose
        attributes are used are stacked down axis 0, the rows down axis 1, and
        the columns down axis 2. Furthermore, frames will be stacked such that
        the resulting geometry forms a right-handed coordinate system in the
        frame-of-reference coordinate system.

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
            attribute "SpacingBetweenSlices" (however, this may not be present
            in many images and may need to be inferred from
            "ImagePositionPatient" attributes of consecutive slices).
        coordinate_system: highdicom.CoordinateSystemNames | str
            Coordinate system (``"PATIENT"`` or ``"SLIDE"`` in which the volume
            is defined).
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID, if known. Corresponds to DICOM attribute
            FrameOfReferenceUID.
        channels: dict[int | str | ChannelDescriptor, Sequence[int | str | float | Enum]] | None, optional
            Specification of channels of the array. Channels are additional
            dimensions of the array beyond the three spatial dimensions. For
            each such additional dimension (if any), an item in this dictionary
            is required to specify the meaning. The dictionary key specifies
            the meaning of the dimension, which must be either an instance of
            highdicom.ChannelDescriptor, specifying a DICOM tag whose attribute
            describes the channel, a a DICOM keyword describing a DICOM
            attribute, or an integer representing the tag of a DICOM attribute.
            The corresponding item of the dictionary is a sequence giving the
            value of the relevant attribute at each index in the array. The
            insertion order of the dictionary is significant as it is used to
            match items to the corresponding dimensions of the array (the first
            item in the dictionary corresponds to axis 3 of the array and so
            on).

        Returns
        -------
        highdicom.Volume:
            New Volume using the given array and DICOM attributes.

        """  # noqa: E501
        affine = create_affine_matrix_from_attributes(
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
            coordinate_system=coordinate_system,
            frame_of_reference_uid=frame_of_reference_uid,
            channels=channels,
        )

    @classmethod
    def from_components(
        cls,
        array: np.ndarray,
        *,
        spacing: Sequence[float] | float,
        coordinate_system: CoordinateSystemNames | str,
        position: Sequence[float] | None = None,
        center_position: Sequence[float] | None = None,
        direction: Sequence[float] | None = None,
        patient_orientation: (
            str |
            Sequence[str | PatientOrientationValuesBiped] |
            None
        ) = None,
        frame_of_reference_uid: str | None = None,
        channels: dict[
            BaseTag | int | str | ChannelDescriptor,
            Sequence[int | str | float | Enum]
        ] | None = None,
    ) -> Self:
        """Construct a Volume from components of the affine matrix.

        Parameters
        ----------
        array: numpy.ndarray
            Three dimensional array of voxel data.
        spacing: Sequence[float]
            Spacing between pixel centers in the the frame of reference
            coordinate system along each of the dimensions of the array. Should
            be either a sequence of length 3 to give the values along the three
            spatial dimensions, or a single float value to be shared by all
            spatial dimensions.
        coordinate_system: highdicom.CoordinateSystemNames | str
            Coordinate system (``"PATIENT"`` or ``"SLIDE"`` in which the volume
            is defined).
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
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID for the frame of reference, if known.
        channels: dict[int | str | ChannelDescriptor, Sequence[int | str | float | Enum]] | None, optional
            Specification of channels of the array. Channels are additional
            dimensions of the array beyond the three spatial dimensions. For
            each such additional dimension (if any), an item in this dictionary
            is required to specify the meaning. The dictionary key specifies
            the meaning of the dimension, which must be either an instance of
            highdicom.ChannelDescriptor, specifying a DICOM tag whose attribute
            describes the channel, a a DICOM keyword describing a DICOM
            attribute, or an integer representing the tag of a DICOM attribute.
            The corresponding item of the dictionary is a sequence giving the
            value of the relevant attribute at each index in the array. The
            insertion order of the dictionary is significant as it is used to
            match items to the corresponding dimensions of the array (the first
            item in the dictionary corresponds to axis 3 of the array and so
            on).

        Returns
        -------
        highdicom.Volume:
            Volume constructed from the provided components.

        """  # noqa: E501
        if patient_orientation is not None:
            if (
                CoordinateSystemNames(coordinate_system) !=
                CoordinateSystemNames.PATIENT
            ):
                raise ValueError(
                    "Argument 'patient_orientation' should be provided only "
                    "for volumes in the patient coordinate system."
                )

        affine = create_affine_matrix_from_components(
            spacing=spacing,
            direction=direction,
            patient_orientation=patient_orientation,
            position=position,
            center_position=center_position,
            spatial_shape=array.shape[:3],
        )
        return cls(
            array=array,
            affine=affine,
            coordinate_system=coordinate_system,
            frame_of_reference_uid=frame_of_reference_uid,
            channels=channels,
        )

    def get_geometry(self) -> VolumeGeometry:
        """Get geometry for this volume.

        Returns
        -------
        highdicom.VolumeGeometry:
            Geometry object matching this volume.

        """
        return VolumeGeometry(
            affine=self._affine.copy(),
            spatial_shape=self.spatial_shape,
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid
        )

    @property
    def dtype(self) -> type:
        """type: Datatype of the array."""
        return self._array.dtype.type

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple[int, ...]: Shape of the underlying array.

        Includes any channel dimensions.

        """
        return tuple(self._array.shape)

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Tuple[int, int, int]: Spatial shape of the array.

        Does not include the channel dimensions.

        """
        return tuple(self._array.shape[:3])

    @property
    def number_of_channel_dimensions(self) -> int:
        """int: Number of channel dimensions."""
        return self._array.ndim - 3

    @property
    def channel_shape(self) -> tuple[int, ...]:
        """Tuple[int, ...]: Channel shape of the array.

        Does not include the spatial dimensions.

        """
        return tuple(self._array.shape[3:])

    @property
    def channel_descriptors(self) -> tuple[ChannelDescriptor, ...]:
        """tuple[highdicom.ChannelDescriptor]
        Descriptor of each channel.

        """
        return tuple(self._channels.keys())

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
            New array of voxel data. The shape (spatial and channel) must match
            the existing array.

        """
        if value.shape != self.shape:
            raise ValueError(
                "Array must match the shape of the existing array."
            )
        self._array = value

    def astype(self, dtype: type) -> Self:
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

    def copy(self) -> Self:
        """Get an unaltered copy of the volume.

        Returns
        -------
        highdicom.Volume:
            Copy of the original volume.

        """
        return self.__class__(
            array=self.array.copy(),  # TODO should this copy?
            affine=self._affine.copy(),
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
        )

    def with_array(
        self,
        array: np.ndarray,
        channels: dict[
            BaseTag | int | str | ChannelDescriptor,
            Sequence[int | str | float | Enum]
        ] | None = None,
    ) -> Self:
        """Get a new volume using a different array.

        The spatial and other metadata will be copied from this volume.
        The original volume will be unaltered.

        By default, the new volume will have the same channels (if any) as the
        existing volume. Different channels may be specified by passing the
        'channels' parameter.

        Parameters
        ----------
        array: np.ndarray
            New 3D or 4D array of voxel data. The spatial shape must match the
            existing array, but the presence and number of channels and/or the
            voxel datatype may differ.
        channels: dict[int | str | ChannelDescriptor, Sequence[int | str | float | Enum]] | None, optional
            Specification of channels as used by the constructor. If not
            specified, the channels are assumed to match those in the original
            volume and therefore the array must have the same shape as the
            array of the original volume.

        Returns
        -------
        highdicom.Volume:
            New volume using the given array and the metadata of this volume.

        """  # noqa: E501
        if array.shape[:3] != self.spatial_shape:
            raise ValueError(
                "Array must match the spatial shape of the existing array."
            )
        if channels is None:
            if array.ndim != 3:
                channels = self._channels
                if array.shape != self.shape:
                    raise ValueError(
                        "Array must match the shape of the existing array."
                    )
        return self.__class__(
            array=array,
            affine=self._affine.copy(),
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
            channels=channels,
        )

    def __getitem__(
        self,
        index: int | slice | tuple[int | slice],
    ) -> Self:
        """Get a spatial sub-volume of this volume as a new volume.

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
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
            channels=self._channels,
        )

    def permute_spatial_axes(self, indices: Sequence[int]) -> Self:
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

        new_array = np.transpose(
            self._array,
            [
                *indices,
                *[d + 3 for d in range(self.number_of_channel_dimensions)]
            ]
        )

        return self.__class__(
            array=new_array,
            affine=new_affine,
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
            channels=self._channels,
        )

    def permute_channel_axes_by_index(self, indices: Sequence[int]) -> Self:
        """Create a new volume by permuting the channel axes.

        Parameters
        ----------
        indices: Sequence[int]
            List of integers containing values in the range 0 (inclusive) to
            the number of channel dimensions (exclusive) in some order, used
            to permute the channels. A value of ``i`` corresponds to the channel
            given by ``volume.channel_identifiers[i]``.

        Returns
        -------
        highdicom.Volume:
            New volume with channel axes permuted in the provided order.

        """
        if len(set(indices)) != len(indices):
            raise ValueError(
                "Set of channel indices must not contain "
                "duplicates."
            )
        expected_indices = set(range(self.number_of_channel_dimensions))
        if set(indices) != expected_indices:
            raise ValueError(
                "Set of channel indices must match exactly those "
                "present in the volume."
            )
        full_indices = [0, 1, 2] + [ind + 3 for ind in indices]

        new_array = np.transpose(self._array, full_indices)

        new_channel_identifiers = [
            self.channel_descriptors[ind] for ind in indices
        ]
        new_channels = {
            iden: self._channels[iden] for iden in new_channel_identifiers
        }

        return self.with_array(
            array=new_array,
            channels=new_channels,
        )

    def permute_channel_axes(
        self,
        channel_identifiers: Sequence[BaseTag | int | str | ChannelDescriptor],
    ) -> Self:
        """Create a new volume by permuting the channel axes.

        Parameters
        ----------
        channel_identifiers: Sequence[pydicom.BaseTag | int | str | highdicom.ChannelDescriptor]
            List of channel identifiers matching those in the volume but in an
            arbitrary order.

        Returns
        -------
        highdicom.Volume:
            New volume with channel axes permuted in the provided order.

        """  # noqa: E501
        channel_identifier_objs = [
            self._get_channel_identifier(iden) for iden in channel_identifiers
        ]

        current_identifiers = self.channel_descriptors
        if len(set(channel_identifier_objs)) != len(channel_identifier_objs):
            raise ValueError(
                "Set of channel identifiers must not contain "
                "duplicates."
            )
        if set(channel_identifier_objs) != set(current_identifiers):
            raise ValueError(
                "Set of channel identifiers must match exactly those "
                "present in the volume."
            )

        permutation_indices = [
            current_identifiers.index(iden) for iden in channel_identifier_objs
        ]

        return self.permute_channel_axes_by_index(permutation_indices)

    def _get_channel_identifier(
        self,
        identifier: ChannelDescriptor | int | str,
    ) -> ChannelDescriptor:
        """Standardize representation of a channel identifier.

        Given a value used to specify a channel, check that such a channel
        exists in the volume and return a channel identifier as a
        highdicom.ChannelDescriptor object.

        Parameters
        ----------
        identifier: highdicom.ChannelDescriptor | int | str
            Identifier. Strings will be matched against keywords and integers
            will be matched against tags.

        Returns
        -------
        highdicom.ChannelDescriptor:
            Channel identifier in standard form.

        """
        if isinstance(identifier, ChannelDescriptor):
            if identifier not in self._channels:
                raise ValueError(
                    f"No channel with identifier '{identifier}' found "
                    'in volume.'
                )

            return identifier
        elif isinstance(identifier, str):
            for c in self.channel_descriptors:
                if c.keyword == identifier:
                    return c
            else:
                raise ValueError(
                    f"No channel identifier with keyword '{identifier}' found "
                    'in volume.'
                )
        elif isinstance(identifier, int):
            t = BaseTag(identifier)
            for c in self.channel_descriptors:
                if c.tag is not None and c.tag == t:
                    return c
            else:
                raise ValueError(
                    f"No channel identifier with tag '{t}' found "
                    'in volume.'
                )
        else:
            raise TypeError(
                f'Invalid type for channel identifier: {type(identifier)}'
            )

    def _get_channel_index(
        self,
        identifier: ChannelDescriptor | int | str,
    ) -> int:
        """Get zero-based channel index for a given channel.

        Parameters
        ----------
        identifier: highdicom.ChannelDescriptor | int | str
            Identifier. Strings will be matched against keywords and integers
            will be matched against tags.

        Returns
        -------
        int:
            Zero-based index of the channel within the channel axes.

        """
        identifier_obj = self._get_channel_identifier(identifier)

        index = self.channel_descriptors.index(identifier_obj)

        return index

    def get_channel_values(
        self,
        channel_identifier: int | str | ChannelDescriptor
    ) -> list[str | int | float | Enum]:
        """Get channel values along a particular dimension.

        Parameters
        ----------
        channel_identifier: highdicom.ChannelDescriptor | int | str
            Identifier of a channel within the image.

        Returns
        -------
        list[str | int | float | Enum]:
            Copy of channel values along the selected dimension.

        """
        iden = self._get_channel_identifier(channel_identifier)
        return self._channels[iden][:]

    def get_channel(self, *, keepdims: bool = False, **kwargs) -> Self:
        """Get a volume corresponding to a particular channel along one or more
        dimensions.

        Parameters
        ----------
        keepdims: bool
            Whether to keep a singleton dimension in the output volume.
        kwargs: dict[str, str | int | float | Enum]
            kwargs where the keyword is the keyword of a channel present in the
            volume and the value is the channel value along that channel.

        Returns
        -------
        highdicom.Volume:
            Volume representing a single channel of the original volume.

        """
        indexer: list[slice | int] = [slice(None)] * self._array.ndim

        new_channels = self._channels.copy()

        for kw, v in kwargs.items():

            iden = self._get_channel_identifier(kw)
            cind = self._get_channel_index(iden)

            iden = self.channel_descriptors[cind]
            if iden.is_enumerated:
                v = iden.value_type(v)
            elif not isinstance(v, iden.value_type):
                raise TypeError(
                    f"Value for argument '{iden}' must be of type "
                    f"'{iden.value_type}'."
                )

            dim_ind = cind + 3
            try:
                ind = self._channels[iden].index(v)
            except IndexError as e:
                raise IndexError(
                    f"Value {v} is not found in channel {iden}."
                ) from e

            if keepdims:
                indexer[dim_ind] = slice(ind, ind + 1)
                new_channels[iden] = [v]
            else:
                indexer[dim_ind] = ind
                del new_channels[iden]

        new_array = self._array[tuple(indexer)]

        return self.with_array(
            array=new_array,
            channels=new_channels,
        )

    def normalize_mean_std(
        self,
        per_channel: bool = True,
        output_mean: float = 0.0,
        output_std: float = 1.0,
    ) -> Self:
        """Normalize the intensities using the mean and variance.

        The resulting volume has zero mean and unit variance.

        Parameters
        ----------
        per_channel: bool, optional
            If True (the default), each channel along each channel dimension is
            normalized by its own mean and variance. If False, all channels are
            normalized together using the overall mean and variance.
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
            self.number_of_channel_dimensions > 0
        ):
            mean = self.array.mean(axis=(0, 1, 2), keepdims=True)
            std = self.array.std(axis=(0, 1, 2), keepdims=True)
        else:
            mean = self.array.mean()
            std = self.array.std()
        new_array = (
            (self.array - mean) / (std / output_std) + output_mean
        )

        return self.with_array(new_array)

    def normalize_min_max(
        self,
        output_min: float = 0.0,
        output_max: float = 1.0,
        per_channel: bool = False,
    ) -> Self:
        """Normalize by mapping its full intensity range to a fixed range.

        Other pixel values are scaled linearly within this range.

        Parameters
        ----------
        output_min: float, optional
            The value to which the minimum intensity is mapped.
        output_max: float, optional
            The value to which the maximum intensity is mapped.
        per_channel: bool, optional
            If True, each channel along each channel dimension is normalized by
            its own min and max. If False (the default), all channels are
            normalized together using the overall min and max.

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
            self.number_of_channel_dimensions > 1
        ):
            imin = self.array.min(axis=(0, 1, 2), keepdims=True)
            imax = self.array.max(axis=(0, 1, 2), keepdims=True)
        else:
            imin = self.array.min()
            imax = self.array.max()

        scale_factor = output_range / (imax - imin)
        new_array = (self.array - imin) * scale_factor + output_min

        return self.with_array(new_array)

    def clip(
        self,
        a_min: float | None,
        a_max: float | None,
    ) -> Self:
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

    def squeeze_channel(
        self,
        channel_descriptors: Sequence[
            int | str | BaseTag | ChannelDescriptor
        ] | None = None,
    ) -> Self:
        """Removes any singleton channel axes.

        Parameters
        ----------
        channel_descriptors: Sequence[str | int | highdicom.ChannelDescriptor] | None
            Identifiers of channels to squeeze. If ``None``, squeeze all
            singleton channels. Otherwise squeeze only the specified channels
            and raise an error if any cannot be squeezed.

        Returns
        -------
        highdicom.Volume:
            Volume with channel axis removed.

        """  # noqa: E501
        if channel_descriptors is None:
            channel_descriptors = self.channel_descriptors
            raise_error = False
        else:
            raise_error = True
            channel_descriptors = [
                ChannelDescriptor(iden) for iden in channel_descriptors
            ]
            for iden in channel_descriptors:
                if iden not in self._channels:
                    raise ValueError(
                        f'No channel with identifier: {iden}'
                    )

        to_squeeze = []
        new_channel_idens = []
        for iden in channel_descriptors:
            cind = self.channel_descriptors.index(iden)
            if self.channel_shape[cind] == 1:
                to_squeeze.append(cind + 3)
            else:
                if raise_error:
                    raise RuntimeError(
                        f'Volume has channels along the dimension {iden} and '
                        'cannot be squeezed.'
                    )
                new_channel_idens.append(iden)

        array = self.array.squeeze(tuple(to_squeeze))
        new_channels = {
            iden: self._channels[iden] for iden in new_channel_idens
        }

        return self.with_array(array, channels=new_channels)

    def pad(
        self,
        pad_width: int | Sequence[int] | Sequence[Sequence[int]],
        *,
        mode: PadModes | str = PadModes.CONSTANT,
        constant_value: float = 0.0,
        per_channel: bool = False,
    ) -> Self:
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
            Value used to pad when mode is ``"CONSTANT"``. With other pad
            modes, this argument is ignored.
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

        if mode in (
            PadModes.MINIMUM,
            PadModes.MAXIMUM,
            PadModes.MEAN,
            PadModes.MEDIAN,
        ):
            used_mode = PadModes.CONSTANT
        else:
            used_mode = mode
            # per_channel result is same as default result, so just ignore it
            per_channel = False

        if (
            self.number_of_channel_dimensions == 0 or
            self.channel_shape == (1, )
        ):
            # Zero or one channels, so can ignore the per_channel logic
            per_channel = False

        new_affine, full_pad_width = self._prepare_pad_width(pad_width)

        if not per_channel:
            # no padding for channel dims
            full_pad_width.extend([[0, 0]] * self.number_of_channel_dimensions)

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
            out_spatial_shape = [
                s + p1 + p2
                for s, (p1, p2) in zip(self.spatial_shape, full_pad_width)
            ]
            # preallocate output array
            new_array = np.zeros([*out_spatial_shape, *self.channel_shape])
            for cind in itertools.product(
                *[range(n) for n in self.channel_shape]
            ):
                indexer = (slice(None), slice(None), slice(None), *cind)
                new_array[indexer] = pad_array(
                    self.array[indexer],
                    constant_value
                )
        else:
            new_array = pad_array(self.array, constant_value)

        return self.__class__(
            array=new_array,
            affine=new_affine,
            coordinate_system=self.coordinate_system,
            frame_of_reference_uid=self.frame_of_reference_uid,
            channels=self._channels,
        )


class VolumeToVolumeTransformer:

    """

    Class for transforming voxel indices between two volumes.

    """

    def __init__(
        self,
        volume_from: Volume | VolumeGeometry,
        volume_to: Volume | VolumeGeometry,
        round_output: bool = False,
        check_bounds: bool = False,
    ):
        """Construct transformation object.

        The resulting object will map volume indices of the "from" volume to
        volume indices of the "to" volume.

        Parameters
        ----------
        volume_from: Union[highdicom.Volume, highdicom.VolumeGeometry]
            Volume to which input volume indices refer.
        volume_to: Union[highdicom.Volume, highdicom.VolumeGeometry]
            Volume to which output volume indices refer.
        round_output: bool, optional
            Whether to round the output to the nearest integer (if ``True``) or
            return with sub-voxel accuracy as floats (if ``False``).
        check_bounds: bool, optional
            Whether to perform a bounds check before returning the output
            indices. Note there is no bounds check on the input indices.

        """  # noqa: E501
        self._affine = volume_to.inverse_affine @ volume_from.affine
        self._output_shape = volume_to.spatial_shape
        self._round_output = round_output
        self._check_bounds = check_bounds

    @property
    def affine(self) -> np.ndarray:
        """numpy.ndarray: 4x4 affine transformation matrix"""
        return self._affine.copy()

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """Transform volume indices between two volumes.

        Parameters
        ----------
        indices: numpy.ndarray
            Array of voxel indices in the "from" volume. Array of integer or
            floating-point values with shape ``(n, 3)``, where *n* is the
            number of coordinates. The order of the three indices corresponds
            to the three spatial dimensions volume in that order. Point ``(0,
            0, 0)`` refers to the center of the voxel at index ``(0, 0, 0)`` in
            the array.

        Returns
        -------
        numpy.ndarray
            Array of indices in the output volume that spatially correspond to
            those in the indices in the input array. This will have dtype an
            integer datatype if ``round_output`` is ``True`` and a floating
            point datatype otherwise. The output datatype will be matched to
            the input datatype if possible, otherwise either ``np.int64`` or
            ``np.float64`` is used.

        Raises
        ------
        ValueError
            If ``check_bounds`` is ``True`` and the output indices would
            otherwise contain invalid indices for the "to" volume.

        """
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError(
                'Argument "indices" must be a two-dimensional array '
                'with shape [n, 3].'
            )
        input_is_int = indices.dtype.kind == 'i'
        augmented_input = np.vstack(
            [
                indices.T,
                np.ones((indices.shape[0], ), dtype=indices.dtype),
            ]
        )
        augmented_output = np.dot(self._affine, augmented_input)
        output_indices = augmented_output[:3, :].T

        if self._round_output:
            output_dtype = indices.dtype if input_is_int else np.int64
            output_indices = np.around(output_indices).astype(output_dtype)
        else:
            if not input_is_int:
                output_indices = output_indices.astype(indices.dtype)

        if self._check_bounds:
            bounds_fail = False
            min_indices = np.min(output_indices, axis=1)
            max_indices = np.max(output_indices, axis=1)

            for shape, min_ind, max_ind in zip(
                self._output_shape,
                min_indices,
                max_indices,
            ):
                if min_ind < -0.5:
                    bounds_fail = True
                    break
                if max_ind > shape - 0.5:
                    bounds_fail = True
                    break

            if bounds_fail:
                raise ValueError("Bounds check failed.")

        return output_indices
