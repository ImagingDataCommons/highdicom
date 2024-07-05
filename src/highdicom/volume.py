from copy import deepcopy
from typing import List, Optional, Sequence, Union, Tuple

import numpy as np

from highdicom._module_utils import is_multiframe_image
from highdicom.enum import (
    CoordinateSystemNames,
    PixelIndexDirections,
)
from highdicom.spatial import (
    _create_affine_transformation_matrix,
    _is_matrix_orthogonal,
    get_image_coordinate_system,
    get_plane_sort_index,
    get_regular_slice_spacing,
    get_series_slice_spacing,
    sort_datasets,
)
from highdicom.content import PlanePositionSequence

from pydicom import Dataset


class VolumeArray:

    """Class representing a 3D array of regularly-spaced frames in 3D space.

    This class combines a 3D NumPy array with an affine matrix describing the
    location of the voxels in the frame of reference coordinate space. A
    VolumeArray is not a DICOM object itself, but represents a volume that may
    be extracted from DICOM image, and/or encoded within a DICOM object,
    potentially following any number of processing steps.

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
    # The indexing convention used for all internal representations of the
    # affine matrix.
    _INTERNAL_INDEX_CONVENTION = (
        PixelIndexDirections.I,
        PixelIndexDirections.D,
        PixelIndexDirections.R,
    )

    def __init__(
        self,
        array: np.ndarray,
        affine: np.ndarray,
        frame_of_reference_uid: Optional[str] = None,
        sop_instance_uids: Optional[Sequence[str]] = None,
        frame_numbers: Optional[Sequence[int]] = None,
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
        if array.ndim not in (3, 4):
            raise ValueError(
                "Argument 'array' must be three or four-dimensional."
            )

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

        self._array = array
        self._affine = affine
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
            if len(frame_numbers) != self._array.shape[0]:
                raise ValueError(
                    "Length of 'frame_numbers' should match first dimension "
                    "of 'array'."
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
            if len(sop_instance_uids) != self._array.shape[0]:
                raise ValueError(
                    "Length of 'sop_instance_uids' should match first "
                    "dimension of 'array'."
                )
            self._sop_instance_uids = list(sop_instance_uids)
        else:
            self._sop_instance_uids = None

    @classmethod
    def from_image_series(
        cls,
        series_datasets: Sequence[Dataset],
    ) -> "VolumeArray":
        """Get volume geometry for a series of single frame images.

        Parameters
        ----------
        series_datasets: Sequence[pydicom.Dataset]
            Series of single frame datasets. There is no requirement on the
            sorting of the datasets.

        Returns
        -------
        VolumeArray:
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

        affine = _create_affine_transformation_matrix(
            image_position=ds.ImagePositionPatient,
            image_orientation=ds.ImageOrientationPatient,
            pixel_spacing=ds.PixelSpacing,
            spacing_between_slices=slice_spacing,
            index_convention=cls._INTERNAL_INDEX_CONVENTION,
        )

        # TODO apply color, modality and VOI lookup
        array = np.stack([ds.pixel_array for ds in series_datasets])

        return cls(
            affine=affine,
            array=array,
            frame_of_reference_uid=frame_of_reference_uid,
            sop_instance_uids=sorted_sop_instance_uids,
        )

    @classmethod
    def from_image(
        cls,
        dataset: Dataset,
    ) -> "VolumeArray":
        """Get volume geometry for a multiframe image.

        Parameters
        ----------
        dataset: pydicom.Dataset
            A multi-frame image dataset.

        Returns
        -------
        VolumeArray:
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

        affine = _create_affine_transformation_matrix(
            image_position=sorted_positions[0],
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=slice_spacing,
            index_convention=cls._INTERNAL_INDEX_CONVENTION,
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
            frame_numbers=sorted_frame_numbers,
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
        sop_instance_uids: Optional[Sequence[str]] = None,
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> "VolumeArray":
        """Create a volume geometry from DICOM attributes.

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
        sop_instance_uids: Union[Sequence[str], None], optional
            Ordered SOP Instance UIDs of each frame, if known, in the situation
            that the volume is formed from a sequence of individual DICOM
            instances, stacked down the first axis (index 0)..
        frame_numbers: Union[Sequence[int], None], optional
            Ordered frame numbers of each frame, if known, in the situation
            that the volume is formed from a sequence of frames of one
            multi-frame DICOM image, stacked down the first axis (index 0)..

        """
        affine = _create_affine_transformation_matrix(
            image_position=image_position,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            spacing_between_slices=spacing_between_slices,
            index_convention=cls._INTERNAL_INDEX_CONVENTION,
        )
        return cls(
            affine=affine,
            array=array,
            frame_of_reference_uid=frame_of_reference_uid,
            sop_instance_uids=sop_instance_uids,
            frame_numbers=frame_numbers,
        )

    @classmethod
    def from_components(
        cls,
        array: np.ndarray,
        position: Sequence[float],
        direction: Sequence[float],
        spacing: Sequence[float],
        frame_of_reference_uid: Optional[str] = None,
        sop_instance_uids: Optional[Sequence[str]] = None,
        frame_numbers: Optional[Sequence[int]] = None,
    ) -> "VolumeArray":
        """Construct a VolumeArray from components.

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
            Sequence of three integers giving the shape of the volume array.
        frame_of_reference_uid: Union[str, None], optional
            Frame of reference UID for the frame of reference, if known.
        sop_instance_uids: Union[Sequence[str], None], optional
            Ordered SOP Instance UIDs of each frame, if known, in the situation
            that the volume is formed from a sequence of individual DICOM
            instances, stacked down the first axis (index 0).
        frame_numbers: Union[Sequence[int], None], optional
            Ordered frame numbers of each frame, if known, in the situation
            that the volume is formed from a sequence of frames of one
            multi-frame DICOM image, stacked down the first axis (index 0).

        Returns
        -------
        highdicom.spatial.VolumeArray:
            Volume geometry constructed from the provided components.

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
        affine = np.row_stack(
            [
                np.column_stack([scaled_direction, position]),
                [0.0, 0.0, 0.0, 1.0]
            ]
        )
        return cls(
            array=array,
            affine=affine,
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
                [(self.shape[d] - 1) // 2 for d in range(3)],
                dtype=np.uint32,
            )
        else:
            center = np.array(
                [(self.shape[d] - 1) / 2.0 for d in range(3)]
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
                if indices[:, d].max() > self.shape[d] - 0.5:
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
        if plane_number < 0 or plane_number >= self.shape[0]:
            raise ValueError("Invalid plane number for volume.")
        index = np.array([[plane_number, 0, 0]])
        position = self.map_indices_to_reference(index)[0]

        return PlanePositionSequence(
            CoordinateSystemNames.PATIENT,
            position,
        )

    def get_plane_positions(self) -> List[PlanePositionSequence]:
        """Get plane positions of all planes in the volume.

        Returns
        -------
        List[highdicom.content.PlanePositionSequence]:
            Plane position of the all planes (stacked down axis 0 of the
            volume).

        """
        indices = np.array(
            [
                [p, 0, 0] for p in range(self.shape[0])
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

    def set_array(self, array: np.ndarray) -> None:
        # TODO make this a proper setter and getter
        """Change the voxel array without changing the affine.

        Parameters
        ----------
        array: np.ndarray
            New 3D or 4D array of voxel data. The spatial shape must match the
            existing array, but the presence and number of channels and/or the
            voxel datatype may differ.

        """
        if array.ndim not in (3, 4):
            raise ValueError(
                "Argument 'array' must be a three or four dimensional array."
            )
        if array.shape[:3] != self.spatial_shape:
            raise ValueError(
                "Array must match the spatial shape of the existing array."
            )
        self._array = array

    @property
    def array(self) -> np.ndarray:
        """numpy.ndarray: Volume array (copied)."""
        return self._array.copy()

    @property
    def sop_instance_uids(self) -> Union[List[str], None]:
        """Union[List[str], None]: SOP Instance UID at each index."""
        if self._sop_instance_uids is not None:
            return self._sop_instance_uids.copy()

    @property
    def frame_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]:

        Frame number at each index down the first dimension.

        """
        if self._frame_numbers is not None:
            return self._frame_numbers.copy()

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
        Three values (spacing between slices, spacing spacing between rows,
        spacing between columns).

        """
        dir_mat = self._affine[:3, :3]
        norms = np.sqrt((dir_mat ** 2).sum(axis=0))
        return norms.tolist()

    @property
    def position(self) -> List[float]:
        """List[float]:

        Pixel spacing in millimeter units for the three spatial directions.
        Three values (spacing between slices, spacing spacing between rows,
        spacing between columns).

        """
        return self._affine[:3, 3].tolist()

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

    def with_array(self, array: np.ndarray) -> 'VolumeArray':
        """Get a new volume using a different array.

        The spatial and other metadata will be copied from this volume.
        The original volume will be unaltered.

        Parameters
        ----------
        array: np.ndarray
            New 3D or 4D array of voxel data. The spatial shape must match the
            existing array, but the presence and number of channels and/or the
            voxel datatype may differ.

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
            sop_instance_uids=deepcopy(self.sop_instance_uids),
            frame_numbers=deepcopy(self.frame_numbers),
        )
