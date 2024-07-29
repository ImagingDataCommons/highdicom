from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import List, Optional, Sequence, Union, Tuple

import numpy as np

from highdicom._module_utils import is_multiframe_image
from highdicom.enum import (
    CoordinateSystemNames,
    PatientOrientationValuesBiped,
)
from highdicom.spatial import (
    _create_affine_transformation_matrix,
    _is_matrix_orthogonal,
    _normalize_patient_orientation,
    _transform_affine_matrix,
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

# TODO add segmentation get_volume
# TODO add basic arithmetric operations
# TODO add normalization
# TODO add padding
# TODO add pixel value transformations


class Volume:

    """Class representing a 3D array of regularly-spaced frames in 3D space.

    This class combines a 3D NumPy array with an affine matrix describing the
    location of the voxels in the frame of reference coordinate space. A
    Volume is not a DICOM object itself, but represents a volume that may
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

    def __init__(
        self,
        array: np.ndarray,
        affine: np.ndarray,
        frame_of_reference_uid: Optional[str] = None,
        source_sop_instance_uids: Optional[Sequence[str]] = None,
        source_frame_numbers: Optional[Sequence[int]] = None,
        source_frame_dimension: int = 0,
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
        source_sop_instance_uids: Optional[Sequence[str]], optional
            SOP instance UIDs corresponding to each slice (stacked down
            dimension 0) of the implied volume. This is relevant if and only if
            the volume is formed from a series of single frame DICOM images.
        source_frame_numbers: Optional[Sequence[int]], optional
            Frame numbers of the source image (if any) corresponding to each
            slice (stacked down dimension 0). This is relevant if and only if
            the volume is formed from a set of frames of a single multiframe
            DICOM image.
        source_frame_dimension: int
            Dimension (as a zero-based dimension index) down which source
            frames were stacked to form the volume. Only applicable if
            ``source_sop_instance_uids`` or ``source_frame_numbers`` is
            provided, otherwise ignored.

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

        if source_frame_dimension not in (0, 1, 2):
            raise ValueError(
                f'Argument "source_frame_dimension" must have value 0, 1, or 2.'
            )

        if source_frame_numbers is not None:
            if any(not isinstance(f, int) for f in source_frame_numbers):
                raise TypeError(
                    "Argument 'source_frame_numbers' should be a sequence of ints."
                )
            if any(f < 1 for f in source_frame_numbers):
                raise ValueError(
                    "Argument 'source_frame_numbers' should contain only "
                    "(strictly) positive integers."
                )
            if len(source_frame_numbers) != self._array.shape[source_frame_dimension]:
                raise ValueError(
                    "Length of 'source_frame_numbers' should match size "
                    "of 'array' along the axis given by 'source_frame_dimension'."
                )
            self._source_frame_numbers = list(source_frame_numbers)
        else:
            self._source_frame_numbers = None
        if source_sop_instance_uids is not None:
            if any(not isinstance(u, str) for u in source_sop_instance_uids):
                raise TypeError(
                    "Argument 'source_sop_instance_uids' should be a sequence of "
                    "str."
                )
            if (
                    len(source_sop_instance_uids) !=
                    self._array.shape[source_frame_dimension]
            ):
                raise ValueError(
                    "Length of 'source_sop_instance_uids' should match size "
                    "of 'array' along the axis given by 'source_frame_dimension'."
                )
            self._source_sop_instance_uids = list(source_sop_instance_uids)
        else:
            self._source_sop_instance_uids = None

        if source_frame_numbers is not None or source_sop_instance_uids is not None:
            self._source_frame_dimension = source_frame_dimension
        else:
            self._source_frame_dimension = None

    @classmethod
    def from_image_series(
        cls,
        series_datasets: Sequence[Dataset],
    ) -> "Volume":
        """Create volume from a series of single frame images.

        Parameters
        ----------
        series_datasets: Sequence[pydicom.Dataset]
            Series of single frame datasets. There is no requirement on the
            sorting of the datasets.

        Returns
        -------
        Volume:
            Volume created from the series.

        """
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
        sorted_source_sop_instance_uids = [
            ds.SOPInstanceUID for ds in series_datasets
        ]

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
        )

        # TODO apply color, modality and VOI lookup
        array = np.stack([ds.pixel_array for ds in series_datasets])

        return cls(
            affine=affine,
            array=array,
            frame_of_reference_uid=frame_of_reference_uid,
            source_sop_instance_uids=sorted_source_sop_instance_uids,
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
        sorted_source_frame_numbers = [f + 1 for f in sort_index]

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
            source_frame_numbers=sorted_source_frame_numbers,
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
        source_sop_instance_uids: Optional[Sequence[str]] = None,
        source_frame_numbers: Optional[Sequence[int]] = None,
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
        source_sop_instance_uids: Union[Sequence[str], None], optional
            Ordered SOP Instance UIDs of each frame, if known, in the situation
            that the volume is formed from a sequence of individual DICOM
            instances, stacked down the first axis (index 0)..
        source_frame_numbers: Union[Sequence[int], None], optional
            Ordered frame numbers of each frame of the source image, in the
            situation that the volume is formed from a sequence of frames of
            one multi-frame DICOM image, stacked down the first axis (index
            0).

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
        )
        return cls(
            affine=affine,
            array=array,
            frame_of_reference_uid=frame_of_reference_uid,
            source_sop_instance_uids=source_sop_instance_uids,
            source_frame_numbers=source_frame_numbers,
        )

    @classmethod
    def from_components(
        cls,
        array: np.ndarray,
        position: Sequence[float],
        direction: Sequence[float],
        spacing: Sequence[float],
        frame_of_reference_uid: Optional[str] = None,
        source_sop_instance_uids: Optional[Sequence[str]] = None,
        source_frame_numbers: Optional[Sequence[int]] = None,
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
        source_sop_instance_uids: Union[Sequence[str], None], optional
            Ordered SOP Instance UIDs of each frame, if known, in the situation
            that the volume is formed from a sequence of individual DICOM
            instances, stacked down the first axis (index 0).
        source_frame_numbers: Union[Sequence[int], None], optional
            Ordered frame numbers of each frame of the source image, in the
            situation that the volume is formed from a sequence of frames of
            one multi-frame DICOM image, stacked down the first axis (index 0).

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
            source_sop_instance_uids=source_sop_instance_uids,
            source_frame_numbers=source_frame_numbers,
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
            dimension of the volume given by ``source_frame_dimension``.

        """
        if self._source_frame_numbers is None:
            raise RuntimeError(
                "Frame information is not present."
            )
        return self._source_frame_numbers.index(frame_number)

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
            dimension of the volume given by ``source_frame_dimension``.

        """
        if self._source_sop_instance_uids is None:
            raise RuntimeError(
                "SOP Instance UID information is not present."
            )
        return self._source_sop_instance_uids.index(sop_instance_uid)

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
    def source_frame_dimension(self) -> Optional[int]:
        """Optional[int]: Dimension along which source frames were stacked.

        Will return either 0, 1, or 2 when the volume was created from a source
        image or image series. Will return ``None`` if the volume was not
        created from a source image or image series.

        """
        return self._source_frame_dimension

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

    @property
    def source_sop_instance_uids(self) -> Union[List[str], None]:
        # TODO account for rotated arrays
        """Union[List[str], None]: SOP Instance UID at each index."""
        if self._source_sop_instance_uids is not None:
            return self._source_sop_instance_uids.copy()

    @property
    def source_frame_numbers(self) -> Union[List[int], None]:
        # TODO account for rotated arrays
        """Union[List[int], None]:

        Frame number within the source image at each index down the first
        dimension.

        """
        if self._source_frame_numbers is not None:
            return self._source_frame_numbers.copy()

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

        return self.__class__(
            array=new_array,
            affine=self.affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
            source_sop_instance_uids=deepcopy(self.source_sop_instance_uids),
            source_frame_numbers=deepcopy(self.source_frame_numbers),
            source_frame_dimension=self.source_frame_dimension or 0,
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
            source_sop_instance_uids=deepcopy(self.source_sop_instance_uids),
            source_frame_numbers=deepcopy(self.source_frame_numbers),
        )

    def __getitem__(
        self,
        index: Union[int, slice, Tuple[Union[int, slice]]],
    ) -> "Volume":
        """Get a sub-volume of this volume as a new volume.

        Parameters
        ----------
        index: Union[int, slice, Tuple[Union[int, slice]]]

        Returns
        -------
        highdicom.Volume:

        """
        if isinstance(index, int):
            # Change the index to a slice of length one so that all dimensions
            # are retained in the output array. Also make into a tuple of
            # length 1 to standardize format
            tuple_index = (slice(index, index + 1), )
        elif isinstance(index, slice):
            # Make into a tuple of length one to standardize the format
            tuple_index = (index, )
        elif isinstance(index, tuple):
            index_list = []
            for item in index:
                if isinstance(item, int):
                    # Change the index to a slice of length one so that all
                    # dimensions are retained in the output array.
                    item = slice(item, item + 1)
                    index_list.append(item)
                elif isinstance(item, slice):
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

        new_array = self._array[tuple_index]

        new_sop_instance_uids = None
        new_frame_numbers = None
        new_vectors = []
        origin_indices = []

        for d in range(0, 3):
            # The index item along this dimension
            if len(tuple_index) > d:
                index_item = tuple_index[d]
                first, _, step = index_item.indices(self.shape[d])
            else:
                index_item = None
                first = 0
                step = 1

            new_vectors.append(self._affine[:3, d] * step)
            origin_indices.append(first)

            if self.source_frame_dimension is not None:
                if d == self.source_frame_dimension:
                    if index_item is not None:
                        # Need to index the source frame lists along this
                        # dimension
                        if self._source_sop_instance_uids is not None:
                            new_sop_instance_uids = (
                                self._source_sop_instance_uids[
                                    index_item
                                ]
                            )
                        if self._source_frame_numbers is not None:
                            new_frame_numbers = self._source_frame_numbers[
                                index_item
                            ]
                    else:
                        # Not indexing along this dimension so the lists are
                        # unchanged
                        new_sop_instance_uids = deepcopy(
                            self.source_sop_instance_uids
                        )
                        new_frame_numbers = deepcopy(
                            self.source_frame_numbers
                        )

        origin_index_arr = np.array([origin_indices])
        new_origin_arr = self.map_indices_to_reference(origin_index_arr).T

        new_rotation = np.column_stack(new_vectors)
        new_affine = np.row_stack(
            [
                np.column_stack([new_rotation, new_origin_arr]),
                np.array([0., 0., 0., 1.0]),
            ]
        )

        return self.__class__(
            array=new_array,
            affine=new_affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
            source_sop_instance_uids=new_sop_instance_uids,
            source_frame_numbers=new_frame_numbers,
            source_frame_dimension=self.source_frame_dimension or 0,
        )

    def permute(self, indices: Sequence[int]) -> 'Volume':
        # TODO add tests for this
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
        if len(indices) != 3 or set(indices) != {0, 1, 2}:
            raise ValueError(
                f'Argument "indices" must consist of the values 0, 1, and 2 '
                'in some order.'
            )

        if self._array.ndim == 3:
            new_array = np.transpose(self._array, indices)
        else:
            new_array = np.transpose(self._array, [*indices, 3])

        new_affine = _transform_affine_matrix(
            affine=self._affine,
            shape=self.spatial_shape,
            permute_indices=indices,
        )

        if self.source_frame_dimension is None:
            new_source_frame_dimension = 0
        else:
            new_source_frame_dimension = indices.index(
                self.source_frame_dimension
            )

        return self.__class__(
            array=new_array,
            affine=new_affine,
            frame_of_reference_uid=self.frame_of_reference_uid,
            source_sop_instance_uids=deepcopy(self.source_sop_instance_uids),
            source_frame_numbers=deepcopy(self.source_frame_numbers),
            source_frame_dimension=new_source_frame_dimension,
        )

    def flip(self, axis: Union[int, Sequence[int]]) -> 'Volume':
        """Flip the spatial axes of the array.

        Note that this flips the array and updates the affine to reflect the
        flip.

        Parameters
        ----------
        axis: Union[int, Sequence[int]]
            Axis or list of axes that should be flipped. These should include
            only the spatial axes (0, 1, and/or 2).

        Returns
        -------
        highdicom.Volume:
            New volume with spatial axes flipped as requested.

        """
        if isinstance(axis, int):
            axis = [axis]

        if len(axis) > 3 or len(set(axis) - {0, 1, 2}) > 0:
            raise ValueError(
                'Arugment "axis" must contain only values 0, 1, and/or 2.'
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

    def to_patient_orientation(
        self,
        patient_orientation: Union[
            str,
            Sequence[Union[str, PatientOrientationValuesBiped]],
        ],
    ) -> 'Volume':
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

        return result.permute(permute_indices)


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
