"""Tools for working with multiframe DICOM images."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import logging
import sqlite3
from typing import (
    Any,
    Iterable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
    cast,
)
import numpy as np
from pydicom import Dataset
from pydicom.tag import BaseTag
from pydicom.datadict import get_entry, tag_for_keyword
from pydicom.multival import MultiValue

from highdicom._module_utils import is_multiframe_image
from highdicom.base import SOPClass, _check_little_endian
from highdicom.enum import (
    CoordinateSystemNames,
)
from highdicom.seg.enum import SpatialLocationsPreservedValues
from highdicom.spatial import (
    get_image_coordinate_system,
    get_volume_positions,
)
from highdicom.uid import UID as hd_UID
from highdicom.utils import (
    iter_tiled_full_frame_data,
)
from highdicom.volume import VolumeGeometry


# Dictionary mapping DCM VRs to appropriate SQLite types
_DCM_SQL_TYPE_MAP = {
        'CS': 'VARCHAR',
        'DS': 'REAL',
        'FD': 'REAL',
        'FL': 'REAL',
        'IS': 'INTEGER',
        'LO': 'TEXT',
        'LT': 'TEXT',
        'PN': 'TEXT',
        'SH': 'TEXT',
        'SL': 'INTEGER',
        'SS': 'INTEGER',
        'ST': 'TEXT',
        'UI': 'TEXT',
        'UL': 'INTEGER',
        'UR': 'TEXT',
        'US or SS': 'INTEGER',
        'US': 'INTEGER',
        'UT': 'TEXT',
    }
_NO_FRAME_REF_VALUE = -1


logger = logging.getLogger(__name__)


class MultiFrameImage(SOPClass):

    """Database manager for frame information in a multiframe image."""

    _coordinate_system: CoordinateSystemNames
    _is_tiled_full: bool
    _single_source_frame_per_frame: bool
    _dim_ind_pointers: List[BaseTag]
    _dim_ind_col_names: Dict[int, str]
    _locations_preserved: Optional[SpatialLocationsPreservedValues]
    _db_con: sqlite3.Connection
    _volume_geometry: Optional[VolumeGeometry]

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> 'MultiFrameImage':
        """Create a MultiFrameImage from an existing pydicom Dataset.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Dataset of a multi-frame image.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        _check_little_endian(dataset)

        # Checks on integrity of input dataset
        if not is_multiframe_image(dataset):
            raise ValueError('Dataset is not a multiframe image.')
        if copy:
            im = deepcopy(dataset)
        else:
            im = dataset
        im.__class__ = cls
        im = cast(cls, im)

        im._build_luts()
        return im

    def _build_luts(self) -> None:
        """Build lookup tables for efficient querying.

        Two lookup tables are currently constructed. The first maps the
        SOPInstanceUIDs of all datasets referenced in the image to a
        tuple containing the StudyInstanceUID, SeriesInstanceUID and
        SOPInstanceUID.

        The second look-up table contains information about each frame of the
        segmentation, including the segment it contains, the instance and frame
        from which it was derived (if these are unique), and its dimension
        index values.

        """
        self._coordinate_system = get_image_coordinate_system(
            self
        )
        referenced_uids = self._get_ref_instance_uids()
        all_referenced_sops = {uids[2] for uids in referenced_uids}

        self._is_tiled_full = (
            hasattr(self, 'DimensionOrganizationType') and
            self.DimensionOrganizationType == 'TILED_FULL'
        )

        self._dim_ind_pointers = [
            dim_ind.DimensionIndexPointer
            for dim_ind in self.DimensionIndexSequence
        ]
        func_grp_pointers = {}
        for dim_ind in self.DimensionIndexSequence:
            ptr = dim_ind.DimensionIndexPointer
            if ptr in self._dim_ind_pointers:
                grp_ptr = getattr(dim_ind, "FunctionalGroupPointer", None)
                func_grp_pointers[ptr] = grp_ptr

        # We mav want to gather additional information that is not one of the
        # indices
        extra_collection_pointers = []
        extra_collection_func_pointers = {}
        slice_spacing_hint = None
        image_position_tag = tag_for_keyword('ImagePositionPatient')
        shared_pixel_spacing: Optional[List[float]] = None
        if self._coordinate_system == CoordinateSystemNames.PATIENT:
            plane_pos_seq_tag = tag_for_keyword('PlanePositionSequence')
            # Include the image position if it is not an index
            if image_position_tag not in self._dim_ind_pointers:
                extra_collection_pointers.append(image_position_tag)
                extra_collection_func_pointers[
                    image_position_tag
                ] = plane_pos_seq_tag

            if hasattr(self, 'SharedFunctionalGroupsSequence'):
                sfgs = self.SharedFunctionalGroupsSequence[0]
                if hasattr(sfgs, 'PixelMeasuresSequence'):
                    measures = sfgs.PixelMeasuresSequence[0]
                    slice_spacing_hint = measures.get('SpacingBetweenSlices')
                    shared_pixel_spacing = measures.get('PixelSpacing')
            if slice_spacing_hint is None or shared_pixel_spacing is None:
                # Get the orientation of the first frame, and in the later loop
                # check whether it is shared.
                if hasattr(self, 'PerFrameFunctionalGroupsSequence'):
                    pfg1 = self.PerFrameFunctionalGroupsSequence[0]
                    if hasattr(pfg1, 'PixelMeasuresSequence'):
                        slice_spacing_hint = pfg1.PixelMeasuresSequence[0].get(
                            'SpacingBetweenSlices'
                        )
                        shared_pixel_spacing = pfg1.get('PixelSpacing')

        dim_ind_positions = {
            dim_ind.DimensionIndexPointer: i
            for i, dim_ind in enumerate(self.DimensionIndexSequence)
        }
        dim_indices: Dict[int, List[int]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }
        dim_values: Dict[int, List[Any]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }

        extra_collection_values: Dict[int, List[Any]] = {
            ptr: [] for ptr in extra_collection_pointers
        }

        # Get the shared orientation
        shared_image_orientation: Optional[List[float]] = None
        if hasattr(self, 'ImageOrientationSlide'):
            shared_image_orientation = self.ImageOrientationSlide
        if hasattr(self, 'SharedFunctionalGroupsSequence'):
            sfgs = self.SharedFunctionalGroupsSequence[0]
            if hasattr(sfgs, 'PlaneOrientationSequence'):
                shared_image_orientation = (
                    sfgs.PlaneOrientationSequence[0].ImageOrientationPatient
                )
        if shared_image_orientation is None:
            # Get the orientation of the first frame, and in the later loop
            # check whether it is shared.
            if hasattr(self, 'PerFrameFunctionalGroupsSequence'):
                pfg1 = self.PerFrameFunctionalGroupsSequence[0]
                if hasattr(pfg1, 'PlaneOrientationSequence'):
                    shared_image_orientation = (
                        pfg1
                        .PlaneOrientationSequence[0]
                        .ImageOrientationPatient
                    )

        self._single_source_frame_per_frame = True

        if self._is_tiled_full:
            # With TILED_FULL, there is no PerFrameFunctionalGroupsSequence,
            # so we have to deduce the per-frame information
            row_tag = tag_for_keyword('RowPositionInTotalImagePixelMatrix')
            col_tag = tag_for_keyword('ColumnPositionInTotalImagePixelMatrix')
            x_tag = tag_for_keyword('XOffsetInSlideCoordinateSystem')
            y_tag = tag_for_keyword('YOffsetInSlideCoordinateSystem')
            z_tag = tag_for_keyword('ZOffsetInSlideCoordinateSystem')
            tiled_full_dim_indices = {row_tag, col_tag}
            if len(tiled_full_dim_indices - set(dim_indices.keys())) > 0:
                raise RuntimeError(
                    'Expected images with '
                    '"DimensionOrganizationType" of "TILED_FULL" '
                    'to have the following dimension index pointers: '
                    'RowPositionInTotalImagePixelMatrix, '
                    'ColumnPositionInTotalImagePixelMatrix.'
                )
            self._single_source_frame_per_frame = False
            (
                channel_numbers,
                _,
                dim_values[col_tag],
                dim_values[row_tag],
                dim_values[x_tag],
                dim_values[y_tag],
                dim_values[z_tag],
            ) = zip(*iter_tiled_full_frame_data(self))

            if hasattr(self, 'SegmentSequence'):
                segment_tag = tag_for_keyword('ReferencedSegmentNumber')
                dim_values[segment_tag] = channel_numbers
            elif hasattr(self, 'OpticalPathSequence'):
                op_tag = tag_for_keyword('OpticalPathIdentifier')
                dim_values[op_tag] = channel_numbers

            # Create indices for each of the dimensions
            for ptr, vals in dim_values.items():
                _, indices = np.unique(vals, return_inverse=True)
                dim_indices[ptr] = (indices + 1).tolist()

            # There is no way to deduce whether the spatial locations are
            # preserved in the tiled full case
            self._locations_preserved = None

            referenced_instances = None
            referenced_frames = None
        else:
            referenced_instances: Optional[List[str]] = []
            referenced_frames: Optional[List[int]] = []

            # Create a list of source images and check for spatial locations
            # preserved
            locations_list_type = List[
                Optional[SpatialLocationsPreservedValues]
            ]
            locations_preserved: locations_list_type = []

            for frame_item in self.PerFrameFunctionalGroupsSequence:
                # Get dimension indices for this frame
                content_seq = frame_item.FrameContentSequence[0]
                indices = content_seq.DimensionIndexValues
                if not isinstance(indices, (MultiValue, list)):
                    # In case there is a single dimension index
                    indices = [indices]
                if len(indices) != len(self._dim_ind_pointers):
                    raise RuntimeError(
                        'Unexpected mismatch between dimension index values in '
                        'per-frames functional groups sequence and items in '
                        'the dimension index sequence.'
                    )
                for ptr in self._dim_ind_pointers:
                    dim_indices[ptr].append(indices[dim_ind_positions[ptr]])
                    grp_ptr = func_grp_pointers[ptr]
                    if grp_ptr is not None:
                        dim_val = frame_item[grp_ptr][0][ptr].value
                    else:
                        dim_val = frame_item[ptr].value
                    dim_values[ptr].append(dim_val)
                for ptr in extra_collection_pointers:
                    grp_ptr = extra_collection_func_pointers[ptr]
                    if grp_ptr is not None:
                        dim_val = frame_item[grp_ptr][0][ptr].value
                    else:
                        dim_val = frame_item[ptr].value
                    extra_collection_values[ptr].append(dim_val)

                frame_source_instances = []
                frame_source_frames = []
                for der_im in getattr(
                    frame_item,
                    'DerivationImageSequence',
                    []
                ):
                    for src_im in getattr(
                        der_im,
                        'SourceImageSequence',
                        []
                    ):
                        frame_source_instances.append(
                            src_im.ReferencedSOPInstanceUID
                        )
                        if hasattr(src_im, 'SpatialLocationsPreserved'):
                            locations_preserved.append(
                                SpatialLocationsPreservedValues(
                                    src_im.SpatialLocationsPreserved
                                )
                            )
                        else:
                            locations_preserved.append(
                                None
                            )

                        if hasattr(src_im, 'ReferencedFrameNumber'):
                            if isinstance(
                                src_im.ReferencedFrameNumber,
                                MultiValue
                            ):
                                frame_source_frames.extend(
                                    [
                                        int(f)
                                        for f in src_im.ReferencedFrameNumber
                                    ]
                                )
                            else:
                                frame_source_frames.append(
                                    int(src_im.ReferencedFrameNumber)
                                )
                        else:
                            frame_source_frames.append(_NO_FRAME_REF_VALUE)

                if (
                    len(set(frame_source_instances)) != 1 or
                    len(set(frame_source_frames)) != 1
                ):
                    self._single_source_frame_per_frame = False
                else:
                    ref_instance_uid = frame_source_instances[0]
                    if ref_instance_uid not in all_referenced_sops:
                        raise AttributeError(
                            f'SOP instance {ref_instance_uid} referenced in '
                            'the source image sequence is not included in the '
                            'Referenced Series Sequence or Studies Containing '
                            'Other Referenced Instances Sequence. This is an '
                            'error with the integrity of the '
                            'object.'
                        )
                    referenced_instances.append(ref_instance_uid)
                    referenced_frames.append(frame_source_frames[0])

                # Check that this doesn't have a conflicting orientation
                if shared_image_orientation is not None:
                    if hasattr(frame_item, 'PlaneOrientationSequence'):
                        iop = (
                            frame_item
                            .PlaneOrientationSequence[0]
                            .ImageOrientationPatient
                        )
                        if iop != shared_image_orientation:
                            shared_image_orientation = None

            # Summarise
            if any(
                isinstance(v, SpatialLocationsPreservedValues) and
                v == SpatialLocationsPreservedValues.NO
                for v in locations_preserved
            ):

                self._locations_preserved = SpatialLocationsPreservedValues.NO
            elif all(
                isinstance(v, SpatialLocationsPreservedValues) and
                v == SpatialLocationsPreservedValues.YES
                for v in locations_preserved
            ):
                self._locations_preserved = SpatialLocationsPreservedValues.YES
            else:
                self._locations_preserved = None

            if not self._single_source_frame_per_frame:
                referenced_instances = None
                referenced_frames = None

        self._db_con = sqlite3.connect(":memory:")

        self._create_ref_instance_table(referenced_uids)

        # Construct the columns and values to put into a frame look-up table
        # table within sqlite. There will be one row per frame in the
        # image
        col_defs = []  # SQL column definitions
        col_data = []  # lists of column data

        # Frame number column
        col_defs.append('FrameNumber INTEGER PRIMARY KEY')
        col_data.append(list(range(1, self.NumberOfFrames + 1)))

        self._dim_ind_col_names = {}
        for i, t in enumerate(dim_indices.keys()):
            vr, vm_str, _, _, kw = get_entry(t)
            if kw == '':
                kw = f'UnknownDimensionIndex{i}'
            ind_col_name = kw + '_DimensionIndexValues'
            self._dim_ind_col_names[t] = ind_col_name

            # Add column for dimension index
            col_defs.append(f'{ind_col_name} INTEGER NOT NULL')
            col_data.append(dim_indices[t])

            # Add column for dimension value
            # For this to be possible, must have a fixed VM
            # and a VR that we can map to a sqlite type
            # Otherwise, we just omit the data from the db
            if kw == 'ReferencedSegmentNumber':
                # Special case since this tag technically has VM 1-n
                vm = 1
            else:
                try:
                    vm = int(vm_str)
                except ValueError:
                    continue
            try:
                sql_type = _DCM_SQL_TYPE_MAP[vr]
            except KeyError:
                continue

            if vm > 1:
                for d in range(vm):
                    data = [el[d] for el in dim_values[t]]
                    col_defs.append(f'{kw}_{d} {sql_type} NOT NULL')
                    col_data.append(data)
            else:
                # Single column
                col_defs.append(f'{kw} {sql_type} NOT NULL')
                col_data.append(dim_values[t])

        for i, t in enumerate(extra_collection_pointers):
            vr, vm_str, _, _, kw = get_entry(t)

            # Add column for dimension value
            # For this to be possible, must have a fixed VM
            # and a VR that we can map to a sqlite type
            # Otherwise, we just omit the data from the db
            vm = int(vm_str)
            sql_type = _DCM_SQL_TYPE_MAP[vr]

            if vm > 1:
                for d in range(vm):
                    data = [el[d] for el in extra_collection_values[t]]
                    col_defs.append(f'{kw}_{d} {sql_type} NOT NULL')
                    col_data.append(data)
            else:
                # Single column
                col_defs.append(f'{kw} {sql_type} NOT NULL')
                col_data.append(dim_values[t])

        # Volume related information
        self._volume_geometry = None
        if (
            self._coordinate_system == CoordinateSystemNames.PATIENT
            and shared_image_orientation is not None
        ):
            if shared_image_orientation is not None:
                if image_position_tag in self._dim_ind_pointers:
                    image_positions = dim_values[image_position_tag]
                else:
                    image_positions = extra_collection_values[
                        image_position_tag
                    ]
                volume_spacing, volume_positions = get_volume_positions(
                    image_positions=image_positions,
                    image_orientation=shared_image_orientation,
                    allow_missing=True,
                    allow_duplicates=True,
                    spacing_hint=slice_spacing_hint,
                )
                if volume_positions is not None:
                    origin_slice_index = volume_positions.index(0)
                    number_of_slices = max(volume_positions) + 1
                    self._volume_geometry = VolumeGeometry.from_attributes(
                        image_position=image_positions[origin_slice_index],
                        image_orientation=shared_image_orientation,
                        rows=self.Rows,
                        columns=self.Columns,
                        pixel_spacing=shared_pixel_spacing,
                        number_of_frames=number_of_slices,
                        spacing_between_slices=volume_spacing,
                    )
                    col_defs.append('VolumePosition INTEGER NOT NULL')
                    col_data.append(volume_positions)

        # Columns related to source frames, if they are usable for indexing
        if (referenced_frames is None) != (referenced_instances is None):
            raise TypeError(
                "'referenced_frames' and 'referenced_instances' should be "
                "provided together or not at all."
            )
        if referenced_instances is not None:
            col_defs.append('ReferencedFrameNumber INTEGER')
            col_defs.append('ReferencedSOPInstanceUID VARCHAR NOT NULL')
            col_defs.append(
                'FOREIGN KEY(ReferencedSOPInstanceUID) '
                'REFERENCES InstanceUIDs(SOPInstanceUID)'
            )
            col_data += [
                referenced_frames,
                referenced_instances,
            ]

        # Build LUT from columns
        all_defs = ", ".join(col_defs)
        cmd = f'CREATE TABLE FrameLUT({all_defs})'
        placeholders = ', '.join(['?'] * len(col_data))
        with self._db_con:
            self._db_con.execute(cmd)
            self._db_con.executemany(
                f'INSERT INTO FrameLUT VALUES({placeholders})',
                zip(*col_data),
            )

    def _get_ref_instance_uids(self) -> List[Tuple[str, str, str]]:
        """List all instances referenced in the image.

        Returns
        -------
        List[Tuple[str, str, str]]
            List of all instances referenced in the image in the format
            (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID).

        """
        instance_data = []
        if hasattr(self, 'ReferencedSeriesSequence'):
            for ref_series in self.ReferencedSeriesSequence:
                for ref_ins in ref_series.ReferencedInstanceSequence:
                    instance_data.append(
                        (
                            self.StudyInstanceUID,
                            ref_series.SeriesInstanceUID,
                            ref_ins.ReferencedSOPInstanceUID
                        )
                    )
        other_studies_kw = 'StudiesContainingOtherReferencedInstancesSequence'
        if hasattr(self, other_studies_kw):
            for ref_study in getattr(self, other_studies_kw):
                for ref_series in ref_study.ReferencedSeriesSequence:
                    for ref_ins in ref_series.ReferencedInstanceSequence:
                        instance_data.append(
                            (
                                ref_study.StudyInstanceUID,
                                ref_series.SeriesInstanceUID,
                                ref_ins.ReferencedSOPInstanceUID,
                            )
                        )

        # There shouldn't be duplicates here, but there's no explicit rule
        # preventing it.
        # Since dictionary ordering is preserved, this trick deduplicates
        # the list without changing the order
        unique_instance_data = list(dict.fromkeys(instance_data))
        if len(unique_instance_data) != len(instance_data):
            counts = Counter(instance_data)
            duplicate_sop_uids = [
                f"'{key[2]}'" for key, value in counts.items() if value > 1
            ]
            display_str = ', '.join(duplicate_sop_uids)
            logger.warning(
                'Duplicate entries found in the ReferencedSeriesSequence. '
                f"SOP Instance UID: '{self.SOPInstanceUID}', "
                f'duplicated referenced SOP Instance UID items: {display_str}.'
            )

        return unique_instance_data

    def _check_indexing_with_source_frames(
        self,
        ignore_spatial_locations: bool = False
    ) -> None:
        """Check if indexing by source frames is possible.

        Raise exceptions with useful messages otherwise.

        Possible problems include:
            * Spatial locations are not preserved.
            * The dataset does not specify that spatial locations are preserved
              and the user has not asserted that they are.
            * At least one frame in the image lists multiple
              source frames.

        Parameters
        ----------
        ignore_spatial_locations: bool
            Allows the user to ignore whether spatial locations are preserved
            in the frames.

        """
        # Checks that it is possible to index using source frames in this
        # dataset
        if self._is_tiled_full:
            raise RuntimeError(
                'Indexing via source frames is not possible when an '
                'image is stored using the DimensionOrganizationType '
                '"TILED_FULL".'
            )
        elif self._locations_preserved is None:
            if not ignore_spatial_locations:
                raise RuntimeError(
                    'Indexing via source frames is not permissible since this '
                    'image does not specify that spatial locations are '
                    'preserved in the course of deriving the image '
                    'from the source image. If you are confident that spatial '
                    'locations are preserved, or do not require that spatial '
                    'locations are preserved, you may override this behavior '
                    "with the 'ignore_spatial_locations' parameter."
                )
        elif self._locations_preserved == SpatialLocationsPreservedValues.NO:
            if not ignore_spatial_locations:
                raise RuntimeError(
                    'Indexing via source frames is not permissible since this '
                    'image specifies that spatial locations are not preserved '
                    'in the course of deriving the image from the '
                    'source image. If you do not require that spatial '
                    ' locations are preserved you may override this behavior '
                    "with the 'ignore_spatial_locations' parameter."
                )
        if not self._single_source_frame_per_frame:
            raise RuntimeError(
                'Indexing via source frames is not permissible since some '
                'frames in the image specify multiple source frames.'
            )

    @property
    def dimension_index_pointers(self) -> List[BaseTag]:
        """List[pydicom.tag.BaseTag]:
            List of tags used as dimension indices.
        """
        return [BaseTag(t) for t in self._dim_ind_pointers]

    def _create_ref_instance_table(
        self,
        referenced_uids: List[Tuple[str, str, str]],
    ) -> None:
        """Create a table of referenced instances.

        The resulting table (called InstanceUIDs) contains Study, Series and
        SOP instance UIDs for each instance referenced by the image.

        Parameters
        ----------
        referenced_uids: List[Tuple[str, str, str]]
            List of UIDs for each instance referenced in the image.
            Each tuple should be in the format
            (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID).

        """
        with self._db_con:
            self._db_con.execute(
                """
                    CREATE TABLE InstanceUIDs(
                        StudyInstanceUID VARCHAR NOT NULL,
                        SeriesInstanceUID VARCHAR NOT NULL,
                        SOPInstanceUID VARCHAR PRIMARY KEY
                    )
                """
            )
            self._db_con.executemany(
                "INSERT INTO InstanceUIDs "
                "(StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID) "
                "VALUES(?, ?, ?)",
                referenced_uids,
            )

    def are_dimension_indices_unique(
        self,
        dimension_index_pointers: Sequence[Union[int, BaseTag]],
    ) -> bool:
        """Check if a list of index pointers uniquely identifies frames.

        For a given list of dimension index pointers, check whether every
        combination of index values for these pointers identifies a unique
        image frame. This is a pre-requisite for indexing using this list of
        dimension index pointers.

        Parameters
        ----------
        dimension_index_pointers: Sequence[Union[int, pydicom.tag.BaseTag]]
            Sequence of tags serving as dimension index pointers.

        Returns
        -------
        bool
            True if dimension indices are unique.

        """
        column_names = []
        for ptr in dimension_index_pointers:
            column_names.append(self._dim_ind_col_names[ptr])
        col_str = ", ".join(column_names)
        cur = self._db_con.cursor()
        n_unique_combos = cur.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM FrameLUT GROUP BY {col_str})"
        ).fetchone()[0]
        return n_unique_combos == self.NumberOfFrames

    def get_source_image_uids(self) -> List[Tuple[hd_UID, hd_UID, hd_UID]]:
        """Get UIDs of source image instances referenced in the image.

        Returns
        -------
        List[Tuple[highdicom.UID, highdicom.UID, highdicom.UID]]
            (Study Instance UID, Series Instance UID, SOP Instance UID) triplet
            for every image instance referenced in the image.

        """
        cur = self._db_con.cursor()
        res = cur.execute(
            'SELECT StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID '
            'FROM InstanceUIDs'
        )

        return [
            (hd_UID(a), hd_UID(b), hd_UID(c)) for a, b, c in res.fetchall()
        ]

    def _get_unique_referenced_sop_instance_uids(self) -> Set[str]:
        """Get set of unique Referenced SOP Instance UIDs.

        Returns
        -------
        Set[str]
            Set of unique Referenced SOP Instance UIDs.

        """
        cur = self._db_con.cursor()
        return {
            r[0] for r in
            cur.execute(
                'SELECT DISTINCT(SOPInstanceUID) from InstanceUIDs'
            )
        }

    def _get_max_referenced_frame_number(self) -> int:
        """Get highest frame number of any referenced frame.

        Absent access to the referenced dataset itself, being less than this
        value is a sufficient condition for the existence of a frame number
        in the source image.

        Returns
        -------
        int
            Highest frame number referenced in the image.

        """
        cur = self._db_con.cursor()
        return cur.execute(
            'SELECT MAX(ReferencedFrameNumber) FROM FrameLUT'
        ).fetchone()[0]

    def is_indexable_as_total_pixel_matrix(self) -> bool:
        """Whether the image can be indexed as a total pixel matrix.

        Returns
        -------
        bool:
            True if the image may be indexed using row and column
            positions in the total pixel matrix. False otherwise.

        """
        row_pos_kw = tag_for_keyword('RowPositionInTotalImagePixelMatrix')
        col_pos_kw = tag_for_keyword('ColumnPositionInTotalImagePixelMatrix')
        return (
            row_pos_kw in self._dim_ind_col_names and
            col_pos_kw in self._dim_ind_col_names
        )

    def _get_unique_dim_index_values(
        self,
        dimension_index_pointers: Sequence[int],
    ) -> Set[Tuple[int, ...]]:
        """Get set of unique dimension index value combinations.

        Parameters
        ----------
        dimension_index_pointers: Sequence[int]
            List of dimension index pointers for which to find unique
            combinations of values.

        Returns
        -------
        Set[Tuple[int, ...]]
            Set of unique dimension index value combinations for the given
            input dimension index pointers.

        """
        cols = [self._dim_ind_col_names[p] for p in dimension_index_pointers]
        cols_str = ', '.join(cols)
        cur = self._db_con.cursor()
        return {
            r for r in
            cur.execute(
                f'SELECT DISTINCT {cols_str} FROM FrameLUT'
            )
        }

    @property
    def volume_geometry(self) -> Optional[VolumeGeometry]:
        """Union[highdicom.VolumeGeometry, None]: Geometry of the volume if the
        image represents a regularly-spaced 3D volume. ``None``
        otherwise.

        """
        return self._volume_geometry

    @contextmanager
    def _generate_temp_table(
        self,
        table_name: str,
        column_defs: Sequence[str],
        column_data: Iterable[Sequence[Any]],
    ) -> Generator[None, None, None]:
        """Context manager that handles a temporary table.

        The temporary table is created with the specified information. Control
        flow then returns to code within the "with" block. After the "with"
        block has completed, the cleanup of the table is automatically handled.

        Parameters
        ----------
        table_name: str
            Name of the temporary table.
        column_defs: Sequence[str]
            SQL syntax strings defining each column in the temporary table, one
            string per column.
        column_data: Iterable[Sequence[Any]]
            Column data to place into the table.

        Yields
        ------
        None:
            Yields control to the "with" block, with the temporary table
            created.

        """
        defs_str = ', '.join(column_defs)
        create_cmd = (f'CREATE TABLE {table_name}({defs_str})')
        placeholders = ', '.join(['?'] * len(column_defs))

        with self._db_con:
            self._db_con.execute(create_cmd)
            self._db_con.executemany(
                f'INSERT INTO {table_name} VALUES({placeholders})',
                column_data
            )

        # Return control flow to "with" block
        yield

        # Clean up the table
        cmd = (f'DROP TABLE {table_name}')
        with self._db_con:
            self._db_con.execute(cmd)
