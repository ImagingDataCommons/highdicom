"""Module for SOP classes of the SEG modality."""
import logging
from collections import Counter, defaultdict
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from os import PathLike
import sqlite3
from typing import (
    Any,
    BinaryIO,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
import warnings

import numpy as np
from pydicom.dataset import Dataset
from pydicom.datadict import get_entry, keyword_for_tag, tag_for_keyword
from pydicom.encaps import encapsulate
from pydicom.multival import MultiValue
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from pydicom.tag import BaseTag, Tag
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
    RLELossless,
    UID,
)
from pydicom.sr.codedict import codes
from pydicom.valuerep import PersonName, format_number_as_ds
from pydicom.sr.coding import Code
from pydicom.filereader import dcmread

from highdicom._module_utils import ModuleUsageValues, get_module_usage
from highdicom.base import SOPClass, _check_little_endian
from highdicom.content import (
    ContentCreatorIdentificationCodeSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
    PixelMeasuresSequence
)
from highdicom.enum import (
    CoordinateSystemNames,
    DimensionOrganizationTypeValues,
)
from highdicom.frame import encode_frame
from highdicom.utils import (
    are_plane_positions_tiled_full,
    compute_plane_position_tiled_full,
    is_tiled_image,
    get_tile_array,
    iter_tiled_full_frame_data,
    tile_pixel_matrix,
)
from highdicom.seg.content import (
    DimensionIndexSequence,
    SegmentDescription,
)
from highdicom.seg.enum import (
    SegmentationFractionalTypeValues,
    SegmentationTypeValues,
    SegmentsOverlapValues,
    SpatialLocationsPreservedValues,
    SegmentAlgorithmTypeValues,
)
from highdicom.seg.utils import iter_segments
from highdicom.spatial import ImageToReferenceTransformer
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import (
    check_person_name,
    _check_code_string,
    _check_long_string,
)
from highdicom.uid import UID as hd_UID


logger = logging.getLogger(__name__)


_NO_FRAME_REF_VALUE = -1


def _get_unsigned_dtype(max_val: Union[int, np.integer]) -> type:
    """Get the smallest unsigned NumPy datatype to accommodate a value.

    Parameters
    ----------
    max_val: int
        The largest non-negative integer that must be accommodated.

    Returns
    -------
    numpy.dtype:
        The selected NumPy datatype.

    """
    if max_val < 256:
        dtype = np.dtype(np.uint8)
    elif max_val < 65536:
        dtype = np.dtype(np.uint16)
    else:
        dtype = np.dtype(np.uint32)  # should be extremely unlikely
    return dtype


def _check_numpy_value_representation(
    max_val: int,
    dtype: Union[np.dtype, str, type]
) -> None:
    """Check whether a given maximum value can be represented by a given dtype.

    Parameters
    ----------
    max_val: int
        The largest non-negative integer that must be accommodated.
    dtype: Union[numpy.dtype, str, type]
        Data type of the array to be checked

    Raises
    ------
    ValueError
        If the given maximum value is too large to be represented by dtype.

    """
    dtype = np.dtype(dtype)
    raise_error = False
    if dtype.kind == 'f':
        if max_val > np.finfo(dtype).max:
            raise_error = True
    elif dtype.kind in ('i', 'u'):
        if max_val > np.iinfo(dtype).max:
            raise_error = True
    if raise_error:
        raise ValueError(
            "The maximum output value of the segmentation array is "
            f"{max_val}, which is too large be represented using dtype "
            f"{dtype}."
        )


class _SegDBManager:

    """Database manager for data associated with a segmentation image."""

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

    def __init__(
        self,
        referenced_uids: List[Tuple[str, str, str]],
        segment_numbers: List[int],
        dim_indices: Dict[int, List[int]],
        dim_values: Dict[int, List[Any]],
        referenced_instances: Optional[List[str]],
        referenced_frames: Optional[List[int]],
    ):
        """

        Parameters
        ----------
        referenced_uids: List[Tuple[str, str, str]]
            Triplet of UIDs for each image instance (Study Instance UID,
            Series Instance UID, SOP Instance UID) that is referenced
            in the segmentation image.
        segment_numbers: List[int]
            Segment numbers for each frame in the segmentation image.
        dim_indices: Dict[int, List[int]]
            Dictionary mapping the integer tag value of each dimension index
            pointer (excluding SegmentNumber) to a list of dimension indices
            for each frame in the segmentation image.
        dim_values: Dict[int, List[Values]]
            Dictionary mapping the integer tag value of each dimension index
            pointer (excluding SegmentNumber) to a list of dimension values
            for each frame in the segmentation image.
        referenced_instances: Optional[List[str]]
            SOP Instance UID of each referenced image instance for each frame
            in the segmentation image. Should be omitted if there is not a
            single referenced image instance per segmentation image frame.
        referenced_frames: Optional[List[int]]
            Number of the corresponding frame in the referenced image
            instance for each frame in the segmentation image. Should be
            omitted if there is not a single referenced image instance per
            segmentation image frame.

        """
        self._db_con: sqlite3.Connection = sqlite3.connect(":memory:")

        self._create_ref_instance_table(referenced_uids)

        self._number_of_frames = len(segment_numbers)

        # Construct the columns and values to put into a frame look-up table
        # table within sqlite. There will be one row per frame in the
        # segmentation instance
        col_defs = []  # SQL column definitions
        col_data = []  # lists of column data

        # Frame number column
        col_defs.append('FrameNumber INTEGER PRIMARY KEY')
        col_data.append(list(range(1, self._number_of_frames + 1)))

        # Segment number column
        col_defs.append('SegmentNumber INTEGER NOT NULL')
        col_data.append(segment_numbers)

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
            try:
                vm = int(vm_str)
            except ValueError:
                continue
            try:
                sql_type = self._DCM_SQL_TYPE_MAP[vr]
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

    def _create_ref_instance_table(
        self,
        referenced_uids: List[Tuple[str, str, str]],
    ) -> None:
        """Create a table of referenced instances.

        The resulting table (called InstanceUIDs) contains Study, Series and
        SOP instance UIDs for each instance referenced by the segmentation
        image.

        Parameters
        ----------
        referenced_uids: List[Tuple[str, str, str]]
            List of UIDs for each instance referenced in the segmentation.
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

    def get_source_image_uids(self) -> List[Tuple[hd_UID, hd_UID, hd_UID]]:
        """Get UIDs of source image instances referenced in the segmentation.

        Returns
        -------
        List[Tuple[highdicom.UID, highdicom.UID, highdicom.UID]]
            (Study Instance UID, Series Instance UID, SOP Instance UID) triplet
            for every image instance referenced in the segmentation.

        """
        cur = self._db_con.cursor()
        res = cur.execute(
            'SELECT StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID '
            'FROM InstanceUIDs'
        )

        return [
            (hd_UID(a), hd_UID(b), hd_UID(c)) for a, b, c in res.fetchall()
        ]

    def are_dimension_indices_unique(
        self,
        dimension_index_pointers: Sequence[Union[int, BaseTag]],
    ) -> bool:
        """Check if a list of index pointers uniquely identifies frames.

        For a given list of dimension index pointers, check whether every
        combination of index values for these pointers identifies a unique
        frame per segment in the segmentation image. This is a pre-requisite
        for indexing using this list of dimension index pointers in the
        :meth:`Segmentation.get_pixels_by_dimension_index_values()` method.

        Parameters
        ----------
        dimension_index_pointers: Sequence[Union[int, pydicom.tag.BaseTag]]
            Sequence of tags serving as dimension index pointers.

        Returns
        -------
        bool
            True if dimension indices are unique.

        """
        column_names = ['SegmentNumber']
        for ptr in dimension_index_pointers:
            column_names.append(self._dim_ind_col_names[ptr])
        col_str = ", ".join(column_names)
        cur = self._db_con.cursor()
        n_unique_combos = cur.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM FrameLUT GROUP BY {col_str})"
        ).fetchone()[0]
        return n_unique_combos == self._number_of_frames

    def are_referenced_sop_instances_unique(self) -> bool:
        """Check if Referenced SOP Instance UIDs uniquely identify frames.

        This is a pre-requisite for requesting segmentation masks defined by
        the SOP Instance UIDs of their source frames, such as using the
        Segmentation.get_pixels_by_source_instance() method and
        _SegDBManager.iterate_indices_by_source_instance() method.

        Returns
        -------
        bool
            True if the ReferencedSOPInstanceUID (in combination with the
            segment number) uniquely identifies frames of the segmentation
            image.

        """
        cur = self._db_con.cursor()
        n_unique_combos = cur.execute(
            'SELECT COUNT(*) FROM '
            '(SELECT 1 FROM FrameLUT GROUP BY ReferencedSOPInstanceUID, '
            'SegmentNumber)'
        ).fetchone()[0]
        return n_unique_combos == self._number_of_frames

    def are_referenced_frames_unique(self) -> bool:
        """Check if Referenced Frame Numbers uniquely identify frames.

        Returns
        -------
        bool
            True if the ReferencedFrameNumber (in combination with the
            segment number) uniquely identifies frames of the segmentation
            image.

        """
        cur = self._db_con.cursor()
        n_unique_combos = cur.execute(
            'SELECT COUNT(*) FROM '
            '(SELECT 1 FROM FrameLUT GROUP BY ReferencedFrameNumber, '
            'SegmentNumber)'
        ).fetchone()[0]
        return n_unique_combos == self._number_of_frames

    def get_unique_sop_instance_uids(self) -> Set[str]:
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

    def get_max_frame_number(self) -> int:
        """Get highest frame number of any referenced frame.

        Absent access to the referenced dataset itself, being less than this
        value is a sufficient condition for the existence of a frame number
        in the source image.

        Returns
        -------
        int
            Highest frame number referenced in the segmentation image.

        """
        cur = self._db_con.cursor()
        return cur.execute(
            'SELECT MAX(ReferencedFrameNumber) FROM FrameLUT'
        ).fetchone()[0]

    def get_unique_dim_index_values(
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

    def is_indexable_as_total_pixel_matrix(self) -> bool:
        """Whether the segmentation can be indexed as a total pixel matrix.

        Returns
        -------
        bool:
            True if the segmentation may be indexed using row and column
            positions in the total pixel matrix. False otherwise.

        """
        row_pos_kw = tag_for_keyword('RowPositionInTotalImagePixelMatrix')
        col_pos_kw = tag_for_keyword('ColumnPositionInTotalImagePixelMatrix')
        return (
            row_pos_kw in self._dim_ind_col_names and
            col_pos_kw in self._dim_ind_col_names
        )

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

    @contextmanager
    def _generate_temp_segment_table(
        self,
        segment_numbers: Sequence[int],
        combine_segments: bool,
        relabel: bool
    ) -> Generator[None, None, None]:
        """Context manager that handles a temporary table for segments.

        The temporary table is named "TemporarySegmentNumbers" with columns
        OutputSegmentNumber and SegmentNumber that are populated with values
        derived from the input. Control flow then returns to code within the
        "with" block. After the "with" block has completed, the cleanup of
        the table is automatically handled.

        Parameters
        ----------
        segment_numbers: Sequence[int]
            Segment numbers to include, in the order desired.
        combine_segments: bool
            Whether the segments will be combined into a label map.
        relabel: bool
            Whether the output segment numbers should be relabelled to 1-n
            (True) or retain their values in the original segmentation object.

        Yields
        ------
        None:
            Yields control to the "with" block, with the temporary table
            created.

        """
        if combine_segments:
            if relabel:
                # Output segment numbers are consecutive and start at 1
                data = enumerate(segment_numbers, 1)
            else:
                # Output segment numbers are the same as the input
                # segment numbers
                data = zip(segment_numbers, segment_numbers)
        else:
            # Output segment numbers are indices along the output
            # array's segment dimension, so are consecutive starting at
            # 0
            data = enumerate(segment_numbers)

        cmd = (
            'CREATE TABLE TemporarySegmentNumbers('
            '    SegmentNumber INTEGER UNIQUE NOT NULL,'
            '    OutputSegmentNumber INTEGER UNIQUE NOT NULL'
            ')'
        )

        with self._db_con:
            self._db_con.execute(cmd)
            self._db_con.executemany(
                'INSERT INTO '
                'TemporarySegmentNumbers('
                '    OutputSegmentNumber, SegmentNumber'
                ')'
                'VALUES(?, ?)',
                data
            )

        # Yield execution to "with" block
        yield

        # Clean up table after user code executes
        with self._db_con:
            self._db_con.execute('DROP TABLE TemporarySegmentNumbers')

    @contextmanager
    def iterate_indices_by_source_instance(
        self,
        source_sop_instance_uids: Sequence[str],
        segment_numbers: Sequence[int],
        combine_segments: bool = False,
        relabel: bool = False,
    ) -> Generator[
            Iterator[
                Tuple[
                    Tuple[Union[slice, int], ...],
                    Tuple[Union[slice, int], ...],
                    int
                ]
            ],
            None,
            None,
        ]:
        """Iterate over segmentation frame indices for given source image
        instances.

        This is intended for the case of a segmentation image that references
        multiple single frame sources images (typically a series). In this
        case, the user supplies a list of SOP Instance UIDs of the source
        images of interest, and this method returns information about the
        frames of the segmentation image relevant to these source images.

        This yields an iterator to the underlying database result that iterates
        over information on the steps required to construct the requested
        segmentation mask from the stored frames of the segmentation image.

        This method is intended to be used as a context manager that yields the
        requested iterator. The iterator is only valid while the context
        manager is active.

        Parameters
        ----------
        source_sop_instance_uids: str
            SOP Instance UID of the source instances for which segmentation
            image frames are requested.
        segment_numbers: Sequence[int]
            Numbers of segments to include.
        combine_segments: bool, optional
            If True, produce indices to combine the different segments into a
            single label map in which the value of a pixel represents its
            segment. If False (the default), segments are binary and stacked
            down the last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the output segment
            numbers are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.

        Yields
        ------
        Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int]]:
            Indices required to construct the requested mask. Each
            triplet denotes the (output indexer, segmentation indexer,
            output segment number) representing a list of "instructions" to
            create the requested output array by copying frames from the
            segmentation dataset and inserting them into the output array with
            a given segment value. Output indexer and segmentation indexer are
            tuples that can be used to index the output and segmentations
            numpy arrays directly.

        """  # noqa: E501
        # Run query to create the iterable of indices needed to construct the
        # desired pixel array. The approach here is to create two temporary
        # tables in the SQLite database, one for the desired source UIDs, and
        # another for the desired segments, then use table joins with the
        # referenced UIDs table and the frame LUT at the relevant rows, before
        # clearing up the temporary tables.

        # Create temporary table of desired frame numbers
        table_name = 'TemporarySOPInstanceUIDs'
        column_defs = [
            'OutputFrameIndex INTEGER UNIQUE NOT NULL',
            'SourceSOPInstanceUID VARCHAR UNIQUE NOT NULL'
        ]
        column_data = enumerate(source_sop_instance_uids)

        # Construct the query The ORDER BY is not logically necessary
        # but seems to improve performance of the downstream numpy
        # operations, presumably as it is more cache efficient
        query = (
            'SELECT '
            '    T.OutputFrameIndex,'
            '    L.FrameNumber - 1,'
            '    S.OutputSegmentNumber '
            'FROM TemporarySOPInstanceUIDs T '
            'INNER JOIN FrameLUT L'
            '    ON T.SourceSOPInstanceUID = L.ReferencedSOPInstanceUID '
            'INNER JOIN TemporarySegmentNumbers S'
            '    ON L.SegmentNumber = S.SegmentNumber '
            'ORDER BY T.OutputFrameIndex'
        )

        with self._generate_temp_table(
            table_name=table_name,
            column_defs=column_defs,
            column_data=column_data,
        ):
            with self._generate_temp_segment_table(
                segment_numbers=segment_numbers,
                combine_segments=combine_segments,
                relabel=relabel
            ):
                yield (
                    (
                        (fo, slice(None), slice(None)),
                        (fi, slice(None), slice(None)),
                        seg_no
                    )
                    for (fo, fi, seg_no) in self._db_con.execute(query)
                )

    @contextmanager
    def iterate_indices_by_source_frame(
        self,
        source_sop_instance_uid: str,
        source_frame_numbers: Sequence[int],
        segment_numbers: Sequence[int],
        combine_segments: bool = False,
        relabel: bool = False,
    ) -> Generator[
            Iterator[
                Tuple[
                    Tuple[Union[slice, int], ...],
                    Tuple[Union[slice, int], ...],
                    int
                ]
            ],
            None,
            None,
        ]:
        """Iterate over frame indices for given source image frames.

        This is intended for the case of a segmentation image that references a
        single multi-frame source image instance. In this case, the user
        supplies a list of frames numbers of interest within the single source
        instance, and this method returns information about the frames
        of the segmentation image relevant to these frames.

        This yields an iterator to the underlying database result that iterates
        over information on the steps required to construct the requested
        segmentation mask from the stored frames of the segmentation image.

        This method is intended to be used as a context manager that yields the
        requested iterator. The iterator is only valid while the context
        manager is active.

        Parameters
        ----------
        source_sop_instance_uid: str
            SOP Instance UID of the source instance that contains the source
            frames.
        source_frame_numbers: Sequence[int]
            A sequence of frame numbers (1-based) within the source instance
            for which segmentations are requested.
        segment_numbers: Sequence[int]
            Sequence containing segment numbers to include.
        combine_segments: bool, optional
            If True, produce indices to combine the different segments into a
            single label map in which the value of a pixel represents its
            segment. If False (the default), segments are binary and stacked
            down the last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the output segment
            numbers are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.

        Yields
        ------
        Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int]]:
            Indices required to construct the requested mask. Each
            triplet denotes the (output indexer, segmentation indexer,
            output segment number) representing a list of "instructions" to
            create the requested output array by copying frames from the
            segmentation dataset and inserting them into the output array with
            a given segment value. Output indexer and segmentation indexer are
            tuples that can be used to index the output and segmentations
            numpy arrays directly.

        """  # noqa: E501
        # Run query to create the iterable of indices needed to construct the
        # desired pixel array. The approach here is to create two temporary
        # tables in the SQLite database, one for the desired frame numbers, and
        # another for the desired segments, then use table joins with the frame
        # LUT to arrive at the relevant rows, before clearing up the temporary
        # tables.

        # Create temporary table of desired frame numbers
        table_name = 'TemporaryFrameNumbers'
        column_defs = [
            'OutputFrameIndex INTEGER UNIQUE NOT NULL',
            'SourceFrameNumber INTEGER UNIQUE NOT NULL'
        ]
        column_data = enumerate(source_frame_numbers)

        # Construct the query The ORDER BY is not logically necessary
        # but seems to improve performance of the downstream numpy
        # operations, presumably as it is more cache efficient
        query = (
            'SELECT '
            '    F.OutputFrameIndex,'
            '    L.FrameNumber - 1,'
            '    S.OutputSegmentNumber '
            'FROM TemporaryFrameNumbers F '
            'INNER JOIN FrameLUT L'
            '    ON F.SourceFrameNumber = L.ReferencedFrameNumber '
            'INNER JOIN TemporarySegmentNumbers S'
            '    ON L.SegmentNumber = S.SegmentNumber '
            'ORDER BY F.OutputFrameIndex'
        )

        with self._generate_temp_table(
            table_name=table_name,
            column_defs=column_defs,
            column_data=column_data,
        ):
            with self._generate_temp_segment_table(
                segment_numbers=segment_numbers,
                combine_segments=combine_segments,
                relabel=relabel
            ):
                yield (
                    (
                        (fo, slice(None), slice(None)),
                        (fi, slice(None), slice(None)),
                        seg_no
                    )
                    for (fo, fi, seg_no) in self._db_con.execute(query)
                )

    @contextmanager
    def iterate_indices_by_dimension_index_values(
        self,
        dimension_index_values: Sequence[Sequence[int]],
        dimension_index_pointers: Sequence[int],
        segment_numbers: Sequence[int],
        combine_segments: bool = False,
        relabel: bool = False,
    ) -> Generator[
            Iterator[
                Tuple[
                    Tuple[Union[slice, int], ...],
                    Tuple[Union[slice, int], ...],
                    int
                ]
            ],
            None,
            None,
        ]:
        """Iterate over frame indices for given dimension index values.

        This is intended to be the most flexible and lowest-level (and there
        also least convenient) method to request information about
        segmentation frames. The user can choose to specify which segmentation
        frames are of interest using arbitrary dimension indices and their
        associated values. This makes no assumptions about the dimension
        organization of the underlying segmentation, except that the given
        dimension indices can be used to uniquely identify frames in the
        segmentation image.

        This yields an iterator to the underlying database result that iterates
        over information on the steps required to construct the requested
        segmentation mask from the stored frames of the segmentation image.

        This method is intended to be used as a context manager that yields the
        requested iterator. The iterator is only valid while the context
        manager is active.

        Parameters
        ----------
        dimension_index_values: Sequence[Sequence[int]]
            Dimension index values for the requested frames.
        dimension_index_pointers: Sequence[Union[int, pydicom.tag.BaseTag]]
            The data element tags that identify the indices used in the
            ``dimension_index_values`` parameter.
        segment_numbers: Sequence[int]
            Sequence containing segment numbers to include.
        combine_segments: bool, optional
            If True, produce indices to combine the different segments into a
            single label map in which the value of a pixel represents its
            segment. If False (the default), segments are binary and stacked
            down the last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the output segment
            numbers are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.

        Yields
        ------
        Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int]]:
            Indices required to construct the requested mask. Each
            triplet denotes the (output indexer, segmentation indexer,
            output segment number) representing a list of "instructions" to
            create the requested output array by copying frames from the
            segmentation dataset and inserting them into the output array with
            a given segment value. Output indexer and segmentation indexer are
            tuples that can be used to index the output and segmentations
            numpy arrays directly.

        """  # noqa: E501
        # Create temporary table of desired dimension indices
        table_name = 'TemporaryDimensionIndexValues'

        dim_ind_cols = [
            self._dim_ind_col_names[p] for p in dimension_index_pointers
        ]
        column_defs = (
            ['OutputFrameIndex INTEGER UNIQUE NOT NULL'] +
            [f'{col} INTEGER NOT NULL' for col in dim_ind_cols]
        )
        column_data = (
            (i, *tuple(row))
            for i, row in enumerate(dimension_index_values)
        )

        # Construct the query The ORDER BY is not logically necessary
        # but seems to improve performance of the downstream numpy
        # operations, presumably as it is more cache efficient
        join_str = ' AND '.join(f'D.{col} = L.{col}' for col in dim_ind_cols)
        query = (
            'SELECT '
            '    D.OutputFrameIndex,'  # frame index of the output array
            '    L.FrameNumber - 1,'  # frame *index* of segmentation image
            '    S.OutputSegmentNumber '  # output segment number
            'FROM TemporaryDimensionIndexValues D '
            'INNER JOIN FrameLUT L'
            f'   ON {join_str} '
            'INNER JOIN TemporarySegmentNumbers S'
            '    ON L.SegmentNumber = S.SegmentNumber '
            'ORDER BY D.OutputFrameIndex'
        )

        with self._generate_temp_table(
            table_name=table_name,
            column_defs=column_defs,
            column_data=column_data,
        ):
            with self._generate_temp_segment_table(
                segment_numbers=segment_numbers,
                combine_segments=combine_segments,
                relabel=relabel
            ):
                yield (
                    (
                        (fo, slice(None), slice(None)),
                        (fi, slice(None), slice(None)),
                        seg_no
                    )
                    for (fo, fi, seg_no) in self._db_con.execute(query)
                )

    @contextmanager
    def iterate_indices_for_tiled_region(
        self,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
        tile_shape: Tuple[int, int],
        segment_numbers: Sequence[int],
        combine_segments: bool = False,
        relabel: bool = False,
    ) -> Generator[
            Iterator[
                Tuple[
                    Tuple[Union[slice, int], ...],
                    Tuple[Union[slice, int], ...],
                    int
                ]
            ],
            None,
            None,
        ]:
        """Iterate over segmentation frame indices for a given region of the
        segmentation's total pixel matrix.

        This is intended for the case of a segmentation image that is stored as
        a tiled representation of total pixel matrix.

        This yields an iterator to the underlying database result that iterates
        over information on the steps required to construct the requested
        segmentation mask from the stored frames of the segmentation image.

        This method is intended to be used as a context manager that yields the
        requested iterator. The iterator is only valid while the context
        manager is active.

        Parameters
        ----------
        row_start: int
            Row index (1-based) in the total pixel matrix of the first row of
            the output array. May be negative (last row is -1).
        row_end: int
            Row index (1-based) in the total pixel matrix one beyond the last
            row of the output array. May be negative (last row is -1).
        column_start: int
            Column index (1-based) in the total pixel matrix of the first
            column of the output array. May be negative (last column is -1).
        column_end: int
            Column index (1-based) in the total pixel matrix one beyond the last
            column of the output array. May be negative (last column is -1).
        tile_shape: Tuple[int, int]
            Shape of each tile (rows, columns).
        segment_numbers: Sequence[int]
            Numbers of segments to include.
        combine_segments: bool, optional
            If True, produce indices to combine the different segments into a
            single label map in which the value of a pixel represents its
            segment. If False (the default), segments are binary and stacked
            down the last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the output segment
            numbers are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.

        Yields
        ------
        Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int]]:
            Indices required to construct the requested mask. Each
            triplet denotes the (output indexer, segmentation indexer,
            output segment number) representing a list of "instructions" to
            create the requested output array by copying frames from the
            segmentation dataset and inserting them into the output array with
            a given segment value. Output indexer and segmentation indexer are
            tuples that can be used to index the output and segmentations
            numpy arrays directly.

        """  # noqa: E501
        th, tw = tile_shape

        oh = row_end - row_start
        ow = column_end - column_start

        row_offset_start = row_start - th + 1
        column_offset_start = column_start - tw + 1

        # Construct the query The ORDER BY is not logically necessary
        # but seems to improve performance of the downstream numpy
        # operations, presumably as it is more cache efficient
        query = (
            'SELECT '
            '    L.RowPositionInTotalImagePixelMatrix,'
            '    L.ColumnPositionInTotalImagePixelMatrix,'
            '    L.FrameNumber - 1,'
            '    S.OutputSegmentNumber '
            'FROM FrameLUT L '
            'INNER JOIN TemporarySegmentNumbers S'
            '    ON L.SegmentNumber = S.SegmentNumber '
            'WHERE ('
            '    L.RowPositionInTotalImagePixelMatrix >= '
            f'        {row_offset_start}'
            f'    AND L.RowPositionInTotalImagePixelMatrix < {row_end}'
            '    AND L.ColumnPositionInTotalImagePixelMatrix >= '
            f'        {column_offset_start}'
            f'    AND L.ColumnPositionInTotalImagePixelMatrix < {column_end}'
            ')'
            'ORDER BY '
            '     L.RowPositionInTotalImagePixelMatrix,'
            '     L.ColumnPositionInTotalImagePixelMatrix,'
            '     S.OutputSegmentNumber'
        )

        with self._generate_temp_segment_table(
            segment_numbers=segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel
        ):
            yield (
                (
                    (
                        slice(
                            max(rp - row_start, 0),
                            min(rp + th - row_start, oh)
                        ),
                        slice(
                            max(cp - column_start, 0),
                            min(cp + tw - column_start, ow)
                        ),
                    ),
                    (
                        fi,
                        slice(
                            max(row_start - rp, 0),
                            min(row_end - rp, th)
                        ),
                        slice(
                            max(column_start - cp, 0),
                            min(column_end - cp, tw)
                        ),
                    ),
                    seg_no
                )
                for (rp, cp, fi, seg_no) in self._db_con.execute(query)
            )


class Segmentation(SOPClass):

    """SOP class for the Segmentation IOD."""

    def __init__(
        self,
        source_images: Sequence[Dataset],
        pixel_array: np.ndarray,
        segmentation_type: Union[str, SegmentationTypeValues],
        segment_descriptions: Sequence[SegmentDescription],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: Union[str, Tuple[str]],
        device_serial_number: str,
        fractional_type: Optional[
            Union[str, SegmentationFractionalTypeValues]
        ] = SegmentationFractionalTypeValues.PROBABILITY,
        max_fractional_value: int = 255,
        content_description: Optional[str] = None,
        content_creator_name: Optional[Union[str, PersonName]] = None,
        transfer_syntax_uid: Union[str, UID] = ExplicitVRLittleEndian,
        pixel_measures: Optional[PixelMeasuresSequence] = None,
        plane_orientation: Optional[PlaneOrientationSequence] = None,
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
        omit_empty_frames: bool = True,
        content_label: Optional[str] = None,
        content_creator_identification: Optional[
            ContentCreatorIdentificationCodeSequence
        ] = None,
        workers: Union[int, Executor] = 0,
        dimension_organization_type: Union[
            DimensionOrganizationTypeValues,
            str,
            None,
        ] = None,
        tile_pixel_array: bool = False,
        tile_size: Union[Sequence[int], None] = None,
        pyramid_uid: Optional[str] = None,
        pyramid_label: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        source_images: Sequence[Dataset]
            One or more single- or multi-frame images (or metadata of images)
            from which the segmentation was derived
        pixel_array: numpy.ndarray
            Array of segmentation pixel data of boolean, unsigned integer or
            floating point data type representing a mask image. The array may
            be a 2D, 3D or 4D numpy array.

            If it is a 2D numpy array, it represents the segmentation of a
            single frame image, such as a planar x-ray or single instance from
            a CT or MR series.

            If it is a 3D array, it represents the segmentation of either a
            series of source images (such as a series of CT or MR images) a
            single 3D multi-frame image (such as a multi-frame CT/MR image), or
            a single 2D tiled image (such as a slide microscopy image).

            If ``pixel_array`` represents the segmentation of a 3D image, the
            first dimension represents individual 2D planes. Unless the
            ``plane_positions`` parameter is provided, the frame in
            ``pixel_array[i, ...]`` should correspond to either
            ``source_images[i]`` (if ``source_images`` is a list of single
            frame instances) or ``source_images[0].pixel_array[i, ...]`` if
            ``source_images`` is a single multiframe instance.

            Similarly, if ``pixel_array`` is a 3D array representing the
            segmentation of a tiled 2D image, the first dimension represents
            individual 2D tiles (for one channel and z-stack) and these tiles
            correspond to the frames in the source image dataset.

            If ``pixel_array`` is an unsigned integer or boolean array with
            binary data (containing only the values ``True`` and ``False`` or
            ``0`` and ``1``) or a floating-point array, it represents a single
            segment. In the case of a floating-point array, values must be in
            the range 0.0 to 1.0.

            Otherwise, if ``pixel_array`` is a 2D or 3D array containing multiple
            unsigned integer values, each value is treated as a different
            segment whose segment number is that integer value. This is
            referred to as a *label map* style segmentation.  In this case, all
            segments from 1 through ``pixel_array.max()`` (inclusive) must be
            described in `segment_descriptions`, regardless of whether they are
            present in the image.  Note that this is valid for segmentations
            encoded using the ``"BINARY"`` or ``"FRACTIONAL"`` methods.

            Note that that a 2D numpy array and a 3D numpy array with a
            single frame along the first dimension may be used interchangeably
            as segmentations of a single frame, regardless of their data type.

            If ``pixel_array`` is a 4D numpy array, the first three dimensions
            are used in the same way as the 3D case and the fourth dimension
            represents multiple segments. In this case
            ``pixel_array[:, :, :, i]`` represents segment number ``i + 1``
            (since numpy indexing is 0-based but segment numbering is 1-based),
            and all segments from 1 through ``pixel_array.shape[-1] + 1`` must
            be described in ``segment_descriptions``.

            Furthermore, a 4D array with unsigned integer data type must
            contain only binary data (``True`` and ``False`` or ``0`` and
            ``1``). In other words, a 4D array is incompatible with the *label
            map* style encoding of the segmentation.

            Where there are multiple segments that are mutually exclusive (do
            not overlap) and binary, they may be passed using either a *label
            map* style array or a 4D array. A 4D array is required if either
            there are multiple segments and they are not mutually exclusive
            (i.e. they overlap) or there are multiple segments and the
            segmentation is fractional.

            Note that if the segmentation of a single source image with
            multiple stacked segments is required, it is necessary to include
            the singleton first dimension in order to give a 4D array.

            For ``"FRACTIONAL"`` segmentations, values either encode the
            probability of a given pixel belonging to a segment
            (if `fractional_type` is ``"PROBABILITY"``)
            or the extent to which a segment occupies the pixel
            (if `fractional_type` is ``"OCCUPANCY"``).

        segmentation_type: Union[str, highdicom.seg.SegmentationTypeValues]
            Type of segmentation, either ``"BINARY"`` or ``"FRACTIONAL"``
        segment_descriptions: Sequence[highdicom.seg.SegmentDescription]
            Description of each segment encoded in `pixel_array`. In the case of
            pixel arrays with multiple integer values, the segment description
            with the corresponding segment number is used to describe each segment.
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the output segmentation series.
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        manufacturer: str
            Name of the manufacturer of the device (developer of the software)
            that creates the instance
        manufacturer_model_name: str
            Name of the device model (name of the software library or
            application) that creates the instance
        software_versions: Union[str, Tuple[str]]
            Version(s) of the software that creates the instance
        device_serial_number: str
            Manufacturer's serial number of the device
        fractional_type: Union[str, highdicom.seg.SegmentationFractionalTypeValues, None], optional
            Type of fractional segmentation that indicates how pixel data
            should be interpreted
        max_fractional_value: int, optional
            Maximum value that indicates probability or occupancy of 1 that
            a pixel represents a given segment
        content_description: Union[str, None], optional
            Description of the segmentation
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the creator of the segmentation (if created manually)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following lossless compressed transfer syntaxes
            are supported for encapsulated format encoding in case of
            FRACTIONAL segmentation type:
            RLE Lossless (``"1.2.840.10008.1.2.5"``),
            JPEG 2000 Lossless (``"1.2.840.10008.1.2.4.90"``), and
            JPEG LS Lossless (``"1.2.840.10008.1.2.4.00"``).
        pixel_measures: Union[highdicom.PixelMeasures, None], optional
            Physical spacing of image pixels in `pixel_array`.
            If ``None``, it will be assumed that the segmentation image has the
            same pixel measures as the source image(s).
        plane_orientation: Union[highdicom.PlaneOrientationSequence, None], optional
            Orientation of planes in `pixel_array` relative to axes of
            three-dimensional patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image as the
            same plane orientation as the source image(s).
        plane_positions: Union[Sequence[highdicom.PlanePositionSequence], None], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space.
            If ``None``, it will be assumed that the segmentation image has the
            same plane position as the source image(s). However, this will only
            work when the first dimension of `pixel_array` matches the number
            of frames in `source_images` (in case of multi-frame source images)
            or the number of `source_images` (in case of single-frame source
            images).
        omit_empty_frames: bool, optional
            If True (default), frames with no non-zero pixels are omitted from
            the segmentation image. If False, all frames are included.
        content_label: Union[str, None], optional
            Content label
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this segmentation.
        workers: Union[int, concurrent.futures.Executor], optional
            Number of worker processes to use for frame compression. If 0, no
            workers are used and compression is performed in the main process
            (this is the default behavior). If negative, as many processes are
            created as the machine has processors.

            Alternatively, you may directly pass an instance of a class derived
            from ``concurrent.futures.Executor`` (most likely an instance of
            ``concurrent.futures.ProcessPoolExecutor``) for highdicom to use.
            You may wish to do this either to have greater control over the
            setup of the executor, or to avoid the setup cost of spawning new
            processes each time this ``__init__`` method is called if your
            application creates a large number of Segmentations.

            Note that if you use worker processes, you must ensure that your
            main process uses the ``if __name__ == "__main__"`` idiom to guard
            against spawned child processes creating further workers.
        dimension_organization_type: Union[highdicom.enum.DimensionOrganizationTypeValues, str, None], optional
            Dimension organization type to use for the output image.
        tile_pixel_array: bool, optional
            If True, `highdicom` will automatically convert an input total
            pixel matrix into a sequence of frames representing tiles of the
            segmentation. This is valid only when the source image supports
            tiling (e.g. VL Whole Slide Microscopy images).

            If True, the input pixel array must consist of a single "frame",
            i.e. must be either a 2D numpy array, a 3D numpy array with a size
            of 1 down the first dimension (axis zero), or a 4D numpy array also
            with a size of 1 down the first dimension. The input pixel array is
            treated as the total pixel matrix of the segmentation, and this is
            tiled along the row and column dimension to create an output image
            with multiple, smaller frames.

            If no ``pixel_measures``, ``plane_positions``,
            ``plane_orientation`` are supplied, the total pixel matrix of the
            segmentation is assumed to correspond to the total pixel matrix of
            the (single) source image. If ``plane_positions`` is supplied, the
            sequence should contain a single item representing the plane
            position of the entire total pixel matrix. Plane positions of
            the newly created tiles will derived automatically from this.

            If False, the pixel array is already considered to consist of one
            or more existing frames, as described above.
        tile_size: Union[Sequence[int], None], optional
            Tile size to use when tiling the input pixel array. If ``None``
            (the default), the tile size is copied from the source image.
            Otherwise the tile size is specified explicitly as (number of rows,
            number of columns). This value is ignored if ``tile_pixel_array``
            is False.
        pyramid_uid: Optional[str], optional
            Unique identifier for the pyramid containing this segmentation.
            Should only be used if this segmentation is part of a
            multi-resolution pyramid.
        pyramid_label: Optional[str], optional
            Human readable label for the pyramid containing this segmentation.
            Should only be used if this segmentation is part of a
            multi-resolution pyramid.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Raises
        ------
        ValueError
            When

                * Length of `source_images` is zero.
                * Items of `source_images` are not all part of the same study
                  and series.
                * Items of `source_images` have different number of rows and
                  columns.
                * Length of `plane_positions` does not match number of segments
                  encoded in `pixel_array`.
                * Length of `plane_positions` does not match number of 2D planes
                  in `pixel_array` (size of first array dimension).

        Note
        ----
        The assumption is made that segments in `pixel_array` are defined in
        the same frame of reference as `source_images`.


        """  # noqa: E501
        if len(source_images) == 0:
            raise ValueError('At least one source image is required.')

        uniqueness_criteria = set(
            (
                image.StudyInstanceUID,
                image.SeriesInstanceUID,
                image.Rows,
                image.Columns,
                getattr(image, 'FrameOfReferenceUID', None),
            )
            for image in source_images
        )
        if len(uniqueness_criteria) > 1:
            raise ValueError(
                'Source images must all be part of the same series and must '
                'have the same image dimensions (number of rows/columns).'
            )

        src_img = source_images[0]
        is_multiframe = hasattr(src_img, 'NumberOfFrames')
        if is_multiframe and len(source_images) > 1:
            raise ValueError(
                'Only one source image should be provided in case images '
                'are multi-frame images.'
            )
        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            JPEG2000Lossless,
            JPEGLSLossless,
            RLELossless,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported.'
            )

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        if pixel_array.ndim not in [3, 4]:
            raise ValueError('Pixel array must be a 2D, 3D, or 4D array.')

        is_tiled = hasattr(src_img, 'TotalPixelMatrixRows')
        if tile_pixel_array and not is_tiled:
            raise ValueError(
                'When argument "tile_pixel_array" is True, the source image '
                'must be a tiled image.'
            )
        if tile_pixel_array and pixel_array.shape[0] != 1:
            raise ValueError(
                'When argument "tile_pixel_array" is True, the input pixel '
                'array must contain only one "frame" representing the entire '
                'entire pixel matrix.'
            )

        super().__init__(
            study_instance_uid=src_img.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid='1.2.840.10008.5.1.4.1.1.66.4',
            manufacturer=manufacturer,
            modality='SEG',
            transfer_syntax_uid=transfer_syntax_uid,
            patient_id=src_img.PatientID,
            patient_name=src_img.PatientName,
            patient_birth_date=src_img.PatientBirthDate,
            patient_sex=src_img.PatientSex,
            accession_number=src_img.AccessionNumber,
            study_id=src_img.StudyID,
            study_date=src_img.StudyDate,
            study_time=src_img.StudyTime,
            referring_physician_name=getattr(
                src_img, 'ReferringPhysicianName', None
            ),
            manufacturer_model_name=manufacturer_model_name,
            device_serial_number=device_serial_number,
            software_versions=software_versions,
            **kwargs
        )

        # Frame of Reference
        has_ref_frame_uid = hasattr(src_img, 'FrameOfReferenceUID')
        if has_ref_frame_uid:
            self.FrameOfReferenceUID = src_img.FrameOfReferenceUID
            self.PositionReferenceIndicator = getattr(
                src_img,
                'PositionReferenceIndicator',
                None
            )
            # Using Container Type Code Sequence attribute would be more
            # elegant, but unfortunately it is a type 2 attribute.
            if (hasattr(src_img, 'ImageOrientationSlide') or
                    hasattr(src_img, 'ImageCenterPointCoordinatesSequence')):
                self._coordinate_system: Optional[CoordinateSystemNames] = \
                    CoordinateSystemNames.SLIDE
            else:
                self._coordinate_system = CoordinateSystemNames.PATIENT
        else:
            # Only allow missing FrameOfReferenceUID if it is not required
            # for this IOD
            usage = get_module_usage('frame-of-reference', src_img.SOPClassUID)
            if usage == ModuleUsageValues.MANDATORY:
                raise ValueError(
                    "Source images have no Frame Of Reference UID, but it is "
                    "required by the IOD."
                )

            # It may be possible to generalize this, but for now only a single
            # source frame is permitted when no frame of reference exists
            if (
                len(source_images) > 1 or
                (is_multiframe and src_img.NumberOfFrames > 1)
            ):
                raise ValueError(
                    "Only a single frame is supported when the source "
                    "image has no Frame of Reference UID."
                )
            if plane_positions is not None:
                raise TypeError(
                    "If source images have no Frame Of Reference UID, the "
                    'argument "plane_positions" may not be specified since the '
                    "segmentation pixel array must be spatially aligned with "
                    "the source images."
                )
            if plane_orientation is not None:
                raise TypeError(
                    "If source images have no Frame Of Reference UID, the "
                    'argument "plane_orientation" may not be specified since '
                    "the segmentation pixel array must be spatially aligned "
                    "with the source images."
                )
            self._coordinate_system = None

        # General Reference

        # Note that appending directly to the SourceImageSequence is typically
        # slow so it's more efficient to build as a Python list then convert
        # later. We save conversion for after the main loop
        source_image_seq: List[Dataset] = []
        referenced_series: Dict[str, List[Dataset]] = defaultdict(list)
        for s_img in source_images:
            ref = Dataset()
            ref.ReferencedSOPClassUID = s_img.SOPClassUID
            ref.ReferencedSOPInstanceUID = s_img.SOPInstanceUID
            source_image_seq.append(ref)
            referenced_series[s_img.SeriesInstanceUID].append(ref)
        self.SourceImageSequence = source_image_seq

        # Common Instance Reference
        ref_image_seq: List[Dataset] = []
        for series_instance_uid, referenced_images in referenced_series.items():
            ref = Dataset()
            ref.SeriesInstanceUID = series_instance_uid
            ref.ReferencedInstanceSequence = referenced_images
            ref_image_seq.append(ref)
        self.ReferencedSeriesSequence = ref_image_seq

        # Image Pixel
        if tile_pixel_array:
            # By default use the same tile size as the source image (even if
            # they are not spatially aligned)
            tile_size = tile_size or (src_img.Rows, src_img.Columns)
            self.Rows, self.Columns = (tile_size)
        else:
            self.Rows = pixel_array.shape[1]
            self.Columns = pixel_array.shape[2]

        # Segmentation Image
        self.ImageType = ['DERIVED', 'PRIMARY']
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = 'MONOCHROME2'
        self.PixelRepresentation = 0

        if content_label is not None:
            _check_code_string(content_label)
            self.ContentLabel = content_label
        else:
            self.ContentLabel = f'{src_img.Modality}_SEG'
        self.ContentDescription = content_description
        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name
        if content_creator_identification is not None:
            if not isinstance(
                content_creator_identification,
                ContentCreatorIdentificationCodeSequence
            ):
                raise TypeError(
                    'Argument "content_creator_identification" must be of type '
                    'ContentCreatorIdentificationCodeSequence.'
                )
            self.ContentCreatorIdentificationCodeSequence = \
                content_creator_identification

        segmentation_type = SegmentationTypeValues(segmentation_type)
        self.SegmentationType = segmentation_type.value
        if self.SegmentationType == SegmentationTypeValues.BINARY.value:
            self.BitsAllocated = 1
            self.HighBit = 0
            if self.file_meta.TransferSyntaxUID.is_encapsulated:
                raise ValueError(
                    'The chosen transfer syntax '
                    f'{self.file_meta.TransferSyntaxUID} '
                    'is not compatible with the BINARY segmentation type'
                )
        elif self.SegmentationType == SegmentationTypeValues.FRACTIONAL.value:
            self.BitsAllocated = 8
            self.HighBit = 7
            segmentation_fractional_type = SegmentationFractionalTypeValues(
                fractional_type
            )
            self.SegmentationFractionalType = segmentation_fractional_type.value
            if max_fractional_value > 2**8:
                raise ValueError(
                    'Maximum fractional value must not exceed image bit depth.'
                )
            self.MaximumFractionalValue = max_fractional_value
        else:
            raise ValueError(
                'Unknown segmentation type "{}"'.format(segmentation_type)
            )

        self.BitsStored = self.BitsAllocated
        self.LossyImageCompression = getattr(
            src_img,
            'LossyImageCompression',
            '00'
        )
        if self.LossyImageCompression == '01':
            self.LossyImageCompressionRatio = \
                src_img.LossyImageCompressionRatio
            self.LossyImageCompressionMethod = \
                src_img.LossyImageCompressionMethod

        # Multi-Resolution Pyramid
        if pyramid_uid is not None:
            if not is_tiled:
                raise TypeError(
                    'Argument "pyramid_uid" should only be specified '
                    'for tiled images.'
                )
            if (
                self._coordinate_system is None or
                self._coordinate_system != CoordinateSystemNames.SLIDE
            ):
                raise TypeError(
                    'Argument "pyramid_uid" should only be specified for '
                    'segmentations in the SLIDE coordinate system.'
                )
            self.PyramidUID = pyramid_uid

            if pyramid_label is not None:
                _check_long_string(pyramid_label)
                self.PyramidLabel = pyramid_label

        elif pyramid_label is not None:
            raise TypeError(
                'Argument "pyramid_label" should not be specified if '
                '"pyramid_uid" is not specified.'
            )

        # Multi-Frame Functional Groups and Multi-Frame Dimensions
        sffg_item = Dataset()
        source_pixel_measures = self._get_pixel_measures_sequence(
            source_image=src_img,
            is_multiframe=is_multiframe,
        )
        if pixel_measures is None:
            pixel_measures = source_pixel_measures

        if has_ref_frame_uid:
            if self._coordinate_system == CoordinateSystemNames.SLIDE:
                source_plane_orientation = PlaneOrientationSequence(
                    coordinate_system=self._coordinate_system,
                    image_orientation=src_img.ImageOrientationSlide
                )
            else:
                if is_multiframe:
                    src_sfg = src_img.SharedFunctionalGroupsSequence[0]
                    source_plane_orientation = deepcopy(
                        src_sfg.PlaneOrientationSequence
                    )
                else:
                    source_plane_orientation = PlaneOrientationSequence(
                        coordinate_system=self._coordinate_system,
                        image_orientation=src_img.ImageOrientationPatient
                    )

            if plane_orientation is None:
                plane_orientation = source_plane_orientation

        self.DimensionIndexSequence = DimensionIndexSequence(
            coordinate_system=self._coordinate_system
        )
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = \
            self.DimensionIndexSequence[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence = [dimension_organization]

        if has_ref_frame_uid:
            if is_multiframe:
                source_plane_positions = \
                    self.DimensionIndexSequence.get_plane_positions_of_image(
                        src_img
                    )
            else:
                source_plane_positions = \
                    self.DimensionIndexSequence.get_plane_positions_of_series(
                        source_images
                    )

        if pixel_measures is not None:
            sffg_item.PixelMeasuresSequence = pixel_measures
        if (
            self._coordinate_system is not None and
            self._coordinate_system == CoordinateSystemNames.PATIENT
        ):
            sffg_item.PlaneOrientationSequence = plane_orientation
        self.SharedFunctionalGroupsSequence = [sffg_item]

        # Check segment numbers
        described_segment_numbers = np.array([
            int(item.SegmentNumber)
            for item in segment_descriptions
        ])
        self._check_segment_numbers(described_segment_numbers)
        number_of_segments = len(described_segment_numbers)
        self.SegmentSequence = segment_descriptions

        # Checks on pixels and overlap
        pixel_array, segments_overlap = self._check_and_cast_pixel_array(
            pixel_array,
            number_of_segments,
            segmentation_type,
        )
        self.SegmentsOverlap = segments_overlap.value

        if has_ref_frame_uid:
            if tile_pixel_array:

                if plane_positions is None:
                    # Use the origin of the source image
                    origin_seq = src_img.TotalPixelMatrixOriginSequence[0]
                    x_offset = origin_seq.XOffsetInSlideCoordinateSystem
                    y_offset = origin_seq.YOffsetInSlideCoordinateSystem
                else:
                    # Use the provided image origin
                    pp = plane_positions[0][0]
                    rp = pp.RowPositionInTotalImagePixelMatrix
                    cp = pp.ColumnPositionInTotalImagePixelMatrix
                    if rp != 1 or cp != 1:
                        raise ValueError(
                            "When specifying a single plane position when "
                            'the "tile_pixel_array" argument is True, the '
                            "plane position must be at the top left corner "
                            "of the total pixel matrix. I.e. it must have "
                            "RowPositionInTotalImagePixelMatrix and "
                            "ColumnPositionInTotalImagePixelMatrix equal to 1."
                        )
                    x_offset = pp.XOffsetInSlideCoordinateSystem
                    y_offset = pp.YOffsetInSlideCoordinateSystem
                orientation = plane_orientation[0].ImageOrientationSlide

                plane_positions = [
                    compute_plane_position_tiled_full(
                        row_index=r,
                        column_index=c,
                        x_offset=x_offset,
                        y_offset=y_offset,
                        rows=self.Rows,
                        columns=self.Columns,
                        image_orientation=orientation,
                        pixel_spacing=pixel_measures[0].PixelSpacing,
                    )
                    for c, r in tile_pixel_matrix(
                        total_pixel_matrix_rows=pixel_array.shape[1],
                        total_pixel_matrix_columns=pixel_array.shape[2],
                        rows=self.Rows,
                        columns=self.Columns,
                    )
                ]

            else:
                if plane_positions is None:
                    if pixel_array.shape[0] != len(source_plane_positions):
                        raise ValueError(
                            'Number of plane positions in source image(s) does '
                            'not match size of first dimension of '
                            '"pixel_array" argument.'
                        )
                    plane_positions = source_plane_positions
                else:
                    if pixel_array.shape[0] != len(plane_positions):
                        raise ValueError(
                            'Number of PlanePositionSequence items provided '
                            'via "plane_positions" argument does not match '
                            'size of first dimension of "pixel_array" argument.'
                        )

            # plane_position_values is an array giving, for each plane of
            # the input array, the raw values of all attributes that
            # describe its position. The first dimension is sorted the same
            # way as the input pixel array and the second is sorted the
            # same way as the dimension index sequence (without segment
            # number) plane_sort_index is a list of indices into the input
            # planes giving the order in which they should be arranged to
            # correctly sort them for inclusion into the segmentation
            plane_position_values, plane_sort_index = \
                self.DimensionIndexSequence.get_index_values(
                    plane_positions
                )

            are_spatial_locations_preserved = (
                all(
                    plane_positions[i] == source_plane_positions[i]
                    for i in range(len(plane_positions))
                ) and
                plane_orientation == source_plane_orientation and
                pixel_measures == source_pixel_measures
            )

        else:
            # Only one spatial location supported
            plane_positions = [None]
            plane_position_values = [None]
            plane_sort_index = np.array([0])
            are_spatial_locations_preserved = True

        if are_spatial_locations_preserved:
            if tile_pixel_array:
                if (
                    pixel_array.shape[1:3] !=
                    (
                        src_img.TotalPixelMatrixRows,
                        src_img.TotalPixelMatrixColumns
                    )
                ):
                    raise ValueError(
                        "Shape of input pixel_array does not match shape of "
                        "the total pixel matrix of the source image."
                    )
            else:
                if pixel_array.shape[1:3] != (src_img.Rows, src_img.Columns):
                    raise ValueError(
                        "Shape of input pixel_array does not match shape of "
                        "the source image."
                    )

        # Dimension Organization Type
        dimension_organization_type = self._check_dimension_organization_type(
            dimension_organization_type=dimension_organization_type,
            is_tiled=is_tiled,
            are_spatial_locations_preserved=are_spatial_locations_preserved,
            omit_empty_frames=omit_empty_frames,
            source_image=src_img,
            plane_positions=plane_positions,
            rows=self.Rows,
            columns=self.Columns,
        )
        if dimension_organization_type is not None:
            self.DimensionOrganizationType = dimension_organization_type.value

        # Find indices such that empty planes are removed
        if omit_empty_frames:
            if tile_pixel_array:
                included_plane_indices, is_empty = \
                    self._get_nonempty_tile_indices(
                        pixel_array,
                        plane_positions=plane_positions,
                        rows=self.Rows,
                        columns=self.Columns,
                    )
            else:
                included_plane_indices, is_empty = \
                    self._get_nonempty_plane_indices(pixel_array)
            if is_empty:
                # Cannot omit empty frames when all frames are empty
                omit_empty_frames = False
                included_plane_indices = list(range(len(plane_positions)))
            else:
                # Remove all empty plane positions from the list of sorted
                # plane position indices
                included_plane_indices_set = set(included_plane_indices)
                plane_sort_index = [
                    ind for ind in plane_sort_index
                    if ind in included_plane_indices_set
                ]
        else:
            included_plane_indices = list(range(len(plane_positions)))

        if has_ref_frame_uid:
            # Get unique values of attributes in the Plane Position Sequence or
            # Plane Position Slide Sequence, which define the position of the
            # plane with respect to the three dimensional patient or slide
            # coordinate system, respectively. These can subsequently be used
            # to look up the relative position of a plane relative to the
            # indexed dimension.
            unique_dimension_values = [
                np.unique(
                    plane_position_values[included_plane_indices, index],
                    axis=0
                )
                for index in range(plane_position_values.shape[1])
            ]
        else:
            unique_dimension_values = [None]

        if (
            has_ref_frame_uid and
            self._coordinate_system == CoordinateSystemNames.SLIDE
        ):
            total_pixel_matrix_size = (
                pixel_array.shape[1:3] if tile_pixel_array else None
            )
            self._add_slide_coordinate_metadata(
                source_image=src_img,
                plane_orientation=plane_orientation,
                plane_position_values=plane_position_values,
                pixel_measures=pixel_measures,
                are_spatial_locations_preserved=are_spatial_locations_preserved,
                is_tiled=is_tiled,
                total_pixel_matrix_size=total_pixel_matrix_size,
            )

        is_encaps = self.file_meta.TransferSyntaxUID.is_encapsulated
        process_pool: Optional[Executor] = None

        if not isinstance(workers, (int, Executor)):
            raise TypeError(
                'Argument "workers" must be of type int or '
                'concurrent.futures.Executor (or a derived class).'
            )
        using_multiprocessing = (
            isinstance(workers, Executor) or workers != 0
        )

        # List of frames. In the case of native transfer syntaxes, we will
        # collect a list of frames as flattened NumPy arrays for bitpacking at
        # the end. In the case of encapsulated transfer syntaxes with no
        # workers, we will accumulate a list of encoded frames to encapsulate
        # at the end
        frames: Union[List[bytes], List[np.ndarray]] = []

        if is_encaps:
            if using_multiprocessing:
                # In the case of encapsulated transfer syntaxes with multiple
                # workers, we will accumulate a list of encoded frames to
                # encapsulate at the end
                frame_futures: List[Future] = []

                # Use the existing executor or create one
                if isinstance(workers, Executor):
                    process_pool = workers
                else:
                    # If workers is negative, pass None to use all processors
                    process_pool = ProcessPoolExecutor(
                        workers if workers > 0 else None
                    )

            # Parameters to use when calling the encode_frame function in
            # either of the above two cases
            encode_frame_kwargs = dict(
                transfer_syntax_uid=self.file_meta.TransferSyntaxUID,
                bits_allocated=self.BitsAllocated,
                bits_stored=self.BitsStored,
                photometric_interpretation=self.PhotometricInterpretation,
                pixel_representation=self.PixelRepresentation
            )
        else:
            if using_multiprocessing:
                warnings.warn(
                    "Setting workers != 0 or passing an instance of "
                    "concurrent.futures.Executor when using a non-encapsulated "
                    "transfer syntax has no effect.",
                    UserWarning
                )
                using_multiprocessing = False

        # Information about individual frames is placed into the
        # PerFrameFunctionalGroupsSequence. Note that a *very* significant
        # efficiency gain is observed when building this as a Python list
        # rather than a pydicom sequence, and then converting to a pydicom
        # sequence at the end
        pffg_sequence: List[Dataset] = []

        for segment_number in described_segment_numbers:
            for plane_index in plane_sort_index:

                if tile_pixel_array:
                    pos = plane_positions[plane_index][0]
                    plane_array = get_tile_array(
                        pixel_array[0],
                        row_offset=pos.RowPositionInTotalImagePixelMatrix,
                        column_offset=pos.ColumnPositionInTotalImagePixelMatrix,
                        tile_rows=self.Rows,
                        tile_columns=self.Columns,
                    )
                else:
                    # Select the relevant existing frame
                    plane_array = pixel_array[plane_index]

                # Pixel array for just this segment and this position
                segment_array = self._get_segment_pixel_array(
                    plane_array,
                    segment_number=segment_number,
                    number_of_segments=number_of_segments,
                    segmentation_type=segmentation_type,
                    max_fractional_value=max_fractional_value,
                )

                # Even though completely empty planes were removed earlier,
                # there may still be planes in which this specific segment is
                # absent. Such frames should be removed
                if (
                    omit_empty_frames and not
                    np.any(segment_array)
                ):
                    logger.debug(
                        f'skip empty plane {plane_index} of segment '
                        f'#{segment_number}'
                    )
                    continue
                logger.debug(
                    f'add plane #{plane_index} for segment #{segment_number}'
                )

                # Get the item of the PerFrameFunctionalGroupsSequence for this
                # segmentation frame
                if has_ref_frame_uid:
                    plane_pos_val = plane_position_values[plane_index]
                    try:
                        dimension_index_values = (
                            self._get_dimension_index_values(
                                unique_dimension_values=unique_dimension_values,
                                plane_position_value=plane_pos_val,
                                coordinate_system=self._coordinate_system,
                            )
                        )
                    except IndexError as error:
                        raise IndexError(
                            'Could not determine position of plane '
                            f'#{plane_index} in three dimensional coordinate '
                            f'system based on dimension index values: {error}'
                        )
                else:
                    dimension_index_values = []

                if (
                    dimension_organization_type !=
                    DimensionOrganizationTypeValues.TILED_FULL
                ):
                    # No per-frame functional group for TILED FULL
                    pffg_item = self._get_pffg_item(
                        segment_number=segment_number,
                        dimension_index_values=dimension_index_values,
                        plane_position=plane_positions[plane_index],
                        source_images=source_images,
                        source_image_index=plane_index,
                        are_spatial_locations_preserved=are_spatial_locations_preserved,  # noqa: E501
                        has_ref_frame_uid=has_ref_frame_uid,
                        coordinate_system=self._coordinate_system,
                    )
                    pffg_sequence.append(pffg_item)

                # Add the segmentation pixel array for this frame to the list
                if is_encaps:
                    if process_pool is None:
                        # Encode this frame and add resulting bytes to the list
                        # for encapsulation at the end
                        frames.append(
                            encode_frame(
                                segment_array,
                                **encode_frame_kwargs,
                            )
                        )
                    else:
                        # Submit this frame for encoding this frame and add the
                        # future to the list for encapsulation at the end
                        future = process_pool.submit(
                            encode_frame,
                            array=segment_array,
                            **encode_frame_kwargs,
                        )
                        frame_futures.append(future)
                else:
                    # Concatenate the 1D array for encoding at the end
                    frames.append(segment_array.flatten())

        if (
            dimension_organization_type !=
            DimensionOrganizationTypeValues.TILED_FULL
        ):
            self.PerFrameFunctionalGroupsSequence = pffg_sequence

        if is_encaps:
            if process_pool is not None:
                frames = [
                    fut.result() for fut in frame_futures
                ]

                # Shutdown the pool if we created it, otherwise it is the
                # caller's responsibility
                if process_pool is not workers:
                    process_pool.shutdown()

            # Encapsulate all pre-compressed frames
            self.NumberOfFrames = len(frames)
            self.PixelData = encapsulate(frames)
        else:
            # Encode the whole pixel array at once
            # This allows for correct bit-packing in cases where
            # number of pixels per frame is not a multiple of 8
            self.NumberOfFrames = len(frames)
            self.PixelData = self._encode_pixels_native(
                np.concatenate(frames)
            )

        # Add a null trailing byte if required
        if len(self.PixelData) % 2 == 1:
            self.PixelData += b'0'

        self.copy_specimen_information(src_img)
        self.copy_patient_and_study_information(src_img)

        # Build lookup tables for efficient decoding
        self._build_luts()

    def add_segments(
        self,
        pixel_array: np.ndarray,
        segment_descriptions: Sequence[SegmentDescription],
        plane_positions: Optional[Sequence[PlanePositionSequence]] = None,
        omit_empty_frames: bool = True,
    ) -> None:
        """To ensure correctness of segmentation images, this
        method was deprecated in highdicom 0.8.0. For more information
        and migration instructions see :ref:`here <add-segments-deprecation>`.

        """  # noqa E510
        raise AttributeError(
            'To ensure correctness of segmentation images, the add_segments '
            'method was deprecated in highdicom 0.8.0. For more information '
            'and migration instructions visit '
            'https://highdicom.readthedocs.io/en/latest/release_notes.html'
            '#deprecation-of-add-segments-method'
        )

    @staticmethod
    def _check_segment_numbers(described_segment_numbers: np.ndarray):
        """Checks on segment numbers extracted from the segment descriptions.

        Segment numbers should start at 1 and increase by 1. This method checks
        this and raises an appropriate exception for the user if the segment
        numbers are incorrect.

        Parameters
        ----------
        described_segment_numbers: np.ndarray
            The segment numbers from the segment descriptions, in the order
            they were passed. 1D array of integers.

        Raises
        ------
        ValueError
            If the ``described_segment_numbers`` do not have the required values

        """
        # Check segment numbers in the segment descriptions are
        # monotonically increasing by 1
        if not (np.diff(described_segment_numbers) == 1).all():
            raise ValueError(
                'Segment descriptions must be sorted by segment number '
                'and monotonically increasing by 1.'
            )
        if described_segment_numbers[0] != 1:
            raise ValueError(
                'Segment descriptions should be numbered starting '
                f'from 1. Found {described_segment_numbers[0]}. '
            )

    @staticmethod
    def _get_pixel_measures_sequence(
        source_image: Dataset,
        is_multiframe: bool,
    ) -> Optional[PixelMeasuresSequence]:
        """Get a Pixel Measures Sequence from the source image.

        This is a helper method used in the constructor.

        Parameters
        ----------
        source_image: pydicom.Dataset
            The first source image.
        is_multiframe: bool
            Whether the source image is multiframe.

        Returns
        -------
        Union[highdicom.PixelMeasuresSequence, None]
            A PixelMeasuresSequence derived from the source image, if this is
            possible. Otherwise None.

        """
        if is_multiframe:
            src_shared_fg = source_image.SharedFunctionalGroupsSequence[0]
            pixel_measures = src_shared_fg.PixelMeasuresSequence
        else:
            if hasattr(source_image, 'FrameOfReferenceUID'):
                pixel_measures = PixelMeasuresSequence(
                    pixel_spacing=source_image.PixelSpacing,
                    slice_thickness=source_image.SliceThickness,
                    spacing_between_slices=source_image.get(
                        'SpacingBetweenSlices',
                        None
                    )
                )
            else:
                pixel_spacing = getattr(source_image, 'PixelSpacing', None)
                if pixel_spacing is not None:
                    pixel_measures = PixelMeasuresSequence(
                        pixel_spacing=pixel_spacing,
                        slice_thickness=source_image.get(
                            'SliceThickness',
                            None
                        ),
                        spacing_between_slices=source_image.get(
                            'SpacingBetweenSlices',
                            None
                        )
                    )
                else:
                    pixel_measures = None

        return pixel_measures

    def _add_slide_coordinate_metadata(
        self,
        source_image: Dataset,
        plane_orientation: PlaneOrientationSequence,
        plane_position_values: np.ndarray,
        pixel_measures: PixelMeasuresSequence,
        are_spatial_locations_preserved: bool,
        is_tiled: bool,
        total_pixel_matrix_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Add metadata related to the slide coordinate system.

        This is a helper method used in the constructor.

        Parameters
        ----------
        source_image: pydicom.Dataset
            The source image (assumed to be a single source image).
        plane_orientation: highdicom.PlaneOrientationSequence
            Plane orientation sequence for the segmentation.
        plane_position_values: numpy.ndarray
            Plane positions of each plane.
        pixel_measures: highdicom.PixelMeasuresSequence
            PixelMeasuresSequence for the segmentation.
        are_spatial_locations_preserved: bool
            Whether spatial locations are preserved between the source image
            and the segmentation.
        is_tiled: bool
            Whether the source image is a tiled image.
        total_pixel_matrix_size: Optional[Tuple[int, int]]
            Size (rows, columns) of the total pixel matrix, if known. If None,
            this will be deduced from the specified plane position values.
            Explicitly providing the total pixel matrix size is required if the
            total pixel matrix is smaller than the total area covered by the
            provided tiles (i.e. the provided plane positions are padded).

        """
        plane_position_names = self.DimensionIndexSequence.get_index_keywords()

        self.ImageOrientationSlide = deepcopy(
            plane_orientation[0].ImageOrientationSlide
        )
        if are_spatial_locations_preserved and is_tiled:
            self.TotalPixelMatrixOriginSequence = deepcopy(
                source_image.TotalPixelMatrixOriginSequence
            )
            self.TotalPixelMatrixRows = source_image.TotalPixelMatrixRows
            self.TotalPixelMatrixColumns = source_image.TotalPixelMatrixColumns
            self.TotalPixelMatrixFocalPlanes = 1
        elif are_spatial_locations_preserved and not is_tiled:
            self.ImageCenterPointCoordinatesSequence = deepcopy(
                source_image.ImageCenterPointCoordinatesSequence
            )
        else:
            row_index = plane_position_names.index(
                'RowPositionInTotalImagePixelMatrix'
            )
            row_offsets = plane_position_values[:, row_index]
            col_index = plane_position_names.index(
                'ColumnPositionInTotalImagePixelMatrix'
            )
            col_offsets = plane_position_values[:, col_index]
            frame_indices = np.lexsort([row_offsets, col_offsets])
            first_frame_index = frame_indices[0]
            last_frame_index = frame_indices[-1]
            x_index = plane_position_names.index(
                'XOffsetInSlideCoordinateSystem'
            )
            x_origin = plane_position_values[first_frame_index, x_index]
            y_index = plane_position_names.index(
                'YOffsetInSlideCoordinateSystem'
            )
            y_origin = plane_position_values[first_frame_index, y_index]
            z_index = plane_position_names.index(
                'ZOffsetInSlideCoordinateSystem'
            )
            z_origin = plane_position_values[first_frame_index, z_index]

            if is_tiled:
                origin_item = Dataset()
                origin_item.XOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(x_origin)
                origin_item.YOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(y_origin)
                self.TotalPixelMatrixOriginSequence = [origin_item]
                self.TotalPixelMatrixFocalPlanes = 1
                if total_pixel_matrix_size is None:
                    self.TotalPixelMatrixRows = int(
                        plane_position_values[last_frame_index, row_index] +
                        self.Rows - 1
                    )
                    self.TotalPixelMatrixColumns = int(
                        plane_position_values[last_frame_index, col_index] +
                        self.Columns - 1
                    )
                else:
                    self.TotalPixelMatrixRows = total_pixel_matrix_size[0]
                    self.TotalPixelMatrixColumns = total_pixel_matrix_size[1]
            else:
                transform = ImageToReferenceTransformer(
                    image_position=(x_origin, y_origin, z_origin),
                    image_orientation=plane_orientation,
                    pixel_spacing=pixel_measures[0].PixelSpacing
                )
                center_image_coordinates = np.array(
                    [[self.Columns / 2, self.Rows / 2]],
                    dtype=float
                )
                center_reference_coordinates = transform(
                    center_image_coordinates
                )
                x_center = center_reference_coordinates[0, 0]
                y_center = center_reference_coordinates[0, 1]
                z_center = center_reference_coordinates[0, 2]
                center_item = Dataset()
                center_item.XOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(x_center)
                center_item.YOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(y_center)
                center_item.ZOffsetInSlideCoordinateSystem = \
                    format_number_as_ds(z_center)
                self.ImageCenterPointCoordinatesSequence = [center_item]

    @staticmethod
    def _check_dimension_organization_type(
        dimension_organization_type: Union[
            DimensionOrganizationTypeValues,
            str,
            None,
        ],
        is_tiled: bool,
        are_spatial_locations_preserved: bool,
        omit_empty_frames: bool,
        source_image: Dataset,
        plane_positions: Sequence[PlanePositionSequence],
        rows: int,
        columns: int,
    ) -> Optional[DimensionOrganizationTypeValues]:
        """Checks that the specified Dimension Organization Type is valid.

        Parameters
        ----------
        dimension_organization_type: Union[highdicom.enum.DimensionOrganizationTypeValues, str, None]
           The specified DimensionOrganizationType for the output Segmentation.
        is_tiled: bool
            Whether the source image is a tiled image.
        are_spatial_locations_preserved: bool
            Whether spatial locations are preserved between the source image
            and the segmentation pixel array.
        omit_empty_frames: bool
            Whether it was specified to omit empty frames.
        source_image: pydicom.Dataset
            Representative dataset of the source images.
        plane_positions: Sequence[highdicom.PlanePositionSequence]
            Plane positions of all frames.
        rows: int
            Number of rows in each frame of the segmentation image.
        columns: int
            Number of columns in each frame of the segmentation image.

        Returns
        -------
        Optional[highdicom.enum.DimensionOrganizationTypeValues]:
            DimensionOrganizationType to use for the output Segmentation.

        """  # noqa: E501
        if is_tiled and dimension_organization_type is None:
            dimension_organization_type = \
                DimensionOrganizationTypeValues.TILED_SPARSE

        if dimension_organization_type is not None:
            dimension_organization_type = DimensionOrganizationTypeValues(
                dimension_organization_type
            )
            tiled_dimension_organization_types = [
                DimensionOrganizationTypeValues.TILED_SPARSE,
                DimensionOrganizationTypeValues.TILED_FULL
            ]

            if (
                dimension_organization_type in
                tiled_dimension_organization_types
            ):
                if not is_tiled:
                    raise ValueError(
                        f"A value of {dimension_organization_type.value} "
                        'for parameter "dimension_organization_type" is '
                        'only valid if the source images are tiled.'
                    )

            if (
                dimension_organization_type ==
                DimensionOrganizationTypeValues.TILED_FULL
            ):
                if not are_plane_positions_tiled_full(
                    plane_positions,
                    rows,
                    columns,
                ):
                    raise ValueError(
                        'A value of "TILED_FULL" for parameter '
                        '"dimension_organization_type" is not permitted unless '
                        'the "plane_positions" of the segmentation do not '
                        'do not follow the relevant requirements. See '
                        'https://dicom.nema.org/medical/dicom/current/output/'
                        'chtml/part03/sect_C.7.6.17.3.html#sect_C.7.6.17.3.'
                    )
                if omit_empty_frames:
                    raise ValueError(
                        'Parameter "omit_empty_frames" should be False if '
                        'using "dimension_organization_type" of "TILED_FULL".'
                    )

        return dimension_organization_type

    @staticmethod
    def _check_and_cast_pixel_array(
        pixel_array: np.ndarray,
        number_of_segments: int,
        segmentation_type: SegmentationTypeValues
    ) -> Tuple[np.ndarray, SegmentsOverlapValues]:
        """Checks on the shape and data type of the pixel array.

        Also checks for overlapping segments and returns the result.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            The segmentation pixel array.
        number_of_segments: int
            The segment numbers from the segment descriptions, in the order
            they were passed. 1D array of integers.
        segmentation_type: highdicom.seg.SegmentationTypeValues
            The segmentation_type parameter.

        Returns
        -------
        pixel_array: numpyp.ndarray
            Input pixel array with the data type simplified if possible.
        segments_overlap: highdicom.seg.SegmentationOverlaps
            The value for the SegmentationOverlaps attribute, inferred from the
            pixel array.

        """
        if pixel_array.ndim == 4:
            # Check that the number of segments in the array matches
            if pixel_array.shape[-1] != number_of_segments:
                raise ValueError(
                    'The number of segments in last dimension of the pixel '
                    f'array ({pixel_array.shape[-1]}) does not match the '
                    'number of described segments '
                    f'({number_of_segments}).'
                )

        if pixel_array.dtype in (np.bool_, np.uint8, np.uint16):
            max_pixel = pixel_array.max()

            if pixel_array.ndim == 3:
                # A label-map style array where pixel values represent
                # segment associations

                # The pixel values in the pixel array must all belong to
                # a described segment
                if max_pixel > number_of_segments:
                    raise ValueError(
                        'Pixel array contains segments that lack '
                        'descriptions.'
                    )

                # By construction of the pixel array, we know that the segments
                # cannot overlap
                segments_overlap = SegmentsOverlapValues.NO
            else:
                # Pixel array is 4D where each segment is stacked down
                # the last dimension
                # In this case, each segment of the pixel array should be binary
                if max_pixel > 1:
                    raise ValueError(
                        'When passing a 4D stack of segments with an integer '
                        'pixel type, the pixel array must be binary.'
                    )

                # Need to check whether or not segments overlap
                if max_pixel == 0:
                    # Empty segments can't overlap (this skips an unnecessary
                    # further test)
                    segments_overlap = SegmentsOverlapValues.NO
                elif pixel_array.shape[-1] == 1:
                    # A single segment does not overlap
                    segments_overlap = SegmentsOverlapValues.NO
                else:
                    sum_over_segments = pixel_array.sum(axis=-1)
                    if np.any(sum_over_segments > 1):
                        segments_overlap = SegmentsOverlapValues.YES
                    else:
                        segments_overlap = SegmentsOverlapValues.NO

        elif pixel_array.dtype in (np.float_, np.float32, np.float64):
            unique_values = np.unique(pixel_array)
            if np.min(unique_values) < 0.0 or np.max(unique_values) > 1.0:
                raise ValueError(
                    'Floating point pixel array values must be in the '
                    'range [0, 1].'
                )
            if segmentation_type == SegmentationTypeValues.BINARY:
                non_boolean_values = np.logical_and(
                    unique_values > 0.0,
                    unique_values < 1.0
                )
                if np.any(non_boolean_values):
                    raise ValueError(
                        'Floating point pixel array values must be either '
                        '0.0 or 1.0 in case of BINARY segmentation type.'
                    )
                pixel_array = pixel_array.astype(np.uint8)

                # Need to check whether or not segments overlap
                if len(unique_values) == 1 and unique_values[0] == 0.0:
                    # All pixels are zero: there can be no overlap
                    segments_overlap = SegmentsOverlapValues.NO
                elif pixel_array.ndim == 3 or pixel_array.shape[-1] == 1:
                    # A single segment does not overlap
                    segments_overlap = SegmentsOverlapValues.NO
                elif pixel_array.sum(axis=-1).max() > 1:
                    segments_overlap = SegmentsOverlapValues.YES
                else:
                    segments_overlap = SegmentsOverlapValues.NO
            else:
                if (pixel_array.ndim == 3) or (pixel_array.shape[-1] == 1):
                    # A single segment does not overlap
                    segments_overlap = SegmentsOverlapValues.NO
                else:
                    # A truly fractional segmentation with multiple segments.
                    # Unclear how overlap should be interpreted in this case
                    segments_overlap = SegmentsOverlapValues.UNDEFINED
        else:
            raise TypeError('Pixel array has an invalid data type.')

        return pixel_array, segments_overlap

    @staticmethod
    def _get_nonempty_plane_indices(
        pixel_array: np.ndarray
    ) -> Tuple[List[int], bool]:
        """Get a list of all indices of original planes that are non-empty.

        Empty planes (without any positive pixels in any of the segments) do
        not need to be included in the segmentation image. This method finds a
        list of indices of the input frames that are non-empty, and therefore
        should be included in the segmentation image.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Segmentation pixel array

        Returns
        -------
        included_plane_indices : List[int]
            List giving for each plane position in the resulting segmentation
            image the index of the corresponding frame in the original pixel
            array.
        is_empty: bool
            Whether the entire image is empty. If so, empty frames should not
            be omitted.

        """
        # This list tracks which source image each non-empty frame came from
        source_image_indices = [
            i for i, frm in enumerate(pixel_array)
            if np.any(frm)
        ]

        if len(source_image_indices) == 0:
            logger.warning(
                'Encoding an empty segmentation with "omit_empty_frames" '
                'set to True. Reverting to encoding all frames since omitting '
                'all frames is not possible.'
            )
            return (list(range(pixel_array.shape[0])), True)

        return (source_image_indices, False)

    @staticmethod
    def _get_nonempty_tile_indices(
        pixel_array: np.ndarray,
        plane_positions: Sequence[PlanePositionSequence],
        rows: int,
        columns: int,
    ) -> Tuple[List[int], bool]:
        """Get a list of all indices of tile locations that are non-empty.

        This is similar to _get_nonempty_plane_indices, but works on a total
        pixel matrix rather than a set of frames. Empty planes (without any
        positive pixels in any of the segments) do not need to be included in
        the segmentation image. This method finds a list of indices of the
        input frames that are non-empty, and therefore should be included in
        the segmentation image.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Segmentation pixel array
        plane_positions: Sequence[highdicom.PlanePositionSequence]
            Plane positions of each tile.
        rows: int
            Number of rows in each tile.
        columns: int
            Number of columns in each tile.

        Returns
        -------
        included_plane_indices : List[int]
            List giving for each plane position in the resulting segmentation
            image the index of the corresponding frame in the original pixel
            array.
        is_empty: bool
            Whether the entire image is empty. If so, empty frames should not
            be omitted.

        """
        # This list tracks which source image each non-empty frame came from
        source_image_indices = [
            i for i, pos in enumerate(plane_positions)
            if np.any(
                get_tile_array(
                    pixel_array[0],
                    row_offset=pos[0].RowPositionInTotalImagePixelMatrix,
                    column_offset=pos[0].ColumnPositionInTotalImagePixelMatrix,
                    tile_rows=rows,
                    tile_columns=columns,
                )
            )
        ]

        if len(source_image_indices) == 0:
            logger.warning(
                'Encoding an empty segmentation with "omit_empty_frames" '
                'set to True. Reverting to encoding all frames since omitting '
                'all frames is not possible.'
            )
            return (list(range(len(plane_positions))), True)

        return (source_image_indices, False)

    @staticmethod
    def _get_segment_pixel_array(
        pixel_array: np.ndarray,
        segment_number: int,
        number_of_segments: int,
        segmentation_type: SegmentationTypeValues,
        max_fractional_value: int
    ) -> np.ndarray:
        """Get pixel data array for a specific segment and plane.

        This is a helper method used during the constructor. Note that the
        pixel array is expected to have been processed using the
        ``_check_and_cast_pixel_array`` method before being passed to this
        method.

        Parameters
        ----------
        pixel_array: numpy.ndarray
            Segmentation pixel array containing all segments for a single plane.
            Array is therefore either (Rows x Columns x Segments) or (Rows x
            Columns) in case of a "label map" style array.
        segment_number: int
            The segment of interest.
        number_of_segments: int
            Number of segments in the the segmentation.
        segmentation_type: highdicom.seg.SegmentationTypeValues
            Desired output segmentation type.
        max_fractional_value: int
            Value for scaling FRACTIONAL segmentations.

        Returns
        -------
        numpy.ndarray:
            Pixel data array consisting of pixel data for a single segment for
            a single plane. Output array has dtype np.uint8 and binary values
            (0 or 1).

        """
        if pixel_array.dtype in (np.float_, np.float32, np.float64):
            # Based on the previous checks and casting, if we get here the
            # output is a FRACTIONAL segmentation Floating-point numbers must
            # be mapped to 8-bit integers in the range [0,
            # max_fractional_value].
            if pixel_array.ndim == 3:
                segment_array = pixel_array[:, :, segment_number - 1]
            else:
                segment_array = pixel_array
            segment_array = np.around(
                segment_array * float(max_fractional_value)
            )
            segment_array = segment_array.astype(np.uint8)
        else:
            if pixel_array.ndim == 2:
                # "Label maps" that must be converted to binary masks.
                if number_of_segments == 1:
                    # We wish to avoid unnecessary comparison or casting
                    # operations here, for efficiency reasons. If there is only
                    # a single segment, the label map pixel array is already
                    # correct
                    if pixel_array.dtype != np.uint8:
                        segment_array = pixel_array.astype(np.uint8)
                    else:
                        segment_array = pixel_array
                else:
                    segment_array = (
                        pixel_array == segment_number
                    ).astype(np.uint8)
            else:
                segment_array = pixel_array[:, :, segment_number - 1]
                if segment_array.dtype != np.uint8:
                    segment_array = segment_array.astype(np.uint8)

            # It may happen that a binary valued array is passed that should be
            # stored as a fractional segmentation. In this case, we also need
            # to stretch pixel values to 8-bit unsigned integer range by
            # multiplying with the maximum fractional value.
            if segmentation_type == SegmentationTypeValues.FRACTIONAL:
                # Avoid an unnecessary multiplication operation if max
                # fractional value is 1
                if int(max_fractional_value) != 1:
                    segment_array *= int(max_fractional_value)

        return segment_array

    @staticmethod
    def _get_dimension_index_values(
        unique_dimension_values: List[np.ndarray],
        plane_position_value: np.ndarray,
        coordinate_system: Optional[CoordinateSystemNames],
    ) -> List[int]:
        """Get Dimension Index Values for a frame.

        The Dimension Index Values are a list of integer indices that describe
        the position of a frame as indices along each of the dimensions of
        the Dimension Index Sequence. See
        :class:`highdicom.seg.DimensionIndexSequence`.

        Parameters
        ----------
        unique_dimension_values: List[numpy.ndarray]
            List of arrays containing, for each dimension in the dimension
            index sequence (except ReferencedSegment), the sorted unique
            values of all planes along that dimension. Each array in the list
            corresponds to one dimension, and has shape (N x m) where N is the
            number of unique values for that dimension and m is the
            multiplicity of values for that dimension.
        plane_position_value: numpy.ndarray
            Plane position of the plane. This is a 1D or 2D array containing
            each of the raw values for this plane of the attributes listed as
            dimension index pointers (except ReferencedSegment). For dimension
            indices where the value multiplicity of all attributes is 1, the
            array will be 1D. If the value multiplicity of attributes is
            greater than 1, these values are stacked along the second
            dimension.
        coordinate_system: Optional[highdicom.CoordinateSystemNames]
            The type of coordinate system used (if any).

        Returns
        -------
        dimension_index_values: List[int]
            The dimension index values (except the segment number) for the
            given plane.

        """
        # Look up the position of the plane relative to the indexed
        # dimension.
        if (
            coordinate_system ==
            CoordinateSystemNames.SLIDE
        ):
            index_values = [
                int(
                    np.where(
                        (unique_dimension_values[idx] == pos)
                    )[0][0] + 1
                )
                for idx, pos in enumerate(plane_position_value)
            ]
        else:
            # In case of the patient coordinate system, the
            # value of the attribute the Dimension Index
            # Sequence points to (Image Position Patient) has a
            # value multiplicity greater than one.
            index_values = [
                int(
                    np.where(
                        (unique_dimension_values[idx] == pos).all(
                            axis=1
                        )
                    )[0][0] + 1
                )
                for idx, pos in enumerate(plane_position_value)
            ]

        return index_values

    @staticmethod
    def _get_pffg_item(
        segment_number: int,
        dimension_index_values: List[int],
        plane_position: PlanePositionSequence,
        source_images: List[Dataset],
        source_image_index: int,
        are_spatial_locations_preserved: bool,
        has_ref_frame_uid: bool,
        coordinate_system: Optional[CoordinateSystemNames],
    ) -> Dataset:
        """Get a single item of the Per Frame Functional Groups Sequence.

        This is a helper method used in the constructor.

        Parameters
        ----------
        segment_number: int
            Segment number of this segmentation frame.
        dimension_index_values: List[int]
            Dimension index values (except segment number) for this frame.
        plane_position: highdicom.seg.PlanePositionSequence
            Plane position of this frame.
        source_images: List[Dataset]
            Full list of source images.
        source_image_index: int
            Index of this frame in the original list of source images.
        are_spatial_locations_preserved: bool
            Whether spatial locations are preserved between the segmentation
            and the source images.
        has_ref_frame_uid: bool
            Whether the sources images have a frame of reference UID.
        coordinate_system: Optional[highdicom.CoordinateSystemNames]
            Coordinate system used, if any.

        Returns
        -------
        pydicom.Dataset
            Dataset representing the item of the
            Per Frame Functional Groups Sequence for this segmentation frame.

        """
        pffg_item = Dataset()
        frame_content_item = Dataset()

        frame_content_item.DimensionIndexValues = (
            [int(segment_number)] + dimension_index_values
        )
        pffg_item.FrameContentSequence = [frame_content_item]
        if has_ref_frame_uid:
            if coordinate_system == CoordinateSystemNames.SLIDE:
                pffg_item.PlanePositionSlideSequence = plane_position
            else:
                pffg_item.PlanePositionSequence = plane_position

        # Determining the source images that map to the frame is not
        # always trivial. Since DerivationImageSequence is a type 2
        # attribute, we leave its value empty.
        pffg_item.DerivationImageSequence = []

        if are_spatial_locations_preserved:
            derivation_image_item = Dataset()
            derivation_code = codes.cid7203.Segmentation
            derivation_image_item.DerivationCodeSequence = [
                CodedConcept.from_code(derivation_code)
            ]

            derivation_src_img_item = Dataset()
            if hasattr(source_images[0], 'NumberOfFrames'):
                # A single multi-frame source image
                src_img_item = source_images[0]
                # Frame numbers are one-based
                derivation_src_img_item.ReferencedFrameNumber = (
                    source_image_index + 1
                )
            else:
                # Multiple single-frame source images
                src_img_item = source_images[source_image_index]
            derivation_src_img_item.ReferencedSOPClassUID = \
                src_img_item.SOPClassUID
            derivation_src_img_item.ReferencedSOPInstanceUID = \
                src_img_item.SOPInstanceUID
            purpose_code = \
                codes.cid7202.SourceImageForImageProcessingOperation
            derivation_src_img_item.PurposeOfReferenceCodeSequence = [
                CodedConcept.from_code(purpose_code)
            ]
            derivation_src_img_item.SpatialLocationsPreserved = 'YES'
            derivation_image_item.SourceImageSequence = [
                derivation_src_img_item,
            ]
            pffg_item.DerivationImageSequence.append(
                derivation_image_item
            )
        else:
            logger.debug('spatial locations not preserved')

        identification = Dataset()
        identification.ReferencedSegmentNumber = int(segment_number)
        pffg_item.SegmentIdentificationSequence = [
            identification,
        ]

        return pffg_item

    def _encode_pixels_native(self, planes: np.ndarray) -> bytes:
        """Encode pixel planes using a native transfer syntax.

        Parameters
        ----------
        planes: numpy.ndarray
            Array representing one or more segmentation image planes. If
            multiple image planes, planes stacked down the first dimension
            (index 0).

        Returns
        -------
        bytes
            Encoded pixels

        """
        if self.SegmentationType == SegmentationTypeValues.BINARY.value:
            return pack_bits(planes)
        else:
            return planes.tobytes()

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> 'Segmentation':
        """Create instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Segmentation image.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.seg.Segmentation
            Representation of the supplied dataset as a highdicom
            Segmentation.

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        _check_little_endian(dataset)
        # Checks on integrity of input dataset
        if dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.66.4':
            raise ValueError('Dataset is not a Segmentation.')
        if copy:
            seg = deepcopy(dataset)
        else:
            seg = dataset
        seg.__class__ = Segmentation

        sf_groups = seg.SharedFunctionalGroupsSequence[0]
        if hasattr(seg, 'PlaneOrientationSequence'):
            plane_ori_seq = sf_groups.PlaneOrientationSequence[0]
            if hasattr(plane_ori_seq, 'ImageOrientationSlide'):
                seg._coordinate_system = CoordinateSystemNames.SLIDE
            elif hasattr(plane_ori_seq, 'ImageOrientationPatient'):
                seg._coordinate_system = CoordinateSystemNames.PATIENT
            else:
                seg._coordinate_system = None
        else:
            seg._coordinate_system = None

        for i, segment in enumerate(seg.SegmentSequence, 1):
            if segment.SegmentNumber != i:
                raise AttributeError(
                    'Segments are expected to start at 1 and be consecutive '
                    'integers.'
                )

        for i, s in enumerate(seg.SegmentSequence, 1):
            if s.SegmentNumber != i:
                raise ValueError(
                    'Segment numbers in the segmentation image must start at '
                    '1 and increase by 1 with the segments sequence.'
                )

        # Convert contained items to highdicom types
        # Segment descriptions
        seg.SegmentSequence = [
            SegmentDescription.from_dataset(ds, copy=False)
            for ds in seg.SegmentSequence
        ]

        # Shared functional group elements
        if hasattr(sf_groups, 'PlanePositionSequence'):
            plane_pos = PlanePositionSequence.from_sequence(
                sf_groups.PlanePositionSequence,
                copy=False,
            )
            sf_groups.PlanePositionSequence = plane_pos
        if hasattr(sf_groups, 'PlaneOrientationSequence'):
            plane_ori = PlaneOrientationSequence.from_sequence(
                sf_groups.PlaneOrientationSequence,
                copy=False,
            )
            sf_groups.PlaneOrientationSequence = plane_ori
        if hasattr(sf_groups, 'PixelMeasuresSequence'):
            pixel_measures = PixelMeasuresSequence.from_sequence(
                sf_groups.PixelMeasuresSequence,
                copy=False,
            )
            sf_groups.PixelMeasuresSequence = pixel_measures

        # Per-frame functional group items
        if hasattr(seg, 'PerFrameFunctionalGroupsSequence'):
            for pffg_item in seg.PerFrameFunctionalGroupsSequence:
                if hasattr(pffg_item, 'PlanePositionSequence'):
                    plane_pos = PlanePositionSequence.from_sequence(
                        pffg_item.PlanePositionSequence,
                        copy=False
                    )
                    pffg_item.PlanePositionSequence = plane_pos
                if hasattr(pffg_item, 'PlaneOrientationSequence'):
                    plane_ori = PlaneOrientationSequence.from_sequence(
                        pffg_item.PlaneOrientationSequence,
                        copy=False,
                    )
                    pffg_item.PlaneOrientationSequence = plane_ori
                if hasattr(pffg_item, 'PixelMeasuresSequence'):
                    pixel_measures = PixelMeasuresSequence.from_sequence(
                        pffg_item.PixelMeasuresSequence,
                        copy=False,
                    )
                    pffg_item.PixelMeasuresSequence = pixel_measures

        seg._build_luts()

        return cast(Segmentation, seg)

    def _get_ref_instance_uids(self) -> List[Tuple[str, str, str]]:
        """List all instances referenced in the segmentation.

        Returns
        -------
        List[Tuple[str, str, str]]
            List of all instances referenced in the segmentation in the format
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
                f"Segmentation SOP Instance UID: '{self.SOPInstanceUID}', "
                f'duplicated referenced SOP Instance UID items: {display_str}.'
            )

        return unique_instance_data

    def _build_luts(self) -> None:
        """Build lookup tables for efficient querying.

        Two lookup tables are currently constructed. The first maps the
        SOPInstanceUIDs of all datasets referenced in the segmentation to a
        tuple containing the StudyInstanceUID, SeriesInstanceUID and
        SOPInstanceUID.

        The second look-up table contains information about each frame of the
        segmentation, including the segment it contains, the instance and frame
        from which it was derived (if these are unique), and its dimension
        index values.

        """
        referenced_uids = self._get_ref_instance_uids()
        all_referenced_sops = {uids[2] for uids in referenced_uids}

        is_tiled_full = (
            hasattr(self, 'DimensionOrganizationType') and
            self.DimensionOrganizationType == 'TILED_FULL'
        )

        segment_numbers = []

        # Get list of all dimension index pointers, excluding the segment
        # number, since this is treated differently
        seg_num_tag = tag_for_keyword('ReferencedSegmentNumber')
        self._dim_ind_pointers = [
            dim_ind.DimensionIndexPointer
            for dim_ind in self.DimensionIndexSequence
            if dim_ind.DimensionIndexPointer != seg_num_tag
        ]

        func_grp_pointers = {}
        for dim_ind in self.DimensionIndexSequence:
            ptr = dim_ind.DimensionIndexPointer
            if ptr in self._dim_ind_pointers:
                grp_ptr = getattr(dim_ind, "FunctionalGroupPointer", None)
                func_grp_pointers[ptr] = grp_ptr

        dim_ind_positions = {
            dim_ind.DimensionIndexPointer: i
            for i, dim_ind in enumerate(self.DimensionIndexSequence)
            if dim_ind.DimensionIndexPointer != seg_num_tag
        }
        dim_indices: Dict[int, List[int]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }
        dim_values: Dict[int, List[Any]] = {
            ptr: [] for ptr in self._dim_ind_pointers
        }

        self._single_source_frame_per_seg_frame = True

        if is_tiled_full:
            # With TILED_FULL, there is no PerFrameFunctionalGroupsSequence,
            # so we have to deduce the per-frame information
            row_tag = tag_for_keyword('RowPositionInTotalImagePixelMatrix')
            col_tag = tag_for_keyword('ColumnPositionInTotalImagePixelMatrix')
            x_tag = tag_for_keyword('XOffsetInSlideCoordinateSystem')
            y_tag = tag_for_keyword('YOffsetInSlideCoordinateSystem')
            z_tag = tag_for_keyword('ZOffsetInSlideCoordinateSystem')
            tiled_full_dim_indices = {row_tag, col_tag, x_tag, y_tag, z_tag}
            if len(set(dim_indices.keys()) - tiled_full_dim_indices) > 0:
                raise RuntimeError(
                    'Expected segmentation images with '
                    '"DimensionOrganizationType" of "TILED_FULL" are expected '
                    'to have the following dimension index pointers: '
                    'SegmentNumber, RowPositionInTotalImagePixelMatrix, '
                    'ColumnPositionInTotalImagePixelMatrix.'
                )
            self._single_source_frame_per_seg_frame = False
            (
                segment_numbers,
                _,
                dim_values[col_tag],
                dim_values[row_tag],
                dim_values[x_tag],
                dim_values[y_tag],
                dim_values[z_tag],
            ) = zip(*iter_tiled_full_frame_data(self))

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
                # Get segment number for this frame
                seg_id_seg = frame_item.SegmentIdentificationSequence[0]
                seg_num = seg_id_seg.ReferencedSegmentNumber
                segment_numbers.append(int(seg_num))

                # Get dimension indices for this frame
                content_seq = frame_item.FrameContentSequence[0]
                indices = content_seq.DimensionIndexValues
                if not isinstance(indices, (MultiValue, list)):
                    # In case there is a single dimension index
                    indices = [indices]
                if len(indices) != len(self._dim_ind_pointers) + 1:
                    # (+1 because referenced segment number is ignored)
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

                frame_source_instances = []
                frame_source_frames = []
                for der_im in frame_item.DerivationImageSequence:
                    for src_im in der_im.SourceImageSequence:
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
                    self._single_source_frame_per_seg_frame = False
                else:
                    ref_instance_uid = frame_source_instances[0]
                    if ref_instance_uid not in all_referenced_sops:
                        raise AttributeError(
                            f'SOP instance {ref_instance_uid} referenced in '
                            'the source image sequence is not included in the '
                            'Referenced Series Sequence or Studies Containing '
                            'Other Referenced Instances Sequence. This is an '
                            'error with the integrity of the Segmentation '
                            'object.'
                        )
                    referenced_instances.append(ref_instance_uid)
                    referenced_frames.append(frame_source_frames[0])

            # Summarise
            if any(
                isinstance(v, SpatialLocationsPreservedValues) and
                v == SpatialLocationsPreservedValues.NO
                for v in locations_preserved
            ):
                Type = Optional[SpatialLocationsPreservedValues]
                self._locations_preserved: Type = \
                    SpatialLocationsPreservedValues.NO
            elif all(
                isinstance(v, SpatialLocationsPreservedValues) and
                v == SpatialLocationsPreservedValues.YES
                for v in locations_preserved
            ):
                self._locations_preserved = SpatialLocationsPreservedValues.YES
            else:
                self._locations_preserved = None

            if not self._single_source_frame_per_seg_frame:
                referenced_instances = None
                referenced_frames = None

        self._db_man = _SegDBManager(
            referenced_uids=referenced_uids,
            segment_numbers=segment_numbers,
            dim_indices=dim_indices,
            dim_values=dim_values,
            referenced_instances=referenced_instances,
            referenced_frames=referenced_frames,
        )

    @property
    def segmentation_type(self) -> SegmentationTypeValues:
        """highdicom.seg.SegmentationTypeValues: Segmentation type."""
        return SegmentationTypeValues(self.SegmentationType)

    @property
    def segmentation_fractional_type(
        self
    ) -> Union[SegmentationFractionalTypeValues, None]:
        """
        highdicom.seg.SegmentationFractionalTypeValues:
            Segmentation fractional type.

        """
        if not hasattr(self, 'SegmentationFractionalType'):
            return None
        return SegmentationFractionalTypeValues(
            self.SegmentationFractionalType
        )

    def iter_segments(self):
        """Iterates over segments in this segmentation image.

        Returns
        -------
        Iterator[Tuple[numpy.ndarray, Tuple[pydicom.dataset.Dataset, ...], pydicom.dataset.Dataset]]
            For each segment in the Segmentation image instance, provides the
            Pixel Data frames representing the segment, items of the Per-Frame
            Functional Groups Sequence describing the individual frames, and
            the item of the Segment Sequence describing the segment

        """  # noqa
        return iter_segments(self)

    @property
    def number_of_segments(self) -> int:
        """int: The number of segments in this SEG image."""
        return len(self.SegmentSequence)

    @property
    def segment_numbers(self) -> range:
        """range: The segment numbers present in the SEG image as a range."""
        return range(1, self.number_of_segments + 1)

    def get_segment_description(
        self,
        segment_number: int
    ) -> SegmentDescription:
        """Get segment description for a segment.

        Parameters
        ----------
        segment_number: int
            Segment number for the segment, as a 1-based index.

        Returns
        -------
        highdicom.seg.SegmentDescription
            Description of the given segment.

        """
        if segment_number < 1 or segment_number > self.number_of_segments:
            raise IndexError(
                f'{segment_number} is an invalid segment number for this '
                'dataset.'
            )
        return self.SegmentSequence[segment_number - 1]

    def get_segment_numbers(
        self,
        segment_label: Optional[str] = None,
        segmented_property_category: Optional[Union[Code, CodedConcept]] = None,
        segmented_property_type: Optional[Union[Code, CodedConcept]] = None,
        algorithm_type: Optional[Union[SegmentAlgorithmTypeValues, str]] = None,
        tracking_uid: Optional[str] = None,
        tracking_id: Optional[str] = None,
    ) -> List[int]:
        """Get a list of segment numbers matching provided criteria.

        Any number of optional filters may be provided. A segment must match
        all provided filters to be included in the returned list.

        Parameters
        ----------
        segment_label: Union[str, None], optional
            Segment label filter to apply.
        segmented_property_category: Union[Code, CodedConcept, None], optional
            Segmented property category filter to apply.
        segmented_property_type: Union[Code, CodedConcept, None], optional
            Segmented property type filter to apply.
        algorithm_type: Union[SegmentAlgorithmTypeValues, str, None], optional
            Segmented property type filter to apply.
        tracking_uid: Union[str, None], optional
            Tracking unique identifier filter to apply.
        tracking_id: Union[str, None], optional
            Tracking identifier filter to apply.

        Returns
        -------
        List[int]
            List of all segment numbers matching the provided criteria.

        Examples
        --------

        Get segment numbers of all segments that both represent tumors and were
        generated by an automatic algorithm from a segmentation object ``seg``:

        >>> from pydicom.sr.codedict import codes
        >>> from highdicom.seg import SegmentAlgorithmTypeValues, Segmentation
        >>> from pydicom import dcmread
        >>> ds = dcmread('data/test_files/seg_image_sm_control.dcm')
        >>> seg = Segmentation.from_dataset(ds)
        >>> segment_numbers = seg.get_segment_numbers(
        ...     segmented_property_type=codes.SCT.ConnectiveTissue,
        ...     algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC
        ... )
        >>> segment_numbers
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        Get segment numbers of all segments identified by a given
        institution-specific tracking ID:

        >>> segment_numbers = seg.get_segment_numbers(
        ...     tracking_id='Segment #4'
        ... )
        >>> segment_numbers
        [4]

        Get segment numbers of all segments identified a globally unique
        tracking UID:

        >>> uid = '1.2.826.0.1.3680043.8.498.42540123542017542395135803252098380233'
        >>> segment_numbers = seg.get_segment_numbers(tracking_uid=uid)
        >>> segment_numbers
        [13]

        """  # noqa: E501
        filter_funcs = []
        if segment_label is not None:
            filter_funcs.append(
                lambda desc: desc.segment_label == segment_label
            )
        if segmented_property_category is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_category == segmented_property_category
            )
        if segmented_property_type is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_type == segmented_property_type
            )
        if algorithm_type is not None:
            algo_type = SegmentAlgorithmTypeValues(algorithm_type)
            filter_funcs.append(
                lambda desc:
                SegmentAlgorithmTypeValues(desc.algorithm_type) == algo_type
            )
        if tracking_uid is not None:
            filter_funcs.append(
                lambda desc: desc.tracking_uid == tracking_uid
            )
        if tracking_id is not None:
            filter_funcs.append(
                lambda desc: desc.tracking_id == tracking_id
            )

        return [
            desc.segment_number
            for desc in self.SegmentSequence
            if all(f(desc) for f in filter_funcs)
        ]

    def get_tracking_ids(
        self,
        segmented_property_category: Optional[Union[Code, CodedConcept]] = None,
        segmented_property_type: Optional[Union[Code, CodedConcept]] = None,
        algorithm_type: Optional[Union[SegmentAlgorithmTypeValues, str]] = None
    ) -> List[Tuple[str, UID]]:
        """Get all unique tracking identifiers in this SEG image.

        Any number of optional filters may be provided. A segment must match
        all provided filters to be included in the returned list.

        The tracking IDs and the accompanying tracking UIDs are returned
        in a list of tuples.

        Note that the order of the returned list is not significant and will
        not in general match the order of segments.

        Parameters
        ----------
        segmented_property_category: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Segmented property category filter to apply.
        segmented_property_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Segmented property type filter to apply.
        algorithm_type: Union[highdicom.seg.SegmentAlgorithmTypeValues, str, None], optional
            Segmented property type filter to apply.

        Returns
        -------
        List[Tuple[str, pydicom.uid.UID]]
            List of all unique (Tracking Identifier, Unique Tracking Identifier)
            tuples that are referenced in segment descriptions in this
            Segmentation image that match all provided filters.

        Examples
        --------

        Read in an example segmentation image in the highdicom test data:

        >>> import highdicom as hd
        >>> from pydicom.sr.codedict import codes
        >>>
        >>> seg = hd.seg.segread('data/test_files/seg_image_ct_binary_overlap.dcm')

        List the tracking IDs and UIDs present in the segmentation image:

        >>> sorted(seg.get_tracking_ids(), reverse=True)  # otherwise its a random order
        [('Spine', '1.2.826.0.1.3680043.10.511.3.10042414969629429693880339016394772'), ('Bone', '1.2.826.0.1.3680043.10.511.3.83271046815894549094043330632275067')]

        >>> for seg_num in seg.segment_numbers:
        ...     desc = seg.get_segment_description(seg_num)
        ...     print(desc.segmented_property_type.meaning)
        Bone
        Spine

        List tracking IDs only for those segments with a segmented property
        category of 'Spine':

        >>> seg.get_tracking_ids(segmented_property_type=codes.SCT.Spine)
        [('Spine', '1.2.826.0.1.3680043.10.511.3.10042414969629429693880339016394772')]

        """  # noqa: E501
        filter_funcs = []
        if segmented_property_category is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_category == segmented_property_category
            )
        if segmented_property_type is not None:
            filter_funcs.append(
                lambda desc:
                desc.segmented_property_type == segmented_property_type
            )
        if algorithm_type is not None:
            algo_type = SegmentAlgorithmTypeValues(algorithm_type)
            filter_funcs.append(
                lambda desc:
                SegmentAlgorithmTypeValues(desc.algorithm_type) == algo_type
            )

        return list({
            (desc.tracking_id, UID(desc.tracking_uid))
            for desc in self.SegmentSequence
            if desc.tracking_id is not None and
            desc.tracking_uid is not None and
            all(f(desc) for f in filter_funcs)
        })

    @property
    def segmented_property_categories(self) -> List[CodedConcept]:
        """Get all unique segmented property categories in this SEG image.

        Returns
        -------
        List[CodedConcept]
            All unique segmented property categories referenced in segment
            descriptions in this SEG image.

        """
        categories = []
        for desc in self.SegmentSequence:
            if desc.segmented_property_category not in categories:
                categories.append(desc.segmented_property_category)

        return categories

    @property
    def segmented_property_types(self) -> List[CodedConcept]:
        """Get all unique segmented property types in this SEG image.

        Returns
        -------
        List[CodedConcept]
            All unique segmented property types referenced in segment
            descriptions in this SEG image.

        """
        types = []
        for desc in self.SegmentSequence:
            if desc.segmented_property_type not in types:
                types.append(desc.segmented_property_type)

        return types

    def _get_pixels_by_seg_frame(
        self,
        output_shape: Union[int, Tuple[int, int]],
        indices_iterator: Iterator[
            Tuple[
                Tuple[Union[slice, int], ...],
                Tuple[Union[slice, int], ...],
                int
            ]
        ],
        segment_numbers: np.ndarray,
        combine_segments: bool = False,
        relabel: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: Union[type, str, np.dtype, None] = None,
    ) -> np.ndarray:
        """Construct a segmentation array given an array of frame numbers.

        The output array is either 4D (combine_segments=False) or 3D
        (combine_segments=True), where dimensions are frames x rows x columns x
        segments.

        Parameters
        ----------
        output_shape: Union[int, Tuple[int, int]]
            Shape of the output array. If an integer is False, this is the
            number of frames in the output array and the number of rows and
            columns are taken to match those of each segmentation frame. If a
            tuple of integers, it contains the number of (rows, columns) in the
            output array and there is no frame dimension (this is the tiled
            case). Note in either case, the segments dimension (if relevant) is
            omitted.
        indices_iterator: Iterator[Tuple[Tuple[Union[slice, int], ...], Tuple[Union[slice, int], ...], int ]]
            An iterable object that yields tuples of (output_indexer,
            segmentation_indexer, output_segment_number) that describes how to
            construct the desired output pixel array from the segmentation
            image's pixel array. 'output_indexer' is a tuple that may be used
            directly to index the output array to place a single frame's pixels
            into the output array. Similarly 'segmentation_indexer' is a tuple
            that may be used directly to index the segmentation pixel array
            to retrieve the pixels to place into the output array.
            with as segment number 'output_segment_number'. Note that in both
            cases the indexers access the frame, row and column dimensions of
            the relevant array, but not the segment dimension (if relevant).
        segment_numbers: np.ndarray
            One dimensional numpy array containing segment numbers
            corresponding to the columns of the seg frames matrix.
        combine_segments: bool
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, np.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.

        Returns
        -------
        pixel_array: np.ndarray
            Segmentation pixel array

        """  # noqa: E501
        if (
            segment_numbers.min() < 1 or
            segment_numbers.max() > self.number_of_segments
        ):
            raise ValueError(
                'Segment numbers array contains invalid values.'
            )

        # Determine output type
        if combine_segments:
            max_output_val = (
                segment_numbers.shape[0] if relabel else segment_numbers.max()
            )
        else:
            max_output_val = 1

        will_be_rescaled = (
            rescale_fractional and
            self.segmentation_type == SegmentationTypeValues.FRACTIONAL and
            not combine_segments
        )
        if dtype is None:
            if will_be_rescaled:
                dtype = np.float32
            else:
                dtype = _get_unsigned_dtype(max_output_val)
        dtype = np.dtype(dtype)

        # Check dtype is suitable
        if dtype.kind not in ('u', 'i', 'f'):
            raise ValueError(
                f'Data type "{dtype}" is not suitable.'
            )

        if will_be_rescaled:
            intermediate_dtype = np.uint8
            if dtype.kind != 'f':
                raise ValueError(
                    'If rescaling a fractional segmentation, the output dtype '
                    'must be a floating-point type.'
                )
        else:
            intermediate_dtype = dtype
        _check_numpy_value_representation(max_output_val, dtype)

        num_segments = len(segment_numbers)
        if self.pixel_array.ndim == 2:
            h, w = self.pixel_array.shape
        else:
            _, h, w = self.pixel_array.shape

        if combine_segments:
            # Check whether segmentation is binary, or fractional with only
            # binary values
            if self.segmentation_type == SegmentationTypeValues.FRACTIONAL:
                if not rescale_fractional:
                    raise ValueError(
                        'In order to combine segments of a FRACTIONAL '
                        'segmentation image, argument "rescale_fractional" '
                        'must be set to True.'
                    )
                # Combining fractional segs is only possible if there are
                # two unique values in the array: 0 and MaximumFractionalValue
                is_binary = np.isin(
                    np.unique(self.pixel_array),
                    np.array([0, self.MaximumFractionalValue]),
                    assume_unique=True
                ).all()
                if not is_binary:
                    raise ValueError(
                        'Combining segments of a FRACTIONAL segmentation is '
                        'only possible if the pixel array contains only 0s '
                        'and the specified MaximumFractionalValue '
                        f'({self.MaximumFractionalValue}).'
                    )
                pixel_array = self.pixel_array // self.MaximumFractionalValue
                pixel_array = pixel_array.astype(np.uint8)
            else:
                pixel_array = self.pixel_array

            if pixel_array.ndim == 2:
                pixel_array = pixel_array[None, :, :]

            # Initialize empty pixel array
            full_output_shape = (
                output_shape
                if isinstance(output_shape, tuple)
                else (output_shape, h, w)
            )
            out_array = np.zeros(
                full_output_shape,
                dtype=intermediate_dtype
            )

            # Loop over the supplied iterable
            for (output_indexer, seg_indexer, seg_n) in indices_iterator:
                pix_value = intermediate_dtype.type(seg_n)

                if not skip_overlap_checks:
                    if np.any(
                        np.logical_and(
                            pixel_array[seg_indexer] > 0,
                            out_array[output_indexer] > 0
                        )
                    ):
                        raise RuntimeError(
                            "Cannot combine segments because segments "
                            "overlap."
                        )
                out_array[output_indexer] = np.maximum(
                    pixel_array[seg_indexer] * pix_value,
                    out_array[output_indexer]
                )

        else:
            # Initialize empty pixel array
            full_output_shape = (
                (*output_shape, num_segments)
                if isinstance(output_shape, tuple)
                else (output_shape, h, w, num_segments)
            )
            out_array = np.zeros(
                full_output_shape,
                dtype=intermediate_dtype
            )

            # loop through output frames
            for (output_indexer, seg_indexer, seg_n) in indices_iterator:

                # Output indexer needs segment index
                output_indexer = (*output_indexer, seg_n)

                # Copy data to to output array
                if self.pixel_array.ndim == 2:
                    # Special case with a single segmentation frame
                    out_array[output_indexer] = \
                        self.pixel_array.copy()
                else:
                    out_array[output_indexer] = \
                        self.pixel_array[seg_indexer].copy()

            if rescale_fractional:
                if self.segmentation_type == SegmentationTypeValues.FRACTIONAL:
                    if out_array.max() > self.MaximumFractionalValue:
                        raise RuntimeError(
                            'Segmentation image contains values greater than '
                            'the MaximumFractionalValue recorded in the '
                            'dataset.'
                        )
                    max_val = self.MaximumFractionalValue
                    out_array = out_array.astype(dtype) / max_val

        return out_array

    def get_source_image_uids(self) -> List[Tuple[hd_UID, hd_UID, hd_UID]]:
        """Get UIDs for all source SOP instances referenced in the dataset.

        Returns
        -------
        List[Tuple[highdicom.UID, highdicom.UID, highdicom.UID]]
            List of tuples containing Study Instance UID, Series Instance UID
            and SOP Instance UID for every SOP Instance referenced in the
            dataset.

        """
        return self._db_man.get_source_image_uids()

    def get_default_dimension_index_pointers(
        self
    ) -> List[BaseTag]:
        """Get the default list of tags used to index frames.

        The list of tags used to index dimensions depends upon how the
        segmentation image was constructed, and is stored in the
        DimensionIndexPointer attribute within the DimensionIndexSequence. The
        list returned by this method matches the order of items in the
        DimensionIndexSequence, but omits the ReferencedSegmentNumber
        attribute, since this is handled differently to other tags when
        indexing frames in highdicom.

        Returns
        -------
        List[pydicom.tag.BaseTag]
            List of tags used as the default dimension index pointers.

        """
        return self._dim_ind_pointers[:]

    def are_dimension_indices_unique(
        self,
        dimension_index_pointers: Sequence[Union[int, BaseTag]]
    ) -> bool:
        """Check if a list of index pointers uniquely identifies frames.

        For a given list of dimension index pointers, check whether every
        combination of index values for these pointers identifies a unique
        frame per segment in the segmentation image. This is a pre-requisite
        for indexing using this list of dimension index pointers in the
        :meth:`Segmentation.get_pixels_by_dimension_index_values()` method.

        Parameters
        ----------
        dimension_index_pointers: Sequence[Union[int, pydicom.tag.BaseTag]]
            Sequence of tags serving as dimension index pointers.

        Returns
        -------
        bool
            True if the specified list of dimension index pointers uniquely
            identifies frames in the segmentation image. False otherwise.

        Raises
        ------
        KeyError
            If any of the elements of the ``dimension_index_pointers`` are not
            valid dimension index pointers in this segmentation image.

        """
        if len(dimension_index_pointers) == 0:
            raise ValueError(
                'Argument "dimension_index_pointers" may not be empty.'
            )
        for ptr in dimension_index_pointers:
            if ptr not in self._dim_ind_pointers:
                kw = keyword_for_tag(ptr)
                if kw == '':
                    kw = '<no keyword>'
                raise KeyError(
                    f'Tag {ptr} ({kw}) is not used as a dimension index '
                    'in this image.'
                )
        return self._db_man.are_dimension_indices_unique(
            dimension_index_pointers
        )

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
            * At least one frame in the segmentation lists multiple
              source frames.

        Parameters
        ----------
        ignore_spatial_locations: bool
            Allows the user to ignore whether spatial locations are preserved
            in the frames.

        """
        # Checks that it is possible to index using source frames in this
        # dataset
        is_tiled_full = (
            hasattr(self, 'DimensionOrganizationType') and
            self.DimensionOrganizationType == 'TILED_FULL'
        )
        if is_tiled_full:
            raise RuntimeError(
                'Indexing via source frames is not possible when a '
                'segmentation is stored using the DimensionOrganizationType '
                '"TILED_FULL".'
            )
        elif self._locations_preserved is None:
            if not ignore_spatial_locations:
                raise RuntimeError(
                    'Indexing via source frames is not permissible since this '
                    'image does not specify that spatial locations are '
                    'preserved in the course of deriving the segmentation '
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
                    'in the course of deriving the segmentation from the '
                    'source image. If you do not require that spatial '
                    ' locations are preserved you may override this behavior '
                    "with the 'ignore_spatial_locations' parameter."
                )
        if not self._single_source_frame_per_seg_frame:
            raise RuntimeError(
                'Indexing via source frames is not permissible since some '
                'frames in the segmentation specify multiple source frames.'
            )

    def get_pixels_by_source_instance(
        self,
        source_sop_instance_uids: Sequence[str],
        segment_numbers: Optional[Sequence[int]] = None,
        combine_segments: bool = False,
        relabel: bool = False,
        ignore_spatial_locations: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: Union[type, str, np.dtype, None] = None,
    ) -> np.ndarray:
        """Get a pixel array for a list of source instances.

        This is intended for retrieving segmentation masks derived from
        (series of) single frame source images.

        The output array will have 4 dimensions under the default behavior, and
        3 dimensions if ``combine_segments`` is set to ``True``.  The first
        dimension represents the source instances. ``pixel_array[i, ...]``
        represents the segmentation of ``source_sop_instance_uids[i]``.  The
        next two dimensions are the rows and columns of the frames,
        respectively.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (4th) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        Parameters
        ----------
        source_sop_instance_uids: str
            SOP Instance UID of the source instances to for which segmentations
            are requested.
        segment_numbers: Union[Sequence[int], None], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        ignore_spatial_locations: bool, optional
           Ignore whether or not spatial locations were preserved in the
           derivation of the segmentation frames from the source frames. In
           some segmentation images, the pixel locations in the segmentation
           frames may not correspond to pixel locations in the frames of the
           source image from which they were derived. The segmentation image
           may or may not specify whether or not spatial locations are
           preserved in this way through use of the optional (0028,135A)
           SpatialLocationsPreserved attribute. If this attribute specifies
           that spatial locations are not preserved, or is absent from the
           segmentation image, highdicom's default behavior is to disallow
           indexing by source frames. To override this behavior and retrieve
           segmentation pixels regardless of the presence or value of the
           spatial locations preserved attribute, set this parameter to True.
        assert_missing_frames_are_empty: bool, optional
            Assert that requested source frame numbers that are not referenced
            by the segmentation image contain no segments. If a source frame
            number is not referenced by the segmentation image, highdicom is
            unable to check that the frame number is valid in the source image.
            By default, highdicom will raise an error if any of the requested
            source frames are not referenced in the source image. To override
            this behavior and return a segmentation frame of all zeros for such
            frames, set this parameter to True.
        rescale_fractional: bool, optional
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.

        Returns
        -------
        pixel_array: np.ndarray
            Pixel array representing the segmentation. See notes for full
            explanation.

        Examples
        --------

        Read in an example from the highdicom test data:

        >>> import highdicom as hd
        >>>
        >>> seg = hd.seg.segread('data/test_files/seg_image_ct_binary.dcm')

        List the source images for this segmentation:

        >>> for study_uid, series_uid, sop_uid in seg.get_source_image_uids():
        ...     print(sop_uid)
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.94
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.95
        1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.96

        Get the segmentation array for a subset of these images:

        >>> pixels = seg.get_pixels_by_source_instance(
        ...     source_sop_instance_uids=[
        ...         '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93',
        ...         '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.94'
        ...     ]
        ... )
        >>> pixels.shape
        (2, 16, 16, 1)

        """
        # Check that indexing in this way is possible
        self._check_indexing_with_source_frames(ignore_spatial_locations)

        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )
        if isinstance(source_sop_instance_uids, str):
            raise TypeError(
                'source_sop_instance_uids should be a sequence of UIDs, not a '
                'single UID'
            )
        if len(source_sop_instance_uids) == 0:
            raise ValueError(
                'Source SOP instance UIDs may not be empty.'
            )

        # Check that the combination of source instances and segment numbers
        # uniquely identify segmentation frames
        if not self._db_man.are_referenced_sop_instances_unique():
            raise RuntimeError(
                'Source SOP instance UIDs and segment numbers do not '
                'uniquely identify frames of the segmentation image.'
            )

        # Check that all frame numbers requested actually exist
        if not assert_missing_frames_are_empty:
            unique_uids = self._db_man.get_unique_sop_instance_uids()
            missing_uids = set(source_sop_instance_uids) - unique_uids
            if len(missing_uids) > 0:
                msg = (
                    f'SOP Instance UID(s) {list(missing_uids)} do not match '
                    'any referenced source instances. To return an empty '
                    'segmentation mask in this situation, use the '
                    '"assert_missing_frames_are_empty" parameter.'
                )
                raise KeyError(msg)

        with self._db_man.iterate_indices_by_source_instance(
            source_sop_instance_uids=source_sop_instance_uids,
            segment_numbers=segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel,
        ) as indices:

            return self._get_pixels_by_seg_frame(
                output_shape=len(source_sop_instance_uids),
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
            )

    def get_pixels_by_source_frame(
        self,
        source_sop_instance_uid: str,
        source_frame_numbers: Sequence[int],
        segment_numbers: Optional[Sequence[int]] = None,
        combine_segments: bool = False,
        relabel: bool = False,
        ignore_spatial_locations: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: Union[type, str, np.dtype, None] = None,
    ):
        """Get a pixel array for a list of frames within a source instance.

        This is intended for retrieving segmentation masks derived from
        multi-frame (enhanced) source images. All source frames for
        which segmentations are requested must belong within the same
        SOP Instance UID.

        The output array will have 4 dimensions under the default behavior, and
        3 dimensions if ``combine_segments`` is set to ``True``.  The first
        dimension represents the source frames. ``pixel_array[i, ...]``
        represents the segmentation of ``source_frame_numbers[i]``.  The
        next two dimensions are the rows and columns of the frames,
        respectively.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (4th) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        Parameters
        ----------
        source_sop_instance_uid: str
            SOP Instance UID of the source instance that contains the source
            frames.
        source_frame_numbers: Sequence[int]
            A sequence of frame numbers (1-based) within the source instance
            for which segmentations are requested.
        segment_numbers: Optional[Sequence[int]], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        ignore_spatial_locations: bool, optional
            Ignore whether or not spatial locations were preserved in the
            derivation of the segmentation frames from the source frames. In
            some segmentation images, the pixel locations in the segmentation
            frames may not correspond to pixel locations in the frames of the
            source image from which they were derived. The segmentation image
            may or may not specify whether or not spatial locations are
            preserved in this way through use of the optional (0028,135A)
            SpatialLocationsPreserved attribute. If this attribute specifies
            that spatial locations are not preserved, or is absent from the
            segmentation image, highdicom's default behavior is to disallow
            indexing by source frames. To override this behavior and retrieve
            segmentation pixels regardless of the presence or value of the
            spatial locations preserved attribute, set this parameter to True.
        assert_missing_frames_are_empty: bool, optional
            Assert that requested source frame numbers that are not referenced
            by the segmentation image contain no segments. If a source frame
            number is not referenced by the segmentation image and is larger
            than the frame number of the highest referenced frame, highdicom is
            unable to check that the frame number is valid in the source image.
            By default, highdicom will raise an error in this situation. To
            override this behavior and return a segmentation frame of all zeros
            for such frames, set this parameter to True.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.

        Returns
        -------
        pixel_array: np.ndarray
            Pixel array representing the segmentation. See notes for full
            explanation.

        Examples
        --------

        Read in an example from the highdicom test data derived from a
        multiframe slide microscopy image:

        >>> import highdicom as hd
        >>>
        >>> seg = hd.seg.segread('data/test_files/seg_image_sm_control.dcm')

        List the source image SOP instance UID for this segmentation:

        >>> sop_uid = seg.get_source_image_uids()[0][2]
        >>> sop_uid
        '1.2.826.0.1.3680043.9.7433.3.12857516184849951143044513877282227'

        Get the segmentation array for 3 of the frames in the multiframe source
        image.  The resulting segmentation array has 3 10 x 10 frames, one for
        each source frame. The final dimension contains the 20 different
        segments present in this segmentation.

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6]
        ... )
        >>> pixels.shape
        (3, 10, 10, 20)

        This time, select only 4 of the 20 segments:

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6],
        ...     segment_numbers=[10, 11, 12, 13]
        ... )
        >>> pixels.shape
        (3, 10, 10, 4)

        Instead create a multiclass label map for each source frame. Note
        that segments 6, 8, and 10 are present in the three chosen frames.

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6],
        ...     combine_segments=True
        ... )
        >>> pixels.shape, np.unique(pixels)
        ((3, 10, 10), array([ 0,  6,  8, 10], dtype=uint8))

        Now relabel the segments to give a pixel map with values between 0
        and 3 (inclusive):

        >>> pixels = seg.get_pixels_by_source_frame(
        ...     source_sop_instance_uid=sop_uid,
        ...     source_frame_numbers=[4, 5, 6],
        ...     segment_numbers=[6, 8, 10],
        ...     combine_segments=True,
        ...     relabel=True
        ... )
        >>> pixels.shape, np.unique(pixels)
        ((3, 10, 10), array([0, 1, 2, 3], dtype=uint8))

        """
        # Check that indexing in this way is possible
        self._check_indexing_with_source_frames(ignore_spatial_locations)

        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )

        if len(source_frame_numbers) == 0:
            raise ValueError(
                'Source frame numbers should not be empty.'
            )
        if not all(f > 0 for f in source_frame_numbers):
            raise ValueError(
                'Frame numbers are 1-based indices and must be > 0.'
            )

        # Check that the combination of frame numbers and segment numbers
        # uniquely identify segmentation frames
        if not self._db_man.are_referenced_frames_unique():
            raise RuntimeError(
                'Source frame numbers and segment numbers do not '
                'uniquely identify frames of the segmentation image.'
            )

        # Check that all frame numbers requested actually exist
        if not assert_missing_frames_are_empty:
            max_frame_number = self._db_man.get_max_frame_number()
            for f in source_frame_numbers:
                if f > max_frame_number:
                    msg = (
                        f'Source frame number {f} is larger than any '
                        'referenced source frame, so highdicom cannot be '
                        'certain that it is valid. To return an empty '
                        'segmentation mask in this situation, use the '
                        "'assert_missing_frames_are_empty' parameter."
                    )
                    raise ValueError(msg)

        with self._db_man.iterate_indices_by_source_frame(
            source_sop_instance_uid=source_sop_instance_uid,
            source_frame_numbers=source_frame_numbers,
            segment_numbers=segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel,
        ) as indices:

            return self._get_pixels_by_seg_frame(
                output_shape=len(source_frame_numbers),
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
            )

    def get_pixels_by_dimension_index_values(
        self,
        dimension_index_values: Sequence[Sequence[int]],
        dimension_index_pointers: Optional[Sequence[int]] = None,
        segment_numbers: Optional[Sequence[int]] = None,
        combine_segments: bool = False,
        relabel: bool = False,
        assert_missing_frames_are_empty: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: Union[type, str, np.dtype, None] = None,
    ):
        """Get a pixel array for a list of dimension index values.

        This is intended for retrieving segmentation masks using the index
        values within the segmentation object, without referring to the
        source images from which the segmentation was derived.

        The output array will have 4 dimensions under the default behavior, and
        3 dimensions if ``combine_segments`` is set to ``True``.  The first
        dimension represents the source frames. ``pixel_array[i, ...]``
        represents the segmentation frame with index
        ``dimension_index_values[i]``.  The next two dimensions are the rows
        and columns of the frames, respectively.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (4th) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        Parameters
        ----------
        dimension_index_values: Sequence[Sequence[int]]
            Dimension index values for the requested frames. Each element of
            the sequence is a sequence of 1-based index values representing the
            dimension index values for a single frame of the output
            segmentation. The order of the index values within the inner
            sequence is determined by the ``dimension_index_pointers``
            parameter, and as such the length of each inner sequence must
            match the length of ``dimension_index_pointers`` parameter.
        dimension_index_pointers: Union[Sequence[Union[int, pydicom.tag.BaseTag]], None], optional
            The data element tags that identify the indices used in the
            ``dimension_index_values`` parameter. Each element identifies a
            data element tag by which frames are ordered in the segmentation
            image dataset. If this parameter is set to ``None`` (the default),
            the value of
            :meth:`Segmentation.get_default_dimension_index_pointers()` is
            used. Valid values of this parameter are are determined by
            the construction of the segmentation image and include any
            permutation of any subset of elements in the
            :meth:`Segmentation.get_default_dimension_index_pointers()` list.
        segment_numbers: Union[Sequence[int], None], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        assert_missing_frames_are_empty: bool, optional
            Assert that requested source frame numbers that are not referenced
            by the segmentation image contain no segments. If a source frame
            number is not referenced by the segmentation image, highdicom is
            unable to check that the frame number is valid in the source image.
            By default, highdicom will raise an error if any of the requested
            source frames are not referenced in the source image. To override
            this behavior and return a segmentation frame of all zeros for such
            frames, set this parameter to True.
        rescale_fractional: bool, optional
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the smallest
            unsigned integer type that accommodates all of the output values
            will be chosen.

        Returns
        -------
        pixel_array: np.ndarray
            Pixel array representing the segmentation. See notes for full
            explanation.

        Examples
        --------

        Read a test image of a segmentation of a slide microscopy image

        >>> import highdicom as hd
        >>> from pydicom.datadict import keyword_for_tag, tag_for_keyword
        >>> from pydicom import dcmread
        >>>
        >>> ds = dcmread('data/test_files/seg_image_sm_control.dcm')
        >>> seg = hd.seg.Segmentation.from_dataset(ds)

        Get the default list of dimension index values

        >>> for tag in seg.get_default_dimension_index_pointers():
        ...     print(keyword_for_tag(tag))
        ColumnPositionInTotalImagePixelMatrix
        RowPositionInTotalImagePixelMatrix
        XOffsetInSlideCoordinateSystem
        YOffsetInSlideCoordinateSystem
        ZOffsetInSlideCoordinateSystem


        Use a subset of these index pointers to index the image

        >>> tags = [
        ...     tag_for_keyword('ColumnPositionInTotalImagePixelMatrix'),
        ...     tag_for_keyword('RowPositionInTotalImagePixelMatrix')
        ... ]
        >>> assert seg.are_dimension_indices_unique(tags)  # True

        It is therefore possible to index using just this subset of
        dimension indices

        >>> pixels = seg.get_pixels_by_dimension_index_values(
        ...     dimension_index_pointers=tags,
        ...     dimension_index_values=[[1, 1], [1, 2]]
        ... )
        >>> pixels.shape
        (2, 10, 10, 20)

        """  # noqa: E501
        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )

        if dimension_index_pointers is None:
            dimension_index_pointers = self._dim_ind_pointers
        else:
            if len(dimension_index_pointers) == 0:
                raise ValueError(
                    'Argument "dimension_index_pointers" must not be empty.'
                )
            for ptr in dimension_index_pointers:
                if ptr not in self._dim_ind_pointers:
                    kw = keyword_for_tag(ptr)
                    if kw == '':
                        kw = '<no keyword>'
                    raise KeyError(
                        f'Tag {Tag(ptr)} ({kw}) is not used as a dimension '
                        'index in this image.'
                    )

        if len(dimension_index_values) == 0:
            raise ValueError(
                'Argument "dimension_index_values" must not be empty.'
            )
        for row in dimension_index_values:
            if len(row) != len(dimension_index_pointers):
                raise ValueError(
                    'Dimension index values must be a sequence of sequences of '
                    'integers, with each inner sequence having a single value '
                    'per dimension index pointer specified.'
                )

        if not self.are_dimension_indices_unique(dimension_index_pointers):
            raise RuntimeError(
                'The chosen dimension indices do not uniquely identify '
                'frames of the segmentation image. You may need to provide '
                'further indices to disambiguate.'
            )

        # Check that all frame numbers requested actually exist
        if not assert_missing_frames_are_empty:
            unique_dim_ind_vals = self._db_man.get_unique_dim_index_values(
                dimension_index_pointers
            )
            queried_dim_inds = set(tuple(r) for r in dimension_index_values)
            missing_dim_inds = queried_dim_inds - unique_dim_ind_vals
            if len(missing_dim_inds) > 0:
                msg = (
                    f'Dimension index values {list(missing_dim_inds)} do not '
                    'match any segmentation frame. To return '
                    'an empty segmentation mask in this situation, '
                    "use the 'assert_missing_frames_are_empty' "
                    'parameter.'
                )
                raise ValueError(msg)

        with self._db_man.iterate_indices_by_dimension_index_values(
            dimension_index_values=dimension_index_values,
            dimension_index_pointers=dimension_index_pointers,
            segment_numbers=segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel,
        ) as indices:

            return self._get_pixels_by_seg_frame(
                output_shape=len(dimension_index_values),
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
            )

    def get_total_pixel_matrix(
        self,
        row_start: int = 1,
        row_end: Optional[int] = None,
        column_start: int = 1,
        column_end: Optional[int] = None,
        segment_numbers: Optional[Sequence[int]] = None,
        combine_segments: bool = False,
        relabel: bool = False,
        rescale_fractional: bool = True,
        skip_overlap_checks: bool = False,
        dtype: Union[type, str, np.dtype, None] = None,
    ):
        """Get the pixel array as a (region of) the total pixel matrix.

        This is intended for retrieving segmentation masks derived from
        multi-frame (enhanced) source images that are tiled. The method
        returns (a region of) the 2D total pixel matrix implied by the
        frames within the segmentation.

        The output array will have 3 dimensions under the default behavior, and
        2 dimensions if ``combine_segments`` is set to ``True``. The first two
        dimensions are the rows and columns of the total pixel matrix,
        respectively. By default, the full total pixel matrix is returned,
        however a smaller region may be requested using the ``row_start``,
        ``row_end``, ``column_start`` and ``column_end`` parameters as 1-based
        indices into the total pixel matrix.

        When ``combine_segments`` is ``False`` (the default behavior), the
        segments are stacked down the final (3rd) dimension of the pixel array.
        If ``segment_numbers`` was specified, then ``pixel_array[:, :, i]``
        represents the data for segment ``segment_numbers[i]``. If
        ``segment_numbers`` was unspecified, then ``pixel_array[:, :, i]``
        represents the data for segment ``parser.segment_numbers[i]``. Note
        that in neither case does ``pixel_array[:, :, i]`` represent
        the segmentation data for the segment with segment number ``i``, since
        segment numbers begin at 1 in DICOM.

        When ``combine_segments`` is ``True``, then the segmentation data from
        all specified segments is combined into a multi-class array in which
        pixel value is used to denote the segment to which a pixel belongs.
        This is only possible if the segments do not overlap and either the
        type of the segmentation is ``BINARY`` or the type of the segmentation
        is ``FRACTIONAL`` but all values are exactly 0.0 or 1.0.  the segments
        do not overlap. If the segments do overlap, a ``RuntimeError`` will be
        raised. After combining, the value of a pixel depends upon the
        ``relabel`` parameter. In both cases, pixels that appear in no segments
        with have a value of ``0``.  If ``relabel`` is ``False``, a pixel that
        appears in the segment with segment number ``i`` (according to the
        original segment numbering of the segmentation object) will have a
        value of ``i``. If ``relabel`` is ``True``, the value of a pixel in
        segment ``i`` is related not to the original segment number, but to the
        index of that segment number in the ``segment_numbers`` parameter of
        this method. Specifically, pixels belonging to the segment with segment
        number ``segment_numbers[i]`` is given the value ``i + 1`` in the
        output pixel array (since 0 is reserved for pixels that belong to no
        segments). In this case, the values in the output pixel array will
        always lie in the range ``0`` to ``len(segment_numbers)`` inclusive.

        Parameters
        ----------
        row_start: int, optional
            1-based row index in the total pixel matrix of the first row to
            include in the output array. May be negative, in which case the
            last row is considered index -1.
        row_end: Union[int, None], optional
            1-based row index in the total pixel matrix of the first row beyond
            the last row to include in the output array. A ``row_end`` value of
            ``n`` will include rows ``n - 1`` and below, similar to standard
            Python indexing. If ``None``, rows up until the final row of the
            total pixel matrix are included. May be negative, in which case the
            last row is considered index -1.
        column_start: int, optional
            1-based column index in the total pixel matrix of the first column
            to include in the output array. May be negative, in which case the
            last column is considered index -1.
        column_end: Union[int, None], optional
            1-based column index in the total pixel matrix of the first column
            beyond the last column to include in the output array. A
            ``column_end`` value of ``n`` will include columns ``n - 1`` and
            below, similar to standard Python indexing. If ``None``, columns up
            until the final column of the total pixel matrix are included. May
            be negative, in which case the last column is considered index -1.
        segment_numbers: Optional[Sequence[int]], optional
            Sequence containing segment numbers to include. If unspecified,
            all segments are included.
        combine_segments: bool, optional
            If True, combine the different segments into a single label
            map in which the value of a pixel represents its segment.
            If False (the default), segments are binary and stacked down the
            last dimension of the output array.
        relabel: bool, optional
            If True and ``combine_segments`` is ``True``, the pixel values in
            the output array are relabelled into the range ``0`` to
            ``len(segment_numbers)`` (inclusive) according to the position of
            the original segment numbers in ``segment_numbers`` parameter.  If
            ``combine_segments`` is ``False``, this has no effect.
        rescale_fractional: bool
            If this is a FRACTIONAL segmentation and ``rescale_fractional`` is
            True, the raw integer-valued array stored in the segmentation image
            output will be rescaled by the MaximumFractionalValue such that
            each pixel lies in the range 0.0 to 1.0. If False, the raw integer
            values are returned. If the segmentation has BINARY type, this
            parameter has no effect.
        skip_overlap_checks: bool
            If True, skip checks for overlap between different segments. By
            default, checks are performed to ensure that the segments do not
            overlap. However, this reduces performance. If checks are skipped
            and multiple segments do overlap, the segment with the highest
            segment number (after relabelling, if applicable) will be placed
            into the output array.
        dtype: Union[type, str, numpy.dtype, None]
            Data type of the returned array. If None, an appropriate type will
            be chosen automatically. If the returned values are rescaled
            fractional values, this will be numpy.float32. Otherwise, the
            smallest unsigned integer type that accommodates all of the output
            values will be chosen.

        Returns
        -------
        pixel_array: np.ndarray
            Pixel array representing the segmentation's total pixel matrix.

        Note
        ----
        This method uses 1-based indexing of rows and columns in order to match
        the conventions used in the DICOM standard. The first row of the total
        pixel matrix is row 1, and the last is ``self.TotalPixelMatrixRows``.
        This is is unlike standard Python and NumPy indexing which is 0-based.
        For negative indices, the two are equivalent with the final row/column
        having index -1.

        """
        # Check whether this segmentation is appropriate for tile-based indexing
        if not is_tiled_image(self):
            raise RuntimeError("Segmentation is not a tiled image.")
        if not self._db_man.is_indexable_as_total_pixel_matrix():
            raise RuntimeError(
                "Segmentation does not have appropriate dimension indices "
                "to be indexed as a total pixel matrix."
            )

        # Checks on validity of the inputs
        if segment_numbers is None:
            segment_numbers = list(self.segment_numbers)
        if len(segment_numbers) == 0:
            raise ValueError(
                'Segment numbers may not be empty.'
            )

        if row_start is None:
            row_start = 1
        if row_end is None:
            row_end = self.TotalPixelMatrixRows + 1
        if column_start is None:
            column_start = 1
        if column_end is None:
            column_end = self.TotalPixelMatrixColumns + 1

        if column_start == 0 or row_start == 0:
            raise ValueError(
                'Arguments "row_start" and "column_start" may not be 0.'
            )

        if row_start > self.TotalPixelMatrixRows + 1:
            raise ValueError(
                'Invalid value for "row_start".'
            )
        elif row_start < 0:
            row_start = self.TotalPixelMatrixRows + row_start + 1
        if row_end > self.TotalPixelMatrixRows + 1:
            raise ValueError(
                'Invalid value for "row_end".'
            )
        elif row_end < 0:
            row_end = self.TotalPixelMatrixRows + row_end + 1

        if column_start > self.TotalPixelMatrixColumns + 1:
            raise ValueError(
                'Invalid value for "column_start".'
            )
        elif column_start < 0:
            column_start = self.TotalPixelMatrixColumns + column_start + 1
        if column_end > self.TotalPixelMatrixColumns + 1:
            raise ValueError(
                'Invalid value for "column_end".'
            )
        elif column_end < 0:
            column_end = self.TotalPixelMatrixColumns + column_end + 1

        output_shape = (
            row_end - row_start,
            column_end - column_start,
        )

        with self._db_man.iterate_indices_for_tiled_region(
            row_start=row_start,
            row_end=row_end,
            column_start=column_start,
            column_end=column_end,
            tile_shape=(self.Rows, self.Columns),
            segment_numbers=segment_numbers,
            combine_segments=combine_segments,
            relabel=relabel,
        ) as indices:

            return self._get_pixels_by_seg_frame(
                output_shape=output_shape,
                indices_iterator=indices,
                segment_numbers=np.array(segment_numbers),
                combine_segments=combine_segments,
                relabel=relabel,
                rescale_fractional=rescale_fractional,
                skip_overlap_checks=skip_overlap_checks,
                dtype=dtype,
            )


def segread(fp: Union[str, bytes, PathLike, BinaryIO]) -> Segmentation:
    """Read a segmentation image stored in DICOM File Format.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing a
        Segmentation image.

    Returns
    -------
    highdicom.seg.Segmentation
        Segmentation image read from the file.

    """
    return Segmentation.from_dataset(dcmread(fp), copy=False)
