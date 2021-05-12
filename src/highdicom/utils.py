import logging
import itertools
from typing import Iterator, List, Optional, Sequence, Tuple, Any

import numpy as np
from pydicom.datadict import tag_for_keyword, keyword_for_tag
from pydicom.dataset import Dataset
from pydicom.tag import Tag, BaseTag
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.multival import MultiValue
from pydicom.uid import UID

from highdicom.content import PlanePositionSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.spatial import map_pixel_into_coordinate_system


logger = logging.getLogger(__name__)


def tile_pixel_matrix(
    total_pixel_matrix_rows: int,
    total_pixel_matrix_columns: int,
    rows: int,
    columns: int,
) -> Iterator[Tuple[int, int]]:
    """Tiles an image into smaller frames (rectangular regions).

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
    return itertools.product(tile_col_indices, tile_row_indices)


def compute_plane_position_tiled_full(
    row_index: int,
    column_index: int,
    x_offset: float,
    y_offset: float,
    rows: int,
    columns: int,
    image_orientation: Sequence[float],
    pixel_spacing: Sequence[float],
    slice_thickness: Optional[float] = None,
    spacing_between_slices: Optional[float] = None,
    slice_index: Optional[float] = None
) -> PlanePositionSequence:
    """Computes the absolute position of a Frame (image plane) in the
    Frame of Reference defined by the three-dimensional slide coordinate
    system given their relative position in the Total Pixel Matrix.

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
    slice_thickness: float, optional
        Thickness of a focal plane in micrometers
    spacing_between_slices: float, optional
        Distance between neighboring focal planes in micrometers
    slice_index: int, optional
        Relative one-based index of the slice in the array of slices
        within the volume

    Returns
    -------
    highdicom.content.PlanePositionSequence
        Positon of the plane in the slide coordinate system

    Raises
    ------
    TypeError
        When only one of `slice_index` and `spacing_between_slices` is provided

    """
    # Offset values are one-based, i.e., the top left pixel in the Total Pixel
    # Matrix has offset (1, 1) rather than (0, 0)
    row_offset_frame = ((row_index - 1) * rows) + 1
    column_offset_frame = ((column_index - 1) * columns) + 1

    provided_3d_params = (
        slice_index is not None,
        spacing_between_slices is not None,
    )
    if not(sum(provided_3d_params) == 0 or sum(provided_3d_params) == 2):
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
        coordinate=(column_offset_frame, row_offset_frame),
        image_position=(x_offset, y_offset, z_offset),
        image_orientation=image_orientation,
        pixel_spacing=pixel_spacing,
    )

    return PlanePositionSequence(
        coordinate_system=CoordinateSystemNames.SLIDE,
        image_position=(x, y, z),
        pixel_matrix_position=(column_offset_frame, row_offset_frame)
    )


def compute_plane_position_slide_per_frame(
        dataset: Dataset
    ) -> List[PlanePositionSequence]:
    """Computes the plane position for each frame in given dataset with
    respect to the slide coordinate system.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        VL Whole Slide Microscopy Image

    Returns
    -------
    List[highdicom.content.PlanePositionSequence]
        Plane Position Sequence per frame

    Raises
    ------
    ValueError
        When `dataset` does not represent a VL Whole Slide Microscopy Image

    """
    if not dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.77.1.6':
        raise ValueError('Expected a VL Whole Slide Microscopy Image')

    image_origin = dataset.TotalPixelMatrixOriginSequence[0]
    image_orientation = (
        float(dataset.ImageOrientationSlide[0]),
        float(dataset.ImageOrientationSlide[1]),
        float(dataset.ImageOrientationSlide[2]),
        float(dataset.ImageOrientationSlide[3]),
        float(dataset.ImageOrientationSlide[4]),
        float(dataset.ImageOrientationSlide[5]),
    )
    tiles_per_column = int(
        np.ceil(dataset.TotalPixelMatrixRows / dataset.Rows)
    )
    tiles_per_row = int(
        np.ceil(dataset.TotalPixelMatrixColumns / dataset.Columns)
    )
    num_focal_planes = getattr(
        dataset,
        'NumberOfFocalPlanes',
        1
    )

    shared_fg = dataset.SharedFunctionalGroupsSequence[0]
    pixel_measures = shared_fg.PixelMeasuresSequence[0]
    pixel_spacing = (
        float(pixel_measures.PixelSpacing[0]),
        float(pixel_measures.PixelSpacing[1]),
    )
    slice_thickness = getattr(
        pixel_measures,
        'SliceThickness',
        1.0
    )
    spacing_between_slices = getattr(
        pixel_measures,
        'SpacingBetweenSlices',
        1.0
    )

    return [
        compute_plane_position_tiled_full(
            row_index=r,
            column_index=c,
            x_offset=image_origin.XOffsetInSlideCoordinateSystem,
            y_offset=image_origin.YOffsetInSlideCoordinateSystem,
            rows=dataset.Rows,
            columns=dataset.Columns,
            image_orientation=image_orientation,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            spacing_between_slices=spacing_between_slices,
            slice_index=s,
        )
        for s, r, c in itertools.product(
            range(1, num_focal_planes + 1),
            range(1, tiles_per_column + 1),  # column direction, top to bottom
            range(1, tiles_per_row + 1),  # row direction, left to right
        )
    ]


class _DicomHelper:

    """A class for checking dicom tags and comparing dicom attributes"""

    @staticmethod
    def istag_file_meta_information_group(t: BaseTag) -> bool:
        return t.group == 0x0002

    @staticmethod
    def istag_repeating_group(t: BaseTag) -> bool:
        g = t.group
        return (g >= 0x5000 and g <= 0x501e) or\
            (g >= 0x6000 and g <= 0x601e)

    @staticmethod
    def istag_group_length(t: BaseTag) -> bool:
        return t.element == 0

    @staticmethod
    def isequal(v1: Any, v2: Any, float_tolerance: float = 1.0e-5) -> bool:
        from pydicom.valuerep import DSfloat

        def is_equal_float(x1: float, x2: float) -> bool:
            return abs(x1 - x2) < float_tolerance
        if type(v1) != type(v2):
            return False
        if isinstance(v1, DataElementSequence):
            for item1, item2 in zip(v1, v2):
                if not _DicomHelper.isequal_dicom_dataset(item1, item2):
                    return False
        if not isinstance(v1, MultiValue):
            v11 = [v1]
            v22 = [v2]
        else:
            v11 = v1
            v22 = v2
        if len(v11) != len(v22):
            return False
        for xx, yy in zip(v11, v22):
            if isinstance(xx, DSfloat) or isinstance(xx, float):
                if not is_equal_float(xx, yy):
                    return False
            else:
                if xx != yy:
                    return False
        return True

    @staticmethod
    def isequal_dicom_dataset(ds1: Dataset, ds2: Dataset) -> bool:
        """Checks if two dicom dataset have the same value in all attributes

        Parameters
        ----------
        ds1: pydicom.dataset.Dataset
            1st dicom dataset
        ds2: pydicom.dataset.Dataset
            2nd dicom dataset

        Returns
        -------
        True if dicom datasets are equal otherwise False

        """
        if type(ds1) != type(ds2):
            return False
        if not isinstance(ds1, Dataset):
            return False
        for k1, elem1 in ds1.items():
            if k1 not in ds2:
                return False
            elem2 = ds2[k1]
            if not _DicomHelper.isequal(elem2.value, elem1.value):
                return False
        return True

    @staticmethod
    def tag2kwstr(tg: BaseTag) -> str:
        """Converts tag to keyword and (group, element) form"""
        return '{}-{:32.32s}'.format(
            str(tg), keyword_for_tag(tg))


class FrameSet:

    """

        A class containing the dicom frames that hold equal distinguishing
        attributes to detect all perframe and shared dicom attributes
    """

    def __init__(
            self,
            single_frame_list: List[Dataset],
            distinguishing_tags: List[BaseTag],
        ) -> None:
        """

        Parameters
        ----------
        single_frame_list: List[pydicom.dataset.Dataset]
            list of single frames that have equal distinguising attributes
        distinguishing_tags: List[pydicom.tag.BaseTag]
            list of distinguishing attributes tags

        """
        self._frames = single_frame_list
        self._distinguishing_attributes_tags = distinguishing_tags
        tmp = [
            tag_for_keyword('AcquisitionDateTime'),
            tag_for_keyword('AcquisitionDate'),
            tag_for_keyword('AcquisitionTime'),
            tag_for_keyword('SpecificCharacterSet')]
        self._excluded_from_perframe_tags =\
            self._distinguishing_attributes_tags + tmp
        self._perframe_tags: List[BaseTag] = []
        self._shared_tags: List[BaseTag] = []
        self._find_per_frame_and_shared_tags()

    @property
    def frames(self) -> List[Dataset]:
        return self._frames[:]

    @property
    def distinguishing_attributes_tags(self) -> List[Tag]:
        return self._distinguishing_attributes_tags[:]

    @property
    def excluded_from_perframe_tags(self) -> List[Tag]:
        return self._excluded_from_perframe_tags[:]

    @property
    def perframe_tags(self) -> List[Tag]:
        return self._perframe_tags[:]

    @property
    def shared_tags(self) -> List[Tag]:
        return self._shared_tags[:]

    @property
    def series_instance_uid(self) -> UID:
        """Returns the series instance uid of the FrameSet"""
        return self._frames[0].SeriesInstanceUID

    @property
    def study_instance_uid(self) -> UID:
        """Returns the study instance uid of the FrameSet"""
        return self._frames[0].StudyInstanceUID

    def get_sop_instance_uid_list(self) -> list:
        """Returns a list containing all SOPInstanceUID of the FrameSet"""
        output_list = [f.SOPInstanceUID for f in self._frames]
        return output_list

    def get_sop_class_uid(self) -> UID:
        """Returns the sop class uid of the FrameSet"""
        return self._frames[0].SOPClassUID

    def _find_per_frame_and_shared_tags(self) -> None:
        """Detects and collects all shared and perframe attributes"""
        rough_shared: dict = {}
        sfs = self.frames
        for ds in sfs:
            for ttag, elem in ds.items():
                if (not ttag.is_private and not
                    _DicomHelper.istag_file_meta_information_group(ttag) and not
                        _DicomHelper.istag_repeating_group(ttag) and not
                        _DicomHelper.istag_group_length(ttag) and not
                        self._istag_excluded_from_perframe(ttag) and
                        ttag != tag_for_keyword('PixelData')):
                    elem = ds[ttag]
                    if ttag not in self._perframe_tags:
                        self._perframe_tags.append(ttag)
                    if ttag in rough_shared:
                        rough_shared[ttag].append(elem.value)
                    else:
                        rough_shared[ttag] = [elem.value]
        to_be_removed_from_shared = []
        for ttag, v in rough_shared.items():
            v = rough_shared[ttag]
            if len(v) < len(self.frames):
                to_be_removed_from_shared.append(ttag)
            else:
                all_values_are_equal = all(
                    _DicomHelper.isequal(v_i, v[0]) for v_i in v)
                if not all_values_are_equal:
                    to_be_removed_from_shared.append(ttag)
        for t in to_be_removed_from_shared:
            del rough_shared[t]
        for t, v in rough_shared.items():
            self._shared_tags.append(t)
            self._perframe_tags.remove(t)

    def _istag_excluded_from_perframe(self, t: BaseTag) -> bool:
        return t in self._excluded_from_perframe_tags


class FrameSetCollection:

    """A class to extract framesets based on distinguishing dicom attributes"""

    def __init__(self, single_frame_list: Sequence[Dataset]) -> None:
        """Forms framesets based on a list of distinguishing attributes.
        The list of "distinguishing" attributes that are used to determine
        commonality is currently fixed, and includes the unique identifying
        attributes at the Patient, Study, Equipment levels, the Modality and
        SOP Class, and ImageType as well as the characteristics of the Pixel
        Data, and those attributes that for cross-sectional images imply
        consistent sampling, such as ImageOrientationPatient, PixelSpacing and
        SliceThickness, and in addition AcquisitionContextSequence and
        BurnedInAnnotation.

        Parameters
        ----------
        single_frame_list: Sequence[pydicom.dataset.Dataset]
            list of mixed or non-mixed single frame dicom images

        Notes
        -----
        Note that Series identification, specifically SeriesInstanceUID is NOT
        a distinguishing attribute; i.e. FrameSets may span Series

        """
        self.mixed_frames = single_frame_list
        self.mixed_frames_copy = self.mixed_frames[:]
        self._distinguishing_attribute_keywords = [
            'PatientID',
            'PatientName',
            'StudyInstanceUID',
            'FrameOfReferenceUID',
            'Manufacturer',
            'InstitutionName',
            'InstitutionAddress',
            'StationName',
            'InstitutionalDepartmentName',
            'ManufacturerModelName',
            'DeviceSerialNumber',
            'SoftwareVersions',
            'GantryID',
            'PixelPaddingValue',
            'Modality',
            'ImageType',
            'BurnedInAnnotation',
            'SOPClassUID',
            'Rows',
            'Columns',
            'BitsStored',
            'BitsAllocated',
            'HighBit',
            'PixelRepresentation',
            'PhotometricInterpretation',
            'PlanarConfiguration',
            'SamplesPerPixel',
            'ProtocolName',
            'ImageOrientationPatient',
            'PixelSpacing',
            'SliceThickness',
            'AcquisitionContextSequence']
        self._frame_sets: List[FrameSet] = []
        frame_counts = []
        frameset_counter = 0
        while len(self.mixed_frames_copy) != 0:
            frameset_counter += 1
            x = self._find_all_similar_to_first_datasets()
            self._frame_sets.append(FrameSet(x[0], x[1]))
            frame_counts.append(len(x[0]))
            # log information
            logger.debug(
                f"Frameset({frameset_counter:02d}) "
                "including {len(x[0]):03d} frames")
            logger.debug('\t Distinguishing tags:')
            for dg_i, dg_tg in enumerate(x[1], 1):
                logger.debug(
                    f'\t\t{dg_i:02d}/{len(x[1])})\t{str(dg_tg)}-'
                    '{keyword_for_tag(dg_tg):32.32s} = '
                    '{str(x[0][0][dg_tg].value):32.32s}')
            logger.debug('\t dicom datasets in this frame set:')
            for dicom_i, dicom_ds in enumerate(x[0], 1):
                logger.debug(
                    f'\t\t{dicom_i}/{len(x[0])})\t '
                    '{dicom_ds["SOPInstanceUID"]}')
        frames = ''
        for i, f_count in enumerate(frame_counts, 1):
            frames += '{: 2d}){:03d}\t'.format(i, f_count)
        frames = '{: 2d} frameset(s) out of all {: 3d} instances:'.format(
            len(frame_counts), len(self.mixed_frames)) + frames
        logger.info(frames)
        self._excluded_from_perframe_tags = {}
        for kwkw in self._distinguishing_attribute_keywords:
            self._excluded_from_perframe_tags[tag_for_keyword(kwkw)] = False
        excluded_kws = [
            'AcquisitionDateTime'
            'AcquisitionDate'
            'AcquisitionTime'
            'SpecificCharacterSet'
        ]
        for kwkw in excluded_kws:
            self._excluded_from_perframe_tags[tag_for_keyword(kwkw)] = False

    def _find_all_similar_to_first_datasets(self) -> tuple:
        """Takes the fist instance from mixed-frames and finds all dicom images
        that have the same distinguishing attributes.

        """
        similar_ds: List[Dataset] = [self.mixed_frames_copy[0]]
        distinguishing_tags_existing = []
        distinguishing_tags_missing = []
        self.mixed_frames_copy = self.mixed_frames_copy[1:]
        for kw in self._distinguishing_attribute_keywords:
            tg = tag_for_keyword(kw)
            if tg in similar_ds[0]:
                distinguishing_tags_existing.append(tg)
            else:
                distinguishing_tags_missing.append(tg)
        logger_msg = set()
        for ds in self.mixed_frames_copy:
            all_equal = True
            for tg in distinguishing_tags_missing:
                if tg in ds:
                    logger_msg.add(
                        '{} is missing in all but {}'.format(
                            _DicomHelper.tag2kwstr(tg), ds['SOPInstanceUID']))
                    all_equal = False
                    break
            if not all_equal:
                continue
            for tg in distinguishing_tags_existing:
                ref_val = similar_ds[0][tg].value
                if tg not in ds:
                    all_equal = False
                    break
                new_val = ds[tg].value
                if not _DicomHelper.isequal(ref_val, new_val):
                    logger_msg.add(
                        'Inequality on distinguishing '
                        'attribute{} -> {} != {} \n series uid = {}'.format(
                            _DicomHelper.tag2kwstr(tg), ref_val, new_val,
                            ds.SeriesInstanceUID))
                    all_equal = False
                    break
            if all_equal:
                similar_ds.append(ds)
        for msg_ in logger_msg:
            logger.info(msg_)
        for ds in similar_ds:
            if ds in self.mixed_frames_copy:
                self.mixed_frames_copy = [
                    nds for nds in self.mixed_frames_copy if nds != ds]
        return (similar_ds, distinguishing_tags_existing)

    @property
    def distinguishing_attribute_keywords(self) -> List[str]:
        """Returns the list of all distinguising attributes found."""

        return self._distinguishing_attribute_keywords[:]

    @property
    def frame_sets(self) -> List[FrameSet]:
        """Returns the list of all FrameSets found."""

        return self._frame_sets
