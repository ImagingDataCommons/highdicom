""" Module for SOP Classes of Legacy Converted Enhanced Image IODs."""
import logging
from typing import Any, List, Union, Callable, Sequence
from numpy import log10, array, ceil, cross, dot, ndarray
from pydicom.datadict import tag_for_keyword, dictionary_VR, keyword_for_tag
from pydicom.dataset import Dataset
from pydicom.tag import Tag, BaseTag
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.multival import MultiValue
from datetime import datetime, timedelta
from pydicom.valuerep import DT, DA, TM
from copy import deepcopy
from pydicom.uid import UID

from highdicom.base import SOPClass
from highdicom._iods import IOD_MODULE_MAP
from highdicom._modules import MODULE_ATTRIBUTE_MAP


logger = logging.getLogger(__name__)


LEGACY_ENHANCED_SOP_CLASS_UID_MAP = {
    # CT Image Storage
    '1.2.840.10008.5.1.4.1.1.2': '1.2.840.10008.5.1.4.1.1.2.2',
    # MR Image Storage
    '1.2.840.10008.5.1.4.1.1.4': '1.2.840.10008.5.1.4.1.1.4.4',
    # PET Image Storage
    '1.2.840.10008.5.1.4.1.1.128': '1.2.840.10008.5.1.4.1.1.128.1',
}


_SOP_CLASS_UID_IOD_KEY_MAP = {
    '1.2.840.10008.5.1.4.1.1.2.2': 'legacy-converted-enhanced-ct-image',
    '1.2.840.10008.5.1.4.1.1.4.4': 'legacy-converted-enhanced-mr-image',
    '1.2.840.10008.5.1.4.1.1.128.1': 'legacy-converted-enhanced-pet-image',
}


class ـDicomHelper:

    """A class for checking dicom tags and comparing dicom attributes"""

    @classmethod
    def istag_file_meta_information_group(cls, t: BaseTag) -> bool:
        return t.group == 0x0002

    @classmethod
    def istag_repeating_group(cls, t: BaseTag) -> bool:
        g = t.group
        return (g >= 0x5000 and g <= 0x501e) or\
            (g >= 0x6000 and g <= 0x601e)

    @classmethod
    def istag_group_length(cls, t: BaseTag) -> bool:
        return t.element == 0

    @classmethod
    def isequal(cls, v1: Any, v2: Any) -> bool:
        from pydicom.valuerep import DSfloat
        float_tolerance = 1.0e-5

        def is_equal_float(x1: float, x2: float) -> bool:
            return abs(x1 - x2) < float_tolerance
        if type(v1) != type(v2):
            return False
        if isinstance(v1, DataElementSequence):
            for item1, item2 in zip(v1, v2):
                cls.isequal_dicom_dataset(item1, item2)
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

    @classmethod
    def isequal_dicom_dataset(cls, ds1: Dataset, ds2: Dataset) -> bool:
        """Checks if two dicom dataset have the same value in all attributes
        Parameters
        ----------
        ds1: Dataset
            1st dicom dataset
        ds2: Dataset
            2nd dicom dataset
        Returns
        -------
        True of dicom datasets are equal otherwise False
        """
        if type(ds1) != type(ds2):
            return False
        if not isinstance(ds1, Dataset):
            return False
        for k1, elem1 in ds1.items():
            if k1 not in ds2:
                return False
            elem2 = ds2[k1]
            if not cls.isequal(elem2.value, elem1.value):
                return False
        return True

    @classmethod
    def tag2str(cls, tg: BaseTag) -> str:
        """Converts tag to hex format like (group, element)

        """
        if not isinstance(tg, BaseTag):
            tg = Tag(tg)
        return '(0x{:0>4x}, 0x{:0>4x})'.format(tg.group, tg.element)

    @classmethod
    def tag2kwstr(cls, tg: BaseTag) -> str:
        """Converts tag to keyword and (group, element) form

        """
        return '{}-{:32.32s}'.format(
            cls.tag2str(tg), keyword_for_tag(tg))


class GeometryOfSlice:

    """A class for checking dicom slices geomtery/parallelization"""

    def __init__(self,
                 row_vector: ndarray,
                 col_vector: ndarray,
                 top_left_corner_pos: ndarray,
                 voxel_spaceing: ndarray,
                 dimensions: tuple) -> None:
        """
        Parameters
        ----------
        row_vector: ndarray
            3D vector representing row of the input slice
        col_vector: ndarray
            3D vector representing column the input  aslice
        top_left_corner_pos: ndarray
            3D point representing top left coner position of the input slice
        voxel_spaceing: ndarray
            Three element array. 1st and 2nd copied from PexelSpacing and the
            3rd copied from SliceThickness
        dimensions: tuple
            3 element tuple holding x as number of cols, y as number of rows
            and z as 1
        """
        self.RowVector = row_vector
        self.ColVector = col_vector
        self.TopLeftCornerPosition = top_left_corner_pos
        self.VoxelSpacing = voxel_spaceing
        self.Dim = dimensions

    def get_normal_vector(self) -> ndarray:
        """Returns the normal vector of the input slice

        """
        n: ndarray = cross(self.RowVector, self.ColVector)
        n[2] = -n[2]
        return n

    def get_distance_along_origin(self) -> float:
        """Returns the shortest distince of the slice from the origin

        """
        n = self.get_normal_vector()
        return float(
            dot(self.TopLeftCornerPosition, n))

    def are_parallel(
            slice1: Any,
            slice2: Any,
            tolerance: float = 0.0001) -> bool:
        """Returns False if two slices are not prallel else True

        """
        if (not isinstance(slice1, GeometryOfSlice) or
                not isinstance(slice2, GeometryOfSlice)):
            logger.warning(
                'slice1 and slice2 are not of the same '
                'type: type(slice1) = {} and type(slice2) = {}'.format(
                    type(slice1), type(slice2)))
            return False
        else:
            n1: ndarray = slice1.get_normal_vector()
            n2: ndarray = slice2.get_normal_vector()
            for el1, el2 in zip(n1, n2):
                if abs(el1 - el2) > tolerance:
                    return False
            return True


class ـPerframeFunctionalGroup(DataElementSequence):

    """A sequence class for perframe functional group"""

    def __init__(self, number_of_frames: int) -> None:
        """
        Parameters
        ----------
        number_of_frames: int
            The perframe functional group sequence will have items equal to
            the whole number of frames
        """
        super().__init__()
        for i in range(0, number_of_frames):
            item = Dataset()
            self.append(item)


class ـSharedFunctionalGroup(DataElementSequence):

    """A sequence class for shared functional group"""

    def __init__(self) -> None:
        """Construncts a shared frame functional group holding only one item

        """
        super().__init__()
        item = Dataset()
        self.append(item)


class FrameSet:

    """
        A class containing the dicom frames that hold equal distinguishing
        attributes to detect all perframe and shared dicom attributes
    """

    def __init__(self, single_frame_list: list,
                 distinguishing_tags: list) -> None:
        """
        Parameters
        ----------
        single_frame_list: list
            list of single frames that have equal distinguising attributes
        distinguishing_tags: list
            list of distinguishing attributes tags
        """
        self._frames = single_frame_list
        self._distinguishing_attributes_tags = distinguishing_tags
        tmp = [
            tag_for_keyword('AcquisitionDateTime'),
            tag_for_keyword('AcquisitionDate'),
            tag_for_keyword('AcquisitionTime'),
            tag_for_keyword('SpecificCharacterSet')]
        self._excluded_fromperframe_tags =\
            self.distinguishing_attributes_tags + tmp
        self._perframe_tags: list = []
        self._shared_tags: list = []
        self._find_per_frame_and_shared_tags()

    @property
    def frames(self) -> List[Dataset]:
        return self._frames[:]

    @property
    def distinguishing_attributes_tags(self) -> List[Tag]:
        return self._distinguishing_attributes_tags[:]

    @property
    def excluded_from_perframe_tags(self) -> List[Tag]:
        return self._excluded_fromperframe_tags[:]

    @property
    def perframe_tags(self) -> List[Tag]:
        return self._perframe_tags[:]

    @property
    def shared_tags(self) -> List[Tag]:
        return self._shared_tags[:]

    @property
    def SeriesInstanceUID(self) -> UID:
        """Returns the sereis instance uid of the FrameSet

        """
        return self._frames[0].SeriesInstanceUID

    @property
    def StudyInstanceUID(self) -> UID:
        """Returns the study instance uid of the FrameSet

        """
        return self._frames[0].StudyInstanceUID

    def get_sop_instance_uid_list(self) -> list:
        """Returns a list containing all SOPInstanceUID of the FrameSet

        """
        OutputList: list = []
        for f in self._frames:
            OutputList.append(f.SOPInstanceUID)
        return OutputList

    def get_sop_class_uid(self) -> UID:
        """Returns the sop class uid of the FrameSet

        """
        return self._frames[0].SOPClassUID

    def _find_per_frame_and_shared_tags(self) -> None:
        """Detects and collects all shared and perframe attributes

        """
        rough_shared: dict = {}
        sfs = self.frames
        for ds in sfs:
            for ttag, elem in ds.items():
                if (not ttag.is_private and not
                    ـDicomHelper.istag_file_meta_information_group(ttag) and not
                        ـDicomHelper.istag_repeating_group(ttag) and not
                        ـDicomHelper.istag_group_length(ttag) and not
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
                all_values_are_equal = True
                for v_i in v:
                    if not ـDicomHelper.isequal(v_i, v[0]):
                        all_values_are_equal = False
                        break
                if not all_values_are_equal:
                    to_be_removed_from_shared.append(ttag)
        from pydicom.datadict import keyword_for_tag
        for t, v in rough_shared.items():
            if keyword_for_tag(t) != 'PatientSex':
                continue
        for t in to_be_removed_from_shared:
            del rough_shared[t]
        for t, v in rough_shared.items():
            self._shared_tags.append(t)
            self._perframe_tags.remove(t)

    def _istag_excluded_from_perframe(self, t: BaseTag) -> bool:
        return t in self.excluded_from_perframe_tags


class FrameSetCollection:

    """A calss to extract framesets based on distinguishing dicom attributes"""

    def __init__(self, single_frame_list: Sequence[Any]) -> None:
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
        single_frame_list: list
            lisf of mixed or non-mixed single frame dicom images
        Note
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
        to_be_removed_from_distinguishing_attribs: set = set()
        self._frame_sets: list = []
        frame_counts = []
        frameset_counter = 0
        while len(self.mixed_frames_copy) != 0:
            frameset_counter += 1
            x = self._find_all_similar_to_first_datasets()
            self._frame_sets.append(FrameSet(x[0], x[1]))
            frame_counts.append(len(x[0]))
            # log information
            logger.debug("Frameset({:02d}) including {:03d} frames".format(
                frameset_counter, len(x[0])))
            logger.debug('\t Distinguishing tags:')
            for dg_i, dg_tg in enumerate(x[1], 1):
                logger.debug('\t\t{:02d}/{})\t{}-{:32.32s} = {:32.32s}'.format(
                    dg_i, len(x[1]), ـDicomHelper.tag2str(dg_tg),
                    keyword_for_tag(dg_tg),
                    str(x[0][0][dg_tg].value)))
            logger.debug('\t dicom datasets in this frame set:')
            for dicom_i, dicom_ds in enumerate(x[0], 1):
                logger.debug('\t\t{}/{})\t {}'.format(
                    dicom_i, len(x[0]), dicom_ds['SOPInstanceUID']))
        frames = ''
        for i, f_count in enumerate(frame_counts, 1):
            frames += '{: 2d}){:03d}\t'.format(i, f_count)
        frames = '{: 2d} frameset(s) out of all {: 3d} instances:'.format(
            len(frame_counts), len(self.mixed_frames)) + frames
        logger.info(frames)
        for kw in to_be_removed_from_distinguishing_attribs:
            self.distinguishing_attribute_keywords.remove(kw)
        self.excluded_from_perframe_tags = {}
        for kwkw in self.distinguishing_attribute_keywords:
            self.excluded_from_perframe_tags[tag_for_keyword(kwkw)] = False
        self.excluded_from_perframe_tags[
            tag_for_keyword('AcquisitionDateTime')] = False
        self.excluded_from_perframe_tags[
            tag_for_keyword('AcquisitionDate')] = False
        self.excluded_from_perframe_tags[
            tag_for_keyword('AcquisitionTime')] = False
        self.ExcludedFromFunctionalGroupsTags = {
            tag_for_keyword('SpecificCharacterSet'): False}

    def _find_all_similar_to_first_datasets(self) -> tuple:
        """Takes the fist instance from mixed-frames and finds all dicom images
        that have the same distinguishing attributes.
        """
        similar_ds: list = [self.mixed_frames_copy[0]]
        distinguishing_tags_existing = []
        distinguishing_tags_missing = []
        self.mixed_frames_copy = self.mixed_frames_copy[1:]
        for kw in self.distinguishing_attribute_keywords:
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
                            ـDicomHelper.tag2kwstr(tg), ds['SOPInstanceUID']))
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
                if not ـDicomHelper.isequal(ref_val, new_val):
                    logger_msg.add(
                        'Inequality on distinguishing '
                        'attribute{} -> {} != {} \n series uid = {}'.format(
                            ـDicomHelper.tag2kwstr(tg), ref_val, new_val,
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


class _CommonLegacyConvertedEnhanceImage(SOPClass):

    """SOP class for common Legacy Converted Enhanced instances."""

    def __init__(
            self,
            legacy_datasets: Sequence[Dataset],
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            sort_key: Callable = None,
        ) -> None:
        """
        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.dataset.Dataset]
            DICOM data sets of legacy single-frame image instances that should
            be converted
        series_instance_uid: str
            UID of the series
        series_number: Union[int, None]
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        """
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')
        sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
        all_framesets = FrameSetCollection(legacy_datasets)
        if len(all_framesets.frame_sets) > 1:
            raise ValueError(
                'Mixed frames sets: the input single frame list contain more '
                'than one multiframe collection')
        frame_set = all_framesets.frame_sets[0]
        if sort_key is None:
            sort_key = _CommonLegacyConvertedEnhanceImage.default_sort_key
        super().__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer="" if 'Manufacturer' not in ref_ds
            else ref_ds.Manufacturer,
            modality="" if 'Modality' not in ref_ds
            else ref_ds.Modality,
            patient_id=None if 'PatientID' not in ref_ds
            else ref_ds.PatientID,
            patient_name=None if 'PatientName' not in ref_ds
            else ref_ds.PatientName,
            patient_birth_date=None if 'PatientBirthDate' not in ref_ds
            else ref_ds.PatientBirthDate,
            patient_sex=None if 'PatientSex' not in ref_ds
            else ref_ds.PatientSex,
            accession_number=None if 'AccessionNumber' not in ref_ds
            else ref_ds.AccessionNumber,
            study_id=None if 'StudyID' not in ref_ds
            else ref_ds.StudyID,
            study_date=None if 'StudyDate' not in ref_ds
            else ref_ds.StudyDate,
            study_time=None if 'StudyTime' not in ref_ds
            else ref_ds.StudyTime,
            referring_physician_name=None if 'ReferringPhysicianName' not in
            ref_ds else ref_ds.ReferringPhysicianName,
        )
        self._legacy_datasets = legacy_datasets
        self._perframe_functional_group = ـPerframeFunctionalGroup(
            len(legacy_datasets))
        tg = tag_for_keyword('PerFrameFunctionalGroupsSequence')
        self[tg] = DataElement(tg, 'SQ', self._perframe_functional_group)
        self._shared_functional_group = ـSharedFunctionalGroup()
        tg = tag_for_keyword('SharedFunctionalGroupsSequence')
        self[tg] = DataElement(tg, 'SQ', self._shared_functional_group)
        self.distinguishing_attributes_tags = self._get_tag_used_dictionary(
            frame_set.distinguishing_attributes_tags)
        self.excluded_from_perframe_tags = self._get_tag_used_dictionary(
            frame_set.excluded_from_perframe_tags)
        self._perframe_tags = self._get_tag_used_dictionary(
            frame_set.perframe_tags)
        self._shared_tags = self._get_tag_used_dictionary(
            frame_set.shared_tags)
        self.ExcludedFromFunctionalGroupsTags = {
            tag_for_keyword('SpecificCharacterSet'): False}

        # --------------------------------------------------------------------
        self.__build_blocks: list = []
        # == == == == == == == == == == == == == == == == == == == == == == ==
        new_ds = []
        for item in sorted(self._legacy_datasets, key=sort_key):
            new_ds.append(item)

        # self = multi_frame_output
        self._module_excepted_list: dict = {
         "patient": [],
         "clinical-trial-subject": [],
         "general-study":
         [
            "StudyInstanceUID",
            "RequestingService"
         ],
         "patient-study":
         [
            "ReasonForVisit",
            "ReasonForVisitCodeSequence"
         ],
         "clinical-trial-study": [],
         "general-series":
         [
            "SeriesInstanceUID",
            "SeriesNumber",
            "SmallestPixelValueInSeries",
            "LargestPixelValueInSeries",
            "PerformedProcedureStepEndDate",
            "PerformedProcedureStepEndTime"
         ],
         "clinical-trial-series": [],
         "general-equipment":
         [
            "InstitutionalDepartmentTypeCodeSequence"
         ],
         "frame-of-reference": [],
         "sop-common":
         [
            "SOPClassUID",
            "SOPInstanceUID",
            "InstanceNumber",
            "SpecificCharacterSet",
            "EncryptedAttributesSequence",
            "MACParametersSequence",
            "DigitalSignaturesSequence"
         ],
         "general-image":
         [
            "ImageType",
            "AcquisitionDate",
            "AcquisitionDateTime",
            "AcquisitionTime",
            "AnatomicRegionSequence",
            "PrimaryAnatomicStructureSequence",
            "IrradiationEventUID",
            "AcquisitionNumber",
            "InstanceNumber",
            "PatientOrientation",
            "ImageLaterality",
            "ImagesInAcquisition",
            "ImageComments",
            "QualityControlImage",
            "BurnedInAnnotation",
            "RecognizableVisualFeatures",
            "LossyImageCompression",
            "LossyImageCompressionRatio",
            "LossyImageCompressionMethod",
            "RealWorldValueMappingSequence",
            "IconImageSequence",
            "PresentationLUTShape"
         ],
         "sr-document-general":
         [
            "ContentDate",
            "ContentTime",
            "ReferencedInstanceSequence",
            "InstanceNumber",
            "VerifyingObserverSequence",
            "AuthorObserverSequence",
            "ParticipantSequence",
            "CustodialOrganizationSequence",
            "PredecessorDocumentsSequence",
            "CurrentRequestedProcedureEvidenceSequence",
            "PertinentOtherEvidenceSequence",
            "CompletionFlag",
            "CompletionFlagDescription",
            "VerificationFlag",
            "PreliminaryFlag",
            "IdenticalDocumentsSequence"
         ]
        }
        self.earliest_date = DA('00010101')
        self.earliest_time = TM('000000')
        self.earliest_date_time = DT('00010101000000')
        self.farthest_future_date = DA('99991231')
        self.farthest_future_time = TM('235959')
        self.farthest_future_date_time = DT('99991231235959')
        self._slices: list = []
        self._tolerance = 0.0001
        self._slice_location_map: dict = {}
        self._byte_data = bytearray()
        self._word_data = bytearray()
        self.earliest_content_date_time = self.farthest_future_date_time
        self._add_common_ct_pet_mr_build_blocks()
        if (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-ct-image'):
            self._add_build_blocks_for_ct()
        elif (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-mr-image'):
            self._add_build_blocks_for_mr()
        elif (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-pet-image'):
            self._add_build_blocks_for_pet()

    def _is_empty_or_empty_items(self, attribute: DataElement) -> bool:
        """Takes a dicom DataElement and check if DataElement is empty or in
        case of Sequence returns True if there is not item or all the items
        are empty.
        """
        if attribute.is_empty:
            return True
        if isinstance(attribute.value, DataElementSequence):
            if len(attribute.value) == 0:
                return True
            for item in attribute.value:
                for tg, v in item.items():
                    v = item[tg]
                    if not self._is_empty_or_empty_items(v):
                        return False
        return False

    def _mark_tag_as_used(self, tg: BaseTag) -> None:
        """Checks what group the input tag belongs to and marks it as used to
        keep track of all used and unused tags
        """
        if tg in self._shared_tags:
            self._shared_tags[tg] = True
        elif tg in self.excluded_from_perframe_tags:
            self.excluded_from_perframe_tags[tg] = True
        elif tg in self._perframe_tags:
            self._perframe_tags[tg] = True

    def _copy_attrib_if_present(self, src_ds: Dataset, dest_ds: Dataset,
                                src_kw_or_tg: str, dest_kw_or_tg: str = None,
                                check_not_to_be_perframe: bool = True,
                                check_not_to_be_empty: bool = False) -> None:
        """Copies a dicom attribute value from a keyword in the source Dataset
        to a similar or different keyword in the destination Dataset
        Parameters
        ----------
        src_ds: Dataset
            Source Dataset to copy the attribute from.
        dest_ds: Dataset
            Destination Dataset to copy the attribute to.
        src_kw_or_tg: str
            The keyword from the souce Dataset to copy its value.
        dest_kw_or_tg: str = None
            The keyword of the destination Dataset, the value is copied to. If
            its value is None, then the destination keyword will be exactly the
            source keyword.
        check_not_to_be_perframe: bool = True
            If this arg is true, then copy is aborted if the src attribute is
            perframe.
        check_not_to_be_empty: bool = False
            If this arg is true, then copy is aborted if the source attribute is
            empty.
        """
        if isinstance(src_kw_or_tg, str):
            src_kw_or_tg = tag_for_keyword(src_kw_or_tg)
        if dest_kw_or_tg is None:
            dest_kw_or_tg = src_kw_or_tg
        elif isinstance(dest_kw_or_tg, str):
            dest_kw_or_tg = tag_for_keyword(dest_kw_or_tg)
        if check_not_to_be_perframe:
            if src_kw_or_tg in self._perframe_tags:
                return
        if src_kw_or_tg in src_ds:
            elem = src_ds[src_kw_or_tg]
            if check_not_to_be_empty:
                if self._is_empty_or_empty_items(elem):
                    return
            new_elem = deepcopy(elem)
            if dest_kw_or_tg == src_kw_or_tg:
                dest_ds[dest_kw_or_tg] = new_elem
            else:
                new_elem1 = DataElement(dest_kw_or_tg,
                                        dictionary_VR(dest_kw_or_tg),
                                        new_elem.value)
                dest_ds[dest_kw_or_tg] = new_elem1
            # now mark the attrib as used/done to keep track of every one of it
            self._mark_tag_as_used(src_kw_or_tg)

    def _get_or_create_attribute(
        self, src: Dataset, kw: Union[str, Tag], default: Any) -> DataElement:
        """Creates a new DataElement with a value copied from the source
        Dataset. If the attribute is absent in source Dataset, then its value
        will be the default value.
        Parameters
        ----------
        src: Dataset
            Source Dataset to copy the value if available.
        kw: Union[str, Tag]
            The keyword of created DataElement.
        default: Any
            The default value created DataElement if the keyword was absent in
            the source Dataset.
        Returns
        -------
        A new DataElement created.
        """
        if kw is str:
            tg = tag_for_keyword(kw)
        else:
            tg = kw
        if kw in src:
            a = deepcopy(src[kw])
        else:
            a = DataElement(tg, dictionary_VR(tg), default)
        from pydicom.valuerep import DT, TM, DA
        if a.VR == 'DA' and isinstance(a.value, str):
            try:
                d_tmp = DA(a.value)
                a.value = DA(default) if d_tmp is None else d_tmp
            except BaseException:
                a.value = DA(default)
        if a.VR == 'DT' and isinstance(a.value, str):
            try:
                dt_tmp = DT(a.value)
                a.value = DT(default) if dt_tmp is None else dt_tmp
            except BaseException:
                a.value = DT(default)
        if a.VR == 'TM' and isinstance(a.value, str):
            try:
                t_tmp = TM(a.value)
                a.value = TM(default) if t_tmp is None else t_tmp
            except BaseException:
                a.value = TM(default)

        self._mark_tag_as_used(tg)
        return a

    def _add_module(self, module_name: str, excepted_attributes: list = [],
                    check_not_to_be_perframe: bool = True,
                    check_not_to_be_empty: bool = False) -> None:
        """Copies all attribute of a particular module to current SOPClass,
        excepting the excepted_attributes, from a reference frame (the first
        frame on the single frame list).
        Parameters
        ----------
        module_name: str:
            A hiphenated module name like `image-pixel`.
        excepted_attributes: list = []
            List of all attributes that are not allowed to be copied
        check_not_to_be_perframe: bool = True
            If this flag is true, then the perframe attributes will not be
            copied.
        check_not_to_be_empty: bool = False
            If this flag is true, then the empty attributes will not be copied.
        """
        attribs: list = MODULE_ATTRIBUTE_MAP[module_name]
        ref_dataset = self._legacy_datasets[0]
        for a in attribs:
            kw: str = a['keyword']
            if kw in excepted_attributes:
                continue
            if len(a['path']) == 0:
                self._copy_attrib_if_present(
                    ref_dataset, self, kw,
                    check_not_to_be_perframe=check_not_to_be_perframe,
                    check_not_to_be_empty=check_not_to_be_empty)

    def _add_module_to_mf_image_pixel(self) -> None:
        """Copies/adds` a/an image_pixel` multiframe module to
        the current SOPClass from its single frame source.
        """
        module_and_excepted_at = {
         "image-pixel":
         [
            "ColorSpace",
            "PixelDataProviderURL",
            "ExtendedOffsetTable",
            "ExtendedOffsetTableLengths",
            "PixelData"
         ]
        }
        for module, except_at in module_and_excepted_at.items():
            self._add_module(
                module,
                excepted_attributes=except_at,
                check_not_to_be_empty=False,
                check_not_to_be_perframe=True)  # don't check the perframe set

    def _add_module_to_mf_enhanced_common_image(self) -> None:
        """Copies/adds a/an `enhanced_common_image` multiframe module to
        the current SOPClass from its single frame source.
        """
        ref_dataset = self._legacy_datasets[0]
        attribs_to_be_added = [
            'ContentQualification',
            'ImageComments',
            'BurnedInAnnotation',
            'RecognizableVisualFeatures',
            'LossyImageCompression',
            'LossyImageCompressionRatio',
            'LossyImageCompressionMethod']
        for kw in attribs_to_be_added:
            self._copy_attrib_if_present(
                ref_dataset, self, kw,
                check_not_to_be_perframe=True,
                check_not_to_be_empty=False)
        sum_compression_ratio: float = 0
        c_ratio_tag = tag_for_keyword('LossyImageCompressionRatio')
        if tag_for_keyword('LossyImageCompression') in self._shared_tags and \
                tag_for_keyword(
                    'LossyImageCompressionMethod') in self._shared_tags and \
                c_ratio_tag in self._perframe_tags:
            for fr_ds in self._legacy_datasets:
                if c_ratio_tag in fr_ds:
                    ratio = fr_ds[c_ratio_tag].value
                    try:
                        sum_compression_ratio += float(ratio)
                    except BaseException:
                        sum_compression_ratio += 1  # supposing uncompressed
                else:
                    sum_compression_ratio += 1
            avg_compression_ratio = sum_compression_ratio /\
                len(self._legacy_datasets)
            avg_ratio_str = '{:.6f}'.format(avg_compression_ratio)
            self[c_ratio_tag] = \
                DataElement(c_ratio_tag, 'DS', avg_ratio_str)

        if tag_for_keyword('PresentationLUTShape') not in self._perframe_tags:
            # actually should really invert the pixel data if MONOCHROME1,
            #           since only MONOCHROME2 is permitted : (
            # also, do not need to check if PhotometricInterpretation is
            #           per-frame, since a distinguishing attribute
            phmi_kw = 'PhotometricInterpretation'
            phmi_a = self._get_or_create_attribute(
                self._legacy_datasets[0], phmi_kw, "MONOCHROME2")
            LUT_shape_default = "INVERTED" if phmi_a.value == 'MONOCHROME1'\
                else "IDENTITY"
            LUT_shape_a = self._get_or_create_attribute(
                self._legacy_datasets[0],
                'PresentationLUTShape',
                LUT_shape_default)
            if not LUT_shape_a.is_empty:
                self['PresentationLUTShape'] = LUT_shape_a
        # Icon Image Sequence - always discard these

    def _add_module_to_mf_contrast_bolus(self) -> None:
        """Copies/adds a/an `contrast_bolus` multiframe module to
        the current SOPClass from its single frame source.
        """
        self._add_module('contrast-bolus')

    def _add_module_to_mf_enhanced_ct_image(self) -> None:
        """Copies/adds a/an `enhanced_ct_image` multiframe module to
        the current SOPClass from its single frame source.
        """
        pass
        # David's code doesn't hold anything for this module ... should ask him

    def _add_module_to_mf_enhanced_pet_image(self) -> None:
        """Copies/adds a/an `enhanced_pet_image` multiframe module to
        the current SOPClass from its single frame source.
        """
        # David's code doesn't hold anything for this module ... should ask him
        kw = 'ContentQualification'
        tg = tag_for_keyword(kw)
        elem = self._get_or_create_attribute(
            self._legacy_datasets[0], kw, 'RESEARCH')
        self[tg] = elem

    def _add_module_to_mf_enhanced_mr_image(self) -> None:
        """Copies/adds a/an `enhanced_mr_image` multiframe module to
        the current SOPClass from its single frame source.
        """
        self._copy_attrib_if_present(
            self._legacy_datasets[0],
            self,
            "ResonantNucleus",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        if 'ResonantNucleus' not in self:
            # derive from ImagedNucleus, which is the one used in legacy MR
            #  IOD, but does not have a standard list of defined terms ...
            #  (could check these : ()
            self._copy_attrib_if_present(
                self._legacy_datasets[0],
                self,
                "ImagedNucleus",
                check_not_to_be_perframe=True,
                check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self._legacy_datasets[0],
            self,
            "KSpaceFiltering",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self._legacy_datasets[0],
            self,
            "MagneticFieldStrength",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self._legacy_datasets[0],
            self,
            "ApplicableSafetyStandardAgency",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self._legacy_datasets[0],
            self,
            "ApplicableSafetyStandardDescription",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)

    def _add_module_to_mf_acquisition_context(self) -> None:
        """Copies/adds a/an `acquisition_context` multiframe module to
        the current SOPClass from its single frame source.
        """
        tg = tag_for_keyword('AcquisitionContextSequence')
        if tg not in self._perframe_tags:
            self[tg] = self._get_or_create_attribute(
                self._legacy_datasets[0],
                tg,
                None)

    def _get_value_for_frame_type(
            self, attrib: DataElement) -> Union[list, None]:
        """Guesses the appropriate FrameType attribute value from ImageType.

        """
        if not isinstance(attrib, DataElement):
            return None
        output = ['', '', '', '']
        v = attrib.value
        lng = len(v)
        output[0] = 'ORIGINAL' if lng == 0 else v[0]
        output[1] = 'PRIMARY'
        output[2] = 'VOLUME' if lng < 3 else v[2]
        output[3] = 'NONE'
        return output

    def _get_frame_type_seq_tag(
            self, modality: str) -> int:
        """Detects the correct tag/keyword for the frame type sq based on the
        modality name.
        """
        seq_kw = '{}{}FrameTypeSequence'
        if modality == 'PET':
            seq_kw = seq_kw.format(modality, '')
        else:
            seq_kw = seq_kw.format(modality, 'Image')
        return tag_for_keyword(seq_kw)

    def _add_module_to_dataset_common_ct_mr_pet_image_description(
            self, source: Dataset, destination: Dataset, level: int) -> None:
        """Copies/adds attributes related to `common_ct_mr_pet_image_description`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        level: int
            If level is `0` then the destination attributes will be in the root
            of dicom Dataset like `ImageType`. If level is not `0`, then the
            destination attributes will be in functional groups items like
            `FrameType`
        """
        frame_type_a = source['ImageType']
        if level == 0:
            frame_type_tg = tag_for_keyword('ImageType')
        else:
            frame_type_tg = tag_for_keyword('FrameType')
        new_val = self._get_value_for_frame_type(frame_type_a)
        destination[frame_type_tg] = DataElement(
            frame_type_tg, frame_type_a.VR, new_val)

        def element_generator(kw: str, val: Any) -> DataElement:
            return DataElement(
                tag_for_keyword(kw),
                dictionary_VR(tag_for_keyword(kw)), val)
        destination['PixelPresentation'] = element_generator(
            'PixelPresentation', "MONOCHROME")
        destination['VolumetricProperties'] = element_generator(
            'VolumetricProperties', "VOLUME")
        destination['VolumeBasedCalculationTechnique'] = element_generator(
            'VolumeBasedCalculationTechnique', "NONE")

    def _add_module_to_mf_common_ct_mr_pet_image_description(
            self, modality: str) -> None:
        """Copies/adds the common attrabutes for ct/mr/pet description
        module to the current SOPClass from its single frame source.
        """
        im_type_tag = tag_for_keyword('ImageType')
        seq_tg = self._get_frame_type_seq_tag(modality)
        if im_type_tag not in self._perframe_tags:
            self._add_module_to_dataset_common_ct_mr_pet_image_description(
                self._legacy_datasets[0], self, 0)
            # ----------------------------
            item = self._shared_functional_group[0]
            inner_item = Dataset()
            self._add_module_to_dataset_common_ct_mr_pet_image_description(
                self._legacy_datasets[0], inner_item, 1)
            item[seq_tg] = DataElement(
                seq_tg, 'SQ', DataElementSequence([inner_item]))
        else:
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                inner_item = Dataset()
                self._add_module_to_dataset_common_ct_mr_pet_image_description(
                    self._legacy_datasets[i], inner_item, 1)
                item[seq_tg] = DataElement(
                    seq_tg, 'SQ', DataElementSequence([inner_item]))

    def _add_module_to_mf_composite_instance_contex(self) -> None:
        """Copies/adds a/an `composite_instance_contex` multiframe module to
        the current SOPClass from its single frame source.
        """
        for module_name, excpeted_a in self._module_excepted_list.items():
            self._add_module(
             module_name,
             excepted_attributes=excpeted_a,
             check_not_to_be_empty=False,
             check_not_to_be_perframe=True)  # don't check the perframe set

    def _add_module_to_dataset_frame_anatomy(
            self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `frame_anatomy`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        # David's code is more complicaated than mine
        # Should check it out later.
        fa_seq_tg = tag_for_keyword('FrameAnatomySequence')
        item = Dataset()
        self._copy_attrib_if_present(source, item, 'AnatomicRegionSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        if len(item) != 0:
            self._copy_attrib_if_present(
                source, item, 'FrameLaterality',
                check_not_to_be_perframe=False,
                check_not_to_be_empty=True)
            if 'FrameLaterality' not in item:
                self._copy_attrib_if_present(
                    source, item, 'ImageLaterality',
                    'FrameLaterality',
                    check_not_to_be_perframe=False,
                    check_not_to_be_empty=True)
            if 'FrameLaterality' not in item:
                self._copy_attrib_if_present(
                    source, item, 'Laterality',
                    'FrameLaterality',
                    check_not_to_be_perframe=False,
                    check_not_to_be_empty=True)
            if 'FrameLaterality' not in item:
                frame_laterality_a = self._get_or_create_attribute(
                    source, 'FrameLaterality', "U")
                item['FrameLaterality'] = frame_laterality_a
            frame_anatomy_a = DataElement(
                fa_seq_tg,
                dictionary_VR(fa_seq_tg),
                DataElementSequence([item]))
            destination['FrameAnatomySequence'] = frame_anatomy_a

    def _has_frame_anatomy(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `frame_anatomy` present in source single frames.
        Otherwise returns false.
        """
        laterality_tg = tag_for_keyword('Laterality')
        im_laterality_tg = tag_for_keyword('ImageLaterality')
        bodypart_tg = tag_for_keyword('BodyPartExamined')
        anatomical_reg_tg = tag_for_keyword('AnatomicRegionSequence')
        return (laterality_tg in tags or
                im_laterality_tg in tags or
                bodypart_tg in tags or
                anatomical_reg_tg)

    def _add_module_to_mf_frame_anatomy(self) -> None:
        """Copies/adds a/an `frame_anatomy` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_frame_anatomy(self._perframe_tags) and
            (self._has_frame_anatomy(self._shared_tags) or
             self._has_frame_anatomy(self.excluded_from_perframe_tags))
            ):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_frame_anatomy(
                self._legacy_datasets[0], item)
        elif self._has_frame_anatomy(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_frame_anatomy(
                    self._legacy_datasets[i], item)

    def _has_pixel_measures(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `pixel_measures` present in source single frames.
        Otherwise returns false.
        """
        pixel_spacing_tg = tag_for_keyword('PixelSpacing')
        slice_thickness_tg = tag_for_keyword('SliceThickness')
        imager_pixel_spacing_tg = tag_for_keyword('ImagerPixelSpacing')
        return (pixel_spacing_tg in tags or
                slice_thickness_tg in tags or
                imager_pixel_spacing_tg in tags)

    def _add_module_to_dataset_pixel_measures(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `pixel_measures`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        self._copy_attrib_if_present(source,
                                     item,
                                     'PixelSpacing',
                                     check_not_to_be_perframe=False)
        self._copy_attrib_if_present(source,
                                     item,
                                     'SliceThickness',
                                     check_not_to_be_perframe=False)
        if 'PixelSpacing' not in item:
            self._copy_attrib_if_present(source,
                                         item,
                                         'ImagerPixelSpacing',
                                         'PixelSpacing',
                                         check_not_to_be_perframe=False,
                                         check_not_to_be_empty=True)
        pixel_measures_kw = 'PixelMeasuresSequence'
        pixel_measures_tg = tag_for_keyword(pixel_measures_kw)
        seq = DataElement(pixel_measures_tg,
                          dictionary_VR(pixel_measures_tg),
                          DataElementSequence([item]))
        destination[pixel_measures_tg] = seq

    def _add_module_to_mf_pixel_measures(self) -> None:
        """Copies/adds a/an `pixel_measures` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_pixel_measures(self._perframe_tags) and
            (self._has_pixel_measures(self._shared_tags) or
            self._has_pixel_measures(self.excluded_from_perframe_tags))
            ):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_pixel_measures(
                self._legacy_datasets[0], item)
        elif self._has_pixel_measures(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_pixel_measures(
                    self._legacy_datasets[i], item)

    def _has_plane_position(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `plane_position` present in source single frames.
        Otherwise returns false.
        """
        image_position_patient_tg = tag_for_keyword('ImagePositionPatient')
        return image_position_patient_tg in tags

    def _add_module_to_dataset_plane_position(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `plane_position`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        self._copy_attrib_if_present(source,
                                     item,
                                     'ImagePositionPatient',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        plane_position_sequence_kw = 'PlanePositionSequence'
        plane_position_sequence_tg = tag_for_keyword(plane_position_sequence_kw)
        seq = DataElement(plane_position_sequence_tg,
                          dictionary_VR(plane_position_sequence_tg),
                          DataElementSequence([item]))
        destination[plane_position_sequence_tg] = seq

    def _add_module_to_mf_plane_position(self) -> None:
        """Copies/adds a/an `plane_position` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_plane_position(self._perframe_tags) and
            (self._has_plane_position(self._shared_tags) or
            self._has_plane_position(self.excluded_from_perframe_tags))
            ):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_plane_position(
                self._legacy_datasets[0], item)
        elif self._has_plane_position(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_plane_position(
                    self._legacy_datasets[i], item)

    def _has_plane_orientation(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `plane_orientation` present in source single frames.
        Otherwise returns false.
        """
        image_orientation_patient_tg = tag_for_keyword(
            'ImageOrientationPatient')
        return image_orientation_patient_tg in tags

    def _add_module_to_dataset_plane_orientation(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `plane_orientation`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        self._copy_attrib_if_present(source,
                                     item,
                                     'ImageOrientationPatient',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        kw = 'PlaneOrientationSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), DataElementSequence([item]))
        destination[tg] = seq

    def _add_module_to_mf_plane_orientation(self) -> None:
        """Copies/adds a/an `plane_orientation` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_plane_orientation(self._perframe_tags) and
            (self._has_plane_orientation(self._shared_tags) or
            self._has_plane_orientation(self.excluded_from_perframe_tags))
            ):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_plane_orientation(
                self._legacy_datasets[0], item)
        elif self._has_plane_orientation(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_plane_orientation(
                    self._legacy_datasets[i], item)

    def _has_frame_voi_lut(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `frame_voi_lut` present in source single frames.
        Otherwise returns false.
        """
        window_width_tg = tag_for_keyword('WindowWidth')
        window_center_tg = tag_for_keyword('WindowCenter')
        window_center_width_explanation_tg = tag_for_keyword(
            'WindowCenterWidthExplanation')
        return (window_width_tg in tags or
                window_center_tg in tags or
                window_center_width_explanation_tg in tags)

    def _add_module_to_dataset_frame_voi_lut(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `frame_voi_lut`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        self._copy_attrib_if_present(source,
                                     item,
                                     'WindowWidth',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(source,
                                     item,
                                     'WindowCenter',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(source,
                                     item,
                                     'WindowCenterWidthExplanation',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        kw = 'FrameVOILUTSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), DataElementSequence([item]))
        destination[tg] = seq

    def _add_module_to_mf_frame_voi_lut(self) -> None:
        """Copies/adds a/an `frame_voi_lut` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_frame_voi_lut(self._perframe_tags) and
            (self._has_frame_voi_lut(self._shared_tags) or
            self._has_frame_voi_lut(self.excluded_from_perframe_tags))
            ):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_frame_voi_lut(
                self._legacy_datasets[0], item)
        elif self._has_frame_voi_lut(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_frame_voi_lut(
                    self._legacy_datasets[i], item)

    def _has_pixel_value_transformation(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `pixel_value_transformation` present in source single frames.
        Otherwise returns false.
        """
        rescale_intercept_tg = tag_for_keyword('RescaleIntercept')
        rescale_slope_tg = tag_for_keyword('RescaleSlope')
        rescale_type_tg = tag_for_keyword('RescaleType')
        return (rescale_intercept_tg in tags or
                rescale_slope_tg in tags or
                rescale_type_tg in tags)

    def _add_module_to_dataset_pixel_value_transformation(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `pixel_value_transformation`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        self._copy_attrib_if_present(source,
                                     item,
                                     'RescaleSlope',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(source,
                                     item,
                                     'RescaleIntercept',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        haveValuesSoAddType = ('RescaleSlope' in item or
                               'RescaleIntercept' in item)
        self._copy_attrib_if_present(source,
                                     item,
                                     'RescaleType',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        value = ''
        modality = '' if 'Modality' not in source\
            else source["Modality"].value
        if haveValuesSoAddType:
            value = 'US'
            if modality == 'CT':
                containes_localizer = False
                image_type_v = [] if 'ImageType' not in source\
                    else source['ImageType'].value
                for i in image_type_v:
                    if i == 'LOCALIZER':
                        containes_localizer = True
                        break
                if not containes_localizer:
                    value = "HU"
            # elif modality == 'PT':
                # value = 'US' if 'Units' not in source\
                #     else source['Units'].value
            else:
                value = 'US'
            tg = tag_for_keyword('RescaleType')
            if "RescaleType" not in item:
                item[tg] = DataElement(tg, dictionary_VR(tg), value)
            elif item[tg].value != value:
                # keep the copied value as LUT explanation
                voi_exp_tg = tag_for_keyword('LUTExplanation')
                item[voi_exp_tg] = DataElement(
                    voi_exp_tg, dictionary_VR(voi_exp_tg), item[tg].value)
                item[tg].value = value
        kw = 'PixelValueTransformationSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), DataElementSequence([item]))
        destination[tg] = seq

    def _add_module_to_mf_pixel_value_transformation(self) -> None:
        """Copies/adds a/an `pixel_value_transformation` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_pixel_value_transformation(self._perframe_tags) and
                (self._has_pixel_value_transformation(self._shared_tags) or
                    self._has_pixel_value_transformation(
                        self.excluded_from_perframe_tags))):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_pixel_value_transformation(
                self._legacy_datasets[0], item)
        elif self._has_pixel_value_transformation(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_pixel_value_transformation(
                    self._legacy_datasets[i], item)

    def _has_referenced_image(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `referenced_image` present in source single frames.
        Otherwise returns false.
        """
        return tag_for_keyword('ReferencedImageSequence') in tags

    def _add_module_to_dataset_referenced_image(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `referenced_image`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        self._copy_attrib_if_present(source,
                                     destination,
                                     'ReferencedImageSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)

    def _add_module_to_mf_referenced_image(self) -> None:
        """Copies/adds a/an `referenced_image` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_referenced_image(self._perframe_tags) and
            (self._has_referenced_image(self._shared_tags) or
            self._has_referenced_image(self.excluded_from_perframe_tags))
            ):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_referenced_image(
                self._legacy_datasets[0], item)
        elif self._has_referenced_image(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_referenced_image(
                    self._legacy_datasets[i], item)

    def _has_derivation_image(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `derivation_image` present in source single frames.
        Otherwise returns false.
        """
        return tag_for_keyword('SourceImageSequence') in tags

    def _add_module_to_dataset_derivation_image(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `derivation_image`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        self._copy_attrib_if_present(source,
                                     item,
                                     'DerivationDescription',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        self._copy_attrib_if_present(source,
                                     item,
                                     'DerivationCodeSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(source,
                                     item,
                                     'SourceImageSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        kw = 'DerivationImageSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), DataElementSequence([item]))
        destination[tg] = seq

    def _add_module_to_mf_derivation_image(self) -> None:
        """Copies/adds a/an `derivation_image` multiframe module to
        the current SOPClass from its single frame source.
        """
        if (not self._has_derivation_image(self._perframe_tags) and
            (self._has_derivation_image(self._shared_tags) or
            self._has_derivation_image(self.excluded_from_perframe_tags))
            ):
            item = self._shared_functional_group[0]
            self._add_module_to_dataset_derivation_image(
                self._legacy_datasets[0], item)
        elif self._has_derivation_image(self._perframe_tags):
            for i in range(0, len(self._legacy_datasets)):
                item = self._perframe_functional_group[i]
                self._add_module_to_dataset_derivation_image(
                    self._legacy_datasets[i], item)

    def _get_tag_used_dictionary(self, input: list) -> dict:
        """Returns a dictionary of input tags with a use flag

        """
        out: dict = {}
        for item in input:
            out[item] = False
        return out

    def _add_module_to_dataset_unassigned_perframe(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `unassigned_perframe`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        for tg in self._eligeible_tags:
            self._copy_attrib_if_present(source,
                                         item,
                                         tg,
                                         check_not_to_be_perframe=False,
                                         check_not_to_be_empty=False)
        kw = 'UnassignedPerFrameConvertedAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), DataElementSequence([item]))
        destination[tg] = seq

    def _add_largest_smallest_pixle_value(self) -> None:
        """Adds the attributes for largest and smallest pixel value to
        current SOPClass object

        """
        ltg = tag_for_keyword("LargestImagePixelValue")
        from sys import float_info
        lval = float_info.min
        if ltg in self._perframe_tags:
            for frame in self._legacy_datasets:
                if ltg in frame:
                    nval = frame[ltg].value
                else:
                    continue
                lval = nval if lval < nval else lval
            if lval > float_info.min:
                self[ltg] = DataElement(ltg, 'SS', int(lval))
        # ==========================
        stg = tag_for_keyword("SmallestImagePixelValue")
        sval = float_info.max
        if stg in self._perframe_tags:
            for frame in self._legacy_datasets:
                if stg in frame:
                    nval = frame[stg].value
                else:
                    continue
                sval = nval if sval < nval else sval
            if sval < float_info.max:
                self[stg] = DataElement(stg, 'SS', int(sval))

        stg = "SmallestImagePixelValue"

    def _add_module_to_mf_unassigned_perframe(self) -> None:
        """Copies/adds a/an `unassigned_perframe` multiframe module to
        the current SOPClass from its single frame source.
        """
        # first collect all not used tags
        # note that this is module is order dependent
        self._add_largest_smallest_pixle_value()
        self._eligeible_tags: List[Tag] = []
        for tg, used in self._perframe_tags.items():
            if not used and tg not in self.ExcludedFromFunctionalGroupsTags:
                self._eligeible_tags.append(tg)
        for i in range(0, len(self._legacy_datasets)):
            item = self._perframe_functional_group[i]
            self._add_module_to_dataset_unassigned_perframe(
                self._legacy_datasets[i], item)

    def _add_module_to_dataset_unassigned_shared(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `unassigned_shared`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        for tg, used in self._shared_tags.items():
            if (not used and
                    tg not in self and
                    tg not in self.ExcludedFromFunctionalGroupsTags):
                self._copy_attrib_if_present(source,
                                             item,
                                             tg,
                                             check_not_to_be_perframe=False,
                                             check_not_to_be_empty=False)
        kw = 'UnassignedSharedConvertedAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), DataElementSequence([item]))
        destination[tg] = seq

    def _add_module_to_mf_unassigned_shared(self) -> None:
        """Copies/adds a/an `unassigned_shared` multiframe module to
        the current SOPClass from its single frame source.
        """
        item = self._shared_functional_group[0]
        self._add_module_to_dataset_unassigned_shared(
            self._legacy_datasets[0], item)

    def _create_empty_element(self, tg: BaseTag) -> DataElement:
        """Creates an empty dicom DataElement for input tag

        """
        return DataElement(tg, dictionary_VR(tg), None)

    def _add_module_to_mf_empty_type2_attributes(self) -> None:
        """Adds empty type2 attributes to the current SOPClass to avoid
        type2 missing error.
        """
        iod_name = _SOP_CLASS_UID_IOD_KEY_MAP[
            self['SOPClassUID'].value]
        modules = IOD_MODULE_MAP[iod_name]
        for module in modules:
            if module['usage'] == 'M':
                mod_key = module['key']
                attrib_list = MODULE_ATTRIBUTE_MAP[mod_key]
                for a in attrib_list:
                    if len(a['path']) == 0 and a['type'] == '2':
                        tg = tag_for_keyword(a['keyword'])
                        if (tg not in self._legacy_datasets[0] and
                           tg not in self and
                           tg not in self._perframe_tags and
                           tg not in self._shared_tags):
                            self[tg] =\
                                self._create_empty_element(tg)

    def _add_module_to_dataset_conversion_source(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `conversion_source`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        self._copy_attrib_if_present(source,
                                     item,
                                     'SOPClassUID',
                                     'ReferencedSOPClassUID',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        self._copy_attrib_if_present(source,
                                     item,
                                     'SOPInstanceUID',
                                     'ReferencedSOPInstanceUID',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        kw = 'ConversionSourceAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), DataElementSequence([item]))
        destination[tg] = seq

    def _add_module_to_mf_conversion_source(self) -> None:
        """Copies/adds a/an `conversion_source` multiframe module to
        the current SOPClass from its single frame source.
        """
        for i in range(0, len(self._legacy_datasets)):
            item = self._perframe_functional_group[i]
            self._add_module_to_dataset_conversion_source(
                self._legacy_datasets[i], item)

            self.EarliestFrameAcquisitionDateTime =\
                self.farthest_future_date_time

    def _build_slices_geometry_frame_content(self) -> None:
        """Instantiates an object of GeometryOfSlice for each sice.

        """
        frame_count = len(self._legacy_datasets)
        for i in range(0, frame_count):
            curr_frame = self._legacy_datasets[i]
            if 'ImagePositionPatient' not in curr_frame:
                image_position_patient_v = None
            else:
                image_position_patient_v =\
                    curr_frame['ImagePositionPatient'].value
            if 'ImageOrientationPatient' not in curr_frame:
                image_orientation_patient_v = None
            else:
                image_orientation_patient_v =\
                    curr_frame['ImageOrientationPatient'].value
            if 'PixelSpacing' not in curr_frame:
                pixel_spacing_v = None
            else:
                pixel_spacing_v = curr_frame['PixelSpacing'].value
            if 'SliceThickness' not in curr_frame:
                slice_thickness_v = 0.0
            else:
                slice_thickness_v = curr_frame['SliceThickness'].value
            # slice_location_v = None \
            #     if 'SliceLocation' not in curr_frame\
            #     else curr_frame['SliceLocation'].value
            rows_v = 0 \
                if 'Rows' not in curr_frame\
                else curr_frame['Rows'].value
            columns_v = 0 \
                if 'Columns' not in curr_frame\
                else curr_frame['Columns'].value
            if (image_orientation_patient_v is not None and
                    image_position_patient_v is not None and
                    pixel_spacing_v is not None):
                row = array(image_orientation_patient_v[0:3])
                col = array(image_orientation_patient_v[3:])
                voxel_spaceing = array([pixel_spacing_v[0],
                                        pixel_spacing_v[1],
                                        slice_thickness_v])
                tpl = array(image_position_patient_v)
                dim = (rows_v, columns_v, 1)
                self._slices.append(GeometryOfSlice(row, col,
                                    tpl, voxel_spaceing, dim))
            else:
                logger.error(
                    "Error in geometry. One or more required "
                    "attributes are not available")
                logger.error("\tImageOrientationPatient = {}".format(
                    image_orientation_patient_v))
                logger.error("\tImagePositionPatient = {}".format(
                    image_position_patient_v))
                logger.error("\tPixelSpacing = {}".format(pixel_spacing_v))
                self._slices = []  # clear the slices
                break

    def _are_all_slices_parallel_frame_content(self) -> bool:
        """Returns true if all slices are prallel otherwise, false.

        """
        slice_count = len(self._slices)
        if slice_count >= 2:
            last_slice = self._slices[0]
            for i in range(1, slice_count):
                curr_slice = self._slices[i]
                if not GeometryOfSlice.are_parallel(
                        curr_slice, last_slice, self._tolerance):
                    return False
                last_slice = curr_slice
            return True
        elif slice_count == 1:
            return True
        else:
            return False

    def _add_stack_info_frame_content(self) -> None:
        """Adds stack information to the FrameContentSequence dicom attribute.

        """
        self._build_slices_geometry_frame_content()
        round_digits = int(ceil(-log10(self._tolerance)))
        source_series_uid = ''
        if self._are_all_slices_parallel_frame_content():
            self._slice_location_map = {}
            for idx, s in enumerate(self._slices):
                not_round_dist = s.get_distance_along_origin()
                dist = round(not_round_dist, round_digits)
                logger.debug(
                    'Slice locaation {} rounded by {} digits to {}'.format(
                        not_round_dist, round_digits, dist
                    ))
                if dist in self._slice_location_map:
                    self._slice_location_map[dist].append(idx)
                else:
                    self._slice_location_map[dist] = [idx]
            distance_index = 1
            frame_content_tg = tag_for_keyword("FrameContentSequence")
            for loc, idxs in sorted(self._slice_location_map.items()):
                if len(idxs) != 1:
                    if source_series_uid == '':
                        source_series_uid = \
                            self._legacy_datasets[0].SeriesInstanceUID
                    logger.warning(
                        'There are {} slices in one location {} on '
                        'series = {}'.format(
                            len(idxs), loc, source_series_uid))
                for frame_index in idxs:
                    frame = self._perframe_functional_group[frame_index]
                    new_item = frame[frame_content_tg].value[0]
                    new_item["StackID"] = self._get_or_create_attribute(
                        self._legacy_datasets[0],
                        "StackID", "0")
                    new_item["InStackPositionNumber"] =\
                        self._get_or_create_attribute(
                        self._legacy_datasets[0],
                        "InStackPositionNumber", distance_index)
                distance_index += 1

    def _has_frame_content(self, tags: dict) -> bool:
        """returns true if attributes specific to
        `frame_content` present in source single frames.
        Otherwise returns false.
        """
        acquisition_date_time_tg = tag_for_keyword('AcquisitionDateTime')
        acquisition_date_tg = tag_for_keyword('AcquisitionDate')
        acquisition_time_tg = tag_for_keyword('AcquisitionTime')
        return (acquisition_date_time_tg in tags or
                acquisition_time_tg in tags or
                acquisition_date_tg in tags)

    def _add_module_to_dataset_frame_content(
        self, source: Dataset, destination: Dataset) -> None:
        """Copies/adds attributes related to `frame_content`
        to destination dicom Dataset
        Parameters
        ----------
        source: Dataset
            the source dicom Dataset from which the modules attributes values
            are copied
        destination: Dataset
            the destination dicom Dataset to which the modules attributes
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        """
        item = Dataset()
        fan_tg = tag_for_keyword('FrameAcquisitionNumber')
        an_tg = tag_for_keyword('AcquisitionNumber')
        if an_tg in source:
            fan_val = source[an_tg].value
        else:
            fan_val = 0
        item[fan_tg] = DataElement(fan_tg, dictionary_VR(fan_tg), fan_val)
        self._mark_tag_as_used(an_tg)
        # ----------------------------------------------------------------
        acquisition_date_time_a = self._get_or_create_attribute(
            source, 'AcquisitionDateTime', self.earliest_date_time)
        # chnage the keyword to FrameAcquisitionDateTime:
        frame_acquisition_date_time_a = DataElement(
            tag_for_keyword('FrameAcquisitionDateTime'),
            'DT', acquisition_date_time_a.value)
        acquisition_date_time_is_perframe = self._has_frame_content(
            self._perframe_tags)
        if frame_acquisition_date_time_a.value == self.earliest_date_time:
            acquisition_date_a = self._get_or_create_attribute(
                source, 'AcquisitionDate', self.earliest_date)
            acquisition_time_a = self._get_or_create_attribute(
                source, 'AcquisitionTime', self.earliest_time)
            d = acquisition_date_a.value
            t = acquisition_time_a.value
            # frame_acquisition_date_time_a.value = (DT(d.strftime('%Y%m%d') +
            #                                     t.strftime('%H%M%S')))
            frame_acquisition_date_time_a.value = DT(str(d) + str(t))
        if frame_acquisition_date_time_a.value > self.earliest_date_time:
            if (frame_acquisition_date_time_a.value <
                    self.EarliestFrameAcquisitionDateTime):
                self.EarliestFrameAcquisitionDateTime =\
                    frame_acquisition_date_time_a.value
            if not acquisition_date_time_is_perframe:
                if ('TriggerTime' in source and
                        'FrameReferenceDateTime' not in source):
                    trigger_time_a = self._get_or_create_attribute(
                        source, 'TriggerTime', self.earliest_time)
                    trigger_time_in_millisecond = int(trigger_time_a.value)
                    if trigger_time_in_millisecond > 0:
                        t_delta = timedelta(trigger_time_in_millisecond)
                        # this is so rediculous. I'm not able to cnvert
                        #      the DT to datetime (cast to superclass)
                        d_t = datetime.combine(
                            frame_acquisition_date_time_a.value.date(),
                            frame_acquisition_date_time_a.value.time())
                        d_t = d_t + t_delta
                        frame_acquisition_date_time_a.value =\
                            DT(d_t.strftime('%Y%m%d%H%M%S'))
            item['FrameAcquisitionDateTime'] = frame_acquisition_date_time_a
        # ---------------------------------
        self._copy_attrib_if_present(
            source, item, "AcquisitionDuration",
            "FrameAcquisitionDuration",
            check_not_to_be_perframe=False,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            source, item,
            'TemporalPositionIndex',
            check_not_to_be_perframe=False,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            source, item, "ImageComments",
            "FrameComments",
            check_not_to_be_perframe=False,
            check_not_to_be_empty=True)
        # -----------------------------------
        seq_tg = tag_for_keyword('FrameContentSequence')
        destination[seq_tg] = DataElement(
            seq_tg, dictionary_VR(seq_tg), DataElementSequence([item]))

    def _add_acquisition_info_frame_content(self) -> None:
        """Adds acquisition information to the FrameContentSequence dicom
        attribute.

        """
        for i in range(0, len(self._legacy_datasets)):
            item = self._perframe_functional_group[i]
            self._add_module_to_dataset_frame_content(
                self._legacy_datasets[i], item)
        if (self.EarliestFrameAcquisitionDateTime <
                self.farthest_future_date_time):
            kw = 'AcquisitionDateTime'
            self[kw] = DataElement(
                tag_for_keyword(kw),
                'DT', self.EarliestFrameAcquisitionDateTime)

    def _add_module_to_mf_frame_content(self) -> None:
        """Copies/adds a/an 'frame_content` multiframe module to
        the current SOPClass from its single frame source.
        """
        self._add_acquisition_info_frame_content()
        self._add_stack_info_frame_content()

    def _is_other_byte_vr_pixel_data(self, vr: str) -> bool:
        """checks if `PixelData` dicom value representation is OB.

        """
        return vr[0] == 'O' and vr[1] == 'B'

    def _is_other_word_vr_pixel_data(self, vr: str) -> bool:
        """checks if `PixelData` dicom value representation is OW.

        """
        return vr[0] == 'O' and vr[1] == 'W'
    # def _has(self, tags: dict) -> bool:        """
    #     image_position_patient_tg = tag_for_keyword('ImagePositionPatient')
    #     return image_position_patient_tg in tags

    def _copy_data_pixel_data(
            self, src: bytearray, word_data: bool = False) -> None:
        """Copies contnet of PixelData from one frame and appends it to the
        content of PixelData for multiframe
        Parameters
        ----------
        src: bytearray
            content of pixel data from source frame(one of the single frames)
        word_data: bool = False
            flag representing if the data is word-wise instead of blyte-wise

        """
        # Make sure that the length complies by row and col
        if word_data:
            des = self._word_data
            ByteCount = 2 * self._number_of_pixels_per_frame
        else:
            des = self._byte_data
            ByteCount = self._number_of_pixels_per_frame
        if len(src) != ByteCount:
            tmp: bytearray = bytearray(ByteCount)
            tmp[:len(src)] = src[:]
            src = tmp
        des.extend(src)

    def _add_module_to_mf_pixel_data(self) -> None:
        """Copies/add`s a/an pixel_data` multiframe module to
        the current SOPClass from its single frame source.
        """
        kw = 'NumberOfFrames'
        tg = tag_for_keyword(kw)
        self._frame_count = len(self._legacy_datasets)
        self[kw] =\
            DataElement(tg, dictionary_VR(tg), self._frame_count)
        row = self._legacy_datasets[0]["Rows"].value
        col = self._legacy_datasets[0]["Columns"].value
        self._number_of_pixels_per_frame = row * col
        self._number_of_pixels = row * col * self._frame_count
        kw = "PixelData"
        for i in range(0, len(self._legacy_datasets)):
            if kw not in self._legacy_datasets[i]:
                continue
            pixel_data_a = self._legacy_datasets[i][kw]
            if self._is_other_byte_vr_pixel_data(pixel_data_a.VR):
                if len(self._word_data) != 0:
                    raise TypeError(
                        'Cannot mix OB and OW Pixel Data '
                        'VR from different frames')
                self._copy_data_pixel_data(pixel_data_a.value, False)
            elif self._is_other_word_vr_pixel_data(pixel_data_a.VR):
                if len(self._byte_data) != 0:
                    raise TypeError(
                        'Cannot mix OB and OW Pixel Data '
                        'VR from different frames')
                self._copy_data_pixel_data(pixel_data_a.value, True)
            else:
                raise TypeError(
                    'Cannot mix OB and OW Pixel Data VR from different frames')
        if len(self._byte_data) != 0:
            MF_PixelData = DataElement(tag_for_keyword(kw),
                                       'OB', bytes(self._byte_data))
        elif len(self._word_data) != 0:
            MF_PixelData = DataElement(tag_for_keyword(kw),
                                       'OW', bytes(self._word_data))
        self[kw] = MF_PixelData

    def _add_module_to_mf_content_date_time(self) -> None:
        """Copies/adds a/an `content_date_time` multiframe module to
        the current SOPClass from its single frame source.
        """
        default_atrs = ["Acquisition", "Series", "Study"]
        for i in range(0, len(self._legacy_datasets)):
            src = self._legacy_datasets[i]
            default_date = self.farthest_future_date
            for def_atr in default_atrs:
                at_tg = tag_for_keyword(def_atr + "Date")
                if at_tg in src:
                    val = src[at_tg].value
                    if isinstance(val, DA):
                        default_date = val
                        break
            kw = 'ContentDate'
            d_a = self._get_or_create_attribute(
                src, kw, default_date)
            d = d_a.value
            default_time = self.farthest_future_time
            for def_atr in default_atrs:
                at_tg = tag_for_keyword(def_atr + "Time")
                if at_tg in src:
                    val = src[at_tg].value
                    if isinstance(val, TM):
                        default_time = val
                        break
            kw = 'ContentTime'
            t_a = self._get_or_create_attribute(
                src, kw, default_time)
            t = t_a.value
            value = DT(d.strftime('%Y%m%d') + t.strftime('%H%M%S.%f'))
            if self.earliest_content_date_time > value:
                self.earliest_content_date_time = value
        if self.earliest_content_date_time < self.farthest_future_date_time:
            n_d = DA(self.earliest_content_date_time.date().strftime('%Y%m%d'))
            n_t = TM(
                self.earliest_content_date_time.time().strftime('%H%M%S.%f'))
            kw = 'ContentDate'
            self[kw] = DataElement(
                tag_for_keyword(kw), 'DA', n_d)
            kw = 'ContentTime'
            self[kw] = DataElement(
                tag_for_keyword(kw), 'TM', n_t)

    def _add_data_element_to_target_contributing_equipment(
            self, target: Dataset, kw: str, value: Any) -> None:
        """Add new data element related to ContributingEquipmentSequence to a
        target dataset(usually an item).
        Parameters
        ----------
        target: Dataset
            destination dicom Dataset.
        kw: str
            keyword if the attribute being added.
        value: Any
            value if the attribute being added.
        """
        tg = tag_for_keyword(kw)
        target[kw] = DataElement(tg, dictionary_VR(tg), value)

    def _add_module_to_mf_contributing_equipment(self) -> None:
        """Copies/adds a/an `contributing_equipment` multiframe module to
        the current SOPClass from its single frame source.
        """
        code_value_tg = tag_for_keyword('CodeValue')
        code_meaning_tg = tag_for_keyword('CodeMeaning')
        coding_scheme_designator_tg = tag_for_keyword('CodingSchemeDesignator')
        purpose_of_reference_code_item = Dataset()
        purpose_of_reference_code_item['CodeValue'] = DataElement(
            code_value_tg,
            dictionary_VR(code_value_tg),
            '109106')
        purpose_of_reference_code_item['CodeMeaning'] = DataElement(
            code_meaning_tg,
            dictionary_VR(code_meaning_tg),
            'Enhanced Multi-frame Conversion Equipment')
        purpose_of_reference_code_item['CodingSchemeDesignator'] = DataElement(
            coding_scheme_designator_tg,
            dictionary_VR(coding_scheme_designator_tg),
            'DCM')
        purpose_of_reference_code_seq = DataElement(
            tag_for_keyword('PurposeOfReferenceCodeSequence'),
            'SQ', DataElementSequence([purpose_of_reference_code_item]))
        item: Dataset = Dataset()
        item[
            'PurposeOfReferenceCodeSequence'] = purpose_of_reference_code_seq
        self._add_data_element_to_target_contributing_equipment(
            item, "Manufacturer", 'HighDicom')
        self._add_data_element_to_target_contributing_equipment(
            item, "InstitutionName", 'HighDicom')
        self._add_data_element_to_target_contributing_equipment(
            item,
            "InstitutionalDepartmentName",
            'Software Development')
        self._add_data_element_to_target_contributing_equipment(
            item,
            "InstitutionAddress",
            'Radialogy Department, B&W Hospital, Boston, MA')
        self._add_data_element_to_target_contributing_equipment(
            item,
            "SoftwareVersions",
            '1.4')  # get sw version
        self._add_data_element_to_target_contributing_equipment(
            item,
            "ContributionDescription",
            'Legacy Enhanced Image created from Classic Images')
        tg = tag_for_keyword('ContributingEquipmentSequence')
        self[tg] = DataElement(tg, 'SQ', DataElementSequence([item]))

    def _add_module_to_mf_instance_creation_date_time(self) -> None:
        """Copies/adds a/an `instance_creation_date_time` multiframe module to
        the current SOPClass from its single frame source.
        """
        nnooww = datetime.now()
        n_d = DA(nnooww.date().strftime('%Y%m%d'))
        n_t = TM(nnooww.time().strftime('%H%M%S'))
        kw = 'InstanceCreationDate'
        self[kw] = DataElement(
            tag_for_keyword(kw), 'DA', n_d)
        kw = 'InstanceCreationTime'
        self[kw] = DataElement(
            tag_for_keyword(kw), 'TM', n_t)

    @classmethod
    def default_sort_key(cls, x: Dataset) -> tuple:
        """The default sort key to sort all single frames before conversion

        """
        out: tuple = tuple()
        if 'SeriesNumber' in x:
            out += (x['SeriesNumber'].value, )
        if 'InstanceNumber' in x:
            out += (x['InstanceNumber'].value, )
        if 'SOPInstanceUID' in x:
            out += (x['SOPInstanceUID'].value, )
        return out

    def _clear_build_blocks(self) -> None:
        """Clears the array containing all methods for multiframe conversion

        """
        self.__build_blocks = []

    def _add_common_ct_pet_mr_build_blocks(self) -> None:
        """Arranges common methods for multiframe conversion and
        put them in place.

        """
        blocks = [
            [self._add_module_to_mf_image_pixel, None],
            [self._add_module_to_mf_composite_instance_contex, None],
            [self._add_module_to_mf_enhanced_common_image, None],
            [self._add_module_to_mf_acquisition_context, None],
            [self._add_module_to_mf_frame_anatomy, None],
            [self._add_module_to_mf_pixel_measures, None],
            [self._add_module_to_mf_plane_orientation, None],
            [self._add_module_to_mf_plane_position, None],
            [self._add_module_to_mf_frame_voi_lut, None],
            [self._add_module_to_mf_pixel_value_transformation, None],
            [self._add_module_to_mf_referenced_image, None],
            [self._add_module_to_mf_conversion_source, None],
            [self._add_module_to_mf_frame_content, None],
            [self._add_module_to_mf_pixel_data, None],
            [self._add_module_to_mf_content_date_time, None],
            [self._add_module_to_mf_instance_creation_date_time, None],
            [self._add_module_to_mf_contributing_equipment, None],
            [self._add_module_to_mf_unassigned_perframe, None],
            [self._add_module_to_mf_unassigned_shared, None],
        ]
        for b in blocks:
            self.__build_blocks.append(b)

    def _add_ct_specific_build_blocks(self) -> None:
        """Arranges CT specific methods for multiframe conversion and
        put them in place.

        """
        blocks = [
            [
                self._add_module_to_mf_common_ct_mr_pet_image_description,
                ('CT',)
            ],
            [self._add_module_to_mf_enhanced_ct_image, None],
            [self._add_module_to_mf_contrast_bolus, None],
        ]
        for b in blocks:
            self.__build_blocks.append(b)

    def _add_mr_specific_build_blocks(self) -> None:
        """Arranges MRI specific methods for multiframe conversion and
        put them in place

        """
        blocks = [
            [
                self._add_module_to_mf_common_ct_mr_pet_image_description,
                ('MR',)
            ],
            [self._add_module_to_mf_enhanced_mr_image, None],
            [self._add_module_to_mf_contrast_bolus, None],
        ]
        for b in blocks:
            self.__build_blocks.append(b)

    def _add_pet_specific_build_blocks(self) -> None:
        """Arranges PET specific methods for multiframe conversion and
        put them in place

        """
        blocks = [
            [
                self._add_module_to_mf_common_ct_mr_pet_image_description,
                ('PET',)
            ],
            [self._add_module_to_mf_enhanced_pet_image, None],
        ]
        for b in blocks:
            self.__build_blocks.append(b)

    def _add_build_blocks_for_mr(self) -> None:
        """Arranges all methods necessary for MRI multiframe conversion and
        put them in place

        """
        self._clear_build_blocks()
        self._add_common_ct_pet_mr_build_blocks()
        self._add_mr_specific_build_blocks()

    def _add_build_blocks_for_pet(self) -> None:
        """Arranges all methods necessary for PET multiframe conversion and
        put them in place

        """
        self._clear_build_blocks()
        self._add_common_ct_pet_mr_build_blocks()
        self._add_pet_specific_build_blocks()

    def _add_build_blocks_for_ct(self) -> None:
        """Arranges all methods necessary for CT multiframe conversion and
        put them in place.

        """
        self._clear_build_blocks()
        self._add_common_ct_pet_mr_build_blocks()
        self._add_ct_specific_build_blocks()

    def _convert2multiframe(self) -> None:
        """Runs all necessary methods to conver from single frame to
        multi-frame.

        """
        logger.debug('Strt singleframe to multiframe conversion')
        for fun, args in self.__build_blocks:
            if not args:
                fun()
            else:
                fun(*args)
        logger.debug('Conversion succeeded')


class LegacyConvertedEnhancedCTImage(_CommonLegacyConvertedEnhanceImage):

    """SOP class for Legacy Converted Enhanced CT Image instances."""

    def __init__(
            self,
            legacy_datasets: Sequence[Dataset],
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            sort_key: Callable = None,
        ) -> None:
        """
        Parameters
        ----------

        Returns
        -------

        Note
        ----

        """
        """
        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.dataset.Dataset]
            DICOM data sets of legacy single-frame image instances that should
            be converted
        series_instance_uid: str
            UID of the series
        series_number: Union[int, None]
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        """
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')
        if ref_ds.Modality != 'CT':
            raise ValueError(
                'Wrong modality for conversion of legacy CT images.'
            )
        if ref_ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.2':
            raise ValueError(
                'Wrong SOP class for conversion of legacy CT images.'
            )
        super().__init__(
            legacy_datasets,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sort_key=sort_key
        )
        self._add_build_blocks_for_ct()
        self._convert2multiframe()


class LegacyConvertedEnhancedPETImage(_CommonLegacyConvertedEnhanceImage):

    """SOP class for Legacy Converted Enhanced PET Image instances."""

    def __init__(
            self,
            legacy_datasets: Sequence[Dataset],
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            sort_key: Callable = None,
        ) -> None:
        """
        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.dataset.Dataset]
            DICOM data sets of legacy single-frame image instances that should
            be converted
        series_instance_uid: str
            UID of the series
        series_number: Union[int, None]
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        """
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')
        if ref_ds.Modality != 'PT':
            raise ValueError(
                'Wrong modality for conversion of legacy PET images.'
            )
        if ref_ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.128':
            raise ValueError(
                'Wrong SOP class for conversion of legacy PET images.'
            )
        super().__init__(
            legacy_datasets,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sort_key=sort_key
        )
        self._add_build_blocks_for_pet()
        self._convert2multiframe()


class LegacyConvertedEnhancedMRImage(_CommonLegacyConvertedEnhanceImage):

    """SOP class for Legacy Converted Enhanced MR Image instances."""

    def __init__(
            self,
            legacy_datasets: Sequence[Dataset],
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            sort_key: Callable = None,
        ) -> None:
        """
        Parameters
        ----------

        Returns
        -------

        Note
        ----

        """
        """
        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.dataset.Dataset]
            DICOM data sets of legacy single-frame image instances that should
            be converted
        series_instance_uid: str
            UID of the series
        series_number: Union[int, None]
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        """
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')
        if ref_ds.Modality != 'MR':
            raise ValueError(
                'Wrong modality for conversion of legacy MR images.'
            )
        if ref_ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.4':
            raise ValueError(
                'Wrong SOP class for conversion of legacy MR images.'
            )
        super().__init__(
            legacy_datasets,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sort_key=sort_key
        )
        self._add_build_blocks_for_mr()
        self._convert2multiframe()
