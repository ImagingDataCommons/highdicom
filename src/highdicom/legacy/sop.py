""" Module for SOjkP Classes of Legacy Converted Enhanced Image IODs.
For the most part the single frame to multi-frame conversion logic is taken
from `PixelMed <https://www.dclunie.com>`_ by David Clunie

"""
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor
from copy import deepcopy
import logging
from sys import float_info
from typing import Any, List, Union, Callable, Sequence, Dict, Tuple

from pydicom.datadict import tag_for_keyword, dictionary_VR, keyword_for_tag
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate, encapsulate_extended
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.tag import Tag, BaseTag
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGLSLossless,
    RLELossless,
    UID,
)
from pydicom.valuerep import DT, DA, TM, DSfloat, format_number_as_ds

from highdicom.image import Image, _Image
from highdicom.base_content import ContributingEquipment
from highdicom.frame import encode_frame
from highdicom.spatial import get_series_volume_positions

# TODO defer these imports
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


_FARTHEST_FUTURE_DATE_TIME = DT('99991231235959')


def _istag_file_meta_information_group(t: BaseTag) -> bool:
    return t.group == 0x0002


def _istag_repeating_group(t: BaseTag) -> bool:
    g = t.group
    return (
        (g >= 0x5000 and g <= 0x501e) or
        (g >= 0x6000 and g <= 0x601e)
    )


def _istag_group_length(t: BaseTag) -> bool:
    return t.element == 0


def _isequal(v1: Any, v2: Any, float_tolerance: float = 1.0e-5) -> bool:
    def is_equal_float(x1: float, x2: float) -> bool:
        return abs(x1 - x2) < float_tolerance
    if type(v1) is not type(v2):
        return False
    if isinstance(v1, DataElementSequence):
        for item1, item2 in zip(v1, v2):
            if not _isequal_dicom_dataset(item1, item2):
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


def _isequal_dicom_dataset(ds1: Dataset, ds2: Dataset) -> bool:
    """Checks if two dicom dataset have the same value in all attributes

    Parameters
    ----------
    ds1: pydicom.Dataset
        1st dicom dataset
    ds2: pydicom.Dataset
        2nd dicom dataset

    Returns
    -------
    True if dicom datasets are equal otherwise False

    """
    if type(ds1) is not type(ds2):
        return False
    if not isinstance(ds1, Dataset):
        return False
    for k1, elem1 in ds1.items():
        if k1 not in ds2:
            return False
        elem2 = ds2[k1]
        if not _isequal(elem2.value, elem1.value):
            return False
    return True


def _display_tag(tg: BaseTag) -> str:
    """Converts tag to keyword and (group, element) form"""
    return f'{str(tg)}-{keyword_for_tag(tg):32.32s}'


def _transcode_frame(
    dataset: Dataset,
    transfer_syntax_uid: str,
) -> bytes:
    """Transcode single frame dataset's pixel data.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Single frame (legacy) dataset whose pixel data should be transcoded.
    transfer_syntax_uid: str
        New transfer syntax.

    Returns
    -------
    bytes:
        Pixel data of the input dataset transcoded to the given
        transfer_syntax_uid.

    """
    return encode_frame(
        array=dataset.pixel_array,
        transfer_syntax_uid=transfer_syntax_uid,
        bits_allocated=dataset.BitsAllocated,
        bits_stored=dataset.BitsStored,
        photometric_interpretation=dataset.PhotometricInterpretation,
        pixel_representation=dataset.get('PixelRepresentation', 0),
        planar_configuration=dataset.get('PlanarConfiguration'),
    )


class _FrameSet:

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
        single_frame_list: List[pydicom.Dataset]
            list of single frames that have equal distinguishing attributes
        distinguishing_tags: List[pydicom.tag.BaseTag]
            list of distinguishing attributes tags

        """
        self._frames = single_frame_list
        self._distinguishing_attributes_tags = distinguishing_tags
        tmp = [
            tag_for_keyword('AcquisitionDateTime'),
            tag_for_keyword('AcquisitionDate'),
            tag_for_keyword('AcquisitionTime'),
            tag_for_keyword('SpecificCharacterSet')
        ]
        self._excluded_from_perframe_tags = (
            self._distinguishing_attributes_tags + tmp
        )
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

    def _find_per_frame_and_shared_tags(self) -> None:
        """Detects and collects all shared and perframe attributes"""
        rough_shared: Dict[BaseTag, List[DataElement]] = defaultdict(list)
        sh_tgs = set()
        pf_tgs = set()
        sfs = self.frames
        for ds in sfs:
            for ttag, elem in ds.items():
                if (
                    not ttag.is_private and
                    not _istag_file_meta_information_group(ttag) and
                    not _istag_repeating_group(ttag) and
                    not _istag_group_length(ttag) and
                    self._istag_excluded_from_perframe(ttag) and
                    ttag != tag_for_keyword('PixelData')
                ):
                    # Since elem could be a RawDataElement so __getattr__ is
                    # safer and gives DataElement type as output
                    elem = ds[ttag]
                    pf_tgs.add(ttag)
                    rough_shared[ttag].append(elem.value)

        sh_tgs = set(rough_shared.keys())
        for ttag, v in rough_shared.items():
            if len(v) < len(self.frames):
                sh_tgs.remove(ttag)
            else:
                all_values_are_equal = all(
                    _isequal(v_i, v[0]) for v_i in v
                )
                if not all_values_are_equal:
                    sh_tgs.remove(ttag)

        pf_tgs -= sh_tgs
        self._shared_tags = list(sh_tgs)
        self._perframe_tags = list(pf_tgs)

    def _istag_excluded_from_perframe(self, t: BaseTag) -> bool:
        return t in self._excluded_from_perframe_tags


class _FrameSetCollection:

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
        single_frame_list: Sequence[pydicom.Dataset]
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
            'AcquisitionContextSequence',
        ]
        self._frame_sets: List[_FrameSet] = []
        frame_counts = []
        frameset_counter = 0
        while len(self.mixed_frames_copy) != 0:
            frameset_counter += 1
            ds_list, distinguishing_tgs = (
                self._find_all_similar_to_first_datasets()
            )
            # removing similar datasets from mixed frames
            for ds in ds_list:
                if ds in self.mixed_frames_copy:
                    self.mixed_frames_copy = [
                        nds for nds in self.mixed_frames_copy if nds != ds
                    ]

            self._frame_sets.append(_FrameSet(ds_list, distinguishing_tgs))
            frame_counts.append(len(ds_list))
            # log information
            logger.debug(
                f"Frameset({frameset_counter:02d}) "
                f"including {len(ds_list):03d} frames"
            )
            logger.debug('\t Distinguishing tags:')
            for dg_i, dg_tg in enumerate(distinguishing_tgs, 1):
                logger.debug(
                    f'\t\t{dg_i:02d}/{len(distinguishing_tgs)})\t{str(dg_tg)}-'
                    f'{keyword_for_tag(dg_tg):32.32s} = '
                    f'{str(ds_list[0][dg_tg].value):32.32s}')

            logger.debug('\t dicom datasets in this frame set:')

            for dicom_i, dicom_ds in enumerate(ds_list, 1):
                logger.debug(
                    f'\t\t{dicom_i}/{len(ds_list)})\t '
                    f'{dicom_ds["SOPInstanceUID"]}')

        frames = ''
        for i, f_count in enumerate(frame_counts, 1):
            frames += f'{i:2d}){f_count:03d}\t'
        frames += (
            f'{len(frame_counts):2d} frameset(s) out of all '
            f'{len(self.mixed_frames):3d} instances:'
        )
        logger.info(frames)
        self._excluded_from_perframe_tags = {}
        for kwkw in self._distinguishing_attribute_keywords:
            self._excluded_from_perframe_tags[tag_for_keyword(kwkw)] = False

        excluded_kws = [
            'AcquisitionDateTime',
            'AcquisitionDate',
            'AcquisitionTime',
            'SpecificCharacterSet',
        ]
        for kwkw in excluded_kws:
            self._excluded_from_perframe_tags[tag_for_keyword(kwkw)] = False

    def _find_all_similar_to_first_datasets(
            self) -> Tuple[List[Dataset], List[BaseTag]]:
        """Takes the fist instance from mixed-frames and finds all dicom images
        that have the same distinguishing attributes.

        Returns
        -------
        Tuple[List[pydicom.Dataset], List[pydicom.tag.BaseTag]]
            a pair of similar datasets and the corresponding list of
            distinguishing tags

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
                        f'{_display_tag(tg)} is missing in all but '
                        f'{ds["SOPInstanceUID"]}'
                    )
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
                if not _isequal(ref_val, new_val):
                    logger_msg.add(
                        'Inequality on distinguishing '
                        f'attribute {_display_tag(tg)} -> '
                        f'{ref_val} != {new_val} \n'
                        f'series uid = {ds.SeriesInstanceUID}'
                    )
                    all_equal = False
                    break

            if all_equal:
                similar_ds.append(ds)

        for msg_ in logger_msg:
            logger.info(msg_)

        return (similar_ds, distinguishing_tags_existing)

    @property
    def distinguishing_attribute_keywords(self) -> List[str]:
        """Returns the list of all distinguishing attributes found."""
        return self._distinguishing_attribute_keywords[:]

    @property
    def frame_sets(self) -> List[_FrameSet]:
        """Returns the list of all FrameSets found."""
        return self._frame_sets


class _CommonLegacyConvertedEnhancedImage(Image):

    """SOP class for common Legacy Converted Enhanced instances."""

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        transfer_syntax_uid: str | None = None,
        use_extended_offset_table: bool = False,
        sort_key: Callable | None = None,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        workers: int | Executor = 0,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.Dataset]
            DICOM data sets of legacy single-frame image instances that should
            be converted
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of data
            elements. If ``None``(the default), the transfer syntax of the
            legacy datasets will be used and the frames will not be re-encoded.
            The following compressed transfer syntaxes are supported: JPEG 2000
            Lossless (``"1.2.840.10008.1.2.4.90"``) and JPEG-LS Lossless
            (``"1.2.840.10008.1.2.4.80"``).
        use_extended_offset_table: bool, optional
            Include an extended offset table instead of a basic offset table
            for encapsulated transfer syntaxes. Extended offset tables avoid
            size limitations on basic offset tables, and separate the offset
            table from the pixel data by placing it into metadata. However,
            they may be less widely supported than basic offset tables. This
            parameter is ignored if using a native (uncompressed) transfer
            syntax. The default value may change in a future release.
        sort_key: Callable | None, optional
            A function by which the single-frame instances will be sorted
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        workers: int | concurrent.futures.Executor, optional
            Number of worker processes to use for frame compression, if
            compression or transcoding is needed. If 0, no workers are used and
            compression is performed in the main process (this is the default
            behavior). If negative, as many processes are created as the
            machine has processors.

            Alternatively, you may directly pass an instance of a class derived
            from ``concurrent.futures.Executor`` (most likely an instance of
            ``concurrent.futures.ProcessPoolExecutor``) for highdicom to use.
            You may wish to do this either to have greater control over the
            setup of the executor, or to avoid the setup cost of spawning new
            processes each time this ``__init__`` method is called if your
            application creates a large number of objects.

            Note that if you use worker processes, you must ensure that your
            main process uses the ``if __name__ == "__main__"`` idiom to guard
            against spawned child processes creating further workers.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('At least one legacy dataset must be provided.')

        sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
        all_framesets = _FrameSetCollection(legacy_datasets)
        if len(all_framesets.frame_sets) > 1:
            raise ValueError(
                'Mixed frames sets: the input single frame list contains more '
                'than one multiframe collection'
            )
        frame_set = all_framesets.frame_sets[0]
        if sort_key is None:
            sort_key = _CommonLegacyConvertedEnhancedImage.default_sort_key

        legacy_datasets = sorted(legacy_datasets, key=sort_key)

        content_date, content_time, acquisition_datetime = self._get_datetimes(
            legacy_datasets
        )

        super(_Image, self).__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            transfer_syntax_uid=transfer_syntax_uid,
            #  Manufacturer is type 2
            manufacturer=getattr(ref_ds, "Manufacturer", None),
            #  Modality is type 1
            modality=ref_ds.Modality,
            #  PatientID is type 2
            patient_id=getattr(ref_ds, "PatientID", None),
            #  PatientName is type 2
            patient_name=getattr(ref_ds, "PatientName", None),
            #  PatientBirthDate is type 2
            patient_birth_date=getattr(ref_ds, "PatientBirthDate", None),
            #  PatientSex is type 2
            patient_sex=getattr(ref_ds, "PatientSex", None),
            #  AccessionNumber is type 2
            accession_number=getattr(ref_ds, "AccessionNumber", None),
            #  StudyID is type 2
            study_id=getattr(ref_ds, "StudyID", None),
            #  StudyDate is type 2
            study_date=getattr(ref_ds, "StudyDate", None),
            #  StudyTime is type 2
            study_time=getattr(ref_ds, "StudyTime", None),
            #  ReferringPhysicianName is type 2
            referring_physician_name=getattr(
                ref_ds, "ReferringPhysicianName", None
            ),
            content_date=content_date,
            content_time=content_time,
            **kwargs
        )
        self._legacy_datasets = legacy_datasets
        self.SharedFunctionalGroupsSequence = [Dataset()]
        self.PerFrameFunctionalGroupsSequence = [
            Dataset() for _ in range(len(legacy_datasets))
        ]
        self._distinguishing_attributes_tags = self._get_tag_used_dictionary(
            frame_set.distinguishing_attributes_tags
        )
        self._excluded_from_perframe_tags = self._get_tag_used_dictionary(
            frame_set.excluded_from_perframe_tags
        )
        self._perframe_tags = self._get_tag_used_dictionary(
            frame_set.perframe_tags
        )
        self._shared_tags = self._get_tag_used_dictionary(
            frame_set.shared_tags
        )
        self.excluded_from_functional_groups_tags = {
            tag_for_keyword('SpecificCharacterSet'): False
        }
        self._build_blocks: List[Any] = []

        self._module_excepted_list: Dict[str, List[str]] = {
            "patient": [],
            "clinical-trial-subject": [],
            "general-study": [
                "StudyInstanceUID",
                "RequestingService"
            ],
            "patient-study": [
                "ReasonForVisit",
                "ReasonForVisitCodeSequence"
            ],
            "clinical-trial-study": [],
            "general-series": [
                "SeriesInstanceUID",
                "SeriesNumber",
                "SmallestPixelValueInSeries",
                "LargestPixelValueInSeries",
                "PerformedProcedureStepEndDate",
                "PerformedProcedureStepEndTime"
            ],
            "clinical-trial-series": [],
            "general-equipment": [
                "InstitutionalDepartmentTypeCodeSequence"
            ],
            "frame-of-reference": [],
            "sop-common": [
                "SOPClassUID",
                "SOPInstanceUID",
                "InstanceNumber",
                "SpecificCharacterSet",
                "EncryptedAttributesSequence",
                "MACParametersSequence",
                "DigitalSignaturesSequence"
            ],
            "general-image": [
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
            "sr-document-general": [
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
        self._copy_pixel_data(
            legacy_datasets,
            transfer_syntax_uid=transfer_syntax_uid,
            workers=workers,
            use_extended_offset_table=use_extended_offset_table,
        )

        if acquisition_datetime is not None:
            self.AcquisitionDateTime = acquisition_datetime

        self._add_common_ct_pet_mr_build_blocks()
        self._build_luts()

    def _get_datetimes(
        self,
        legacy_datasets: list[Dataset],
    ) -> Tuple[DA | None, TM | None, DT | None]:
        """Choose appropriate date/times for the new object.

        Chooses the content date, content time, and acquisition datetime for
        the new dataset using attributes of the legacy datasets.

        Content date and content time are chosen as the earliest acquisition,
        series, or study datetime in the source datasets. Acquisition datetime
        is chosen as the earliest acquisition datetime in the source datasets.

        Parameters
        ----------
        legacy_datasets: list[pydicom.Dataset]
            Legacy datasets

        Returns
        -------
        content_date: pydicom.valuerep.DA | None
            Content date for the new dataset.
        content_time: pydicom.valuerep.TM | None
            Content time for the new dataset.
        acquisition_datetime: pydicom.valuerep.DT | None
            Acquisition datetime for the new dataset.

        """
        earliest_content_date_time = _FARTHEST_FUTURE_DATE_TIME
        earliest_acquisition_date_time = _FARTHEST_FUTURE_DATE_TIME

        for src in legacy_datasets:
            frame_acquisition_datetime = None
            frame_content_datetime = None

            if 'AcquisitionDateTime' in src:
                frame_content_datetime = DT(src.AcquisitionDateTime)
                frame_acquisition_datetime = frame_content_datetime
            elif 'AcquisitionDate' in src and 'AcquisitionTime' in src:
                frame_content_datetime = DT.combine(
                    DA(src.AcquisitionDate),
                    TM(src.AcquisitionTime)
                )
                frame_acquisition_datetime = frame_content_datetime
            elif 'SeriesDate' in src and 'SeriesTime' in src:
                frame_content_datetime = DT.combine(
                    DA(src.SeriesDate),
                    TM(src.SeriesTime)
                )
            elif 'StudyDate' in src and 'StudyTime' in src:
                if src.StudyDate is not None and src.StudyTime is not None:
                    frame_content_datetime = DT.combine(
                        DA(src.StudyDate),
                        TM(src.StudyTime)
                    )

            if frame_content_datetime is not None:
                earliest_content_date_time = min(
                    frame_content_datetime,
                    earliest_content_date_time
                )
            if frame_acquisition_datetime is not None:
                earliest_acquisition_date_time = min(
                    frame_acquisition_datetime,
                    earliest_acquisition_date_time
                )

        content_date = None
        content_time = None
        acquisition_datetime = None
        if earliest_content_date_time < _FARTHEST_FUTURE_DATE_TIME:
            content_date = DA(earliest_content_date_time.date())
            content_time = TM(earliest_content_date_time.time())

        if earliest_acquisition_date_time < _FARTHEST_FUTURE_DATE_TIME:
            acquisition_datetime = DT(earliest_acquisition_date_time)

        return content_date, content_time, acquisition_datetime

    def _is_empty_or_empty_items(self, attribute: DataElement) -> bool:
        """Takes a dicom DataElement and check if DataElement is empty or in
        case of Sequence returns True if there is no item or all the items
        are empty.

        Parameters
        ----------
        attrib: pydicom.dataelem.DataElement
            input DICOM attribute whose emptiness will be checked.

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
        elif tg in self._excluded_from_perframe_tags:
            self._excluded_from_perframe_tags[tg] = True
        elif tg in self._perframe_tags:
            self._perframe_tags[tg] = True

    def _copy_attrib_if_present(
        self,
        src_ds: Dataset,
        dest_ds: Dataset,
        src_kw_or_tg: str | int,
        dest_kw_or_tg: str | int | None = None,
        ignore_if_perframe: bool = True,
        ignore_if_empty: bool = False,
    ) -> None:
        """Copies a dicom attribute value from a keyword in the source Dataset
        to the same keyword or a different keyword in the destination Dataset

        Parameters
        ----------
        src_ds: pydicom.Dataset
            Source dataset to copy the attribute from.
        dest_ds: pydicom.Dataset
            Destination dataset to copy the attribute to.
        src_kw_or_tg: str
            The keyword from the source dataset to copy.
        dest_kw_or_tg: str | int | None, optional
            The keyword of the destination dataset, to copy the value to. If
            None, then the source keyword is used.
        ignore_if_perframe: bool
            If true, then copy is aborted if the source attribute is per-frame.
        ignore_if_empty: bool
            If true, copy is aborted if the source attribute is empty.

        """
        if isinstance(src_kw_or_tg, str):
            src_kw_or_tg = tag_for_keyword(src_kw_or_tg)
        if dest_kw_or_tg is None:
            dest_kw_or_tg = src_kw_or_tg
        elif isinstance(dest_kw_or_tg, str):
            dest_kw_or_tg = tag_for_keyword(dest_kw_or_tg)
        if ignore_if_perframe and src_kw_or_tg in self._perframe_tags:
            return
        if src_kw_or_tg in src_ds:
            elem = src_ds[src_kw_or_tg]
            if ignore_if_empty:
                if self._is_empty_or_empty_items(elem):
                    return
            new_elem = deepcopy(elem)
            if dest_kw_or_tg == src_kw_or_tg:
                dest_ds[dest_kw_or_tg] = new_elem
            else:
                new_elem1 = DataElement(
                    dest_kw_or_tg,
                    dictionary_VR(dest_kw_or_tg),
                    new_elem.value
                )
                dest_ds[dest_kw_or_tg] = new_elem1
            # now mark the attrib as used/done to keep track of every one of it
            self._mark_tag_as_used(src_kw_or_tg)

    def _add_module(
        self,
        module_name: str,
        excepted_attributes: List[str] | None = None,
        ignore_if_perframe: bool = True,
        ignore_if_empty: bool = False
    ) -> None:
        """Copies all attribute of a particular module to current SOPClass,
        excepting the excepted_attributes, from a reference frame (the first
        frame on the single frame list).

        Parameters
        ----------
        module_name: str:
            A hyphenated module name like `image-pixel`.
        excepted_attributes: List[str] = []
            List of all attributes that are not allowed to be copied
        ignore_if_perframe: bool
            If True, then the perframe attributes will not be
            copied.
        ignore_if_empty: bool
            If True, then the empty attributes will not be copied.

        """
        attribs: List[dict] = MODULE_ATTRIBUTE_MAP[module_name]
        ref_dataset = self._legacy_datasets[0]
        for a in attribs:
            kw: str = a['keyword']
            if excepted_attributes is not None and kw in excepted_attributes:
                continue
            if len(a['path']) == 0:
                self._copy_attrib_if_present(
                    ref_dataset,
                    self,
                    kw,
                    ignore_if_perframe=ignore_if_perframe,
                    ignore_if_empty=ignore_if_empty
                )

    def _add_image_pixel_module(self) -> None:
        """Copies/adds an `image_pixel` multiframe module to
        the current SOPClass from its single frame source.

        """
        self._add_module(
            "image-pixel",
            excepted_attributes=[
                "ColorSpace",
                "PixelDataProviderURL",
                "ExtendedOffsetTable",
                "ExtendedOffsetTableLengths",
                "PixelData",
            ],
            ignore_if_empty=False,
            ignore_if_perframe=True
        )

    def _add_enhanced_common_image_module(self) -> None:
        """Copies/adds an `enhanced_common_image` multiframe module to
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
            'LossyImageCompressionMethod'
        ]

        for kw in attribs_to_be_added:
            self._copy_attrib_if_present(
                ref_dataset,
                self,
                kw,
                ignore_if_perframe=True,
                ignore_if_empty=False
            )

        sum_compression_ratio = 0.0
        c_ratio_tag = tag_for_keyword('LossyImageCompressionRatio')
        if (
            tag_for_keyword('LossyImageCompression') in self._shared_tags and
            tag_for_keyword(
                'LossyImageCompressionMethod'
            ) in self._shared_tags and
            c_ratio_tag in self._perframe_tags
        ):
            for fr_ds in self._legacy_datasets:
                if 'LossyImageCompressionRatio' in fr_ds:
                    ratio = fr_ds.LossyImageCompressionRatio
                    try:
                        sum_compression_ratio += float(ratio)
                    except BaseException:
                        sum_compression_ratio += 1  # supposing uncompressed
                else:
                    sum_compression_ratio += 1
            avg_compression_ratio = (
                sum_compression_ratio / len(self._legacy_datasets)
            )
            avg_ratio_str = '{:.6f}'.format(avg_compression_ratio)
            self.LossyImageCompressionRatio = avg_ratio_str

        if tag_for_keyword('PresentationLUTShape') not in self._perframe_tags:
            # actually should really invert the pixel data if MONOCHROME1,
            #           since only MONOCHROME2 is permitted :(
            # also, do not need to check if PhotometricInterpretation is
            #           per-frame, since a distinguishing attribute
            phmi = self._legacy_datasets[0].get(
                'PhotometricInterpretation',
                'MONOCHROME2'
            )
            lut_shape_default = (
                "INVERTED" if phmi == 'MONOCHROME1'
                else "IDENTITY"
            )
            lut_shape = self._legacy_datasets[0].get(
                'PresentationLUTShape',
                lut_shape_default,
            )
            if lut_shape is None:
                lut_shape = lut_shape_default

            self.PresentationLUTShape = lut_shape

    def _add_contrast_bolus_module(self) -> None:
        """Copies/adds a `contrast_bolus` multiframe module to
        the current SOPClass from its single frame source.

        """
        self._add_module('contrast-bolus')

    def _add_enhanced_ct_image_module(self) -> None:
        """Copies/adds an `enhanced_ct_image` multiframe module to
        the current SOPClass from its single frame source.

        """
        pass
        # David's code doesn't hold anything for this module ... should ask him

    def _add_enhanced_pet_image_module(self) -> None:
        """Copies/adds an `enhanced_pet_image` multiframe module to
        the current SOPClass from its single frame source.

        """
        self.ContentQualification = (
            self._legacy_datasets[0].get('ContentQualification', 'RESEARCH')
        )

    def _add_enhanced_mr_image_module(self) -> None:
        """Copies/adds an `enhanced_mr_image` multiframe module to
        the current SOPClass from its single frame source.

        """
        self._copy_attrib_if_present(
            self._legacy_datasets[0],
            self,
            "ResonantNucleus",
            ignore_if_perframe=True,
            ignore_if_empty=True,
        )
        if 'ResonantNucleus' not in self:
            # derive from ImagedNucleus, which is the one used in legacy MR
            #  IOD, but does not have a standard list of defined terms ...
            #  (could check these :()
            self._copy_attrib_if_present(
                self._legacy_datasets[0],
                self,
                "ImagedNucleus",
                ignore_if_perframe=True,
                ignore_if_empty=True,
            )
        attr_to_be_copied = [
            "KSpaceFiltering",
            "MagneticFieldStrength",
            "ApplicableSafetyStandardAgency",
            "ApplicableSafetyStandardDescription",
        ]
        for attr in attr_to_be_copied:
            self._copy_attrib_if_present(
                self._legacy_datasets[0],
                self,
                attr,
                ignore_if_perframe=True,
                ignore_if_empty=True,
            )

    def _add_acquisition_context_module(self) -> None:
        """Copies/adds an `acquisition_context` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            tag_for_keyword('AcquisitionContextSequence')
            not in self._perframe_tags
        ):
            self.AcquisitionContextSequence = self._legacy_datasets[0].get(
                'AcquisitionContextSequence'
            )

    def _add_common_ct_mr_pet_image_description_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
        is_root: bool,
    ) -> None:
        """Copies/adds attributes related to
        `common_ct_mr_pet_image_description` to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source fataset from which the module's attribute values are copied
        destination: pydicom.Dataset
            Destination Dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.
        is_root: int
            Whether the destination represents the root of the Dataset, which
            uses ``ImageType``. Otherwise then the destination attribute will
            be ``FrameType``.

        """
        frame_type = source.ImageType
        dest_kw = 'ImageType' if is_root else 'FrameType'
        lng = len(frame_type)
        new_val = [
            'ORIGINAL' if lng == 0 else frame_type[0],
            'PRIMARY',
            'VOLUME' if lng < 3 else frame_type[2],
            'NONE',
        ]
        setattr(destination, dest_kw, new_val)

        destination.PixelPresentation = 'MONOCHROME'
        destination.VolumetricProperties = 'VOLUME'
        destination.VolumeBasedCalculationTechnique = 'NONE'

    def _add_common_ct_mr_pet_image_description_module(
        self,
        modality: str,
    ) -> None:
        """Copies/adds the common attributes for ct/mr/pet description
        module to the current SOPClass from its single frame source.

        """
        image_or_empty = '' if modality == 'PET' else 'Image'
        seq_kw = f'{modality}{image_or_empty}FrameTypeSequence'

        if tag_for_keyword('ImageType') not in self._perframe_tags:
            self._add_common_ct_mr_pet_image_description_module_to_dataset(
                self._legacy_datasets[0], self, 0
            )
            inner_item = Dataset()
            self._add_common_ct_mr_pet_image_description_module_to_dataset(
                self._legacy_datasets[0], inner_item, 1
            )
            setattr(
                self.SharedFunctionalGroupsSequence[0],
                seq_kw,
                [inner_item]
            )
        else:
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                inner_item = Dataset()
                self._add_common_ct_mr_pet_image_description_module_to_dataset(
                    leg_ds, inner_item, 1
                )
                setattr(pffg, seq_kw, [inner_item])

    def _add_composite_instance_contex_module(self) -> None:
        """Copies/adds a `composite_instance_contex` multiframe module to
        the current SOPClass from its single frame source.

        """
        for module_name, excepted_attrs in self._module_excepted_list.items():
            self._add_module(
                module_name,
                excepted_attributes=excepted_attrs,
                ignore_if_empty=False,
                ignore_if_perframe=True,
            )

    def _add_frame_anatomy_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `frame_anatomy`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'AnatomicRegionSequence',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        if len(item) != 0:
            self._copy_attrib_if_present(
                source,
                item,
                'FrameLaterality',
                ignore_if_perframe=False,
                ignore_if_empty=True,
            )
            if 'FrameLaterality' not in item:
                self._copy_attrib_if_present(
                    source,
                    item,
                    'ImageLaterality',
                    'FrameLaterality',
                    ignore_if_perframe=False,
                    ignore_if_empty=True,
                )
            if 'FrameLaterality' not in item:
                self._copy_attrib_if_present(
                    source,
                    item,
                    'Laterality',
                    'FrameLaterality',
                    ignore_if_perframe=False,
                    ignore_if_empty=True,
                )
            if 'FrameLaterality' not in item:
                item.FrameLaterality = source.get('FrameLaterality', 'U')

            destination.FrameAnatomySequence = [item]

    def _has_frame_anatomy(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `frame_anatomy` present in source single frames.
        Otherwise returns false.

        """
        laterality_tg = tag_for_keyword('Laterality')
        im_laterality_tg = tag_for_keyword('ImageLaterality')
        bodypart_tg = tag_for_keyword('BodyPartExamined')
        anatomical_reg_tg = tag_for_keyword('AnatomicRegionSequence')
        return (
            laterality_tg in tags or
            im_laterality_tg in tags or
            bodypart_tg in tags or
            anatomical_reg_tg
        )

    def _add_frame_anatomy_module(self) -> None:
        """Copies/adds a `frame_anatomy` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_frame_anatomy(self._perframe_tags) and
            (
                self._has_frame_anatomy(self._shared_tags) or
                self._has_frame_anatomy(self._excluded_from_perframe_tags)
            )
        ):
            item = self.SharedFunctionalGroupsSequence[0]
            self._add_frame_anatomy_module_to_dataset(
                self._legacy_datasets[0],
                item
            )
        elif self._has_frame_anatomy(self._perframe_tags):
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                self._add_frame_anatomy_module_to_dataset(leg_ds, pffg)

    def _has_pixel_measures(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `pixel_measures` present in source single frames.
        Otherwise returns false.

        """
        pixel_spacing_tg = tag_for_keyword('PixelSpacing')
        slice_thickness_tg = tag_for_keyword('SliceThickness')
        imager_pixel_spacing_tg = tag_for_keyword('ImagerPixelSpacing')
        return (
            pixel_spacing_tg in tags or
            slice_thickness_tg in tags or
            imager_pixel_spacing_tg in tags
        )

    def _add_pixel_measures_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `pixel_measures`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'PixelSpacing',
            ignore_if_perframe=False,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'SliceThickness',
            ignore_if_perframe=False,
        )
        if 'PixelSpacing' not in item:
            self._copy_attrib_if_present(
                source,
                item,
                'ImagerPixelSpacing',
                'PixelSpacing',
                ignore_if_perframe=False,
                ignore_if_empty=True,
            )

        destination.PixelMeasuresSequence = [item]

    def _add_pixel_measures_module(self) -> None:
        """Copies/adds a `pixel_measures` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_pixel_measures(self._perframe_tags) and
            (
                self._has_pixel_measures(self._shared_tags) or
                self._has_pixel_measures(self._excluded_from_perframe_tags)
            )
        ):
            item = self.SharedFunctionalGroupsSequence[0]
            self._add_pixel_measures_module_to_dataset(
                self._legacy_datasets[0], item
            )
        elif self._has_pixel_measures(self._perframe_tags):
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                self._add_pixel_measures_module_to_dataset(leg_ds, pffg)

    def _has_plane_position(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `plane_position` present in source single frames.
        Otherwise returns false.

        """
        image_position_patient_tg = tag_for_keyword('ImagePositionPatient')
        return image_position_patient_tg in tags

    def _add_plane_position_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `plane_position`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'ImagePositionPatient',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )

        destination.PlanePositionSequence = [item]

    def _add_plane_position_module(self) -> None:
        """Copies/adds a `plane_position` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_plane_position(self._perframe_tags) and
            (
                self._has_plane_position(self._shared_tags) or
                self._has_plane_position(self._excluded_from_perframe_tags)
            )
        ):
            self._add_plane_position_module_to_dataset(
                self._legacy_datasets[0],
                self.SharedFunctionalGroupsSequence[0],
            )
        elif self._has_plane_position(self._perframe_tags):
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                self._add_plane_position_module_to_dataset(leg_ds, pffg)

    def _has_plane_orientation(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `plane_orientation` present in source single frames.
        Otherwise returns false.

        """
        image_orientation_patient_tg = tag_for_keyword(
            'ImageOrientationPatient'
        )
        return image_orientation_patient_tg in tags

    def _add_plane_orientation_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `plane_orientation`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'ImageOrientationPatient',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        destination.PlaneOrientationSequence = [item]

    def _add_plane_orientation_module(self) -> None:
        """Copies/adds a `plane_orientation` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_plane_orientation(self._perframe_tags) and
            (
                self._has_plane_orientation(self._shared_tags) or
                self._has_plane_orientation(self._excluded_from_perframe_tags)
            )
        ):
            item = self.SharedFunctionalGroupsSequence[0]
            self._add_plane_orientation_module_to_dataset(
                self._legacy_datasets[0], item
            )
        elif self._has_plane_orientation(self._perframe_tags):
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                self._add_plane_orientation_module_to_dataset(leg_ds, pffg)

    def _has_frame_voi_lut(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `frame_voi_lut` present in source single frames.
        Otherwise returns false.

        """
        window_width_tg = tag_for_keyword('WindowWidth')
        window_center_tg = tag_for_keyword('WindowCenter')
        window_center_width_explanation_tg = tag_for_keyword(
            'WindowCenterWidthExplanation'
        )
        return (
            window_width_tg in tags or
            window_center_tg in tags or
            window_center_width_explanation_tg in tags
        )

    def _add_frame_voi_lut_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `frame_voi_lut`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'WindowWidth',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'WindowCenter',
            ignore_if_perframe=False,
            ignore_if_empty=False
        )
        self._copy_attrib_if_present(
            source,
            item,
            'WindowCenterWidthExplanation',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        destination.FrameVOILUTSequence = [item]

    def _add_frame_voi_lut_module(self) -> None:
        """Copies/adds a `frame_voi_lut` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_frame_voi_lut(self._perframe_tags) and
            (
                self._has_frame_voi_lut(self._shared_tags) or
                self._has_frame_voi_lut(self._excluded_from_perframe_tags)
            )
        ):
            item = self.SharedFunctionalGroupsSequence[0]
            self._add_frame_voi_lut_module_to_dataset(
                self._legacy_datasets[0], item
            )
        elif self._has_frame_voi_lut(self._perframe_tags):
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                self._add_frame_voi_lut_module_to_dataset(leg_ds, pffg)

    def _has_pixel_value_transformation(
        self,
        tags: Dict[BaseTag, bool]
    ) -> bool:
        """returns true if attributes specific to
        `pixel_value_transformation` present in source single frames.
        Otherwise returns false.

        """
        rescale_intercept_tg = tag_for_keyword('RescaleIntercept')
        rescale_slope_tg = tag_for_keyword('RescaleSlope')
        rescale_type_tg = tag_for_keyword('RescaleType')
        return (
            rescale_intercept_tg in tags or
            rescale_slope_tg in tags or
            rescale_type_tg in tags
        )

    def _add_pixel_value_transformation_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `pixel_value_transformation`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'RescaleSlope',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'RescaleIntercept',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        have_values_so_add_type = (
            'RescaleSlope' in item or
            'RescaleIntercept' in item
        )
        self._copy_attrib_if_present(
            source,
            item,
            'RescaleType',
            ignore_if_perframe=False,
            ignore_if_empty=True,
        )
        value = ''
        if have_values_so_add_type:
            value = 'US'  # unspecified
            if source.get('Modality', '') == 'CT':
                image_type_v = (
                    [] if 'ImageType' not in source
                    else source['ImageType'].value
                )
                if not any(
                    i == 'LOCALIZER' for i in image_type_v
                ):
                    value = "HU"
            else:
                value = 'US'

            if 'RescaleType' not in item:
                item.RescaleType = value
            elif item.RescaleType != value:
                # keep the copied value as LUT explanation
                item.LUTExplanation = item.RescaleType
                item.RescaleType = value

        destination.PixelValueTransformationSequence = [item]

    def _add_pixel_value_transformation_module(self) -> None:
        """Copies/adds a `pixel_value_transformation` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_pixel_value_transformation(self._perframe_tags) and
            (
                self._has_pixel_value_transformation(self._shared_tags) or
                self._has_pixel_value_transformation(
                    self._excluded_from_perframe_tags
                )
            )
        ):
            item = self.SharedFunctionalGroupsSequence[0]
            self._add_pixel_value_transformation_module_to_dataset(
                self._legacy_datasets[0], item
            )
        elif self._has_pixel_value_transformation(self._perframe_tags):
            for item, legacy in zip(
                self.PerFrameFunctionalGroupsSequence,
                self._legacy_datasets
            ):
                self._add_referenced_image_module_to_dataset(legacy, item)

    def _has_referenced_image(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `referenced_image` present in source single frames.
        Otherwise returns false.

        """
        return tag_for_keyword('ReferencedImageSequence') in tags

    def _add_referenced_image_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `referenced_image`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        self._copy_attrib_if_present(
            source,
            destination,
            'ReferencedImageSequence',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )

    def _add_referenced_image_module(self) -> None:
        """Copies/adds a `referenced_image` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_referenced_image(self._perframe_tags) and
            (
                self._has_referenced_image(self._shared_tags) or
                self._has_referenced_image(self._excluded_from_perframe_tags)
            )
        ):
            item = self.SharedFunctionalGroupsSequence[0]
            self._add_referenced_image_module_to_dataset(
                self._legacy_datasets[0], item
            )
        elif self._has_referenced_image(self._perframe_tags):
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                self._add_referenced_image_module_to_dataset(leg_ds, pffg)

    def _has_derivation_image(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `derivation_image` present in source single frames.
        Otherwise returns false.

        """
        return tag_for_keyword('SourceImageSequence') in tags

    def _add_derivation_image_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `derivation_image`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'DerivationDescription',
            ignore_if_perframe=False,
            ignore_if_empty=True,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'DerivationCodeSequence',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'SourceImageSequence',
            ignore_if_perframe=False,
            ignore_if_empty=False,
        )
        destination.DerivationImageSequence = [item]

    def _add_derivation_image_module(self) -> None:
        """Copies/adds a `derivation_image` multiframe module to
        the current SOPClass from its single frame source.

        """
        if (
            not self._has_derivation_image(self._perframe_tags) and
            (
                self._has_derivation_image(self._shared_tags) or
                self._has_derivation_image(self._excluded_from_perframe_tags)
            )
        ):
            item = self.SharedFunctionalGroupsSequence[0]
            self._add_derivation_image_module_to_dataset(
                self._legacy_datasets[0], item
            )
        elif self._has_derivation_image(self._perframe_tags):
            for leg_ds, pffg in zip(
                self._legacy_datasets,
                self.PerFrameFunctionalGroupsSequence
            ):
                self._add_derivation_image_module_to_dataset(leg_ds, pffg)

    def _get_tag_used_dictionary(
        self,
        input: List[BaseTag]
    ) -> Dict[BaseTag, bool]:
        """Returns a dictionary of input tags with a use flag

        Parameters
        ----------
        input: List[pydicom.tag.BaseTag]
            list of tags to build dictionary holding their used flag.

        Returns
        -------
        dict: Dict[pydicom.tag.BaseTag, bool]
            a dictionary type of tags with used flag.

        """
        return {item: False for item in input}

    def _add_unassigned_perframe_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `unassigned_perframe`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        for tg in self._eligible_tags:
            self._copy_attrib_if_present(
                source,
                item,
                tg,
                ignore_if_perframe=False,
                ignore_if_empty=False,
            )
        destination.UnassignedPerFrameConvertedAttributesSequence = [item]

    def _add_largest_smallest_pixel_value(self) -> None:
        """Adds the attributes for largest and smallest pixel value to
        current SOPClass object

        """
        ltg = tag_for_keyword("LargestImagePixelValue")
        lval = float_info.min
        if ltg in self._perframe_tags:
            for frame in self._legacy_datasets:
                if ltg in frame:
                    nval = frame[ltg].value
                else:
                    continue
                lval = nval if lval < nval else lval
            if lval > float_info.min:
                self.LargestImagePixelValue = int(lval)

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
                self.SmallestImagePixelValue = int(sval)

    def _add_unassigned_perframe_module(self) -> None:
        """Copies/adds an `unassigned_perframe` multiframe module to
        the current SOPClass from its single frame source.

        """
        # first collect all not used tags
        # note that this is module is order dependent
        self._add_largest_smallest_pixel_value()
        self._eligible_tags: List[BaseTag] = []
        for tg, used in self._perframe_tags.items():
            if not used and tg not in self.excluded_from_functional_groups_tags:
                self._eligible_tags.append(tg)

        for leg_ds, pffg in zip(
            self._legacy_datasets,
            self.PerFrameFunctionalGroupsSequence
        ):
            self._add_unassigned_perframe_module_to_dataset(leg_ds, pffg)

    def _add_unassigned_shared_module(self) -> None:
        """Copies/adds an `unassigned_shared` multiframe module to
        the current SOPClass from its single frame source.

        """
        item = Dataset()
        for tg, used in self._shared_tags.items():
            if (
                not used and
                tg not in self and
                tg not in self.excluded_from_functional_groups_tags
            ):
                self._copy_attrib_if_present(
                    self._legacy_datasets[0],
                    item,
                    tg,
                    ignore_if_perframe=False,
                    ignore_if_empty=False,
                )

        setattr(
            self.SharedFunctionalGroupsSequence[0],
            'UnassignedSharedConvertedAttributesSequence',
            [item]
        )

    def _add_conversion_source_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `conversion_source`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()
        self._copy_attrib_if_present(
            source,
            item,
            'SOPClassUID',
            'ReferencedSOPClassUID',
            ignore_if_perframe=False,
            ignore_if_empty=True,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'SOPInstanceUID',
            'ReferencedSOPInstanceUID',
            ignore_if_perframe=False,
            ignore_if_empty=True,
        )
        destination.ConversionSourceAttributesSequence = [item]

    def _add_conversion_source_module(self) -> None:
        """Copies/adds a `conversion_source` multiframe module to
        the current SOPClass from its single frame source.

        """
        for leg_ds, pffg in zip(
            self._legacy_datasets,
            self.PerFrameFunctionalGroupsSequence
        ):
            self._add_conversion_source_module_to_dataset(leg_ds, pffg)

    def _add_stack_info_frame_content(self) -> None:
        """Adds stack info to the FrameContentSequence dicom attribute."""
        spacing, position_indices = get_series_volume_positions(
            self._legacy_datasets,
            allow_missing_positions=True,
            allow_duplicate_positions=True,
        )

        if spacing is not None and position_indices is not None:
            stack_id = '1'

            for pffg, pos in zip(
                self.PerFrameFunctionalGroupsSequence,
                position_indices
            ):
                pffg.FrameContentSequence[0].StackID = stack_id
                pffg.FrameContentSequence[0].InStackPositionNumber = int(pos)

            sfgs = self.SharedFunctionalGroupsSequence[0]
            if 'PixelMeasuresSequence' in sfgs:
                (
                    sfgs.PixelMeasuresSequence[0].SpacingBetweenSlices
                ) = format_number_as_ds(spacing)

    def _has_frame_content(self, tags: Dict[BaseTag, bool]) -> bool:
        """returns true if attributes specific to
        `frame_content` present in source single frames.
        Otherwise returns false.

        """
        acquisition_date_time_tg = tag_for_keyword('AcquisitionDateTime')
        acquisition_date_tg = tag_for_keyword('AcquisitionDate')
        acquisition_time_tg = tag_for_keyword('AcquisitionTime')
        return (
            acquisition_date_time_tg in tags or
            acquisition_time_tg in tags or
            acquisition_date_tg in tags
        )

    def _add_frame_content_module_to_dataset(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Copies/adds attributes related to `frame_content`
        to destination dicom Dataset

        Parameters
        ----------
        source: pydicom.Dataset
            Source dataset from which the module's attribute values
            are copied
        destination: pydicom.Dataset
            Destination dataset to which the module's attribute
            values are copied. The destination Dataset usually is an item
            from a perframe/shared functional group sequence.

        """
        item = Dataset()

        item.FrameAcquisitionNumber = source.get('AcquisitionNumber', 0)
        self._mark_tag_as_used(tag_for_keyword('AcquisitionNumber'))

        fa_dt: DT | None = None
        if 'AcquisitionDateTime' in source:
            fa_dt = DT(source.AcquisitionDateTime)
        elif 'AcquisitionDate' in source and 'AcquisitionTime' in source:
            fa_dt = DT(
                datetime.combine(
                    DA(source.AcquisitionDate),
                    TM(source.AcquisitionTime)
                )
            )

        if fa_dt is not None:
            if not self._has_frame_content(self._perframe_tags):
                if (
                    'TriggerTime' in source and
                    'FrameReferenceDateTime' not in source
                ):
                    trigger_time_in_millisecond = int(source.TriggerTime)
                    if trigger_time_in_millisecond > 0:
                        t_delta = timedelta(trigger_time_in_millisecond)
                        fa_dt += t_delta

            destination.FrameAcquisitionDateTime = fa_dt

        self._copy_attrib_if_present(
            source,
            item,
            'AcquisitionDuration',
            'FrameAcquisitionDuration',
            ignore_if_perframe=False,
            ignore_if_empty=True,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'TemporalPositionIndex',
            ignore_if_perframe=False,
            ignore_if_empty=True,
        )
        self._copy_attrib_if_present(
            source,
            item,
            'ImageComments',
            'FrameComments',
            ignore_if_perframe=False,
            ignore_if_empty=True,
        )
        destination.FrameContentSequence = [item]

    def _add_acquisition_info_frame_content(self) -> None:
        """Adds acquisition information to the FrameContentSequence dicom
        attribute.

        """
        for leg_ds, pffg in zip(
            self._legacy_datasets,
            self.PerFrameFunctionalGroupsSequence
        ):
            self._add_frame_content_module_to_dataset(leg_ds, pffg)

    def _add_frame_content_module(self) -> None:
        """Copies/adds a 'frame_content` multiframe module to
        the current SOPClass from its single frame source.

        """
        self._add_acquisition_info_frame_content()
        self._add_stack_info_frame_content()

    def _copy_pixel_data(
        self,
        legacy_datasets: list[Dataset],
        transfer_syntax_uid: str | None,
        workers: int | Executor = 0,
        use_extended_offset_table: bool = False,
    ) -> None:
        """Set pixel data by optionally transcoding and combining legacy frames.

        Parameters
        ----------
        legacy_datasets: list[pydicom.Dataset]
            Legacy datasets (in order) whose pixel data is to combined.
        transfer_syntax_uid: str | None
            Transfer syntax UID to use to encode the frames in the new object.
            If None, the transfer syntax of the original objects is used.
        workers: int | concurrent.futures.Executor, optional
            Number of worker processes to use for frame compression, if
            compression or transcoding is needed. If 0, no workers are used and
            compression is performed in the main process (this is the default
            behavior). If negative, as many processes are created as the
            machine has processors.

            Alternatively, you may directly pass an instance of a class derived
            from ``concurrent.futures.Executor`` (most likely an instance of
            ``concurrent.futures.ProcessPoolExecutor``) for highdicom to use.
            You may wish to do this either to have greater control over the
            setup of the executor, or to avoid the setup cost of spawning new
            processes each time this ``__init__`` method is called if your
            application creates a large number of objects.

            Note that if you use worker processes, you must ensure that your
            main process uses the ``if __name__ == "__main__"`` idiom to guard
            against spawned child processes creating further workers.
        use_extended_offset_table: bool, optional
            Include an extended offset table instead of a basic offset table
            for encapsulated transfer syntaxes. Extended offset tables avoid
            size limitations on basic offset tables, and separate the offset
            table from the pixel data by placing it into metadata. However,
            they may be less widely supported than basic offset tables. This
            parameter is ignored if using a native (uncompressed) transfer
            syntax. The default value may change in a future release.

        """
        allowed_transfer_syntaxes = (
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            JPEG2000Lossless,
            JPEG2000,
            JPEGLSLossless,
            JPEGBaseline8Bit,
            RLELossless,
        )
        if (
            transfer_syntax_uid is not None and
            (transfer_syntax_uid not in allowed_transfer_syntaxes)
        ):
            raise ValueError(
                f"Transfer syntax '{transfer_syntax_uid}' not recognized or "
                'not supported.'
            )

        self.NumberOfFrames = len(legacy_datasets)

        src_tx_uid = legacy_datasets[0].file_meta.TransferSyntaxUID
        if transfer_syntax_uid is None:
            dst_tx_uid = src_tx_uid
        else:
            dst_tx_uid = UID(transfer_syntax_uid)

        if not isinstance(workers, (int, Executor)):
            raise TypeError(
                'Argument "workers" must be of type int or '
                'concurrent.futures.Executor (or a derived class).'
            )
        using_multiprocessing = (
            isinstance(workers, Executor) or workers != 0
        )

        frames: list[bytes]

        if (
            (dst_tx_uid != src_tx_uid) and
            (src_tx_uid.is_encapsulated or dst_tx_uid.is_encapsulated)
        ):
            if using_multiprocessing:
                # Use the existing executor or create one
                if isinstance(workers, Executor):
                    process_pool = workers
                else:
                    # If workers is negative, pass None to use all processors
                    process_pool = ProcessPoolExecutor(
                        workers if workers > 0 else None
                    )

                futures = [
                    process_pool.submit(
                        _transcode_frame,
                        dataset=ds,
                        transfer_syntax_uid=dst_tx_uid,
                    )
                    for ds in legacy_datasets
                ]

                frames = [fut.result() for fut in futures]

                if process_pool is not workers:
                    process_pool.shutdown()

            else:
                frames = [
                    _transcode_frame(ds, dst_tx_uid)
                    for ds in legacy_datasets
                ]

        else:
            # No transcoding is required, just concatenate frames
            frames = [ds.PixelData for ds in legacy_datasets]

        if dst_tx_uid.is_encapsulated:
            if use_extended_offset_table:
                (
                    self.PixelData,
                    self.ExtendedOffsetTable,
                    self.ExtendedOffsetTableLengths,
                ) = encapsulate_extended(frames)
            else:
                self.PixelData = encapsulate(frames)
        else:
            self.PixelData = b''.join(frames)

    @staticmethod
    def default_sort_key(
        x: Dataset) -> Tuple[Union[int, str, UID], ...]:
        """The default sort key to sort all single frames before conversion

        Parameters
        ----------
        x: pydicom.Dataset
            input Dataset to be sorted.

        Returns
        -------
        tuple: Tuple[Union[int, str, UID]]
            a sort key of three elements.
                1st priority: SeriesNumber
                2nd priority: InstanceNumber
                3rd priority: SOPInstanceUID

        """
        out: tuple = tuple()
        if 'SeriesNumber' in x:
            out += (x.SeriesNumber, )
        if 'InstanceNumber' in x:
            out += (x.InstanceNumber, )
        if 'SOPInstanceUID' in x:
            out += (x.SOPInstanceUID, )
        return out

    def _add_common_ct_pet_mr_build_blocks(self) -> None:
        """Arranges common methods for multiframe conversion and
        put them in place.

        """
        self._build_blocks.extend(
            [
                [self._add_image_pixel_module, None],
                [self._add_composite_instance_contex_module, None],
                [self._add_enhanced_common_image_module, None],
                [self._add_acquisition_context_module, None],
                [self._add_frame_anatomy_module, None],
                [self._add_pixel_measures_module, None],
                [self._add_plane_orientation_module, None],
                [self._add_plane_position_module, None],
                [self._add_frame_voi_lut_module, None],
                [self._add_pixel_value_transformation_module, None],
                [self._add_referenced_image_module, None],
                [self._add_conversion_source_module, None],
                [self._add_frame_content_module, None],
                [self._add_unassigned_perframe_module, None],
                [self._add_unassigned_shared_module, None],
            ]
        )

    def _run_conversion(self) -> None:
        """Runs all necessary methods to convert from single frame to
        multi-frame.

        """
        for fun, args in self._build_blocks:
            if not args:
                fun()
            else:
                fun(*args)


class LegacyConvertedEnhancedCTImage(_CommonLegacyConvertedEnhancedImage):

    """SOP class for Legacy Converted Enhanced CT Image instances."""

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        sort_key: Callable | None = None,
        transfer_syntax_uid: str | None = None,
        use_extended_offset_table: bool = False,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        workers: int | Executor = 0,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.Dataset]
            DICOM data sets of legacy single-frame image instances that should
            be converted
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        transfer_syntax_uid: str | None, optional
            UID of transfer syntax that should be used for encoding of data
            elements. If ``None``(the default), the transfer syntax of the
            legacy datasets will be used and the frames will not be re-encoded.
            The following compressed transfer syntaxes are supported: JPEG 2000
            Lossless (``"1.2.840.10008.1.2.4.90"``) and JPEG-LS Lossless
            (``"1.2.840.10008.1.2.4.80"``).
        use_extended_offset_table: bool, optional
            Include an extended offset table instead of a basic offset table
            for encapsulated transfer syntaxes. Extended offset tables avoid
            size limitations on basic offset tables, and separate the offset
            table from the pixel data by placing it into metadata. However,
            they may be less widely supported than basic offset tables. This
            parameter is ignored if using a native (uncompressed) transfer
            syntax. The default value may change in a future release.
        sort_key: Callable | None, optional
            A function by which the single-frame instances will be sorted
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        workers: int | concurrent.futures.Executor, optional
            Number of worker processes to use for frame compression, if
            compression or transcoding is needed. If 0, no workers are used and
            compression is performed in the main process (this is the default
            behavior). If negative, as many processes are created as the
            machine has processors.

            Alternatively, you may directly pass an instance of a class derived
            from ``concurrent.futures.Executor`` (most likely an instance of
            ``concurrent.futures.ProcessPoolExecutor``) for highdicom to use.
            You may wish to do this either to have greater control over the
            setup of the executor, or to avoid the setup cost of spawning new
            processes each time this ``__init__`` method is called if your
            application creates a large number of objects.

            Note that if you use worker processes, you must ensure that your
            main process uses the ``if __name__ == "__main__"`` idiom to guard
            against spawned child processes creating further workers.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

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
            transfer_syntax_uid=transfer_syntax_uid,
            use_extended_offset_table=use_extended_offset_table,
            workers=workers,
            contributing_equipment=contributing_equipment,
            sort_key=sort_key,
            **kwargs
        )
        self._add_ct_specific_build_blocks()
        self._run_conversion()

    def _add_ct_specific_build_blocks(self) -> None:
        """Arranges CT specific methods for multiframe conversion and
        put them in place.

        """
        self._build_blocks.extend(
            [
                [
                    self._add_common_ct_mr_pet_image_description_module,
                    ('CT', )
                ],
                [self._add_enhanced_ct_image_module, None],
                [self._add_contrast_bolus_module, None],
            ]
        )


class LegacyConvertedEnhancedPETImage(_CommonLegacyConvertedEnhancedImage):

    """SOP class for Legacy Converted Enhanced PET Image instances."""

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        transfer_syntax_uid: str | None = None,
        use_extended_offset_table: bool = False,
        sort_key: Callable | None = None,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        workers: int | Executor = 0,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.Dataset]
            DICOM data sets of legacy single-frame image instances that should
            be converted
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        transfer_syntax_uid: str | None, optional
            UID of transfer syntax that should be used for encoding of data
            elements. If ``None``(the default), the transfer syntax of the
            legacy datasets will be used and the frames will not be re-encoded.
            The following compressed transfer syntaxes are supported: JPEG 2000
            Lossless (``"1.2.840.10008.1.2.4.90"``) and JPEG-LS Lossless
            (``"1.2.840.10008.1.2.4.80"``).
        use_extended_offset_table: bool, optional
            Include an extended offset table instead of a basic offset table
            for encapsulated transfer syntaxes. Extended offset tables avoid
            size limitations on basic offset tables, and separate the offset
            table from the pixel data by placing it into metadata. However,
            they may be less widely supported than basic offset tables. This
            parameter is ignored if using a native (uncompressed) transfer
            syntax. The default value may change in a future release.
        sort_key: Callable | None, optional
            A function by which the single-frame instances will be sorted
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        workers: int | concurrent.futures.Executor, optional
            Number of worker processes to use for frame compression, if
            compression or transcoding is needed. If 0, no workers are used and
            compression is performed in the main process (this is the default
            behavior). If negative, as many processes are created as the
            machine has processors.

            Alternatively, you may directly pass an instance of a class derived
            from ``concurrent.futures.Executor`` (most likely an instance of
            ``concurrent.futures.ProcessPoolExecutor``) for highdicom to use.
            You may wish to do this either to have greater control over the
            setup of the executor, or to avoid the setup cost of spawning new
            processes each time this ``__init__`` method is called if your
            application creates a large number of objects.

            Note that if you use worker processes, you must ensure that your
            main process uses the ``if __name__ == "__main__"`` idiom to guard
            against spawned child processes creating further workers.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

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
            transfer_syntax_uid=transfer_syntax_uid,
            use_extended_offset_table=use_extended_offset_table,
            sort_key=sort_key,
            workers=workers,
            contributing_equipment=contributing_equipment,
            **kwargs
        )
        self._add_pet_specific_build_blocks()
        self._run_conversion()

    def _add_pet_specific_build_blocks(self) -> None:
        """Arranges PET specific methods for multiframe conversion and
        put them in place

        """
        self._build_blocks.extend(
            [
                [
                    self._add_common_ct_mr_pet_image_description_module,
                    ('PET', )
                ],
                [self._add_enhanced_pet_image_module, None],
            ]
        )


class LegacyConvertedEnhancedMRImage(_CommonLegacyConvertedEnhancedImage):

    """SOP class for Legacy Converted Enhanced MR Image instances."""

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        transfer_syntax_uid: str | None = None,
        use_extended_offset_table: bool = False,
        sort_key: Callable | None = None,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        workers: int | Executor = 0,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.Dataset]
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
        transfer_syntax_uid: str | None, optional
            UID of transfer syntax that should be used for encoding of data
            elements. If ``None``(the default), the transfer syntax of the
            legacy datasets will be used and the frames will not be re-encoded.
            The following compressed transfer syntaxes are supported: JPEG 2000
            Lossless (``"1.2.840.10008.1.2.4.90"``) and JPEG-LS Lossless
            (``"1.2.840.10008.1.2.4.80"``).
        use_extended_offset_table: bool, optional
            Include an extended offset table instead of a basic offset table
            for encapsulated transfer syntaxes. Extended offset tables avoid
            size limitations on basic offset tables, and separate the offset
            table from the pixel data by placing it into metadata. However,
            they may be less widely supported than basic offset tables. This
            parameter is ignored if using a native (uncompressed) transfer
            syntax. The default value may change in a future release.
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        workers: int | concurrent.futures.Executor, optional
            Number of worker processes to use for frame compression, if
            compression or transcoding is needed. If 0, no workers are used and
            compression is performed in the main process (this is the default
            behavior). If negative, as many processes are created as the
            machine has processors.

            Alternatively, you may directly pass an instance of a class derived
            from ``concurrent.futures.Executor`` (most likely an instance of
            ``concurrent.futures.ProcessPoolExecutor``) for highdicom to use.
            You may wish to do this either to have greater control over the
            setup of the executor, or to avoid the setup cost of spawning new
            processes each time this ``__init__`` method is called if your
            application creates a large number of objects.

            Note that if you use worker processes, you must ensure that your
            main process uses the ``if __name__ == "__main__"`` idiom to guard
            against spawned child processes creating further workers.
        sort_key: Callable | None, optional
            A function by which the single-frame instances will be sorted

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
            transfer_syntax_uid=transfer_syntax_uid,
            use_extended_offset_table=use_extended_offset_table,
            workers=workers,
            contributing_equipment=contributing_equipment,
            sort_key=sort_key,
            **kwargs
        )
        self._add_mr_specific_build_blocks()
        self._run_conversion()

    def _add_mr_specific_build_blocks(self) -> None:
        """Arranges MRI specific methods for multiframe conversion and
        put them in place

        """
        self._build_blocks.extend(
            [
                [
                    self._add_common_ct_mr_pet_image_description_module,
                    ('MR', )
                ],
                [self._add_enhanced_mr_image_module, None],
                [self._add_contrast_bolus_module, None],
            ]
        )
