""" Module for SOP Classes of Legacy Converted Enhanced Image IODs."""
from __future__ import annotations
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from numpy import log10, array, ceil, cross, dot, ndarray
from pydicom.datadict import tag_for_keyword, dictionary_VR, keyword_for_tag
from pydicom.dataset import Dataset
from pydicom.tag import Tag, BaseTag
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence as DicomSequence
from pydicom.multival import MultiValue
from datetime import date, datetime, time, timedelta
from pydicom.valuerep import DT, DA, TM
from copy import deepcopy
from pydicom.uid import UID
from highdicom.base import SOPClass
from highdicom.legacy import SOP_CLASS_UIDS
from highdicom._iods import IOD_MODULE_MAP
from highdicom._modules import MODULE_ATTRIBUTE_MAP
# logger = logging.getLogger(__name__)
LEGACY_ENHANCED_SOP_CLASS_UID_MAP = {
    # CT Image Storage
    '1.2.840.10008.5.1.4.1.1.2': '1.2.840.10008.5.1.4.1.1.2.2',
    # MR Image Storage
    '1.2.840.10008.5.1.4.1.1.4': '1.2.840.10008.5.1.4.1.1.4.4',
    # PET Image Storage
    '1.2.840.10008.5.1.4.1.1.128': '1.2.840.10008.5.1.4.1.1.128.1',
}
_SOP_CLASS_UID_IOD_KEY_MAP = {
    '1.2.840.10008.5.1.4.1.1.2.2':  'legacy-converted-enhanced-ct-image',
    '1.2.840.10008.5.1.4.1.1.4.4':  'legacy-converted-enhanced-mr-image',
    '1.2.840.10008.5.1.4.1.1.128.1': 'legacy-converted-enhanced-pet-image',
}


def _convert_legacy_to_enhanced(
        sf_datasets: Sequence[Dataset],
        mf_dataset: Optional[Dataset] = None
) -> Dataset:
    """Converts one or more MR, CT or PET Image instances into one
    Legacy Converted Enhanced MR/CT/PET Image instance by copying information
    from `sf_datasets` into `mf_dataset`.
    Parameters
    ----------
    sf_datasets: Sequence[pydicom.dataset.Dataset]
        DICOM data sets of single-frame legacy image instances
    mf_dataset: pydicom.dataset.Dataset, optional
        DICOM data set of multi-frame enhanced image instance
    Returns
    -------
    pydicom.dataset.Dataset
        DICOM data set of enhanced multi-frame image instance
    Note
    ----
    Frames will be included into the Pixel Data element in the order in
    which instances are provided via `sf_datasets`.
    """
    logger = logging.getLogger(__name__)
    try:
        ref_ds = sf_datasets[0]
    except IndexError:
        raise ValueError('No data sets of single-frame legacy images provided.')
    if mf_dataset is None:
        mf_dataset = Dataset()
    transfer_syntaxes = set()
    series = set()
    studies = set()
    modalities = set()
    for ds in sf_datasets:
        transfer_syntaxes.add(ds.file_meta.TransferSyntaxUID)
        series.add(ds.SeriesInstanceUID)
        studies.add(ds.StudyInstanceUID)
        modalities.add(ds.Modality)
    if len(series) > 1:
        raise ValueError(
            'All instances must belong to the same series.')
    if len(studies) > 1:
        raise ValueError(
            'All instances must belong to the same study.')
    if len(modalities) > 1:
        raise ValueError(
            'All instances must have the same modality.')
    if len(transfer_syntaxes) > 1:
        raise ValueError(
            'All instances must have the same transfer syntaxes.')
    sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
    mf_dataset.NumberOfFrames = len(sf_datasets)
    # We will ignore some attributes, because they will get assigned new
    # values in the legacy converted enhanced image instance.
    ignored_attributes = {
        tag_for_keyword('NumberOfFrames'),
        tag_for_keyword('InstanceNumber'),
        tag_for_keyword('SOPClassUID'),
        tag_for_keyword('SOPInstanceUID'),
        tag_for_keyword('PixelData'),
        tag_for_keyword('SeriesInstanceUID'),
    }
    mf_attributes = []
    iod_key = _SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid]
    for module_item in IOD_MODULE_MAP[iod_key]:
        module_key = module_item['key']
        for attr_item in MODULE_ATTRIBUTE_MAP[module_key]:
            # Only root-level attributes
            if len(attr_item['path']) > 0:
                continue
            tag = tag_for_keyword(attr_item['keyword'])
            if tag in ignored_attributes:
                continue
            mf_attributes.append(tag)
    # Assign attributes that are not defined at the root level of the
    # Lecacy Converted Enhanced MR/CT/PET Image IOD to the appropriate
    # sequence attributes of the SharedFunctinoalGroupsSequence or
    # PerFrameFunctionalGroupsSequence attributes. Collect all unassigned
    # attributes (we will deal with them later on).
    # IODs only cover the modules, but not functional group macros.
    # Therefore, we need to handle those separately.
    assigned_attributes = {
        # shared
        tag_for_keyword('ImageOrientationPatient'),
        tag_for_keyword('PixelSpacing'),
        tag_for_keyword('SliceThickness'),
        tag_for_keyword('SpacingBetweenSlices'),
        # per-frame
        tag_for_keyword('ImageType'),
        tag_for_keyword('AcquisitionDate'),
        tag_for_keyword('AcquisitionTime'),
        tag_for_keyword('InstanceNumber'),
        tag_for_keyword('SOPClassUID'),
        tag_for_keyword('SOPInstanceUID'),
        tag_for_keyword('ImagePositionPatient'),
        tag_for_keyword('WindowCenter'),
        tag_for_keyword('WindowWidth'),
        tag_for_keyword('ReferencedImageSequence'),
        tag_for_keyword('SourceImageSequence'),
        tag_for_keyword('BodyPartExamined'),
        tag_for_keyword('IrradiationEventUID'),
        tag_for_keyword('RescaleIntercept'),
        tag_for_keyword('RescaleSlope'),
        tag_for_keyword('RescaleType'),
    }
    if ref_ds.ImageType[0] == 'ORIGINAL':
        mf_dataset.VolumeBasedCalculationTechnique = 'NONE'
    else:
        mf_dataset.VolumeBasedCalculationTechnique = 'MIXED'
    pixel_representation = sf_datasets[0].PixelRepresentation
    volumetric_properties = 'VOLUME'
    unique_image_types = set()
    unassigned_dataelements: Dict[str, List[Dataset]] = defaultdict(list)
    # Per-Frame Functional Groups
    perframe_items = []
    for i, ds in enumerate(sf_datasets):
        perframe_item = Dataset()
        # Frame Content (M)
        frame_content_item = Dataset()
        if 'AcquisitionDate' in ds and 'AcquisitionTime' in ds:
            frame_content_item.FrameAcquisitionDateTime = '{}{}'.format(
                ds.AcquisitionDate,
                ds.AcquisitionTime)
        frame_content_item.FrameAcquisitionNumber = ds.InstanceNumber
        perframe_item.FrameContentSequence = [
            frame_content_item,
        ]
        # Plane Position (Patient) (M)
        plane_position_item = Dataset()
        plane_position_item.ImagePositionPatient = ds.ImagePositionPatient
        perframe_item.PlanePositionSequence = [
            plane_position_item,
        ]
        frame_type = list(ds.ImageType)
        if len(frame_type) < 4:
            if frame_type[0] == 'ORIGINAL':
                frame_type.append('NONE')
            else:
                logger.warn('unknown derived pixel contrast')
                frame_type.append('OTHER')
        unique_image_types.add(tuple(frame_type))
        frame_type_item = Dataset()
        frame_type_item.FrameType = frame_type
        frame_type_item.PixelRepresentation = pixel_representation
        frame_type_item.VolumetricProperties = volumetric_properties
        if frame_type[0] == 'ORIGINAL':
            frame_type_item.FrameVolumeBasedCalculationTechnique = 'NONE'
        else:
            frame_type_item.FrameVolumeBasedCalculationTechnique = 'MIXED'
        if sop_class_uid == '1.2.840.10008.5.1.4.1.1.4.4':
            # MR Image Frame Type (M)
            perframe_item.MRImageFrameTypeSequence = [
                frame_type_item,
            ]
        elif sop_class_uid == '1.2.840.10008.5.1.4.1.1.2.2':
            # CT Image Frame Type (M)
            perframe_item.CTImageFrameTypeSequence = [
                frame_type_item,
            ]
            # CT Pixel Value Transformation (M)
            pixel_val_transform_item = Dataset()
            pixel_val_transform_item.RescaleIntercept = ds.RescaleIntercept
            pixel_val_transform_item.RescaleSlope = ds.RescaleSlope
            try:
                pixel_val_transform_item.RescaleType = ds.RescaleType
            except AttributeError:
                pixel_val_transform_item.RescaleType = 'US'
            perframe_item.PixelValueTransformationSequence = [
                pixel_val_transform_item,
            ]
        elif sop_class_uid == '1.2.840.10008.5.1.4.1.1.128.1':
            # PET Image Frame Type (M)
            perframe_item.PETImageFrameTypeSequence = [
                frame_type_item,
            ]
        # Frame VOI LUT (U)
        try:
            frame_voi_lut_item = Dataset()
            frame_voi_lut_item.WindowCenter = ds.WindowCenter
            frame_voi_lut_item.WindowWidth = ds.WindowWidth
            perframe_item.FrameVOILUTSequence = [
                frame_voi_lut_item,
            ]
        except AttributeError:
            pass
        # Referenced Image (C)
        try:
            perframe_item.ReferencedImageSequence = \
                ds.ReferencedImageSequence
        except AttributeError:
            pass
        # Derivation Image (C)
        try:
            perframe_item.SourceImageSequence = ds.SourceImageSequence
        except AttributeError:
            pass
        # Frame Anatomy (C)
        try:
            frame_anatomy_item = Dataset()
            frame_anatomy_item.BodyPartExamined = ds.BodyPartExamined
            perframe_item.FrameAnatomySequence = [
                frame_anatomy_item,
            ]
        except AttributeError:
            pass
        # Image Frame Conversion Source (C)
        conv_src_attr_item = Dataset()
        conv_src_attr_item.ReferencedSOPClassUID = ds.SOPClassUID
        conv_src_attr_item.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        perframe_item.ConversionSourceAttributesSequence = [
            conv_src_attr_item,
        ]
        # Irradiation Event Identification (C) - CT/PET only
        try:
            irradiation_event_id_item = Dataset()
            irradiation_event_id_item.IrradiationEventUID = \
                ref_ds.IrradiationEventUID
            perframe_item.IrradiationEventIdentificationSequence = [
                irradiation_event_id_item,
            ]
        except AttributeError:
            pass
        # Temporal Position (U)
        try:
            temporal_position_item = Dataset()
            temporal_position_item.TemporalPositionTimeOffset = \
                ref_ds.TemporalPositionTimeOffset
            perframe_item.TemporalPositionSequence = [
                temporal_position_item,
            ]
        except AttributeError:
            pass
        # Cardiac Synchronization (U                # TODO: http: # dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html# sect_C.7.6.16.2.7  # noqa
        # Contrast/Bolus Usage (U) - MR/CT onl              # TODO: http: # dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html# sect_C.7.6.16.2.12  # noqa
        # Respiratory Synchronization (U                # TODO: http: # dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html# sect_C.7.6.16.2.17  # noqa
        # Real World Value Mapping (U) - PET onl                # TODO: http: # dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html# sect_C.7.6.16.2.11  # noqa
        perframe_items.append(perframe_item)
        # All other attributes that are not assigned to functional groups.
        for tag, da in ds.items():
            if tag in assigned_attributes:
                continue
            elif tag in mf_attributes:
                mf_dataset.add(da)
            else:
                if tag not in ignored_attributes:
                    unassigned_dataelements[tag].append(da)
    # All remaining unassigned attributes will be collected in either the
    # UnassignedSharedConvertedAttributesSequence or the
    # UnassignedPerFrameConvertedAttributesSequence, depending on whether
    # values vary accross frames (original single-frame image instances).
    unassigned_shared_ca_item = Dataset()
    unassigned_perframe_ca_items = [
        Dataset()
        for _ in range(len(sf_datasets))
    ]
    for tag, dataelements in unassigned_dataelements.items():
        values = [str(da.value) for da in dataelements]
        unique_values = set(values)
        if len(unique_values) == 1:
            unassigned_shared_ca_item.add(dataelements[0])
        else:
            for i, da in enumerate(dataelements):
                unassigned_perframe_ca_items[i].add(da)
    mf_dataset.ImageType = list(list(unique_image_types)[0])
    if len(unique_image_types) > 1:
        mf_dataset.ImageType[2] = 'MIXED'
    mf_dataset.PixelRepresentation = pixel_representation
    mf_dataset.VolumetricProperties = volumetric_properties
    # Shared Functional Groups
    shared_item = Dataset()
    # Pixel Measures (M)
    pixel_measures_item = Dataset()
    pixel_measures_item.PixelSpacing = ref_ds.PixelSpacing
    pixel_measures_item.SliceThickness = ref_ds.SliceThickness
    try:
        pixel_measures_item.SpacingBetweenSlices = \
            ref_ds.SpacingBetweenSlices
    except AttributeError:
        pass
    shared_item.PixelMeasuresSequence = [
        pixel_measures_item,
    ]
    # Plane Orientation (Patient) (M)
    plane_orientation_item = Dataset()
    plane_orientation_item.ImageOrientationPatient = \
        ref_ds.ImageOrientationPatient
    shared_item.PlaneOrientationSequence = [
        plane_orientation_item,
    ]
    shared_item.UnassignedSharedConvertedAttributesSequence = [
        unassigned_shared_ca_item,
    ]
    mf_dataset.SharedFunctionalGroupsSequence = [
        shared_item,
    ]
    for i, ca_item in enumerate(unassigned_perframe_ca_items):
        perframe_items[i].UnassignedPerFrameConvertedAttributesSequence = [
            ca_item,
        ]
    mf_dataset.PerFrameFunctionalGroupsSequence = perframe_items
    mf_dataset.AcquisitionContextSequence = []
    # TODO: Encapsulated Pixel Data with compressed frame items.
    # Create the Pixel Data element of the mulit-frame image instance using
    # native encoding (simply concatenating pixels of individual frames)
    # Sometimes there may be numpy types such as " > i2". The (* 1) hack
    # ensures that pixel values have the correct integer type.
    mf_dataset.PixelData = b''.join([
        (ds.pixel_array * 1).data for ds in sf_datasets
    ])
    return mf_dataset


class LegacyConvertedEnhancedMRImage(SOPClass):
    """SOP class for Legacy Converted Enhanced MR Image instances."""

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        **kwargs: Any) -> None:
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
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`
        """
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')
        if ref_ds.Modality != 'MR':
            raise ValueError(
                'Wrong modality for conversion of legacy MR images.')
        if ref_ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.4':
            raise ValueError(
                'Wrong SOP class for conversion of legacy MR images.')
        sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
        super().__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=ref_ds.Manufacturer,
            modality=ref_ds.Modality,
            transfer_syntax_uid=None,  # FIXME: frame encoding
            patient_id=ref_ds.PatientID,
            patient_name=ref_ds.PatientName,
            patient_birth_date=ref_ds.PatientBirthDate,
            patient_sex=ref_ds.PatientSex,
            accession_number=ref_ds.AccessionNumber,
            study_id=ref_ds.StudyID,
            study_date=ref_ds.StudyDate,
            study_time=ref_ds.StudyTime,
            referring_physician_name=ref_ds.ReferringPhysicianName,
            **kwargs)
        _convert_legacy_to_enhanced(legacy_datasets, self)
        self.PresentationLUTShape = 'IDENTITY'


class LegacyConvertedEnhancedCTImage(SOPClass):
    """SOP class for Legacy Converted Enhanced CT Image instances."""

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        **kwargs: Any) -> None:
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
                'Wrong modality for conversion of legacy CT images.')
        if ref_ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.2':
            raise ValueError(
                'Wrong SOP class for conversion of legacy CT images.')
        sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
        super().__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=ref_ds.Manufacturer,
            modality=ref_ds.Modality,
            transfer_syntax_uid=None,  # FIXME: frame encoding
            patient_id=ref_ds.PatientID,
            patient_name=ref_ds.PatientName,
            patient_birth_date=ref_ds.PatientBirthDate,
            patient_sex=ref_ds.PatientSex,
            accession_number=ref_ds.AccessionNumber,
            study_id=ref_ds.StudyID,
            study_date=ref_ds.StudyDate,
            study_time=ref_ds.StudyTime,
            referring_physician_name=ref_ds.ReferringPhysicianName,
            **kwargs)
        _convert_legacy_to_enhanced(legacy_datasets, self)


class LegacyConvertedEnhancedPETImage(SOPClass):
    """SOP class for Legacy Converted Enhanced PET Image instances."""

    def __init__(
            self,
            legacy_datasets: Sequence[Dataset],
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            **kwargs: Any) -> None:
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
                'Wrong modality for conversion of legacy PET images.')
        if ref_ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.128':
            raise ValueError(
                'Wrong SOP class for conversion of legacy PET images.')
        sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
        super().__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=ref_ds.Manufacturer,
            modality=ref_ds.Modality,
            transfer_syntax_uid=None,  # FIXME: frame encoding
            patient_id=ref_ds.PatientID,
            patient_name=ref_ds.PatientName,
            patient_birth_date=ref_ds.PatientBirthDate,
            patient_sex=ref_ds.PatientSex,
            accession_number=ref_ds.AccessionNumber,
            study_id=ref_ds.StudyID,
            study_date=ref_ds.StudyDate,
            study_time=ref_ds.StudyTime,
            referring_physician_name=ref_ds.ReferringPhysicianName,
            **kwargs)
        _convert_legacy_to_enhanced(legacy_datasets, self)


from abc import ABC, abstractmethod


class Abstract_MultiframeModuleAdder(ABC):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):

        self.ExcludedFromPerFrameTags = excluded_from_perframe_tags
        self.ExcludedFromFunctionalGroupsTags = excluded_from_functional_tags
        self._PerFrameTags = perframe_tags
        self._SharedTags = shared_tags
        self.TargetDataset = multi_frame_output
        self.SingleFrameSet = sf_datasets
        self.EarliestDate = DA('00010101')
        self.EarliestTime = TM('000000')
        self.EarliestDateTime = DT('00010101000000')
        self.FarthestFutureDate = DA('99991231')
        self.FarthestFutureTime = TM('235959')
        self.FarthestFutureDateTime = DT('99991231235959')

    def _is_empty_or_empty_items(self, attribute: DataElement) -> bool:
        if attribute.is_empty:
            return True
        if type(attribute.value) == Sequence:
            if len(attribute.value) == 0:
                return True
            for item in attribute.value:
                for tg, v in item.items():
                    v = item[tg]
                    if not self._is_empty_or_empty_items(v):
                        return False
        return False

    def _mark_tag_as_used(self, tg: BaseTag) -> None:
        if tg in self._SharedTags:
            self._SharedTags[tg] = True
        elif tg in self.ExcludedFromPerFrameTags:
            self.ExcludedFromPerFrameTags[tg] = True
        elif tg in self._PerFrameTags:
            self._PerFrameTags[tg] = True

    def _copy_attrib_if_present(self, src_ds: Dataset, dest_ds: Dataset,
                                src_kw_or_tg: str, dest_kw_or_tg: str = None,
                                check_not_to_be_perframe: bool = True,
                                check_not_to_be_empty: bool = False) -> None:
        if type(src_kw_or_tg) == str:
            src_kw_or_tg = tag_for_keyword(src_kw_or_tg)
        if dest_kw_or_tg is None:
            dest_kw_or_tg = src_kw_or_tg
        elif type(dest_kw_or_tg) == str:
            dest_kw_or_tg = tag_for_keyword(dest_kw_or_tg)
        if check_not_to_be_perframe:
            if src_kw_or_tg in self._PerFrameTags:
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

    def _get_perframe_item(self, index: int) -> Dataset:
        if index > len(self.SingleFrameSet):
            return None
        pf_kw: str = 'PerFrameFunctionalGroupsSequence'
        pf_tg = tag_for_keyword(pf_kw)
        if pf_tg not in self.TargetDataset:
            seq = []
            for i in range(0, len(self.SingleFrameSet)):
                seq.append(Dataset())
            self.TargetDataset[pf_tg] = DataElement(pf_tg,
                                                    'SQ',
                                                    DicomSequence(seq))
        return self.TargetDataset[pf_tg].value[index]

    def _get_shared_item(self) -> Dataset:
        sf_kw = 'SharedFunctionalGroupsSequence'
        sf_tg = tag_for_keyword(sf_kw)
        if sf_kw not in self.TargetDataset:
            seq = [Dataset()]
            self.TargetDataset[sf_tg] = DataElement(sf_tg,
                                                    'SQ',
                                                    DicomSequence(seq))
        return self.TargetDataset[sf_tg].value[0]

    def _get_or_create_attribute(
        self, src: Dataset, kw: Union[str, Tag], default: Any) -> DataElement:
        if kw is str:
            tg = tag_for_keyword(kw)
        else:
            tg = kw
        if kw in src:
            a = deepcopy(src[kw])
        else:
            a = DataElement(tg, dictionary_VR(tg), default)
        from pydicom.valuerep import DT, TM, DA
        if a.VR == 'DA' and type(a.value) == str:
            try:
                d_tmp = DA(a.value)
                a.value = DA(default) if d_tmp is None else d_tmp
            except: 
                a.value = DA(default)
        if a.VR == 'DT' and type(a.value) == str:
            try:
                dt_tmp = DT(a.value)
                a.value = DT(default) if dt_tmp is None else dt_tmp
            except:
                a.value = DT(default)
        if a.VR == 'TM' and type(a.value) == str:
            try:
                t_tmp = TM(a.value)
                a.value = TM(default) if t_tmp is None else t_tmp
            except:
                a.value = TM(default)

        self._mark_tag_as_used(tg)
        return a

    def _add_module(self, module_name: str, excepted_attributes: list = [],
                    check_not_to_be_perframe: bool = True,
                    check_not_to_be_empty: bool = False) -> None:
        # sf_sop_instance_uid = sf_datasets[0]
        # mf_sop_instance_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[
        #   sf_sop_instance_uid]
        # iod_name = _SOP_CLASS_UID_IOD_KEY_MAP[mf_sop_instance_uid]
        # modules = IOD_MODULE_MAP[iod_name]
        from copy import deepcopy
        attribs: list = MODULE_ATTRIBUTE_MAP[module_name]
        ref_dataset = self.SingleFrameSet[0]
        for a in attribs:
            kw: str = a['keyword']
            if kw in excepted_attributes:
                continue
            if len(a['path']) == 0:
                self._copy_attrib_if_present(
                    ref_dataset, self.TargetDataset, kw,
                    check_not_to_be_perframe=check_not_to_be_perframe,
                    check_not_to_be_empty=check_not_to_be_empty)

    @abstractmethod
    def AddModule(self) -> None:
        pass


class ImagePixelModule(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
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


class CompositeInstanceContex(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)
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

    def AddModule(self) -> None:
        for module_name, excpeted_a in self._module_excepted_list.items():
            self._add_module(
             module_name,
             excepted_attributes=excpeted_a,
             check_not_to_be_empty=False,
             check_not_to_be_perframe=True)  # don't check the perframe set


class CommonCTMRPETImageDescriptionMacro(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset,
                 modality: str = 'CT'):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)
        self.Modality = modality

    def _get_value_for_frame_type(self,
                                  attrib: DataElement) -> Union[list, None]:
        if type(attrib) != DataElement:
            return None
        output = ['', '', '', '']
        v = attrib.value
        lng = len(v)
        output[0] = 'ORIGINAL' if lng == 0 else v[0]
        output[1] = 'PRIMARY'
        output[2] = 'VOLUME' if lng < 3 else v[2]
        output[3] = 'NONE'
        return output

    def _get_frame_type_seq_tag(self) -> int:
        seq_kw = '{}{}FrameTypeSequence'
        if self.Modality == 'PET':
            seq_kw = seq_kw.format(self.Modality, '')
        else:
            seq_kw = seq_kw.format(self.Modality, 'Image')
        return tag_for_keyword(seq_kw)

    def _add_module_to_functional_group(self, src_fg: Dataset,
                                        dest_fg: Dataset, level: int) -> None:
        FrameType_a = src_fg['ImageType']
        if level == 0:
            FrameType_tg = tag_for_keyword('ImageType')
        else:
            FrameType_tg = tag_for_keyword('FrameType')
        new_val = self._get_value_for_frame_type(FrameType_a)
        dest_fg[FrameType_tg] = DataElement(FrameType_tg,
                                            FrameType_a.VR, new_val)

        def element_generator(kw: str, val: Any) -> DataElement:
            return DataElement(
                tag_for_keyword(kw),
                dictionary_VR(tag_for_keyword(kw)), val)
        dest_fg['PixelPresentation'] = element_generator(
            'PixelPresentation', "MONOCHROME")
        dest_fg['VolumetricProperties'] = element_generator(
            'VolumetricProperties', "VOLUME")
        dest_fg['VolumeBasedCalculationTechnique'] = element_generator(
            'VolumeBasedCalculationTechnique', "NONE")

    def AddModule(self) -> None:
        im_type_tag = tag_for_keyword('ImageType')
        seq_tg = self._get_frame_type_seq_tag()
        if im_type_tag not in self._PerFrameTags:
            self._add_module_to_functional_group(self.SingleFrameSet[0],
                                                 self.TargetDataset, 0)
            # ----------------------------
            item = self._get_shared_item()
            inner_item = Dataset()
            self._add_module_to_functional_group(self.SingleFrameSet[0],
                                                 inner_item, 1)
            item[seq_tg] = DataElement(seq_tg, 'SQ', [inner_item])
        else:
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                inner_item = Dataset()
                self._add_module_to_functional_group(self.SingleFrameSet[i],
                                                     inner_item, 1)
                item[seq_tg] = DataElement(seq_tg, 'SQ', [inner_item])


class EnhancedCommonImageModule(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
        # ct_mr = CommonCTMRImageDescriptionMacro(self.SingleFrameSet
        # , self.ExcludedFromPerFrameTags
        # , self._PerFrameTags
        # , self._SharedTags
        # , self.TargetDataset)
        # ct_mr.AddModule()
        # Acquisition Number
        # Acquisition DateTime - should be able to find earliest amongst all
        #   frames, if present (required if ORIGINAL)
        # Acquisition Duration - should be able to work this out, but type 2C,
        #   so can send empty
        # Referenced Raw Data Sequence - optional - ignore - too hard to merge
        # Referenced Waveform Sequence - optional - ignore - too hard to merge
        # Referenced Image Evidence Sequence - should add if we have references
        # Source Image Evidence Sequence - should add if we have sources : (
        # Referenced Presentation State Sequence - should merge if present in
        #   any source frame
        # Samples per Pixel - handled by distinguishingAttribute copy
        # Photometric Interpretation - handled by distinguishingAttribute copy
        # Bits Allocated - handled by distinguishingAttribute copy
        # Bits Stored - handled by distinguishingAttribute copy
        # High Bit - handled by distinguishingAttribute copy
        ref_dataset = self.SingleFrameSet[0]
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
                ref_dataset, self.TargetDataset, kw,
                check_not_to_be_perframe=True,
                check_not_to_be_empty=False)
        sum_compression_ratio = 0
        c_ratio_tag = tag_for_keyword('LossyImageCompressionRatio')
        if  tag_for_keyword('LossyImageCompression') in self._SharedTags and \
                tag_for_keyword(
                    'LossyImageCompressionMethod') in self._SharedTags and \
                c_ratio_tag in self._PerFrameTags:
            for fr_ds in self.SingleFrameSet:
                if c_ratio_tag in fr_ds:
                    ratio = fr_ds[c_ratio_tag].value
                    try:
                        sum_compression_ratio += float(ratio)
                    except:
                        sum_compression_ratio += 1 #  supposing uncompressed
                else:
                    supe_compression_ratio += 1
            avg_compression_ratio = sum_compression_ratio /\
                len(self.SingleFrameSet)
            avg_ratio_str = '{:.6f}'.format(avg_compression_ratio)
            self.TargetDataset[c_ratio_tag] = \
                DataElement(c_ratio_tag, 'DS', avg_ratio_str)

        if tag_for_keyword('PresentationLUTShape') not in self._PerFrameTags:
            # actually should really invert the pixel data if MONOCHROME1,
            #           since only MONOCHROME2 is permitted : (
            # also, do not need to check if PhotometricInterpretation is
            #           per-frame, since a distinguishing attribute
            phmi_kw = 'PhotometricInterpretation'
            phmi_a = self._get_or_create_attribute(self.SingleFrameSet[0],
                                                   phmi_kw,
                                                   "MONOCHROME2")
            LUT_shape_default = "INVERTED" if phmi_a.value == 'MONOCHROME1'\
                else "IDENTITY"
            LUT_shape_a = self._get_or_create_attribute(self.SingleFrameSet[0],
                                                        'PresentationLUTShape',
                                                        LUT_shape_default)
            if not LUT_shape_a.is_empty:
                self.TargetDataset['PresentationLUTShape'] = LUT_shape_a
        # Icon Image Sequence - always discard these


class ContrastBolusModule(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
        self._add_module('contrast-bolus')


class EnhancedCTImageModule(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
        pass
        # David's code doesn't hold anything for this module ... should ask him


class EnhancedPETImageModule(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
        # David's code doesn't hold anything for this module ... should ask him
        kw = 'ContentQualification'
        tg = tag_for_keyword(kw)
        elem = self._get_or_create_attribute(
            self.SingleFrameSet[0], kw, 'RESEARCH')
        self.TargetDataset[tg] = elem


class EnhancedMRImageModule(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
        self._copy_attrib_if_present(
            self.SingleFrameSet[0],
            self.TargetDataset,
            "ResonantNucleus",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        if 'ResonantNucleus' not in self.TargetDataset:
            # derive from ImagedNucleus, which is the one used in legacy MR
            #  IOD, but does not have a standard list of defined terms ...
            #  (could check these : ()
            self._copy_attrib_if_present(
                self.SingleFrameSet[0],
                self.TargetDataset,
                "ImagedNucleus",
                check_not_to_be_perframe=True,
                check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.SingleFrameSet[0],
            self.TargetDataset,
            "KSpaceFiltering",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.SingleFrameSet[0],
            self.TargetDataset,
            "MagneticFieldStrength",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.SingleFrameSet[0],
            self.TargetDataset,
            "ApplicableSafetyStandardAgency",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.SingleFrameSet[0],
            self.TargetDataset,
            "ApplicableSafetyStandardDescription",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)


class AcquisitionContextModule(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
        tg = tag_for_keyword('AcquisitionContextSequence')
        if tg not in self._PerFrameTags:
            self.TargetDataset[tg] = self._get_or_create_attribute(
                self.SingleFrameSet[0],
                tg,
                None)


class FrameAnatomyFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        # David's code is more complicaated than mine
        # Should check it out later.
        fa_seq_tg = tag_for_keyword('FrameAnatomySequence')
        item = Dataset()
        self._copy_attrib_if_present(src_fg, item, 'AnatomicRegionSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        if len(item) != 0:
            self._copy_attrib_if_present(src_fg, item, 'FrameLaterality',
                                        check_not_to_be_perframe=False,
                                        check_not_to_be_empty=True)
            if 'FrameLaterality' not in item:
                self._copy_attrib_if_present(src_fg, item, 'ImageLaterality',
                                            'FrameLaterality',
                                            check_not_to_be_perframe=False,
                                            check_not_to_be_empty=True)
            if 'FrameLaterality' not in item:
                self._copy_attrib_if_present(src_fg, item, 'Laterality',
                                            'FrameLaterality',
                                            check_not_to_be_perframe=False,
                                            check_not_to_be_empty=True)
            if 'FrameLaterality' not in item:
                FrameLaterality_a = self._get_or_create_attribute(
                    src_fg, 'FrameLaterality', "U")
                item['FrameLaterality'] = FrameLaterality_a
            FrameAnatomy_a = DataElement(fa_seq_tg,
                                        dictionary_VR(fa_seq_tg),
                                        [item])
            dest_fg['FrameAnatomySequence'] = FrameAnatomy_a

    def _contains_right_attributes(self, tags: dict) -> bool:
        laterality_tg = tag_for_keyword('Laterality')
        im_laterality_tg = tag_for_keyword('ImageLaterality')
        bodypart_tg = tag_for_keyword('BodyPartExamined')
        anatomical_reg_tg = tag_for_keyword('AnatomicRegionSequence')
        return (laterality_tg in tags or
                im_laterality_tg in tags or
                bodypart_tg in tags or
                anatomical_reg_tg)

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
             self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class PixelMeasuresFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _contains_right_attributes(self, tags: dict) -> bool:
        PixelSpacing_tg = tag_for_keyword('PixelSpacing')
        SliceThickness_tg = tag_for_keyword('SliceThickness')
        ImagerPixelSpacing_tg = tag_for_keyword('ImagerPixelSpacing')
        return (PixelSpacing_tg in tags or
                SliceThickness_tg in tags or
                ImagerPixelSpacing_tg in tags)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'PixelSpacing',
                                     check_not_to_be_perframe=False)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'SliceThickness',
                                     check_not_to_be_perframe=False)
        if 'PixelSpacing' not in item:
            self._copy_attrib_if_present(src_fg,
                                         item,
                                         'ImagerPixelSpacing',
                                         'PixelSpacing',
                                         check_not_to_be_perframe=False,
                                         check_not_to_be_empty=True)
        pixel_measures_kw = 'PixelMeasuresSequence'
        pixel_measures_tg = tag_for_keyword(pixel_measures_kw)
        seq = DataElement(pixel_measures_tg,
                          dictionary_VR(pixel_measures_tg),
                          [item])
        dest_fg[pixel_measures_tg] = seq

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
            self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class PlanePositionFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _contains_right_attributes(self, tags: dict) -> bool:
        ImagePositionPatient_tg = tag_for_keyword('ImagePositionPatient')
        return ImagePositionPatient_tg in tags

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'ImagePositionPatient',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        PlanePositionSequence_kw = 'PlanePositionSequence'
        PlanePositionSequence_tg = tag_for_keyword(PlanePositionSequence_kw)
        seq = DataElement(PlanePositionSequence_tg,
                          dictionary_VR(PlanePositionSequence_tg),
                          [item])
        dest_fg[PlanePositionSequence_tg] = seq

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
            self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class PlaneOrientationFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _contains_right_attributes(self, tags: dict) -> bool:
        ImageOrientationPatient_tg = tag_for_keyword('ImageOrientationPatient')
        return ImageOrientationPatient_tg in tags

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'ImageOrientationPatient',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        kw = 'PlaneOrientationSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
            self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class FrameVOILUTFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _contains_right_attributes(self, tags: dict) -> bool:
        WindowWidth_tg = tag_for_keyword('WindowWidth')
        WindowCenter_tg = tag_for_keyword('WindowCenter')
        WindowCenterWidthExplanation_tg = tag_for_keyword(
            'WindowCenterWidthExplanation')
        return (WindowWidth_tg in tags or
                WindowCenter_tg in tags or
                WindowCenterWidthExplanation_tg in tags)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'WindowWidth',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'WindowCenter',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'WindowCenterWidthExplanation',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        kw = 'FrameVOILUTSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
            self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class PixelValueTransformationFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _contains_right_attributes(self, tags: dict) -> bool:
        RescaleIntercept_tg = tag_for_keyword('RescaleIntercept')
        RescaleSlope_tg = tag_for_keyword('RescaleSlope')
        RescaleType_tg = tag_for_keyword('RescaleType')
        return (RescaleIntercept_tg in tags or
                RescaleSlope_tg in tags or
                RescaleType_tg in tags)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'RescaleSlope',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'RescaleIntercept',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        haveValuesSoAddType = ('RescaleSlope' in item or
                               'RescaleIntercept' in item)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'RescaleType',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        
        value = ''
        modality = '' if 'Modality' not in src_fg\
            else src_fg["Modality"].value
        if haveValuesSoAddType:
            value = 'US'
            if modality == 'CT':
                containes_localizer = False
                ImageType_v = [] if 'ImageType' not in src_fg\
                    else src_fg['ImageType'].value
                for i in ImageType_v:
                    if i == 'LOCALIZER':
                        containes_localizer = True
                        break
                if not containes_localizer:
                    value = "HU"
            # elif modality == 'PT':
                # value = 'US' if 'Units' not in src_fg\
                #     else src_fg['Units'].value
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
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
            self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class ReferencedImageFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _contains_right_attributes(self, tags: dict) -> bool:
        return tag_for_keyword('ReferencedImageSequence') in tags

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        self._copy_attrib_if_present(src_fg,
                                     dest_fg,
                                     'ReferencedImageSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
            self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class DerivationImageFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _contains_right_attributes(self, tags: dict) -> bool:
        return tag_for_keyword('SourceImageSequence') in tags

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'DerivationDescription',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'DerivationCodeSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'SourceImageSequence',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=False)
        kw = 'DerivationImageSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self) -> None:
        if (not self._contains_right_attributes(self._PerFrameTags) and
            (self._contains_right_attributes(self._SharedTags) or
            self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0], item)
        elif self._contains_right_attributes(self._PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.SingleFrameSet[i], item)


class UnassignedPerFrame(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        for tg in self._eligeible_tags:
            self._copy_attrib_if_present(src_fg,
                                         item,
                                         tg,
                                         check_not_to_be_perframe=False,
                                         check_not_to_be_empty=False)
        kw = 'UnassignedPerFrameConvertedAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def _add_largest_smallest_pixle_value(self) -> None:
        ltg = tag_for_keyword("LargestImagePixelValue")
        from sys import float_info
        lval = float_info.min
        if ltg in self._PerFrameTags:
            for frame in self.SingleFrameSet:
                if ltg in frame:
                    nval = frame[ltg].value
                else:
                    continue
                lval = nval if lval < nval else lval
            if lval > float_info.min:
                self.TargetDataset[ltg] = DataElement(ltg, 'SS', int(lval))
    # ==========================
        stg = tag_for_keyword("SmallestImagePixelValue")
        sval = float_info.max
        if stg in self._PerFrameTags:
            for frame in self.SingleFrameSet:
                if stg in frame:
                    nval = frame[stg].value
                else:
                    continue
                sval = nval if sval < nval else sval
            if sval < float_info.max:
                self.TargetDataset[stg] = DataElement(stg, 'SS', int(sval))

        stg = "SmallestImagePixelValue"

    def AddModule(self) -> None:
        # first collect all not used tags
        # note that this is module is order dependent
        self._add_largest_smallest_pixle_value()
        self._eligeible_tags: List[Tag] = []
        for tg, used in self._PerFrameTags.items():
            if not used and tg not in self.ExcludedFromFunctionalGroupsTags:
                self._eligeible_tags.append(tg)
        for i in range(0, len(self.SingleFrameSet)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(
                self.SingleFrameSet[i], item)


class UnassignedShared(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        for tg, used in self._SharedTags.items():
            if (not used and
                    tg not in self.TargetDataset and
                    tg not in self.ExcludedFromFunctionalGroupsTags):
                self._copy_attrib_if_present(src_fg,
                                             item,
                                             tg,
                                             check_not_to_be_perframe=False,
                                             check_not_to_be_empty=False)
        kw = 'UnassignedSharedConvertedAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self) -> None:
        item = self._get_shared_item()
        self._add_module_to_functional_group(self.SingleFrameSet[0], item)


class EmptyType2Attributes(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def CreateEmptyElement(self, tg: BaseTag) -> DataElement:
        return DataElement(tg, dictionary_VR(tg), None)

    def AddModule(self) -> None:
        iod_name = _SOP_CLASS_UID_IOD_KEY_MAP[
            self.TargetDataset['SOPClassUID'].value]
        modules = IOD_MODULE_MAP[iod_name]
        for module in modules:
            if module['usage'] == 'M':
                mod_key = module['key']
                attrib_list = MODULE_ATTRIBUTE_MAP[mod_key]
                for a in attrib_list:
                    if len(a['path']) == 0 and a['type'] == '2':
                        tg = tag_for_keyword(a['keyword'])
                        if (tg not in self.SingleFrameSet[0] and
                           tg not in self.TargetDataset and
                           tg not in self._PerFrameTags and
                           tg not in self._SharedTags):
                            self.TargetDataset[tg] =\
                                self.CreateEmptyElement(tg)


class ConversionSourceFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'SOPClassUID',
                                     'ReferencedSOPClassUID',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        self._copy_attrib_if_present(src_fg,
                                     item,
                                     'SOPInstanceUID',
                                     'ReferencedSOPInstanceUID',
                                     check_not_to_be_perframe=False,
                                     check_not_to_be_empty=True)
        kw = 'ConversionSourceAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self) -> None:
        for i in range(0, len(self.SingleFrameSet)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(
                self.SingleFrameSet[i], item)


class FrameContentFunctionalGroup(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)
        self.EarliestFrameAcquisitionDateTime = self.FarthestFutureDateTime
        self._slices: list = []
        self._tolerance = 0.0001
        self._slice_location_map: dict = {}

    def _build_slices_geometry(self) -> None:
        logger = logging.getLogger(__name__)
        frame_count = len(self.SingleFrameSet)
        for i in range(0, frame_count):
            curr_frame = self.SingleFrameSet[i]
            ImagePositionPatient_v = None \
                if 'ImagePositionPatient' not in curr_frame\
                else curr_frame['ImagePositionPatient'].value
            ImageOrientationPatient_v = None \
                if 'ImageOrientationPatient' not in curr_frame\
                else curr_frame['ImageOrientationPatient'].value
            PixelSpacing_v = None \
                if 'PixelSpacing' not in curr_frame\
                else curr_frame['PixelSpacing'].value
            SliceThickness_v = 0.0 \
                if 'SliceThickness' not in curr_frame\
                else curr_frame['SliceThickness'].value
            # SliceLocation_v = None \
            #     if 'SliceLocation' not in curr_frame\
            #     else curr_frame['SliceLocation'].value
            Rows_v = 0 \
                if 'Rows' not in curr_frame\
                else curr_frame['Rows'].value
            Columns_v = 0 \
                if 'Columns' not in curr_frame\
                else curr_frame['Columns'].value
            if (ImageOrientationPatient_v is not None and
                    ImagePositionPatient_v is not None and
                    PixelSpacing_v is not None):
                row = array(ImageOrientationPatient_v[0:3])
                col = array(ImageOrientationPatient_v[3:])
                voxel_spaceing = array([PixelSpacing_v[0],
                                        PixelSpacing_v[1],
                                        SliceThickness_v])
                tpl = array(ImagePositionPatient_v)
                dim = (Rows_v, Columns_v, 1)
                self._slices.append(GeometryOfSlice(row, col,
                                    tpl, voxel_spaceing, dim))
            else:
                logger.error(
                    "Error in geometry. One or more required "
                    "attributes are not available")
                logger.error("\tImageOrientationPatient = {}".format(
                    ImageOrientationPatient_v))
                logger.error("\tImagePositionPatient = {}".format(
                    ImagePositionPatient_v))
                logger.error("\tPixelSpacing = {}".format(PixelSpacing_v))
                self._slices = []  # clear the slices
                break

    def _are_all_slices_parallel(self) -> bool:
        slice_count = len(self._slices)
        if slice_count >= 2:
            last_slice = self._slices[0]
            for i in range(1, slice_count):
                curr_slice = self._slices[i]
                if not GeometryOfSlice.AreParallel(
                        curr_slice, last_slice, self._tolerance):
                    return False
                last_slice = curr_slice
            return True
        elif slice_count == 1:
            return True
        else:
            return False

    def _add_stack_info(self) -> None:
        logger = logging.getLogger(__name__)
        self._build_slices_geometry()
        round_digits = int(ceil(-log10(self._tolerance)))
        if self._are_all_slices_parallel():
            self._slice_location_map = {}
            for idx, s in enumerate(self._slices):
                not_round_dist = s.GetDistanceAlongOrigin()
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
                    logger.warning(
                        'There are {} slices in one location {}'.format(
                            len(idxs), loc)
                        )
                for frame_index in idxs:
                    frame = self._get_perframe_item(frame_index)
                    new_item = frame[frame_content_tg].value[0]
                    new_item["StackID"] = self._get_or_create_attribute(
                        self.SingleFrameSet[0],
                        "StackID", "0")
                    new_item["InStackPositionNumber"] =\
                        self._get_or_create_attribute(
                        self.SingleFrameSet[0],
                        "InStackPositionNumber", distance_index)
                distance_index += 1

    def _contains_right_attributes(self, tags: dict) -> bool:
        AcquisitionDateTime_tg = tag_for_keyword('AcquisitionDateTime')
        AcquisitionDate_tg = tag_for_keyword('AcquisitionDate')
        AcquisitionTime_tg = tag_for_keyword('AcquisitionTime')
        return (AcquisitionDateTime_tg in tags or
                AcquisitionTime_tg in tags or
                AcquisitionDate_tg in tags)

    def _add_module_to_functional_group(
        self, src_fg: Dataset, dest_fg: Dataset) -> None:
        item = Dataset()
        fan_tg = tag_for_keyword('FrameAcquisitionNumber')
        an_tg = tag_for_keyword('AcquisitionNumber')
        if an_tg in src_fg:
            fan_val = src_fg[an_tg].value
        else:
            fan_val = 0
        item[fan_tg] = DataElement(fan_tg, dictionary_VR(fan_tg), fan_val)
        self._mark_tag_as_used(an_tg)
        # ----------------------------------------------------------------
        AcquisitionDateTime_a = self._get_or_create_attribute(
            src_fg, 'AcquisitionDateTime',  self.EarliestDateTime)
        # chnage the keyword to FrameAcquisitionDateTime:
        FrameAcquisitionDateTime_a = DataElement(
            tag_for_keyword('FrameAcquisitionDateTime'),
            'DT', AcquisitionDateTime_a.value)
        AcquisitionDateTime_is_perframe = self._contains_right_attributes(
            self._PerFrameTags)
        if FrameAcquisitionDateTime_a.value == self.EarliestDateTime:
            AcquisitionDate_a = self._get_or_create_attribute(
                src_fg, 'AcquisitionDate', self.EarliestDate)
            AcquisitionTime_a = self._get_or_create_attribute(
                src_fg, 'AcquisitionTime', self.EarliestTime)
            d = AcquisitionDate_a.value
            t = AcquisitionTime_a.value
            # FrameAcquisitionDateTime_a.value = (DT(d.strftime('%Y%m%d') +
            #                                     t.strftime('%H%M%S')))
            FrameAcquisitionDateTime_a.value = DT(str(d) + str(t))
        if FrameAcquisitionDateTime_a.value > self.EarliestDateTime:
            if (FrameAcquisitionDateTime_a.value <
                    self.EarliestFrameAcquisitionDateTime):
                self.EarliestFrameAcquisitionDateTime =\
                    FrameAcquisitionDateTime_a.value
            if not AcquisitionDateTime_is_perframe:
                if ('TriggerTime' in src_fg and
                        'FrameReferenceDateTime' not in src_fg):
                    TriggerTime_a = self._get_or_create_attribute(
                        src_fg, 'TriggerTime', self.EarliestTime)
                    trigger_time_in_millisecond = int(TriggerTime_a.value)
                    if trigger_time_in_millisecond > 0:
                        t_delta = timedelta(trigger_time_in_millisecond)
                        # this is so rediculous. I'm not able to cnvert
                        #      the DT to datetime (cast to superclass)
                        d_t = datetime.combine(
                            FrameAcquisitionDateTime_a.value.date(),
                            FrameAcquisitionDateTime_a.value.time())
                        d_t = d_t + t_delta
                        FrameAcquisitionDateTime_a.value =\
                            DT(d_t.strftime('%Y%m%d%H%M%S'))
            item['FrameAcquisitionDateTime'] = FrameAcquisitionDateTime_a
        # ---------------------------------
        self._copy_attrib_if_present(
            src_fg, item, "AcquisitionDuration",
            "FrameAcquisitionDuration",
            check_not_to_be_perframe=False,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            src_fg, item,
            'TemporalPositionIndex',
            check_not_to_be_perframe=False,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            src_fg, item, "ImageComments",
            "FrameComments",
            check_not_to_be_perframe=False,
            check_not_to_be_empty=True)
        # -----------------------------------
        seq_tg = tag_for_keyword('FrameContentSequence')
        dest_fg[seq_tg] = DataElement(seq_tg, dictionary_VR(seq_tg), [item])
    # Also we want to add the earliest frame acq date time to the multiframe:

    def _add_acquisition_info(self) -> None:
        for i in range(0, len(self.SingleFrameSet)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(
                self.SingleFrameSet[i], item)
        if self.EarliestFrameAcquisitionDateTime < self.FarthestFutureDateTime:
            kw = 'AcquisitionDateTime'
            self.TargetDataset[kw] = DataElement(
                tag_for_keyword(kw),
                'DT', self.EarliestFrameAcquisitionDateTime)

    def AddModule(self) -> None:
        self._add_acquisition_info()
        self._add_stack_info()


class PixelData(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)
        self._byte_data = bytearray()
        self._word_data = bytearray()

    def _is_other_byte_vr(self, vr: str) -> bool:
        return vr[0] == 'O' and vr[1] == 'B'

    def _is_other_word_vr(self, vr: str) -> bool:
        return vr[0] == 'O' and vr[1] == 'W'
    # def _contains_right_attributes(self, tags: dict) -> bool:
    #     ImagePositionPatient_tg = tag_for_keyword('ImagePositionPatient')
    #     return ImagePositionPatient_tg in tags

    def _copy_data(self, src: bytearray, word_data: bool = False) -> None:
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

    def AddModule(self) -> None:
        kw = 'NumberOfFrames'
        tg = tag_for_keyword(kw)
        self._frame_count = len(self.SingleFrameSet)
        self.TargetDataset[kw] =\
            DataElement(tg, dictionary_VR(tg), self._frame_count)
        row = self.SingleFrameSet[0]["Rows"].value
        col = self.SingleFrameSet[0]["Columns"].value
        self._number_of_pixels_per_frame = row * col
        self._number_of_pixels = row * col * self._frame_count
        kw = "PixelData"
        for i in range(0, len(self.SingleFrameSet)):
            PixelData_a = self.SingleFrameSet[i][kw]
            if self._is_other_byte_vr(PixelData_a.VR):
                if len(self._word_data) != 0:
                    raise TypeError(
                        'Cannot mix OB and OW Pixel Data '
                        'VR from different frames')
                self._copy_data(PixelData_a.value, False)
            elif self._is_other_word_vr(PixelData_a.VR):
                if len(self._byte_data) != 0:
                    raise TypeError(
                        'Cannot mix OB and OW Pixel Data '
                        'VR from different frames')
                self._copy_data(PixelData_a.value, True)
            else:
                raise TypeError(
                    'Cannot mix OB and OW Pixel Data VR from different frames')
        if len(self._byte_data) != 0:
            MF_PixelData = DataElement(tag_for_keyword(kw),
                                       'OB', bytes(self._byte_data))
        elif len(self._word_data) != 0:
            MF_PixelData = DataElement(tag_for_keyword(kw),
                                       'OW', bytes(self._word_data))
        self.TargetDataset[kw] = MF_PixelData


class ContentDateTime(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)
        self.EarliestContentDateTime = self.FarthestFutureDateTime

    def AddModule(self) -> None:
        default_atrs = ["Acquisition", "Series", "Study"]
        for i in range(0, len(self.SingleFrameSet)):
            src = self.SingleFrameSet[i]
            default_date = self.FarthestFutureDate
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
            default_time = self.FarthestFutureTime
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
            if self.EarliestContentDateTime > value:
                self.EarliestContentDateTime = value
        if self.EarliestContentDateTime < self.FarthestFutureDateTime:
            n_d = DA(self.EarliestContentDateTime.date().strftime('%Y%m%d'))
            n_t = TM(self.EarliestContentDateTime.time().strftime('%H%M%S.%f'))
            kw = 'ContentDate'
            self.TargetDataset[kw] = DataElement(
                tag_for_keyword(kw), 'DA', n_d)
            kw = 'ContentTime'
            self.TargetDataset[kw] = DataElement(
                tag_for_keyword(kw), 'TM', n_t)


class InstanceCreationDateTime(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def AddModule(self) -> None:
        nnooww = datetime.now()
        n_d = DA(nnooww.date().strftime('%Y%m%d'))
        n_t = TM(nnooww.time().strftime('%H%M%S'))
        kw = 'InstanceCreationDate'
        self.TargetDataset[kw] = DataElement(
            tag_for_keyword(kw), 'DA', n_d)
        kw = 'InstanceCreationTime'
        self.TargetDataset[kw] = DataElement(
            tag_for_keyword(kw), 'TM', n_t)


class ContributingEquipmentSequence(Abstract_MultiframeModuleAdder):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):
        super().__init__(
            sf_datasets,
            excluded_from_perframe_tags,
            excluded_from_functional_tags,
            perframe_tags,
            shared_tags,
            multi_frame_output)

    def _add_data_element_to_target(self, target: Dataset,
                                    kw: str, value: Any) -> None:
        tg = tag_for_keyword(kw)
        target[kw] = DataElement(tg, dictionary_VR(tg), value)

    def AddModule(self) -> None:
        CodeValue_tg = tag_for_keyword('CodeValue')
        CodeMeaning_tg = tag_for_keyword('CodeMeaning')
        CodingSchemeDesignator_tg = tag_for_keyword('CodingSchemeDesignator')
        PurposeOfReferenceCode_item = Dataset()
        PurposeOfReferenceCode_item['CodeValue'] = DataElement(
            CodeValue_tg,
            dictionary_VR(CodeValue_tg),
            '109106')
        PurposeOfReferenceCode_item['CodeMeaning'] = DataElement(
            CodeMeaning_tg,
            dictionary_VR(CodeMeaning_tg),
            'Enhanced Multi-frame Conversion Equipment')
        PurposeOfReferenceCode_item['CodingSchemeDesignator'] = DataElement(
            CodingSchemeDesignator_tg,
            dictionary_VR(CodingSchemeDesignator_tg),
            'DCM')
        PurposeOfReferenceCode_seq = DataElement(
            tag_for_keyword('PurposeOfReferenceCodeSequence'),
            'SQ', [PurposeOfReferenceCode_item])
        item: Dataset = Dataset()
        item[
            'PurposeOfReferenceCodeSequence'] = PurposeOfReferenceCode_seq
        self._add_data_element_to_target(item, "Manufacturer", 'HighDicom')
        self._add_data_element_to_target(item, "InstitutionName", 'HighDicom')
        self._add_data_element_to_target(
            item,
            "InstitutionalDepartmentName",
            'Software Development')
        self._add_data_element_to_target(
            item,
            "InstitutionAddress",
            'Radialogy Department, B&W Hospital, Boston, MA')
        self._add_data_element_to_target(
            item,
            "SoftwareVersions",
            '1.4')  # get sw version
        self._add_data_element_to_target(
            item,
            "ContributionDescription",
            'Legacy Enhanced Image created from Classic Images')
        tg = tag_for_keyword('ContributingEquipmentSequence')
        self.TargetDataset[tg] = DataElement(tg, 'SQ', [item])


class LegacyConvertedEnhanceImage(SOPClass):
    """SOP class for Legacy Converted Enhanced PET Image instances."""

    def __init__(
            self,
            frame_set: FrameSet,
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            sort_key: Callable = None,
            **kwargs: Any) -> None:
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
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`
        """
        legacy_datasets = frame_set.Frames
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')
        sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
        if sort_key is None:
            sort_key = LegacyConvertedEnhanceImage.default_sort_key
        super().__init__(
            study_instance_uid=None if 'StudyInstanceUID' not in ref_ds
            else ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=None if 'Manufacturer' not in ref_ds
            else ref_ds.Manufacturer,
            modality=None if 'Modality' not in ref_ds
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
            **kwargs)
        self._legacy_datasets = legacy_datasets
        self.DistinguishingAttributesTags = self._get_tag_used_dictionary(
            frame_set.DistinguishingAttributesTags)
        self.ExcludedFromPerFrameTags = self._get_tag_used_dictionary(
            frame_set.ExcludedFromPerFrameTags)
        self._PerFrameTags = self._get_tag_used_dictionary(
            frame_set.PerFrameTags)
        self._SharedTags = self._get_tag_used_dictionary(
            frame_set.SharedTags)
        self.ExcludedFromFunctionalGroupsTags = {
            tag_for_keyword('SpecificCharacterSet'): False}
        # --------------------------------------------------------------------
        self.__build_blocks: list = []
        # == == == == == == == == == == == == == == == == == == == == == == ==
        new_ds = []
        for item in sorted(self._legacy_datasets, key=sort_key):
            new_ds.append(item)
        self._legacy_datasets = new_ds
        if (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-ct-image'):
            self.AddBuildBlocksForCT()
        elif (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-mr-image'):
            self.AddBuildBlocksForMR()
        elif (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-pet-image'):
            self.AddBuildBlocksForPET()

    def _get_tag_used_dictionary(self, input: list) -> dict:
        out: dict = {}
        for item in input:
            out[item] = False
        return out

    def default_sort_key(x: Dataset) -> tuple:
        out: tuple = tuple()
        if 'SeriesNumber' in x:
            out += (x['SeriesNumber'].value, )
        if 'InstanceNumber' in x:
            out += (x['InstanceNumber'].value, )
        if 'SOPInstanceUID' in x:
            out += (x['SOPInstanceUID'].value, )
        return out

    def AddNewBuildBlock(
        self, element: Abstract_MultiframeModuleAdder) -> None:
        if not isinstance(element, Abstract_MultiframeModuleAdder):
            raise ValueError('Build block must be an instance '
                             'of Abstract_MultiframeModuleAdder')
        self.__build_blocks.append(element)

    def ClearBuildBlocks(self) -> None:
        self.__build_blocks = []

    def AddCommonCT_PET_MR_BuildBlocks(self) -> None:
        Blocks = [
            ImagePixelModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            CompositeInstanceContex(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            EnhancedCommonImageModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            AcquisitionContextModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            FrameAnatomyFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            PixelMeasuresFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            PlaneOrientationFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            PlanePositionFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            FrameVOILUTFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            PixelValueTransformationFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            ReferencedImageFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            ConversionSourceFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            FrameContentFunctionalGroup(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            PixelData(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            ContentDateTime(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            InstanceCreationDateTime(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            ContributingEquipmentSequence(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            UnassignedPerFrame(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            UnassignedShared(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self)  # ,
            # StackInformation(
            #     self._legacy_datasets,
            #     self.ExcludedFromPerFrameTags,
            #     self.ExcludedFromFunctionalGroupsTags,
            #     self._PerFrameTags,
            #     self._SharedTags,
            #      self),
            # EmptyType2Attributes(
            #     self._legacy_datasets,
            #     self.ExcludedFromPerFrameTags,
            #     self.ExcludedFromFunctionalGroupsTags,
            #     self._PerFrameTags,
            #     self._SharedTags,
            #     self)
        ]
        for b in Blocks:
            self.AddNewBuildBlock(b)

    def AddCTSpecificBuildBlocks(self) -> None:
        Blocks = [
            CommonCTMRPETImageDescriptionMacro(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self,
                'CT'),
            EnhancedCTImageModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            ContrastBolusModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self)
        ]
        for b in Blocks:
            self.AddNewBuildBlock(b)

    def AddMRSpecificBuildBlocks(self) -> None:
        Blocks = [
            CommonCTMRPETImageDescriptionMacro(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self,
                'MR'),
            EnhancedMRImageModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self),
            ContrastBolusModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self)
        ]
        for b in Blocks:
            self.AddNewBuildBlock(b)

    def AddPETSpecificBuildBlocks(self) -> None:
        Blocks = [
            CommonCTMRPETImageDescriptionMacro(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self,
                'PET'),
            EnhancedPETImageModule(
                self._legacy_datasets,
                self.ExcludedFromPerFrameTags,
                self.ExcludedFromFunctionalGroupsTags,
                self._PerFrameTags,
                self._SharedTags,
                self)
        ]
        for b in Blocks:
            self.AddNewBuildBlock(b)

    def AddBuildBlocksForCT(self) -> None:
        self.ClearBuildBlocks()
        self.AddCommonCT_PET_MR_BuildBlocks()
        self.AddCTSpecificBuildBlocks()

    def AddBuildBlocksForMR(self) -> None:
        self.ClearBuildBlocks()
        self.AddCommonCT_PET_MR_BuildBlocks()
        self.AddMRSpecificBuildBlocks()

    def AddBuildBlocksForPET(self) -> None:
        self.ClearBuildBlocks()
        self.AddCommonCT_PET_MR_BuildBlocks()
        self.AddPETSpecificBuildBlocks()

    def BuildMultiFrame(self) -> None:
        logger = logging.getLogger(__name__)
        logger.debug('Strt singleframe to multiframe conversion')
        for builder in self.__build_blocks:
            builder.AddModule()
        logger.debug('Conversion succeeded')


class GeometryOfSlice:
    def __init__(self,
                 row_vector: ndarray,
                 col_vector: ndarray,
                 top_left_corner_pos: ndarray,
                 voxel_spaceing: ndarray,
                 dimensions: tuple):
        self.RowVector = row_vector
        self.ColVector = col_vector
        self.TopLeftCornerPosition = top_left_corner_pos
        self.VoxelSpacing = voxel_spaceing
        self.Dim = dimensions

    def GetNormalVector(self) -> ndarray:
        n: ndarray = cross(self.RowVector, self.ColVector)
        n[2] = -n[2]
        return n

    def GetDistanceAlongOrigin(self) -> float:
        n = self.GetNormalVector()
        return float(
            dot(self.TopLeftCornerPosition, n))

    def AreParallel(slice1: GeometryOfSlice,
                    slice2: GeometryOfSlice,
                    tolerance: float = 0.0001) -> bool:
        logger = logging.getLogger(__name__)
        if (type(slice1) != GeometryOfSlice or
                type(slice2) != GeometryOfSlice):
            logger.warning(
                'slice1 and slice2 are not of the same '
                'type: type(slice1) = {} and type(slice2) = {}'.format(
                    type(slice1), type(slice2)
                ))
            return False
        else:
            n1: ndarray = slice1.GetNormalVector()
            n2: ndarray = slice2.GetNormalVector()
            for el1, el2 in zip(n1, n2):
                if abs(el1 - el2) > tolerance:
                    return False
            return True


class DicomHelper:
    def __init__(self) -> None:
        pass

    def istag_file_meta_information_group(t: BaseTag) -> bool:
        return t.group == 0x0002

    def istag_repeating_group(t: BaseTag) -> bool:
        g = t.group
        return (g >= 0x5000 and g <= 0x501e) or\
            (g >= 0x6000 and g <= 0x601e)

    def istag_group_length(t: BaseTag) -> bool:
        return t.element == 0

    def isequal(v1: Any, v2: Any) -> bool:
        from pydicom.valuerep import DSfloat
        float_tolerance = 1.0e-5

        def is_equal_float(x1: float, x2: float) -> bool:
            return abs(x1 - x2) < float_tolerance
        if type(v1) != type(v2):
            return False
        if type(v1) == DicomSequence:
            for item1, item2 in zip(v1, v2):
                DicomHelper.isequal_dicom_dataset(item1, item2)
        if type(v1) != MultiValue:
            v11 = [v1]
            v22 = [v2]
        else:
            v11 = v1
            v22 = v2
        if len(v11) != len(v22):
            return False
        for xx, yy in zip(v11, v22):
            if type(xx) == DSfloat or type(xx) == float:
                if not is_equal_float(xx, yy):
                    return False
            else:
                if xx != yy:
                    return False
        return True

    def isequal_dicom_dataset(ds1: Dataset, ds2: Dataset) -> bool:
        if type(ds1) != type(ds2):
            return False
        if type(ds1) != Dataset:
            return False
        for k1, elem1 in ds1.items():
            if k1 not in ds2:
                return False
            elem2 = ds2[k1]
            if not DicomHelper.isequal(elem2.value, elem1.value):
                return False
        return True

    def tag2str(tg: BaseTag) -> str:
        if not isinstance(tg, BaseTag):
            tg = Tag(tg)
        return '(0x{:0>4x}, 0x{:0>4x})'.format(tg.group, tg.element)

    def tag2kwstr(tg: BaseTag) -> str:
        return '{}-{:32.32s}'.format(
            DicomHelper.tag2str(tg), keyword_for_tag(tg))


class FrameSet:
    def __init__(self, single_frame_list: list,
                 distinguishing_tags: list):
        self._Frames = single_frame_list
        self._DistinguishingAttributesTags = distinguishing_tags
        tmp = [
            tag_for_keyword('AcquisitionDateTime'),
            tag_for_keyword('AcquisitionDate'),
            tag_for_keyword('AcquisitionTime'),
            tag_for_keyword('SpecificCharacterSet')]
        self._ExcludedFromPerFrameTags =\
            self.DistinguishingAttributesTags + tmp
        self._PerFrameTags: list = []
        self._SharedTags: list = []
        self._find_per_frame_and_shared_tags()

    @property
    def Frames(self) -> List[Dataset]:
        return self._Frames[:]

    @property
    def DistinguishingAttributesTags(self) -> List[Tag]:
        return self._DistinguishingAttributesTags[:]

    @property
    def ExcludedFromPerFrameTags(self) -> List[Tag]:
        return self._ExcludedFromPerFrameTags[:]

    @property
    def PerFrameTags(self) -> List[Tag]:
        return self._PerFrameTags[:]

    @property
    def SharedTags(self) -> List[Tag]:
        return self._SharedTags[:]
    
    @property
    def SeriesInstanceUID(self) -> UID:
        return self._Frames[0].SeriesInstanceUID
    
    @property
    def StudyInstanceUID(self) -> UID:
        return self._Frames[0].StudyInstanceUID


    def GetSOPInstanceUIDList(self) -> list:
        OutputList: list = []
        for f in self._Frames:
            OutputList.append(f.SOPInstanceUID)
        return OutputList

    def GetSOPClassUID(self) -> UID:
        return self._Frames[0].SOPClassUID

    def _find_per_frame_and_shared_tags(self) -> None:
        logger = logging.getLogger(__name__)
        rough_shared: dict = {}
        sfs = self.Frames
        for ds in sfs:
            for ttag, elem in ds.items():
                if (not ttag.is_private and not
                    DicomHelper.istag_file_meta_information_group(ttag) and not
                        DicomHelper.istag_repeating_group(ttag) and not
                        DicomHelper.istag_group_length(ttag) and not
                        self._istag_excluded_from_perframe(ttag) and
                        ttag != tag_for_keyword('PixelData')):
                    elem = ds[ttag]
                    if ttag not in self._PerFrameTags:
                        self._PerFrameTags.append(ttag)
                    if ttag in rough_shared:
                        rough_shared[ttag].append(elem.value)
                    else:
                        rough_shared[ttag] = [elem.value]
        to_be_removed_from_shared = []
        for ttag, v in rough_shared.items():
            v = rough_shared[ttag]
            if len(v) < len(self.Frames):
                to_be_removed_from_shared.append(ttag)
            else:
                all_values_are_equal = True
                for v_i in v:
                    if not DicomHelper.isequal(v_i, v[0]):
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
            self._SharedTags.append(t)
            self._PerFrameTags.remove(t)

    def _istag_excluded_from_perframe(self, t: BaseTag) -> bool:
        return t in self.ExcludedFromPerFrameTags


class FrameSetCollection:
    def __init__(self, single_frame_list: list):
        logger = logging.getLogger(__name__)
        self.MixedFrames = single_frame_list
        self.MixedFramesCopy = self.MixedFrames[:]
        self._DistinguishingAttributeKeywords = [
            'PatientID',
            'PatientName',
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
        self._FrameSets: list = []
        frame_counts = []
        frameset_counter = 0
        while len(self.MixedFramesCopy) != 0:
            frameset_counter += 1
            x = self._find_all_similar_to_first_datasets()
            self._FrameSets.append(FrameSet(x[0], x[1]))
            frame_counts.append(len(x[0]))
            # log information
            logger.debug("Frameset({:02d}) including {:03d} frames".format(
                frameset_counter, len(x[0])))
            logger.debug('\t Distinguishing tags:')
            for dg_i, dg_tg in enumerate(x[1], 1):
                logger.debug('\t\t{:02d}/{})\t{}-{:32.32s} = {:32.32s}'.format(
                    dg_i, len(x[1]), DicomHelper.tag2str(dg_tg),
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
            len(frame_counts), len(self.MixedFrames)) + frames
        logger.info(frames)        
        for kw in to_be_removed_from_distinguishing_attribs:
            self.DistinguishingAttributeKeywords.remove(kw)
        self.ExcludedFromPerFrameTags = {}
        for i in self.DistinguishingAttributeKeywords:
            self.ExcludedFromPerFrameTags[tag_for_keyword(i)] = False
        self.ExcludedFromPerFrameTags[
            tag_for_keyword('AcquisitionDateTime')] = False
        self.ExcludedFromPerFrameTags[
            tag_for_keyword('AcquisitionDate')] = False
        self.ExcludedFromPerFrameTags[
            tag_for_keyword('AcquisitionTime')] = False
        self.ExcludedFromFunctionalGroupsTags = {
            tag_for_keyword('SpecificCharacterSet'): False}

    def _find_all_similar_to_first_datasets(self) -> tuple:
        logger = logging.getLogger(__name__)
        similar_ds: list = [self.MixedFramesCopy[0]]
        distinguishing_tags_existing = []
        distinguishing_tags_missing = []
        self.MixedFramesCopy = self.MixedFramesCopy[1:]
        for kw in self.DistinguishingAttributeKeywords:
            tg = tag_for_keyword(kw)
            if tg in similar_ds[0]:
                distinguishing_tags_existing.append(tg)
            else:
                distinguishing_tags_missing.append(tg)
        logger_msg = set()
        for ds in self.MixedFramesCopy:
            all_equal = True
            for tg in distinguishing_tags_missing:
                if tg in ds:
                    logger.info()
                    logger_msg.add(
                        '{} is missing in all but {}'.format(
                            DicomHelper.tag2kwstr(tg), ds['SOPInstanceUID']))
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
                if not DicomHelper.isequal(ref_val, new_val):
                    logger_msg.add(
                        'Inequality on distinguishing '
                        'attribute{} -> {} != {} \n series uid = {}'.format(
                            DicomHelper.tag2kwstr(tg), ref_val, new_val,
                            ds.SeriesInstanceUID))
                    all_equal = False
                    break
            if all_equal:
                similar_ds.append(ds)
        for msg_ in logger_msg:
            logger.info(msg_)
        for ds in similar_ds:
            if ds in self.MixedFramesCopy:
                self.MixedFramesCopy.remove(ds)
        return (similar_ds, distinguishing_tags_existing)

    @property
    def DistinguishingAttributeKeywords(self) -> List[str]:
        return self._DistinguishingAttributeKeywords[:]

    @property
    def FrameSets(self) -> List[FrameSet]:
        return self._FrameSets
