"""Module for SOP Classes of Legacy Converted Enhanced Image IODs."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydicom.datadict import tag_for_keyword, dictionary_VR
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence as DicomSequence
from pydicom.multival import MultiValue
from datetime import date, datetime, time
from pydicom.valuerep import DT, DA, TM
from copy import deepcopy

from pydicom.uid import UID

from highdicom.base import SOPClass
from highdicom.legacy import SOP_CLASS_UIDS
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
    '1.2.840.10008.5.1.4.1.1.2.2':   'legacy-converted-enhanced-ct-image',
    '1.2.840.10008.5.1.4.1.1.4.4':   'legacy-converted-enhanced-mr-image',
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
            'All instances must belong to the same series.'
        )
    if len(studies) > 1:
        raise ValueError(
            'All instances must belong to the same study.'
        )
    if len(modalities) > 1:
        raise ValueError(
            'All instances must have the same modality.'
        )
    if len(transfer_syntaxes) > 1:
        raise ValueError(
            'All instances must have the same transfer syntaxes.'
        )

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
                ds.AcquisitionTime
            )
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

        # Cardiac Synchronization (U)
        # TODO: http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html#sect_C.7.6.16.2.7  # noqa

        # Contrast/Bolus Usage (U) - MR/CT only
        # TODO: http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html#sect_C.7.6.16.2.12  # noqa

        # Respiratory Synchronization (U)
        # TODO: http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html#sect_C.7.6.16.2.17  # noqa

        # Real World Value Mapping (U) - PET only
        # TODO: http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html#sect_C.7.6.16.2.11  # noqa

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
    # Sometimes there may be numpy types such as ">i2". The (* 1) hack
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
        **kwargs: Any
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
                'Wrong modality for conversion of legacy MR images.'
            )
        if ref_ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.4':
            raise ValueError(
                'Wrong SOP class for conversion of legacy MR images.'
            )

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
            **kwargs
        )
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
        **kwargs: Any
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
            **kwargs
        )
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
            **kwargs: Any
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
            **kwargs
        )
        _convert_legacy_to_enhanced(legacy_datasets, self)
from abc import ABC, abstractmethod
class Abstract_MultiframeModuleAdder(ABC):
 
    def __init__(self, sf_datasets:Sequence[Dataset]
                    , excluded_from_perframe_tags:Sequence[Tag]
                    , excluded_from_functional_tags:Sequence[Tag]
                    , perframe_tags: Sequence[Tag]
                    , shared_tags: Sequence[Tag]
                    , multi_frame_output:Dataset=None ):
        self.ExcludedFromPerFrameTags =  excluded_from_perframe_tags
        self.ExcludedFromFunctionalGroupsTags = excluded_from_functional_tags
        self.PerFrameTags = perframe_tags 
        self.SharedTags = shared_tags
        self.TargetDataset = multi_frame_output
        self.SingleFrameSet = sf_datasets
        self.EarliestDate = DA('00010101')
        self.EarliestTime = TM('000000')
        self.EarliestDateTime = DT('00010101000000')
    def _is_empty_or_empty_items(self, attribute:DataElement)->bool:
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
        return True
                   
    def _mark_tag_as_used(self, tg:Tag):
        if tg in self.SharedTags:
                self.SharedTags[tg] = True 
        elif tg in self.ExcludedFromPerFrameTags:
            self.ExcludedFromPerFrameTags[tg] = True
        elif tg in self.PerFrameTags:
            self.PerFrameTags[tg] = True

    def _copy_attrib_if_present(self, src_ds:Dataset, dest_ds:Dataset, 
                                src_kw_or_tg:str, dest_kw_or_tg:str=None, 
                                check_not_to_be_perframe=True, check_not_to_be_empty=False):
        if type(src_kw_or_tg) == str:
            src_kw_or_tg = tag_for_keyword(src_kw_or_tg)
        if dest_kw_or_tg == None:
            dest_kw_or_tg = src_kw_or_tg
        elif type(dest_kw_or_tg) == str:
            dest_kw_or_tg = tag_for_keyword(dest_kw_or_tg)
        if check_not_to_be_perframe:
            if  src_kw_or_tg in self.PerFrameTags:
                return 
        if src_kw_or_tg in src_ds:
            elem = src_ds[src_kw_or_tg]
            if check_not_to_be_empty:
                if _is_empty_or_empty_items(elem):
                    return 
            new_elem = deepcopy(elem)
            if dest_kw_or_tg == src_kw_or_tg:
                dest_ds[dest_kw_or_tg] = new_elem
            else:
                new_elem1 = DataElement(dest_kw_or_tg,
                    dictionary_VR(dest_kw_or_tg), newelem.value)
                dest_ds[dest_kw_or_tg] = new_elem1
            # now mark the attrib as used/done to keep track of every one of it
            self._mark_tag_as_used(src_kw_or_tg)
            
            

            
    def _get_perframe_item(self, index:int)->Dataset:
        if index > len(self.SingleFrameSet):
            return None
        pf_kw = 'PerFrameFunctionalGroupsSequence'
        pf_tg = tag_for_keyword(pf_kw)
        if not pf_kw in self.TargetDataset:
            seq = []
            for i in range(0, len(self.SingleFrameSet)):
                seq.append(Dataset())
            self.TargetDataset[pf_tg] = DataElement(pf_tg, 'SQ', DicomSequence(seq))
        return self.TargetDataset[pf_tg].value[index]
    def _get_shared_item(self)->Dataset:
        sf_kw = 'SharedFunctionalGroupsSequence'
        sf_tg = tag_for_keyword(sf_kw)
        if not sf_kw in self.TargetDataset:
            seq = [Dataset()]
            self.TargetDataset[sf_tg] = DataElement(sf_tg, 'SQ', DicomSequence(seq))
        return self.TargetDataset[sf_tg].value[0]
    def _get_or_create_attribute(self, src:Dataset, kw:str, default)->DataElement:
        tg = tag_for_keyword(kw)
        if kw in src:
            a = deepcopy(src[kw])
        else:
            a = DataElement(tg, 
                dictionary_VR(tg), default )
        from pydicom.valuerep import DT, TM, DA
        if a.VR == 'DA' and type(a.value)==str:
            a.value = DA(a.value)
        if a.VR == 'DT' and type(a.value)==str:
            a.value = DT(a.value)
        if a.VR == 'TM' and type(a.value)==str:
            a.value = TM(a.value)

        self._mark_tag_as_used(tg)
        return a
    def _add_module(self, module_name: str, check_not_to_be_perframe=True):
        # sf_sop_instance_uid = sf_datasets[0]
        # mf_sop_instance_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[sf_sop_instance_uid]
        # iod_name = _SOP_CLASS_UID_IOD_KEY_MAP[mf_sop_instance_uid]
        # modules = IOD_MODULE_MAP[iod_name]
        from copy import deepcopy
        attribs = MODULE_ATTRIBUTE_MAP[module_name]
        ref_dataset = self.SingleFrameSet[0]
        for a in attribs:
            if len(a['path']) == 0:
                self._copy_attrib_if_present(ref_dataset, self.TargetDataset, a['keyword'],
                                        check_not_to_be_perframe=check_not_to_be_perframe)

    @abstractmethod
    def AddModule(self):
        pass
class PixelImageModule(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def AddModule(self):
        self._add_module('image-pixel', False) # don't check the perframe set
class CommonCTMRPETImageDescriptionMacro(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
        , excluded_from_perframe_tags:Sequence[Tag]
        , excluded_from_functional_tags:Sequence[Tag]
        , perframe_tags: Sequence[Tag]
        , shared_tags: Sequence[Tag]
        , multi_frame_output:Dataset=None
        , modality:str='CT'):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
        self.Modality = modality
    def _get_value_for_frame_type(self, attrib:DataElement):
        if type(attrib) == DataElement:
            return None
        output = ['', '', '', '']
        v = attrib.value
        l = len(v)
        output[0] = 'ORIGINAL' if l == 0 else v[0]
        output[1] = 'PRIMARY'
        output[2] = 'VOLUME' if l<3 else v[2]
        output[3] = 'NONE'
        return output

    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        seq_kw = '{}{}FrameTypeSequence'
        if self.Modality == 'PET':
            seq_kw = seq_kw.format(self.Modality, '')
        else:
            seq_kw = seq_kw.format(self.Modality, 'Image')
        seq_tg = tag_for_keyword(seq_kw)
        
        FrameType_a = src_fg['ImageType']
        new_val = self._get_value_for_frame_type(FrameType_a)
        inner_item = Dataset()
        FrameType_tg = tag_for_keyword('FrameType')
        inner_item[FrameType_tg] = DataElement(FrameType_tg,
            FrameType_a.VR, new_val)
        dest_fg[seq_tg] = DataElement(seq_tg, dictionary_VR(seq_tg),
            [inner_item]) 

    def AddModule(self):
        im_type_tag = tag_for_keyword('ImageType')
        fm_type_tag = tag_for_keyword('FrameType')

        if not im_type_tag in  self.PerFrameTags:
            im_type_a = self.SingleFrameSet[0][im_type_tag]
            new_val = self._get_value_for_frame_type(im_type_a)
            self.TargetDataset[im_type_tag] = DataElement(im_type_tag,
                im_type_a.VR, new_val)
            #----------------------------
            
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item) 
        else:
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item) 
                
class EnhancedCommonImageModule(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def AddModule(self):
        # ct_mr = CommonCTMRImageDescriptionMacro(self.SingleFrameSet
        # , self.ExcludedFromPerFrameTags
        # , self.PerFrameTags
        # , self.SharedTags
        # , self.TargetDataset)
        # ct_mr.AddModule()

		
		# Acquisition Number
		# Acquisition DateTime						- should be able to find earliest amongst all frames, if present (required if ORIGINAL)
		# Acquisition Duration						- should be able to work this out, but type 2C, so can send empty
		
		# Referenced Raw Data Sequence				- optional - ignore - too hard to merge
		# Referenced Waveform Sequence				- optional - ignore - too hard to merge
		# Referenced Image Evidence Sequence		- should add if we have references :(
		# Source Image Evidence Sequence			- should add if we have sources :(
		# Referenced Presentation State Sequence	- should merge if present in any source frame :(
		
		# Samples per Pixel						- handled by distinguishingAttribute copy
		# Photometric Interpretation				- handled by distinguishingAttribute copy
		# Bits Allocated							- handled by distinguishingAttribute copy
		# Bits Stored								- handled by distinguishingAttribute copy
		# High Bit									- handled by distinguishingAttribute copy
        ref_dataset = self.SingleFrameSet[0]
        attribs_to_be_added = ['ContentQualification',
            'ImageComments',
            'BurnedInAnnotation',
            'RecognizableVisualFeatures',
            'LossyImageCompression',
            'LossyImageCompressionRatio',
            'LossyImageCompressionMethod']
        for kw in attribs_to_be_added:
            self._copy_attrib_if_present(ref_dataset, self.TargetDataset, kw)
        

        if not tag_for_keyword('PresentationLUTShape') in self.PerFrameTags :
            # actually should really invert the pixel data if MONOCHROME1, since only MONOCHROME2 is permitted :(
            # also, do not need to check if PhotometricInterpretation is per-frame, since a distinguishing attribute
            phmi_kw = 'PhotometricInterpretation'
            phmi_tg = tag_for_keyword(phmi_kw)
            phmi_a = self._get_or_create_attribute(self.SingleFrameSet[0], phmi_kw, "MONOCHROME2")
            LUT_shape_default = "INVERTED" if phmi_a.value == 'MONOCHROME1' else "IDENTITY"
            LUT_shape_a = self._get_or_create_attribute(self.SingleFrameSet[0],
                                                        'PresentationLUTShape', 
                                                        LUT_shape_default)
            if not LUT_shape_a.is_empty:
                self.TargetDataset['PresentationLUTShape'] = LUT_shape_a
        # Icon Image Sequence							- always discard these
class ContrastBolusModule(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def AddModule(self):
        self._add_module('contrast-bolus')
class EnhancedCTImageModule(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def AddModule(self):
        pass
        #David's code doesn't hold anything for this module ... should ask him
class AcquisitionContextModule(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def AddModule(self):
        self._copy_attrib_if_present(self.SingleFrameSet
                                    , self.TargetDataset
                                    , 'AcquisitionContextSequence'
                                    , check_not_to_be_perframe=True)#check not to be in perframe
class FrameAnatomyFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        #David's code is more complicaated than mine
        #Should check it out later.
        fa_seq_tg = tag_for_keyword('FrameAnatomySequence')
        item = Dataset()
        self._copy_attrib_if_present(item, src_fg, 'AnatomicRegionSequence'
                                    , check_not_to_be_perframe=False)
        self._copy_attrib_if_present(item, src_fg, 'FrameLaterality'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=True)
        
        if not 'FrameLaterality' in item:
            self._copy_attrib_if_present(item, src_fg, 'ImageLaterality'
                                    , 'FrameLaterality'
                                    , check_not_to_be_perframe=False)
        if not 'FrameLaterality' in item:
            self._copy_attrib_if_present(item, src_fg, 'Laterality'
                                    , 'FrameLaterality'
                                    , check_not_to_be_perframe=False)

        FrameAnatomy_a = DataElement(fa_seq_tg, dictionary_VR(fa_seq_tg),
                                        [item] )
        dest_fg['FrameAnatomySequence'] = FrameAnatomy_a
    def _contains_right_attributes(self, tags:dict) ->bool:
        laterality_tg = tag_for_keyword('Laterality')
        im_laterality_tg = tag_for_keyword('ImageLaterality')
        bodypart_tg = tag_for_keyword('BodyPartExamined')
        anatomical_reg_tg = tag_for_keyword('AnatomicRegionSequence')

        return (laterality_tg in tags 
                or im_laterality_tg in tags 
                or bodypart_tg in tags 
                or anatomical_reg_tg)

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)

class PixelMeasuresFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        PixelSpacing_tg = tag_for_keyword('PixelSpacing' )
        SliceThickness_tg = tag_for_keyword('SliceThickness')
        ImagerPixelSpacing_tg = tag_for_keyword('ImagerPixelSpacing')

        return (PixelSpacing_tg in tags
                or SliceThickness_tg in tags
                or ImagerPixelSpacing_tg in tags)
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    ,item
                                    ,'PixelSpacing'
                                    , check_not_to_be_perframe=False)
        self._copy_attrib_if_present(src_fg
                                    ,item
                                    ,'SliceThickness'
                                    , check_not_to_be_perframe=False)
        if not 'PixelSpacing' in item:
            self._copy_attrib_if_present(src_fg
                                        ,item
                                        ,'ImagerPixelSpacing'
                                        ,'PixelSpacing'
                                        , check_not_to_be_perframe=False
                                        , check_not_to_be_empty=True)
        pixel_measures_kw = 'PixelMeasuresSequence'
        pixel_measures_tg = tag_for_keyword(pixel_measures_kw)
        seq = DataElement(pixel_measures_tg
                        , dictionary_VR(pixel_measures_tg)
                        , [item])
        dest_fg[pixel_measures_tg] = seq

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class PlanePositionFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        ImagePositionPatient_tg = tag_for_keyword('ImagePositionPatient' )

        return ImagePositionPatient_tg in tags
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'ImagePositionPatient'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        PlanePositionSequence_kw = 'PlanePositionSequence'
        PlanePositionSequence_tg = tag_for_keyword(PlanePositionSequence_kw)
        seq = DataElement(PlanePositionSequence_tg
                        , dictionary_VR(PlanePositionSequence_tg)
                        , [item])
        dest_fg[PlanePositionSequence_tg] = seq

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class PlanePositionFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        ImagePositionPatient_tg = tag_for_keyword('ImagePositionPatient' )

        return ImagePositionPatient_tg in tags
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'ImagePositionPatient'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        PlanePositionSequence_kw = 'PlanePositionSequence'
        PlanePositionSequence_tg = tag_for_keyword(PlanePositionSequence_kw)
        seq = DataElement(PlanePositionSequence_tg
                        , dictionary_VR(PlanePositionSequence_tg)
                        , [item])
        dest_fg[PlanePositionSequence_tg] = seq

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class PlaneOrientationFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        ImageOrientationPatient_tg = tag_for_keyword('ImageOrientationPatient' )

        return ImageOrientationPatient_tg in tags
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'ImageOrientationPatient'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        kw = 'PlaneOrientationSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class FrameVOILUTFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        WindowWidth_tg = tag_for_keyword('WindowWidth')
        WindowCenter_tg = tag_for_keyword('WindowCenter')
        WindowCenterWidthExplanation_tg = tag_for_keyword('WindowCenterWidthExplanation')

        return (WindowWidth_tg in tags
                or WindowCenter_tg in tags
                or WindowCenterWidthExplanation_tg in tags)
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'WindowWidth'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'WindowCenter'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'WindowCenterWidthExplanation'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        kw = 'FrameVOILUTSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class PixelValueTransformationFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        RescaleIntercept_tg = tag_for_keyword('RescaleIntercept')
        RescaleSlope_tg = tag_for_keyword('RescaleSlope')
        RescaleType_tg = tag_for_keyword('RescaleType')

        return (RescaleIntercept_tg in tags
                or RescaleSlope_tg in tags
                or RescaleType_tg in tags)
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'RescaleSlope'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'RescaleIntercept'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        haveValuesSoAddType = 'RescaleSlope' in item or 'RescaleIntercept' in item
        self._copy_attrib_if_present(src_fg
                                    , item
                                    , 'RescaleType'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=True)
        if not "RescaleType" in item:
            value = ''
            modality = '' if not 'Modality' in src_fg else src_fg["Modality"].value
            if haveValuesSoAddType:
                value = 'US'
                if modality == 'CT':
                    containes_localizer = False
                    ImageType_v = [] if not 'ImageType' in src_fg else src_fg['ImageType'].value
                    for i in ImageType_v:
                        if i=='LOCALIZER':
                            containes_localizer = True
                            break
                    if not containes_localizer:
                        value = "HU"
                elif modality == 'PT':
                    value = 'US' if not 'Units' in src_fg else src_fg['Units'].value
            if value != '':
                tg =  tag_for_keyword('RescaleType')
                item[tg]= DataElement(tg, dictionary_VR(tg), value)        

        kw = 'PixelValueTransformationSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class ReferencedImageFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        return tag_for_keyword('ReferencedImageSequence') in tags
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        self._copy_attrib_if_present(src_fg
                                    , dest_fg
                                    ,'ReferencedImageSequence'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class DerivationImageFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _contains_right_attributes(self, tags:dict) ->bool:
        return tag_for_keyword('SourceImageSequence') in tags
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'DerivationDescription'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=True)
        haveValuesSoAddType = 'RescaleSlope' in item or 'RescaleIntercept' in item
        self._copy_attrib_if_present(src_fg
                                    , item
                                    , 'DerivationCodeSequence'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'SourceImageSequence'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=False)
        kw = 'DerivationImageSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def AddModule(self):
        if (not self._contains_right_attributes(self.PerFrameTags)
            and (self._contains_right_attributes(self.SharedTags)
            or self._contains_right_attributes(self.ExcludedFromPerFrameTags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.SingleFrameSet[0],item)
        elif self._contains_right_attributes(self.PerFrameTags):
            for i in range(0, len(self.SingleFrameSet)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class UnassignedPerFrame(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        for tg, used in self.PerFrameTags.items():
            if not used not in self.ExcludedFromFunctionalGroupsTags:
                self._copy_attrib_if_present(src_fg
                                            , item
                                            ,tg
                                            , check_not_to_be_perframe=False
                                            , check_not_to_be_empty=False)
        
        kw = 'UnassignedPerFrameConvertedAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq
        

    def AddModule(self):
        for i in range(0, len(self.SingleFrameSet)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(self.SingleFrameSet[i],item)
class UnassignedShared(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        for tg, used in self.SharedTags.items():
            if (not used 
                and tg not in self.ExcludedFromFunctionalGroupsTags):
                self._copy_attrib_if_present(src_fg
                                            , item
                                            ,tg
                                            , check_not_to_be_perframe=False
                                            , check_not_to_be_empty=False)
        
        kw = 'UnassignedSharedConvertedAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq
        

    def AddModule(self):
        item = self._get_shared_item()
        self._add_module_to_functional_group(self.SingleFrameSet[0],item)
                    
class ConversionSourceFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'ReferencedSOPClassUID'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=True)
        self._copy_attrib_if_present(src_fg
                                    , item
                                    ,'ReferencedSOPInstanceUID'
                                    , check_not_to_be_perframe=False
                                    , check_not_to_be_empty=True)
        kw = 'ConversionSourceAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq
        
        
        

    def AddModule(self):
        for i in range(0, len(self.SingleFrameSet)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    
class FrameContentFunctionalGroup(Abstract_MultiframeModuleAdder):
    def __init__(self, sf_datasets:Sequence[Dataset]
    , excluded_from_perframe_tags:Sequence[Tag]
    , excluded_from_functional_tags:Sequence[Tag]
    , perframe_tags: Sequence[Tag]
    , shared_tags: Sequence[Tag]
    , multi_frame_output:Dataset=None):
        
        super().__init__(
            sf_datasets
            , excluded_from_perframe_tags
            , excluded_from_functional_tags
            , perframe_tags
            , shared_tags
            , multi_frame_output
        )
        self.EarliestFrameAcquisitionDateTime = DT('99991231235959')
    def _contains_right_attributes(self, tags:dict) ->bool:
        AcquisitionDateTime_tg = tag_for_keyword('AcquisitionDateTime')
        AcquisitionDate_tg = tag_for_keyword('AcquisitionDate')
        AcquisitionTime_tg = tag_for_keyword('AcquisitionTime')

        return (AcquisitionDateTime_tg in tags
                or AcquisitionTime_tg in tags
                or AcquisitionDate_tg in tags)
    def _add_module_to_functional_group(self, src_fg:Dataset, dest_fg:Dataset):
        item = Dataset()
        item['AcquisitionNumber'] = self._get_or_create_attribute(src_fg, 'AcquisitionNumber',0) 
        AcquisitionDateTime_a = self._get_or_create_attribute(src_fg,'AcquisitionDateTime',  self.EarliestDateTime)
        AcquisitionDateTime_is_perframe = self._contains_right_attributes(self.PerFrameTags)
        if AcquisitionDateTime_a.value == self.EarliestDateTime:
            AcquisitionDate_a = self._get_or_create_attribute(src_fg,'AcquisitionDate', self.EarliestDate)
        
            AcquisitionTime_a = self._get_or_create_attribute(src_fg,'AcquisitionTime', self.EarliestTime)
            d = AcquisitionDate_a.value
            t = AcquisitionTime_a.value
            AcquisitionDateTime_a.value = DT(d.original_string+t.original_string)
        if AcquisitionDateTime_a.value > self.EarliestDateTime:
            if AcquisitionDateTime_a.value < self.EarliestFrameAcquisitionDateTime:
                self.EarliestFrameAcquisitionDateTime = AcquisitionDateTime_a.value
            if not AcquisitionDateTime_is_perframe:
                if 'TriggerTime' in src_fg and not 'FrameReferenceDateTime' in src_fg:
                    TriggerTime_a = self._get_or_create_attribute(src_fg,'TriggerTime', self.EarliestTime)
                    AcquisitionDateTime_a.value = DT(
                        AcquisitionDate_a.value.original_string+TriggerTime_a.value.original_string)
            item['AcquisitionDateTime'] =AcquisitionDateTime_a
        #---------------------------------
        self._copy_attrib_if_present(item, src_fg, "AcquisitionDuration", 
            "FrameAcquisitionDuration", check_not_to_be_perframe=False, check_not_to_be_empty=True)
        self._copy_attrib_if_present(item, src_fg, 'TemporalPositionIndex'
            , check_not_to_be_perframe=False, check_not_to_be_empty=True)
        self._copy_attrib_if_present(item, src_fg, "ImageComments", 
            "FrameComments", check_not_to_be_perframe=False, check_not_to_be_empty=True)
        #-----------------------------------
        seq_tg = tag_for_keyword('FrameContentSequence')
        dest_fg[seq_tg] = DataElement(seq_tg, dictionary_VR(seq_tg), [item]) 
        

    def AddModule(self):
        for i in range(0, len(self.SingleFrameSet)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(self.SingleFrameSet[i],item)
                    


        
class LegacyConvertedEnhanceImage(SOPClass):

    """SOP class for Legacy Converted Enhanced PET Image instances."""

    def __init__(
            self,
            legacy_datasets: Sequence[Dataset],
            series_instance_uid: str,
            series_number: int,
            sop_instance_uid: str,
            instance_number: int,
            **kwargs: Any
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
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """
        
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')

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
            **kwargs
        )
        self._legacy_datasets = legacy_datasets
        self.DistinguishingAttributeKeywords = [
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
        to_be_removed_from_distinguishing_attribs = set()
        for kw in self.DistinguishingAttributeKeywords:
            x = []
            not_present_attribute_count = 0
            for ds in legacy_datasets:
                if kw in ds:
                    if len(x) == 0:
                        x.append(ds[kw])
                    else:
                        already_has_new_value = False
                        for x_elem in x:
                            if self._isequal(x_elem.value, ds[kw].value):
                                already_has_new_value = True
                                break
                        if not already_has_new_value:
                            x.append(ds[kw])
                else:
                    to_be_removed_from_distinguishing_attribs.add(kw)
                    not_present_attribute_count += 1
            if not_present_attribute_count != len(legacy_datasets) \
                and not_present_attribute_count!=0:
                    raise ValueError('One or more datasets lack {} distinguishing attributes'.format(kw))
            if len(x)>1:
                error_msg = 'All instances must have the same value for {}.\n\tExisting values:'
                for x_elem in x:
                    error_msg += '\n\t\t{}'.format(x_elem.value)
                raise ValueError(error_msg)
        for kw in to_be_removed_from_distinguishing_attribs:
            self.DistinguishingAttributeKeywords.remove(kw)
            
        self.ExcludedFromPerFrameTags = {}
        for i in self.DistinguishingAttributeKeywords:
            self.ExcludedFromPerFrameTags[tag_for_keyword(i)] = False
        
        self.ExcludedFromPerFrameTags[tag_for_keyword('AcquisitionDateTime')] = False
        self.ExcludedFromPerFrameTags[tag_for_keyword('AcquisitionDate')] = False
        self.ExcludedFromPerFrameTags[tag_for_keyword('AcquisitionTime')] = False
        
        self.ExcludedFromFunctionalGroupsTags={tag_for_keyword('SpecificCharacterSet'): False}
        #---------------------------------------------------------------------
        self.PerFrameTags = {}
        self.SharedTags = {}
        self._find_per_frame_and_shared_tags()
        #----------------------------------------------------------------------
        self.__build_blocks = []
    def _find_per_frame_and_shared_tags(self):
        rough_shared = {}
        sfs = self._legacy_datasets
        for ds in sfs:
            for ttag, elem in ds.items():
                if (not ttag.is_private
                and not self._istag_file_meta_information_group(ttag)
                and not self._istag_repeating_group(ttag)
                and not self._istag_group_length(ttag)
                and not self._istag_excluded_from_perframe(ttag)):
                    elem = ds[ttag]
                    self.PerFrameTags[ttag] = False
                    if  ttag in rough_shared:
                        rough_shared[ttag].append(elem.value)
                    else:
                        rough_shared[ttag] = [elem.value]
        to_be_removed_from_shared = []
        for ttag, v in rough_shared.items():
            v = rough_shared[ttag]
            N = len(v)
            if len(v) < len(self._legacy_datasets):
                to_be_removed_from_shared.append(ttag)
            else:
                all_values_are_equal = True
                for v_i in v:
                    if not self._isequal(v_i,v[0]):
                        all_values_are_equal = False
                        break
                if not all_values_are_equal:
                    to_be_removed_from_shared.append(ttag)
        from pydicom.datadict import keyword_for_tag
        for t, v in rough_shared.items():
            if keyword_for_tag(t)!='PatientSex':
                continue


        for t in to_be_removed_from_shared:
            del rough_shared[t]
        for t, v in rough_shared.items():
            self.SharedTags[t] = False
            del self.PerFrameTags[t]
        # for t in self.SharedTags:
        #     print(keyword_for_tag(t))
        # print('perframe ---------------------------')
        # for t in self.PerFrameTags:
        #     print (keyword_for_tag(t))


    def _istag_excluded_from_perframe(self, t:Tag)->bool:
        return t in self.ExcludedFromPerFrameTags
                    
               
    def _istag_file_meta_information_group(self, t:Tag)->bool:
        return t.group == 0x0002
    def _istag_repeating_group(self, t:Tag)->bool:
        g = t.group
        return (g >= 0x5000 and g <= 0x501e) or\
            (g >= 0x6000 and g <= 0x601e)
    def _istag_group_length(self, t:Tag)->bool:
        return t.element == 0
    def _isequal(self, v1, v2):
        float_tolerance = 1.0e-5
        is_equal_float = lambda x1, x2: abs(x1-x2)<float_tolerance
        if type(v1) != type(v2):
            return False
        if type(v1)==DicomSequence:
            for item1, item2 in zip(v1, v2):
                self._isequal_dicom_dataset(item1, item2)

        if type(v1) != MultiValue:
            v11 =[v1]
            v22 =[v2]
        else:
            v11 =v1
            v22 =v2
        for xx, yy in zip(v11, v22):
            if type(xx) == float:
                if not is_equal_float(xx, yy):
                    return False
            else:
                if xx != yy:
                    return False
        return True
    def _isequal_dicom_dataset(self, ds1, ds2 )->bool:
        if type(ds1) != type(ds2):
            return False
        if type(ds1) != Dataset:
            return False
        for k1, elem1 in ds1.items():
            if not k1 in ds2:
                return False
            elem2 = ds2[k1] 
            return self._isequal(elem2.value, elem1.value)   
    def AddNewBuildBlock(self, element:Abstract_MultiframeModuleAdder):
        if not isinstance(element, Abstract_MultiframeModuleAdder) :
            raise ValueError('Build block must be an instance of Abstract_MultiframeModuleAdder')
        self.__build_blocks.append(element)
    def AddBuildBlocksForCT(self):
        Blocks= [PixelImageModule(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,CommonCTMRPETImageDescriptionMacro(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self
                , 'CT')
            ,EnhancedCommonImageModule(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,ContrastBolusModule(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,EnhancedCTImageModule(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,AcquisitionContextModule(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,FrameAnatomyFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,PixelMeasuresFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,PlaneOrientationFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,PlanePositionFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,FrameVOILUTFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,PixelValueTransformationFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,ReferencedImageFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,ConversionSourceFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,FrameContentFunctionalGroup(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,UnassignedPerFrame(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ,UnassignedShared(self._legacy_datasets
                , self.ExcludedFromPerFrameTags
                , self.ExcludedFromFunctionalGroupsTags
                , self.PerFrameTags
                , self.SharedTags
                , self)
            ]
        for b in Blocks:
            self.AddNewBuildBlock(b)
    
    def BuildMultiFrame(self):
        for builder in self.__build_blocks:
            builder.AddModule()

            
                    








       
