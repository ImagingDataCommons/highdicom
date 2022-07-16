"""Module for SOP Classes of Legacy Converted Enhanced Image IODs."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
)

from highdicom.base import SOPClass
from highdicom.frame import encode_frame
from highdicom._iods import IOD_MODULE_MAP, SOP_CLASS_UID_IOD_KEY_MAP
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
    mf_dataset: Union[pydicom.dataset.Dataset, None], optional
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
    iod_key = SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid]
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
            frame_content_item.FrameAcquisitionDateTime = datetime.combine(
                ds.AcquisitionDate, ds.AcquisitionTime
            ).strftime('%Y%m%d%H%M%S')
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
        if pixel_representation == 0:
            frame_type_item.PixelPresentation = 'MONOCHROME'
        else:
            frame_type_item.PixelPresentation = 'COLOR'
        frame_type_item.VolumetricProperties = volumetric_properties
        if frame_type[0] == 'ORIGINAL':
            frame_type_item.VolumeBasedCalculationTechnique = 'NONE'
        else:
            frame_type_item.VolumeBasedCalculationTechnique = 'MIXED'

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
    if pixel_representation == 0:
        mf_dataset.PixelPresentation = 'MONOCHROME'
    else:
        mf_dataset.PixelPresentation = 'COLOR'
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

    # Create the Pixel Data element of the mulit-frame image instance using
    # native encoding (simply concatenating pixels of individual frames)
    # Sometimes there may be numpy types such as ">i2". The (* 1) hack
    # ensures that pixel values have the correct integer type.
    encoded_frames = [
        encode_frame(
            ds.pixel_array * 1,
            transfer_syntax_uid=mf_dataset.file_meta.TransferSyntaxUID,
            bits_allocated=ds.BitsAllocated,
            bits_stored=ds.BitsStored,
            photometric_interpretation=ds.PhotometricInterpretation,
            pixel_representation=ds.PixelRepresentation
        )
        for ds in sf_datasets
    ]
    if mf_dataset.file_meta.TransferSyntaxUID.is_encapsulated:
        mf_dataset.PixelData = encapsulate(encoded_frames)
    else:
        mf_dataset.PixelData = b''.join(encoded_frames)

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
        transfer_syntax_uid: str = ExplicitVRLittleEndian,
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
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following compressed transfer syntaxes
            are supported: JPEG 2000 Lossless (``"1.2.840.10008.1.2.4.90"``)
            and JPEG-LS Lossless (``"1.2.840.10008.1.2.4.80"``).
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

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            JPEG2000Lossless,
            JPEGLSLossless,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported'
            )

        super().__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=ref_ds.Manufacturer,
            modality=ref_ds.Modality,
            transfer_syntax_uid=transfer_syntax_uid,
            patient_id=ref_ds.PatientID,
            patient_name=ref_ds.PatientName,
            patient_birth_date=ref_ds.PatientBirthDate,
            patient_sex=ref_ds.PatientSex,
            accession_number=ref_ds.AccessionNumber,
            study_id=ref_ds.StudyID,
            study_date=ref_ds.StudyDate,
            study_time=ref_ds.StudyTime,
            referring_physician_name=getattr(
                ref_ds, 'ReferringPhysicianName', None
            ),
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
        transfer_syntax_uid: str = ExplicitVRLittleEndian,
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
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following compressed transfer syntaxes
            are supported: JPEG 2000 Lossless (``"1.2.840.10008.1.2.4.90"``)
            and JPEG-LS Lossless (``"1.2.840.10008.1.2.4.80"``).
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
            transfer_syntax_uid=transfer_syntax_uid,
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
        transfer_syntax_uid: str = ExplicitVRLittleEndian,
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
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements. The following compressed transfer syntaxes
            are supported: JPEG 2000 Lossless (``"1.2.840.10008.1.2.4.90"``)
            and JPEG-LS Lossless (``"1.2.840.10008.1.2.4.80"``).
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

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            JPEG2000Lossless,
            JPEGLSLossless,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported'
            )

        super().__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=ref_ds.Manufacturer,
            modality=ref_ds.Modality,
            transfer_syntax_uid=transfer_syntax_uid,
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
