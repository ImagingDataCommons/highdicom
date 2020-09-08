""" Module for SOP Classes of Legacy Converted Enhanced Image IODs."""
from __future__ import annotations
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from numpy import log10, array, ceil, cross, dot, ndarray
from pydicom.datadict import tag_for_keyword, dictionary_VR, keyword_for_tag
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence as DicomSequence
from pydicom.multival import MultiValue
from datetime import date, datetime, time, timedelta
from pydicom.valuerep import DT, DA, TM, DSfloat
from copy import deepcopy
from pydicom.uid import UID
from highdicom.base import SOPClass
from sys import float_info
from highdicom.legacy import SOP_CLASS_UIDS
from highdicom._iods import IOD_MODULE_MAP
from highdicom._modules import MODULE_ATTRIBUTE_MAP
from abc import ABC, abstractmethod


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
    '1.2.840.10008.5.1.4.1.1.2.2':  'legacy-converted-enhanced-ct-image',
    '1.2.840.10008.5.1.4.1.1.4.4':  'legacy-converted-enhanced-mr-image',
    '1.2.840.10008.5.1.4.1.1.128.1': 'legacy-converted-enhanced-pet-image',
}


class Abstract_MultiframeModuleAdder(ABC):

    def __init__(self, sf_datasets: Sequence[Dataset],
                 excluded_from_perframe_tags: dict,
                 excluded_from_functional_tags: dict,
                 perframe_tags: dict,
                 shared_tags: dict,
                 multi_frame_output: Dataset):

        self.excluded_from_per_frame_tags = excluded_from_perframe_tags
        self.excluded_from_functional_group_tags = excluded_from_functional_tags
        self._perframe_tags = perframe_tags
        self._shared_tags = shared_tags
        self.target_dataset = multi_frame_output
        self.single_frame_set = sf_datasets
        self.earliest_date = DA('00010101')
        self.earliest_time = TM('000000')
        self.earliest_date_time = DT('00010101000000')
        self.farthest_future_date = DA('99991231')
        self.farthest_future_time = TM('235959')
        self.farthest_future_date_time = DT('99991231235959')

    def _is_empty_or_empty_items(self, attribute: DataElement) -> bool:
        if attribute.is_empty:
            return True
        if isinstance(attribute.value, Sequence):
            if len(attribute.value) == 0:
                return True
            for item in attribute.value:
                for tg, v in item.items():
                    v = item[tg]
                    if not self._is_empty_or_empty_items(v):
                        return False
        return False

    def _mark_tag_as_used(self, tg: Tag) -> None:
        if tg in self._shared_tags:
            self._shared_tags[tg] = True
        elif tg in self.excluded_from_per_frame_tags:
            self.excluded_from_per_frame_tags[tg] = True
        elif tg in self._perframe_tags:
            self._perframe_tags[tg] = True

    def _copy_attrib_if_present(self, src_ds: Dataset, dest_ds: Dataset,
                                src_kw_or_tg: str, dest_kw_or_tg: str = None,
                                check_not_to_be_perframe: bool = True,
                                check_not_to_be_empty: bool = False) -> None:
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

    def _get_perframe_item(self, index: int) -> Dataset:
        if index > len(self.single_frame_set):
            return None
        pf_kw: str = 'PerFrameFunctionalGroupsSequence'
        pf_tg = tag_for_keyword(pf_kw)
        if pf_tg not in self.target_dataset:
            seq = []
            for i in range(0, len(self.single_frame_set)):
                seq.append(Dataset())
            self.target_dataset[pf_tg] = DataElement(pf_tg,
                                                     'SQ',
                                                     DicomSequence(seq))
        return self.target_dataset[pf_tg].value[index]

    def _get_shared_item(self) -> Dataset:
        sf_kw = 'SharedFunctionalGroupsSequence'
        sf_tg = tag_for_keyword(sf_kw)
        if sf_kw not in self.target_dataset:
            seq = [Dataset()]
            self.target_dataset[sf_tg] = DataElement(sf_tg,
                                                     'SQ',
                                                     DicomSequence(seq))
        return self.target_dataset[sf_tg].value[0]

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
        if a.VR == 'DA' and isinstance(a.value, str):
            try:
                a.value = DA(a.value)
            except BaseException:
                a.value = DA(default)
        if a.VR == 'DT' and isinstance(a.value, str):
            try:
                a.value = DT(a.value)
            except BaseException:
                a.value = DT(default)
        if a.VR == 'TM' and isinstance(a.value, str):
            try:
                a.value = TM(a.value)
            except BaseException:
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
        attribs: list = MODULE_ATTRIBUTE_MAP[module_name]
        ref_dataset = self.single_frame_set[0]
        for a in attribs:
            kw: str = a['keyword']
            if kw in excepted_attributes:
                continue
            if len(a['path']) == 0:
                self._copy_attrib_if_present(
                    ref_dataset, self.target_dataset, kw,
                    check_not_to_be_perframe=check_not_to_be_perframe,
                    check_not_to_be_empty=check_not_to_be_empty)

    @abstractmethod
    def add_module(self) -> None:
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

    def add_module(self) -> None:
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

    def add_module(self) -> None:
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
        self.modality = modality

    def _get_value_for_frame_type(self,
                                  attrib: DataElement) -> Union[list, None]:
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

    def _get_frame_type_seq_tag(self) -> int:
        seq_kw = '{}{}FrameTypeSequence'
        if self.modality == 'PET':
            seq_kw = seq_kw.format(self.modality, '')
        else:
            seq_kw = seq_kw.format(self.modality, 'Image')
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

    def add_module(self) -> None:
        im_type_tag = tag_for_keyword('ImageType')
        seq_tg = self._get_frame_type_seq_tag()
        if im_type_tag not in self._perframe_tags:
            self._add_module_to_functional_group(self.single_frame_set[0],
                                                 self.target_dataset, 0)
            # ----------------------------
            item = self._get_shared_item()
            inner_item = Dataset()
            self._add_module_to_functional_group(self.single_frame_set[0],
                                                 inner_item, 1)
            item[seq_tg] = DataElement(seq_tg, 'SQ', [inner_item])
        else:
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                inner_item = Dataset()
                self._add_module_to_functional_group(self.single_frame_set[i],
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

    def add_module(self) -> None:
        # ct_mr = CommonCTMRImageDescriptionMacro(self.single_frame_set
        # , self.excluded_from_per_frame_tags
        # , self._perframe_tags
        # , self._shared_tags
        # , self.target_dataset)
        # ct_mr.add_module()
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
        ref_dataset = self.single_frame_set[0]
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
                ref_dataset, self.target_dataset, kw,
                check_not_to_be_perframe=True,
                check_not_to_be_empty=False)
        if tag_for_keyword('PresentationLUTShape') not in self._perframe_tags:
            # actually should really invert the pixel data if MONOCHROME1,
            #           since only MONOCHROME2 is permitted : (
            # also, do not need to check if PhotometricInterpretation is
            #           per-frame, since a distinguishing attribute
            phmi_kw = 'PhotometricInterpretation'
            phmi_a = self._get_or_create_attribute(self.single_frame_set[0],
                                                   phmi_kw,
                                                   "MONOCHROME2")
            LUT_shape_default = "INVERTED" if phmi_a.value == 'MONOCHROME1'\
                else "IDENTITY"
            LUT_shape_a = self._get_or_create_attribute(
                self.single_frame_set[0],
                'PresentationLUTShape',
                LUT_shape_default)
            if not LUT_shape_a.is_empty:
                self.target_dataset['PresentationLUTShape'] = LUT_shape_a
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

    def add_module(self) -> None:
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

    def add_module(self) -> None:
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

    def add_module(self) -> None:
        # David's code doesn't hold anything for this module ... should ask him
        kw = 'ContentQualification'
        tg = tag_for_keyword(kw)
        elem = self._get_or_create_attribute(
            self.single_frame_set[0], kw, 'RESEARCH')
        self.target_dataset[tg] = elem


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

    def add_module(self) -> None:
        self._copy_attrib_if_present(
            self.single_frame_set[0],
            self.target_dataset,
            "ResonantNucleus",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        if 'ResonantNucleus' not in self.target_dataset:
            # derive from ImagedNucleus, which is the one used in legacy MR
            #  IOD, but does not have a standard list of defined terms ...
            #  (could check these : ()
            self._copy_attrib_if_present(
                self.single_frame_set[0],
                self.target_dataset,
                "ImagedNucleus",
                check_not_to_be_perframe=True,
                check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.single_frame_set[0],
            self.target_dataset,
            "KSpaceFiltering",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.single_frame_set[0],
            self.target_dataset,
            "MagneticFieldStrength",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.single_frame_set[0],
            self.target_dataset,
            "ApplicableSafetyStandardAgency",
            check_not_to_be_perframe=True,
            check_not_to_be_empty=True)
        self._copy_attrib_if_present(
            self.single_frame_set[0],
            self.target_dataset,
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

    def add_module(self) -> None:
        tg = tag_for_keyword('AcquisitionContextSequence')
        if tg not in self._perframe_tags:
            self.target_dataset[tg] = self._get_or_create_attribute(
                self.single_frame_set[0],
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

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
             self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
            self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
            self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
            self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
            self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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
        if "RescaleType" not in item:
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
                elif modality == 'PT':
                    value = 'US' if 'Units' not in src_fg\
                        else src_fg['Units'].value
            if value != '':
                tg = tag_for_keyword('RescaleType')
                item[tg] = DataElement(tg, dictionary_VR(tg), value)
        kw = 'PixelValueTransformationSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
            self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
            self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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

    def add_module(self) -> None:
        if (not self._contains_right_attributes(self._perframe_tags) and
            (self._contains_right_attributes(self._shared_tags) or
            self._contains_right_attributes(self.excluded_from_per_frame_tags))
            ):
            item = self._get_shared_item()
            self._add_module_to_functional_group(self.single_frame_set[0], item)
        elif self._contains_right_attributes(self._perframe_tags):
            for i in range(0, len(self.single_frame_set)):
                item = self._get_perframe_item(i)
                self._add_module_to_functional_group(
                    self.single_frame_set[i], item)


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
        lval = float_info.min
        if ltg in self._perframe_tags:
            for frame in self.single_frame_set:
                if ltg in frame:
                    nval = frame[ltg].value
                else:
                    continue
                lval = nval if lval < nval else lval
            if lval > float_info.min:
                self.target_dataset[ltg] = DataElement(ltg, 'SS', int(lval))
    # ==========================
        stg = tag_for_keyword("SmallestImagePixelValue")
        sval = float_info.max
        if stg in self._perframe_tags:
            for frame in self.single_frame_set:
                if stg in frame:
                    nval = frame[stg].value
                else:
                    continue
                sval = nval if sval < nval else sval
            if sval < float_info.max:
                self.target_dataset[stg] = DataElement(stg, 'SS', int(sval))

        stg = "SmallestImagePixelValue"

    def add_module(self) -> None:
        # first collect all not used tags
        # note that this is module is order dependent
        self._add_largest_smallest_pixle_value()
        self._eligeible_tags: List[Tag] = []
        for tg, used in self._perframe_tags.items():
            if not used and tg not in self.excluded_from_functional_group_tags:
                self._eligeible_tags.append(tg)
        for i in range(0, len(self.single_frame_set)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(
                self.single_frame_set[i], item)


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
        for tg, used in self._shared_tags.items():
            if (not used and
                    tg not in self.target_dataset and
                    tg not in self.excluded_from_functional_group_tags):
                self._copy_attrib_if_present(src_fg,
                                             item,
                                             tg,
                                             check_not_to_be_perframe=False,
                                             check_not_to_be_empty=False)
        kw = 'UnassignedSharedConvertedAttributesSequence'
        tg = tag_for_keyword(kw)
        seq = DataElement(tg, dictionary_VR(tg), [item])
        dest_fg[tg] = seq

    def add_module(self) -> None:
        item = self._get_shared_item()
        self._add_module_to_functional_group(self.single_frame_set[0], item)


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

    def create_empty_element(self, tg: Tag) -> DataElement:
        return DataElement(tg, dictionary_VR(tg), None)

    def add_module(self) -> None:
        iod_name = _SOP_CLASS_UID_IOD_KEY_MAP[
            self.target_dataset['SOPClassUID'].value]
        modules = IOD_MODULE_MAP[iod_name]
        for module in modules:
            if module['usage'] == 'M':
                mod_key = module['key']
                attrib_list = MODULE_ATTRIBUTE_MAP[mod_key]
                for a in attrib_list:
                    if len(a['path']) == 0 and a['type'] == '2':
                        tg = tag_for_keyword(a['keyword'])
                        if (tg not in self.single_frame_set[0] and
                           tg not in self.target_dataset and
                           tg not in self._perframe_tags and
                           tg not in self._shared_tags):
                            self.target_dataset[tg] =\
                                self.create_empty_element(tg)


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

    def add_module(self) -> None:
        for i in range(0, len(self.single_frame_set)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(
                self.single_frame_set[i], item)


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
        self.earliest_frame_acquisition_date_time = \
            self.farthest_future_date_time
        self._slices: list = []
        self._tolerance = 0.0001
        self._slice_location_map: dict = {}

    def _build_slices_geometry(self) -> None:
        frame_count = len(self.single_frame_set)
        for i in range(0, frame_count):
            curr_frame = self.single_frame_set[i]
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
                print("Error in geometri ...")
                self._slices = []  # clear the slices
                break

    def _are_all_slices_parallel(self) -> bool:
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

    def _add_stack_info(self) -> None:
        self._build_slices_geometry()
        round_digits = int(ceil(-log10(self._tolerance)))
        if self._are_all_slices_parallel():
            self._slice_location_map = {}
            for idx, s in enumerate(self._slices):
                dist = round(s.get_distance_along_origin(), round_digits)
                if dist in self._slice_location_map:
                    self._slice_location_map[dist].append(idx)
                else:
                    self._slice_location_map[dist] = [idx]
            distance_index = 1
            frame_content_tg = tag_for_keyword("FrameContentSequence")
            for loc, idxs in sorted(self._slice_location_map.items()):
                if len(idxs) != 1:
                    print('Error')
                for frame_index in idxs:
                    frame = self._get_perframe_item(frame_index)
                    new_item = frame[frame_content_tg].value[0]
                    new_item["StackID"] = self._get_or_create_attribute(
                        self.single_frame_set[0],
                        "StackID", "0")
                    new_item["InStackPositionNumber"] =\
                        self._get_or_create_attribute(
                        self.single_frame_set[0],
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
            src_fg, 'AcquisitionDateTime',  self.earliest_date_time)
        # chnage the keyword to FrameAcquisitionDateTime:
        FrameAcquisitionDateTime_a = DataElement(
            tag_for_keyword('FrameAcquisitionDateTime'),
            'DT', AcquisitionDateTime_a.value)
        AcquisitionDateTime_is_perframe = self._contains_right_attributes(
            self._perframe_tags)
        if FrameAcquisitionDateTime_a.value == self.earliest_date_time:
            AcquisitionDate_a = self._get_or_create_attribute(
                src_fg, 'AcquisitionDate', self.earliest_date)
            AcquisitionTime_a = self._get_or_create_attribute(
                src_fg, 'AcquisitionTime', self.earliest_time)
            d = AcquisitionDate_a.value
            t = AcquisitionTime_a.value
            # FrameAcquisitionDateTime_a.value = (DT(d.strftime('%Y%m%d') +
            #                                     t.strftime('%H%M%S')))
            FrameAcquisitionDateTime_a.value = DT(str(d) + str(t))
        if FrameAcquisitionDateTime_a.value > self.earliest_date_time:
            if (FrameAcquisitionDateTime_a.value <
                    self.earliest_frame_acquisition_date_time):
                self.earliest_frame_acquisition_date_time =\
                    FrameAcquisitionDateTime_a.value
            if not AcquisitionDateTime_is_perframe:
                if ('TriggerTime' in src_fg and
                        'FrameReferenceDateTime' not in src_fg):
                    TriggerTime_a = self._get_or_create_attribute(
                        src_fg, 'TriggerTime', self.earliest_time)
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
        for i in range(0, len(self.single_frame_set)):
            item = self._get_perframe_item(i)
            self._add_module_to_functional_group(
                self.single_frame_set[i], item)
        if self.earliest_frame_acquisition_date_time <\
            self.farthest_future_date_time:
            kw = 'AcquisitionDateTime'
            self.target_dataset[kw] = DataElement(
                tag_for_keyword(kw),
                'DT', self.earliest_frame_acquisition_date_time)

    def add_module(self) -> None:
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

    def add_module(self) -> None:
        kw = 'NumberOfframes'
        tg = tag_for_keyword(kw)
        self._frame_count = len(self.single_frame_set)
        self.target_dataset[kw] =\
            DataElement(tg, dictionary_VR(tg), self._frame_count)
        row = self.single_frame_set[0]["Rows"].value
        col = self.single_frame_set[0]["Columns"].value
        self._number_of_pixels_per_frame = row * col
        self._number_of_pixels = row * col * self._frame_count
        kw = "PixelData"
        for i in range(0, len(self.single_frame_set)):
            PixelData_a = self.single_frame_set[i][kw]
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
        self.target_dataset[kw] = MF_PixelData


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
        self.earliest_content_date_time = self.farthest_future_date_time

    def add_module(self) -> None:
        for i in range(0, len(self.single_frame_set)):
            src = self.single_frame_set[i]
            kw = 'ContentDate'
            d = DA(
                self.farthest_future_date if kw not in src else src[kw].value)
            kw = 'ContentTime'
            t = TM(
                self.farthest_future_time if kw not in src else src[kw].value)
            value = DT(d.strftime('%Y%m%d') + t.strftime('%H%M%S.%f'))
            if self.earliest_content_date_time > value:
                self.earliest_content_date_time = value
        if self.earliest_content_date_time < self.farthest_future_date_time:
            n_d = DA(self.earliest_content_date_time.date().strftime('%Y%m%d'))
            n_t = TM(
                self.earliest_content_date_time.time().strftime('%H%M%S.%f'))
            kw = 'ContentDate'
            self.target_dataset[kw] = DataElement(
                tag_for_keyword(kw), 'DA', n_d)
            kw = 'ContentTime'
            self.target_dataset[kw] = DataElement(
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

    def add_module(self) -> None:
        nnooww = datetime.now()
        n_d = DA(nnooww.date().strftime('%Y%m%d'))
        n_t = TM(nnooww.time().strftime('%H%M%S'))
        kw = 'InstanceCreationDate'
        self.target_dataset[kw] = DataElement(
            tag_for_keyword(kw), 'DA', n_d)
        kw = 'InstanceCreationTime'
        self.target_dataset[kw] = DataElement(
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

    def add_module(self) -> None:
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
        self.target_dataset[tg] = DataElement(tg, 'SQ', [item])


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
        legacy_datasets = frame_set.frames
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('No DICOM data sets of provided.')
        sop_class_uid = LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID]
        if sort_key is None:
            sort_key = LegacyConvertedEnhanceImage.default_sort_key
        super().__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=sop_class_uid,
            instance_number=instance_number,
            manufacturer=ref_ds.Manufacturer,
            modality=ref_ds.modality,
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
        self._legacy_datasets = legacy_datasets
        self.distinguishing_attributes_tags = self._get_tag_used_dictionary(
            frame_set.distinguishing_attributes_tags)
        self.excluded_from_per_frame_tags = self._get_tag_used_dictionary(
            frame_set.excluded_from_per_frame_tags)
        self._perframe_tags = self._get_tag_used_dictionary(
            frame_set.perframe_tags)
        self._shared_tags = self._get_tag_used_dictionary(
            frame_set.shared_tags)
        self.excluded_from_functional_group_tags = {
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
            self.add_build_blocks_for_ct()
        elif (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-mr-image'):
            self.add_build_blocks_for_mr()
        elif (_SOP_CLASS_UID_IOD_KEY_MAP[sop_class_uid] ==
                'legacy-converted-enhanced-pet-image'):
            self.add_build_blocks_for_pet()

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

    def add_new_build_block(
        self, element: Abstract_MultiframeModuleAdder) -> None:
        if not isinstance(element, Abstract_MultiframeModuleAdder):
            raise ValueError('Build block must be an instance '
                             'of Abstract_MultiframeModuleAdder')
        self.__build_blocks.append(element)

    def clear_build_blocks(self) -> None:
        self.__build_blocks = []

    def add_common_ct_pet_mr_build_blocks(self) -> None:
        Blocks = [
            ImagePixelModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            CompositeInstanceContex(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            EnhancedCommonImageModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            AcquisitionContextModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            FrameAnatomyFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            PixelMeasuresFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            PlaneOrientationFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            PlanePositionFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            FrameVOILUTFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            PixelValueTransformationFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            ReferencedImageFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            ConversionSourceFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            FrameContentFunctionalGroup(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            PixelData(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            ContentDateTime(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            InstanceCreationDateTime(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            ContributingEquipmentSequence(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            UnassignedPerFrame(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            UnassignedShared(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self)  # ,
            # StackInformation(
            #     self._legacy_datasets,
            #     self.excluded_from_per_frame_tags,
            #     self.excluded_from_functional_group_tags,
            #     self._perframe_tags,
            #     self._shared_tags,
            #      self),
            # EmptyType2Attributes(
            #     self._legacy_datasets,
            #     self.excluded_from_per_frame_tags,
            #     self.excluded_from_functional_group_tags,
            #     self._perframe_tags,
            #     self._shared_tags,
            #     self)
        ]
        for b in Blocks:
            self.add_new_build_block(b)

    def add_ct_specific_build_blocks(self) -> None:
        Blocks = [
            CommonCTMRPETImageDescriptionMacro(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self,
                'CT'),
            EnhancedCTImageModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            ContrastBolusModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self)
        ]
        for b in Blocks:
            self.add_new_build_block(b)

    def AddMRSpecificBuildBlocks(self) -> None:
        Blocks = [
            CommonCTMRPETImageDescriptionMacro(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self,
                'MR'),
            EnhancedMRImageModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self),
            ContrastBolusModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self)
        ]
        for b in Blocks:
            self.add_new_build_block(b)

    def add_pet_specific_build_blocks(self) -> None:
        Blocks = [
            CommonCTMRPETImageDescriptionMacro(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self,
                'PET'),
            EnhancedPETImageModule(
                self._legacy_datasets,
                self.excluded_from_per_frame_tags,
                self.excluded_from_functional_group_tags,
                self._perframe_tags,
                self._shared_tags,
                self)
        ]
        for b in Blocks:
            self.add_new_build_block(b)

    def add_build_blocks_for_ct(self) -> None:
        self.clear_build_blocks()
        self.add_common_ct_pet_mr_build_blocks()
        self.add_ct_specific_build_blocks()

    def add_build_blocks_for_mr(self) -> None:
        self.clear_build_blocks()
        self.add_common_ct_pet_mr_build_blocks()
        self.add_mr_specific_build_blocks()

    def add_build_blocks_for_pet(self) -> None:
        self.clear_build_blocks()
        self.add_common_ct_pet_mr_build_blocks()
        self.add_pet_specific_build_blocks()

    def BuildMultiFrame(self) -> None:
        for builder in self.__build_blocks:
            builder.add_module()


class GeometryOfSlice:
    def __init__(self,
                 row_vector: ndarray,
                 col_vector: ndarray,
                 top_left_corner_pos: ndarray,
                 voxel_spaceing: ndarray,
                 dimensions: tuple):
        self.row_vector = row_vector
        self.col_vector = col_vector
        self.top_left_corner_position = top_left_corner_pos
        self.voxel_spacing = voxel_spaceing
        self.dim = dimensions

    def get_normal_vector(self) -> ndarray:
        n: ndarray = cross(self.row_vector, self.col_vector)
        n[2] = -n[2]
        return n

    def get_distance_along_origin(self) -> float:
        n = self.get_normal_vector()
        return float(
            dot(self.top_left_corner_position, n))

    def are_parallel(slice1: GeometryOfSlice,
                     slice2: GeometryOfSlice,
                     tolerance: float = 0.0001) -> bool:
        if (isinstance(slice1, GeometryOfSlice) == False) or\
                (isinstance(slice2, GeometryOfSlice) == False):
            print('Error')
            return False
        else:
            n1: ndarray = slice1.get_normal_vector()
            n2: ndarray = slice2.get_normal_vector()
            for el1, el2 in zip(n1, n2):
                if abs(el1 - el2) > tolerance:
                    return False
            return True


class DicomHelper:
    def __init__(self) -> None:
        pass

    def istag_file_meta_information_group(t: Tag) -> bool:
        return t.group == 0x0002

    def istag_repeating_group(t: Tag) -> bool:
        g = t.group
        return (g >= 0x5000 and g <= 0x501e) or\
            (g >= 0x6000 and g <= 0x601e)

    def istag_group_length(t: Tag) -> bool:
        return t.element == 0

    def isequal(v1: Any, v2: Any) -> bool:
        float_tolerance = 1.0e-5

        def is_equal_float(x1: float, x2: float) -> bool:
            return abs(x1 - x2) < float_tolerance
        if not type(v1) == type(v2):
            return False
        if isinstance(v1, DicomSequence):
            for item1, item2 in zip(v1, v2):
                DicomHelper.isequal_dicom_dataset(item1, item2)
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

    def isequal_dicom_dataset(ds1: Dataset, ds2: Dataset) -> bool:
        if type(ds1) != type(ds2):
            return False
        if not isinstance(ds1, Dataset):
            return False
        for k1, elem1 in ds1.items():
            if k1 not in ds2:
                return False
            elem2 = ds2[k1]
            if not DicomHelper.isequal(elem2.value, elem1.value):
                return False
        return True


class FrameSet:
    def __init__(self, single_frame_list: list,
                 distinguishing_tags: list):
        self._frames = single_frame_list
        self._distinguishing_attributes_tags = distinguishing_tags
        tmp = [
            tag_for_keyword('AcquisitionDateTime'),
            tag_for_keyword('AcquisitionDate'),
            tag_for_keyword('AcquisitionTime'),
            tag_for_keyword('SpecificCharacterSet')]
        self._excluded_from_per_frame_tags =\
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
    def excluded_from_per_frame_tags(self) -> List[Tag]:
        return self._excluded_from_per_frame_tags[:]

    @property
    def perframe_tags(self) -> List[Tag]:
        return self._perframe_tags[:]

    @property
    def shared_tags(self) -> List[Tag]:
        return self._shared_tags[:]

    def GetSOPInstanceUIDList(self) -> list:
        OutputList: list = []
        for f in self._frames:
            OutputList.append(f.SOPInstanceUID)
        return OutputList

    def GetSOPClassUID(self) -> UID:
        return self._frames[0].SOPClassUID

    def _find_per_frame_and_shared_tags(self) -> None:
        rough_shared: dict = {}
        sfs = self.frames
        for ds in sfs:
            for ttag, elem in ds.items():
                if (not ttag.is_private and not
                    DicomHelper.istag_file_meta_information_group(ttag) and not
                        DicomHelper.istag_repeating_group(ttag) and not
                        DicomHelper.istag_group_length(ttag) and not
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
                    if not DicomHelper.isequal(v_i, v[0]):
                        all_values_are_equal = False
                        break
                if not all_values_are_equal:
                    to_be_removed_from_shared.append(ttag)
        for t, v in rough_shared.items():
            if keyword_for_tag(t) != 'PatientSex':
                continue
        for t in to_be_removed_from_shared:
            del rough_shared[t]
        for t, v in rough_shared.items():
            self._shared_tags.append(t)
            self._perframe_tags.remove(t)

    def _istag_excluded_from_perframe(self, t: Tag) -> bool:
        return t in self.excluded_from_per_frame_tags


class FrameSetCollection:
    def __init__(self, single_frame_list: list):
        self.mixed_frames = single_frame_list
        self.mixed_frames_copy = self.mixed_frames[:]
        self._distinguishing_attribute_keywords = [
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
        self._frame_sets: list = []
        while len(self.mixed_frames_copy) != 0:
            x = self._find_all_similar_to_first_datasets()
            self._frame_sets.append(FrameSet(x[0], x[1]))
        for kw in to_be_removed_from_distinguishing_attribs:
            self.distinguishing_attribute_keywords.remove(kw)
        self.excluded_from_per_frame_tags = {}
        for i in self.distinguishing_attribute_keywords:
            self.excluded_from_per_frame_tags[tag_for_keyword(i)] = False
        self.excluded_from_per_frame_tags[
            tag_for_keyword('AcquisitionDateTime')] = False
        self.excluded_from_per_frame_tags[
            tag_for_keyword('AcquisitionDate')] = False
        self.excluded_from_per_frame_tags[
            tag_for_keyword('AcquisitionTime')] = False
        self.excluded_from_functional_group_tags = {
            tag_for_keyword('SpecificCharacterSet'): False}

    def _find_all_similar_to_first_datasets(self) -> tuple:
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
        for ds in self.mixed_frames_copy:
            all_equal = True
            for tg in distinguishing_tags_missing:
                if tg in ds:
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
                    all_equal = False
                    break
            if all_equal:
                similar_ds.append(ds)
        for ds in similar_ds:
            if ds in self.mixed_frames_copy:
                self.mixed_frames_copy.remove(ds)
        return (similar_ds, distinguishing_tags_existing)

    @property
    def distinguishing_attribute_keywords(self) -> List[str]:
        return self._distinguishing_attribute_keywords[:]

    @property
    def FrameSets(self) -> List[FrameSet]:
        return self._frame_sets
