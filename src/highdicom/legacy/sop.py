""" Module for SOP Classes of Legacy Converted Enhanced Image IODs.
For the most part the single frame to multi-frame conversion logic is taken
from `PixelMed <https://www.dclunie.com>`_ by David Clunie

"""
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import pkgutil
from sys import float_info
from typing import cast, Any, Union, Callable, Sequence, Tuple

from pydicom.datadict import tag_for_keyword, dictionary_VR
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate, encapsulate_extended
from pydicom.tag import BaseTag
from pydicom.uid import (
    LegacyConvertedEnhancedCTImageStorage,
    LegacyConvertedEnhancedMRImageStorage,
    LegacyConvertedEnhancedPETImageStorage,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGLSLossless,
    RLELossless,
    UID,
)
from pydicom.valuerep import DT, DA, TM, format_number_as_ds

from highdicom.image import Image, _Image
from highdicom.base_content import ContributingEquipment
from highdicom.frame import encode_frame
from highdicom.spatial import get_series_volume_positions

# TODO defer these imports
from highdicom._modules import MODULE_ATTRIBUTE_MAP
from highdicom.sr.coding import CodedConcept


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


# List of attributes required to be consistent between each frame for the
# conversion to be valid
_CONSISTENT_KEYWORDS = [
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
]


def _istag_file_meta_information_group(t: BaseTag) -> bool:
    return t.group == 0x0002


def _istag_repeating_group(t: BaseTag) -> bool:
    g = t.group
    return (
        (g >= 0x5000 and g <= 0x501e) or
        (g >= 0x6000 and g <= 0x601e)
    )


@lru_cache(maxsize=1)
def _get_anatomic_region_mapping() -> dict[str, CodedConcept]:
    """Get a mapping from body part examined to SCT codes.

    This mapping is defined in the standard at :dcm:`Annex L
    <part16/chapter_L.html>` and intended to modernize body parts expressed in
    the old "BodyPartExamined" attribute to the more standardized
    "AnatomicRegionSequence", using SNOMED controlled terminology.

    Returns
    -------
    dict[str, highdicom.sr.CodedConcept]
        Mapping from old-style BodyPartExamined values to SNOMED codes used for
        AnatomicRegionSequence.

    """
    data_file = pkgutil.get_data(
        'highdicom',
        '_standard/anatomic_regions.json'
    )
    anatomic_regions = json.loads(data_file.decode('utf-8'))

    return {
        k: CodedConcept(value=v[1], scheme_designator=v[0], meaning=v[2])
        for k, v in anatomic_regions.items()
    }


def _istag_group_length(t: BaseTag) -> bool:
    return t.element == 0


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


@dataclass
class _AttributeConfig:
    """Configuration for how to copy an attribute to the converted instance.

    Parameters
    ----------
    dest_kw: str
        Keyword where the attribute will be placed.
    src_kws: list[str] | None, optional
        List of source keywords to search for the value to place into the
        destination keyword. If None, the dest_kw is used as the (only) source
        keyword. If a list is provided, the keywords are searched in order
        until a value is found to place into the destination. In this case, the
        dest_kw is not used unless explicitly specified.
    default_val: Any | None
        Default value to use if attribute is not found in the source dataset.
    """
    dest_kw: str
    src_kws: list[str] | None = None
    default_val: Any = None

    def get_source_keywords(self) -> list[str]:
        """Get a list of all potential source keywords."""
        if self.src_kws is not None:
            # Use the src_kws, if specified
            return self.src_kws

        # Otherwise use the destination keyword as the source keyword
        return [self.dest_kw]


@dataclass
class _FunctionalGroupConfig:
    """Configuration for creating a functional group.

    Parameters
    ----------
    sequence_name: str
        Name of the sequence where the functional group will be placed.
    attribute_configs: list[highdicom.legacy.sop._AttributeConfig]
        Configurations for each attribute to place into the sequence.
    further_source_attributes: list[str]
        Keywords for further attributes that are not copied to the destination
        but should be checked for presence and consistency in the source to
        determine whether this functional group should be included and whether
        it should be per-frame or shared. Typically, these will be attributes
        that are processed in the custom_logic_callback rather than simply
        copied over to the destination.
    custom_logic_callback: Callable[[pydicom.Dataset, pydicom.Dataset], None] | None
        Callback implementing custom logic to call after the other parameters
        have been copied. Takes the source and desitnation datasets as input
        parameters and has no return value.

    """  # noqa: E501
    sequence_name: str
    attribute_configs: list[_AttributeConfig]
    further_source_attributes: list[str] | None = None
    custom_logic_callback: Callable[[Dataset, Dataset], None] | None = None


@dataclass
class _ModuleConfig:
    """Configuration for creating a module in the destination dataset.

    Parameters
    ----------
    module_name: str
        Name of the module (in highdicom's module list). This format uses lower
        case and hyphens to separate words.
    attribute_configs: list[highdicom.legacy.sop._AttributeConfig] | None
        Attribute-level configurations. If ``auto_discover_attributes`` is
        True, configurations are only required for attributes that deviate from
        the default behavior.
    skip_attributes: list[str] | None
        List of attributes to skip.
    auto_discover_attributes: bool
        Whether to use built-in standard information to list attributes
        automatically.
    custom_logic_callback: Callable[[], None] | None
        Callback implementing custom logic to call after the other parameters
        have been copied. Takes no parameters and has no return value.

    """
    module_name: str
    attribute_configs: list[_AttributeConfig] | None = None
    skip_attributes: list[str] | None = None
    auto_discover_attributes: bool = True
    custom_logic_callback: Callable[[], None] | None = None


class _LegacyConversionRunner:
    """Utility class for running a conversion process and associated state.

    This class holds state required by the conversion process, but no longer
    needed after the process has concluded.

    """

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        destination: Dataset,
    ) -> None:
        self._legacy_datasets = list(legacy_datasets)
        self._destination = destination
        self._tag_shared_dict = {}
        self._unused_tags = set()

    def _get_tags_present(self) -> dict[BaseTag, bool]:
        """Find tags present in any dataset and whether they are shared.

        Returns
        -------
        tag_shared_dict: dict[pydicom.tag.BaseTag, bool]
            Dictionary whose keys include all tags present in any dataset. The
            corresponding value is a boolean that indicates whether that tag
            is consistent (in presence and value) across all datasets.

        """
        # Tags that should not be included in the list because they require
        # some special treatment
        exclusions = {
            # Acquisition times require merging across the instances
            tag_for_keyword('AcquisitionDateTime'),
            tag_for_keyword('AcquisitionDate'),
            tag_for_keyword('AcquisitionTime'),

            # TODO why is this here?
            tag_for_keyword('SpecificCharacterSet'),
        }

        all_tags_present = set()
        for ds in self._legacy_datasets:
            for t in ds.keys():
                if (
                    not _istag_file_meta_information_group(t) and
                    not _istag_repeating_group(t) and
                    not _istag_group_length(t) and
                    t not in exclusions
                ):
                    all_tags_present.add(t)

        tag_shared_dict: dict[BaseTag, bool] = {}

        for t in all_tags_present:
            present_in_all = all(t in ds for ds in self._legacy_datasets)

            same_value_in_all = False
            if present_in_all:
                ref_val = self._legacy_datasets[0][t].value
                same_value_in_all = all(
                    ds[t].value == ref_val for ds in self._legacy_datasets
                )

            tag_shared_dict[t] = present_in_all and same_value_in_all

        return tag_shared_dict

    def _check_attribute_consistency(
        self,
        check_geometry: bool = True,
        check_series_instance_uid: bool = True,
    ) -> None:
        """Check whether a list of instances is valid for legacy conversion.

        Checks that a specified list of attributes is consistent for all
        provided datasets.

        Parameters
        ----------
        tag_shared_dict: dict[pydicom.uid.BaseTag, bool]
            Dictionary whose keys include all tags present in any dataset. The
            corresponding value is a boolean that indicates whether that tag
            is consistent (in presence and value) across all datasets.
        check_geometry: bool, optional
            If True, require that the orientations, slice thickness, and pixel
            spacings of all instances are the same.
        check_series_instance_uid: bool, optional
            If True, require that all instances belong to the same series. This
            is not a strict requirement of the standard.

        Raises
        ------
        ValueError:
            If the legacy datasets have inconsistencies in attributes that are
            required to be consistent.

        """
        # List of attributes that must be consistent across instances for the
        # conversion to be valid
        consistent_attributes = _CONSISTENT_KEYWORDS.copy()

        if check_geometry:
            consistent_attributes.extend(
                [
                    'ImageOrientationPatient',
                    'PixelSpacing',
                    'SliceThickness',
                ]
            )
        if check_series_instance_uid:
            consistent_attributes.append('SeriesInstanceUID')

        inconsistencies = []

        for attr in consistent_attributes:
            t = BaseTag(tag_for_keyword(attr))

            if not self._tag_shared_dict.get(t, True):
                inconsistencies.append(attr)

        if len(inconsistencies) > 0:
            inconsistencies_str = ", ".join(inconsistencies)
            raise ValueError(
                "The legacy instances provided are not a valid source for a "
                "legacy conversion because the presence and/or values the "
                "following attribute(s) is not consistent between instances: "
                f"{inconsistencies_str}."
            )

    def _copy_attrib_if_present(
        self,
        src_ds: Dataset,
        dest_ds: Dataset,
        src_kw: str,
        dest_kw: str | None = None,
        ignore_if_perframe: bool = True,
    ) -> None:
        """Copies a dicom attribute value from a keyword in the source Dataset
        to the same keyword or a different keyword in the destination Dataset

        Parameters
        ----------
        src_ds: pydicom.Dataset
            Source dataset to copy the attribute from.
        dest_ds: pydicom.Dataset
            Destination dataset to copy the attribute to.
        src_kw: str
            The keyword from the source dataset to copy.
        dest_kw: str | None, optional
            The keyword of the destination dataset, to copy the value to. If
            None, then the source keyword is used.
        ignore_if_perframe: bool
            If true, then copy is aborted if the source attribute is per-frame.

        """
        src_tg = BaseTag(cast(int, tag_for_keyword(src_kw)))

        if dest_kw is None:
            dest_tg = src_tg
        else:
            dest_tg = BaseTag(cast(int, tag_for_keyword(dest_kw)))

        if ignore_if_perframe:
            if not self._tag_shared_dict.get(src_tg, True):
                return

        if src_tg in src_ds:
            elem = src_ds[src_tg]

            new_elem = deepcopy(elem)
            if dest_tg == src_tg:
                dest_ds[dest_tg] = new_elem
            else:
                new_elem1 = DataElement(
                    dest_tg,
                    dictionary_VR(dest_tg),
                    new_elem.value
                )
                dest_ds[dest_tg] = new_elem1

            # Mark this tag as used
            if src_tg in self._unused_tags:
                self._unused_tags.remove(src_tg)

    def run(self) -> None:
        """Run conversion."""
        self._tag_shared_dict = self._get_tags_present()
        self._check_attribute_consistency()

        # List of attributes that should never be placed into the
        # UnassignedPerFrameConvertedAttributesSequence or
        # UnassignedSharedConvertedAttributesSequence
        excluded_keywords = [
            *_CONSISTENT_KEYWORDS,
            'PixelData',
            'SeriesInstanceUID',
            'AcquisitionDate',
            'AcquisitionTime',
            'AcquisitionDateTime',
        ]
        excluded_tags = [tag_for_keyword(kw) for kw in excluded_keywords]
        self._unused_tags = {
            t for t in self._tag_shared_dict.keys()
            if t not in excluded_tags
        }

        module_configs = [
            _ModuleConfig('patient'),
            _ModuleConfig('clinical-trial-subject'),
            _ModuleConfig(
                'general-study',
                skip_attributes=[
                    'StudyInstanceUID',
                    'RequestingService',
                ]
            ),
            _ModuleConfig(
                'patient-study',
                skip_attributes=[
                    'ReasonForVisit',
                    'ReasonForVisitCodeSequence'
                ],
            ),
            _ModuleConfig('clinical-trial-study'),
            _ModuleConfig(
                'general-series',
                skip_attributes=[
                    'SeriesInstanceUID',
                    'SeriesNumber',
                    'SmallestPixelValueInSeries',
                    'LargestPixelValueInSeries',
                    'PerformedProcedureStepEndDate',
                    'PerformedProcedureStepEndTime'
                ],
            ),
            _ModuleConfig('clinical-trial-series'),
            _ModuleConfig(
                'general-equipment',
                skip_attributes=[
                    'InstitutionalDepartmentTypeCodeSequence'
                ],
            ),
            _ModuleConfig('frame-of-reference'),
            _ModuleConfig(
                'sop-common',
                skip_attributes=[
                    'SOPClassUID',
                    'SOPInstanceUID',
                    'InstanceNumber',
                    'SpecificCharacterSet',
                    'EncryptedAttributesSequence',
                    'MACParametersSequence',
                    'DigitalSignaturesSequence'
                ],
            ),
            _ModuleConfig(
                'general-image',
                skip_attributes=[
                    'ImageType',
                    'AcquisitionDate',
                    'AcquisitionDateTime',
                    'AcquisitionTime',
                    'AnatomicRegionSequence',
                    'PrimaryAnatomicStructureSequence',
                    'IrradiationEventUID',
                    'AcquisitionNumber',
                    'InstanceNumber',
                    'PatientOrientation',
                    'ImageLaterality',
                    'ImagesInAcquisition',
                    'ImageComments',
                    'QualityControlImage',
                    'BurnedInAnnotation',
                    'RecognizableVisualFeatures',
                    'LossyImageCompression',
                    'LossyImageCompressionRatio',
                    'LossyImageCompressionMethod',
                    'RealWorldValueMappingSequence',
                    'IconImageSequence',
                    'PresentationLUTShape'
                ],
            ),
            _ModuleConfig(
                'image-pixel',
                skip_attributes=[
                    'ColorSpace',
                    'PixelDataProviderURL',
                    'ExtendedOffsetTable',
                    'ExtendedOffsetTableLengths',
                    'PixelData',
                ],
            ),
            _ModuleConfig(
                'acquisition-context',
                attribute_configs=[
                    _AttributeConfig(
                        'AcquisitionContextSequence',
                        default_val=[]
                    )
                ],
            ),
        ]

        if (
            self._destination.SOPClassUID ==
            LegacyConvertedEnhancedMRImageStorage
        ):
            module_configs.append(
                _ModuleConfig(
                    'enhanced-mr-image',
                    attribute_configs=[
                        _AttributeConfig(
                            'ResonantNucleus',
                            src_kws=['ResonantNucleus', 'ImagedNucleus'],
                        ),
                    ],
                    skip_attributes=['ImageType'],
                    custom_logic_callback=(
                        self._common_enhanced_image_custom_logic
                    ),
                )
            )
            module_configs.append(_ModuleConfig('contrast-bolus'))
        elif (
            self._destination.SOPClassUID ==
            LegacyConvertedEnhancedCTImageStorage
        ):
            module_configs.append(
                _ModuleConfig(
                    'enhanced-ct-image',
                    skip_attributes=['ImageType'],
                    custom_logic_callback=(
                        self._common_enhanced_image_custom_logic
                    )
                )
            )
            module_configs.append(_ModuleConfig('contrast-bolus'))
        elif (
            self._destination.SOPClassUID ==
            LegacyConvertedEnhancedPETImageStorage
        ):
            module_configs.append(
                _ModuleConfig(
                    'enhanced-pet-image',
                    skip_attributes=['ImageType'],
                    custom_logic_callback=(
                        self._common_enhanced_image_custom_logic
                    )
                )
            )

        for config in module_configs:
            self._add_module(config)

        self._destination.SharedFunctionalGroupsSequence = [Dataset()]
        self._destination.PerFrameFunctionalGroupsSequence = [
            Dataset() for _ in range(len(self._legacy_datasets))
        ]

        self._frame_type_seq_kw = {
            LegacyConvertedEnhancedMRImageStorage: 'MRImageFrameTypeSequence',
            LegacyConvertedEnhancedCTImageStorage: 'CTImageFrameTypeSequence',
            LegacyConvertedEnhancedPETImageStorage: 'PETFrameTypeSequence',
        }[self._destination.SOPClassUID]

        functional_group_configs = [
            # TODO require if BodyPartExamined present in any source image
            _FunctionalGroupConfig(
                'FrameAnatomySequence',
                [
                    _AttributeConfig('AnatomicRegionSequence'),
                    _AttributeConfig('PrimaryAnatomicStructureSequence'),
                ],
                further_source_attributes=['BodyPartExamined'],
                custom_logic_callback=self._frame_anatomy_custom_logic,
            ),
            _FunctionalGroupConfig(
                'PixelMeasuresSequence',
                [
                    _AttributeConfig(
                        'PixelSpacing',
                        src_kws=['PixelSpacing', 'ImagerPixelSpacing'],
                    ),
                    _AttributeConfig('SliceThickness'),
                ],
            ),
            _FunctionalGroupConfig(
                'PlanePositionSequence',
                [_AttributeConfig('ImagePositionPatient')],
            ),
            _FunctionalGroupConfig(
                'PlaneOrientationSequence',
                [_AttributeConfig('ImageOrientationPatient')],
            ),
            _FunctionalGroupConfig(
                'FrameVOILUTSequence',
                [
                    _AttributeConfig('WindowWidth'),
                    _AttributeConfig('WindowCenter'),
                    _AttributeConfig('WindowCenterWidthExplanation'),
                ],
            ),
            _FunctionalGroupConfig(
                'DerivationImageSequence',
                [
                    _AttributeConfig('DerivationDescription'),
                    _AttributeConfig('DerivationCodeSequence'),
                    _AttributeConfig('SourceImageSequence'),
                ],
            ),
            # TODO Referenced Image Sequence - source attributes in sequence
            _FunctionalGroupConfig(
                'PixelValueTransformationSequence',
                [
                    _AttributeConfig('RescaleSlope'),
                    _AttributeConfig('RescaleIntercept'),
                    _AttributeConfig('RescaleType'),
                ],
                custom_logic_callback=(
                    self._pixel_value_transformation_custom_logic
                )
            ),
            _FunctionalGroupConfig(
                'FrameContentSequence',
                [
                    _AttributeConfig(
                        'FrameAcquisitionDuration',
                        src_kws=['AcquisitionDuration']
                    ),
                    _AttributeConfig('TemporalPositionIndex'),
                    # TODO improve?
                    _AttributeConfig('AcquisitionNumber', default_val=0),
                    _AttributeConfig(
                        'FrameComments',
                        src_kws=['ImageComments']
                    ),
                ],
                custom_logic_callback=self._frame_content_custom_logic,
            ),
            _FunctionalGroupConfig(
                'ConversionSourceAttributesSequence',
                [
                    _AttributeConfig(
                        'ReferencedSOPClassUID',
                        src_kws=['SOPClassUID']
                    ),
                    _AttributeConfig(
                        'ReferencedSOPInstanceUID',
                        src_kws=['SOPInstanceUID']
                    ),
                ],
            ),
            _FunctionalGroupConfig(
                self._frame_type_seq_kw,
                [
                    _AttributeConfig(
                        'PixelPresentation',
                        src_kws=[],
                        default_val='MONOCHROME',
                    ),
                    _AttributeConfig(
                        'VolumetricProperties',
                        src_kws=[],
                        default_val='VOLUME',  # TODO check
                    ),
                    _AttributeConfig(
                        'VolumeBasedCalculationTechnique',
                        src_kws=[],
                        default_val='NONE',
                    ),
                ],
                custom_logic_callback=self._frame_type_custom_logic,
            ),
        ]

        if (
            self._destination.SOPClassUID in
            (
                LegacyConvertedEnhancedCTImageStorage,
                LegacyConvertedEnhancedPETImageStorage,
            )
        ):
            functional_group_configs.append(
                _FunctionalGroupConfig(
                    'IrradiationEventIdentificationSequence',
                    [_AttributeConfig('IrradiationEventUID')],
                )
            )

        for config in functional_group_configs:
            self._add_functional_group(config)

        # Miscellaneous other tasks
        self._add_stack_info_frame_content()
        self._add_image_type()
        self._add_unassigned_attributes()

    def _add_module(self, config: _ModuleConfig) -> None:
        """Add module to the destination dataset.

        Parameters
        ----------
        config: highdicom.legacy.sop._ModuleConfig
            Configuration defining behavior for this module.

        """
        if config.skip_attributes is not None:
            skip_attributes = config.skip_attributes
        else:
            skip_attributes = []

        attribute_configs: dict[str, _AttributeConfig] = {}

        if config.attribute_configs is not None:
            for attr_config in config.attribute_configs:
                attribute_configs[attr_config.dest_kw] = attr_config

        if config.auto_discover_attributes:
            for a in MODULE_ATTRIBUTE_MAP[config.module_name]:
                if len(a['path']) > 0:
                    continue

                if a['keyword'] in skip_attributes:
                    continue

                if a['keyword'] not in attribute_configs:
                    attribute_configs[a['keyword']] = _AttributeConfig(
                        a['keyword']
                    )

        ref_dataset = self._legacy_datasets[0]

        for a in attribute_configs.values():
            self._copy_attrib_if_present(
                ref_dataset,
                self._destination,
                a.dest_kw,
            )
            if (
                a.default_val is not None and
                a.dest_kw not in self._destination
            ):
                setattr(self._destination, a.dest_kw, a.default_val)

        if config.custom_logic_callback is not None:
            config.custom_logic_callback()

    def _common_enhanced_image_custom_logic(self):
        """Custom logic applicable to Enhanced CT/MR/PET Image modules."""
        self._destination.PresentationLUTShape = 'IDENTITY'

        if any(
            hasattr(src, 'LossyImageCompressionRatio')
            for src in self._legacy_datasets
        ):
            sum_compression_ratio = 0.0
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
            self._destination.LossyImageCompressionRatio = avg_ratio_str

    def _copy_to_functional_group(
        self,
        source: Dataset,
        destination: Dataset,
        sequence_name: str,
        attribute_configs: list[_AttributeConfig],
        custom_logic_callback: Callable | None = None,
    ) -> None:
        """
        Parameters
        ----------
        source: pydicom.Dataset
            Dataset to copy from.
        destination: pydicom.Dataset
            Dataset to copy to.
        sequence_name: str
            Name of the sequence where the functional group will be placed.
        attribute_configs: list[highdicom.legacy.sop._AttributeConfig]
            Configurations for each attribute to place into the sequence.
        custom_logic_callback: Callable[[pydicom.Dataset, pydicom.Dataset], None] | None
            Callback implementing custom logic to call after the other parameters
            have been copied. Takes the source and desitnation datasets as input
            parameters and has no return value.

        """  # noqa: E501
        item = Dataset()

        for a_cfg in attribute_configs:
            for src_kw in a_cfg.get_source_keywords():
                if src_kw in source:
                    self._copy_attrib_if_present(
                        source,
                        item,
                        src_kw,
                        a_cfg.dest_kw,
                        ignore_if_perframe=False,
                    )
                    break
            else:
                # Information was not found in the source dataset. Use the
                # default value if there is one
                if a_cfg.default_val is not None:
                    setattr(item, a_cfg.dest_kw, a_cfg.default_val)

        if custom_logic_callback is not None:
            custom_logic_callback(source, item)

        setattr(destination, sequence_name, [item])

    def _add_functional_group(self, config: _FunctionalGroupConfig) -> None:
        """Add a functional group to the destination dataset.

        Parameters
        ----------
        config: highdicom.legacy._FunctionalGroupConfig
            Configuration object for the functional group to be added.

        """
        # If any attribute is per-frame, the whole functional group is
        # per-frame
        any_attr_is_per_frame = False
        any_attr_exists = False
        for a_cfg in config.attribute_configs:
            if a_cfg.default_val is not None:
                any_attr_exists = True

            for kw in a_cfg.get_source_keywords():
                tag = BaseTag(cast(int, tag_for_keyword(kw)))

                if tag in self._tag_shared_dict:
                    any_attr_exists = True
                    if not self._tag_shared_dict[tag]:
                        any_attr_is_per_frame = True
                        break

            if any_attr_is_per_frame:
                # We already have all the information we need
                break

        if config.further_source_attributes is not None:
            for kw in config.further_source_attributes:
                tag = tag_for_keyword(kw)

                if tag in self._tag_shared_dict:
                    any_attr_exists = True

                    if not self._tag_shared_dict[tag]:
                        any_attr_is_per_frame = True
                        break

                if any_attr_is_per_frame:
                    # We already have all the information we need
                    break

        if any_attr_is_per_frame:
            # At least one attribute is per-frame, so need to place everything
            # in per-frame functional groups
            for src, pffg in zip(
                self._legacy_datasets,
                self._destination.PerFrameFunctionalGroupsSequence,
            ):
                self._copy_to_functional_group(
                    source=src,
                    destination=pffg,
                    sequence_name=config.sequence_name,
                    attribute_configs=config.attribute_configs,
                    custom_logic_callback=config.custom_logic_callback,
                )
        elif any_attr_exists:
            # Use the shared functional groups
            self._copy_to_functional_group(
                source=self._legacy_datasets[0],
                destination=self._destination.SharedFunctionalGroupsSequence[0],
                sequence_name=config.sequence_name,
                attribute_configs=config.attribute_configs,
                custom_logic_callback=config.custom_logic_callback,
            )

        # Else nothing to copy

    def _pixel_value_transformation_custom_logic(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Custom logic for the PixelValueTransformationSequence.

        This is needed to handle setting the RescaleType and LUTExplanation.

        Parameters
        ----------
        source: pydicom.Dataset
            Dataset to copy from.
        destination: pydicom.Dataset
            Dataset to copy to.

        """
        if (
            'RescaleSlope' in destination or
            'RescaleIntercept' in destination
        ):
            value = 'US'  # unspecified
            if source.get('Modality', '') == 'CT':
                image_type_v = (
                    [] if 'ImageType' not in source
                    else source['ImageType'].value
                )
                if not any(
                    i == 'LOCALIZER' for i in image_type_v
                ):
                    value = 'HU'
            else:
                value = 'US'

            if 'RescaleType' not in destination:
                destination.RescaleType = value
            elif destination.RescaleType != value:
                # keep the copied value as LUT explanation
                destination.LUTExplanation = destination.RescaleType
                destination.RescaleType = value

    def _frame_type_custom_logic(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Custom logic for the {CTImage/MRImage/PET} FrameTypeSequence.

        This is needed to handle the frame type attibute.

        Parameters
        ----------
        source: pydicom.Dataset
            Dataset to copy from.
        destination: pydicom.Dataset
            Dataset to copy to.

        """
        frame_type = source.ImageType
        dest_kw = 'FrameType'
        lng = len(frame_type)
        new_val = [
            'ORIGINAL' if lng == 0 else frame_type[0],
            'PRIMARY',
            'VOLUME' if lng < 3 else frame_type[2],
            'NONE',
        ]
        setattr(destination, dest_kw, new_val)

    def _frame_anatomy_custom_logic(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Custom logic for the Frame Anatomy Sequence.

        This is needed to handle mapping "BodyPartExamined" to
        "AnatomicRegionSequence".

        Parameters
        ----------
        source: pydicom.Dataset
            Dataset to copy from.
        destination: pydicom.Dataset
            Dataset to copy to.

        """
        if not hasattr(destination, 'AnatomicRegionSequence'):
            if hasattr(source, 'BodyPartExamined'):
                # Attempt to map to AnatomicRegionSequence. This mapping is
                # required by the standard but the AnatomicRegionSequence may be
                # omitted in the body part examined is not present or has a
                # non-standard value.
                mapping = _get_anatomic_region_mapping()
                if source.BodyPartExamined in mapping:
                    code = mapping[source.BodyPartExamined]

                    destination.AnatomicRegionSequence = [code]
            else:
                pass
                # TODO should remove the entire sequence here :(

        # Determine the required frame laterality. First check the modifier of
        # the primary anatomic structure and map following Part 3 Section 10.5
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_10.5.html
        modifier_mapping = {
            '7771000': 'L',
            '24028007': 'R',
            '66459002': 'U',
            '51440002': 'B',
        }

        if hasattr(destination, 'PrimaryAnatomicStructureSequence'):
            pri_struct_seq = (
                destination.PrimaryAnatomicStructureSequence[0]
            )
            if 'PrimaryAnatomicStructureModifierSequence' in pri_struct_seq:
                modifier_seq = (
                    pri_struct_seq.PrimaryAnatomicStructureModifierSequence[0]
                )
                modifier_val = modifier_seq.CodeValue

                if modifier_val in modifier_mapping:
                    destination.FrameLaterality = modifier_mapping[
                        modifier_val
                    ]

        # Now check the anatomic region modifier
        if 'FrameLaterality' not in destination:
            anatomic_region = (
                destination.AnatomicRegionSequence[0]
            )
            if 'AnatomicRegionModifierSequence' in anatomic_region:
                modifier_seq = (
                    anatomic_region.AnatomicRegionModifierSequence[0]
                )
                modifier_val = modifier_seq.CodeValue

                if modifier_val in modifier_mapping:
                    destination.FrameLaterality = modifier_mapping[
                        modifier_val
                    ]

        # Check laterality information in the original source
        if 'FrameLaterality' not in destination:
            for kw in ['FrameLaterality', 'ImageLaterality', 'Laterality']:
                if kw in source:
                    destination.FrameLaterality = getattr(source, kw)
                    break
            else:
                # No laterality information, just assume unilateral
                destination.FrameLaterality = 'U'

    def _frame_content_custom_logic(
        self,
        source: Dataset,
        destination: Dataset,
    ) -> None:
        """Custom logic for the FrameContentSequence.

        This is needed to handle setting acquisition timing related parameters.

        Parameters
        ----------
        source: pydicom.Dataset
            Dataset to copy from.
        destination: pydicom.Dataset
            Dataset to copy to.

        """
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
            acq_time_kws = [
                'AcquisitionDateTime',
                'AcquisitionDate',
                'AcquisitionTime'
            ]
            if all(
                self._tag_shared_dict.get(tag_for_keyword(t), True)
                for t in acq_time_kws
            ):
                if (
                    'TriggerTime' in source and
                    'FrameReferenceDateTime' not in source
                ):
                    trigger_time_in_millisecond = int(source.TriggerTime)
                    if trigger_time_in_millisecond > 0:
                        t_delta = timedelta(trigger_time_in_millisecond)
                        fa_dt += t_delta

            destination.FrameAcquisitionDateTime = fa_dt

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
                self._destination.PerFrameFunctionalGroupsSequence,
                position_indices
            ):
                if 'FrameContentSequence' not in pffg:
                    pffg.FrameContentSequence = [Dataset()]

                pffg.FrameContentSequence[0].StackID = stack_id
                pffg.FrameContentSequence[0].InStackPositionNumber = (
                    int(pos) + 1
                )

            sfgs = self._destination.SharedFunctionalGroupsSequence[0]
            if 'PixelMeasuresSequence' in sfgs:
                (
                    sfgs.PixelMeasuresSequence[0].SpacingBetweenSlices
                ) = format_number_as_ds(spacing)

    def _add_referenced_image_functional_group(self) -> None:
        """Add ReferencedImageSequence to the functional groups.

        This doesn't fit the pattern of the other functional groups because the
        attributes exist in a sequence within the source images, so handle
        separately here.

        """
        t = BaseTag(cast(int, tag_for_keyword('ReferencedImageSequence')))
        if t in self._tag_shared_dict:
            if self._tag_shared_dict[t]:
                # Reference are shared
                (
                    self
                    ._destination
                    .SharedFunctionalGroupsSequence[0]
                    .ReferencedImageSequence
                ) = deepcopy(self._legacy_datasets[0].ReferencedImageSequence)
            else:
                # Refernces are per-frame
                for src, pffg in zip(
                    self._legacy_datasets,
                    self._destination.PerFrameFunctionalGroupsSequence,
                ):
                    pffg.ReferencedImageSequence = deepcopy(
                        src.ReferencedImageSequence
                    )

    def _add_unassigned_attributes(self) -> None:
        """Add all unassigned attributes.

        Unassiged attributes are those in the legacy datasets that have not yet
        been used for any value in the converted instance. These are placed
        into the relevant sequence in either the shared or per-frame functional
        groups.

        This may include private attributes.

        """
        if len(self._unused_tags) == 0:
            # Nothing to do
            return

        # Shared
        if any(self._tag_shared_dict[t] for t in self._unused_tags):
            (
                self
                ._destination
                .SharedFunctionalGroupsSequence[0]
                .UnassignedSharedConvertedAttributesSequence
            ) = [Dataset()]

        # Per-frame
        if any(not self._tag_shared_dict[t] for t in self._unused_tags):
            for pffg in self._destination.PerFrameFunctionalGroupsSequence:
                pffg.UnassignedPerFrameConvertedAttributesSequence = [
                    Dataset()
                ]

        for t in self._unused_tags:
            if self._tag_shared_dict[t]:
                (
                    self
                    ._destination
                    .SharedFunctionalGroupsSequence[0]
                    .UnassignedSharedConvertedAttributesSequence
                )[0][t] = self._legacy_datasets[0][t]

            else:
                for src, pffg in zip(
                    self._legacy_datasets,
                    self._destination.PerFrameFunctionalGroupsSequence,
                ):
                    (
                        pffg
                        .UnassignedPerFrameConvertedAttributesSequence
                    )[0][t] = src[t]

    def _add_image_type(self) -> None:
        """Set the (top-level) ImageType of the new dataset.

        The ImageType summarizes individual frame types, see
        :dcm:`Sect C.8.16.1 {part03/sect_C.8.16.html#sect_C.8.16.1>`

        """
        # If the frame type is shared, no need to aggregate
        if hasattr(
            self._destination.SharedFunctionalGroupsSequence[0],
            self._frame_type_seq_kw
        ):
            self._destination.ImageType = getattr(
                self._destination.SharedFunctionalGroupsSequence[0],
                self._frame_type_seq_kw
            )[0].FrameType
            return

        frame_v1 = {
            getattr(pffg, self._frame_type_seq_kw)[0].FrameType[0]
            for pffg in self._destination.PerFrameFunctionalGroupsSequence
        }
        if len(frame_v1) > 1:
            v1 = 'MIXED'
        else:
            v1 = list(frame_v1)[0]

        # V2 cannot be MIXED - take the most common
        v2_counter = Counter(
            getattr(pffg, self._frame_type_seq_kw)[0].FrameType[1]
            for pffg in self._destination.PerFrameFunctionalGroupsSequence
        )
        v2 = v2_counter.most_common(1)[0][0]

        # V3 cannot be MIXED - take the most common
        v3_counter = Counter(
            getattr(pffg, self._frame_type_seq_kw)[0].FrameType[2]
            for pffg in self._destination.PerFrameFunctionalGroupsSequence
        )
        v3 = v3_counter.most_common(1)[0][0]

        frame_v4 = {
            getattr(pffg, self._frame_type_seq_kw)[0].FrameType[3]
            for pffg in self._destination.PerFrameFunctionalGroupsSequence
        }
        if len(frame_v4) > 1:
            v4 = 'MIXED'
        else:
            v4 = list(frame_v1)[0]

        self._destination.ImageType = [v1, v2, v3, v4]


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

        """  # noqa: E501
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('At least one legacy dataset must be provided.')

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
            sop_class_uid=LEGACY_ENHANCED_SOP_CLASS_UID_MAP[ref_ds.SOPClassUID],
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
        self._add_contributing_equipment(
            contributing_equipment,
            legacy_datasets[0],
        )

        self._copy_pixel_data(
            legacy_datasets,
            transfer_syntax_uid=transfer_syntax_uid,
            workers=workers,
            use_extended_offset_table=use_extended_offset_table,
        )

        if acquisition_datetime is not None:
            self.AcquisitionDateTime = acquisition_datetime

        runner = _LegacyConversionRunner(legacy_datasets, self)
        runner.run()

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

        """  # noqa: E501
        try:
            ref_ds = legacy_datasets[0]
        except IndexError:
            raise ValueError('At least one legacy dataset must be provided.')
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

        """  # noqa: E501
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

        """  # noqa: E501
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
