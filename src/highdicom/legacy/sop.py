"""Module for SOP Classes of Legacy Converted Enhanced Image IODs."""

from collections import Counter, defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from os import PathLike
import pkgutil
from sys import float_info
from typing import (
    Any,
    BinaryIO,
    Union,
    Callable,
    Generator,
    Sequence,
    Tuple,
)
from typing_extensions import Self

from pydicom.datadict import keyword_for_tag
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

from highdicom.base_content import ContributingEquipment
from highdicom.enum import PhotometricInterpretationValues
from highdicom.frame import encode_frame
from highdicom.image import Image, _Image
from highdicom.spatial import get_series_volume_positions

from highdicom._module_utils import (
    AttributeTypeValues,
    ModuleUsageValues,
    construct_module_tree,
    get_module_usage,
)
from highdicom.sr.coding import CodedConcept


logger = logging.getLogger(__name__)


LEGACY_ENHANCED_SOP_CLASS_UID_MAP = {
    # CT Image Storage
    "1.2.840.10008.5.1.4.1.1.2": "1.2.840.10008.5.1.4.1.1.2.2",
    # MR Image Storage
    "1.2.840.10008.5.1.4.1.1.4": "1.2.840.10008.5.1.4.1.1.4.4",
    # PET Image Storage
    "1.2.840.10008.5.1.4.1.1.128": "1.2.840.10008.5.1.4.1.1.128.1",
}


_SOP_CLASS_UID_IOD_KEY_MAP = {
    "1.2.840.10008.5.1.4.1.1.2.2": "legacy-converted-enhanced-ct-image",
    "1.2.840.10008.5.1.4.1.1.4.4": "legacy-converted-enhanced-mr-image",
    "1.2.840.10008.5.1.4.1.1.128.1": "legacy-converted-enhanced-pet-image",
}


_FARTHEST_FUTURE_DATE_TIME = DT("99991231235959")


# List of attributes required to be consistent between each frame for the
# conversion to be valid
_CONSISTENT_KEYWORDS = [
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "FrameOfReferenceUID",
    "Manufacturer",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "InstitutionalDepartmentName",
    "ManufacturerModelName",
    "DeviceSerialNumber",
    "SoftwareVersions",
    "GantryID",
    "PixelPaddingValue",
    "Modality",
    "SOPClassUID",
    "Rows",
    "Columns",
    "BitsStored",
    "BitsAllocated",
    "HighBit",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "PlanarConfiguration",
    "SamplesPerPixel",
    "ProtocolName",
    "SpecificCharacterSet",
]


def _istag_file_meta_information_group(t: BaseTag) -> bool:
    return t.group == 0x0002


def _istag_repeating_group(t: BaseTag) -> bool:
    g = t.group
    return (g >= 0x5000 and g <= 0x501E) or (g >= 0x6000 and g <= 0x601E)


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
    data_file = pkgutil.get_data("highdicom", "_standard/anatomic_regions.json")

    if data_file is None:
        raise FileNotFoundError(
            "Error loading anatomic regions JSON data file."
        )

    anatomic_regions = json.loads(data_file.decode("utf-8"))

    return {
        k: CodedConcept(value=v[1], scheme_designator=v[0], meaning=v[2])
        for k, v in anatomic_regions.items()
    }


def _istag_group_length(t: BaseTag) -> bool:
    return t.element == 0


def _transcode_frame(
    dataset: Dataset,
    transfer_syntax_uid: str,
    photometric_interpretation: PhotometricInterpretationValues,
) -> bytes:
    """Transcode single frame dataset's pixel data.

    Parameters
    ----------
    dataset: pydicom.Dataset
        Single frame (legacy) dataset whose pixel data should be transcoded.
    transfer_syntax_uid: str
        New transfer syntax.
    photometric_interpretation: highdicom.PhotometricInterpretationValues
        Photometric interpretation to use with the new transfer syntax.

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
        photometric_interpretation=photometric_interpretation,
        pixel_representation=dataset.get("PixelRepresentation", 0),
        planar_configuration=dataset.get("PlanarConfiguration"),
    )


@dataclass
class _AttributeConfig:
    """Configuration for how to copy an attribute to the converted instance.

    Parameters
    ----------
    dest_kw: str | None
        Keyword where the attribute will be placed. If explicitly set to None,
    src_kws: list[str] | None, optional
        List of source keywords to search for the value to place into the
        destination keyword. If None, the dest_kw is used as the (only) source
        keyword. If a list is provided, the keywords are searched in order
        until a value is found to place into the destination. In this case, the
        dest_kw is not used unless explicitly specified.
    default_val: Any | None
        Default value to use if attribute is not found in the source dataset.
    defer_copy: bool
        If true, the attribute will not directly be placed into the
        destination, but will still be considered for determining whether all
        required information is present. This is generally used for attributes
        that will be processed by custom logic.

    """

    dest_kw: str
    src_kws: list[str] | None = None
    default_val: Any = None
    defer_copy: bool = False

    def get_source_keywords(self) -> list[str]:
        """Get a list of all potential source keywords."""
        if self.src_kws is not None:
            # Use the src_kws, if specified
            return self.src_kws

        # Otherwise use the destination keyword as the source keyword
        return [self.dest_kw]


def default_sort_key(x: Dataset) -> Tuple[Union[int, str, UID], ...]:
    """The default sort key to sort single frames before conversion.

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
    if "SeriesNumber" in x:
        out += (x.SeriesNumber,)
    if "InstanceNumber" in x:
        out += (x.InstanceNumber,)
    if "SOPInstanceUID" in x:
        out += (x.SOPInstanceUID,)

    return out


class _LegacyConversionRunner:
    """Utility class for running a conversion process and associated state.

    This class holds state required by the conversion process, but no longer
    needed after the process has concluded.

    """

    def __init__(
        self,
        legacy_datasets: Sequence[Dataset],
        destination: Dataset,
        transfer_syntax_uid: str | None = None,
        use_extended_offset_table: bool = False,
        workers: int | Executor = 0,
        sort_key: Callable | None = None,
    ) -> None:
        """

        Parameters
        ----------
        legacy_datasets: Sequence[pydicom.Dataset]
            Legacy (single-frame) datasets to be converted.
        destination: pydicom.Dataset
            Existing Legacy Converted Enhanced dataset to copy attributes to.
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
        workers: int | concurrent.futures.Executor, optional
            Number of worker processes (or executor object) to use for frame
            compression, if compression or transcoding is needed.
        sort_key: Callable | None, optional
            A function by which the single-frame instances will be sorted to
            determine the order of frames in the newly created instance.

        """
        if sort_key is None:
            sort_key = default_sort_key

        self._legacy_datasets = sorted(list(legacy_datasets), key=sort_key)
        self._destination = destination
        self._transfer_syntax_uid = transfer_syntax_uid
        self._workers = workers
        self._use_extended_offset_table = use_extended_offset_table
        self._keyword_shared_dict: dict[str, bool] = {}
        self._unused_keywords: set[str] = set()
        self._private_tag_shared_dict: dict[BaseTag, bool]

        (self._keyword_shared_dict, self._private_tag_shared_dict) = (
            self._find_shared_and_perframe_attributes()
        )
        self._check_attribute_consistency()

        incoming_pi = (
            self._legacy_datasets[0].PhotometricInterpretation
        )

        if incoming_pi == "MONOCHROME1":
            raise ValueError(
                "Conversion of images with Photometric Interpretation "
                "'MONOCHROME1' is not supported."
            )

        # List of attributes that should never be placed into the
        # UnassignedPerFrameConvertedAttributesSequence or
        # UnassignedSharedConvertedAttributesSequence
        excluded_keywords = [
            *_CONSISTENT_KEYWORDS,
            "PixelData",
            "SeriesInstanceUID",
            "AcquisitionDate",
            "AcquisitionTime",
            "AcquisitionDateTime",
            "ImageType",
            "SeriesDescription",
        ]
        self._unused_keywords = {
            kw for kw in self._keyword_shared_dict.keys()
            if kw not in excluded_keywords
        }

        # The name of the "Enhanced X Image" module within the IOD being
        # created
        enhanced_image_module_name = {
            LegacyConvertedEnhancedPETImageStorage: "enhanced-pet-image",
            LegacyConvertedEnhancedCTImageStorage: "enhanced-ct-image",
            LegacyConvertedEnhancedMRImageStorage: "enhanced-mr-image",
        }[self._destination.SOPClassUID]

        # Attribute configs specific to the above module
        # created
        enhanced_image_module_attribute_configs = {
            LegacyConvertedEnhancedPETImageStorage: [],
            LegacyConvertedEnhancedCTImageStorage: [],
            LegacyConvertedEnhancedMRImageStorage: [
                _AttributeConfig(
                    "ResonantNucleus",
                    src_kws=["ResonantNucleus", "ImagedNucleus"],
                ),
            ],
        }[self._destination.SOPClassUID]

        self._add_module("patient")
        self._add_module("clinical-trial-subject")
        self._add_module(
            "general-study",
            skip_attributes=[
                "StudyInstanceUID",
                "RequestingService",
            ],
        )
        self._add_module(
            "patient-study",
            skip_attributes=["ReasonForVisit", "ReasonForVisitCodeSequence"],
        )
        self._add_module("clinical-trial-study")
        self._add_module(
            "general-series",
            skip_attributes=[
                "SeriesInstanceUID",
                "SeriesNumber",
                "SeriesDescription",
                "SmallestPixelValueInSeries",
                "LargestPixelValueInSeries",
                "PerformedProcedureStepEndDate",
                "PerformedProcedureStepEndTime",
            ],
        )
        self._add_module("clinical-trial-series")
        self._add_module(
            "general-equipment",
            skip_attributes=["InstitutionalDepartmentTypeCodeSequence"],
        )
        self._add_module("frame-of-reference")
        self._add_module(
            "sop-common",
            skip_attributes=[
                "SOPClassUID",
                "SOPInstanceUID",
                "InstanceNumber",
                "EncryptedAttributesSequence",
                "MACParametersSequence",
                "DigitalSignaturesSequence",
            ],
        )
        self._add_module(
            enhanced_image_module_name,
            attribute_configs=[
                _AttributeConfig("VolumetricProperties", default_val="VOLUME"),
                _AttributeConfig(
                    "VolumeBasedCalculationTechnique",
                    default_val="NONE",
                ),
                _AttributeConfig(
                    "PixelPresentation",
                    default_val=(
                        "COLOR"
                        if self._legacy_datasets[0].SamplesPerPixel == 3
                        else "MONOCHROME"
                    ),
                ),
                _AttributeConfig(
                    "PresentationLUTShape",
                    src_kws=[],
                    default_val="IDENTITY",
                ),
                _AttributeConfig(
                    "ContentQualification",
                    default_val="RESEARCH",
                ),
                # Modality-spcific configs
                *enhanced_image_module_attribute_configs,
            ],
            skip_attributes=[
                "ImageType",
                "AcquisitionDateTime",
                "ReferencedWaveformSequence",
                "ReferencedImageEvidenceSequence",
                "AcquisitionNumber",
                "InstanceNumber",
                "ImageComments",
                "BurnedInAnnotation",
                "RecognizableVisualFeatures",
                "LossyImageCompression",
                "LossyImageCompressionRatio",
                "IconImageSequence",
            ],
            custom_logic_callback=self._common_enhanced_image_custom_logic,
        )
        self._add_module(
            "image-pixel",
            skip_attributes=[
                "ColorSpace",
                "PixelDataProviderURL",
                "ExtendedOffsetTable",
                "ExtendedOffsetTableLengths",
                "SmallestImagePixelValue",
                "LargestImagePixelValue",
                "PhotometricInterpretation",
                "PixelData",
            ],
        )
        self._add_module(
            "acquisition-context",
            attribute_configs=[
                _AttributeConfig("AcquisitionContextSequence", default_val=[])
            ],
        )

        if (
            self._destination.SOPClassUID !=
            LegacyConvertedEnhancedPETImageStorage
        ):
            self._add_module("contrast-bolus")

        self._destination.SharedFunctionalGroupsSequence = [Dataset()]
        self._destination.PerFrameFunctionalGroupsSequence = [
            Dataset() for _ in range(len(self._legacy_datasets))
        ]

        self._frame_type_seq_kw = {
            LegacyConvertedEnhancedMRImageStorage: "MRImageFrameTypeSequence",
            LegacyConvertedEnhancedCTImageStorage: "CTImageFrameTypeSequence",
            LegacyConvertedEnhancedPETImageStorage: "PETFrameTypeSequence",
        }[self._destination.SOPClassUID]

        self._add_functional_group(
            "DerivationImageSequence",
            [_AttributeConfig("SourceImageSequence")],
            optional_attributes=[
                _AttributeConfig("DerivationDescription"),
                _AttributeConfig("DerivationCodeSequence"),
            ],
            required=False,
        )
        self._add_functional_group(
            self._frame_type_seq_kw,
            [
                _AttributeConfig(
                    "FrameType",
                    src_kws=["ImageType"],
                    defer_copy=True,  # handled in callback
                ),
                _AttributeConfig(
                    "PixelPresentation",
                    src_kws=[],
                    default_val="MONOCHROME",
                ),
                _AttributeConfig(
                    "VolumetricProperties",
                    src_kws=[],
                    default_val="VOLUME",  # TODO check
                ),
                _AttributeConfig(
                    "VolumeBasedCalculationTechnique",
                    src_kws=[],
                    default_val="NONE",
                ),
            ],
            custom_logic_callback=self._frame_type_custom_logic,
        )

        if self._check_frame_anatomy_condition():
            self._add_functional_group(
                "FrameAnatomySequence",
                [
                    _AttributeConfig(
                        "AnatomicRegionSequence",
                        ["AnatomicRegionSequence", "BodyPartExamined"],
                        defer_copy=True,  # handle this in callback
                    ),
                ],
                optional_attributes=[
                    _AttributeConfig("PrimaryAnatomicStructureSequence"),
                ],
                custom_logic_callback=self._frame_anatomy_custom_logic,
                required=False,
            )

        self._add_functional_group(
            "FrameContentSequence",
            [],
            optional_attributes=[
                _AttributeConfig(
                    "FrameAcquisitionNumber",
                    ["AcquisitionNumber"],
                ),
                _AttributeConfig(
                    "FrameAcquisitionDuration",
                    src_kws=["AcquisitionDuration"]
                ),
                _AttributeConfig("TemporalPositionIndex"),
                _AttributeConfig(
                    "FrameComments",
                    src_kws=["ImageComments"]
                ),
            ],
            custom_logic_callback=self._frame_content_custom_logic,
            can_be_shared=False,
            required=True,
        )
        self._add_functional_group(
            "PlanePositionSequence",
            [_AttributeConfig("ImagePositionPatient")],
        )
        self._add_functional_group(
            "PlaneOrientationSequence",
            [_AttributeConfig("ImageOrientationPatient")],
        )
        self._add_functional_group(
            "ConversionSourceAttributesSequence",
            [
                _AttributeConfig(
                    "ReferencedSOPClassUID",
                    src_kws=["SOPClassUID"]
                ),
                _AttributeConfig(
                    "ReferencedSOPInstanceUID",
                    src_kws=["SOPInstanceUID"]
                ),
            ],
            can_be_shared=False,
        )
        self._add_functional_group(
            "PixelMeasuresSequence",
            [
                _AttributeConfig(
                    "PixelSpacing",
                    src_kws=["PixelSpacing", "ImagerPixelSpacing"],
                ),
                _AttributeConfig("SliceThickness"),
            ],
        )
        self._add_functional_group(
            "FrameVOILUTSequence",
            [
                _AttributeConfig("WindowWidth"),
                _AttributeConfig("WindowCenter"),
            ],
            optional_attributes=[
                _AttributeConfig("WindowCenterWidthExplanation"),
            ],
        )
        # Pixel Value Transformation is technically optional for MR, but we
        # will populate with 0/1 rescale anyway for consistency
        self._add_functional_group(
            "PixelValueTransformationSequence",
            [
                _AttributeConfig("RescaleSlope", default_val=1),
                _AttributeConfig("RescaleIntercept", default_val=0),
            ],
            optional_attributes=[
                _AttributeConfig("RescaleType"),
            ],
            custom_logic_callback=(
                self._pixel_value_transformation_custom_logic
            ),
        )

        if self._destination.SOPClassUID in (
            LegacyConvertedEnhancedCTImageStorage,
            LegacyConvertedEnhancedPETImageStorage,
        ):
            self._add_functional_group(
                "IrradiationEventIdentificationSequence",
                [_AttributeConfig("IrradiationEventUID")],
                required=False,
            )

        # Functional grops where entire existing sequences are copied from the
        # source image
        self._copy_existing_sequence_to_functional_groups(
            "ReferencedImageSequence"
        )
        self._copy_existing_sequence_to_functional_groups(
            "RealWorldValueMappingSequence"
        )

        # Miscellaneous other tasks
        self._add_image_type()
        self._add_stack_info_frame_content()
        self._add_largest_smallest_pixel_value()
        self._add_common_instance_reference()
        self._add_unassigned_attributes()
        self._copy_pixel_data()

    def _find_shared_and_perframe_attributes(
        self,
    ) -> tuple[dict[str, bool], dict[BaseTag, bool]]:
        """Find attributes present in any dataset and whether they are shared.

        Returns
        -------
        keyword_shared_dict: dict[str, bool]
            Dictionary whose keys include all (standard) keywords present in
            any dataset. The corresponding value is a boolean that indicates
            whether that tag is consistent (in presence and value) across all
            datasets.
        private_tag_shared_dict: dict[pydicom.tag.BaseTag, bool]
            Dictionary whose keys include all (standard) keywords present in
            any dataset. The corresponding value is a boolean that indicates
            whether that tag is consistent (in presence and value) across all
            datasets.

        """
        all_keywords_present = set()
        all_private_tags_present = set()
        for ds in self._legacy_datasets:
            for t in ds.keys():
                if (
                    not _istag_file_meta_information_group(t) and not
                    _istag_repeating_group(t) and not
                    _istag_group_length(t)
                ):
                    if t.is_private:
                        if not t.is_private_creator:
                            all_private_tags_present.add(t)
                    else:
                        all_keywords_present.add(keyword_for_tag(t))

        keyword_shared_dict: dict[str, bool] = {}

        for kw in all_keywords_present:
            present_in_all = all(kw in ds for ds in self._legacy_datasets)

            same_value_in_all = False
            if present_in_all:
                ref_val = getattr(self._legacy_datasets[0], kw)
                same_value_in_all = all(
                    getattr(ds, kw) == ref_val for ds in self._legacy_datasets
                )

            keyword_shared_dict[kw] = present_in_all and same_value_in_all

        private_tag_shared_dict: dict[BaseTag, bool] = {}

        for t in all_private_tags_present:
            present_in_all = all(t in ds for ds in self._legacy_datasets)

            same_value_in_all = False
            if present_in_all:
                ref_val = self._legacy_datasets[0][t].value
                same_value_in_all = all(
                    ds[t].value == ref_val for ds in self._legacy_datasets
                )

            private_tag_shared_dict[t] = present_in_all and same_value_in_all

        return keyword_shared_dict, private_tag_shared_dict

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
                    "ImageOrientationPatient",
                    "PixelSpacing",
                    "SliceThickness",
                ]
            )
        if check_series_instance_uid:
            consistent_attributes.append("SeriesInstanceUID")

        inconsistencies = []

        for attr in consistent_attributes:
            if not self._keyword_shared_dict.get(attr, True):
                inconsistencies.append(attr)

        if len(inconsistencies) > 0:
            inconsistencies_str = ", ".join(inconsistencies)
            raise ValueError(
                "The legacy instances provided are not a valid source for a "
                "legacy conversion because the presence and/or values of the "
                "following attribute(s) is not consistent across instances: "
                f"{inconsistencies_str}."
            )

        # Additionally check transfer syntex UID
        for ds in self._legacy_datasets:
            if (
                ds.file_meta.TransferSyntaxUID !=
                self._legacy_datasets[0].file_meta.TransferSyntaxUID
            ):
                raise ValueError(
                    "Legacy instances have inconsistent transfer syntaxes."
                )

        # Further check that no duplicates were passed
        sop_instance_uids = {ds.SOPInstanceUID for ds in self._legacy_datasets}
        if len(sop_instance_uids) < len(self._legacy_datasets):
            raise ValueError(
                "Duplicate SOP Instance UIDs found in input datasets."
            )

    def _mark_keyword_used(self, kw: str) -> None:
        """Record that a keyword in the source files has been used.

        Parameters
        ----------
        kw: str
            Attribute keyword to mark as used.

        """
        if kw in self._unused_keywords:
            self._unused_keywords.remove(kw)

    def _add_module(
        self,
        module_name: str,
        attribute_configs: list[_AttributeConfig] | None = None,
        skip_attributes: list[str] | None = None,
        custom_logic_callback: Callable[[], None] | None = None,
    ) -> None:
        """Add module to the destination dataset.

        Parameters
        ----------
        module_name: str
            Name of the module (in highdicom's module list). This format uses
            lower case and hyphens to separate words, e.g. "general-series".
        attribute_configs: list[highdicom.legacy.sop._AttributeConfig] | None
            Attribute-level configurations. Configurations are only required for
            attributes that deviate from the default behavior.
        skip_attributes: list[str] | None
            List of attributes to skip.
        custom_logic_callback: Callable[[], None] | None
            Callback implementing custom logic to call after the other
            parameters have been copied. Takes no parameters and has no return
            value.

        """
        module_usage = get_module_usage(
            module_name,
            self._destination.SOPClassUID
        )
        if skip_attributes is None:
            skip_attributes = []
        if attribute_configs is None:
            attribute_configs = []

        ref_dataset = self._legacy_datasets[0]
        module_tree = construct_module_tree(module_name)

        def iter_attribute_configs(
            only_required: bool = False,
        ) -> Generator[
            tuple[
                str,
                AttributeTypeValues,
                str | None,
                bool,
                Any,
                bool,
            ],
            None,
            None,
        ]:
            """Loop over attributes in this module.

            Parameters
            ----------
            only_required: bool
                Only yield required (type 1, excluding type 1C) attributes. If
                False, return all attributes regardless of usage type.

            Yields
            ------
            dest_kw: str
                Destination keyword.
            usage_type: AttributeTypeValues
                Usage type (required, conditional, optional etc).
            src_kw: str | None
                Keyword in the source legacy files in which data is found, if
                ``None``, no relevant attribute was found in the legacy
                datasets (but there may still be a default value). str | None
            is_shared: bool
                Whether the attribute is shared by all legacy datasets (True)
                or differs between legacy datasets (False).
            value: Any
                The value to place into the destination. May come from a source
                dataset or from a configured default value.
            defer_copy: bool
                Whether to defer the copying of this attribute to a callback.

            """
            for dest_kw, info in module_tree["attributes"].items():
                if dest_kw in skip_attributes:
                    continue

                if (
                    only_required and
                    info["type"] != AttributeTypeValues.REQUIRED
                ):
                    continue

                # Check for a provided configuration
                for a_cfg in attribute_configs:
                    if a_cfg.dest_kw == dest_kw:
                        default_val = a_cfg.default_val
                        src_kws = a_cfg.get_source_keywords()
                        defer_copy = a_cfg.defer_copy
                        break
                else:
                    # Default behavior if no configuration is found
                    default_val = None
                    src_kws = [dest_kw]
                    defer_copy = False

                for src_kw in src_kws:
                    if src_kw in self._keyword_shared_dict:
                        if self._keyword_shared_dict[src_kw]:
                            yield (
                                dest_kw,
                                info["type"],
                                src_kw,
                                True,
                                getattr(ref_dataset, src_kw),
                                defer_copy,
                            )
                        else:
                            yield (
                                dest_kw,
                                info["type"],
                                src_kw,
                                False,
                                None,
                                defer_copy,
                            )

                        break
                else:
                    # No value found, use default
                    yield (
                        dest_kw,
                        info["type"],
                        None,
                        True,
                        default_val,
                        defer_copy,
                    )

        # First determine whether the module should be included
        if module_usage != ModuleUsageValues.MANDATORY:
            for dest_kw, _, _, _, val, _ in iter_attribute_configs(True):
                if val is None:
                    # We have no value for one of the required attributes, so
                    # we should skip the entire module entirely
                    logger.debug(
                        f"Skipping optional module {module_name} "
                        f"because no value for required attribute {dest_kw} "
                        "can be found."
                    )
                    return

        for (
            dest_kw,
            usage_type,
            src_kw,
            is_shared,
            val,
            defer_copy,
        ) in iter_attribute_configs():
            if val is not None:
                if not defer_copy:
                    setattr(self._destination, dest_kw, deepcopy(val))
                    if src_kw is not None:
                        self._mark_keyword_used(src_kw)
            elif not is_shared:
                match usage_type:
                    case AttributeTypeValues.REQUIRED:
                        # Most of these cases will have been caught earlier
                        # by the initial consistency checks
                        raise AttributeError(
                            "Unable to determine value for "
                            f"required attribute '{dest_kw}' because "
                            "the value is inconsistent between the "
                            "legacy files."
                        )
                    case AttributeTypeValues.REQUIRED_EMPTY_IF_UNKNOWN:
                        # This shouldn't really happen, but probably
                        # failing if it did would be overly strict. Issue
                        # a warning and leave the attribute empty
                        logger.warning(
                            f"Setting type 2 attribute '{dest_kw}' to "
                            "empty because value is inconsistent "
                            "between source images."
                        )
                        setattr(self._destination, dest_kw, None)
            else:
                # No value found
                match usage_type:
                    case AttributeTypeValues.REQUIRED:
                        raise AttributeError(
                            "Unable to determine value for required "
                            f"attribute '{dest_kw}' because the required "
                            "information is not present in the legacy files."
                        )
                    case AttributeTypeValues.REQUIRED_EMPTY_IF_UNKNOWN:
                        setattr(self._destination, dest_kw, None)

        if custom_logic_callback is not None:
            custom_logic_callback()

    def _copy_to_functional_group(
        self,
        source: Dataset,
        destination: Dataset,
        sequence_name: str,
        attribute_configs: list[_AttributeConfig],
        custom_logic_callback: Callable[[Dataset, Dataset], None] | None = None,
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
            have been copied. Takes the source and destination datasets as input
            parameters and has no return value.

        """  # noqa: E501
        item = Dataset()

        for a_cfg in attribute_configs:
            if a_cfg.defer_copy:
                continue

            for src_kw in a_cfg.get_source_keywords():
                if src_kw in source:
                    setattr(
                        item,
                        a_cfg.dest_kw,
                        getattr(source, src_kw),
                    )
                    self._mark_keyword_used(src_kw)
                    break
            else:
                # Information was not found in the source dataset. Use the
                # default value if there is one
                if a_cfg.default_val is not None:
                    setattr(item, a_cfg.dest_kw, a_cfg.default_val)

        if custom_logic_callback is not None:
            custom_logic_callback(source, item)

        setattr(destination, sequence_name, [item])

    def _add_functional_group(
        self,
        sequence_name: str,
        required_attributes: list[_AttributeConfig],
        optional_attributes: list[_AttributeConfig] | None = None,
        custom_logic_callback: Callable[[Dataset, Dataset], None] | None = None,
        required: bool = True,
        can_be_shared: bool = True,
    ) -> None:
        """Add a functional group to the destination dataset.

        Parameters
        ----------
        sequence_name: str
            Name of the sequence where the functional group will be placed.
        required_attributes: list[highdicom.legacy.sop._AttributeConfig]
            Configurations for each attribute required in the output sequence.
        optional_attributes: list[highdicom.legacy.sop._AttributeConfig]
            Configurations for optional attributes that will be placed into the
            output sequence if possible.
        further_source_attributes: list[str]
            Keywords for further attributes that are not copied to the destination
            but should be checked for presence and consistency in the source to
            determine whether this functional group should be included and whether
            it should be per-frame or shared. Typically, these will be attributes
            that are processed in the custom_logic_callback rather than simply
            copied over to the destination.
        custom_logic_callback: Callable[[pydicom.Dataset, pydicom.Dataset], None] | None
            Callback implementing custom logic to call after the other parameters
            have been copied. Takes the source and destination datasets as input
            parameters and has no return value.
        required: bool
            Whether this functional group is required. If so, an error will be
            raised if it cannot be populated.
        can_be_shared: bool
            Whether it is permissible to place this in the shared functional
            groups.

        """  # noqa: E501
        if optional_attributes is None:
            optional_attributes = []

        # If any attribute is per-frame, the whole functional group is
        # per-frame
        any_attr_is_per_frame = False
        any_attr_exists = False
        all_required_attrs_exist = True
        for configs, required_attr in zip(
            [required_attributes, optional_attributes], [True, False]
        ):
            for a_cfg in configs:
                for kw in a_cfg.get_source_keywords():
                    if kw in self._keyword_shared_dict:
                        any_attr_exists = True
                        if not self._keyword_shared_dict[kw]:
                            any_attr_is_per_frame = True
                        break
                else:
                    if a_cfg.default_val is not None:
                        any_attr_exists = True
                    elif required_attr:
                        all_required_attrs_exist = False
                        if required:
                            raise AttributeError(
                                "Cannot determine value for required attribute "
                                f"'{a_cfg.dest_kw}' in the '{sequence_name}'."
                            )

        if not required and not all_required_attrs_exist:
            # Do not include this functional group. Nothing more to do
            return

        attribute_configs = required_attributes + optional_attributes

        if any_attr_is_per_frame or not can_be_shared:
            # At least one attribute is per-frame, so need to place everything
            # in per-frame functional groups
            for src, pffg in zip(
                self._legacy_datasets,
                self._destination.PerFrameFunctionalGroupsSequence,
            ):
                self._copy_to_functional_group(
                    source=src,
                    destination=pffg,
                    sequence_name=sequence_name,
                    attribute_configs=attribute_configs,
                    custom_logic_callback=custom_logic_callback,
                )
        elif any_attr_exists or required:
            # Use the shared functional groups
            self._copy_to_functional_group(
                source=self._legacy_datasets[0],
                destination=self._destination.SharedFunctionalGroupsSequence[0],
                sequence_name=sequence_name,
                attribute_configs=attribute_configs,
                custom_logic_callback=custom_logic_callback,
            )

        # Else nothing to copy

    def _common_enhanced_image_custom_logic(self):
        """Custom logic applicable to Enhanced CT/MR/PET Image modules."""
        # These attributes should be YES if any source has YES, NO if all
        # sources have NO, and skipped otherwise
        for kw in [
            "BurnedInAnnotation",
            "RecognizableVisualFeatures",
            "LossyImageCompression",
        ]:
            if any(
                hasattr(src, kw) and getattr(src, kw) == "YES"
                for src in self._legacy_datasets
            ):
                setattr(self._destination, kw, "YES")
                self._mark_keyword_used(kw)
            elif all(
                hasattr(src, kw) and getattr(src, kw) == "NO"
                for src in self._legacy_datasets
            ):
                setattr(self._destination, kw, "NO")
                self._mark_keyword_used(kw)

        if any(
            hasattr(src, "LossyImageCompressionRatio")
            for src in self._legacy_datasets
        ):
            sum_compression_ratio = 0.0
            for fr_ds in self._legacy_datasets:
                if "LossyImageCompressionRatio" in fr_ds:
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
            avg_ratio_str = "{:.6f}".format(avg_compression_ratio)
            self._destination.LossyImageCompressionRatio = avg_ratio_str
            self._mark_keyword_used("LossyImageCompressionRatio")

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
        value = "US"  # unspecified
        if source.get("Modality", "") == "CT":
            image_type_v = (
                [] if "ImageType" not in source else source["ImageType"].value
            )
            if not any(i == "LOCALIZER" for i in image_type_v):
                value = "HU"
        else:
            value = "US"

        if "RescaleType" not in destination:
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
        dest_kw = "FrameType"
        lng = len(frame_type)
        new_val = [
            "ORIGINAL" if lng == 0 else frame_type[0],
            "PRIMARY",
            "VOLUME" if lng < 3 else frame_type[2],
            "NONE",
        ]
        setattr(destination, dest_kw, new_val)

    def _check_frame_anatomy_condition(self) -> bool:
        """Determine whether to include a frame anatomy sequence.

        The frame anatomy functional group is conditionally required, but the
        condition requires custom logic to express. The condition is given as:

        "Required if Body Part Examined (0018,0015) is present and contains a
        Value defined in Annex L “Correspondence of Anatomic Region Codes and
        Body Part Examined Defined Terms” in PS3.16, or Anatomic Region
        Sequence (0008,2218) was present in any of the Classic Images that were
        converted."

        See :dcm:`Section A.70.4.html <part03/sect_A.70.4.html>` for Legacy
        Converted CT and similarly for other IODs.

        """
        mapping = _get_anatomic_region_mapping()

        for ds in self._legacy_datasets:
            if "AnatomicRegionSequence" in ds:
                continue

            if "BodyPartExamined" in ds and ds.BodyPartExamined in mapping:
                continue

            return False

        return True

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
        if hasattr(source, "AnatomicRegionSequence"):
            destination.AnatomicRegionSequence = deepcopy(
                source.AnatomicRegionSequence
            )
        else:
            # Due to earlier checks, should have BodyPartExamined if we get
            # here
            # Attempt to map to AnatomicRegionSequence. This mapping is
            # required by the standard but the AnatomicRegionSequence may be
            # omitted in the body part examined is not present or has a
            # non-standard value.
            mapping = _get_anatomic_region_mapping()
            if source.BodyPartExamined in mapping:
                code = mapping[source.BodyPartExamined]

                destination.AnatomicRegionSequence = [code]

        # Determine the required frame laterality. First check the modifier of
        # the primary anatomic structure and map following Part 3 Section 10.5
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_10.5.html
        modifier_mapping = {
            "7771000": "L",
            "24028007": "R",
            "66459002": "U",
            "51440002": "B",
        }

        if hasattr(destination, "PrimaryAnatomicStructureSequence"):
            pri_struct_seq = destination.PrimaryAnatomicStructureSequence[0]
            if "PrimaryAnatomicStructureModifierSequence" in pri_struct_seq:
                modifier_seq = (
                    pri_struct_seq
                    .PrimaryAnatomicStructureModifierSequence[0]
                )
                modifier_val = modifier_seq.CodeValue

                if modifier_val in modifier_mapping:
                    destination.FrameLaterality = modifier_mapping[modifier_val]

        # Now check the anatomic region modifier
        if "FrameLaterality" not in destination:
            anatomic_region = destination.AnatomicRegionSequence[0]
            if "AnatomicRegionModifierSequence" in anatomic_region:
                modifier_seq = anatomic_region.AnatomicRegionModifierSequence[0]
                modifier_val = modifier_seq.CodeValue

                if modifier_val in modifier_mapping:
                    destination.FrameLaterality = modifier_mapping[modifier_val]

        # Check laterality information in the original source
        if "FrameLaterality" not in destination:
            for kw in ["FrameLaterality", "ImageLaterality", "Laterality"]:
                if kw in source:
                    destination.FrameLaterality = getattr(source, kw)
                    break
            else:
                # No laterality information, just assume unilateral
                destination.FrameLaterality = "U"

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
        if "AcquisitionDateTime" in source:
            fa_dt = DT(source.AcquisitionDateTime)
        elif "AcquisitionDate" in source and "AcquisitionTime" in source:
            fa_dt = DT(
                datetime.combine(
                    DA(source.AcquisitionDate),
                    TM(source.AcquisitionTime)
                )
            )

        if fa_dt is not None:
            acq_time_kws = [
                "AcquisitionDateTime",
                "AcquisitionDate",
                "AcquisitionTime",
            ]
            if all(
                self._keyword_shared_dict.get(kw, True) for kw in acq_time_kws
            ):
                if (
                    "TriggerTime" in source and
                    "FrameReferenceDateTime" not in source
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
            stack_id = "1"

            for pffg, pos in zip(
                self._destination.PerFrameFunctionalGroupsSequence,
                position_indices
            ):
                if "FrameContentSequence" not in pffg:
                    pffg.FrameContentSequence = [Dataset()]

                pffg.FrameContentSequence[0].StackID = str(stack_id)
                pffg.FrameContentSequence[0].InStackPositionNumber = (
                    int(pos) + 1
                )

            sfgs = self._destination.SharedFunctionalGroupsSequence[0]
            if "PixelMeasuresSequence" in sfgs:
                (
                    sfgs.PixelMeasuresSequence[0].SpacingBetweenSlices
                ) = format_number_as_ds(spacing)

    def _copy_existing_sequence_to_functional_groups(
        self,
        keyword: str,
    ) -> None:
        """Add an existing sequence to the functional groups.

        These sequences don't fit the pattern of the other functional groups
        because the attributes already exist in a sequence within the source
        images and the whole sequence should be copied to the functional
        groups.

        Parameters
        ----------
        keyword: str
            Keyword of the sequence to copy.

        """
        if keyword in self._keyword_shared_dict:
            if self._keyword_shared_dict[keyword]:
                # Reference are shared
                setattr(
                    self._destination.SharedFunctionalGroupsSequence[0],
                    keyword,
                    deepcopy(getattr(self._legacy_datasets[0], keyword)),
                )
            else:
                # Refernces are per-frame
                for src, pffg in zip(
                    self._legacy_datasets,
                    self._destination.PerFrameFunctionalGroupsSequence,
                ):
                    setattr(pffg, keyword, deepcopy(getattr(src, keyword)))

            self._mark_keyword_used(keyword)

    def _add_unassigned_attributes(self) -> None:
        """Add all unassigned attributes.

        Unassiged attributes are those in the legacy datasets that have not yet
        been used for any value in the converted instance. These are placed
        into the relevant sequence in either the shared or per-frame functional
        groups.

        This includes private attributes.

        """
        if (
            len(self._unused_keywords) == 0 and
            len(self._private_tag_shared_dict) == 0
        ):
            # Nothing to do
            return

        # Shared
        if any(
            self._keyword_shared_dict[kw] for kw in self._unused_keywords
        ) or any(
            self._private_tag_shared_dict.values()
        ):
            (
                self._destination.SharedFunctionalGroupsSequence[
                    0
                ].UnassignedSharedConvertedAttributesSequence
            ) = [Dataset()]

        # Per-frame
        if any(
            not self._keyword_shared_dict[kw] for kw in self._unused_keywords
        ) or any(
            not shared for shared in self._private_tag_shared_dict.values()
        ):
            for pffg in self._destination.PerFrameFunctionalGroupsSequence:
                pffg.UnassignedPerFrameConvertedAttributesSequence = [Dataset()]

        for kw in self._unused_keywords:
            if self._keyword_shared_dict[kw]:
                setattr(
                    self._destination.SharedFunctionalGroupsSequence[
                        0
                    ].UnassignedSharedConvertedAttributesSequence[0],
                    kw,
                    deepcopy(getattr(self._legacy_datasets[0], kw)),
                )

            else:
                for src, pffg in zip(
                    self._legacy_datasets,
                    self._destination.PerFrameFunctionalGroupsSequence,
                ):
                    seq = pffg.UnassignedPerFrameConvertedAttributesSequence[0]
                    if kw in src:
                        setattr(
                            seq,
                            kw,
                            deepcopy(getattr(src, kw)),
                        )

        # Private tags (these will all be unused at this stage)
        for t, shared in self._private_tag_shared_dict.items():
            if shared:
                self._copy_private_attribute(
                    t,
                    self._legacy_datasets[0],
                    (
                        self._destination.SharedFunctionalGroupsSequence[
                            0
                        ].UnassignedSharedConvertedAttributesSequence[0]
                    ),
                )

            else:
                for src, pffg in zip(
                    self._legacy_datasets,
                    self._destination.PerFrameFunctionalGroupsSequence,
                ):
                    seq = pffg.UnassignedPerFrameConvertedAttributesSequence[0]
                    if t in src:
                        self._copy_private_attribute(t, src, seq)

    @staticmethod
    def _copy_private_attribute(
        tag: BaseTag, source: Dataset, destination: Dataset
    ) -> None:
        """Copy a private attribute from source to destination.

        In addition to copying the tag and value, ensures that the private
        creator information is also copied.

        Parameters
        ----------
        tag: pydicom.tag.BaseTag
            Tag for the attribute to be copied.
        source: pydicom.Dataset
            Dataset to copy from.
        destination: pydicom.Dataset
            Dataset to copy to.

        """
        # Ensure the private creator information is also copied
        creator_tag = tag.private_creator

        if creator_tag not in source:
            # Skip private tags without creator information
            return

        if creator_tag not in destination:
            destination[creator_tag] = deepcopy(source[creator_tag])

        # Copy the attribute itself
        destination[tag] = deepcopy(source[tag])

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
                self._frame_type_seq_kw,
            )[0].FrameType
            return

        frame_v1 = {
            getattr(pffg, self._frame_type_seq_kw)[0].FrameType[0]
            for pffg in self._destination.PerFrameFunctionalGroupsSequence
        }
        if len(frame_v1) > 1:
            v1 = "MIXED"
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
            v4 = "MIXED"
        else:
            v4 = list(frame_v4)[0]

        self._destination.ImageType = [v1, v2, v3, v4]
        print([v1, v2, v3, v4])

    def _add_largest_smallest_pixel_value(self) -> None:
        """Adds the attributes for largest and smallest pixel value.

        These must be aggregated using min/max across the series.

        """
        kw = "LargestImagePixelValue"
        if kw in self._keyword_shared_dict:
            lval = float_info.min
            for frame in self._legacy_datasets:
                if kw in frame:
                    lval = max(lval, getattr(frame, kw))

            if lval > float_info.min:
                setattr(self._destination, kw, int(lval))
                self._mark_keyword_used(kw)

        kw = "SmallestImagePixelValue"
        if kw in self._keyword_shared_dict:
            lval = float_info.max
            for frame in self._legacy_datasets:
                if kw in frame:
                    lval = min(lval, getattr(frame, kw))

            if lval < float_info.max:
                setattr(self._destination, kw, int(lval))
                self._mark_keyword_used(kw)

    def _add_common_instance_reference(self) -> None:
        """Add the Common Instance Reference Module."""
        sops_per_series = defaultdict(set)
        sops_per_study = defaultdict(lambda: defaultdict(set))

        studies_kw = (
            "StudiesContainingOtherReferencedInstancesSequence"
        )

        # Gather deduplicated information on instances to include
        for ds in self._legacy_datasets:
            # The legacy datasets themselves should be added to the
            # referenced instance sequence, since they are listed as
            # the conversion source in the functional groups
            sops_per_series[ds.SeriesInstanceUID].add(
                (ds.SOPClassUID, ds.SOPInstanceUID)
            )

            # Any referenced series items in the legacy datasets should
            # also be added, but need to be deduplicated across the legacy
            # intances
            for series_item in ds.get("ReferencedSeriesSequence", []):
                for instance_item in series_item.get(
                    "ReferencedInstanceSequence", []
                ):
                    sops_per_series[series_item.SeriesInstanceUID].add(
                        (
                            instance_item.ReferencedSOPClassUID,
                            instance_item.ReferencedSOPInstanceUID
                        )
                    )

            # Referenced instances fromother studies in the legacy datasets
            # should also be added, but need to be deduplicated across the
            # legacy intances
            for study_item in ds.get(studies_kw, []):
                for series_item in study_item.get(
                    "ReferencedSeriesSequence", []
                ):
                    for instance_item in series_item.get(
                        "ReferencedInstanceSequence", []
                    ):
                        sops_per_study[
                            study_item.StudyInstanceUID
                        ][
                            series_item.SeriesInstanceUID
                        ].add(
                            (
                                instance_item.ReferencedSOPClassUID,
                                instance_item.ReferencedSOPInstanceUID
                            )
                        )

        # Construct the sequences
        series_items = []
        for series_uid, instance_uids in sops_per_series.items():
            series_item = Dataset()
            series_item.SeriesInstanceUID = series_uid

            instance_items = []
            for sop_class_uid, sop_instance_uid in instance_uids:
                instance_item = Dataset()
                instance_item.ReferencedSOPInstanceUID = sop_instance_uid
                instance_item.ReferencedSOPClassUID = sop_class_uid
                instance_items.append(instance_item)

            series_item.ReferencedInstanceSequence = instance_items
            series_items.append(series_item)

        if len(series_items) > 0:
            self._destination.ReferencedSeriesSequence = series_items

        study_items = []
        for study_uid, sop_per_study_series in sops_per_study.items():
            study_item = Dataset()
            study_item.StudyInstanceUID = study_uid

            series_items = []
            for series_uid, instance_uids in sop_per_study_series.items():
                series_item = Dataset()
                series_item.SeriesInstanceUID = series_uid

                instance_items = []
                for sop_class_uid, sop_instance_uid in instance_uids:
                    instance_item = Dataset()
                    instance_item.ReferencedSOPInstanceUID = sop_instance_uid
                    instance_item.ReferencedSOPClassUID = sop_class_uid
                    instance_items.append(instance_item)

                series_item.ReferencedInstanceSequence = instance_items
                series_items.append(series_item)

            if len(series_items) > 0:
                study_item.ReferencedSeriesSequence = series_items

            study_items.append(study_item)

        if len(study_items) > 0:
            setattr(self._destination, studies_kw, study_items)

    def _copy_pixel_data(self) -> None:
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
        if self._transfer_syntax_uid is not None and (
            self._transfer_syntax_uid not in allowed_transfer_syntaxes
        ):
            raise ValueError(
                f"Transfer syntax '{self._transfer_syntax_uid}' not recognized "
                "or not supported."
            )

        self._destination.NumberOfFrames = len(self._legacy_datasets)

        src_tx_uid = self._legacy_datasets[0].file_meta.TransferSyntaxUID
        if self._transfer_syntax_uid is None:
            dst_tx_uid = src_tx_uid
            outgoing_pi = PhotometricInterpretationValues(
                self._legacy_datasets[0].PhotometricInterpretation
            )
        else:
            dst_tx_uid = UID(self._transfer_syntax_uid)

            # Deduce the PhotometricInterpretation for the converted image
            samples_per_pixel = self._legacy_datasets[0].SamplesPerPixel
            if samples_per_pixel == 1:
                # Monochrome 1 is disallowed earlier
                outgoing_pi = PhotometricInterpretationValues.MONOCHROME2
            else:
                # Photometric interpretation depends on transfer syntax
                outgoing_pi = {
                    ImplicitVRLittleEndian: PhotometricInterpretationValues.RGB,
                    ExplicitVRLittleEndian: PhotometricInterpretationValues.RGB,
                    JPEGLSLossless: PhotometricInterpretationValues.RGB,
                    RLELossless: PhotometricInterpretationValues.RGB,
                    JPEG2000Lossless: PhotometricInterpretationValues.YBR_RCT,
                    JPEG2000: PhotometricInterpretationValues.YBR_ICT,
                    JPEGBaseline8Bit:
                        PhotometricInterpretationValues.YBR_FULL_422,
                }[dst_tx_uid]

        self._destination.PhotometricInterpretation = outgoing_pi.value

        if not isinstance(self._workers, (int, Executor)):
            raise TypeError(
                'Argument "workers" must be of type int or '
                "concurrent.futures.Executor (or a derived class)."
            )
        using_multiprocessing = (
            isinstance(self._workers, Executor) or self._workers != 0
        )

        frames: list[bytes]

        if (dst_tx_uid != src_tx_uid) and (
            src_tx_uid.is_encapsulated or dst_tx_uid.is_encapsulated
        ):
            if using_multiprocessing:
                # Use the existing executor or create one
                if isinstance(self._workers, Executor):
                    process_pool = self._workers
                else:
                    # If workers is negative, pass None to use all processors
                    process_pool = ProcessPoolExecutor(
                        self._workers if self._workers > 0 else None
                    )

                futures = [
                    process_pool.submit(
                        _transcode_frame,
                        dataset=ds,
                        transfer_syntax_uid=dst_tx_uid,
                        photometric_interpretation=outgoing_pi,
                    )
                    for ds in self._legacy_datasets
                ]

                frames = [fut.result() for fut in futures]

                if process_pool is not self._workers:
                    process_pool.shutdown()

            else:
                frames = [
                    _transcode_frame(ds, dst_tx_uid, outgoing_pi)
                    for ds in self._legacy_datasets
                ]

        else:
            # No transcoding is required, just concatenate frames
            frames = [ds.PixelData for ds in self._legacy_datasets]

        if dst_tx_uid.is_encapsulated:
            if self._use_extended_offset_table:
                (
                    self._destination.PixelData,
                    self._destination.ExtendedOffsetTable,
                    self._destination.ExtendedOffsetTableLengths,
                ) = encapsulate_extended(frames)
            else:
                self._destination.PixelData = encapsulate(frames)
        else:
            self._destination.PixelData = b"".join(frames)


class _CommonLegacyConvertedEnhancedImage(Image):
    """SOP class for common Legacy Converted Enhanced instances."""

    # Override these on the sub-classes
    _MODALITY_ATTRIBUTE = ""
    _MODALITY_NAME = ""
    _LEGACY_SOP_CLASS_UID = ""
    _ENHANCED_SOP_CLASS_UID = ""

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
        contributing_equipment: Sequence[ContributingEquipment] | None = None,
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
            A function by which the single-frame instances will be sorted to
            determine the order of frames in the newly created instance.
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
            raise ValueError("At least one legacy dataset must be provided.")

        if ref_ds.Modality != self._MODALITY_ATTRIBUTE:
            raise ValueError(
                "Wrong modality for conversion of legacy "
                f"{self._MODALITY_NAME} images."
            )
        if ref_ds.SOPClassUID != self._LEGACY_SOP_CLASS_UID:
            raise ValueError(
                f"Wrong SOP class for conversion of legacy "
                f"{self._MODALITY_NAME} images."
            )

        content_date, content_time, acquisition_datetime = self._get_datetimes(
            legacy_datasets
        )

        super(_Image, self).__init__(
            study_instance_uid=ref_ds.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=LEGACY_ENHANCED_SOP_CLASS_UID_MAP[
                self._LEGACY_SOP_CLASS_UID
            ],
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
                ref_ds,
                "ReferringPhysicianName",
                None
            ),
            content_date=content_date,
            content_time=content_time,
            **kwargs,
        )
        self._add_contributing_equipment(
            contributing_equipment,
            legacy_datasets[0],
        )

        if acquisition_datetime is not None:
            self.AcquisitionDateTime = acquisition_datetime

        _LegacyConversionRunner(
            legacy_datasets,
            destination=self,
            workers=workers,
            transfer_syntax_uid=transfer_syntax_uid,
            use_extended_offset_table=use_extended_offset_table,
            sort_key=sort_key,
        )

        self._build_luts()

    def _get_datetimes(
        self,
        legacy_datasets: Sequence[Dataset],
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

            if "AcquisitionDateTime" in src:
                frame_content_datetime = DT(src.AcquisitionDateTime)
                frame_acquisition_datetime = frame_content_datetime
            elif "AcquisitionDate" in src and "AcquisitionTime" in src:
                frame_content_datetime = DT.combine(
                    DA(src.AcquisitionDate), TM(src.AcquisitionTime)
                )
                frame_acquisition_datetime = frame_content_datetime
            elif "SeriesDate" in src and "SeriesTime" in src:
                frame_content_datetime = DT.combine(
                    DA(src.SeriesDate), TM(src.SeriesTime)
                )
            elif "StudyDate" in src and "StudyTime" in src:
                if src.StudyDate is not None and src.StudyTime is not None:
                    frame_content_datetime = DT.combine(
                        DA(src.StudyDate), TM(src.StudyTime)
                    )

            if frame_content_datetime is not None:
                earliest_content_date_time = min(
                    frame_content_datetime, earliest_content_date_time
                )
            if frame_acquisition_datetime is not None:
                earliest_acquisition_date_time = min(
                    frame_acquisition_datetime, earliest_acquisition_date_time
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

    @classmethod
    def from_dataset(cls, dataset: Dataset, copy: bool = False) -> Self:
        """Create instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a LegacyConvertedEnhanced image.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        Self:
            Representation of the supplied dataset as a highdicom Legacy
            Converted Enhanced image.

        """
        if dataset.SOPClassUID != cls._ENHANCED_SOP_CLASS_UID:
            raise ValueError(
                "Dataset is not a Legacy Converted Enhanced "
                f"{cls._MODALITY_NAME} image."
            )
        return super().from_dataset(dataset, copy=copy)


class LegacyConvertedEnhancedCTImage(_CommonLegacyConvertedEnhancedImage):
    """SOP class for Legacy Converted Enhanced CT Image instances."""

    _MODALITY_ATTRIBUTE = "CT"
    _MODALITY_NAME = "CT"
    _LEGACY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.2"
    _ENHANCED_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.2.2"


class LegacyConvertedEnhancedPETImage(_CommonLegacyConvertedEnhancedImage):
    """SOP class for Legacy Converted Enhanced PET Image instances."""

    _MODALITY_ATTRIBUTE = "PT"
    _MODALITY_NAME = "PET"
    _LEGACY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.128"
    _ENHANCED_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.128.1"


class LegacyConvertedEnhancedMRImage(_CommonLegacyConvertedEnhancedImage):
    """SOP class for Legacy Converted Enhanced MR Image instances."""

    _MODALITY_ATTRIBUTE = "MR"
    _MODALITY_NAME = "MR"
    _LEGACY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.4"
    _ENHANCED_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.4.4"


def lcectimread(
    fp: str | bytes | PathLike | BinaryIO,
    lazy_frame_retrieval: bool = False
) -> LegacyConvertedEnhancedCTImage:
    """Read an LegacyConvertedEnhancedCTImage from DICOM file.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing an
        Legacy Converted Enhanced CT Image.
    lazy_frame_retrieval: bool
        If True, the returned image will retrieve frames from the file as
        requested, rather than loading in the entire object to memory
        initially. This may be a good idea if file reading is slow and you are
        likely to need only a subset of the frames in the image.

    Returns
    -------
    highdicom.legacy.LegacyConvertedEnhancedCTImage:
        Image read from the file.

    """
    return LegacyConvertedEnhancedCTImage.from_file(
        fp,
        lazy_frame_retrieval=lazy_frame_retrieval
    )


def lcemrimread(
    fp: str | bytes | PathLike | BinaryIO,
    lazy_frame_retrieval: bool = False
) -> LegacyConvertedEnhancedMRImage:
    """Read an LegacyConvertedEnhancedMRImage from DICOM file.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing an
        Legacy Converted Enhanced MR Image.
    lazy_frame_retrieval: bool
        If True, the returned image will retrieve frames from the file as
        requested, rather than loading in the entire object to memory
        initially. This may be a good idea if file reading is slow and you are
        likely to need only a subset of the frames in the image.

    Returns
    -------
    highdicom.legacy.LegacyConvertedEnhancedMRImage:
        Image read from the file.

    """
    return LegacyConvertedEnhancedMRImage.from_file(
        fp,
        lazy_frame_retrieval=lazy_frame_retrieval
    )


def lcepetimread(
    fp: str | bytes | PathLike | BinaryIO,
    lazy_frame_retrieval: bool = False
) -> LegacyConvertedEnhancedPETImage:
    """Read an LegacyConvertedEnhancedPETImage from DICOM file.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing an
        Legacy Converted Enhanced PET Image.
    lazy_frame_retrieval: bool
        If True, the returned image will retrieve frames from the file as
        requested, rather than loading in the entire object to memory
        initially. This may be a good idea if file reading is slow and you are
        likely to need only a subset of the frames in the image.

    Returns
    -------
    highdicom.legacy.LegacyConvertedEnhancedPETImage:
        Image read from the file.

    """
    return LegacyConvertedEnhancedPETImage.from_file(
        fp,
        lazy_frame_retrieval=lazy_frame_retrieval
    )
