"""Content that is specific to Segmentation IODs."""
from copy import deepcopy
from typing import cast
from collections.abc import Sequence
from typing_extensions import Self
import warnings

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code

from highdicom.color import CIELabColor
from highdicom.content import (
    AlgorithmIdentificationSequence,
    AnatomicRegion,
    PrimaryAnatomicStructure,
)
from highdicom.enum import (
    CoordinateSystemNames,
)
from highdicom.image import _DimensionIndexSequence
from highdicom.seg.enum import SegmentAlgorithmTypeValues
from highdicom.sr.coding import CodedConcept, _ModifiedCodedConcept
from highdicom._standard_utils import (
    check_required_attributes,
)
from highdicom.volume import ChannelDescriptor


class SegmentedPropertyType(_ModifiedCodedConcept):
    """Dataset describing the property a segmentation represents.

    This class may be used instead of a plain
    :class:`highdicom.sr.CodedConcept` in order to include modifier codes that
    describe characteristics of the segmented property, such as laterality.

    See :dcm:`CID 7151 <part16/sect_CID_7151.html>` "Segmentation Property
    Types" for the appropriate context group for the root concept, and
    :dcm:`CID 2 <part16/sect_CID_2.html>` "Anatomic Modifier" for the context
    group for modifier codes.

    """
    _MODIFIER_SEQUENCE_NAME = "SegmentedPropertyTypeModifierCodeSequence"


class SegmentDescription(Dataset):

    """Dataset describing a segment based on the Segment Description macro.

    Note that this does **not** correspond to the "Segment Description"
    attribute (0062,0006), which is just one attribute within the Segment
    Description macro.

    """

    def __init__(
        self,
        segment_number: int,
        segment_label: str,
        segmented_property_category: Code | CodedConcept,
        segmented_property_type: Code | CodedConcept | SegmentedPropertyType,
        algorithm_type: SegmentAlgorithmTypeValues | str,
        algorithm_identification: None | (
            AlgorithmIdentificationSequence
        ) = None,
        tracking_uid: str | None = None,
        tracking_id: str | None = None,
        anatomic_regions: None | (
            Sequence[Code | CodedConcept | AnatomicRegion]
        ) = None,
        primary_anatomic_structures: None | (
            Sequence[Code | CodedConcept | PrimaryAnatomicStructure]
        ) = None,
        display_color: CIELabColor | None = None,
    ) -> None:
        """
        Parameters
        ----------
        segment_number: int
            Number of the segment.
        segment_label: str
            Label of the segment
        segmented_property_category: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, highdicom.seg.SegmentedPropertyType]
            Category of the property the segment represents,
            e.g. ``Code("49755003", "SCT", "Morphologically Abnormal Structure")``
            (see :dcm:`CID 7150 <part16/sect_CID_7150.html>`
            "Segmentation Property Categories")
        segmented_property_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, highdicom.seg.SegmentedPropertyType]
            Property the segment represents,
            e.g. ``Code("108369006", "SCT", "Neoplasm")``
            (see :dcm:`CID 7151 <part16/sect_CID_7151.html>` "Segmentation
            Property Types"). Use a
            :class:`highdicom.seg.SegmentedPropertyType` to pass a code with
            modifiers (such as specifying laterality).
        algorithm_type: Union[str, highdicom.seg.SegmentAlgorithmTypeValues]
            Type of algorithm
        algorithm_identification: Union[highdicom.AlgorithmIdentificationSequence, None], optional
            Information useful for identification of the algorithm, such
            as its name or version. Required unless the algorithm type is `MANUAL`
        tracking_uid: Union[str, None], optional
            Unique tracking identifier (universally unique). This is intended
            to provide an identifier to track this segment across multiple
            imaging studies.
        tracking_id: Union[str, None], optional
            Tracking identifier (unique only with the domain of use). This is
            intended to provide an identifier to track this segment across
            multiple imaging studies.
        anatomic_regions: Union[Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, highdicom.AnatomicRegion]], None], optional
            Anatomic region(s) into which segment falls,
            e.g. ``Code("41216001", "SCT", "Prostate")``
            (see :dcm:`CID 4 <part16/sect_CID_4.html>`
            "Anatomic Region", :dcm:`CID 4031 <part16/sect_CID_4031.html>`
            "Common Anatomic Regions", as as well as other CIDs for
            domain-specific anatomic regions). Use a list of
            :class:`highdicom.AnatomicRegion` to pass a code with
            modifiers (such as specifying laterality).
        primary_anatomic_structures: Union[Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, highdicom.PrimaryAnatomicStructure]], None], optional
            Anatomic structure(s) the segment represents (see CIDs for
            domain-specific primary anatomic structures). Use a list of
            :class:`highdicom.PrimaryAnatomicStructure` to pass a code with
            modifiers (such as specifying laterality).
        display_color: Union[highdicom.color.CIELabColor, None], optional
            A recommended color to render this segment.

        Note
        ----
        When segment descriptions are passed to a segmentation instance they
        must have consecutive segment numbers, starting at 1 for the first
        segment added.

        """  # noqa: E501
        super().__init__()
        if segment_number < 1 or segment_number > 65535:
            raise ValueError(
                "Segment number must be a positive integer below 65536."
            )
        self.SegmentNumber = segment_number
        self.SegmentLabel = segment_label

        if isinstance(segmented_property_category, _ModifiedCodedConcept):
            raise TypeError(
                "Argument 'segmented_property_category' should not have "
                f"type {segmented_property_category.__class__.__name__}."
            )

        self.SegmentedPropertyCategoryCodeSequence = [
            CodedConcept.from_code(segmented_property_category)
        ]

        if (
            isinstance(segmented_property_type, _ModifiedCodedConcept) and not
            isinstance(segmented_property_type, SegmentedPropertyType)
        ):
            raise TypeError(
                "Argument 'segmented_property_type' should not have "
                f"type {segmented_property_type.__class__.__name__}."
            )

        if not isinstance(segmented_property_type, SegmentedPropertyType):
            segmented_property_type = SegmentedPropertyType(
                segmented_property_type
            )

        self.SegmentedPropertyTypeCodeSequence = [
            segmented_property_type
        ]

        algorithm_type = SegmentAlgorithmTypeValues(algorithm_type)
        self.SegmentAlgorithmType = algorithm_type.value
        if algorithm_identification is None:
            if (
                self.SegmentAlgorithmType !=
                SegmentAlgorithmTypeValues.MANUAL.value
            ):
                raise TypeError(
                    "Algorithm identification sequence is required "
                    "unless the segmentation type is MANUAL"
                )
        else:
            self.SegmentAlgorithmName = \
                algorithm_identification[0].AlgorithmName
            self.SegmentationAlgorithmIdentificationSequence = \
                algorithm_identification

        num_given_tracking_identifiers = sum([
            tracking_id is not None,
            tracking_uid is not None
        ])

        if num_given_tracking_identifiers == 2:
            self.TrackingID = tracking_id
            self.TrackingUID = tracking_uid
        elif num_given_tracking_identifiers == 1:
            raise TypeError(
                'Tracking ID and Tracking UID must both be provided.'
            )

        if anatomic_regions is not None and len(anatomic_regions) > 0:
            checked_regions = []

            for region in anatomic_regions:
                if (
                    isinstance(region, _ModifiedCodedConcept) and not
                    isinstance(region, AnatomicRegion)
                ):
                    raise TypeError(
                        "Items of 'anatomic_regions' should not have "
                        f"type {region.__class__.__name__}."
                    )

                if isinstance(region, AnatomicRegion):
                    checked_regions.append(region)
                else:
                    checked_regions.append(
                        AnatomicRegion(region)
                    )

            self.AnatomicRegionSequence = checked_regions

        if (
            primary_anatomic_structures is not None and
            len(primary_anatomic_structures) > 0
        ):
            checked_structures = []

            for structure in primary_anatomic_structures:
                if (
                    isinstance(structure, _ModifiedCodedConcept) and not
                    isinstance(structure, PrimaryAnatomicStructure)
                ):
                    raise TypeError(
                        "Items of 'primary_anatomic_structures' should not "
                        f"have type {structure.__class__.__name__}."
                    )

                if isinstance(structure, PrimaryAnatomicStructure):
                    checked_structures.append(structure)
                else:
                    checked_structures.append(
                        PrimaryAnatomicStructure(structure)
                    )

            self.PrimaryAnatomicStructureSequence = checked_structures

        if display_color is not None:
            if not isinstance(display_color, CIELabColor):
                raise TypeError(
                    '"display_color" must be of type '
                    'highdicom.color.CIELabColor.'
                )
            self.RecommendedDisplayCIELabValue = list(
                display_color.value
            )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True
    ) -> Self:
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an item of the Segment Sequence.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.seg.SegmentDescription
            Segment description.

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        check_required_attributes(
            dataset,
            module='segmentation-image',
            base_path=['SegmentSequence']
        )
        if copy:
            desc = deepcopy(dataset)
        else:
            desc = dataset
        desc.__class__ = cls

        # Convert sub sequences to highdicom types
        desc.SegmentedPropertyCategoryCodeSequence = [
            CodedConcept.from_dataset(
                desc.SegmentedPropertyCategoryCodeSequence[0],
                copy=False,
            )
        ]
        desc.SegmentedPropertyTypeCodeSequence = [
            SegmentedPropertyType.from_dataset(
                desc.SegmentedPropertyTypeCodeSequence[0],
                copy=False,
            )
        ]
        if hasattr(desc, 'SegmentationAlgorithmIdentificationSequence'):
            desc.SegmentationAlgorithmIdentificationSequence = \
                AlgorithmIdentificationSequence.from_sequence(
                    desc.SegmentationAlgorithmIdentificationSequence,
                    copy=False,
                )
        if hasattr(desc, 'AnatomicRegionSequence'):
            desc.AnatomicRegionSequence = [
                AnatomicRegion.from_dataset(ds, copy=False)
                for ds in desc.AnatomicRegionSequence
            ]
        if hasattr(desc, 'PrimaryAnatomicStructureSequence'):
            desc.PrimaryAnatomicStructureSequence = [
                PrimaryAnatomicStructure.from_dataset(ds, copy=False)
                for ds in desc.PrimaryAnatomicStructureSequence
            ]
        return cast(Self, desc)

    @property
    def segment_number(self) -> int:
        """int: Number of the segment."""
        return int(self.SegmentNumber)

    @property
    def segment_label(self) -> str:
        """str: Label of the segment."""
        return str(self.SegmentLabel)

    @property
    def segmented_property_category(self) -> CodedConcept:
        """highdicom.sr.CodedConcept:
            Category of the property the segment represents.

        """
        return self.SegmentedPropertyCategoryCodeSequence[0]

    @property
    def segmented_property_type(self) -> SegmentedPropertyType:
        """highdicom.seg.SegmentedPropertyType:
            Type of the property the segment represents.

        """
        return self.SegmentedPropertyTypeCodeSequence[0]

    @property
    def algorithm_type(self) -> SegmentAlgorithmTypeValues:
        """highdicom.seg.SegmentAlgorithmTypeValues:
            Type of algorithm used to create the segment.

        """
        return SegmentAlgorithmTypeValues(self.SegmentAlgorithmType)

    @property
    def algorithm_identification(
        self
    ) -> AlgorithmIdentificationSequence | None:
        """Union[highdicom.AlgorithmIdentificationSequence, None]
            Information useful for identification of the algorithm, if any.

        """
        if hasattr(self, 'SegmentationAlgorithmIdentificationSequence'):
            return self.SegmentationAlgorithmIdentificationSequence
        return None

    @property
    def tracking_uid(self) -> str | None:
        """Union[str, None]:
            Tracking unique identifier for the segment, if any.

        """
        if 'TrackingUID' in self:
            return self.TrackingUID
        return None

    @property
    def tracking_id(self) -> str | None:
        """Union[str, None]: Tracking identifier for the segment, if any."""
        if 'TrackingID' in self:
            return self.TrackingID
        return None

    @property
    def anatomic_regions(self) -> list[AnatomicRegion]:
        """List[highdicom.AnatomicRegion]:
            List of anatomic regions into which the segment falls.
            May be empty.

        """
        if not hasattr(self, 'AnatomicRegionSequence'):
            return []
        return list(self.AnatomicRegionSequence)

    @property
    def primary_anatomic_structures(self) -> list[PrimaryAnatomicStructure]:
        """List[highdicom.PrimaryAnatomicStructure]:
            List of anatomic anatomic structures the segment represents.
            May be empty.

        """
        if not hasattr(self, 'PrimaryAnatomicStructureSequence'):
            return []
        return list(self.PrimaryAnatomicStructureSequence)


class DimensionIndexSequence(_DimensionIndexSequence):

    """Sequence of data elements describing dimension indices for the patient
    or slide coordinate system based on the Dimension Index functional
    group macro.

    Note
    ----
    The order of indices is fixed.

    Note
    ----
    This class is deprecated and will be removed in a future version of
    highdicom. User code should generally avoid this class.

    """

    def __init__(
        self,
        coordinate_system: str | CoordinateSystemNames | None,
        include_segment_number: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.CoordinateSystemNames, None]
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
            imaging. If None, the imaging does not belong within a frame of
            reference.
        include_segment_number: bool
            Include the segment number as a dimension index.

        """
        warnings.warn(
            "The highdicom.seg.DimensionIndexSequence class is deprecated and "
            "will be removed in a future version of the library. User code "
            "should typically avoid this class.",
            UserWarning,
            stacklevel=2,
        )
        channel_dimensions = (
            [ChannelDescriptor(0x0062_000b)]  # ReferencedSegmentNumber
            if include_segment_number else None
        )
        super().__init__(
            coordinate_system=coordinate_system,
            functional_groups_module=(
                'segmentation-multi-frame-functional-groups'
            ),
            channel_dimensions=channel_dimensions,
        )
