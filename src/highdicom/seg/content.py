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
)
from highdicom.enum import (
    CoordinateSystemNames,
)
from highdicom.image import DimensionIndexSequence as BaseDimensionIndexSequence
from highdicom.seg.enum import SegmentAlgorithmTypeValues
from highdicom.sr.coding import CodedConcept
from highdicom._module_utils import (
    check_required_attributes,
)
from highdicom.volume import ChannelDescriptor


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
        segmented_property_type: Code | CodedConcept,
        algorithm_type: SegmentAlgorithmTypeValues | str,
        algorithm_identification: None | (
            AlgorithmIdentificationSequence
        ) = None,
        tracking_uid: str | None = None,
        tracking_id: str | None = None,
        anatomic_regions: None | (
            Sequence[Code | CodedConcept]
        ) = None,
        primary_anatomic_structures: None | (
            Sequence[Code | CodedConcept]
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
        segmented_property_category: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Category of the property the segment represents,
            e.g. ``Code("49755003", "SCT", "Morphologically Abnormal Structure")``
            (see :dcm:`CID 7150 <part16/sect_CID_7150.html>`
            "Segmentation Property Categories")
        segmented_property_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Property the segment represents,
            e.g. ``Code("108369006", "SCT", "Neoplasm")``
            (see :dcm:`CID 7151 <part16/sect_CID_7151.html>`
            "Segmentation Property Types")
        algorithm_type: Union[str, highdicom.seg.SegmentAlgorithmTypeValues]
            Type of algorithm
        algorithm_identification: Union[highdicom.AlgorithmIdentificationSequence, None], optional
            Information useful for identification of the algorithm, such
            as its name or version. Required unless the algorithm type is `MANUAL`
        tracking_uid: Union[str, None], optional
            Unique tracking identifier (universally unique)
        tracking_id: Union[str, None], optional
            Tracking identifier (unique only with the domain of use)
        anatomic_regions: Union[Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]], None], optional
            Anatomic region(s) into which segment falls,
            e.g. ``Code("41216001", "SCT", "Prostate")``
            (see :dcm:`CID 4 <part16/sect_CID_4.html>`
            "Anatomic Region", :dcm:`CID 4031 <part16/sect_CID_4031.html>`
            "Common Anatomic Regions", as as well as other CIDs for
            domain-specific anatomic regions)
        primary_anatomic_structures: Union[Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]], None], optional
            Anatomic structure(s) the segment represents
            (see CIDs for domain-specific primary anatomic structures)
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
        self.SegmentedPropertyCategoryCodeSequence = [
            CodedConcept.from_code(segmented_property_category)
        ]
        self.SegmentedPropertyTypeCodeSequence = [
            CodedConcept.from_code(segmented_property_type)
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
        if anatomic_regions is not None:
            self.AnatomicRegionSequence = [
                CodedConcept.from_code(region)
                for region in anatomic_regions
            ]
        if primary_anatomic_structures is not None:
            self.PrimaryAnatomicStructureSequence = [
                CodedConcept.from_code(structure)
                for structure in primary_anatomic_structures
            ]
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
            CodedConcept.from_dataset(
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
                CodedConcept.from_dataset(ds, copy=False)
                for ds in desc.AnatomicRegionSequence
            ]
        if hasattr(desc, 'PrimaryAnatomicStructureSequence'):
            desc.PrimaryAnatomicStructureSequence = [
                CodedConcept.from_dataset(ds, copy=False)
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
    def segmented_property_type(self) -> CodedConcept:
        """highdicom.sr.CodedConcept:
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
    def anatomic_regions(self) -> list[CodedConcept]:
        """List[highdicom.sr.CodedConcept]:
            List of anatomic regions into which the segment falls.
            May be empty.

        """
        if not hasattr(self, 'AnatomicRegionSequence'):
            return []
        return list(self.AnatomicRegionSequence)

    @property
    def primary_anatomic_structures(self) -> list[CodedConcept]:
        """List[highdicom.sr.CodedConcept]:
            List of anatomic anatomic structures the segment represents.
            May be empty.

        """
        if not hasattr(self, 'PrimaryAnatomicStructureSequence'):
            return []
        return list(self.PrimaryAnatomicStructureSequence)


class DimensionIndexSequence(BaseDimensionIndexSequence):

    """Sequence of data elements describing dimension indices for the patient
    or slide coordinate system based on the Dimension Index functional
    group macro.

    Note
    ----
    The order of indices is fixed.

    Note
    ----
    This class is deprecated and will be removed in a future version of
    highdicom. User code should generally avoid this class, and if necessary,
    the more general :class:`highdicom.DimensionIndexSequence` should be used
    instead.

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
            "should typically avoid this class, or, if required, use the more "
            "general highdicom.DimensionIndexSequence instead.",
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
