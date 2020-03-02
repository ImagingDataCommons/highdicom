"""Data Elements that are specific to the Segmentation IOD."""
from typing import Optional, Sequence, Union

import numpy as np
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code

from highdicom.content import AlgorithmIdentificationSequence
from highdicom.enum import CoordinateSystemNames
from highdicom.seg.enum import SegmentAlgorithmTypeValues
from highdicom.sr.coding import CodedConcept


class SegmentDescription(Dataset):

    """Dataset describing a segment based on the Segment Description macro."""

    def __init__(
            self,
            segment_number: int,
            segment_label: str,
            segmented_property_category: Union[Code, CodedConcept],
            segmented_property_type: Union[Code, CodedConcept],
            algorithm_type: Union[SegmentAlgorithmTypeValues, str],
            algorithm_identification: AlgorithmIdentificationSequence,
            tracking_uid: Optional[str] = None,
            tracking_id: Optional[str] = None,
            anatomic_regions: Optional[
                Sequence[Union[Code, CodedConcept]]
            ] = None,
            primary_anatomic_structures: Optional[
                Sequence[Union[Code, CodedConcept]]
            ] = None
        ) -> None:
        """
        Parameters
        ----------
        segment_number: int
            Number of the segment
        segment_label: str
            Label of the segment
        segmented_property_category: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]
            Category of the property the segment represents,
            e.g. ``Code("49755003", "SCT", "Morphologically Abnormal Structure")``
            (see CID 7150 Segmentation Property Categories)
        segmented_property_type: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]
            Property the segment represents,
            e.g. ``Code("108369006", "SCT", "Neoplasm")``
            (see CID 7151 Segmentation Property Types)
        algorithm_type: Union[str, highdicom.seg.enum.SegmentAlgorithmTypeValues]
            Type of algorithm
        algorithm_identification: highdicom.content.AlgorithmIdentificationSequence, optional
            Information useful for identification of the algorithm, such
            as its name or version
        tracking_uid: str, optional
            Unique tracking identifier (universally unique)
        tracking_id: str, optional
            Tracking identifier (unique only with the domain of use)
        anatomic_regions: Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]], optional
            Anatomic region(s) into which segment falls,
            e.g. ``Code("41216001", "SCT", "Prostate")``
            (see CID 4 Anatomic Region, CID 4031 Common Anatomic Regions, as
            as well as other CIDs for domain-specific anatomic regions)
        primary_anatomic_structures: Sequence[Union[highdicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]], optional
            Anatomic structure(s) the segment represents
            (see CIDs for domain-specific primary anatomic structures)

        """  # noqa
        super().__init__()
        self.SegmentNumber = segment_number
        self.SegmentLabel = segment_label
        self.SegmentedPropertyCategoryCodeSequence = [
            CodedConcept(
                segmented_property_category.value,
                segmented_property_category.scheme_designator,
                segmented_property_category.meaning,
                segmented_property_category.scheme_version
            ),
        ]
        self.SegmentedPropertyTypeCodeSequence = [
            CodedConcept(
                segmented_property_type.value,
                segmented_property_type.scheme_designator,
                segmented_property_type.meaning,
                segmented_property_type.scheme_version
            ),
        ]
        algorithm_type = SegmentAlgorithmTypeValues(algorithm_type)
        self.SegmentAlgorithmType = algorithm_type.value
        self.SegmentAlgorithmName = algorithm_identification[0].AlgorithmName
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
                CodedConcept(
                    region.value,
                    region.scheme_designator,
                    region.meaning,
                    region.scheme_version
                )
                for region in anatomic_regions
            ]
        if primary_anatomic_structures is not None:
            self.PrimaryAnatomicStructureSequence = [
                CodedConcept(
                    structure.value,
                    structure.scheme_designator,
                    structure.meaning,
                    structure.scheme_version
                )
                for structure in primary_anatomic_structures
            ]


class DimensionIndexSequence(DataElementSequence):

    """Sequence of data elements describing dimension indices for the patient
    or slide coordinate system based on the Dimension Index functional
    group macro.

    Note
    ----
    The order of indices is fixed.

    """

    def __init__(
            self,
            coordinate_system: Union[str, CoordinateSystemNames]
        ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.enum.CoordinateSystemNames]
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
            imaging

        """
        super().__init__()
        coordinate_system = CoordinateSystemNames(coordinate_system)
        if coordinate_system == CoordinateSystemNames.SLIDE:
            dim_uid = '1.2.826.0.1.3680043.9.7433.2.4'

            segment_number_index = Dataset()
            segment_number_index.DimensionIndexPointer = tag_for_keyword(
                'ReferencedSegmentNumber'
            )
            segment_number_index.FunctionalGroupPointer = tag_for_keyword(
                'SegmentIdentificationSequence'
            )
            segment_number_index.DimensionOrganizationUID = dim_uid
            segment_number_index.DimensionDescriptionLabel = 'Segment Number'

            x_image_dimension_index = Dataset()
            x_image_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'XOffsetInSlideCoordinateSystem'
            )
            x_image_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            x_image_dimension_index.DimensionOrganizationUID = dim_uid
            x_image_dimension_index.DimensionDescriptionLabel = \
                'X Offset in Slide Coordinate System'

            y_image_dimension_index = Dataset()
            y_image_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'YOffsetInSlideCoordinateSystem'
            )
            y_image_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            y_image_dimension_index.DimensionOrganizationUID = dim_uid
            y_image_dimension_index.DimensionDescriptionLabel = \
                'Y Offset in Slide Coordinate System'

            z_image_dimension_index = Dataset()
            z_image_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'ZOffsetInSlideCoordinateSystem'
            )
            z_image_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            z_image_dimension_index.DimensionOrganizationUID = dim_uid
            z_image_dimension_index.DimensionDescriptionLabel = \
                'Z Offset in Slide Coordinate System'

            col_image_dimension_index = Dataset()
            col_image_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'ColumnPositionInTotalImagePixelMatrix'
            )
            col_image_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            col_image_dimension_index.DimensionOrganizationUID = dim_uid
            col_image_dimension_index.DimensionDescriptionLabel = \
                'Column Position In Total Image Pixel Matrix'

            row_image_dimension_index = Dataset()
            row_image_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'RowPositionInTotalImagePixelMatrix'
            )
            row_image_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            row_image_dimension_index.DimensionOrganizationUID = dim_uid
            row_image_dimension_index.DimensionDescriptionLabel = \
                'Row Position In Total Image Pixel Matrix'

            self.extend([
                segment_number_index,
                x_image_dimension_index,
                y_image_dimension_index,
                z_image_dimension_index,
                col_image_dimension_index,
                row_image_dimension_index,
            ])

        elif coordinate_system == CoordinateSystemNames.PATIENT:
            dim_uid = '1.2.826.0.1.3680043.9.7433.2.3'

            segment_number_index = Dataset()
            segment_number_index.DimensionIndexPointer = tag_for_keyword(
                'ReferencedSegmentNumber'
            )
            segment_number_index.FunctionalGroupPointer = tag_for_keyword(
                'SegmentIdentificationSequence'
            )
            segment_number_index.DimensionOrganizationUID = dim_uid
            segment_number_index.DimensionDescriptionLabel = 'Segment Number'

            image_position_index = Dataset()
            image_position_index.DimensionIndexPointer = tag_for_keyword(
                'ImagePositionPatient'
            )
            image_position_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSequence'
            )
            image_position_index.DimensionOrganizationUID = dim_uid
            image_position_index.DimensionDescriptionLabel = \
                'Image Position Patient'

            self.extend([
                segment_number_index,
                image_position_index,
            ])
