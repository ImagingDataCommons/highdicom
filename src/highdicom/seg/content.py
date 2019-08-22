"""Sequence items for the Segmentation IOD."""
from typing import Dict, Optional, Sequence, Union, Tuple

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.coding import Code
from pydicom.codedict import codes

from highdicom.enum import ImagingTargets
from highdicom.seg.enum import SegmentAlgorithmTypes
from highdicom.sr.coding import CodedConcept


class SegmentationAlgorithmIdentification(Dataset):

    """Dataset describing an item of the Segmentation Algorithm Identification
    Sequence.
    """

    def __init__(self, name: str,
                 family: Union[Code, CodedConcept],
                 version: str,
                 source: Optional[str] = None,
                 parameters: Optional[Dict[str, str]] = None):
        """
        Parameters
        ----------
        name: str
            Name of the algorithm
        family: Union[pydicom.sr.coding.Code, pydicom.sr.coding.CodedConcept]
            Kind of algorithm family
        version: str
            Version of the algorithm
        source: str, optional
            Source of the algorithm, e.g. name of the algorithm manufacturer
        parameters: Dict[str: str], optional
            Name and actual value of the parameters with which the algorithm
            was invoked

        """  # noqa
        super().__init__()
        self.AlgorithmName = name
        self.AlgorithmVersion = version
        self.AlgorithmFamilyCodeSequence = [
            family,
        ]
        self.AlgorithmParameters = ','.join([
            '='.join([key, value])
            for key, value in parameters.items()
        ])


class Segment(Dataset):

    """Dataset describing a segment based on the Specimen Description macro."""

    def __init__(self,
                 segment_number: int,
                 segment_label: str,
                 segmented_property_category: Union[Code, CodedConcept],
                 segmented_property_type: Union[Code, CodedConcept],
                 algorithm_type: Union[SegmentAlgorithmTypes, str],
                 algorithm_identification: SegmentationAlgorithmIdentification,
                 tracking_uid: Optional[str] = None,
                 tracking_id: Optional[str] = None,
                 anatomic_regions: Optional[Sequence[Union[Code, CodedConcept]]] = None,
                 primary_anatomic_structures: Optional[Sequence[Union[Code, CodedConcept]]] = None
            ) -> None:
        """
        Parameters
        ----------
        segment_number: int
            Number of the segment
        segment_label: str
            Label of the segment
        segmented_property_category: Union[pydicom.sr.coding.Code, pydicom.sr.coding.CodedConcept]
            Category of the property the segment represents,
            e.g. ``Code("49755003", "SCT", "Morphologically Abnormal Structure")``
            (see CID 7150 Segmentation Property Categories)
        segmented_property_type: Union[pydicom.sr.coding.Code, pydicom.sr.coding.CodedConcept]
            Property the segment represents,
            e.g. ``Code("108369006", "SCT", "Neoplasm")``
            (see CID 7151 Segmentation Property Types)
        algorithm_type: Union[str, highdicom.seg.enum.SegmentAlgorithmTypes]
            Type of algorithm
        algorithm_identification: SegmentationAlgorithmIdentificationSequence, optional
            Information useful for identification of the algorithm, such
            as its name or version
        tracking_uid: str, optional
            Unique tracking identifier (universally unique)
        tracking_id: str, optional
            Tracking identifier (unique only with the domain of use)
        anatomic_regions: Sequence[Union[Code, CodedConcept]], optional
            Anatomic region(s) into which segment falls,
            e.g. ``Code("41216001", "SCT", "Prostate")``
            (see CID 4 Anatomic Region, CID 4031 Common Anatomic Regions, as
            as well as other CIDs for domain-specific anatomic regions)
        primary_anatomic_structures: Sequence[Union[Code, CodedConcept]], optional
            Anatomic structure(s) the segment represents
            (see CIDs for domain-specific primary anatomic structures)

        """
        super().__init__()
        self.SegmentNumber = segment_number
        self.SegmentLabel = segment_label
        self.SegmentedPropertyCategoryCodeSequence = [
            segmented_property_category,
        ]
        self.SegmentedPropertyTypeCodeSequence = [
            segmented_property_type,
        ]
        self.SegmentAlgorithmType = SegmentAlgorithmTypes(algorithm_type).value
        self.SegmentAlgorithmName = algorithm_identification.AlgorithmName
        self.SegmentationAlgorithmIdentificationSequence = [
            algorithm_identification,
        ]
        num_given_tracking_identifiers = sum(
            tracking_id is not None,
            tracking_uid is not None
        )
        if num_given_tracking_identifiers == 2:
            self.TrackingID = tracking_id
            self.TrackingUID = tracking_uid
        elif num_given_tracking_identifiers == 1:
            raise TypeError(
                'Tracking ID and Tracking UID must both be provided.'
            )
        if anatomic_regions is not None:
            self.AnatomicRegionSequence = anatomic_regions
        if primary_anatomic_structures is not None:
            self.PrimaryAnatomicStructureSequence = primary_anatomic_structures


class DerivationImageSequence(DataElementSequence):

    """Sequence of data elements providing references to the source image of a
    segmentation image based on the Derivation Image functional group macros.
    """

    def __init__(
            self,
            referenced_sop_class_uid: str,
            referenced_sop_instance_uid: str,
            referenced_frame_numbers: Sequence[int]
        ) -> None:
        """
        Parameters
        ----------
        referenced_sop_class_uid: str
            SOP Class UID of the referenced source image
        referenced_sop_instance_uid: str
            SOP Instance UID of the referenced source image
        referenced_frame_numbers: Sequence[int]
            Frame number within the reference source image

        """
        super().__init__()
        derivation_item = Dataset()
        source_image = Dataset()
        source_image.ReferencedSOPClassUID = referenced_sop_class_uid
        source_image.ReferencedSOPInstanceUID = referenced_sop_instance_uid
        source_image.PurposeOfReferenceCodeSequence = [
            codes.DCM.SourceImageForImageProcessingOperation,
        ]
        item.DerivationCodeSequence = [
            codes.DCM.Segmentation,
        ]
        item.SourceImageSequence = [
            source_image,
        ]
        self.append(item)


class PixelMeasuresSequence(DataElementSequence):

    """Sequence of data elements describing physical spacing of an image based
    on the Pixel Measures functional group macro.
    """

    def __init__(
            self,
            pixel_spacing: Tuple[float, float],
            slice_thickness: float,
            spacing_between_slices: Optional[float] = None,
        ) -> None:
        """
        Parameters
        ----------
        pixel_spacing: Tuple[float, float]
            Distance in physical space between neighboring pixels in
            millimeters along the row and column dimension of the image
        slice_thickness: float
            Depth of physical space volume the image represents in millimeter
        spacing_between_slices: float, optional
            Distance in physical space between two consecutive images in
            millimeters. Only required for certain modalities, such as MR or CT.

        """
        super().__init__()
        item = Dataset()
        item.PixelSpacing = pixel_spacing
        item.SliceThickness = slice_thickness
        if spacing_between_slices is not None:
            item.SpacingBetweenSlices = spacing_between_slices
        self.append(item)


class PlanePositionSequence(DataElementSequence):

    """Sequence of data elements describing the position of an individual plane
    (frame) in the patient coordinate system based on the Plane Position
    (Patient) functional group macro.
    """

    def __init__(
            self,
            image_position: Tuple[float, float, float],
        ) -> None:
        """
        Parameters
        ----------
        image_position: Tuple[float, float, float]
            Offset of the first row and first column of the image in millimeter
            along the x, y, and z axis of the three-dimensional slide coordinate
            system

        """
        item = Dataset()
        item.ImagePositionPatient = image_position
        self.append(item)


class PlanePositionSlideSequence(DataElementSequence):

    """Sequence of data elements describing the position of an individual plane
    (frame) in the slide coordinate system based on the Plane Position (Slide)
    functional group macro.
    """

    def __init__(
            self,
            image_position: Tuple[float, float, float],
            pixel_matrix_position: Tuple[int, int],
        ) -> None:
        """
        Parameters
        ----------
        image_position: Tuple[float, float, float]
            Offset of the first row and first column of the image in millimeter
            along the x, y, and z axis of the three-dimensional slide coordinate
            system
        pixel_matrix_position: Tuple[int, int]
            Offset of the first row and first column of the image in
            pixels along the row and column axis of the two-dimensional total
            pixel matrix. Required if `imaging_target` is ``"slide"``.

        """
        item = Dataset()
        item.XOffsetInSlideCoordinateSystem = image_position[0]
        item.YOffsetInSlideCoordinateSystem = image_position[1]
        item.ZOffsetInSlideCoordinateSystem = image_position[2]
        item.RowPositionInTotalImagePixelMatrix = pixel_matrix_position[0]
        item.ColumnPositionInTotalImagePixelMatrix = pixel_matrix_position[1]
        self.append(item)


class PlaneOrientationSequence(DataElementSequence):

    """Sequence of data elements describing the image position in the patient
    or slide coordinate system based on either the Plane Orientation (Patient)
    or the Plane Orientation (Slide) functional group macro, respectively.
    """

    def __init__(
            self,
            imaging_target: Union[str, ImagingTargets],
            image_orientation: Tuple[float, float, float, float, float, float]
        ) -> None:
        """
        Parameters
        ----------
        imaging_target: Union[str, highdicom.enum.ImagingTargets]
            Subject (``"patient"`` or ``"slide"``) that was the target of
            imaging
        image_orientation: Tuple[float, float, float, float, float, float]
            Direction cosines for the first row (first triplet) and the first
            column (second triplet) of an image with respect to the x, y, and z
            axis of the three-dimensional slide coordinate system

        """
        super().__init__()
        imaging_target = ImagingTargets(imaging_target)
        item = Dataset()
        if imaging_target == ImagingTargets.SLIDE:
            item.ImageOrientationSlide = image_orientation
        elif imaging_target == ImagingTargets.PATIENT:
            item.ImageOrientationPatient = image_orientation
        self.append(item)


class DimensionIndexSequence(DataElementSequence):

    """Sequence of data elements describing dimension indices for the patient
    or slide coordinate system based on the Dimension Index functional
    group macro.
    """

    def __init__(
            self,
            imaging_target: Union[str, ImagingTargets]
        ) -> None:
        """
        Parameters
        ----------
        imaging_target: Union[str, highdicom.enum.ImagingTargets]
            Subject (``"patient"`` or ``"slide"``) that was the target of
            imaging

        """
        imaging_target = ImagingTargets(imaging_target)
        if imaging_target == ImagingTargets.SLIDE:
            dim_uid = '1.2.826.0.1.3680043.9.7433.2.4'

            segment_number_index = Dataset()
            segment_number_index.DimensionIndexPointer = tag_for_keyword(
                'ReferencedSegmentNumber'
            )
            segment_number_index.FunctionalGroupPointer = tag_for_keyword(
                'SegmentIdentificationSequence'
            )
            segment_number_index.DimensionOrganizationUID = dim_uid
            segment_number_index.DimensionDescriptionLabel = \
                'Segment Number'

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
            y_image_dimension_index.DimensionOrganizationUID = dim
            y_image_dimension_index.DimensionDescriptionLabel = \
                'Y Offset in Slide Coordinate System'

            z_image_dimension_index = Dataset()
            z_image_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'ZOffsetInSlideCoordinateSystem'
            )
            z_image_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            z_image_dimension_index.DimensionOrganizationUID = dim
            z_image_dimension_index.DimensionDescriptionLabel = \
                'Z Offset in Slide Coordinate System'

            self.extend([
                segment_number_index,
                x_image_dimension_index,
                y_image_dimension_index,
                z_image_dimension_index,
            ])
        elif imaging_target == ImagingTargets.PATIENT:
            dim_uid = '1.2.826.0.1.3680043.9.7433.2.3'

            dimension_organization = Dataset()
            dimension_organization.DimensionOrganizationUID = dim_uid
            self.DimensionOrganizationSequence = [dimension_organization]

            segment_number_index = Dataset()
            segment_number_index.DimensionIndexPointer = tag_for_keyword(
                'ReferencedSegmentNumber'
            )
            segment_number_index.FunctionalGroupPointer = tag_for_keyword(
                'SegmentIdentificationSequence'
            )
            segment_number_index.DimensionOrganizationUID = dim_uid

            image_position_index = Dataset()
            image_position_index.DimensionIndexPointer = tag_for_keyword(
                'ImagePositionPatient'
            )
            image_position_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSequence'
            )
            image_position_index.DimensionOrganizationUID = dim_uid

            self.extend([
                segment_number_index,
                image_position_index,
            ])

