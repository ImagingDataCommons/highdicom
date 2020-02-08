"""Sequence items for the Segmentation IOD."""
from typing import Dict, Optional, Sequence, Union, Tuple

import numpy as np
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes

from highdicom.enum import CoordinateSystemNames
from highdicom.seg.enum import SegmentAlgorithmTypes
from highdicom.sr.coding import CodedConcept


SLIDE_DIMENSION_ORGANIZATION_UID = '1.2.826.0.1.3680043.9.7433.2.4'

PATIENT_DIMENSION_ORGANIZATION_UID = '1.2.826.0.1.3680043.9.7433.2.3'


class AlgorithmIdentificationSequence(DataElementSequence):

    """Sequence of data elements describing information useful for
    identification of an algorithm.
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
        item = Dataset()
        item.AlgorithmName = name
        item.AlgorithmVersion = version
        item.AlgorithmFamilyCodeSequence = [
            CodedConcept(
                family.value,
                family.scheme_designator,
                family.meaning,
                family.scheme_version,
            ),
        ]
        if source is not None:
            item.AlgorithmSource = source
        if parameters is not None:
            item.AlgorithmParameters = ','.join([
                '='.join([key, value])
                for key, value in parameters.items()
            ])
        self.append(item)


class SegmentDescription(Dataset):

    """Dataset describing a segment based on the Segment Description macro."""

    def __init__(
            self,
            segment_number: int,
            segment_label: str,
            segmented_property_category: Union[Code, CodedConcept],
            segmented_property_type: Union[Code, CodedConcept],
            algorithm_type: Union[SegmentAlgorithmTypes, str],
            algorithm_identification: AlgorithmIdentificationSequence,
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
        algorithm_identification: AlgorithmIdentificationSequence, optional
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
        self.SegmentAlgorithmType = SegmentAlgorithmTypes(algorithm_type).value
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


class Surface(Dataset):

    """Dataset representing an item of the Surface Sequence attribute."""

    def __init__(
            self,
            number: int,
            points: np.ndarray,
            is_processed: Optional[bool] = None,
            processing_ratio: Optional[float] = None,
            processing_algorithm_identification:
                Optional[AlgorithmIdentificationSequence] = None,
            is_finite_volume: Optional[bool] = None,
            is_manifold: Optional[bool] = None
        ):
        """
        Parameters
        ----------
        number: int
            One-based index number of the surface
        points: numpy.ndarray
            Array of shape (n, 3), where *n* is the number of points defining
            the surface of a mesh (polyhedral object in the three-dimensional
            slide or patient coordinate system) or a point cloud, where each
            point is defined by a (x, y, z) coordinate triplet
        is_processed: bool, optional
            Whether the surface has been processed to reduce the number of
            points
        processing_ratio: float, optional
            Ratio of number of remaining points to number of original points
            if surface has been processed
        processing_algorithm_identification: highdicom.seg.content.AlgorithmIdentificationSequence, optional
            Identifying information about the algorithm that was used to
            process the surface
        is_finite_volume: bool, optional
            Whether the surface has a finite volume,
            i.e. is topologically closed
        is_manifold: bool, optional
            Whether the surface is a manifold

        Note
        ----
        When `is_finite_volume` or `is_manifold` are not specified, the value
        of attributes ``FiniteVolume`` and ``Manifold`` are set to ``"UNKOWN"``,
        respectively.

        """  # noqa
        super().__init__()
        self.SurfaceNumber = number

        if is_processed is not None:
            if is_processed:
                self.SurfaceProcessing = 'YES'
                if processing_ratio is None:
                    raise TypeError(
                        'Surface processing ratio must be specified if '
                        'surface has been processed.'
                    )
                self.SurfaceProcessingRatio = float(processing_ratio)
                if processing_algorithm_identification is None:
                    raise TypeError(
                        'Surface processing algorithm identification must be '
                        'specified if surface has been processed.'
                    )
                self.SurfaceProcessingAlgorithmIdentificationSequence = \
                    processing_algorithm_identification
            else:
                self.SurfaceProcessing = 'NO'
        else:
            self.SurfaceProcessing = None
        if is_finite_volume is not None:
            self.FiniteVolume = 'YES' if is_finite_volume else 'NO'
        else:
            self.FiniteVolume = 'UNKNOWN'
        if is_manifold is not None:
            self.Manifold = 'YES' if is_manifold else 'NO'
        else:
            self.Manifold = 'UNKNOWN'

        if points.shape[1] != 3:
            raise ValueError(
                'Points must supposed to be 3D spatial coordinates and must '
                'be represented as a vector of length 3.'
            )
        points_item = Dataset()
        points_item.NumberOfSurfacePoints = points.shape[0]
        points_item.PointCoordinateData = points.flatten().tolist()
        self.SurfacePointsSequence = [points_item]
        # TODO: compute bounding box
        # self.PointsBoundingBox = []
        self.SurfacePointsNormalsSequence = []
        # TODO: compute normals at points
        # normals = np.array()
        # normals_item = Dataset()
        # normals_item.NumberOfVectors = normals.shape[0]
        # normals_item.VectorDimensionality = normals.shape[1]
        # normals_item.VectorCoordinateData = normals.flatten().tolist()
        # self.SurfacePointsNormalsSequence = [normals_item]
        # TODO: compute primitives
        mesh_item = Dataset()
        mesh_item.LongVertexPointIndexList = None
        mesh_item.LongEdgePointIndexList = None
        mesh_item.LongTrianglePointIndexList = None
        mesh_item.TriangleStripSequence = []
        mesh_item.TriangleFanSequence = []
        mesh_item.LineSequence = []
        mesh_item.FacetSequence = []
        self.SurfaceMeshPrimitivesSequence = [mesh_item]


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
            millimeters. Only required for certain modalities, such as MR.

        """
        super().__init__()
        item = Dataset()
        item.PixelSpacing = list(pixel_spacing)
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
            Offset of the first row and first column of the plane (frame) in
            millimeter along the x, y, and z axis of the patient coordinate
            system

        """
        super().__init__()
        item = Dataset()
        item.ImagePositionPatient = list(image_position)
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
            Offset of the first row and first column of the plane (frame) in
            millimeter along the x, y, and z axis of the slide coordinate
            system
        pixel_matrix_position: Tuple[int, int]
            Offset of the first row and first column of the plane (frame) in
            pixels along the row and column direction of the total pixel matrix

        """
        super().__init__()
        item = Dataset()
        item.XOffsetInSlideCoordinateSystem = image_position[0]
        item.YOffsetInSlideCoordinateSystem = image_position[1]
        item.ZOffsetInSlideCoordinateSystem = image_position[2]
        item.RowPositionInTotalImagePixelMatrix = pixel_matrix_position[0]
        item.ColumnPositionInTotalImagePixelMatrix = pixel_matrix_position[1]
        self.append(item)

    @classmethod
    def compute_for_tiled_full(
            cls,
            row_index: int,
            column_index: int,
            depth_index: int,
            x_offset: float,
            y_offset: float,
            z_offset: float,
            rows: int,
            columns: int,
            image_orientation: Tuple[float, float, float, float, float, float],
            pixel_spacing: Tuple[float, float],
            slice_thickness: float,
            spacing_between_slices: float
        ) -> 'PlanePositionSlideSequence':
        """Computes the position of a plane (frame) for Dimension Orientation
        Type TILED_FULL.

        Parameters
        ----------
        row_index: int
            Relative one-based index value for a given frame along the row
            direction of the the tiled total pixel matrix, which is defined by
            the first triplet in `image_orientation`
        column_index: int
            Relative one-based index value for a given frame along the column
            direction of the the tiled total pixel matrix, which is defined by
            the second triplet in `image_orientation`
        depth_index: int
            Relative one-based index value for a given frame along the depth
            direction from the glass slide to the coverslip (focal plane)
        x_offset_image: float
            X offset of the total pixel matrix in the slide coordinate system
        y_offset_image: float
            Y offset of the total pixel matrix in the slide coordinate system
        z_offset_image: float
            Z offset of the total pixel matrix (focal plane) in the slide
            coordinate system
        rows: int
            Number of rows per tile
        columns: int
            Number of columns per tile
        image_orientation: Tuple[float, float, float, float, float, float]
            Cosines of row (first triplet) and column (second triplet) direction
            for x, y and z axis of the slide coordinate system
        pixel_spacing: Tuple[float, float]
            Physical distance between the centers of neighboring pixels along
            the row and column direction
        slice_thickness: float
            Physical thickness of a focal plane
        spacing_between_slices: float
            Physical distance between neighboring focal planes

        """
        row_offset_frame = ((row_index - 1) * rows) + 1
        column_offset_frame = ((column_index - 1) * columns) + 1
        # We need to take rotation of pixel matrix relative to slide into account.
        # According to the standard, we only have to deal with planar rotations by
        # 180 degrees along the row and/or column direction.
        if tuple([float(v) for v in image_orientation[:3]]) == (0.0, -1.0, 0.0):
            x_func = np.subtract
        else:
            x_func = np.add
        x_offset_frame = x_func(
            x_offset,
            (row_offset_frame * pixel_spacing[1])
        )
        if tuple([float(v) for v in image_orientation[3:]]) == (-1.0, 0.0, 0.0):
            y_func = np.subtract
        else:
            y_func = np.add
        y_offset_frame = y_func(
            y_offset,
            (column_offset_frame * pixel_spacing[0])
        )
        z_offset_frame = np.sum([
            z_offset,
            (float(depth_index - 1) * slice_thickness),
            (float(depth_index - 1) * spacing_between_slices)
        ])
        return cls(
            image_position=(x_offset_frame, y_offset_frame, z_offset_frame),
            pixel_matrix_position=(row_offset_frame, column_offset_frame)
        )


class PlaneOrientationSequence(DataElementSequence):

    """Sequence of data elements describing the image position in the patient
    or slide coordinate system based on either the Plane Orientation (Patient)
    or the Plane Orientation (Slide) functional group macro, respectively.
    """

    def __init__(
            self,
            coordinate_system: Union[str, CoordinateSystemNames],
            image_orientation: Tuple[float, float, float, float, float, float]
        ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.enum.CoordinateSystemNames]
            Subject (``"patient"`` or ``"slide"``) that was the target of
            imaging
        image_orientation: Tuple[float, float, float, float, float, float]
            Direction cosines for the first row (first triplet) and the first
            column (second triplet) of an image with respect to the x, y, and z
            axis of the three-dimensional slide coordinate system

        """
        super().__init__()
        coordinate_system = CoordinateSystemNames(coordinate_system)
        item = Dataset()
        if coordinate_system == CoordinateSystemNames.SLIDE:
            item.ImageOrientationSlide = list(image_orientation)
        elif coordinate_system == CoordinateSystemNames.PATIENT:
            item.ImageOrientationPatient = list(image_orientation)
        self.append(item)


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
            Subject (``"patient"`` or ``"slide"``) that was the target of
            imaging

        """
        super().__init__()
        coordinate_system = CoordinateSystemNames(coordinate_system)
        if coordinate_system == CoordinateSystemNames.SLIDE:
            dim_uid = SLIDE_DIMENSION_ORGANIZATION_UID

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
            dim_uid = PATIENT_DIMENSION_ORGANIZATION_UID

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
