"""Data Elements that are specific to the Segmentation IOD."""
from typing import Optional, Sequence, Union

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code

from highdicom.content import AlgorithmIdentificationSequence
from highdicom.seg.enum import SegmentAlgorithmTypes
from highdicom.sr.coding import CodedConcept


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
        processing_algorithm_identification: highdicom.content.AlgorithmIdentificationSequence, optional
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
