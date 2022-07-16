"""Content that is specific to Segmentation IODs."""
from copy import deepcopy
from typing import cast, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code

from highdicom.content import (
    AlgorithmIdentificationSequence,
    PlanePositionSequence,
)
from highdicom.enum import CoordinateSystemNames
from highdicom.seg.enum import SegmentAlgorithmTypeValues
from highdicom.spatial import map_pixel_into_coordinate_system
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID
from highdicom.utils import compute_plane_position_slide_per_frame
from highdicom._module_utils import check_required_attributes


class SegmentDescription(Dataset):

    """Dataset describing a segment based on the Segment Description macro."""

    def __init__(
        self,
        segment_number: int,
        segment_label: str,
        segmented_property_category: Union[Code, CodedConcept],
        segmented_property_type: Union[Code, CodedConcept],
        algorithm_type: Union[SegmentAlgorithmTypeValues, str],
        algorithm_identification: Optional[
            AlgorithmIdentificationSequence
        ] = None,
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
        primary_anatomic_structures: Union[Sequence[Union[highdicom.sr.Code, highdicom.sr.CodedConcept]], None], optional
            Anatomic structure(s) the segment represents
            (see CIDs for domain-specific primary anatomic structures)

        Notes
        -----
        When segment descriptions are passed to a segmentation instance they
        must have consecutive segment numbers, starting at 1 for the first
        segment added.

        """  # noqa: E501
        super().__init__()
        if segment_number < 1:
            raise ValueError("Segment number must be a positive integer")
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

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'SegmentDescription':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an item of the Segment Sequence.

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
        desc = deepcopy(dataset)
        desc.__class__ = cls

        # Convert sub sequences to highdicom types
        desc.SegmentedPropertyCategoryCodeSequence = [
            CodedConcept.from_dataset(
                desc.SegmentedPropertyCategoryCodeSequence[0]
            )
        ]
        desc.SegmentedPropertyTypeCodeSequence = [
            CodedConcept.from_dataset(
                desc.SegmentedPropertyTypeCodeSequence[0]
            )
        ]
        if hasattr(desc, 'SegmentationAlgorithmIdentificationSequence'):
            desc.SegmentationAlgorithmIdentificationSequence = \
                AlgorithmIdentificationSequence.from_sequence(
                    desc.SegmentationAlgorithmIdentificationSequence
                )
        if hasattr(desc, 'AnatomicRegionSequence'):
            desc.AnatomicRegionSequence = [
                CodedConcept.from_dataset(ds)
                for ds in desc.AnatomicRegionSequence
            ]
        if hasattr(desc, 'PrimaryAnatomicStructureSequence'):
            desc.PrimaryAnatomicStructureSequence = [
                CodedConcept.from_dataset(ds)
                for ds in desc.PrimaryAnatomicStructureSequence
            ]
        return cast(SegmentDescription, desc)

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
    ) -> Union[AlgorithmIdentificationSequence, None]:
        """Union[highdicom.AlgorithmIdentificationSequence, None]
            Information useful for identification of the algorithm, if any.

        """
        if hasattr(self, 'SegmentationAlgorithmIdentificationSequence'):
            return self.SegmentationAlgorithmIdentificationSequence
        return None

    @property
    def tracking_uid(self) -> Union[str, None]:
        """Union[str, None]:
            Tracking unique identifier for the segment, if any.

        """
        if 'TrackingUID' in self:
            return self.TrackingUID
        return None

    @property
    def tracking_id(self) -> Union[str, None]:
        """Union[str, None]: Tracking identifier for the segment, if any."""
        if 'TrackingID' in self:
            return self.TrackingID
        return None

    @property
    def anatomic_regions(self) -> List[CodedConcept]:
        """List[highdicom.sr.CodedConcept]:
            List of anatomic regions into which the segment falls.
            May be empty.

        """
        if not hasattr(self, 'AnatomicRegionSequence'):
            return []
        return list(self.AnatomicRegionSequence)

    @property
    def primary_anatomic_structures(self) -> List[CodedConcept]:
        """List[highdicom.sr.CodedConcept]:
            List of anatomic anatomic structures the segment represents.
            May be empty.

        """
        if not hasattr(self, 'PrimaryAnatomicStructureSequence'):
            return []
        return list(self.PrimaryAnatomicStructureSequence)


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
        coordinate_system: Union[str, CoordinateSystemNames, None]
    ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.CoordinateSystemNames, None]
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
            imaging. If None, the imaging does not belong within a frame of
            reference.

        """
        super().__init__()
        if coordinate_system is None:
            self._coordinate_system = None
        else:
            self._coordinate_system = CoordinateSystemNames(coordinate_system)

        if self._coordinate_system is None:
            dim_uid = UID()

            segment_number_index = Dataset()
            segment_number_index.DimensionIndexPointer = tag_for_keyword(
                'ReferencedSegmentNumber'
            )
            segment_number_index.FunctionalGroupPointer = tag_for_keyword(
                'SegmentIdentificationSequence'
            )
            segment_number_index.DimensionOrganizationUID = dim_uid
            segment_number_index.DimensionDescriptionLabel = 'Segment Number'

            self.append(segment_number_index)

        elif self._coordinate_system == CoordinateSystemNames.SLIDE:
            dim_uid = UID()

            segment_number_index = Dataset()
            segment_number_index.DimensionIndexPointer = tag_for_keyword(
                'ReferencedSegmentNumber'
            )
            segment_number_index.FunctionalGroupPointer = tag_for_keyword(
                'SegmentIdentificationSequence'
            )
            segment_number_index.DimensionOrganizationUID = dim_uid
            segment_number_index.DimensionDescriptionLabel = 'Segment Number'

            x_axis_index = Dataset()
            x_axis_index.DimensionIndexPointer = tag_for_keyword(
                'XOffsetInSlideCoordinateSystem'
            )
            x_axis_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            x_axis_index.DimensionOrganizationUID = dim_uid
            x_axis_index.DimensionDescriptionLabel = \
                'X Offset in Slide Coordinate System'

            y_axis_index = Dataset()
            y_axis_index.DimensionIndexPointer = tag_for_keyword(
                'YOffsetInSlideCoordinateSystem'
            )
            y_axis_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            y_axis_index.DimensionOrganizationUID = dim_uid
            y_axis_index.DimensionDescriptionLabel = \
                'Y Offset in Slide Coordinate System'

            z_axis_index = Dataset()
            z_axis_index.DimensionIndexPointer = tag_for_keyword(
                'ZOffsetInSlideCoordinateSystem'
            )
            z_axis_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            z_axis_index.DimensionOrganizationUID = dim_uid
            z_axis_index.DimensionDescriptionLabel = \
                'Z Offset in Slide Coordinate System'

            row_dimension_index = Dataset()
            row_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'ColumnPositionInTotalImagePixelMatrix'
            )
            row_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            row_dimension_index.DimensionOrganizationUID = dim_uid
            row_dimension_index.DimensionDescriptionLabel = \
                'Column Position In Total Image Pixel Matrix'

            column_dimension_index = Dataset()
            column_dimension_index.DimensionIndexPointer = tag_for_keyword(
                'RowPositionInTotalImagePixelMatrix'
            )
            column_dimension_index.FunctionalGroupPointer = tag_for_keyword(
                'PlanePositionSlideSequence'
            )
            column_dimension_index.DimensionOrganizationUID = dim_uid
            column_dimension_index.DimensionDescriptionLabel = \
                'Row Position In Total Image Pixel Matrix'

            # Organize frames for each segment similar to TILED_FULL, first
            # along the row dimension (column indices from left to right) and
            # then along the column dimension (row indices from top to bottom)
            # of the Total Pixel Matrix.
            self.extend([
                segment_number_index,
                row_dimension_index,
                column_dimension_index,
                x_axis_index,
                y_axis_index,
                z_axis_index,
            ])

        elif self._coordinate_system == CoordinateSystemNames.PATIENT:
            dim_uid = UID()

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

        else:
            raise ValueError(
                f'Unknown coordinate system "{self._coordinate_system}"'
            )

    def get_plane_positions_of_image(
        self,
        image: Dataset
    ) -> List[PlanePositionSequence]:
        """Gets plane positions of frames in multi-frame image.

        Parameters
        ----------
        image: Dataset
            Multi-frame image

        Returns
        -------
        List[highdicom.PlanePositionSequence]
            Plane position of each frame in the image

        """
        is_multiframe = hasattr(image, 'NumberOfFrames')
        if not is_multiframe:
            raise ValueError('Argument "image" must be a multi-frame image.')

        if self._coordinate_system is None:
            raise ValueError(
                'Cannot calculate plane positions when images do not exist '
                'within a frame of reference.'
            )
        elif self._coordinate_system == CoordinateSystemNames.SLIDE:
            if hasattr(image, 'PerFrameFunctionalGroupsSequence'):
                plane_positions = [
                    item.PlanePositionSlideSequence
                    for item in image.PerFrameFunctionalGroupsSequence
                ]
            else:
                # If Dimension Organization Type is TILED_FULL, plane
                # positions are implicit and need to be computed.
                plane_positions = compute_plane_position_slide_per_frame(image)
        else:
            plane_positions = [
                item.PlanePositionSequence
                for item in image.PerFrameFunctionalGroupsSequence
            ]

        return plane_positions

    def get_plane_positions_of_series(
        self,
        images: Sequence[Dataset]
    ) -> List[PlanePositionSequence]:
        """Gets plane positions for series of single-frame images.

        Parameters
        ----------
        images: Sequence[Dataset]
            Series of single-frame images

        Returns
        -------
        List[highdicom.PlanePositionSequence]
            Plane position of each frame in the image

        """
        is_multiframe = any([hasattr(img, 'NumberOfFrames') for img in images])
        if is_multiframe:
            raise ValueError(
                'Argument "images" must be a series of single-frame images.'
            )

        if self._coordinate_system is None:
            raise ValueError(
                'Cannot calculate plane positions when images do not exist '
                'within a frame of reference.'
            )
        elif self._coordinate_system == CoordinateSystemNames.SLIDE:
            plane_positions = []
            for img in images:
                # Unfortunately, the image position is not specified relative to
                # the top left corner but to the center of the image.
                # Therefore, we need to compute the offset and subtract it.
                center_item = img.ImageCenterPointCoordinatesSequence[0]
                x_center = center_item.XOffsetInSlideCoordinateSystem
                y_center = center_item.YOffsetInSlideCoordinateSystem
                z_center = center_item.ZOffsetInSlideCoordinateSystem
                offset_coordinate = map_pixel_into_coordinate_system(
                    index=((img.Columns / 2, img.Rows / 2)),
                    image_position=(x_center, y_center, z_center),
                    image_orientation=img.ImageOrientationSlide,
                    pixel_spacing=img.PixelSpacing
                )
                center_coordinate = np.array((0., 0., 0.), dtype=float)
                origin_coordinate = center_coordinate - offset_coordinate
                plane_positions.append(
                    PlanePositionSequence(
                        coordinate_system=CoordinateSystemNames.SLIDE,
                        image_position=origin_coordinate,
                        pixel_matrix_position=(1, 1)
                    )
                )
        else:
            plane_positions = [
                PlanePositionSequence(
                    coordinate_system=CoordinateSystemNames.PATIENT,
                    image_position=img.ImagePositionPatient
                )
                for img in images
            ]

        return plane_positions

    def get_index_position(self, pointer: str) -> int:
        """Get relative position of a given dimension in the dimension index.

        Parameters
        ----------
        pointer: str
            Name of the dimension (keyword of the attribute),
            e.g., ``"ReferencedSegmentNumber"``

        Returns
        -------
        int
            Zero-based relative position

        Examples
        --------
        >>> dimension_index = DimensionIndexSequence("SLIDE")
        >>> i = dimension_index.get_index_position("ReferencedSegmentNumber")
        >>> dimension_description = dimension_index[i]
        >>> dimension_description
        (0020, 9164) Dimension Organization UID          ...
        (0020, 9165) Dimension Index Pointer             AT: (0062, 000b)
        (0020, 9167) Functional Group Pointer            AT: (0062, 000a)
        (0020, 9421) Dimension Description Label         LO: 'Segment Number'

        """
        indices = [
            i
            for i, indexer in enumerate(self)
            if indexer.DimensionIndexPointer == tag_for_keyword(pointer)
        ]
        if len(indices) == 0:
            raise ValueError(
                f'Dimension index does not contain a dimension "{pointer}".'
            )
        return indices[0]

    def get_index_values(
        self,
        plane_positions: Sequence[PlanePositionSequence]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get values of indexed attributes that specify position of planes.

        Parameters
        ----------
        plane_positions: Sequence[highdicom.PlanePositionSequence]
            Plane position of frames in a multi-frame image or in a series of
            single-frame images

        Returns
        -------
        dimension_index_values: numpy.ndarray
            2D array of dimension index values
        plane_indices: numpy.ndarray
            1D array of planes indices for sorting frames according to their
            spatial position specified by the dimension index

        Note
        ----
        Includes only values of indexed attributes that specify the spatial
        position of planes relative to the total pixel matrix or the frame of
        reference, and excludes values of the Referenced Segment Number
        attribute.

        """
        if self._coordinate_system is None:
            raise RuntimeError(
                'Cannot calculate index values for multiple plane '
                'positions when images do not exist within a frame of '
                'reference.'
            )

        # For each dimension other than the Referenced Segment Number,
        # obtain the value of the attribute that the Dimension Index Pointer
        # points to in the element of the Plane Position Sequence or
        # Plane Position Slide Sequence.
        # Per definition, this is the Image Position Patient attribute
        # in case of the patient coordinate system, or the
        # X/Y/Z Offset In Slide Coordinate System and the Column/Row
        # Position in Total Image Pixel Matrix attributes in case of the
        # the slide coordinate system.
        plane_position_values = np.array([
            [
                np.array(p[0][indexer.DimensionIndexPointer].value)
                for indexer in self[1:]
            ]
            for p in plane_positions
        ])

        # Build an array that can be used to sort planes according to the
        # Dimension Index Value based on the order of the items in the
        # Dimension Index Sequence.
        _, plane_sort_indices = np.unique(
            plane_position_values,
            axis=0,
            return_index=True
        )

        return (plane_position_values, plane_sort_indices)

    def get_index_keywords(self) -> List[str]:
        """Get keywords of attributes that specify the position of planes.

        Returns
        -------
        List[str]
            Keywords of indexed attributes

        Note
        ----
        Includes only keywords of indexed attributes that specify the spatial
        position of planes relative to the total pixel matrix or the frame of
        reference, and excludes the keyword of the Referenced Segment Number
        attribute.

        Examples
        --------
        >>> dimension_index = DimensionIndexSequence('SLIDE')
        >>> plane_positions = [
        ...     PlanePositionSequence('SLIDE', [10.0, 0.0, 0.0], [1, 1]),
        ...     PlanePositionSequence('SLIDE', [30.0, 0.0, 0.0], [1, 2]),
        ...     PlanePositionSequence('SLIDE', [50.0, 0.0, 0.0], [1, 3])
        ... ]
        >>> values, indices = dimension_index.get_index_values(plane_positions)
        >>> names = dimension_index.get_index_keywords()
        >>> for name in names:
        ...     print(name)
        ColumnPositionInTotalImagePixelMatrix
        RowPositionInTotalImagePixelMatrix
        XOffsetInSlideCoordinateSystem
        YOffsetInSlideCoordinateSystem
        ZOffsetInSlideCoordinateSystem
        >>> index = names.index("XOffsetInSlideCoordinateSystem")
        >>> print(values[:, index])
        [10. 30. 50.]

        """
        return [
            keyword_for_tag(indexer.DimensionIndexPointer)
            for indexer in self[1:]
        ]
