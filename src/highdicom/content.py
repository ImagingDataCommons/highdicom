"""Generic Data Elements that can be included in a variety of IODs."""
from typing import Dict, Optional, Sequence, Union, Tuple

import numpy as np
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code

from highdicom.enum import CoordinateSystemNames
from highdicom.seg.enum import SegmentAlgorithmTypes
from highdicom.sr.coding import CodedConcept


SLIDE_DIMENSION_ORGANIZATION_UID = '1.2.826.0.1.3680043.9.7433.2.4'

PATIENT_DIMENSION_ORGANIZATION_UID = '1.2.826.0.1.3680043.9.7433.2.3'


class AlgorithmIdentificationSequence(DataElementSequence):

    """Sequence of data elements describing information useful for
    identification of an algorithm.
    """

    def __init__(
            self,
            name: str,
            family: Union[Code, CodedConcept],
            version: str,
            source: Optional[str] = None,
            parameters: Optional[Dict[str, str]] = None
        ):
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
    (Patient) functional group macro or in the slide coordinate system based
    on the Plane Position (Slide) functional group macro.
    """

    def __init__(
            self,
            coordinate_system: Union[str, CoordinateSystemNames],
            image_position: Tuple[float, float, float],
            pixel_matrix_position: Optional[Tuple[int, int]] = None
        ) -> None:
        """
        Parameters
        ----------
        image_position: Tuple[float, float, float]
            Offset of the first row and first column of the plane (frame) in
            millimeter along the x, y, and z axis of the patient coordinate
            system
        pixel_matrix_position: Tuple[int, int], optional
            Offset of the first row and first column of the plane (frame) in
            pixels along the row and column direction of the total pixel matrix
            (only required if `coordinate_system` is ``"SLIDE"``)

        """
        super().__init__()
        item = Dataset()
        if coordinate_system == CoordinateSystemNames.SLIDE:
            if pixel_matrix_position is None:
                raise TypeError(
                    'Position in Pixel Matrix must be specified for '
                    'slide coordinate system.'
                )
            item.XOffsetInSlideCoordinateSystem = image_position[0]
            item.YOffsetInSlideCoordinateSystem = image_position[1]
            item.ZOffsetInSlideCoordinateSystem = image_position[2]
            item.RowPositionInTotalImagePixelMatrix = pixel_matrix_position[0]
            item.ColumnPositionInTotalImagePixelMatrix = pixel_matrix_position[1]
        elif coordinate_system == CoordinateSystemNames.PATIENT:
            item.ImagePositionPatient = list(image_position)
        self.append(item)

    def __eq__(self, other) -> bool:
        """Determines whether two image planes have the same position.

        Parameters
        ----------
        other: highdicom.content.PlanePositionSequence
            Plane position of other image that should be compared

        Returns
        -------
        bool
            Whether the two image planes have the same position

        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                'Can only compare image position between instances of '
                'class "{}".'.format(self.__class__.__name__)
            )
        if hasattr(self[0], 'ImagePositionPatient'):
            return np.array_equal(
                np.array(other[0].ImagePositionPatient),
                np.array(self[0].ImagePositionPatient)
            )
        else:
            return np.array_equal(
                np.array([
                    other[0].XOffsetInSlideCoordinateSystem,
                    other[0].YOffsetInSlideCoordinateSystem,
                    other[0].ZOffsetInSlideCoordinateSystem,
                ]),
                np.array([
                    self[0].XOffsetInSlideCoordinateSystem,
                    self[0].YOffsetInSlideCoordinateSystem,
                    self[0].ZOffsetInSlideCoordinateSystem,
                ]),
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
            Subject (``"PATIENT"`` or ``"SLIDE"``) that was the target of
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

    def __eq__(self, other) -> bool:
        """Determines whether two image planes have the same orientation.

        Parameters
        ----------
        other: highdicom.content.PlaneOrientationSequence
            Plane position of other image that should be compared

        Returns
        -------
        bool
            Whether the two image planes have the same orientation

        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                'Can only compare orientation between instances of '
                'class "{}".'.format(self.__class__.__name__)
            )
        if hasattr(self[0], 'ImageOrientationPatient'):
            if not hasattr(other[0], 'ImageOrientationPatient'):
                raise AttributeError(
                    'Can only compare orientation between images that '
                    'share the same coordinate system.'
                )
            return np.array_equal(
                np.array(other[0].ImageOrientationPatient),
                np.array(self[0].ImageOrientationPatient)
            )
        elif hasattr(self[0], 'ImageOrientationSlide'):
            if not hasattr(other[0], 'ImageOrientationSlide'):
                raise AttributeError(
                    'Can only compare orientations between images that '
                    'share the same coordinate system.'
                )
            return np.array_equal(
                np.array(other[0].ImageOrientationSlide),
                np.array(self[0].ImageOrientationSlide)
            )
        else:
            return False


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
