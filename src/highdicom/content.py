"""Generic Data Elements that can be included in a variety of IODs."""
import datetime
from typing import Any, Dict, List, Optional, Union, Sequence, Tuple

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes

from highdicom.enum import (
    CoordinateSystemNames,
    UniversalEntityIDTypeValues,
)
from highdicom.sr.coding import CodedConcept
from highdicom.sr.value_types import (
    CodeContentItem,
    ContentSequence,
    DateTimeContentItem,
    NumContentItem,
    TextContentItem,
)


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
        family: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]
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
            millimeter along the x, y, and z axis of the three-dimensional
            patient or slide coordinate system
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
            row_position, col_position = pixel_matrix_position
            item.XOffsetInSlideCoordinateSystem = image_position[0]
            item.YOffsetInSlideCoordinateSystem = image_position[1]
            item.ZOffsetInSlideCoordinateSystem = image_position[2]
            item.RowPositionInTotalImagePixelMatrix = row_position
            item.ColumnPositionInTotalImagePixelMatrix = col_position
        elif coordinate_system == CoordinateSystemNames.PATIENT:
            item.ImagePositionPatient = list(image_position)
        self.append(item)

    def __eq__(self, other: Any) -> bool:
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
            axis of the three-dimensional coordinate system

        """
        super().__init__()
        coordinate_system = CoordinateSystemNames(coordinate_system)
        item = Dataset()
        if coordinate_system == CoordinateSystemNames.SLIDE:
            item.ImageOrientationSlide = list(image_orientation)
        elif coordinate_system == CoordinateSystemNames.PATIENT:
            item.ImageOrientationPatient = list(image_orientation)
        self.append(item)

    def __eq__(self, other: Any) -> bool:
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


class IssuerOfIdentifier(Dataset):

    """Dataset describing the issuer or a specimen or container identifier."""

    def __init__(
            self,
            issuer_of_identifier: str,
            issuer_of_identifier_type: Optional[
                Union[str, UniversalEntityIDTypeValues]
            ] = None
        ):
        """
        Parameters
        ----------
        issuer_of_identifier: str
            Identifier of the entity that created the examined specimen
        issuer_of_identifier_type: Union[str, highdicom.enum.UniversalEntityIDTypeValues], optional
            Type of identifier of the entity that created the examined specimen
            (required if `issuer_of_specimen_id` is a Unique Entity ID)

        """  # noqa
        super().__init__()
        if issuer_of_identifier_type is None:
            self.LocalNamespaceEntityID = issuer_of_identifier
        else:
            self.UniversalEntityID = issuer_of_identifier
            issuer_of_identifier_type = UniversalEntityIDTypeValues(
                issuer_of_identifier_type
            )
            self.UniversalEntityIDType = issuer_of_identifier_type.value


class SpecimenCollection(ContentSequence):

    """Sequence of structured reporting content item describing a specimen
    collection procedure.
    """

    def __init__(
            self,
            procedure: Union[Code, CodedConcept]
        ):
        """
        Parameters
        ----------
        procedure: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]
            Procedure used to collect the examined specimen

        """  # noqa
        super().__init__()
        item = CodeContentItem(
            name=codes.SCT.SpecimenCollection,
            value=procedure
        )
        self.append(item)


class SpecimenSampling(ContentSequence):

    """Sequence of structured reporting content item describing a specimen
    sampling procedure according to structured reporting template TID 8002
    Specimen Sampling.
    """

    def __init__(
            self,
            method: Union[Code, CodedConcept],
            parent_specimen_id: str,
            parent_specimen_type: Union[Code, CodedConcept],
            issuer_of_parent_specimen_id: Optional[IssuerOfIdentifier] = None
        ):
        """
        Parameters
        ----------
        method: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]
            Method used to sample the examined specimen from a parent specimen
        parent_specimen_id: str
            Identifier of the parent specimen
        parent_specimen_type: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]
            Type of the parent specimen
        issuer_of_parent_specimen_id: highdicom.content.IssuerOfIdentifier, optional
            Issuer who created the parent specimen

        """  # noqa
        super().__init__()
        # CID 8110
        method_item = CodeContentItem(
            name=codes.DCM.SamplingMethod,
            value=method
        )
        self.append(method_item)
        parent_specimen_identitier_item = TextContentItem(
            name=codes.DCM.ParentSpecimenIdentifier,
            value=parent_specimen_id
        )
        self.append(parent_specimen_identitier_item)
        if issuer_of_parent_specimen_id is not None:
            try:
                entity_id = issuer_of_parent_specimen_id.UniversalEntityID
            except AttributeError:
                entity_id = issuer_of_parent_specimen_id.LocalNamespaceEntityID
            issuer_of_parent_specimen_identitier_item = TextContentItem(
                name=codes.DCM.IssuerOfParentSpecimenIdentifier,
                value=entity_id
            )
            self.append(issuer_of_parent_specimen_identitier_item)
        # CID 8103
        parent_specimen_type_item = CodeContentItem(
            name=codes.DCM.ParentSpecimenType,
            value=parent_specimen_type
        )
        self.append(parent_specimen_type_item)


class SpecimenStaining(ContentSequence):

    """Sequence of structured reporting content item describing a specimen
    staining procedure according to structured reporting template TID 8003
    Specimen Staining.
    """

    def __init__(
            self,
            substances: Sequence[Union[Code, CodedConcept]]
        ):
        """
        Parameters
        ----------
        substances: Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]]
            Substances used to stain examined specimen(s)

        """  # noqa
        super().__init__()
        # CID 8112
        for s in substances:
            item = CodeContentItem(
                name=codes.SCT.UsingSubstance,
                value=s
            )
            self.append(item)


class SpecimenPreparationStep(ContentSequence):

    """Dataset describing a specimen preparation step according to
    structured reporting template TID 8001 Specimen Preparation.
    """

    def __init__(
            self,
            specimen_id: str,
            processing_type: Union[Code, CodedConcept],
            processing_procedure: Union[
                SpecimenCollection,
                SpecimenSampling,
                SpecimenStaining,
            ],
            processing_description: Optional[
                Union[str, Code, CodedConcept]
            ] = None,
            processing_datetime: Optional[datetime.datetime] = None,
            issuer_of_specimen_id: Optional[IssuerOfIdentifier] = None,
            fixative: Optional[Union[Code, CodedConcept]] = None,
            embedding_medium: Optional[Union[Code, CodedConcept]] = None
        ):
        """
        Parameters
        ----------
        specimen_id: str
            Identifier of the processed specimen
        processing_type: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept]
            Type of processing
        processing_procedure: Union[highdicom.content.SpecimenCollection, highdicom.content.SpecimenSampling, highdicom.content.SpecimenStaining]
            Procedure used during processing
        processing_datetime: datetime.datetime, optional
            Datetime of processing
        processing_description: Union[str, pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept], optional
            Description of processing
        issuer_of_specimen_id: highdicom.content.IssuerOfIdentifier, optional
        fixative: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept], optional
            Fixative used during processing
        embedding_medium: Union[pydicom.sr.coding.Code, highdicom.sr.coding.CodedConcept], optional
            Embedding medium used during processing

        """  # noqa
        super().__init__()
        specimen_identifier_item = TextContentItem(
            name=codes.DCM.SpecimenIdentifier,
            value=specimen_id
        )
        self.append(specimen_identifier_item)
        if issuer_of_specimen_id is not None:
            self.append(issuer_of_specimen_id)
        # CID 8111
        processing_type_item = CodeContentItem(
            name=codes.DCM.ProcessingType,
            value=processing_type
        )
        self.append(processing_type_item)
        if processing_datetime is not None:
            processing_datetime_item = DateTimeContentItem(
                name=codes.DCM.DateTimeOfProcessing,
                value=processing_datetime
            )
            self.append(processing_datetime_item)
        if processing_description is not None:
            if isinstance(processing_description, str):
                processing_description_item = TextContentItem(
                    name=codes.DCM.ProcessingStepDescription,
                    value=processing_description
                )
            else:
                processing_description_item = CodeContentItem(
                    name=codes.DCM.ProcessingStepDescription,
                    value=processing_description
                )
            self.append(processing_description_item)
        accepted_procedure_types = (
            SpecimenCollection,
            SpecimenSampling,
            SpecimenStaining,
        )
        if not isinstance(processing_procedure, accepted_procedure_types):
            raise TypeError(
                'Unknown procedure of specimen preparation step.'
            )
        self.extend(processing_procedure)
        if fixative is not None:
            tissue_fixative_item = CodeContentItem(
                name=codes.SCT.TissueFixative,
                value=fixative
            )
            self.append(tissue_fixative_item)
        if embedding_medium is not None:
            embedding_medium_item = CodeContentItem(
                name=codes.SCT.EmbeddingMedium,
                value=embedding_medium
            )
            self.append(embedding_medium_item)


class SpecimenDescription(Dataset):

    """Dataset describing a specimen."""

    def __init__(
            self,
            specimen_id: str,
            specimen_uid: str,
            specimen_location: Optional[
                Union[str, Tuple[float, float, float]]
            ] = None,
            specimen_preparation_steps: Optional[
                Sequence[SpecimenPreparationStep]
            ] = None,
            issuer_of_specimen_id: Optional[IssuerOfIdentifier] = None
        ):
        """
        Parameters
        ----------
        specimen_id: str
            Identifier of the examined specimen
        specimen_uid: str
            Unique identifier of the examined specimen
        specimen_location: Union[str, Tuple[float, float, float]], optional
            Location of the examined specimen relative to the container
            provided either in form of text or in form of spatial x, y, z
            coordinates specifying the position (offset) relative to the
            three-dimensional slide coordinate system
        specimen_preparation_steps: Sequence[highdicom.content.SpecimenPreparationStep], optional
            Steps that were applied during the preparation of the examined
            specimen in the laboratory prior to image acquisition
        issuer_of_specimen_id: highdicom.content.IssuerOfIdentifier, optional
            Description of the issuer of the specimen identifier

        """  # noqa
        super().__init__()
        self.SpecimenIdentifier = specimen_id
        self.SpecimenUID = specimen_uid
        self.SpecimenPreparationSequence: List[Dataset] = []
        if specimen_preparation_steps is not None:
            for step in specimen_preparation_steps:
                if not isinstance(step, ContentSequence):
                    raise TypeError(
                        'Each specimen preparation step must be provided as '
                        'a sequence of content items.'
                    )
                step_item = Dataset()
                step_item.SpecimenPreparationStepContentItemSequence = step
                self.SpecimenPreparationSequence.append(step_item)
        if specimen_location is not None:
            loc_seq = []
            if isinstance(specimen_location, str):
                loc_item = TextContentItem(
                    name=codes.DCM.LocationOfSpecimen,
                    value=specimen_location
                )
                loc_seq.append(loc_item)
            elif isinstance(specimen_location, tuple):
                names = (
                    codes.DCM.LocationOfSpecimenXOffset,
                    codes.DCM.LocationOfSpecimenYOffset,
                    codes.DCM.LocationOfSpecimenZOffset,
                )
                for i, coordinate in enumerate(specimen_location):
                    loc_item = NumContentItem(
                        name=names[i],
                        value=coordinate,
                    )
                    loc_seq.append(loc_item)
            self.SpecimenLocalizationContentItemSequence = loc_seq
        self.IssuerOfTheSpecimenIdentifierSequence: List[Dataset] = []
        if issuer_of_specimen_id is not None:
            self.IssuerOfTheSpecimenIdentifierSequence.append(
                issuer_of_specimen_id
            )
