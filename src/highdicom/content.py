"""Generic Data Elements that can be included in a variety of IODs."""
from collections import Counter
import datetime
from copy import deepcopy
from typing import Any, cast, Dict, List, Optional, Union, Sequence, Tuple

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DataElementSequence
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes
from pydicom.valuerep import DS, format_number_as_ds
from pydicom._storage_sopclass_uids import SegmentationStorage

from highdicom.enum import (
    CoordinateSystemNames,
    VOILUTFunctionValues,
    UniversalEntityIDTypeValues,
    RescaleTypeValues
)
from highdicom.sr.coding import CodedConcept
from highdicom.sr.value_types import (
    CodeContentItem,
    ContentSequence,
    DateTimeContentItem,
    NumContentItem,
    TextContentItem,
)
from highdicom.uid import UID
from highdicom.valuerep import (
    _check_long_string,
    _check_long_text,
    _check_short_text
)
from highdicom._module_utils import (
    check_required_attributes,
    iod_has_pixel_data
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
        family: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Kind of algorithm family
        version: str
            Version of the algorithm
        source: str, optional
            Source of the algorithm, e.g. name of the algorithm manufacturer
        parameters: Dict[str, str], optional
            Name and actual value of the parameters with which the algorithm
            was invoked

        """  # noqa: E501
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

    @classmethod
    def from_sequence(
        cls,
        sequence: DataElementSequence
    ) -> 'AlgorithmIdentificationSequence':
        """Construct instance from an existing data element sequence.

        Parameters
        ----------
        sequence: pydicom.sequence.Sequence
            Data element sequence representing the
            AlgorithmIdentificationSequence Sequence.

        Returns
        -------
        highdicom.seg.content.AlgorithmIdentificationSequence
            Algorithm identification sequence.

        """
        if not isinstance(sequence, DataElementSequence):
            raise TypeError(
                'Sequence should be of type pydicom.sequence.Sequence.'
            )
        if len(sequence) != 1:
            raise ValueError('Sequence should contain a single item.')
        check_required_attributes(
            sequence[0],
            module='segmentation-image',
            base_path=[
                'SegmentSequence',
                'SegmentationAlgorithmIdentificationSequence'
            ]
        )
        algo_id_sequence = deepcopy(sequence)
        algo_id_sequence.__class__ = AlgorithmIdentificationSequence
        return cast(AlgorithmIdentificationSequence, algo_id_sequence)

    @property
    def name(self) -> str:
        """str: Name of the algorithm."""
        return self[0].AlgorithmName

    @property
    def family(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: Kind of the algorithm family."""
        return CodedConcept.from_dataset(
            self[0].AlgorithmFamilyCodeSequence[0]
        )

    @property
    def version(self) -> str:
        """str: Version of the algorithm."""
        return self[0].AlgorithmVersion

    @property
    def source(self) -> Optional[str]:
        """Union[str, None]:
               Source of the algorithm, e.g. name of the algorithm
               manufacturer, if any

        """
        return getattr(self[0], 'AlgorithmSource', None)

    @property
    def parameters(self) -> Optional[Dict[str, str]]:
        """Union[Dict[str, str], None]:
               Dictionary mapping algorithm parameter names to values,
               if any

        """
        if not hasattr(self[0], 'AlgorithmParameters'):
            return None
        parameters = {}
        for param in self[0].AlgorithmParameters.split(','):
            split = param.split('=')
            if len(split) != 2:
                raise ValueError('Malformed parameter string')
            parameters[split[0]] = split[1]
        return parameters


class ContentCreatorIdentificationCodeSequence(DataElementSequence):

    """Sequence identifying the person who created the content."""
    def __init__(
        self,
        person_identification_codes: Sequence[Union[Code, CodedConcept]],
        institution_name: str,
        person_address: Optional[str] = None,
        person_telephone_numbers: Optional[Sequence[str]] = None,
        person_telecom_information: Optional[str] = None,
        institution_code: Union[Code, CodedConcept, None] = None,
        institution_address: Optional[str] = None,
        institutional_department_name: Optional[str] = None,
        institutional_department_type_code: Union[
            Code,
            CodedConcept,
            None
        ] = None,
    ):
        """

        Parameters
        ----------
        person_identification_codes: Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None]]
            Coded description(s) identifying the person.
        institution_name: str
            Name of the to which the identified individual is responsible or
            accountable.
        person_address: Union[str, None]
            Mailing address of the person.
        person_telephone_numbers: Union[Sequence[str], None], optional
            Person's telephone number(s).
        person_telecom_information: Union[str, None], optional
            The person's telecommunication contact information, including
            email or other addresses.
        institution_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Coded concept identifying the institution.
        institution_address: Union[str, None], optional
            Mailing address of the institution.
        institutional_department_name: Union[str, None], optional
            Name of the department, unit or service within the healthcare
            facility.
        institutional_department_type_code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            A coded description of the type of Department or Service.

        """  # noqa: E501
        super().__init__()
        item = Dataset()

        if len(person_identification_codes) < 1:
            raise ValueError(
                'Sequence "person_identification_codes" must not be empty.'
            )
        for name_code in person_identification_codes:
            if not isinstance(name_code, (Code, CodedConcept)):
                raise TypeError(
                    'Items of "person_identification_codes" must be of type '
                    'pydicom.sr.coding.Code or '
                    'highdicom.sr.CodedConcept.'
                )
        item.PersonIdentificationCodeSequence = [
            CodedConcept.from_code(c) for c in person_identification_codes
        ]

        _check_long_string(institution_name)
        item.InstitutionName = institution_name

        if institution_code is not None:
            if not isinstance(institution_code, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "institution_code" must be of type '
                    'pydicom.sr.coding.Code or '
                    'highdicom.sr.CodedConcept.'
                )
            item.InstitutionCodeSequence = [institution_code]

        if person_address is not None:
            _check_short_text(person_address)
            item.PersonAddress = person_address

        if person_telephone_numbers is not None:
            if len(person_telephone_numbers) < 1:
                raise ValueError(
                    'Sequence "person_telephone_numbers" must not be empty.'
                )
            for phone_number in person_telephone_numbers:
                _check_long_string(phone_number)
            item.PersonTelephoneNumbers = person_telephone_numbers

        if person_telecom_information is not None:
            _check_long_text(person_telecom_information)
            item.PersonTelecomInformation = person_telecom_information

        if institution_address is not None:
            _check_short_text(institution_address)
            item.InstitutionAddress = institution_address

        if institutional_department_name is not None:
            _check_long_string(institutional_department_name)
            item.InstitutionalDepartmentName = institutional_department_name

        if institutional_department_type_code is not None:
            if not isinstance(
                institutional_department_type_code,
                (Code, CodedConcept)
            ):
                raise TypeError(
                    'Argument "institutional_department_type_code" must be of '
                    'type pydicom.sr.coding.Code or '
                    'highdicom.sr.CodedConcept.'
                )
            item.InstitutionalDepartmentTypeCodeSequence = [
                CodedConcept.from_code(institutional_department_type_code)
            ]

        self.append(item)


class PixelMeasuresSequence(DataElementSequence):

    """Sequence of data elements describing physical spacing of an image based
    on the Pixel Measures functional group macro.
    """

    def __init__(
        self,
        pixel_spacing: Sequence[float],
        slice_thickness: float,
        spacing_between_slices: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        pixel_spacing: Sequence[float]
            Distance in physical space between neighboring pixels in
            millimeters along the row and column dimension of the image. First
            value represents the spacing between rows (vertical) and second
            value represents the spacing between columns (horizontal).
        slice_thickness: float
            Depth of physical space volume the image represents in millimeter
        spacing_between_slices: float, optional
            Distance in physical space between two consecutive images in
            millimeters. Only required for certain modalities, such as MR.

        """
        super().__init__()
        item = Dataset()
        item.PixelSpacing = [DS(ps, auto_format=True) for ps in pixel_spacing]
        item.SliceThickness = slice_thickness
        if spacing_between_slices is not None:
            item.SpacingBetweenSlices = spacing_between_slices
        self.append(item)

    @classmethod
    def from_sequence(
        cls,
        sequence: DataElementSequence
    ) -> 'PixelMeasuresSequence':
        """Create a PixelMeasuresSequence from an existing Sequence.

        Parameters
        ----------
        sequence: pydicom.sequence.Sequence
            Sequence to be converted.

        Returns
        -------
        highdicom.PixelMeasuresSequence
            Plane position sequence.

        Raises
        ------
        TypeError:
            If sequence is not of the correct type.
        ValueError:
            If sequence does not contain exactly one item.
        AttributeError:
            If sequence does not contain the attributes required for a
            pixel measures sequence.

        """
        if not isinstance(sequence, DataElementSequence):
            raise TypeError(
                'Sequence must be of type pydicom.sequence.Sequence'
            )
        if len(sequence) != 1:
            raise ValueError('Sequence must contain a single item.')
        req_kws = ['SliceThickness', 'PixelSpacing']
        if not all(hasattr(sequence[0], kw) for kw in req_kws):
            raise AttributeError(
                'Sequence does not have the required attributes for '
                'a Pixel Measures Sequence.'
            )

        pixel_measures = deepcopy(sequence)
        pixel_measures.__class__ = PixelMeasuresSequence
        return cast(PixelMeasuresSequence, pixel_measures)


class PlanePositionSequence(DataElementSequence):

    """Sequence of data elements describing the position of an individual plane
    (frame) in the patient coordinate system based on the Plane Position
    (Patient) functional group macro or in the slide coordinate system based
    on the Plane Position (Slide) functional group macro.
    """

    def __init__(
        self,
        coordinate_system: Union[str, CoordinateSystemNames],
        image_position: Sequence[float],
        pixel_matrix_position: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.CoordinateSystemNames]
            Frame of reference coordinate system
        image_position: Sequence[float]
            Offset of the first row and first column of the plane (frame) in
            millimeter along the x, y, and z axis of the three-dimensional
            patient or slide coordinate system
        pixel_matrix_position: Tuple[int, int], optional
            Offset of the first column and first row of the plane (frame) in
            pixels along the row and column direction of the total pixel matrix
            (only required if `coordinate_system` is ``"SLIDE"``)

        Note
        ----
        The values of both `image_position` and `pixel_matrix_position` are
        one-based.

        """
        super().__init__()
        item = Dataset()

        coordinate_system = CoordinateSystemNames(coordinate_system)
        if coordinate_system == CoordinateSystemNames.SLIDE:
            if pixel_matrix_position is None:
                raise TypeError(
                    'Position in Pixel Matrix must be specified for '
                    'slide coordinate system.'
                )
            col_position, row_position = pixel_matrix_position
            x, y, z = image_position
            item.XOffsetInSlideCoordinateSystem = DS(x, auto_format=True)
            item.YOffsetInSlideCoordinateSystem = DS(y, auto_format=True)
            item.ZOffsetInSlideCoordinateSystem = DS(z, auto_format=True)
            item.RowPositionInTotalImagePixelMatrix = row_position
            item.ColumnPositionInTotalImagePixelMatrix = col_position
        elif coordinate_system == CoordinateSystemNames.PATIENT:
            item.ImagePositionPatient = [
                DS(ip, auto_format=True) for ip in image_position
            ]
        else:
            raise ValueError(
                f'Unknown coordinate system "{coordinate_system.value}".'
            )
        self.append(item)

    def __eq__(self, other: Any) -> bool:
        """Determines whether two image planes have the same position.

        Parameters
        ----------
        other: highdicom.PlanePositionSequence
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
                    other[0].RowPositionInTotalImagePixelMatrix,
                    other[0].ColumnPositionInTotalImagePixelMatrix,
                ]),
                np.array([
                    self[0].XOffsetInSlideCoordinateSystem,
                    self[0].YOffsetInSlideCoordinateSystem,
                    self[0].ZOffsetInSlideCoordinateSystem,
                    self[0].RowPositionInTotalImagePixelMatrix,
                    self[0].ColumnPositionInTotalImagePixelMatrix,
                ]),
            )

    @classmethod
    def from_sequence(
        cls,
        sequence: DataElementSequence
    ) -> 'PlanePositionSequence':
        """Create a PlanePositionSequence from an existing Sequence.

        The coordinate system is inferred from the attributes in the sequence.

        Parameters
        ----------
        sequence: pydicom.sequence.Sequence
            Sequence to be converted.

        Returns
        -------
        highdicom.PlanePositionSequence:
            Plane position sequence.

        Raises
        ------
        TypeError:
            If sequence is not of the correct type.
        ValueError:
            If sequence does not contain exactly one item.
        AttributeError:
            If sequence does not contain the attributes required for a
            plane position sequence.

        """
        if not isinstance(sequence, DataElementSequence):
            raise TypeError(
                'Sequence must be of type pydicom.sequence.Sequence'
            )
        if len(sequence) != 1:
            raise ValueError('Sequence must contain a single item.')
        if not hasattr(sequence[0], 'ImagePositionPatient'):
            check_required_attributes(
                dataset=sequence[0],
                module='segmentation-multi-frame-functional-groups',
                base_path=[
                    'PerFrameFunctionalGroupsSequence',
                    'PlanePositionSlideSequence'
                ]
            )

        plane_position = deepcopy(sequence)
        plane_position.__class__ = PlanePositionSequence
        return cast(PlanePositionSequence, plane_position)


class PlaneOrientationSequence(DataElementSequence):

    """Sequence of data elements describing the image position in the patient
    or slide coordinate system based on either the Plane Orientation (Patient)
    or the Plane Orientation (Slide) functional group macro, respectively.
    """

    def __init__(
        self,
        coordinate_system: Union[str, CoordinateSystemNames],
        image_orientation: Sequence[float]
    ) -> None:
        """
        Parameters
        ----------
        coordinate_system: Union[str, highdicom.CoordinateSystemNames]
            Frame of reference coordinate system
        image_orientation: Sequence[float]
            Direction cosines for the first row (first triplet) and the first
            column (second triplet) of an image with respect to the X, Y, and Z
            axis of the three-dimensional coordinate system

        """
        super().__init__()
        item = Dataset()
        coordinate_system = CoordinateSystemNames(coordinate_system)
        image_orientation_ds = [
            DS(io, auto_format=True) for io in image_orientation
        ]
        if coordinate_system == CoordinateSystemNames.SLIDE:
            item.ImageOrientationSlide = image_orientation_ds
        elif coordinate_system == CoordinateSystemNames.PATIENT:
            item.ImageOrientationPatient = image_orientation_ds
        else:
            raise ValueError(
                f'Unknown coordinate system "{coordinate_system.value}".'
            )
        self.append(item)

    def __eq__(self, other: Any) -> bool:
        """Determines whether two image planes have the same orientation.

        Parameters
        ----------
        other: highdicom.PlaneOrientationSequence
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

    @classmethod
    def from_sequence(
        cls,
        sequence: DataElementSequence
    ) -> 'PlaneOrientationSequence':
        """Create a PlaneOrientationSequence from an existing Sequence.

        The coordinate system is inferred from the attributes in the sequence.

        Parameters
        ----------
        sequence: pydicom.sequence.Sequence
            Sequence to be converted.

        Returns
        -------
        highdicom.PlaneOrientationSequence:
            Plane orientation sequence.

        Raises
        ------
        TypeError:
            If sequence is not of the correct type.
        ValueError:
            If sequence does not contain exactly one item.
        AttributeError:
            If sequence does not contain the attributes required for a
            plane orientation sequence.

        """
        if not isinstance(sequence, DataElementSequence):
            raise TypeError(
                'Sequence must be of type pydicom.sequence.Sequence'
            )
        if len(sequence) != 1:
            raise ValueError('Sequence must contain a single item.')
        if not hasattr(sequence[0], 'ImageOrientationPatient'):
            if not hasattr(sequence[0], 'ImageOrientationSlide'):
                raise AttributeError(
                    'The sequence does not contain required attributes for '
                    'either the PATIENT or SLIDE coordinate system.'
                )

        plane_orientation = deepcopy(sequence)
        plane_orientation.__class__ = PlaneOrientationSequence
        return cast(PlaneOrientationSequence, plane_orientation)


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

        """  # noqa: E501
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

    """Sequence of SR content item describing a specimen collection procedure.
    """

    def __init__(
        self,
        procedure: Union[Code, CodedConcept]
    ):
        """
        Parameters
        ----------
        procedure: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Procedure used to collect the examined specimen

        """  # noqa: E501
        super().__init__(is_root=True)
        item = CodeContentItem(
            name=codes.SCT.SpecimenCollection,
            value=procedure
        )
        self.append(item)


class SpecimenSampling(ContentSequence):

    """Sequence of SR content item describing a specimen sampling procedure.

    See SR template
    :dcm:`TID 8002 Specimen Sampling <part16/chapter_C.html#sect_TID_8002>`.

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
        method: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Method used to sample the examined specimen from a parent specimen
        parent_specimen_id: str
            Identifier of the parent specimen
        parent_specimen_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Type of the parent specimen
        issuer_of_parent_specimen_id: highdicom.IssuerOfIdentifier, optional
            Issuer who created the parent specimen

        """  # noqa: E501
        super().__init__(is_root=True)
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

    """Sequence of SR content item describing a specimen staining procedure

    See SR template
    :dcm:`TID 8003 Specimen Staining <part16/chapter_C.html#sect_TID_8003>`.

    """

    def __init__(
        self,
        substances: Sequence[Union[Code, CodedConcept, str]]
    ):
        """
        Parameters
        ----------
        substances: Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, str]]
            Substances used to stain examined specimen(s)

        """  # noqa: E501
        super().__init__(is_root=True)
        # CID 8112
        for s in substances:
            if isinstance(s, (Code, CodedConcept)):
                item = CodeContentItem(
                    name=codes.SCT.UsingSubstance,
                    value=s
                )
            elif isinstance(s, str):
                item = TextContentItem(
                    name=codes.SCT.UsingSubstance,
                    value=s
                )
            else:
                raise TypeError(
                    'Items of argument "substances" must have type '
                    'CodedConcept, Code, or str.'
                )
            self.append(item)


class SpecimenProcessing(ContentSequence):

    """Sequence of SR content item describing a specimen processing procedure.
    """

    def __init__(
        self,
        description: Union[Code, CodedConcept]
    ):
        """
        Parameters
        ----------
        description: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Description of the processing

        """  # noqa: E501
        super().__init__(is_root=True)
        # CID 8112
        item = CodeContentItem(
            name=codes.DCM.ProcessingStepDescription,
            value=description
        )
        self.append(item)


class SpecimenPreparationStep(ContentSequence):

    """Dataset describing a specimen preparation step according to structured
    reporting template
    :dcm:`TID 8001 Specimen Preparation <part16/chapter_C.html#sect_TID_8001>`.

    """

    def __init__(
        self,
        specimen_id: str,
        processing_procedure: Union[
            SpecimenCollection,
            SpecimenSampling,
            SpecimenStaining,
            SpecimenProcessing,
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
        processing_procedure: Union[highdicom.SpecimenCollection, highdicom.SpecimenSampling, highdicom.SpecimenStaining]
            Procedure used during processing
        processing_datetime: datetime.datetime, optional
            Datetime of processing
        processing_description: Union[str, pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
            Description of processing
        issuer_of_specimen_id: highdicom.IssuerOfIdentifier, optional
        fixative: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
            Fixative used during processing
        embedding_medium: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept], optional
            Embedding medium used during processing

        """  # noqa: E501
        super().__init__(is_root=True)
        specimen_identifier_item = TextContentItem(
            name=codes.DCM.SpecimenIdentifier,
            value=specimen_id
        )
        self.append(specimen_identifier_item)
        if issuer_of_specimen_id is not None:
            try:
                entity_id = issuer_of_specimen_id.UniversalEntityID
            except AttributeError:
                entity_id = issuer_of_specimen_id.LocalNamespaceEntityID
            issuer_of_specimen_id_item = TextContentItem(
                name=codes.DCM.IssuerOfSpecimenIdentifier,
                value=entity_id
            )
            self.append(issuer_of_specimen_id_item)

        if isinstance(processing_procedure, SpecimenCollection):
            processing_type = codes.SCT.SpecimenCollection
        elif isinstance(processing_procedure, SpecimenProcessing):
            processing_type = codes.SCT.SpecimenProcessing
        elif isinstance(processing_procedure, SpecimenStaining):
            processing_type = codes.SCT.Staining
        elif isinstance(processing_procedure, SpecimenSampling):
            processing_type = codes.SCT.SamplingOfTissueSpecimen
        else:
            raise TypeError(
                'Argument "processing_procedure" must have type '
                'SpecimenCollection, SpecimenSampling, SpecimenProcessing, '
                'or SpecimenStaining.'
            )

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
            processing_description_item: Union[
                TextContentItem,
                CodeContentItem,
            ]
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

        self.extend(processing_procedure)
        if fixative is not None:
            tissue_fixative_item = CodeContentItem(
                name=codes.SCT.TissueFixative,
                value=fixative
            )
            self.append(tissue_fixative_item)
        if embedding_medium is not None:
            embedding_medium_item = CodeContentItem(
                name=codes.SCT.TissueEmbeddingMedium,
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
        issuer_of_specimen_id: Optional[IssuerOfIdentifier] = None,
        primary_anatomic_structures: Optional[
            Sequence[Union[Code, CodedConcept]]
        ] = None
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
            provided either in form of text or in form of spatial X, Y, Z
            coordinates specifying the position (offset) relative to the
            three-dimensional slide coordinate system in millimeter (X, Y) and
            micrometer (Z) unit.
        specimen_preparation_steps: Sequence[highdicom.SpecimenPreparationStep], optional
            Steps that were applied during the preparation of the examined
            specimen in the laboratory prior to image acquisition
        issuer_of_specimen_id: highdicom.IssuerOfIdentifier, optional
            Description of the issuer of the specimen identifier
        primary_anatomic_structures: Sequence[Union[pydicom.sr.Code, highdicom.sr.CodedConcept]]
            Body site at which specimen was collected

        """  # noqa: E501
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
            loc_item: Union[TextContentItem, NumContentItem]
            loc_seq: List[Union[TextContentItem, NumContentItem]] = []
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
                units = (
                    codes.UCUM.Millimeter,
                    codes.UCUM.Millimeter,
                    codes.UCUM.Micrometer,
                )
                for i, coordinate in enumerate(specimen_location):
                    loc_item = NumContentItem(
                        name=names[i],
                        value=coordinate,
                        unit=units[i]
                    )
                    loc_seq.append(loc_item)
            self.SpecimenLocalizationContentItemSequence = loc_seq

        self.IssuerOfTheSpecimenIdentifierSequence: List[Dataset] = []
        if issuer_of_specimen_id is not None:
            self.IssuerOfTheSpecimenIdentifierSequence.append(
                issuer_of_specimen_id
            )

        if primary_anatomic_structures is not None:
            if not isinstance(primary_anatomic_structures, Sequence):
                raise TypeError(
                    'Argument "primary_anatomic_structures" must be a '
                    'sequence.'
                )
            if len(primary_anatomic_structures) == 0:
                raise ValueError(
                    'Argument "primary_anatomic_structures" must not be '
                    'empty.'
                )
            self.PrimaryAnatomicStructureSequence: List[Dataset] = []
            for structure in primary_anatomic_structures:
                if isinstance(structure, CodedConcept):
                    self.PrimaryAnatomicStructureSequence.append(structure)
                elif isinstance(structure, Code):
                    self.PrimaryAnatomicStructureSequence.append(
                        CodedConcept(*structure)
                    )
                else:
                    raise TypeError(
                        'Items of argument "primary_anatomic_structures" '
                        'must have type Code or CodedConcept.'
                    )


class ReferencedImageSequence(DataElementSequence):

    """Sequence describing references to images, frames and/or segments."""

    def __init__(
        self,
        referenced_images: Optional[Sequence[Dataset]] = None,
        referenced_frame_number: Union[int, Sequence[int], None] = None,
        referenced_segment_number: Union[int, Sequence[int], None] = None,
    ):
        """

        Parameters
        ----------
        referenced_images: Union[Sequence[pydicom.Dataset], None], optional
            Images to which the VOI LUT described in this dataset applies. Note
            that if unspecified, the VOI LUT applies to every image referenced
            in the presentation state object that this dataset is included in.
        referenced_frame_number: Union[int, Sequence[int], None], optional
            Frame number(s) within a referenced multiframe image to which this
            VOI LUT applies.
        referenced_segment_number: Union[int, Sequence[int], None], optional
            Segment number(s) within a referenced segmentation image to which
            this VOI LUT applies.

        """
        super().__init__()

        if len(referenced_images) == 0:
            raise ValueError(
                'Argument "referenced_images" must not be empty.'
            )

        # Check for duplicate instances
        sop_uid_counts = Counter(
            ins.SOPInstanceUID for ins in referenced_images
        )
        most_common, max_count = sop_uid_counts.most_common(1)[0]
        if max_count > 1:
            raise ValueError("Found duplicate instances in referenced images.")

        multiple_images = len(referenced_images) > 1
        if referenced_frame_number is not None:
            if multiple_images:
                raise ValueError(
                    'Specifying "referenced_frame_number" is not supported '
                    'with multiple referenced images.'
                )
            if not hasattr(referenced_images[0], 'NumberOfFrames'):
                raise TypeError(
                    'Specifying "referenced_frame_number" is not valid '
                    'when the referenced image is not a multi-frame image.'
                )
            if isinstance(referenced_frame_number, Sequence):
                _referenced_frame_numbers = referenced_frame_number
            else:
                _referenced_frame_numbers = [referenced_frame_number]
            for f in _referenced_frame_numbers:
                if f < 1 or f > referenced_images[0].NumberOfFrames:
                    raise ValueError(
                        f'Frame number {f} is invalid for referenced '
                        'image.'
                    )
        if referenced_segment_number is not None:
            if multiple_images:
                raise ValueError(
                    'Specifying "referenced_segment_number" is not '
                    'supported with multiple referenced images.'
                )
            if referenced_images[0].SOPClassUID != SegmentationStorage:
                raise TypeError(
                    '"referenced_segment_number" is only valid when the '
                    'referenced image is a segmentation image.'
                )
            number_of_segments = len(referenced_images[0].SegmentSequence)
            if isinstance(referenced_segment_number, Sequence):
                _referenced_segment_numbers = referenced_segment_number
            else:
                _referenced_segment_numbers = [referenced_segment_number]
            for s in _referenced_segment_numbers:
                if s < 1 or s > number_of_segments:
                    raise ValueError(
                        f'Segment number {s} is invalid for referenced '
                        'image.'
                    )
            if referenced_frame_number is not None:
                # Check that the one of the specified segments exists
                # in each of the referenced frame
                for f in _referenced_frame_numbers:
                    f_ind = f - 1
                    seg_num = (
                        referenced_images[0].
                        PerFrameFunctionalGroupsSequence[f_ind].
                        SegmentIdentificationSequence[0].
                        ReferencedSegmentNumber
                    )
                    if seg_num not in _referenced_segment_numbers:
                        raise ValueError(
                            f'Referenced frame {f} does not contain any of '
                            'the referenced segments.'
                        )
        for im in referenced_images:
            if not iod_has_pixel_data(im.SOPClassUID):
                raise ValueError(
                    'Dataset provided in "referenced_images" does not '
                    'represent an image.'
                )
            ref_im = Dataset()
            ref_im.ReferencedSOPInstanceUID = im.SOPInstanceUID
            ref_im.ReferencedSOPClassUID = im.SOPClassUID
            if referenced_segment_number is not None:
                ref_im.ReferencedSegmentNumber = referenced_segment_number
            if referenced_frame_number is not None:
                ref_im.ReferencedFrameNumber = referenced_frame_number
            self.append(ref_im)


class LUT(Dataset):

    """Dataset describing a pixel value lookup table."""

    def __init__(
        self,
        first_mapped_value: int,
        lut_data: np.ndarray,
        lut_explanation: Optional[str] = None
    ):
        """

        Parameters
        ----------
        first_mapped_value: int
            Pixel value that will be mapped to the first value in the
            lookup-table.
        lut_data: np.ndarray
            Lookup table data. Must be of type uint8 or uint16.
        lut_explanation: Union[str, None], optional
            Free-form text explanation of the meaning of the LUT.

        Note
        ----
        After the LUT is applied, a pixel in the image with value equal to
        ``first_mapped_value`` is mapped to an output value of ``lut_data[0]``,
        an input value of ``first_mapped_value + 1`` is mapped to
        ``lut_data[1]``, and so on.

        """
        super().__init__()
        if not isinstance(first_mapped_value, int):
            raise TypeError('Argument "first_mapped_value" must be an integer.')
        if first_mapped_value < 0:
            raise ValueError(
                'Argument "first_mapped_value" must be non-negative.'
            )
        if first_mapped_value >= 2 ** 16:
            raise ValueError(
                'Argument "first_mapped_value" must be less than 2^16.'
            )

        if not isinstance(lut_data, np.ndarray):
            raise TypeError('Argument "lut_data" must be of type np.ndarray')
        if lut_data.ndim != 1:
            raise ValueError("Numpy array must have a single dimension.")
        len_data = lut_data.size
        if len_data == 0:
            raise ValueError('Argument "lut_data" must not be empty.')
        if len_data > 2**16:
            raise ValueError(
                'Length of lut_data must be no greater than 2^16 elements.'
            )
        elif len_data == 2**16:
            # Per the standard, this is recorded as 0
            len_data = 0
        if lut_data.dtype.type == np.uint8:
            bits_allocated = 8
        elif lut_data.dtype.type == np.uint16:
            bits_allocated = 16
        else:
            raise ValueError(
                "Numpy array must have dtype uint8 or uint16."
            )
        self.LUTData = lut_data.tobytes()

        self.LUTDescriptor = [
            len_data,
            first_mapped_value,
            bits_allocated
        ]

        if lut_explanation is not None:
            _check_long_string(lut_explanation)
            self.LUTExplanation = lut_explanation

    @property
    def lut_data(self) -> np.ndarray:
        """np.ndarray: LUT data formatted as np.ndarray."""
        if self.bits_allocated == 8:
            np_dtype = np.uint8
        elif self.bits_allocated == 16:
            np_dtype = np.uint16
        else:
            raise RuntimeError("Invalid LUT descriptor.")
        return np.frombuffer(self.LUTData, np_dtype)

    @property
    def first_mapped_value(self) -> int:
        """int: Pixel value that will be mapped to the first value in the
        LUT.
        """
        return self.LUTDescriptor[1]

    @property
    def bits_allocated(self) -> int:
        """int: Bits allocated for the LUT data. 8 or 16."""
        return self.LUTDescriptor[2]


class ModalityLUT(LUT):

    """Dataset describing a modality lookup table."""

    def __init__(
        self,
        modality_lut_type: Union[RescaleTypeValues, str],
        first_mapped_value: int,
        lut_data: np.ndarray,
        lut_explanation: Optional[str] = None
    ):
        """

        Parameters
        ----------
         modality_lut_type: Union[highdicom.RescaleTypeValues, str]
            String or enumerated value specifying the units of the output of
            the LUT operation.
          first_mapped_value: int
            Pixel value that will be mapped to the first value in the
            lookup-table.
        lut_data: np.ndarray
            Lookup table data. Must be of type uint8 or uint16.
        lut_explanation: Union[str, None], optional
            Free-form text explanation of the meaning of the LUT.

        Note
        ----
        After the LUT is applied, a pixel in the image with value equal to
        ``first_mapped_value`` is mapped to an output value of ``lut_data[0]``,
        an input value of ``first_mapped_value + 1`` is mapped to
        ``lut_data[1]``, and so on.

        """
        super().__init__(
            first_mapped_value=first_mapped_value,
            lut_data=lut_data,
            lut_explanation=lut_explanation
        )
        if isinstance(modality_lut_type, RescaleTypeValues):
            self.ModalityLUTType = modality_lut_type.value
        else:
            _check_long_string(modality_lut_type)
            self.ModalityLUTType = modality_lut_type


class VOILUT(Dataset):

    """Dataset describing a value-of-interest lookup table."""

    def __init__(
        self,
        window_center: Union[float, Sequence[float], None] = None,
        window_width: Union[float, Sequence[float], None] = None,
        window_explanation: Union[str, Sequence[str], None] = None,
        voi_lut_function: Union[VOILUTFunctionValues, str, None] = None,
        voi_luts: Optional[Sequence[LUT]] = None,
    ):
        """

        Parameters
        ----------
        window_center: Union[float, Sequence[float], None], optional
            Center value of the intensity window used for display.
        window_width: Union[float, Sequence[float], None], optional
            Width of the intensity window used for display.
        window_explanation: Union[str, Sequence[str], None], optional
            Free-form explanation of the window center and width.
        voi_lut_function: Union[highdicom.VOILUTFunctionValues, str, None], optional
            Description of the LUT function parametrized by ``window_center``.
            and ``window_width``.
        voi_luts: Union[Sequence[highdicom.LUT], None], optional
            Intensity lookup tables used for display.

        Note
        ----
        Either ``window_center`` and ``window_width`` should be provided or
        ``voi_luts`` should be provided, or both. ``window_explanation`` should
        only be provided if ``window_center`` is provided.

        """  # noqa: E501
        super().__init__()

        if window_center is not None:
            if window_width is None:
                raise TypeError(
                    'Providing "window_center" is invalid if "window_width" '
                    'is not provided.'
                )
            window_is_sequence = isinstance(window_center, Sequence)
            if window_is_sequence:
                if len(window_center) == 0:
                    raise TypeError(
                        'Argument "window_center" must not be an empty '
                        'sequence.'
                    )
                self.WindowCenter = [
                    format_number_as_ds(x) for x in window_center
                ]
            else:
                self.WindowCenter = format_number_as_ds(window_center)
        if window_width is not None:
            if window_center is None:
                raise TypeError(
                    'Providing "window_width" is invalid if "window_center" '
                    'is not provided.'
                )
            if isinstance(window_width, Sequence):
                if (
                    not window_is_sequence or
                    (len(window_width) != len(window_center))
                ):
                    raise ValueError(
                        'Length of "window_width" must match length of '
                        '"window_center".'
                    )
                if len(window_width) == 0:
                    raise TypeError(
                        'Argument "window_width" must not be an empty sequence.'
                    )
                self.WindowWidth = [
                    format_number_as_ds(x) for x in window_width
                ]
            else:
                if window_is_sequence:
                    raise TypeError(
                        'Length of "window_width" must match length of '
                        '"window_center".'
                    )
                self.WindowWidth = format_number_as_ds(window_width)
        if window_explanation is not None:
            if window_center is None:
                raise TypeError(
                    'Providing "window_explanation" is invalid if '
                    '"window_center" is not provided.'
                )
            if isinstance(window_explanation, str):
                if window_is_sequence:
                    raise TypeError(
                        'Length of "window_explanation" must match length of '
                        '"window_center".'
                    )
                _check_long_string(window_explanation)
            elif isinstance(window_explanation, Sequence):
                if (
                    not window_is_sequence or
                    (len(window_explanation) != len(window_center))
                ):
                    raise ValueError(
                        'Length of "window_explanation" must match length of '
                        '"window_center".'
                    )
                if len(window_explanation) == 0:
                    raise TypeError(
                        'Argument "window_explanation" must not be an empty '
                        'sequence.'
                    )
                for exp in window_explanation:
                    _check_long_string(exp)
            self.WindowCenterWidthExplanation = window_explanation
        if voi_lut_function is not None:
            if window_center is None:
                raise TypeError(
                    'Providing "voi_lut_function" is invalid if '
                    '"window_center" is not provided.'
                )
            self.VOILUTFunction = VOILUTFunctionValues(voi_lut_function).value

        if voi_luts is not None:
            if len(voi_luts) == 0:
                raise ValueError('"voi_luts" should not be empty.')
            for lut in voi_luts:
                if not isinstance(lut, LUT):
                    raise TypeError(
                        'Items of "voi_luts" should be of type highdicom.LUT.'
                    )
            self.VOILUTSequence = list(voi_luts)
        else:
            if window_center is None:
                raise TypeError(
                    'At least one of "window_center" or "voi_luts" should be '
                    'provided.'
                )


class PaletteColorLookupTable(Dataset):

    """Dataset describing a palette color lookup table."""

    def __init__(
        self,
        red_palette_color_lut_data: np.ndarray,
        green_palette_color_lut_data: np.ndarray,
        blue_palette_color_lut_data: np.ndarray,
        red_first_mapped_value: int,
        green_first_mapped_value: int,
        blue_first_mapped_value: int,
        palette_color_lut_uid: Union[UID, str, None] = None
    ):
        """Add attributes from the Palette Color Lookup Table module.

        Parameters
        ----------
        red_palette_color_lut_data: np.ndarray
            Array of values for the red color lookup table data. Must be a 1D
            array of uint16 values, where the first entry is the red output
            value of the palette color lookup table operation when the input
            pixel is ``"red_first_mapped_value"``, and so on.
        green_palette_color_lut_data: np.ndarray
            Array of values for the green color lookup table data. Otherwise as
            described for ``red_palette_color_lut_data``.
        blue_palette_color_lut_data: np.ndarray
            Array of values for the blue color lookup table data. Otherwise as
            described for ``red_palette_color_lut_data``.
        red_first_mapped_value: int
            Integer representing the first input value mapped by the red palette
            lookup table operation.
        green_first_mapped_value: int
            Integer representing the first input value mapped by the green
            lookup table operation.
        blue_first_mapped_value: int
            Integer representing the first input value mapped by the blue
            palette lookup table operation.
        palette_color_lut_uid: Union[UID, str, None], optional
            Unique identifier for the palette color lookup table.

        """
        super().__init__()
        colors = ['red', 'green', 'blue']
        all_lut_data = [
            red_palette_color_lut_data,
            green_palette_color_lut_data,
            blue_palette_color_lut_data
        ]
        all_first_values = [
            red_first_mapped_value,
            green_first_mapped_value,
            blue_first_mapped_value
        ]

        if palette_color_lut_uid is not None:
            self.PaletteColorLookupTableUID = palette_color_lut_uid

        for color, lut_data, first_mapped_value in zip(
            colors,
            all_lut_data,
            all_first_values
        ):
            if not isinstance(first_mapped_value, int):
                raise TypeError(
                    f'Argument "{color}_first_mapped_value" must be an integer.'
                )
            if first_mapped_value < 0:
                raise ValueError(
                    'Argument "first_mapped_value" must be non-negative.'
                )
            if first_mapped_value >= 2 ** 16:
                raise ValueError(
                    f'Argument "{color}_first_mapped_value" must be less than '
                    '2^16.'
                )

            if not isinstance(lut_data, np.ndarray):
                raise TypeError(
                    f'Argument "f{color}_palette_color_lut_data" must be of '
                    'type np.ndarray.'
                )
            if lut_data.ndim != 1:
                raise ValueError(
                    f'Argument "f{color}_palette_color_lut_data" '
                    'must have a single dimension.'
                )
            len_data = lut_data.size
            if len_data == 0:
                raise ValueError(
                    f'Argument "f{color}_palette_color_lut_data" '
                    'must not be empty.'
                )
            if len_data > 2**16:
                raise ValueError(
                    f'Length of "f{color}_palette_color_lut_data" must be no '
                    'greater than 2^16 elements.'
                )
            elif len_data == 2**16:
                # Per the standard, this is recorded as 0
                len_data = 0

            if lut_data.dtype.type != np.uint16:
                raise ValueError(
                    f'Argument "f{color}_palette_color_lut_data" must have '
                    'dtype uint16.'
                )

            descriptor = [
                len_data,
                first_mapped_value,
                16  # always 16 as part of Palette Color LUT module
            ]
            setattr(
                self,
                f'{color.title()}PaletteColorLookupTableDescriptor',
                descriptor
            )
            setattr(
                self,
                f'{color.title()}PaletteColorLookupTableData',
                lut_data.tobytes()
            )
