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
    PresentationLUTShapeValues,
    RescaleTypeValues,
    UniversalEntityIDTypeValues,
    VOILUTFunctionValues,
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
    does_iod_have_pixel_data
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
            Algorithm Identification Sequence

        Returns
        -------
        highdicom.seg.content.AlgorithmIdentificationSequence
            Algorithm Identification Sequence

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

    """Sequence of data elements for identifying the person who created content.

    """

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
        person_identification_codes: Sequence[Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]]
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
        slice_thickness: Optional[float],
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
        slice_thickness: Union[float, None]
            Depth of physical space volume the image represents in millimeter.
        spacing_between_slices: Union[float, None], optional
            Distance in physical space between two consecutive images in
            millimeters. Only required for certain modalities, such as MR.

        """
        super().__init__()
        item = Dataset()
        item.PixelSpacing = [DS(ps, auto_format=True) for ps in pixel_spacing]
        if slice_thickness is not None:
            item.SliceThickness = DS(slice_thickness, auto_format=True)
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
            Plane Measures Sequence.

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
            Plane Position Sequence.

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
            Plane Orientation Sequence.

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

    """Sequence of SR content items describing a specimen collection procedure.
    """

    def __init__(
        self,
        procedure: Union[Code, CodedConcept]
    ):
        """
        Parameters
        ----------
        procedure: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Surgical procedure used to collect the examined specimen

        """  # noqa: E501
        super().__init__(is_root=False, is_sr=False)
        item = CodeContentItem(
            name=codes.SCT.SpecimenCollection,
            value=procedure
        )
        self.append(item)

    @property
    def procedure(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: Surgical procedure"""
        items = self.find(codes.SCT.SpecimenCollection)
        if len(items) == 0:
            raise AttributeError(
                'Could not find content item "Specimen Collection".'
            )
        return items[0].value


class SpecimenSampling(ContentSequence):

    """Sequence of SR content items describing a specimen sampling procedure.

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
        super().__init__(is_root=False, is_sr=False)
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

    @property
    def method(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: Sampling method"""
        items = self.find(codes.DCM.SamplingMethod)
        if len(items) == 0:
            raise AttributeError(
                'Could not find content item "Sampling Method".'
            )
        return items[0].value

    @property
    def parent_specimen_id(self) -> str:
        """str: Parent specimen identifier"""
        items = self.find(codes.DCM.ParentSpecimenIdentifier)
        if len(items) == 0:
            raise AttributeError(
                'Could not find content item "Parent Specimen Identifier".'
            )
        return items[0].value

    @property
    def parent_specimen_type(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: Parent specimen type"""
        items = self.find(codes.DCM.ParentSpecimenType)
        if len(items) == 0:
            raise AttributeError(
                'Could not find content item "Parent Specimen Type".'
            )
        return items[0].value


class SpecimenStaining(ContentSequence):

    """Sequence of SR content items describing a specimen staining procedure

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
        super().__init__(is_root=False, is_sr=False)
        if len(substances) == 0:
            raise ValueError(
                'Argument "substances" must contain at least one item.'
            )
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

    @property
    def substances(self) -> List[CodedConcept]:
        """List[highdicom.sr.CodedConcept]: Substances used for staining"""
        items = self.find(codes.SCT.UsingSubstance)
        return [item.value for item in items]


class SpecimenProcessing(ContentSequence):

    """Sequence of SR content items describing a specimen processing procedure.

    """

    def __init__(
        self,
        description: Union[Code, CodedConcept, str]
    ):
        """
        Parameters
        ----------
        description: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, str]
            Description of the processing

        """  # noqa: E501
        super().__init__(is_root=False, is_sr=False)
        # CID 8112
        if isinstance(description, str):
            item = TextContentItem(
                name=codes.DCM.ProcessingStepDescription,
                value=description
            )
        else:
            item = CodeContentItem(
                name=codes.DCM.ProcessingStepDescription,
                value=description
            )
        self.append(item)

    @property
    def description(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: Processing step description"""
        items = self.find(codes.DCM.ProcessingStepDescription)
        if len(items) == 0:
            raise AttributeError(
                'Could not find content item "Processing Step Description".'
            )
        return items[0].value


class SpecimenPreparationStep(Dataset):

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
        processing_procedure: Union[highdicom.SpecimenCollection, highdicom.SpecimenSampling, highdicom.SpecimenStaining, highdicom.SpecimenProcessing]
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
        super().__init__()
        sequence = ContentSequence(is_root=False, is_sr=False)
        specimen_identifier_item = TextContentItem(
            name=codes.DCM.SpecimenIdentifier,
            value=specimen_id
        )
        sequence.append(specimen_identifier_item)
        if issuer_of_specimen_id is not None:
            try:
                entity_id = issuer_of_specimen_id.UniversalEntityID
            except AttributeError:
                entity_id = issuer_of_specimen_id.LocalNamespaceEntityID
            issuer_of_specimen_id_item = TextContentItem(
                name=codes.DCM.IssuerOfSpecimenIdentifier,
                value=entity_id
            )
            sequence.append(issuer_of_specimen_id_item)

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
        sequence.append(processing_type_item)

        if processing_datetime is not None:
            processing_datetime_item = DateTimeContentItem(
                name=codes.DCM.DateTimeOfProcessing,
                value=processing_datetime
            )
            sequence.append(processing_datetime_item)
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
            sequence.append(processing_description_item)

        self._processing_procedure = processing_procedure
        sequence.extend(processing_procedure)
        if fixative is not None:
            tissue_fixative_item = CodeContentItem(
                name=codes.SCT.TissueFixative,
                value=fixative
            )
            sequence.append(tissue_fixative_item)
        if embedding_medium is not None:
            embedding_medium_item = CodeContentItem(
                name=codes.SCT.TissueEmbeddingMedium,
                value=embedding_medium
            )
            sequence.append(embedding_medium_item)
        self.SpecimenPreparationStepContentItemSequence = sequence

    @property
    def specimen_id(self) -> str:
        """str: Specimen identifier"""
        items = self.SpecimenPreparationStepContentItemSequence.find(
            codes.DCM.SpecimenIdentifier
        )
        if len(items) == 0:
            raise AttributeError(
                'Could not find content item "Specimen Identifier".'
            )
        return items[0].value

    @property
    def processing_type(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: Processing type"""
        items = self.SpecimenPreparationStepContentItemSequence.find(
            codes.DCM.ProcessingType
        )
        if len(items) == 0:
            raise AttributeError(
                'Could not find content item "Processing Type".'
            )
        return items[0].value

    @property
    def processing_procedure(self) -> Union[
        SpecimenCollection,
        SpecimenSampling,
        SpecimenStaining,
        SpecimenProcessing,
    ]:
        """Union[highdicom.SpecimenCollection, highdicom.SpecimenSampling,
        highdicom.SpecimenStaining, highdicom.SpecimenProcessing]:

        Procedure used during processing

        """  # noqa: E501
        return self._processing_procedure

    @property
    def fixative(self) -> Union[CodedConcept, None]:
        """highdicom.sr.CodedConcept: Tissue fixative"""
        items = self.SpecimenPreparationStepContentItemSequence.find(
            codes.SCT.TissueFixative
        )
        if len(items) == 0:
            return None
        return items[0].value

    @property
    def embedding_medium(self) -> Union[CodedConcept, None]:
        """highdicom.sr.CodedConcept: Tissue embedding medium"""
        items = self.SpecimenPreparationStepContentItemSequence.find(
            codes.SCT.TissueEmbeddingMedium
        )
        if len(items) == 0:
            return None
        return items[0].value

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
    ) -> 'SpecimenPreparationStep':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset

        Returns
        -------
        highdicom.SpecimenPreparationStep
            Specimen Preparation Step

        """
        instance = deepcopy(dataset)
        sequence = ContentSequence.from_sequence(
            dataset.SpecimenPreparationStepContentItemSequence,
            is_root=False,
            is_sr=False
        )
        instance.SpecimenPreparationStepContentItemSequence = sequence
        instance.__class__ = cls
        # Order of template TID 8001 "Specimen Preparation" is significant
        specimen_identifier_items = sequence.find(codes.DCM.SpecimenIdentifier)
        if len(specimen_identifier_items) != 1:
            raise ValueError(
                'Specimen Preparation Step Content Item Sequence must contain '
                'exactly one content item "Specimen Identifier".'
            )
        processing_type_items = sequence.find(codes.DCM.ProcessingType)
        if len(processing_type_items) != 1:
            raise ValueError(
                'Specimen Preparation Step Content Item Sequence must contain '
                'exactly one content item "Processing Type".'
            )
        processing_type = processing_type_items[0].value

        instance._processing_procedure: Union[
            SpecimenCollection,
            SpecimenSampling,
            SpecimenStaining,
            SpecimenProcessing,
        ]
        if processing_type == codes.SCT.SpecimenCollection:
            collection_items = sequence.find(codes.SCT.SpecimenCollection)
            if len(collection_items) != 1:
                raise ValueError(
                    'Specimen Preparation Step Content Item Sequence must '
                    'contain exactly one content item "Specimen Collection" '
                    'when processing type is "Specimen Collection".'
                )
            instance._processing_procedure = SpecimenCollection(
                procedure=collection_items[0].value
            )
        elif processing_type == codes.SCT.SpecimenProcessing:
            description_items = sequence.find(
                codes.DCM.ProcessingStepDescription
            )
            if len(description_items) != 1:
                raise ValueError(
                    'Specimen Preparation Step Content Item Sequence must '
                    'contain exactly one content item "Processing Step '
                    'Description" when processing type is "Specimen .'
                    'Processing".'
                )
            instance._processing_procedure = SpecimenProcessing(
                description=description_items[0].value
            )
        elif processing_type == codes.SCT.Staining:
            substance_items = sequence.find(codes.SCT.UsingSubstance)
            if len(substance_items) == 0:
                raise ValueError(
                    'Specimen Preparation Step Content Item Sequence must '
                    'contain one or more content item "Using Substance" '
                    'when processing type is "Staining".'
                )
            instance._processing_procedure = SpecimenStaining(
                substances=[item.value for item in substance_items]
            )
        elif processing_type == codes.SCT.SamplingOfTissueSpecimen:
            sampling_method_items = sequence.find(codes.DCM.SamplingMethod)
            if len(sampling_method_items) != 1:
                raise ValueError(
                    'Specimen Preparation Step Content Item Sequence must '
                    'contain exactly one content item "Sampling Method" '
                    'when processing type is "Sampling of Tissue Specimen".'
                )
            parent_specimen_id_items = sequence.find(
                codes.DCM.ParentSpecimenIdentifier
            )
            if len(parent_specimen_id_items) != 1:
                raise ValueError(
                    'Specimen Preparation Step Content Item Sequence must '
                    'contain exactly one content item "Parent Specimen '
                    'Identifier" when processing type is "Sampling of Tissue '
                    'Specimen".'
                )
            parent_specimen_type_items = sequence.find(
                codes.DCM.ParentSpecimenType
            )
            if len(parent_specimen_type_items) != 1:
                raise ValueError(
                    'Specimen Preparation Step Content Item Sequence must '
                    'contain exactly one content item "Parent Specimen '
                    'Type" when processing type is "Sampling of Tissue '
                    'Specimen".'
                )
            issuer_of_parent_specimen_type_items = sequence.find(
                codes.DCM.IssuerOfParentSpecimenIdentifier
            )
            if len(issuer_of_parent_specimen_type_items) > 0:
                issuer = issuer_of_parent_specimen_type_items[0].value
            else:
                issuer = None
            instance._processing_procedure = SpecimenSampling(
                method=sampling_method_items[0].value,
                parent_specimen_id=parent_specimen_id_items[0].value,
                parent_specimen_type=parent_specimen_type_items[0].value,
                issuer_of_parent_specimen_id=issuer
            )
        else:
            raise ValueError(
                'Specimen Preparation Step Content Item Sequence must contain '
                'a content item "Processing Type" with one of the following '
                'values: "Specimen Collection", "Specimen Processing", '
                '"Staining", or "Sampling of Tissue Specimen".'
            )

        cast(SpecimenPreparationStep, instance)
        return instance


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
            for step_item in specimen_preparation_steps:
                if not isinstance(step_item, SpecimenPreparationStep):
                    raise TypeError(
                        'Items of "specimen_preparation_steps" must have '
                        'type SpecimenPreparationStep.'
                    )
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

    @property
    def specimen_id(self) -> str:
        """str: Specimen identifier"""
        return str(self.SpecimenIdentifier)

    @property
    def specimen_uid(self) -> UID:
        """highdicom.UID: Unique specimen identifier"""
        return UID(self.SpecimenUID)

    @property
    def specimen_preparation_steps(self) -> List[SpecimenPreparationStep]:
        """highdicom.SpecimenPreparationStep: Specimen preparation steps"""
        return list(self.SpecimenPreparationSequence)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'SpecimenDescription':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an item of Specimen Description Sequence

        Returns
        -------
        highdicom.SpecimenDescription
            Constructed object

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        check_required_attributes(
            dataset,
            module='specimen',
            base_path=['SpecimenDescriptionSequence'],
            check_optional_sequences=True
        )
        desc = deepcopy(dataset)
        desc.__class__ = cls

        # Convert sub sequences to highdicom types
        desc.SpecimenPreparationSequence = [
            SpecimenPreparationStep.from_dataset(step)
            for step in desc.SpecimenPreparationSequence
        ]
        if hasattr(desc, 'PrimaryAnatomicStructureSequence'):
            desc.PrimaryAnatomicStructureSequence = [
                CodedConcept.from_dataset(ds)
                for ds in desc.PrimaryAnatomicStructureSequence
            ]
        if hasattr(desc, 'SpecimenTypeCodeSequence'):
            desc.SpecimenTypeCodeSequence = [
                CodedConcept.from_dataset(ds)
                for ds in desc.SpecimenTypeCodeSequence
            ]

        return desc


class ReferencedImageSequence(DataElementSequence):

    """Sequence of data elements describing a set of referenced images."""

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
        for ref_im in referenced_images:
            if not isinstance(ref_im, Dataset):
                raise TypeError(
                    'Argument "referenced_images" must be a sequence of '
                    'pydicom.Dataset instances.'
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
            if not does_iod_have_pixel_data(im.SOPClassUID):
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

    """Dataset describing a lookup table (LUT)."""

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
        lut_data: numpy.ndarray
            Lookup table data. Must be of type uint16.
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
            raise TypeError('Argument "lut_data" must have type numpy.ndarray.')
        if lut_data.ndim != 1:
            raise ValueError(
                'Argument "lut_data" must be an array with a single dimension.'
            )
        len_data = lut_data.size
        if len_data == 0:
            raise ValueError('Argument "lut_data" must not be empty.')
        if len_data > 2**16:
            raise ValueError(
                'Length of argument "lut_data" must be no greater than '
                '2^16 elements.'
            )
        elif len_data == 2**16:
            # Per the standard, this is recorded as 0
            len_data = 0
        # Note 8 bit LUT data is unsupported pending clarification on the
        # standard
        if lut_data.dtype.type == np.uint16:
            bits_per_entry = 16
        else:
            raise ValueError(
                "Numpy array must have dtype uint16."
            )
        # The LUT data attribute has VR OW (16-bit other words)
        self.LUTData = lut_data.astype(np.uint16).tobytes()

        self.LUTDescriptor = [
            len_data,
            int(first_mapped_value),
            bits_per_entry
        ]

        if lut_explanation is not None:
            _check_long_string(lut_explanation)
            self.LUTExplanation = lut_explanation

    @property
    def lut_data(self) -> np.ndarray:
        """numpy.ndarray: LUT data"""
        if self.bits_per_entry == 8:
            raise RuntimeError("8 bit LUTs are currently unsupported.")
        elif self.bits_per_entry == 16:
            dtype = np.uint16
        else:
            raise RuntimeError("Invalid LUT descriptor.")
        length = self.LUTDescriptor[0]
        data = self.LUTData
        # The LUT data attributes have VR OW (16-bit other words)
        array = np.frombuffer(data, dtype=np.uint16)
        # Needs to be casted according to third descriptor value.
        array = array.astype(dtype)
        if len(array) != length:
            raise RuntimeError(
                'Length of LUTData does not match the value expected from the '
                f'LUTDescriptor. Expected {length}, found {len(array)}.'
            )
        return array

    @property
    def number_of_entries(self) -> int:
        """int: Number of entries in the lookup table."""
        value = int(self.LUTDescriptor[0])
        # Part 3 Section C.7.6.3.1.5 Palette Color Lookup Table Descriptor
        # "When the number of table entries is equal to 2^16
        # then this value shall be 0".
        # That's because the descriptor attributes have VR US, which cannot
        # encode the value of 2^16, but only values in the range [0, 2^16 - 1].
        if value == 0:
            return 2**16
        else:
            return value

    @property
    def first_mapped_value(self) -> int:
        """int: Pixel value that will be mapped to the first value in the
        lookup table.
        """
        return int(self.LUTDescriptor[1])

    @property
    def bits_per_entry(self) -> int:
        """int: Bits allocated for the lookup table data. 8 or 16."""
        return int(self.LUTDescriptor[2])


class ModalityLUT(LUT):

    """Dataset describing an item of the Modality LUT Sequence."""

    def __init__(
        self,
        lut_type: Union[RescaleTypeValues, str],
        first_mapped_value: int,
        lut_data: np.ndarray,
        lut_explanation: Optional[str] = None
    ):
        """

        Parameters
        ----------
        lut_type: Union[highdicom.RescaleTypeValues, str]
            String or enumerated value specifying the units of the output of
            the LUT operation.
        first_mapped_value: int
            Pixel value that will be mapped to the first value in the
            lookup-table.
        lut_data: numpy.ndarray
            Lookup table data. Must be of type uint16.
        lut_explanation: Union[str, None], optional
            Free-form text explanation of the meaning of the LUT.

        """
        super().__init__(
            first_mapped_value=first_mapped_value,
            lut_data=lut_data,
            lut_explanation=lut_explanation
        )
        if isinstance(lut_type, RescaleTypeValues):
            self.ModalityLUTType = lut_type.value
        else:
            _check_long_string(lut_type)
            self.ModalityLUTType = lut_type


class VOILUT(LUT):

    """Dataset describing an item of the VOI LUT Sequence."""

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
        lut_data: numpy.ndarray
            Lookup table data. Must be of type uint16.
        lut_explanation: Union[str, None], optional
            Free-form text explanation of the meaning of the LUT.

        """
        super().__init__(
            first_mapped_value=first_mapped_value,
            lut_data=lut_data,
            lut_explanation=lut_explanation
        )


class VOILUTTransformation(Dataset):

    """Dataset describing the VOI LUT Transformation as part of the Pixel
    Transformation Sequence to transform modality pixel values into pixel
    values that are of interest to a user or an application.

    """

    def __init__(
        self,
        window_center: Union[float, Sequence[float], None] = None,
        window_width: Union[float, Sequence[float], None] = None,
        window_explanation: Union[str, Sequence[str], None] = None,
        voi_lut_function: Union[VOILUTFunctionValues, str, None] = None,
        voi_luts: Optional[Sequence[VOILUT]] = None,
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
        voi_luts: Union[Sequence[highdicom.VOILUT], None], optional
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
                    format_number_as_ds(float(x)) for x in window_center
                ]
            else:
                self.WindowCenter = format_number_as_ds(float(window_center))
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
                    format_number_as_ds(float(x)) for x in window_width
                ]
            else:
                if window_is_sequence:
                    raise TypeError(
                        'Length of "window_width" must match length of '
                        '"window_center".'
                    )
                self.WindowWidth = format_number_as_ds(float(window_width))
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
                raise ValueError('Argument "voi_luts" should not be empty.')
            for lut in voi_luts:
                if not isinstance(lut, VOILUT):
                    raise TypeError(
                        'Items of "voi_luts" should be of type VOILUT.'
                    )
            self.VOILUTSequence = list(voi_luts)
        else:
            if window_center is None:
                raise TypeError(
                    'At least one of "window_center" or "luts" should be '
                    'provided.'
                )


class ModalityLUTTransformation(Dataset):

    """Dataset describing the Modality LUT Transformation as part of the Pixel
    Transformation Sequence to transform the manufacturer dependent pixel
    values into pixel values that are meaningful for the modality and are
    manufacturer independent.

    """

    def __init__(
        self,
        rescale_intercept: Optional[Union[int, float]] = None,
        rescale_slope: Optional[Union[int, float]] = None,
        rescale_type: Optional[Union[RescaleTypeValues, str]] = None,
        modality_lut: Optional[ModalityLUT] = None,
    ):
        """

        Parameters
        ----------
        rescale_intercept: Union[int, float, None], optional
            Intercept of linear function used for rescaling pixel values.
        rescale_slope: Union[int, float, None], optional
            Slope of linear function used for rescaling pixel values.
        rescale_type: Union[highdicom.RescaleTypeValues, str, None], optional
            String or enumerated value specifying the units of the output of
            the Modality LUT or rescale operation.
        modality_lut: Union[highdicom.ModalityLUT, None], optional
            Lookup table specifying a pixel rescaling operation to apply to
            the stored values to give modality values.

        Note
        ----
        Either `modality_lut` may be specified or all three of `rescale_slope`,
        `rescale_intercept`, and `rescale_type` may be specified. All four
        parameters should not be specified simultaneously.

        """
        super().__init__()
        if modality_lut is not None:
            if rescale_intercept is not None:
                raise TypeError(
                    'Argument "rescale_intercept" must not be specified when '
                    '"modality_lut" is specified.'
                )
            if rescale_slope is not None:
                raise TypeError(
                    'Argument "rescale_slope" must not be specified when '
                    '"modality_lut" is specified.'
                )
            if rescale_type is not None:
                raise TypeError(
                    'Argument "rescale_type" must not be specified when '
                    '"modality_lut" is specified.'
                )
            if not isinstance(modality_lut, ModalityLUT):
                raise TypeError(
                    'Argument "modality_lut" must be of type '
                    'highdicom.ModalityLUT.'
                )
            self.ModalityLUTSequence = [modality_lut]
        else:
            if rescale_intercept is None:
                raise TypeError(
                    'Argument "rescale_intercept" must be specified when '
                    '"modality_lut" is not specified.'
                )
            if rescale_slope is None:
                raise TypeError(
                    'Argument "rescale_slope" must be specified when '
                    '"modality_lut" is not specified.'
                )
            if rescale_type is None:
                raise TypeError(
                    'Argument "rescale_type" must be specified when '
                    '"modality_lut" is not specified.'
                )
            self.RescaleIntercept = format_number_as_ds(rescale_intercept)
            self.RescaleSlope = format_number_as_ds(rescale_slope)
            if isinstance(rescale_type, RescaleTypeValues):
                self.RescaleType = rescale_type.value
            else:
                _check_long_string(rescale_type)
                self.RescaleType = rescale_type


class PresentationLUT(LUT):

    """Dataset describing an item of the Presentation LUT Sequence."""

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
        lut_data: numpy.ndarray
            Lookup table data. Must be of type uint16.
        lut_explanation: Union[str, None], optional
            Free-form text explanation of the meaning of the LUT.

        """
        super().__init__(
            first_mapped_value=first_mapped_value,
            lut_data=lut_data,
            lut_explanation=lut_explanation
        )


class PresentationLUTTransformation(Dataset):

    """Dataset describing the Presentation LUT Transformation as part of the
    Pixel Transformation Sequence to transform polarity pixel values into
    device-indendent presentation values (P-Values).

    """

    def __init__(
        self,
        presentation_lut_shape: Optional[
            Union[PresentationLUTShapeValues, str]
        ] = None,
        presentation_lut: Optional[PresentationLUT] = None,
    ):
        """

        Parameters
        ----------
        presentation_lut_shape: Union[highdicom.pr.PresentationLUTShapeValues, str, None], optional
            Shape of the presentation LUT
        presentation_lut: Optional[highdicom.PresentationLUT], optional
            Presentation LUT

        Note
        -----
        Only one of ``presentation_lut_shape`` or ``presentation_lut`` should
        be provided.

        """  # noqa: E501
        super().__init__()
        if presentation_lut is not None:
            if presentation_lut_shape is not None:
                raise TypeError(
                    'Only one of arguments "presentation_lut" or '
                    '"presentation_lut_shape" should be provided.'
                )
            if not isinstance(presentation_lut, PresentationLUT):
                raise TypeError(
                    'Argument "presentation_lut" must be of '
                    'type PresentationLUT.'
                )
            self.PresentationLUTSequence = [presentation_lut]
        else:
            presentation_lut_shape = (
                presentation_lut_shape or
                PresentationLUTShapeValues.IDENTITY
            )
            self.PresentationLUTShape = PresentationLUTShapeValues(
                presentation_lut_shape
            ).value


class PaletteColorLUT(Dataset):

    """Dataset describing a palette color lookup table (LUT)."""

    def __init__(
        self,
        first_mapped_value: int,
        lut_data: np.ndarray,
        color: str
    ):
        """

        Parameters
        ----------
        first_mapped_value: int
            Pixel value that will be mapped to the first value in the
            lookup table.
        lut_data: numpy.ndarray
            Lookup table data. Must be of type uint16.
        color: str
            Text representing the color (``red``, ``green``, or
            ``blue``).

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
            raise TypeError('Argument "lut_data" must have type numpy.ndarray.')
        if lut_data.ndim != 1:
            raise ValueError(
                'Argument "lut_data" must be an array with a single dimension.'
            )
        len_data = lut_data.shape[0]
        if len_data == 0:
            raise ValueError('Argument "lut_data" must not be empty.')
        if len_data > 2**16:
            raise ValueError(
                'Length of argument "lut_data" must be no greater than '
                '2^16 elements.'
            )
        elif len_data == 2**16:
            # Per the standard, this is recorded as 0
            number_of_entries = 0
        else:
            number_of_entries = len_data
        # Note 8 bit LUT data is unsupported pending clarification on the
        # standard
        if lut_data.dtype.type == np.uint16:
            bits_per_entry = 16
        else:
            raise ValueError(
                "Numpy array must have dtype uint16."
            )

        if color.lower() not in ('red', 'green', 'blue'):
            raise ValueError(
                'Argument "color" must be either "red", "green", or "blue".'
            )
        self._attr_name_prefix = f'{color.title()}PaletteColorLookupTable'

        # The Palette Color Lookup Table Data attributes have VR OW
        # (16-bit other words)
        setattr(
            self,
            f'{self._attr_name_prefix}Data',
            lut_data.astype(np.uint16).tobytes()
        )
        setattr(
            self,
            f'{self._attr_name_prefix}Descriptor',
            [number_of_entries, int(first_mapped_value), bits_per_entry]
        )

    @property
    def lut_data(self) -> np.ndarray:
        """numpy.ndarray: lookup table data"""
        if self.bits_per_entry == 8:
            raise RuntimeError("8 bit LUTs are currently unsupported.")
        elif self.bits_per_entry == 16:
            dtype = np.uint16
        else:
            raise RuntimeError("Invalid LUT descriptor.")
        length = self.number_of_entries
        data = getattr(self, f'{self._attr_name_prefix}Data')
        # The LUT data attributes have VR OW (16-bit other words)
        array = np.frombuffer(data, dtype=np.uint16)
        # Needs to be casted according to third descriptor value.
        array = array.astype(dtype)
        if len(array) != length:
            raise RuntimeError(
                'Length of Lookup Table Data does not match the value '
                'expected from the Lookup Table Descriptor. '
                f'Expected {length}, found {len(array)}.'
            )
        return array

    @property
    def number_of_entries(self) -> int:
        """int: Number of entries in the lookup table."""
        descriptor = getattr(self, f'{self._attr_name_prefix}Descriptor')
        value = int(descriptor[0])
        if value == 0:
            return 2**16
        return value

    @property
    def first_mapped_value(self) -> int:
        """int: Pixel value that will be mapped to the first value in the
        lookup table.
        """
        descriptor = getattr(self, f'{self._attr_name_prefix}Descriptor')
        return int(descriptor[1])

    @property
    def bits_per_entry(self) -> int:
        """int: Bits allocated for the lookup table data. 8 or 16."""
        descriptor = getattr(self, f'{self._attr_name_prefix}Descriptor')
        return int(descriptor[2])


class SegmentedPaletteColorLUT(Dataset):

    """Dataset describing a segmented palette color lookup table (LUT)."""

    def __init__(
        self,
        first_mapped_value: int,
        segmented_lut_data: np.ndarray,
        color: str
    ):
        """

        Parameters
        ----------
        first_mapped_value: int
            Pixel value that will be mapped to the first value in the
            lookup table.
        segmented_lut_data: numpy.ndarray
            Segmented lookup table data. Must be of type uint16.
        color: str
            Free-form text explanation of the color (``red``, ``green``, or
            ``blue``).

        Note
        ----
        After the LUT is applied, a pixel in the image with value equal to
        ``first_mapped_value`` is mapped to an output value of ``lut_data[0]``,
        an input value of ``first_mapped_value + 1`` is mapped to
        ``lut_data[1]``, and so on.

        See :dcm:`here <part03/sect_C.7.9.2.html>` for details of how the
        segmented LUT data is encoded. Highdicom may provide utilities to
        assist in creating these arrays in a future release.

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

        if not isinstance(segmented_lut_data, np.ndarray):
            raise TypeError(
                'Argument "segmented_lut_data" must have type numpy.ndarray.'
            )
        if segmented_lut_data.ndim != 1:
            raise ValueError(
                'Argument "segmented_lut_data" must be an array with a '
                'single dimension.'
            )
        len_data = segmented_lut_data.size
        if len_data == 0:
            raise ValueError('Argument "segmented_lut_data" must not be empty.')
        if len_data > 2**16:
            raise ValueError(
                'Length of argument "segmented_lut_data" must be no greater '
                'than 2^16 elements.'
            )
        elif len_data == 2**16:
            # Per the standard, this is recorded as 0
            len_data = 0
        # Note 8 bit LUT data is currently unsupported pending clarification on
        # the standard
        if segmented_lut_data.dtype.type == np.uint16:
            bits_per_entry = 16
            self._dtype = np.uint16
        else:
            raise ValueError(
                "Numpy array must have dtype uint16."
            )

        if color.lower() not in ('red', 'green', 'blue'):
            raise ValueError(
                'Argument "color" must be either "red", "green", or "blue".'
            )
        self._attr_name_prefix = f'{color.title()}PaletteColorLookupTable'

        # The Segmented Palette Color Lookup Table Data attributes have VR OW
        # (16-bit other words)
        setattr(
            self,
            f'Segmented{self._attr_name_prefix}Data',
            segmented_lut_data.astype(np.uint16).tobytes()
        )

        expanded_lut_values = []
        i = 0
        offset = 0
        while i < len(segmented_lut_data):
            opcode = segmented_lut_data[i]
            i += 1
            if opcode == 0:
                # Discrete segment type (constant)
                length = segmented_lut_data[i]
                i += 1
                value = segmented_lut_data[i]
                i += 1
                expanded_lut_values.extend([
                    value for _ in range(length)
                ])
                offset += length
            elif opcode == 1:
                # Linear segment type (interpolation)
                length = segmented_lut_data[i]
                i += 1
                start_value = expanded_lut_values[offset - 1]
                end_value = segmented_lut_data[i]
                i += 1
                step = (end_value - start_value) / (length - 1)
                expanded_lut_values.extend([
                    start_value + int(np.round(j * step))
                    for j in range(length)

                ])
                offset += length
            elif opcode == 2:
                # TODO
                raise ValueError(
                  'Indirect segment type is not yet supported for '
                  'Segmented Palette Color Lookup Table.'
                )
            else:
                raise ValueError(
                  f'Encountered unexpected segment type {opcode} for '
                  'Segmented Palette Color Lookup Table.'
                )

        self._lut_data = np.array(
            expanded_lut_values,
            dtype=self._dtype
        )

        len_data = len(expanded_lut_values)
        if len_data == 2**16:
            number_of_entries = 0
        else:
            number_of_entries = len_data
        setattr(
            self,
            f'{self._attr_name_prefix}Descriptor',
            [number_of_entries, int(first_mapped_value), bits_per_entry]
        )

    @property
    def segmented_lut_data(self) -> np.ndarray:
        """numpy.ndarray: segmented lookup table data"""
        length = self.number_of_entries
        data = getattr(self, f'Segmented{self._attr_name_prefix}Data')
        # The LUT data attributes have VR OW (16-bit other words)
        array = np.frombuffer(data, dtype=np.uint16)
        # Needs to be casted according to third descriptor value.
        array = array.astype(self._dtype)
        if len(array) != length:
            raise RuntimeError(
                'Length of LUTData does not match the value expected from the '
                f'LUTDescriptor. Expected {length}, found {len(array)}.'
            )
        return array

    @property
    def lut_data(self) -> np.ndarray:
        """numpy.ndarray: expanded lookup table data"""
        return self._lut_data

    @property
    def number_of_entries(self) -> int:
        """int: Number of entries in the lookup table."""
        descriptor = getattr(self, f'{self._attr_name_prefix}Descriptor')
        value = int(descriptor[0])
        # Part 3 Section C.7.6.3.1.5 Palette Color Lookup Table Descriptor
        # "When the number of table entries is equal to 2^16
        # then this value shall be 0".
        # That's because the descriptor attributes have VR US, which cannot
        # encode the value of 2^16, but only values in the range [0, 2^16 - 1].
        if value == 0:
            return 2**16
        else:
            return value

    @property
    def first_mapped_value(self) -> int:
        """int: Pixel value that will be mapped to the first value in the
        lookup table.
        """
        descriptor = getattr(self, f'{self._attr_name_prefix}Descriptor')
        return int(descriptor[1])

    @property
    def bits_per_entry(self) -> int:
        """int: Bits allocated for the lookup table data. 8 or 16."""
        descriptor = getattr(self, f'{self._attr_name_prefix}Descriptor')
        return int(descriptor[2])


class PaletteColorLUTTransformation(Dataset):

    """Dataset describing the Palette Color LUT Transformation as part of the
    Pixel Transformation Sequence to transform grayscale into RGB color pixel
    values.

    """

    def __init__(
        self,
        red_lut: Union[PaletteColorLUT, SegmentedPaletteColorLUT],
        green_lut: Union[PaletteColorLUT, SegmentedPaletteColorLUT],
        blue_lut: Union[PaletteColorLUT, SegmentedPaletteColorLUT],
        palette_color_lut_uid: Union[UID, str, None] = None
    ):
        """

        Parameters
        ----------
        red_lut: Union[highdicom.PaletteColorLUT, highdicom.SegmentedPaletteColorLUT]
            Lookup table for the red output color channel.
        green: Union[highdicom.PaletteColorLUT, highdicom.SegmentedPaletteColorLUT]
            Lookup table for the green output color channel.
        blue_lut: Union[highdicom.PaletteColorLUT, highdicom.SegmentedPaletteColorLUT]
            Lookup table for the blue output color channel.
        palette_color_lut_uid: Union[highdicom.UID, str, None], optional
            Unique identifier for the palette color lookup table.

        """  # noqa: E501
        super().__init__()

        # Checks on inputs
        self._color_luts = {
            'Red': red_lut,
            'Green': green_lut,
            'Blue': blue_lut
        }
        for lut in self._color_luts.values():
            if not isinstance(lut, (PaletteColorLUT, SegmentedPaletteColorLUT)):
                raise TypeError(
                    'Arguments "red_lut", "green_lut", and "blue_lut" must be '
                    'of type PaletteColorLUT or SegmentedPaletteColorLUT.'
                )
        if not hasattr(red_lut, 'RedPaletteColorLookupTableDescriptor'):
            raise ValueError(
                'Argument "red_lut" does not correspond to red color.'
            )
        if not hasattr(green_lut, 'GreenPaletteColorLookupTableDescriptor'):
            raise ValueError(
                'Argument "green_lut" does not correspond to green color.'
            )
        if not hasattr(blue_lut, 'BluePaletteColorLookupTableDescriptor'):
            raise ValueError(
                'Argument "blue_lut" does not correspond to blue color.'
            )

        red_length = red_lut.number_of_entries
        green_length = green_lut.number_of_entries
        blue_length = blue_lut.number_of_entries
        if len(set([red_length, green_length, blue_length])) != 1:
            raise ValueError(
                'All three palette color LUTs must have the same number of '
                'entries.'
            )
        red_bits = red_lut.bits_per_entry
        green_bits = green_lut.bits_per_entry
        blue_bits = blue_lut.bits_per_entry
        if len(set([red_bits, green_bits, blue_bits])) != 1:
            raise ValueError(
                'All three palette color LUTs must have the same number of '
                'bits per entry.'
            )
        red_fmv = red_lut.first_mapped_value
        green_fmv = green_lut.first_mapped_value
        blue_fmv = blue_lut.first_mapped_value
        if len(set([red_fmv, green_fmv, blue_fmv])) != 1:
            raise ValueError(
                'All three palette color LUTs must have the same '
                'first mapped value.'
            )

        for name, lut in self._color_luts.items():
            desc_attr = f'{name}PaletteColorLookupTableDescriptor'
            setattr(
                self,
                desc_attr,
                getattr(lut, desc_attr)
            )
            if isinstance(lut, SegmentedPaletteColorLUT):
                data_attr = f'Segmented{name}PaletteColorLookupTableData'
            else:
                data_attr = f'{name}PaletteColorLookupTableData'
            setattr(
                self,
                data_attr,
                getattr(lut, data_attr)
            )

        if palette_color_lut_uid is not None:
            self.PaletteColorLookupTableUID = palette_color_lut_uid

        # To cache the array
        self._lut_data = None

    @property
    def red_lut(self) -> Union[PaletteColorLUT, SegmentedPaletteColorLUT]:
        """Union[highdicom.PaletteColorLUT, highdicom.SegmentedPaletteColorLUT]:
            Lookup table for the red output color channel

        """
        return self._color_luts['Red']

    @property
    def green_lut(self) -> Union[PaletteColorLUT, SegmentedPaletteColorLUT]:
        """Union[highdicom.PaletteColorLUT, highdicom.SegmentedPaletteColorLUT]:
            Lookup table for the green output color channel

        """
        return self._color_luts['Green']

    @property
    def blue_lut(self) -> Union[PaletteColorLUT, SegmentedPaletteColorLUT]:
        """Union[highdicom.PaletteColorLUT, highdicom.SegmentedPaletteColorLUT]:
            Lookup table for the blue output color channel

        """
        return self._color_luts['Blue']
