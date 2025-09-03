"""Sub-templates common to multiple root-level templates"""
from abc import ABC
import logging
from typing import List, Optional, Self, Sequence, TypedDict, Union

from pydicom import Dataset
from highdicom.sr.value_types import (
    Code,
    CodeContentItem,
    CodedConcept,
    ContentSequence,
    NumContentItem,
    TextContentItem,
    PnameContentItem,
    RelationshipTypeValues,
    UIDRefContentItem
)
from highdicom.sr.content import ContentItem
from pydicom.sr.codedict import codes
from copy import deepcopy
from highdicom.uid import UID

logger = logging.getLogger(__name__)


class Template(ContentSequence):

    """Abstract base class for a DICOM SR template."""

    def __init__(
        self,
        items: Sequence[ContentItem] | None = None,
        is_root: bool = False
    ) -> None:
        """

        Parameters
        ----------
        items: Sequence[ContentItem], optional
            content items
        is_root: bool
            Whether this template exists at the root of the SR document
            content tree.

        """
        super().__init__(items, is_root=is_root)


class Units(TypedDict):
    value: Union[float, None]
    code_value: str
    code_meaning: str


class CIDUnits(ABC):
    name: str
    coding_scheme_designator: str = "UCUM"
    units: List[Units]

    def add_items(self, content: ContentSequence) -> None:

        """ Adds units as NumContentItems to a content sequence """

        for unit in self.units:
            if unit["value"] is not None:
                item = NumContentItem(
                    name=self.name,
                    value=unit["value"],
                    unit=CodedConcept(
                        value=unit["code_value"],
                        meaning=unit["code_meaning"],
                        scheme_designator=self.coding_scheme_designator
                    ),
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(item)


class AgeUnit(CIDUnits):
    name = codes.DCM.SubjectAge

    def __init__(self, year: Union[int, None] = None,
                 month: Union[int, None] = None,
                 week: Union[int, None] = None,
                 day: Union[int, None] = None,
                 hour: Union[int, None] = None,
                 minute: Union[int, None] = None) -> None:
        self.units = [
            {'value': year, 'code_value': "a", 'code_meaning': "year"},
            {'value': month, 'code_value': "mo", 'code_meaning': "month"},
            {'value': week, 'code_value': "wk", 'code_meaning': "week"},
            {'value': day, 'code_value': "d", 'code_meaning': "day"},
            {'value': hour, 'code_value': "h", 'code_meaning': "hour"},
            {'value': minute, 'code_value': "min", 'code_meaning': "minute"}
        ]


class PressureUnit(CIDUnits):
    def __init__(self, mmHg: Union[int, None] = None,
                 kPa: Union[int, None] = None) -> None:
        self.units = [
            {'value': mmHg, 'code_value': "mm[Hg]", 'code_meaning': "mmHg"},
            {'value': kPa, 'code_value': "kPa", 'code_meaning': "kPa"}
        ]


class AlgorithmIdentification(Template):

    """:dcm:`TID 4019 <part16/sect_TID_4019.html>`
    Algorithm Identification"""

    def __init__(
        self,
        name: str,
        version: str,
        parameters: Sequence[str] | None = None
    ) -> None:
        """

        Parameters
        ----------
        name: str
            name of the algorithm
        version: str
            version of the algorithm
        parameters: Union[Sequence[str], None], optional
            parameters of the algorithm

        """
        super().__init__()
        name_item = TextContentItem(
            name=CodedConcept(
                value='111001',
                meaning='Algorithm Name',
                scheme_designator='DCM'
            ),
            value=name,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        self.append(name_item)
        version_item = TextContentItem(
            name=CodedConcept(
                value='111003',
                meaning='Algorithm Version',
                scheme_designator='DCM'
            ),
            value=version,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        self.append(version_item)
        if parameters is not None:
            for param in parameters:
                parameter_item = TextContentItem(
                    name=CodedConcept(
                        value='111002',
                        meaning='Algorithm Parameters',
                        scheme_designator='DCM'
                    ),
                    value=param,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                self.append(parameter_item)


class LanguageOfContentItemAndDescendants(Template):

    """:dcm:`TID 1204 <part16/chapter_A.html#sect_TID_1204>`
     Language of Content Item and Descendants"""

    def __init__(self, language: CodedConcept):
        """

        Parameters
        ----------
        language: highdicom.sr.CodedConcept
            language used for content items included in report

        """
        super().__init__()
        language_item = CodeContentItem(
            name=CodedConcept(
                value='121049',
                meaning='Language of Content Item and Descendants',
                scheme_designator='DCM',
            ),
            value=language,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        self.append(language_item)


class PersonObserverIdentifyingAttributes(Template):

    """:dcm:`TID 1003 <part16/chapter_A.html#sect_TID_1003>`
     Person Observer Identifying Attributes"""

    def __init__(
        self,
        name: str,
        login_name: str | None = None,
        organization_name: str | None = None,
        role_in_organization: CodedConcept | Code | None = None,
        role_in_procedure: CodedConcept | Code | None = None
    ):
        """

        Parameters
        ----------
        name: str
            name of the person
        login_name: Union[str, None], optional
            login name of the person
        organization_name: Union[str, None], optional
            name of the person's organization
        role_in_organization: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            role of the person within the organization
        role_in_procedure: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            role of the person in the reported procedure

        """  # noqa: E501
        super().__init__()
        name_item = PnameContentItem(
            name=CodedConcept(
                value='121008',
                meaning='Person Observer Name',
                scheme_designator='DCM',
            ),
            value=name,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(name_item)
        if login_name is not None:
            login_name_item = TextContentItem(
                name=CodedConcept(
                    value='128774',
                    meaning='Person Observer\'s Login Name',
                    scheme_designator='DCM',
                ),
                value=login_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(login_name_item)
        if organization_name is not None:
            organization_name_item = TextContentItem(
                name=CodedConcept(
                    value='121009',
                    meaning='Person Observer\'s Organization Name',
                    scheme_designator='DCM',
                ),
                value=organization_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(organization_name_item)
        if role_in_organization is not None:
            role_in_organization_item = CodeContentItem(
                name=CodedConcept(
                    value='121010',
                    meaning='Person Observer\'s Role in the Organization',
                    scheme_designator='DCM',
                ),
                value=role_in_organization,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(role_in_organization_item)
        if role_in_procedure is not None:
            role_in_procedure_item = CodeContentItem(
                name=CodedConcept(
                    value='121011',
                    meaning='Person Observer\'s Role in this Procedure',
                    scheme_designator='DCM',
                ),
                value=role_in_procedure,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(role_in_procedure_item)

    @property
    def name(self) -> str:
        """str: name of the person"""
        return self[0].value

    @property
    def login_name(self) -> str | None:
        """Union[str, None]: login name of the person"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverLoginName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Login Name" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def organization_name(self) -> str | None:
        """Union[str, None]: name of the person's organization"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverOrganizationName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Organization Name" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def role_in_organization(self) -> str | None:
        """Union[str, None]: role of the person in the organization"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverRoleInTheOrganization
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Role in Organization" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def role_in_procedure(self) -> str | None:
        """Union[str, None]: role of the person in the procedure"""
        matches = [
            item for item in self
            if item.name == codes.DCM.PersonObserverRoleInThisProcedure
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Role in Procedure" content item '
                'in "Person Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> Self:
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1003 "Person Observer Identifying Attributes"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.PersonObserverIdentifyingAttributes
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('name', codes.DCM.PersonObserverName),
            ('login_name', codes.DCM.PersonObserverLoginName),
            ('organization_name',
             codes.DCM.PersonObserverOrganizationName),
            ('role_in_organization',
             codes.DCM.PersonObserverRoleInTheOrganization),
            ('role_in_procedure',
             codes.DCM.PersonObserverRoleInThisProcedure),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class DeviceObserverIdentifyingAttributes(Template):

    """:dcm:`TID 1004 <part16/chapter_A.html#sect_TID_1004>`
     Device Observer Identifying Attributes
    """

    def __init__(
        self,
        uid: str,
        name: str | None = None,
        manufacturer_name: str | None = None,
        model_name: str | None = None,
        serial_number: str | None = None,
        physical_location: str | None = None,
        role_in_procedure: Code | CodedConcept | None = None
    ):
        """

        Parameters
        ----------
        uid: str
            device UID
        name: Union[str, None], optional
            name of device
        manufacturer_name: Union[str, None], optional
            name of device's manufacturer
        model_name: Union[str, None], optional
            name of the device's model
        serial_number: Union[str, None], optional
            serial number of the device
        physical_location: Union[str, None], optional
            physical location of the device during the procedure
        role_in_procedure: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            role of the device in the reported procedure

        """  # noqa: E501
        super().__init__()
        device_observer_item = UIDRefContentItem(
            name=CodedConcept(
                value='121012',
                meaning='Device Observer UID',
                scheme_designator='DCM',
            ),
            value=uid,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(device_observer_item)
        if name is not None:
            name_item = TextContentItem(
                name=CodedConcept(
                    value='121013',
                    meaning='Device Observer Name',
                    scheme_designator='DCM',
                ),
                value=name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(name_item)
        if manufacturer_name is not None:
            manufacturer_name_item = TextContentItem(
                name=CodedConcept(
                    value='121014',
                    meaning='Device Observer Manufacturer',
                    scheme_designator='DCM',
                ),
                value=manufacturer_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(manufacturer_name_item)
        if model_name is not None:
            model_name_item = TextContentItem(
                name=CodedConcept(
                    value='121015',
                    meaning='Device Observer Model Name',
                    scheme_designator='DCM',
                ),
                value=model_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(model_name_item)
        if serial_number is not None:
            serial_number_item = TextContentItem(
                name=CodedConcept(
                    value='121016',
                    meaning='Device Observer Serial Number',
                    scheme_designator='DCM',
                ),
                value=serial_number,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(serial_number_item)
        if physical_location is not None:
            physical_location_item = TextContentItem(
                name=codes.DCM.DeviceObserverPhysicalLocationDuringObservation,
                value=physical_location,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(physical_location_item)
        if role_in_procedure is not None:
            role_in_procedure_item = CodeContentItem(
                name=codes.DCM.DeviceRoleInProcedure,
                value=role_in_procedure,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(role_in_procedure_item)

    @property
    def uid(self) -> UID:
        """highdicom.UID: unique device identifier"""
        return UID(self[0].value)

    @property
    def name(self) -> str | None:
        """Union[str, None]: name of device"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Name" content item '
                'in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def manufacturer_name(self) -> str | None:
        """Union[str, None]: name of device manufacturer"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverManufacturer
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Manufacturer" content '
                'name in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def model_name(self) -> str | None:
        """Union[str, None]: name of device model"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverModelName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Model Name" content '
                'item in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def serial_number(self) -> str | None:
        """Union[str, None]: device serial number"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverSerialNumber
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Serial Number" content '
                'item in "Device Observer Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def physical_location(self) -> str | None:
        """Union[str, None]: location of device"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceObserverPhysicalLocationDuringObservation  # noqa: E501
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Observer Physical Location '
                'During Observation" content item in "Device Observer '
                'Identifying Attributes" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> Self:
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1004 "Device Observer Identifying Attributes"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.templates.DeviceObserverIdentifyingAttributes
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('name', codes.DCM.DeviceObserverName),
            ('uid', codes.DCM.DeviceObserverUID),
            ('manufacturer_name', codes.DCM.DeviceObserverManufacturer),
            ('model_name', codes.DCM.DeviceObserverModelName),
            ('serial_number', codes.DCM.DeviceObserverSerialNumber),
            ('physical_location',
             codes.DCM.DeviceObserverPhysicalLocationDuringObservation),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class ObserverContext(Template):

    """:dcm:`TID 1002 <part16/chapter_A.html#sect_TID_1002>`
     Observer Context"""

    def __init__(
        self,
        observer_type: CodedConcept,
        observer_identifying_attributes: (
            PersonObserverIdentifyingAttributes |
            DeviceObserverIdentifyingAttributes
        )
    ):
        """

        Parameters
        ----------
        observer_type: highdicom.sr.CodedConcept
            type of observer (see :dcm:`CID 270 <part16/sect_CID_270.html>`
            "Observer Type" for options)
        observer_identifying_attributes: Union[highdicom.sr.PersonObserverIdentifyingAttributes, highdicom.sr.DeviceObserverIdentifyingAttributes]
            observer identifying attributes

        """  # noqa: E501
        super().__init__()
        observer_type_item = CodeContentItem(
            name=CodedConcept(
                value='121005',
                meaning='Observer Type',
                scheme_designator='DCM',
            ),
            value=observer_type,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(observer_type_item)
        if observer_type == codes.cid270.Person:
            if not isinstance(observer_identifying_attributes,
                              PersonObserverIdentifyingAttributes):
                raise TypeError(
                    'Observer identifying attributes must have '
                    f'type {PersonObserverIdentifyingAttributes.__name__} '
                    f'for observer type "{observer_type.meaning}".'
                )
        elif observer_type == codes.cid270.Device:
            if not isinstance(observer_identifying_attributes,
                              DeviceObserverIdentifyingAttributes):
                raise TypeError(
                    'Observer identifying attributes must have '
                    f'type {DeviceObserverIdentifyingAttributes.__name__} '
                    f'for observer type "{observer_type.meaning}".'
                )
        else:
            raise ValueError(
                'Argument "oberver_type" must be either "Person" or "Device".'
            )
        self.extend(observer_identifying_attributes)

    @property
    def observer_type(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: observer type"""
        return self[0].value

    @property
    def observer_identifying_attributes(self) -> (
        PersonObserverIdentifyingAttributes |
        DeviceObserverIdentifyingAttributes
    ):
        """Union[highdicom.sr.PersonObserverIdentifyingAttributes, highdicom.sr.DeviceObserverIdentifyingAttributes]:
        observer identifying attributes
        """  # noqa: E501
        if self.observer_type == codes.DCM.Device:
            return DeviceObserverIdentifyingAttributes.from_sequence(self)
        elif self.observer_type == codes.DCM.Person:
            return PersonObserverIdentifyingAttributes.from_sequence(self)
        else:
            raise ValueError(
                f'Unexpected observer type "{self.observer_type.meaning}"'
            )


class SubjectContextFetus(Template):

    """:dcm:`TID 1008 <part16/chapter_A.html#sect_TID_1008>`
     Subject Context Fetus"""

    def __init__(self, subject_id: str):
        """

        Parameters
        ----------
        subject_id: str
            identifier of the fetus for longitudinal tracking

        """
        super().__init__()
        subject_id_item = TextContentItem(
            name=CodedConcept(
                value='121030',
                meaning='Subject ID',
                scheme_designator='DCM'
            ),
            value=subject_id,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(subject_id_item)

    @property
    def subject_id(self) -> str:
        """str: subject identifier"""
        return self[0].value

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> Self:
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1008 "Subject Context, Fetus"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.SubjectContextFetus
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('subject_id', codes.DCM.SubjectID),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class SubjectContextSpecimen(Template):

    """:dcm:`TID 1009 <part16/chapter_A.html#sect_TID_1009>`
     Subject Context Specimen"""

    def __init__(
        self,
        uid: str,
        identifier: str | None = None,
        container_identifier: str | None = None,
        specimen_type: Code | CodedConcept | None = None
    ):
        """

        Parameters
        ----------
        uid: str
            Unique identifier of the observed specimen
        identifier: Union[str, None], optional
            Identifier of the observed specimen (may have limited scope,
            e.g., only relevant with respect to the corresponding container)
        container_identifier: Union[str, None], optional
            Identifier of the container holding the specimen (e.g., a glass
            slide)
        specimen_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
            Type of the specimen (see
            :dcm:`CID 8103 <part16/sect_CID_8103.html>`
            "Anatomic Pathology Specimen Types" for options)

        """  # noqa: E501
        super().__init__()
        specimen_uid_item = UIDRefContentItem(
            name=CodedConcept(
                value='121039',
                meaning='Specimen UID',
                scheme_designator='DCM'
            ),
            value=uid,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(specimen_uid_item)
        if identifier is not None:
            specimen_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='121041',
                    meaning='Specimen Identifier',
                    scheme_designator='DCM'
                ),
                value=identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(specimen_identifier_item)
        if specimen_type is not None:
            specimen_type_item = CodeContentItem(
                name=CodedConcept(
                    value='371439000',
                    meaning='Specimen Type',
                    scheme_designator='SCT'
                ),
                value=specimen_type,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(specimen_type_item)
        if container_identifier is not None:
            container_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='111700',
                    meaning='Specimen Container Identifier',
                    scheme_designator='DCM'
                ),
                value=container_identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(container_identifier_item)

    @property
    def specimen_uid(self) -> str:
        """str: unique specimen identifier"""
        return self[0].value

    @property
    def specimen_identifier(self) -> str | None:
        """Union[str, None]: specimen identifier"""
        matches = [
            item for item in self
            if item.name == codes.DCM.SpecimenIdentifier
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Specimen Identifier" content '
                'item in "Subject Context Specimen" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def container_identifier(self) -> str | None:
        """Union[str, None]: specimen container identifier"""
        matches = [
            item for item in self
            if item.name == codes.DCM.SpecimenContainerIdentifier
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Specimen Container Identifier" content '
                'item in "Subject Context Specimen" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def specimen_type(self) -> CodedConcept | None:
        """Union[highdicom.sr.CodedConcept, None]: type of specimen"""
        matches = [
            item for item in self
            if item.name == codes.SCT.SpecimenType
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Specimen Type" content '
                'item in "Subject Context Specimen" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_image(
        cls,
        image: Dataset,
    ) -> Self:
        """Deduce specimen information from an existing image.

        This is appropriate, for example, when copying the specimen information
        from a source image into a derived SR or similar object.

        Parameters
        ----------
        image: pydicom.Dataset
            An image from which to infer specimen information. There is no
            limitation on the type of image, however it must have the Specimen
            module included.

        Raises
        ------
        ValueError:
            If the input image does not contain specimen information.

        """
        if not hasattr(image, 'ContainerIdentifier'):
            raise ValueError("Image does not contain specimen information.")

        description = image.SpecimenDescriptionSequence[0]

        # Specimen type code sequence is optional
        if hasattr(description, 'SpecimenTypeCodeSequence'):
            specimen_type: CodedConcept | None = CodedConcept.from_dataset(
                description.SpecimenTypeCodeSequence[0]
            )
        else:
            specimen_type = None

        return cls(
            container_identifier=image.ContainerIdentifier,
            identifier=description.SpecimenIdentifier,
            uid=description.SpecimenUID,
            specimen_type=specimen_type,
        )

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> Self:
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1009 "Subject Context, Specimen"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.SubjectContextSpecimen
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('uid', codes.DCM.SpecimenUID),
            ('identifier', codes.DCM.SpecimenIdentifier),
            ('container_identifier', codes.DCM.SpecimenContainerIdentifier),
            ('specimen_type', codes.SCT.SpecimenType),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class SubjectContextDevice(Template):

    """:dcm:`TID 1010 <part16/chapter_A.html#sect_TID_1010>`
     Subject Context Device"""

    def __init__(
        self,
        name: str,
        uid: str | None = None,
        manufacturer_name: str | None = None,
        model_name: str | None = None,
        serial_number: str | None = None,
        physical_location: str | None = None
    ):
        """

        Parameters
        ----------
        name: str
            name of the observed device
        uid: Union[str, None], optional
            unique identifier of the observed device
        manufacturer_name: Union[str, None], optional
            name of the observed device's manufacturer
        model_name: Union[str, None], optional
            name of the observed device's model
        serial_number: Union[str, None], optional
            serial number of the observed device
        physical_location: str, optional
            physical location of the observed device during the procedure

        """
        super().__init__()
        device_name_item = TextContentItem(
            name=codes.DCM.DeviceSubjectName,
            value=name,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(device_name_item)
        if uid is not None:
            device_uid_item = UIDRefContentItem(
                name=codes.DCM.DeviceSubjectUID,
                value=uid,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(device_uid_item)
        if manufacturer_name is not None:
            manufacturer_name_item = TextContentItem(
                name=CodedConcept(
                    value='121194',
                    meaning='Device Subject Manufacturer',
                    scheme_designator='DCM',
                ),
                value=manufacturer_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(manufacturer_name_item)
        if model_name is not None:
            model_name_item = TextContentItem(
                name=CodedConcept(
                    value='121195',
                    meaning='Device Subject Model Name',
                    scheme_designator='DCM',
                ),
                value=model_name,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(model_name_item)
        if serial_number is not None:
            serial_number_item = TextContentItem(
                name=CodedConcept(
                    value='121196',
                    meaning='Device Subject Serial Number',
                    scheme_designator='DCM',
                ),
                value=serial_number,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(serial_number_item)
        if physical_location is not None:
            physical_location_item = TextContentItem(
                name=codes.DCM.DeviceSubjectPhysicalLocationDuringObservation,
                value=physical_location,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(physical_location_item)

    @property
    def device_name(self) -> str:
        """str: name of device"""
        return self[0].value

    @property
    def device_uid(self) -> str | None:
        """Union[str, None]: unique device identifier"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectUID
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject UID" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_manufacturer_name(self) -> str | None:
        """Union[str, None]: name of device manufacturer"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectManufacturer
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Manufacturer" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_model_name(self) -> str | None:
        """Union[str, None]: name of device model"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectModelName
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Model Name" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_serial_number(self) -> str | None:
        """Union[str, None]: device serial number"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectSerialNumber
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Serial Number" content '
                'item in "Subject Context Device" template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @property
    def device_physical_location(self) -> str | None:
        """Union[str, None]: location of device"""
        matches = [
            item for item in self
            if item.name == codes.DCM.DeviceSubjectPhysicalLocationDuringObservation  # noqa: E501
        ]
        if len(matches) > 1:
            logger.warning(
                'found more than one "Device Subject Physical Location '
                'During Observation" content item in "Subject Context Device" '
                'template'
            )
        if len(matches) > 0:
            return matches[0].value
        return None

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = False
    ) -> Self:
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing SR Content Items of template
            TID 1010 "Subject Context, Device"
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.sr.SubjectContextDevice
            Content Sequence containing SR Content Items

        """
        attr_codes = [
            ('name', codes.DCM.DeviceSubjectName),
            ('uid', codes.DCM.DeviceSubjectUID),
            ('manufacturer_name', codes.DCM.DeviceSubjectManufacturer),
            ('model_name', codes.DCM.DeviceSubjectModelName),
            ('serial_number', codes.DCM.DeviceSubjectSerialNumber),
            (
                'physical_location',
                codes.DCM.DeviceSubjectPhysicalLocationDuringObservation
            ),
        ]
        kwargs = {}
        for dataset in sequence:
            dataset_copy = deepcopy(dataset)
            content_item = ContentItem._from_dataset_derived(dataset_copy)
            for param, name in attr_codes:
                if content_item.name == name:
                    kwargs[param] = content_item.value
        return cls(**kwargs)


class SubjectContext(Template):

    """:dcm:`TID 1006 <part16/chapter_A.html#sect_TID_1006>`
     Subject Context"""

    def __init__(
        self,
        subject_class: CodedConcept,
        subject_class_specific_context: (
            SubjectContextFetus |
            SubjectContextSpecimen |
            SubjectContextDevice
        )
    ):
        """

        Parameters
        ----------
        subject_class: highdicom.sr.CodedConcept
            type of subject if the subject of the report is not the patient
            (see :dcm:`CID 271 <part16/sect_CID_271.html>`
            "Observation Subject Class" for options)
        subject_class_specific_context: Union[highdicom.sr.SubjectContextFetus, highdicom.sr.SubjectContextSpecimen, highdicom.sr.SubjectContextDevice], optional
            additional context information specific to `subject_class`

        """  # noqa: E501
        super().__init__()
        subject_class_item = CodeContentItem(
            name=CodedConcept(
                value='121024',
                meaning='Subject Class',
                scheme_designator='DCM'
            ),
            value=subject_class,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(subject_class_item)
        if isinstance(subject_class_specific_context, SubjectContextSpecimen):
            if subject_class != codes.DCM.Specimen:
                raise TypeError(
                    'Subject class specific context doesn\'t match '
                    'subject class "Specimen".'
                )
        elif isinstance(subject_class_specific_context, SubjectContextFetus):
            if subject_class != codes.DCM.Fetus:
                raise TypeError(
                    'Subject class specific context doesn\'t match '
                    'subject class "Fetus".'
                )
        elif isinstance(subject_class_specific_context, SubjectContextDevice):
            if subject_class != codes.DCM.Device:
                raise TypeError(
                    'Subject class specific context doesn\'t match '
                    'subject class "Device".'
                )
        else:
            raise TypeError('Unexpected subject class specific context.')
        self.extend(subject_class_specific_context)

    @classmethod
    def from_image(cls, image: Dataset) -> Self | None:
        """Get a subject context inferred from an existing image.

        Currently this is only supported for subjects that are specimens.

        Parameters
        ----------
        image: pydicom.Dataset
            Dataset of an existing DICOM image object
            containing metadata on the imaging subject. Highdicom will attempt
            to infer the subject context from this image. If successful, it
            will be returned as a ``SubjectContext``, otherwise ``None``.

        Returns
        -------
        Optional[highdicom.sr.SubjectContext]:
            SubjectContext, if it can be inferred from the image. Otherwise,
            ``None``.

        """
        try:
            subject_context_specimen = SubjectContextSpecimen.from_image(
                image
            )
        except ValueError:
            pass
        else:
            return cls(
                subject_class=codes.DCM.Specimen,
                subject_class_specific_context=subject_context_specimen,
            )

        return None

    @property
    def subject_class(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: type of subject"""
        return self[0].value

    @property
    def subject_class_specific_context(self) -> (
        SubjectContextFetus |
        SubjectContextSpecimen |
        SubjectContextDevice
    ):
        """Union[highdicom.sr.SubjectContextFetus, highdicom.sr.SubjectContextSpecimen, highdicom.sr.SubjectContextDevice]:
        subject class specific context
        """  # noqa: E501
        if self.subject_class == codes.DCM.Specimen:
            return SubjectContextSpecimen.from_sequence(sequence=self)
        elif self.subject_class == codes.DCM.Fetus:
            return SubjectContextFetus.from_sequence(sequence=self)
        elif self.subject_class == codes.DCM.Device:
            return SubjectContextDevice.from_sequence(sequence=self)
        else:
            raise ValueError('Unexpected subject class "{item.meaning}".')


class ObservationContext(Template):

    """:dcm:`TID 1001 <part16/chapter_A.html#sect_TID_1001>`
     Observation Context"""

    def __init__(
        self,
        observer_person_context: ObserverContext | None = None,
        observer_device_context: ObserverContext | None = None,
        subject_context: SubjectContext | None = None
    ):
        """

        Parameters
        ----------
        observer_person_context: Union[highdicom.sr.ObserverContext, None], optional
            description of the person that reported the observation
        observer_device_context: Union[highdicom.sr.ObserverContext, None], optional
            description of the device that was involved in reporting the
            observation
        subject_context: Union[highdicom.sr.SubjectContext, None], optional
            description of the imaging subject in case it is not the patient
            for which the report is generated (e.g., a pathology specimen in
            a whole-slide microscopy image, a fetus in an ultrasound image, or
            a pacemaker device in a chest X-ray image)

        """  # noqa: E501
        super().__init__()
        if observer_person_context is not None:
            if not isinstance(observer_person_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_person_context" must '
                    f'have type {ObserverContext.__name__}'
                )
            self.extend(observer_person_context)
        if observer_device_context is not None:
            if not isinstance(observer_device_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_device_context" must '
                    f'have type {ObserverContext.__name__}'
                )
            self.extend(observer_device_context)
        if subject_context is not None:
            if not isinstance(subject_context, SubjectContext):
                raise TypeError(
                    f'Argument "subject_context" must have '
                    f'type {SubjectContext.__name__}'
                )
            self.extend(subject_context)


class LanguageOfValue(CodeContentItem):
    """:dcm:`TID 1201 <part16/chapter_A.html#sect_TID_1201>`
    Language of Value
    """

    def __init__(self,
                 language: Union[Code, CodedConcept],
                 country_of_language: Optional[Union[Code, CodedConcept]] = None
                 ) -> None:
        super().__init__(
            name=codes.DCM.LanguageOfValue,
            value=language,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        content = ContentSequence()
        if country_of_language is not None:
            country_of_language_item = CodeContentItem(
                name=codes.DCM.CountryOfLanguage,
                value=country_of_language,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            content.append(country_of_language_item)
        if len(content) > 0:
            self.ContentSequence = content
