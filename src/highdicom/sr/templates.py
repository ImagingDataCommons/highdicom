"""DICOM structured reporting templates."""
from typing import Optional, Sequence, Union

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes

from highdicom.sr.coding import CodedConcept
from highdicom.sr.content import (
    FindingSite,
    LongitudinalTemporalOffsetFromEvent,
    ImageRegion,
    ImageRegion3D,
    VolumeSurface,
    RealWorldValueMap,
    ReferencedSegment,
    ReferencedSegmentationFrame,
    SourceImageForMeasurement,
)
from highdicom.sr.enum import RelationshipTypeValues
from highdicom.sr.value_types import (
    CodeContentItem,
    ContainerContentItem,
    ContentItem,
    ContentSequence,
    NumContentItem,
    TextContentItem,
    UIDRefContentItem,
)
from highdicom.uid import UID


DEFAULT_LANGUAGE = CodedConcept(
    value='en-US',
    scheme_designator='RFC5646',
    meaning='English (United States)'
)


class Template(ContentSequence):

    """Abstract base class for a DICOM SR template."""

    def __init__(self, items: Optional[Sequence[ContentItem]] = None) -> None:
        """
        Parameters
        ----------
        items: Sequence[ContentItem], optional
            content items

        """
        super().__init__(items)


class AlgorithmIdentification(Template):

    """TID 4019 Algorithm Identification"""

    def __init__(
            self,
            name: str,
            version: str,
            parameters: Optional[Sequence[str]] = None
        ) -> None:
        """
        Parameters
        ----------
        name: str
            name of the algorithm
        version: str
            version of the algorithm
        parameters: Sequence[str], optional
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
                        meaning='Algorithm Parameter',
                        scheme_designator='DCM'
                    ),
                    value=param,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                self.append(parameter_item)


class TrackingIdentifier(Template):

    """TID 4108 Tracking Identifier"""

    def __init__(
            self,
            uid: Optional[str] = None,
            identifier: Optional[str] = None
        ):
        """
        Parameters
        ----------
        uid: Union[pydicom.uid.UID, str], optional
            globally unique identifier
        identifier: str, optional
            human readable identifier

        """
        super().__init__()
        if uid is None:
            uid = UID()
        if identifier is not None:
            tracking_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='112039',
                    meaning='Tracking Identifier',
                    scheme_designator='DCM'
                ),
                value=identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(tracking_identifier_item)
        tracking_uid_item = UIDRefContentItem(
            name=CodedConcept(
                value='112040',
                meaning='Tracking Unique Identifier',
                scheme_designator='DCM'
            ),
            value=uid,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(tracking_uid_item)


class TimePointContext(Template):

    """TID 1502 Time Point Context"""

    def __init__(
            self,
            time_point: str,
            time_point_type: Optional[Union[CodedConcept, Code]] = None,
            time_point_order: Optional[int] = None,
            subject_time_point_identifier: Optional[
                LongitudinalTemporalOffsetFromEvent
            ] = None,
            protocol_time_point_identifier: Optional[str] = None,
            temporal_offset_from_event: Optional[
                LongitudinalTemporalOffsetFromEvent
            ] = None,
            temporal_event_type: Optional[Union[CodedConcept, Code]] = None
        ):
        """
        Parameters
        ----------
        time_point: str
            actual value represention of the time point
        time_point_type: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded type of time point, e.g., "Baseline" or "Posttreatment"
            (see CID 646 "Time Point Types" for options)
        time_point_order: int, optional
            number indicating the order of a time point relative to other
            time points in a time series
        subject_time_point_identifier: str, optional
           identifier of a specific time point in a time series, which is
           unique within an appropriate local context and specific to a
           particular subject (patient)
        protocol_time_point_identifier: str, optional
           identifier of a specific time point in a time series, which is
           unique within an appropriate local context and specific to a
           particular protocol using the same value for different subjects
        temporal_offset_from_event: highdicom.sr.content.LongitudinalTemporalOffsetFromEvent, optional
            offset in time from a particular event of significance, e.g., the
            baseline of an imaging study or enrollment into a clincal trial
        temporal_event_type: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            type of event to which `temporal_offset_from_event` is relative,
            e.g., "Baseline" or "Enrollment"
            (required if `temporal_offset_from_event` is provided)

        """  # noqa
        time_point_item = TextContentItem(
            name=CodedConcept(
                value='C2348792',
                meaning='Time Point',
                scheme_designator='UMLS'
            ),
            value=time_point,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        self.append(time_point_item)
        if time_point_type is not None:
            time_point_type_item = CodeContentItem(
                name=CodedConcept(
                    value='126072',
                    meaning='Time Point Type',
                    scheme_designator='DCM'
                ),
                value=time_point_type,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(time_point_type_item)
        if time_point_order is not None:
            time_point_order_item = NumContentItem(
                name=CodedConcept(
                    value='126073',
                    meaning='Time Point Order',
                    scheme_designator='DCM'
                ),
                value=time_point_order,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(time_point_order_item)
        if subject_time_point_identifier is not None:
            subject_time_point_identifier_item = NumContentItem(
                name=CodedConcept(
                    value='126070',
                    meaning='Subject Time Point Identifier',
                    scheme_designator='DCM'
                ),
                value=subject_time_point_identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(subject_time_point_identifier_item)
        if protocol_time_point_identifier is not None:
            protocol_time_point_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='126071',
                    meaning='Protocol Time Point Identifier',
                    scheme_designator='DCM'
                ),
                value=protocol_time_point_identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(protocol_time_point_identifier_item)
        if temporal_offset_from_event is not None:
            if not isinstance(temporal_offset_from_event,
                              LongitudinalTemporalOffsetFromEvent):
                raise TypeError(
                    'Argument "temporal_offset_from_event" must have type '
                    'LongitudinalTemporalOffsetFromEvent.'
                )
            self.append(temporal_offset_from_event)


class MeasurementStatisticalProperties(Template):

    """TID 311 Measurement Statistical Properties"""

    def __init__(
            self,
            values: Sequence[NumContentItem],
            description: Optional[str] = None,
            authority: Optional[str] = None
        ):
        """
        Parameters
        ----------
        values: Sequence[highdicom.sr.value_types.NumContentItem]
            reference values of the population of measurements, e.g., its
            mean or standard deviation
            (see CID 226 "Population Statistical Descriptors" and
            CID 227 227 "Sample Statistical Descriptors" for options)
        description: str, optional
            description of the reference population of measurements
        authority: str, optional
            authority for a description of the reference population of
            measurements

        """
        super().__init__()
        if not isinstance(values, (list, tuple)):
            raise TypeError('Argument "values" must be a list.')
        for concept in values:
            if not isinstance(concept, NumContentItem):
                raise ValueError(
                    'Items of argument "values" must have type '
                    'NumContentItem.'
                )
        self.extend(values)
        if description is not None:
            description_item = TextContentItem(
                name=CodedConcept(
                    value='121405',
                    meaning='Population Description',
                    scheme_designator='DCM'
                ),
                value=description,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(description_item)
        if authority is not None:
            authority_item = TextContentItem(
                name=CodedConcept(
                    value='121406',
                    meaning='Reference Authority',
                    scheme_designator='DCM'
                ),
                value=authority,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(authority_item)


class NormalRangeProperties(Template):

    """TID 312 Normal Range Properties"""

    def __init__(
            self,
            values: Sequence[NumContentItem],
            description: Optional[str] = None,
            authority: Optional[str] = None
        ):
        """
        Parameters
        ----------
        values: Sequence[highdicom.sr.value_types.NumContentItem]
            reference values of the normal range, e.g., its upper and lower
            bound
            (see CID 223 "Normal Range Values" for options)
        description: str, optional
            description of the normal range
        authority: str, optional
            authority for the description of the normal range

        """
        super().__init__()
        if not isinstance(values, (list, tuple)):
            raise TypeError('Argument "values" must be a list.')
        for concept in values:
            if not isinstance(concept, NumContentItem):
                raise ValueError(
                    'Items of argument "values" must have type '
                    'NumContentItem.'
                )
        self.extend(values)
        if description is not None:
            description_item = TextContentItem(
                name=codes.DCM.NormalRangeDescription,
                value=description,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(description_item)
        if authority is not None:
            authority_item = TextContentItem(
                name=codes.DCM.NormalRangeAuthority,
                value=authority,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(authority_item)


class MeasurementProperties(Template):

    """TID 310 Measurement Properties"""

    def __init__(
            self,
            normality: Optional[Union[CodedConcept, Code]] = None,
            level_of_significance: Optional[Union[CodedConcept, Code]] = None,
            selection_status: Optional[Union[CodedConcept, Code]] = None,
            measurement_statistical_properties: Optional[
                MeasurementStatisticalProperties
            ] = None,
            normal_range_properties: Optional[NormalRangeProperties] = None,
            upper_measurement_uncertainty: Optional[Union[int, float]] = None,
            lower_measurement_uncertainty: Optional[Union[int, float]] = None
        ):
        """
        Parameters
        ----------
        normality: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            the extend to which the measurement is considered normal or abnormal
            (see CID 222 "Normality Codes" for options)
        level_of_significance: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            the extend to which the measurement is considered normal or abnormal
            (see CID 220 "Level of Significance" for options)
        selection_status: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            how the measurement value was selected or computed from a set of
            available values
            (see CID 224 "Selection Method" for options)
        measurement_statistical_properties: highdicom.sr.templates.MeasurementStatisticalProperties, optional
            statistical properties of a reference population for a measurement
            and/or the position of a measurement in such a reference population
        normal_range_properties: highdicom.sr.templates.NormalRangeProperties, optional
            statistical properties of a reference population for a measurement
            and/or the position of a measurement in such a reference population
        upper_measurement_uncertainty: Union[int, float], optional
            upper range of measurment uncertainty
        lower_measurement_uncertainty: Union[int, float], optional
            lower range of measurment uncertainty

        """  # noqa
        super().__init__()
        if normality is not None:
            normality_item = CodeContentItem(
                name=CodedConcept(
                    value='121402',
                    meaning='Normality',
                    scheme_designator='DCM'
                ),
                value=normality,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(normality_item)
        if measurement_statistical_properties is not None:
            if not isinstance(measurement_statistical_properties,
                              MeasurementStatisticalProperties):
                raise TypeError(
                    'Argument "measurment_statistical_properties" must have '
                    'type MeasurementStatisticalProperties.'
                )
            self.extend(measurement_statistical_properties)
        if normal_range_properties is not None:
            if not isinstance(normal_range_properties,
                              NormalRangeProperties):
                raise TypeError(
                    'Argument "normal_range_properties" must have '
                    'type NormalRangeProperties.'
                )
            self.extend(normal_range_properties)
        if level_of_significance is not None:
            level_of_significance_item = CodeContentItem(
                name=CodedConcept(
                    value='121403',
                    meaning='Level of Significance',
                    scheme_designator='DCM'
                ),
                value=level_of_significance,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(level_of_significance_item)
        if selection_status is not None:
            selection_status_item = CodeContentItem(
                name=CodedConcept(
                    value='121404',
                    meaning='Selection Status',
                    scheme_designator='DCM'
                ),
                value=selection_status,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(selection_status_item)
        if upper_measurement_uncertainty is not None:
            upper_measurement_uncertainty_item = CodeContentItem(
                name=CodedConcept(
                    value='371886008',
                    meaning='+, range of upper measurement uncertainty',
                    scheme_designator='SCT'
                ),
                value=upper_measurement_uncertainty,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(upper_measurement_uncertainty_item)
        if lower_measurement_uncertainty is not None:
            lower_measurement_uncertainty_item = CodeContentItem(
                name=CodedConcept(
                    value='371885007',
                    meaning='-, range of lower measurement uncertainty',
                    scheme_designator='SCT'
                ),
                value=lower_measurement_uncertainty,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            self.append(lower_measurement_uncertainty_item)


class PersonObserverIdentifyingAttributes(Template):

    """TID 1003 Person Observer Identifying Attributes"""

    def __init__(
            self,
            name: str,
            login_name: Optional[str] = None,
            organization_name: Optional[str] = None,
            role_in_organization: Optional[Union[CodedConcept, Code]] = None,
            role_in_procedure: Optional[Union[CodedConcept, Code]] = None
        ):
        """
        Parameters
        ----------
        name: str
            name of the person
        login_name: str
            login name of the person
        organization_name: str, optional
            name of the person's organization
        role_in_organization: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            role of the person within the organization
        role_in_procedure: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            role of the person in the reported procedure

        """  # noqa
        super().__init__()
        name_item = TextContentItem(
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


class DeviceObserverIdentifyingAttributes(Template):

    """TID 1004 Device Observer Identifying Attributes"""

    def __init__(
            self,
            uid: str,
            name: Optional[str] = None,
            manufacturer_name: Optional[str] = None,
            model_name: Optional[str] = None,
            serial_number: Optional[str] = None,
            physical_location: Optional[str] = None,
            role_in_procedure: Optional[str] = None
        ):
        """
        Parameters
        ----------
        uid: str
            device UID
        name: str, optional
            name of device
        manufacturer_name: str, optional
            name of device's manufacturer
        model_name: str, optional
            name of the device's model
        serial_number: str, optional
            serial number of the device
        physical_location: str, optional
            physical location of the device during the procedure
        role_in_procedure: str, optional
            role of the device in the reported procedure

        """
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
        if manufacturer_name is not None:
            manufacturer_name_item = TextContentItem(
                name=CodedConcept(
                    value='121013',
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


class ObserverContext(Template):

    """TID 1002 Observer Context"""

    def __init__(
            self,
            observer_type: CodedConcept,
            observer_identifying_attributes: Union[
                PersonObserverIdentifyingAttributes,
                DeviceObserverIdentifyingAttributes
            ]
        ):
        """
        Parameters
        ----------
        observer_type: highdicom.sr.coding.CodedConcept
            type of observer
            (see CID 270 "Observer Type" for options)
        observer_identifying_attributes: Union[highdicom.sr.templates.PersonObserverIdentifyingAttributes, highdicom.sr.templates.DeviceObserverIdentifyingAttributes]
            observer identifying attributes

        """  # noqa
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
                    'type {} for observer type "{}".'.format(
                        PersonObserverIdentifyingAttributes.__name__,
                        observer_type.meaning
                    )
                )
        elif observer_type == codes.cid270.Device:
            if not isinstance(observer_identifying_attributes,
                              DeviceObserverIdentifyingAttributes):
                raise TypeError(
                    'Observer identifying attributes must have '
                    'type {} for observer type "{}".'.format(
                        DeviceObserverIdentifyingAttributes.__name__,
                        observer_type.meaning,
                    )
                )
        else:
            raise ValueError(
                'Argument "oberver_type" must be either "Person" or "Device".'
            )
        self.extend(observer_identifying_attributes)


class SubjectContextFetus(Template):

    """TID 1008 Subject Context Fetus"""

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


class SubjectContextSpecimen(Template):

    """TID 1009 Subject Context Specimen"""

    def __init__(
            self,
            uid: str,
            identifier: Optional[str] = None,
            container_identifier: Optional[str] = None,
            specimen_type: Optional[str] = None
        ):
        """
        Parameters
        ----------
        uid: str
            unique identifier of the observed specimen
        identifier: str, optional
            identifier of the observed specimen (may have limited scope,
            e.g., only relevant with respect to the corresponding container)
        container_identifier: str, optional
            identifier of the container holding the speciment (e.g., a glass
            slide)
        specimen_type: highdicom.sr.coding.CodedConcept, optional
            type of the specimen
            (see CID 8103 "Anatomic Pathology Specimen Types" for options)

        """
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
        if specimen_type is not None:
            specimen_type_item = CodeContentItem(
                name=CodedConcept(
                    value='121042',
                    meaning='Specimen Type',
                    scheme_designator='DCM'
                ),
                value=specimen_type,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            self.append(specimen_type_item)


class SubjectContextDevice(Template):

    """TID 1010 Subject Context Device"""

    def __init__(
            self,
            name: str,
            uid: Optional[str] = None,
            manufacturer_name: Optional[str] = None,
            model_name: Optional[str] = None,
            serial_number: Optional[str] = None,
            physical_location: Optional[str] = None
        ):
        """
        Parameters
        ----------
        name: str
            name of the observed device
        uid: str, optional
            unique identifier of the observed device
        manufacturer_name: str, optional
            name of the observed device's manufacturer
        model_name: str, optional
            name of the observed device's model
        serial_number: str, optional
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


class SubjectContext(Template):

    """TID 1006 Subject Context"""

    def __init__(
            self,
            subject_class: CodedConcept,
            subject_class_specific_context: Optional[
                Union[
                    SubjectContextFetus,
                    SubjectContextSpecimen,
                    SubjectContextDevice
                ]
            ] = None
        ):
        """
        Parameters
        ----------
        subject_class: highdicom.sr.coding.CodedConcept
            type of subject if the subject of the report is not the patient
            (see CID 271 "Observation Subject Class" for options)
        subject_class_specific_context: Union[highdicom.sr.templates.SubjectContextFetus, highdicom.sr.templates.SubjectContextSpecimen, highdicom.sr.templates.SubjectContextDevice], optional
            additional context information specific to `subject_class`

        """  # noqa
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
        if subject_class_specific_context is not None:
            self.extend(subject_class_specific_context)


class ObservationContext(Template):

    """TID 1001 Observation Context"""

    def __init__(
            self,
            observer_person_context: Optional[ObserverContext] = None,
            observer_device_context: Optional[ObserverContext] = None,
            subject_context: Optional[SubjectContext] = None
        ):
        """
        Parameters
        ----------
        observer_person_context: [highdicom.sr.templates.ObserverContext, None], optional
            description of the person that reported the observation
        observer_device_context: highdicom.sr.templates.ObserverContext, optional
            description of the device that was involved in reporting the
            observation
        subject_context: highdicom.sr.templates.SubjectContext, optional
            description of the imaging subject in case it is not the patient
            for which the report is generated (e.g., a pathology specimen in
            a whole-slide microscopy image, a fetus in an ultrasound image, or
            a pacemaker device in a chest X-ray image)

        """  # noqa
        super().__init__()
        if observer_person_context is not None:
            if not isinstance(observer_person_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_person_context" must '
                    'have type {}'.format(
                        ObserverContext.__name__
                    )
                )
            self.extend(observer_person_context)
        if observer_device_context is not None:
            if not isinstance(observer_device_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_device_context" must '
                    'have type {}'.format(
                        ObserverContext.__name__
                    )
                )
            self.extend(observer_device_context)
        if subject_context is not None:
            if not isinstance(subject_context, SubjectContext):
                raise TypeError(
                    'Argument "subject_context" must have type {}'.format(
                        SubjectContext.__name__
                    )
                )
            self.extend(subject_context)


class LanguageOfContentItemAndDescendants(Template):

    """TID 1204 Language of Content Item and Descendants"""

    def __init__(self, language: CodedConcept):
        """
        Parameters
        ----------
        language: highdicom.sr.coding.CodedConcept
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


class Measurement(Template):

    """TID 300 Measurement"""

    def __init__(
            self,
            name: Union[CodedConcept, Code],
            tracking_identifier: Optional[TrackingIdentifier] = None,
            value: Optional[Union[int, float]] = None,
            unit: Optional[Union[CodedConcept, Code]] = None,
            qualifier: Optional[Union[CodedConcept, Code]] = None,
            algorithm_id: Optional[AlgorithmIdentification] = None,
            derivation: Optional[Union[CodedConcept, Code]] = None,
            finding_sites: Optional[Sequence[FindingSite]] = None,
            method: Optional[Union[CodedConcept, Code]] = None,
            properties: Optional[MeasurementProperties] = None,
            referenced_images: Optional[
                Sequence[SourceImageForMeasurement]
            ] = None,
            referenced_real_world_value_map: Optional[RealWorldValueMap] = None
        ):
        """
        Parameters
        ----------
        name: highdicom.sr.coding.CodedConcept
            name of the measurement
            (see CID 7469 "Generic Intensity and Size Measurements" and
            CID 7468 "Texture Measurements" for options)
        tracking_identifier: highdicom.sr.templates.TrackingIdentifier, optional
            identifier for tracking measurements
        value: Union[int, float], optional
            numeric measurement value
        unit: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            unit of the numeric measurement value
            (see CID 7181 "Abstract Multi-dimensional Image Model Component
            Units" for options)
        qualifier: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            qualification of numeric measurement value or as an alternative
            qualitative description
        algorithm_id: highdicom.sr.templates.AlgorithmIdentification, optional
            identification of algorithm used for making measurements
        derivation: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            how the value was computed
            (see CID 7464 "General Region of Interest Measurement Modifiers"
            for options)
        finding_sites: Sequence[highdicom.sr.content.FindingSite], optional
            Coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        method: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            measurement method
            (see CID 6147 "Response Criteria" for options)
        properties: highdicom.sr.templates.MeasurementProperties, optional
            measurement properties, including evaluations of its normality
            and/or significance, its relationship to a reference population,
            and an indication of its selection from a set of measurements
        referenced_images: Sequence[highdicom.sr.content.SourceImageForMeasurement], optional
            referenced images which were used as sources for the measurement
        referenced_real_world_value_map: highdicom.sr.content.RealWorldValueMap, optional
            referenced real world value map for referenced source images

        """  # noqa
        super().__init__()
        value_item = NumContentItem(
            name=name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        value_item.ContentSequence = ContentSequence()
        if tracking_identifier is not None:
            if not isinstance(tracking_identifier, TrackingIdentifier):
                raise TypeError(
                    'Argument "tracking_identifier" must have type '
                    'TrackingIdentifier.'
                )
            value_item.ContentSequence.extend(tracking_identifier)
        if method is not None:
            method_item = CodeContentItem(
                name=CodedConcept(
                    value='370129005',
                    meaning='Measurement Method',
                    scheme_designator='SCT'
                ),
                value=method,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            value_item.ContentSequence.append(method_item)
        if derivation is not None:
            derivation_item = CodeContentItem(
                name=CodedConcept(
                    value='121401',
                    meaning='Derivation',
                    scheme_designator='DCM'
                ),
                value=derivation,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            value_item.ContentSequence.append(derivation_item)
        if finding_sites is not None:
            if not isinstance(finding_sites, (list, tuple, set)):
                raise TypeError(
                    'Argument "finding_sites" must be a sequence.'

                )
            for site in finding_sites:
                if not isinstance(site, FindingSite):
                    raise TypeError(
                        'Items of argument "finding_sites" must have '
                        'type FindingSite.'
                    )
                value_item.ContentSequence.append(site)
        if properties is not None:
            if not isinstance(properties, MeasurementProperties):
                raise TypeError(
                    'Argument "properties" must have '
                    'type MeasurementProperties.'
                )
            value_item.ContentSequence.extend(properties)
        if referenced_images is not None:
            for image in referenced_images:
                if not isinstance(image, SourceImageForMeasurement):
                    raise TypeError(
                        'Arguments "referenced_images" must have type '
                        'SourceImageForMeasurement.'
                    )
                value_item.ContentSequence.append(image)
        if referenced_real_world_value_map is not None:
            if not isinstance(referenced_real_world_value_map,
                              RealWorldValueMap):
                raise TypeError(
                    'Argument "referenced_real_world_value_map" must have type '
                    'RealWorldValueMap.'
                )
            value_item.ContentSequence.append(referenced_real_world_value_map)
        if algorithm_id is not None:
            if not isinstance(algorithm_id, AlgorithmIdentification):
                raise TypeError(
                    'Argument "algorithm_id" must have type '
                    'AlgorithmIdentification.'
                )
            value_item.ContentSequence.extend(algorithm_id)
        self.append(value_item)


class MeasurementsAndQualitativeEvaluations(Template):

    """TID 1501 Measurement and Qualitative Evaluation Group"""

    def __init__(
            self,
            tracking_identifier: TrackingIdentifier,
            referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
            time_point_context: Optional[TimePointContext] = None,
            finding_type: Optional[Union[CodedConcept, Code]] = None,
            method: Optional[Union[CodedConcept, Code]] = None,
            algorithm_id: Optional[AlgorithmIdentification] = None,
            finding_sites: Optional[Sequence[FindingSite]] = None,
            session: Optional[str] = None,
            measurements: Sequence[Measurement] = None,
            qualitative_evaluations: Optional[Sequence[CodeContentItem]] = None
        ):
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.templates.TrackingIdentifier
            Identifier for tracking measurements
        referenced_real_world_value_map: highdicom.sr.content.RealWorldValueMap, optional
            Referenced real world value map for region of interest
        time_point_context: highdicom.sr.templates.TimePointContext, optional
            Description of the time point context
        finding_type: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            Type of object that was measured, e.g., organ or tumor
        method: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded measurement method
            (see CID 6147 "Response Criteria" for options)
        algorithm_id: highdicom.sr.templates.AlgorithmIdentification, optional
            identification of algorithm used for making measurements
        finding_sites: Sequence[highdicom.sr.content.FindingSite], optional
            Coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        session: str, optional
            Description of the session
        measurements: Sequence[highdicom.sr.templates.Measurement], optional
            Numeric measurements
        qualitative_evaluations: Sequence[highdicom.sr.value_types.CodeContentItem], optional
            Coded name-value pairs that describe measurements in qualitative
            terms

        """  # noqa
        super().__init__()
        group_item = ContainerContentItem(
            name=CodedConcept(
                value='125007',
                meaning='Measurement Group',
                scheme_designator='DCM'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        group_item.ContentSequence = ContentSequence()
        if not isinstance(tracking_identifier, TrackingIdentifier):
            raise TypeError(
                'Argument "tracking_identifier" must have type '
                'TrackingIdentifier.'
            )
        if len(tracking_identifier) != 2:
            raise ValueError(
                'Argument "tracking_identifier" must include a '
                'human readable tracking identifier and a tracking unique '
                'identifier.'
            )
        group_item.ContentSequence.extend(tracking_identifier)
        if session is not None:
            session_item = TextContentItem(
                name=CodedConcept(
                    value='C67447',
                    meaning='Activity Session',
                    scheme_designator='NCIt'
                ),
                value=session,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
            )
            group_item.ContentSequence.append(session_item)
        if finding_type is not None:
            finding_type_item = CodeContentItem(
                name=CodedConcept(
                    value='121071',
                    meaning='Finding',
                    scheme_designator='DCM'
                ),
                value=finding_type,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            group_item.ContentSequence.append(finding_type_item)
        if method is not None:
            method_item = CodeContentItem(
                name=CodedConcept(
                    value='370129005',
                    meaning='Measurement Method',
                    scheme_designator='SCT'
                ),
                value=method,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            group_item.ContentSequence.append(method_item)
        if finding_sites is not None:
            if not isinstance(finding_sites, (list, tuple, set)):
                raise TypeError(
                    'Argument "finding_sites" must be a sequence.'

                )
            for site in finding_sites:
                if not isinstance(site, FindingSite):
                    raise TypeError(
                        'Items of argument "finding_sites" must have '
                        'type FindingSite.'
                    )
                group_item.ContentSequence.append(site)
        if algorithm_id is not None:
            if not isinstance(algorithm_id, AlgorithmIdentification):
                raise TypeError(
                    'Argument "algorithm_id" must have type '
                    'AlgorithmIdentification.'
                )
            group_item.ContentSequence.extend(algorithm_id)
        if time_point_context is not None:
            if not isinstance(time_point_context, TimePointContext):
                raise TypeError(
                    'Argument "time_point_context" must have type '
                    'TimePointContext.'
                )
            group_item.ContentSequence.append(time_point_context)
        if referenced_real_world_value_map is not None:
            if not isinstance(referenced_real_world_value_map,
                              RealWorldValueMap):
                raise TypeError(
                    'Argument "referenced_real_world_value_map" must have type '
                    'RealWorldValueMap.'
                )
            group_item.ContentSequence.append(referenced_real_world_value_map)
        if measurements is not None:
            for measurement in measurements:
                if not isinstance(measurement, Measurement):
                    raise TypeError(
                        'Items of argument "measurements" must have '
                        'type Measurement.'
                    )
                group_item.ContentSequence.extend(measurement)
            group_item.ContentSequence.extend(measurement)
        if qualitative_evaluations is not None:
            for evaluation in qualitative_evaluations:
                if not isinstance(evaluation, CodeContentItem):
                    raise TypeError(
                        'Items of argument "qualitative_evaluations" must have '
                        'type CodeContentItem.'
                    )
                group_item.ContentSequence.append(evaluation)
        self.append(group_item)


class _ROIMeasurementsAndQualitativeEvaluations(
        MeasurementsAndQualitativeEvaluations):

    """Abstract base class for ROI Measurements and Qualitative Evaluation
    templates."""

    def __init__(
            self,
            tracking_identifier: TrackingIdentifier,
            referenced_regions: Optional[
                Union[Sequence[ImageRegion], Sequence[ImageRegion3D]]
            ] = None,
            referenced_segment: Optional[
                Union[ReferencedSegment, ReferencedSegmentationFrame]
            ] = None,
            referenced_volume_surface: Optional[VolumeSurface] = None,
            referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
            time_point_context: Optional[TimePointContext] = None,
            finding_type: Optional[Union[CodedConcept, Code]] = None,
            method: Optional[Union[CodedConcept, Code]] = None,
            algorithm_id: Optional[AlgorithmIdentification] = None,
            finding_sites: Optional[Sequence[FindingSite]] = None,
            session: Optional[str] = None,
            measurements: Sequence[Measurement] = None,
            qualitative_evaluations: Optional[Sequence[CodeContentItem]] = None
        ):
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_regions: Union[Sequence[highdicom.sr.content.ImageRegion], Sequence[highdicom.sr.content.ImageRegion3D]], optional
            regions of interest in source image(s)
        referenced_segment: Union[highdicom.sr.content.ReferencedSegment, highdicom.sr.content.ReferencedSegmentationFrame], optional
            segmentation for region of interest in source image
        referenced_volume_surface: hidicom.sr.content.VolumeSurface, optional
            surface segmentation for region of interest in source image
        referenced_real_world_value_map: highdicom.sr.content.RealWorldValueMap, optional
            referenced real world value map for region of interest
        time_point_context: highdicom.sr.templates.TimePointContext, optional
            description of the time point context
        finding_type: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            type of object that was measured, e.g., organ or tumor
        method: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded measurement method
            (see CID 6147 "Response Criteria" for options)
        algorithm_id: highdicom.sr.templates.AlgorithmIdentification, optional
            identification of algorithm used for making measurements
        finding_sites: Sequence[highdicom.sr.content.FindingSite], optional
            Coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        session: str, optional
            description of the session
        measurements: Sequence[highdicom.sr.templates.Measurement], optional
            numeric measurements
        qualitative_evaluations: Sequence[highdicom.sr.value_types.CodeContentItem], optional
            coded name-value (question-answer) pairs that describe the
            measurements in qualitative terms

        Note
        ----
        Either a segmentation, a list of regions, or a volume needs to
        referenced together with the corresponding source image(s) or series.
        Derived classes determine which of the above will be allowed.

        """  # noqa
        super().__init__(
            tracking_identifier=tracking_identifier,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations
        )
        group_item = self[0]
        were_references_provided = [
            referenced_regions is not None,
            referenced_volume_surface is not None,
            referenced_segment is not None,
        ]
        if sum(were_references_provided) == 0:
            raise ValueError(
                'One of the following arguments must be provided: '
                '"referenced_regions", "referenced_volume_surface", or '
                '"referenced_segment".'
            )
        elif sum(were_references_provided) > 1:
            raise ValueError(
                'Only one of the following arguments should be provided: '
                '"referenced_regions", "referenced_volume_surface", or '
                '"referenced_segment".'
            )
        if referenced_regions is not None:
            if len(referenced_regions) == 0:
                raise ValueError(
                    'Argument "referenced_region" must have non-zero length.'
                )
            for region in referenced_regions:
                if not isinstance(region, (ImageRegion, ImageRegion3D)):
                    raise TypeError(
                        'Items of argument "referenced_regions" must have type '
                        'ImageRegion or ImageRegion3D.'
                    )
                group_item.ContentSequence.append(region)
        elif referenced_volume_surface is not None:
            if not isinstance(referenced_volume_surface,
                              VolumeSurface):
                raise TypeError(
                    'Items of argument "referenced_volume_surface" must have '
                    'type VolumeSurface.'
                )
            group_item.ContentSequence.append(referenced_volume_surface)
        elif referenced_segment is not None:
            if not isinstance(
                    referenced_segment,
                    (ReferencedSegment, ReferencedSegmentationFrame)
                ):
                raise TypeError(
                    'Argument "referenced_segment" must have type '
                    'ReferencedSegment or '
                    'ReferencedSegmentationFrame.'
                )
            group_item.ContentSequence.append(referenced_segment)


class PlanarROIMeasurementsAndQualitativeEvaluations(
        _ROIMeasurementsAndQualitativeEvaluations):

    """TID 1410 Planar ROI Measurements and Qualitative Evaluations"""

    def __init__(
            self,
            tracking_identifier: TrackingIdentifier,
            referenced_region: Optional[
                Union[ImageRegion, ImageRegion3D]
            ] = None,
            referenced_segment: Optional[
                Union[ReferencedSegment, ReferencedSegmentationFrame]
            ] = None,
            referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
            time_point_context: Optional[TimePointContext] = None,
            finding_type: Optional[Union[CodedConcept, Code]] = None,
            method: Optional[Union[CodedConcept, Code]] = None,
            algorithm_id: Optional[AlgorithmIdentification] = None,
            finding_sites: Optional[Sequence[FindingSite]] = None,
            session: Optional[str] = None,
            measurements: Sequence[Measurement] = None,
            qualitative_evaluations: Optional[Union[CodedConcept, Code]] = None
        ):
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_region: Union[highdicom.sr.content.ImageRegion, highdicom.sr.content.ImageRegion3D], optional
            region of interest in source image
        referenced_segment: highdicom.sr.content.ReferencedSegmentationFrame, optional
            segmentation for region of interest in source image
        referenced_real_world_value_map: highdicom.sr.content.RealWorldValueMap, optional
            referenced real world value map for region of interest
        time_point_context: highdicom.sr.templates.TimePointContext, optional
            description of the time point context
        finding_type: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            type of object that was measured, e.g., organ or tumor
        method: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded measurement method
            (see CID 6147 "Response Criteria" for options)
        algorithm_id: highdicom.sr.templates.AlgorithmIdentification, optional
            identification of algorithm used for making measurements
        finding_sites: Sequence[highdicom.sr.content.FindingSite], optional
            Coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        session: str, optional
            description of the session
        measurements: Sequence[highdicom.sr.templates.Measurement], optional
            measurements for a region of interest
        qualitative_evaluations: Sequence[highdicom.sr.value_types.CodeContentItem], optional
            coded name-value (question-answer) pairs that describe the
            measurements in qualitative terms for a region of interest

        Note
        ----
        Either a segmentation or a region needs to referenced
        together with the corresponding source image from which the
        segmentation or region was obtained.

        """  # noqa
        were_references_provided = [
            referenced_region is not None,
            referenced_segment is not None,
        ]
        if sum(were_references_provided) == 0:
            raise ValueError(
                'One of the following arguments must be provided: '
                '"referenced_region", "referenced_segment".'
            )
        elif sum(were_references_provided) > 1:
            raise ValueError(
                'Only one of the following arguments should be provided: '
                '"referenced_region", "referenced_segment".'
            )
        referenced_regions: Optional[
            Union[Sequence[ImageRegion], Sequence[ImageRegion3D]]
        ] = None
        if referenced_region is not None:
            referenced_regions = [referenced_region]
        super().__init__(
            tracking_identifier=tracking_identifier,
            referenced_regions=referenced_regions,
            referenced_segment=referenced_segment,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations
        )


class VolumetricROIMeasurementsAndQualitativeEvaluations(
        _ROIMeasurementsAndQualitativeEvaluations):

    """TID 1411 Volumetric ROI Measurements and Qualitative Evaluations"""

    def __init__(
            self,
            tracking_identifier: TrackingIdentifier,
            referenced_regions: Optional[
                Union[ImageRegion, ImageRegion3D]
            ] = None,
            referenced_volume_surface: Optional[VolumeSurface] = None,
            referenced_segment: Optional[
                Union[ReferencedSegment, ReferencedSegmentationFrame]
            ] = None,
            referenced_real_world_value_map: Optional[RealWorldValueMap] = None,
            time_point_context: Optional[TimePointContext] = None,
            finding_type: Optional[Union[CodedConcept, Code]] = None,
            method: Optional[Union[CodedConcept, Code]] = None,
            algorithm_id: Optional[AlgorithmIdentification] = None,
            finding_sites: Optional[Sequence[FindingSite]] = None,
            session: Optional[str] = None,
            measurements: Sequence[Measurement] = None,
            qualitative_evaluations: Optional[Union[CodedConcept, Code]] = None
        ):
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_regions: Union[Sequence[highdicom.sr.content.ImageRegion], Sequence[highdicom.sr.content.ImageRegion3D]], optional
            regions of interest in source image(s)
        referenced_volume_surface: highdicom.sr.content.VolumeSurface, optional
            volume of interest in source image(s)
        referenced_segment: highdicom.sr.content.ReferencedSegment, optional
            segmentation for region of interest in source image
        referenced_real_world_value_map: highdicom.sr.content.RealWorldValueMap, optional
            referenced real world value map for region of interest
        time_point_context: highdicom.sr.templates.TimePointContext, optional
            description of the time point context
        finding_type: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            type of object that was measured, e.g., organ or tumor
        method: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded measurement method
            (see CID 6147 "Response Criteria" for options)
        algorithm_id: highdicom.sr.templates.AlgorithmIdentification, optional
            identification of algorithm used for making measurements
        finding_sites: Sequence[highdicom.sr.content.FindingSite], optional
            Coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        session: str, optional
            description of the session
        measurements: Sequence[highdicom.sr.templates.Measurement], optional
            measurements for a volume of interest
        qualitative_evaluations: Sequence[highdicom.sr.value_types.CodeContentItem], optional
            coded name-value (question-answer) pairs that describe the
            measurements in qualitative terms for a volume of interest

        Note
        ----
        Either a segmentation, a list of regions or volume needs to referenced
        together with the corresponding source image(s) or series.

        """  # noqa
        super().__init__(
            measurements=measurements,
            tracking_identifier=tracking_identifier,
            referenced_regions=referenced_regions,
            referenced_volume_surface=referenced_volume_surface,
            referenced_segment=referenced_segment,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            qualitative_evaluations=qualitative_evaluations
        )


class MeasurementsDerivedFromMultipleROIMeasurements(Template):

    """TID 1420 Measurements Derived From Multiple ROI Measurements"""

    def __init__(
            self,
            derivation: CodedConcept,
            measurement_groups: Union[
                Sequence[PlanarROIMeasurementsAndQualitativeEvaluations],
                Sequence[VolumetricROIMeasurementsAndQualitativeEvaluations]
            ],
            measurement_properties: Optional[MeasurementProperties] = None
        ):
        """
        Parameters
        ----------
        derivation: Sequence[highdicom.sr.coding.CodedConcept]
            methods for derivation of measurements from multiple ROIs
            measurements
            (see CID 7465 "Measurements Derived From Multiple ROI Measurements"
            for otions)
        measurement_groups: Union[Sequence[highdicom.sr.templates.PlanarROIMeasurementsAndQualitativeEvaluations], Sequence[highdicom.sr.templates.VolumetricROIMeasurementsAndQualitativeEvaluations]]
            one or more groups of either planar or volumetric ROI measurements
            and qualitative evaluations
        measurement_properties: highdicom.sr.templates.MeasurementProperties, optional
            measurement properties, including evaluations of its normality
            and/or significance, its relationship to a reference population,
            and an indication of its selection from a set of measurements

        """  # noqa
        value_item = NumContentItem(
            name=derivation,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        value_item.ContentSequence = ContentSequence()
        for group in measurement_groups:
            allowed_group_types = (
                PlanarROIMeasurementsAndQualitativeEvaluations,
                VolumetricROIMeasurementsAndQualitativeEvaluations,
            )
            if not isinstance(group, allowed_group_types):
                raise TypeError(
                    'Items of argument "measurement_groups" must have type '
                    'PlanarROIMeasurementsAndQualitativeEvaluations or '
                    'VolumetricROIMeasurementsAndQualitativeEvaluations.'
                )
            group[0].RelationshipType = 'R-INFERRED FROM'
            value_item.ContentSequence.extend(group)
        if measurement_properties is not None:
            if not isinstance(measurement_properties, MeasurementProperties):
                raise TypeError(
                    'Argument "measurement_properties" must have '
                    'type MeasurementProperties.'
                )
            value_item.ContentSequence.extend(measurement_properties)
        # TODO: how to do R-INFERRED FROM relationship?
        self.append(value_item)


class MeasurementReport(Template):

    """TID 1500 Measurement Report"""

    def __init__(
            self,
            observation_context: ObservationContext,
            procedure_reported: Union[CodedConcept, Code],
            imaging_measurements: Optional[
                Sequence[
                    Union[
                        PlanarROIMeasurementsAndQualitativeEvaluations,
                        VolumetricROIMeasurementsAndQualitativeEvaluations,
                        MeasurementsAndQualitativeEvaluations,
                    ]
                ]
            ] = None,
            derived_imaging_measurements: Optional[
                Sequence[MeasurementsDerivedFromMultipleROIMeasurements]
            ] = None,
            title: Optional[Union[CodedConcept, Code]] = None,
            language_of_content_item_and_descendants: Optional[
                LanguageOfContentItemAndDescendants
            ] = None
        ):
        """
        Parameters
        ----------
        observation_context: highdicom.sr.templates.ObservationContext
            description of the observation context
        procedure_reported: Union[Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], Sequence[Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]]]
            one or more coded description(s) of the procedure
            (see CID 100 "Quantitative Diagnostic Imaging Procedures" for
            options)
        imaging_measurements: Sequence[Union[highdicom.sr.templates.PlanarROIMeasurementsAndQualitativeEvaluations, highdicom.sr.templates.VolumetricROIMeasurementsAndQualitativeEvaluations, highdicom.sr.templates.MeasurementsAndQualitativeEvaluations]], optional
            measurements and qualitative evaluations of images or regions
            within images
        derived_imaging_measurements: Sequence[highdicom.sr.templates.MeasurementsDerivedFromMultipleROIMeasurements], optional
            measurements derived from other measurements of images or regions
            within images
            qualitative evaluations of images
        title: highdicom.sr.coding.CodedConcept, optional
            title of the report
            (see CID 7021 "Measurement Report Document Titles" for options)
        language_of_content_item_and_descendants: highdicom.sr.templates.LanguageOfContentItemAndDescendants, optional
            specification of the language of report content items
            (defaults to English)

        Note
        ----
        Only one of `imaging_measurements` or `derived_imaging_measurements`
        shall be specified.

        """ # noqa
        super().__init__()
        if title is None:
            title = codes.cid7021.ImagingMeasurementReport
        if not isinstance(title, (CodedConcept, Code, )):
            raise TypeError(
                'Argument "title" must have type CodedConcept or Code.'
            )
        item = ContainerContentItem(
            name=title,
            template_id='1500'
        )
        item.ContentSequence = ContentSequence()
        if language_of_content_item_and_descendants is None:
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(
            language_of_content_item_and_descendants
        )
        item.ContentSequence.extend(observation_context)
        if isinstance(procedure_reported, (CodedConcept, Code, )):
            procedure_reported = [procedure_reported]
        for procedure in procedure_reported:
            procedure_item = CodeContentItem(
                name=CodedConcept(
                    value='121058',
                    meaning='Procedure reported',
                    scheme_designator='DCM',
                ),
                value=procedure,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            item.ContentSequence.append(procedure_item)
        image_library_item = ImageLibrary()
        item.ContentSequence.extend(image_library_item)

        num_arguments_provided = sum([
            imaging_measurements is not None,
            derived_imaging_measurements is not None
        ])
        if num_arguments_provided > 1:
            raise ValueError(
                'Only one of the following arguments can be provided: '
                '"imaging_measurements", "derived_imaging_measurement".'
            )
        if imaging_measurements is not None:
            measurement_types = (
                PlanarROIMeasurementsAndQualitativeEvaluations,
                VolumetricROIMeasurementsAndQualitativeEvaluations,
                MeasurementsAndQualitativeEvaluations,
            )
            container_item = ContainerContentItem(
                name=CodedConcept(
                    value='126010',
                    meaning='Imaging Measurements',
                    scheme_designator='DCM'
                ),
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            container_item.ContentSequence = []
            for measurements in imaging_measurements:
                if not isinstance(measurements, measurement_types):
                    raise TypeError(
                        'Measurements must have one of the following types: '
                        '"{}"'.format(
                            '", "'.join(
                                [
                                    t.__name__
                                    for t in measurement_types
                                ]
                            )
                        )
                    )
                container_item.ContentSequence.extend(measurements)
        elif derived_imaging_measurements is not None:
            derived_measurement_types = (
                MeasurementsDerivedFromMultipleROIMeasurements,
            )
            container_item = ContainerContentItem(
                name=CodedConcept(
                    value='126011',
                    meaning='Derived Imaging Measurements',
                    scheme_designator='DCM'
                ),
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            container_item.ContentSequence = []
            for measurements in derived_imaging_measurements:
                if not isinstance(measurements, derived_measurement_types):
                    raise TypeError(
                        'Measurements must have one of the following types: '
                        '"{}"'.format(
                            '", "'.join(
                                [
                                    t.__name__
                                    for t in derived_measurement_types
                                ]
                            )
                        )
                    )
                container_item.ContentSequence.extend(measurements)
        else:
            raise TypeError(
                'One of the following arguments must be provided: '
                '"imaging_measurements", "derived_imaging_measurements".'
            )
        item.ContentSequence.append(container_item)
        self.append(item)


class ImageLibrary(Template):

    """TID 1600 Image Library"""

    def __init__(self) -> None:
        """
        Note
        ----
        Image Library Entry Descriptors are not included.

        """
        # We didn't implement this on purpose.
        super().__init__()
        library_item = ContainerContentItem(
            name=CodedConcept(
                value='111028',
                meaning='Image Library',
                scheme_designator='DCM'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        self.append(library_item)
