"""DICOM structured reporting templates."""
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.coding import Code
from pydicom.codedict import codes

from highdicom.sr.coding import CodedConcept
from highdicom.sr.value_types import (
    CodeContentItem,
    ContainerContentItem,
    ContentSequence,
    ImageContentItem,
    NumContentItem,
    RelationshipTypes,
    TextContentItem,
    UIDRefContentItem,
)

from highdicom.sr.content import (
    FindingSite,
    LongitudinalTemporalOffsetFromEvent,
    ImageRegion,
    ImageRegion3D,
    VolumeSurface,
    RealWorldValueMap,
    ReferencedSegmentation,
    ReferencedSegmentationFrame,
)


DEFAULT_LANGUAGE = CodedConcept(
    value='en-US',
    scheme_designator='RFC5646',
    meaning='English (United States)'
)


class Template(ContentSequence):

    """Abstract base class for a DICOM SR template."""

    def __init__(self, iterable=None):
        super(Template, self).__init__(iterable)


class Measurement(Template):

    """TID 300 Measurement"""

    def __init__(self, name, tracking_identifier=None, value=None, unit=None,
                 qualifier=None, algorithm_id=None, derivation=None,
                 finding_sites=None, method=None, properties=None,
                 referenced_image=None, referenced_real_world_value_map=None):
        """
        Parameters
        ----------
        name: pydicom.sr.coding.CodedConcept
            name of the measurement
            (see CID 7469 "Generic Intensity and Size Measurements" and
            CID 7468 "Texture Measurements" for options)
        tracking_identifier: Union[pydicom.sr.templates.TrackingIdentifier, None], optional
            identifier for tracking measurements
        value: Union[int, float, None], optional
            numeric measurement value
        unit: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            unit of the numeric measurement value
            (see CID 7181 "Abstract Multi-dimensional Image Model Component
            Units" for options)
        qualifier: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            qualification of numeric measurement value or as an alternative
            qualitative description
        algorithm_id: Union[pydicom.sr.templates.AlgorithmIdentification, None], optional
            identification of algorithm
        derivation: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            how the value was computed
            (see CID 7464 "General Region of Interest Measurement Modifiers"
            for options)
        finding_sites: Union[List[pydicom.sr.template.FindingSite], None], optional
            coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken
        method: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            measurement method
            (see CID 6147 "Response Criteria" for options)
        properties: Union[pydicom.sr.templates.MeasurementProperties, None], optional
            measurement properties, including evaluations of its normality
            and/or significance, its relationship to a reference population,
            and an indication of its selection from a set of measurements
        referenced_image: Union[pydicom.sr.content_item.SourceImageForProcessing, None], optional
            referenced image from which quantitative features were extracted
        referenced_real_world_value_map: Union[pydicom.sr.content_items.RealWorldValueMap, None], optional
            referenced real world value map for region of interest

        """  # noqa
        super(Measurement, self).__init__()
        value_item = NumContentItem(
            name=name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            relationship_type=RelationshipTypes.CONTAINS
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
                relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
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
                relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
            )
            value_item.ContentSequence.append(derivation_item)
        if finding_sites is not None:
            if not isinstance(finding_sites, (list, tuple, set)):
                raise TypeError(
                    'Argument "finding_sites" must have type list.'

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
        if referenced_image is not None:
            if not isinstance(referenced_image, SourceImageForProcessing):
                raise TypeError(
                    'Argument "referenced_image" must have type '
                    'SourceImageForProcessing.'
                )
            value_item.ContentSequence.append(referenced_image)
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


class MeasurementProperties(Template):

    """TID 310 Measurement Properties"""

    def __init__(self, normality=None, level_of_significance=None,
                 selection_status=None, measurement_statistical_properties=None,
                 normal_range_properties=None,
                 upper_measurement_uncertainty=None,
                 lower_measurement_uncertainty=None):
        """
        Parameters
        ----------
        normality: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            the extend to which the measurement is considered normal or abnormal
            (see CID 222 "Normality Codes" for options)
        level_of_significance: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            the extend to which the measurement is considered normal or abnormal
            (see CID 220 "Level of Significance" for options)
        selection_status: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            how the measurement value was selected or computed from a set of
            available values
            (see CID 224 "Selection Method" for options)
        measurement_statistical_properties: Union[pydicom.sr.templates.MeasurementStatisticalProperties, None], optional
            statistical properties of a reference population for a measurement
            and/or the position of a measurement in such a reference population
        normal_range_properties: Union[pydicom.sr.templates.NormalRangeProperties, None], optional
            statistical properties of a reference population for a measurement
            and/or the position of a measurement in such a reference population
        upper_measurement_uncertainty: Union[int, float, None], optional
            upper range of measurment uncertainty
        lower_measurement_uncertainty: Union[int, float, None], optional
            lower range of measurment uncertainty

        """  # noqa
        super(MeasurementProperties, self).__init__()
        if normality is not None:
            normality_item = CodeContentItem(
                name=CodedConcept(
                    value='121402',
                    meaning='Normality',
                    scheme_designator='DCM'
                ),
                value=normality,
                relationship_type=RelationshipTypes.HAS_PROPERTIES
            )
            self.append(normality_item)
        if measurement_statistical_properties is not None:
            if not isinstance(measurment_statistical_properties,
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
                relationship_type=RelationshipTypes.HAS_PROPERTIES
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
                relationship_type=RelationshipTypes.HAS_PROPERTIES
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
                relationship_type=RelationshipTypes.HAS_PROPERTIES
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
                relationship_type=RelationshipTypes.HAS_PROPERTIES
            )
            self.append(lower_measurement_uncertainty_item)


class MeasurementStatisticalProperties(Template):

    """TID 311 Measurement Statistical Properties"""

    def __init__(self, values, description=None, authority=None):
        """
        Parameters
        ----------
        values: List[pydicom.sr.value_types.NumContentItem]
            reference values of the population of measurements, e.g., its
            mean or standard deviation
            (see CID 226 "Population Statistical Descriptors" and
            CID 227 227 "Sample Statistical Descriptors" for options)
        description: Union[str, None], optional
            description of the reference population of measurements
        authority: Union[str, None], optional
            authority for a description of the reference population of
            measurements

        """
        super(MeasurementStatisticalProperties, self).__init__()
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
                value=authority,
                relationship_type=RelationshipTypes.HAS_PROPERTIES
            )
            self.append(authority_item)
        if authority is not None:
            authority_item = TextContentItem(
                name=CodedConcept(
                    value='121406',
                    meaning='Population Authority',
                    scheme_designator='DCM'
                ),
                value=authority,
                relationship_type=RelationshipTypes.HAS_PROPERTIES
            )
            self.append(authority_item)


class NormalRangeProperties(Template):

    """TID 312 Normal Range Properties"""

    def __init__(self, values, description=None, authority=None):
        """
        Parameters
        ----------
        values: List[pydicom.sr.value_types.NumContentItem]
            reference values of the normal range, e.g., its upper and lower
            bound
            (see CID 223 "Normal Range Values" for options)
        description: Union[str, None], optional
            description of the normal range
        authority: Union[str, None], optional
            authority for the description of the normal range

        """
        super(NormalRangeProperties, self).__init__()
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
                    value='121407',
                    meaning='Normal Range Description',
                    scheme_designator='DCM'
                ),
                value=authority,
                relationship_type=RelationshipTypes.HAS_PROPERTIES
            )
            self.append(authority_item)
        if authority is not None:
            authority_item = TextContentItem(
                name=CodedConcept(
                    value='121408',
                    meaning='Normal Range Authority',
                    scheme_designator='DCM'
                ),
                value=authority,
                relationship_type=RelationshipTypes.HAS_PROPERTIES
            )
            self.append(authority_item)


class ObservationContext(Template):

    """TID 1001 Observation Context"""

    def __init__(self, observer_person_context=None,
                 observer_device_context=None,
                 subject_context=None):
        """
        Parameters
        ----------
        observer_person_context: [pydicom.sr.templates.ObserverContext, None], optional
            description of the person that reported the observation
        observer_device_context: Union[pydicom.sr.templates.ObserverContext, None], optional
            description of the device that was involved in reporting the
            observation
        subject_context: Union[pydicom.sr.templates.SubjectContext, None], optional
            description of the imaging subject in case it is not the patient
            for which the report is generated (e.g., a pathology specimen in
            a whole-slide microscopy image, a fetus in an ultrasound image, or
            a pacemaker device in a chest X-ray image)

        """
        super(ObservationContext, self).__init__()
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


class ObserverContext(Template):

    """TID 1002 Observer Context"""

    def __init__(self, observer_type, observer_identifying_attributes):
        """
        Parameters
        ----------
        observer_type: pydicom.sr.coding.CodedConcept
            type of observer
            (see CID 270 "Observer Type" for options)
        observer_identifying_attributes: Union[pydicom.sr.templates.PersonObserverIdentifyingAttributes, pydicom.sr.templates.DeviceObserverIdentifyingAttributes]
            observer identifying attributes

        """  # noqa
        super(ObserverContext, self).__init__()
        observer_type_item = CodeContentItem(
            name=CodedConcept(
                value='121005',
                meaning='Observer Type',
                scheme_designator='DCM',
            ),
            value=observer_type,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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


class PersonObserverIdentifyingAttributes(Template):

    """TID 1003 Person Observer Identifying Attributes"""

    def __init__(self, name, login_name=None, organization_name=None,
                 role_in_organization=None, role_in_procedure=None):
        """
        Parameters
        ----------
        name: str
            name of the person
        login_name: str
            login name of the person
        organization_name: Union[str, None], optional
            name of the person's organization
        role_in_organization: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            role of the person within the organization
        role_in_procedure: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            role of the person in the reported procedure

        """  # noqa
        super(PersonObserverIdentifyingAttributes, self).__init__()
        name_item = TextContentItem(
            name=CodedConcept(
                value='121008',
                meaning='Person Observer Name',
                scheme_designator='DCM',
            ),
            value=name,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(role_in_procedure_item)


class DeviceObserverIdentifyingAttributes(Template):

    """TID 1004 Device Observer Identifying Attributes"""

    def __init__(self, uid,
                 name=None,
                 manufacturer_name=None,
                 model_name=None,
                 serial_number=None,
                 physical_location=None,
                 role_in_procedure=None):
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
        role_in_procedure: Union[str, None], optional
            role of the device in the reported procedure

        """
        super(DeviceObserverIdentifyingAttributes, self).__init__()
        device_observer_item = UIDRefContentItem(
            name=CodedConcept(
                value='121012',
                meaning='Device Observer UID',
                scheme_designator='DCM',
            ),
            value=uid,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(serial_number_item)
        if physical_location is not None:
            physical_location_item = TextContentItem(
                name=CodedConcept(
                    value='121017',
                    meaning='Device Observer Physical Location During Observation',
                    scheme_designator='DCM',
                ),
                value=physical_location,
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(physical_location_item)
        if role_in_procedure is not None:
            role_in_procedure_item = CodeContentItem(
                name=CodedConcept(
                    value='113876',
                    meaning='Device Role in Procedure',
                    scheme_designator='DCM',
                ),
                value=role_in_procedure,
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(role_in_procedure_item)


class SubjectContext(Template):

    """TID 1006 Subject Context"""

    def __init__(self, subject_class, subject_class_specific_context=None):
        """
        Parameters
        ----------
        subject_class: pydicom.sr.coding.CodedConcept
            type of subject if the subject of the report is not the patient
            (see CID 271 "Observation Subject Class" for options)
        subject_class_specific_context: Union[pydicom.sr.templates.SubjectContextFetus, pydicom.sr.templates.SubjectContextSpecimen, pydicom.sr.templates.SubjectContextDevice, None], optional
            additional context information specific to `subject_class`

        """  # noqa
        super(SubjectContext, self).__init__()
        subject_class_item = CodeContentItem(
            name=CodedConcept(
                value='121024',
                meaning='Subject Class',
                scheme_designator='DCM'
            ),
            value=subject_class,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
        )
        self.append(subject_class_item)
        self.extend(subject_class_specific_context)


class SubjectContextFetus(Template):

    """TID 1008 Subject Context Fetus"""

    def __init__(self, subject_id):
        """
        Parameters
        ----------
        subject_id: str
            identifier of the fetus for longitudinal tracking

        """
        super(SubjectContextFetus, self).__init__()
        subject_id_item = TextContentItem(
            name=CodedConcept(
                value='121030',
                meaning='Subject ID',
                scheme_designator='DCM'
            ),
            value=subject_id,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
        )
        self.append(subject_id_item)


class SubjectContextSpecimen(Template):

    """TID 1009 Subject Context Specimen"""

    def __init__(self, uid, identifier=None,
                 container_identifier=None, specimen_type=None):
        """
        Parameters
        ----------
        uid: str
            unique identifier of the observed specimen
        identifier: str
            identifier of the observed specimen (may have limited scope,
            e.g., only relevant with respect to the corresponding container)
        container_identifier: str
            identifier of the container holding the speciment (e.g., a glass
            slide)
        specimen_type: pydicom.sr.coding.CodedConcept
            type of the specimen
            (see CID 8103 "Anatomic Pathology Specimen Types" for options)

        """
        super(SubjectContextSpecimen, self).__init__()
        specimen_uid_item = UIDRefContentItem(
            name=CodedConcept(
                value='121039',
                meaning='Specimen UID',
                scheme_designator='DCM'
            ),
            value=uid,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(container_identifier_item)
        if specimen_type is not None:
            specimen_type_item = CodeContentItem(
                name=CodedConcept(
                    value='R-00254',
                    meaning='Specimen Type',
                    scheme_designator='DCM'
                ),
                value=specimen_type,
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(specimen_type_item)


class SubjectContextDevice(Template):

    """TID 1010 Subject Context Device"""

    def __init__(self, name, uid=None, manufacturer_name=None,
                 model_name=None, serial_number=None, physical_location=None):
        """
        Parameters
        ----------
        name: str
            name of the observed device
        uid: Union[str, None], optional
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
        super(SubjectContextDevice, self).__init__()
        device_name_item = TextContentItem(
            name=CodedConcept(
                value='121193',
                meaning='Device Subject Name',
                scheme_designator='DCM'
            ),
            value=device_name,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
        )
        self.append(device_name_item)
        if device_uid is not None:
            device_uid_item = UIDRefContentItem(
                name=CodedConcept(
                    value='121198',
                    meaning='Device Subject UID',
                    scheme_designator='DCM'
                ),
                value=device_uid,
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(serial_number_item)
        if physical_location is not None:
            physical_location_item = TextContentItem(
                name=CodedConcept(
                    value='121197',
                    meaning='Device Subject Physical Location During Observation',
                    scheme_designator='DCM',
                ),
                value=physical_location,
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(physical_location_item)


class LanguageOfContentItemAndDescendants(Template):

    """TID 1204 Language of Content Item and Descendants"""

    def __init__(self, language):
        """
        Parameters
        ----------
        language: pydicom.sr.coding.CodedConcept
            language used for content items included in report

        """
        super(LanguageOfContentItemAndDescendants, self).__init__()
        language_item = CodeContentItem(
            name=CodedConcept(
                value='121049',
                meaning='Language of Content Item and Descendants',
                scheme_designator='DCM',
            ),
            value=language,
            relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
        )
        self.append(language_item)


class _MeasurementsAndQualitatitiveEvaluations(Template):

    """Abstract base class for ROI Measurements and Qualitative Evaluation
    templates."""

    def __init__(self, tracking_identifier,
                 referenced_image=None, referenced_real_world_value_map=None,
                 time_point_context=None, finding_type=None, session=None,
                 measurements=None, qualitative_evaluations=None):
        """
        Parameters
        ----------
        tracking_identifier: pydicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_image: Union[pydicom.sr.content_item.SourceImageForProcessing, None], optional
            referenced image from which quantitative features were extracted
        referenced_real_world_value_map: Union[pydicom.sr.content_items.RealWorldValueMap, None], optional
            referenced real world value map for region of interest
        time_point_context: Union[pydicom.sr.templates.TimePointContext, None], optional
            description of the time point context
        finding_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            type of object that was measured, e.g., organ or tumor
        session: Union[str, None], optional
            description of the session
        measurements: Union[pydicom.sr.templates.ROIMeasurements, None], optional
            measurements
        qualitative_evaluations: Union[List[pydicom.sr.coding.CodeContentItem], None], optional
            coded name-value (question-answer) pairs that describe the
            measurements in qualitative terms

        """  # noqa
        super(_MeasurementsAndQualitatitiveEvaluations, self).__init__()
        group_item = ContainerContentItem(
            name=CodedConcept(
                value='125007',
                meaning='Measurement Group',
                scheme_designator='DCM'
            )
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.CONTAINS
            )
            group_item.ContentSequence.append(finding_type_item)
        if time_point_context is not None:
            if not isinstance(time_point_context, TimePointContext):
                raise TypeError(
                    'Argument "time_point_context" must have type '
                    'TimePointContext.'
                )
            group_item.ContentSequence.append(time_point_context)
        if referenced_image is not None:
            if not isinstance(referenced_image, SourceImageForProcessing):
                raise TypeError(
                    'Argument "referenced_image" must have type '
                    'SourceImageForProcessing.'
                )
            group_item.ContentSequence.append(referenced_image)
        if referenced_real_world_value_map is not None:
            if not isinstance(referenced_real_world_value_map,
                              RealWorldValueMap):
                raise TypeError(
                    'Argument "referenced_real_world_value_map" must have type '
                    'RealWorldValueMap.'
                )
            group_item.ContentSequence.append(referenced_real_world_value_map)
        if measurements is not None:
            group_item.ContentSequence.extend(measurements)
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
        _MeasurementsAndQualitatitiveEvaluations):

    """Abstract base class for ROI Measurements and Qualitative Evaluation
    templates."""

    def __init__(self, tracking_identifier,
                 referenced_regions=None, referenced_volume_surface=None,
                 referenced_segmentation=None,
                 referenced_real_world_value_map=None,
                 time_point_context=None, finding_type=None, session=None,
                 measurements=None, qualitative_evaluations=None):
        """
        Parameters
        ----------
        tracking_identifier: pydicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_regions: Union[List[pydicom.sr.content_items.ImageRegion], List[pydicom.sr.content_items.ImageRegion3D], None], optional
            regions of interest in source image(s)
        referenced_segmentation: Union[pydicom.sr.content_items.ReferencedSegmentation, pydicom.sr.content_items.ReferencedSegmentationFrame, None], optional
            segmentation for region of interest in source image
        referenced_real_world_value_map: Union[pydicom.sr.content_items.RealWorldValueMap, None], optional
            referenced real world value map for region of interest
        time_point_context: Union[pydicom.sr.templates.TimePointContext, None], optional
            description of the time point context
        finding_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            type of object that was measured, e.g., organ or tumor
        session: Union[str, None], optional
            description of the session
        measurements: Union[pydicom.sr.templates.ROIMeasurements, None], optional
            measurements
        qualitative_evaluations: Union[List[pydicom.sr.coding.CodeContentItem], None], optional
            coded name-value (question-answer) pairs that describe the
            measurements in qualitative terms

        Note
        ----
        Either a segmentation, a list of regions, or a volume needs to
        referenced together with the corresponding source image(s) or series.
        Derived classes determine which of the above will be allowed.

        """  # noqa
        super(_ROIMeasurementsAndQualitativeEvaluations, self).__init__(
            tracking_identifier=tracking_identifier,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations
        )
        group_item = self[0]
        were_references_provided = [
            referenced_regions is not None,
            referenced_volume_surface is not None,
            referenced_segmentation is not None,
        ]
        if sum(were_references_provided) == 0:
            raise ValueError(
                'One of the following arguments must be provided: '
                '"referenced_regions", "referenced_volume_surface", or '
                '"referenced_segmentation".'
            )
        elif sum(were_references_provided) > 1:
            raise ValueError(
                'Only one of the following arguments should be provided: '
                '"referenced_regions", "referenced_volume_surface", or '
                '"referenced_segmentation".'
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
                    'Items of argument "referenced_volume_surface" must have type '
                    'VolumeSurface.'
                )
            group_item.ContentSequence.append(referenced_volume_surface)
        elif referenced_segmentation is not None:
            if not isinstance(referenced_segmentation,
                              (ReferencedSegmentation,
                               ReferencedSegmentationFrame)
                             ):
                raise TypeError(
                    'Argument "referenced_segmentation" must have type '
                    'ReferencedSegmentation or '
                    'ReferencedSegmentationFrame.'
                )
            group_item.ContentSequence.append(referenced_segmentation)


class PlanarROIMeasurementsAndQualitativeEvaluations(
        _ROIMeasurementsAndQualitativeEvaluations):

    """TID 1410 Planar ROI Measurements and Qualitative Evaluations"""

    def __init__(self, tracking_identifier,
                 referenced_region=None,
                 referenced_segmentation=None,
                 referenced_real_world_value_map=None,
                 time_point_context=None, finding_type=None, session=None,
                 measurements=None, qualitative_evaluations=None):
        """
        Parameters
        ----------
        tracking_identifier: pydicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_region: Union[pydicom.sr.content_items.ImageRegion, pydicom.sr.content_items.ImageRegion3D, None], optional
            region of interest in source image
        referenced_segmentation: Union[pydicom.sr.content_items.ReferencedSegmentationFrame, None], optional
            segmentation for region of interest in source image
        referenced_real_world_value_map: Union[pydicom.sr.content_items.RealWorldValueMap, None], optional
            referenced real world value map for region of interest
        time_point_context: Union[pydicom.sr.templates.TimePointContext, None], optional
            description of the time point context
        finding_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            type of object that was measured, e.g., organ or tumor
        session: Union[str, None], optional
            description of the session
        measurements: Union[pydicom.sr.templates.ROIMeasurements, None], optional
            measurements for a region of interest
        qualitative_evaluations: Union[List[pydicom.sr.coding.CodeContentItem], None], optional
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
            referenced_segmentation is not None,
        ]
        if sum(were_references_provided) == 0:
            raise ValueError(
                'One of the following arguments must be provided: '
                '"referenced_region", "referenced_segmentation".'
            )
        elif sum(were_references_provided) > 1:
            raise ValueError(
                'Only one of the following arguments should be provided: '
                '"referenced_region", "referenced_segmentation".'
            )
        super(PlanarROIMeasurementsAndQualitativeEvaluations, self).__init__(
            tracking_identifier=tracking_identifier,
            referenced_regions=[referenced_region],
            referenced_segmentation=referenced_segmentation,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations
        )


class VolumetricROIMeasurementsAndQualitativeEvaluations(
        _ROIMeasurementsAndQualitativeEvaluations):

    """TID 1411 Volumetric ROI Measurements and Qualitative Evaluations"""

    def __init__(self, tracking_identifier,
                 referenced_regions=None,
                 referenced_volume_surface=None,
                 referenced_segmentation=None,
                 referenced_real_world_value_map=None,
                 time_point_context=None, finding_type=None, session=None,
                 measurements=None, qualitative_evaluations=None):
        """
        Parameters
        ----------
        tracking_identifier: pydicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_regions: Union[List[pydicom.sr.content_items.ImageRegion], List[pydicom.sr.content_items.ImageRegion3D], None], optional
            regions of interest in source image(s)
        referenced_volume_surface: Union[pydicom.sr.content_items.VolumeSurface, None], optional
            volume of interest in source image(s)
        referenced_segmentation: Union[pydicom.sr.content_items.ReferencedSegmentation, None], optional
            segmentation for region of interest in source image
        referenced_real_world_value_map: Union[pydicom.sr.content_items.RealWorldValueMap, None], optional
            referenced real world value map for region of interest
        time_point_context: Union[pydicom.sr.templates.TimePointContext, None], optional
            description of the time point context
        finding_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            type of object that was measured, e.g., organ or tumor
        session: Union[str, None], optional
            description of the session
        measurements: Union[pydicom.sr.templates.ROIMeasurements, None], optional
            measurements for a volume of interest
        qualitative_evaluations: Union[List[pydicom.sr.coding.CodeContentItem], None], optional
            coded name-value (question-answer) pairs that describe the
            measurements in qualitative terms for a volume of interest

        Note
        ----
        Either a segmentation, a list of regions or volume needs to referenced
        together with the corresponding source image(s) or series.

        """  # noqa
        super(VolumetricROIMeasurementsAndQualitativeEvaluations, self).__init__(
            measurements=measurements,
            tracking_identifier=tracking_identifier,
            referenced_regions=referenced_regions,
            referenced_volume_surface=referenced_volume_surface,
            referenced_segmentation=referenced_segmentation,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            session=session,
            qualitative_evaluations=qualitative_evaluations
        )


class MeasurementsDerivedFromMultipleROIMeasurements(Template):

    """TID 1420 Measurements Derived From Multiple ROI Measurements"""

    def __init__(self, derivation, measurement_groups,
                 measurement_properties=None):
        """
        Parameters
        ----------
        derivation: List[pydicom.sr.coding.CodedConcept]
            methods for derivation of measurements from multiple ROIs
            measurements
            (see CID 7465 "Measurements Derived From Multiple ROI Measurements"
            for otions)
        measurement_groups: Union[List[pydicom.sr.templates.PlanarROIMeasurementsAndQualitativeEvaluations], List[pydicom.sr.templates.VolumetricROIMeasurementsAndQualitativeEvaluations]]
            one or more groups of either planar or volumetric ROI measurements
            and qualitative evaluations
        measurement_properties: Union[pydicom.sr.templates.MeasurementProperties, None], optional
            measurement properties, including evaluations of its normality
            and/or significance, its relationship to a reference population,
            and an indication of its selection from a set of measurements

        """  # noqa
        value_item = NumContentItem(
            name=derivation
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


class MeasurementAndQualitativeEvaluationGroup(
        _MeasurementsAndQualitatitiveEvaluations):

    """TID 1501 Measurement and Qualitative Evaluation Group"""

    def __init__(self, tracking_identifier, referenced_image=None,
                 referenced_real_world_value_map=None, time_point_context=None,
                 finding_type=None, session=None, measurements=None,
                 qualitative_evaluations=None):
        """
        Parameters
        ----------
        tracking_identifier: pydicom.sr.templates.TrackingIdentifier
            identifier for tracking measurements
        referenced_image: Union[pydicom.sr.content_item.SourceImageForProcessing, None], optional
            referenced image from which quantitative features were extracted
        referenced_real_world_value_map: Union[pydicom.sr.content_items.RealWorldValueMap, None], optional
            referenced real world value map for region of interest
        time_point_context: Union[pydicom.sr.templates.TimePointContext, None], optional
            description of the time point context
        finding_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            type of object that was measured, e.g., organ or tumor
        session: Union[str, None], optional
            description of the session
        measurements: Union[pydicom.sr.templates.ROIMeasurements, None], optional
            measurements
        qualitative_evaluations: Union[List[pydicom.sr.coding.CodeContentItem], None], optional
            coded name-value (question-answer) pairs that describe the
            measurements in qualitative terms

        """  # noqa
        super(MeasurementAndQualitativeEvaluationGroup, self).__init__(
            measurements=measurements,
            tracking_identifier=tracking_identifier,
            referenced_image=referenced_image,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            session=session,
            qualitative_evaluations=qualitative_evaluations
        )


class ROIMeasurements(Template):

    """TID 1419 ROI Measurements"""

    def __init__(self, measurements, method=None, finding_sites=None):
        """
        Parameters
        ----------
        measurements: List[pydicom.sr.templates.Measurement]
            individual measurements
        method: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded measurement method
            (see CID 6147 "Response Criteria" for options)
        finding_sites: Union[List[pydicom.sr.templates.FindingSite], None], optional
            coded description of one or more anatomic locations corresonding
            to the image region from which measurement was taken

        """  # noqa
        super(ROIMeasurements, self).__init__()
        if method is not None:
            method_item = CodeContentItem(
                name=CodedConcept(
                    value='370129005',
                    meaning='Measurement Method',
                    scheme_designator='SCT'
                ),
                value=method,
                relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
            )
            self.append(method_item)
        if finding_sites is not None:
            for site in finding_sites:
                if not isinstance(site, FindingSite):
                    raise TypeError(
                        'Items of argument "finding_sites" must have '
                        'type FindingSite.'
                    )
                self.append(site)
        if len(measurements) == 0:
            raise ValueError('Argument "measurements" must not be empty.')
        for m in measurements:
            if not isinstance(m, Measurement):
                raise TypeError(
                    'Items of argument "measurements" must have type '
                    'Measurement.'
                )
            self.extend(m)


class MeasurementReport(Template):

    """TID 1500 Measurement Report"""

    def __init__(self,
                 observation_context,
                 procedure_reported,
                 imaging_measurements=None,
                 derived_imaging_measurements=None,
                 qualitative_evaluations=None,
                 title=codes.cid7021.ImagingMeasurementReport,
                 language_of_content_item_and_descendants=None
                ):
        """
        Parameters
        ----------
        observation_context: pydicom.sr.templates.ObservationContext
            description of the observation context
        procedure_reported: Union[Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], List[Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]]]
            one or more coded description(s) of the procedure
            (see CID 100 Quantitative Diagnostic Imaging Procedures for options)
        imaging_measurements: Union[List[pydicom.sr.templates._MeasurementsAndQualitatitiveEvaluations], None], optional
            imaging measurements and qualitative evaluations, i.e.
            the items contained by one or more instances of
            ``pydicom.sr.templates.PlanarROIMeasurementsAndQualitativeEvaluations``,
            ``pydicom.sr.templates.VolumetricROIMeasurementsAndQualitativeEvaluations``,
            or ``pydicom.sr.templates.MeasurementAndQualitativeEvaluationGroup``
        derived_imaging_measurements: Union[List[Union[pydicom.sr.templates.MeasurementsDerivedFromMultipleROIMeasurements]], None], optional
            measurements derived from multiple imaging measurements
        qualitative_evaluations: Union[List[Union[pydicom.sr.coding.CodeContentItem, pydicom.sr.value_types.TextContentItem]], None], optional
            qualitative evaluations of images
        title: pydicom.sr.coding.CodedConcept, optional
            title of the report
            (see CID 7021 "Measurement Report Document Titles" for options)
        language_of_content_item_and_descendants: Union[pydicom.sr.templates.LanguageOfContentItemAndDescendants, None], optional
            specification of the language of report content items
            (defaults to English)

        Note
        ----
        Only one of `imaging_measurements`, `derived_imaging_measurement`, or
        `qualitative_evaluations` should be specified.

        """ # noqa
        super(MeasurementReport, self).__init__()
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
                relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
            )
            item.ContentSequence.append(procedure_item)
        image_library_item = ImageLibrary()
        item.ContentSequence.extend(image_library_item)

        num_arguments_provided = sum([
            imaging_measurements is not None,
            derived_imaging_measurements is not None,
            qualitative_evaluations is not None
        ])
        if num_arguments_provided > 1:
            raise ValueError(
                'Only one of the following arguments can be provided: '
                '"imaging_measurements", "derived_imaging_measurement", '
                '"qualitative_evaluations".'
            )
        if imaging_measurements is not None:
            container_item = ContainerContentItem(
                name=CodedConcept(
                    value='126010',
                    meaning='Imaging Measurements',
                    scheme_designator='DCM'
                ),
                relationship_type=RelationshipTypes.CONTAINS
            )
            container_item.ContentSequence = imaging_measurements
        elif derived_imaging_measurements is not None:
            container_item = ContainerContentItem(
                name=CodedConcept(
                    value='126011',
                    meaning='Derived Imaging Measurements',
                    scheme_designator='DCM'
                ),
                relationship_type=RelationshipTypes.CONTAINS
            )
            container_item.ContentSequence = derived_imaging_measurements
        elif qualitative_evaluations is not None:
            container_item = ContainerContentItem(
                name=CodedConcept(
                    value='C0034375',
                    meaning='Qualitative Evaluations',
                    scheme_designator='UMLS'
                ),
                relationship_type=RelationshipTypes.CONTAINS
            )
            container_item.ContentSequence = qualitative_evaluations
        item.ContentSequence.append(container_item)
        self.append(item)


class TimePointContext(Template):

    """TID 1502 Time Point Context"""

    def __init__(self, time_point, time_point_type=None, time_point_order=None,
                 subject_time_point_identifier=None,
                 protocol_time_point_identifier=None):
        """
        Parameters
        ----------
        time_point: str
            actual value represention of the time point
        time_point_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded type of time point, e.g., "Baseline" or "Posttreatment"
            (see CID 646 "Time Point Types" for options)
        time_point_order: Union[int, None], optional
            number indicating the order of a time point relative to other
            time points in a time series
        subject_time_point_identifier: Union[str, None], optional
           identifier of a specific time point in a time series, which is
           unique within an appropriate local context and specific to a
           particular subject (patient)
        protocol_time_point_identifier: Union[str, None], optional
           identifier of a specific time point in a time series, which is
           unique within an appropriate local context and specific to a
           particular protocol using the same value for different subjects
        temporal_offset_from_event: Union[pydicom.sr.template.LongitudinalTemporalOffsetFromEvent, None], optional
            offset in time from a particular event of significance, e.g., the
            baseline of an imaging study or enrollment into a clincal trial
        temporal_event_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
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
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(subject_time_point_identifier_item)
        if protocol_time_point_identifier is not None:
            protocol_time_point_identifier_item = NumContentItem(
                name=CodedConcept(
                    value='126071',
                    meaning='Protocol Time Point Identifier',
                    scheme_designator='DCM'
                ),
                value=protocol_time_point_identifier,
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
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


class ImageLibrary(Template):

    """TID 1600 Image Library"""

    def __init__(self):
        """
        Note
        ----
        Image Library Entry Descriptors are not included.

        """
        super(ImageLibrary, self).__init__()
        library_item = ContainerContentItem(
            name=CodedConcept(
                value='111028',
                meaning='Image Library',
                scheme_designator='DCM'
            ),
            relationship_type=RelationshipTypes.CONTAINS
        )
        self.append(library_item)


class AlgorithmIdentification(Template):

    """TID 4019 Algorithm Identification"""

    def __init__(self, name, version, parameters=None):
        """
        Parameters
        ----------
        name: str
            name of the algorithm
        version: str
            version of the algorithm
        parameters: Union[List[str], None], optional
            parameters of the algorithm

        """
        super(AlgorithmIdentification, self).__init__()
        name_item = TextContentItem(
            name=CodedConcept(
                value='111001',
                meaning='Algorithm Name',
                scheme_designator='DCM'
            ),
            value=name,
            relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
        )
        self.append(name_item)
        version_item = TextContentItem(
            name=CodedConcept(
                value='111003',
                meaning='Algorithm Version',
                scheme_designator='DCM'
            ),
            value=version,
            relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
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
                    relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
                )
                self.append(parameter_item)


class TrackingIdentifier(Template):

    """TID 4108 Tracking Identifier"""

    def __init__(self, uid, identifier=None):
        """
        Parameters
        ----------
        uid: Union[pydicom.uid.UID, str]
            globally unique identifier
        identifier: Union[str, None], optional
            human readable identifier

        """
        super(TrackingIdentifier, self).__init__()
        if identifier is not None:
            tracking_identifier_item = TextContentItem(
                name=CodedConcept(
                    value='112039',
                    meaning='Tracking Identifier',
                    scheme_designator='DCM'
                ),
                value=identifier,
                relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
            )
            self.append(tracking_identifier_item)
        tracking_uid_item = UIDRefContentItem(
            name=CodedConcept(
                value='112040',
                meaning='Tracking Unique Identifier',
                scheme_designator='DCM'
            ),
            value=uid,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
        )
        self.append(tracking_uid_item)
