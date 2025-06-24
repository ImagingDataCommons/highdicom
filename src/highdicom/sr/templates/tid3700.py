from datetime import datetime
from typing import Optional, Sequence, Union
from highdicom.sr.value_types import (
    Code,
    CodeContentItem,
    CodedConcept,
    ContentSequence,
    DateTimeContentItem,
    NumContentItem,
    TextContentItem,
    CompositeContentItem,
    ContainerContentItem,
    RelationshipTypeValues,
    TcoordContentItem,
    UIDRefContentItem,
    WaveformContentItem
)
from highdicom.sr.templates.common import (
    AgeUnit,
    AlgorithmIdentification,
    LanguageOfContentItemAndDescendants,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
    PressureUnit,
    Template
)
from pydicom.valuerep import DT
from pydicom.sr.codedict import codes

from highdicom.sr.templates.tid3802 import CardiovascularPatientHistory

BPM = Code(
    value='bpm',
    scheme_designator='UCUM',
    meaning='beats per minute',
    scheme_version=None
)
BEATS = Code(
    value='beats',
    scheme_designator='UCUM',
    meaning='beats',
    scheme_version=None
)
KG = Code(
    value='kg',
    scheme_designator='UCUM',
    meaning='kilogram',
    scheme_version=None
)


class ECGWaveFormInformation(Template):
    """:dcm:`TID 3708 <part16/sect_TID_3708.html>`
    ECG Waveform Information
    """

    def __init__(
        self,
        procedure_datetime: Union[str, datetime, DT],
        source_of_measurement: Optional[WaveformContentItem] = None,
        lead_system: Optional[Union[CodedConcept, Code]] = None,
        acquisition_device_type: Optional[str] = None,
        equipment_identification: Optional[str] = None,
        person_observer_identifying_attributes: Optional[
            PersonObserverIdentifyingAttributes] = None,
        room_identification: Optional[str] = None,
        ecg_control_numeric_variables: Optional[Sequence[float]] = None,
        ecg_control_text_variables: Optional[Sequence[str]] = None,
        algorithm_identification: Optional[AlgorithmIdentification] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.LN.CurrentProcedureDescriptions,
            template_id='3708',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        procedure_datetime_item = DateTimeContentItem(
            name=codes.DCM.ProcedureDatetime,
            value=procedure_datetime,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(procedure_datetime_item)
        if source_of_measurement is not None:
            if not isinstance(source_of_measurement, WaveformContentItem):
                raise TypeError(
                    'Argument "source_of_measurement" must have ' +
                    'type WaveformContentItem.'
                )
            content.append(source_of_measurement)
        if lead_system is not None:
            if not isinstance(lead_system, (CodedConcept, Code)):
                raise TypeError(
                    'Argument "lead_system" must have type ' +
                    'Code or CodedConcept.'
                )
            lead_system_item = CodeContentItem(
                name=CodedConcept(
                    value='10:11345',
                    meaning='Lead System',
                    scheme_designator='MDC'
                ),
                value=lead_system,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(lead_system_item)
        if acquisition_device_type is not None:
            if not isinstance(acquisition_device_type, str):
                raise TypeError(
                    'Argument "acquisition_device_type" must have type str.'
                )
            acquisition_device_type_item = TextContentItem(
                name=codes.DCM.AcquisitionDeviceType,
                value=acquisition_device_type,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(acquisition_device_type_item)
        if equipment_identification is not None:
            if not isinstance(equipment_identification, str):
                raise TypeError(
                    'Argument "equipment_identification" must have type str.'
                )
            equipment_identification_item = TextContentItem(
                name=codes.DCM.EquipmentIdentification,
                value=equipment_identification,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(equipment_identification_item)
        if person_observer_identifying_attributes is not None:
            if not isinstance(person_observer_identifying_attributes,
                              PersonObserverIdentifyingAttributes):
                raise TypeError(
                    'Argument "person_observer_identifying_attributes" ' +
                    'must have type PersonObserverIdentifyingAttributes.'
                )
            content.extend(person_observer_identifying_attributes)
        if room_identification is not None:
            if not isinstance(room_identification, (list, tuple, set)):
                raise TypeError(
                    'Argument "room_identifications" must be a sequence.'
                )
            room_identification_item = TextContentItem(
                name=codes.DCM.RoomIdentification,
                value=room_identification,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(room_identification_item)
        if ecg_control_numeric_variables is not None:
            if not isinstance(ecg_control_numeric_variables, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "ecg_control_numeric_variables" must be ' +
                    'a sequence.'
                )
            for ecg_control_numeric_variable in ecg_control_numeric_variables:
                if not isinstance(ecg_control_numeric_variable, float):
                    raise TypeError(
                        'Items of argument "ecg_control_numeric_variable" ' +
                        'must have type float.'
                    )
                ecg_control_numeric_variable_item = NumContentItem(
                    name=CodedConcept(
                        value='3690',
                        meaning='ECG Control Numeric Variable',
                        scheme_designator='CID'
                    ),
                    value=ecg_control_numeric_variable,
                    unit=codes.UCUM.NoUnits,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(ecg_control_numeric_variable_item)
        if ecg_control_text_variables is not None:
            if not isinstance(ecg_control_text_variables, (list, tuple, set)):
                raise TypeError(
                    'Argument "ecg_control_text_variables" must be a sequence.'
                )
            for ecg_control_text_variable in ecg_control_text_variables:
                if not isinstance(ecg_control_text_variable, str):
                    raise TypeError(
                        'Items of argument "ecg_control_numeric_variable" ' +
                        'must have type str.'
                    )
                ecg_control_text_variable_item = TextContentItem(
                    name=CodedConcept(
                        value='3691',
                        meaning='ECG Control Text Variable',
                        scheme_designator='CID'
                    ),
                    value=ecg_control_text_variable,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(ecg_control_text_variable_item)
        if algorithm_identification is not None:
            if not isinstance(algorithm_identification,
                              AlgorithmIdentification):
                raise TypeError(
                    'Argument "algorithm_identification" must have type ' +
                    'AlgorithmIdentification.'
                )
            content.extend(algorithm_identification)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class ECGMeasurementSource(Template):
    """:dcm:`TID 3715 <part16/sect_TID_3715.html>`
    ECG Measurement Source
    """

    def __init__(
        self,
        beat_number: Optional[str],
        measurement_method: Optional[Union[Code, CodedConcept]],
        source_of_measurement: Optional[TcoordContentItem]
    ) -> None:
        item = ContainerContentItem(
            name=CodedConcept(
                value='3715',
                meaning='ECG Measurement Source',
                scheme_designator='TID'
            ),
            template_id='3715',
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        content = ContentSequence()
        if not isinstance(beat_number, str):
            raise TypeError(
                'Argument "beat_number" must have type str.'
            )
        # TODO: Beat number str can be up to three numeric characters
        beat_number_item = TextContentItem(
            name=codes.DCM.BeatNumber,
            value=beat_number,
            # TODO: The Relationship Type is not defined in the DICOM template
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(beat_number_item)
        if source_of_measurement is not None:
            if not isinstance(source_of_measurement, TcoordContentItem):
                raise TypeError(
                    'Argument "source_of_measurement" must have type ' +
                    'TcoordContentItem.'
                )
            content.append(source_of_measurement)
        if measurement_method is not None:
            if not isinstance(measurement_method, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "measurement_method" must be a ' +
                    'Code or CodedConcept.'
                )
            measurement_method_item = CodeContentItem(
                name=codes.SCT.MeasurementMethod,
                value=measurement_method,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            content.append(measurement_method_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class QTcIntervalGlobal(NumContentItem):
    """:mdc:`MDC 2:15876`
    QTc interval global
    """

    def __init__(
        self,
        value: float,
        algorithm_name: Optional[Union[Code, CodedConcept]] = None
    ) -> None:
        super().__init__(
            name=codes.MDC.QtcIntervalGlobal,
            value=value,
            unit=codes.UCUM.Millisecond,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if algorithm_name is not None:
            if not isinstance(algorithm_name, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "algorithm_name" must have type ' +
                    'Code or CodedConcept.'
                )
            algorithm_name_item = CodeContentItem(
                name=codes.DCM.AlgorithmName,
                value=algorithm_name,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(algorithm_name_item)
        if len(content) > 0:
            self.ContentSequence = content


class NumberOfEctopicBeats(NumContentItem):
    """:dcm:`DCM 122707`
    Number of Ectopic Beats
    """

    def __init__(
        self,
        value: float,
        associated_morphologies: Optional[Sequence[Union[Code,
                                                         CodedConcept]]] = None
    ) -> None:
        super().__init__(
            name=codes.DCM.NumberOfEctopicBeats,
            value=value,
            unit=BEATS,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if associated_morphologies is not None:
            if not isinstance(associated_morphologies, (list, tuple, set)):
                raise TypeError(
                    'Argument "associated_morphologies" must be a sequence.'
                )
            for associated_morphology in associated_morphologies:
                if not isinstance(associated_morphology, (Code, CodedConcept)):
                    raise TypeError(
                        'Items of argument "associated_morphology" must ' +
                        'have type Code or CodedConcept.'
                    )
                associated_morphology_item = CodeContentItem(
                    name=codes.SCT.AssociatedMorphology,
                    value=associated_morphology,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(associated_morphology_item)
        if len(content) > 0:
            self.ContentSequence = content


class ECGGlobalMeasurements(Template):
    """:dcm:`TID 3713 <part16/sect_TID_3713.html>`
    ECG Global Measurements
    """

    def __init__(
        self,
        ventricular_heart_rate: float,
        qt_interval_global: float,
        pr_interval_global: float,
        qrs_duration_global: float,
        rr_interval_global: float,
        ecg_measurement_source: Optional[ECGMeasurementSource] = None,
        atrial_heart_rate: Optional[float] = None,
        qtc_interval_global: Optional[QTcIntervalGlobal] = None,
        ecg_global_waveform_durations: Optional[Sequence[float]] = None,
        ecg_axis_measurements: Optional[Sequence[float]] = None,
        count_of_all_beats: Optional[float] = None,
        number_of_ectopic_beats: Optional[NumberOfEctopicBeats] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.DCM.ECGGlobalMeasurements,
            template_id='3713',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if not isinstance(ventricular_heart_rate, float):
            raise TypeError(
                'Argument "ventricular_heart_rate" must have type float.'
            )
        ventricular_heart_rate_item = NumContentItem(
            name=CodedConcept(
                value='2:16016',
                meaning='Ventricular Heart Rate',
                scheme_designator='MDC'
            ),
            value=ventricular_heart_rate,
            unit=BPM,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(ventricular_heart_rate_item)
        if not isinstance(qt_interval_global, float):
            raise TypeError(
                'Argument "qt_interval_global" must have type float.'
            )
        qt_interval_global_item = NumContentItem(
            name=codes.MDC.QTIntervalGlobal,
            value=qt_interval_global,
            unit=codes.UCUM.Millisecond,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(qt_interval_global_item)
        if not isinstance(pr_interval_global, float):
            raise TypeError(
                'Argument "qt_interval_global" must have type float.'
            )
        pr_interval_global_item = NumContentItem(
            name=codes.MDC.PRIntervalGlobal,
            value=pr_interval_global,
            unit=codes.UCUM.Millisecond,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(pr_interval_global_item)
        if not isinstance(qrs_duration_global, float):
            raise TypeError(
                'Argument "qrs_duration_global" must have type float.'
            )
        qrs_duration_global_item = NumContentItem(
            name=codes.MDC.QRSDurationGlobal,
            value=qrs_duration_global,
            unit=codes.UCUM.Millisecond,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(qrs_duration_global_item)
        if not isinstance(rr_interval_global, float):
            raise TypeError(
                'Argument "rr_interval_global" must have type float.'
            )
        rr_interval_global_item = NumContentItem(
            name=codes.MDC.RRIntervalGlobal,
            value=rr_interval_global,
            unit=codes.UCUM.Millisecond,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(rr_interval_global_item)
        if ecg_measurement_source is not None:
            if not isinstance(ecg_measurement_source, ECGMeasurementSource):
                raise TypeError(
                    'Argument "ecg_measurement_source" must have type ' +
                    'ECGMeasurementSource.'
                )
            content.extend(ecg_measurement_source)
        if atrial_heart_rate is not None:
            if not isinstance(atrial_heart_rate, float):
                raise TypeError(
                    'Argument "atrial_heart_rate" must have type float.'
                )
            atrial_heart_rate_item = NumContentItem(
                name=CodedConcept(
                    value='2:16020',
                    meaning='Atrial Heart Rate',
                    scheme_designator='MDC'
                ),
                value=atrial_heart_rate,
                unit=BPM,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(atrial_heart_rate_item)
        if qtc_interval_global is not None:
            if not isinstance(qtc_interval_global, QTcIntervalGlobal):
                raise TypeError(
                    'Argument "qtc_interval_global" must have type ' +
                    'QTcIntervalGlobal.'
                )
            content.append(qtc_interval_global)
        if ecg_global_waveform_durations is not None:
            if not isinstance(ecg_global_waveform_durations, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "ecg_global_waveform_durations" must ' +
                    'be a sequence.'
                )
            for ecg_global_waveform_duration in ecg_global_waveform_durations:
                if not isinstance(ecg_global_waveform_duration, float):
                    raise TypeError(
                        'Items of argument "ecg_global_waveform_duration" ' +
                        'must have type float.'
                    )
                ecg_global_waveform_duration_item = NumContentItem(
                    name=CodedConcept(
                        value='3687',
                        meaning='ECG Global Waveform Duration',
                        scheme_designator='CID'
                    ),
                    value=ecg_global_waveform_duration,
                    unit=codes.UCUM.Millisecond,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(ecg_global_waveform_duration_item)
        if ecg_axis_measurements is not None:
            if not isinstance(ecg_axis_measurements, (list, tuple, set)):
                raise TypeError(
                    'Argument "ecg_axis_measurements" must be a sequence.'
                )
            for ecg_axis_measurement in ecg_axis_measurements:
                if not isinstance(ecg_axis_measurement, float):
                    raise TypeError(
                        'Items of argument "ecg_axis_measurement" must ' +
                        'have type float.'
                    )
                ecg_axis_measurement_item = NumContentItem(
                    name=CodedConcept(
                        value='3229',
                        meaning='ECG Axis Measurement',
                        scheme_designator='CID'
                    ),
                    value=ecg_axis_measurement,
                    unit=codes.UCUM.Degree,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(ecg_axis_measurement_item)
        if count_of_all_beats is not None:
            if not isinstance(count_of_all_beats, float):
                raise TypeError(
                    'Argument "count_of_all_beats" must have type float.'
                )
            count_of_all_beats_item = NumContentItem(
                name=CodedConcept(
                    value='2:16032',
                    meaning='Count of all beats',
                    scheme_designator='MDC'
                ),
                value=count_of_all_beats,
                unit=BEATS,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(count_of_all_beats_item)
        if number_of_ectopic_beats is not None:
            if not isinstance(number_of_ectopic_beats, NumberOfEctopicBeats):
                raise TypeError(
                    'Argument "number_of_ectopic_beats" must have type NumberOfEctopicBeats.'
                )
            content.append(number_of_ectopic_beats)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class ECGLeadMeasurements(Template):
    """:dcm:`TID 3714 <part16/sect_TID_3714.html>`
    ECG Lead Measurements
    """

    def __init__(
        self,
        lead_id: Union[Code, CodedConcept],
        ecg_measurement_source: Optional[ECGMeasurementSource] = None,
        electrophysiology_waveform_durations: Optional[Sequence[float]] = None,
        electrophysiology_waveform_voltages: Optional[Sequence[float]] = None,
        st_segment_finding: Optional[Union[Code, CodedConcept]] = None,
        findings: Optional[Sequence[Union[Code, CodedConcept]]] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.DCM.ECGLeadMeasurements,
            template_id='3714',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if not isinstance(lead_id, (Code, CodedConcept)):
            raise TypeError(
                'Argument "lead_id" must be a Code or CodedConcept.'
            )
        lead_id_item = CodeContentItem(
            name=codes.DCM.LeadID,
            value=lead_id,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        content.append(lead_id_item)
        if ecg_measurement_source is not None:
            if not isinstance(ecg_measurement_source, ECGMeasurementSource):
                raise TypeError(
                    'Argument "ecg_measurement_source" must be a ' +
                    'ECGMeasurementSource.'
                )
            content.extend(ecg_measurement_source)
        if electrophysiology_waveform_durations is not None:
            if not isinstance(electrophysiology_waveform_durations, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "electrophysiology_waveform_durations" ' +
                    'must be a sequence.'
                )
            for electrophysiology_waveform_duration in \
                    electrophysiology_waveform_durations:
                if not isinstance(electrophysiology_waveform_duration, float):
                    raise TypeError(
                        'Items of argument ' +
                        '"electrophysiology_waveform_duration" ' +
                        'must have type float.'
                    )
                electrophysiology_waveform_duration_item = NumContentItem(
                    name=CodedConcept(
                        value='3687',
                        meaning='Electrophysiology Waveform Duration',
                        scheme_designator='CID'
                    ),
                    value=electrophysiology_waveform_duration,
                    unit=codes.UCUM.Millisecond,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(electrophysiology_waveform_duration_item)
        if electrophysiology_waveform_voltages is not None:
            if not isinstance(electrophysiology_waveform_voltages, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "electrophysiology_waveform_voltages" ' +
                    'must be a sequence.'
                )
            for electrophysiology_waveform_voltage in \
                    electrophysiology_waveform_voltages:
                if not isinstance(electrophysiology_waveform_voltage, float):
                    raise TypeError(
                        'Items of argument' +
                        ' "electrophysiology_waveform_voltage" ' +
                        'must have type float.'
                    )
                electrophysiology_waveform_voltage_item = NumContentItem(
                    name=CodedConcept(
                        value='3687',
                        meaning='Electrophysiology Waveform Duration',
                        scheme_designator='CID'
                    ),
                    value=electrophysiology_waveform_voltage,
                    unit=codes.UCUM.Millivolt,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(electrophysiology_waveform_voltage_item)
        if st_segment_finding is not None:
            if not isinstance(st_segment_finding, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "st_segment_finding" must be a ' +
                    'Code or CodedConcept.'
                )
            st_segment_finding_item = CodeContentItem(
                name=CodedConcept(
                    value='365416000',
                    meaning='ST Segment Finding',
                    scheme_designator='SCT'
                ),
                value=st_segment_finding,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(st_segment_finding_item)
        if findings is not None:
            if not isinstance(findings, (list, tuple, set)):
                raise TypeError(
                    'Argument "findings" must be a sequence.'
                )
            for finding in findings:
                if not isinstance(finding, (Code, CodedConcept)):
                    raise TypeError(
                        'Items of argument "finding" must be a ' +
                        'Code or CodedConcept.'
                    )
                finding_item = CodeContentItem(
                    name=codes.DCM.Finding,
                    value=finding,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(finding_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class QuantitativeAnalysis(Template):
    """:dcm:`EV 122144`
    Quantitative Analysis
    """

    def __init__(
        self,
        ecg_global_measurements: Optional[ECGGlobalMeasurements] = None,
        ecg_lead_measurements: Optional[Sequence[ECGLeadMeasurements]] = None,
    ) -> None:
        item = ContainerContentItem(
            name=CodedConcept(
                value='122144',
                meaning='Quantitative Analysis',
                scheme_designator='DCM'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if ecg_global_measurements is not None:
            if not isinstance(ecg_global_measurements, ECGGlobalMeasurements):
                raise TypeError(
                    'Argument "ecg_global_measurements" must be a ' +
                    'ECGGlobalMeasurements.'
                )
            content.extend(ecg_global_measurements)
        if ecg_lead_measurements is not None:
            if not isinstance(ecg_lead_measurements, (list, tuple, set)):
                raise TypeError(
                    'Argument "ecg_lead_measurements" must be a sequence.'
                )
            for ecg_lead_measurement in ecg_lead_measurements:
                if not isinstance(ecg_lead_measurement, ECGLeadMeasurements):
                    raise TypeError(
                        'Items of argument "ecg_lead_measurement" must ' +
                        'have type ECGLeadMeasurements.'
                    )
                content.extend(ecg_lead_measurement)
        item.ContentSequence = content
        super().__init__([item])


class IndicationsForProcedure(Template):
    """:ln:`EV 18785-6`
    Indications for Procedure
    """

    def __init__(
        self,
        findings: Optional[Sequence[Union[Code, CodedConcept]]] = None,
        finding_text: Optional[str] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.LN.IndicationsForProcedure,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if findings is not None:
            if not isinstance(findings, (list, tuple, set)):
                raise TypeError(
                    'Argument "findings" must be a sequence.'
                )
            for finding in findings:
                if not isinstance(finding, (CodedConcept, Code, )):
                    raise TypeError(
                        'Argument "findings" must have type ' +
                        'Code or CodedConcept.'
                    )
                finding_item = CodeContentItem(
                    name=codes.DCM.Finding,
                    value=finding,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(finding_item)
        if finding_text is not None:
            if not isinstance(finding_text, str):
                raise TypeError(
                    'Argument "finding_text" must have type str.'
                )
            finding_text_item = TextContentItem(
                name=codes.DCM.Finding,
                value=finding_text,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(finding_text_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class PatientCharacteristicsForECG(Template):
    """:dcm:`TID 3704 <part16/sect_TID_3704.html>`
    Patient Characteristics for ECG
    """

    def __init__(
        self,
        subject_age: AgeUnit,
        subject_sex: str,
        patient_height: Optional[float] = None,
        patient_weight: Optional[float] = None,
        systolic_blood_pressure: Optional[PressureUnit] = None,
        diastolic_blood_pressure: Optional[PressureUnit] = None,
        patient_state: Optional[Union[Code, CodedConcept]] = None,
        pacemaker_in_situ: Optional[Union[Code, CodedConcept]] = None,
        icd_in_situ: Optional[Union[Code, CodedConcept]] = None
    ):
        item = ContainerContentItem(
            name=codes.DCM.PatientCharacteristics,
            template_id='3704',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if not isinstance(subject_age, AgeUnit):
            raise TypeError(
                'Argument "subject_age" must have type AgeUnit.'
            )
        subject_age.add_items(content)
        if not isinstance(subject_sex, str):
            raise TypeError(
                'Argument "subject_sex" must have type str.'
            )
        subject_sex_item = TextContentItem(
            name=codes.DCM.SubjectSex,
            value=subject_sex,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(subject_sex_item)
        if patient_height is not None:
            if not isinstance(patient_height, float):
                raise TypeError(
                    'Argument "patient_height" must have type float.'
                )
            patient_height_item = NumContentItem(
                name=CodedConcept(
                    value='8302-2',
                    meaning='Patient Height',
                    scheme_designator='LN'
                ),
                value=patient_height,
                unit=codes.UCUM.Centimeter,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(patient_height_item)
        if patient_weight is not None:
            if not isinstance(patient_weight, float):
                raise TypeError(
                    'Argument "patient_weight" must have type float.'
                )
            patient_weight_item = NumContentItem(
                name=CodedConcept(
                    value='29463-7',
                    meaning='Patient Weight',
                    scheme_designator='LN'
                ),
                value=patient_weight,
                unit=KG,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(patient_weight_item)
        if systolic_blood_pressure is not None:
            if not isinstance(systolic_blood_pressure, PressureUnit):
                raise TypeError(
                    'Argument "systolic_blood_pressure" must have ' +
                    'type PressureUnit.'
                )
            systolic_blood_pressure.name = codes.SCT.SystolicBloodPressure
            systolic_blood_pressure.add_items(content)
        if diastolic_blood_pressure is not None:
            if not isinstance(diastolic_blood_pressure, PressureUnit):
                raise TypeError(
                    'Argument "diastolic_blood_pressure" must have ' +
                    'type PressureUnit.'
                )
            diastolic_blood_pressure.name = codes.SCT.DiastolicBloodPressure
            diastolic_blood_pressure.add_items(content)
        if patient_state is not None:
            if not isinstance(patient_state, (CodedConcept, Code)):
                raise TypeError(
                    'Argument "patient_state" must have type ' +
                    'Code or CodedConcept.'
                )
            patient_state_item = CodeContentItem(
                name=codes.DCM.PatientState,
                value=patient_state,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(patient_state_item)
        if pacemaker_in_situ is not None:
            if not isinstance(pacemaker_in_situ, (CodedConcept, Code)):
                raise TypeError(
                    'Argument "pacemaker_in_situ" must have type ' +
                    'Code or CodedConcept.'
                )
            pacemaker_in_situ_item = CodeContentItem(
                name=CodedConcept(
                    value='441509002',
                    meaning='Pacemaker in situ',
                    scheme_designator='SCT'
                ),
                value=pacemaker_in_situ,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(pacemaker_in_situ_item)
        if icd_in_situ is not None:
            if not isinstance(icd_in_situ, (CodedConcept, Code)):
                raise TypeError(
                    'Argument "icd_in_situ" must have type ' +
                    'Code or CodedConcept.'
                )
            icd_in_situ_item = CodeContentItem(
                name=CodedConcept(
                    value='443325000',
                    meaning='ICD in situ',
                    scheme_designator='SCT'
                ),
                value=icd_in_situ,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(icd_in_situ_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class PriorECGStudy(Template):
    """:dcm:`TID 3702 <part16/sect_TID_3702.html>`
    Prior ECG Study
    """

    def __init__(
        self,
        comparison_with_prior_study_done: Union[CodedConcept, Code],
        procedure_datetime: Optional[Union[str, datetime, DT]] = None,
        procedure_study_instance_uid: Optional[UIDRefContentItem] = None,
        prior_report_for_current_patient: Optional[CompositeContentItem] = None,
        source_of_measurement: Optional[WaveformContentItem] = None
    ):
        item = ContainerContentItem(
            name=codes.LN.PriorProcedureDescriptions,
            template_id='3702',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        comparison_with_prior_study_done_item = CodeContentItem(
            name=CodedConcept(
                value='122140',
                meaning='Comparison with Prior Study Done',
                scheme_designator='DCM'
            ),
            value=comparison_with_prior_study_done,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content.append(comparison_with_prior_study_done_item)
        if procedure_datetime is not None:
            if not isinstance(procedure_datetime, (str, datetime, DT)):
                raise TypeError(
                    'Argument "procedure_datetime" must have type ' +
                    'str, datetime.datetime or DT.'
                )
            procedure_datetime_item = DateTimeContentItem(
                name=codes.DCM.ProcedureDatetime,
                value=procedure_datetime,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(procedure_datetime_item)
        if procedure_study_instance_uid is not None:
            if not isinstance(procedure_study_instance_uid,
                              UIDRefContentItem):
                raise TypeError(
                    'Argument "procedure_stdy_instance_uid" must have ' +
                    'type UIDRefContentItem.'
                )
            content.append(procedure_study_instance_uid)
        if prior_report_for_current_patient is not None:
            if not isinstance(prior_report_for_current_patient,
                              CompositeContentItem):
                raise TypeError(
                    'Argument "prior_report_for_current_patient" must ' +
                    'have type CompositeContentItem.'
                )
            content.append(prior_report_for_current_patient)
        if source_of_measurement is not None:
            if not isinstance(source_of_measurement, WaveformContentItem):
                raise TypeError(
                    'Argument "source_of_measurement" must have type ' +
                    'WaveformContentItem.'
                )
            content.append(source_of_measurement)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class ECGFinding(Template):
    """:sct:`EV 271921002`
    ECG Finding
    """

    def __init__(
        self,
        value: Union[CodedConcept, Code],
        equivalent_meaning_of_value: Optional[str] = None,
        ecg_findings: Optional[Sequence["ECGFinding"]] = None
    ) -> None:
        item = CodeContentItem(
            name=CodedConcept(
                value='271921002',
                meaning='ECG Finding',
                scheme_designator='SCT'
            ),
            value=value,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if equivalent_meaning_of_value is not None:
            if not isinstance(equivalent_meaning_of_value, str):
                raise TypeError(
                    'Argument "equivalent_meaning_of_value" must have type str.'
                )
            equivalent_meaning_of_value_item = TextContentItem(
                name=codes.DCM.EquivalentMeaningOfValue,
                value=equivalent_meaning_of_value,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            content.append(equivalent_meaning_of_value_item)
        if ecg_findings is not None:
            if not isinstance(ecg_findings, (list, tuple, set)):
                raise TypeError(
                    'Argument "ecg_findings" must be a sequence.'
                )
            for ecg_finding in ecg_findings:
                if not isinstance(ecg_finding, ECGFinding):
                    raise TypeError(
                        'Items of argument "ecg_finding" must have ' +
                        'type ECGFinding.'
                    )
                content.extend(ecg_finding)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class ECGQualitativeAnalysis(Template):
    """:dcm:`TID 3717 <part16/sect_TID_3717.html>`
    ECG Qualitative Analysis
    """

    def __init__(
        self,
        ecg_finding_text: Optional[str] = None,
        ecg_finding_codes: Optional[Sequence[ECGFinding]] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.DCM.QualitativeAnalysis,
            template_id='3717',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if ecg_finding_text is None and ecg_finding_codes is None:
            raise ValueError(
                'Either argument "ecg_finding_text" or "ECG_finding_code" ' +
                'must be given.'
            )
        if ecg_finding_text is not None:
            if not isinstance(ecg_finding_text, str):
                raise TypeError(
                    'Argument "ecg_finding_text" must have type str.'
                )
            ecg_finding_text_item = TextContentItem(
                name=CodedConcept(
                    value='271921002',
                    meaning='ECG Finding',
                    scheme_designator='SCT'
                ),
                value=ecg_finding_text,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(ecg_finding_text_item)
        if ecg_finding_codes is not None:
            if not isinstance(ecg_finding_codes, (list, tuple, set)):
                raise TypeError(
                    'Argument "ecg_finding_code" must be a sequence.'
                )
            for ecg_finding_code in ecg_finding_codes:
                if not isinstance(ecg_finding_code, ECGFinding):
                    raise TypeError(
                        'Items of argument "ecg_finding" must have ' +
                        'type ECGFinding.'
                    )
                content.extend(ecg_finding_code)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class SummaryECG(Template):
    """:dcm:`TID 3719 <part16/sect_TID_3719.html>`
    Summary, ECG
    """

    def __init__(
        self,
        summary: Optional[str] = None,
        ecg_overall_finding: Optional[Union[Code, CodedConcept]] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.LN.Summary,
            template_id='3719',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if summary is not None:
            if not isinstance(summary, str):
                raise TypeError(
                    'Argument "summary" must have type str.'
                )
            summary_item = TextContentItem(
                name=codes.LN.Summary,
                value=summary,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(summary_item)
        if ecg_overall_finding is not None:
            if not isinstance(ecg_overall_finding, (Code, CodedConcept)):
                raise TypeError(
                    'Argument "ECG_overall_finding" must have type ' +
                    'Code or CodedConcept.'
                )
            ecg_overall_finding_item = CodeContentItem(
                name=CodedConcept(
                    value='18810-2',
                    meaning='ECG overall finding',
                    scheme_designator='LN'
                ),
                value=ecg_overall_finding,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(ecg_overall_finding_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class ECGReport(Template):
    """:dcm:`TID 3700 <part16/chapter_A.html#sect_TID_3700>`
    ECG Report
    """

    def __init__(
        self,
        language_of_content_item_and_descendants:
            LanguageOfContentItemAndDescendants,
        observer_contexts: Sequence[ObserverContext],
        ecg_waveform_information: ECGWaveFormInformation,
        quantitative_analysis: QuantitativeAnalysis,
        procedure_reported: Optional[
            Union[CodedConcept, Code]] = None,
        indications_for_procedure: Optional[
            IndicationsForProcedure] = None,
        cardiovascular_patient_history: Optional[
            CardiovascularPatientHistory] = None,
        patient_characteristics_for_ecg: Optional[
            PatientCharacteristicsForECG] = None,
        prior_ecg_study: Optional[PriorECGStudy] = None,
        ecg_qualitative_analysis: Optional[ECGQualitativeAnalysis] = None,
        summary_ecg: Optional[SummaryECG] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.LN.ECGReport,
            template_id='3700'
        )
        content = ContentSequence()
        if not isinstance(language_of_content_item_and_descendants,
                          LanguageOfContentItemAndDescendants):
            raise TypeError(
                'Argument "language_of_content_item_and_descendants" must ' +
                'have type LanguageOfContentItemAndDescendants.'
            )
        content.extend(language_of_content_item_and_descendants)
        if not isinstance(observer_contexts, (list, tuple, set)):
            raise TypeError(
                'Argument "observer_contexts" must be a sequence.'
            )
        for observer_context in observer_contexts:
            if not isinstance(observer_context, ObserverContext):
                raise TypeError(
                    'Items of argument "observer_context" must have ' +
                    'type ObserverContext.'
                )
            content.extend(observer_context)
        if not isinstance(ecg_waveform_information, ECGWaveFormInformation):
            raise TypeError(
                'Argument "ecg_waveform_information" must have type ' +
                'ECGWaveFormInformation.'
            )
        content.extend(ecg_waveform_information)
        if not isinstance(quantitative_analysis, QuantitativeAnalysis):
            raise TypeError(
                'Argument "quantitative_analysis" must have type ' +
                'QuantitativeAnalysis.'
            )
        content.extend(quantitative_analysis)
        if procedure_reported is not None:
            if not isinstance(procedure_reported, (CodedConcept, Code)):
                raise TypeError(
                    'Argument "procedure_reported" must have type ' +
                    'Code or CodedConcept.'
                )
            procedure_reported_item = CodeContentItem(
                name=codes.DCM.ProcedureReported,
                value=procedure_reported,
                relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
            )
            content.append(procedure_reported_item)
        if indications_for_procedure is not None:
            if not isinstance(indications_for_procedure,
                              IndicationsForProcedure):
                raise TypeError(
                    'Argument "indications_for_procedure" must have ' +
                    'type IndicationsForProcedure.'
                )
            content.extend(indications_for_procedure)
        if cardiovascular_patient_history is not None:
            if not isinstance(cardiovascular_patient_history,
                              CardiovascularPatientHistory):
                raise TypeError(
                    'Argument "cardiovascular_patient_history" must have ' +
                    'type CardiovascularPatientHistory.'
                )
            content.extend(cardiovascular_patient_history)
        if patient_characteristics_for_ecg is not None:
            if not isinstance(patient_characteristics_for_ecg,
                              PatientCharacteristicsForECG):
                raise TypeError(
                    'Argument "patient_characteristics_for_ecg" must have ' +
                    'type PatientCharacteristicsForECG.'
                )
            content.extend(patient_characteristics_for_ecg)
        if prior_ecg_study is not None:
            if not isinstance(prior_ecg_study, PriorECGStudy):
                raise TypeError(
                    'Argument "prior_ecg_study" must have type PriorECGStudy.'
                )
            content.extend(prior_ecg_study)
        if ecg_qualitative_analysis is not None:
            if not isinstance(ecg_qualitative_analysis, ECGQualitativeAnalysis):
                raise TypeError(
                    'Argument "ecg_qualitative_analysis" must have type ' +
                    'ECGQualitativeAnalysis.'
                )
            content.extend(ecg_qualitative_analysis)
        if summary_ecg is not None:
            if not isinstance(summary_ecg, SummaryECG):
                raise TypeError(
                    'Argument "summary_ecg" must have type SummaryECG.'
                )
            content.extend(summary_ecg)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item], is_root=True)
