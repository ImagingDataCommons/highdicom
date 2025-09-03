"""Classes implementing structured reporting templates"""
from highdicom.sr.templates.common import (
    DeviceObserverIdentifyingAttributes,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
    AlgorithmIdentification,
    LanguageOfContentItemAndDescendants,
    ObservationContext,
    SubjectContext,
    SubjectContextDevice,
    SubjectContextFetus,
    SubjectContextSpecimen,
    AgeUnit,
    PressureUnit,
    LanguageOfValue
)
from highdicom.sr.templates.tid1500 import (
    TrackingIdentifier,
    TimePointContext,
    MeasurementStatisticalProperties,
    NormalRangeProperties,
    MeasurementProperties,
    QualitativeEvaluation,
    Measurement,
    MeasurementsAndQualitativeEvaluations,
    PlanarROIMeasurementsAndQualitativeEvaluations,
    VolumetricROIMeasurementsAndQualitativeEvaluations,
    ImageLibraryEntryDescriptors,
    MeasurementReport,
    ImageLibraryEntry,
    ImageLibrary
)
from highdicom.sr.templates.tid2000 import (
    EquivalentMeaningsOfConceptNameText,
    EquivalentMeaningsOfConceptNameCode,
    ReportNarrativeCode,
    ReportNarrativeText,
    DiagnosticImagingReportHeading,
    BasicDiagnosticImagingReport
)
from highdicom.sr.templates.tid3700 import (
    ECGWaveFormInformation,
    ECGMeasurementSource,
    QTcIntervalGlobal,
    NumberOfEctopicBeats,
    ECGGlobalMeasurements,
    ECGLeadMeasurements,
    QuantitativeAnalysis,
    IndicationsForProcedure,
    PatientCharacteristicsForECG,
    PriorECGStudy,
    ECGFinding,
    ECGQualitativeAnalysis,
    SummaryECG,
    ECGReport
)
from highdicom.sr.templates.tid3802 import (
    Therapy,
    ProblemProperties,
    ProblemList,
    SocialHistory,
    ProcedureProperties,
    PastSurgicalHistory,
    RelevantDiagnosticTestsAndOrLaboratoryData,
    MedicationTypeText,
    MedicationTypeCode,
    HistoryOfMedicationUse,
    FamilyHistoryOfClinicalFinding,
    HistoryOfFamilyMemberDiseases,
    MedicalDeviceUse,
    HistoryOfMedicalDeviceUse,
    CardiovascularPatientHistory
)

__all__ = [
    # Common
    'DeviceObserverIdentifyingAttributes',
    'ObserverContext',
    'PersonObserverIdentifyingAttributes',
    'AlgorithmIdentification',
    'LanguageOfContentItemAndDescendants',
    'ObservationContext',
    'SubjectContext',
    'SubjectContextDevice',
    'SubjectContextFetus',
    'SubjectContextSpecimen',
    'AgeUnit',
    'PressureUnit',
    'LanguageOfValue',

    # TID 1500
    'TrackingIdentifier',
    'TimePointContext',
    'MeasurementStatisticalProperties',
    'NormalRangeProperties',
    'MeasurementProperties',
    'QualitativeEvaluation',
    'Measurement',
    'MeasurementsAndQualitativeEvaluations',
    'PlanarROIMeasurementsAndQualitativeEvaluations',
    'VolumetricROIMeasurementsAndQualitativeEvaluations',
    'ImageLibraryEntryDescriptors',
    'MeasurementReport',
    'ImageLibraryEntry',
    'ImageLibrary',

    # TID 2000
    'EquivalentMeaningsOfConceptNameText',
    'EquivalentMeaningsOfConceptNameCode',
    'ReportNarrativeCode',
    'ReportNarrativeText',
    'DiagnosticImagingReportHeading',
    'BasicDiagnosticImagingReport',

    # TID 3700
    'ECGWaveFormInformation',
    'ECGMeasurementSource',
    'QTcIntervalGlobal',
    'NumberOfEctopicBeats',
    'ECGGlobalMeasurements',
    'ECGLeadMeasurements',
    'QuantitativeAnalysis',
    'IndicationsForProcedure',
    'PatientCharacteristicsForECG',
    'PriorECGStudy',
    'ECGFinding',
    'ECGQualitativeAnalysis',
    'SummaryECG',
    'ECGReport',

    # TID 3802
    'Therapy',
    'ProblemProperties',
    'ProblemList',
    'SocialHistory',
    'ProcedureProperties',
    'PastSurgicalHistory',
    'RelevantDiagnosticTestsAndOrLaboratoryData',
    'MedicationTypeText',
    'MedicationTypeCode',
    'HistoryOfMedicationUse',
    'FamilyHistoryOfClinicalFinding',
    'HistoryOfFamilyMemberDiseases',
    'MedicalDeviceUse',
    'HistoryOfMedicalDeviceUse',
    'CardiovascularPatientHistory'
]
