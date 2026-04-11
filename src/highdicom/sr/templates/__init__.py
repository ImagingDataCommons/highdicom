"""DICOM structured reporting templates.

This package contains implementations of DICOM SR templates defined in
DICOM PS3.16. Each sub-module covers one top-level template family:

- :mod:`tid1500` — TID 1500 Measurement Report (general imaging measurements)
- :mod:`tid6000` — Supplement 247 Eyecare Measurement Templates (TID 6001–6009)
"""

# Re-export everything from tid1500 for backwards compatibility.
# All names that were previously importable from highdicom.sr.templates
# remain importable without change.
from highdicom.sr.templates.tid1500 import (
    AlgorithmIdentification,
    DeviceObserverIdentifyingAttributes,
    ImageLibrary,
    ImageLibraryEntry,
    ImageLibraryEntryDescriptors,
    LanguageOfContentItemAndDescendants,
    Measurement,
    MeasurementProperties,
    MeasurementReport,
    MeasurementsAndQualitativeEvaluations,
    MeasurementStatisticalProperties,
    NormalRangeProperties,
    ObserverContext,
    ObservationContext,
    PersonObserverIdentifyingAttributes,
    PlanarROIMeasurementsAndQualitativeEvaluations,
    QualitativeEvaluation,
    SubjectContext,
    SubjectContextDevice,
    SubjectContextFetus,
    SubjectContextSpecimen,
    Template,
    TimePointContext,
    TrackingIdentifier,
    VolumetricROIMeasurementsAndQualitativeEvaluations,
)

# Supplement 247 — Eyecare Measurement Templates
from highdicom.sr.templates.tid6000 import (
    CircumpapillaryRNFLKeyMeasurements,
    MacularThicknessKeyMeasurements,
    OphthalmologyMeasurementsGroup,
)

__all__ = [
    # TID 1500 family
    "AlgorithmIdentification",
    "DeviceObserverIdentifyingAttributes",
    "ImageLibrary",
    "ImageLibraryEntry",
    "ImageLibraryEntryDescriptors",
    "LanguageOfContentItemAndDescendants",
    "Measurement",
    "MeasurementProperties",
    "MeasurementReport",
    "MeasurementsAndQualitativeEvaluations",
    "MeasurementStatisticalProperties",
    "NormalRangeProperties",
    "ObserverContext",
    "ObservationContext",
    "PersonObserverIdentifyingAttributes",
    "PlanarROIMeasurementsAndQualitativeEvaluations",
    "QualitativeEvaluation",
    "SubjectContext",
    "SubjectContextDevice",
    "SubjectContextFetus",
    "SubjectContextSpecimen",
    "Template",
    "TimePointContext",
    "TrackingIdentifier",
    "VolumetricROIMeasurementsAndQualitativeEvaluations",
    # Supplement 247 family
    "OphthalmologyMeasurementsGroup",
    "CircumpapillaryRNFLKeyMeasurements",
    "MacularThicknessKeyMeasurements",
]
