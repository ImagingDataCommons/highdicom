"""DICOM structured reporting templates.

This package contains implementations of DICOM SR templates defined in
DICOM PS3.16. Each sub-module covers one top-level template family:

- :mod:`tid1500` — TID 1500 Measurement Report (general imaging measurements)
- :mod:`tid2120` — Eyecare Measurement Templates (TID 2120, 2123, 2124),
  ratified in DICOM PS3.16 2025b (formerly circulated as Supplement 247
  draft TIDs 6001, 6004, 6005)
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

# Eyecare Measurement Templates (TID 2120, 2123, 2124)
from highdicom.sr.templates.tid2120 import (
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
    # Eyecare Measurement Templates family
    "OphthalmologyMeasurementsGroup",
    "CircumpapillaryRNFLKeyMeasurements",
    "MacularThicknessKeyMeasurements",
]
