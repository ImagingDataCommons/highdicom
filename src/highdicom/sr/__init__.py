"""Package for creationg of Structured Report (SR) instances."""
from highdicom.sr.coding import CodedConcept
from highdicom.sr.content import (
    FindingSite,
    ImageRegion,
    ImageRegion3D,
    LongitudinalTemporalOffsetFromEvent,
    SourceImageForMeasurement,
    SourceImageForMeasurementGroup,
    SourceImageForSegmentation,
    SourceImageForRegion,
    SourceSeriesForSegmentation,
    RealWorldValueMap,
    ReferencedSegment,
    ReferencedSegmentationFrame,
    VolumeSurface,
)
from highdicom.sr.enum import (
    GraphicTypeValues,
    GraphicTypeValues3D,
    PixelOriginInterpretationValues,
    RelationshipTypeValues,
    TemporalRangeTypeValues,
    ValueTypeValues,
)
from highdicom.sr.sop import (
    EnhancedSR,
    ComprehensiveSR,
    Comprehensive3DSR,
)
from highdicom.sr.templates import (
    AlgorithmIdentification,
    DeviceObserverIdentifyingAttributes,
    ImageLibrary,
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
    TrackingIdentifier,
    TimePointContext,
    VolumetricROIMeasurementsAndQualitativeEvaluations,
)
from highdicom.sr import utils
from highdicom.sr.value_types import (
    ContentItem,
    ContentSequence,
    CodeContentItem,
    ContainerContentItem,
    CompositeContentItem,
    DateContentItem,
    DateTimeContentItem,
    ImageContentItem,
    NumContentItem,
    PnameContentItem,
    ScoordContentItem,
    Scoord3DContentItem,
    TcoordContentItem,
    TextContentItem,
    TimeContentItem,
    UIDRefContentItem,
)

SOP_CLASS_UIDS = {
    '1.2.840.10008.5.1.4.1.1.88.11',  # Basic Text SR
    '1.2.840.10008.5.1.4.1.1.88.22',  # Enhanced SR
    '1.2.840.10008.5.1.4.1.1.88.33',  # Comprehensive SR
    '1.2.840.10008.5.1.4.1.1.88.34',  # Comprehensive 3D SR
    '1.2.840.10008.5.1.4.1.1.88.35',  # Extensible SR
    '1.2.840.10008.5.1.4.1.1.88.40',  # Procedure Log
    '1.2.840.10008.5.1.4.1.1.88.50',  # Mammography CAD SR
    '1.2.840.10008.5.1.4.1.1.88.65',  # Chest CAD SR
    '1.2.840.10008.5.1.4.1.1.88.67',  # X-Ray Radiation Dose SR
    '1.2.840.10008.5.1.4.1.1.88.68',  # Radiopharmaceutical Radiation Dose SR
    '1.2.840.10008.5.1.4.1.1.88.69',  # Colon CAD SR
    '1.2.840.10008.5.1.4.1.1.88.70',  # Implantation Plan SR
    '1.2.840.10008.5.1.4.1.1.88.71',  # Acquisition Context SR
    '1.2.840.10008.5.1.4.1.1.88.72',  # Simplified Adult Echo SR
    '1.2.840.10008.5.1.4.1.1.88.73',  # Patient Radiation Dose SR
}

__all__ = [
    'AlgorithmIdentification',
    'EnhancedSR',
    'CodedConcept',
    'ContentSequence',
    'ComprehensiveSR',
    'Comprehensive3DSR',
    'ContentItem',
    'ContentSequence',
    'CodeContentItem',
    'ContainerContentItem',
    'CompositeContentItem',
    'DateContentItem',
    'DateTimeContentItem',
    'DeviceObserverIdentifyingAttributes',
    'FindingSite',
    'GraphicTypeValues',
    'GraphicTypeValues3D',
    'ImageContentItem',
    'ImageLibrary',
    'ImageLibraryEntryDescriptors',
    'ImageRegion',
    'ImageRegion3D',
    'LanguageOfContentItemAndDescendants',
    'LongitudinalTemporalOffsetFromEvent',
    'Measurement',
    'MeasurementProperties',
    'MeasurementReport',
    'MeasurementsAndQualitativeEvaluations',
    'MeasurementStatisticalProperties',
    'NormalRangeProperties',
    'NumContentItem',
    'ObserverContext',
    'ObservationContext',
    'PersonObserverIdentifyingAttributes',
    'PlanarROIMeasurementsAndQualitativeEvaluations',
    'PixelOriginInterpretationValues',
    'PnameContentItem',
    'QualitativeEvaluation',
    'RealWorldValueMap',
    'ReferencedSegment',
    'ReferencedSegmentationFrame',
    'RelationshipTypeValues',
    'ScoordContentItem',
    'Scoord3DContentItem',
    'SourceImageForMeasurement',
    'SourceImageForMeasurementGroup',
    'SourceImageForSegmentation',
    'SourceImageForRegion',
    'SourceSeriesForSegmentation',
    'SubjectContext',
    'SubjectContextDevice',
    'SubjectContextFetus',
    'SubjectContextSpecimen',
    'TcoordContentItem',
    'TemporalRangeTypeValues',
    'TextContentItem',
    'TimeContentItem',
    'TimePointContext',
    'TrackingIdentifier',
    'UIDRefContentItem',
    'utils',
    'ValueTypeValues',
    'VolumeSurface',
    'VolumetricROIMeasurementsAndQualitativeEvaluations',
]
