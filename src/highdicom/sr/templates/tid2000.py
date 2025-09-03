"""Templates related to TID2000 Basic Diagnostic Imaging Report"""
from typing import Optional, Sequence, Union
from highdicom.sr.value_types import (
    Code,
    CodeContentItem,
    CodedConcept,
    ContentSequence,
    TextContentItem,
    ContainerContentItem,
    ImageContentItem,
    RelationshipTypeValues
)
from highdicom.sr.templates.common import (
    LanguageOfValue,
    Template,
    LanguageOfContentItemAndDescendants,
    ObservationContext
)
from pydicom.sr.codedict import codes


class EquivalentMeaningsOfConceptNameText(TextContentItem):
    """:dcm:`TID 1210 <part16/chapter_A.html#sect_TID_1210>`
    Equivalent Meaning(s) of Concept Name Text
    """

    def __init__(self,
                 value: str,
                 language_of_value: Optional[LanguageOfValue] = None
                 ) -> None:
        super().__init__(
            name=codes.DCM.EquivalentMeaningOfConceptName,
            value=value,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD)
        content = ContentSequence()
        if language_of_value is not None:
            content.append(language_of_value)
        if len(content) > 0:
            self.ContentSequence = content


class EquivalentMeaningsOfConceptNameCode(CodeContentItem):
    """:dcm:`TID 1210 <part16/chapter_A.html#sect_TID_1210>`
    Equivalent Meaning(s) of Concept Name Code
    """

    def __init__(self,
                 value: Union[Code, CodedConcept],
                 language_of_value: Optional[LanguageOfValue] = None
                 ) -> None:
        super().__init__(
            name=codes.DCM.EquivalentMeaningOfConceptName,
            value=value,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD)
        content = ContentSequence()
        if language_of_value is not None:
            content.append(language_of_value)
        if len(content) > 0:
            self.ContentSequence = content


class ReportNarrativeCode(CodeContentItem):
    """:dcm:`TID 2002 <part16/chapter_A.html#sect_TID_2002>`
    Report Narrative Code
    """

    def __init__(self,
                 value: Union[Code, CodedConcept],
                 basic_diagnostic_imaging_report_observations: Optional[
                     Sequence[ImageContentItem]] = None
                 ) -> None:
        super().__init__(
            name=CodedConcept(
                value='7002',
                meaning='Diagnostic Imaging Report Element',
                scheme_designator='CID'
            ),
            value=value,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if not isinstance(basic_diagnostic_imaging_report_observations, (
            list, tuple, set
        )):
            raise TypeError(
                'Argument "basic_diagnostic_imaging_report_observations" ' +
                'must be a sequence.'
            )
        for basic_diagnostic_imaging_report_observation in \
                basic_diagnostic_imaging_report_observations:
            if not isinstance(basic_diagnostic_imaging_report_observation,
                              ImageContentItem):
                raise TypeError(
                    'Items of argument' +
                    ' "basic_diagnostic_imaging_report_observation" ' +
                    'must have type ImageContentItem.'
                )
            content.append(basic_diagnostic_imaging_report_observation)
        if len(content) > 0:
            self.ContentSequence = content


class ReportNarrativeText(TextContentItem):
    """:dcm:`TID 2002 <part16/chapter_A.html#sect_TID_2002>`
    Report Narrative Text
    """

    def __init__(self,
                 value: str,
                 basic_diagnostic_imaging_report_observations: Optional[
                     Sequence[ImageContentItem]] = None
                 ) -> None:
        super().__init__(
            name=CodedConcept(
                value='7002',
                meaning='Diagnostic Imaging Report Element',
                scheme_designator='CID'
            ),
            value=value,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if not isinstance(basic_diagnostic_imaging_report_observations, (
            list, tuple, set
        )):
            raise TypeError(
                'Argument "basic_diagnostic_imaging_report_observations" ' +
                'must be a sequence.'
            )
        for basic_diagnostic_imaging_report_observation in \
                basic_diagnostic_imaging_report_observations:
            if not isinstance(basic_diagnostic_imaging_report_observation,
                              ImageContentItem):
                raise TypeError(
                    'Items of argument ' +
                    ' "basic_diagnostic_imaging_report_observation" ' +
                    'must have type ImageContentItem.'
                )
            content.append(basic_diagnostic_imaging_report_observation)
        if len(content) > 0:
            self.ContentSequence = content


class DiagnosticImagingReportHeading(Template):
    """:dcm:`CID 7001 <part16/sect_CID_7001.html>`
    Diagnostic Imaging Report Heading
    """

    def __init__(
        self,
        report_narrative: Union[ReportNarrativeCode, ReportNarrativeText],
        observation_context: Optional[ObservationContext] = None
    ) -> None:
        item = ContainerContentItem(
            name=CodedConcept(
                value='7001',
                meaning='Diagnostic Imaging Report Heading',
                scheme_designator='CID'
            ),
            template_id='7001',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        item.ContentSequence = ContentSequence()
        if not isinstance(report_narrative, (
            ReportNarrativeCode, ReportNarrativeText
        )):
            raise TypeError(
                'Argument "report_narrative" must have type ' +
                'ReportNarrativeCode or ReportNarrativeText.'
            )
        item.ContentSequence.append(report_narrative)
        if observation_context is not None:
            if not isinstance(observation_context, ObservationContext):
                raise TypeError(
                    'Argument "observation_context" must have type ' +
                    'ObservationContext.'
                )
            item.ContentSequence.extend(observation_context)
        super().__init__([item])

class BasicDiagnosticImagingReport(Template):
    """:dcm:`TID 2000 <part16/chapter_A.html#sect_TID_2000>`
    Basic Diagnostic Imaging Report
    """

    def __init__(
        self,
        language_of_content_item_and_descendants:
            LanguageOfContentItemAndDescendants,
        observation_context: ObservationContext,
        procedures_reported: Optional[Sequence[
            Union[CodedConcept, Code]]] = None,
        acquisition_device_types: Optional[Sequence[
            Union[CodedConcept, Code]]] = None,
        target_regions: Optional[Sequence[
            Union[CodedConcept, Code]]] = None,
        equivalent_meanings_of_concept_name: Optional[Sequence[
            Union[EquivalentMeaningsOfConceptNameText,
                  EquivalentMeaningsOfConceptNameCode]]] = None,
        diagnostic_imaging_report_headings: Optional[Sequence[
            DiagnosticImagingReportHeading]] = None
    ) -> None:
        item = ContainerContentItem(
            name=CodedConcept(
                value='7000',
                meaning='Diagnostic Imaging Report Document Title',
                scheme_designator='CID'
            ),
            template_id='2000'
        )
        item.ContentSequence = ContentSequence()
        if not isinstance(language_of_content_item_and_descendants,
                          LanguageOfContentItemAndDescendants):
            raise TypeError(
                'Argument "language_of_content_item_and_descendants" must ' +
                'have type LanguageOfContentItemAndDescendants.'
            )
        item.ContentSequence.extend(language_of_content_item_and_descendants)
        if not isinstance(observation_context, ObservationContext):
            raise TypeError(
                'Argument "observation_context" must have type ' +
                'ObservationContext.'
            )
        item.ContentSequence.extend(observation_context)
        if procedures_reported is not None:
            if not isinstance(procedures_reported, (list, tuple, set)):
                raise TypeError(
                    'Argument "procedures_reported" must be a sequence.'
                )
            for procedure_reported in procedures_reported:
                if not isinstance(procedure_reported, (CodedConcept, Code)):
                    raise TypeError(
                        'Items of argument "procedure_reported" must have ' +
                        'type Code or CodedConcept.'
                    )
                procedure_reported_item = CodeContentItem(
                    name=codes.DCM.ProcedureReported,
                    value=procedure_reported,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                item.ContentSequence.append(procedure_reported_item)
        if acquisition_device_types is not None:
            if not isinstance(acquisition_device_types, (list, tuple, set)):
                raise TypeError(
                    'Argument "acquisition_device_types" must be a sequence.'
                )
            for acquisition_device_type in acquisition_device_types:
                if not isinstance(acquisition_device_type, (
                    CodedConcept, Code
                )):
                    raise TypeError(
                        'Items of argument "acquisition_device_type" must ' +
                        'have type Code or CodedConcept.'
                    )
                acquisition_device_type_item = CodeContentItem(
                    name=codes.DCM.AcquisitionDeviceType,
                    value=acquisition_device_type,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                item.ContentSequence.append(acquisition_device_type_item)
        if target_regions is not None:
            if not isinstance(target_regions, (list, tuple, set)):
                raise TypeError(
                    'Argument "target_regions" must be a sequence.'
                )
            for target_region in target_regions:
                if not isinstance(target_region, (CodedConcept, Code)):
                    raise TypeError(
                        'Items of argument "target_region" must have type ' +
                        'Code or CodedConcept.'
                    )
                target_region_item = CodeContentItem(
                    name=codes.DCM.TargetRegion,
                    value=target_region,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                item.ContentSequence.append(target_region_item)
        if equivalent_meanings_of_concept_name is not None:
            if not isinstance(equivalent_meanings_of_concept_name, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "equivalent_meanings_of_concept_name" ' +
                    'must be a sequence.'
                )
            for equivalent_meaning_of_concept_name in \
                    equivalent_meanings_of_concept_name:
                if not isinstance(equivalent_meaning_of_concept_name, (
                    EquivalentMeaningsOfConceptNameCode,
                    EquivalentMeaningsOfConceptNameText
                )):
                    raise TypeError(
                        'Items of argument' +
                        ' "equivalent_meaning_of_concept_name" ' +
                        'must have type EquivalentMeaningsOfConceptNameCode ' +
                        'or EquivalentMeaningsOfConceptNameText.'
                    )
                item.ContentSequence.append(equivalent_meaning_of_concept_name)
        if diagnostic_imaging_report_headings is not None:
            if not isinstance(diagnostic_imaging_report_headings, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "diagnostic_imaging_report_headings" ' +
                    'must be a sequence.'
                )
            for diagnostic_imaging_report_heading in \
                    diagnostic_imaging_report_headings:
                if not isinstance(diagnostic_imaging_report_heading,
                                  DiagnosticImagingReportHeading):
                    raise TypeError(
                        'Items of argument' +
                        ' "diagnostic_imaging_report_heading" ' +
                        'must have type DiagnosticImagingReportHeading.'
                    )
                item.ContentSequence.extend(diagnostic_imaging_report_heading)
        super().__init__([item], is_root=True)
