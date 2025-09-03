from datetime import datetime
from typing import Optional, Sequence, Union

from pydicom.sr.codedict import codes
from pydicom.valuerep import DT

from highdicom.sr.value_types import (
    Code,
    CodeContentItem,
    CodedConcept,
    DateTimeContentItem,
    NumContentItem,
    TextContentItem,
    CompositeContentItem,
    ContainerContentItem,
    PnameContentItem,
    RelationshipTypeValues
)
from highdicom.sr.templates.common import Template

from highdicom.sr.value_types import (
    ContentSequence
)

MILLIGRAM_PER_DECILITER = Code(
    value='mg/dL',
    scheme_designator='UCUM',
    meaning='milligram per deciliter',
    scheme_version=None
)


class Therapy(CodeContentItem):
    """:sct:`277132007`
    Therapy
    """

    def __init__(
        self,
        value: Union[Code, CodedConcept],
        status: Optional[Union[Code, CodedConcept]] = None
    ) -> None:
        super().__init__(
            name=CodedConcept(
                value='277132007',
                meaning='Therapy',
                scheme_designator='SCT'
            ),
            value=value,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if status is not None:
            status_item = CodeContentItem(
                name=CodedConcept(
                    value='33999-4',
                    meaning='Status',
                    scheme_designator='LN'
                ),
                value=status,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(status_item)
        if len(content) > 0:
            self.ContentSequence = content


class ProblemProperties(Template):
    """:dcm:`TID 3829 <part16/sect_TID_3829.html>`
    Problem Properties
    """

    def __init__(
        self,
        # TODO: Concern Type should be a class for itself
        concern_type: Union[Code, CodedConcept],
        datetime_concern_noted: Optional[Union[str, datetime, DT]] = None,
        datetime_concern_resolved: Optional[Union[str, datetime, DT]] = None,
        health_status: Optional[Union[Code, CodedConcept]] = None,
        therapies: Optional[Sequence[Therapy]] = None,
        comment: Optional[str] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.DCM.Concern,
            template_id='3829',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        concern_item = CodeContentItem(
            name=CodedConcept(
                value='3769',
                meaning='Concern Type',
                scheme_designator='CID'
            ),
            value=concern_type,
            relationship_type=RelationshipTypeValues.HAS_PROPERTIES
        )
        content.append(concern_item)
        if datetime_concern_noted is not None:
            datetime_concern_noted_item = DateTimeContentItem(
                name=codes.DCM.DatetimeConcernNoted,
                value=datetime_concern_noted,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(datetime_concern_noted_item)
        if datetime_concern_resolved is not None:
            datetime_concern_resolved_item = DateTimeContentItem(
                name=codes.DCM.DatetimeConcernResolved,
                value=datetime_concern_resolved,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(datetime_concern_resolved_item)
        if health_status is not None:
            health_status_item = CodeContentItem(
                name=CodedConcept(
                    value='11323-3',
                    meaning='Health status',
                    scheme_designator='LN'
                ),
                value=health_status,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(health_status_item)
        if therapies is not None:
            if not isinstance(therapies, (list, tuple, set)):
                raise TypeError(
                    'Argument "therapies" must be a sequence.'
                )
            for therapy in therapies:
                if not isinstance(therapy, Therapy):
                    raise TypeError(
                        'Items of argument "therapies" must have type Therapy.'
                    )
                content.append(therapy)
        if comment is not None:
            comment_item = TextContentItem(
                name=codes.DCM.Comment,
                value=comment,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(comment_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class ProblemList(Template):
    """:ln:`11450-4`
    Problem List
    """

    def __init__(
        self,
        concern_types: Optional[
            Sequence[str]] = None,
        cardiac_patient_risk_factors: Optional[
            Sequence[ProblemProperties]] = None,
        history_of_diabetes_mellitus: Optional[
            ProblemProperties] = None,
        history_of_hypertension: Optional[
            ProblemProperties] = None,
        history_of_hypercholesterolemia: Optional[
            ProblemProperties] = None,
        arrhythmia: Optional[
            ProblemProperties] = None,
        history_of_myocardial_infarction: Optional[
            ProblemProperties] = None,
        history_of_kidney_disease: Optional[
            ProblemProperties] = None
    ) -> None:
        item = ContainerContentItem(
            name=CodedConcept(
                value='11450-4',
                meaning='Problem List',
                scheme_designator='LN'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if concern_types is not None:
            if not isinstance(concern_types, (list, tuple, set)):
                raise TypeError(
                    'Argument "concern_types" must be a sequence.'
                )
            for concern_type in concern_types:
                if not isinstance(concern_type, str):
                    raise TypeError(
                        'Items of argument "concern_types" must have type str.'
                    )
                concern_type_item = TextContentItem(
                    name=CodedConcept(
                        value='3769',
                        meaning='Concern Type',
                        scheme_designator='CID'
                    ),
                    value=concern_type,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(concern_type_item)
        if cardiac_patient_risk_factors is not None:
            if not isinstance(cardiac_patient_risk_factors, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "cardiac_patient_risk_factors" must ' +
                    'be a sequence.'
                )
            for cardiac_patient_risk_factor in cardiac_patient_risk_factors:
                if not isinstance(cardiac_patient_risk_factor,
                                  ProblemProperties):
                    raise TypeError(
                        'Items of argument "cardiac_patient_risk_factors" ' +
                        'must have type ProblemProperties.'
                    )
                content.extend(cardiac_patient_risk_factor)
        if history_of_diabetes_mellitus is not None:
            if not isinstance(history_of_diabetes_mellitus, ProblemProperties):
                raise TypeError(
                    'Argument "history_of_diabetes_mellitus" must ' +
                    'be a ProblemProperties.'
                )
            content.extend(history_of_diabetes_mellitus)
        if history_of_hypertension is not None:
            if not isinstance(history_of_hypertension, ProblemProperties):
                raise TypeError(
                    'Argument "history_of_hypertension" must ' +
                    'be a ProblemProperties.'
                )
            content.extend(history_of_hypertension)
        if history_of_hypercholesterolemia is not None:
            if not isinstance(history_of_hypercholesterolemia,
                              ProblemProperties):
                raise TypeError(
                    'Argument "history_of_hypercholesterolemia" must ' +
                    'be a ProblemProperties.'
                )
            content.extend(history_of_hypercholesterolemia)
        if arrhythmia is not None:
            if not isinstance(arrhythmia, ProblemProperties):
                raise TypeError(
                    'Argument "arrhythmia" must be a ProblemProperties.'
                )
            content.extend(arrhythmia)
        if history_of_myocardial_infarction is not None:
            if not isinstance(history_of_myocardial_infarction,
                              ProblemProperties):
                raise TypeError(
                    'Argument "history_of_myocardial_infarction" ' +
                    'must be a ProblemProperties.'
                )
            content.extend(history_of_myocardial_infarction)
        if history_of_kidney_disease is not None:
            if not isinstance(history_of_kidney_disease, ProblemProperties):
                raise TypeError(
                    'Argument "history_of_kidney_disease" must ' +
                    'be a ProblemProperties.'
                )
            content.extend(history_of_kidney_disease)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class SocialHistory(Template):
    """:ln:`29762-2`
    Social History
    """

    def __init__(
        self,
        social_history: Optional[str] = None,
        social_histories: Optional[Sequence[str]] = None,
        tobacco_smoking_behavior: Optional[
            Union[Code, CodedConcept]] = None,
        drug_misuse_behavior: Optional[Union[Code,
                                             CodedConcept]] = None,
    ) -> None:
        item = ContainerContentItem(
            name=CodedConcept(
                value='29762-2',
                meaning='Social History',
                scheme_designator='LN'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if social_history is not None:
            social_history_item = TextContentItem(
                name=CodedConcept(
                    value='160476009',
                    meaning='Social History',
                    scheme_designator='SCT'
                ),
                value=social_history,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(social_history_item)
        if social_histories is not None:
            if not isinstance(social_histories, (list, tuple, set)):
                raise TypeError(
                    'Argument "social_histories" must be a sequence.'
                )
            for history in social_histories:
                if not isinstance(history, str):
                    raise TypeError(
                        'Items of argument "social_histories" must ' +
                        'have type str.'
                    )
                social_history_item = TextContentItem(
                    name=CodedConcept(
                        value='3774',
                        meaning='Social History',
                        scheme_designator='CID'
                    ),
                    value=history,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(social_history_item)
        if tobacco_smoking_behavior is not None:
            tobacco_smoking_behavior_item = CodeContentItem(
                name=codes.SCT.TobaccoSmokingBehavior,
                value=tobacco_smoking_behavior,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(tobacco_smoking_behavior_item)
        if drug_misuse_behavior is not None:
            drug_misuse_behavior_item = CodeContentItem(
                name=codes.SCT.DrugMisuseBehavior,
                value=drug_misuse_behavior,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(drug_misuse_behavior_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class ProcedureProperties(Template):
    """:dcm:`TID 3830 <part16/sect_TID_3830.html>`
    Procedure Properties
    """

    def __init__(
        self,
        name: Union[CodedConcept, Code],
        value: Union[CodedConcept, Code],
        procedure_datetime: Optional[
            Union[str, datetime, DT]] = None,
        clinical_reports: Optional[
            Sequence[CompositeContentItem]] = None,
        clinical_reports_text: Optional[Sequence[str]] = None,
        service_delivery_location: Optional[str] = None,
        service_performer_person: Optional[str] = None,
        service_performer_organisation: Optional[str] = None,
        comment: Optional[str] = None,
        procedure_results: Optional[
            Sequence[Union[Code, CodedConcept]]] = None
    ) -> None:
        item = CodeContentItem(
            name=name,
            value=value,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        content = ContentSequence()
        if procedure_datetime is not None:
            procedure_datetime_item = DateTimeContentItem(
                name=codes.DCM.ProcedureDatetime,
                value=procedure_datetime,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(procedure_datetime_item)
        if clinical_reports is not None:
            if not isinstance(clinical_reports, (list, tuple, set)):
                raise TypeError(
                    'Argument "clinical_reports" must be a sequence.'
                )
            for clinical_report in clinical_reports:
                if not isinstance(clinical_report, CompositeContentItem):
                    raise TypeError(
                        'Items of argument "clinical_reports" must ' +
                        'have type CompositeContentItem.'
                    )
                content.append(clinical_report)
        if clinical_reports_text is not None:
            if not isinstance(clinical_reports_text, (list, tuple, set)):
                raise TypeError(
                    'Argument "clinical_reports_text" must be a sequence.'
                )
            for clinical_report in clinical_reports_text:
                if not isinstance(clinical_report, str):
                    raise TypeError(
                        'Items of argument "clinical_reports" must ' +
                        'have type str.'
                    )
                clinical_report_item = TextContentItem(
                    name=codes.SCT.ClinicalReport,
                    value=clinical_report,
                    relationship_type=RelationshipTypeValues.HAS_PROPERTIES
                )
                content.append(clinical_report_item)
        if service_delivery_location is not None:
            service_delivery_location_item = TextContentItem(
                name=codes.DCM.ServiceDeliveryLocation,
                value=service_delivery_location,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(service_delivery_location_item)
        if service_performer_person is not None:
            service_performer_person_item = PnameContentItem(
                name=codes.DCM.ServicePerformer,
                value=service_performer_person,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(service_performer_person_item)
        if service_performer_organisation is not None:
            service_performer_organisation_item = TextContentItem(
                name=codes.DCM.ServicePerformer,
                value=service_performer_organisation,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(service_performer_organisation_item)
        if comment is not None:
            comment_item = TextContentItem(
                name=codes.DCM.Comment,
                value=comment,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(comment_item)
        if procedure_results is not None:
            if not isinstance(procedure_results, (list, tuple, set)):
                raise TypeError(
                    'Argument "procedure_results" must be a sequence.'
                )
            for procedure_result in procedure_results:
                if not isinstance(procedure_result, (Code, CodedConcept)):
                    raise TypeError(
                        'Items of argument "clinical_reports" must have ' +
                        'type Code or CodedConcept.'
                    )
                procedure_result_item = CodeContentItem(
                    name=codes.DCM.ProcedureResult,
                    value=procedure_result,
                    relationship_type=RelationshipTypeValues.HAS_PROPERTIES
                )
                content.append(procedure_result_item)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])


class PastSurgicalHistory(Template):
    """:ln:`10167-5`
    Past Surgical History
    """

    def __init__(
        self,
        histories: Optional[Sequence[str]] = None,
        procedure_properties: Optional[
            Sequence[ProcedureProperties]] = None
    ) -> None:
        super().__init__()
        item = ContainerContentItem(
            name=CodedConcept(
                value='10167-5',
                meaning='Past Surgical History',
                scheme_designator='LN'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if histories is not None:
            if not isinstance(histories, (list, tuple, set)):
                raise TypeError(
                    'Argument "histories" must be a sequence.'
                )
            for history in histories:
                if not isinstance(history, str):
                    raise TypeError(
                        'Items of argument "social_histories" must ' +
                        'have type str.'
                    )
                history_item = TextContentItem(
                    name=codes.LN.History,
                    value=history,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(history_item)
        if procedure_properties is not None:
            if not isinstance(procedure_properties, (list, tuple, set)):
                raise TypeError(
                    'Argument "procedure_properties" must be a sequence.'
                )
            for procedure_property in procedure_properties:
                if not isinstance(procedure_property, ProcedureProperties):
                    raise TypeError(
                        'Items of argument "procedure_properties" must ' +
                        'have type ProcedureProperties.'
                    )
                content.extend(procedure_property)
        if len(content) > 0:
            item.ContentSequence = content
        self.append(item)


class RelevantDiagnosticTestsAndOrLaboratoryData(Template):
    """:ln:`30954-2`
    Relevant Diagnostic Tests and/or Laboratory Data
    """

    def __init__(
        self,
        histories: Optional[Sequence[str]] = None,
        procedure_properties: Optional[
            Sequence[ProcedureProperties]] = None,
        cholesterol_in_HDL: Optional[float] = None,
        cholesterol_in_LDL: Optional[float] = None
    ) -> None:
        super().__init__()
        item = ContainerContentItem(
            name=codes.LN.RelevantDiagnosticTestsAndOrLaboratoryData,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if histories is not None:
            if not isinstance(histories, (list, tuple, set)):
                raise TypeError(
                    'Argument "histories" must be a sequence.'
                )
            for history in histories:
                if not isinstance(history, str):
                    raise TypeError(
                        'Items of argument "social_histories" must ' +
                        'have type str.'
                    )
                history_item = TextContentItem(
                    name=codes.LN.History,
                    value=history,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(history_item)
        if procedure_properties is not None:
            if not isinstance(procedure_properties, (list, tuple, set)):
                raise TypeError(
                    'Argument "procedure_properties" must be a sequence.'
                )
            for procedure_property in procedure_properties:
                if not isinstance(procedure_property, ProcedureProperties):
                    raise TypeError(
                        'Items of argument "procedure_properties" must ' +
                        'have type ProcedureProperties.'
                    )
                content.extend(procedure_property)
        if cholesterol_in_HDL is not None:
            cholesterol_in_HDL_item = NumContentItem(
                name=CodedConcept(
                    value='2086-7',
                    meaning='Cholesterol in HDL',
                    scheme_designator='LN'
                ),
                value=cholesterol_in_HDL,
                unit=MILLIGRAM_PER_DECILITER,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(cholesterol_in_HDL_item)
        if cholesterol_in_LDL is not None:
            cholesterol_in_LDL_item = NumContentItem(
                name=CodedConcept(
                    value='2089-1',
                    meaning='Cholesterol in LDL',
                    scheme_designator='LN'
                ),
                value=cholesterol_in_LDL,
                unit=MILLIGRAM_PER_DECILITER,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(cholesterol_in_LDL_item)
        if len(content) > 0:
            item.ContentSequence = content
        self.append(item)


class MedicationTypeText(TextContentItem):
    """:dcm:`DT 111516 <part16/chapter_D.html#DCM_111516>`
    Medication Type Text
    """

    def __init__(
        self,
        value: str,
        status: Optional[Union[Code, CodedConcept]]
    ) -> None:
        super().__init__(
            name=codes.DCM.MedicationType,
            value=value,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if status is not None:
            status_item = CodeContentItem(
                name=CodedConcept(
                    value='33999-4',
                    meaning='Status',
                    scheme_designator='LN'
                ),
                value=status,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(status_item)
        if len(content) > 0:
            self.ContentSequence = content


class MedicationTypeCode(CodeContentItem):
    """:dcm:`DT 111516 <part16/chapter_D.html#DCM_111516>`
    Medication Type Code
    """

    def __init__(
        self,
        value: Union[Code, CodedConcept],
        dosage: Optional[float],
        status: Optional[Union[Code, CodedConcept]]
    ) -> None:
        super().__init__(
            name=codes.DCM.MedicationType,
            value=value,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if dosage is not None:
            dosage_item = NumContentItem(
                name=codes.SCT.Dosage,
                value=dosage,
                unit=codes.UCUM.NoUnits,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(dosage_item)
        if status is not None:
            status_item = CodeContentItem(
                name=CodedConcept(
                    value='33999-4',
                    meaning='Status',
                    scheme_designator='LN'
                ),
                value=status,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(status_item)
        if len(content) > 0:
            self.ContentSequence = content


class HistoryOfMedicationUse(Template):
    """:ln:`10160-0`
    History of Medication Use
    """

    def __init__(
        self,
        medication_types_text: Optional[
            Sequence[MedicationTypeText]] = None,
        medication_types_code: Optional[
            Sequence[MedicationTypeCode]] = None,
    ) -> None:
        super().__init__()
        item = ContainerContentItem(
            name=CodedConcept(
                value='10160-0',
                meaning='History of Medication Use',
                scheme_designator='LN'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if medication_types_text is not None:
            if not isinstance(medication_types_text, (list, tuple, set)):
                raise TypeError(
                    'Argument "medication_types_text" must be a sequence.'
                )
            for medication_type_text in medication_types_text:
                if not isinstance(medication_type_text, MedicationTypeText):
                    raise TypeError(
                        'Items of argument "medication_types_text" must ' +
                        'have type MedicationTypeText.'
                    )
                content.append(medication_type_text)
        if medication_types_code is not None:
            if not isinstance(medication_types_code, (list, tuple, set)):
                raise TypeError(
                    'Argument "medication_types_code" must be a sequence.'
                )
            for medication_type_code in medication_types_code:
                if not isinstance(medication_type_code, MedicationTypeCode):
                    raise TypeError(
                        'Items of argument "medication_types_code" must ' +
                        'have type MedicationTypeCode.'
                    )
                content.append(medication_type_code)
        if len(content) > 0:
            item.ContentSequence = content
        self.append(item)


class FamilyHistoryOfClinicalFinding(CodeContentItem):
    """:sct:`416471007`
    Family history of clinical finding
    """

    def __init__(
        self,
        value: Union[Code, CodedConcept],
        subject_relationship: Union[Code, CodedConcept]
    ) -> None:
        super().__init__(
            name=codes.SCT.FamilyHistoryOfClinicalFinding,
            value=value,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        content = ContentSequence()
        subject_relationship_item = CodeContentItem(
            name=CodedConcept(
                value='408732007',
                meaning='Subject relationship',
                scheme_designator='SCT'
            ),
            value=subject_relationship,
            relationship_type=RelationshipTypeValues.HAS_PROPERTIES
        )
        content.append(subject_relationship_item)
        if len(content) > 0:
            self.ContentSequence = content


class HistoryOfFamilyMemberDiseases(Template):
    """:ln:`10157-6`
    History of Family Member Diseases
    """

    def __init__(
        self,
        histories: Optional[Sequence[str]] = None,
        family_histories_of_clinical_findings: Optional[
            Sequence[FamilyHistoryOfClinicalFinding]] = None
    ) -> None:
        super().__init__()
        item = ContainerContentItem(
            name=CodedConcept(
                value='10157-6',
                meaning='History of Family Member Diseases',
                scheme_designator='LN'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if histories is not None:
            if not isinstance(histories, (list, tuple, set)):
                raise TypeError(
                    'Argument "histories" must be a sequence.'
                )
            for history in histories:
                if not isinstance(history, str):
                    raise TypeError(
                        'Items of argument "social_histories" must ' +
                        'have type str.'
                    )
                history_item = TextContentItem(
                    name=codes.LN.History,
                    value=history,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                content.append(history_item)
        if family_histories_of_clinical_findings is not None:
            if not isinstance(family_histories_of_clinical_findings, (
                list, tuple, set
            )):
                raise TypeError(
                    'Argument "family_histories_of_clinical_findings" ' +
                    'must be a sequence.'
                )
            for family_history_of_clinical_finding in \
                    family_histories_of_clinical_findings:
                if not isinstance(family_history_of_clinical_finding,
                                  FamilyHistoryOfClinicalFinding):
                    raise TypeError(
                        'Items of argument ' +
                        '"family_histories_of_clinical_findings" ' +
                        'must have type FamilyHistoryOfClinicalFinding.'
                    )
                content.append(family_history_of_clinical_finding)
        if len(content) > 0:
            item.ContentSequence = content
        self.append(item)


class MedicalDeviceUse(Template):
    """:dcm:`CID 3831 <part16/sect_TID_3831.html>`
    Medical Device Use
    """

    def __init__(
        self,
        datetime_started: Optional[
            Union[str, datetime, DT]] = None,
        datetime_ended: Optional[
            Union[str, datetime, DT]] = None,
        status: Optional[Union[Code, CodedConcept]] = None,
        comment: Optional[str] = None
    ) -> None:
        super().__init__()
        item = ContainerContentItem(
            name=CodedConcept(
                value='46264-8',
                meaning='Medical Device use',
                scheme_designator='LN'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if datetime_started is not None:
            datetime_started_item = DateTimeContentItem(
                name=codes.DCM.DatetimeStarted,
                value=datetime_started,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(datetime_started_item)
        if datetime_ended is not None:
            datetime_ended_item = DateTimeContentItem(
                name=codes.DCM.DatetimeEnded,
                value=datetime_ended,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(datetime_ended_item)
        if status is not None:
            status_item = CodeContentItem(
                name=CodedConcept(
                    value='33999-4',
                    meaning='Status',
                    scheme_designator='LN'
                ),
                value=status,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(status_item)
        if comment is not None:
            comment_item = TextContentItem(
                name=codes.DCM.Comment,
                value=comment,
                relationship_type=RelationshipTypeValues.HAS_PROPERTIES
            )
            content.append(comment_item)
        if len(content) > 0:
            item.ContentSequence = content
        self.append(item)


class HistoryOfMedicalDeviceUse(Template):
    """:ln:`46264-8`
    History of medical device use
    """

    def __init__(
        self,
        history: Optional[str] = None,
        medical_device_uses: Optional[
            Sequence[MedicalDeviceUse]] = None,
    ) -> None:
        super().__init__()
        item = ContainerContentItem(
            name=CodedConcept(
                value='46264-8',
                meaning='History of medical device use',
                scheme_designator='LN'
            ),
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if history is not None:
            history_item = TextContentItem(
                name=codes.LN.History,
                value=history,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(history_item)
        if medical_device_uses is not None:
            if not isinstance(medical_device_uses, (list, tuple, set)):
                raise TypeError(
                    'Argument "medical_device_uses" must be a sequence.'
                )
            for medical_device_use in medical_device_uses:
                if not isinstance(medical_device_use, MedicalDeviceUse):
                    raise TypeError(
                        'Items of argument "medical_device_uses" must have ' +
                        'type MedicalDeviceUse.'
                    )
                content.extend(medical_device_use)
        if len(content) > 0:
            item.ContentSequence = content
        self.append(item)


class CardiovascularPatientHistory(Template):
    """:dcm:`TID 3802 <part16/sect_TID_3802.html>`
    Cardiovascular Patient History
    """

    def __init__(
        self,
        history: Optional[str] = None,
        problem_list: Optional[ProblemList] = None,
        social_history: Optional[SocialHistory] = None,
        past_surgical_history: Optional[PastSurgicalHistory] = None,
        relevant_diagnostic_tests_and_or_laboratory_data: Optional[
            RelevantDiagnosticTestsAndOrLaboratoryData] = None,
        history_of_medication_use: Optional[HistoryOfMedicationUse] = None,
        history_of_family_member_diseases: Optional[
            HistoryOfFamilyMemberDiseases] = None,
        history_of_medical_device_use: Optional[
            HistoryOfMedicalDeviceUse] = None
    ) -> None:
        item = ContainerContentItem(
            name=codes.LN.History,
            template_id='3802',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        content = ContentSequence()
        if history is not None:
            history_item = TextContentItem(
                name=codes.LN.History,
                value=history,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            content.append(history_item)
        if problem_list is not None:
            if not isinstance(problem_list, ProblemList):
                raise TypeError(
                    'Argument "problem_list" must be a ProblemList.'
                )
            content.extend(problem_list)
        if social_history is not None:
            if not isinstance(social_history, SocialHistory):
                raise TypeError(
                    'Argument "social_history" must be a SocialHistory.'
                )
            content.extend(social_history)
        if past_surgical_history is not None:
            if not isinstance(past_surgical_history, PastSurgicalHistory):
                raise TypeError(
                    'Argument "past_surgical_history" must be a ' +
                    'PastSurgicalHistory.'
                )
            content.extend(past_surgical_history)
        if relevant_diagnostic_tests_and_or_laboratory_data is not None:
            if not isinstance(relevant_diagnostic_tests_and_or_laboratory_data,
                              RelevantDiagnosticTestsAndOrLaboratoryData):
                raise TypeError(
                    'Argument ' +
                    '"relevant_diagnostic_tests_and_or_laboratory_data" ' +
                    'must be a RelevantDiagnosticTestsAndOrLaboratoryData.'
                )
            content.extend(relevant_diagnostic_tests_and_or_laboratory_data)
        if history_of_medication_use is not None:
            if not isinstance(history_of_medication_use,
                              HistoryOfMedicationUse):
                raise TypeError(
                    'Argument "history_of_medication_use" must be ' +
                    'a HistoryOfMedicationUse.'
                )
            content.extend(history_of_medication_use)
        if history_of_family_member_diseases is not None:
            if not isinstance(history_of_family_member_diseases,
                              HistoryOfFamilyMemberDiseases):
                raise TypeError(
                    'Argument "history_of_family_member_diseases" must ' +
                    'be a HistoryOfFamilyMemberDiseases.'
                )
            content.extend(history_of_family_member_diseases)
        if history_of_medical_device_use is not None:
            if not isinstance(history_of_medical_device_use,
                              HistoryOfMedicalDeviceUse):
                raise TypeError(
                    'Argument "history_of_medical_device_use" must be a ' +
                    'HistoryOfMedicalDeviceUse.'
                )
            content.extend(history_of_medical_device_use)
        if len(content) > 0:
            item.ContentSequence = content
        super().__init__([item])
