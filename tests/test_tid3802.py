import unittest
from datetime import datetime

from pydicom.sr.codedict import codes

from highdicom.sr.enum import RelationshipTypeValues
from highdicom.sr.templates import (
    CardiovascularPatientHistory,
    FamilyHistoryOfClinicalFinding,
    HistoryOfFamilyMemberDiseases,
    HistoryOfMedicalDeviceUse,
    HistoryOfMedicationUse,
    MedicalDeviceUse,
    MedicationTypeCode,
    MedicationTypeText,
    PastSurgicalHistory,
    ProblemList,
    ProblemProperties,
    ProcedureProperties,
    RelevantDiagnosticTestsAndOrLaboratoryData,
    SocialHistory,
    Therapy
)
from highdicom.sr.value_types import (
    Code,
    CodedConcept,
    CompositeContentItem
)


class TestTherapy(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.name_code = CodedConcept('T', 'Beta-blocker', '99')
        self.status = Code('active', 'Active', '99')

    def test_construction(self):
        item = Therapy(self.name_code)
        self.assertEqual(item.ValueType, 'CODE')
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_construction_with_status(self):
        item = Therapy(self.name_code, status=self.status)
        self.assertTrue(any(
            ci.ConceptNameCodeSequence[0].CodeValue == '33999-4'
            for ci in item.ContentSequence))

    def test_type_guard(self):
        with self.assertRaises(TypeError):
            Therapy('bad')

    def test_from_dataset(self):
        item = Therapy(self.name_code)
        rebuilt = Therapy.from_dataset(item)
        self.assertIsInstance(rebuilt, Therapy)


class TestProblemProperties(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.concern = Code('C', 'Concern', '99')
        self.health = Code('healthy', 'Healthy', '99')
        self.therapy = Therapy(CodedConcept('Drug', 'Statin', '99'))
        self.note = 'Patient recovering well'
        self.dt1 = datetime.now()
        self.dt2 = datetime.now()

    def test_construction(self):
        tpl = ProblemProperties(self.concern)
        self.assertIn(
            self.concern, tpl[0].ContentSequence[0].ConceptCodeSequence)

    def test_construction_with_optionals(self):
        tpl = ProblemProperties(
            self.concern,
            datetime_concern_noted=self.dt1,
            datetime_concern_resolved=self.dt2,
            health_status=self.health,
            therapies=[self.therapy],
            comment=self.note,
        )
        seq = tpl[0].ContentSequence
        self.assertTrue(any(ci.ValueType == 'DATETIME' for ci in seq))
        self.assertIn(self.therapy, seq)
        self.assertTrue(
            any(ci.ValueType == 'TEXT' and ci.TextValue == self.note
                for ci in seq))

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            ProblemProperties('bad')
        with self.assertRaises(TypeError):
            ProblemProperties(self.concern, therapies='bad')
        with self.assertRaises(TypeError):
            ProblemProperties(self.concern, therapies=[123])


class TestProblemList(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.concern_strings = ['Chest pain', 'Hypertension']
        self.pp_a = ProblemProperties(Code('C', 'Concern', '99'))
        self.pp_b = ProblemProperties(Code('C', 'Concern', '99'))

    def test_construction(self):
        tpl = ProblemList()
        self.assertFalse(hasattr(tpl[0], 'ContentSequence'))

    def test_concern_types_only(self):
        tpl = ProblemList(concern_types=self.concern_strings)
        seq = tpl[0].ContentSequence
        self.assertEqual(len(seq), len(self.concern_strings))
        self.assertTrue(all(ci.ValueType == 'TEXT' for ci in seq))

    def test_problem_properties(self):
        tpl = ProblemList(
            cardiac_patient_risk_factors=[self.pp_a],
            history_of_diabetes_mellitus=self.pp_b,
        )
        seq = tpl[0].ContentSequence
        self.assertIn(self.pp_a[0], seq)
        self.assertIn(self.pp_b[0], seq)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            ProblemList(concern_types='bad')
        with self.assertRaises(TypeError):
            ProblemList(concern_types=[123])
        with self.assertRaises(TypeError):
            ProblemList(cardiac_patient_risk_factors='bad')
        with self.assertRaises(TypeError):
            ProblemList(cardiac_patient_risk_factors=[123])
        with self.assertRaises(TypeError):
            ProblemList(history_of_diabetes_mellitus='bad')


class TestSocialHistory(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.single_text = 'Lives alone'
        self.multi_texts = ['Never smoker', 'Works night shifts']
        self.smoke_code = Code('smk', 'Smoker', '99')
        self.drug_code = CodedConcept('drug', 'Drug misuse', '99')

    def test_construction(self):
        tpl = SocialHistory()
        self.assertFalse(hasattr(tpl[0], 'ContentSequence'))

    def test_single_text(self):
        tpl = SocialHistory(social_history=self.single_text)
        itm = tpl[0].ContentSequence[0]
        self.assertEqual(itm.TextValue, self.single_text)

    def test_multiple_texts(self):
        tpl = SocialHistory(social_histories=self.multi_texts)
        seq = tpl[0].ContentSequence
        self.assertEqual(len(seq), len(self.multi_texts))

    def test_codes(self):
        tpl = SocialHistory(
            tobacco_smoking_behavior=self.smoke_code,
            drug_misuse_behavior=self.drug_code,
        )
        seq = tpl[0].ContentSequence
        self.assertTrue(any(
            ci.ConceptNameCodeSequence[0] == codes.SCT.TobaccoSmokingBehavior
            for ci in seq))
        self.assertTrue(
            any(ci.ConceptNameCodeSequence[0] == codes.SCT.DrugMisuseBehavior
                for ci in seq))

    def test_construction_with_optionals(self):
        tpl = SocialHistory(
            social_history=self.single_text,
            social_histories=self.multi_texts,
            tobacco_smoking_behavior=self.smoke_code,
            drug_misuse_behavior=self.drug_code,
        )
        seq = tpl[0].ContentSequence
        self.assertGreaterEqual(len(seq), 4)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            SocialHistory(social_histories='bad')
        with self.assertRaises(TypeError):
            SocialHistory(social_histories=[123])


class TestProcedureProperties(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.name = CodedConcept('ABC', 'Angioplasty', '99')
        self.value = Code('done', 'Done', '99')
        self.dt = datetime.now()
        self.rep_item = CompositeContentItem(
            Code('report', 'reported', '99'),
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2',
            referenced_sop_instance_uid='1.2.3.4.5.6.7.8.9.10',
            relationship_type=RelationshipTypeValues.HAS_PROPERTIES
        )
        self.perf_person = 'Doe^John'
        self.perf_org = 'Charité Berlin'
        self.comment = 'Procedure uneventful'
        self.result_code = CodedConcept('R', 'Normal', '99')

    def test_construction(self):
        item = ProcedureProperties(self.name, self.value)[0]
        self.assertEqual(item.ConceptNameCodeSequence[0], self.name)
        self.assertEqual(item.ConceptCodeSequence[0], self.value)
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_construction_with_optionals(self):
        item = ProcedureProperties(
            self.name,
            self.value,
            procedure_datetime=self.dt,
            clinical_reports=[self.rep_item],
            clinical_reports_text=['Rep A', 'Rep B'],
            service_delivery_location='Ward 12',
            service_performer_person=self.perf_person,
            service_performer_organisation=self.perf_org,
            comment=self.comment,
            procedure_results=[self.result_code],
        )[0]
        seq = item.ContentSequence
        self.assertTrue(any(ci.ValueType == 'DATETIME' for ci in seq))
        self.assertIn(self.rep_item, seq)
        self.assertTrue(any(ci.TextValue == 'Rep A' for ci in seq
                            if getattr(ci, "TextValue", None) is not None))
        self.assertTrue(any(ci.TextValue == self.perf_org for ci in seq
                            if getattr(ci, "TextValue", None) is not None))
        self.assertTrue(any(ci.TextValue == self.comment for ci in seq
                            if getattr(ci, "TextValue", None) is not None))
        self.assertTrue(
            any(ci.ConceptCodeSequence[0] == self.result_code for ci in seq
                if getattr(ci, "ConceptCodeSequence", None) is not None))

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            ProcedureProperties('bad', self.value)
        with self.assertRaises(TypeError):
            ProcedureProperties(self.name, 'bad')
        with self.assertRaises(TypeError):
            ProcedureProperties(self.name, self.value, clinical_reports='x')
        with self.assertRaises(TypeError):
            ProcedureProperties(self.name, self.value, clinical_reports=[1])
        with self.assertRaises(TypeError):
            ProcedureProperties(self.name, self.value,
                                clinical_reports_text=123)
        with self.assertRaises(TypeError):
            ProcedureProperties(self.name, self.value,
                                clinical_reports_text=[1])
        with self.assertRaises(TypeError):
            ProcedureProperties(self.name, self.value, procedure_results='x')
        with self.assertRaises(TypeError):
            ProcedureProperties(self.name, self.value, procedure_results=[123])


class TestPastSurgicalHistory(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.hist_strings = ['Appendectomy', 'CABG']
        self.pp = ProcedureProperties(
            CodedConcept('ABC', 'Angioplasty', '99'),
            Code('done', 'Done', '99')
        )

    def test_construction(self):
        tpl = PastSurgicalHistory()
        self.assertFalse(hasattr(tpl[0], 'ContentSequence'))

    def test_only_histories(self):
        tpl = PastSurgicalHistory(histories=self.hist_strings)
        self.assertEqual(len(tpl[0].ContentSequence), len(self.hist_strings))

    def test_only_properties(self):
        tpl = PastSurgicalHistory(procedure_properties=[self.pp])
        self.assertIn(self.pp[0], tpl[0].ContentSequence)

    def test_construction_with_optionals(self):
        tpl = PastSurgicalHistory(
            histories=self.hist_strings, procedure_properties=[self.pp])
        seq = tpl[0].ContentSequence
        self.assertGreaterEqual(len(seq), len(self.hist_strings) + 1)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            PastSurgicalHistory(histories='bad')
        with self.assertRaises(TypeError):
            PastSurgicalHistory(histories=[123])
        with self.assertRaises(TypeError):
            PastSurgicalHistory(procedure_properties='bad')
        with self.assertRaises(TypeError):
            PastSurgicalHistory(procedure_properties=[123])


class TestRelevantDiagnosticTestsAndOrLaboratoryData(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.histories = ['Blood test 2023-06-12']
        self.pp = ProcedureProperties(
            CodedConcept('ABC', 'Angioplasty', '99'),
            Code('done', 'Done', '99')
        )
        self.hdl = 55.0
        self.ldl = 130.0

    def test_construction(self):
        tpl = RelevantDiagnosticTestsAndOrLaboratoryData()
        self.assertFalse(hasattr(tpl[0], 'ContentSequence'))

    def test_construction_with_optionals(self):
        tpl = RelevantDiagnosticTestsAndOrLaboratoryData(
            histories=self.histories,
            procedure_properties=[self.pp],
            cholesterol_in_HDL=self.hdl,
            cholesterol_in_LDL=self.ldl,
        )
        seq = tpl[0].ContentSequence
        self.assertEqual(sum(ci.ValueType == 'TEXT' for ci in seq), 1)
        self.assertIn(self.pp[0], seq)
        self.assertTrue(any(ci.ValueType == 'NUM' and
                            ci.ConceptNameCodeSequence[0].CodeValue == '2086-7'
                            for ci in seq))
        self.assertTrue(any(ci.ValueType == 'NUM' and
                            ci.ConceptNameCodeSequence[0].CodeValue == '2089-1'
                            for ci in seq))

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            RelevantDiagnosticTestsAndOrLaboratoryData(histories='bad')
        with self.assertRaises(TypeError):
            RelevantDiagnosticTestsAndOrLaboratoryData(histories=[123])
        with self.assertRaises(TypeError):
            RelevantDiagnosticTestsAndOrLaboratoryData(procedure_properties='x')
        with self.assertRaises(TypeError):
            RelevantDiagnosticTestsAndOrLaboratoryData(procedure_properties=[1])


class TestMedicationTypeText(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.status = Code('active', 'Active', '99')

    def test_construction(self):
        itm = MedicationTypeText('Aspirin', self.status)
        self.assertEqual(itm.ValueType, 'TEXT')
        self.assertTrue(any(
            ci.ConceptNameCodeSequence[0].CodeValue == '33999-4'
            for ci in itm.ContentSequence))

    def test_from_dataset(self):
        itm = MedicationTypeText('Ibuprofen', self.status)
        rebuilt = MedicationTypeText.from_dataset(itm)
        self.assertIsInstance(rebuilt, MedicationTypeText)


class TestMedicationTypeCode(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.code_val = CodedConcept('RX', 'Metformin', '99')
        self.status = Code('stopped', 'Stopped', '99')

    def test_construction(self):
        itm = MedicationTypeCode(
            self.code_val,
            dosage=None,
            status=self.status
        )
        self.assertEqual(itm.ValueType, 'CODE')
        self.assertFalse(
            any(ci.ValueType == 'NUM'
                for ci in getattr(itm, 'ContentSequence', [])))

    def test_with_dosage(self):
        itm = MedicationTypeCode(
            self.code_val, dosage=500.0, status=self.status)
        self.assertTrue(
            any(ci.ValueType == 'NUM' for ci in itm.ContentSequence))

    def test_from_dataset(self):
        itm = MedicationTypeCode(
            self.code_val, dosage=250.0, status=self.status)
        rebuilt = MedicationTypeCode.from_dataset(itm)
        self.assertIsInstance(rebuilt, MedicationTypeCode)


class TestHistoryOfMedicationUse(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.txt_item = MedicationTypeText(
            'Aspirin', Code('active', 'Active', '99'))
        self.code_item = MedicationTypeCode(
            CodedConcept('RX', 'Metformin', '99'),
            dosage=500.0,
            status=Code('active', 'Active', '99'))

    def test_text_only(self):
        tpl = HistoryOfMedicationUse(medication_types_text=[self.txt_item])
        self.assertIn(self.txt_item, tpl[0].ContentSequence)

    def test_code_only(self):
        tpl = HistoryOfMedicationUse(medication_types_code=[self.code_item])
        self.assertIn(self.code_item, tpl[0].ContentSequence)

    def test_construction_with_optionals(self):
        tpl = HistoryOfMedicationUse(
            medication_types_text=[self.txt_item],
            medication_types_code=[self.code_item],
        )
        seq = tpl[0].ContentSequence
        self.assertIn(self.txt_item, seq)
        self.assertIn(self.code_item, seq)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            HistoryOfMedicationUse(medication_types_text='bad')
        with self.assertRaises(TypeError):
            HistoryOfMedicationUse(medication_types_code='bad')
        with self.assertRaises(TypeError):
            HistoryOfMedicationUse(medication_types_text=[123])


class TestFamilyHistoryOfClinicalFinding(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.val = CodedConcept('FH', 'Diabetes', '99')
        self.rel = Code('mother', 'Mother', '99')

    def test_construction(self):
        itm = FamilyHistoryOfClinicalFinding(self.val, self.rel)
        self.assertEqual(itm.ValueType, 'CODE')
        self.assertTrue(any(
            ci.ConceptNameCodeSequence[0].CodeValue == '408732007'
            for ci in itm.ContentSequence))

    def test_from_dataset(self):
        itm = FamilyHistoryOfClinicalFinding(self.val, self.rel)
        rebuilt = FamilyHistoryOfClinicalFinding.from_dataset(itm)
        self.assertIsInstance(rebuilt, FamilyHistoryOfClinicalFinding)


class TestHistoryOfFamilyMemberDiseases(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.hist_strings = ['Father: myocardial infarction at 55']
        self.val = CodedConcept('FH', 'Diabetes', '99')
        self.rel = Code('father', 'Father', '99')
        self.fh_item = FamilyHistoryOfClinicalFinding(self.val, self.rel)

    def test_construction(self):
        tpl = HistoryOfFamilyMemberDiseases()
        self.assertFalse(hasattr(tpl[0], 'ContentSequence'))

    def test_strings_only(self):
        tpl = HistoryOfFamilyMemberDiseases(histories=self.hist_strings)
        self.assertEqual(len(tpl[0].ContentSequence), len(self.hist_strings))

    def test_findings_only(self):
        tpl = HistoryOfFamilyMemberDiseases(
            family_histories_of_clinical_findings=[self.fh_item]
        )
        self.assertIn(self.fh_item, tpl[0].ContentSequence)

    def test_both(self):
        tpl = HistoryOfFamilyMemberDiseases(
            histories=self.hist_strings,
            family_histories_of_clinical_findings=[self.fh_item],
        )
        seq = tpl[0].ContentSequence
        self.assertIn(self.fh_item, seq)
        self.assertEqual(sum(ci.ValueType == 'TEXT' for ci in seq), 1)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            HistoryOfFamilyMemberDiseases(histories='bad')
        with self.assertRaises(TypeError):
            HistoryOfFamilyMemberDiseases(histories=[123])
        with self.assertRaises(TypeError):
            HistoryOfFamilyMemberDiseases(
                family_histories_of_clinical_findings='x')
        with self.assertRaises(TypeError):
            HistoryOfFamilyMemberDiseases(
                family_histories_of_clinical_findings=[123])


class TestMedicalDeviceUse(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.start = datetime.now()
        self.end = datetime.now()
        self.status = Code('active', 'Active', '99')
        self.comment = 'No issues'

    def test_construction(self):
        tpl = MedicalDeviceUse()
        self.assertFalse(hasattr(tpl[0], 'ContentSequence'))

    def test_construction_with_optionals(self):
        tpl = MedicalDeviceUse(
            datetime_started=self.start,
            datetime_ended=self.end,
            status=self.status,
            comment=self.comment,
        )
        seq = tpl[0].ContentSequence
        self.assertTrue(any(ci.ValueType == 'DATETIME' for ci in seq))
        self.assertTrue(any(ci.ValueType == 'CODE' for ci in seq))
        self.assertTrue(
            any(ci.ValueType == 'TEXT' and
                ci.TextValue == self.comment for ci in seq))


class TestHistoryOfMedicalDeviceUse(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.note = 'Pacemaker implanted 2018'
        self.device_use = MedicalDeviceUse(
            status=Code('active', 'Active', '99'))

    def test_note_only(self):
        tpl = HistoryOfMedicalDeviceUse(history=self.note)
        self.assertEqual(tpl[0].ContentSequence[0].TextValue, self.note)

    def test_devices_only(self):
        tpl = HistoryOfMedicalDeviceUse(medical_device_uses=[self.device_use])
        self.assertIn(self.device_use[0], tpl[0].ContentSequence)

    def test_construction_with_optionals(self):
        tpl = HistoryOfMedicalDeviceUse(
            history=self.note,
            medical_device_uses=[self.device_use],
        )
        self.assertGreaterEqual(len(tpl[0].ContentSequence), 2)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            HistoryOfMedicalDeviceUse(medical_device_uses='x')
        with self.assertRaises(TypeError):
            HistoryOfMedicalDeviceUse(medical_device_uses=[123])


class TestCardiovascularPatientHistory(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.note = 'Patient reports occasional chest pain'
        self.problem_list = ProblemList()
        self.social_history = SocialHistory()
        self.past_surg = PastSurgicalHistory()
        self.labs = RelevantDiagnosticTestsAndOrLaboratoryData()
        self.med_use = HistoryOfMedicationUse(
            medication_types_text=[MedicationTypeText(
                'Aspirin',
                Code('active', 'Active', '99'))]
        )
        self.family = HistoryOfFamilyMemberDiseases()
        self.devices = HistoryOfMedicalDeviceUse(
            history='Pacemaker implanted 2018'
        )

    def test_history_only(self):
        tpl = CardiovascularPatientHistory(history=self.note)
        seq = tpl[0].ContentSequence
        self.assertEqual(seq[0].TextValue, self.note)

    def test_all_sections(self):
        tpl = CardiovascularPatientHistory(
            history=self.note,
            problem_list=self.problem_list,
            social_history=self.social_history,
            past_surgical_history=self.past_surg,
            relevant_diagnostic_tests_and_or_laboratory_data=self.labs,
            history_of_medication_use=self.med_use,
            history_of_family_member_diseases=self.family,
            history_of_medical_device_use=self.devices,
        )
        seq = tpl[0].ContentSequence
        self.assertTrue(any(ci.TextValue == self.note for ci in seq))
        self.assertIn(next(iter(self.problem_list)), seq)
        self.assertIn(next(iter(self.social_history)), seq)
        self.assertIn(next(iter(self.past_surg)), seq)
        self.assertIn(next(iter(self.labs)), seq)
        self.assertIn(next(iter(self.med_use)), seq)
        self.assertIn(next(iter(self.family)), seq)
        self.assertIn(next(iter(self.devices)), seq)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            CardiovascularPatientHistory(problem_list='bad')
        with self.assertRaises(TypeError):
            CardiovascularPatientHistory(social_history='bad')
        with self.assertRaises(TypeError):
            CardiovascularPatientHistory(past_surgical_history='bad')
        with self.assertRaises(TypeError):
            CardiovascularPatientHistory(
                relevant_diagnostic_tests_and_or_laboratory_data='bad')
        with self.assertRaises(TypeError):
            CardiovascularPatientHistory(history_of_medication_use='bad')
        with self.assertRaises(TypeError):
            CardiovascularPatientHistory(
                history_of_family_member_diseases='bad')
        with self.assertRaises(TypeError):
            CardiovascularPatientHistory(history_of_medical_device_use='bad')
