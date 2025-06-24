import unittest

from pydicom.sr.codedict import codes
from highdicom.sr import (
    LanguageOfContentItemAndDescendants,
    PersonObserverIdentifyingAttributes
)
from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import RelationshipTypeValues
from highdicom.sr.value_types import Code, ImageContentItem
from highdicom.sr.templates import (
    BasicDiagnosticImagingReport,
    DiagnosticImagingReportHeading,
    EquivalentMeaningsOfConceptNameText,
    EquivalentMeaningsOfConceptNameCode,
    ReportNarrativeCode,
    ReportNarrativeText,
    LanguageOfValue,
    ObservationContext,
    ObserverContext
)


class TestEquivalentMeaningsOfConceptNameText(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.lang = LanguageOfValue(
            language=Code('en-US', 'English-US', 'RFC5646')
        )

    def test_construction(self):
        item = EquivalentMeaningsOfConceptNameText('alt term')
        self.assertEqual(item.ValueType, 'TEXT')
        self.assertEqual(item.RelationshipType,
                         RelationshipTypeValues.HAS_CONCEPT_MOD.value)
        self.assertEqual(
            item.ConceptNameCodeSequence[0],
            codes.DCM.EquivalentMeaningOfConceptName)
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_construction_with_language(self):
        item = EquivalentMeaningsOfConceptNameText(
            'alt', language_of_value=self.lang)
        self.assertTrue(hasattr(item, 'ContentSequence'))
        self.assertIn(self.lang, item.ContentSequence)

    def test_from_dataset(self):
        ds = EquivalentMeaningsOfConceptNameText('alt')
        round_item = EquivalentMeaningsOfConceptNameText.from_dataset(ds)
        self.assertIsInstance(round_item, EquivalentMeaningsOfConceptNameText)


class TestEquivalentMeaningsOfConceptNameCode(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.code_val = Code('99TST', 'Test', '99TEST')
        self.concept_val = CodedConcept('100', 'Concept', '99TEST')
        self.lang = LanguageOfValue(
            language=Code('de-DE', 'German', 'RFC5646'))

    def test_accepts_code(self):
        item = EquivalentMeaningsOfConceptNameCode(self.code_val)
        self.assertEqual(item.ValueType, 'CODE')
        self.assertEqual(item.ConceptCodeSequence[0], self.code_val)
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_accepts_coded_concept(self):
        item = EquivalentMeaningsOfConceptNameCode(
            self.concept_val, language_of_value=self.lang)
        self.assertEqual(item.ConceptCodeSequence[0], self.concept_val)
        self.assertIn(self.lang, item.ContentSequence)

    def test_rejects_str(self):
        with self.assertRaises(TypeError):
            EquivalentMeaningsOfConceptNameCode(
                "string")

    def test_from_dataset(self):
        item = EquivalentMeaningsOfConceptNameCode(self.code_val)
        recovered = EquivalentMeaningsOfConceptNameCode.from_dataset(item)
        self.assertIsInstance(recovered, EquivalentMeaningsOfConceptNameCode)


class TestReportNarrativeCode(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.code = CodedConcept('888', 'Narr', '99TEST')
        self.empty_obs = []
        self.img = ImageContentItem(
            name=codes.DCM.SourceImageForSegmentation,
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2',
            referenced_sop_instance_uid='1.2.3.4',
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )

    def test_construction(self):
        item = ReportNarrativeCode(
            self.code,
            basic_diagnostic_imaging_report_observations=self.empty_obs)
        self.assertEqual(item.ValueType, 'CODE')
        self.assertEqual(item.ConceptCodeSequence[0], self.code)
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_construction_with_observation(self):
        item = ReportNarrativeCode(
            self.code,
            basic_diagnostic_imaging_report_observations=[self.img])
        self.assertIn(self.img, item.ContentSequence)

    def test_obs_not_iterable(self):
        with self.assertRaises(TypeError):
            ReportNarrativeCode(
                self.code, basic_diagnostic_imaging_report_observations=42)

    def test_obs_wrong_item_type(self):
        with self.assertRaises(TypeError):
            ReportNarrativeCode(
                self.code,
                basic_diagnostic_imaging_report_observations=[None])

    def test_from_dataset(self):
        item = ReportNarrativeCode(self.code, self.empty_obs)
        recovered = ReportNarrativeCode.from_dataset(item)
        self.assertIsInstance(recovered, ReportNarrativeCode)


class TestReportNarrativeText(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.empty_obs = []
        self.img = ImageContentItem(
            name=codes.DCM.SourceImageForSegmentation,
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2',
            referenced_sop_instance_uid='1.2.3.4',
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )

    def test_construction(self):
        item = ReportNarrativeText(
            'free text',
            basic_diagnostic_imaging_report_observations=self.empty_obs)
        self.assertEqual(item.ValueType, 'TEXT')
        self.assertEqual(item.ConceptNameCodeSequence[0].CodeValue, '7002')
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_construction_with_observation(self):
        item = ReportNarrativeText(
            'obs',
            basic_diagnostic_imaging_report_observations=[self.img])
        self.assertIn(self.img, item.ContentSequence)

    def test_obs_not_iterable(self):
        with self.assertRaises(TypeError):
            ReportNarrativeText(
                'x', basic_diagnostic_imaging_report_observations=123)

    def test_obs_wrong_item_type(self):
        with self.assertRaises(TypeError):
            ReportNarrativeText(
                'x',
                basic_diagnostic_imaging_report_observations=['bad'])

    def test_from_dataset(self):
        item = ReportNarrativeText(
            'abc',
            basic_diagnostic_imaging_report_observations=self.empty_obs)
        recovered = ReportNarrativeText.from_dataset(item)
        self.assertIsInstance(recovered, ReportNarrativeText)


class TestDiagnosticImagingReportHeading(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.narr_text = ReportNarrativeText('txt', [])
        self.narr_code = ReportNarrativeCode(CodedConcept('1', 'n', '99'), [])
        person = PersonObserverIdentifyingAttributes(
            name='Doe^John'
        )
        self.obs_ctx = ObservationContext(
            observer_person_context=ObserverContext(
                observer_type=codes.cid270.Person,
                observer_identifying_attributes=person
            )
        )

    def test_construction_with_text_narrative(self):
        head = DiagnosticImagingReportHeading(self.narr_text)
        container = head[0]
        self.assertEqual(
            container.ContentTemplateSequence[0].TemplateIdentifier, '7001')
        self.assertIn(self.narr_text, container.ContentSequence)

    def test_construction_with_code_narrative(self):
        head = DiagnosticImagingReportHeading(self.narr_code)
        self.assertIn(self.narr_code, head[0].ContentSequence)

    def test_construction_with_observation_context(self):
        head = DiagnosticImagingReportHeading(
            self.narr_text, observation_context=self.obs_ctx)
        for item in self.obs_ctx:
            self.assertIn(item, head[0].ContentSequence)

    def test_invalid_narrative_type(self):
        with self.assertRaises(TypeError):
            DiagnosticImagingReportHeading("bad")

    def test_invalid_observation_context(self):
        with self.assertRaises(TypeError):
            DiagnosticImagingReportHeading(
                self.narr_text, observation_context="bad")


class TestBasicDiagnosticImagingReport(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.lang = LanguageOfContentItemAndDescendants(
            language=Code('en-US', 'English (US)', 'RFC5646')
        )
        person = PersonObserverIdentifyingAttributes(
            name='Doe^John'
        )
        self.obs_ctx = ObservationContext(
            observer_person_context=ObserverContext(
                observer_type=codes.cid270.Person,
                observer_identifying_attributes=person
            )
        )
        self.proc_code = CodedConcept('113724', 'CT Chest', 'SCT')
        self.dev_code = Code('99DEV', 'CT scanner', '99TEST')
        self.reg_code = Code('T-D3000', 'Lung', 'SRT')

        self.test_code_item = EquivalentMeaningsOfConceptNameCode(
            value=Code('99ALT', 'Alt-code', '99TEST')
        )
        self.test_text_item = EquivalentMeaningsOfConceptNameText(
            value='Alternative term'
        )
        self.heading = DiagnosticImagingReportHeading(
            report_narrative=ReportNarrativeText(
                value='Simple narrative',
                basic_diagnostic_imaging_report_observations=[]
            )
        )

    def test_construction(self):
        report = BasicDiagnosticImagingReport(
            language_of_content_item_and_descendants=self.lang,
            observation_context=self.obs_ctx,
        )
        self.assertEqual(len(report), 1)
        container = report[0]
        self.assertEqual(container.ValueType, 'CONTAINER')
        self.assertEqual(
            container.ContentTemplateSequence[0].TemplateIdentifier,
            '2000'
        )
        self.assertIn(self.lang[0], container.ContentSequence)
        self.assertTrue(
            any(ci.RelationshipType ==
                RelationshipTypeValues.HAS_OBS_CONTEXT.value
                for ci in container.ContentSequence),
            'ObservationContext items missing'
        )

    def test_construction_with_optionals(self):
        report = BasicDiagnosticImagingReport(
            language_of_content_item_and_descendants=self.lang,
            observation_context=self.obs_ctx,
            procedures_reported=[self.proc_code],
            acquisition_device_types=[self.dev_code],
            target_regions=[self.reg_code],
            equivalent_meanings_of_concept_name=[
                self.test_text_item, self.test_code_item],
            diagnostic_imaging_report_headings=[self.heading],
        )
        container = report[0]
        seq = container.ContentSequence
        self.assertTrue(
            any(item.ConceptNameCodeSequence[0] == codes.DCM.ProcedureReported
                for item in seq),
            '"Procedure Reported" items missing'
        )
        self.assertTrue(
            any(item.ConceptNameCodeSequence[0] ==
                codes.DCM.AcquisitionDeviceType
                for item in seq),
            '"Acquisition Device Type" items missing'
        )
        self.assertTrue(
            any(item.ConceptNameCodeSequence[0] == codes.DCM.TargetRegion
                for item in seq),
            '"Target Region" items missing'
        )
        for equiv in (self.test_text_item, self.test_code_item):
            self.assertIn(equiv, seq)
        for child in self.heading:
            self.assertIn(child, seq)

    def test_mandatory_args_typecheck(self):
        with self.assertRaises(TypeError):
            BasicDiagnosticImagingReport(
                language_of_content_item_and_descendants='wrong type',
                observation_context=self.obs_ctx,
            )

        with self.assertRaises(TypeError):
            BasicDiagnosticImagingReport(
                language_of_content_item_and_descendants=self.lang,
                observation_context='wrong type',
            )

    def test_sequence_arg_must_be_iterable(self):
        not_seq = 123
        for kw in (
            dict(procedures_reported=not_seq),
            dict(acquisition_device_types=not_seq),
            dict(target_regions=not_seq),
            dict(equivalent_meanings_of_concept_name=not_seq),
            dict(diagnostic_imaging_report_headings=not_seq),
        ):
            with self.assertRaises(TypeError):
                BasicDiagnosticImagingReport(
                    language_of_content_item_and_descendants=self.lang,
                    observation_context=self.obs_ctx,
                    **kw
                )

    def test_sequence_item_type_validation(self):
        wrong = 'string'
        wrong_cases = (
            dict(procedures_reported=[wrong]),
            dict(acquisition_device_types=[wrong]),
            dict(target_regions=[wrong]),
            dict(equivalent_meanings_of_concept_name=[wrong]),
            dict(diagnostic_imaging_report_headings=[wrong]),
        )
        for kw in wrong_cases:
            with self.assertRaises(TypeError):
                BasicDiagnosticImagingReport(
                    language_of_content_item_and_descendants=self.lang,
                    observation_context=self.obs_ctx,
                    **kw
                )
