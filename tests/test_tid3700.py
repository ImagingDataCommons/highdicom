import unittest
from datetime import datetime

from highdicom.sr import PersonObserverIdentifyingAttributes
from highdicom.sr.value_types import (
    Code,
    CodedConcept,
    CompositeContentItem,
    TemporalRangeTypeValues,
    UIDRefContentItem,
    WaveformContentItem,
    TcoordContentItem,
    RelationshipTypeValues,
)
from highdicom.sr.templates import (
    CardiovascularPatientHistory,
    ECGFinding,
    ECGGlobalMeasurements,
    ECGLeadMeasurements,
    ECGQualitativeAnalysis,
    ECGReport,
    ECGWaveFormInformation,
    ECGMeasurementSource,
    IndicationsForProcedure,
    NumberOfEctopicBeats,
    ObserverContext,
    PatientCharacteristicsForECG,
    PriorECGStudy,
    QTcIntervalGlobal,
    QuantitativeAnalysis,
    SummaryECG,
    LanguageOfContentItemAndDescendants,
    AgeUnit,
    PressureUnit
)
from pydicom.sr.codedict import codes


class TestECGWaveFormInformation(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.proc_dt = datetime.now()
        self.wf_item = WaveformContentItem(
            name=Code('123', 'Waveform', '99TEST'),
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2',
            referenced_sop_instance_uid='1.2.3.4',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        self.lead_code = Code('123', 'Lead-X', '99TEST')
        self.acq_dev = 'Recorder-X'
        self.equip_id = 'SN-42'
        self.room_id = ['Room-1']
        self.num_vars = [1.1, 2.2]
        self.txt_vars = ['on', 'off']

    def test_construction(self):
        tpl = ECGWaveFormInformation(self.proc_dt)
        cont = tpl[0]
        print(cont)
        self.assertEqual(
            cont.ContentTemplateSequence[0].TemplateIdentifier,
            '3708'
        )
        self.assertEqual(cont.ContentSequence[0].ValueType, 'DATETIME')

    def test_construction_with_optionals(self):
        tpl = ECGWaveFormInformation(
            self.proc_dt,
            source_of_measurement=self.wf_item,
            lead_system=self.lead_code,
            acquisition_device_type=self.acq_dev,
            equipment_identification=self.equip_id,
            room_identification=self.room_id,
            ecg_control_numeric_variables=self.num_vars,
            ecg_control_text_variables=self.txt_vars,
        )
        seq = tpl[0].ContentSequence
        self.assertIn(self.wf_item, seq)
        self.assertTrue(any(ci.ValueType == 'NUM' for ci in seq))
        self.assertTrue(any(ci.ValueType == 'TEXT' and
                            ci.ConceptNameCodeSequence[0].CodeValue == '3691'
                            for ci in seq))

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(self.proc_dt, source_of_measurement=123)
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(self.proc_dt, lead_system='bad')
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(self.proc_dt, acquisition_device_type=42)
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(self.proc_dt, equipment_identification=[])
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(self.proc_dt, room_identification='oops')
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(
                self.proc_dt, ecg_control_numeric_variables='x')
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(
                self.proc_dt, ecg_control_numeric_variables=['x'])
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(self.proc_dt, ecg_control_text_variables=0)
        with self.assertRaises(TypeError):
            ECGWaveFormInformation(
                self.proc_dt,
                ecg_control_text_variables=[1])


class TestECGMeasurementSource(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.beat = '5'
        self.method_code = CodedConcept('555', 'Auto', '99TEST')
        self.tcoord_item = TcoordContentItem(
            name=Code('123', 'Tcoord', '99TEST'),
            temporal_range_type=TemporalRangeTypeValues.POINT,
            referenced_date_time=[datetime.now()],
            relationship_type=RelationshipTypeValues.CONTAINS
        )

    def test_construction(self):
        tpl = ECGMeasurementSource(self.beat, None, None)
        cont = tpl[0]
        self.assertEqual(
            cont.ContentTemplateSequence[0].TemplateIdentifier,
            '3715'
        )
        self.assertEqual(cont.ContentSequence[0].ValueType, 'TEXT')
        self.assertEqual(cont.ContentSequence[0].TextValue, self.beat)

    def test_construction_with_optionals(self):
        tpl = ECGMeasurementSource(
            self.beat, self.method_code, self.tcoord_item)
        seq = tpl[0].ContentSequence
        self.assertIn(self.tcoord_item, seq)
        self.assertTrue(
            any(ci.ValueType == 'CODE' and
                ci.ConceptNameCodeSequence[0] == codes.SCT.MeasurementMethod
                for ci in seq))

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            ECGMeasurementSource(123, None, None)
        with self.assertRaises(TypeError):
            ECGMeasurementSource(self.beat, 'str', None)
        with self.assertRaises(TypeError):
            ECGMeasurementSource(self.beat, None, 42)


class TestQTcIntervalGlobal(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.alg_code = Code('ALG', 'Baz', '99TEST')

    def test_construction(self):
        item = QTcIntervalGlobal(400.0)
        self.assertEqual(item.ValueType, 'NUM')
        unit = item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0]
        self.assertEqual(unit, codes.UCUM.Millisecond)
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_construction_with_algorithm(self):
        item = QTcIntervalGlobal(420.5, algorithm_name=self.alg_code)
        self.assertIn(
            self.alg_code, item.ContentSequence[0].ConceptCodeSequence)
        self.assertEqual(item.ContentSequence[0].RelationshipType,
                         RelationshipTypeValues.HAS_PROPERTIES.value)

    def test_type_guard(self):
        with self.assertRaises(TypeError):
            QTcIntervalGlobal(300.0, algorithm_name='bad')

    def test_from_dataset(self):
        item = QTcIntervalGlobal(450.0)
        rebuilt = QTcIntervalGlobal.from_dataset(item)
        self.assertIsInstance(rebuilt, QTcIntervalGlobal)


class TestNumberOfEctopicBeats(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.morph_code = CodedConcept('999', 'PVC', '99TEST')

    def test_construction(self):
        item = NumberOfEctopicBeats(7)
        unit = item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0]
        self.assertEqual(unit.CodeValue, 'beats')
        self.assertFalse(hasattr(item, 'ContentSequence'))

    def test_construction_with_morphologies(self):
        item = NumberOfEctopicBeats(
            9, associated_morphologies=[self.morph_code])
        self.assertIn(self.morph_code,
                      item.ContentSequence[0].ConceptCodeSequence)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            NumberOfEctopicBeats(3, associated_morphologies=123)
        with self.assertRaises(TypeError):
            NumberOfEctopicBeats(3, associated_morphologies=['bad'])

    def test_from_dataset(self):
        item = NumberOfEctopicBeats(2)
        rebuilt = NumberOfEctopicBeats.from_dataset(item)
        self.assertIsInstance(rebuilt, NumberOfEctopicBeats)


class TestECGGlobalMeasurements(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.base_vals = dict(
            ventricular_heart_rate=60.0,
            qt_interval_global=350.0,
            pr_interval_global=160.0,
            qrs_duration_global=90.0,
            rr_interval_global=1000.0,
        )
        self.msrc = ECGMeasurementSource(
            '1', None,
            TcoordContentItem(
                name=Code('123', 'Tcoord', '99TEST'),
                temporal_range_type=TemporalRangeTypeValues.POINT,
                referenced_date_time=[datetime.now()],
                relationship_type=RelationshipTypeValues.CONTAINS
            ))
        self.qtc = QTcIntervalGlobal(400.0)
        self.durations = [120.0, 130.0]
        self.axes = [45.0, -30.0]
        self.all_beats = 700.0
        self.ectopic_num = NumberOfEctopicBeats(
            value=5.0
        )

    def test_construction(self):
        tpl = ECGGlobalMeasurements(**self.base_vals)
        cont = tpl[0]
        self.assertEqual(
            cont.ContentTemplateSequence[0].TemplateIdentifier,
            '3713'
        )
        self.assertEqual(
            cont.ConceptNameCodeSequence[0], codes.DCM.ECGGlobalMeasurements)
        self.assertEqual(
            sum(1 for ci in cont.ContentSequence if ci.ValueType == 'NUM'),
            5
        )
        bpm_unit = cont.ContentSequence[0]\
            .MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0]
        self.assertEqual(bpm_unit, Code('bpm', 'UCUM', 'beats per minute'))

    def test_construction_with_optionals(self):
        tpl = ECGGlobalMeasurements(
            **self.base_vals,
            ecg_measurement_source=self.msrc,
            atrial_heart_rate=75.0,
            qtc_interval_global=self.qtc,
            ecg_global_waveform_durations=self.durations,
            ecg_axis_measurements=self.axes,
            count_of_all_beats=self.all_beats,
            number_of_ectopic_beats=self.ectopic_num,
        )
        seq = tpl[0].ContentSequence
        self.assertTrue(any(item in seq for item in self.msrc))
        self.assertIn(self.qtc, seq)
        self.assertTrue(any(ci.ContentSequence[0].TextValue == '1' if
                            getattr(ci, "MeasuredValueSequence", None) is
                            None else getattr(ci,
                                              "MeasuredValueSequence"
                                              )[0].NumericValue ==
                            75.0 for ci in seq))
        self.assertEqual(
            sum(
                1 for ci in seq
                if ci.ConceptNameCodeSequence[0].CodeValue == '3687'),
            len(self.durations)
        )
        self.assertEqual(
            sum(
                1 for ci in seq
                if ci.ConceptNameCodeSequence[0].CodeValue == '3229'),
            len(self.axes)
        )
        self.assertTrue(
            any(ci.MeasuredValueSequence[0].NumericValue == self.all_beats if
                getattr(ci, "MeasuredValueSequence", None) is not None
                else False for ci in seq))
        self.assertTrue(
            any(ci.MeasuredValueSequence[0].NumericValue ==
                self.ectopic_num.value if getattr(ci,
                                                  "MeasuredValueSequence",
                                                  None) is
                not None else False
                for ci in seq))

    def test_type_guard_mandatory(self):
        bad = self.base_vals.copy()
        bad['ventricular_heart_rate'] = 'fast'
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(**bad)

    def test_bad_measurement_source(self):
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(
                **self.base_vals, ecg_measurement_source='bad')

    def test_bad_atrial_rate(self):
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(**self.base_vals, atrial_heart_rate='x')

    def test_bad_qtc_interval(self):
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(**self.base_vals, qtc_interval_global=123)

    def test_bad_duration_sequence(self):
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(
                **self.base_vals, ecg_global_waveform_durations=123)
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(
                **self.base_vals, ecg_global_waveform_durations=['x'])

    def test_bad_axis_sequence(self):
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(**self.base_vals, ecg_axis_measurements='x')
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(
                **self.base_vals, ecg_axis_measurements=[None])

    def test_bad_all_beats(self):
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(**self.base_vals, count_of_all_beats='many')

    def test_bad_ectopic_beats(self):
        with self.assertRaises(TypeError):
            ECGGlobalMeasurements(
                **self.base_vals, number_of_ectopic_beats='PVC')


class TestECGLeadMeasurements(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.lead_code = Code('lead1', 'Lead I', '99')
        # Create a dummy ECGMeasurementSource that is iterable.
        self.ecg_source = ECGMeasurementSource("5", None, None)
        self.durations = [100.0, 200.5]
        self.voltages = [0.5, 1.2]
        self.st_finding = Code('st1', 'ST abnormality', '99')
        self.findings = [Code('st1', 'ST abnormality', '99'),
                         Code('st2', 'ST abnormality', '99')]

    def test_construction(self):
        meas = ECGLeadMeasurements(self.lead_code)
        container = meas[0]
        seq = container.ContentSequence
        self.assertEqual(len(seq), 1)
        lead_item = seq[0]
        self.assertEqual(lead_item.value, self.lead_code)

    def test_invalid_lead_id(self):
        with self.assertRaises(TypeError):
            ECGLeadMeasurements("bad_lead")

    def test_ecg_measurement_source(self):
        meas = ECGLeadMeasurements(
            self.lead_code, ecg_measurement_source=self.ecg_source)
        container = meas[0]
        seq = container.ContentSequence
        dummy_item = next(iter(self.ecg_source))
        self.assertIn(dummy_item, seq)

    def test_valid_durations(self):
        meas = ECGLeadMeasurements(
            self.lead_code, electrophysiology_waveform_durations=self.durations)
        seq = meas[0].ContentSequence
        self.assertEqual(len(seq), 1 + len(self.durations))
        dur_item = seq[1]
        self.assertEqual(dur_item.value, self.durations[0])
        self.assertEqual(dur_item.unit, codes.UCUM.Millisecond)

    def test_invalid_durations(self):
        with self.assertRaises(TypeError):
            ECGLeadMeasurements(
                self.lead_code, electrophysiology_waveform_durations=123)
        with self.assertRaises(TypeError):
            ECGLeadMeasurements(
                self.lead_code, electrophysiology_waveform_durations=["bad"])

    def test_valid_voltages(self):
        meas = ECGLeadMeasurements(
            self.lead_code, electrophysiology_waveform_voltages=self.voltages)
        seq = meas[0].ContentSequence
        self.assertEqual(len(seq), 1 + len(self.voltages))
        voltage_item = seq[1]
        self.assertEqual(voltage_item.value, self.voltages[0])
        self.assertEqual(voltage_item.unit, codes.UCUM.Millivolt)

    def test_invalid_voltages(self):
        with self.assertRaises(TypeError):
            ECGLeadMeasurements(
                self.lead_code, electrophysiology_waveform_voltages="bad")
        with self.assertRaises(TypeError):
            ECGLeadMeasurements(
                self.lead_code, electrophysiology_waveform_voltages=[None])

    def test_valid_st_segment_finding(self):
        meas = ECGLeadMeasurements(
            self.lead_code, st_segment_finding=self.st_finding)
        seq = meas[0].ContentSequence
        self.assertEqual(len(seq), 2)
        st_item = seq[1]
        self.assertEqual(st_item.value, self.st_finding)

    def test_invalid_st_segment_finding(self):
        with self.assertRaises(TypeError):
            ECGLeadMeasurements(self.lead_code, st_segment_finding=123)

    def test_valid_findings(self):
        meas = ECGLeadMeasurements(self.lead_code, findings=self.findings)
        seq = meas[0].ContentSequence
        self.assertEqual(len(seq), 1 + len(self.findings))
        finding_item = seq[1]
        self.assertEqual(finding_item.value, self.findings[0])

    def test_invalid_findings(self):
        with self.assertRaises(TypeError):
            ECGLeadMeasurements(self.lead_code, findings="bad")
        with self.assertRaises(TypeError):
            ECGLeadMeasurements(self.lead_code, findings=["bad"])

    def test_construction_all_optionals(self):
        meas = ECGLeadMeasurements(
            self.lead_code,
            ecg_measurement_source=self.ecg_source,
            electrophysiology_waveform_durations=self.durations,
            electrophysiology_waveform_voltages=self.voltages,
            st_segment_finding=self.st_finding,
            findings=self.findings,
        )
        seq = meas[0].ContentSequence
        expected_count = (1 + len(self.durations) + len(self.voltages) +
                          1 + len(self.findings) + len(list(self.ecg_source)))
        self.assertEqual(len(seq), expected_count)


class TestQuantitativeAnalysis(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.global_meas = ECGGlobalMeasurements(
            ventricular_heart_rate=60.0,
            qt_interval_global=350.0,
            pr_interval_global=160.0,
            qrs_duration_global=90.0,
            rr_interval_global=1000.0,
        )
        self.lead_meas = ECGLeadMeasurements(Code('lead1', 'Lead I', '99'))

    def test_construction(self):
        tpl = QuantitativeAnalysis()
        self.assertEqual(len(tpl[0].ContentSequence), 0)

    def test_construction_with_global(self):
        tpl = QuantitativeAnalysis(ecg_global_measurements=self.global_meas)
        for item in self.global_meas:
            self.assertIn(item, tpl[0].ContentSequence)

    def test_construction_with_leads(self):
        tpl = QuantitativeAnalysis(ecg_lead_measurements=[self.lead_meas])
        self.assertIn(self.lead_meas[0], tpl[0].ContentSequence)

    def test_construction_with_optionals(self):
        tpl = QuantitativeAnalysis(
            ecg_global_measurements=self.global_meas,
            ecg_lead_measurements=[self.lead_meas],
        )
        seq = tpl[0].ContentSequence
        self.assertTrue(any(item in seq for item in self.global_meas))
        self.assertIn(self.lead_meas[0], seq)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            QuantitativeAnalysis(ecg_global_measurements='bad')
        with self.assertRaises(TypeError):
            QuantitativeAnalysis(ecg_lead_measurements='bad')
        with self.assertRaises(TypeError):
            QuantitativeAnalysis(ecg_lead_measurements=[123])


class TestIndicationsForProcedure(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.code1 = Code('A', 'Alpha', '99')
        self.concept = CodedConcept('B', 'Bravo', '99')
        self.text = 'Chest pain'

    def test_construction(self):
        tpl = IndicationsForProcedure()
        self.assertFalse(hasattr(tpl[0], 'ContentSequence'))

    def test_construction_with_codes(self):
        tpl = IndicationsForProcedure(findings=[self.code1, self.concept])
        seq = tpl[0].ContentSequence
        self.assertEqual(len(seq), 2)
        self.assertTrue(all(ci.ValueType == 'CODE' for ci in seq))

    def test_construction_with_text(self):
        tpl = IndicationsForProcedure(finding_text=self.text)
        itm = tpl[0].ContentSequence[0]
        self.assertEqual(itm.TextValue, self.text)

    def test_construction_with_optionals(self):
        tpl = IndicationsForProcedure(
            findings=[self.code1], finding_text=self.text)
        seq = tpl[0].ContentSequence
        self.assertEqual(len(seq), 2)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            IndicationsForProcedure(findings='bad')
        with self.assertRaises(TypeError):
            IndicationsForProcedure(findings=[42])
        with self.assertRaises(TypeError):
            IndicationsForProcedure(finding_text=123)


class TestPatientCharacteristicsForECG(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.age = AgeUnit(2000, 1, 1)
        self.sys_bp = PressureUnit(1, 1)
        self.dia_bp = PressureUnit(2, 2)
        self.state = Code('REST', 'Resting', '99')
        self.pace = CodedConcept('P', 'Pacemaker', '99')
        self.icd = CodedConcept('ICD', 'ICD in situ', '99')

    def test_construction(self):
        tpl = PatientCharacteristicsForECG(self.age, 'M')
        cont = tpl[0]
        self.assertEqual(
            cont.ContentTemplateSequence[0].TemplateIdentifier,
            '3704'
        )
        self.assertTrue(
            any(ci.ValueType == 'TEXT' and
                ci.ConceptNameCodeSequence[0] == codes.DCM.SubjectSex
                for ci in cont.ContentSequence))

    def test_construction_with_optionals(self):
        tpl = PatientCharacteristicsForECG(
            self.age, 'F',
            patient_height=180.0,
            patient_weight=80.0,
            systolic_blood_pressure=self.sys_bp,
            diastolic_blood_pressure=self.dia_bp,
            patient_state=self.state,
            pacemaker_in_situ=self.pace,
            icd_in_situ=self.icd,
        )
        seq = tpl[0].ContentSequence
        self.assertTrue(any(
            ci.ValueType == 'NUM' and
            ci.ConceptNameCodeSequence[0].CodeValue == '8302-2'
            for ci in seq))
        self.assertTrue(any(
            ci.ValueType == 'NUM' and
            ci.ConceptNameCodeSequence[0].CodeValue == '29463-7'
            for ci in seq))
        self.assertTrue(any(
            ci.ValueType == 'CODE' and
            ci.ConceptNameCodeSequence[0] == codes.DCM.PatientState
            for ci in seq))

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            PatientCharacteristicsForECG('age', 'M')
        with self.assertRaises(TypeError):
            PatientCharacteristicsForECG(self.age, 1)
        with self.assertRaises(TypeError):
            PatientCharacteristicsForECG(
                self.age, 'M', patient_height='tall')
        with self.assertRaises(TypeError):
            PatientCharacteristicsForECG(
                self.age, 'M', patient_weight='heavy')
        with self.assertRaises(TypeError):
            PatientCharacteristicsForECG(
                self.age, 'M', systolic_blood_pressure='bad')
        with self.assertRaises(TypeError):
            PatientCharacteristicsForECG(
                self.age, 'M', patient_state='bad')


class TestPriorECGStudy(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.base_code = Code('Y', 'Yes', '99')
        self.uid_item = UIDRefContentItem(
            name=codes.DCM.SeriesInstanceUID,
            value='1.2.3.4.5.6',
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        self.rep_item = CompositeContentItem(
            name=codes.DCM.SeriesInstanceUID,
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2',
            referenced_sop_instance_uid='1.2.3.4',
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )
        self.wf_item = WaveformContentItem(
            name=Code('123', 'Waveform', '99TEST'),
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.2',
            referenced_sop_instance_uid='1.2.3.4',
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        self.dt = datetime.now()

    def test_construction(self):
        tpl = PriorECGStudy(self.base_code)
        self.assertEqual(tpl[0].ConceptNameCodeSequence[0],
                         codes.LN.PriorProcedureDescriptions)
        self.assertTrue(
            any(ci.ValueType == 'CODE' for ci in tpl[0].ContentSequence))

    def test_construction_with_optionals(self):
        tpl = PriorECGStudy(
            self.base_code,
            procedure_datetime=self.dt,
            procedure_study_instance_uid=self.uid_item,
            prior_report_for_current_patient=self.rep_item,
            source_of_measurement=self.wf_item,
        )
        seq = tpl[0].ContentSequence
        self.assertIn(self.uid_item, seq)
        self.assertIn(self.rep_item, seq)
        self.assertIn(self.wf_item, seq)
        self.assertTrue(any(ci.ValueType == 'DATETIME' for ci in seq))

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            PriorECGStudy('bad')
        with self.assertRaises(TypeError):
            PriorECGStudy(self.base_code, procedure_datetime=123)
        with self.assertRaises(TypeError):
            PriorECGStudy(self.base_code, procedure_study_instance_uid='bad')
        with self.assertRaises(TypeError):
            PriorECGStudy(self.base_code,
                          prior_report_for_current_patient='bad')
        with self.assertRaises(TypeError):
            PriorECGStudy(self.base_code, source_of_measurement='bad')


class TestECGFinding(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.val_code = CodedConcept('Z', 'Finding', '99')
        self.eq_text = 'equivalent'
        self.sub_finding = ECGFinding(self.val_code)

    def test_construction(self):
        itm = ECGFinding(self.val_code)[0]
        print(itm)
        self.assertEqual(itm.ValueType, 'CODE')
        self.assertEqual(itm.ConceptCodeSequence[0], self.val_code)

    def test_construction_with_equivalent_and_nested(self):
        itm = ECGFinding(
            self.val_code,
            equivalent_meaning_of_value=self.eq_text, ecg_findings=[
                self.sub_finding])[0]
        seq = itm.ContentSequence
        self.assertTrue(
            any(ci.ValueType == 'TEXT' and ci.TextValue == self.eq_text
                for ci in seq))
        self.assertIn(self.sub_finding[0], seq)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            ECGFinding('bad')
        with self.assertRaises(TypeError):

            ECGFinding(self.val_code, equivalent_meaning_of_value=123)
        with self.assertRaises(TypeError):
            ECGFinding(self.val_code, ecg_findings='bad')
        with self.assertRaises(TypeError):
            ECGFinding(self.val_code, ecg_findings=[123])


class TestECGQualitativeAnalysis(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.text = 'Normal ECG'
        self.finding_code = ECGFinding(CodedConcept('X', 'Some', '99'))

    def test_text_only(self):
        tpl = ECGQualitativeAnalysis(ecg_finding_text=self.text)
        self.assertTrue(any(ci.ValueType == 'TEXT' and ci.TextValue ==
                        self.text for ci in tpl[0].ContentSequence))

    def test_codes_only(self):
        tpl = ECGQualitativeAnalysis(ecg_finding_codes=[self.finding_code])
        self.assertIn(self.finding_code[0], tpl[0].ContentSequence)

    def test_both(self):
        tpl = ECGQualitativeAnalysis(
            ecg_finding_text=self.text, ecg_finding_codes=[self.finding_code])
        seq = tpl[0].ContentSequence
        self.assertTrue(any(ci.ValueType == 'TEXT' for ci in seq) and
                        self.finding_code[0] in seq)

    def test_required_argument(self):
        with self.assertRaises(ValueError):
            ECGQualitativeAnalysis()

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            ECGQualitativeAnalysis(ecg_finding_text=123)
        with self.assertRaises(TypeError):
            ECGQualitativeAnalysis(ecg_finding_codes='bad')
        with self.assertRaises(TypeError):
            ECGQualitativeAnalysis(ecg_finding_codes=[123])


class TestSummaryECG(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.sum_text = 'ECG appears normal'
        # matches constructor expectation (str)
        self.overall_code = CodedConcept('X', 'Finding', '99')

    def test_text_only(self):
        tpl = SummaryECG(summary=self.sum_text)
        self.assertTrue(any(ci.ValueType == 'TEXT' and ci.TextValue ==
                        self.sum_text for ci in tpl[0].ContentSequence))

    def test_code_only(self):
        tpl = SummaryECG(ecg_overall_finding=self.overall_code)
        self.assertTrue(
            any(ci.ValueType == 'CODE' for ci in tpl[0].ContentSequence))

    def test_both(self):
        tpl = SummaryECG(summary=self.sum_text,
                         ecg_overall_finding=self.overall_code)
        self.assertEqual(len(tpl[0].ContentSequence), 2)

    def test_type_guards(self):
        with self.assertRaises(TypeError):
            SummaryECG(summary=123)
        with self.assertRaises(TypeError):
            SummaryECG(ecg_overall_finding="Overall")


class TestECGReport(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.lang = LanguageOfContentItemAndDescendants(
            language=Code('en-US', 'US-English', 'RFC5646')
        )
        person = PersonObserverIdentifyingAttributes(
            name='Doe^John'
        )
        self.obs_ctx = ObserverContext(
            observer_type=codes.cid270.Person,
            observer_identifying_attributes=person
        )
        self.wf_info = ECGWaveFormInformation(datetime.now())
        self.quant = QuantitativeAnalysis(
            ecg_global_measurements=ECGGlobalMeasurements(
                ventricular_heart_rate=60.0,
                qt_interval_global=350.0,
                pr_interval_global=160.0,
                qrs_duration_global=90.0,
                rr_interval_global=1000.0,
            )
        )
        self.procedure_code = Code('PROC', 'Rest ECG', '99')
        self.indication = IndicationsForProcedure()
        self.cardio_hist = CardiovascularPatientHistory()
        self.patient_chars = PatientCharacteristicsForECG(
            AgeUnit(2000, 1, 1), "F"
        )
        self.prior = PriorECGStudy(
            Code('X', 'Finding', '99')
        )
        self.qual = ECGQualitativeAnalysis(ecg_finding_text="Finding")
        self.summary = SummaryECG()

    def test_construction(self):
        rep = ECGReport(
            self.lang,
            [self.obs_ctx],
            self.wf_info,
            self.quant,
        )
        cont = rep[0]
        self.assertEqual(
            cont.ContentTemplateSequence[0].TemplateIdentifier,
            '3700'
        )
        self.assertIn(self.wf_info[0], cont.ContentSequence)

    def test_construction_with_optionals(self):
        rep = ECGReport(
            self.lang,
            [self.obs_ctx],
            self.wf_info,
            self.quant,
            procedure_reported=self.procedure_code,
            indications_for_procedure=self.indication,
            cardiovascular_patient_history=self.cardio_hist,
            patient_characteristics_for_ecg=self.patient_chars,
            prior_ecg_study=self.prior,
            ecg_qualitative_analysis=self.qual,
            summary_ecg=self.summary,
        )
        seq = rep[0].ContentSequence
        self.assertTrue(
            any(ci.ConceptNameCodeSequence[0] == codes.DCM.ProcedureReported
                for ci in seq))

    def test_guard_mandatory(self):
        with self.assertRaises(TypeError):
            ECGReport('bad', [self.obs_ctx], self.wf_info,
                      self.quant)
        with self.assertRaises(TypeError):
            ECGReport(self.lang, 'bad', self.wf_info,
                      self.quant)
        with self.assertRaises(TypeError):
            ECGReport(self.lang, [123], self.wf_info,
                      self.quant)
        with self.assertRaises(TypeError):
            ECGReport(self.lang, [self.obs_ctx], 'bad',
                      self.quant)
        with self.assertRaises(TypeError):
            ECGReport(self.lang, [self.obs_ctx],
                      self.wf_info, 'bad')
