"""Tests for Supplement 247 Eyecare Measurement SR templates (TID 6001/6004/6005).

Tests cover:
- OphthalmologyMeasurementsGroup (TID 6001)
- CircumpapillaryRNFLKeyMeasurements (TID 6004)
- MacularThicknessKeyMeasurements (TID 6005)
- Two ComprehensiveSR roundtrips (serialise → dcmread → verify values survive)

Content item navigation paths (post laterality-nesting fix):
  group[0]                        → ContainerContentItem "Measurement Group"
  group[0].ContentSequence[0]     → Finding Site (CodeContentItem)
  group[0].ContentSequence[0].ContentSequence[0] → Laterality (CodeContentItem)
  group[0].ContentSequence[1]     → first Measurement NumContentItem (no tracking)
  group[0].ContentSequence[1]     → Tracking Identifier text (with tracking)
  group[0].ContentSequence[2]     → first Measurement NumContentItem (with tracking)

  report[0]                       → root ContainerContentItem
  report[0].ContentSequence[0]    → Language item
  report[0].ContentSequence[1]    → AlgorithmId item 1
  report[0].ContentSequence[2]    → AlgorithmId item 2
  report[0].ContentSequence[3]    → first OphthalmologyMeasurementsGroup container
  report[0].ContentSequence[4]    → second group container (bilateral)
"""

from io import BytesIO
from pathlib import Path

import pytest
from pydicom import dcmread
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

from highdicom.sr.sop import ComprehensiveSR
from highdicom.sr.templates import AlgorithmIdentification, Measurement
from highdicom.sr.templates.tid6000 import (
    AverageMacularThickness,
    CircumpapillaryRNFLKeyMeasurements,
    MacularCenterSubfieldThickness,
    MacularInnerSuperiorThickness,
    MacularInnerNasalThickness,
    MacularInnerInferiorThickness,
    MacularInnerTemporalThickness,
    MacularOuterSuperiorThickness,
    MacularOuterNasalThickness,
    MacularOuterInferiorThickness,
    MacularOuterTemporalThickness,
    MacularThicknessKeyMeasurements,
    MacularTotalVolume,
    OphthalmologyMeasurementsGroup,
    RNFLAverageThickness,
    RNFLInferiorThickness,
    RNFLSuperiorThickness,
    RNFLTemporalThickness,
    RNFLNasalThickness,
    RetinalROIRadius,
    UCUM_MICROLITER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / 'data' / 'test_files'


def _make_rnfl_measurement(value: float = 121.0) -> Measurement:
    return Measurement(
        name=RNFLAverageThickness,
        value=value,
        unit=codes.UCUM.Micrometer,
    )


def _make_macular_measurement(value: float = 281.4) -> Measurement:
    return Measurement(
        name=MacularCenterSubfieldThickness,
        value=value,
        unit=codes.UCUM.Micrometer,
    )


def _make_algo(name: str = 'Revo FC130', version: str = '1.0') -> AlgorithmIdentification:
    return AlgorithmIdentification(name=name, version=version)


# ---------------------------------------------------------------------------
# OphthalmologyMeasurementsGroup (TID 6001)
# ---------------------------------------------------------------------------


class TestOphthalmologyMeasurementsGroup:
    def test_basic_construction(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
        )
        assert len(group) == 1

    def test_container_name_is_measurement_group(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
        )
        container = group[0]
        assert container.ConceptNameCodeSequence[0].CodeValue == '125007'

    def test_template_id(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
        )
        container = group[0]
        assert container.ContentTemplateSequence[0].TemplateIdentifier == '6001'

    def test_finding_site_default_is_eye(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
        )
        container = group[0]
        site_item = container.ContentSequence[0]
        # Concept name: Finding Site
        assert site_item.ConceptNameCodeSequence[0].CodeValue == '363698007'
        # Value: Eye
        assert site_item.ConceptCodeSequence[0].CodeValue == '81745001'

    def test_finding_site_custom(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Left,
            measurements=[_make_rnfl_measurement()],
            finding_site=codes.cid4209.OpticNerveHead,
        )
        container = group[0]
        site_item = container.ContentSequence[0]
        assert site_item.ConceptCodeSequence[0].CodeValue == \
            codes.cid4209.OpticNerveHead.value

    def test_laterality_nested_inside_finding_site(self):
        """Row 3 >> in TID 60x1: laterality is child of Finding Site, not container."""
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
        )
        container = group[0]
        site_item = container.ContentSequence[0]
        # Finding Site must have its own ContentSequence with Laterality inside
        assert hasattr(site_item, 'ContentSequence')
        lat_item = site_item.ContentSequence[0]
        assert lat_item.ConceptNameCodeSequence[0].CodeValue == '272741003'

    def test_laterality_right(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
        )
        lat = group[0].ContentSequence[0].ContentSequence[0]
        # Right eye: EV (24028007, SCT, "Right")
        assert lat.ConceptCodeSequence[0].CodeValue == '24028007'

    def test_laterality_left(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Left,
            measurements=[_make_rnfl_measurement()],
        )
        lat = group[0].ContentSequence[0].ContentSequence[0]
        # Left eye: EV (7771000, SCT, "Left")
        assert lat.ConceptCodeSequence[0].CodeValue == '7771000'

    def test_measurement_present(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement(121.0)],
        )
        container = group[0]
        # Without tracking: measurement is at index 1
        meas_item = container.ContentSequence[1]
        assert float(meas_item.MeasuredValueSequence[0].NumericValue) == 121.0

    def test_measurement_unit_micrometer(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
        )
        meas_item = group[0].ContentSequence[1]
        assert meas_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == 'um'

    def test_tracking_identifier(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement()],
            tracking_identifier='RNFL-OD-001',
        )
        container = group[0]
        # With tracking: [0]=site, [1]=tracking, [2]=measurement
        tracking_item = container.ContentSequence[1]
        assert tracking_item.ConceptNameCodeSequence[0].CodeValue == '112039'
        assert tracking_item.TextValue == 'RNFL-OD-001'

    def test_measurement_index_shifts_with_tracking(self):
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_rnfl_measurement(99.0)],
            tracking_identifier='track-01',
        )
        # With tracking identifier, measurement moves to index 2
        meas_item = group[0].ContentSequence[2]
        assert float(meas_item.MeasuredValueSequence[0].NumericValue) == 99.0

    def test_multiple_measurements(self):
        meas_list = [
            Measurement(name=RNFLAverageThickness, value=121.0, unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLInferiorThickness, value=145.0, unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLSuperiorThickness, value=138.0, unit=codes.UCUM.Micrometer),
        ]
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=meas_list,
        )
        container = group[0]
        # [0]=site, [1],[2],[3]=3 measurements
        assert len(container.ContentSequence) == 4
        vals = [
            float(container.ContentSequence[i].MeasuredValueSequence[0].NumericValue)
            for i in (1, 2, 3)
        ]
        assert vals == [121.0, 145.0, 138.0]

    def test_empty_measurements_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            OphthalmologyMeasurementsGroup(
                laterality=codes.cid244.Right,
                measurements=[],
            )

    def test_wrong_measurement_type_raises(self):
        with pytest.raises(TypeError, match="type Measurement"):
            OphthalmologyMeasurementsGroup(
                laterality=codes.cid244.Right,
                measurements=["not a measurement"],
            )


# ---------------------------------------------------------------------------
# CircumpapillaryRNFLKeyMeasurements (TID 6004)
# ---------------------------------------------------------------------------


class TestCircumpapillaryRNFLKeyMeasurements:
    def _make_group(self, laterality=None) -> OphthalmologyMeasurementsGroup:
        lat = laterality or codes.cid244.Right
        return OphthalmologyMeasurementsGroup(
            laterality=lat,
            measurements=[_make_rnfl_measurement()],
        )

    def test_basic_construction(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        assert len(report) == 1

    def test_root_container_code(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        root = report[0]
        assert root.ConceptNameCodeSequence[0].CodeValue == 'nnn102'
        assert root.ConceptNameCodeSequence[0].CodingSchemeDesignator == '99OPHTHALMO'

    def test_template_id(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        assert report[0].ContentTemplateSequence[0].TemplateIdentifier == '6004'

    def test_content_sequence_structure(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        root = report[0]
        # [0]=Language, [1],[2]=AlgoId (2 items), [3]=Measurement Group
        assert len(root.ContentSequence) == 4

    def test_algo_id_present(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo('MyOCT', '2.3'),
            measurement_groups=[self._make_group()],
        )
        root = report[0]
        algo_name_item = root.ContentSequence[1]
        assert algo_name_item.TextValue == 'MyOCT'

    def test_bilateral(self):
        od_group = self._make_group(codes.cid244.Right)
        os_group = self._make_group(codes.cid244.Left)
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[od_group, os_group],
        )
        root = report[0]
        # [0]=Language, [1],[2]=AlgoId, [3]=OD group, [4]=OS group
        assert len(root.ContentSequence) == 5

    def test_laterality_in_bilateral(self):
        od_group = self._make_group(codes.cid244.Right)
        os_group = self._make_group(codes.cid244.Left)
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[od_group, os_group],
        )
        root = report[0]
        od_container = root.ContentSequence[3]
        os_container = root.ContentSequence[4]

        od_lat = od_container.ContentSequence[0].ContentSequence[0]
        os_lat = os_container.ContentSequence[0].ContentSequence[0]
        assert od_lat.ConceptCodeSequence[0].CodeValue == '24028007'  # Right
        assert os_lat.ConceptCodeSequence[0].CodeValue == '7771000'   # Left

    def test_more_than_two_groups_raises(self):
        with pytest.raises(ValueError, match="at most two"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                measurement_groups=[
                    self._make_group(),
                    self._make_group(),
                    self._make_group(),
                ],
            )

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                measurement_groups=[],
            )

    def test_wrong_algo_type_raises(self):
        with pytest.raises(TypeError, match="AlgorithmIdentification"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id="not an algo",
                measurement_groups=[self._make_group()],
            )

    def test_wrong_group_type_raises(self):
        with pytest.raises(TypeError, match="OphthalmologyMeasurementsGroup"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                measurement_groups=["not a group"],
            )

    def test_rnfl_codes_are_99ophthalmo(self):
        """All RNFL measurement concept names use scheme 99OPHTHALMO."""
        for concept in (
            RNFLAverageThickness,
            RNFLInferiorThickness,
            RNFLSuperiorThickness,
            RNFLTemporalThickness,
            RNFLNasalThickness,
            RetinalROIRadius,
        ):
            assert concept.scheme_designator == '99OPHTHALMO', (
                f"{concept.meaning} should use 99OPHTHALMO, got {concept.scheme_designator}"
            )

    def test_full_rnfl_quadrants(self):
        """All five RNFL quadrant measurements encode correctly."""
        meas_list = [
            Measurement(name=RNFLAverageThickness,  value=121.0, unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLInferiorThickness,  value=145.0, unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLSuperiorThickness,  value=138.0, unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLTemporalThickness,  value=80.0,  unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLNasalThickness,     value=95.0,  unit=codes.UCUM.Micrometer),
        ]
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=meas_list,
        )
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[group],
        )
        root = report[0]
        mg_container = root.ContentSequence[3]
        # [0]=site, [1..5]=5 measurements
        assert len(mg_container.ContentSequence) == 6
        avg = mg_container.ContentSequence[1]
        assert float(avg.MeasuredValueSequence[0].NumericValue) == 121.0


# ---------------------------------------------------------------------------
# MacularThicknessKeyMeasurements (TID 6005)
# ---------------------------------------------------------------------------


class TestMacularThicknessKeyMeasurements:
    def _make_group(self, value: float = 281.4) -> OphthalmologyMeasurementsGroup:
        return OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_macular_measurement(value)],
        )

    def test_basic_construction(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        assert len(report) == 1

    def test_root_container_code(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        root = report[0]
        assert root.ConceptNameCodeSequence[0].CodeValue == 'nnn103'
        assert root.ConceptNameCodeSequence[0].CodingSchemeDesignator == '99OPHTHALMO'

    def test_template_id(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        assert report[0].ContentTemplateSequence[0].TemplateIdentifier == '6005'

    def test_content_sequence_structure(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        root = report[0]
        # [0]=Language, [1],[2]=AlgoId (2 items), [3]=Measurement Group
        assert len(root.ContentSequence) == 4

    def test_etdrs_codes_are_loinc(self):
        """ETDRS subfield measurements use LOINC (LN) scheme."""
        loinc_concepts = [
            MacularCenterSubfieldThickness,
            MacularInnerSuperiorThickness,
            MacularInnerNasalThickness,
            MacularInnerInferiorThickness,
            MacularInnerTemporalThickness,
            MacularOuterSuperiorThickness,
            MacularOuterNasalThickness,
            MacularOuterInferiorThickness,
            MacularOuterTemporalThickness,
            MacularTotalVolume,
        ]
        for concept in loinc_concepts:
            assert concept.scheme_designator == 'LN', (
                f"{concept.meaning} should use LN, got {concept.scheme_designator}"
            )

    def test_average_macular_thickness_is_99ophthalmo(self):
        assert AverageMacularThickness.scheme_designator == '99OPHTHALMO'
        assert AverageMacularThickness.value == 'nnn250'

    def test_ucum_microliter_constant(self):
        """UCUM_MICROLITER is 'uL' from scheme UCUM."""
        assert UCUM_MICROLITER.value == 'uL'
        assert UCUM_MICROLITER.scheme_designator == 'UCUM'

    def test_total_volume_uses_ul(self):
        """MacularTotalVolume measurement must use UCUM_MICROLITER unit."""
        meas = Measurement(
            name=MacularTotalVolume,
            value=8.42,
            unit=UCUM_MICROLITER,
        )
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[meas],
        )
        meas_item = group[0].ContentSequence[1]
        assert meas_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == 'uL'

    def test_full_etdrs_grid(self):
        """Nine ETDRS subfields + total volume encode correctly."""
        meas_list = [
            Measurement(name=MacularCenterSubfieldThickness,  value=281.4, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularInnerSuperiorThickness,   value=338.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularInnerNasalThickness,      value=350.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularInnerInferiorThickness,   value=335.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularInnerTemporalThickness,   value=320.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularOuterSuperiorThickness,   value=290.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularOuterNasalThickness,      value=305.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularOuterInferiorThickness,   value=285.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularOuterTemporalThickness,   value=265.0, unit=codes.UCUM.Micrometer),
            Measurement(name=MacularTotalVolume,              value=8.42,  unit=UCUM_MICROLITER),
        ]
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=meas_list,
        )
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[group],
        )
        root = report[0]
        mg_container = root.ContentSequence[3]
        # [0]=site, [1..10]=10 measurements
        assert len(mg_container.ContentSequence) == 11

        # Verify CMT value
        cmt_item = mg_container.ContentSequence[1]
        assert float(cmt_item.MeasuredValueSequence[0].NumericValue) == 281.4

        # Verify total volume uses uL
        vol_item = mg_container.ContentSequence[10]
        assert vol_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == 'uL'

    def test_bilateral_macular(self):
        od = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[_make_macular_measurement(281.4)],
        )
        os = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Left,
            measurements=[_make_macular_measurement(275.0)],
        )
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[od, os],
        )
        root = report[0]
        # [0]=Language, [1],[2]=AlgoId, [3]=OD, [4]=OS
        assert len(root.ContentSequence) == 5

        od_meas = root.ContentSequence[3].ContentSequence[1]
        os_meas = root.ContentSequence[4].ContentSequence[1]
        assert float(od_meas.MeasuredValueSequence[0].NumericValue) == 281.4
        assert float(os_meas.MeasuredValueSequence[0].NumericValue) == 275.0

    def test_more_than_two_groups_raises(self):
        with pytest.raises(ValueError, match="at most two"):
            MacularThicknessKeyMeasurements(
                algorithm_id=_make_algo(),
                measurement_groups=[
                    self._make_group(),
                    self._make_group(),
                    self._make_group(),
                ],
            )


# ---------------------------------------------------------------------------
# Roundtrip tests — template → ComprehensiveSR → dcmwrite → dcmread
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Verify that SR content survives serialisation to DICOM bytes."""

    @pytest.fixture
    def ref_ds(self):
        return dcmread(str(_DATA_DIR / 'ct_image.dcm'))

    def test_rnfl_roundtrip(self, ref_ds):
        """TID 6004: numeric value 121.0 µm and code 'nnn400' survive dcmread."""
        meas = Measurement(
            name=RNFLAverageThickness,
            value=121.0,
            unit=codes.UCUM.Micrometer,
        )
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[meas],
        )
        template = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[group],
        )

        sr = ComprehensiveSR(
            evidence=[ref_ds],
            content=template,
            series_instance_uid=generate_uid(),
            series_number=1,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer='Revo',
        )
        with BytesIO() as buf:
            sr.save_as(buf)
            buf.seek(0)
            sr_back = dcmread(buf)

        # After dcmread, sr_back.ContentSequence lists the ROOT container's
        # children directly: [0]=Language, [1][2]=AlgoId, [3]=MeasGroup
        mg = sr_back.ContentSequence[3]
        assert mg.ValueType == 'CONTAINER'
        meas_item = mg.ContentSequence[1]
        assert float(meas_item.MeasuredValueSequence[0].NumericValue) == 121.0
        assert meas_item.ConceptNameCodeSequence[0].CodeValue == 'nnn400'
        assert meas_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == 'um'

    def test_macular_roundtrip_loinc_code_survives(self, ref_ds):
        """TID 6005: LOINC code '57109-1' (CMT) survives serialisation."""
        meas = Measurement(
            name=MacularCenterSubfieldThickness,
            value=281.4,
            unit=codes.UCUM.Micrometer,
        )
        group = OphthalmologyMeasurementsGroup(
            laterality=codes.cid244.Right,
            measurements=[meas],
        )
        template = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[group],
        )

        sr = ComprehensiveSR(
            evidence=[ref_ds],
            content=template,
            series_instance_uid=generate_uid(),
            series_number=1,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer='Cirrus',
        )
        with BytesIO() as buf:
            sr.save_as(buf)
            buf.seek(0)
            sr_back = dcmread(buf)

        # After dcmread, sr_back.ContentSequence lists the ROOT container's
        # children directly: [0]=Language, [1][2]=AlgoId, [3]=MeasGroup
        mg = sr_back.ContentSequence[3]
        assert mg.ValueType == 'CONTAINER'
        meas_item = mg.ContentSequence[1]
        assert float(meas_item.MeasuredValueSequence[0].NumericValue) == 281.4
        assert meas_item.ConceptNameCodeSequence[0].CodeValue == '57109-1'
        assert meas_item.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'LN'
