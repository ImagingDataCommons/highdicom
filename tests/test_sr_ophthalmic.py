"""Tests for Eyecare Measurement SR templates (TID 2120/2123/2124),
ratified in DICOM PS3.16 2025b (formerly circulated as Supplement 247 draft
TIDs 6001/6004/6005).

Tests cover:
- OphthalmologyMeasurementsGroup (TID 2120)
- CircumpapillaryRNFLKeyMeasurements (TID 2123)
- MacularThicknessKeyMeasurements (TID 2124)
- Two ComprehensiveSR roundtrips (serialise → dcmread → verify values survive)
"""

from io import BytesIO
from pathlib import Path

import pytest
from pydicom import dcmread
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

from highdicom.sr.sop import ComprehensiveSR
from highdicom.sr.templates import (
    AlgorithmIdentification,
    Measurement,
    TrackingIdentifier,
)
from highdicom.sr.templates.tid2120 import (
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
    UCUM_MICROLITER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / 'data' / 'test_files'


def _make_tracking_id(identifier: str = 'ophthalmic-group') -> TrackingIdentifier:
    # The inherited MeasurementsAndQualitativeEvaluations base class requires
    # both a human-readable identifier and a UID to be present (not just a
    # UID), so a non-None identifier must always be supplied.
    return TrackingIdentifier(identifier=identifier)


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
# OphthalmologyMeasurementsGroup (TID 2120)
# ---------------------------------------------------------------------------


class TestOphthalmologyMeasurementsGroup:
    def test_basic_construction(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        assert len(group) == 1

    def test_container_name_is_measurement_group(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        container = group[0]
        assert container.ConceptNameCodeSequence[0].CodeValue == '125007'

    def test_template_id(self):
        """TID 2120 is invoked here, not the TID 1501 base class it reuses."""
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        container = group[0]
        assert container.ContentTemplateSequence[0].TemplateIdentifier == '2120'

    def test_finding_site_default_is_eye(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        site = group.finding_sites[0]
        assert site.value == codes.cid4209.Eye

    def test_finding_site_custom(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Left,
            measurements=[_make_rnfl_measurement()],
            finding_site=codes.cid4209.OpticNerveHead,
        )
        site = group.finding_sites[0]
        assert site.value == codes.cid4209.OpticNerveHead

    def test_laterality_nested_inside_finding_site(self):
        """Laterality is a modifier of Finding Site, not of the container."""
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        site = group.finding_sites[0]
        assert hasattr(site, 'ContentSequence')
        assert site.ContentSequence[0].ConceptNameCodeSequence[0].CodeValue == \
            '272741003'

    def test_laterality_right(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        site = group.finding_sites[0]
        # Right eye: EV (24028007, SCT, "Right")
        assert site.laterality.value == '24028007'

    def test_laterality_left(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Left,
            measurements=[_make_rnfl_measurement()],
        )
        site = group.finding_sites[0]
        # Left eye: EV (7771000, SCT, "Left")
        assert site.laterality.value == '7771000'

    def test_topographical_modifier(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
            topographical_modifier=codes.cid4209.OpticNerveHead,
        )
        site = group.finding_sites[0]
        assert site.topographical_modifier == codes.cid4209.OpticNerveHead

    def test_measurement_present(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement(121.0)],
        )
        measurements = group.get_measurements()
        assert len(measurements) == 1
        assert measurements[0].value == 121.0

    def test_measurement_unit_micrometer(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        assert group.get_measurements()[0].unit == codes.UCUM.Micrometer

    def test_tracking_identifier(self):
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id('RNFL-OD-001'),
            laterality=codes.cid247.Right,
            measurements=[_make_rnfl_measurement()],
        )
        assert group.tracking_identifier == 'RNFL-OD-001'
        assert group.tracking_uid is not None

    def test_tracking_identifier_requires_readable_text(self):
        """The inherited base class requires both a human-readable
        identifier and a UID; a bare TrackingIdentifier() (UID only) is
        rejected even though TID 2120 itself marks tracking identifier as
        optional overall."""
        with pytest.raises(ValueError, match="tracking"):
            OphthalmologyMeasurementsGroup(
                tracking_identifier=TrackingIdentifier(),
                laterality=codes.cid247.Right,
                measurements=[_make_rnfl_measurement()],
            )

    def test_multiple_measurements(self):
        meas_list = [
            Measurement(name=RNFLAverageThickness, value=121.0, unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLInferiorThickness, value=145.0, unit=codes.UCUM.Micrometer),
            Measurement(name=RNFLSuperiorThickness, value=138.0, unit=codes.UCUM.Micrometer),
        ]
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=meas_list,
        )
        vals = [m.value for m in group.get_measurements()]
        assert vals == [121.0, 145.0, 138.0]

    def test_no_measurements_allowed(self):
        """TID 2120's Measurement row is conditional on the invoking root
        template's requirements, not mandatory at this level, so an empty/
        absent list is accepted here."""
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[],
        )
        assert group.get_measurements() == []

    def test_wrong_measurement_type_raises(self):
        with pytest.raises(TypeError, match="type Measurement"):
            OphthalmologyMeasurementsGroup(
                tracking_identifier=_make_tracking_id(),
                laterality=codes.cid247.Right,
                measurements=["not a measurement"],
            )


# ---------------------------------------------------------------------------
# CircumpapillaryRNFLKeyMeasurements (TID 2123)
# ---------------------------------------------------------------------------


class TestCircumpapillaryRNFLKeyMeasurements:
    def _make_group(self, laterality=None) -> OphthalmologyMeasurementsGroup:
        lat = laterality or codes.cid247.Right
        return OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
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
        assert root.ConceptNameCodeSequence[0].CodeValue == '131242'
        assert root.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'

    def test_template_id(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        assert report[0].ContentTemplateSequence[0].TemplateIdentifier == '2123'

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
        od_group = self._make_group(codes.cid247.Right)
        os_group = self._make_group(codes.cid247.Left)
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[od_group, os_group],
        )
        root = report[0]
        # [0]=Language, [1],[2]=AlgoId, [3]=OD group, [4]=OS group
        assert len(root.ContentSequence) == 5

    def test_laterality_in_bilateral(self):
        od_group = self._make_group(codes.cid247.Right)
        os_group = self._make_group(codes.cid247.Left)
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[od_group, os_group],
        )
        root = report[0]
        od_container = root.ContentSequence[3]
        os_container = root.ContentSequence[4]

        od_lat = od_group.finding_sites[0].laterality
        os_lat = os_group.finding_sites[0].laterality
        assert od_lat.value == '24028007'  # Right
        assert os_lat.value == '7771000'   # Left
        # sanity: the containers landed at the expected root positions
        assert od_container.ConceptNameCodeSequence[0].CodeValue == '125007'
        assert os_container.ConceptNameCodeSequence[0].CodeValue == '125007'

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

    def test_rnfl_codes_are_final_dcm_codes(self):
        """RNFL measurement concept names use final, ratified DCM codes."""
        expected = {
            RNFLAverageThickness: '131264',
            RNFLInferiorThickness: '131265',
            RNFLSuperiorThickness: '131266',
            RNFLTemporalThickness: '131267',
            RNFLNasalThickness: '131268',
        }
        for concept, value in expected.items():
            assert concept.scheme_designator == 'DCM'
            assert concept.value == value

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
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=meas_list,
        )
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[group],
        )
        assert len(report) == 1
        vals = [m.value for m in group.get_measurements()]
        assert vals == [121.0, 145.0, 138.0, 80.0, 95.0]


# ---------------------------------------------------------------------------
# MacularThicknessKeyMeasurements (TID 2124)
# ---------------------------------------------------------------------------


class TestMacularThicknessKeyMeasurements:
    def _make_group(self, value: float = 281.4) -> OphthalmologyMeasurementsGroup:
        return OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
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
        assert root.ConceptNameCodeSequence[0].CodeValue == '131243'
        assert root.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'

    def test_template_id(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[self._make_group()],
        )
        assert report[0].ContentTemplateSequence[0].TemplateIdentifier == '2124'

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

    def test_average_macular_thickness_is_final_dcm_code(self):
        assert AverageMacularThickness.scheme_designator == 'DCM'
        assert AverageMacularThickness.value == '131255'

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
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[meas],
        )
        assert group.get_measurements()[0].unit == UCUM_MICROLITER

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
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=meas_list,
        )
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[group],
        )
        assert len(report) == 1

        measurements = group.get_measurements()
        assert len(measurements) == 10
        assert measurements[0].value == 281.4
        assert measurements[-1].unit == UCUM_MICROLITER

    def test_bilateral_macular(self):
        od = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            measurements=[_make_macular_measurement(281.4)],
        )
        os_ = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Left,
            measurements=[_make_macular_measurement(275.0)],
        )
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[od, os_],
        )
        root = report[0]
        # [0]=Language, [1],[2]=AlgoId, [3]=OD, [4]=OS
        assert len(root.ContentSequence) == 5

        assert od.get_measurements()[0].value == 281.4
        assert os_.get_measurements()[0].value == 275.0

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
        """TID 2123: numeric value 121.0 µm and code '131264' survive dcmread."""
        meas = Measurement(
            name=RNFLAverageThickness,
            value=121.0,
            unit=codes.UCUM.Micrometer,
        )
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
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
        meas_items = [
            item for item in mg.ContentSequence if item.ValueType == 'NUM'
        ]
        assert len(meas_items) == 1
        meas_item = meas_items[0]
        assert float(meas_item.MeasuredValueSequence[0].NumericValue) == 121.0
        assert meas_item.ConceptNameCodeSequence[0].CodeValue == '131264'
        assert meas_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == 'um'

    def test_macular_roundtrip_loinc_code_survives(self, ref_ds):
        """TID 2124: LOINC code '57109-1' (CMT) survives serialisation."""
        meas = Measurement(
            name=MacularCenterSubfieldThickness,
            value=281.4,
            unit=codes.UCUM.Micrometer,
        )
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
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
        meas_items = [
            item for item in mg.ContentSequence if item.ValueType == 'NUM'
        ]
        assert len(meas_items) == 1
        meas_item = meas_items[0]
        assert float(meas_item.MeasuredValueSequence[0].NumericValue) == 281.4
        assert meas_item.ConceptNameCodeSequence[0].CodeValue == '57109-1'
        assert meas_item.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'LN'
