"""Tests for Eyecare Measurement SR templates (TID 2120/2123/2124),
ratified in DICOM PS3.16 2025b (formerly circulated as Supplement 247 draft
TIDs 6001/6004/6005).

Tests cover:
- OphthalmologyMeasurementsGroup (TID 2120, generic)
- RNFLSectorMeasurementsGroup / RNFLClockfaceMeasurementsGroup (TID 2120,
  specialized for TID 2123 rows 5/6)
- MacularMeasurementsGroup (TID 2120, specialized for TID 2124 row 5)
- CircumpapillaryRNFLKeyMeasurements (TID 2123), including the bilateral
  RNFL-symmetry conformance rule
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
    GarwayHeathSectors,
    MacularCenterSubfieldThickness,
    MacularInnerSuperiorThickness,
    MacularInnerNasalThickness,
    MacularInnerInferiorThickness,
    MacularInnerTemporalThickness,
    MacularMeasurementsGroup,
    MacularOuterSuperiorThickness,
    MacularOuterNasalThickness,
    MacularOuterInferiorThickness,
    MacularOuterTemporalThickness,
    MacularThicknessKeyMeasurements,
    MacularTotalVolume,
    OphthalmologyMeasurementsGroup,
    QuadrantSectors,
    RNFLAverageThickness,
    RNFLClockfaceMeasurementsGroup,
    RNFLClockfaceMethod,
    RNFLInferiorThickness,
    RNFLSectorMeasurementsGroup,
    RNFLSuperiorThickness,
    RNFLSymmetry,
    RNFLTemporalThickness,
    RNFLNasalThickness,
    RetinalROIWidth,
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


def _make_sector_group(
    laterality=None, identifier: str = 'RNFL-sector', **kwargs
) -> RNFLSectorMeasurementsGroup:
    kwargs.setdefault('average', 121.0)
    return RNFLSectorMeasurementsGroup(
        tracking_identifier=_make_tracking_id(identifier),
        laterality=laterality or codes.cid247.Right,
        sector_method=GarwayHeathSectors,
        retinal_roi_width=3.4,
        **kwargs,
    )


def _make_clockface_group(
    laterality=None, identifier: str = 'RNFL-clockface', **kwargs
) -> RNFLClockfaceMeasurementsGroup:
    kwargs.setdefault('clockface_measurements', {1: 100.0, 6: 95.0})
    return RNFLClockfaceMeasurementsGroup(
        tracking_identifier=_make_tracking_id(identifier),
        laterality=laterality or codes.cid247.Right,
        **kwargs,
    )


def _make_macular_group(
    value: float = 281.4, laterality=None
) -> MacularMeasurementsGroup:
    return MacularMeasurementsGroup(
        tracking_identifier=_make_tracking_id('Macula'),
        laterality=laterality or codes.cid247.Right,
        center_subfield=value,
    )


# ---------------------------------------------------------------------------
# OphthalmologyMeasurementsGroup (TID 2120, generic)
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
# RNFLSectorMeasurementsGroup (TID 2120, sector role, TID 2123 Row 5)
# ---------------------------------------------------------------------------


class TestRNFLSectorMeasurementsGroup:
    def test_basic_construction(self):
        group = _make_sector_group()
        assert len(group) == 1

    def test_roi_width_and_average_present(self):
        group = _make_sector_group(average=121.0)
        measurements = {m.name: m.value for m in group.get_measurements()}
        assert measurements[RetinalROIWidth] == 3.4
        assert measurements[RNFLAverageThickness] == 121.0

    def test_roi_width_unit_is_mm(self):
        group = _make_sector_group()
        roi = next(
            m for m in group.get_measurements() if m.name == RetinalROIWidth
        )
        assert roi.unit == codes.UCUM.Millimeter

    def test_all_optional_quadrants(self):
        group = _make_sector_group(
            average=121.0,
            inferior=145.0,
            superior=138.0,
            temporal=80.0,
            nasal=95.0,
            nasal_superior=90.0,
            nasal_inferior=92.0,
            temporal_inferior=78.0,
            temporal_superior=82.0,
        )
        # ROI width + 9 optional measurements
        assert len(group.get_measurements()) == 10

    def test_method_is_sector_method(self):
        group = _make_sector_group()
        assert group.method == GarwayHeathSectors

    def test_invalid_sector_method_raises(self):
        with pytest.raises(ValueError, match="CID 4282"):
            RNFLSectorMeasurementsGroup(
                tracking_identifier=_make_tracking_id(),
                laterality=codes.cid247.Right,
                sector_method=codes.SCT.Eye,
                retinal_roi_width=3.4,
            )

    def test_accepts_any_cid4282_method(self):
        group = RNFLSectorMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            sector_method=QuadrantSectors,
            retinal_roi_width=3.0,
        )
        assert group.method == QuadrantSectors

    def test_template_id_is_2120(self):
        group = _make_sector_group()
        assert group[0].ContentTemplateSequence[0].TemplateIdentifier == '2120'


# ---------------------------------------------------------------------------
# RNFLClockfaceMeasurementsGroup (TID 2120, clockface role, TID 2123 Row 6)
# ---------------------------------------------------------------------------


class TestRNFLClockfaceMeasurementsGroup:
    def test_basic_construction(self):
        group = _make_clockface_group()
        assert len(group) == 1

    def test_method_is_fixed_clockface_method(self):
        group = _make_clockface_group()
        assert group.method == RNFLClockfaceMethod

    def test_positions_encode_correctly(self):
        group = _make_clockface_group(
            clockface_measurements={1: 100.0, 2: 105.0, 12: 98.0}
        )
        measurements = {m.name.value: m.value for m in group.get_measurements()}
        assert measurements['131276'] == 100.0  # position 1
        assert measurements['131277'] == 105.0  # position 2
        assert measurements['131287'] == 98.0   # position 12

    def test_roi_width_optional_and_encodes(self):
        group = _make_clockface_group(retinal_roi_width=3.6)
        roi = next(
            m for m in group.get_measurements() if m.name == RetinalROIWidth
        )
        assert roi.value == 3.6
        assert roi.unit == codes.UCUM.Millimeter

    def test_empty_measurements_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            RNFLClockfaceMeasurementsGroup(
                tracking_identifier=_make_tracking_id(),
                laterality=codes.cid247.Right,
                clockface_measurements={},
            )

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="1 to 12"):
            RNFLClockfaceMeasurementsGroup(
                tracking_identifier=_make_tracking_id(),
                laterality=codes.cid247.Right,
                clockface_measurements={0: 100.0},
            )
        with pytest.raises(ValueError, match="1 to 12"):
            RNFLClockfaceMeasurementsGroup(
                tracking_identifier=_make_tracking_id(),
                laterality=codes.cid247.Right,
                clockface_measurements={13: 100.0},
            )


# ---------------------------------------------------------------------------
# CircumpapillaryRNFLKeyMeasurements (TID 2123)
# ---------------------------------------------------------------------------


class TestCircumpapillaryRNFLKeyMeasurements:
    def test_basic_construction_sector_only(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            sector_measurement_groups=[_make_sector_group()],
        )
        assert len(report) == 1

    def test_basic_construction_clockface_only(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            clockface_measurement_groups=[_make_clockface_group()],
        )
        assert len(report) == 1

    def test_sector_and_clockface_combined(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            sector_measurement_groups=[_make_sector_group()],
            clockface_measurement_groups=[_make_clockface_group()],
        )
        root = report[0]
        # [0]=Language, [1][2]=AlgoId, [3]=sector group, [4]=clockface group
        assert len(root.ContentSequence) == 5

    def test_root_container_code(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            sector_measurement_groups=[_make_sector_group()],
        )
        root = report[0]
        assert root.ConceptNameCodeSequence[0].CodeValue == '131242'
        assert root.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'

    def test_template_id(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            sector_measurement_groups=[_make_sector_group()],
        )
        assert report[0].ContentTemplateSequence[0].TemplateIdentifier == '2123'

    def test_algo_id_present(self):
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo('MyOCT', '2.3'),
            sector_measurement_groups=[_make_sector_group()],
        )
        root = report[0]
        algo_name_item = root.ContentSequence[1]
        assert algo_name_item.TextValue == 'MyOCT'

    def test_bilateral_sector_requires_symmetry(self):
        od = _make_sector_group(codes.cid247.Right, identifier='OD')
        os_ = _make_sector_group(codes.cid247.Left, identifier='OS')
        with pytest.raises(ValueError, match="rnfl_symmetry"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                sector_measurement_groups=[od, os_],
            )

    def test_bilateral_sector_with_symmetry_succeeds(self):
        od = _make_sector_group(codes.cid247.Right, identifier='OD')
        os_ = _make_sector_group(codes.cid247.Left, identifier='OS')
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            sector_measurement_groups=[od, os_],
            rnfl_symmetry=97.3,
        )
        root = report[0]
        symmetry_items = [
            item for item in root.ContentSequence
            if getattr(item, 'ValueType', None) == 'NUM' and
            item.ConceptNameCodeSequence[0].CodeValue == '131273'
        ]
        assert len(symmetry_items) == 1
        assert float(symmetry_items[0].MeasuredValueSequence[0].NumericValue) == 97.3

    def test_bilateral_across_sector_and_clockface_requires_symmetry(self):
        """Bilaterality is judged across both group lists combined."""
        od = _make_sector_group(codes.cid247.Right, identifier='OD')
        os_ = _make_clockface_group(codes.cid247.Left, identifier='OS')
        with pytest.raises(ValueError, match="rnfl_symmetry"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                sector_measurement_groups=[od],
                clockface_measurement_groups=[os_],
            )

    def test_unilateral_rejects_symmetry(self):
        with pytest.raises(ValueError, match="only applicable"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                sector_measurement_groups=[_make_sector_group()],
                rnfl_symmetry=97.3,
            )

    def test_neither_group_list_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            CircumpapillaryRNFLKeyMeasurements(algorithm_id=_make_algo())

    def test_more_than_two_sector_groups_raises(self):
        with pytest.raises(ValueError, match="at most two"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                sector_measurement_groups=[
                    _make_sector_group(identifier='1'),
                    _make_sector_group(identifier='2'),
                    _make_sector_group(identifier='3'),
                ],
            )

    def test_more_than_two_clockface_groups_raises(self):
        with pytest.raises(ValueError, match="at most two"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                clockface_measurement_groups=[
                    _make_clockface_group(identifier='1'),
                    _make_clockface_group(identifier='2'),
                    _make_clockface_group(identifier='3'),
                ],
            )

    def test_wrong_algo_type_raises(self):
        with pytest.raises(TypeError, match="AlgorithmIdentification"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id="not an algo",
                sector_measurement_groups=[_make_sector_group()],
            )

    def test_wrong_sector_group_type_raises(self):
        with pytest.raises(TypeError, match="RNFLSectorMeasurementsGroup"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                sector_measurement_groups=["not a group"],
            )

    def test_wrong_clockface_group_type_raises(self):
        with pytest.raises(TypeError, match="RNFLClockfaceMeasurementsGroup"):
            CircumpapillaryRNFLKeyMeasurements(
                algorithm_id=_make_algo(),
                clockface_measurement_groups=["not a group"],
            )

    def test_rnfl_codes_are_final_dcm_codes(self):
        """RNFL measurement concept names use final, ratified DCM codes."""
        expected = {
            RNFLAverageThickness: '131264',
            RNFLInferiorThickness: '131265',
            RNFLSuperiorThickness: '131266',
            RNFLTemporalThickness: '131267',
            RNFLNasalThickness: '131268',
            RetinalROIWidth: '131274',
            RNFLSymmetry: '131273',
        }
        for concept, value in expected.items():
            assert concept.scheme_designator == 'DCM'
            assert concept.value == value


# ---------------------------------------------------------------------------
# MacularMeasurementsGroup (TID 2120, macular role, TID 2124 Row 5)
# ---------------------------------------------------------------------------


class TestMacularMeasurementsGroup:
    def test_basic_construction(self):
        group = _make_macular_group()
        assert len(group) == 1

    def test_center_subfield_value(self):
        group = _make_macular_group(281.4)
        measurements = {m.name: m.value for m in group.get_measurements()}
        assert measurements[MacularCenterSubfieldThickness] == 281.4

    def test_total_volume_unit_is_ul(self):
        group = MacularMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            total_volume=8.42,
        )
        vol = next(
            m for m in group.get_measurements() if m.name == MacularTotalVolume
        )
        assert vol.unit == UCUM_MICROLITER

    def test_full_etdrs_grid(self):
        group = MacularMeasurementsGroup(
            tracking_identifier=_make_tracking_id(),
            laterality=codes.cid247.Right,
            center_point=270.0,
            center_subfield=281.4,
            inner_superior=338.0,
            inner_nasal=350.0,
            inner_inferior=335.0,
            inner_temporal=320.0,
            outer_superior=290.0,
            outer_nasal=305.0,
            outer_inferior=285.0,
            outer_temporal=265.0,
            total_volume=8.42,
            average_thickness=300.1,
        )
        # 11 micrometer-unit measurements + total volume
        assert len(group.get_measurements()) == 12

    def test_no_measurements_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            MacularMeasurementsGroup(
                tracking_identifier=_make_tracking_id(),
                laterality=codes.cid247.Right,
            )

    def test_template_id_is_2120(self):
        group = _make_macular_group()
        assert group[0].ContentTemplateSequence[0].TemplateIdentifier == '2120'


# ---------------------------------------------------------------------------
# MacularThicknessKeyMeasurements (TID 2124)
# ---------------------------------------------------------------------------


class TestMacularThicknessKeyMeasurements:
    def test_basic_construction(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[_make_macular_group()],
        )
        assert len(report) == 1

    def test_root_container_code(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[_make_macular_group()],
        )
        root = report[0]
        assert root.ConceptNameCodeSequence[0].CodeValue == '131243'
        assert root.ConceptNameCodeSequence[0].CodingSchemeDesignator == 'DCM'

    def test_template_id(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[_make_macular_group()],
        )
        assert report[0].ContentTemplateSequence[0].TemplateIdentifier == '2124'

    def test_content_sequence_structure(self):
        report = MacularThicknessKeyMeasurements(
            algorithm_id=_make_algo(),
            measurement_groups=[_make_macular_group()],
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

    def test_bilateral_macular(self):
        od = _make_macular_group(281.4, codes.cid247.Right)
        os_ = _make_macular_group(275.0, codes.cid247.Left)
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
                    _make_macular_group(),
                    _make_macular_group(),
                    _make_macular_group(),
                ],
            )

    def test_wrong_group_type_raises(self):
        with pytest.raises(TypeError, match="MacularMeasurementsGroup"):
            MacularThicknessKeyMeasurements(
                algorithm_id=_make_algo(),
                measurement_groups=["not a group"],
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
        group = _make_sector_group(average=121.0)
        template = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=_make_algo(),
            sector_measurement_groups=[group],
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
        meas_items = {
            item.ConceptNameCodeSequence[0].CodeValue: item
            for item in mg.ContentSequence if item.ValueType == 'NUM'
        }
        avg_item = meas_items['131264']
        assert float(avg_item.MeasuredValueSequence[0].NumericValue) == 121.0
        assert avg_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == 'um'
        # Retinal ROI width (mandatory) also survives
        roi_item = meas_items['131274']
        assert float(roi_item.MeasuredValueSequence[0].NumericValue) == 3.4

    def test_macular_roundtrip_loinc_code_survives(self, ref_ds):
        """TID 2124: LOINC code '57109-1' (CMT) survives serialisation."""
        group = _make_macular_group(281.4)
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
