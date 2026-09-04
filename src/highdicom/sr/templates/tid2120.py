"""DICOM SR templates for Eyecare Measurement Templates.

Implements TIDs 2120, 2123, and 2124 as ratified in DICOM PS3.16 2025b
(formerly circulated in Supplement 247 draft as TIDs 6001, 6004, and 6005;
Supplement 247 has since been incorporated into the standard and its
templates renumbered).

Reference: :dcm:`TID 2120 <part16/sect_TID_2120.html>`,
:dcm:`TID 2123 <part16/sect_TID_2123.html>`,
:dcm:`TID 2124 <part16/sect_TID_2124.html>`

Conformance enforcement
------------------------
Per the ratified standard, TID 2123 *Circumpapillary Retinal Nerve Fiber
Layer Key Measurements* invokes TID 2120 in two structurally distinct
roles: a "sector" group (:dcm:`CID 4282 <part16/sect_CID_4282.html>` sector
methods, e.g. Garway-Heath or quadrant sectors, with a mandatory Retinal ROI
width measurement plus optional :dcm:`CID 4283 <part16/sect_CID_4283.html>`
sector values) and a separate "clockface" group (fixed RNFL Clockface
Method, :dcm:`CID 4284 <part16/sect_CID_4284.html>` clockface-position
values), plus a bilateral RNFL-symmetry measurement that becomes mandatory
once both eyes are represented.

Rather than accept an arbitrary list of measurements and hope the caller
assembled the right combination for whichever role and CID applies,
:class:`RNFLSectorMeasurementsGroup`, :class:`RNFLClockfaceMeasurementsGroup`,
and :class:`MacularMeasurementsGroup` accept the individual measurement
values as named parameters and build the correctly-coded
:class:`~highdicom.sr.Measurement` instances internally, so that a
successful construction guarantees a conformant object.
:class:`OphthalmologyMeasurementsGroup` remains available directly for
other TID 2120 invocations not covered by these specializations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

from highdicom.sr.coding import CodedConcept
from highdicom.sr.content import (
    FindingSite,
    RealWorldValueMap,
    SourceImageForMeasurementGroup,
)
from highdicom.sr.value_types import (
    ContentSequence,
    ContainerContentItem,
)
from highdicom.sr.templates.tid1500 import (
    AlgorithmIdentification,
    DEFAULT_LANGUAGE,
    LanguageOfContentItemAndDescendants,
    Measurement,
    MeasurementsAndQualitativeEvaluations,
    QualitativeEvaluation,
    Template,
    TimePointContext,
    TrackingIdentifier,
)

# ---------------------------------------------------------------------------
# Supplementary UCUM unit not yet in pydicom.sr.codedict
# ---------------------------------------------------------------------------
#: UCUM microliter — used for macular total volume (``LN 57118-2``).
#: Not exposed as a convenience attribute in pydicom 3.x; defined here until
#: pydicom adds it to ``codes.UCUM``.
UCUM_MICROLITER: Code = Code('uL', 'UCUM', 'microliter')

# ---------------------------------------------------------------------------
# Container concept names for the root templates TID 2123 and TID 2124.
# Final DCM codes per the ratified standard (formerly Sup247 draft
# placeholders nnn102/nnn103 under a private coding scheme).
# ---------------------------------------------------------------------------
_CONCEPT_CIRCUMPAPILLARY_RNFL_KEY = CodedConcept(
    value='131242', scheme_designator='DCM',
    meaning='Circumpapillary RNFL Key Measurements'
)
_CONCEPT_MACULAR_THICKNESS_KEY = CodedConcept(
    value='131243', scheme_designator='DCM',
    meaning='Macular Thickness Key Measurements'
)

# ---------------------------------------------------------------------------
# CID 4282 — Retinal Sector Methods. Used as the ``method`` value for
# RNFLSectorMeasurementsGroup.
# ---------------------------------------------------------------------------

#: Semicircular sectors. DCM 131301.
SemicircularSectors: CodedConcept = CodedConcept(
    value='131301', scheme_designator='DCM', meaning='Semicircular sectors'
)
#: Quadrant sectors. DCM 131302.
QuadrantSectors: CodedConcept = CodedConcept(
    value='131302', scheme_designator='DCM', meaning='Quadrant sectors'
)
#: SNIT (Superior-Nasal-Inferior-Temporal) rectangular sectors. DCM 131303.
SNITRectangularSectors: CodedConcept = CodedConcept(
    value='131303', scheme_designator='DCM',
    meaning='SNIT rectangular sectors'
)
#: Garway-Heath sectors. DCM 131305.
GarwayHeathSectors: CodedConcept = CodedConcept(
    value='131305', scheme_designator='DCM', meaning='Garway-Heath sectors'
)
#: Quadrant-octant sectors. DCM 131306.
QuadrantOctantSectors: CodedConcept = CodedConcept(
    value='131306', scheme_designator='DCM',
    meaning='Quadrant-octant sectors'
)

_SECTOR_METHODS = (
    SemicircularSectors,
    QuadrantSectors,
    SNITRectangularSectors,
    GarwayHeathSectors,
    QuadrantOctantSectors,
)

#: RNFL Clockface Method — the fixed method value for
#: RNFLClockfaceMeasurementsGroup. DCM 131308.
RNFLClockfaceMethod: CodedConcept = CodedConcept(
    value='131308', scheme_designator='DCM',
    meaning='RNFL Clockface Method'
)

# ---------------------------------------------------------------------------
# CID 4283 — RNFL Sector Measurements. All values in µm.
# Final DCM codes per the ratified standard (formerly Sup247 draft
# placeholders nnn400-nnn404 under a private coding scheme).
# ---------------------------------------------------------------------------

#: Average circumpapillary RNFL thickness. DCM 131264.
RNFLAverageThickness: CodedConcept = CodedConcept(
    value='131264', scheme_designator='DCM',
    meaning='RNFL average thickness'
)

#: Inferior sector RNFL thickness. DCM 131265.
RNFLInferiorThickness: CodedConcept = CodedConcept(
    value='131265', scheme_designator='DCM',
    meaning='RNFL inferior sector thickness'
)

#: Superior sector RNFL thickness. DCM 131266.
RNFLSuperiorThickness: CodedConcept = CodedConcept(
    value='131266', scheme_designator='DCM',
    meaning='RNFL superior sector thickness'
)

#: Temporal sector RNFL thickness. DCM 131267.
RNFLTemporalThickness: CodedConcept = CodedConcept(
    value='131267', scheme_designator='DCM',
    meaning='RNFL temporal sector thickness'
)

#: Nasal sector RNFL thickness. DCM 131268.
RNFLNasalThickness: CodedConcept = CodedConcept(
    value='131268', scheme_designator='DCM',
    meaning='RNFL nasal sector thickness'
)

#: Nasal-superior sector RNFL thickness. DCM 131269.
RNFLNasalSuperiorThickness: CodedConcept = CodedConcept(
    value='131269', scheme_designator='DCM',
    meaning='RNFL nasal-superior sector thickness'
)

#: Nasal-inferior sector RNFL thickness. DCM 131270.
RNFLNasalInferiorThickness: CodedConcept = CodedConcept(
    value='131270', scheme_designator='DCM',
    meaning='RNFL nasal-inferior sector thickness'
)

#: Temporal-inferior sector RNFL thickness. DCM 131271.
RNFLTemporalInferiorThickness: CodedConcept = CodedConcept(
    value='131271', scheme_designator='DCM',
    meaning='RNFL temporal-inferior sector thickness'
)

#: Temporal-superior sector RNFL thickness. DCM 131272.
RNFLTemporalSuperiorThickness: CodedConcept = CodedConcept(
    value='131272', scheme_designator='DCM',
    meaning='RNFL temporal-superior sector thickness'
)

#: Symmetry between right and left eye RNFL measurements, in percent.
#: Mandatory (TID 2123 row 7) whenever RNFL measurements are provided for
#: both eyes. DCM 131273.
RNFLSymmetry: CodedConcept = CodedConcept(
    value='131273', scheme_designator='DCM', meaning='RNFL symmetry'
)

# ---------------------------------------------------------------------------
# CID 4284 — RNFL Clockface Measurements.
# ---------------------------------------------------------------------------

#: Width of the circumpapillary circle scan region, in mm. Shared between
#: the sector (CID 4283) and clockface (CID 4284) roles. DCM 131274.
RetinalROIWidth: CodedConcept = CodedConcept(
    value='131274', scheme_designator='DCM', meaning='Retinal ROI width'
)

#: Mapping of clockface position (1-12) to its CodedConcept, DCM
#: 131276-131287, each an RNFL thickness measurement in µm at that
#: clockface position.
RNFL_CLOCKFACE_POSITION_CODES: dict[int, CodedConcept] = {
    position: CodedConcept(
        value=str(131275 + position), scheme_designator='DCM',
        meaning=f'RNFL clockface position {position} thickness'
    )
    for position in range(1, 13)
}

# ---------------------------------------------------------------------------
# CID 4285 — Macular Thickness Key Measurements.
#
# The nine ETDRS grid subfields and total volume use LOINC codes; average
# macular thickness uses a final DCM code (formerly Sup247 draft placeholder
# nnn250 under a private coding scheme).
# ---------------------------------------------------------------------------

#: Center point thickness (single foveal pixel/A-scan). LOINC 57108-3.
MacularCenterPointThickness: CodedConcept = CodedConcept(
    value='57108-3', scheme_designator='LN',
    meaning='Macular grid.center point thickness by OCT'
)

#: Central subfield (1 mm diameter circle; CMT). LOINC 57109-1.
MacularCenterSubfieldThickness: CodedConcept = CodedConcept(
    value='57109-1', scheme_designator='LN',
    meaning='Macular grid.center subfield thickness by OCT'
)

#: Inner superior subfield (1–3 mm annulus, superior sector). LOINC 57110-9.
MacularInnerSuperiorThickness: CodedConcept = CodedConcept(
    value='57110-9', scheme_designator='LN',
    meaning='Macular grid.inner superior subfield thickness by OCT'
)

#: Inner nasal subfield. LOINC 57111-7.
MacularInnerNasalThickness: CodedConcept = CodedConcept(
    value='57111-7', scheme_designator='LN',
    meaning='Macular grid.inner nasal subfield thickness by OCT'
)

#: Inner inferior subfield. LOINC 57112-5.
MacularInnerInferiorThickness: CodedConcept = CodedConcept(
    value='57112-5', scheme_designator='LN',
    meaning='Macular grid.inner inferior subfield thickness by OCT'
)

#: Inner temporal subfield. LOINC 57113-3.
MacularInnerTemporalThickness: CodedConcept = CodedConcept(
    value='57113-3', scheme_designator='LN',
    meaning='Macular grid.inner temporal subfield thickness by OCT'
)

#: Outer superior subfield (3–6 mm annulus, superior sector). LOINC 57114-1.
MacularOuterSuperiorThickness: CodedConcept = CodedConcept(
    value='57114-1', scheme_designator='LN',
    meaning='Macular grid.outer superior subfield thickness by OCT'
)

#: Outer nasal subfield. LOINC 57115-8.
MacularOuterNasalThickness: CodedConcept = CodedConcept(
    value='57115-8', scheme_designator='LN',
    meaning='Macular grid.outer nasal subfield thickness by OCT'
)

#: Outer inferior subfield. LOINC 57116-6.
MacularOuterInferiorThickness: CodedConcept = CodedConcept(
    value='57116-6', scheme_designator='LN',
    meaning='Macular grid.outer inferior subfield thickness by OCT'
)

#: Outer temporal subfield. LOINC 57117-4.
MacularOuterTemporalThickness: CodedConcept = CodedConcept(
    value='57117-4', scheme_designator='LN',
    meaning='Macular grid.outer temporal subfield thickness by OCT'
)

#: Total macular volume within the 6 mm ETDRS circle, in µL. LOINC 57118-2.
MacularTotalVolume: CodedConcept = CodedConcept(
    value='57118-2', scheme_designator='LN',
    meaning='Macular grid.total volume by OCT'
)

#: Average macular thickness over the full ETDRS grid, in µm. DCM 131255.
AverageMacularThickness: CodedConcept = CodedConcept(
    value='131255', scheme_designator='DCM',
    meaning='Average macular thickness'
)


# ---------------------------------------------------------------------------
# TID 2120 — Ophthalmology Measurements Group
# ---------------------------------------------------------------------------

class OphthalmologyMeasurementsGroup(MeasurementsAndQualitativeEvaluations):
    """:dcm:`TID 2120 <part16/sect_TID_2120.html>`
    Ophthalmology Measurements Group

    Type: Extensible · Order: Non-Significant · Root: No

    Sub-template that wraps a set of ophthalmic measurements for a single
    eye. Per the standard, TID 2120 is a proper subset of
    :dcm:`TID 1501 <part16/chapter_A.html#sect_TID_1501>`
    *Measurement and Qualitative Evaluation Group* with the finding site
    constrained to ``EV (81745001, SCT, "Eye")`` and laterality mandatory,
    so this class subclasses
    :class:`~highdicom.sr.MeasurementsAndQualitativeEvaluations` directly
    rather than reimplementing shared TID 1501 behavior (tracking identifier,
    source images, etc.).

    This template is invoked by the root templates
    :class:`CircumpapillaryRNFLKeyMeasurements` (TID 2123, via
    :class:`RNFLSectorMeasurementsGroup` and
    :class:`RNFLClockfaceMeasurementsGroup`) and
    :class:`MacularThicknessKeyMeasurements` (TID 2124, via
    :class:`MacularMeasurementsGroup`). It remains directly usable for other
    TID 2120 invocations not covered by those specializations.
    """

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        laterality: CodedConcept | Code,
        measurements: Sequence[Measurement] | None = None,
        finding_site: CodedConcept | Code | None = None,
        topographical_modifier: CodedConcept | Code | None = None,
        referenced_real_world_value_map: RealWorldValueMap | None = None,
        time_point_context: TimePointContext | None = None,
        finding_type: CodedConcept | Code | None = None,
        method: CodedConcept | Code | None = None,
        algorithm_id: AlgorithmIdentification | None = None,
        session: str | None = None,
        qualitative_evaluations: Sequence[QualitativeEvaluation] | None = None,
        finding_category: CodedConcept | Code | None = None,
        source_images: Sequence[SourceImageForMeasurementGroup] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements.
        laterality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Laterality of the eye measured. See
            :dcm:`CID 247 <part16/sect_CID_247.html>` "Laterality" for
            options; typically ``codes.cid247.Right`` or
            ``codes.cid247.Left``. Encoded as a modifier of the Finding
            Site item.
        measurements: Union[Sequence[highdicom.sr.Measurement], None], optional
            One or more :class:`~highdicom.sr.Measurement` instances
            (TID 300), each encoding a single numeric ophthalmic finding
            (e.g. average RNFL thickness, or a macular subfield thickness).
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Specific anatomic site within the eye. See
            :dcm:`CID 4209 <part16/sect_CID_4209.html>`
            "Ophthalmic Anatomic Location" for options. Defaults to
            ``codes.cid4209.Eye`` (``EV (81745001, SCT, "Eye")``), which is
            the value-set constraint imposed by the standard.
        topographical_modifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded modifier of the finding site.
        referenced_real_world_value_map: Union[highdicom.sr.RealWorldValueMap, None], optional
            Referenced real world value map for the eye.
        time_point_context: Union[highdicom.sr.TimePointContext, None], optional
            Description of the time point context.
        finding_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of observed finding.
        method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded measurement method.
        algorithm_id: Union[highdicom.sr.AlgorithmIdentification, None], optional
            Identification of algorithm used for making measurements.
        session: Union[str, None], optional
            Description of the session.
        qualitative_evaluations: Union[Sequence[highdicom.sr.QualitativeEvaluation], None], optional
            Coded name-value pairs that describe qualitative evaluations.
        finding_category: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Category of observed finding.
        source_images: Union[Sequence[highdicom.sr.SourceImageForMeasurementGroup], None], optional
            Images that were the source of the measurements. If not
            provided, all images listed in the document tree of the
            containing SR document are assumed to be source images.

        """  # noqa: E501
        if finding_site is None:
            finding_site = codes.cid4209.Eye

        finding_sites = [
            FindingSite(
                anatomic_location=finding_site,
                laterality=laterality,
                topographical_modifier=topographical_modifier,
            )
        ]

        super().__init__(
            tracking_identifier=tracking_identifier,
            referenced_real_world_value_map=referenced_real_world_value_map,
            time_point_context=time_point_context,
            finding_type=finding_type,
            method=method,
            algorithm_id=algorithm_id,
            finding_sites=finding_sites,
            session=session,
            measurements=measurements,
            qualitative_evaluations=qualitative_evaluations,
            finding_category=finding_category,
            source_images=source_images,
        )
        # The base class hardcodes template_id='1501' since it implements
        # TID 1501. This container instead invokes TID 2120, which imposes
        # additional constraints (mandatory laterality, eye-only finding
        # site) beyond what TID 1501 requires, so the Content Template
        # Sequence must identify it as such.
        self[0].ContentTemplateSequence[0].TemplateIdentifier = '2120'


def _laterality_of(group: OphthalmologyMeasurementsGroup) -> CodedConcept | None:
    sites = group.finding_sites
    if not sites:
        return None
    return sites[0].laterality


# ---------------------------------------------------------------------------
# TID 2120, sector role — invoked by TID 2123 Row 5
# ---------------------------------------------------------------------------

class RNFLSectorMeasurementsGroup(OphthalmologyMeasurementsGroup):
    """TID 2120 Ophthalmology Measurements Group, invoked in the "sector"
    role of :dcm:`TID 2123 <part16/sect_TID_2123.html>` Row 5: a retinal
    sector method (:dcm:`CID 4282 <part16/sect_CID_4282.html>`) with the
    mandatory Retinal ROI width measurement plus optional
    :dcm:`CID 4283 <part16/sect_CID_4283.html>` sector thickness values.
    """

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        laterality: CodedConcept | Code,
        sector_method: CodedConcept | Code,
        retinal_roi_width: float,
        average: float | None = None,
        inferior: float | None = None,
        superior: float | None = None,
        temporal: float | None = None,
        nasal: float | None = None,
        nasal_superior: float | None = None,
        nasal_inferior: float | None = None,
        temporal_inferior: float | None = None,
        temporal_superior: float | None = None,
        finding_site: CodedConcept | Code | None = None,
        topographical_modifier: CodedConcept | Code | None = None,
        source_images: Sequence[SourceImageForMeasurementGroup] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements.
        laterality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Laterality of the eye measured (:dcm:`CID 247 <part16/sect_CID_247.html>`).
        sector_method: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Retinal sector method used, one of the
            :dcm:`CID 4282 <part16/sect_CID_4282.html>` codes exposed as
            module-level constants: :data:`SemicircularSectors`,
            :data:`QuadrantSectors`, :data:`SNITRectangularSectors`,
            :data:`GarwayHeathSectors`, :data:`QuadrantOctantSectors`.
        retinal_roi_width: float
            Width of the circumpapillary circle scan region, in mm.
            Mandatory per Row 5.
        average: Union[float, None], optional
            Average circumpapillary RNFL thickness, in µm.
        inferior: Union[float, None], optional
            Inferior sector RNFL thickness, in µm.
        superior: Union[float, None], optional
            Superior sector RNFL thickness, in µm.
        temporal: Union[float, None], optional
            Temporal sector RNFL thickness, in µm.
        nasal: Union[float, None], optional
            Nasal sector RNFL thickness, in µm.
        nasal_superior: Union[float, None], optional
            Nasal-superior sector RNFL thickness, in µm.
        nasal_inferior: Union[float, None], optional
            Nasal-inferior sector RNFL thickness, in µm.
        temporal_inferior: Union[float, None], optional
            Temporal-inferior sector RNFL thickness, in µm.
        temporal_superior: Union[float, None], optional
            Temporal-superior sector RNFL thickness, in µm.
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Specific anatomic site within the eye. Defaults to
            ``codes.cid4209.Eye``.
        topographical_modifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded modifier of the finding site.
        source_images: Union[Sequence[highdicom.sr.SourceImageForMeasurementGroup], None], optional
            Images that were the source of the measurements.

        Raises
        ------
        ValueError
            If ``sector_method`` is not one of the CID 4282 codes.

        """  # noqa: E501
        sector_method = CodedConcept.from_code(sector_method)
        if sector_method not in _SECTOR_METHODS:
            raise ValueError(
                'Argument "sector_method" must be one of the CID 4282 '
                'Retinal Sector Methods codes (e.g. GarwayHeathSectors, '
                'QuadrantSectors).'
            )

        measurements = [
            Measurement(
                name=RetinalROIWidth,
                value=retinal_roi_width,
                unit=codes.UCUM.Millimeter,
            ),
        ]
        optional_measurements = {
            RNFLAverageThickness: average,
            RNFLInferiorThickness: inferior,
            RNFLSuperiorThickness: superior,
            RNFLTemporalThickness: temporal,
            RNFLNasalThickness: nasal,
            RNFLNasalSuperiorThickness: nasal_superior,
            RNFLNasalInferiorThickness: nasal_inferior,
            RNFLTemporalInferiorThickness: temporal_inferior,
            RNFLTemporalSuperiorThickness: temporal_superior,
        }
        for name, value in optional_measurements.items():
            if value is not None:
                measurements.append(
                    Measurement(
                        name=name, value=value, unit=codes.UCUM.Micrometer
                    )
                )

        super().__init__(
            tracking_identifier=tracking_identifier,
            laterality=laterality,
            measurements=measurements,
            finding_site=finding_site,
            topographical_modifier=topographical_modifier,
            method=sector_method,
            source_images=source_images,
        )


# ---------------------------------------------------------------------------
# TID 2120, clockface role — invoked by TID 2123 Row 6
# ---------------------------------------------------------------------------

class RNFLClockfaceMeasurementsGroup(OphthalmologyMeasurementsGroup):
    """TID 2120 Ophthalmology Measurements Group, invoked in the
    "clockface" role of :dcm:`TID 2123 <part16/sect_TID_2123.html>` Row 6:
    the fixed :data:`RNFLClockfaceMethod` with
    :dcm:`CID 4284 <part16/sect_CID_4284.html>` clockface-position
    thickness values.
    """

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        laterality: CodedConcept | Code,
        clockface_measurements: Mapping[int, float],
        retinal_roi_width: float | None = None,
        finding_site: CodedConcept | Code | None = None,
        topographical_modifier: CodedConcept | Code | None = None,
        source_images: Sequence[SourceImageForMeasurementGroup] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements.
        laterality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Laterality of the eye measured (:dcm:`CID 247 <part16/sect_CID_247.html>`).
        clockface_measurements: Mapping[int, float]
            Mapping of clockface position (1-12, per CID 4284) to the RNFL
            thickness in µm measured at that position. Positions not
            measured may be omitted; at least one must be provided.
        retinal_roi_width: Union[float, None], optional
            Width of the circumpapillary circle scan region, in mm.
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Specific anatomic site within the eye. Defaults to
            ``codes.cid4209.Eye``.
        topographical_modifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded modifier of the finding site.
        source_images: Union[Sequence[highdicom.sr.SourceImageForMeasurementGroup], None], optional
            Images that were the source of the measurements.

        Raises
        ------
        ValueError
            If ``clockface_measurements`` is empty or contains a key
            outside 1-12.

        """  # noqa: E501
        if not clockface_measurements:
            raise ValueError(
                'Argument "clockface_measurements" must contain at least '
                'one clockface position.'
            )
        for position in clockface_measurements:
            if position not in RNFL_CLOCKFACE_POSITION_CODES:
                raise ValueError(
                    f'Clockface position {position!r} is invalid; keys of '
                    '"clockface_measurements" must be integers from 1 to '
                    '12 (CID 4284).'
                )

        measurements = []
        if retinal_roi_width is not None:
            measurements.append(
                Measurement(
                    name=RetinalROIWidth,
                    value=retinal_roi_width,
                    unit=codes.UCUM.Millimeter,
                )
            )
        for position, value in clockface_measurements.items():
            measurements.append(
                Measurement(
                    name=RNFL_CLOCKFACE_POSITION_CODES[position],
                    value=value,
                    unit=codes.UCUM.Micrometer,
                )
            )

        super().__init__(
            tracking_identifier=tracking_identifier,
            laterality=laterality,
            measurements=measurements,
            finding_site=finding_site,
            topographical_modifier=topographical_modifier,
            method=RNFLClockfaceMethod,
            source_images=source_images,
        )


# ---------------------------------------------------------------------------
# TID 2123 — Circumpapillary Retinal Nerve Fiber Layer Key Measurements
# ---------------------------------------------------------------------------

class CircumpapillaryRNFLKeyMeasurements(Template):
    """:dcm:`TID 2123 <part16/sect_TID_2123.html>`
    Circumpapillary Retinal Nerve Fiber Layer Key Measurements

    Type: Extensible · Order: Non-Significant · Root: Yes

    Root template for circumpapillary retinal nerve fiber layer (RNFL)
    thickness measurements obtained by ophthalmic tomography (OCT). Wraps
    one or two :class:`RNFLSectorMeasurementsGroup` instances (Row 5), one
    or two :class:`RNFLClockfaceMeasurementsGroup` instances (Row 6), or
    both, plus an RNFL-symmetry measurement (Row 7) that is required
    whenever both eyes are represented across the two group lists combined.

    Usage example::

        from highdicom.sr.templates import AlgorithmIdentification, TrackingIdentifier
        from highdicom.sr.templates.tid2120 import (
            CircumpapillaryRNFLKeyMeasurements,
            GarwayHeathSectors,
            RNFLSectorMeasurementsGroup,
        )
        from pydicom.sr.codedict import codes

        algo = AlgorithmIdentification(name='Revo FC130', version='1.0')
        group = RNFLSectorMeasurementsGroup(
            tracking_identifier=TrackingIdentifier(identifier='RNFL-OD'),
            laterality=codes.cid247.Right,
            sector_method=GarwayHeathSectors,
            retinal_roi_width=3.4,
            average=121.0,
        )
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=algo,
            sector_measurement_groups=[group],
        )
    """

    def __init__(
        self,
        algorithm_id: AlgorithmIdentification,
        sector_measurement_groups: (
            Sequence[RNFLSectorMeasurementsGroup] | None
        ) = None,
        clockface_measurement_groups: (
            Sequence[RNFLClockfaceMeasurementsGroup] | None
        ) = None,
        rnfl_symmetry: float | None = None,
        language_of_content_item_and_descendants: (
            LanguageOfContentItemAndDescendants | None
        ) = None,
    ) -> None:
        """
        Parameters
        ----------
        algorithm_id: highdicom.sr.AlgorithmIdentification
            Identification of the algorithm used to produce the RNFL
            measurements (TID 4019). Mandatory.
        sector_measurement_groups: Union[Sequence[highdicom.sr.RNFLSectorMeasurementsGroup], None], optional
            One or two :class:`RNFLSectorMeasurementsGroup` instances, one
            per eye (Row 5).
        clockface_measurement_groups: Union[Sequence[highdicom.sr.RNFLClockfaceMeasurementsGroup], None], optional
            One or two :class:`RNFLClockfaceMeasurementsGroup` instances,
            one per eye (Row 6).
        rnfl_symmetry: Union[float, None], optional
            Symmetry between right and left eye RNFL measurements, as a
            percentage (Row 7). Required when ``sector_measurement_groups``
            and/or ``clockface_measurement_groups`` together represent both
            eyes; must not be provided otherwise.
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all SR content items. Defaults to
            English (``en-US``, RFC 5646) when not provided.

        Raises
        ------
        TypeError
            If ``algorithm_id`` is not an
            :class:`~highdicom.sr.AlgorithmIdentification`, or if any group
            does not have the expected type.
        ValueError
            If neither group list is provided, if either contains more than
            two items, or if ``rnfl_symmetry`` is inconsistent with whether
            both eyes are represented.

        """  # noqa: E501
        sector_measurement_groups = list(sector_measurement_groups or [])
        clockface_measurement_groups = list(clockface_measurement_groups or [])

        if not sector_measurement_groups and not clockface_measurement_groups:
            raise ValueError(
                'At least one of "sector_measurement_groups" or '
                '"clockface_measurement_groups" must be provided.'
            )
        if len(sector_measurement_groups) > 2:
            raise ValueError(
                'Argument "sector_measurement_groups" must contain at most '
                'two items (one per eye).'
            )
        if len(clockface_measurement_groups) > 2:
            raise ValueError(
                'Argument "clockface_measurement_groups" must contain at '
                'most two items (one per eye).'
            )
        for g in sector_measurement_groups:
            if not isinstance(g, RNFLSectorMeasurementsGroup):
                raise TypeError(
                    'Each item of "sector_measurement_groups" must have '
                    'type RNFLSectorMeasurementsGroup.'
                )
        for g in clockface_measurement_groups:
            if not isinstance(g, RNFLClockfaceMeasurementsGroup):
                raise TypeError(
                    'Each item of "clockface_measurement_groups" must have '
                    'type RNFLClockfaceMeasurementsGroup.'
                )
        if not isinstance(algorithm_id, AlgorithmIdentification):
            raise TypeError(
                'Argument "algorithm_id" must have type '
                'AlgorithmIdentification.'
            )

        all_groups = [*sector_measurement_groups, *clockface_measurement_groups]
        lateralities = [_laterality_of(g) for g in all_groups]
        is_bilateral = (
            any(lat == codes.cid247.Right for lat in lateralities) and
            any(lat == codes.cid247.Left for lat in lateralities)
        )
        if is_bilateral and rnfl_symmetry is None:
            raise ValueError(
                'Argument "rnfl_symmetry" is required (TID 2123 Row 7) '
                'when RNFL measurements are provided for both eyes.'
            )
        if not is_bilateral and rnfl_symmetry is not None:
            raise ValueError(
                'Argument "rnfl_symmetry" is only applicable (TID 2123 '
                'Row 7) when RNFL measurements are provided for both eyes.'
            )

        item = ContainerContentItem(
            name=_CONCEPT_CIRCUMPAPILLARY_RNFL_KEY,
            template_id='2123',
        )
        item.ContentSequence = ContentSequence()

        if language_of_content_item_and_descendants is None:
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(language_of_content_item_and_descendants)

        item.ContentSequence.extend(algorithm_id)

        for group in sector_measurement_groups:
            item.ContentSequence.extend(group)
        for group in clockface_measurement_groups:
            item.ContentSequence.extend(group)

        if rnfl_symmetry is not None:
            item.ContentSequence.extend(
                Measurement(
                    name=RNFLSymmetry,
                    value=rnfl_symmetry,
                    unit=codes.UCUM.Percent,
                )
            )

        super().__init__([item], is_root=True)


# ---------------------------------------------------------------------------
# TID 2120, macular role — invoked by TID 2124 Row 5
# ---------------------------------------------------------------------------

class MacularMeasurementsGroup(OphthalmologyMeasurementsGroup):
    """TID 2120 Ophthalmology Measurements Group, invoked by
    :dcm:`TID 2124 <part16/sect_TID_2124.html>` Row 5:
    :dcm:`CID 4285 <part16/sect_CID_4285.html>` Macular Thickness Key
    Measurements (ETDRS grid subfields, total volume, average thickness).
    """

    def __init__(
        self,
        tracking_identifier: TrackingIdentifier,
        laterality: CodedConcept | Code,
        center_point: float | None = None,
        center_subfield: float | None = None,
        inner_superior: float | None = None,
        inner_nasal: float | None = None,
        inner_inferior: float | None = None,
        inner_temporal: float | None = None,
        outer_superior: float | None = None,
        outer_nasal: float | None = None,
        outer_inferior: float | None = None,
        outer_temporal: float | None = None,
        total_volume: float | None = None,
        average_thickness: float | None = None,
        finding_site: CodedConcept | Code | None = None,
        topographical_modifier: CodedConcept | Code | None = None,
        source_images: Sequence[SourceImageForMeasurementGroup] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tracking_identifier: highdicom.sr.TrackingIdentifier
            Identifier for tracking measurements.
        laterality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Laterality of the eye measured (:dcm:`CID 247 <part16/sect_CID_247.html>`).
        center_point: Union[float, None], optional
            Center point thickness (single foveal pixel/A-scan), in µm.
        center_subfield: Union[float, None], optional
            Central subfield thickness (CMT; 1 mm diameter circle), in µm.
        inner_superior: Union[float, None], optional
            Inner superior subfield thickness, in µm.
        inner_nasal: Union[float, None], optional
            Inner nasal subfield thickness, in µm.
        inner_inferior: Union[float, None], optional
            Inner inferior subfield thickness, in µm.
        inner_temporal: Union[float, None], optional
            Inner temporal subfield thickness, in µm.
        outer_superior: Union[float, None], optional
            Outer superior subfield thickness, in µm.
        outer_nasal: Union[float, None], optional
            Outer nasal subfield thickness, in µm.
        outer_inferior: Union[float, None], optional
            Outer inferior subfield thickness, in µm.
        outer_temporal: Union[float, None], optional
            Outer temporal subfield thickness, in µm.
        total_volume: Union[float, None], optional
            Total macular volume within the 6 mm ETDRS circle, in µL.
        average_thickness: Union[float, None], optional
            Average macular thickness over the full ETDRS grid, in µm.
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Specific anatomic site within the eye. Defaults to
            ``codes.cid4209.Eye``.
        topographical_modifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Coded modifier of the finding site.
        source_images: Union[Sequence[highdicom.sr.SourceImageForMeasurementGroup], None], optional
            Images that were the source of the measurements.

        Raises
        ------
        ValueError
            If no measurement value is provided.

        """  # noqa: E501
        micrometer_measurements = {
            MacularCenterPointThickness: center_point,
            MacularCenterSubfieldThickness: center_subfield,
            MacularInnerSuperiorThickness: inner_superior,
            MacularInnerNasalThickness: inner_nasal,
            MacularInnerInferiorThickness: inner_inferior,
            MacularInnerTemporalThickness: inner_temporal,
            MacularOuterSuperiorThickness: outer_superior,
            MacularOuterNasalThickness: outer_nasal,
            MacularOuterInferiorThickness: outer_inferior,
            MacularOuterTemporalThickness: outer_temporal,
            AverageMacularThickness: average_thickness,
        }
        if (
            all(v is None for v in micrometer_measurements.values()) and
            total_volume is None
        ):
            raise ValueError(
                'At least one macular thickness measurement must be '
                'provided.'
            )

        measurements = [
            Measurement(name=name, value=value, unit=codes.UCUM.Micrometer)
            for name, value in micrometer_measurements.items()
            if value is not None
        ]
        if total_volume is not None:
            measurements.append(
                Measurement(
                    name=MacularTotalVolume,
                    value=total_volume,
                    unit=UCUM_MICROLITER,
                )
            )

        super().__init__(
            tracking_identifier=tracking_identifier,
            laterality=laterality,
            measurements=measurements,
            finding_site=finding_site,
            topographical_modifier=topographical_modifier,
            source_images=source_images,
        )


# ---------------------------------------------------------------------------
# TID 2124 — Macular Thickness Key Measurements
# ---------------------------------------------------------------------------

class MacularThicknessKeyMeasurements(Template):
    """:dcm:`TID 2124 <part16/sect_TID_2124.html>`
    Macular Thickness Key Measurements

    Type: Extensible · Order: Non-Significant · Root: Yes

    Root template for macular thickness measurements obtained by ophthalmic
    tomography (OCT) using the Early Treatment of Diabetic Retinopathy Study
    (ETDRS) grid. Wraps one or two :class:`MacularMeasurementsGroup`
    instances, one per eye.

    Usage example::

        from highdicom.sr.templates import AlgorithmIdentification, TrackingIdentifier
        from highdicom.sr.templates.tid2120 import (
            MacularMeasurementsGroup,
            MacularThicknessKeyMeasurements,
        )
        from pydicom.sr.codedict import codes

        algo = AlgorithmIdentification(name='Cirrus HD-OCT', version='11.0')
        group = MacularMeasurementsGroup(
            tracking_identifier=TrackingIdentifier(identifier='Macula-OD'),
            laterality=codes.cid247.Right,
            center_subfield=288.49,
        )
        report = MacularThicknessKeyMeasurements(
            algorithm_id=algo,
            measurement_groups=[group],
        )
    """

    def __init__(
        self,
        algorithm_id: AlgorithmIdentification,
        measurement_groups: Sequence[MacularMeasurementsGroup],
        language_of_content_item_and_descendants: (
            LanguageOfContentItemAndDescendants | None
        ) = None,
    ) -> None:
        """
        Parameters
        ----------
        algorithm_id: highdicom.sr.AlgorithmIdentification
            Identification of the algorithm used to produce the macular
            thickness measurements (TID 4019). Mandatory.
        measurement_groups: Sequence[highdicom.sr.MacularMeasurementsGroup]
            One or two :class:`MacularMeasurementsGroup` instances
            (TID 2120), one per eye. Mandatory.
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all SR content items. Defaults to
            English (``en-US``) when not provided.

        Raises
        ------
        TypeError
            If ``algorithm_id`` is not an
            :class:`~highdicom.sr.AlgorithmIdentification`, or if any
            element of ``measurement_groups`` is not a
            :class:`MacularMeasurementsGroup`.
        ValueError
            If ``measurement_groups`` is empty or contains more than two
            items.

        """  # noqa: E501

        if not measurement_groups:
            raise ValueError(
                "Argument 'measurement_groups' must contain at least one item."
            )
        if len(measurement_groups) > 2:
            raise ValueError(
                "Argument 'measurement_groups' must contain at most two items "
                "(one per eye)."
            )
        for g in measurement_groups:
            if not isinstance(g, MacularMeasurementsGroup):
                raise TypeError(
                    'Each item of "measurement_groups" must have type '
                    'MacularMeasurementsGroup.'
                )
        if not isinstance(algorithm_id, AlgorithmIdentification):
            raise TypeError(
                'Argument "algorithm_id" must have type '
                'AlgorithmIdentification.'
            )

        item = ContainerContentItem(
            name=_CONCEPT_MACULAR_THICKNESS_KEY,
            template_id='2124',
        )
        item.ContentSequence = ContentSequence()

        if language_of_content_item_and_descendants is None:
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(language_of_content_item_and_descendants)

        item.ContentSequence.extend(algorithm_id)

        for group in measurement_groups:
            item.ContentSequence.extend(group)

        super().__init__([item], is_root=True)
