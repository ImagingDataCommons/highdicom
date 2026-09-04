"""DICOM SR templates for Supplement 247 — Eyecare Measurement Templates.

Implements TIDs 2120, 2123, and 2124 as ratified in DICOM PS3.16 2025b
(formerly circulated in Supplement 247 draft as TIDs 6001, 6004, and 6005;
Supplement 247 has since been incorporated into the standard and its
templates renumbered).

Reference: :dcm:`TID 2120 <part16/sect_TID_2120.html>`,
:dcm:`TID 2123 <part16/sect_TID_2123.html>`,
:dcm:`TID 2124 <part16/sect_TID_2124.html>`

Scope of this implementation
-----------------------------
TID 2123 *Circumpapillary Retinal Nerve Fiber Layer Key Measurements* is
defined by the ratified standard as invoking TID 2120 in two structurally
distinct roles: a "sector" group (:dcm:`CID 4282 <part16/sect_CID_4282.html>`
sector methods, e.g. Garway-Heath or quadrant sectors) and a separate
"clockface" group (12 clockface-position measurements), plus an optional
bilateral RNFL-symmetry measurement. This implementation does not yet
distinguish these roles: :class:`CircumpapillaryRNFLKeyMeasurements` accepts
a flat list of :class:`OphthalmologyMeasurementsGroup` instances (one or two,
per eye), same as :class:`MacularThicknessKeyMeasurements`. Enforcing the
sector/clockface split (and validating measurements against the CID that
matches each role) is deferred to a follow-up that also addresses
constructor-level conformance enforcement more broadly.
"""

from __future__ import annotations

from collections.abc import Sequence

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

    This template is invoked 1–2 times (once per eye) by the root templates
    :class:`CircumpapillaryRNFLKeyMeasurements` (TID 2123) and
    :class:`MacularThicknessKeyMeasurements` (TID 2124).
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


# ---------------------------------------------------------------------------
# TID 2123 — Circumpapillary Retinal Nerve Fiber Layer Key Measurements
# ---------------------------------------------------------------------------

class CircumpapillaryRNFLKeyMeasurements(Template):
    """:dcm:`TID 2123 <part16/sect_TID_2123.html>`
    Circumpapillary Retinal Nerve Fiber Layer Key Measurements

    Type: Extensible · Order: Non-Significant · Root: Yes

    Root template for circumpapillary retinal nerve fiber layer (RNFL)
    thickness measurements obtained by ophthalmic tomography (OCT).

    See the module docstring for the scope of this implementation relative
    to the ratified standard's sector/clockface group split, which is not
    yet enforced here.

    Usage example::

        from highdicom.sr.templates import (
            AlgorithmIdentification,
            CircumpapillaryRNFLKeyMeasurements,
            Measurement,
            OphthalmologyMeasurementsGroup,
            TrackingIdentifier,
        )
        from highdicom.sr.templates.tid2120 import RNFLAverageThickness
        from pydicom.sr.codedict import codes

        algo = AlgorithmIdentification(name='Revo FC130', version='1.0')
        meas = [
            Measurement(
                name=RNFLAverageThickness,
                value=121.0,
                unit=codes.UCUM.Micrometer,
            ),
        ]
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=TrackingIdentifier(identifier='RNFL-OD'),
            laterality=codes.cid247.Right,
            measurements=meas,
        )
        report = CircumpapillaryRNFLKeyMeasurements(
            algorithm_id=algo,
            measurement_groups=[group],
        )
    """

    def __init__(
        self,
        algorithm_id: AlgorithmIdentification,
        measurement_groups: Sequence[OphthalmologyMeasurementsGroup],
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
        measurement_groups: Sequence[highdicom.sr.OphthalmologyMeasurementsGroup]
            One or two :class:`OphthalmologyMeasurementsGroup` instances
            (TID 2120), one per eye. Two groups are used for bilateral
            studies (one per eye).
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all SR content items. Defaults to
            English (``en-US``, RFC 5646) when not provided.

        Raises
        ------
        TypeError
            If ``algorithm_id`` is not an
            :class:`~highdicom.sr.AlgorithmIdentification`, or if any
            element of ``measurement_groups`` is not an
            :class:`OphthalmologyMeasurementsGroup`.
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
            if not isinstance(g, OphthalmologyMeasurementsGroup):
                raise TypeError(
                    'Each item of "measurement_groups" must have type '
                    'OphthalmologyMeasurementsGroup.'
                )
        if not isinstance(algorithm_id, AlgorithmIdentification):
            raise TypeError(
                'Argument "algorithm_id" must have type '
                'AlgorithmIdentification.'
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

        for group in measurement_groups:
            item.ContentSequence.extend(group)

        super().__init__([item], is_root=True)


# ---------------------------------------------------------------------------
# TID 2124 — Macular Thickness Key Measurements
# ---------------------------------------------------------------------------

class MacularThicknessKeyMeasurements(Template):
    """:dcm:`TID 2124 <part16/sect_TID_2124.html>`
    Macular Thickness Key Measurements

    Type: Extensible · Order: Non-Significant · Root: Yes

    Root template for macular thickness measurements obtained by ophthalmic
    tomography (OCT) using the Early Treatment of Diabetic Retinopathy Study
    (ETDRS) grid.

    The nine ETDRS subfield concept names use their LOINC codes
    (``57108-3`` through ``57117-4``). Total macular volume uses
    ``LN 57118-2``; the unit must be ``UCUM_MICROLITER`` (``uL``). Average
    macular thickness uses DCM ``131255``.

    Usage example::

        from highdicom.sr.templates import (
            AlgorithmIdentification,
            MacularThicknessKeyMeasurements,
            Measurement,
            OphthalmologyMeasurementsGroup,
            TrackingIdentifier,
        )
        from highdicom.sr.templates.tid2120 import MacularCenterSubfieldThickness
        from pydicom.sr.codedict import codes

        algo = AlgorithmIdentification(name='Cirrus HD-OCT', version='11.0')
        meas = [
            Measurement(
                name=MacularCenterSubfieldThickness,
                value=288.49,
                unit=codes.UCUM.Micrometer,
            ),
        ]
        group = OphthalmologyMeasurementsGroup(
            tracking_identifier=TrackingIdentifier(identifier='Macula-OD'),
            laterality=codes.cid247.Right,
            measurements=meas,
        )
        report = MacularThicknessKeyMeasurements(
            algorithm_id=algo,
            measurement_groups=[group],
        )
    """

    def __init__(
        self,
        algorithm_id: AlgorithmIdentification,
        measurement_groups: Sequence[OphthalmologyMeasurementsGroup],
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
        measurement_groups: Sequence[highdicom.sr.OphthalmologyMeasurementsGroup]
            One or two :class:`OphthalmologyMeasurementsGroup` instances
            (TID 2120), one per eye. Each group carries the laterality
            and the individual macular thickness measurements. Mandatory.
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all SR content items. Defaults to
            English (``en-US``) when not provided.

        Raises
        ------
        TypeError
            If ``algorithm_id`` is not an
            :class:`~highdicom.sr.AlgorithmIdentification`, or if any
            element of ``measurement_groups`` is not an
            :class:`OphthalmologyMeasurementsGroup`.
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
            if not isinstance(g, OphthalmologyMeasurementsGroup):
                raise TypeError(
                    'Each item of "measurement_groups" must have type '
                    'OphthalmologyMeasurementsGroup.'
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
