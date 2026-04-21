"""DICOM SR templates for Supplement 247 — Eyecare Measurement Templates.

Implements TIDs 6001, 6004, and 6005 as defined in DICOM Supplement 247
"Eyecare Measurement Templates", incorporated into DICOM PS3.16 2025c.

Reference: DICOM PS3.16 Supplement 247
  https://dicom.nema.org/Dicom/News/January2025/docs/sups/sup247.pdf

Note on provisional concept codes
----------------------------------
Supplement 247 was balloted with several DCM concept codes shown as
``nnnXXX`` placeholders — the final numeric values had not yet been
assigned by DICOM WG-09 at time of publication.  Those codes are defined
here as private ``CodedConcept`` objects under scheme ``99OPHTHALMO``
(≤ 16 chars, required for DICOM VR SH) and are marked with
``# Sup247 draft — nnnXXX`` inline comments.

Codes pending a companion PR to pydicom (``pydicom.sr.codedict``):

Container concept names (99OPHTHALMO, draft placeholders):
  - ``nnn102`` — "RNFL Key Measurements"
  - ``nnn103`` — "Macular Thickness Key Measurements"

CID 42x3 RNFL Key Measurements (99OPHTHALMO):
  - ``nnn400`` – average thickness
  - ``nnn401–nnn404`` – inferior / superior / temporal / nasal quadrants
  - ``nnn406`` – retinal ROI radius

CID 42x4 Macular Thickness Key Measurements:
  - ``nnn250`` – average macular thickness  (99OPHTHALMO)
  - ``LN 57108-3`` through ``LN 57118-2`` – ETDRS grid (LOINC, stable,
    not yet exposed as convenience attributes in ``pydicom.sr.codedict``)

No ``coding_ophthalmology.py`` is created in this PR; per CPBridge guidance
in ImagingDataCommons/highdicom#406, new CIDs belong in a parallel pydicom PR.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import RelationshipTypeValues
from highdicom.sr.value_types import (
    CodeContentItem,
    ContainerContentItem,
    ContentSequence,
    TextContentItem,
    UIDRefContentItem,
)
from highdicom.sr.templates.tid1500 import (
    AlgorithmIdentification,
    DEFAULT_LANGUAGE,
    LanguageOfContentItemAndDescendants,
    Measurement,
    Template,
)

# ---------------------------------------------------------------------------
# Supplementary UCUM unit not yet in pydicom.sr.codedict
# ---------------------------------------------------------------------------
#: UCUM microliter — used for macular total volume (``LN 57118-2``).
#: Not exposed as a convenience attribute in pydicom 3.x; defined here until
#: pydicom adds it to ``codes.UCUM``.
UCUM_MICROLITER: Code = Code('uL', 'UCUM', 'microliter')

# ---------------------------------------------------------------------------
# Provisional concept codes — Supplement 247 draft (nnnXXX placeholders).
# ---------------------------------------------------------------------------
_SCHEME = '99OPHTHALMO'

# Container concept names for root templates TID 6004 and 6005.
# Sup247 draft: EV (nnn102, DCM, "RNFL Key Measurements") etc.
_CONCEPT_RNFL_KEY = CodedConcept(
    value='nnn102', scheme_designator=_SCHEME,
    meaning='RNFL Key Measurements'
)
_CONCEPT_MACULAR_THICKNESS_KEY = CodedConcept(
    value='nnn103', scheme_designator=_SCHEME,
    meaning='Macular Thickness Key Measurements'
)

# ---------------------------------------------------------------------------
# CID 42x3 — RNFL Key Measurements (Supplement 247 Table CID 42x3)
# All measurements in µm; unit: (um, UCUM, "um").
# Codes are Sup247 draft placeholders under 99OPHTHALMO.
# ---------------------------------------------------------------------------

#: Average circumpapillary RNFL thickness.  Sup247 draft nnn400.
RNFLAverageThickness: CodedConcept = CodedConcept(
    value='nnn400', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer average thickness'
)

#: Inferior quadrant RNFL thickness.  Sup247 draft nnn401.
RNFLInferiorThickness: CodedConcept = CodedConcept(
    value='nnn401', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer inferior thickness'
)

#: Superior quadrant RNFL thickness.  Sup247 draft nnn402.
RNFLSuperiorThickness: CodedConcept = CodedConcept(
    value='nnn402', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer superior thickness'
)

#: Temporal quadrant RNFL thickness.  Sup247 draft nnn403.
RNFLTemporalThickness: CodedConcept = CodedConcept(
    value='nnn403', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer temporal thickness'
)

#: Nasal quadrant RNFL thickness.  Sup247 draft nnn404.
RNFLNasalThickness: CodedConcept = CodedConcept(
    value='nnn404', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer nasal thickness'
)

#: Radius of the circumpapillary circle scan used for RNFL measurement, in mm.
#: Sup247 draft nnn406.
RetinalROIRadius: CodedConcept = CodedConcept(
    value='nnn406', scheme_designator=_SCHEME,
    meaning='Retinal ROI radius'
)

# ---------------------------------------------------------------------------
# CID 42x4 — Macular Thickness Key Measurements (Supplement 247 Table CID 42x4)
#
# The nine ETDRS grid subfields and total volume use LOINC codes that are
# finalised in DICOM 2025c.  The "average macular thickness" concept (nnn250)
# remains a Sup247 draft placeholder.
# ---------------------------------------------------------------------------

#: Center point thickness (single foveal pixel/A-scan).  LOINC 57108-3.
MacularCenterPointThickness: CodedConcept = CodedConcept(
    value='57108-3', scheme_designator='LN',
    meaning='Macular grid.center point thickness by OCT'
)

#: Central subfield (1 mm diameter circle; CMT).  LOINC 57109-1.
MacularCenterSubfieldThickness: CodedConcept = CodedConcept(
    value='57109-1', scheme_designator='LN',
    meaning='Macular grid.center subfield thickness by OCT'
)

#: Inner superior subfield (1–3 mm annulus, superior sector).  LOINC 57110-9.
MacularInnerSuperiorThickness: CodedConcept = CodedConcept(
    value='57110-9', scheme_designator='LN',
    meaning='Macular grid.inner superior subfield thickness by OCT'
)

#: Inner nasal subfield.  LOINC 57111-7.
MacularInnerNasalThickness: CodedConcept = CodedConcept(
    value='57111-7', scheme_designator='LN',
    meaning='Macular grid.inner nasal subfield thickness by OCT'
)

#: Inner inferior subfield.  LOINC 57112-5.
MacularInnerInferiorThickness: CodedConcept = CodedConcept(
    value='57112-5', scheme_designator='LN',
    meaning='Macular grid.inner inferior subfield thickness by OCT'
)

#: Inner temporal subfield.  LOINC 57113-3.
MacularInnerTemporalThickness: CodedConcept = CodedConcept(
    value='57113-3', scheme_designator='LN',
    meaning='Macular grid.inner temporal subfield thickness by OCT'
)

#: Outer superior subfield (3–6 mm annulus, superior sector).  LOINC 57114-1.
MacularOuterSuperiorThickness: CodedConcept = CodedConcept(
    value='57114-1', scheme_designator='LN',
    meaning='Macular grid.outer superior subfield thickness by OCT'
)

#: Outer nasal subfield.  LOINC 57115-8.
MacularOuterNasalThickness: CodedConcept = CodedConcept(
    value='57115-8', scheme_designator='LN',
    meaning='Macular grid.outer nasal subfield thickness by OCT'
)

#: Outer inferior subfield.  LOINC 57116-6.
MacularOuterInferiorThickness: CodedConcept = CodedConcept(
    value='57116-6', scheme_designator='LN',
    meaning='Macular grid.outer inferior subfield thickness by OCT'
)

#: Outer temporal subfield.  LOINC 57117-4.
MacularOuterTemporalThickness: CodedConcept = CodedConcept(
    value='57117-4', scheme_designator='LN',
    meaning='Macular grid.outer temporal subfield thickness by OCT'
)

#: Total macular volume within the 6 mm ETDRS circle, in µL.  LOINC 57118-2.
MacularTotalVolume: CodedConcept = CodedConcept(
    value='57118-2', scheme_designator='LN',
    meaning='Macular grid.total volume by OCT'
)

#: Average macular thickness over the full ETDRS grid, in µm.
#: Sup247 draft nnn250.
AverageMacularThickness: CodedConcept = CodedConcept(
    value='nnn250', scheme_designator=_SCHEME,
    meaning='Average macular thickness'
)


# ---------------------------------------------------------------------------
# TID 6001 (Sup247 TID 60x1) — Ophthalmology Measurements Group
# ---------------------------------------------------------------------------

class OphthalmologyMeasurementsGroup(Template):
    """:dcm:`TID 6001 <part16/chapter_A.html#sect_TID_6001>`
    Ophthalmology Measurements Group

    Type: Extensible · Order: Non-Significant · Root: No

    Sub-template that wraps a set of ophthalmic measurements for a single
    eye.  Structurally equivalent to TID 1501
    *Measurement and Qualitative Evaluation Group* but specialised for
    ophthalmology: the finding site is constrained to
    ``EV (81745001, SCT, "Eye")`` and laterality is mandatory.

    Per Supplement 247 Table TID 60x1, the content item hierarchy is::

        Row 1  CONTAINER (125007, DCM, "Measurement Group")         [M]
        Row 2   > Finding Site (363698007, SCT)  →  Eye (81745001)  [M]
        Row 3  >>   Laterality (272741003, SCT)  →  CID 244         [M]
        Row 6   > Tracking Identifier (112039, DCM)                  [U]
        Row 8   > Measurement (TID 300) × 1-n                       [MC]

    Note on HTML anchor: the URL fragment ``#sect_TID_6001`` will resolve
    once DICOM PS3.16 is updated to incorporate Supplement 247.  Until then
    refer to the supplement PDF directly:
    https://dicom.nema.org/Dicom/News/January2025/docs/sups/sup247.pdf

    This template is invoked 1–2 times (once per eye) by the root templates
    :class:`CircumpapillaryRNFLKeyMeasurements` (TID 6004) and
    :class:`MacularThicknessKeyMeasurements` (TID 6005).
    """

    def __init__(
        self,
        laterality: CodedConcept | Code,
        measurements: Sequence[Measurement],
        finding_site: CodedConcept | Code | None = None,
        tracking_identifier: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        laterality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Laterality of the eye measured.  See
            :dcm:`CID 244 <part16/sect_CID_244.html>` "Laterality
            Left-Right Only" for options; typically ``codes.cid244.Right``
            or ``codes.cid244.Left``.  Encoded as a HAS_CONCEPT_MOD of
            the Finding Site item (Table TID 60x1, row 3, double-indent).
        measurements: Sequence[highdicom.sr.Measurement]
            One or more :class:`~highdicom.sr.Measurement` instances
            (TID 300), each encoding a single numeric ophthalmic finding
            (e.g. average RNFL thickness from CID 42x3, or a macular
            subfield thickness from CID 42x4).  Corresponds to row 8 of
            TID 60x1 — mandatory when the invoking template supplies a
            non-empty ``$Measurement`` parameter.
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Specific anatomic site within the eye.  See
            :dcm:`CID 4209 <part16/sect_CID_4209.html>`
            "Ophthalmic Anatomic Location" for options.  Defaults to
            ``codes.cid4209.Eye`` (``EV (81745001, SCT, "Eye")``), which is
            the value-set constraint imposed by Supplement 247 row 2.
        tracking_identifier: str, optional
            Free-text identifier for tracking this measurement group across
            studies or analyses.  Encoded as
            ``(112039, DCM, "Tracking Identifier")`` with relationship type
            HAS_OBS_CONTEXT (row 6 of TID 60x1).

        """  # noqa: E501
        super().__init__()

        if finding_site is None:
            finding_site = codes.cid4209.Eye

        if not measurements:
            raise ValueError(
                "Argument 'measurements' must contain at least one item."
            )
        for meas in measurements:
            if not isinstance(meas, Measurement):
                raise TypeError(
                    'Each item of "measurements" must have type Measurement.'
                )

        item = ContainerContentItem(
            name=codes.DCM.MeasurementGroup,
            template_id='6001',
            relationship_type=RelationshipTypeValues.CONTAINS,
        )
        item.ContentSequence = ContentSequence()

        # Row 2 — Finding Site (M, HAS_CONCEPT_MOD of container).
        # Value set constraint: EV (81745001, SCT, "Eye").
        site_item = CodeContentItem(
            name=CodedConcept(
                value='363698007',
                scheme_designator='SCT',
                meaning='Finding Site',
            ),
            value=finding_site,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD,
        )
        # Row 3 (>>) — Laterality is HAS_CONCEPT_MOD of the Finding Site
        # item, NOT of the container.  The double-indent (>>) in Table TID
        # 60x1 encodes this nested modifier relationship.
        site_item.ContentSequence = ContentSequence()
        lat_item = CodeContentItem(
            name=CodedConcept(
                value='272741003',
                scheme_designator='SCT',
                meaning='Laterality',
            ),
            value=laterality,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD,
        )
        site_item.ContentSequence.append(lat_item)
        item.ContentSequence.append(site_item)

        # Row 6 — Tracking Identifier (U, HAS_OBS_CONTEXT).
        if tracking_identifier is not None:
            tracking_item = TextContentItem(
                name=CodedConcept(
                    value='112039',
                    scheme_designator='DCM',
                    meaning='Tracking Identifier',
                ),
                value=tracking_identifier,
                relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT,
            )
            item.ContentSequence.append(tracking_item)

        # Row 8 — Measurements (MC, 1-n, CONTAINS via TID 300).
        for meas in measurements:
            item.ContentSequence.extend(meas)

        self.append(item)


# ---------------------------------------------------------------------------
# TID 6004 (Sup247 TID 60x4) — Retinal Nerve Fiber Layer Key Measurements
# ---------------------------------------------------------------------------

class CircumpapillaryRNFLKeyMeasurements(Template):
    """:dcm:`TID 6004 <part16/chapter_A.html#sect_TID_6004>`
    Retinal Nerve Fiber Layer Key Measurements

    Type: Extensible · Order: Non-Significant · Root: Yes

    Root template for circumpapillary retinal nerve fiber layer (RNFL)
    thickness measurements obtained by ophthalmic tomography (OCT).
    Encodes the key quadrant and average RNFL values as defined by
    CID 42x3 *RNFL Key Measurements* of Supplement 247.

    Per Supplement 247 Table TID 60x4, the content item hierarchy is::

        Row 1  CONTAINER (nnn102, 99OPHTHALMO, "RNFL Key Measurements")
        Row 2   > Language of Content Item (TID 1204)                [U]
        Row 4   > Algorithm Identification (TID 4019)                [M]
        Row 5   > Ophthalmology Measurements Group (TID 6001) × 1-2  [M]

    (Row 3, Observer Context, is not implemented in this PR.
    Row 6, bilateral symmetry measurement, is an optional extension.)

    Note: the HTML anchor ``#sect_TID_6004`` will resolve once DICOM PS3.16
    is updated to incorporate Supplement 247.  Until then refer to:
    https://dicom.nema.org/Dicom/News/January2025/docs/sups/sup247.pdf

    Usage example::

        from highdicom.sr.templates import (
            AlgorithmIdentification,
            CircumpapillaryRNFLKeyMeasurements,
            Measurement,
            OphthalmologyMeasurementsGroup,
        )
        from highdicom.sr.templates.tid6000 import RNFLAverageThickness
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
            laterality=codes.cid244.Right,
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
            measurements (TID 4019).  Mandatory per Supplement 247
            Table TID 60x4 row 4 (HAS_OBS_CONTEXT).
        measurement_groups: Sequence[highdicom.sr.OphthalmologyMeasurementsGroup]
            One or two :class:`OphthalmologyMeasurementsGroup` instances
            (TID 6001), one per eye.  Each group carries the laterality
            and the individual RNFL thickness measurements from CID 42x3.
            Two groups are used for bilateral studies (one per eye).
            Mandatory per Supplement 247 Table TID 60x4 row 5.
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all SR content items (row 2 of
            TID 60x4, HAS_CONCEPT_MOD).  Defaults to English
            (``en-US``, RFC 5646) when not provided.

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
            name=_CONCEPT_RNFL_KEY,
            template_id='6004',
        )
        item.ContentSequence = ContentSequence()

        # Row 2 — Language of Content Item and Descendants (U).
        if language_of_content_item_and_descendants is None:
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(language_of_content_item_and_descendants)

        # Row 4 — Algorithm Identification (M, TID 4019, HAS_OBS_CONTEXT).
        item.ContentSequence.extend(algorithm_id)

        # Row 5 — Ophthalmology Measurements Group (M, 1-2, CONTAINS).
        for group in measurement_groups:
            item.ContentSequence.extend(group)

        super().__init__([item], is_root=True)


# ---------------------------------------------------------------------------
# TID 6005 (Sup247 TID 60x5) — Macular Thickness Key Measurements
# ---------------------------------------------------------------------------

class MacularThicknessKeyMeasurements(Template):
    """:dcm:`TID 6005 <part16/chapter_A.html#sect_TID_6005>`
    Macular Thickness Key Measurements

    Type: Extensible · Order: Non-Significant · Root: Yes

    Root template for macular thickness measurements obtained by ophthalmic
    tomography (OCT) using the Early Treatment of Diabetic Retinopathy Study
    (ETDRS) grid.  Encodes the central subfield thickness (CMT) and the
    inner/outer ring sector values defined by CID 42x4 *Macular Thickness
    Key Measurements* of Supplement 247.

    The nine ETDRS subfield concept names use their LOINC codes
    (``57108-3`` through ``57117-4``), which are finalised in the standard.
    Total macular volume uses ``LN 57118-2``; the unit must be
    ``UCUM_MICROLITER`` (``uL``).  Average macular thickness (``nnn250``)
    remains a Supplement 247 draft placeholder under scheme ``99OPHTHALMO``.

    Per Supplement 247 Table TID 60x5, the content item hierarchy is::

        Row 1  CONTAINER (nnn103, 99OPHTHALMO, "Macular Thickness Key Measurements")
        Row 2   > Language of Content Item (TID 1204)                [U]
        Row 4   > Algorithm Identification (TID 4019)                [M]
        Row 5   > Ophthalmology Measurements Group (TID 6001) × 1-2  [M]

    (Row 3, Observer Context, is not implemented in this PR.)

    Note: the HTML anchor ``#sect_TID_6005`` will resolve once DICOM PS3.16
    is updated to incorporate Supplement 247.  Until then refer to:
    https://dicom.nema.org/Dicom/News/January2025/docs/sups/sup247.pdf

    Usage example::

        from highdicom.sr.templates import (
            AlgorithmIdentification,
            MacularThicknessKeyMeasurements,
            Measurement,
            OphthalmologyMeasurementsGroup,
        )
        from highdicom.sr.templates.tid6000 import MacularCenterSubfieldThickness
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
            laterality=codes.cid244.Right,
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
            thickness measurements (TID 4019).  Mandatory per Supplement 247
            Table TID 60x5 row 4 (HAS_OBS_CONTEXT).
        measurement_groups: Sequence[highdicom.sr.OphthalmologyMeasurementsGroup]
            One or two :class:`OphthalmologyMeasurementsGroup` instances
            (TID 6001), one per eye.  Each group carries the laterality
            and the individual macular thickness measurements from CID 42x4.
            ETDRS subfield measurements use LOINC codes; the unit for all
            thickness values is ``codes.UCUM.Micrometer`` (``um``).
            Total volume (``MacularTotalVolume``) uses :data:`UCUM_MICROLITER`
            (``uL``).  Mandatory per Supplement 247 Table TID 60x5 row 5.
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all SR content items (row 2,
            HAS_CONCEPT_MOD).  Defaults to English (``en-US``) when not
            provided.

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
            template_id='6005',
        )
        item.ContentSequence = ContentSequence()

        # Row 2 — Language of Content Item and Descendants (U).
        if language_of_content_item_and_descendants is None:
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(language_of_content_item_and_descendants)

        # Row 4 — Algorithm Identification (M, TID 4019, HAS_OBS_CONTEXT).
        item.ContentSequence.extend(algorithm_id)

        # Row 5 — Ophthalmology Measurements Group (M, 1-2, CONTAINS).
        for group in measurement_groups:
            item.ContentSequence.extend(group)

        super().__init__([item], is_root=True)
