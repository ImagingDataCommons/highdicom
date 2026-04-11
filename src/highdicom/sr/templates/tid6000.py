"""DICOM SR templates for Supplement 247 — Eyecare Measurement Templates.

Implements TIDs 6001–6005 as defined in DICOM Supplement 247
"Eyecare Measurement Templates", incorporated into DICOM PS3.16 2025c.

Reference: DICOM PS3.16 Supplement 247
  https://dicom.nema.org/Dicom/News/January2025/docs/sups/sup247.pdf

Note on provisional concept codes
----------------------------------
Supplement 247 was balloted as a draft at the time of this implementation.
Several DCM concept codes appear in the draft as ``nnnXXX`` placeholders
(i.e. final code values had not yet been assigned by DICOM WG-09).
Those placeholders are marked with ``# Sup247 draft — nnnXXX`` comments
and defined as private ``CodedConcept`` objects using the scheme designator
``99OPHTHALMO`` (≤ 16 chars, required for DICOM VR SH).

A companion PR to pydicom will replace these with the final CID entries
once the official code assignments are published.  All code objects that
already have stable identifiers (LOINC, SNOMED-CT, or established DCM
values) use the canonical pydicom ``codes.*`` accessors.
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
    NumContentItem,
)
from highdicom.sr.templates.tid1500 import (
    AlgorithmIdentification,
    LanguageOfContentItemAndDescendants,
    Measurement,
    Template,
)

# ---------------------------------------------------------------------------
# Provisional concept codes — Supplement 247 draft (nnnXXX placeholders).
# Scheme 99HIGHDICOM-EYECARE is a private designator that will be replaced
# by official DCM codes once DICOM WG-09 assigns them.
# ---------------------------------------------------------------------------
_SCHEME = '99OPHTHALMO'

# Container concept names for the root templates (TID 60x3–60x9)
# Sup247 draft table rows: EV (nnn101–nnn107, DCM, "...")
_CONCEPT_OPTIC_DISC_KEY = CodedConcept(
    value='nnn101', scheme_designator=_SCHEME,
    meaning='Optic Disc Key Measurements'
)
_CONCEPT_RNFL_KEY = CodedConcept(
    value='nnn102', scheme_designator=_SCHEME,
    meaning='RNFL Key Measurements'
)
_CONCEPT_MACULAR_THICKNESS_KEY = CodedConcept(
    value='nnn103', scheme_designator=_SCHEME,
    meaning='Macular Thickness Key Measurements'
)

# ---------------------------------------------------------------------------
# CID 42x3 — RNFL Key Measurements (all µm, UCUM "um")
# Sup247 draft Table CID 42x3: nnn400–nnn404, nnn406, nnn411–nnn422
# ---------------------------------------------------------------------------
RNFLAverageThickness = CodedConcept(
    value='nnn400', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer average thickness'
)
RNFLInferiorThickness = CodedConcept(
    value='nnn401', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer inferior thickness'
)
RNFLSuperiorThickness = CodedConcept(
    value='nnn402', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer superior thickness'
)
RNFLTemporalThickness = CodedConcept(
    value='nnn403', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer temporal thickness'
)
RNFLNasalThickness = CodedConcept(
    value='nnn404', scheme_designator=_SCHEME,
    meaning='Retinal nerve fiber layer nasal thickness'
)
RetinalROIRadius = CodedConcept(
    value='nnn406', scheme_designator=_SCHEME,
    meaning='Retinal ROI radius'
)

# ---------------------------------------------------------------------------
# CID 42x4 — Macular Thickness Key Measurements
# LOINC codes are already finalised in DICOM 2025c for the ETDRS grid.
# nnn250 (Average macular thickness) remains a Sup247 placeholder.
# ---------------------------------------------------------------------------
# LOINC ETDRS grid codes — stable, no pydicom shortcut yet
MacularCenterPointThickness = CodedConcept(
    value='57108-3', scheme_designator='LN',
    meaning='Macular grid.center point thickness by OCT'
)
MacularCenterSubfieldThickness = CodedConcept(
    value='57109-1', scheme_designator='LN',
    meaning='Macular grid.center subfield thickness by OCT'
)
MacularInnerSuperiorThickness = CodedConcept(
    value='57110-9', scheme_designator='LN',
    meaning='Macular grid.inner superior subfield thickness by OCT'
)
MacularInnerNasalThickness = CodedConcept(
    value='57111-7', scheme_designator='LN',
    meaning='Macular grid.inner nasal subfield thickness by OCT'
)
MacularInnerInferiorThickness = CodedConcept(
    value='57112-5', scheme_designator='LN',
    meaning='Macular grid.inner inferior subfield thickness by OCT'
)
MacularInnerTemporalThickness = CodedConcept(
    value='57113-3', scheme_designator='LN',
    meaning='Macular grid.inner temporal subfield thickness by OCT'
)
MacularOuterSuperiorThickness = CodedConcept(
    value='57114-1', scheme_designator='LN',
    meaning='Macular grid.outer superior subfield thickness by OCT'
)
MacularOuterNasalThickness = CodedConcept(
    value='57115-8', scheme_designator='LN',
    meaning='Macular grid.outer nasal subfield thickness by OCT'
)
MacularOuterInferiorThickness = CodedConcept(
    value='57116-6', scheme_designator='LN',
    meaning='Macular grid.outer inferior subfield thickness by OCT'
)
MacularOuterTemporalThickness = CodedConcept(
    value='57117-4', scheme_designator='LN',
    meaning='Macular grid.outer temporal subfield thickness by OCT'
)
MacularTotalVolume = CodedConcept(
    value='57118-2', scheme_designator='LN',
    meaning='Macular grid.total volume by OCT'
)
# Sup247 draft nnn250 — provisional
AverageMacularThickness = CodedConcept(
    value='nnn250', scheme_designator=_SCHEME,
    meaning='Average macular thickness'
)


# ---------------------------------------------------------------------------
# TID 6001 (Sup247 TID 60x1) — Ophthalmology Measurements Group
# ---------------------------------------------------------------------------

class OphthalmologyMeasurementsGroup(Template):
    """:dcm:`TID 6001 <part16/chapter_A.html#sect_TID_6001>`
    Ophthalmology Measurements Group

    A sub-template specialised for ophthalmic measurements, structurally
    equivalent to TID 1501 *Measurement and Qualitative Evaluation Group*
    but restricted to ophthalmic anatomy (finding site "Eye") and with
    mandatory laterality.

    This template is invoked by the root eyecare templates (TID 6003–6009)
    once per eye measured.  It wraps the actual measurement
    :class:`~highdicom.sr.Measurement` items together with the anatomic
    context (finding site and laterality) required by Supplement 247.

    Reference: DICOM PS3.16 Supplement 247, Table TID 60x1
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
            Laterality of the eye measured (see
            :dcm:`CID 244 <part16/sect_CID_244.html>` "Laterality" for
            options; typically ``codes.cid244.Right`` or
            ``codes.cid244.Left``).
        measurements: Sequence[highdicom.sr.Measurement]
            One or more measurements belonging to this group.  Each
            :class:`~highdicom.sr.Measurement` (TID 300) encodes a single
            numeric finding (e.g. average RNFL thickness).
        finding_site: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Specific anatomic site within the eye (see
            :dcm:`CID 4209 <part16/sect_CID_4209.html>` "Ophthalmic
            Anatomic Location" for options).  Defaults to
            ``codes.cid4209.Eye`` when not provided.
        tracking_identifier: str, optional
            Free-text tracking identifier for this measurement group (row 6
            of TID 60x1, concept ``(112039, DCM, "Tracking Identifier")``).

        """  # noqa: E501
        super().__init__()

        if finding_site is None:
            finding_site = codes.cid4209.Eye

        item = ContainerContentItem(
            name=codes.DCM.MeasurementGroup,
            template_id='6001',
            relationship_type=RelationshipTypeValues.CONTAINS,
        )
        item.ContentSequence = ContentSequence()

        # Row 2 — Finding Site (mandatory)
        site_item = CodeContentItem(
            name=CodedConcept(
                value='363698007',
                scheme_designator='SCT',
                meaning='Finding Site',
            ),
            value=finding_site,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD,
        )
        item.ContentSequence.append(site_item)

        # Row 3 — Laterality (mandatory)
        lat_item = CodeContentItem(
            name=CodedConcept(
                value='272741003',
                scheme_designator='SCT',
                meaning='Laterality',
            ),
            value=laterality,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD,
        )
        item.ContentSequence.append(lat_item)

        # Row 6 — Tracking Identifier (optional)
        if tracking_identifier is not None:
            from highdicom.sr.value_types import TextContentItem
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

        # Rows 8/10 — Measurements (mandatory, 1-n)
        if not measurements:
            raise ValueError(
                "Argument 'measurements' must contain at least one item."
            )
        for meas in measurements:
            if not isinstance(meas, Measurement):
                raise TypeError(
                    'Each item of "measurements" must have type Measurement.'
                )
            item.ContentSequence.extend(meas)

        self.append(item)


# ---------------------------------------------------------------------------
# TID 6004 (Sup247 TID 60x4) — Retinal Nerve Fiber Layer Key Measurements
# ---------------------------------------------------------------------------

class CircumpapillaryRNFLKeyMeasurements(Template):
    """:dcm:`TID 6004 <part16/chapter_A.html#sect_TID_6004>`
    Retinal Nerve Fiber Layer Key Measurements

    Root template for circumpapillary RNFL thickness measurements obtained
    by ophthalmic tomography (OCT).  Encodes the key quadrant and average
    RNFL values as defined by CID 42x3 (RNFL Key Measurements) of
    Supplement 247.

    Usage::

        from highdicom.sr.templates import (
            AlgorithmIdentification,
            CircumpapillaryRNFLKeyMeasurements,
            Measurement,
            OphthalmologyMeasurementsGroup,
        )
        from highdicom.sr.templates.tid6000 import (
            RNFLAverageThickness,
            RNFLSuperiorThickness,
            RNFLInferiorThickness,
        )
        from pydicom.sr.codedict import codes
        from pydicom.sr.coding import Code

        algo = AlgorithmIdentification(name='Revo FC', version='1.0')
        meas = [
            Measurement(
                name=RNFLAverageThickness,
                value=121.0,
                unit=codes.UCUM.Micrometer,
            ),
            Measurement(
                name=RNFLSuperiorThickness,
                value=135.0,
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

    Reference: DICOM PS3.16 Supplement 247, Table TID 60x4
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
            measurements (TID 4019, mandatory per Supplement 247 row 4).
        measurement_groups: Sequence[highdicom.sr.OphthalmologyMeasurementsGroup]
            One or two measurement groups (TID 6001), one per eye.  Each
            group carries the laterality and the individual RNFL thickness
            measurements from CID 42x3.
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all content items (defaults to English
            when omitted).

        """  # noqa: E501
        super().__init__()

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

        item = ContainerContentItem(
            name=_CONCEPT_RNFL_KEY,
            template_id='6004',
        )
        item.ContentSequence = ContentSequence()

        # Row 2 — Language (optional)
        if language_of_content_item_and_descendants is None:
            from highdicom.sr.templates.tid1500 import DEFAULT_LANGUAGE
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(language_of_content_item_and_descendants)

        # Row 4 — Algorithm Identification (mandatory)
        if not isinstance(algorithm_id, AlgorithmIdentification):
            raise TypeError(
                'Argument "algorithm_id" must have type '
                'AlgorithmIdentification.'
            )
        item.ContentSequence.extend(algorithm_id)

        # Row 5 — Measurement Groups (1-2, mandatory)
        for group in measurement_groups:
            item.ContentSequence.extend(group)

        super().__init__([item], is_root=True)


# ---------------------------------------------------------------------------
# TID 6005 (Sup247 TID 60x5) — Macular Thickness Key Measurements
# ---------------------------------------------------------------------------

class MacularThicknessKeyMeasurements(Template):
    """:dcm:`TID 6005 <part16/chapter_A.html#sect_TID_6005>`
    Macular Thickness Key Measurements

    Root template for ETDRS-grid macular thickness measurements obtained
    by ophthalmic tomography (OCT).  Encodes the central subfield (CMT)
    and the inner/outer ring sector values defined by CID 42x4 (Macular
    Thickness Key Measurements) of Supplement 247.

    The ETDRS grid concept names are encoded using their LOINC codes
    (``57108-3`` through ``57118-2``), which are finalised in the standard.

    Usage::

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

    Reference: DICOM PS3.16 Supplement 247, Table TID 60x5
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
            thickness measurements (TID 4019, mandatory per Supplement 247
            row 4).
        measurement_groups: Sequence[highdicom.sr.OphthalmologyMeasurementsGroup]
            One or two measurement groups (TID 6001), one per eye.  Each
            group carries the laterality and the individual macular thickness
            measurements from CID 42x4.
        language_of_content_item_and_descendants: Union[highdicom.sr.LanguageOfContentItemAndDescendants, None], optional
            Language specification for all content items (defaults to English
            when omitted).

        """  # noqa: E501
        super().__init__()

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

        item = ContainerContentItem(
            name=_CONCEPT_MACULAR_THICKNESS_KEY,
            template_id='6005',
        )
        item.ContentSequence = ContentSequence()

        # Row 2 — Language (optional)
        if language_of_content_item_and_descendants is None:
            from highdicom.sr.templates.tid1500 import DEFAULT_LANGUAGE
            language_of_content_item_and_descendants = \
                LanguageOfContentItemAndDescendants(DEFAULT_LANGUAGE)
        item.ContentSequence.extend(language_of_content_item_and_descendants)

        # Row 4 — Algorithm Identification (mandatory)
        if not isinstance(algorithm_id, AlgorithmIdentification):
            raise TypeError(
                'Argument "algorithm_id" must have type '
                'AlgorithmIdentification.'
            )
        item.ContentSequence.extend(algorithm_id)

        # Row 5 — Measurement Groups (1-2, mandatory)
        for group in measurement_groups:
            item.ContentSequence.extend(group)

        super().__init__([item], is_root=True)
