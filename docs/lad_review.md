# Highdicom Review Through the LAD Framework

**Purpose:** Evaluate [highdicom](https://github.com/ImagingDataCommons/highdicom) for potential adoption in [heudiconv](https://github.com/nipy/heudiconv) as a more generic, robust, and maintained handler for DICOM metadata.

**Review date:** 2026-02-20

**Framework:** [LAD (LLM-Assisted Development)](https://github.com/chrisfoulon/LAD) by [@chrisfoulon](https://github.com/chrisfoulon) -- a systematic prompt kit for autonomous feature development and enterprise-grade quality assessment. This review uses LAD's embedded quality dimensions (code quality, testing, documentation, integration context, risk management) as the evaluation lens.

---

## LAD Integration Context Assessment

The LAD framework categorizes integration strategies as INTEGRATE / ENHANCE / NEW / DEPRECATION. For heudiconv adopting highdicom, this is an **ENHANCE** scenario -- extending heudiconv's existing DICOM metadata handling by replacing lower-level pydicom/dcmstack usage with highdicom's higher-level abstractions.

---

## 1. Code Quality (LAD Phase 0/1 Criteria)

| LAD Criterion | Highdicom Status | Score |
|---|---|---|
| **PEP 8 / Linting** | flake8 enforced in CI (line length 80) | Pass |
| **Type Hints (PEP 484)** | Full coverage, mypy 1.15.0 checked in CI | Pass |
| **Docstrings (NumPy style)** | Comprehensive numpydoc on all public API | Pass |
| **Complexity** | No explicit radon/max-complexity configured, but modules are well-factored into focused subpackages | Adequate |
| **Python Version** | 3.10-3.14 tested | Pass |

**LAD verdict:** Meets enterprise-grade code quality standards. The type hint coverage is particularly strong -- heudiconv currently has annotation constraints due to nipype's function reparsing (limited to 3.7-compatible syntax), so highdicom's modern typing would be a step up for the DICOM layer.

---

## 2. Testing Standards (LAD Phase 4a-4c Criteria)

| LAD Criterion | Highdicom Status |
|---|---|
| **Coverage Threshold** | >=80% enforced in CI via `--cov-fail-under=80` (LAD targets 90%) |
| **Test Suite Size** | 26,428 lines across 17 modules -- substantial |
| **Test Design** | Component-appropriate: unit tests for content/spatial, integration tests for I/O and SOP classes |
| **Test Data** | 1.3GB of real DICOM test files covering modalities and transfer syntaxes |
| **CI Matrix** | Python 3.10, 3.11, 3.13, 3.14 x with/without libjpeg |

**LAD Root Cause Taxonomy risks:**
- **INFRASTRUCTURE**: Clean -- pydicom >=3.0.1 is the only heavy dependency
- **API_COMPATIBILITY**: Stable API within 0.x series; ~28 releases show incremental evolution, not breaking changes
- **TEST_DESIGN**: Tests are comprehensive, especially for seg (7k lines) and sr (6k lines)

**LAD verdict:** Solid testing foundation. The 80% floor is below LAD's 90% target but realistic for a domain-specific library with large generated modules (`_modules.py` at 28MB). Test data is real-world DICOM, not synthetic -- a significant quality signal.

---

## 3. Documentation Standards (LAD Multi-Level Assessment)

| LAD Level | Highdicom Coverage |
|---|---|
| **Level 1 (Plain English)** | Good README, overview docs, peer-reviewed paper (JDI 2022) |
| **Level 2 (API Reference)** | Full Sphinx autodoc hosted at readthedocs.io, 29 .rst files |
| **Level 3 (Code Examples)** | Jupyter notebooks in `examples/`, Dockerized environment, per-module usage guides |

**LAD verdict:** Exceeds typical research software documentation. The peer-reviewed publication is a strong signal -- design decisions are academically documented and defensible.

---

## 4. Risk Management (LAD Phase 4a Regression Risk)

**Bus Factor: 1** -- This is the single most significant risk under the LAD framework.

| Risk Factor | Assessment |
|---|---|
| **Maintainer concentration** | CPBridge: 685/1686 commits (65%). hackermd (20%) appears less active recently |
| **Institutional backing** | NCI Imaging Data Commons, MGH/BWH, QIICR -- federally funded |
| **Response time** | 0-1 days on issues -- excellent |
| **Release cadence** | 6 releases in 2025, active feature branches for v0.28.0 |
| **Community size** | 220 stars, 48 forks, 19 contributors, ~41K monthly PyPI downloads |
| **Deprecation risk** | Low -- backed by NCI IDC which is a major NIH initiative |

**CI Status (as of review date):** The latest master commit (22b0b79) has a failing CI run. The failure is a time-of-day-dependent bug in `test_series_datetime` -- the test passes `series_time=12:34:56` without explicit `content_time`, which defaults to `now.time()`. When CI runs before 12:34:56 UTC, the validation `series_time > content_time` fires incorrectly. The root cause is a logic bug in `base.py:296-300`: the time comparison is performed even when `series_date != content_date`, which is semantically wrong (if a series is from a past date, its time-of-day is irrelevant relative to today's content time). A fix has been prepared -- see the Appendix below.

**LAD Regression Risk Matrix:**
- **Low risk**: API stability (28 releases in 0.x with incremental evolution)
- **Medium risk**: Single-maintainer dependency
- **Low risk**: Institutional defunding (federal/NIH multi-source funding)

---

## 5. LAD Integration Feasibility for Heudiconv

### Current heudiconv DICOM stack

- `pydicom >=1.0.0` -- direct low-level DICOM parsing
- `dcmstack >=0.8` -- DICOM metadata aggregation (primary extraction layer)
- Custom CSA header parsing for Siemens private tags
- Custom `SeqInfo` NamedTuple for sequence metadata

### What highdicom would give heudiconv

| Heudiconv Pain Point | Highdicom Solution |
|---|---|
| CSA header workarounds | Not directly -- highdicom focuses on standard DICOM, not vendor-private tags |
| Vendor-specific sequence handling | Partial -- standard-compliant metadata; vendor quirks still need custom code |
| Series grouping complexity | `Image` class provides structured access to multi-frame relationships |
| Coordinate/spatial transforms | `PixelToReferenceTransformer` and spatial module -- far richer than current approach |
| Type safety on DICOM attributes | Full enum coverage (`PhotometricInterpretationValues`, orientation types, etc.) |
| Volume construction from series | `get_volume_from_series()` -- direct match for heudiconv's core use case |
| Metadata validation | Standard compliance validation on construction -- prevents invalid objects |

### What highdicom would NOT help with

- Siemens CSA header parsing (private tags are outside DICOM standard)
- dcm2niix integration (orthogonal concern)
- BIDS-specific mapping logic (domain-specific to heudiconv)
- nipype workflow integration

---

## 6. LAD Maintenance Opportunity Detection

Adopting highdicom would let heudiconv address:

**High Priority (fix during adoption):**
- Replace raw pydicom attribute access with typed highdicom enums -- eliminates a class of KeyError/AttributeError bugs
- Use `Volume`/`VolumeGeometry` instead of manual frame-of-reference computation
- Leverage highdicom's tolerance for real-world DICOM deviations (design principle #4: "Tolerate minor deviations")

**Medium Priority (boy scout rule):**
- Retire dcmstack dependency if highdicom covers its metadata extraction role
- Modernize type annotations in DICOM-handling code (highdicom requires 3.10+, which could force a heudiconv floor bump)

---

## Overall LAD Assessment

| LAD Dimension | Rating | Notes |
|---|---|---|
| Code Quality | **Strong** | Type hints, linting, docstrings all enforced |
| Testing | **Strong** | >=80% coverage, 26K lines of tests, real DICOM data |
| Documentation | **Strong** | Multi-level docs, peer-reviewed paper, RTD hosting |
| Community Health | **Moderate** | Active but bus-factor-1; strong institutional backing mitigates |
| Integration Fit | **Partial** | Excellent for standard DICOM metadata; does not cover vendor-private tags |
| Risk | **Low-Medium** | Main risk is maintainer concentration; offset by NIH/NCI backing |
| API Stability | **Strong** | 28 releases in 0.x with incremental evolution |

---

## Recommendation

Highdicom is a well-engineered, LAD-compliant library for **standard DICOM metadata handling**. For heudiconv adoption:

1. **Good fit for**: Replacing raw pydicom access patterns, spatial transforms, volume construction, metadata validation, typed enums for DICOM attributes
2. **Not a replacement for**: dcmstack's vendor-specific metadata aggregation, CSA header parsing, dcm2niix integration
3. **Key concern**: It would add a dependency on a bus-factor-1 project, though the institutional backing (NCI IDC) and MIT license provide safety nets (forkability)
4. **Migration path**: Incremental -- start using highdicom's `Image`/`Volume` for new code paths while keeping dcmstack for vendor-specific legacy paths. This is an ENHANCE, not a wholesale REPLACE

The strongest argument for adoption is that highdicom enforces DICOM standard compliance at the type level -- something heudiconv currently achieves through convention and manual validation. The strongest argument for caution is that heudiconv's hardest problems (vendor-specific quirks, private tags, CSA headers) are explicitly outside highdicom's scope.

---

## Appendix: CI Fix for Time-of-Day-Dependent Test Failure

The latest master (22b0b79) fails CI due to a time-of-day-dependent bug. The root cause is in `src/highdicom/base.py`, where the `series_time > content_time` validation runs even when `series_date` and `content_date` differ -- making the comparison semantically meaningless.

**Fix in `base.py` (line ~296):** Only compare times when dates are equal:

```python
# Before (buggy):
if content_time is not None:
    if series_time > content_time:
        raise ValueError(...)

# After (fixed):
if (
    content_time is not None and
    content_date is not None and
    series_date == content_date and
    series_time > content_time
):
    raise ValueError(...)
```

**Test fixes in `test_base.py`:**
- `test_series_datetime`: Made deterministic by explicitly setting `content_date`/`content_time`
- `test_series_datetime_earlier_date` (new): Verifies that `series_time > content_time` is allowed when `series_date < content_date`
- `test_series_time_after_content_same_date` (new): Verifies that `series_time > content_time` is rejected when dates are equal

**Result:** 1783 passed, 0 failed, 119 skipped, coverage 83.94% (>=80% threshold).

---

*This review was generated using the [LAD framework](https://github.com/chrisfoulon/LAD) by [@chrisfoulon](https://github.com/chrisfoulon) and Claude Code.*
