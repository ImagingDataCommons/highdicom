"""Enumerated values."""
from enum import Enum


class CoordinateSystemNames(Enum):

    """Enumerated values for coordinate system names."""

    PATIENT = 'PATIENT'
    SLIDE = 'SLIDE'


class ContentQualificationValues(Enum):

    """Enumerated values for Content Qualification attribute."""

    PRODUCT = 'PRODUCT'
    RESEARCH = 'RESEARCH'
    SERVICE = 'SERVICE'


class DimensionOrganizationTypeValues(Enum):

    """Enumerated values for Dimension Organization Type attribute."""

    THREE_DIMENSIONAL = '3D'
    THREE_DIMENSIONAL_TEMPORAL = '3D_TEMPORAL'
    TILED_FULL = 'TILED_FULL'
    TILED_SPARSE = 'TILED_SPARSE'


class PhotometricInterpretationValues(Enum):

    """Enumerated values for Photometric Interpretation attribute."""

    MONOCHROME1 = 'MONOCHROME1'
    MONOCHROME2 = 'MONOCHROME2'
    PALETTE_COLOR = 'PALETTE COLOR'
    RGB = 'RGB'
    YBR_FULL = 'YBR_FULL'
    YBR_FULL_422 = 'YBR_FULL_422'
    YBR_PARTIAL_420 = 'YBR_PARTIAL_420'
    YBR_ICT = 'YBR_ICT'
    YBR_RCT = 'YBR_RCT'


class LateralityValues(Enum):

    """Enumerated values for Laterality attribute."""

    R = 'R'
    L = 'L'


class AnatomicalOrientationTypeValues(Enum):

    """Enumerated values for Anatomical Orientation Type attribute."""

    BIPED = 'BIPED'
    QUADRUPED = 'QUADRUPED'


class PatientOrientationValuesBiped(Enum):

    """Enumerated values for Patient Orientation attribute
    if Anatomical Orientation Type attribute has value ``"BIPED"``.
    """

    A = 'A'
    P = 'P'
    R = 'R'
    L = 'L'
    H = 'H'
    F = 'F'


class PatientOrientationValuesQuadruped(Enum):

    """Enumerated values for Patient Orientation attribute
    if Anatomical Orientation Type attribute has value ``"QUADRUPED"``.
    """

    LE = 'LE'
    RT = 'RT'
    D = 'D'
    V = 'V'
    CR = 'CR'
    CD = 'CD'
    R = 'R'
    M = 'M'
    L = 'L'
    PR = 'PR'
    DI = 'DI'
    PA = 'PA'
    PL = 'PL'


class UniversalEntityIDTypeValues(Enum):

    """Enumerated values for Universal Entity ID Type attribute."""

    DNS = 'DNS'
    EUI64 = 'EUI64'
    ISO = 'ISO'
    URI = 'URI'
    UUID = 'UUID'
    X400 = 'X400'
    X500 = 'X500'
