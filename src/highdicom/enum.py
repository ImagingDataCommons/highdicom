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


class PatientSexValues(Enum):

    """Enumerated values for Patient's Sex attribute."""

    M = 'M'
    """Male"""

    F = 'F'
    """Female"""

    O = 'O'  # noqa: E741
    """Other"""


class PhotometricInterpretationValues(Enum):

    """Enumerated values for Photometric Interpretation attribute.

    See :dcm:`Section C.7.6.3.1.2<part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2>`
    for more information.

    """

    MONOCHROME1 = 'MONOCHROME1'
    MONOCHROME2 = 'MONOCHROME2'
    PALETTE_COLOR = 'PALETTE COLOR'
    RGB = 'RGB'
    YBR_FULL = 'YBR_FULL'
    YBR_FULL_422 = 'YBR_FULL_422'
    YBR_PARTIAL_420 = 'YBR_PARTIAL_420'
    YBR_ICT = 'YBR_ICT'
    YBR_RCT = 'YBR_RCT'


class PlanarConfigurationValues(Enum):

    """Enumerated values for Planar Representation attribute."""

    COLOR_BY_PIXEL = 0
    COLOR_BY_PLANE = 1


class PixelRepresentationValues(Enum):

    """Enumerated values for Planar Representation attribute."""

    UNSIGNED_INTEGER = 0
    COMPLEMENT = 1


class RescaleTypeValues(Enum):

    """Enumerated values for attribute Rescale Type.

    This specifies the units of the result of the rescale operation.
    Other values may be used, but they are not defined by the DICOM standard.

    """

    OD = 'OD'
    """The number in the LUT represents thousands of optical density.

    That is, a value of 2140 represents an optical density of 2.140.

    """

    HU = 'HU'
    """Hounsfield Units (CT)."""

    US = 'US'
    """Unspecified."""

    MGML = 'MGML'
    """Milligrams per milliliter."""

    Z_EFF = 'Z_EFF'
    """Effective Atomic Number (i.e., Effective-Z)."""

    ED = 'ED'
    """Electron density in 1023 electrons/ml."""

    EDW = 'EDW'
    """Electron density normalized to water.

    Units are N/Nw where N is number of electrons per unit volume, and Nw is
    number of electrons in the same unit of water at standard temperature and
    pressure.

    """

    HU_MOD = 'HU_MOD'
    """Modified Hounsfield Unit."""

    PCT = 'PCT'
    """Percentage (%)"""


class VOILUTFunctionValues(Enum):

    """Enumerated values for attribute VOI LUT Function."""

    LINEAR = 'LINEAR'
    LINEAR_EXACT = 'LINEAR_EXACT'
    SIGMOID = 'SIGMOID'


class PresentationLUTShapeValues(Enum):

    """Enumerated values for the Presentation LUT Shape attribute."""

    IDENTITY = 'IDENTITY'
    """No further translation of values is performed."""

    INVERSE = 'INVERSE'
    """

    A value of INVERSE shall mean the same as a value of IDENTITY, except that
    the minimum output value shall convey the meaning of the maximum available
    luminance, and the maximum value shall convey the minimum available
    luminance.

    """


class LateralityValues(Enum):

    """Enumerated values for Laterality attribute."""

    R = 'R'
    """Right"""

    L = 'L'
    """Left"""


class AnatomicalOrientationTypeValues(Enum):

    """Enumerated values for Anatomical Orientation Type attribute."""

    BIPED = 'BIPED'
    QUADRUPED = 'QUADRUPED'


class PatientOrientationValuesBiped(Enum):

    """Enumerated values for Patient Orientation attribute
    if Anatomical Orientation Type attribute has value ``"BIPED"``.
    """

    A = 'A'
    """Anterior"""

    P = 'P'
    """Posterior"""

    R = 'R'
    """Right"""

    L = 'L'
    """Left"""

    H = 'H'
    """Head"""

    F = 'F'
    """Foot"""


class PatientOrientationValuesQuadruped(Enum):

    """Enumerated values for Patient Orientation attribute
    if Anatomical Orientation Type attribute has value ``"QUADRUPED"``.
    """

    LE = 'LE'
    """Left"""

    RT = 'RT'
    """Right"""

    D = 'D'
    """Dorsal"""

    V = 'V'
    """Ventral"""

    CR = 'CR'
    """Cranial"""

    CD = 'CD'
    """Caudal"""

    R = 'R'
    """Rostral"""

    M = 'M'
    """Medial"""

    L = 'L'
    """Lateral"""

    PR = 'PR'
    """Proximal"""

    DI = 'DI'
    """Distal"""

    PA = 'PA'
    """Palmar"""

    PL = 'PL'
    """Plantar"""


class UniversalEntityIDTypeValues(Enum):

    """Enumerated values for Universal Entity ID Type attribute."""

    DNS = 'DNS'
    """An Internet dotted name. Either in ASCII or as integers."""

    EUI64 = 'EUI64'
    """An IEEE Extended Unique Identifier."""

    ISO = 'ISO'
    """An International Standards Organization Object Identifier."""

    URI = 'URI'
    """Uniform Resource Identifier."""

    UUID = 'UUID'
    """The DCE Universal Unique Identifier."""

    X400 = 'X400'
    """An X.400 MHS identifier."""

    X500 = 'X500'
    """An X.500 directory name."""
