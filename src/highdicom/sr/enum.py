"""Enumerate values specific to Structured Report IODs."""
from enum import Enum


class ValueTypeValues(Enum):

    """Enumerated values for attribute Value Type."""

    CODE = 'CODE'
    COMPOSITE = 'COMPOSITE'
    CONTAINER = 'CONTAINER'
    DATE = 'DATE'
    DATETIME = 'DATETIME'
    IMAGE = 'IMAGE'
    NUM = 'NUM'
    PNAME = 'PNAME'
    SCOORD = 'SCOORD'
    SCOORD3D = 'SCOORD3D'
    TCOORD = 'TCOORD'
    TEXT = 'TEXT'
    TIME = 'TIME'
    UIDREF = 'UIDREF'
    WAVEFORM = 'WAVEFORM'


class GraphicTypeValues(Enum):

    """Enumerated values for attribute Graphic Type."""

    CIRCLE = 'CIRCLE'
    ELLIPSE = 'ELLIPSE'
    ELLIPSOID = 'ELLIPSOID'
    MULTIPOINT = 'MULTIPOINT'
    POINT = 'POINT'
    POLYLINE = 'POLYLINE'


class GraphicTypeValues3D(Enum):

    """Enumerated values for attribute Graphic Type 3D."""

    ELLIPSE = 'ELLIPSE'
    ELLIPSOID = 'ELLIPSOID'
    MULTIPOINT = 'MULTIPOINT'
    POINT = 'POINT'
    POLYLINE = 'POLYLINE'
    POLYGON = 'POLYGON'


class TemporalRangeTypeValues(Enum):

    """Enumerated values for attribute Temporal Range Type."""

    BEGIN = 'BEGIN'
    END = 'END'
    MULTIPOINT = 'MULTIPOINT'
    MULTISEGMENT = 'MULTISEGMENT'
    POINT = 'POINT'
    SEGMENT = 'SEGMENT'


class RelationshipTypeValues(Enum):

    """Enumerated values for attribute Relationship Type."""

    CONTAINS = 'CONTAINS'
    HAS_ACQ_CONTEXT = 'HAS ACQ CONTEXT'
    HAS_CONCEPT_MOD = 'HAS CONCEPT MOD'
    HAS_OBS_CONTEXT = 'HAS OBS CONTEXT'
    HAS_PROPERTIES = 'HAS PROPERTIES'
    INFERRED_FROM = 'INFERRED FROM'
    SELECTED_FROM = 'SELECTED FROM'


class PixelOriginInterpretationValues(Enum):

    """Enumerated values for attribute Pixel Origin Interpretation."""

    FRAME = 'FRAME'
    VOLUME = 'VOLUME'


class PlanarROITypes(Enum):

    """Planar Region of Interest (ROI) types.

    Correspond to SR Content Content Items contained in Measurement Group
    defined by TID 1410 Planar ROI Measurements and Qualitative Evaluations.

    """

    REGION = 'REGION'
    SEGMENT = 'SEGMENT'


class VolumetricROITypes(Enum):

    """Volumetric Region of Interest (ROI) types.

    Correspond to SR Content Content Items contained in Measurement Group
    defined by TID 1411 Volumetric ROI Measurements and Qualitative Evaluations.

    """

    REGIONS = 'REGIONS'
    SEGMENT = 'SEGMENT'
    VOLUME_SURFACE = 'VOLUME_SURFACE'
