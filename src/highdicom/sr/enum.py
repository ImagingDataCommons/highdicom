"""Enumerate values specific to Structured Report IODs."""
from enum import Enum


class ValueTypeValues(Enum):

    """Enumerated values for attribute Value Type.

    See :dcm:`Table C.17.3.2.1 <part03/sect_C.17.3.2.html#sect_C.17.3.2.1>`.

    """

    CODE = 'CODE'
    """Coded expression of the concept."""

    COMPOSITE = 'COMPOSITE'
    """Reference to UIDs of Composite SOP Instances."""

    CONTAINER = 'CONTAINER'
    """The content of the CONTAINER. The value of a CONTAINER Content Item is
    the collection of Content Items that it contains."""

    DATE = 'DATE'
    """Calendar date."""

    DATETIME = 'DATETIME'
    """Concatenated date and time."""

    IMAGE = 'IMAGE'
    """Reference to UIDs of Image Composite SOP Instances."""

    NUM = 'NUM'
    """Numeric value and associated Unit of Measurement."""

    PNAME = 'PNAME'
    """Name of person."""

    SCOORD = 'SCOORD'
    """Listing of spatial coordinates defined in 2D pixel matrix."""

    SCOORD3D = 'SCOORD3D'
    """Listing of spatial 3D coordinates."""

    TCOORD = 'TCOORD'
    """Listing of temporal coordinates."""

    TEXT = 'TEXT'
    """Textual expression of the concept."""

    TIME = 'TIME'
    """Time of day."""

    UIDREF = 'UIDREF'
    """Unique Identifier."""

    WAVEFORM = 'WAVEFORM'
    """Reference to UIDs of Waveform Composite SOP Instances."""


class GraphicTypeValues(Enum):

    """Enumerated values for attribute Graphic Type.

    See :dcm:`C.18.6.1.1 <part03/sect_C.18.6.html#sect_C.18.6.1.1>`.

    """

    CIRCLE = 'CIRCLE'
    """A circle defined by two (column,row) pairs. The first point is the
    central pixel. The second point is a pixel on the perimeter of the
    circle."""

    ELLIPSE = 'ELLIPSE'
    """An ellipse defined by four pixel (column,row) pairs, the first two
    points specifying the endpoints of the major axis and the second two points
    specifying the endpoints of the minor axis of an ellipse."""

    MULTIPOINT = 'MULTIPOINT'
    """Multiple pixels each denoted by an (column,row) pair."""

    POINT = 'POINT'
    """A single pixel denoted by a single (column,row) pair."""

    POLYLINE = 'POLYLINE'
    """A series of connected line segments with ordered vertices denoted by
    (column,row) pairs; if the first and last vertices are the same it is a
    closed polygon."""


class GraphicTypeValues3D(Enum):

    """Enumerated values for attribute Graphic Type 3D.

    See :dcm:`C.18.9.1.2 <part03/sect_C.18.9.html#sect_C.18.9.1.2>`.

    """

    ELLIPSE = 'ELLIPSE'
    """An ellipse defined by four (x,y,z) triplets, the first two triplets
    specifying the endpoints of the major axis and the second two triplets
    specifying the endpoints of the minor axis."""

    ELLIPSOID = 'ELLIPSOID'
    """A three-dimensional geometric surface whose plane sections are either
    ellipses or circles and contains three intersecting orthogonal axes, "a",
    "b", and "c"; the ellipsoid is defined by six (x,y,z) triplets, the first
    and second triplets specifying the endpoints of axis "a", the third and
    fourth triplets specifying the endpoints of axis "b", and the fifth and
    sixth triplets specifying the endpoints of axis "c"."""

    MULTIPOINT = 'MULTIPOINT'
    """Multiple locations each denoted by an (x,y,z) triplet; the points need
    not be coplanar."""

    POINT = 'POINT'
    """A single location denoted by a single (x,y,z) triplet."""

    POLYLINE = 'POLYLINE'
    """A series of connected line segments with ordered vertices denoted by
    (x,y,z) triplets; the points need not be coplanar."""

    POLYGON = 'POLYGON'
    """A series of connected line segments with ordered vertices denoted by
    (x,y,z) triplets, where the first and last vertices shall be the same
    forming a polygon; the points shall be coplanar."""


class TemporalRangeTypeValues(Enum):

    """Enumerated values for attribute Temporal Range Type.

    See :dcm:`C.18.7.1.1 <part03/sect_C.18.7.html#sect_C.18.7.1.1>`.

    """

    BEGIN = 'BEGIN'
    """A range beginning at one temporal point, and extending beyond the end of
    the acquired data."""

    END = 'END'
    """A range beginning before the start of the acquired data, and extending
    to (and including) the identified temporal point."""

    MULTIPOINT = 'MULTIPOINT'
    """Multiple temporal points."""

    MULTISEGMENT = 'MULTISEGMENT'
    """Multiple segments, each denoted by two temporal points."""

    POINT = 'POINT'
    """A single temporal point."""

    SEGMENT = 'SEGMENT'
    """A range between two temporal points."""


class RelationshipTypeValues(Enum):

    """Enumerated values for attribute Relationship Type.

    See :dcm:`C.17.3.2.4 <part03/sect_C.17.3.2.4.html#sect_C.17.3.2.4>`.

    """

    CONTAINS = 'CONTAINS'
    """Source Item contains Target Content Item."""

    HAS_ACQ_CONTEXT = 'HAS ACQ CONTEXT'
    """The Target Content Item describes the conditions present during data
    acquisition of the Source Content Item."""

    HAS_CONCEPT_MOD = 'HAS CONCEPT MOD'
    """Used to qualify or describe the Concept Name of the Source Content Item,
    such as to create a post-coordinated description of a concept, or to
    further describe a concept."""

    HAS_OBS_CONTEXT = 'HAS OBS CONTEXT'
    """Target Content Items shall convey any specialization of Observation
    Context needed for unambiguous documentation of the Source Content Item."""

    HAS_PROPERTIES = 'HAS PROPERTIES'
    """Description of properties of the Source Content Item."""

    INFERRED_FROM = 'INFERRED FROM'
    """Source Content Item conveys a measurement or other inference made from
    the Target Content Items. Denotes the supporting evidence for a measurement
    or judgment."""

    SELECTED_FROM = 'SELECTED FROM'
    """Source Content Item conveys spatial or temporal coordinates selected
    from the Target Content Item(s)."""


class PixelOriginInterpretationValues(Enum):

    """Enumerated values for attribute Pixel Origin Interpretation."""

    FRAME = 'FRAME'
    """Relative to the individual frame."""

    VOLUME = 'VOLUME'
    """Relative to the Total Image Matrix (for tiled images)."""
