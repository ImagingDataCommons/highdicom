from typing import NamedTuple


class ImagePosition(NamedTuple):

    """Named tuple for image positions in the three-dimensional patient or
    slide coordinate system.
    """

    x: float
    y: float
    z: float


