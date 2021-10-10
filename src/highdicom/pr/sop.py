"""Module for SOP Classes of Presentation State (PR) IODs."""

from highdicom.base import SOPClass


class GrayscaleSoftcopyPresentationState(SOPClass):

    """SOP class for a Grayscale Softcopy Presentation State (GSPS) object.

    A GSPS object includes instructions for the presentation of an image by
    software.

    """

    def __init__(
    ):

        """
        """
        # TODO check graphic annotation units are not matrix unless its a whole
        # slide image
        pass
