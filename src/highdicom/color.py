import logging
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image, ImageCms
from PIL.ImageCms import (
    applyTransform,
    getProfileDescription,
    getProfileName,
    ImageCmsProfile,
    ImageCmsTransform,
    isIntentSupported,
)


logger = logging.getLogger(__name__)


class CIELabColor:

    """Class to represent a color value in CIELab color space."""

    def __init__(
        self,
        l_star: float,
        a_star: float,
        b_star: float
    ):
        """

        Parameters
        ----------
        l_star: float
            Lightness value in the range 0.0 (black) to 100.0 (white).
        a_star: float
            Red-green value from -128.0 (red) to 127.0 (green).
        b_star: float
            Blue-yellow value from -128.0 (blue) to 127.0 (yellow).

        """
        if l_star < 0.0 or l_star > 100.0:
            raise ValueError(
                'Value for "l_star" must lie between 0.0 (black) and 100.0'
                ' (white).'
            )
        if a_star < -128.0 or a_star > 127.0:
            raise ValueError(
                'Value for "a_star" must lie between -128.0 (red) and 127.0'
                ' (green).'
            )
        if b_star < -128.0 or b_star > 127.0:
            raise ValueError(
                'Value for "b_star" must lie between -128.0 (blue) and 127.0'
                ' (yellow).'
            )
        l_val = int(l_star * 0xFFFF / 100.0)
        a_val = int((a_star + 128.0) * 0xFFFF / 255.0)
        b_val = int((b_star + 128.0) * 0xFFFF / 255.0)
        self._value = (l_val, a_val, b_val)

    @property
    def value(self) -> Tuple[int, int, int]:
        """Tuple[int]:
            Value formatted as a triplet of 16 bit unsigned integers.
        """
        return self._value


class ColorManager:

    """Class for color management using ICC profiles."""

    def __init__(self, icc_profile: bytes):
        """

        Parameters
        ----------
        icc_profile: bytes
            ICC profile

        Raises
        ------
        ValueError
            When ICC Profile cannot be read.

        """
        try:
            self._icc_transform = self._build_icc_transform(icc_profile)
        except OSError:
            raise ValueError('Could not read ICC Profile.')

    def transform_frame(self, array: np.ndarray) -> np.ndarray:
        """Transforms a frame by applying the ICC profile.

        Parameters
        ----------
        array: numpy.ndarray
            Pixel data of a color image frame in form of an array with
            dimensions (Rows x Columns x SamplesPerPixel)

        Returns
        -------
        numpy.ndarray
            Color corrected pixel data of a image frame in form of an array
            with dimensions (Rows x Columns x SamplesPerPixel)

        Raises
        ------
        ValueError
            When `array` does not have 3 dimensions and thus does not represent
            a color image frame.

        """
        if array.ndim != 3:
            raise ValueError(
                'Array has incorrect dimensions for a color image frame.'
            )
        image = Image.fromarray(array)
        applyTransform(image, self._icc_transform, inPlace=True)
        return np.asarray(image)

    @staticmethod
    def _build_icc_transform(icc_profile: bytes) -> ImageCmsTransform:
        """Builds an ICC Transformation object.

        Parameters
        ----------
        icc_profile: bytes
            ICC Profile

        Returns
        -------
        PIL.ImageCms.ImageCmsTransform
            ICC Transformation object

        """
        profile: bytes
        try:
            profile = ImageCmsProfile(BytesIO(icc_profile))
        except OSError:
            raise ValueError('Cannot read ICC Profile in image metadata.')
        name = getProfileName(profile).strip()
        description = getProfileDescription(profile).strip()
        logger.debug(f'found ICC Profile "{name}": "{description}"')

        logger.debug('build ICC Transform')
        if hasattr(ImageCms, "Intent"):
            # This is the API for pillow>=10.0.0
            intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
            direction = ImageCms.Direction.INPUT
        else:
            # This is the API for pillow<10.0.0
            # Ideally we would simply require pillow>=10.0.0, but unfortunately
            # this would rule out supporting python < 3.8. Once we drop support
            # for 3.7 and below, we can require pillow>=10.0.0 and drop this
            # branch
            intent = ImageCms.INTENT_RELATIVE_COLORIMETRIC
            direction = ImageCms.DIRECTION_INPUT
        if not isIntentSupported(
            profile,
            intent=intent,
            direction=direction,
        ):
            raise ValueError(
                'ICC Profile does not support desired '
                'color transformation intent.'
            )
        return ImageCms.buildTransform(
            inputProfile=profile,
            outputProfile=ImageCms.createProfile('sRGB'),
            inMode='RGB',  # according to PS3.3 C.11.15.1.1
            outMode='RGB'
        )
