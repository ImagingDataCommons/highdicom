import logging
from io import BytesIO

import numpy as np
from PIL import Image, ImageCms, ImageColor
from PIL.ImageCms import (
    applyTransform,
    getProfileDescription,
    getProfileName,
    ImageCmsProfile,
    ImageCmsTransform,
    isIntentSupported,
)
from typing_extensions import Self


logger = logging.getLogger(__name__)


def _rgb_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    # Adapted from ColorUtilities module of pixelmed:
    # https://www.dclunie.com/pixelmed/software/javadoc/com/pixelmed/utils/ColorUtilities.html

    def convert_component(c: float) -> float:
        c = c / 255.0
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        return c / 12.92

    r = convert_component(r) * 100
    g = convert_component(g) * 100
    b = convert_component(b) * 100

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    return x, y, z


def _xyz_to_cielab(x: float, y: float, z: float) -> tuple[float, float, float]:
    # Adapted from ColorUtilities module of pixelmed:
    # https://www.dclunie.com/pixelmed/software/javadoc/com/pixelmed/utils/ColorUtilities.html

    x = x / 95.047
    y = y / 100.0
    z = z / 108.883

    def convert_component(c: float) -> float:
        if  c > 0.008856:
            return c ** (1.0 / 3)
        return 7.787 * x + (16.0 / 116)

    x = convert_component(x)
    y = convert_component(y)
    z = convert_component(z)

    l = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return l, a, b


def _cielab_to_xyz(l_star: float, a_star: float, b_star: float) -> tuple[float, float, float]:
    # Adapted from ColorUtilities module of pixelmed:
    # https://www.dclunie.com/pixelmed/software/javadoc/com/pixelmed/utils/ColorUtilities.html

    y = ( l_star + 16 ) / 116
    x = a_star / 500 + y
    z = y - b_star / 200

    def convert_component(c: float) -> float:
        c3 = c ** 3

        if c3 > 0.008856:
            return c3
        return (c - 16.0 / 116) / 7.787

    x = convert_component(x)
    y = convert_component(y)
    z = convert_component(z)

    x = 95.047  * x
    y = 100.0  * y
    z = 108.883  * z

    return x, y, z

def _xyz_to_rgb(x: float, y: float, z: float) -> tuple[float, float, float]:
    # Adapted from ColorUtilities module of pixelmed:
    # https://www.dclunie.com/pixelmed/software/javadoc/com/pixelmed/utils/ColorUtilities.html
    x = x / 100
    y = y / 100
    z = z / 100

    r = x *  3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y *  1.8758 + z *  0.0415
    b = x *  0.0557 + y * -0.2040 + z *  1.0570

    def convert_component(c: float) -> float:
        if c > 0.0031308:
            return 1.055 * (c ** (1 / 2.4)) - 0.055
        return 12.92 * c

    r = convert_component(r) * 255
    g = convert_component(g) * 255
    b = convert_component(b) * 255

    return r, g, b


def _rgb_to_cielab(r: float, g: float, b: float) -> tuple[float, float, float]:
    return _xyz_to_cielab(*_rgb_to_xyz(r, g, b))


def _cielab_to_rgb(l_star: float, a_star: float, b_star: float) -> tuple[float, float, float]:
    return _xyz_to_rgb(*_cielab_to_xyz(l_star, a_star, b_star))


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

    @classmethod
    def from_rgb(cls, r: float, g: float, b: float) -> Self:
        return cls(*_rgb_to_cielab(r, g, b))

    @classmethod
    def from_color(cls, color: str) -> Self:
        return cls.from_rgb(*ImageColor.getrgb(color))

    def to_rgb(self) -> tuple[int, int, int]:
        r, g, b = _cielab_to_rgb(self.l_star, self.a_star, self.b_star)
        return round(r), round(g), round(b)

    @property
    def l_star(self) -> float:
        return self._value[0] * (100.0 / 0xFFFF)

    @property
    def a_star(self) -> float:
        return self._value[1] * (255.0 / 0xFFFF) - 128.0

    @property
    def b_star(self) -> float:
        return self._value[2] * (255.0 / 0xFFFF) - 128.0

    @property
    def value(self) -> tuple[int, int, int]:
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
        except OSError as e:
            raise ValueError('Could not read ICC Profile.') from e

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
        except OSError as e:
            raise ValueError(
                'Cannot read ICC Profile in image metadata.'
            ) from e
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
