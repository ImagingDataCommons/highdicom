"""Utiliies for working with colors."""
import logging
from io import BytesIO
from collections.abc import Sequence

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


# White points, defined by their 4-figure CIE x,y chromaticities, as in the
# sample code of the CSS Color Module Level 4
# (https://www.w3.org/TR/css-color-4/#color-conversion-code).
#
# DICOM encodes CIELab values in the same way as the ICC Profile Connection
# Space (PCS, see PS3.3 C.10.7.1.1), i.e. relative to the D50 illuminant,
# whereas the native white point of sRGB is D65. The conversions below
# therefore chromatically adapt between the two using the linear Bradford
# transform. Note that for neutral (gray) colors the results are the same
# either way, for chromatic colors the a* and b* values differ slightly.
_D50_WHITEPOINT_X = 0.3457 / 0.3585
_D50_WHITEPOINT_Y = 1.0
_D50_WHITEPOINT_Z = (1.0 - 0.3457 - 0.3585) / 0.3585


def _rgb_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert an sRGB color to CIE XYZ representation.

    Outputs are relative to the D50 white point (ICC PCS XYZ) and scaled
    between 0.0 and the white point (0.9642, 1.0, 0.8251). As a private
    function, no checks are performed that input values are valid, and output
    values are not clipped.

    Parameters
    ----------
    r: float
        Red component between 0 and 255 (inclusive).
    g: float
        Green component between 0 and 255 (inclusive).
    b: float
        Blue component between 0 and 255 (inclusive).

    Returns
    -------
    x: float
        X component as a float between 0.0 and 0.9642.
    y: float
        Y component as a float between 0.0 and 1.0.
    z: float
        Z component as a float between 0.0 and 0.8251.

    """
    # Matches the implementation of DCMTK's IODCIELabUtil::rgb2Xyz()
    def invert_gamma(c: float) -> float:
        # sRGB transfer function (IEC 61966-2-1), gamma-encoded to linear
        c = c / 255.0
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        return c / 12.92

    r = invert_gamma(r)
    g = invert_gamma(g)
    b = invert_gamma(b)

    # Linear RGB to CIE XYZ (D65)
    x65 = (
        (506752.0 / 1228815.0) * r +
        (87881.0 / 245763.0) * g +
        (12673.0 / 70218.0) * b
    )
    y65 = (
        (87098.0 / 409605.0) * r +
        (175762.0 / 245763.0) * g +
        (12673.0 / 175545.0) * b
    )
    z65 = (
        (7918.0 / 409605.0) * r +
        (87881.0 / 737289.0) * g +
        (1001167.0 / 1053270.0) * b
    )

    # Chromatic adaptation from D65 to D50 (linear Bradford)
    x = (
        1.0479297925449969 * x65 +
        0.022946870601609652 * y65 -
        0.05019226628920524 * z65
    )
    y = (
        0.02962780877005599 * x65 +
        0.9904344267538799 * y65 -
        0.017073799063418826 * z65
    )
    z = (
        -0.009243040646204504 * x65 +
        0.015055191490298152 * y65 +
        0.7518742814281371 * z65
    )

    return x, y, z


def _xyz_to_rgb(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert a CIE XYZ color to sRGB representation.

    Inputs are relative to the D50 white point (ICC PCS XYZ) and scaled
    between 0.0 and the white point (0.9642, 1.0, 0.8251). As a private
    function, no checks are performed that input values are valid, and output
    values are not clipped. Consequently colors that lie outside the sRGB
    gamut give values outside the 0.0 to 255.0 range.

    Parameters
    ----------
    x: float
        X component as a float between 0.0 and 0.9642.
    y: float
        Y component as a float between 0.0 and 1.0.
    z: float
        Z component as a float between 0.0 and 0.8251.

    Returns
    -------
    r: float
        Red component between 0.0 and 255.0 (inclusive).
    g: float
        Green component between 0.0 and 255.0 (inclusive).
    b: float
        Blue component between 0.0 and 255.0 (inclusive).

    """
    # Matches the implementation of DCMTK's IODCIELabUtil::xyz2Rgb(), except
    # that out-of-gamut values are not clipped here (this is left to the
    # caller, which may instead want to report them as out of gamut)
    # Chromatic adaptation from D50 to D65 (linear Bradford)
    x65 = (
        0.955473421488075 * x -
        0.02309845494876471 * y +
        0.06325924320057072 * z
    )
    y65 = (
        -0.0283697093338637 * x +
        1.0099953980813041 * y +
        0.021041441191917323 * z
    )
    z65 = (
        0.012314014864481998 * x -
        0.020507649298898964 * y +
        1.330365926242124 * z
    )

    # CIE XYZ (D65) to linear RGB
    r = (
        (12831.0 / 3959.0) * x65 -
        (329.0 / 214.0) * y65 -
        (1974.0 / 3959.0) * z65
    )
    g = (
        (-851781.0 / 878810.0) * x65 +
        (1648619.0 / 878810.0) * y65 +
        (36519.0 / 878810.0) * z65
    )
    b = (
        (705.0 / 12673.0) * x65 -
        (2585.0 / 12673.0) * y65 +
        (705.0 / 667.0) * z65
    )

    def apply_gamma(c: float) -> float:
        # sRGB transfer function (IEC 61966-2-1), linear to gamma-encoded
        if c > 0.0031308:
            return 1.055 * (c ** (1.0 / 2.4)) - 0.055
        return 12.92 * c

    return apply_gamma(r) * 255, apply_gamma(g) * 255, apply_gamma(b) * 255


def _xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert a CIE XYZ color to CIE Lab representation.

    Both input and output are relative to the D50 white point (ICC PCS). As a
    private function, no checks are performed that input values are valid, and
    output values are not clipped.

    Parameters
    ----------
    x: float
        X component as a float between 0.0 and 0.9642.
    y: float
        Y component as a float between 0.0 and 1.0.
    z: float
        Z component as a float between 0.0 and 0.8251.

    Returns
    -------
    l_star: float
        Lightness value in the range 0.0 (black) to 100.0 (white).
    a_star: float
        Red-green value from -128.0 (red) to 127.0 (green).
    b_star: float
        Blue-yellow value from -128.0 (blue) to 127.0 (yellow).

    """
    # Matches the implementation of DCMTK's IODCIELabUtil::xyz2Lab()
    x = x / _D50_WHITEPOINT_X
    y = y / _D50_WHITEPOINT_Y
    z = z / _D50_WHITEPOINT_Z

    def convert_component(c: float) -> float:
        # epsilon = (6/29)^3 = 216/24389, kappa = (29/3)^3 = 24389/27
        if c > 216.0 / 24389.0:
            return c ** (1.0 / 3.0)
        return ((24389.0 / 27.0) * c + 16.0) / 116.0

    x = convert_component(x)
    y = convert_component(y)
    z = convert_component(z)

    lightness = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return lightness, a, b


def _lab_to_xyz(
    l_star: float,
    a_star: float,
    b_star: float,
) -> tuple[float, float, float]:
    """Convert a CIE Lab color to CIE XYZ representation.

    Both input and output are relative to the D50 white point (ICC PCS).
    Outputs are scaled between 0.0 and the white point (0.9642, 1.0, 0.8251).
    As a private function, no checks are performed that input values are
    valid, and output values are not clipped.

    Parameters
    ----------
    l_star: float
        Lightness value in the range 0.0 (black) to 100.0 (white).
    a_star: float
        Red-green value from -128.0 (red) to 127.0 (green).
    b_star: float
        Blue-yellow value from -128.0 (blue) to 127.0 (yellow).

    Returns
    -------
    x: float
        X component as a float between 0.0 and 0.9642.
    y: float
        Y component as a float between 0.0 and 1.0.
    z: float
        Z component as a float between 0.0 and 0.8251.

    """
    # Matches the implementation of DCMTK's IODCIELabUtil::lab2Xyz()
    y = (l_star + 16) / 116
    x = a_star / 500 + y
    z = y - b_star / 200

    def convert_component(c: float) -> float:
        # c > 6/29 is equivalent to c ** 3 > epsilon
        if c > 6.0 / 29.0:
            return c ** 3
        return (116.0 * c - 16.0) / (24389.0 / 27.0)

    x = _D50_WHITEPOINT_X * convert_component(x)
    y = _D50_WHITEPOINT_Y * convert_component(y)
    z = _D50_WHITEPOINT_Z * convert_component(z)

    return x, y, z


def _rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert an sRGB color to CIE Lab representation.

    The output is relative to the D50 white point (ICC PCS), as used by DICOM.
    As a private function, no checks are performed that input values are
    valid, and output values are not clipped.

    Parameters
    ----------
    r: float
        Red component between 0 and 255 (inclusive).
    g: float
        Green component between 0 and 255 (inclusive).
    b: float
        Blue component between 0 and 255 (inclusive).

    Returns
    -------
    l_star: float
        Lightness value in the range 0.0 (black) to 100.0 (white).
    a_star: float
        Red-green value from -128.0 (red) to 127.0 (green).
    b_star: float
        Blue-yellow value from -128.0 (blue) to 127.0 (yellow).

    """
    return _xyz_to_lab(*_rgb_to_xyz(r, g, b))


def _lab_to_rgb(
    l_star: float,
    a_star: float,
    b_star: float,
) -> tuple[float, float, float]:
    """Convert a CIE Lab color to sRGB representation.

    The input is relative to the D50 white point (ICC PCS), as used by DICOM.
    As a private function, no checks are performed that input values are
    valid, and output values are not clipped. Lab colors that cannot be
    represented in sRGB will have values outside to 0.0 to 255.0 range.

    Parameters
    ----------
    l_star: float
        Lightness value in the range 0.0 (black) to 100.0 (white).
    a_star: float
        Red-green value from -128.0 (red) to 127.0 (green).
    b_star: float
        Blue-yellow value from -128.0 (blue) to 127.0 (yellow).

    Returns
    -------
    r: float
        Red component.
    g: float
        Green component.
    b: float
        Blue component.

    """
    return _xyz_to_rgb(*_lab_to_xyz(l_star, a_star, b_star))


class CIELabColor:

    """Class to represent a color value in CIELab color space.

    Various places in DICOM use the CIE-Lab color space to represent colors.
    This class is used to pass these colors around and convert them to and from
    RGB representation.

    Examples
    --------

    Construct a CIE-Lab color directly and convert it to RGB:

    >>> import highdicom as hd
    >>>
    >>> color = hd.color.CIELabColor(50.0, 34.0, 12.4)
    >>> print(color.to_rgb())
    (175, 94, 100)

    Construct a CIE-Lab color from an RGB color and examine the Lab components:

    >>> import highdicom as hd
    >>>
    >>> color = hd.color.CIELabColor.from_rgb(0, 255, 0)
    >>> print(color.l_star, color.a_star, color.b_star)
    87.81872281986725 -79.27237354085602 80.99610894941634

    Construct a CIE-Lab color from the name of a well-known color:

    >>> import highdicom as hd
    >>>
    >>> color = hd.color.CIELabColor.from_string('turquoise')
    >>> print(color.l_star, color.a_star, color.b_star)
    80.96131837949187 -45.13618677042801 -4.680933852140072

    Within DICOM files, the three components are represented using scaled and
    shifted unsigned 16 bit integer values. You can move between these
    representations like this:

    >>> import highdicom as hd
    >>>
    >>> color = hd.color.CIELabColor.from_string('orange')
    >>> # Print the values that would actually be stored in a DICOM file
    >>> print(color.value)
    (49538, 39968, 53230)
    >>> # Create a color directly from these values
    >>> color2 = hd.color.CIELabColor.from_dicom_value((49538, 39968, 53230))
    >>> print(color2.to_rgb())
    (255, 165, 0)

    """

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
        l_val = round(l_star * 0xFFFF / 100.0)
        a_val = round((a_star + 128.0) * 0xFFFF / 255.0)
        b_val = round((b_star + 128.0) * 0xFFFF / 255.0)
        self._value = (l_val, a_val, b_val)

    @property
    def value(self) -> tuple[int, int, int]:
        """

        Tuple[int]:
            Value formatted as a triplet of 16 bit unsigned integers (as stored
            within DICOM). This consists of a triplet of 16-bit unsigned
            integers for the L*, a*, and b* components in that order. The L*
            component is linearly scaled from the typical range of 0 to 100.0
            to the 16 bit integer range (0 to 65535, or ``0xFFFF``) and rounded
            to the nearest integer. The a* and b* components are mapped from
            their typical range (-128.0 to 127.0) by shifting to an unsigned
            integer range by adding 128.0, then linearly scaling this to the 16
            bit integer range and rounding to the nearest integer. Thus, -128.0
            is represented as 0 (``0x0000``), 0.0 as 32896 (``0x8080``), and
            127.0 as 65535 (``0xFFFF``).

        """
        return self._value

    @property
    def l_star(self) -> float:
        """float: L* component as value between 0 and 100.0."""
        return self._value[0] * (100.0 / 0xFFFF)

    @property
    def a_star(self) -> float:
        """float: a* component as value between -128.0 and 127.0."""
        return self._value[1] * (255.0 / 0xFFFF) - 128.0

    @property
    def b_star(self) -> float:
        """float: b* component as value between -128.0 and 127.0."""
        return self._value[2] * (255.0 / 0xFFFF) - 128.0

    @property
    def lab(self) -> tuple[float, float, float]:
        """

        float:
            L* component as value between 0 and 100.0.
        float:
            a* component as value between -128.0 and 127.0.
        float:
            b* component as value between -128.0 and 127.0.

        """
        return (self.l_star, self.a_star, self.b_star)

    @classmethod
    def from_dicom_value(cls, value: Sequence[int]) -> Self:
        """Create a color from the DICOM integer representation.

        Parameters
        ----------
        value: Sequence[int]
            The DICOM representation of a CIELab color. This consists of a
            triplet of 16-bit unsigned integers for the L*, a*, and b*
            components in that order. The L* component should be linearly
            scaled from the typical range of 0 to 100.0 to the 16 bit integer
            range (0 to 65535, or ``0xFFFF``) and rounded to the nearest
            integer. The a* and b* components should be mapped from their
            typical range (-128.0 to 127.0) by shifting to an unsigned integer
            range by adding 128.0, then linearly scaling this to the 16 bit
            integer range and rounding to the nearest integer. Thus, -128.0
            should be represented as 0 (``0x0000``), 0.0 as 32896 (``0x8080``),
            and 127.0 as 65535 (``0xFFFF``).

        Returns
        -------
        Self
            Color constructed from the supplied DICOM values.

        """
        if len(value) != 3:
            raise ValueError("Argument 'value' must have length 3.")

        for v in value:
            if not isinstance(v, int):
                raise TypeError('Elements must be integers.')

            if v < 0 or v > 0xFFFF:
                raise ValueError(
                    "All values must lie in range 0 to 0xFFFF"
                )

        c = cls.__new__(cls)
        c._value = tuple(value)
        return c

    @classmethod
    def from_rgb(cls, r: float, g: float, b: float) -> Self:
        """Create the color from RGB values.

        Parameters
        ----------
        r: int | float
            Red component value in range 0 to 255 (inclusive).
        g: float
            Green component value in range 0 to 255 (inclusive).
        b: float
            Blue component value in range 0 to 255 (inclusive).

        Returns
        -------
        Self
            Color constructed from the supplied RGB values.

        Note
        ----

        Some valid CIELab colors lie outside the valid RGB range, and therefore
        cannot be created with this method.

        """
        for c in [r, g, b]:
            if not (0 <= c <= 255):
                raise ValueError(
                    'Each RGB component must lie in the range 0 to 255.'
                )

        return cls(*_rgb_to_lab(r, g, b))

    @classmethod
    def from_string(cls, color: str) -> Self:
        """Construct from a string representing a color.

        Parameters
        ----------
        color: str
            Should be a string understood by PIL's ``getrgb()`` function (see
            `here
            <https://pillow.readthedocs.io/en/stable/reference/ImageColor.html#color-names>`_
            for the documentation of that function or `here
            <https://drafts.csswg.org/css-color-4/#named-colors>`_ for the
            original list of colors). This includes many case-insensitive color
            names (e.g. ``"red"``, ``"Crimson"``, or ``"INDIGO"``), hex codes
            (e.g. ``"#ff7733"``) or decimal integers in the format of this
            example: ``"RGB(255, 255, 0)"``.

        Returns
        -------
        Self
            Color constructed from the supplied string.

        """
        return cls.from_rgb(*ImageColor.getrgb(color))

    def to_rgb(self, clip: bool = False) -> tuple[int, int, int]:
        """Get an RGB representation of this color.

        Note that the full gamut of representable CIE-Lab colors is a super-set
        of those representable with RGB. By default, if the color is not
        representable as an RGB color, a ``ValueError`` will be raised.

        Parameters
        ----------
        clip: bool, optional
            If the color cannot be represented in RGB, clip the values to the
            range 0 to 255 to give the closest representable RGB color. If
            ``False``, colors that cannot be represented in RGB will raise a
            ``ValueError``.

        Returns
        -------
        int:
            Red component, between 0 and 255 (inclusive).
        int:
            Green component, between 0 and 255 (inclusive).
        int:
            Blue component, between 0 and 255 (inclusive).

        """
        r, g, b = _lab_to_rgb(self.l_star, self.a_star, self.b_star)

        def _check_component(c):
            if 0 <= round(c) <= 255:
                return round(c)
            else:
                if clip:
                    return max(min(c, 255), 0)
                else:
                    raise ValueError(
                        'This color is not representable in RGB color space. '
                        "Use 'clip=True' to clip to the nearest representable "
                        'value.'
                    )

        return _check_component(r), _check_component(g), _check_component(b)


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
