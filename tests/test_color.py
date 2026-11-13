import unittest

import numpy as np
import pytest
from PIL.ImageCms import ImageCmsProfile, createProfile

from highdicom.color import (
    _D50_WHITEPOINT_X,
    _D50_WHITEPOINT_Y,
    _D50_WHITEPOINT_Z,
    _lab_to_rgb,
    _rgb_to_lab,
    _rgb_to_xyz,
    _xyz_to_rgb,
    CIELabColor,
    ColorManager,
)


@pytest.mark.parametrize(
    'l_in,a_in,b_in,out',
    [
        [0.0, -128.0, -128.0, (0x0000, 0x0000, 0x0000)],
        [100.0, -128.0, -128.0, (0xFFFF, 0x0000, 0x0000)],
        [100.0, 0.0, 0.0, (0xFFFF, 0x8080, 0x8080)],
        [100.0, 0.0, 0.0, (0xFFFF, 0x8080, 0x8080)],
        [100.0, 127.0, 127.0, (0xFFFF, 0xFFFF, 0xFFFF)],
        [100.0, -128.0, 127.0, (0xFFFF, 0x0000, 0xFFFF)],
    ]
)
def test_cielab(l_in, a_in, b_in, out):
    color = CIELabColor(l_in, a_in, b_in)
    assert color.value == out


@pytest.mark.parametrize(
    # Expected values are CIELab relative to the D50 white point (the
    # illuminant of the ICC Profile Connection Space, which is what DICOM
    # uses), as produced by the sample code of the CSS Color Module Level 4
    # and by DCMTK's IODCIELabUtil
    'r,g,b,l_out,a_out,b_out',
    [
        [0, 0, 0, 0.0, 0.0, 0.0],
        [255, 0, 0, 54.291, 80.805, 69.891],
        [0, 255, 0, 87.819, -79.271, 80.995],
        [0, 0, 255, 29.568, 68.287, -112.030],
        [0, 255, 255, 90.666, -50.656, -14.962],
        [255, 255, 0, 97.607, -15.750, 93.394],
        [255, 0, 255, 60.169, 93.540, -60.501],
        [255, 255, 255, 100.0, 0.0, 0.0],
        [45, 123, 198, 49.75, -3.84, -46.70],
    ]
)
def test_from_rgb(r, g, b, l_out, a_out, b_out):
    color = CIELabColor.from_rgb(r, g, b)

    assert abs(color.l_star - l_out) < 0.1
    assert abs(color.a_star - a_out) < 0.1
    assert abs(color.b_star - b_out) < 0.1

    l_star, a_star, b_star = color.lab
    assert abs(l_star - l_out) < 0.1
    assert abs(a_star - a_out) < 0.1
    assert abs(b_star - b_out) < 0.1

    assert color.to_rgb() == (r, g, b)


def test_to_rgb_invalid():
    # A color that cannot be represented with RGB
    color = CIELabColor(93.21, 117.12, -100.7)

    with pytest.raises(ValueError):
        color.to_rgb()

    # With clip=True, will clip to closest representable value
    r, g, b = color.to_rgb(clip=True)
    assert r == 255
    assert g == 107
    assert b == 255


@pytest.mark.parametrize(
    'color,r_out,g_out,b_out',
    [
        ['black', 0, 0, 0],
        ['white', 255, 255, 255],
        ['red', 255, 0, 0],
        ['green', 0, 128, 0],
        ['blue', 0, 0, 255],
        ['yellow', 255, 255, 0],
        ['orange', 255, 165, 0],
        ['DARKORCHID', 153, 50, 204],
        ['LawnGreen', 124, 252, 0],
        ['#232489', 0x23, 0x24, 0x89],
        ['#567832', 0x56, 0x78, 0x32],
        ['#a6e83c', 0xa6, 0xe8, 0x3c],
    ]
)
def test_from_string(color, r_out, g_out, b_out):
    color = CIELabColor.from_string(color)
    r, g, b = color.to_rgb()

    assert r == r_out
    assert g == g_out
    assert b == b_out


def test_from_dicom():
    v = (1000, 3456, 4218)
    color = CIELabColor.from_dicom_value(v)
    assert color.value == v


@pytest.mark.parametrize(
    'l_in,a_in,b_in',
    [
        (-1.0, -128.0, -128.0),
        (100.1, -128.0, -128.0),
        (100.0, -128.1, 127.0),
        (100.0, -128.0, 127.1),
    ]
)
def test_cielab_invalid(l_in, a_in, b_in):
    with pytest.raises(ValueError):
        CIELabColor(l_in, a_in, b_in)


# Reference colors used by DCMTK's test suite for IODCIELabUtil: sRGB (8 bit)
# and the corresponding D50 CIELab (ICC PCS) values, as computed by the sample
# code of the CSS Color Module Level 4. The first part are CSS named colors,
# the second part are colors from 3D Slicer's "General Anatomy" color table.
REFERENCE_COLORS = [
    ('black', 0, 0, 0, 0.000, 0.000, 0.000),
    ('white', 255, 255, 255, 100.000, 0.000, 0.000),
    ('gray', 128, 128, 128, 53.585, 0.000, 0.000),
    ('silver', 192, 192, 192, 77.704, 0.000, 0.000),
    ('red', 255, 0, 0, 54.291, 80.805, 69.891),
    ('lime', 0, 255, 0, 87.819, -79.271, 80.995),
    ('blue', 0, 0, 255, 29.568, 68.287, -112.030),
    ('yellow', 255, 255, 0, 97.607, -15.750, 93.394),
    ('cyan', 0, 255, 255, 90.666, -50.656, -14.962),
    ('magenta', 255, 0, 255, 60.169, 93.540, -60.501),
    ('maroon', 128, 0, 0, 26.165, 48.473, 39.439),
    ('navy', 0, 0, 128, 11.335, 40.964, -67.203),
    ('olive', 128, 128, 0, 52.150, -9.448, 56.024),
    ('teal', 0, 128, 128, 47.986, -30.387, -8.975),
    ('purple', 128, 0, 128, 29.692, 56.112, -36.293),
    ('orange', 255, 165, 0, 75.590, 27.516, 79.121),
    ('cornflowerblue', 100, 149, 237, 61.233, 3.047, -50.188),
    ('darkslategray', 47, 79, 79, 31.141, -12.261, -3.937),
    ('hotpink', 255, 105, 180, 65.860, 63.258, -9.644),
    ('indigo', 75, 0, 130, 19.715, 47.029, -54.278),
    ('lavender', 230, 230, 250, 91.742, 2.775, -9.724),
    ('midnightblue', 25, 25, 112, 14.929, 25.955, -50.904),
    ('peru', 205, 133, 63, 62.253, 23.948, 48.413),
    ('springgreen', 0, 255, 127, 88.436, -72.499, 45.977),
    ('CSF space', 85, 188, 255, 72.318, -15.277, -42.679),
    ('aorta', 224, 97, 76, 57.648, 49.583, 37.620),
    ('bile', 0, 145, 30, 52.261, -51.068, 46.791),
    ('bone', 241, 214, 145, 86.748, 2.835, 37.696),
    ('brain', 250, 250, 225, 97.780, -3.106, 12.048),
    ('capillary', 183, 156, 220, 68.539, 19.645, -28.946),
    ('cartilage', 111, 184, 210, 70.737, -18.519, -20.706),
    ('gray matter', 200, 200, 235, 81.452, 5.183, -17.387),
    ('liver', 221, 130, 101, 64.063, 33.878, 31.516),
    ('muscle', 192, 104, 88, 54.284, 34.766, 25.456),
    ('pancreas', 249, 180, 111, 78.903, 20.532, 45.266),
    ('skin', 177, 122, 101, 56.608, 20.206, 20.612),
    ('spleen', 157, 108, 162, 52.334, 27.009, -21.229),
    ('thyroid gland', 62, 162, 114, 59.883, -39.102, 16.066),
    ('vein', 0, 151, 206, 57.968, -19.204, -38.353),
    ('white matter', 250, 250, 210, 97.515, -4.805, 19.293),
]


@pytest.mark.parametrize('name,r,g,b,l_out,a_out,b_out', REFERENCE_COLORS)
def test_reference_colors(name, r, g, b, l_out, a_out, b_out):
    # RGB -> CIELab must match the D50 reference values
    l_star, a_star, b_star = _rgb_to_lab(r, g, b)
    assert abs(l_star - l_out) < 0.03
    assert abs(a_star - a_out) < 0.03
    assert abs(b_star - b_out) < 0.03

    # The reference values are rounded to 3 decimals, which may shift a
    # component that lies exactly on a gamut boundary by one
    r_out, g_out, b_out_ = (round(c) for c in _lab_to_rgb(l_out, a_out, b_out))
    assert abs(r_out - r) <= 1
    assert abs(g_out - g) <= 1
    assert abs(b_out_ - b) <= 1


def test_white_point():
    # sRGB white maps exactly onto the D50 white point of the ICC PCS
    assert _rgb_to_xyz(255, 255, 255) == pytest.approx(
        (_D50_WHITEPOINT_X, _D50_WHITEPOINT_Y, _D50_WHITEPOINT_Z),
        abs=1e-9,
    )
    assert _rgb_to_lab(255, 255, 255) == pytest.approx((100.0, 0.0, 0.0))

    # The D50 white point of ICC v4.3 Table 14 maps back onto sRGB white
    assert _xyz_to_rgb(0.96422, 1.0, 0.82521) == pytest.approx(
        (255.0, 255.0, 255.0), abs=0.1
    )

    # White and black have the expected DICOM encodings (ICC v4.3 Table 14)
    assert CIELabColor.from_rgb(255, 255, 255).value == (0xFFFF, 0x8080, 0x8080)
    assert CIELabColor.from_rgb(0, 0, 0).value == (0x0000, 0x8080, 0x8080)


@pytest.mark.parametrize(
    'value,rgb_out',
    [
        # Values checked against DCMTK (and PixelMed), all of which lie
        # marginally outside the sRGB gamut and therefore require clipping
        [(35732, 48892, 14692), (181, 82, 255)],
        [(0, 0x8000, 0x8000), (0, 0, 1)],
        # Saturated blue, slightly outside the sRGB gamut in linear RGB
        [(19378, 50557, 3680), (0, 0, 255)],
    ]
)
def test_dicom_value_to_rgb_out_of_gamut(value, rgb_out):
    color = CIELabColor.from_dicom_value(value)

    with pytest.raises(ValueError):
        color.to_rgb()

    assert color.to_rgb(clip=True) == rgb_out


@pytest.mark.parametrize(
    'r,g,b',
    # Values from DCMTK's (and PixelMed's) test suite
    [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (225, 190, 150),
        (200, 200, 200),
        (128, 174, 128),
        (221, 130, 101),
        (0x51, 0x5d, 0xe5),
        (0x4c, 0x6e, 0xda),
    ]
)
def test_rgb_round_trip(r, g, b):
    # An 8 bit RGB triple must survive the round trip through the 16 bit
    # DICOM CIELab encoding unchanged
    assert CIELabColor.from_rgb(r, g, b).to_rgb() == (r, g, b)


def test_rgb_round_trip_cube():
    # As above, but for a sample of the whole RGB cube
    for r in range(0, 256, 17):
        for g in range(0, 256, 17):
            for b in range(0, 256, 17):
                assert CIELabColor.from_rgb(r, g, b).to_rgb() == (r, g, b)


class TestColorManager(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._icc_profile = ImageCmsProfile(createProfile('sRGB')).tobytes()

    def test_construction(self) -> None:
        ColorManager(self._icc_profile)

    def test_construction_without_profile(self) -> None:
        with pytest.raises(TypeError):
            ColorManager()  # type: ignore

    def test_transform_frame(self) -> None:
        manager = ColorManager(self._icc_profile)
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 255
        output = manager.transform_frame(frame)
        assert output.shape == frame.shape
        assert output.dtype == frame.dtype

    def test_transform_frame_wrong_shape(self) -> None:
        manager = ColorManager(self._icc_profile)
        frame = np.ones((10, 10), dtype=np.uint8) * 255
        with pytest.raises(ValueError):
            manager.transform_frame(frame)
