import unittest

import numpy as np
import pytest
from PIL.ImageCms import ImageCmsProfile, createProfile

from highdicom.color import ColorManager, CIELabColor


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
    # Examples generated from colormine.org
    'r,g,b,l_out,a_out,b_out',
    [
        [0, 0, 0, 0.0, 0.0, 0.0],
        [255, 0, 0, 53.23, 80.11, 67.22],
        [0, 255, 0, 87.74, -86.18, 83.18],
        [0, 0, 255, 32.30, 79.20, -107.86],
        [0, 255, 255, 91.11, -48.08, -14.14],
        [255, 255, 0, 97.14, -21.56, 94.48],
        [255, 0, 255, 60.32, 98.25, -60.84],
        [255, 255, 255, 100.0, 0.0, -0.01],
        [45, 123, 198, 50.45, 2.59, -45.75],
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
    assert g == 125
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
