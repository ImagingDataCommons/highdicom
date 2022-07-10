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
