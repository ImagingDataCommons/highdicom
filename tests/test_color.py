import unittest

import numpy as np
import pytest
from PIL.ImageCms import ImageCmsProfile, createProfile

from highdicom.color import ColorManager


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
