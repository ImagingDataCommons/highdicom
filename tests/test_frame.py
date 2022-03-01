from unittest import TestCase
from pathlib import Path

import numpy as np
import pytest
from pydicom.uid import (
    JPEG2000Lossless,
    JPEGLSLossless,
    JPEGBaseline8Bit,
)

from highdicom.frame import decode_frame, encode_frame


class TestDecodeFrame(TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        self._test_files_dir = file_path.parent.parent.joinpath(
            'data',
            'test_files'
        )

    def test_jpeg_rgb(self):
        filepath = str(self._test_files_dir.joinpath('frame_rgb.jpeg'))
        with open(filepath, 'br') as fp:
            compressed_frame = fp.read()
        rows = 80
        columns = 80
        samples_per_pixel = 3
        bits_allocated = 8
        frame = decode_frame(
            value=compressed_frame,
            transfer_syntax_uid=JPEGBaseline8Bit,
            rows=rows,
            columns=columns,
            samples_per_pixel=samples_per_pixel,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='YBR_FULL_422',
            pixel_representation=0,
            planar_configuration=0
        )
        assert frame.shape[0] == rows
        assert frame.shape[1] == columns
        assert frame.shape[2] == samples_per_pixel
        assert str(frame.dtype) == f'uint{bits_allocated}'
        np.testing.assert_allclose(
            frame[0, 0, :],
            [255, 255, 255],
            atol=1.
        )
        np.testing.assert_allclose(
            frame[0, 30, :],
            [255, 0, 0],
            atol=1.
        )
        np.testing.assert_allclose(
            frame[0, 50, :],
            [0, 255, 0],
            atol=1.
        )
        np.testing.assert_allclose(
            frame[0, 70, :],
            [0, 0, 255],
            atol=1.
        )

    def test_jpeg_rgb_empty(self):
        filepath = str(self._test_files_dir.joinpath('frame_rgb_empty.jpeg'))
        with open(filepath, 'br') as fp:
            compressed_frame = fp.read()
        rows = 16
        columns = 32
        samples_per_pixel = 3
        bits_allocated = 8
        frame = decode_frame(
            value=compressed_frame,
            transfer_syntax_uid=JPEGBaseline8Bit,
            rows=rows,
            columns=columns,
            samples_per_pixel=samples_per_pixel,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='YBR_FULL',
            pixel_representation=0,
            planar_configuration=0
        )
        assert frame.shape[0] == rows
        assert frame.shape[1] == columns
        assert frame.shape[2] == samples_per_pixel
        assert str(frame.dtype) == f'uint{bits_allocated}'
        assert frame[0, 0, 0] == 255

    def test_jpeg_rgb_wrong_photometric_interpretation(self):
        with pytest.raises(ValueError):
            decode_frame(
                value=b'',
                transfer_syntax_uid=JPEGBaseline8Bit,
                rows=16,
                columns=32,
                samples_per_pixel=3,
                bits_allocated=8,
                bits_stored=8,
                photometric_interpretation='MONOCHROME',
                pixel_representation=0,
                planar_configuration=0
            )

    def test_jpeg_rgb_missing_planar_configuration(self):
        with pytest.raises(ValueError):
            decode_frame(
                value=b'',
                transfer_syntax_uid=JPEGBaseline8Bit,
                rows=16,
                columns=32,
                samples_per_pixel=3,
                bits_allocated=8,
                bits_stored=8,
                photometric_interpretation='RGB',
                pixel_representation=0,
            )


class TestEncodeFrame(TestCase):

    def setUp(self):
        super().setUp()

    def test_jpeg_rgb(self):
        bits_allocated = 8
        frame = np.ones((16, 32, 3), dtype=np.dtype(f'uint{bits_allocated}'))
        frame *= 255
        compressed_frame = encode_frame(
            frame,
            transfer_syntax_uid=JPEGBaseline8Bit,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='YBR_FULL_422',
            pixel_representation=0,
            planar_configuration=0
        )
        assert compressed_frame.startswith(b'\xFF\xD8')
        assert compressed_frame.endswith(b'\xFF\xD9')

    def test_jpeg_monochrome(self):
        bits_allocated = 8
        frame = np.zeros((16, 32), dtype=np.dtype(f'uint{bits_allocated}'))
        compressed_frame = encode_frame(
            frame,
            transfer_syntax_uid=JPEGBaseline8Bit,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='MONOCHROME1',
            pixel_representation=0
        )
        assert compressed_frame.startswith(b'\xFF\xD8')
        assert compressed_frame.endswith(b'\xFF\xD9')

    def test_jpeg2000_rgb(self):
        bits_allocated = 8
        frame = np.ones((16, 32, 3), dtype=np.dtype(f'uint{bits_allocated}'))
        frame *= 255
        compressed_frame = encode_frame(
            frame,
            transfer_syntax_uid=JPEG2000Lossless,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='YBR_FULL',
            pixel_representation=0,
            planar_configuration=0
        )
        assert compressed_frame.startswith(b'\x00\x00\x00\x0C\x6A\x50\x20')
        assert compressed_frame.endswith(b'\xFF\xD9')
        decoded_frame = decode_frame(
            value=compressed_frame,
            transfer_syntax_uid=JPEG2000Lossless,
            rows=frame.shape[0],
            columns=frame.shape[1],
            samples_per_pixel=frame.shape[2],
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='YBR_FULL',
            pixel_representation=0,
            planar_configuration=0
        )
        np.testing.assert_array_equal(frame, decoded_frame)

    def test_jpeg2000_monochrome(self):
        bits_allocated = 16
        frame = np.zeros((16, 32), dtype=np.dtype(f'uint{bits_allocated}'))
        compressed_frame = encode_frame(
            frame,
            transfer_syntax_uid=JPEG2000Lossless,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='MONOCHROME2',
            pixel_representation=0,
        )
        assert compressed_frame.startswith(b'\x00\x00\x00\x0C\x6A\x50\x20')
        assert compressed_frame.endswith(b'\xFF\xD9')
        decoded_frame = decode_frame(
            value=compressed_frame,
            transfer_syntax_uid=JPEG2000Lossless,
            rows=frame.shape[0],
            columns=frame.shape[1],
            samples_per_pixel=1,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='MONOCHROME2',
            pixel_representation=0,
            planar_configuration=0
        )
        np.testing.assert_array_equal(frame, decoded_frame)

    def test_jpegls_rgb(self):
        bits_allocated = 8
        frame = np.ones((16, 32, 3), dtype=np.dtype(f'uint{bits_allocated}'))
        frame *= 255
        compressed_frame = encode_frame(
            frame,
            transfer_syntax_uid=JPEGLSLossless,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='YBR_FULL',
            pixel_representation=0,
            planar_configuration=0
        )
        assert compressed_frame.startswith(b'\xFF\xD8')
        assert compressed_frame.endswith(b'\xFF\xD9')
        decoded_frame = decode_frame(
            value=compressed_frame,
            transfer_syntax_uid=JPEGLSLossless,
            rows=frame.shape[0],
            columns=frame.shape[1],
            samples_per_pixel=frame.shape[2],
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='YBR_FULL',
            pixel_representation=0,
            planar_configuration=0
        )
        np.testing.assert_array_equal(frame, decoded_frame)

    def test_jpegls_monochrome(self):
        bits_allocated = 16
        frame = np.zeros((16, 32), dtype=np.dtype(f'uint{bits_allocated}'))
        compressed_frame = encode_frame(
            frame,
            transfer_syntax_uid=JPEGLSLossless,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='MONOCHROME2',
            pixel_representation=0,
        )
        assert compressed_frame.startswith(b'\xFF\xD8')
        assert compressed_frame.endswith(b'\xFF\xD9')
        decoded_frame = decode_frame(
            value=compressed_frame,
            transfer_syntax_uid=JPEG2000Lossless,
            rows=frame.shape[0],
            columns=frame.shape[1],
            samples_per_pixel=1,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
            photometric_interpretation='MONOCHROME2',
            pixel_representation=0,
            planar_configuration=0
        )
        np.testing.assert_array_equal(frame, decoded_frame)

    def test_jpeg_rgb_wrong_photometric_interpretation(self):
        frame = np.ones((16, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            encode_frame(
                frame,
                transfer_syntax_uid=JPEGBaseline8Bit,
                bits_allocated=8,
                bits_stored=8,
                photometric_interpretation='RGB',
                pixel_representation=0,
                planar_configuration=0
            )

    def test_jpeg_rgb_wrong_planar_configuration(self):
        frame = np.ones((16, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            encode_frame(
                frame,
                transfer_syntax_uid=JPEGBaseline8Bit,
                bits_allocated=8,
                bits_stored=8,
                photometric_interpretation='YBR_FULL_422',
                pixel_representation=0,
                planar_configuration=1
            )

    def test_jpeg2000_monochrome_wrong_photometric_interpretation(self):
        frame = np.zeros((16, 32), dtype=np.uint16)
        with pytest.raises(ValueError):
            encode_frame(
                frame,
                transfer_syntax_uid=JPEG2000Lossless,
                bits_allocated=16,
                bits_stored=16,
                photometric_interpretation='MONOCHROME',
                pixel_representation=0,
            )

    def test_jpeg2000_monochrome_wrong_pixel_representation(self):
        frame = np.zeros((16, 32), dtype=np.uint16)
        with pytest.raises(ValueError):
            encode_frame(
                frame,
                transfer_syntax_uid=JPEG2000Lossless,
                bits_allocated=16,
                bits_stored=16,
                photometric_interpretation='MONOCHROME2',
                pixel_representation=1,
            )
