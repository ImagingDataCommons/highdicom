from pathlib import Path

import numpy as np
import pytest
from pydicom.uid import (
    JPEG2000,
    ExplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
    JPEGLSNearLossless,
    JPEGBaseline8Bit,
    RLELossless,
)

from highdicom.frame import decode_frame, encode_frame


def test_decode_jpeg_rgb():
    pytest.importorskip("libjpeg")
    test_files_dir = Path(__file__).parents[1].joinpath(
        'data',
        'test_files'
    )
    filepath = str(test_files_dir.joinpath('frame_rgb.jpeg'))
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


def test_decode_jpeg_rgb_empty():
    pytest.importorskip("libjpeg")
    test_files_dir = Path(__file__).parents[1].joinpath(
        'data',
        'test_files'
    )
    filepath = str(test_files_dir.joinpath('frame_rgb_empty.jpeg'))
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


def test_decode_jpeg_rgb_wrong_photometric_interpretation():
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


def test_decode_jpeg_rgb_missing_planar_configuration():
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


def test_encode_native():
    bits_allocated = 16
    r, c = 16, 32
    dtype = np.dtype(f'uint{bits_allocated}')
    frame = np.random.randint(0, 1000, (r, c), dtype=dtype)
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=ExplicitVRLittleEndian,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
        planar_configuration=0
    )
    reconstructed = np.frombuffer(
        compressed_frame,
        dtype=dtype
    ).reshape(r, c)
    assert np.array_equal(reconstructed, frame)


def test_encode_native_from_big_endian():
    bits_allocated = 16
    r, c = 16, 32
    dtype = np.dtype('>u2')
    frame = np.random.randint(0, 1000, (r, c), dtype=dtype)
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=ExplicitVRLittleEndian,  # change endianness
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
        planar_configuration=0
    )
    reconstructed = np.frombuffer(
        compressed_frame,
        dtype='<u2',  # little endian
    ).reshape(r, c)
    assert np.array_equal(reconstructed, frame)


def test_encode_jpeg_rgb():
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


def test_encode_jpeg_monochrome():
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


def test_encode_jpeg2000_rgb():
    pytest.importorskip("openjpeg")
    bits_allocated = 8
    frame = np.ones((48, 32, 3), dtype=np.dtype(f'uint{bits_allocated}'))
    frame[2:4, 5:30, 0] = 7
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEG2000,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='YBR_ICT',
        pixel_representation=0,
        planar_configuration=0
    )
    assert compressed_frame.startswith(b"\xFF\x4F\xFF\x51")
    assert compressed_frame.endswith(b'\xFF\xD9')
    decoded_frame = decode_frame(
        value=compressed_frame,
        transfer_syntax_uid=JPEG2000,
        rows=frame.shape[0],
        columns=frame.shape[1],
        samples_per_pixel=frame.shape[2],
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='YBR_ICT',
        pixel_representation=0,
        planar_configuration=0
    )
    np.testing.assert_allclose(frame, decoded_frame, atol=2)


def test_enncode_jpeg2000_monochrome():
    pytest.importorskip("openjpeg")
    bits_allocated = 8
    frame = np.zeros((48, 32), dtype=np.dtype(f'uint{bits_allocated}'))
    frame[2:4, 5:30] = 7
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEG2000,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
    )
    assert compressed_frame.startswith(b"\xFF\x4F\xFF\x51")
    assert compressed_frame.endswith(b'\xFF\xD9')
    decoded_frame = decode_frame(
        value=compressed_frame,
        transfer_syntax_uid=JPEG2000,
        rows=frame.shape[0],
        columns=frame.shape[1],
        samples_per_pixel=1,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
        planar_configuration=0
    )
    np.testing.assert_allclose(frame, decoded_frame, atol=2)


def test_encode_jpeg2000lossless_rgb():
    pytest.importorskip("openjpeg")
    bits_allocated = 8
    frame = np.ones((48, 32, 3), dtype=np.dtype(f'uint{bits_allocated}'))
    frame *= 255
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEG2000Lossless,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='YBR_RCT',
        pixel_representation=0,
        planar_configuration=0
    )
    assert compressed_frame.startswith(b"\xFF\x4F\xFF\x51")
    assert compressed_frame.endswith(b'\xFF\xD9')
    decoded_frame = decode_frame(
        value=compressed_frame,
        transfer_syntax_uid=JPEG2000Lossless,
        rows=frame.shape[0],
        columns=frame.shape[1],
        samples_per_pixel=frame.shape[2],
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='YBR_RCT',
        pixel_representation=0,
        planar_configuration=0
    )
    np.testing.assert_array_equal(frame, decoded_frame)


def test_encode_jpeg2000lossless_monochrome():
    pytest.importorskip("openjpeg")
    bits_allocated = 16
    frame = np.zeros((48, 32), dtype=np.dtype(f'uint{bits_allocated}'))
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEG2000Lossless,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
    )
    assert compressed_frame.startswith(b"\xFF\x4F\xFF\x51")
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


def test_encode_jpeg2000lossless_single_bit():
    pytest.importorskip("openjpeg")
    bits_allocated = 1
    frame = np.zeros((48, 32), dtype=np.dtype('uint8'))
    frame[12:45, 3:6] = 1
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEG2000Lossless,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
    )
    assert compressed_frame.startswith(b"\xFF\x4F\xFF\x51")
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


def test_encode_jpegls_rgb():
    pytest.importorskip("libjpeg")
    bits_allocated = 8
    frame = np.ones((16, 32, 3), dtype=np.dtype(f'uint{bits_allocated}'))
    frame *= 255
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEGLSLossless,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='RGB',
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
        photometric_interpretation='RGB',
        pixel_representation=0,
        planar_configuration=0
    )
    np.testing.assert_array_equal(frame, decoded_frame)


def test_encode_jpegls_monochrome():
    pytest.importorskip("libjpeg")
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
        transfer_syntax_uid=JPEGLSLossless,
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


def test_encode_jpeglsnearlossless_rgb():
    pytest.importorskip("libjpeg")
    bits_allocated = 8
    frame = np.ones((16, 32, 3), dtype=np.dtype(f'uint{bits_allocated}'))
    frame *= 255
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEGLSNearLossless,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='RGB',
        pixel_representation=0,
        planar_configuration=0
    )
    assert compressed_frame.startswith(b'\xFF\xD8')
    assert compressed_frame.endswith(b'\xFF\xD9')
    decoded_frame = decode_frame(
        value=compressed_frame,
        transfer_syntax_uid=JPEGLSNearLossless,
        rows=frame.shape[0],
        columns=frame.shape[1],
        samples_per_pixel=frame.shape[2],
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='RGB',
        pixel_representation=0,
        planar_configuration=0
    )
    np.testing.assert_allclose(frame, decoded_frame)


def test_encode_jpeglsnearlossless_monochrome():
    pytest.importorskip("libjpeg")
    bits_allocated = 16
    frame = np.zeros((16, 32), dtype=np.dtype(f'uint{bits_allocated}'))
    compressed_frame = encode_frame(
        frame,
        transfer_syntax_uid=JPEGLSNearLossless,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
    )
    assert compressed_frame.startswith(b'\xFF\xD8')
    assert compressed_frame.endswith(b'\xFF\xD9')
    decoded_frame = decode_frame(
        value=compressed_frame,
        transfer_syntax_uid=JPEGLSNearLossless,
        rows=frame.shape[0],
        columns=frame.shape[1],
        samples_per_pixel=1,
        bits_allocated=bits_allocated,
        bits_stored=bits_allocated,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=0,
        planar_configuration=0
    )
    np.testing.assert_allclose(frame, decoded_frame)


def test_encode_jpeg_rgb_wrong_photometric_interpretation():
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


def test_encode_jpeg_rgb_wrong_planar_configuration():
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


def test_encode_jpeg2000_monochrome_wrong_photometric_interpretation():
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


def test_encode_jpeg2000_monochrome_wrong_pixel_representation():
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


@pytest.mark.parametrize(
    'transfer_syntax_uid,bits',
    [
        (JPEG2000Lossless, 16),
        (JPEG2000Lossless, 8),
        (JPEGLSLossless, 16),
        (JPEGLSLossless, 8),
        (JPEGLSNearLossless, 16),
    ]
)
def test_encode_signed_monochrome_roundtrip(
    transfer_syntax_uid,
    bits,
):
    if transfer_syntax_uid == JPEG2000Lossless:
        pytest.importorskip("openjpeg")
    else:
        pytest.importorskip("libjpeg")

    rng = np.random.default_rng(0)
    low = -(2 ** (bits - 1))
    high = 2 ** (bits - 1)
    frame = rng.integers(
        low, high, (64, 64), dtype=f'int{bits}'
    )
    # ensure the extremes are exercised
    frame[0, 0] = low
    frame[0, 1] = high - 1

    encoded = encode_frame(
        frame,
        transfer_syntax_uid=transfer_syntax_uid,
        bits_allocated=bits,
        bits_stored=bits,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=1,
    )

    decoded = decode_frame(
        encoded,
        transfer_syntax_uid=transfer_syntax_uid,
        rows=64,
        columns=64,
        samples_per_pixel=1,
        bits_allocated=bits,
        bits_stored=bits,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=1,
    )
    assert decoded.dtype == frame.dtype
    np.testing.assert_array_equal(decoded, frame)


def test_encode_jpeg2000_signed_lossy_monochrome():
    pytest.importorskip("openjpeg")
    frame = np.random.default_rng(0).integers(
        -1000, 1000, (64, 64), dtype=np.int16
    )
    encoded = encode_frame(
        frame,
        transfer_syntax_uid=JPEG2000,
        bits_allocated=16,
        bits_stored=16,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=1,
    )
    decoded = decode_frame(
        encoded,
        transfer_syntax_uid=JPEG2000,
        rows=64,
        columns=64,
        samples_per_pixel=1,
        bits_allocated=16,
        bits_stored=16,
        photometric_interpretation='MONOCHROME2',
        pixel_representation=1,
    )
    assert decoded.dtype == np.int16
    np.testing.assert_allclose(decoded, frame, atol=5)


@pytest.mark.parametrize(
    'transfer_syntax_uid',
    [JPEG2000, JPEG2000Lossless, JPEGLSLossless, JPEGLSNearLossless]
)
def test_encode_signed_palette_color_rejected(transfer_syntax_uid):
    if transfer_syntax_uid in (JPEG2000, JPEG2000Lossless):
        pytest.importorskip("openjpeg")
    else:
        pytest.importorskip("libjpeg")

    frame = np.zeros((64, 64), dtype=np.int16)
    with pytest.raises(ValueError):
        encode_frame(
            frame,
            transfer_syntax_uid=transfer_syntax_uid,
            bits_allocated=16,
            bits_stored=16,
            photometric_interpretation='PALETTE COLOR',
            pixel_representation=1,
        )


@pytest.mark.parametrize(
    'transfer_syntax_uid,photometric_interpretation',
    [
        (JPEG2000, 'YBR_ICT'),
        (JPEG2000Lossless, 'YBR_RCT'),
        (JPEGLSLossless, 'RGB'),
        (JPEGLSNearLossless, 'RGB'),
    ]
)
def test_encode_signed_color_rejected(
    transfer_syntax_uid,
    photometric_interpretation,
):
    if transfer_syntax_uid in (JPEG2000, JPEG2000Lossless):
        pytest.importorskip("openjpeg")
    else:
        pytest.importorskip("libjpeg")

    frame = np.zeros((64, 64, 3), dtype=np.int8)
    with pytest.raises(ValueError):
        encode_frame(
            frame,
            transfer_syntax_uid=transfer_syntax_uid,
            bits_allocated=8,
            bits_stored=8,
            photometric_interpretation=photometric_interpretation,
            pixel_representation=1,
            planar_configuration=0,
        )


@pytest.mark.parametrize(
    'transfer_syntax_uid',
    [JPEG2000, JPEGLSNearLossless, JPEGBaseline8Bit]
)
def test_palette_color_rejected_for_lossy(transfer_syntax_uid):
    if transfer_syntax_uid == JPEG2000:
        pytest.importorskip("openjpeg")
    elif transfer_syntax_uid == JPEGLSNearLossless:
        pytest.importorskip("libjpeg")

    frame = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_frame(
            frame,
            transfer_syntax_uid=transfer_syntax_uid,
            bits_allocated=8,
            bits_stored=8,
            photometric_interpretation='PALETTE COLOR',
            pixel_representation=0,
        )


@pytest.mark.parametrize(
    'transfer_syntax_uid',
    [JPEG2000Lossless, JPEGLSLossless, RLELossless]
)
def test_palette_color_bits_allocated(transfer_syntax_uid):
    if transfer_syntax_uid == JPEG2000Lossless:
        pytest.importorskip("openjpeg")
    elif transfer_syntax_uid == JPEGLSLossless:
        pytest.importorskip("libjpeg")

    # 8 and 16 bit are valid
    for bits, dtype in [(8, np.uint8), (16, np.uint16)]:
        frame = np.zeros((64, 64), dtype=dtype)
        encode_frame(
            frame,
            transfer_syntax_uid=transfer_syntax_uid,
            bits_allocated=bits,
            bits_stored=bits,
            photometric_interpretation='PALETTE COLOR',
            pixel_representation=0,
        )
    # 1 bit is not
    with pytest.raises(ValueError):
        encode_frame(
            np.zeros((64, 64), dtype=np.uint8),
            transfer_syntax_uid=transfer_syntax_uid,
            bits_allocated=1,
            bits_stored=1,
            photometric_interpretation='PALETTE COLOR',
            pixel_representation=0,
        )


@pytest.mark.parametrize(
    'transfer_syntax_uid',
    [JPEGLSLossless, JPEGLSNearLossless]
)
def test_jpegls_bits_stored_too_small(transfer_syntax_uid):
    pytest.importorskip("libjpeg")

    frame = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_frame(
            frame,
            transfer_syntax_uid=transfer_syntax_uid,
            bits_allocated=8,
            bits_stored=1,
            photometric_interpretation='MONOCHROME2',
            pixel_representation=0,
        )


@pytest.mark.parametrize('planar_configuration', [0, 1])
def test_rle_color_roundtrip(planar_configuration):
    photometric_interpretation = 'RGB'
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)
    encoded = encode_frame(
        frame,
        transfer_syntax_uid=RLELossless,
        bits_allocated=8,
        bits_stored=8,
        photometric_interpretation=photometric_interpretation,
        pixel_representation=0,
        planar_configuration=planar_configuration,
    )
    decoded = decode_frame(
        encoded,
        transfer_syntax_uid=RLELossless,
        rows=20,
        columns=30,
        samples_per_pixel=3,
        bits_allocated=8,
        bits_stored=8,
        photometric_interpretation=photometric_interpretation,
        pixel_representation=0,
        planar_configuration=planar_configuration,
    )
    np.testing.assert_array_equal(decoded, frame)


@pytest.mark.parametrize(
    'bits,dtype', [(8, np.int8), (16, np.int16)]
)
@pytest.mark.parametrize(
    'photometric_interpretation',
    ['MONOCHROME1', 'MONOCHROME2']
)
def test_rle_signed_monochrome_roundtrip(
    bits, dtype, photometric_interpretation
):
    rng = np.random.default_rng(0)
    frame = rng.integers(-100, 100, (20, 30)).astype(dtype)
    encoded = encode_frame(
        frame,
        transfer_syntax_uid=RLELossless,
        bits_allocated=bits,
        bits_stored=bits,
        photometric_interpretation=photometric_interpretation,
        pixel_representation=1,
    )
    decoded = decode_frame(
        encoded,
        transfer_syntax_uid=RLELossless,
        rows=20,
        columns=30,
        samples_per_pixel=1,
        bits_allocated=bits,
        bits_stored=bits,
        photometric_interpretation=photometric_interpretation,
        pixel_representation=1,
    )
    np.testing.assert_array_equal(decoded, frame)


def test_rle_ybr_full_encode():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for planar_configuration in (0, 1):
        encode_frame(
            frame,
            transfer_syntax_uid=RLELossless,
            bits_allocated=8,
            bits_stored=8,
            photometric_interpretation='YBR_FULL',
            pixel_representation=0,
            planar_configuration=planar_configuration,
        )


@pytest.mark.parametrize(
    'dtype,shape,photometric_interpretation,pixel_representation,planar',
    [
        # monochrome with planar configuration
        (np.uint8, (16, 16), 'MONOCHROME2', 0, 0),
        # wrong photometric interpretation (monochrome)
        (np.uint8, (16, 16), 'RGB', 0, None),
        # signed palette color
        (np.int8, (16, 16), 'PALETTE COLOR', 1, None),
        # color with disallowed photometric interpretation
        (np.uint8, (16, 16, 3), 'YBR_ICT', 0, 0),
        (np.uint8, (16, 16, 3), 'MONOCHROME2', 0, 0),
        # signed color
        (np.int8, (16, 16, 3), 'RGB', 1, 0),
    ]
)
def test_rle_invalid_combinations(
    dtype,
    shape,
    photometric_interpretation,
    pixel_representation,
    planar,
):
    array = np.zeros(shape, dtype=dtype)
    bits = array.dtype.itemsize * 8
    with pytest.raises(ValueError):
        encode_frame(
            array,
            transfer_syntax_uid=RLELossless,
            bits_allocated=bits,
            bits_stored=bits,
            photometric_interpretation=photometric_interpretation,
            pixel_representation=pixel_representation,
            planar_configuration=planar,
        )
