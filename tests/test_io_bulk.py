from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from highdicom.io import ImageFileReader
from highdicom import Image


def _write_native_dataset(
    filename: Path,
    pixel_array: np.ndarray,
    photometric: str = 'MONOCHROME2',
    icc_profile: bytes | None = None,
) -> None:
    if pixel_array.ndim == 2:
        frames = pixel_array[np.newaxis, ...]
    else:
        frames = pixel_array
    samples_per_pixel = 1 if frames.ndim == 3 else frames.shape[-1]
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(
        filename,
        {},
        file_meta=meta,
        preamble=b'\x00' * 128,
    )
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = 'OT'
    dataset.Rows = frames.shape[1]
    dataset.Columns = frames.shape[2]
    dataset.SamplesPerPixel = samples_per_pixel
    dataset.PhotometricInterpretation = photometric
    if samples_per_pixel > 1:
        dataset.PlanarConfiguration = 0
    dataset.BitsAllocated = frames.dtype.itemsize * 8
    dataset.BitsStored = dataset.BitsAllocated
    dataset.HighBit = dataset.BitsStored - 1
    dataset.PixelRepresentation = int(frames.dtype.kind == 'i')
    if frames.shape[0] > 1:
        dataset.NumberOfFrames = frames.shape[0]
    if icc_profile is not None:
        dataset.ICCProfile = icc_profile
    dataset.PixelData = frames.tobytes()
    dataset.save_as(filename, write_like_original=False)


def test_read_frames_preserves_order_and_matches_read_frame():
    pixels = np.stack([
        np.full((4, 5), 11, dtype=np.uint8),
        np.full((4, 5), 22, dtype=np.uint8),
        np.full((4, 5), 33, dtype=np.uint8),
    ])
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'multi.dcm'
        _write_native_dataset(filename, pixels)
        with ImageFileReader(filename) as reader:
            bulk = reader.read_frames(indices=[2, 0, 2])
            assert len(bulk) == 3
            np.testing.assert_array_equal(bulk[0], pixels[2])
            np.testing.assert_array_equal(bulk[1], pixels[0])
            np.testing.assert_array_equal(bulk[2], pixels[2])
            np.testing.assert_array_equal(
                bulk[1], reader.read_frame(0)
            )


def test_read_frames_single_frame_returns_one_item():
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'single.dcm'
        _write_native_dataset(filename, pixels)
        with ImageFileReader(filename) as reader:
            bulk = reader.read_frames()
            assert len(bulk) == 1
            np.testing.assert_array_equal(bulk[0], pixels)
            np.testing.assert_array_equal(
                bulk[0], reader.read_frame(0)
            )


def test_read_frames_matches_bulk_color_data_and_icc_path():
    pixels = np.array([
        [[[0, 32, 64], [96, 128, 160]],
         [[192, 224, 255], [16, 48, 80]]],
        [[[8, 40, 72], [104, 136, 168]],
         [[200, 232, 248], [24, 56, 88]]],
    ], dtype=np.uint8)
    # A compact sRGB ICC profile created by Pillow is sufficient to exercise
    # highdicom's colour-correction path without a patient-data fixture.
    from PIL import ImageCms
    profile = ImageCms.ImageCmsProfile(
        ImageCms.createProfile('sRGB')
    ).tobytes()
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'color.dcm'
        _write_native_dataset(filename, pixels, 'RGB', profile)
        with ImageFileReader(filename) as reader:
            raw = reader.read_frames(correct_color=False)
            corrected = reader.read_frames(correct_color=True)
            assert len(raw) == len(corrected) == 2
            for index in range(2):
                np.testing.assert_array_equal(
                    raw[index], reader.read_frame(index, correct_color=False)
                )
                np.testing.assert_array_equal(
                    corrected[index], reader.read_frame(index)
                )


def test_read_frames_rejects_invalid_indices_before_decoding():
    pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'invalid.dcm'
        _write_native_dataset(filename, pixels)
        with ImageFileReader(filename) as reader:
            with pytest.raises(ValueError, match='Frame index'):
                reader.read_frames(indices=[-1])
            with pytest.raises(ValueError, match='Frame index'):
                reader.read_frames(indices=[reader.number_of_frames])


def test_read_frames_uses_only_requested_indices(monkeypatch):
    pixels = np.stack([
        np.full((3, 4), 10 + index, dtype=np.uint8)
        for index in range(6)
    ])
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'selective.dcm'
        _write_native_dataset(filename, pixels)
        import highdicom.io

        original_iter_pixels = highdicom.io.iter_pixels
        calls = []

        def spy_iter_pixels(source, **kwargs):
            calls.append(kwargs['indices'])
            yield from original_iter_pixels(source, **kwargs)

        monkeypatch.setattr(highdicom.io, 'iter_pixels', spy_iter_pixels)
        with ImageFileReader(filename) as reader:
            selected = reader.read_frames(indices=[4, 1, 4])

        assert calls == [[4, 1, 4]]
        np.testing.assert_array_equal(selected[0], pixels[4])
        np.testing.assert_array_equal(selected[1], pixels[1])
        np.testing.assert_array_equal(selected[2], pixels[4])


def test_read_raw_frames_matches_single_frame_contract():
    pixels = np.stack([
        np.full((3, 4), 10 + index, dtype=np.uint8)
        for index in range(3)
    ])
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'raw.dcm'
        _write_native_dataset(filename, pixels)
        with ImageFileReader(filename) as reader:
            raw = reader.read_raw_frames(indices=[2, 0, 2])
            assert raw == [
                reader.read_frame_raw(2),
                reader.read_frame_raw(0),
                reader.read_frame_raw(2),
            ]
            assert reader.read_raw_frames(indices=[]) == []


def test_lazy_image_methods_use_selective_reader_path():
    pixels = np.stack([
        np.full((3, 4), 10 + index, dtype=np.uint8)
        for index in range(4)
    ])
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'image.dcm'
        _write_native_dataset(filename, pixels)
        image = Image.from_file(filename, lazy_frame_retrieval=True)

        stored = image.get_stored_frames([4, 2], as_indices=False)
        displayed = image.get_frames([4, 2], as_indices=False, dtype=np.uint8)
        np.testing.assert_array_equal(stored, pixels[[3, 1]])
        np.testing.assert_array_equal(displayed, pixels[[3, 1]])

        assembled = image._get_pixels_by_frame(
            2,
            iter([
                (3, (slice(None), slice(None)),
                 (0, slice(None), slice(None)), ()),
                (1, (slice(None), slice(None)),
                 (1, slice(None), slice(None)), ()),
            ]),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(assembled, pixels[[3, 1]])


def test_read_frames_color_correction_requires_icc_profile():
    pixels = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    with TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / 'no_icc.dcm'
        _write_native_dataset(filename, pixels, 'RGB')
        with ImageFileReader(filename) as reader:
            with pytest.raises(ValueError, match='ICC Profile'):
                reader.read_frames()
