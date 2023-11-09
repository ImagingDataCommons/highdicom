import unittest
from pathlib import Path
from random import shuffle

import numpy as np
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.filebase import DicomBytesIO, DicomFileLike

from highdicom.io import ImageFileReader


class TestImageFileReader(unittest.TestCase):
    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        self._test_dir = file_path.parent.parent.joinpath(
            'data',
            'test_files'
        )

    def test_read_single_frame_ct_image_native(self):
        filename = str(self._test_dir.joinpath('ct_image.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.filename == filename
            assert reader.number_of_frames == 1
            frame = reader.read_frame(0)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 2
            assert frame.dtype == np.int16
            assert frame.shape == (
                reader.metadata.Rows,
                reader.metadata.Columns,
            )
            np.testing.assert_array_equal(frame, pixel_array)

    def test_read_multi_frame_ct_image_native(self):
        filename = str(get_testdata_file('eCT_Supplemental.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 2
            frame_index = 0
            frame = reader.read_frame(frame_index)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 2
            assert frame.dtype == np.uint16
            assert frame.shape == (
                reader.metadata.Rows,
                reader.metadata.Columns,
            )
            np.testing.assert_array_equal(frame, pixel_array[frame_index])

    def test_read_multi_frame_sm_image_native(self):
        filename = str(self._test_dir.joinpath('sm_image.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 25
            indices = list(range(reader.number_of_frames))
            shuffle(indices)
            for i in indices:
                frame = reader.read_frame(i, correct_color=False)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    reader.metadata.SamplesPerPixel,
                )
                np.testing.assert_array_equal(frame, pixel_array[i, ...])

    def test_read_multi_frame_sm_image_numbers_native(self):
        filename = str(self._test_dir.joinpath('sm_image_numbers.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 25
            indices = list(range(reader.number_of_frames))
            shuffle(indices)
            for i in indices:
                frame = reader.read_frame(i, correct_color=False)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    reader.metadata.SamplesPerPixel,
                )
                np.testing.assert_array_equal(frame, pixel_array[i, ...])

    def test_read_multi_frame_seg_image_sm_numbers_bitpacked(self):
        filename = str(self._test_dir.joinpath('seg_image_sm_numbers.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 20
            assert len(reader.metadata.SegmentSequence) == 20
            indices = list(range(reader.number_of_frames))
            shuffle(indices)
            for i in indices:
                frame = reader.read_frame(i)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 2
                assert frame.dtype == np.uint8
                assert frame.max() == 1
                assert len(np.unique(frame)) == 2
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                )
                np.testing.assert_array_equal(frame, pixel_array[i, ...])

    def test_read_multi_frame_sm_image_dots_native(self):
        filename = str(self._test_dir.joinpath('sm_image_dots.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 25
            indices = list(range(reader.number_of_frames))
            shuffle(indices)
            for i in indices:
                frame = reader.read_frame(i, correct_color=False)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    reader.metadata.SamplesPerPixel,
                )
                np.testing.assert_array_equal(frame, pixel_array[i, ...])

    def test_read_multi_frame_seg_image_sm_dots_bitpacked(self):
        filename = str(self._test_dir.joinpath('seg_image_sm_dots.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 62
            assert len(reader.metadata.SegmentSequence) == 50
            indices = list(range(reader.number_of_frames))
            shuffle(indices)
            for i in indices:
                frame = reader.read_frame(i)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 2
                assert frame.dtype == np.uint8
                assert frame.max() == 1
                assert len(np.unique(frame)) == 2
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                )
                np.testing.assert_array_equal(frame, pixel_array[i, ...])

    def test_read_ybr_422_native(self):
        # Reading a frame using YBR_422 photometric interpretation and no
        # compression
        filename = str(get_testdata_file('SC_ybr_full_422_uncompressed.dcm'))
        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 1
            frame = reader.read_frame(0, correct_color=False)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.dtype == np.uint8
            assert frame.shape == (
                reader.metadata.Rows,
                reader.metadata.Columns,
                reader.metadata.SamplesPerPixel,
            )
            np.testing.assert_array_equal(frame, pixel_array)

    def test_read_single_frame_ct_image_dicom_bytes_io(self):
        filename = str(self._test_dir.joinpath("ct_image.dcm"))
        dcm = DicomBytesIO(open(filename, "rb").read())

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(dcm) as reader:
            assert reader.number_of_frames == 1
            frame = reader.read_frame(0)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 2
            assert frame.dtype == np.int16
            assert frame.shape == (
                reader.metadata.Rows,
                reader.metadata.Columns,
            )
            np.testing.assert_array_equal(frame, pixel_array)

    def test_read_single_frame_ct_image_dicom_file_like_opened(self):
        # Test reading frames from an opened DicomFileLike file
        filename = self._test_dir.joinpath("ct_image.dcm")
        dcm = DicomFileLike(filename.open("rb"))

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(dcm) as reader:
            assert reader.number_of_frames == 1
            frame = reader.read_frame(0)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 2
            assert frame.dtype == np.int16
            assert frame.shape == (
                reader.metadata.Rows,
                reader.metadata.Columns,
            )
            np.testing.assert_array_equal(frame, pixel_array)
