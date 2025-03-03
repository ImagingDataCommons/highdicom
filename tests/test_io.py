import unittest
from pathlib import Path
from random import shuffle
from tempfile import TemporaryDirectory

import numpy as np
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.filebase import DicomBytesIO, DicomFileLike, DicomFile
import pytest

from highdicom.io import ImageFileReader
from tests.utils import find_readable_images


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

    def test_read_multi_frame_sm_image_jpegls(self):
        filename = str(self._test_dir.joinpath('sm_image_jpegls.dcm'))
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

    def test_read_single_frame_sm_image_jpegls_dicomfile(self):
        filename = str(self._test_dir.joinpath("sm_image_jpegls.dcm"))
        dcm = DicomFile(filename, "rb")

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(dcm) as reader:
            assert reader.number_of_frames == 25
            for fno in range(reader.number_of_frames):
                frame = reader.read_frame(fno)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    3,
                )
                np.testing.assert_array_equal(frame, pixel_array[fno])

    def test_read_single_frame_sm_image_jpegls_dicom_bytes_io(self):
        filename = str(self._test_dir.joinpath("sm_image_jpegls.dcm"))
        dcm = DicomBytesIO(open(filename, "rb").read())

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(dcm) as reader:
            assert reader.number_of_frames == 25
            for fno in range(reader.number_of_frames):
                frame = reader.read_frame(fno)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    3,
                )
                np.testing.assert_array_equal(frame, pixel_array[fno])

    def test_read_single_frame_sm_image_jpegls_nobot_dicom_bytes_io(self):
        filename = str(self._test_dir.joinpath("sm_image_jpegls_nobot.dcm"))
        dcm = DicomBytesIO(open(filename, "rb").read())

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(dcm) as reader:
            assert reader.number_of_frames == 25
            for fno in range(reader.number_of_frames):
                frame = reader.read_frame(fno)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    3,
                )
                np.testing.assert_array_equal(frame, pixel_array[fno])

    def test_read_single_frame_sm_image_jpegls_dicom_file_like_opened(self):
        # Test reading frames from an opened DicomFileLike file
        filename = self._test_dir.joinpath("sm_image_jpegls.dcm")
        dcm = DicomFileLike(filename.open("rb"))

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(dcm) as reader:
            assert reader.number_of_frames == 25
            for fno in range(reader.number_of_frames):
                frame = reader.read_frame(fno)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    3,
                )
                np.testing.assert_array_equal(frame, pixel_array[fno])

    def test_read_single_frame_sm_image_jpegls_nobot_dicom_file_like_opened(
        self
    ):
        # Test reading frames from an opened DicomFileLike file
        filename = self._test_dir.joinpath("sm_image_jpegls_nobot.dcm")
        dcm = DicomFileLike(filename.open("rb"))

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(dcm) as reader:
            assert reader.number_of_frames == 25
            for fno in range(reader.number_of_frames):
                frame = reader.read_frame(fno)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 3
                assert frame.dtype == np.uint8
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                    3,
                )
                np.testing.assert_array_equal(frame, pixel_array[fno])

    def test_read_rle_no_bot(self):
        # This image is RLE compressed but has no BOT, requiring searching
        # through the pixel data for delimiter tags
        filename = Path(get_testdata_file('rtdose_rle.dcm'))

        dataset = dcmread(filename)
        pixel_array = dataset.pixel_array
        with ImageFileReader(filename) as reader:
            assert reader.number_of_frames == 15
            for f in range(reader.number_of_frames):
                frame = reader.read_frame(f, correct_color=False)
                assert isinstance(frame, np.ndarray)
                assert frame.ndim == 2
                assert frame.dtype == np.uint32
                assert frame.shape == (
                    reader.metadata.Rows,
                    reader.metadata.Columns,
                )
                np.testing.assert_array_equal(frame, pixel_array[f])

    def test_disallow_deflated_dataset(self):
        # Files with a deflated transfer
        msg = (
            'Deflated transfer syntaxes cannot be used with the '
            'ImageFileReader.'
        )
        filename = get_testdata_file('image_dfl.dcm')

        with pytest.raises(ValueError, match=msg):
            with ImageFileReader(filename) as reader:
                reader.read_frame(1)

    def test_extended_offsets(self):
        # Surprisingly, there are no pydicom test files with extended offsets
        # Instead, we start with an image with no basic offsets, and mock one
        # up
        filename = get_testdata_file('rtdose_rle.dcm')

        # First use the image file reader to infer the offsets
        with ImageFileReader(filename) as reader:
            reader.read_frame(1)
            offsets = reader._offset_table

        # Add the extended offset table to the dataset
        dataset = dcmread(filename)
        dataset_with_eot = dcmread(filename)
        dataset_with_eot.ExtendedOffsetTable = np.array(
            offsets,
            np.uint64
        ).tobytes()

        with TemporaryDirectory() as d:
            new_filename = d + '/test.dcm'
            dataset_with_eot.save_as(new_filename)

            with ImageFileReader(new_filename) as reader:
                for i in range(dataset.NumberOfFrames):
                    frame = reader.read_frame(i)
                    assert np.array_equal(frame, dataset.pixel_array[i])


@pytest.mark.parametrize(
    'filename,dependency',
    find_readable_images(),
)
def test_all_images(filename, dependency):
    if dependency is not None:
        pytest.importorskip(dependency)

    dataset = dcmread(filename)
    pixel_array = dataset.pixel_array

    is_color = dataset.SamplesPerPixel == 3
    number_of_frames = dataset.get('NumberOfFrames', 1)
    is_multiframe = number_of_frames > 1

    if is_color:
        ndim = 3
        shape = (
            dataset.Rows,
            dataset.Columns,
            3
        )
    else:
        ndim = 2
        shape = (
            dataset.Rows,
            dataset.Columns,
        )

    with ImageFileReader(filename) as reader:
        assert reader.number_of_frames == number_of_frames
        for f in range(reader.number_of_frames):
            frame = reader.read_frame(f, correct_color=False)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == ndim
            assert frame.dtype == pixel_array.dtype
            assert frame.shape == shape
            expected_frame = (
                pixel_array[f] if is_multiframe else pixel_array
            )
            np.testing.assert_array_equal(frame, expected_frame)
