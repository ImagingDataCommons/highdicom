"""Input/Output of datasets based on DICOM Part10 files."""
import logging
import sys
import traceback
from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset
from pydicom.encaps import get_frame_offsets
from pydicom.filebase import DicomFile, DicomFileLike
from pydicom.filereader import (
    data_element_offset_to_value,
    dcmread,
    read_file_meta_info
)
from pydicom.pixel_data_handlers.numpy_handler import unpack_bits
from pydicom.tag import TupleTag, ItemTag, SequenceDelimiterTag
from pydicom.uid import UID

from highdicom.frame import decode_frame
from highdicom.color import ColorManager

logger = logging.getLogger(__name__)


_FLOAT_PIXEL_DATA_TAGS = {0x7FE00008, 0x7FE00009, }
_UINT_PIXEL_DATA_TAGS = {0x7FE00010, }
_PIXEL_DATA_TAGS = _FLOAT_PIXEL_DATA_TAGS.union(_UINT_PIXEL_DATA_TAGS)

_JPEG_SOI_MARKER = b'\xFF\xD8'  # also JPEG-LS
_JPEG_EOI_MARKER = b'\xFF\xD9'  # also JPEG-LS
_JPEG2000_SOC_MARKER = b'\xFF\x4F'
_JPEG2000_EOC_MARKER = b'\xFF\xD9'
_START_MARKERS = {_JPEG_SOI_MARKER, _JPEG2000_SOC_MARKER}
_END_MARKERS = {_JPEG_EOI_MARKER, _JPEG2000_EOC_MARKER}


def _get_bot(fp: DicomFileLike, number_of_frames: int) -> List[int]:
    """Tries to read the value of the Basic Offset Table (BOT) item and builds
    it in case it is empty.

    Parameters
    ----------
    fp: pydicom.filebase.DicomFileLike
        Pointer for DICOM PS3.10 file stream positioned at the first byte of
        the Pixel Data element
    number_of_frames: int
        Number of frames contained in the Pixel Data element

    Returns
    -------
    List[int]
        Offset of each Frame item in bytes from the first byte of the Pixel Data
        element following the BOT item

    Note
    ----
    Moves the pointer to the first byte of the open file following the BOT item
    (the first byte of the first Frame item).

    """
    logger.debug('read Basic Offset Table')
    basic_offset_table = _read_bot(fp)

    first_frame_offset = fp.tell()
    tag = TupleTag(fp.read_tag())
    if int(tag) != ItemTag:
        raise ValueError(
            'Reading of Basic Offset Table failed. '
            f'Encountered unexpected Tag "{tag}".'
        )
    fp.seek(first_frame_offset, 0)

    # Basic Offset Table item must be present, but it may be empty
    if len(basic_offset_table) == 0:
        logger.debug('Basic Offset Table item is empty')
    if len(basic_offset_table) != number_of_frames:
        logger.debug('build Basic Offset Table item')
        basic_offset_table = _build_bot(fp, number_of_frames)

    return basic_offset_table


def _read_bot(fp: DicomFileLike) -> List[int]:
    """Read Basic Offset Table (BOT) item of encapsulated Pixel Data element.

    Parameters
    ----------
    fp: pydicom.filebase.DicomFileLike
        Pointer for DICOM PS3.10 file stream positioned at the first byte of
        the Pixel Data element

    Returns
    -------
    List[int]
        Offset of each Frame item in bytes from the first byte of the Pixel Data
        element following the BOT item

    Note
    ----
    Moves the pointer to the first byte of the open file following the BOT item
    (the first byte of the first Frame item).

    Raises
    ------
    IOError
        When file pointer is not positioned at first byte of Pixel Data element

    """
    tag = TupleTag(fp.read_tag())
    if int(tag) not in _PIXEL_DATA_TAGS:
        raise IOError(
            'Expected file pointer at first byte of Pixel Data element.'
        )
    # Skip Pixel Data element header (tag, VR, length)
    pixel_data_element_value_offset = data_element_offset_to_value(
        fp.is_implicit_VR, 'OB'
    )
    fp.seek(pixel_data_element_value_offset - 4, 1)
    is_empty, offsets = get_frame_offsets(fp)
    return offsets


def _build_bot(fp: DicomFileLike, number_of_frames: int) -> List[int]:
    """Build Basic Offset Table (BOT) item of encapsulated Pixel Data element.

    Parameters
    ----------
    fp: pydicom.filebase.DicomFileLike
        Pointer for DICOM PS3.10 file stream positioned at the first byte of
        the Pixel Data element following the empty Basic Offset Table (BOT)
    number_of_frames: int
        Total number of frames in the dataset

    Returns
    -------
    List[int]
        Offset of each Frame item in bytes from the first byte of the Pixel Data
        element following the BOT item

    Note
    ----
    Moves the pointer back to the first byte of the Pixel Data element
    following the BOT item (the first byte of the first Frame item).

    Raises
    ------
    IOError
        When file pointer is not positioned at first byte of first Frame item
        after Basic Offset Table item or when parsing of Frame item headers
        fails
    ValueError
        When the number of offsets doesn't match the specified number of frames

    """
    initial_position = fp.tell()
    offset_values = []
    current_offset = 0
    i = 0
    while True:
        frame_position = fp.tell()
        tag = TupleTag(fp.read_tag())
        if int(tag) == SequenceDelimiterTag:
            break
        if int(tag) != ItemTag:
            fp.seek(initial_position, 0)
            raise IOError(
                'Building Basic Offset Table (BOT) failed. '
                f'Expected tag of Frame item #{i} at position {frame_position}.'
            )
        length = fp.read_UL()
        if length % 2:
            fp.seek(initial_position, 0)
            raise IOError(
                'Building Basic Offset Table (BOT) failed. '
                f'Length of Frame item #{i} is not a multiple of 2.'
            )
        elif length == 0:
            fp.seek(initial_position, 0)
            raise IOError(
                'Building Basic Offset Table (BOT) failed. '
                f'Length of Frame item #{i} is zero.'
            )

        first_two_bytes = fp.read(2, True)
        if not fp.is_little_endian:
            first_two_bytes = first_two_bytes[::-1]
        # In case of fragmentation, we only want to get the offsets to the
        # first fragment of a given frame. We can identify those based on the
        # JPEG and JPEG 2000 markers that should be found at the beginning and
        # end of the compressed byte stream.
        if first_two_bytes in _START_MARKERS:
            current_offset = frame_position - initial_position
            offset_values.append(current_offset)

        i += 1
        fp.seek(length - 2, 1)  # minus the first two bytes

    if len(offset_values) != number_of_frames:
        raise ValueError(
            'Number of frame items does not match specified Number of Frames.'
        )
    else:
        basic_offset_table = offset_values

    fp.seek(initial_position, 0)
    return basic_offset_table


class ImageFileReader(object):

    """Reader for DICOM datasets representing Image Information Entities.

    It provides efficient access to individual Frame items contained in the
    Pixel Data element without loading the entire element into memory.

    Examples
    --------
    >>> from pydicom.data import get_testdata_file
    >>> from highdicom.io import ImageFileReader
    >>> test_filepath = get_testdata_file('eCT_Supplemental.dcm')
    >>>
    >>> with ImageFileReader(test_filepath) as image:
    ...     print(image.metadata.SOPInstanceUID)
    ...     for i in range(image.number_of_frames):
    ...         frame = image.read_frame(i)
    ...         print(frame.shape)
    1.3.6.1.4.1.5962.1.1.10.3.1.1166562673.14401
    (512, 512)
    (512, 512)

    """

    def __init__(self, filename: Union[str, Path, DicomFileLike]):
        """
        Parameters
        ----------
        filename: Union[str, pathlib.Path, pydicom.filebase.DicomfileLike]
            DICOM Part10 file containing a dataset of an image SOP Instance

        """
        if isinstance(filename, DicomFileLike):
            fp = filename
            is_little_endian, is_implicit_VR = self._check_file_format(fp)
            try:
                if fp.is_little_endian != is_little_endian:
                    raise ValueError(
                        'Transfer syntax of file object has incorrect value '
                        'for attribute "is_little_endian".'
                    )
            except AttributeError:
                raise AttributeError(
                    'Transfer syntax of file object does not have '
                    'attribute "is_little_endian".'
                )
            try:
                if fp.is_implicit_VR != is_implicit_VR:
                    raise ValueError(
                        'Transfer syntax of file object has incorrect value '
                        'for attribute "is_implicit_VR".'
                    )
            except AttributeError:
                raise AttributeError(
                    'Transfer syntax of file object does not have '
                    'attribute "is_implicit_VR".'
                )
            self._fp = fp
            self._filename = Path(fp.name)
        elif isinstance(filename, (str, Path)):
            self._filename = Path(filename)
            self._fp = None
        else:
            raise TypeError(
                'Argument "filename" must either an open DICOM file object or '
                'the path to a DICOM file stored on disk.'
            )
        self._metadata = None

    @property
    def filename(self) -> str:
        """str: Path to the image file"""
        return str(self._filename)

    def __enter__(self) -> 'ImageFileReader':
        self.open()
        return self

    def __exit__(self, except_type, except_value, except_trace) -> None:
        self._fp.close()
        if except_value:
            sys.stderr.write(
                'Error while accessing file "{}":\n{}'.format(
                    self._filename, str(except_value)
                )
            )
            for tb in traceback.format_tb(except_trace):
                sys.stderr.write(tb)
            raise

    def open(self) -> None:
        """Open file for reading.

        Raises
        ------
        FileNotFoundError
            When file cannot be found
        OSError
            When file cannot be opened
        IOError
            When DICOM metadata cannot be read from file
        ValueError
            When DICOM dataset contained in file does not represent an image

        Note
        ----
        Builds a Basic Offset Table to speed up subsequent frame-level access.

        """
        logger.debug('read File Meta Information')
        if self._fp is None:
            try:
                self._fp = DicomFile(str(self._filename), mode='rb')
            except FileNotFoundError:
                raise FileNotFoundError(f'File not found: "{self._filename}"')
            except Exception:
                raise OSError(
                    f'Could not open file for reading: "{self._filename}"'
                )
        is_little_endian, is_implicit_VR = self._check_file_format(self._fp)
        self._fp.is_little_endian = is_little_endian
        self._fp.is_implicit_VR = is_implicit_VR

    def _check_file_format(self, fp) -> Tuple[bool, bool]:
        """Check whether file object represents a DICOM Part 10 file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            DICOM file object

        Returns
        -------
        is_little_endian: bool
            Whether the data set is encoded in little endian transfer syntax
        is_implicit_VR: bool
            Whether value representations of data elements in the data set
            are implicit

        Raises
        ------
        InvalidDicomError
            If the file object does not represent a DICOM Part 10 file

        """
        file_meta = read_file_meta_info(str(self._filename))
        transfer_syntax_uid = UID(file_meta.TransferSyntaxUID)
        return (
            transfer_syntax_uid.is_little_endian,
            transfer_syntax_uid.is_implicit_VR,
        )

    def _read_metadata(self) -> None:
        """Read metadata from file.

        Caches the metadata and additional information such as the offset of
        the Pixel Data element and the Basic Offset Table to speed up
        subsequent access to individual frame items.
        """
        logger.debug('read metadata elements')
        if self._fp is None:
            raise IOError('File has not been opened for reading.')

        try:
            metadata = dcmread(self._fp, stop_before_pixels=True)
        except Exception as err:
            raise IOError(f'DICOM metadata cannot be read from file: "{err}"')

        # Cache Transfer Syntax UID, since we need it to decode frame items
        self._transfer_syntax_uid = UID(metadata.file_meta.TransferSyntaxUID)

        # Construct a new Dataset that is fully decoupled from the file, i.e.,
        # that does not contain any File Meta Information
        del metadata.file_meta
        self._metadata = Dataset(metadata)

        self._pixel_data_offset = self._fp.tell()
        # Determine whether dataset contains a Pixel Data element
        try:
            tag = TupleTag(self._fp.read_tag())
        except EOFError:
            raise ValueError(
                'Dataset does not represent an image information entity.'
            )
        if int(tag) not in _PIXEL_DATA_TAGS:
            raise ValueError(
                'Dataset does not represent an image information entity.'
            )
        self._as_float = False
        if int(tag) in _FLOAT_PIXEL_DATA_TAGS:
            self._as_float = True

        # Reset the file pointer to the beginning of the Pixel Data element
        self._fp.seek(self._pixel_data_offset, 0)

        logger.debug('build Basic Offset Table')
        number_of_frames = int(getattr(self._metadata, 'NumberOfFrames', 1))
        if self._transfer_syntax_uid.is_encapsulated:
            try:
                self._basic_offset_table = _get_bot(self._fp, number_of_frames)
            except Exception as err:
                raise IOError(f'Failed to build Basic Offset Table: "{err}"')
            self._first_frame_offset = self._fp.tell()
        else:
            if self._fp.is_implicit_VR:
                header_offset = 4 + 4  # tag and length
            else:
                header_offset = 4 + 2 + 2 + 4  # tag, VR, reserved and length
            self._first_frame_offset = self._pixel_data_offset + header_offset
            n_pixels = self._pixels_per_frame
            bits_allocated = self._metadata.BitsAllocated
            if bits_allocated == 1:
                self._basic_offset_table = [
                    int(np.floor(i * n_pixels / 8))
                    for i in range(number_of_frames)
                ]
            else:
                self._basic_offset_table = [
                    i * self._bytes_per_frame_uncompressed
                    for i in range(number_of_frames)
                ]

        if len(self._basic_offset_table) != number_of_frames:
            raise ValueError(
                'Length of Basic Offset Table does not match Number of Frames.'
            )

        # Build the ICC Transformation object. This takes some time and should
        # be done only once to speedup subsequent color corrections.

        icc_profile: Union[bytes, None] = None
        self._color_manager: Union[ColorManager, None] = None
        if self.metadata.SamplesPerPixel > 1:
            try:
                icc_profile = self.metadata.ICCProfile
            except AttributeError:
                try:
                    if len(self.metadata.OpticalPathSequence) > 1:
                        # This should not happen in case of a color image.
                        logger.warning(
                            'color image contains more than one optical path'
                        )
                    optical_path_item = self.metadata.OpticalPathSequence[0]
                    icc_profile = optical_path_item.ICCProfile
                except (IndexError, AttributeError):
                    logger.warning('no ICC Profile found in image metadata.')

            if icc_profile is not None:
                try:
                    self._color_manager = ColorManager(icc_profile)
                except ValueError:
                    logger.warning('could not read ICC Profile')

    @property
    def metadata(self) -> Dataset:
        """pydicom.dataset.Dataset: Metadata"""
        if self._metadata is None:
            self._read_metadata()
        return self._metadata

    @property
    def _pixels_per_frame(self) -> int:
        """int: Number of pixels per frame"""
        return int(np.prod([
            self.metadata.Rows,
            self.metadata.Columns,
            self.metadata.SamplesPerPixel
        ]))

    @property
    def _bytes_per_frame_uncompressed(self) -> int:
        """int: Number of bytes per frame when uncompressed"""
        n_pixels = self._pixels_per_frame
        bits_allocated = self.metadata.BitsAllocated
        if bits_allocated == 1:
            # Determine the nearest whole number of bytes needed to contain
            #   1-bit pixel data. e.g. 10 x 10 1-bit pixels is 100 bits, which
            #   are packed into 12.5 -> 13 bytes
            return n_pixels // 8 + (n_pixels % 8 > 0)
        else:
            return n_pixels * bits_allocated // 8

    def close(self) -> None:
        """Closes file."""
        self._fp.close()

    def read_frame_raw(self, index: int) -> bytes:
        """Reads the raw pixel data of an individual frame item.

        Parameters
        ----------
        index: int
            Zero-based frame index

        Returns
        -------
        bytes
            Pixel data of a given frame item encoded in the transfer syntax.

        Raises
        ------
        IOError
            When frame could not be read

        """
        if index > self.number_of_frames:
            raise ValueError('Frame index exceeds number of frames in image.')
        logger.debug(f'read frame #{index}')

        frame_offset = self._basic_offset_table[index]
        self._fp.seek(self._first_frame_offset + frame_offset, 0)
        if self._transfer_syntax_uid.is_encapsulated:
            try:
                stop_at = self._basic_offset_table[index + 1] - frame_offset
            except IndexError:
                # For the last frame, there is no next offset available.
                stop_at = -1
            n = 0
            # A frame may consist of multiple items (fragments).
            fragments = []
            while True:
                tag = TupleTag(self._fp.read_tag())
                if n == stop_at or int(tag) == SequenceDelimiterTag:
                    break
                if int(tag) != ItemTag:
                    raise ValueError(f'Failed to read frame #{index}.')
                length = self._fp.read_UL()
                fragments.append(self._fp.read(length))
                n += 4 + 4 + length
            frame_data = b''.join(fragments)
        else:
            frame_data = self._fp.read(self._bytes_per_frame_uncompressed)

        if len(frame_data) == 0:
            raise IOError(f'Failed to read frame #{index}.')

        return frame_data

    def read_frame(self, index: int, correct_color: bool = True) -> np.ndarray:
        """Reads and decodes the pixel data of an individual frame item.

        Parameters
        ----------
        index: int
            Zero-based frame index
        correct_color: bool, optional
            Whether colors should be corrected by applying an ICC
            transformation. Will only be performed if metadata contain an
            ICC Profile. Default = True.

        Returns
        -------
        numpy.ndarray
            Array of decoded pixels of the frame with shape (Rows x Columns)
            in case of a monochrome image or (Rows x Columns x SamplesPerPixel)
            in case of a color image.

        Raises
        ------
        IOError
            When frame could not be read

        """
        frame_data = self.read_frame_raw(index)

        logger.debug(f'decode frame #{index}')

        if self.metadata.BitsAllocated == 1:
            unpacked_frame = unpack_bits(frame_data)
            rows, columns = self.metadata.Rows, self.metadata.Columns
            n_pixels = self._pixels_per_frame
            pixel_offset = int(((index * n_pixels / 8) % 1) * 8)
            pixel_array = unpacked_frame[pixel_offset:pixel_offset + n_pixels]
            return pixel_array.reshape(rows, columns)

        frame_array = decode_frame(
            frame_data,
            rows=self.metadata.Rows,
            columns=self.metadata.Columns,
            samples_per_pixel=self.metadata.SamplesPerPixel,
            transfer_syntax_uid=self._transfer_syntax_uid,
            bits_allocated=self.metadata.BitsAllocated,
            bits_stored=self.metadata.BitsStored,
            photometric_interpretation=self.metadata.PhotometricInterpretation,
            pixel_representation=self.metadata.PixelRepresentation,
            planar_configuration=getattr(
                self.metadata, 'PlanarConfiguration', None
            )
        )

        # We don't use the color_correct_frame() function here, since we cache
        # the ICC transform on the reader instance for improved performance.
        if correct_color and self.metadata.SamplesPerPixel > 1:
            if self._color_manager is None:
                raise ValueError(
                    f'Cannot correct color of frame #{index} '
                    'because the image does either not contain an ICC Profile '
                    'or contains a malformatted ICC Profile. '
                    'See logged warning messages for details. '
                    'To read the frame without color correction set '
                    '"correct_color" to False.'
                )
            logger.debug(f'correct color of frame #{index}')
            return self._color_manager.transform_frame(frame_array)

        return frame_array

    @property
    def number_of_frames(self) -> int:
        """int: Number of frames"""
        try:
            return int(self.metadata.NumberOfFrames)
        except AttributeError:
            return 1
