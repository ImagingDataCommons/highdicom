"""Input/Output of datasets based on DICOM Part10 files."""
import logging
import sys
import traceback
from io import BytesIO
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageCms
from PIL.ImageCms import (
    applyTransform,
    getProfileDescription,
    getProfileName,
    ImageCmsProfile,
    ImageCmsTransform,
    isIntentSupported,
)
from pydicom.dataset import Dataset
from pydicom.encaps import get_frame_offsets, read_item, defragment_data
from pydicom.filebase import DicomFile
from pydicom.filereader import (
    data_element_offset_to_value,
    dcmread,
    read_file_meta_info,
)
from pydicom.pixel_data_handlers.numpy_handler import unpack_bits
from pydicom.pixel_data_handlers.pillow_handler import (
    PillowSupportedTransferSyntaxes
)
from pydicom.pixel_data_handlers.util import pixel_dtype, get_expected_length
from pydicom.tag import TupleTag, ItemTag, SequenceDelimiterTag
from pydicom.uid import (
    JPEGBaseline,
    JPEG2000Lossless,
    UID,
    UncompressedPixelTransferSyntaxes,
)

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


def _get_bot(fp: DicomFile, number_of_frames: int) -> List[int]:
    """Tries to read the value of the Basic Offset Table (BOT) item and builds
    it case it is not found.

    Parameters
    ----------
    fp: pydicom.filebase.DicomFile
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
        raise ValueError('Reading of Basic Offset Table failed')
    fp.seek(first_frame_offset, 0)

    # Basic Offset Table item must be present, but it may be empty
    if len(basic_offset_table) == 0:
        logger.debug('Basic Offset Table item is empty')
    if len(basic_offset_table) != number_of_frames:
        logger.debug('build Basic Offset Table item')
        basic_offset_table = _build_bot(
            fp,
            number_of_frames=number_of_frames
        )

    return basic_offset_table


def _read_bot(fp: DicomFile) -> List[int]:
    """Reads the Basic Offset Table (BOT) item of an encapsulated Pixel Data
    element.

    Parameters
    ----------
    fp: pydicom.filebase.DicomFile
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


def _build_bot(fp: DicomFile, number_of_frames: int) -> List[int]:
    """Builds a Basic Offset Table (BOT) item of an encapsulated Pixel Data
    element.

    Parameters
    ----------
    fp: pydicom.filebase.DicomFile
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

        first_two_bytes = fp.read(2, 1)
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


def _build_icc_transform(metadata: Dataset) -> Union[ImageCmsTransform, None]:
    """Builds an ICC Transformation object.

    Parameters
    ----------
    metadata: pydicom.dataset.Dataset
        DICOM metadata of an image instance

    Returns
    -------
    Union[PIL.ImageCms.ImageCmsTransform, None]
        ICC Transformation object

    """
    profile: Union[bytes, None]
    try:
        icc_profile = metadata.ICCProfile
    except AttributeError:
        try:
            if len(metadata.OpticalPathSequence) > 1:
                # This should not happen in case of a color image, but
                # better safe than sorry.
                logger.warning(
                    'metadata describes more than one optical path'
                )
            icc_profile = metadata.OpticalPathSequence[0].ICCProfile
        except (IndexError, AttributeError):
            message = 'no ICC Profile found'
            if metadata.SamplesPerPixel > 1:
                logger.warning(message)
            else:
                logger.debug(message)
            icc_profile = None

    if icc_profile is not None:
        profile = ImageCmsProfile(BytesIO(icc_profile))
        name = getProfileName(profile).strip()
        description = getProfileDescription(profile).strip()
        logger.info(f'found ICC Profile "{name}": "{description}"')

        logger.debug('build ICC Transform')
        intent = ImageCms.INTENT_RELATIVE_COLORIMETRIC
        if not isIntentSupported(
            profile,
            intent=intent,
            direction=ImageCms.DIRECTION_INPUT
        ):
            raise ValueError(
                'ICC Profile does not support desired '
                'color transformation intent.'
            )
        return ImageCms.buildTransform(
            inputProfile=profile,
            outputProfile=ImageCms.createProfile('sRGB'),
            inMode='RGB',  # according to PS3.3 C.11.15.1.1
            outMode='RGB'
        )


def _apply_icc_transform(
        image: Image.Image,
        transform: ImageCmsTransform
    ) -> np.ndarray:
    """Applies an ICC transformation to correct the color of an image.

    Parameters
    ----------
    image: PIL.Image.Image
        Image
    transform: PIL.ImageCms.ImageCmsTransform
        ICC transformation object

    """
    applyTransform(image, transform, inPlace=True)


class ImageFileReader(object):

    """Reader for DICOM datasets representing Image Information Entities.

    It provides efficient access to individual Frame items contained in the
    Pixel Data element without loading the entire element into memory.

    Attributes
    ----------
    filename: str
        Path to the DICOM Part10 file on disk

    Examples
    --------
    >>> from highdicom.io import ImageFileReader
    >>> with ImageFileReader('/path/to/file.dcm') as image:
    ...     print(image.metadata)
    ...     for i in range(image.number_of_frames):
    ...         frame = image.read_frame(i)
    ...         print(frame.shape)

    """

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename: str
            Path to a DICOM Part10 file containing a dataset of an image
            SOP Instance

        """
        self.filename = filename

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, except_type, except_value, except_trace):
        self._fp.close()
        if except_value:
            sys.stdout.write(
                'Error while accessing file "{}":\n{}'.format(
                    self.filename, str(except_value)
                )
            )
            for tb in traceback.format_tb(except_trace):
                sys.stdout.write(tb)
            sys.exit(1)

    def open(self):
        """Opens file and reads metadata from it.

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
        logger.info('read File Meta Information')
        file_meta = read_file_meta_info(self.filename)
        transfer_syntax_uid = UID(file_meta.TransferSyntaxUID)
        try:
            self._fp = DicomFile(str(self.filename), mode='rb')
            self._fp.is_little_endian = transfer_syntax_uid.is_little_endian
            self._fp.is_implicit_VR = transfer_syntax_uid.is_implicit_VR
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found: "{self.filename}"')
        except Exception:
            raise OSError(
                f'Could not open file for reading: "{self.filename}"'
            )
        logger.info('read metadata elements')
        try:
            self._metadata = dcmread(self._fp, stop_before_pixels=True)
        except Exception as err:
            raise IOError(
                f'DICOM metadata cannot be read from file "{self.filename}": '
                f'"{err}"'
            )
        self._pixel_data_offset = self._fp.tell()
        # Determine whether dataset contains a Pixel Data element
        tag = TupleTag(self._fp.read_tag())
        self._fp.seek(self._pixel_data_offset, 0)
        if int(tag) not in _PIXEL_DATA_TAGS:
            raise ValueError('Dataset does not represent an image')
        self._as_float = False
        if int(tag) in _FLOAT_PIXEL_DATA_TAGS:
            self._as_float = True
        # Build the ICC Transformation object. This takes some time and should
        # be done only once to speedup subsequent color corrections.
        self._icc_transform = _build_icc_transform(self._metadata)

        logger.info('build Basic Offset Table')
        transfer_syntax_uid = self.metadata.file_meta.TransferSyntaxUID
        if transfer_syntax_uid.is_encapsulated:
            try:
                self._basic_offset_table = _get_bot(
                    self._fp,
                    number_of_frames=self.number_of_frames
                )
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
            bits_allocated = self.metadata.BitsAllocated
            if bits_allocated == 1:
                self._basic_offset_table = [
                    int(np.floor(i * n_pixels / 8))
                    for i in range(self.number_of_frames)
                ]
            else:
                self._basic_offset_table = [
                    i * self._bytes_per_frame_uncompressed
                    for i in range(self.number_of_frames)
                ]

        if len(self._basic_offset_table) != self.number_of_frames:
            raise ValueError(
                'Length of Basic Offset Table does not match Number of Frames.'
            )

    @property
    def metadata(self) -> Dataset:
        """pydicom.dataset.Dataset: Metadata"""
        try:
            return self._metadata
        except AttributeError:
            raise IOError('File has not been opened for reading.')

    @property
    def _pixels_per_frame(self) -> int:
        """int: Number of pixels per frame"""
        return np.prod([
            self.metadata.Rows,
            self.metadata.Columns,
            self.metadata.SamplesPerPixel
        ])

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

    def close(self):
        """Closes file."""
        self._fp.close()

    def read_frame(self, index: int, correct_color: bool = True) -> np.ndarray:
        """Reads an individual frame.

        Parameters
        ----------
        index: int
            Zero-based frame index
        correct_color: bool, optional
            Whether colors should be corrected by applying an ICC
            transformation. Will only be performed if metadata contain an
            ICC Profile.

        Returns
        -------
        numpy.ndarray
            Array of decoded pixels of the frame with shape (Rows x Columns)
            in case of a monochrome image or (Rows x Columns x SamplesPerPixel)
            in case of a color image.

        """
        if index > self.number_of_frames:
            raise ValueError('Frame index exceeds number of frames in image.')
        logger.info(f'read frame #{index}')
        frame_offset = self._basic_offset_table[index]
        self._fp.seek(self._first_frame_offset + frame_offset, 0)
        if self.metadata.file_meta.TransferSyntaxUID.is_encapsulated:
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

        logger.info(f'decode frame #{index}')

        bits_allocated = self.metadata.BitsAllocated
        if bits_allocated == 1:
            unpacked_frame = unpack_bits(frame_data)
            rows, columns = self.metadata.Rows, self.metadata.Columns
            n_pixels = self._pixels_per_frame
            pixel_offset = int(((index * n_pixels / 8) % 1) * 8)
            pixel_array = unpacked_frame[pixel_offset:pixel_offset + n_pixels]
            return pixel_array.reshape(rows, columns)

        frame_array = self._decode_frame(frame_data)

        if correct_color and self._icc_transform is not None:
            logger.info(f'correct color of frame #{index}')
            image = Image.fromarray(frame_array)
            _apply_icc_transform(image, self._icc_transform)
            return np.asarray(image)

        return frame_array

    @property
    def number_of_frames(self) -> int:
        """int: Number of frames"""
        try:
            return int(self.metadata.NumberOfFrames)
        except AttributeError:
            return 1

    def _decode_frame(self, value: bytes) -> np.ndarray:
        """Decodes pixel data of an individual frame.

        Parameters
        ----------
        value: bytes
            Pixel data of a frame (potentially compressed in case
            of encapsulated format encoding, depending on the transfer syntax)

        Returns
        -------
        numpy.ndarray
            Decoded pixel data

        Raises
        ------
        ValueError
            When transfer syntax is not supported.
        AttributeError
            When `metadata` does not have any of the following attributes:
            ``Rows``, ``Columns``, ``BitsAllocated``, ``SamplesPerPixel``,
            ``PhotometricInterpretation``, ``PixelRepresentation``,
            ``PlanarConfiguration``

        """
        required_attributes = {
            'Rows',
            'Columns',
            'BitsAllocated',
            'SamplesPerPixel',
            'PhotometricInterpretation',
            'PixelRepresentation',
        }
        for attr in required_attributes:
            if not hasattr(self.metadata, attr):
                raise AttributeError(
                    f'Image is missing required attribute "{attr}".'
                )
        if self.metadata.SamplesPerPixel > 1:
            if not hasattr(self.metadata, 'PlanarConfiguration'):
                raise AttributeError(
                    f'Color image is missing required attribute "{attr}".'
                )

        transfer_syntax_uid = self.metadata.file_meta.TransferSyntaxUID
        if transfer_syntax_uid in UncompressedPixelTransferSyntaxes:
            return self._decode_frame_native(value)
        elif transfer_syntax_uid == JPEGBaseline:
            return self._decode_frame_JPEG_baseline(value)
        elif transfer_syntax_uid == JPEG2000Lossless:
            return self._decode_frame_JPEG2000_lossless(value)
        else:
            raise ValueError(
                f'Transfer Syntax "{transfer_syntax_uid}" is not supported '
                'for decoding of frames.'
            )

    def _decode_frame_JPEG_baseline(self, value: bytes) -> np.ndarray:
        """Decodes pixel data of an individual frame with transfer syntax
        ``"1.2.840.10008.1.2.4.50"`` (JPEG Baseline).

        Parameters
        ----------
        value: bytes
            Pixel data of a frame

        Returns
        -------
        numpy.ndarray
            Decoded pixel data

        Raises
        ------
        ValueError
            When EOI or SOI marker segments are not found in byte stream

        Note
        ----
        Will apply ICC Profile if present in `metadata.`

        """
        if not value.startswith(_JPEG_SOI_MARKER):
            raise ValueError('Not valid JPEG. Missing SOI marker segment.')
        if _JPEG_EOI_MARKER not in value[-6:]:  # may be zero-padded
            raise ValueError('Not valid JPEG. Missing EOI marker segment.')

        n_bytes = self._bytes_per_frame_uncompressed
        image = Image.open(BytesIO(value[:n_bytes]))
        if self.metadata.PhotometricInterpretation == 'RGB':
            # RGB color images, which were not transformed into YCbCr color
            # space upon JPEG compression, need to be handled separately.
            # Pillow assumes that images were transformed into YCbCr color
            # space prior to JPEG compression. However, with photometric
            # interpretation RGB, no color transformation was performed.
            # Setting the value of "mode" to YCbCr signals Pillow to not
            # apply any color transformation upon decompression.
            color_mode = 'YCbCr'
            image.tile = [(
                'jpeg',
                image.tile[0][1],
                image.tile[0][2],
                (color_mode, ''),
            )]
            image.mode = color_mode
            image.rawmode = color_mode

        return np.asarray(image)

    def _decode_frame_JPEG2000_lossless(self, value: bytes) -> np.ndarray:
        """Decodes pixel data of an individual frame with transfer syntax
        ``"1.2.840.10008.1.2.4.90"`` (JPEG 2000 Lossless).

        Parameters
        ----------
        value: bytes
            Compressed pixel data of a frame

        Returns
        -------
        numpy.ndarray
            Decoded pixel data

        Raises
        ------
        ValueError
            When EOC or SOC marker segments are not found in byte stream

        Note
        ----
        Will apply ICC Profile if present in the `metadata.`

        """
        if not value.startswith(_JPEG2000_SOC_MARKER):
            raise ValueError('Not valid JPEG 2000. Missing SOC marker segment.')
        if _JPEG2000_EOC_MARKER not in value[-6:]:  # may be zero-padded
            raise ValueError('Not valid JPEG 2000. Missing EOC marker segment.')

        n_bytes = self._bytes_per_frame_uncompressed
        image = Image.open(BytesIO(value[:n_bytes]))
        return np.asarray(image)

    def _decode_frame_bitpacked(
        self,
        value: bytes,
        ignore_bits: int
    ) -> np.ndarray:
        """Decodes bitpacked pixel data of an individual frame with one of the
        following transfer syntaxes:

            - ``"1.2.840.10008.1.2"`` (Implicit VR Little Endian)
            - ``"1.2.840.10008.1.2.1"`` (Explicit VR Little Endian)
            - ``"1.2.840.10008.1.2.2"`` (Explicit VR Big Endian)

        Parameters
        ----------
        value: bytes
            Bitpacked pixel data of a frame

        Returns
        -------
        numpy.ndarray
            Decoded pixel data

        """
        rows = self.metadata.Rows
        columns = self.metadata.Columns
        bits_allocated = self.metadata.BitsAllocated
        if bits_allocated != 1:
            raise ValueError('Expected image with 1 bit allocated per pixel.')
        # Skip any trailing padding bits
        n_pixels = self._pixels_per_frame
        pixel_array = unpack_bits(value)[:n_pixels]
        return pixel_array.reshape(rows, columns)

    def _decode_frame_native(self, value: bytes) -> np.ndarray:
        """Decodes pixel data of an individual frame with one of the following
        transfer syntaxes:

            - ``"1.2.840.10008.1.2"`` (Implicit VR Little Endian)
            - ``"1.2.840.10008.1.2.1"`` (Explicit VR Little Endian)
            - ``"1.2.840.10008.1.2.2"`` (Explicit VR Big Endian)

        Parameters
        ----------
        value: bytes
            Uncompressed pixel data of a frame

        Returns
        -------
        numpy.ndarray
            Decoded pixel data

        """
        rows = self.metadata.Rows
        columns = self.metadata.Columns
        samples_per_pixel = self.metadata.SamplesPerPixel
        photometric_interpretation = self.metadata.PhotometricInterpretation

        dtype = pixel_dtype(self.metadata, as_float=self._as_float)
        n_pixels = self._pixels_per_frame
        # Skip the trailing padding byte(s) if present
        n_bytes = self._bytes_per_frame_uncompressed
        pixel_array = np.frombuffer(value[:n_bytes], dtype=dtype)
        if photometric_interpretation == 'YBR_FULL_422':
            # YBR_FULL_422 data needs to be resampled
            # Y1 Y2 B1 R1 -> Y1 B1 R1 Y2 B1 R1
            out = np.zeros(n_pixels // 2 * 3, dtype=dtype)
            out[::6] = pixel_array[::4]  # Y1
            out[3::6] = pixel_array[1::4]  # Y2
            out[1::6], out[4::6] = pixel_array[2::4], pixel_array[2::4]  # B
            out[2::6], out[5::6] = pixel_array[3::4], pixel_array[3::4]  # R
            pixel_array = out

        if samples_per_pixel > 1:
            planar_configuration = self.metadata.PlanarConfiguration
            if planar_configuration == 0:
                return pixel_array.reshape(rows, columns, samples_per_pixel)
            elif planar_configuration == 1:
                pixel_array = pixel_array.reshape(
                    samples_per_pixel, rows, columns
                )
                return pixel_array.transpose(1, 2, 0)
            else:
                raise ValueError(
                    f'Unexpected Planar Configuration "{planar_configuration}"'
                )
        else:
            return pixel_array.reshape(rows, columns)
