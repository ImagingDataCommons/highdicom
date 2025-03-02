"""Input/Output of datasets based on DICOM Part10 files."""
import logging
from os import PathLike
import sys
import traceback
from typing_extensions import Self
from pathlib import Path
import weakref
from typing import BinaryIO

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.encaps import parse_basic_offsets
from pydicom.filebase import (
    DicomBytesIO,
    DicomFile,
    DicomIO,
    ReadableBuffer,
)
from pydicom.filereader import (
    data_element_offset_to_value,
    dcmread,
    read_file_meta_info,
    read_partial
)
from pydicom.tag import (
    ItemTag,
    SequenceDelimiterTag,
    TagListType,
    TupleTag,
)
from pydicom.uid import UID, DeflatedExplicitVRLittleEndian

from highdicom.frame import decode_frame
from highdicom.color import ColorManager
from highdicom.uid import UID as hd_UID

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


def _wrapped_dcmread(
    fp: str | PathLike | BinaryIO | ReadableBuffer | bytes,
    defer_size: str | int | float | None = None,
    stop_before_pixels: bool = False,
    force: bool = False,
    specific_tags: TagListType | None = None,
) -> pydicom.Dataset:
    """A wrapper around dcmread to support reading from bytes.

    Parameters match those of dcmread, but additional `fp` may be a raw bytes
    object containing the contents of a DICOM file.

    """
    if isinstance(fp, bytes):
        _fp = DicomBytesIO(fp)
    else:
        _fp = fp

    return dcmread(
        _fp,
        defer_size=defer_size,
        stop_before_pixels=stop_before_pixels,
        force=force,
        specific_tags=specific_tags,
    )


def _get_bot(fp: DicomIO, number_of_frames: int) -> list[int]:
    """Tries to read the value of the Basic Offset Table (BOT) item and builds
    it in case it is empty.

    Parameters
    ----------
    fp: pydicom.filebase.DicomIO
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


def _read_bot(fp: DicomIO) -> list[int]:
    """Read Basic Offset Table (BOT) item of encapsulated Pixel Data element.

    Parameters
    ----------
    fp: pydicom.filebase.DicomIO
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
    OSError
        When file pointer is not positioned at first byte of Pixel Data element

    """
    tag = TupleTag(fp.read_tag())
    if int(tag) not in _PIXEL_DATA_TAGS:
        raise OSError(
            'Expected file pointer at first byte of Pixel Data element.'
        )
    # Skip Pixel Data element header (tag, VR, length)
    pixel_data_element_value_offset = data_element_offset_to_value(
        fp.is_implicit_VR, 'OB'
    )
    fp.seek(pixel_data_element_value_offset - 4, 1)
    offsets = parse_basic_offsets(fp)
    return offsets


def _build_bot(fp: DicomIO, number_of_frames: int) -> list[int]:
    """Build Basic Offset Table (BOT) item of encapsulated Pixel Data element.

    Parameters
    ----------
    fp: pydicom.filebase.DicomIO
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
    OSError
        When file pointer is not positioned at first byte of first Frame item
        after Basic Offset Table item or when parsing of Frame item headers
        fails
    ValueError
        When the number of offsets doesn't match the specified number of frames

    """
    initial_position = fp.tell()

    # We will keep two lists, one of all fragment boundaries (regardless of
    # whether or not they are frame boundaries) and the other of just those
    # fragment boundaries that are known to be frame boundaries (as identified
    # by JPEG start markers).
    frame_offset_values = []
    fragment_offset_values = []
    i = 0
    while True:
        frame_position = fp.tell()
        tag = TupleTag(fp.read_tag())
        if int(tag) == SequenceDelimiterTag:
            break
        if int(tag) != ItemTag:
            fp.seek(initial_position, 0)
            raise OSError(
                'Building Basic Offset Table (BOT) failed. '
                f'Expected tag of Frame item #{i} at position {frame_position}.'
            )
        length = fp.read_UL()
        if length % 2:
            fp.seek(initial_position, 0)
            raise OSError(
                'Building Basic Offset Table (BOT) failed. '
                f'Length of Frame item #{i} is not a multiple of 2.'
            )
        elif length == 0:
            fp.seek(initial_position, 0)
            raise OSError(
                'Building Basic Offset Table (BOT) failed. '
                f'Length of Frame item #{i} is zero.'
            )

        current_offset = frame_position - initial_position
        fragment_offset_values.append(current_offset)

        # In case of fragmentation, we only want to get the offsets to the
        # first fragment of a given frame. We can identify those based on the
        # JPEG and JPEG 2000 markers that should be found at the beginning and
        # end of the compressed byte stream.
        first_two_bytes = fp.read(2)
        if not fp.is_little_endian:
            first_two_bytes = first_two_bytes[::-1]

        if first_two_bytes in _START_MARKERS:
            frame_offset_values.append(current_offset)

        i += 1
        fp.seek(length - 2, 1)  # minus the first two bytes

    if len(frame_offset_values) == number_of_frames:
        basic_offset_table = frame_offset_values
    elif len(fragment_offset_values) == number_of_frames:
        # This covers RLE and others that have no frame markers but have a
        # single fragment per frame
        basic_offset_table = fragment_offset_values
    else:
        raise ValueError(
            'Number of frame items does not match specified Number of Frames.'
        )

    fp.seek(initial_position, 0)
    return basic_offset_table


def _read_eot(
    extended_offset_table: bytes,
    number_of_frames: int
) -> list[int]:
    """Read an extended offset table.

    Parameters
    ----------
    extended_offset_table: bytes
        Value of the ExtendedOffsetTable attribute.
    number_of_frames: int
        Number of frames contained in the Pixel Data element

    Returns
    -------
    List[int]
        Offset of each Frame item in bytes from the first byte of the Pixel Data
        element following the BOT item

    """
    result = np.frombuffer(extended_offset_table, dtype=np.uint64).tolist()

    if len(result) != number_of_frames:
        raise ValueError(
            'The number of items in the extended offset table does nt match '
            'the specified number of frames.'
        )

    return result


def _stop_after_group_2(tag: pydicom.tag.BaseTag, vr: str, length: int) -> bool:
    """
    Stop DCM reading after first tag groups
    """
    return tag.group > 2


class ImageFileReader:

    """Reader for DICOM datasets representing Image Information Entities.

    It provides efficient, "lazy", access to individual frame items contained
    in the Pixel Data element without loading the entire element into memory.

    Note
    ----
    As of highdicom 0.24.0, users should prefer the :class:`highdicom.Image`
    class with lazy frame retrieval (e.g. as output by the
    :func:`highdicom.imread` function when ``lazy_frame_retrieval=True``) to
    this class in most situations. The :class:`highdicom.Image` class offers
    the same lazy frame-level access, but additionally has several higher-level
    features, including the ability to apply pixel transformations to loaded
    frames, construct total pixel matrices, and construct volumes.

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

    def __init__(self, filename: str | Path | DicomIO):
        """
        Parameters
        ----------
        filename: Union[str, pathlib.Path, pydicom.filebase.DicomIO]
            DICOM Part10 file containing a dataset of an image SOP Instance

        """
        if isinstance(filename, DicomIO):
            fp = filename
            self._fp = fp
            if hasattr(filename, "name") and filename.name is not None:
                self._filename = Path(fp.name)
            else:
                self._filename = None

            # Since we did not open the file-like object, we should not close
            # it
            self._should_close = False
        elif isinstance(filename, (str, Path)):
            self._filename = Path(filename)
            self._fp = None

            # Since we did open the file-like object, we should close it
            self._should_close = True
        else:
            raise TypeError(
                'Argument "filename" must be either an open DicomIO object '
                'or the path to a DICOM file stored on disk.'
            )
        self._metadata: Dataset | weakref.ReferenceType | None = None
        self._voi_lut = None
        self._palette_color_lut = None
        self._modality_lut = None
        self._enter_depth = 0

    def _change_metadata_ownership(self) -> FileDataset:
        """Set the metadata using a weakref.

        This is used by imread to allow an Image object to take ownership of
        the metadata and this file reader without creating potentially
        problematic reference cycles.

        Returns
        -------
        metadata: pydicom.FileDataset
            Dataset containing the metadata.

        """
        with self:
            self._read_metadata()
            # The file meta was stripped from metadata previously, add it back
            # here to give a FileDataset
            metadata = FileDataset(
                self._filename,
                dataset=self.metadata,
                file_meta=self._file_meta,
                is_implicit_VR=self.transfer_syntax_uid.is_implicit_VR,
                is_little_endian=self.transfer_syntax_uid.is_little_endian,
            )
        self._metadata = weakref.ref(metadata)
        return metadata

    @property
    def filename(self) -> str:
        """str: Path to the image file"""
        return str(self._filename)

    def __enter__(self) -> Self:
        if self._enter_depth == 0:
            self.open()
        self._enter_depth += 1
        return self

    def __exit__(self, except_type, except_value, except_trace) -> None:
        self._enter_depth -= 1
        if self._enter_depth < 1:
            if self._should_close:
                self._fp.close()
                self._fp = None
            if except_value:
                sys.stderr.write(
                    f'Error while accessing file "{self._filename}":\n'
                    f'{except_value}'
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
        OSError
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
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'File not found: "{self._filename}"'
                ) from e
            except Exception as e:
                raise OSError(
                    f'Could not open file for reading: "{self._filename}"'
                ) from e
        if not hasattr(self._fp, 'is_implicit_VR'):
            self._fp.seek(0)
            is_little_endian, is_implicit_VR = self._check_file_format(self._fp)
            self._fp.is_little_endian = is_little_endian
            self._fp.is_implicit_VR = is_implicit_VR

    def _check_file_format(
            self,
            fp: DicomIO
    ) -> tuple[bool, bool]:
        """Check whether file object represents a DICOM Part 10 file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomIO
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
        if self._filename is None:
            # fileobj type is BinaryIO but works fine with a DicomBytesIO
            file_meta = read_partial(
                fileobj=fp,  # type: ignore
                stop_when=_stop_after_group_2
            ).file_meta
            fp.seek(0)
        else:
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
            raise OSError('File has not been opened for reading.')

        try:
            metadata = dcmread(self._fp, stop_before_pixels=True)
        except Exception as err:
            raise OSError(
                f'DICOM metadata cannot be read from file: "{err}"'
            ) from err

        # Cache file meta, since we need it to decode frame items
        self._file_meta = metadata.file_meta

        # Construct a new Dataset that is fully decoupled from the file, i.e.,
        # that does not contain any File Meta Information
        del metadata.file_meta
        self._metadata = Dataset(metadata)

        self._pixel_data_offset = self._fp.tell()

        if self.transfer_syntax_uid == DeflatedExplicitVRLittleEndian:
            # The entire file is compressed with DEFLATE. These cannot be used
            # since the entire file must be decompressed to read or build the
            # basic/extended offset
            raise ValueError(
                'Deflated transfer syntaxes cannot be used with the '
                'ImageFileReader.'
            )

        # Determine whether dataset contains a Pixel Data element
        try:
            tag = TupleTag(self._fp.read_tag())
        except EOFError as e:
            raise ValueError(
                'Dataset does not represent an image information entity.'
            ) from e
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
        number_of_frames = int(getattr(metadata, 'NumberOfFrames', 1))
        if self.transfer_syntax_uid.is_encapsulated:
            if 'ExtendedOffsetTable' in metadata:
                # Try the extended offset table first
                self._offset_table = _read_eot(
                    metadata.ExtendedOffsetTable,
                    number_of_frames
                )
                # tag, VR, reserved and length for PixelData plus tag and
                # length of the BOT (which should be empty if extended offsets
                # are present)
                header_offset = 4 + 2 + 2 + 4 + 4 + 4
                self._first_frame_offset = (
                    self._pixel_data_offset + 20
                )
            else:
                # Fall back to the basic offset table
                try:
                    self._offset_table = _get_bot(self._fp, number_of_frames)
                except Exception as err:
                    raise OSError(
                        f'Failed to build Basic Offset Table: "{err}"'
                    ) from err
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
                self._offset_table = [
                    int(np.floor(i * n_pixels / 8))
                    for i in range(number_of_frames)
                ]
            else:
                self._offset_table = [
                    i * self._bytes_per_frame_uncompressed
                    for i in range(number_of_frames)
                ]

        if len(self._offset_table) != number_of_frames:
            raise ValueError(
                'Length of Basic Offset Table does not match Number of Frames.'
            )

        # Build the ICC Transformation object. This takes some time and should
        # be done only once to speedup subsequent color corrections.

        icc_profile: bytes | None = None
        self._color_manager: ColorManager | None = None
        if metadata.SamplesPerPixel > 1:
            try:
                icc_profile = metadata.ICCProfile
            except AttributeError:
                try:
                    if len(metadata.OpticalPathSequence) > 1:
                        # This should not happen in case of a color image.
                        logger.warning(
                            'color image contains more than one optical path'
                        )
                    optical_path_item = metadata.OpticalPathSequence[0]
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
        if isinstance(self._metadata, weakref.ReferenceType):
            return self._metadata()
        if self._metadata is None:
            self._read_metadata()
        return self._metadata

    @property
    def transfer_syntax_uid(self) -> hd_UID:
        """highdicom.UID: Transfer Syntax UID of the file."""
        self.metadata  # ensure metadata has been read
        return hd_UID(self._file_meta.TransferSyntaxUID)

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
            if self.metadata.PhotometricInterpretation == 'YBR_FULL_422':
                # Account for subsampling of CB and CR when calculating
                # expected number of samples
                # See https://dicom.nema.org/medical/dicom/current/output/chtml
                # /part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2
                n_pixels = self.metadata.Rows * self.metadata.Columns * 2
            return n_pixels * bits_allocated // 8

    def close(self) -> None:
        """Closes file."""
        if self._should_close:
            try:
                self._fp.close()
            except AttributeError:
                return

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
        OSError
            When frame could not be read

        """
        if index > self.number_of_frames:
            raise ValueError('Frame index exceeds number of frames in image.')
        logger.debug(f'read frame #{index}')

        frame_offset = self._offset_table[index]
        self._fp.seek(self._first_frame_offset + frame_offset, 0)
        if self.transfer_syntax_uid.is_encapsulated:
            try:
                stop_at = self._offset_table[index + 1] - frame_offset
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
            raise OSError(f'Failed to read frame #{index}.')

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
        OSError
            When frame could not be read

        """
        frame_data = self.read_frame_raw(index)

        logger.debug(f'decode frame #{index}')

        frame_array = decode_frame(
            frame_data,
            rows=self.metadata.Rows,
            columns=self.metadata.Columns,
            samples_per_pixel=self.metadata.SamplesPerPixel,
            transfer_syntax_uid=self.transfer_syntax_uid,
            bits_allocated=self.metadata.BitsAllocated,
            bits_stored=self.metadata.BitsStored,
            photometric_interpretation=self.metadata.PhotometricInterpretation,
            pixel_representation=self.metadata.PixelRepresentation,
            planar_configuration=getattr(
                self.metadata, 'PlanarConfiguration', None
            ),
            index=index,
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
        if 'NumberOfFrames' in self.metadata:
            return self.metadata.NumberOfFrames
        else:
            return 1
