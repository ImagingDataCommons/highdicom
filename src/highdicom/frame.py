import logging
from io import BytesIO
from typing import Optional, Union

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from pydicom.pixel_data_handlers.rle_handler import rle_encode_frame
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGLSLossless,
    UID,
    RLELossless,
)

from highdicom.enum import (
    PhotometricInterpretationValues,
    PixelRepresentationValues,
    PlanarConfigurationValues,
)

logger = logging.getLogger(__name__)


def encode_frame(
    array: np.ndarray,
    transfer_syntax_uid: str,
    bits_allocated: int,
    bits_stored: int,
    photometric_interpretation: Union[PhotometricInterpretationValues, str],
    pixel_representation: Union[PixelRepresentationValues, int] = 0,
    planar_configuration: Optional[Union[PlanarConfigurationValues, int]] = None
) -> bytes:
    """Encode pixel data of an individual frame.

    Parameters
    ----------
    array: numpy.ndarray
        Pixel data in form of an array with dimensions
        (Rows x Columns x SamplesPerPixel) in case of a color image and
        (Rows x Columns) in case of a monochrome image
    transfer_syntax_uid: int
        Transfer Syntax UID
    bits_allocated: int
        Number of bits that need to be allocated per pixel sample
    bits_stored: int
        Number of bits that are required to store a pixel sample
    photometric_interpretation: int
        Photometric interpretation
    pixel_representation: Union[highdicom.PixelRepresentationValues, int, None], optional
        Whether pixel samples are represented as unsigned integers or
        2's complements
    planar_configuration: Union[highdicom.PlanarConfigurationValues, int, None], optional
        Whether color samples are encoded by pixel (``R1G1B1R2G2B2...``) or
        by plane (``R1R2...G1G2...B1B2...``).

    Returns
    -------
    bytes
        Encoded pixel data (potentially compressed in case of encapsulated
        format encoding, depending on the transfer snytax)

    Raises
    ------
    ValueError
        When `transfer_syntax_uid` is not supported or when
        `planar_configuration` is missing in case of a color image frame.

    Note
    ----
    In case of color image frames, the `photometric_interpretation` parameter
    describes the color space of the **encoded** pixel data and data may be
    converted from RGB color space into the specified color space upon
    encoding.  For example, the JPEG codec converts pixels from RGB into
    YBR color space prior to compression to take advantage of the correlation
    between RGB color bands and improve compression efficiency. Therefore,
    pixels are supposed to be provided via `array` in RGB color space, but
    `photometric_interpretation` needs to specify a YBR color space.

    """  # noqa: E501
    rows = array.shape[0]
    cols = array.shape[1]
    if array.ndim > 2:
        if planar_configuration is None:
            raise ValueError(
                'Planar configuration needs to be specified for encoding of '
                'color image frames.'
            )
        planar_configuration = PlanarConfigurationValues(
            planar_configuration
        ).value
        samples_per_pixel = array.shape[2]
    else:
        samples_per_pixel = 1

    pixel_representation = PixelRepresentationValues(
        pixel_representation
    ).value
    photometric_interpretation = PhotometricInterpretationValues(
        photometric_interpretation
    ).value

    uncompressed_transfer_syntaxes = {
        ExplicitVRLittleEndian,
        ImplicitVRLittleEndian,
    }
    compressed_transfer_syntaxes = {
        JPEGBaseline8Bit,
        JPEG2000Lossless,
        JPEGLSLossless,
        RLELossless,
    }
    supported_transfer_syntaxes = uncompressed_transfer_syntaxes.union(
        compressed_transfer_syntaxes
    )
    if transfer_syntax_uid not in supported_transfer_syntaxes:
        raise ValueError(
            f'Transfer Syntax "{transfer_syntax_uid}" is not supported. '
            'Only the following are supported: "{}"'.format(
                '", "'.join(supported_transfer_syntaxes)
            )
        )
    if transfer_syntax_uid in uncompressed_transfer_syntaxes:
        if samples_per_pixel > 1:
            if planar_configuration != 0:
                raise ValueError(
                    'Planar configuration must be 0 for color image frames '
                    'with native encoding.'
                )
        if bits_allocated == 1:
            if (rows * cols * samples_per_pixel) % 8 != 0:
                raise ValueError(
                    'Frame cannot be bit packed because its size is not a '
                    'multiple of 8.'
                )
            return pack_bits(array.flatten())
        else:
            return array.flatten().tobytes()

    else:
        compression_lut = {
            JPEGBaseline8Bit: (
                'jpeg',
                {
                    'quality': 95
                },
            ),
            JPEG2000Lossless: (
                'jpeg2000',
                {
                    'tile_size': None,
                    'num_resolutions': 1,
                    'irreversible': False,
                },
            ),
            JPEGLSLossless: (
                'JPEG-LS',
                {
                    'near_lossless': 0,
                }
            )
        }

        if transfer_syntax_uid == JPEGBaseline8Bit:
            if samples_per_pixel == 1:
                if planar_configuration is not None:
                    raise ValueError(
                        'Planar configuration must be absent for encoding of '
                        'monochrome image frames with JPEG Baseline codec.'
                    )
                if photometric_interpretation not in (
                        'MONOCHROME1', 'MONOCHROME2'
                    ):
                    raise ValueError(
                        'Photometric intpretation must be either "MONOCHROME1" '
                        'or "MONOCHROME2" for encoding of monochrome image '
                        'frames with JPEG Baseline codec.'
                    )
            elif samples_per_pixel == 3:
                if photometric_interpretation != 'YBR_FULL_422':
                    raise ValueError(
                        'Photometric intpretation must be "YBR_FULL_422" for '
                        'encoding of color image frames with '
                        'JPEG Baseline codec.'
                    )
                if planar_configuration != 0:
                    raise ValueError(
                        'Planar configuration must be 0 for encoding of '
                        'color image frames with JPEG Baseline codec.'
                    )
            else:
                raise ValueError(
                    'Samples per pixel must be 1 or 3 for '
                    'encoding of image frames with JPEG Baseline codec.'
                )
            if bits_allocated != 8 or bits_stored != 8:
                raise ValueError(
                    'Bits allocated and bits stored must be 8 for '
                    'encoding of image frames with JPEG Baseline codec.'
                )
            if pixel_representation != 0:
                raise ValueError(
                    'Pixel representation must be 0 for '
                    'encoding of image frames with JPEG Baseline codec.'
                )

        elif transfer_syntax_uid == JPEG2000Lossless:
            if samples_per_pixel == 1:
                if planar_configuration is not None:
                    raise ValueError(
                        'Planar configuration must be absent for encoding of '
                        'monochrome image frames with Lossless JPEG 2000 codec.'
                    )
                if photometric_interpretation not in (
                        'MONOCHROME1', 'MONOCHROME2'
                    ):
                    raise ValueError(
                        'Photometric intpretation must be either "MONOCHROME1" '
                        'or "MONOCHROME2" for encoding of monochrome image '
                        'frames with Lossless JPEG 2000 codec.'
                    )
                if bits_allocated not in (8, 16):
                    raise ValueError(
                        'Bits Allocated must be 8 or 16 for encoding of '
                        'monochrome image frames with Lossless JPEG 2000 codec.'
                    )
            elif samples_per_pixel == 3:
                if photometric_interpretation != 'YBR_FULL':
                    raise ValueError(
                        'Photometric interpretation must be "YBR_FULL" for '
                        'encoding of color image frames with '
                        'Lossless JPEG 2000 codec.'
                    )
                if planar_configuration != 0:
                    raise ValueError(
                        'Planar configuration must be 0 for encoding of '
                        'color image frames with Lossless JPEG 2000 codec.'
                    )
                if bits_allocated != 8:
                    raise ValueError(
                        'Bits Allocated must be 8 for encoding of '
                        'color image frames with Lossless JPEG 2000 codec.'
                    )
            else:
                raise ValueError(
                    'Samples per pixel must be 1 or 3 for '
                    'encoding of image frames with Lossless JPEG 2000 codec.'
                )
            if pixel_representation != 0:
                raise ValueError(
                    'Pixel representation must be 0 for '
                    'encoding of image frames with Lossless JPEG 2000 codec.'
                )

        elif transfer_syntax_uid == JPEGLSLossless:
            import pillow_jpls  # noqa
            if samples_per_pixel == 1:
                if planar_configuration is not None:
                    raise ValueError(
                        'Planar configuration must be absent for encoding of '
                        'monochrome image frames with Lossless JPEG-LS codec.'
                    )
                if photometric_interpretation not in (
                        'MONOCHROME1', 'MONOCHROME2'
                    ):
                    raise ValueError(
                        'Photometric intpretation must be either "MONOCHROME1" '
                        'or "MONOCHROME2" for encoding of monochrome image '
                        'frames with Lossless JPEG-LS codec.'
                    )
                if bits_allocated not in (8, 16):
                    raise ValueError(
                        'Bits Allocated must be 8 or 16 for encoding of '
                        'monochrome image frames with Lossless JPEG-LS codec.'
                    )
            elif samples_per_pixel == 3:
                if photometric_interpretation != 'YBR_FULL':
                    raise ValueError(
                        'Photometric interpretation must be "YBR_FULL" for '
                        'encoding of color image frames with '
                        'Lossless JPEG-LS codec.'
                    )
                if planar_configuration != 0:
                    raise ValueError(
                        'Planar configuration must be 0 for encoding of '
                        'color image frames with Lossless JPEG-LS codec.'
                    )
                if bits_allocated != 8:
                    raise ValueError(
                        'Bits Allocated must be 8 for encoding of '
                        'color image frames with Lossless JPEG-LS codec.'
                    )
            else:
                raise ValueError(
                    'Samples per pixel must be 1 or 3 for '
                    'encoding of image frames with Lossless JPEG-LS codec.'
                )
            if pixel_representation != 0:
                raise ValueError(
                    'Pixel representation must be 0 for '
                    'encoding of image frames with Lossless JPEG-LS codec.'
                )

        if transfer_syntax_uid in compression_lut.keys():
            image_format, kwargs = compression_lut[transfer_syntax_uid]
            if samples_per_pixel == 3:
                image = Image.fromarray(array, mode='RGB')
            else:
                image = Image.fromarray(array)
            with BytesIO() as buf:
                image.save(buf, format=image_format, **kwargs)
                data = buf.getvalue()
        elif transfer_syntax_uid == RLELossless:
            data = rle_encode_frame(array)
        else:
            raise ValueError(
                f'Transfer Syntax "{transfer_syntax_uid}" is not supported.'
            )
    return data


def decode_frame(
    value: bytes,
    transfer_syntax_uid: str,
    rows: int,
    columns: int,
    samples_per_pixel: int,
    bits_allocated: int,
    bits_stored: int,
    photometric_interpretation: Union[PhotometricInterpretationValues, str],
    pixel_representation: Union[PixelRepresentationValues, int] = 0,
    planar_configuration: Optional[Union[PlanarConfigurationValues, int]] = None
) -> np.ndarray:
    """Decode pixel data of an individual frame.

    Parameters
    ----------
    value: bytes
        Pixel data of a frame (potentially compressed in case
        of encapsulated format encoding, depending on the transfer syntax)
    transfer_syntax_uid: str
        Transfer Syntax UID
    rows: int
        Number of pixel rows in the frame
    columns: int
        Number of pixel columns in the frame
    samples_per_pixel: int
        Number of (color) samples per pixel
    bits_allocated: int
        Number of bits that need to be allocated per pixel sample
    bits_stored: int
        Number of bits that are required to store a pixel sample
    photometric_interpretation: Union[str, highdicom.PhotometricInterpretationValues]
        Photometric interpretation
    pixel_representation: Union[highdicom.PixelRepresentationValues, int, None], optional
        Whether pixel samples are represented as unsigned integers or
        2's complements
    planar_configuration: Union[highdicom.PlanarConfigurationValues, int, None], optional
        Whether color samples are encoded by pixel (``R1G1B1R2G2B2...``) or
        by plane (``R1R2...G1G2...B1B2...``).

    Returns
    -------
    numpy.ndarray
        Decoded pixel data

    Raises
    ------
    ValueError
        When transfer syntax is not supported.

    Note
    ----
    In case of color image frames, the `photometric_interpretation` parameter
    describes the color space of the **encoded** pixel data and data may be
    converted from the specified color space into RGB color space upon
    decoding.  For example, the JPEG codec generally converts pixels from RGB into
    YBR color space prior to compression to take advantage of the correlation
    between RGB color bands and improve compression efficiency. In case of an
    image data set with an encapsulated Pixel Data element containing JPEG
    compressed image frames, the value of the Photometric Interpretation
    element specifies the color space in which image frames were compressed.
    If `photometric_interpretation` specifies a YBR color space, then this
    function assumes that pixels were converted from RGB to YBR color space
    during encoding prior to JPEG compression and need to be converted back
    into RGB color space after JPEG decompression during decoding. If
    `photometric_interpretation` specifies an RGB color space, then the
    function assumes that no color space conversion was performed during
    encoding and therefore no conversion needs to be performed during decoding
    either. In both case, the function is supposed to return decoded pixel data
    of color image frames in RGB color space.

    """  # noqa: E501
    # The pydicom library does currently not support reading individual frames.
    # This hack creates a small dataset containing only a single frame, which
    # can then be decoded using the pydicom API.
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = UID(transfer_syntax_uid)
    ds = Dataset()
    ds.file_meta = file_meta
    ds.Rows = rows
    ds.Columns = columns
    ds.SamplesPerPixel = samples_per_pixel
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1

    pixel_representation = PixelRepresentationValues(
        pixel_representation
    ).value
    ds.PixelRepresentation = pixel_representation
    photometric_interpretation = PhotometricInterpretationValues(
        photometric_interpretation
    ).value
    ds.PhotometricInterpretation = photometric_interpretation
    if samples_per_pixel > 1:
        if planar_configuration is None:
            raise ValueError(
                'Planar configuration needs to be specified for decoding of '
                'color image frames.'
            )
        planar_configuration = PlanarConfigurationValues(
            planar_configuration
        ).value
        ds.PlanarConfiguration = planar_configuration

    if UID(file_meta.TransferSyntaxUID).is_encapsulated:
        ds.PixelData = encapsulate(frames=[value])
    else:
        ds.PixelData = value

    array = ds.pixel_array

    # In case of the JPEG baseline transfer syntax, the pixel_array property
    # does not convert the pixel data into the correct (or let's say expected)
    # color space after decompression.
    if (
        'YBR' in ds.PhotometricInterpretation and
        ds.SamplesPerPixel == 3 and
        transfer_syntax_uid == JPEGBaseline8Bit
    ):
        image = Image.fromarray(array, mode='YCbCr')
        image = image.convert(mode='RGB')
        array = np.asarray(image)

    return array
