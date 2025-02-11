import logging
from io import BytesIO
from typing import cast

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.pixels.utils import pack_bits, unpack_bits
from pydicom.pixels.encoders.base import get_encoder
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEG2000,
    JPEGBaseline8Bit,
    JPEGLSLossless,
    JPEGLSNearLossless,
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
    photometric_interpretation: PhotometricInterpretationValues | str,
    pixel_representation: PixelRepresentationValues | int = 0,
    planar_configuration: PlanarConfigurationValues | int | None = None
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
    photometric_interpretation: Union[PhotometricInterpretationValues, str]
        Photometric interpretation that will be used to store data.  Usually,
        this will match the photometric interpretation of the input pixel
        array, however for ``"JPEGBaseline8Bit"``, ``"JPEG2000"``, and
        ``"JPEG2000Lossless"`` transfer syntaxes with color images, the pixel
        data must be passed in in RGB format and will be converted and stored
        as ``"YBR_FULL_422"`` (``"JPEGBaseline8Bit"``), ``"YBR_ICT"``
        (``"JPEG2000"``), or ``"YBR_RCT"`` (``"JPEG2000Lossless"``). In these
        cases the values of photometric metric passed must match those given
        above.
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
        format encoding, depending on the transfer syntax)

    Raises
    ------
    ValueError
        When ``transfer_syntax_uid`` is not supported or when
        ``planar_configuration`` is missing in case of a color image frame.

    Note
    ----
    In case of color image frames, the ``photometric_interpretation`` parameter
    describes the color space of the **encoded** pixel data and data may be
    converted from RGB color space into the specified color space upon
    encoding.  For example, the JPEG codec converts pixels from RGB into
    YBR color space prior to compression to take advantage of the correlation
    between RGB color bands and improve compression efficiency. Therefore,
    pixels are supposed to be provided via ``array`` in RGB color space, but
    ``photometric_interpretation``` needs to specify a YBR color space.

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
        JPEG2000,
        JPEG2000Lossless,
        JPEGLSLossless,
        JPEGLSNearLossless,
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
        allowable_pis = {
            1: ['MONOCHROME1', 'MONOCHROME2', 'PALETTE COLOR'],
            3: ['RGB', 'YBR_FULL'],
        }[samples_per_pixel]
        if photometric_interpretation not in allowable_pis:
            raise ValueError(
                'Photometric_interpretation of '
                f"'{photometric_interpretation}' "
                f'not supported for samples_per_pixel={samples_per_pixel}.'
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

    elif transfer_syntax_uid == JPEGBaseline8Bit:
        if samples_per_pixel == 1:
            if planar_configuration is not None:
                raise ValueError(
                    'Planar configuration must be absent for encoding of '
                    'monochrome image frames with JPEG Baseline codec.'
                )
            if photometric_interpretation not in (
                    'MONOCHROME1', 'MONOCHROME2', 'PALETTE COLOR'
                ):
                raise ValueError(
                    'Photometric intpretation must be either "MONOCHROME1", '
                    '"MONOCHROME2", or "PALETTE COLOR" for encoding of '
                    'monochrome image frames with JPEG Baseline codec.'
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

        # Pydicom does not have an encoder for JPEGBaseline8Bit so
        # we do this manually
        if samples_per_pixel == 3:
            image = Image.fromarray(array, mode='RGB')
        else:
            image = Image.fromarray(array)
        with BytesIO() as buf:
            image.save(buf, format='jpeg', quality=95)
            data = buf.getvalue()
    else:
        name = {
            JPEG2000: "JPEG 2000",
            JPEG2000Lossless: "Lossless JPEG 2000",
            JPEGLSLossless: "Lossless JPEG-LS",
            JPEGLSNearLossless: "Near-Lossless JPEG-LS",
            RLELossless: "RLE Lossless",
        }[transfer_syntax_uid]

        kwargs = {}

        if samples_per_pixel not in (1, 3):
            raise ValueError(
                'Samples per pixel must be 1 or 3 for '
                f'encoding of image frames with {name} codec.'
            )

        if transfer_syntax_uid != RLELossless:
            if pixel_representation != 0:
                raise ValueError(
                    'Pixel representation must be 0 for '
                    f'encoding of image frames with {name} codec.'
                )
            if samples_per_pixel == 1:
                if planar_configuration is not None:
                    raise ValueError(
                        'Planar configuration must be absent for encoding of '
                        f'monochrome image frames with {name} codec.'
                    )
                if photometric_interpretation not in (
                        'MONOCHROME1', 'MONOCHROME2', 'PALETTE COLOR'
                    ):
                    raise ValueError(
                        'Photometric intpretation must be either '
                        '"MONOCHROME1", "MONOCHROME2", or "PALETTE COLOR" '
                        'for encoding of monochrome image frames with '
                        f'{name} codec.'
                    )
                if transfer_syntax_uid == JPEG2000Lossless:
                    if bits_allocated not in (1, 8, 16):
                        raise ValueError(
                            'Bits Allocated must be 1, 8, or 16 for encoding '
                            f'of monochrome image frames with with {name} '
                            'codec.'
                        )
                else:
                    if bits_allocated not in (8, 16):
                        raise ValueError(
                            'Bits Allocated must be 8 or 16 for encoding of '
                            f'monochrome image frames with with {name} codec.'
                        )
            elif samples_per_pixel == 3:
                if planar_configuration != 0:
                    raise ValueError(
                        'Planar configuration must be 0 for encoding of '
                        f'color image frames with {name} codec.'
                    )
                if bits_allocated not in (8, 16):
                    raise ValueError(
                        'Bits Allocated must be 8 or 16 for encoding of '
                        f'color image frames with {name} codec.'
                    )

                required_pi = {
                    JPEG2000: PhotometricInterpretationValues.YBR_ICT,
                    JPEG2000Lossless: PhotometricInterpretationValues.YBR_RCT,
                    JPEGLSLossless: PhotometricInterpretationValues.RGB,
                    JPEGLSNearLossless: PhotometricInterpretationValues.RGB,
                }[transfer_syntax_uid]

                if photometric_interpretation != required_pi.value:
                    raise ValueError(
                        f'Photometric interpretation must be '
                        f'"{required_pi.value}" for encoding of color image '
                        f'frames with {name} codec.'
                    )

        if transfer_syntax_uid == JPEG2000:
            kwargs = {'j2k_psnr': [100]}

        if transfer_syntax_uid in (JPEG2000, JPEG2000Lossless):
            # This seems to be an openjpeg limitation
            if array.shape[0] < 32 or array.shape[1] < 32:
                raise ValueError(
                    'Images smaller than 32 pixels along both dimensions '
                    f'cannot be encoded with {name} codec.'
                )

        if transfer_syntax_uid == JPEG2000Lossless and bits_allocated == 1:
            # Single bit JPEG2000 compression. Pydicom doesn't (yet) support
            # this case
            if array.dtype != bool:
                if array.max() > 1:
                    raise ValueError(
                        'Array must contain only 0 and 1 for bits_allocated = 1'
                    )
                array = array.astype(bool)

            try:
                from openjpeg.utils import encode_array
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Highdicom requires the pylibjpeg-openjpeg package to "
                    "compress frames using the JPEG2000Lossless transfer "
                    "syntax."
                )

            data = encode_array(
                array,
                bits_stored=1,
                photometric_interpretation=2,
                use_mct=False,
            )
        else:
            encoder = get_encoder(transfer_syntax_uid)

            data = encoder.encode(
                array,
                rows=array.shape[0],
                columns=array.shape[1],
                samples_per_pixel=samples_per_pixel,
                number_of_frames=1,
                bits_allocated=bits_allocated,
                bits_stored=bits_stored,
                photometric_interpretation=photometric_interpretation,
                pixel_representation=pixel_representation,
                planar_configuration=planar_configuration,
                **kwargs,
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
    photometric_interpretation: PhotometricInterpretationValues | str,
    pixel_representation: PixelRepresentationValues | int = 0,
    planar_configuration: PlanarConfigurationValues | int | None = None,
    index: int = 0,
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
    index: int, optional
        The (zero-based) index of the frame in the original dataset. This is
        only required situation: when the bits allocated is 1, the transfer
        syntax is not encapsulated (i.e. is native) and the number of pixels
        per frame is not a multiple of 8. In this case, the index is required
        to know how many bits need to be stripped from the start and/or end of
        the byte array. In all other situations, this parameter is not
        required and will have no effect (since decoding a frame does not
        depend on the index of the frame).

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
    is_encapsulated = UID(transfer_syntax_uid).is_encapsulated

    # This is a special case since there may be extra bits that need stripping
    # from the start and/or end
    if bits_allocated == 1 and not is_encapsulated:
        unpacked_frame = cast(np.ndarray, unpack_bits(value))
        n_pixels = (rows * columns * samples_per_pixel)
        pixel_offset = int(((index * n_pixels / 8) % 1) * 8)
        pixel_array = unpacked_frame[pixel_offset:pixel_offset + n_pixels]
        return pixel_array.reshape(rows, columns)

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

    if is_encapsulated:
        ds.PixelData = encapsulate(frames=[value])
    else:
        ds.PixelData = value

    array = ds.pixel_array

    return array
