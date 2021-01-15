import logging
from io import BytesIO

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from pydicom.pixel_data_handlers.rle_handler import rle_encode_frame
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline,
    RLELossless,
)


logger = logging.getLogger(__name__)


def encode_frame(
    array: np.ndarray,
    metadata: Dataset,
) -> bytes:
    """Encodes pixel data of an individual frame.

    Parameters
    ----------
    array: numpy.ndarray
        Pixel data in form of an array with dimensions
        (Rows x Columns x SamplesPerPixel) in case of a color image and
        (Rows x Columns) in case of a monochrome image
    metadata: pydicom.dataset.Dataset
        Metadata of the corresponding image dataset

    Returns
    -------
    bytes
        Pixel data (potentially compressed in case of encapsulated format
        encoding, depending on the transfer snytax)

    Raises
    ------
    AttributeError
        When required attribute is missing in `metadata`.
    ValueError
        When transfer syntax is not supported.

    """
    transfer_syntax_uid = metadata.file_meta.TransferSyntaxUID
    required_attributes = {
        'Rows',
        'Columns',
        'BitsAllocated',
        'BitsStored',
        'SamplesPerPixel',
    }
    for attr in required_attributes:
        if not hasattr(metadata, attr):
            raise AttributeError(
                'Cannot encode frame. '
                f'Image metadata is missing required attribute "{attr}".'
            )
    bits_allocated = metadata.BitsAllocated
    rows = metadata.Rows
    cols = metadata.Columns
    samples_per_pixel = metadata.SamplesPerPixel

    uncompressed_transfer_syntaxes = {
        ExplicitVRLittleEndian,
        ImplicitVRLittleEndian,
    }
    compressed_transfer_syntaxes = {
        JPEGBaseline,
        JPEG2000Lossless,
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
            JPEGBaseline: (
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
        }
        if transfer_syntax_uid in compression_lut.keys():
            image_format, kwargs = compression_lut[transfer_syntax_uid]
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
    metadata: Dataset,
) -> np.ndarray:
    """Decodes pixel data of an individual frame.

    Parameters
    ----------
    value: bytes
        Pixel data of a frame (potentially compressed in case
        of encapsulated format encoding, depending on the transfer syntax)
    metadata: pydicom.dataset.Dataset
        Metadata of the corresponding image dataset

    Returns
    -------
    numpy.ndarray
        Decoded pixel data

    Raises
    ------
    AttributeError
        When required attribute is missing in `metadata`.
    ValueError
        When transfer syntax is not supported.

    """
    # The pydicom library does currently not support reading individual frames.
    # This hack creates a small dataset containing only a single frame, which
    # can then be decoded using the pydicom API.
    ds = Dataset()
    ds.file_meta = metadata.file_meta
    required_attributes = {
        'Rows',
        'Columns',
        'BitsAllocated',
        'BitsStored',
        'SamplesPerPixel',
        'PhotometricInterpretation',
        'PixelRepresentation',
    }
    for attr in required_attributes:
        try:
            setattr(ds, attr, getattr(metadata, attr))
        except AttributeError:
            raise AttributeError(
                'Cannot decode frame. '
                f'Image metadata is missing required attribute "{attr}".'
            )

    if metadata.SamplesPerPixel > 1:
        attr = 'PlanarConfiguration'
        try:
            setattr(ds, attr, getattr(metadata, attr))
        except AttributeError:
            raise AttributeError(
                'Cannot decode frame. '
                f'Image metadata is missing required attribute "{attr}".'
            )

    transfer_syntax_uid = metadata.file_meta.TransferSyntaxUID

    if transfer_syntax_uid.is_encapsulated:
        if (transfer_syntax_uid == JPEGBaseline and
                metadata.PhotometricInterpretation == 'RGB'):
            # RGB color images, which were not transformed into YCbCr color
            # space upon JPEG compression, need to be handled separately.
            # Pillow assumes that images were transformed into YCbCr color
            # space prior to JPEG compression. However, with photometric
            # interpretation RGB, no color transformation was performed.
            # Setting the value of "mode" to YCbCr signals Pillow to not
            # apply any color transformation upon decompression.
            image = Image.open(BytesIO(value))
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
        else:
            ds.PixelData = encapsulate(frames=[value])
    else:
        ds.PixelData = value

    return ds.pixel_array
