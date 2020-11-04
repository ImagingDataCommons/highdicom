import logging
from io import BytesIO

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from pydicom.pixel_data_handlers.rle_handler import rle_encode_frame
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEGBaseline,
    JPEG2000Lossless,
    RLELossless,
)


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
    ValueError
        When transfer syntax is not supported.

    """
    transfer_syntax_uid = metadata.file_meta.TransferSyntaxUID
    bits_allocated = metadata.BitsAllocated

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
