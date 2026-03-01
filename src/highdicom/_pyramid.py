"""Tools for constructing multi-resolution pyramids."""
from typing import Any
from collections.abc import Generator, Sequence

import numpy as np
from PIL import Image
from pydicom import Dataset
from pydicom.uid import VLWholeSlideMicroscopyImageStorage

from highdicom.content import PixelMeasuresSequence
from highdicom.enum import InterpolationMethods
from highdicom.uid import UID


def iter_derived_pyramid_levels(
    source_images: Sequence[Dataset],
    pixel_arrays: Sequence[np.ndarray],
    *,
    interpolator: InterpolationMethods | str = InterpolationMethods.NEAREST,
    downsample_factors: Sequence[float] | None = None,
    sop_instance_uids: list[str] | None = None,
) -> Generator[
        tuple[
            Dataset,
            np.ndarray,
            PixelMeasuresSequence,
            int,
            str,
        ],
        None,
        None,
    ]:
    """Create levels of derived multi-resolution pyramid with downsampling.

    A multi-resolution pyramid represents the same derived image array at
    multiple resolutions. This function is a general-purpose function for
    iterating through downsampled versions of an image for inclusion in a
    derived pyramid series.

    This function handles multiple related scenarios:

    * Constructing a derived pyramid of a source image pyramid given a
      derived pixel array of the highest resolution source image.
      Highdicom performs the downsampling automatically to match the
      resolution of the other source images. For this case, pass multiple
      ``source_images`` and a single item in ``pixel_arrays``.
    * Constructing a derived pyramid of a source image pyramid given
      user-provided derived pixel arrays for each level in the source pyramid.
      For this case, pass multiple ``source_images`` and a matching number of
      ``pixel_arrays``.
    * Constructing a derived pyramid of a single source image given multiple
      user-provided downsampled derived pixel arrays. For this case, pass a
      single item in ``source_images``, and multiple items in
      ``pixel_arrays``).
    * Constructing a derived pyramid of a single source image and a single
      derived pixel array by downsampling by a given list of
      ``downsample_factors``. For this case, pass a single item in
      ``source_images``, a single item in ``pixel_arrays``, and a list of one
      or more desired ``downsample_factors``.

    In all cases, the items in both ``source_images`` and ``pixel_arrays``
    should be sorted in pyramid order from highest resolution (smallest
    spacing) to lowest resolution (largest spacing), and the pixel array
    in ``pixel_arrays[0]`` must be derived from the source image in
    ``source_images[0]`` with spatial locations preserved (a one-to-one
    correspondence between pixels in the source image's total pixel matrix and
    the provided derived pixel array).

    In all cases, the provided pixel arrays should be total pixel matrices.
    Tiling is performed automatically.

    Parameters
    ----------
    source_images: Sequence[pydicom.Dataset]
        List of source images. If there are multiple source images, they should
        be from the same series and pyramid.
    pixel_arrays: Sequence[numpy.ndarray]
        List of derived pixel arrays. Each should be a total pixel matrix, i.e.
        have shape (rows, columns), (1, rows, columns), or (1, rows, columns,
        segments/channels).
    interpolator: highdicom.InterpolationMethods | str, optional
        Interpolation method used for the downsampling operations.
    downsample_factors: Optional[Sequence[float]], optional
        Factors by which to downsample the pixel array to create each of the
        output levels. This should be provided if and only if a single source
        image and single pixel array are provided. Note that the original array
        is always used to create the first output, so the number of created
        levels is one greater than the number of items in this list. Items must
        be numbers greater than 1 and sorted in ascending order. A downsampling
        factor of *n* implies that the output array is *1/n* time the size of
        input pixel array. For example a list ``[2, 4, 8]`` would be produce 4
        output levels. The first is the same size as the original pixel array,
        the next is half the size, the next is a quarter of the size of the
        original, and the last is one eighth the size of the original. Output
        sizes are rounded to the nearest integer.
    sop_instance_uids: Optional[List[str]], optional
        SOP instance UIDS of the output instances. If not specified, UIDs are
        generated automatically using highdicom's prefix.

    Yields
    ------
    source_image: 
        Dataset to use as the source image at this level. This will be one of
        the inputs provided to ``source_images``.
    pixel_array: numpy.ndarray
        Pixel array at this level. This will either be one of the inputs
        provided to ``pixel_arrays`` or an array created by downsampling the
        single provided pixel array.
    pixel_measures: highdicom.PixelMeasuresSequence
        Pixel measures object describing the pixel spacing of the pixel array
        at this level.
    instance_number: int
        Instance number to include in the instance at this level.
    sop_instance_uid: str
        SOP Instance UID to include in the instance at this level.

    """
    n_sources = len(source_images)
    n_pix_arrays = len(pixel_arrays)

    if n_sources == 0:
        raise ValueError(
            'Argument "source_images" must not be empty.'
        )
    if n_pix_arrays == 0:
        raise ValueError(
            'Argument "pixel_arrays" must not be empty.'
        )

    if n_sources == 1 and n_pix_arrays == 1:
        if downsample_factors is None:
            raise TypeError(
                'Argument "downsample_factors" must be provided when providing '
                'only a single source image and pixel array.'
            )
        if len(downsample_factors) < 1:
            raise ValueError('Argument "downsample_factors" may not be empty.')
        if any(f <= 1.0 for f in downsample_factors):
            raise ValueError(
                'All items in "downsample_factors" must be greater than 1.'
            )
        if len(downsample_factors) > 1:
            if any(
                z1 > z2 for z1, z2 in zip(
                    downsample_factors[:-1],
                    downsample_factors[1:]
                )
            ):
                raise ValueError(
                    'Items in argument "downsample_factors" must be sorted in '
                    'ascending order.'
                )
        n_outputs = len(downsample_factors) + 1  # including original
    else:
        if downsample_factors is not None:
            raise TypeError(
                'Argument "downsample_factors" must not be provided when  '
                'multiple source images or pixel arrays are provided.'
            )
        if n_sources > 1 and n_pix_arrays > 1:
            if n_sources != n_pix_arrays:
                raise ValueError(
                    'If providing multiple source images and multiple pixel '
                    'arrays, the number of items in the two lists must match.'
                )
            n_outputs = n_sources
        else:
            # Either n_sources > 1 or n_pix_arrays > 1 but not both
            n_outputs = max(n_sources, n_pix_arrays)

    if sop_instance_uids is not None:
        if len(sop_instance_uids) != n_outputs:
            raise ValueError(
                'Number of specified SOP Instance UIDs does not match number '
                'of output images.'
            )

    # Check that source images are WSI
    for im in source_images:
        if im.SOPClassUID != VLWholeSlideMicroscopyImageStorage:
            raise ValueError(
                'Source images must have IOD VLWholeSlideMicroscopyImage'
            )

    # Check the source images are appropriately ordered
    for index in range(1, len(source_images)):
        r0 = source_images[index - 1].TotalPixelMatrixRows
        c0 = source_images[index - 1].TotalPixelMatrixColumns
        r1 = source_images[index].TotalPixelMatrixRows
        c1 = source_images[index].TotalPixelMatrixColumns

        if r0 <= r1 or c0 <= c1:
            raise ValueError(
                'Items in argument "source_images" must be strictly ordered in '
                'decreasing resolution.'
            )

    # Check that the source images are from the same series and pyramid
    if len(source_images) > 1:
        series_uid = source_images[0].SeriesInstanceUID
        if not all(
            dcm.SeriesInstanceUID == series_uid
            for dcm in source_images[1:]
        ):
            raise ValueError(
                'All source images should belong to the same series.'
            )
        if not all(hasattr(dcm, 'PyramidUID') for dcm in source_images):
            raise ValueError(
                'All source images should belong to the same pyramid '
                '(share a Pyramid UID).'
            )
        pyramid_uid = source_images[0].PyramidUID
        if not all(
            dcm.PyramidUID == pyramid_uid
            for dcm in source_images[1:]
        ):
            raise ValueError(
                'All source images should belong to the same pyramid '
                '(share a Pyramid UID).'
            )

    # Check that pixel arrays have an appropriate shape
    if len({p.ndim for p in pixel_arrays}) != 1:
        raise ValueError(
            'Each item of argument "pixel_arrays" must have the same number of '
            'dimensions.'
        )
    if pixel_arrays[0].ndim == 4:
        n_channels = pixel_arrays[0].shape[3]
    else:
        n_channels = None

    # Map the highdicom interpolation methods enum to value used by Pillow
    interpolator = InterpolationMethods(interpolator)
    resampler = {
        InterpolationMethods.NEAREST: Image.Resampling.NEAREST,
        InterpolationMethods.LINEAR: Image.Resampling.BILINEAR,
    }[interpolator]

    # Checks on consistency of the pixel arrays
    dtype = pixel_arrays[0].dtype
    for pixel_array in pixel_arrays:
        if pixel_array.ndim not in (2, 3, 4):
            raise ValueError(
                'Each item of argument "pixel_arrays" must be a NumPy array '
                'with 2, 3, or 4 dimensions.'
            )
        if pixel_array.ndim > 2 and pixel_array.shape[0] != 1:
            raise ValueError(
                'Each item of argument "pixel_arrays" must contain a single '
                'frame, with a size of 1 along dimension 0.'
            )
        if pixel_array.dtype != dtype:
            raise TypeError(
                'Each item of argument "pixel_arrays" must have '
                'the same datatype.'
            )
        if pixel_array.ndim == 4:
            if pixel_array.shape[3] != n_channels:
                raise ValueError(
                    'Each item of argument "pixel_arrays" must have '
                    'the same shape down axis 3.'
                )

    # Check the pixel arrays are appropriately ordered
    for index in range(1, len(pixel_arrays)):
        arr0 = pixel_arrays[index - 1]
        arr1 = pixel_arrays[index]

        if arr0.ndim == 2:
            r0 = arr0.shape[:2]
            c0 = arr0.shape[:2]
        else:
            r0 = arr0.shape[1:3]
            c0 = arr0.shape[1:3]

        if arr1.ndim == 2:
            r1 = arr1.shape[:2]
            c1 = arr1.shape[:2]
        else:
            r1 = arr1.shape[1:3]
            c1 = arr1.shape[1:3]

        if r0 <= r1 or c0 <= c1:
            raise ValueError(
                'Items in argument "pixel_arrays" must be strictly ordered in '
                'decreasing size.'
            )

    # Check that input dimensions match
    for index, (source_image, pixel_array) in enumerate(
        zip(source_images, pixel_arrays)
    ):
        src_shape = (
            source_image.TotalPixelMatrixRows,
            source_image.TotalPixelMatrixColumns
        )
        pix_shape = (
            pixel_array.shape[1:3] if pixel_array.ndim > 2
            else pixel_array.shape
        )
        if pix_shape != src_shape:
            raise ValueError(
                "The shape of each provided pixel array must match the shape "
                "of the total pixel matrix of the corresponding source image. "
                f"Got pixel array of shape {pix_shape} for a source image of "
                f"shape {src_shape} at index {index}."
            )

    if n_pix_arrays == 1:
        # Create a pillow image for use later with resizing
        if pixel_arrays[0].ndim == 2:
            pil_images = [Image.fromarray(pixel_arrays[0])]
        elif pixel_arrays[0].ndim == 3:
            # Remove frame dimension before casting
            pil_images = [Image.fromarray(pixel_arrays[0][0])]
        else:  # ndim = 4
            # One "Image" for each channel
            pil_images = [
                Image.fromarray(pixel_arrays[0][0, :, :, i])
                for i in range(pixel_arrays[0].shape[3])
            ]

    # Work "up" pyramid from high to low resolution
    for output_level in range(n_outputs):
        if n_sources > 1:
            source_image = source_images[output_level]
        else:
            source_image = source_images[0]

        if n_pix_arrays > 1:
            pixel_array = pixel_arrays[output_level]
        else:
            if output_level == 0:
                pixel_array = pixel_arrays[0]
            else:
                if n_sources > 1:
                    output_size = (
                        source_image.TotalPixelMatrixColumns,
                        source_image.TotalPixelMatrixRows
                    )
                else:
                    f = downsample_factors[output_level - 1]
                    output_size = (
                        int(source_images[0].TotalPixelMatrixColumns / f),
                        int(source_images[0].TotalPixelMatrixRows / f)
                    )

                # Resize each channel individually
                resized_images = [
                    np.array(im.resize(output_size, resampler))
                    for im in pil_images
                ]
                if len(resized_images) > 1:
                    pixel_array = np.stack(resized_images, axis=-1)[None]
                else:
                    pixel_array = resized_images[0]

        # Standardize shape of pixel array to include singleton frame dimension
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[None]

        if n_sources == 1:
            source_pixel_measures = (
                source_image
                .SharedFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
            )
            src_pixel_spacing = source_pixel_measures.PixelSpacing
            src_slice_thickness = source_pixel_measures.SliceThickness

            if pixel_arrays[0].ndim == 2:
                # No frame dimension
                orig_n_rows = pixel_arrays[0].shape[0]
                orig_n_cols = pixel_arrays[0].shape[1]
            else:
                # Ignore 0th frame dimension
                orig_n_rows = pixel_arrays[0].shape[1]
                orig_n_cols = pixel_arrays[0].shape[2]

            row_spacing = (
                src_pixel_spacing[0] *
                (orig_n_rows / pixel_array.shape[1])
            )
            column_spacing = (
                src_pixel_spacing[1] *
                (orig_n_cols / pixel_array.shape[2])
            )
            pixel_measures = PixelMeasuresSequence(
                pixel_spacing=(row_spacing, column_spacing),
                slice_thickness=src_slice_thickness
            )
        else:
            # This will be copied from the source image
            pixel_measures = None

        if sop_instance_uids is None:
            sop_instance_uid = UID()
        else:
            sop_instance_uid = sop_instance_uids[output_level]

        yield (
            source_image,
            pixel_array,
            pixel_measures,
            output_level + 1,  # instance number
            sop_instance_uid,
        )
