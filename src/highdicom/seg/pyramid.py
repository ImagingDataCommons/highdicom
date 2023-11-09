"""Tools for constructing multi-resolution segmentation pyramids."""
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from pydicom import Dataset
from pydicom.uid import VLWholeSlideMicroscopyImageStorage

from highdicom.content import PixelMeasuresSequence
from highdicom.seg.sop import Segmentation
from highdicom.seg.enum import (
    SegmentationTypeValues,
)
from highdicom.seg.content import (
    SegmentDescription,
)
from highdicom.uid import UID


def create_segmentation_pyramid(
    source_images: Sequence[Dataset],
    pixel_arrays: Sequence[np.ndarray],
    segmentation_type: Union[str, SegmentationTypeValues],
    segment_descriptions: Sequence[SegmentDescription],
    series_instance_uid: str,
    series_number: int,
    manufacturer: str,
    manufacturer_model_name: str,
    software_versions: Union[str, Tuple[str]],
    device_serial_number: str,
    downsample_factors: Optional[Sequence[float]] = None,
    sop_instance_uids: Optional[List[str]] = None,
    pyramid_uid: Optional[str] = None,
    pyramid_label: Optional[str] = None,
    **kwargs: Any
) -> List[Segmentation]:
    """Construct a multi-resolution segmentation pyramid series.

    A multi-resolution pyramid represents the same segmentation array at
    multiple resolutions.

    This function handles multiple related scenarios:

    * Constructing a segmentation of a source image pyramid given a
      segmentation pixel array of the highest resolution source image.
      Highdicom performs the downsampling automatically to match the
      resolution of the other source images. For this case, pass multiple
      ``source_images`` and a single item in ``pixel_arrays``.
    * Constructing a segmentation of a source image pyramid given user-provided
      segmentation pixel arrays for each level in the source pyramid. For this
      case, pass multiple ``source_images`` and a matching number of
      ``pixel_arrays``.
    * Constructing a segmentation of a single source image given multiple
      user-provided downsampled segmentation pixel arrays. For this case, pass
      a single item in ``source_images``, and multiple items in
      ``pixel_arrays``).
    * Constructing a segmentation of a single source image and a single
      segmentation pixel array by downsampling by a given list of
      ``downsample_factors``. For this case, pass a single item in
      ``source_images``, a single item in ``pixel_arrays``, and a list of one
      or more desired ``downsample_factors``.

    In all cases, the items in both ``source_images`` and ``pixel_arrays``
    should be sorted in pyramid order from highest resolution (smallest
    spacing) to lowest resolution (largest spacing), and the pixel array
    in ``pixel_arrays[0]`` must be the segmentation of the source image in
    ``source_images[0]`` with spatial locations preserved (a one-to-one
    correspondence between pixels in the source image's total pixel matrix and
    the provided segmentation pixel array).

    In all cases, the provided pixel arrays should be total pixel matrices.
    Tiling is performed automatically.

    Parameters
    ----------
    source_images: Sequence[pydicom.Dataset]
        List of source images. If there are multiple source images, they should
        be from the same series and pyramid.
    pixel_arrays: Sequence[numpy.ndarray]
        List of segmentation pixel arrays. Each should be a total pixel matrix.
    segmentation_type: Union[str, highdicom.seg.SegmentationTypeValues]
        Type of segmentation, either ``"BINARY"`` or ``"FRACTIONAL"``
    segment_descriptions: Sequence[highdicom.seg.SegmentDescription]
        Description of each segment encoded in `pixel_array`. In the case of
        pixel arrays with multiple integer values, the segment description
        with the corresponding segment number is used to describe each segment.
    series_number: int
        Number of the output segmentation series.
    manufacturer: str
        Name of the manufacturer of the device (developer of the software)
        that creates the instance
    manufacturer_model_name: str
        Name of the device model (name of the software library or
        application) that creates the instance
    software_versions: Union[str, Tuple[str]]
        Version(s) of the software that creates the instance.
    device_serial_number: str
        Manufacturer's serial number of the device
    downsample_factors: Optional[Sequence[float]], optional
        Factors by which to downsample the pixel array to create each of the
        output segmentation objects. This should be provided if and only if a
        single source image and single pixel array are provided. Note that the
        original array is always used to create the first segmentation output,
        so the number of created segmententation instances is one greater than
        the number of items in this list. Items must be numbers greater than
        1 and sorted in ascending order. A downsampling factor of *n* implies
        that the output array is *1/n* time the size of input pixel array. For
        example a list ``[2, 4, 8]`` would be produce 4 output segmentation
        instances. The first is the same size as the original pixel array, the
        next is half the size, the next is a quarter of the size of the
        original, and the last is one eighth the size of the original.
        Output sizes are rounded to the nearest integer.
    series_instance_uid: Optional[str], optional
        UID of the output segmentation series. If not specified, UIDs are
        generated automatically using highdicom's prefix.
    sop_instance_uids: Optional[List[str]], optional
        SOP instance UIDS of the output instances. If not specified, UIDs are
        generated automatically using highdicom's prefix.
    pyramid_uid: Optional[str], optional
        UID for the output imaging pyramid. If not specified, a UID is generated
        using highdicom's prefix.
    pyramid_label: Optional[str], optional
        A human readable label for the output pyramid.
    **kwargs: Any
        Any further parameters are passed directly to the constructor of the
        :class:highdicom.seg.Segmentation object. However the following
        parameters are disallowed: ``instance_number``, ``sop_instance_uid``,
        ``plane_orientation``, ``plane_positions``, ``pixel_measures``,
        ``pixel_array``, ``tile_pixel_array``.

    Note
    ----
    Downsampling is performed via simple nearest neighbor interpolation. If
    more control is needed over the downsampling process (for example
    anti-aliasing), explicitly pass the downsampled arrays.

    """
    # Disallow duplicate items in kwargs
    kwarg_keys = set(kwargs.keys())
    disallowed_keys = {
        'instance_number',
        'sop_instance_uid',
        'plane_orientation',
        'plane_positions',
        'pixel_array',
        'tile_pixel_array',
    }
    error_keys = kwarg_keys & disallowed_keys
    if len(error_keys) > 0:
        raise TypeError(
            f'kwargs supplied to the create_segmentation_pyramid function '
            f'should not contain a value for parameter {error_keys[0]}.'
        )

    if pyramid_uid is None:
        pyramid_uid = UID()
    if series_instance_uid is None:
        series_instance_uid = UID()

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

    # Check that source images are WSI
    for im in source_images:
        if im.SOPClassUID != VLWholeSlideMicroscopyImageStorage:
            raise ValueError(
                'Source images must have IOD VLWholeSlideMicroscopyImageStorage'
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
                'decreasing resolution.'
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
        mask_image = Image.fromarray(pixel_arrays[0])

    all_segs = []

    # Work "up" pyramid from high to low resolution
    for output_level in range(n_outputs):
        if n_sources > 1:
            source_image = source_images[output_level]
        else:
            source_image = source_images[0]

        if n_pix_arrays > 1:
            pixel_array = pixel_arrays[output_level]
        else:
            need_resize = True
            if n_sources > 1:
                output_size = (
                    source_image.TotalPixelMatrixColumns,
                    source_image.TotalPixelMatrixRows
                )
            else:
                if output_level == 0:
                    pixel_array = pixel_arrays[0]
                    need_resize = False
                else:
                    f = downsample_factors[output_level - 1]
                    output_size = (
                        int(source_images[0].TotalPixelMatrixColumns / f),
                        int(source_images[0].TotalPixelMatrixRows / f)
                    )

            if need_resize:
                pixel_array = np.array(
                    mask_image.resize(output_size, Image.Resampling.NEAREST)
                )

        if n_sources == 1:
            source_pixel_measures = (
                source_image
                .SharedFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
            )
            src_pixel_spacing = source_pixel_measures.PixelSpacing
            src_slice_thickness = source_pixel_measures.SliceThickness
            row_spacing = (
                src_pixel_spacing[0] *
                (pixel_arrays[0].shape[0] / pixel_array.shape[0])
            )
            column_spacing = (
                src_pixel_spacing[1] *
                (pixel_arrays[0].shape[1] / pixel_array.shape[1])
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

        # Create the output segmentation
        seg = Segmentation(
            source_images=[source_image],
            pixel_array=pixel_array,
            segmentation_type=segmentation_type,
            segment_descriptions=segment_descriptions,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=output_level + 1,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            pyramid_uid=pyramid_uid,
            pyramid_label=pyramid_label,
            tile_pixel_array=True,
            plane_orientation=None,
            plane_positions=None,
            pixel_measures=pixel_measures,
            **kwargs,
        )

        all_segs.append(seg)

    return all_segs
