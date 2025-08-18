"""Tools for constructing multi-resolution segmentation pyramids."""
from typing import Any
from collections.abc import Sequence

import numpy as np
from pydicom import Dataset

from highdicom.enum import InterpolationMethods
from highdicom._pyramid import iter_derived_pyramid_levels
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
    segmentation_type: str | SegmentationTypeValues,
    segment_descriptions: Sequence[SegmentDescription],
    series_instance_uid: str | None,
    series_number: int,
    manufacturer: str,
    manufacturer_model_name: str,
    software_versions: str | tuple[str],
    device_serial_number: str,
    downsample_factors: Sequence[float] | None = None,
    sop_instance_uids: list[str] | None = None,
    pyramid_uid: str | None = None,
    pyramid_label: str | None = None,
    **kwargs: Any
) -> list[Segmentation]:
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
        List of segmentation pixel arrays. Each should be a total pixel matrix,
        i.e. have shape (rows, columns), (1, rows, columns), or (1, rows,
        columns, segments). Otherwise all options supported by the constructor
        of :class:`highdicom.seg.Segmentation` are permitted.
    segmentation_type: Union[str, highdicom.seg.SegmentationTypeValues]
        Type of segmentation, either ``"BINARY"`` or ``"FRACTIONAL"``
    segment_descriptions: Sequence[highdicom.seg.SegmentDescription]
        Description of each segment encoded in `pixel_array`. In the case of
        pixel arrays with multiple integer values, the segment description
        with the corresponding segment number is used to describe each segment.
    series_instance_uid: str | None
        UID for the output segmentation series. If ``None``, a UID will be
        generated using highdicom's prefix.
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
        :class:`highdicom.seg.Segmentation` object. However the following
        parameters are disallowed: ``instance_number``, ``sop_instance_uid``,
        ``plane_orientation``, ``plane_positions``, ``pixel_measures``,
        ``pixel_array``, ``tile_pixel_array``.

    Note
    ----
    Downsampling is performed via simple nearest neighbor interpolation (for
    ``BINARY`` segmentations) or bi-linear interpolation (for ``FRACTIONAL``
    segmentations). If more control is needed over the downsampling process
    (for example anti-aliasing), explicitly pass the downsampled arrays.

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
            f'should not contain a value for parameter {list(error_keys)[0]}.'
        )

    segmentation_type = SegmentationTypeValues(segmentation_type)

    if pyramid_uid is None:
        pyramid_uid = UID()
    if series_instance_uid is None:
        series_instance_uid = UID()

    dtype = pixel_arrays[0].dtype
    if dtype in (np.bool_, np.uint8, np.uint16):
        interpolator = InterpolationMethods.NEAREST
    elif dtype in (np.float32, np.float64):
        if segmentation_type == SegmentationTypeValues.FRACTIONAL:
            interpolator = InterpolationMethods.LINEAR
        else:
            # This is a floating point image that will ultimately be treated as
            # binary
            interpolator = InterpolationMethods.NEAREST
    else:
        raise TypeError('Pixel array has an invalid data type.')

    all_segs = []

    for (
        source_image,
        pixel_array,
        pixel_measures,
        instance_number,
        sop_instance_uid,
    ) in iter_derived_pyramid_levels(
        source_images=source_images,
        pixel_arrays=pixel_arrays,
        interpolator=interpolator,
        downsample_factors=downsample_factors,
        sop_instance_uids=sop_instance_uids,
    ):
        # Create the output segmentation
        seg = Segmentation(
            source_images=[source_image],
            pixel_array=pixel_array,
            segmentation_type=segmentation_type,
            segment_descriptions=segment_descriptions,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
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
