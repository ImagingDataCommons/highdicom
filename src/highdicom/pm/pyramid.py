"""Tools for constructing multi-resolution parametric map pyramids."""
import datetime
from typing import Any
from collections.abc import Sequence

import numpy as np
from pydicom import Dataset

from highdicom.content import VOILUTTransformation
from highdicom.enum import InterpolationMethods
from highdicom._pyramid import iter_derived_pyramid_levels
from highdicom.pm.sop import ParametricMap
from highdicom.pm.content import (
    RealWorldValueMapping,
)
from highdicom.uid import UID


def create_parametric_map_pyramid(
    source_images: Sequence[Dataset],
    pixel_arrays: Sequence[np.ndarray],
    interpolator: InterpolationMethods | str,
    series_instance_uid: str | None,
    series_number: int,
    manufacturer: str,
    manufacturer_model_name: str,
    software_versions: str | tuple[str],
    device_serial_number: str,
    contains_recognizable_visual_features: bool,
    real_world_value_mappings: (
        Sequence[RealWorldValueMapping] |
        Sequence[Sequence[RealWorldValueMapping]]
    ),
    voi_lut_transformations: (
        Sequence[VOILUTTransformation] | None
    ) = None,
    downsample_factors: Sequence[float] | None = None,
    sop_instance_uids: list[str] | None = None,
    pyramid_uid: str | None = None,
    pyramid_label: str | None = None,
    **kwargs: Any
) -> list[ParametricMap]:
    """Construct a multi-resolution parametric map pyramid series.

    A multi-resolution pyramid represents the same parametric map array at
    multiple resolutions.

    This function handles multiple related scenarios:

    * Constructing a parametric map of a source image pyramid given a
      parametric map pixel array of the highest resolution source image.
      Highdicom performs the downsampling automatically to match the resolution
      of the other source images. For this case, pass multiple
      ``source_images`` and a single item in ``pixel_arrays``.
    * Constructing a parametric map of a source image pyramid given
      user-provided parametric map pixel arrays for each level in the source
      pyramid. For this case, pass multiple ``source_images`` and a matching
      number of ``pixel_arrays``.
    * Constructing a parametric map of a single source image given multiple
      user-provided downsampled parametric map pixel arrays. For this case,
      pass a single item in ``source_images``, and multiple items in
      ``pixel_arrays``).
    * Constructing a parametric map of a single source image and a single
      parametric map pixel array by downsampling by a given list of
      ``downsample_factors``. For this case, pass a single item in
      ``source_images``, a single item in ``pixel_arrays``, and a list of one
      or more desired ``downsample_factors``.

    In all cases, the items in both ``source_images`` and ``pixel_arrays``
    should be sorted in pyramid order from highest resolution (smallest
    spacing) to lowest resolution (largest spacing), and the pixel array
    in ``pixel_arrays[0]`` must be the parametric map of the source image in
    ``source_images[0]`` with spatial locations preserved (a one-to-one
    correspondence between pixels in the source image's total pixel matrix and
    the provided parametric map pixel array).

    In all cases, the provided pixel arrays should be total pixel matrices.
    Tiling is performed automatically.

    Parameters
    ----------
    source_images: Sequence[pydicom.Dataset]
        List of source images. If there are multiple source images, they should
        be from the same series and pyramid.
    pixel_arrays: Sequence[numpy.ndarray]
        List of parametric maps pixel arrays. Each should be a total pixel
        matrix, i.e. have shape (rows, columns), (1, rows, columns), or (1,
        rows, columns, channels). Otherwise all options supported by the
        constructor of :class:`highdicom.pm.ParametricMap` are permitted.
    interpolator: highdicom.InterpolationMethods | str, optional
        Interpolation method used for the downsampling operations.
    series_instance_uid: str | None
        UID for the output parametric series. If ``None``, a UID will be
        generated using highdicom's prefix.
    series_number: int
        Number of the output parametric map series.
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
    contains_recognizable_visual_features: bool
        Whether the image contains recognizable visible features of the
        patient
    real_world_value_mappings: Union[Sequence[highdicom.pm.RealWorldValueMapping], Sequence[Sequence[highdicom.pm.RealWorldValueMapping]]
        Descriptions of how stored values map to real-world values. Each
        channel encoded in each item of ``pixel_arrays`` shall be described
        with one or more real-world value mappings. Multiple mappings might be
        used for different representations such as log versus linear scales or
        for different representations in different units. If `pixel_array` is a
        2D or 3D array and only one channel exists at each spatial image
        position, then one or more real-world value mappings shall be provided
        in a flat sequence. If `pixel_array` is a 4D array and multiple
        channels exist at each spatial image position, then one or more
        mappings shall be provided for each channel in a nested sequence of
        length ``m``, where ``m`` shall match the channel dimension of each
        item of ``pixel_arrays``.

        In some situations the mapping may be difficult to describe (e.g., in
        case of a transformation performed by a deep convolutional neural
        network). The real-world value mapping may then simply describe an
        identity function that maps stored values to unit-less real-world
        values.
    voi_lut_transformations: Sequence[highdicom.VOILUTTransformation] | None, optional
        One or more VOI transformations that describe a pixel transformation to
        apply to frames.
    downsample_factors: Optional[Sequence[float]], optional
        Factors by which to downsample the pixel array to create each of the
        output parametric map objects. This should be provided if and only if a
        single source image and single pixel array are provided. Note that the
        original array is always used to create the first parametric map
        output, so the number of created segmententation instances is one
        greater than the number of items in this list. Items must be numbers
        greater than 1 and sorted in ascending order. A downsampling factor of
        *n* implies that the output array is *1/n* time the size of input pixel
        array. For example a list ``[2, 4, 8]`` would be produce 4 output
        parametric map instances. The first is the same size as the original
        pixel array, the next is half the size, the next is a quarter of the
        size of the original, and the last is one eighth the size of the
        original. Output sizes are rounded to the nearest integer.
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
        :class:`highdicom.pm.ParametricMap` object. However the following
        parameters are disallowed: ``instance_number``, ``sop_instance_uid``,
        ``plane_orientation``, ``plane_positions``, ``pixel_measures``,
        ``pixel_array``, ``tile_pixel_array``.

    """  # noqa: E501
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
            f'kwargs supplied to the create_parametric_map_pyramid function '
            f'should not contain a value for parameter {list(error_keys)[0]}.'
        )

    if pyramid_uid is None:
        pyramid_uid = UID()
    if series_instance_uid is None:
        series_instance_uid = UID()

    now = datetime.datetime.now()
    series_date = kwargs.get('series_date')
    if series_date is None:
        # Series date should not be after content date
        series_date = kwargs.get('content_date')
        if series_date is None:
            series_date = now.date()
    series_time = kwargs.get('series_time')
    if series_time is None:
        # Series time should not be after content time
        series_time = kwargs.get('content_time')
        if series_time is None:
            series_time = now.time()

    all_pmaps = []

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
        pmap = ParametricMap(
            source_images=[source_image],
            pixel_array=pixel_array,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            contains_recognizable_visual_features=(
                contains_recognizable_visual_features
            ),
            real_world_value_mappings=real_world_value_mappings,
            voi_lut_transformations=voi_lut_transformations,
            pyramid_uid=pyramid_uid,
            pyramid_label=pyramid_label,
            tile_pixel_array=True,
            plane_orientation=None,
            plane_positions=None,
            pixel_measures=pixel_measures,
            series_date=series_date,
            series_time=series_time,
            **kwargs,
        )

        all_pmaps.append(pmap)

    return all_pmaps
