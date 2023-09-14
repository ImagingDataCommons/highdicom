"""Tools for constructing multi-resolution segmentation pyramids."""
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from pydicom import Dataset

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

    """
    if pyramid_uid is None:
        pyramid_uid = UID()

    n_sources = len(source_images)
    n_pix_arrays = len(pixel_arrays)

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
        n_outputs = len(downsample_factors) + 1  # including original
    elif downsample_factors is not None:
        raise TypeError(
            'Argument "downsample_factors" must not be provided when multiple '
            'source images or pixel arrays are provided.'
        )
    if n_sources > 1 and n_pix_arrays > 1:
        if n_sources != n_pix_arrays:
            raise ValueError(
                "If providing multiple source images and multiple pixel "
                "arrays, the number of items in the two lists must match."
            )
        n_outputs = n_sources

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
            row_spacing = (
                source_image.PixelSpacing[0] *
                (pixel_arrays[0].shape[0] / pixel_array.shape[0])
            )
            column_spacing = (
                source_image.PixelSpacing[1] *
                (pixel_arrays[0].shape[1] / pixel_array.shape[1])
            )
            pixel_measures = PixelMeasuresSequence(
                pixel_spacing=(row_spacing, column_spacing),
                slice_thickness=source_image.SliceThickness,
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
