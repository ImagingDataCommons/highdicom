"""Utilities for working with SEG image instances."""
from typing import Iterator

import numpy as np
from pydicom.dataset import Dataset


def iter_segments(dataset: Dataset) -> Iterator:
    """Iterates over segments of a Segmentation image instance.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Segmentation image instance

    Returns
    -------
    Iterator[Tuple[numpy.ndarray, Tuple[pydicom.dataset.Dataset, ...], pydicom.dataset.Dataset]]
        For each segment in the Segmentation image instance, provides the
        Pixel Data frames representing the segment, items of the Per-Frame
        Functional Groups Sequence describing the individual frames, and
        the item of the Segment Sequence describing the segment

    Raises
    ------
    AttributeError
        When data set does not contain Content Sequence attribute.

    """  # noqa: E501
    if not hasattr(dataset, 'PixelData'):
        raise AttributeError(
            'Data set does not contain a Pixel Data attribute.'
        )
    segment_description_lut = {
        int(item.SegmentNumber): item
        for item in dataset.SegmentSequence
    }
    segment_number_per_frame = np.array([
        int(item.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        for item in dataset.PerFrameFunctionalGroupsSequence
    ])
    pixel_array = dataset.pixel_array
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]
    for i in np.unique(segment_number_per_frame):
        indices = np.where(segment_number_per_frame == i)[0]
        yield (
            pixel_array[indices, ...],
            tuple([
                dataset.PerFrameFunctionalGroupsSequence[index]
                for index in indices
            ]),
            segment_description_lut[i],
        )
