"""Utilities for working with SEG image instances."""
from typing import Generator, Optional, Union

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code


def iter_segments(dataset: Dataset) -> Generator:
    """Iterates over segments of a Segmentation image instance.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Segmentation image instance

    Returns
    -------
    Generator[Tuple[numpy.ndarray, Tuple[pydicom.dataset.Dataset], pydicom.dataset.Dataset]]
        Pixel pata frames and description (items of the Per-Frame Functional
        Groups Sequence and item of the Segment Sequence) of each segment

    Raises
    ------
    AttributeError
        When data set does not contain Content Sequence attribute.

    """  # noqa
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
    for i in np.unique(segment_number_per_frame):
        indices = np.where(segment_number_per_frame == i)[0]
        yield (
            dataset.pixel_array[indices, ...],
            tuple([
                dataset.PerFrameFunctionalGroupsSequence[index]
                for index in indices
            ]),
            segment_description_lut[i],
        )
