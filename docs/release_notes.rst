.. _releasenotes:

Release Notes
=============

Brief release notes may be found on `on Github
<https://github.com/MGHComputationalPathology/highdicom/releases>`_. This page
contains migration notes for major breaking changes to the library's API.

.. _add-segments-deprecation:

Deprecation of `add_segments` method
------------------------------------

Prior to highdicom 0.8.0, it was possible to add further segments to
:class:`highdicom.seg.Segmentation` image after its construction using the
`add_segments` method. This was found to produce incorrect Dimension Index
Values if the empty frames did not match within all segments added.

To create the Dimension Index Values correctly, the constructor needs access to
all segments in the image when it is first created. Therefore, the
`add_segments` method was removed in highdicom 0.8.0. Instead, in highdicom
0.8.0 and later, multiple segments can be passed to the constructor by stacking
their arrays along the fourth dimension.

Given code that adds segments like this, in highdicom 0.7.0 and earlier:

.. code-block:: python

    import numpy as np
    import highdicom as hd

    # Create initial segment mask and description
    mask_1 = np.array(
        # ...
    )
    description_1 = hd.seg.SegmentDescription(
        # ...
    )
    seg = hd.seg.Segmentation(
        # ...
        pixel_array=mask_1,
        segment_descriptions=[description_1],
        # ...
    )

    # Create a second segment and add to the existing segmentation
    mask_2 = np.array(
        # ...
    )
    description_2 = hd.seg.SegmentDescription(
        # ...
    )

    seg.add_segments(
        # ...
        pixel_array=mask_2,
        segment_descriptions=[description_2],
        # ...
    )


This can be migrated to highdicom 0.8.0 and later by concatenating the arrays
along the fourth dimension and calling the constructor at the end.

.. code-block:: python

    import numpy as np
    import highdicom as hd

    # Create initial segment mask and description
    mask_1 = np.array(
        # ...
    )
    description_1 = hd.seg.SegmentDescription(
        # ...
    )

    # Create a second segment and description
    mask_2 = np.array(
        # ...
    )
    description_2 = hd.seg.SegmentDescription(
        # ...
    )

    combined_segments = np.concatenate([mask_1, mask_2], axis=-1)
    combined_descriptions = [description_1, description_2]

    seg = hd.seg.Segmentation(
        # ...
        pixel_array=combined_segments,
        segment_descriptions=combined_descriptions,
        # ...
    )


Note that segments must always be stacked down the fourth dimension (with index
3) of the ``pixel_array``. In order to create a segmentation with multiple
segments for a single source frame, it is required to add a new dimension
(with length 1) as the first dimension (index 0) of the array.
