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


.. _correct-coordinate-mapping:

Correct coordinate mapping
--------------------------

Prior to highdicom 0.14.1, mappings between image coordinates and reference
coordinates did not take into account that there are two image coordinate
systems, which are shifted by 0.5 pixels.

1. **Pixel indices**: (column, row) indices into the pixel matrix. The values
   are zero-based integers in the range [0, Columns - 1] and [0, Rows - 1].
   Pixel indices are defined relative to the centers of pixels and the (0, 0)
   index is located at the center of the top left corner hand pixel of the
   total pixel matrix.
2. **Image coordinates**: (column, row) coordinates in the pixel matrix at
   sub-pixel resolution. The values are floating-point numbers in the range
   [0, Columns] and [0, Rows]. Image coordinates are defined relative to the
   top left corner of the pixels and the (0.0, 0.0) point is located at the top
   left corner of the top left corner hand pixel of the total pixel matrix.

To account for these differences, introduced two additional transformer classes
in highdicom 0.14.1. and made changes to the existing ones.
The existing transformer class now map between image coordinates and reference
coordinates (:class:`highdicom.spatial.ImageToReferenceTransformer` and
:class:`highdicom.spatial.ReferenceToImageTransformer`).
While the new transformer classes map between pixel indices and reference
coordinates (:class:`highdicom.spatial.PixelToReferenceTransformer` and
:class:`highdicom.spatial.ReferenceToPixelTransformer`).
Note that you want to use the former classes for converting between spatial
coordinates (SCOORD) (:class:`highdicom.sr.ScoordContentItem`) and 3D spatial
coordinates (SCOORD3D) (:class:`highdicom.sr.Scoord3DContentItem`) and the
latter for determining the position of a pixel in the frame of reference or for
projecting a coordinate in the frame of reference onto the image plane.

To make the distinction between pixel indices and image coordinates as clear as
possible, we renamed the parameter of the
:func:`highdicom.spatial.map_pixel_into_coordinate_system` function from
``coordinate`` to ``index`` and enforce that the values that are provided via
the argument are integers rather than floats.
In addition, the return value of
:func:`highdicom.spatial.map_coordinate_into_pixel_matrix` is now a tuple of
integers.
