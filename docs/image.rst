.. _image:

Images
======

Highdicom's :class:`highdicom.Image` class is a fundamental class that provides
methods for working with existing DICOM images. It inherits from pydicom's
``pydicom.Dataset`` class, and therefore you can access individual DICOM
attributes just like you can for any dataset. However, the
:class:`highdicom.Image` class also provides further functionality to make it
easier to access frames with pixel transforms applied and arrange frames based
on metadata attributes.

Most of highdicom's classes correspond to individual Information Object
Definitions defined within the DICOM standard, for example `Segmentation
Image`, `Parametric Map`, or `Comprehensive3DSR`. However this is **not** the
case for the :class:`highdicom.Image` class, which captures behavior common to
a large number of different IODs. Any IOD that includes pixel data can be
loaded into an Image object. This includes both single frame ("legacy") and
newer multi-frame objects.

Reading Images
--------------

You can read in an image from a file using the :func:`highdicom.imread()`
function:

.. code-block:: python

    import highdicom as hd

    # This is a test file in the highdicom git repository
    im = hd.imread('data/test_files/ct_image.dcm')

Alternatively, you can convert an existing ``pydicom.Dataset`` instance that
represents an image to a :class:`highdicom.Image` instances using the
:meth:`highdicom.Image.from_dataset()` method.

.. code-block:: python

    import pydicom
    import highdicom as hd

    # This is a test file in the highdicom git repository
    dcm = pydicom.dcmread('data/test_files/ct_image.dcm')

    im = hd.Image.from_dataset(dcm)

:class:`highdicom.Image` instances cannot be created directly using a
constructor, they must always be created from an existing DICOM object.

Accessing Frames
----------------

The :class:`highdicom.Image` class has three methods for accessing individual
frames of the image:

* :meth:`highdicom.Image.get_raw_frame()`: This returns the raw bytes
  containing the information for a single frame as a Python ``bytes`` object.
  If the image uses a compressed transfer syntax (such as JPEG or its
  variants), the compressed bytestream is returned. This method is intended for
  advanced users and use cases.
* :meth:`highdicom.Image.get_stored_frame()`: This returns the frame as a NumPy
  array with minimal processing. The raw bytes are decompressed if necessary
  and reshaped to form the frame of the correct shape, but no further pixel
  transforms are applied. These are referred to as "stored values" within the
  DICOM standard. Note that the pydicom `.pixel_array` property returns stored
  values for all frames at once.
* :meth:`highdicom.Image.get_frame()`: In addition to the above, this method
  applies pixel transforms stored in the file to the stored values before
  returning them. The transforms applied are configurable through parameters
  (see :doc:`pixel_transforms` for more details on pixel transforms), but by
  default any pixel transform found in the dataset except the value-of-interest
  (VOI) transform is applied. This should be your default way of accessing
  image frames in most cases, since it will typtically return the pixels as the
  creator of the object intended them to be understood. By default, the
  returned frames have datatype `numpy.float64`, but this can be controlled
  using the `dtype` parameter.

For all methods, the first parameter ``frame_number`` is an integer giving the
number of the frame, where the first frame has index 1. This one-based indexing
may be unnatural for Python programming (which generally uses 0-based
indexing). The reason for this choice is that the DICOM standard numbers frames
starting at 1, and in particular if a DICOM object contains references to its
frames, or those of other objects, 1-based frame numbers are used. If you
prefer to use 0-based indexing, you can specify ``as_index=True``.

.. code-block:: python

    import numpy as np
    import highdicom as hd


    # This is a test file in the highdicom git repository
    im = hd.imread('data/test_files/ct_image.dcm')

    # Get raw bytes for the first frame
    first_frame = im.get_raw_frame(1)
    print(type(first_frame))
    # <class  'bytes'>

    # Get stored values for the first frame
    first_frame = im.get_stored_frame(1)
    print(first_frame.min(), first_frame.max())
    # 128 2191

    # Get pixels after rescale/slope applied
    first_frame = im.get_frame(1)
    print(first_frame.dtype)
    # float64
    print(first_frame.min(), first_frame.max())
    # -896.0 1167.0

    # Specify an integer datatype
    first_frame = im.get_frame(1, dtype=np.int32)
    print(first_frame.dtype)
    # int32

    # Alternative, using 0-based index
    first_frame = im.get_frame(0, as_index=True)

These three methods process the raw pixel data "lazily" as needed to avoid
processing unnecessary frames. If you know that you are likely to access frames
multiple times, you can force caching of the stored values by accessing the
``.pixel_array`` property (inherited from ``pydicom.Dataset``).

Additionally, there are two methods for accessing multiple frames at a time:

* :meth:`highdicom.Image.get_stored_frames()`: Returns a stack of multiple
  stored frames. The first parameter is a list (or other iterable) of frame
  numbers. If omitted, all frames are returned in the order they are stored in
  the image.
* :meth:`highdicom.Image.get_frames()`: Returns a stack of multiple
  frames with pixel transforms applied. The first parameter is a list (or other
  iterable) of frame numbers. If omitted, all frames are returned in the order
  they are stored in the image.

Accessing Total Pixel Matrices
------------------------------

Digital pathology images in DICOM format are typically stored as "tiled"
images, where frames are arranged in a 2D pattern across a plane to form a
large "total pixel matrix". For such images, you typically want to work with
the large 2D total pixel matrix that is formed by correctly arranging the tiles
into a 2D array rather than 3D arrays of stacked frames. `highdicom` provides
the :meth:`highdicom.Image.get_total_pixel_matrix()` method for this purpose.

Called without any parameters, it returns a 2D array containing the full total
pixel matrix. The two dimensions are the spatial dimensions. Behind the scenes
highdicom has stitched together the required frames stored in the original file
for you.

.. code-block:: python

    import highdicom as hd

    # Read in a tiled test file from the highdicom repo
    im = hd.imread('data/test_files/sm_image.dcm')

    # Get the full total pixel matrix
    tpm = im.get_total_pixel_matrix()

    expected_shape = (
        im.TotalPixelMatrixRows,
        im.TotalPixelMatrixColumns,
        3,  # RGB channels
    )
    assert tpm.shape == expected_shape

Furthermore, you can request a sub-region of the full total pixel matrix by
specifying the start and/or stop indices for the rows and/or columns within the
total pixel matrix. Note that this method follows DICOM 1-based convention for
indexing rows and columns, i.e. the first row and column of the total pixel
matrix are indexed by the number 1 (not 0 as is common within Python). Negative
indices are also supported to index relative to the last row or column, with -1
being the index of the last row or column. Like for standard Python indexing,
the stop indices are specified as one beyond the final row/column in the
returned array. The requested region does not have to start or stop
at the edges of the underlying frames: `highdicom` stitches together only the
relevant parts of the frames to create the requested image for you.

.. code-block:: python

    import highdicom as hd

    # Read in a tiled test file from the highdicom repo
    im = hd.imread('data/test_files/sm_image.dcm')

    # Get a region of the total pixel matrix
    tpm = im.get_total_pixel_matrix(
        row_start=15,
        row_end=25,
        column_start=26,
    )

    expected_shape = (10, 25, 3)
    assert tpm.shape == expected_shape

Accessing Volumes
-----------------

Many multi-frame images, especially from radiology modalities such as CT, MRI,
DBT, and PET, contain frames that can be arranged together to form voxels on a
regularly-sampled rectangular 3D grid. The :meth:`highdicom.Image.get_volume()`
method checks for this case and, if possible, returns a 3D voxel array array
with the affine matrix describing its position in the frame of reference
coordinate system, as a :class:`highdicom.Volume`. To just check whether it is
possible to form a volume from the frames, use the
:class:`highdicim.Image.get_volume_geometry()` method, which will return
``None`` if no volume can be formed.

.. code-block:: python

    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load an enhanced (multiframe) CT image
    im = hd.imread(get_testdata_file('eCT_Supplemental.dcm'))

    geometry = im.get_volume_geometry()

    assert geometry is not None

    vol = im.get_volume()
    print(vol.spatial_shape)
    # (2, 512, 512)

    print(vol.affine)
    # [[   0.          0.         -0.388672   99.5     ]
    #  [  -0.          0.388672    0.       -301.5     ]
    #  [  10.          0.          0.       -159.      ]
    #  [   0.          0.          0.          1.      ]]

Further parameters allow you to access a sub-region of the volume and control
the pixel transforms applied to the frames.

Any single frame image that defines its position within the frame-of-reference
coordinate system can accessed as a volume, as can any image with a total pixel
matrix. In these cases, the first spatial dimension will always have shape 1.

See :doc:`volume` for an overview of the :class:`highdicom.Volume` class.

.. _lazy:

Lazy Frame Retrieval
--------------------

The :func:`highdicom.imread()` function provides the ``lazy_frame_retrieval``
parameter. If used, the metadata is loaded from the file without the pixel
data. Pixel data is subsequently loaded from the file whenever it is needed by
one of the :class:`highdicom.Image` object's methods. This can save loaded
unneeded pixel data from file when only a subset of it is needed.

In this example, lazy frame retrieval is used to avoid loading all frames of a
tiled image:

.. code-block:: python

    import highdicom as hd

    # Read in a tiled test file from the highdicom repo
    im = hd.imread(
        'data/test_files/sm_image.dcm',
        lazy_frame_retrieval=True
    )

    # Get a region of the total pixel matrix
    tpm = im.get_total_pixel_matrix(row_end=20)

Whether this saves time depends on your usage patterns and hardware.
Furthermore in certain situations highdicom needs to parse the entire pixel
data element in order to determine frame boundaries. This occurs when the
frames are compressed using an encapsulated transfer syntax but there is no
offset table giving the locations of frame boundaries within the file. An
offset table can take the form of either a `basic offset table <BOT>`_ (BOT) at
the start of the PixelData element or an `extended offset table <EOT>`_ (EOT)
as a separate attribute in the metadata. These offset tables are not required,
but often one of them is included in images. Without an offset table, the
potential speed benefits of using lazy frame retrieval are usually eliminated,
even if only a small number of frames are loaded.

.. _BOT: https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_A.4.html
.. _EOT: http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#sect_C.7.6.3.1.8
