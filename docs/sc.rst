.. _sc:

Secondary Capture (SC) Images
=============================

Secondary Capture (SC) images are intended to store conversions of non-DICOM
images to DICOM. Although it is not their intended purpose, in practice they
are also often used as a general-purpose IOD for images that don't have a
dedicated IOD to store things like screenshots and overlays of annotations on
images.

Within `highdicom`, Secondary Capture images are implemented in the
:class:`highdicom.sc.SCImage` class.

Basic Construction
------------------

To create an SC image, you pass the pixel array as a NumPy array along with
various required and optional pieces of metadata. You must include the
:ref:`basic` and (unless you are using the alternative
:meth:`highdicom.sc.SCImage.from_ref_dataset` constructor, see below) the
:ref:`patient_metadata`, as well as the following:

- ``coordinate_system`` (``str`` or
  :class:`highdicom.enum.CoordinateSystemNames`): The coordinate system within
  which the image exists (``'SLIDE'`` when the image is of a microscopic slide
  or ``'PATIENT'`` otherwise).
- ``patient_orientation`` (sequence of two ``str`` or
  :class:`highdicom.enum.PatientOrientationValuesBiped`): Required if and only
  if using the ``'PATINT'`` coordinate system. Orientation of the patient
  moving from the first to the last row (first item) and from the first to the
  last column (second item) of the image (required if `coordinate_system` is
  ``"PATIENT"``)
- ``photometric_interpretation`` (``str`` or
  :class:`highdicom.enum.PhotometricInterpretationValues`): This determines how
  the pixel values should be interpreted. The most common options are
  ``'MONOCHROME2'`` for a grayscale image or ``'RGB'`` for a color image.
- ``bits_allocated`` (``int``): TODO fix this mess

The image itself is passed as a ``numpy.ndarray`` to the ``pixel_array``
parameter. This should be either a 2D array of shape *rows* by *columns* for a
grayscale image, or a 3D array of shape *rows* by *columns* by *3* to specify 3
color channels. The array must have unsigned integer pixel values of type
``bool``, ``numpy.uint8``, or ``numpy.uint16``.

Therefore a basic example of constructing an SC image would look like this:

.. code-block:: python

    import highdicom as hd
    import numpy as np

    # Ranoom grayscale image of size 100x100
    pixel_array = np.random.randint(0, 255, (100, 100))

    sc_image = hd.sc.SCImage(
        pixel_array=pixel_array.astype(np.uint8),
        photometric_interpretation=hd.PhotometricInterpretationValues.MONOCHROME2,
        bits_allocated=8,
        coordinate_system=hd.CoordinateSystemNames.PATIENT,
        study_instance_uid=hd.UID(),
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        series_number=100,
        instance_number=1,
        manufacturer='Manufacturer',
        series_description='Example Secondary Capture',
        patient_name='Blogs^Joseph',
        patient_id='ABC12345',
        patient_birth_date='19871225',
        patient_orientation=('L', 'P'),
    )

    sc_image.save_as('sc.dcm')


Construction From a Referenced Image
------------------------------------

Often you will want to create an SC image to add to a study that already
exists. In this case, you can use the alternative
:meth:`highdicom.sc.SCImage.from_ref_dataset` constructor method to provide
patient, study, and clinical metadata, rather than providing this manually.

Here is a basic example:

.. code-block:: python

    import highdicom as hd
    import numpy as np


    # Load an example from the highdicom test files
    ref_dataset = hd.imread('data/test_files/ct_image.dcm')

    # Ranoom grayscale image of size 100x100
    pixel_array = np.random.randint(0, 255, (100, 100))

    sc_image = hd.sc.SCImage.from_ref_dataset(
        ref_dataset=ref_dataset,
        pixel_array=pixel_array.astype(np.uint8),
        photometric_interpretation=hd.PhotometricInterpretationValues.MONOCHROME2,
        bits_allocated=8,
        coordinate_system=hd.CoordinateSystemNames.PATIENT,
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        series_number=100,
        instance_number=1,
        manufacturer='Manufacturer',
        series_description='Example Secondary Capture',
        patient_orientation=('L', 'P'),
    )

    sc_image.save_as('sc.dcm')

Compression
-----------

Generally you will want to compress your SC image by providing the
``transfer_syntax_uid`` parameter. See :ref:`compression` for more detail.

CT With Annotation Example
--------------------------

In this example, we use a secondary capture to store an image containing a
labeled bounding box region drawn over a CT image.

.. code-block:: python

    import highdicom as hd
    import numpy as np
    from pydicom.uid import JPEG2000Lossless
    from PIL import Image, ImageDraw

    # Read in the source CT image
    image_dataset = hd.imread('/path/to/image.dcm')

    # Create an image for display by windowing the original image and drawing a
    # bounding box over it using Pillow's ImageDraw module

    # First get the original image with a soft tissue window (center 40, width 400)
    # applied, rescaled to the range 0 to 255.
    windowed_image = image_dataset.get_frame(
        1,
        apply_voi_transform=True,
        voi_transform_selector=hd.VOILUTTransformation(
            window_center=40,
            window_width=400,
        ),
        voi_output_range=(0, 255),
    )
    windowed_image = windowed_image.astype(np.uint8)

    # Create RGB channels
    windowed_image = np.tile(windowed_image[:, :, np.newaxis], [1, 1, 3])

    # Cast to a PIL image for easy drawing of boxes and text
    pil_image = Image.fromarray(windowed_image)

    # Draw a red bounding box over part of the image
    x0 = 10
    y0 = 10
    x1 = 60
    y1 = 60
    draw_obj = ImageDraw.Draw(pil_image)
    draw_obj.rectangle(
        [x0, y0, x1, y1],
        outline='red',
        fill=None,
        width=3
    )

    # Add some text
    draw_obj.text(xy=[10, 70], text='Region of Interest', fill='red')

    # Convert to numpy array
    pixel_array = np.array(pil_image)

    # The patient orientation defines the directions of the rows and columns of the
    # image, relative to the anatomy of the patient.  In a standard CT axial image,
    # the rows are oriented leftwards and the columns are oriented posteriorly, so
    # the patient orientation is ['L', 'P']
    patient_orientation=['L', 'P']

    # Create the secondary capture image. By using the `from_ref_dataset`
    # constructor, all the patient and study information will be copied from the
    # original image dataset
    sc_image = hd.sc.SCImage.from_ref_dataset(
        ref_dataset=image_dataset,
        pixel_array=pixel_array,
        photometric_interpretation=hd.PhotometricInterpretationValues.RGB,
        bits_allocated=8,
        coordinate_system=hd.CoordinateSystemNames.PATIENT,
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        series_number=100,
        instance_number=1,
        manufacturer='Manufacturer',
        pixel_spacing=image_dataset.PixelSpacing,
        patient_orientation=patient_orientation,
        transfer_syntax_uid=JPEG2000Lossless,
        series_description='Example Secondary Capture',
    )

    # Save the file
    sc_image.save_as('sc_output.dcm')

Creating an SC Series
---------------------

Unlike some of the other types of image in highdicom, the basic Secondary
Capture IOD contains only a single frame. If you want to create a Secondary
Capture with multiple frames, you should create a series. You can do this by
creating multiple instances in a loop. Be sure to use a single
`series_instance_uid` and provide consecutive values for `instance_number`.

Here is an example for a multiframe CT scan that is in a NumPy array called
``ct_to_save`` where we do not have the original DICOM files on hand. We want
to overlay a segmentation that is stored in a NumPy array called "seg_out".

.. code-block:: python

    import os
    import highdicom as hd
    import numpy as np
    from pydicom.uid import JPEG2000Lossless

    pixel_spacing = [1.0, 1.0]
    sz = ct_to_save.shape[2]
    series_instance_uid = hd.UID()
    study_instance_uid = hd.UID()

    for iz in range(sz):
        this_slice = ct_to_save[:, :, iz]

        # Window the image to a soft tissue window (center 40, width 400)
        # and rescale to the range 0 to 255
        windowed_image = hd.pixels.apply_voi_window(
            this_slice,
            window_center=40,
            window_width=400,

        )

        # Create RGB channels
        pixel_array = np.tile(windowed_image[:, :, np.newaxis], [1, 1, 3])

        # transparency level
        alpha = 0.1

        pixel_array[:, :, 0] = 255 * (1 - alpha) * seg_out[:, :, iz] + alpha * pixel_array[:, :, 0]
        pixel_array[:, :, 1] = alpha * pixel_array[:, :, 1]
        pixel_array[:, :, 2] = alpha * pixel_array[:, :, 2]

        patient_orientation = ['L', 'P']

        # Create the secondary capture image
        sc_image = hd.sc.SCImage(
            pixel_array=pixel_array.astype(np.uint8),
            photometric_interpretation=hd.PhotometricInterpretationValues.RGB,
            bits_allocated=8,
            coordinate_system=hd.CoordinateSystemNames.PATIENT,
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            sop_instance_uid=hd.UID(),
            series_number=100,
            instance_number=iz + 1,
            manufacturer='Manufacturer',
            pixel_spacing=pixel_spacing,
            patient_orientation=patient_orientation,
            series_description='Example Secondary Capture',
            transfer_syntax_uid=JPEG2000Lossless,
        )

        sc_image.save_as(os.path.join("output", 'sc_output_' + str(iz) + '.dcm'))
