.. _sitk_lib:

SimpleITK
=========

`SimpleITK`_ is a widely-used library for volumetric image processing. While
many simple processing operations can be performed within highdicom itself,
SimpleITK provides a much larger set of tools and many existing image
processing pipelines are built in SimpleITK. By integrating highdicom with
SimpleITK you can therefore benefit from both ITK's large set of processing
features and highdicom's support for DICOM reading and writing with minimal
"glue code". In particular, correctly transferring spatial metadata between
highdicom and SimpleITK representations is straightforwardly handled for you.
In order to use ITK with highdicom, the ``SimpleITK`` python package must be
installed separately. Version 2.2.1 or later is required.

.. _sitk_vol:

Volume Conversions
------------------

Highdicom supports conversions with the ``SimpleITK.Image`` class through the
:meth:`highdicom.Volume.to_simpleitk` and :meth:`highdicom.Volume.from_simpleitk`
methods. Like highdicom, SimpleITK uses the "LPS" convention. However, when
converting to and from NumPy arrays, SimpleITK reverses the order of
dimensions. This permutation is handled automatically by highdicom and requires
no intervention by the user.


Creating a SimpleITK Image from a Volume:

.. code-block:: python

    import highdicom as hd


    vol = hd.Volume(...)

    simpleitk_image = vol.to_simpleitk()

Creating a volume from a SimpleITK Image:

.. code-block:: python

    import SimpleITK as sitk
    import highdicom as hd


    simpleitk_image = sitk.Image(...)

    vol = hd.Volume.from_simpleitk(simpleitk_image)


.. _`ITK`: https://itk.org/
.. _`SimpleITK`: https://simpleitk.org/


Use Cases
---------

**Load a DICOM Segmentation into SimpleITK:**

This allows you to work primarily in SimpleITK, while still benefitting from
highdicom's full feature set for reading segmentations, such as correctly
combining multiple segments, filtering segments, filling in missing slices,
lazily retrieving frames, etc. This behavior is not limited to Segmentations,
and would work equally well with any DICOM image the :class:`highdicom.Image`
class supports. The resulting SimpleITK image will have its spatial affine
matrix correctly populated from the source DICOM file.

.. code-block:: python

    import highdicom as hd

    # Here we load in an example DICOM segmentation from the highdicom repo test
    # data that contains two segments. Parameters of the get_volume() method
    # control the volume that is extracted. For example, here we choose one of the
    # two segments before converting to ITK
    sitk_image = (
        hd.seg.segread("data/test_files/seg_image_ct_binary_overlap.dcm")
        .get_volume(
            segment_numbers=[2],
            relabel=True,
            combine_segments=True,
        )
        .to_simpleitk()
    )

**Create a DICOM Segmentation from a SimpleITK image:**

This allows you to build processing pipelines primarily in SimpleITK, but then
benefit from highdicom's full versatile support for creating DICOM
segmentations, including support for multiple segmentation types, multiple
compression methods (transfer syntaxes) etc. The spatial metadata is correctly
carried throughout this whole snippet and stored in the segmentation without
requiring any user code.


.. code-block:: python

    import SimpleITK as sitk
    import highdicom as hd
    from pydicom.sr.codedict import codes
    from pydicom.uid import JPEGLSLossless


    # An example CT file in highdicom test data repo
    ct_file = "data/test_files/ct_image.dcm"

    # Read the image file using itk
    sitk_image = sitk.ReadImage(ct_file, outputPixelType=sitk.sitkInt16)

    # We do still need to load the source CT image file with highdicom (or
    # pydicom) to get the metadata, but we can skip the pixel data
    source_image = hd.imread(ct_file, lazy_frame_retrieval=True)


    def sitk_segmentation_method(image: sitk.Image) -> sitk.Image:
        """Toy example of a processing method built with SimpleITK.

        Here we just apply a simple intensity threshold at 300HU as an example.

        """
        thresholder = sitk.BinaryThresholdImageFilter()
        thresholder.SetLowerThreshold(300)
        thresholder.SetUpperThreshold(10000)
        thresholder.SetInsideValue(1)
        return thresholder.Execute(image)


    # Run the ITK processing
    seg_image = sitk_segmentation_method(sitk_image)

    # Create a highdicom Volume from the ITK image to pass to the
    # highdicom.Segmentation constructor
    seg_volume = hd.Volume.from_simpleitk(seg_image)

    # In this case, the Volume has shape (128, 128, 1). highdicom.seg.Segmentation
    # stores volumes split into frames split down the first dimension, so to
    # get the most sensible output we will transpose the volume. We could do
    # this explicitly, or, more straightforwardly, just transpose to match to
    # the orientation of the source image
    seg_volume = seg_volume.match_orientation(source_image.get_volume_geometry())

    # Now to start creating the segmentation. First we have to describe the segment
    # Here, we describe a bone segment produced by an automatic algorithm)
    bone_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='Bone',
        segmented_property_category=codes.SCT.AnatomicalStructure,
        segmented_property_type=codes.SCT.Bone,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name='Thresholder 5000X',
            version='v1.0',
            family=codes.cid7162.MorphologicalOperations
        )
    )

    # Now construct the DICOM Segmentation Image from the ITK segmentation mask.
    # Highdicom will take the spatial information from the itk image (via the
    # highdicom Volume) and compare it to the source image metadata to establish 
    # the spatial relationship between the segmentation mask and the source image
    # automatically
    seg = hd.seg.Segmentation(
        source_images=[source_image],
        pixel_array=seg_volume,
        segmentation_type=hd.seg.SegmentationTypeValues.LABELMAP,
        segment_descriptions=[bone_description],
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Foo Corp.',
        manufacturer_model_name='Bone Segmentation Algorithm',
        software_versions='0.0.1',
        device_serial_number='1234567890',
        transfer_syntax_uid=JPEGLSLossless,
        series_description='Bone Threshold Segmentation',
    )

    # Save output file
    seg.save_as("bone_segmentation.dcm")
