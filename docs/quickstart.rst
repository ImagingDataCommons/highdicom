.. _quick-start:

Quick Start
===========

This section gives simple examples of how to perform various tasks with
*highdicom*.

.. _accessing-frames:

Loading Images and Accessing Frames
-----------------------------------

The :class:`highdicom.Image` class is used to work with general DICOM images,
including accessing and arranging their frames.

This basic example loads an existing DICOM file and accesses an individual frame,
with and without the value-of-interest (VOI) transform applied.
See :doc:`image` for an overview of the :class:`highdicom.Image` class and
:doc:`pixel_transforms` for an overview of pixel transforms.

.. code-block:: python

    import highdicom as hd


    # This is a test file in the highdicom git repository
    im = hd.imread('data/test_files/dx_image.dcm')

    # Get pixels with rescale/slope applied (default behavior)
    # Note that in DICOM, frame numbers are 1-based
    first_frame = im.get_frame(1)

    # Get pixels with rescale/slope and VOI applied
    first_frame = im.get_frame(1, apply_voi_transform=True)

Constructing Total Pixel Matrices
---------------------------------

Load an existing tiled digital pathology image from a DICOM file and access its
total pixel matrix.
See :doc:`image` for an overview of the :class:`highdicom.Image` class.

.. code-block:: python

    import highdicom as hd


    # This is a test file in the highdicom git repository
    im = hd.imread('data/test_files/sm_image.dcm')

    # Returns a (50, 50, 3) numpy array of the total pixel matrix
    total_pixel_matrix = im.get_total_pixel_matrix()

Working with Volumes
--------------------

If a DICOM image contains frames that can be arranged into a regular 3D
volumetric grid, a :class:`highdicom.Volume` object can be created from it and
used to preprocess an image.
See :doc:`volume` for an overview of the :class:`highdicom.Volume` class.

.. code-block:: python

    import numpy as np
    from pydicom.data import get_testdata_file

    import highdicom as hd

    # Load an enhanced (multiframe) CT image from the pydicom test files
    im = hd.imread(get_testdata_file('eCT_Supplemental.dcm'))

    # Get a Volume object
    volume = im.get_volume()

    # Access the volume's affine matrix and other properties
    print(volume.affine)
    # [[   0.          0.         -0.388672   99.5     ]
    #  [  -0.          0.388672    0.       -301.5     ]
    #  [  10.          0.          0.       -159.      ]
    #  [   0.          0.          0.          1.      ]]

    print(volume.spatial_shape)
    # (2, 512, 512)

    print(volume.spacing)
    # (10.0, 0.388672, 0.388672)

    print(volume.unit_vectors())
    # (array([ 0., -0.,  1.]), array([0., 1., 0.]), array([-1.,  0.,  0.]))

    # Ensure the volume is arranged in foot-posterior-left orientation
    volume = volume.to_patient_orientation("FPL")

    # Center-crop to a given shape
    volume = volume.crop_to_spatial_shape((2, 224, 224))

    # Access the numpy array
    assert isinstance(volume.array, np.ndarray)


.. _creating-seg:

Creating Segmentation (SEG) images
----------------------------------

DICOM Segmentations are used to store segmentations of other DICOM images.
Highdicom uses the :class:`highdicom.seg.Segmentation` to create and read DICOM
Segmentations.
For an in-depth overview of DICOM segmentations, see :doc:`seg`.

This simple example derives a Segmentation image from a series of single-frame
Computed Tomography (CT) images:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes

    # Path to directory containing single-frame legacy CT Image instances
    # stored as PS3.10 files
    series_dir = Path('path/to/series/directory')
    image_files = series_dir.glob('*.dcm')

    # Read CT Image data sets from PS3.10 files on disk
    image_datasets = [hd.imread(str(f)) for f in image_files]

    # Create a binary segmentation mask
    mask = np.zeros(
        shape=(
            len(image_datasets),
            image_datasets[0].Rows,
            image_datasets[0].Columns
        ),
        dtype=np.bool
    )
    mask[1:-1, 10:-10, 100:-100] = True

    # Describe the algorithm that created the segmentation
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='test',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Describe the segment
    description_segment_1 = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='first segment',
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='test segmentation of computed tomography image'
    )

    # Create the Segmentation instance
    seg_dataset = hd.seg.Segmentation(
        source_images=image_datasets,
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description_segment_1],
        series_instance_uid=hd.UID(),
        series_number=2,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Manufacturer',
        manufacturer_model_name='Model',
        software_versions='v1',
        device_serial_number='Device XYZ',
    )

    print(seg_dataset)

    seg_dataset.save_as("seg.dcm")


Derive a Segmentation image from a multi-frame Slide Microscopy (SM) image:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path('/path/to/image/file')

    # Read SM Image data set from PS3.10 files on disk
    image_dataset = hd.imread(str(image_file))

    # Create a binary segmentation mask
    mask = np.max(image_dataset.pixel_array, axis=3) > 1

    # Describe the algorithm that created the segmentation
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='test',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Describe the segment
    description_segment_1 = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label='first segment',
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='test segmentation of slide microscopy image'
    )

    # Create the Segmentation instance
    seg_dataset = hd.seg.Segmentation(
        source_images=[image_dataset],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description_segment_1],
        series_instance_uid=hd.UID(),
        series_number=2,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Manufacturer',
        manufacturer_model_name='Model',
        software_versions='v1',
        device_serial_number='Device XYZ'
    )

    print(seg_dataset)

.. _parsing-seg:

Parsing Segmentation (SEG) images
---------------------------------

Finding relevant segments in a segmentation image instance and retrieving masks
for them:

.. code-block:: python

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes

    # Read SEG Image data set from PS3.10 files on disk into a Segmentation
    # object
    # This example is a test file in the highdicom git repository
    seg = hd.seg.segread('data/test_files/seg_image_ct_binary_overlap.dcm')

    # Check the number of segments
    assert seg.number_of_segments == 2

    # Get a SegmentDescription object, containing various metadata about a given
    # segment
    segment_description = seg.get_segment_description(2)

    # The segment description has various properties
    assert segment_description.segment_number == 2
    assert segment_description.segment_label == 'second segment'
    assert segment_description.tracking_id == 'Spine'
    assert segment_description.tracking_uid == '1.2.826.0.1.3680043.10.511.3.10042414969629429693880339016394772'
    assert segment_description.segmented_property_type == codes.SCT.Spine

    # You can also use get_segment_numbers() to find segments (identified by their
    # segment number) using one or more filters on these properties. For example,
    # to find segments that have segmented property type "Bone"
    bone_segment_numbers = seg.get_segment_numbers(
      segmented_property_type=codes.SCT.Bone
    )
    assert bone_segment_numbers ==  [1]

    # Retrieve the segmentation mask as a highdicom.Volume (with spatial metadata)
    seg_volume = seg.get_volume()

    # Accessing using a volume fills in any missing slices, which are assumed to be
    # empty
    assert seg_volume.array.shape == (165, 16, 16, 2)

    # Access the spatial affine matrix of the resulting volume as a numpy array
    print(seg_volume.affine)
    # [[   0.      ,    0.      ,    0.488281, -125.      ],
    #  [   0.      ,    0.488281,    0.      , -128.100006],
    #  [  -1.25    ,    0.      ,    0.      ,  105.519997],
    #  [   0.      ,    0.      ,    0.      ,    1.      ]])

    # List SOP Instance UIDs of the images from which the segmentation was
    # derived
    for study_uid, series_uid, sop_uid in seg.get_source_image_uids():
      print(study_uid, series_uid, sop_uid)
      # '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.1, 1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.2, 1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93'
      # ...

    # Here is a list of known SOP Instance UIDs that are a subset of those
    # from which the segmentation was derived
    source_image_uids = [
      '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.93',
      '1.3.6.1.4.1.5962.1.1.0.0.0.1196530851.28319.0.94',
    ]

    # Retrieve a binary segmentation mask for these images for the bone segment
    mask = seg.get_pixels_by_source_instance(
      source_sop_instance_uids=source_image_uids,
      segment_numbers=bone_segment_numbers,
    )
    # Output is a numpy array of shape (instances x rows x columns x segments)
    assert mask.shape == (2, 16, 16, 1)
    assert np.unique(mask).tolist() == [0, 1]

    # Alternatively, retrieve the segmentation mask for the full list of segments
    # (2 in this case), and combine the resulting array into a "label mask", where
    # pixel value represents segment number
    mask = seg.get_pixels_by_source_instance(
      source_sop_instance_uids=source_image_uids,
      combine_segments=True,
      skip_overlap_checks=True,  # the segments in this image overlap
    )
    # Output is a numpy array of shape (instances x rows x columns)
    assert mask.shape == (2, 16, 16)
    assert np.unique(mask).tolist() == [0, 1, 2]

For more information see :doc:`seg`.

.. _creating-sr:

Creating Structured Report (SR) documents
-----------------------------------------

Structured Reports store measurements or observations on an image, including
qualitative evaluations, numerical measurements and/or regions of interest
represented using vector graphics.
For a full overview of SRs, see :doc:`generalsr` and :doc:`tid1500`.

Create a Structured Report document that contains a numeric area measurement for
a planar region of interest (ROI) in a single-frame computed tomography (CT)
image:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes
    from pydicom.uid import generate_uid
    from highdicom.sr.content import FindingSite
    from highdicom.sr.templates import Measurement, TrackingIdentifier

    # Path to single-frame CT image instance stored as PS3.10 file
    image_file = Path('/path/to/image/file')

    # Read CT Image data set from PS3.10 files on disk
    image_dataset = hd.imread(str(image_file))

    # Describe the context of reported observations: the person that reported
    # the observations and the device that was used to make the observations
    observer_person_context = hd.sr.ObserverContext(
        observer_type=codes.DCM.Person,
        observer_identifying_attributes=hd.sr.PersonObserverIdentifyingAttributes(
            name='Foo'
        )
    )
    observer_device_context = hd.sr.ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
            uid=hd.UID()
        )
    )
    observation_context = hd.sr.ObservationContext(
        observer_person_context=observer_person_context,
        observer_device_context=observer_device_context,
    )

    # Describe the image region for which observations were made
    # (in physical space based on the frame of reference)
    referenced_region = hd.sr.ImageRegion3D(
        graphic_type=hd.sr.GraphicTypeValues3D.POLYGON,
        graphic_data=np.array([
            (165.0, 200.0, 134.0),
            (170.0, 200.0, 134.0),
            (170.0, 220.0, 134.0),
            (165.0, 220.0, 134.0),
            (165.0, 200.0, 134.0),
        ]),
        frame_of_reference_uid=image_dataset.FrameOfReferenceUID
    )

    # Describe the anatomic site at which observations were made
    finding_sites = [
        FindingSite(
            anatomic_location=codes.SCT.CervicoThoracicSpine,
            topographical_modifier=codes.SCT.VertebralForamen
        ),
    ]

    # Describe the imaging measurements for the image region defined above
    measurements = [
        Measurement(
            name=codes.SCT.AreaOfDefinedRegion,
            tracking_identifier=hd.sr.TrackingIdentifier(uid=generate_uid()),
            value=1.7,
            unit=codes.UCUM.SquareMillimeter,
            properties=hd.sr.MeasurementProperties(
                normality=hd.sr.CodedConcept(
                    value="17621005",
                    meaning="Normal",
                    scheme_designator="SCT"
                ),
                level_of_significance=codes.SCT.NotSignificant
            )
        )
    ]
    imaging_measurements = [
        hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(
                uid=hd.UID(),
                identifier='Planar ROI Measurements'
            ),
            referenced_region=referenced_region,
            finding_type=codes.SCT.SpinalCord,
            measurements=measurements,
            finding_sites=finding_sites
        )
    ]

    # Create the report content
    measurement_report = hd.sr.MeasurementReport(
        observation_context=observation_context,
        procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
        imaging_measurements=imaging_measurements
    )

    # Create the Structured Report instance
    sr_dataset = hd.sr.Comprehensive3DSR(
        evidence=[image_dataset],
        content=measurement_report,
        series_number=1,
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Manufacturer'
    )

    print(sr_dataset)

.. _parsing-sr:

Parsing Structured Report (SR) documents
----------------------------------------

Highdicom has special support for parsing structured reports conforming to the
TID1500 "Measurement Report" template using specialized Python classes for
templates.
For more information see :doc:`tid1500parsing`.

.. code-block:: python

    import numpy as np
    import highdicom as hd
    from pydicom.sr.codedict import codes

    # This example is in the highdicom test data files in the repository
    sr = hd.sr.srread("data/test_files/sr_document_with_multiple_groups.dcm")

    # First we explore finding measurement groups. There are three types of
    # measurement groups (image measurement, planar roi measurement groups, and
    # volumetric roi measurement groups)

    # Get a list of all image measurement groups referencing an image with a
    # particular SOP Instance UID
    groups = sr.content.get_image_measurement_groups(
        referenced_sop_instance_uid="1.3.6.1.4.1.5962.1.1.1.1.1.20040119072730.12322",
    )
    assert len(groups) == 1

    # Get a list of all image measurement groups with a particular tracking UID
    groups = sr.content.get_image_measurement_groups(
        tracking_uid="1.2.826.0.1.3680043.10.511.3.77718622501224431322963356892468048",
    )
    assert len(groups) == 1

    # Get a list of all planar ROI measurement groups with finding type "Nodule"
    # AND finding site "Lung"
    groups = sr.content.get_planar_roi_measurement_groups(
        finding_type=codes.SCT.Nodule,
        finding_site=codes.SCT.Lung,
    )
    assert len(groups) == 1

    # Get a list of all volumetric ROI measurement groups (with no filters)
    groups = sr.content.get_volumetric_roi_measurement_groups()
    assert len(groups) == 1

    # Get a list of all planar ROI measurement groups with graphic type CIRCLE
    groups = sr.content.get_planar_roi_measurement_groups(
        graphic_type=hd.sr.GraphicTypeValues.CIRCLE,
    )
    assert len(groups) == 1

    # Get a list of all planar ROI measurement groups stored as regions
    groups = sr.content.get_planar_roi_measurement_groups(
        reference_type=codes.DCM.ImageRegion,
    )
    assert len(groups) == 2

    # Get a list of all volumetric ROI measurement groups stored as volume
    # surfaces
    groups = sr.content.get_volumetric_roi_measurement_groups(
        reference_type=codes.DCM.VolumeSurface,
    )
    assert len(groups) == 1

    # Next, we explore the properties of measurement groups that can
    # be conveniently accessed with Python properties

    # Use the first (only) image measurement group as an example
    group = sr.content.get_image_measurement_groups()[0]

    # tracking_identifier returns a Python str
    assert group.tracking_identifier == "Image0001"

    # tracking_uid returns a hd.UID, a subclass of str
    assert group.tracking_uid == "1.2.826.0.1.3680043.10.511.3.77718622501224431322963356892468048"

    # source_images returns a list of hd.sr.SourceImageForMeasurementGroup,
    # which in turn have some properties to access data
    assert isinstance(group.source_images[0], hd.sr.SourceImageForMeasurementGroup)
    ref_sop_uid = group.source_images[0].referenced_sop_instance_uid
    assert ref_sop_uid == "1.3.6.1.4.1.5962.1.1.1.1.1.20040119072730.12322"

    # for the various optional pieces of information in a measurement, accessing
    # the relevant property returns None if the information is not present
    assert group.finding_type is None

    # Now use the first planar ROI group as a second example
    group = sr.content.get_planar_roi_measurement_groups()[0]

    # finding_type returns a CodedConcept
    assert group.finding_type == codes.SCT.Nodule

    # finding_sites returns a list of hd.sr.FindingSite objects
    assert isinstance(group.finding_sites[0], hd.sr.FindingSite)
    # the value of a finding site is a CodedConcept
    assert group.finding_sites[0].value == codes.SCT.Lung

    # reference_type returns a CodedConcept (the same values used above for
    # filtering)
    assert group.reference_type == codes.DCM.ImageRegion

    # since this has reference type ImageRegion, we can access the referenced
    # using 'roi', which will return an hd.sr.ImageRegion object
    assert isinstance(group.roi, hd.sr.ImageRegion)

    # the graphic type and actual ROI coordinates (as a numpy array) can be
    # accessed with the graphic_type and value properties of the roi
    assert group.roi.graphic_type == hd.sr.GraphicTypeValues.CIRCLE
    assert isinstance(group.roi.value, np.ndarray)
    assert group.roi.value.shape == (2, 2)

    # Next, we explore getting individual measurements out of measurement
    # groups

    # Use the first planar measurement group as an example
    group = sr.content.get_planar_roi_measurement_groups()[0]

    # Get a list of all measurements
    measurements = group.get_measurements()

    # Get the first measurements for diameter
    measurement = group.get_measurements(name=codes.SCT.Diameter)[0]

    # Access the measurement's name
    assert measurement.name == codes.SCT.Diameter

    # Access the measurement's value
    assert measurement.value == 10.0

    # Access the measurement's unit
    assert measurement.unit == codes.UCUM.mm

    # Get the diameter measurement in this group
    evaluation = group.get_qualitative_evaluations(
        name=codes.DCM.LevelOfSignificance
    )[0]

    # Access the measurement's name
    assert evaluation.name == codes.DCM.LevelOfSignificance

    # Access the measurement's value
    assert evaluation.value == codes.SCT.NotSignificant


Additionally, there are low-level utilities that you can use to find content
items in the content tree of any structured report documents:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    from pydicom.sr.codedict import codes

    # Path to SR document instance stored as PS3.10 file
    document_file = Path('/path/to/document/file')

    # Load document from file on disk
    sr_dataset = dcmread(str(document_file))

    # Find all content items that may contain other content items.
    containers = hd.sr.utils.find_content_items(
        dataset=sr_dataset,
        relationship_type=RelationshipTypeValues.CONTAINS
    )
    print(containers)

    # Query content of SR document, where content is structured according
    # to TID 1500 "Measurement Report"
    if sr_dataset.ContentTemplateSequence[0].TemplateIdentifier == 'TID1500':
        # Determine who made the observations reported in the document
        observers = hd.sr.utils.find_content_items(
            dataset=sr_dataset,
            name=codes.DCM.PersonObserverName
        )
        print(observers)

        # Find all imaging measurements reported in the document
        measurements = hd.sr.utils.find_content_items(
            dataset=sr_dataset,
            name=codes.DCM.ImagingMeasurements,
            recursive=True
        )
        print(measurements)

        # Find all findings reported in the document
        findings = hd.sr.utils.find_content_items(
            dataset=sr_dataset,
            name=codes.DCM.Finding,
            recursive=True
        )
        print(findings)

        # Find regions of interest (ROI) described in the document
        # in form of spatial coordinates (SCOORD)
        regions = hd.sr.utils.find_content_items(
            dataset=sr_dataset,
            value_type=ValueTypeValues.SCOORD,
            recursive=True
        )
        print(regions)


.. _creating-ann:

Creating Microscopy Bulk Simple Annotation (ANN) objects
--------------------------------------------------------

Microscopy Bulk Simple Annotations store large numbers of annotations of
objects in microscopy images in a space-efficient way.
For more information see :ref:`ann` and the documentation of the
:class:`highdicom.ann.MicroscopyBulkSimpleAnnotations` class.


.. code-block:: python

    from pydicom.sr.codedict import codes
    from pydicom.sr.coding import Code
    import highdicom as hd
    import numpy as np

    # Load a slide microscopy image from the highdicom test data (if you have
    # cloned the highdicom git repo)
    sm_image = hd.imread('data/test_files/sm_image.dcm')

    # Graphic data containing two nuclei, each represented by a single point
    # expressed in 2D image coordinates
    graphic_data = [
        np.array([[34.6, 18.4]]),
        np.array([[28.7, 34.9]]),
    ]

    # You may optionally include measurements corresponding to each annotation
    # This is a measurement object representing the areas of each of the two
    # nuclei
    area_measurement = hd.ann.Measurements(
        name=codes.SCT.Area,
        unit=codes.UCUM.SquareMicrometer,
        values=np.array([20.4, 43.8]),
    )

    # An annotation group represents a single set of annotations of the same
    # type. Multiple such groups may be included in a bulk annotations object
    # This group represents nuclei annotations produced by a manual "algorithm"
    nuclei_group = hd.ann.AnnotationGroup(
        number=1,
        uid=hd.UID(),
        label='nuclei',
        annotated_property_category=codes.SCT.AnatomicalStructure,
        annotated_property_type=Code('84640000', 'SCT', 'Nucleus'),
        algorithm_type=hd.ann.AnnotationGroupGenerationTypeValues.MANUAL,
        graphic_type=hd.ann.GraphicTypeValues.POINT,
        graphic_data=graphic_data,
        measurements=[area_measurement],
    )

    # Include the annotation group in a bulk annotation object
    bulk_annotations = hd.ann.MicroscopyBulkSimpleAnnotations(
        source_images=[sm_image],
        annotation_coordinate_type=hd.ann.AnnotationCoordinateTypeValues.SCOORD,
        annotation_groups=[nuclei_group],
        series_instance_uid=hd.UID(),
        series_number=10,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='MGH Pathology',
        manufacturer_model_name='MGH Pathology Manual Annotations',
        software_versions='0.0.1',
        device_serial_number='1234',
        content_description='Nuclei Annotations',
    )

    bulk_annotations.save_as('nuclei_annotations.dcm')

.. _parsing-ann:

Parsing Microscopy Bulk Simple Annotation (ANN) objects
-------------------------------------------------------

The following example demonstrates loading in a small bulk microscopy
annotations file, finding an annotation group representing annotation of
nuclei, and extracting the graphic data for the annotation as well as the area
measurements corresponding to those annotations.
For more information see :ref:`ann`.

.. code-block:: python

    from pydicom.sr.codedict import codes
    from pydicom.sr.coding import Code
    import highdicom as hd

    # Load a bulk annotation file and convert to highdicom object
    ann_dataset = hd.ann.annread('data/test_files/sm_annotations.dcm')

    # Search for annotation groups by filtering for annotated property type of
    # 'nucleus', and take the first such group
    group = ann.get_annotation_groups(
        annotated_property_type=Code('84640000', 'SCT', 'Nucleus'),
    )[0]

    # Determine the graphic type and the number of annotations
    assert group.number_of_annotations == 2
    assert group.graphic_type == hd.ann.GraphicTypeValues.POINT

    # Get the graphic data as a list of numpy arrays, we have to pass the
    # coordinate type from the parent object here
    graphic_data = group.get_graphic_data(
        coordinate_type=ann.AnnotationCoordinateType
    )

    # For annotations of graphic type "POINT" and coordinate type "SCOORD" (2D
    # image coordinates), each annotation is a (1 x 2) NumPy array
    assert graphic_data[0].shape == (1, group.number_of_annotations)

    # Annotations may also optionally contain measurements
    names, values, units = group.get_measurements(name=codes.SCT.Area)

    # The name and the unit are returned as a list of CodedConcepts
    # and the values are returned in a numpy array of shape (number of
    # annotations x number of measurements)
    assert names[0] == codes.SCT.Area
    assert units[0] == codes.UCUM.SquareMicrometer
    assert values.shape == (group.number_of_annotations, 1)


.. _creating-sc:

Creating Secondary Capture (SC) images
--------------------------------------

Secondary captures are a way to store images that were not created directly
by an imaging modality within a DICOM file. They are often used to store
screenshots or overlays, and are widely supported by viewers. However other
methods of displaying image derived information, such as segmentation images
and structured reports should be preferred if they are supported because they
can capture more detail about how the derived information was obtained and
what it represents.

In this example, we use a secondary capture to store an image containing a
labeled bounding box region drawn over a CT image.

.. code-block:: python

    import highdicom as hd
    import numpy as np
    from pydicom.uid import RLELossless
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
        transfer_syntax_uid=RLELossless
    )

    # Save the file
    sc_image.save_as('sc_output.dcm')


To save a 3D image as a series of output slices, simply loop over the 2D
slices and ensure that the individual output instances share a common series
instance UID.  Here is an example for a CT scan that is in a NumPy array called
"ct_to_save" where we do not have the original DICOM files on hand. We want to
overlay a segmentation that is stored in a NumPy array called "seg_out".

.. code-block:: python

    import highdicom as hd
    import numpy as np
    import os

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
        )

        sc_image.save_as(os.path.join("output", 'sc_output_' + str(iz) + '.dcm'))


Creating Grayscale Softcopy Presentation State (GSPS) Objects
-------------------------------------------------------------

A presentation state contains information about how another image should be
rendered, and may include "annotations" in the form of basic shapes, polylines,
and text overlays. Note that a GSPS is not recommended for storing annotations
for any purpose except visualization. A structured report would usually be
preferred for storing annotations for clinical or research purposes.

.. code-block:: python

    import highdicom as hd

    import numpy as np
    from pydicom.valuerep import PersonName


    # Read in an example CT image
    image_dataset = hd.imread('path/to/image.dcm')

    # Create an annotation containing a polyline
    polyline = hd.pr.GraphicObject(
        graphic_type=hd.pr.GraphicTypeValues.POLYLINE,
        graphic_data=np.array([
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0]]
        ),  # coordinates of polyline vertices
        units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for graphic data
        tracking_id='Finding1',  # site-specific ID
        tracking_uid=hd.UID()  # highdicom will generate a unique ID
    )

    # Create a text object annotation
    text = hd.pr.TextObject(
        text_value='Important Finding!',
        bounding_box=np.array(
            [30.0, 30.0, 40.0, 40.0]  # left, top, right, bottom
        ),
        units=hd.pr.AnnotationUnitsValues.PIXEL,  # units for bounding box
        tracking_id='Finding1Text',  # site-specific ID
        tracking_uid=hd.UID()  # highdicom will generate a unique ID
    )

    # Create a single layer that will contain both graphics
    # There may be multiple layers, and each GraphicAnnotation object
    # belongs to a single layer
    layer = hd.pr.GraphicLayer(
        layer_name='LAYER1',
        order=1,  # order in which layers are displayed (lower first)
        description='Simple Annotation Layer',
    )

    # A GraphicAnnotation may contain multiple text and/or graphic objects
    # and is rendered over all referenced images
    annotation = hd.pr.GraphicAnnotation(
        referenced_images=[image_dataset],
        graphic_layer=layer,
        graphic_objects=[polyline],
        text_objects=[text]
    )

    # Assemble the components into a GSPS object
    gsps = hd.pr.GrayscaleSoftcopyPresentationState(
        referenced_images=[image_dataset],
        series_instance_uid=hd.UID(),
        series_number=123,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='Manufacturer',
        manufacturer_model_name='Model',
        software_versions='v1',
        device_serial_number='Device XYZ',
        content_label='ANNOTATIONS',
        graphic_layers=[layer],
        graphic_annotations=[annotation],
        institution_name='MGH',
        institutional_department_name='Radiology',
        content_creator_name=PersonName.from_named_components(
            family_name='Doe',
            given_name='John'
        ),
    )

    # Save the GSPS file
    gsps.save_as('gsps.dcm')


.. .. _creation-legacy:

.. Creating Legacy Converted Enhanced Images
.. -----------------------------------------

.. .. code-block:: python

..     from highdicom.legacy.sop import LegacyConvertedEnhancedCTImage
