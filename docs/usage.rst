.. _user-guide:

User guide
==========

Creating and parsing DICOM objects using the :mod:`highdicom` package.

.. _creating-seg:

Creating Segmentation (SEG) images
----------------------------------

Derive a Segmentation image from a series of single-frame Computed Tomography
(CT) images:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    import numpy as np
    from pydicom.sr.codedict import codes
    from pydicom.filereader import dcmread

    # Path to directory containing single-frame legacy CT Image instances
    # stored as PS3.10 files
    series_dir = Path('path/to/series/directory')
    image_files = series_dir.glob('*.dcm')

    # Read CT Image data sets from PS3.10 files on disk
    image_datasets = [dcmread(str(f)) for f in image_files]

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
    from pydicom.filereader import dcmread

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path('/path/to/image/file')

    # Read SM Image data set from PS3.10 files on disk
    image_dataset = dcmread(str(image_file))

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
    seg_dataset = Segmentation(
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

Iterating over segments in a segmentation image instance:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    from pydicom.filereader import dcmread

    # Path to multi-frame SEG image instance stored as PS3.10 file
    seg_file = Path('/path/to/seg/file')

    # Read SEG Image data set from PS3.10 files on disk
    seg_dataset = dcmread(str(seg_file))

    # Iterate over segments and print the information about the frames
    # that encode the segment across different image positions
    for frames, frame_descriptions, description in hd.seg.utils.iter_segments(seg_dataset):
        print(frames.shape)
        print(
            set([
                item.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                for item in frame_descriptions
            ])
        )
        print(description.SegmentNumber)


.. _creating-sr:

Creating Structured Report (SR) documents
-----------------------------------------

Create a Structured Report document that contains a numeric area measurement for
a planar region of interest (ROI) in a single-frame computed tomography (CT)
image:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    import numpy as np
    from pydicom.filereader import dcmread
    from pydicom.sr.codedict import codes
    from pydicom.uid import generate_uid
    from highdicom.sr.content import FindingSite
    from highdicom.sr.templates import Measurement, TrackingIdentifier

    # Path to single-frame CT image instance stored as PS3.10 file
    image_file = Path('/path/to/image/file')

    # Read CT Image data set from PS3.10 files on disk
    image_dataset = dcmread(str(image_file))

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
        content=measurement_report[0],
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

Finding relevant content in the nested SR content tree:

.. code-block:: python

    from pathlib import Path

    import highdicom as hd
    from pydicom.filereader import dcmread
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
    # to TID 1500 "Measurment Report"
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
    from pydicom import dcmread
    from pydicom.uid import RLELossless
    from PIL import Image, ImageDraw

    # Read in the source CT image
    image_dataset = dcmread('/path/to/image.dcm')

    # Create an image for display by windowing the original image and drawing a
    # bounding box over it using Pillow's ImageDraw module
    slope = getattr(image_dataset, 'RescaleSlope', 1)
    intercept = getattr(image_dataset, 'RescaleIntercept', 0)
    original_image = image_dataset.pixel_array * slope + intercept

    # Window the image to a soft tissue window (center 40, width 400)
    # and rescale to the range 0 to 255
    lower = -160
    upper = 240
    windowed_image = np.clip(original_image, lower, upper)
    windowed_image = (windowed_image - lower) * 255 / (upper - lower)
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
        lower = -160
        upper = 240
        windowed_image = np.clip(this_slice, lower, upper)
        windowed_image = (windowed_image - lower) * 255 / (upper - lower)

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

.. .. _creation-legacy:

.. Creating Legacy Converted Enhanced Images
.. -----------------------------------------

.. .. code-block:: python

..     from highdicom.legacy.sop import LegacyConvertedEnhancedCTImage
