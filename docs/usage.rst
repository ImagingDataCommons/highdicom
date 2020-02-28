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

    import numpy as np
    from pydicom.sr.codedict import codes
    from pydicom.filereader import dcmread
    from pydicom.uid import generate_uid

    from highdicom.content import AlgorithmIdentificationSequence
    from highdicom.seg.content import SegmentDescription
    from highdicom.seg.enum import (
        SegmentAlgorithmTypeValues,
        SegmentationTypeValues
    )
    from highdicom.seg.sop import Segmentation

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
    algorithm_identification = AlgorithmIdentificationSequence(
        name='test',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Describe the segment
    description_segment_1 = SegmentDescription(
        segment_number=1,
        segment_label='first segment',
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=generate_uid(),
        tracking_id='test segmentation of computed tomography image'
    )

    # Create the Segmentation instance
    seg_dataset = Segmentation(
        source_images=image_datasets,
        pixel_array=mask,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=[description_segment_1],
        series_instance_uid=generate_uid(),
        series_number=2,
        sop_instance_uid=generate_uid(),
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

    import numpy as np
    from pydicom.sr.codedict import codes
    from pydicom.filereader import dcmread
    from pydicom.uid import generate_uid

    from highdicom.content import AlgorithmIdentificationSequence
    from highdicom.seg.content import SegmentDescription
    from highdicom.seg.enum import (
        SegmentAlgorithmTypeValues,
        SegmentationTypeValues
    )
    from highdicom.seg.sop import Segmentation

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path('/path/to/image/file')

    # Read SM Image data set from PS3.10 files on disk
    image_dataset = dcmread(str(image_file))

    # Create a binary segmentation mask
    mask = np.max(image_dataset.pixel_array, axis=3) > 1

    # Describe the algorithm that created the segmentation
    algorithm_identification = AlgorithmIdentificationSequence(
        name='test',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Describe the segment
    description_segment_1 = SegmentDescription(
        segment_number=1,
        segment_label='first segment',
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=generate_uid(),
        tracking_id='test segmentation of slide microscopy image'
    )

    # Create the Segmentation instance
    seg_dataset = Segmentation(
        source_images=[image_dataset],
        pixel_array=mask,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=[description_segment_1],
        series_instance_uid=generate_uid(),
        series_number=2,
        sop_instance_uid=generate_uid(),
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

    from pydicom.filereader import dcmread

    from highdicom.seg.utils import iter_segments

    # Path to multi-frame SEG image instance stored as PS3.10 file
    seg_file = Path('/path/to/seg/file')

    # Read SEG Image data set from PS3.10 files on disk
    seg_dataset = dcmread(str(seg_file))

    # Iterate over segments and print the information about the frames
    # that encode the segment across different image positions
    for frames, frame_descriptions, description in iter_segments(seg_dataset):
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

    import numpy as np
    from pydicom.uid import generate_uid
    from pydicom.filereader import dcmread
    from pydicom.sr.codedict import codes

    from highdicom.sr.content import ImageRegion3D
    from highdicom.sr.sop import Comprehensive3DSR
    from highdicom.sr.templates import (
        DeviceObserverIdentifyingAttributes,
        FindingSite,
        Measurement,
        MeasurementProperties,
        MeasurementReport,
        ObservationContext,
        ObserverContext,
        PersonObserverIdentifyingAttributes,
        PlanarROIMeasurementsAndQualitativeEvaluations,
        TrackingIdentifier,
    )
    from highdicom.sr.value_types import CodedConcept
    from highdicom.sr.enum import GraphicTypeValues3D

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path('/path/to/image/file')

    # Read SM Image data set from PS3.10 files on disk
    image_dataset = dcmread(str(image_file))

    # Describe the context of reported observations: the person that reported
    # the observations and the device that was used to make the observations
    observer_person_context = ObserverContext(
        observer_type=codes.DCM.Person,
        observer_identifying_attributes=PersonObserverIdentifyingAttributes(
            name='Foo'
        )
    )
    observer_device_context = ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
            uid=generate_uid()
        )
    )
    observation_context = ObservationContext(
        observer_person_context=observer_person_context,
        observer_device_context=observer_device_context,
    )

    # Describe the image region for which observations were made
    # (in physical space based on the frame of reference)
    referenced_region = ImageRegion3D(
        graphic_type=GraphicTypesValues3D.POLYGON,
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
            tracking_identifier=TrackingIdentifier(uid=generate_uid()),
            value=1.7,
            unit=codes.UCUM.SquareMillimeter,
            properties=MeasurementProperties(
                normality=CodedConcept(
                    value="17621005",
                    meaning="Normal",
                    scheme_designator="SCT"
                ),
                level_of_significance=codes.SCT.NotSignificant
            )
        )
    ]
    imaging_measurements = [
        PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(
                uid=generate_uid(),
                identifier='Planar ROI Measurements'
            ),
            referenced_region=referenced_region,
            finding_type=codes.SCT.SpinalCord,
            measurements=measurements,
            finding_sites=finding_sites
        )
    ]

    # Create the report content
    measurement_report = MeasurementReport(
        observation_context=observation_context,
        procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
        imaging_measurements=imaging_measurements
    )

    # Create the Structured Report instance
    sr_dataset = Comprehensive3DSR(
        evidence=[image_dataset],
        content=measurement_report[0],
        series_number=1,
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
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

    from pydicom.filereader import dcmread
    from pydicom.sr.codedict import codes

    from highdicom.sr.enum import ValueTypeValues, RelationshipTypeValues
    from highdicom.sr.utils import find_content_items


    # Path to SR document instance stored as PS3.10 file
    document_file = Path('/path/to/document/file')

    # Load document from file on disk
    sr_dataset = dcmread(str(document_file))

    # Find all content items that may contain other content items.
    containers = find_content_items(
        dataset=sr_dataset,
        relationship_type=RelationshipTypeValues.CONTAINS
    )
    print(containers)

    # Query content of SR document, where content is structured according
    # to TID 1500 "Measurment Report"
    if sr_dataset.ContentTemplateSequence[0].TemplateIdentifier == 'TID1500':
        # Determine who made the observations reported in the document
        observers = find_content_items(
            dataset=sr_dataset,
            name=codes.DCM.PersonObserverName
        )
        print(observers)

        # Find all imaging measurements reported in the document
        measurements = find_content_items(
            dataset=sr_dataset,
            name=codes.DCM.ImagingMeasurements,
            recursive=True
        )
        print(measurements)

        # Find all findings reported in the document
        findings = find_content_items(
            dataset=sr_dataset,
            name=codes.DCM.Finding,
            recursive=True
        )
        print(findings)

        # Find regions of interest (ROI) described in the document
        # in form of spatial coordinates (SCOORD)
        regions = find_content_items(
            dataset=sr_dataset,
            value_type=ValueTypeValues.SCOORD,
            recursive=True
        )
        print(regions)

.. .. _creation-legacy:

.. Creating Legacy Converted Enhanced Images
.. -----------------------------------------

.. .. code-block:: python

..     from highdicom.legacy.sop import LegacyConvertedEnhancedCTImage
