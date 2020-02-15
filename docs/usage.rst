.. _user-guide:

User guide
==========

Creation of derived DICOM objects using the :mod:`highdicom` package.

.. _seg:

Segmentation (SEG) images
-------------------------

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
        SegmentAlgorithmTypes,
        SegmentationTypes
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
        algorithm_type=SegmentAlgorithmTypes.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=generate_uid(),
        tracking_id='test segmentation of computed tomography image'
    )

    # Create the Segmentation instance
    seg_dataset = Segmentation(
        source_images=image_datasets,
        pixel_array=mask,
        segmentation_type=SegmentationTypes.BINARY,
        segment_descriptions=[description_segment_1],
        series_instance_uid=generate_uid(),
        series_number=2,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer='Manufacturer',
        software_versions='v1',
        device_serial_number='Device XYZ',
        manufacturer_model_name='The best one'
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
        SegmentAlgorithmTypes,
        SegmentationTypes
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
        algorithm_type=SegmentAlgorithmTypes.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=generate_uid(),
        tracking_id='test segmentation of slide microscopy image'
    )

    # Create the Segmentation instance
    seg_dataset = Segmentation(
        source_images=[image_dataset],
        pixel_array=mask,
        segmentation_type=SegmentationTypes.BINARY,
        segment_descriptions=[description_segment_1],
        series_instance_uid=generate_uid(),
        series_number=2,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer='Manufacturer',
        software_versions='v1',
        device_serial_number='Device XYZ'
    )

    print(seg_dataset)

.. _sr:

Structured Reports (SR) documents
---------------------------------

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
    from highdicom.sr.value_types import (
        CodedConcept,
        GraphicTypes3D,
    )

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
        graphic_type=GraphicTypes3D.POLYGON,
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


.. _legacy:

Legacy Converted Enhanced Images
--------------------------------

.. code-block:: python

    from highdicom.legacy.sop import LegacyConvertedEnhancedCTImage
