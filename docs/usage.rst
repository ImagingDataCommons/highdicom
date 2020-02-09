.. _user-guide:

User guide
==========

Creation of derived DICOM objects using the :mod:`highdicom` package.

.. _seg:

Segmentation (SEG) images
-------------------------

Segmentation objects contain segmentation masks for images contained in other DICOM objects.

Highdicom makes it simple to construct segmentation objects from a segmentation mask contained in a numpy array
and the original DICOM objects to which the segmentation refers.

In this example, we create a segmentation object from some test images by simple thresholding.

.. code-block:: python

   import pydicom
   import numpy as np

   from pydicom.sr.codedict import codes
   from highdicom.seg.content import (AlgorithmIdentificationSequence, SegmentAlgorithmTypes,
                                      SegmentDescription)
   from highdicom.seg.enum import SegmentAlgorithmTypes, SegmentationTypes
   from highdicom.seg.sop import Segmentation


   # The segmentation object records various information about the algorithm used to produce the segmentation
   # Here we define this information
   ALGORITHM_NAME = 'My First Segmentation Algorithm'
   ALGORITHM_VERSION = '0.1'
   ALGORITHM_MANUFACTURER = 'Me'
   DEVICE_SERIAL_NUMBER = '123.456'

   # For this example we will use a series of CT images from the pydicom test data
   # These will be the images that the segmentation is performed on
   datasets = [pydicom.read_file(f) for f in pydicom.data.get_testdata_files('77654033/CT2')]

   # Create a 3D pixel array of the raw pixel data
   image_array = np.stack([d.pixel_array for d in datasets], axis=0)

   # Create an example segmentation mask by simple thresholding of the pixels
   seg_array = image_array > image_array.mean()

   # The segmentation object requires certain information about the information that produced the segmentation
   # In particular, the family of algorithms should be described by a suitable coding scheme.
   # Here, we use the DCM coding scheme, which pydicom makes available as a python enum.
   algo_identifier = AlgorithmIdentificationSequence(name=ALGORITHM_NAME,
                                                     family=codes.DCM.ArtificialIntelligence,
                                                     version=ALGORITHM_VERSION,
                                                     source=ALGORITHM_MANUFACTURER)

   # We also need to describe each segment (class) in the segmentation, and the algorithm that produced it.
   # This should be done using a suitable coding scheme. Here we use the SNOMED (SCT) coding scheme to describe a
   # tumor
   seg_desc = [SegmentDescription(segment_number=1,
                                  segment_label='Tumor',
                                  segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
                                  segmented_property_type=codes.SCT.Tumor,
                                  algorithm_type=SegmentAlgorithmTypes.AUTOMATIC.value,
                                  algorithm_identification=algo_identifier)]

   # With this information, we can construct a segmentation object straightforwardly
   # We also need to generate a SOP Instance UID and Series Instance UID, and provide a series number.
   seg = Segmentation(source_images=datasets,
                      pixel_array=seg_array,
                      segment_descriptions=seg_desc,
                      series_instance_uid=pydicom.uid.generate_uid(),
                      series_number=100,
                      segmentation_type=SegmentationTypes.BINARY,
                      sop_instance_uid=pydicom.uid.generate_uid(),
                      instance_number=1,
                      manufacturer=ALGORITHM_MANUFACTURER,
                      manufacturer_model_name=ALGORITHM_NAME,
                      software_versions=ALGORITHM_VERSION,
                      device_serial_number=DEVICE_SERIAL_NUMBER)

   # We now have a pydicom dataset that can be used like any other image
   # For example it can be saved to disk like this
   seg.save_as('test_segmentation.dcm')

.. _sr:

Structured Reports (SR) documents
---------------------------------

.. code-block:: python

    from highdicom.sr.sop import Comprehensive3DSR


.. _legacy:

Legacy Converted Enhanced Images
--------------------------------

.. code-block:: python

    from highdicom.legacy.sop import LegacyConvertedEnhancedCTImage
