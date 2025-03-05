Documentation of the Highdicom Package
======================================

``highdicom`` is a pure Python package providing a high-level application programming interface (API) for working with DICOM files, with a focus on common operations required for machine learning, computer vision, and other similar computational analyses. Broadly speaking, the package helps with three types of task:

1. Reading existing DICOM image files of a wide variety of modalities (covering radiology, pathology, and more) and formatting the frames to prepare them for computational analysis.
2. Storing image-derived information, for example from computational analyses or human annotation, in derived DICOM objects for communication and storage. This includes:

  - Annotations
  - Parametric Map images
  - Segmentation images
  - Structured Report documents (containing numerical results, qualitative evaluations, and/or vector graphic annotations)
  - Secondary Capture images
  - Key Object Selection documents
  - Legacy Converted Enhanced CT/PET/MR images (e.g., for single frame to multi-frame conversion)
  - Softcopy Presentation State instances (including Grayscale, Color, and Pseudo-Color)

3. Reading existing derived DICOM files and filtering and accessing the information contained within them.

For new users looking to get an overview of the library's capabilities and perform basic tasks, we recommend starting with the :ref:`quick-start` page. For a detailed introduction to many of the library's capabilities, see the rest of the :ref:`user-guide`. Documentation of all classes and functions may be found in the :ref:`api-docs`.

For questions, suggestions, or to report bugs, please use the issue tracker on our GitHub `repository`_.

.. _repository: https://github.com/ImagingDataCommons/highdicom

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   overview
   installation
   usage
   development
   code_of_conduct
   conformance
   citation
   license
   release_notes
   package



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
