.. _introduction:

Introduction
============

The ``highdicom`` build distribution provides an application programming interface (API) for creating DICOM objects for image-derived information, focusing on Information Object Definitions (IODs) relevant for quantitative imaging, computer vision and machine learning such as Segmentation (SEG) images and Structured Reporting (SR) documents.

The :mod:`highdicom` Python package contains several subpackages for different modalities and SOP classes, for example :mod:`highdicom.seg` for SEG images and :mod:`highdicom.sr` for SR documents.

Motivation and goals
--------------------

The DICOM standard is crucial for achieving interoperability between image analysis applications and image storage and communication systems during both development and clinical deployment.
However, the standard is vast and complex and implementing it correctly can be challenging - even for DICOM experts.
The main goal of *highdicom* is to abstract the complexity of the standard and allow developers of image analysis applications to focus on the algorithm and the data analysis rather than low-level data encoding.
To this end, *highdicom* provides a high-level, intuitive application programming interface (API) that enables developers to create high-quality DICOM objects in a few lines of Python code.
Importantly, the API is compatible with digital pathology and radiology imaging modalities, including Slide Microscopy (SM), Computed Tomography (CT) and Magnetic Resonance (MR) imaging.

Design
------

The `highdicom` Python package exposes an object-orientated application programming interface (API) for the construction of DICOM Service Object Pair (SOP) instances according to the corresponding IODs.
Each SOP class is implemented in form of a Python class that inherits from `pydicom.dataset.Dataset`.
The class constructor accepts the image-derived information (e.g. pixel data as a `numpy.ndarray`) as well as required contextual information (e.g. patient identifier) as specified by the corresponding IOD.
DICOM validation is built-in and is automatically performed upon object construction to ensure created SOP instances are compliant with the standard.

