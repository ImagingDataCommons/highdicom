[![Build Status](https://travis-ci.com/mghcomputationalpathology/highdicom.svg?branch=master)](https://travis-ci.com/mghcomputationalpathology/highdicom)
[![PyPi Distribution](https://img.shields.io/pypi/v/highdicom.svg)](https://pypi.python.org/pypi/highdicom/)
[![Python Versions](https://img.shields.io/pypi/pyversions/highdicom.svg)](https://pypi.org/project/highdicom/)

# High DICOM

A library that provides high-level DICOM abstractions for the Python programming language to facilitate the creation and handling of DICOM objects for image-derived information, including image annotations, and image analysis results.
It currently provides tools for creating and decoding the following DICOM information object definitions (IODs):
* Annotations
* Parametric Map images
* Segmentation images
* Structured Report documents
* Secondary Capture images
* Key Object Selection documents
* Legacy Converted Enhanced CT/PET/MR images (e.g., for single frame to multi-frame conversion)

## Documentation

Please refer to the online documentation at [highdicom.readthedocs.io](https://highdicom.readthedocs.io), which includes installation instructions, a user guide with examples, a developer guide, and complete documentation of the application programming interface of the `highdicom` package.

## Citation

For more information about the motivation of the library and the design of highdicom's API, please see the following article:

> [Highdicom: A Python library for standardized encoding of image annotations and machine learning model outputs in pathology and radiology](https://arxiv.org/abs/2106.07806)
> C.P. Bridge, C. Gorman, S. Pieper, S.W. Doyle, J.K. Lennerz, J. Kalpathy-Cramer, D.A. Clunie, A.Y. Fedorov, and M.D. Herrmann

If you use highdicom in your research, please cite the above article.

## Support

The developers gratefully acknowledge their support:
* The [Alliance for Digital Pathology](https://digitalpathologyalliance.org/)
* The [MGH & BWH Center for Clinical Data Science](https://www.ccds.io/)
* [Quantitative Image Informatics for Cancer Research (QIICR)](http://qiicr.org)
* [Radiomics](http://radiomics.io)
* The [NCI Imaging Data Commons](https://imaging.datacommons.cancer.gov/)
