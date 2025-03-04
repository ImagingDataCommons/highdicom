[![Build Status](https://github.com/imagingdatacommons/highdicom/actions/workflows/run_unit_tests.yml/badge.svg)](https://github.com/imagingdatacommons/highdicom/actions)
[![Documentation Status](https://readthedocs.org/projects/highdicom/badge/?version=latest)](https://highdicom.readthedocs.io/en/latest/?badge=latest)
[![PyPi Distribution](https://img.shields.io/pypi/v/highdicom.svg)](https://pypi.python.org/pypi/highdicom/)
[![Python Versions](https://img.shields.io/pypi/pyversions/highdicom.svg)](https://pypi.org/project/highdicom/)
[![Downloads](https://pepy.tech/badge/highdicom)](https://pepy.tech/project/highdicom)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

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
* Softcopy Presentation State instances (including Grayscale, Color, and Pseudo-Color)

## Documentation

Please refer to the online documentation at [highdicom.readthedocs.io](https://highdicom.readthedocs.io), which includes installation instructions, a user guide with examples, a developer guide, and complete documentation of the application programming interface of the `highdicom` package.

## Citation

For more information about the motivation of the library and the design of highdicom's API, please see the following article:

> [Highdicom: A Python library for standardized encoding of image annotations and machine learning model outputs in pathology and radiology](https://link.springer.com/article/10.1007/s10278-022-00683-y)
> C.P. Bridge, C. Gorman, S. Pieper, S.W. Doyle, J.K. Lennerz, J. Kalpathy-Cramer, D.A. Clunie, A.Y. Fedorov, and M.D. Herrmann.
> Journal of Digital Imaging, August 2022

If you use highdicom in your research, please cite the above article.

## Support

The developers gratefully acknowledge their support:
* The [Alliance for Digital Pathology](https://digitalpathologyalliance.org/)
* The [MGH & BWH Center for Clinical Data Science](https://www.ccds.io/)
* [Quantitative Image Informatics for Cancer Research (QIICR)](https://qiicr.org/)
* [Radiomics](https://www.radiomics.io/)

This software is maintained in part by the [NCI Imaging Data Commons](https://imaging.datacommons.cancer.gov/) project, 
which has been funded in whole or in part with Federal funds from the NCI, NIH, under task order no. HHSN26110071
under contract no. HHSN261201500003l. 
