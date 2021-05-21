"""Errors raised by highdicom processes."""


class DicomAttributeError(Exception):
    """DICOM standard compliance error.

    Exception indicating that a user-provided DICOM dataset is not in
    compliance with the DICOM standard due to a missing attribute.

    """
    pass
