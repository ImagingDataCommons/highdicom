import logging

import pydicom

logger = logging.getLogger(__name__)


class UID(pydicom.uid.UID):

    """Unique DICOM identifier with a highdicom-specific UID prefix."""

    def __new__(cls: type) -> str:
        prefix = '1.2.826.0.1.3680043.10.511.3.'
        identifier = pydicom.uid.generate_uid(prefix=prefix)
        return super().__new__(cls, identifier)
