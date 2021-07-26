import logging
from typing import Optional, Type, TypeVar

import pydicom

logger = logging.getLogger(__name__)


T = TypeVar('T', bound='UID')


class UID(pydicom.uid.UID):

    """Unique DICOM identifier with a highdicom-specific UID prefix."""

    def __new__(cls: Type[T], value: Optional[str] = None) -> T:
        if value is None:
            prefix = '1.2.826.0.1.3680043.10.511.3.'
            value = pydicom.uid.generate_uid(prefix=prefix)
        return super().__new__(cls, value)
