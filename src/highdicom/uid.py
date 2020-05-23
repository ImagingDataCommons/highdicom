import logging
from typing import Optional

import pydicom

logger = logging.getLogger(__name__)


def generate_uid() -> str:
    '''Generates a unique DICOM identifier using a highdicom specific
    UID prefix.

    Returns
    -------
    str
        unique identifier

    '''
    prefix = '1.2.826.0.1.3680043.10.511.3.'
    return pydicom.uid.generate_uid(prefix=prefix)
