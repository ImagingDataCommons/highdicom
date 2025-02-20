import logging
from uuid import UUID
from typing import TypeVar
from typing_extensions import Self

import pydicom

logger = logging.getLogger(__name__)


T = TypeVar('T', bound='UID')


class UID(pydicom.uid.UID):

    """Unique DICOM identifier.

    If an object is constructed without a value being provided, a value will be
    automatically generated using the highdicom-specific root.
    """

    def __new__(cls: type[T], value: str | None = None) -> T:
        if value is None:
            prefix = '1.2.826.0.1.3680043.10.511.3.'
            value = pydicom.uid.generate_uid(prefix=prefix)
        return super().__new__(cls, value)

    @classmethod
    def from_uuid(cls, uuid: str) -> Self:
        """Create a DICOM UID from a UUID using the 2.25 root.

        Parameters
        ----------
        uuid: str
            UUID

        Returns
        -------
        highdicom.UID
            UID

        Examples
        --------
        >>> from uuid import uuid4
        >>> import highdicom as hd
        >>> uuid = str(uuid4())
        >>> uid = hd.UID.from_uuid(uuid)

        """
        value = f'2.25.{UUID(uuid).int}'
        return cls(value)
