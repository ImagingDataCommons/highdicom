from typing import Optional, Sequence

from pydicom.dataset import Dataset


class CodingSchemeResourceItem(Dataset):

    """Class for items of the Coding Scheme Resource Sequence."""

    def __init__(self, url: str, url_type: str) -> None:
        """
        Parameters
        ----------
        url: str
            unique resource locator
        url_type: str
            type of resource `url` points to (options: `{"DOC", "OWL", "CSV"}`)

        """
        super().__init__()
        self.CodingSchemeURL = str(url)
        if url_type not in {"DOC", "OWL", "CSV"}:
            raise ValueError('Unknonw URL type.')
        self.CodingSchemeURLType = str(url_type)


class CodingSchemeIdentificationItem(Dataset):

    """Class for items of the Coding Scheme Identification Sequence."""

    def __init__(
        self,
        designator: str,
        name: Optional[str] = None,
        version: Optional[str] = None,
        registry: Optional[str] = None,
        uid: Optional[str] = None,
        external_id: Optional[str] = None,
        responsible_organization: Optional[str] = None,
        resources: Optional[Sequence[CodingSchemeResourceItem]] = None
    ) -> None:
        """
        Parameters
        ----------
        designator: str
            value of the Coding Scheme Designator attribute of a `CodedConcept`
        name: str, optional
            name of the scheme
        version: str, optional
            version of the scheme
        registry: str, optional
            name of an external registry where scheme may be obtained from;
            required if scheme is registered
        uid: str, optional
            unique identifier of the scheme; required if the scheme is
            registered by an ISO 8824 object identifier compatible with the
            UI value representation (VR)
        external_id: str, optional
            external identifier of the scheme; required if the scheme is
            registered and `uid` is not available
        responsible_organization: str, optional
            name of the organization that is responsible for the scheme
        resources: Sequence[pydicom.sr.coding.CodingSchemeResourceItem], optional
            one or more resources related to the scheme

        """  # noqa: E501
        super().__init__()
        self.CodingSchemeDesignator = str(designator)
        if name is not None:
            self.CodingSchemeName = str(name)
        if version is not None:
            self.CodingSchemeVersion = str(version)
        if responsible_organization is not None:
            self.CodingSchemeResponsibleOrganization = \
                str(responsible_organization)
        if registry is not None:
            self.CodingSchemeRegistry = str(registry)
            if uid is None and external_id is None:
                raise ValueError(
                    'UID or external ID is required if coding scheme is '
                    'registered.'
                )
            if uid is not None and external_id is not None:
                raise ValueError(
                    'Either UID or external ID should be specified for '
                    'registered coding scheme.'
                )
            if uid is not None:
                self.CodingSchemeUID = str(uid)
            elif external_id is not None:
                self.CodingSchemeExternalID = str(external_id)
        if resources is not None:
            self.CodingSchemeResourcesSequence: Sequence[Dataset] = []
            for r in resources:
                if not isinstance(r, CodingSchemeResourceItem):
                    raise TypeError(
                        'Resources must have type CodingSchemeResourceItem.'
                    )
                self.CodingSchemeResourcesSequence.append(r)
