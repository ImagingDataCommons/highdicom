import logging
from typing import Any, Optional, Sequence, Union

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code

logger = logging.getLogger(__name__)


class CodedConcept(Dataset):

    """Coded concept of a DICOM SR document content module attribute."""

    def __init__(
            self,
            value: str,
            scheme_designator: str,
            meaning: str,
            scheme_version: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        value: str
            code
        scheme_designator: str
            designator of coding scheme
        meaning: str
            meaning of the code
        scheme_version: str, optional
            version of coding scheme

        """
        super(CodedConcept, self).__init__()
        if len(value) > 16:
            if value.startswith('urn') or '://' in value:
                self.URNCodeValue = str(value)
            else:
                self.LongCodeValue = str(value)
        else:
            self.CodeValue = str(value)
        self.CodeMeaning = str(meaning)
        self.CodingSchemeDesignator = str(scheme_designator)
        if scheme_version is not None:
            self.CodingSchemeVersion = str(scheme_version)
        # TODO: Enhanced Code Sequence Macro Attributes

    def __eq__(self, other: Any) -> bool:
        """Compares `self` and `other` for equality.

        Parameters
        ----------
        other: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code]
            code

        Returns
        -------
        bool
            whether `self` and `other` are considered equal

        """
        return Code.__eq__(self, other)

    def __ne__(self, other: Any) -> bool:
        """Compares `self` and `other` for inequality.

        Parameters
        ----------
        other: Union[CodedConcept, pydicom.sr.coding.Code]
            code

        Returns
        -------
        bool
            whether `self` and `other` are not considered equal

        """
        return not (self == other)

    @property
    def value(self) -> str:
        """str: value of either `CodeValue`, `LongCodeValue` or `URNCodeValue`
        attribute"""
        return getattr(
            self, 'CodeValue',
            getattr(
                self, 'LongCodeValue',
                getattr(
                    self, 'URNCodeValue',
                    None
                )
            )
        )

    @property
    def meaning(self) -> str:
        """str: meaning of the code"""
        return self.CodeMeaning

    @property
    def scheme_designator(self) -> str:
        """str: designator of the coding scheme (e.g. ``"DCM"``)"""

        return self.CodingSchemeDesignator

    @property
    def scheme_version(self) -> str:
        """Union[str, None]: version of the coding scheme (if specified)"""
        return getattr(self, 'CodingSchemeVersion', None)


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

        """  # noqa
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
