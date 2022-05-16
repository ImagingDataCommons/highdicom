import logging
from typing import Any, Optional, Union

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
        scheme_version: Union[str, None], optional
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
        if len(meaning) > 64:
            raise ValueError('Code meaning can have maximally 64 characters.')
        self.CodeMeaning = str(meaning)
        self.CodingSchemeDesignator = str(scheme_designator)
        if scheme_version is not None:
            self.CodingSchemeVersion = str(scheme_version)
        # TODO: Enhanced Code Sequence Macro Attributes

    def __hash__(self) -> int:
        return hash(self.scheme_designator + self.value)

    def __eq__(self, other: Any) -> bool:
        """Compares `self` and `other` for equality.

        Parameters
        ----------
        other: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            code

        Returns
        -------
        bool
            whether `self` and `other` are considered equal

        """
        this = Code(
            self.value,
            self.scheme_designator,
            self.meaning,
            self.scheme_version
        )
        return Code.__eq__(this, other)

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

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'CodedConcept':
        """Construct a CodedConcept from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a coded concept.

        Returns
        -------
        highdicom.sr.CodedConcept:
            Coded concept representation of the dataset.

        Raises
        ------
        TypeError:
            If the passed dataset is not a pydicom dataset.
        AttributeError:
            If the dataset does not contain the required elements for a
            coded concept.

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be a pydicom.dataset.Dataset.'
            )
        for kw in ['CodeValue', 'CodeMeaning', 'CodingSchemeDesignator']:
            if not hasattr(dataset, kw):
                raise AttributeError(
                    'Dataset does not contain the following attribute '
                    f'required for coded concepts: {kw}.'
                )
        return cls(
            value=dataset.CodeValue,
            scheme_designator=dataset.CodingSchemeDesignator,
            meaning=dataset.CodeMeaning,
            scheme_version=getattr(dataset, 'CodingSchemeVersion', None)
        )

    @classmethod
    def from_code(cls, code: Union[Code, 'CodedConcept']) -> 'CodedConcept':
        """Construct a CodedConcept for a pydicom Code.

        Parameters
        ----------
        code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Code.

        Returns
        -------
        highdicom.sr.CodedConcept:
            CodedConcept dataset for the code.

        """
        if isinstance(code, cls):
            return code
        return cls(*code)

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
    def scheme_version(self) -> Optional[str]:
        """Union[str, None]: version of the coding scheme (if specified)"""
        return getattr(self, 'CodingSchemeVersion', None)
