from collections.abc import Sequence
from copy import deepcopy
import logging
from typing import Union
from typing_extensions import Self

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
        scheme_version: str | None = None
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
        super().__init__()
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

    def __eq__(self, other: object) -> bool:
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
        if isinstance(other, (Code, CodedConcept)):
            this = Code(
                self.value,
                self.scheme_designator,
                self.meaning,
                self.scheme_version
            )
            return Code.__eq__(this, other)
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
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
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True
    ) -> Self:
        """Construct a CodedConcept from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a coded concept.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

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
        code_value_kws = ['CodeValue', 'LongCodeValue', 'URNCodeValue']
        num_code_values = sum(hasattr(dataset, kw) for kw in code_value_kws)
        if num_code_values != 1:
            raise AttributeError(
                'Dataset should have exactly one of the following attributes: '
                f'{", ".join(code_value_kws)}.'
             )
        for kw in ['CodeMeaning', 'CodingSchemeDesignator']:
            if not hasattr(dataset, kw):
                raise AttributeError(
                    'Dataset does not contain the following attribute '
                    f'required for coded concepts: {kw}.'
                )
        if copy:
            concept = deepcopy(dataset)
        else:
            concept = dataset
        concept.__class__ = cls
        return concept

    @classmethod
    def from_code(cls, code: Union[Code, 'CodedConcept']) -> Self:
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
    def scheme_version(self) -> str | None:
        """Union[str, None]: version of the coding scheme (if specified)"""
        return getattr(self, 'CodingSchemeVersion', None)


class _ModifiedCodedConcept(CodedConcept):

    """Base class for Coded Concepts that have a modifier sequence.

    The name of the modifier sequence varies depending on context and is
    controlled by a class variable.

    """
    _MODIFIER_SEQUENCE_NAME: str  # overrider on subclasses

    def __init__(
        self,
        coded_concept: Code | CodedConcept,
        modifiers: Sequence[Code | CodedConcept] | None = None,
    ):
        """

        Parameters
        ----------
        coded_concept: pydicom.sr.coding.Code | highdicom.sr.CodedConcept
            The root coded concept. The class docstring gives the appropriate
            context group. If this is an instance of the same class, its
            modifiers are carried over (and any further `modifiers` are
            appended to them). Instances of other modified classes are not
            accepted, since their modifiers belong to a different context.
        modifiers: Sequence[pydicom.sr.coding.Code | highdicom.sr.CodedConcept] | None
            Modifier codes that optionally describe characterstics of the root
            concept. The class docstring gives the appropriate context group.

        Raises
        ------
        TypeError:
            If `coded_concept` is an instance of a different modified coded
            concept class, or if `coded_concept` or any item of `modifiers`
            has an invalid type.

        """  # noqa: E501
        if not isinstance(coded_concept, (Code, CodedConcept)):
            raise TypeError(
                "Argument 'coded_concept' must be either "
                "pydicom.sr.coding.Code or highdicom.sr.CodedConcept."
            )

        if (
            isinstance(coded_concept, _ModifiedCodedConcept) and not
            isinstance(coded_concept, type(self))
        ):
            raise TypeError(
                "Argument 'coded_concept' must not be an instance of "
                f"{type(coded_concept).__name__} since this would discard "
                "its modifiers."
            )

        super().__init__(
            value=coded_concept.value,
            scheme_designator=coded_concept.scheme_designator,
            meaning=coded_concept.meaning,
            scheme_version=coded_concept.scheme_version,
        )

        checked_modifiers: list[CodedConcept] = []
        if isinstance(coded_concept, _ModifiedCodedConcept):
            checked_modifiers.extend(coded_concept.modifiers)

        for modifier in (modifiers or []):
            if isinstance(modifier, Code):
                checked_modifiers.append(CodedConcept.from_code(modifier))
            elif isinstance(modifier, CodedConcept):
                checked_modifiers.append(modifier)
            else:
                raise TypeError(
                    "Items of 'modifiers' must be either "
                    "pydicom.sr.coding.Code or highdicom.sr.CodedConcept."
                )

        if len(checked_modifiers) > 0:
            setattr(self, self._MODIFIER_SEQUENCE_NAME, checked_modifiers)

    @classmethod
    def from_code(cls, code: Union[Code, 'CodedConcept']) -> Self:
        """Construct a _ModifiedCodedConcept for a pydicom Code.

        Parameters
        ----------
        code: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept]
            Code.

        Returns
        -------
        Self:
            CodedConcept dataset for the code.

        """
        if isinstance(code, cls):
            return code
        return cls(code)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True
    ) -> Self:
        """Construct a modified Coded Concept from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a coded concept.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        Self:
            Coded concept representation of the dataset.

        Raises
        ------
        TypeError:
            If the passed dataset is not a pydicom dataset.
        AttributeError:
            If the dataset does not contain the required elements for a
            coded concept.

        """
        concept = super().from_dataset(dataset, copy=copy)

        if hasattr(concept, cls._MODIFIER_SEQUENCE_NAME):
            modifiers = [
                # no need for a further copy
                CodedConcept.from_dataset(m, copy=False)
                for m in getattr(concept, cls._MODIFIER_SEQUENCE_NAME)
            ]
            setattr(concept, cls._MODIFIER_SEQUENCE_NAME, modifiers)

        return concept

    @property
    def unmodified(self) -> CodedConcept:
        """highdicom.sr.CodedConcept: The coded concept with any modifiers
        removed.

        """
        return CodedConcept(
            value=self.value,
            scheme_designator=self.scheme_designator,
            meaning=self.meaning,
            scheme_version=self.scheme_version,
        )

    @property
    def modifiers(self) -> tuple[CodedConcept, ...]:
        """tuple[highdicom.sr.CodedConcept, ...]: tuple of concept modifiers
        """
        if not hasattr(self, self._MODIFIER_SEQUENCE_NAME):
            return ()

        # No copy is needed: this is an in-place (and idempotent) class change
        # of items that already represent coded concepts
        return tuple(
            CodedConcept.from_dataset(ds, copy=False) for ds in
            getattr(self, self._MODIFIER_SEQUENCE_NAME)
        )

    def __hash__(self) -> int:
        parts = [
            self.scheme_designator + self.value,
            *[
                mod.scheme_designator + mod.value
                for mod in self.modifiers
            ]
        ]
        # Sort so that order is not significant
        parts = sorted(parts)

        return hash(''.join(parts))

    def __eq__(self, other: object) -> bool:
        """Compares `self` and `other` for equality.

        Two modified codes are considered equal if the name of the sequence
        storing the modifiers is the same, and the unmodified code and all
        modifiers are the same. The order of modifiers is not significant for
        determining equality.

        Parameters
        ----------
        other: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, Self]
            code

        Returns
        -------
        bool
            whether `self` and `other` are considered equal

        """
        # Allow equality with the same modified subclass, CodedConcept or code,
        # but not another subclass of _ModifiedCodedConcept
        if (
            isinstance(other, _ModifiedCodedConcept) and not
            isinstance(other, self.__class__)
        ):
            return False

        if isinstance(other, (Code, CodedConcept)):
            this = Code(
                self.value,
                self.scheme_designator,
                self.meaning,
                self.scheme_version
            )
            self_modifiers = self.modifiers
            other_modifiers = (
                other.modifiers if
                isinstance(other, _ModifiedCodedConcept) else ()
            )

            if len(self_modifiers) != len(other_modifiers):
                return False

            # Sort in order to give permutation invariance
            sorted_self_modifiers = sorted(
                self_modifiers,
                key=lambda x: (x.scheme_designator, x.value)
            )
            sorted_other_modifiers = sorted(
                other_modifiers,
                key=lambda x: (x.scheme_designator, x.value)
            )

            return (
                Code.__eq__(this, other) and
                all(
                    a == b for a, b in
                    zip(sorted_self_modifiers, sorted_other_modifiers)
                )
            )

        return super().__eq__(other)
