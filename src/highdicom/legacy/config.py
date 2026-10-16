"""Configuration classes for legacy attribute harmonization."""
from collections.abc import Callable, Sequence
from inspect import signature

from pydicom.tag import BaseTag
from pydicom.datadict import keyword_for_tag, tag_for_keyword


class HarmonizationConfig:

    """Configuration for harmonizing an attribute.

    """

    def __init__(
        self,
        destination_keyword: str,
        destination_sequence_keyword: str,
        source_tags: Sequence[BaseTag],
        private_creator_name: str | None = None,
        mapping: Callable | None = None
    ) -> None:
        """

        Parameters
        ----------
        destination_keyword: str
            DICOM keyword of the (standard) attribute that the result will be
            placed into.
        destination_sequence_keyword: str
            DICOM keyword of the (standard) sequence that the resulting
            attribute will be placed into.
        source_tags: Sequence[pydicom.tag.BaseTag]
            List of tags for attributes that are required in the legacy images
            in order to determine the value for the destination attribute. Tags
            may be either standard (even group number) or private (odd group
            number), and the two types may be mixed. However, if any tag is
            private, ``private_creator_name`` must be provided and must apply
            to all private tags in this list.
        private_creator_name: str | None, optional
            Name of the private creator of the source attributes, if any are
            private. May be omitted if all items of ``source_tags`` are
            standard.
        mapping: Callable | None, optional
            A callable that takes as input a list containing the values of the
            attributes defined in ``source_tags``, with order corresponding
            between the two lists, and returns the output value for the
            destination attribute. If this parameter is ``None`` (the default),
            the ``source_tags`` list must contain a single item, and the value
            of that attribute is used as-is for placement into the output
            attribute.

        """  # noqa: E501
        if tag_for_keyword(destination_keyword) is None:
            raise ValueError(
                f"{destination_keyword} is not a valid DICOM keyword."
            )

        if len(source_tags) == 0:
            raise ValueError(
                "'source_tags' may not be empty."
            )
        if not all(isinstance(t, BaseTag) for t in source_tags):
            raise TypeError(
                "Items of 'source_tags' must be of type pydicom.BaseTag"
            )

        # Mask the variable position bits of private tags
        source_tags = [
            BaseTag(t & 0xFFFF_00FF) if t.is_private else t
            for t in source_tags
        ]

        for t in source_tags:
            if not t.is_private and keyword_for_tag(t) is None:
                raise ValueError(f"Unrecognized standard tag: {t}")

        if (
            private_creator_name is None and
            any(t.is_private for t in source_tags)
        ):
            raise TypeError(
                "'private_creator_name' must be specified because one or more "
                "tags are private."
            )

        if mapping is None:
            if len(source_tags) is not None:
                raise TypeError(
                    "'mapping' must be provided when multiple source tags are "
                    "specified."
                )
        else:
            if not isinstance(mapping, Callable):
                raise TypeError(
                    "'mapping' must be a callable object"
                )

            if len(signature(mapping).parameters) != len(source_tags):
                raise TypeError(
                    "Number of parameters taken by 'mapping' must match number "
                    "of items in 'source_tags'"
                )

        self.destination_keyword = destination_keyword
        self.destination_sequence_keyword = destination_sequence_keyword
        self.source_tags = source_tags
        self.private_creator_name = private_creator_name
        self.mapping = mapping
