"""Content that is specific to Key Object Selection IODs."""
from typing import cast, List, Optional, Sequence, Union

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes

from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import RelationshipTypeValues, ValueTypeValues
from highdicom.sr.value_types import (
    ContainerContentItem,
    ContentSequence,
    TextContentItem,
    ImageContentItem,
    CompositeContentItem,
    WaveformContentItem,
)
from highdicom.sr.templates import (
    DeviceObserverIdentifyingAttributes,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
)


class KeyObjectSelection(ContentSequence):

    """Sequence of structured reporting content item describing a selection
    of DICOM objects according to structured reporting template
    :dcm:`TID 2010 Key Object Selection <part16/chapter_C.html#sect_TID_2010>`.
    """

    def __init__(
        self,
        document_title: Union[Code, CodedConcept],
        referenced_objects: Sequence[Dataset],
        observer_person_context: Optional[ObserverContext] = None,
        observer_device_context: Optional[ObserverContext] = None,
        description: Optional[str] = None
    ):
        """
        Parameters
        ----------
        document_title: Union[pydicom.sr.coding.Code, highdicom.srCodedConcept]
            Coded title of the document
            (see :dcm:`CID 7010 <part16/sect_CID_7010.html>`)
        referenced_objects: Sequence[pydicom.dataset.Dataset]
            Metadata of selected objects that should be referenced
        observer_person_context: Union[highdicom.sr.ObserverContext, None], optional
            Observer context describing the person that selected the objects
        observer_device_context: Union[highdicom.sr.ObserverContext, None], optional
            Observer context describing the device that selected the objects
        description: Union[str, None], optional
            Description of the selected objects

        """  # noqa: E501
        super().__init__(is_root=True)
        item = ContainerContentItem(
            name=document_title,  # CID 7010
            template_id='2010'
        )
        item.ContentSequence = ContentSequence()

        if observer_person_context is not None:
            if not isinstance(observer_person_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_person_context" must have type '
                    'ObserverContext.'
                )
            if observer_person_context.observer_type != codes.DCM.Person:
                raise ValueError(
                    'Argument "observer_person_context" must have Observer '
                    'Type "Person".'
                )
            item.ContentSequence.extend(observer_person_context)
        if observer_device_context is not None:
            if not isinstance(observer_device_context, ObserverContext):
                raise TypeError(
                    'Argument "observer_device_context" must have type '
                    'ObserverContext.'
                )
            if observer_device_context.observer_type != codes.DCM.Device:
                raise ValueError(
                    'Argument "observer_device_context" must have Observer '
                    'Type "Device".'
                )
            item.ContentSequence.extend(observer_device_context)

        if description is not None:
            description_item = TextContentItem(
                name=CodedConcept(
                    value='113012',
                    scheme_designator='DCM',
                    meaning='Key Object Description'
                ),
                value=description,
                relationship_type=RelationshipTypeValues.CONTAINS
            )
            item.ContentSequence.append(description_item)

        if len(referenced_objects) == 0:
            raise ValueError('At least one object must be referenced.')

        # PS3.3 C.17.3 SR Document Content Module
        # Though many Templates in PS3.16 do not require that the Purpose of
        # Reference be conveyed in the Concept Name, a generic Concept Name,
        # such as (260753009, SCT, "Source"), may be used, since anonymous
        # (unnamed) Content Items may be undesirable for some implementations
        # (e.g., for which the name of a name-value pair is required).
        name = CodedConcept(
            value='260753009',
            scheme_designator='SCT',
            meaning='Source',
        )
        for ds in referenced_objects:
            reference_item: Union[ImageContentItem, CompositeContentItem]
            if 'Rows' in ds and 'Columns' in ds:
                reference_item = ImageContentItem(
                    name=name,
                    referenced_sop_class_uid=ds.SOPClassUID,
                    referenced_sop_instance_uid=ds.SOPInstanceUID,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                item.ContentSequence.append(reference_item)
            else:
                reference_item = CompositeContentItem(
                    name=name,
                    referenced_sop_class_uid=ds.SOPClassUID,
                    referenced_sop_instance_uid=ds.SOPInstanceUID,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
                item.ContentSequence.append(reference_item)

        self.append(item)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Dataset],
        is_root: bool = True
    ) -> 'KeyObjectSelection':
        """Construct object from a sequence of datasets.

        Parameters
        ----------
        sequence: Sequence[pydicom.dataset.Dataset]
            Datasets representing "Key Object Selection" SR Content Items
            of Value Type CONTAINER (sequence shall only contain a single item)
        is_root: bool, optional
            Whether the sequence is used to contain SR Content Items that are
            intended to be added to an SR document at the root of the document
            content tree

        Returns
        -------
        highdicom.ko.KeyObjectSelection
            Content Sequence containing root CONTAINER SR Content Item

        """
        if len(sequence) == 0:
            raise ValueError('Sequence contains no SR Content Items.')
        if len(sequence) > 1:
            raise ValueError(
                'Sequence contains more than one SR Content Item.'
            )
        dataset = sequence[0]
        if dataset.ValueType != ValueTypeValues.CONTAINER.value:
            raise ValueError(
                'Item #1 of sequence is not an appropriate SR Content Item '
                'because it does not have Value Type CONTAINER.'
            )
        if dataset.ContentTemplateSequence[0].TemplateIdentifier != '2010':
            raise ValueError(
                'Item #1 of sequence is not an appropriate SR Content Item '
                'because it does not have Template Identifier "2010".'
            )
        instance = ContentSequence.from_sequence(sequence, is_root=True)
        instance.__class__ = KeyObjectSelection
        return cast(KeyObjectSelection, instance)

    def get_observer_contexts(
        self,
        observer_type: Optional[Union[CodedConcept, Code]] = None
    ) -> List[ObserverContext]:
        """Get observer contexts.

        Parameters
        ----------
        observer_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            Type of observer ("Device" or "Person") for which should be filtered

        Returns
        -------
        List[highdicom.sr.ObserverContext]
            Observer contexts

        """  # noqa: E501
        root_item = self[0]
        matches = [
            (i, item) for i, item in enumerate(root_item.ContentSequence, 1)
            if item.name == codes.DCM.ObserverType
        ]
        observer_contexts = []
        attributes: Union[
            DeviceObserverIdentifyingAttributes,
            PersonObserverIdentifyingAttributes,
        ]
        for i, (index, item) in enumerate(matches):
            if observer_type is not None:
                if item.value != observer_type:
                    continue
            try:
                next_index = matches[i + 1][0]
            except IndexError:
                next_index = -1
            if item.value == codes.DCM.Device:
                attributes = DeviceObserverIdentifyingAttributes.from_sequence(
                    sequence=root_item.ContentSequence[index:next_index]
                )
            elif item.value == codes.DCM.Person:
                attributes = PersonObserverIdentifyingAttributes.from_sequence(
                    sequence=root_item.ContentSequence[index:next_index]
                )
            else:
                raise ValueError('Unexpected observer type "{item.meaning}".')
            context = ObserverContext(
                observer_type=item.value,
                observer_identifying_attributes=attributes
            )
            observer_contexts.append(context)
        return observer_contexts

    def get_references(
        self,
        value_type: Optional[ValueTypeValues] = None,
        sop_class_uid: Optional[str] = None
    ) -> List[
        Union[ImageContentItem, CompositeContentItem, WaveformContentItem]
    ]:
        """Get referenced objects.

        Parameters
        ----------
        value_type: Union[highdicom.sr.ValueTypeValues, None], optional
            Value type of content items that reference objects
        sop_class_uid: Union[str, None], optional
            SOP Class UID of referenced object

        Returns
        -------
        List[Union[highdicom.sr.ImageContentItem, highdicom.sr.CompositeContentItem, highdicom.sr.WaveformContentItem]]
            Content items that reference objects

        """  # noqa: E501
        supported_value_types = {
            ValueTypeValues.IMAGE,
            ValueTypeValues.COMPOSITE,
            ValueTypeValues.WAVEFORM,
        }
        if value_type is not None:
            value_type = ValueTypeValues(value_type)
            if value_type not in supported_value_types:
                raise ValueError(
                    f'Value type "{value_type.value}" is not supported for '
                    'referencing selected objects.'
                )
            expected_value_types = {value_type}
        else:
            expected_value_types = supported_value_types
        return [
            item for item in self[0].ContentSequence
            if (item.value_type in expected_value_types) and (
                sop_class_uid is None or (
                    item.referenced_sop_class_uid == sop_class_uid
                )
            )
        ]
