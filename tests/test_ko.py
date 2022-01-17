from io import BytesIO
import unittest
from pathlib import Path

from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
import pytest

from highdicom.ko.content import KeyObjectSelection
from highdicom.ko.sop import KeyObjectSelectionDocument
from highdicom.sr.enum import ValueTypeValues
from highdicom.sr.templates import (
    DeviceObserverIdentifyingAttributes,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
)
from highdicom.sr.value_types import (
    ContainerContentItem,
    ContentSequence,
    CompositeContentItem,
    ImageContentItem,
)
from highdicom.uid import UID


class TestKeyObjectSelection(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._document_title = codes.DCM.Manifest

        self._sm_object = Dataset()
        self._sm_object.Modality = 'SM'
        self._sm_object.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'
        self._sm_object.SOPInstanceUID = UID()
        self._sm_object.Rows = 512
        self._sm_object.Columns = 512

        self._seg_object = Dataset()
        self._seg_object.Modality = 'SEG'
        self._seg_object.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
        self._seg_object.SOPInstanceUID = UID()
        self._seg_object.Rows = 512
        self._seg_object.Columns = 512

        self._pm_object = Dataset()
        self._pm_object.Modality = 'OT'
        self._pm_object.SOPClassUID = '1.2.840.10008.5.1.4.1.1.30'
        self._pm_object.SOPInstanceUID = UID()
        self._pm_object.Rows = 512
        self._pm_object.Columns = 512

        self._sr_object = Dataset()
        self._sr_object.Modality = 'SR'
        self._sr_object.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.34'
        self._sr_object.SOPInstanceUID = UID()

        self._observer_person_context = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name='Foo^Bar'
            )
        )
        self._observer_device_context = ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
                uid=UID(),
                name='Device'
            )
        )

    def test_construction(self):
        content = KeyObjectSelection(
            document_title=self._document_title,
            referenced_objects=[
                self._sm_object,
                self._seg_object,
                self._pm_object,
                self._sr_object,
            ],
            observer_person_context=self._observer_person_context,
            observer_device_context=self._observer_device_context,
            description='Special Selection'
        )
        assert isinstance(content, KeyObjectSelection)
        assert isinstance(content, ContentSequence)
        assert len(content) == 1
        container = content[0]
        assert isinstance(container, ContainerContentItem)
        assert container.ContentTemplateSequence[0].TemplateIdentifier == '2010'
        # Oberver Context (Person): 2
        # Oberver Context (Device): 3
        # Description: 1
        # Referenced Objects: 4
        assert len(container.ContentSequence) == 10
        observer_type = container.ContentSequence[0]
        assert observer_type.name == codes.DCM.ObserverType
        assert observer_type.value == codes.DCM.Person
        observer_type = container.ContentSequence[2]
        assert observer_type.name == codes.DCM.ObserverType
        assert observer_type.value == codes.DCM.Device
        sm_reference = container.ContentSequence[6]
        assert isinstance(sm_reference, ImageContentItem)
        seg_reference = container.ContentSequence[7]
        assert isinstance(seg_reference, ImageContentItem)
        pm_reference = container.ContentSequence[8]
        assert isinstance(pm_reference, ImageContentItem)
        sr_reference = container.ContentSequence[9]
        assert isinstance(sr_reference, CompositeContentItem)

        observer_contexts = content.get_observer_contexts()
        assert len(observer_contexts) == 2
        observer_contexts = content.get_observer_contexts(
            observer_type=codes.DCM.Person
        )
        assert len(observer_contexts) == 1
        observer_contexts = content.get_observer_contexts(
            observer_type=codes.DCM.Device
        )
        assert len(observer_contexts) == 1

        references = content.get_references()
        assert len(references) == 4
        references = content.get_references(
            value_type=ValueTypeValues.IMAGE
        )
        assert len(references) == 3
        references = content.get_references(
            value_type=ValueTypeValues.COMPOSITE
        )
        assert len(references) == 1
        references = content.get_references(
            value_type=ValueTypeValues.WAVEFORM
        )
        assert len(references) == 0

    def test_construction_with_missing_parameter(self):
        with pytest.raises(TypeError):
            KeyObjectSelection(
                document_title=self._document_title
            )

    def test_construction_with_wrong_parameter_value(self):
        with pytest.raises(ValueError):
            KeyObjectSelection(
                document_title=self._document_title,
                referenced_objects=[]
            )


class TestKeyObjectSelectionDocument(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')

        self._sm_image = dcmread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )
        self._seg_image = dcmread(
            str(data_dir.joinpath('test_files', 'seg_image_sm_control.dcm'))
        )

        self._evidence = [
            self._sm_image,
            self._seg_image,
        ]

        self._content = KeyObjectSelection(
            document_title=codes.DCM.Manifest,
            referenced_objects=[
                self._sm_image,
                self._seg_image,
            ]
        )

    def test_construction(self):
        document = KeyObjectSelectionDocument(
            evidence=self._evidence,
            content=self._content,
            series_instance_uid=UID(),
            series_number=10,
            sop_instance_uid=UID(),
            instance_number=1,
            manufacturer='MGH Computational Pathology',
            institution_name='Massachusetts General Hospital',
            institutional_department_name='Pathology'
        )
        assert isinstance(document, KeyObjectSelectionDocument)
        assert isinstance(document.content, KeyObjectSelection)

        assert document.Modality == 'KO'
        assert hasattr(document, 'CurrentRequestedProcedureEvidenceSequence')
        assert len(document.CurrentRequestedProcedureEvidenceSequence) > 0
        assert hasattr(document, 'ReferencedPerformedProcedureStepSequence')

        study_uid, series_uid, instance_uid = document.resolve_reference(
            self._sm_image.SOPInstanceUID
        )
        assert study_uid == self._sm_image.StudyInstanceUID
        assert series_uid == self._sm_image.SeriesInstanceUID
        assert instance_uid == self._sm_image.SOPInstanceUID

    def test_construction_from_dataset(self):
        document = KeyObjectSelectionDocument(
            evidence=self._evidence,
            content=self._content,
            series_instance_uid=UID(),
            series_number=10,
            sop_instance_uid=UID(),
            instance_number=1,
            manufacturer='MGH Computational Pathology',
            institution_name='Massachusetts General Hospital',
            institutional_department_name='Pathology'
        )
        assert isinstance(document, KeyObjectSelectionDocument)
        assert isinstance(document.content, KeyObjectSelection)

        with BytesIO() as fp:
            document.save_as(fp)
            fp.seek(0)
            document_reread = dcmread(fp)

        test_document = KeyObjectSelectionDocument.from_dataset(document_reread)
        assert isinstance(test_document, KeyObjectSelectionDocument)
        assert isinstance(test_document.content, KeyObjectSelection)
        assert test_document.Modality == 'KO'
        assert hasattr(
            test_document, 'CurrentRequestedProcedureEvidenceSequence'
        )
        assert len(test_document.CurrentRequestedProcedureEvidenceSequence) > 0
        assert hasattr(
            test_document, 'ReferencedPerformedProcedureStepSequence'
        )

        study_uid, series_uid, instance_uid = test_document.resolve_reference(
            self._sm_image.SOPInstanceUID
        )
        assert study_uid == self._sm_image.StudyInstanceUID
        assert series_uid == self._sm_image.SeriesInstanceUID
        assert instance_uid == self._sm_image.SOPInstanceUID
