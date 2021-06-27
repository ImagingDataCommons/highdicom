"""Utilities for working with SR document instances."""
from typing import List, Optional, Union

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes

from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import ValueTypeValues, RelationshipTypeValues
from highdicom.sr.value_types import ContentItem


def find_content_items(
    dataset: Dataset,
    name: Optional[Union[CodedConcept, Code]] = None,
    value_type: Optional[Union[ValueTypeValues, str]] = None,
    relationship_type: Optional[Union[RelationshipTypeValues, str]] = None,
    recursive: bool = False
) -> List[Dataset]:
    """Finds content items in a Structured Report document that match a given
    query.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        SR document instance
    name: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code], optional
        Coded name that items should have
    value_type: Union[highdicom.sr.ValueTypeValues, str], optional
        Type of value that items should have
        (e.g. ``highdicom.sr.ValueTypeValues.CONTAINER``)
    relationship_type: Union[highdicom.sr.RelationshipTypeValues, str], optional
        Type of relationship that items should have with its parent
        (e.g. ``highdicom.sr.RelationshipTypeValues.CONTAINS``)
    recursive: bool, optional
        Whether search should be performed recursively, i.e. whether contained
        child content items should also be queried

    Returns
    -------
    List[pydicom.dataset.Dataset]
        flat list of all content items that matched the query

    Raises
    ------
    AttributeError
        When data set does not contain Content Sequence attribute.

    """  # noqa
    def has_name(item: ContentItem, name: Optional[str]) -> bool:
        if name is None:
            return True
        return item.name == name

    def has_value_type(
            item: ContentItem,
            value_type: Optional[Union[ValueTypeValues, str]]
    ) -> bool:
        if value_type is None:
            return True
        value_type = ValueTypeValues(value_type)
        return item.value_type == value_type.value

    def has_relationship_type(
            item: ContentItem,
            relationship_type: Optional[Union[RelationshipTypeValues, str]]
    ) -> bool:
        if relationship_type is None:
            return True
        if getattr(item, 'relationship_type', None) is None:
            return False
        relationship_type = RelationshipTypeValues(relationship_type)
        return item.relationship_type == relationship_type.value

    if not hasattr(dataset, 'ContentSequence'):
        raise AttributeError(
            'Data set does not contain a Content Sequence attribute.'
        )

    def search_tree(
        node: Dataset,
        name: Optional[Union[CodedConcept, Code]],
        value_type: Optional[Union[ValueTypeValues, str]],
        relationship_type: Optional[Union[RelationshipTypeValues, str]],
        recursive: bool
    ) -> List:
        matched_content_items = []
        for i, content_item in enumerate(node.ContentSequence):
            name_code = content_item.ConceptNameCodeSequence[0]
            item = ContentItem(
                value_type=content_item.ValueType,
                name=CodedConcept(
                    value=name_code.CodeValue,
                    scheme_designator=name_code.CodingSchemeDesignator,
                    meaning=name_code.CodeMeaning
                ),
                relationship_type=content_item.get('RelationshipType', None)
            )
            if (has_name(item, name) and
                    has_value_type(item, value_type) and
                    has_relationship_type(item, relationship_type)):
                matched_content_items.append(content_item)
            if hasattr(content_item, 'ContentSequence') and recursive:
                matched_content_items += search_tree(
                    node=content_item,
                    name=name,
                    value_type=value_type,
                    relationship_type=relationship_type,
                    recursive=recursive
                )
        return matched_content_items

    return search_tree(
        node=dataset,
        name=name,
        value_type=value_type,
        relationship_type=relationship_type,
        recursive=recursive
    )


def get_coded_name(item: Dataset) -> CodedConcept:
    """Gets the concept name of a SR Content Item.

    Parameters
    ----------
    item: pydicom.dataset.Dataset
        Content Item

    Returns
    -------
    highdicom.sr.CodedConcept
        Concept name

    """
    try:
        name = item.ConceptNameCodeSequence[0]
    except AttributeError:
        raise AttributeError(
            'Dataset does not contain attribute "ConceptNameCodeSequence" and '
            'thus doesn\'t represent a SR Content Item.'
        )
    return CodedConcept(
        value=name.CodeValue,
        scheme_designator=name.CodingSchemeDesignator,
        meaning=name.CodeMeaning,
        scheme_version=name.get('CodingSchemeVersion', None)
    )


def get_coded_value(item: Dataset) -> CodedConcept:
    """Gets the value of a SR Content Item with Value Type CODE.

    Parameters
    ----------
    item: pydicom.dataset.Dataset
        Content Item

    Returns
    -------
    highdicom.sr.CodedConcept
        Value

    """
    try:
        value = item.ConceptCodeSequence[0]
    except AttributeError:
        raise AttributeError(
            'Dataset does not contain attribute "ConceptCodeSequence" and '
            'thus doesn\'t represent a SR Content Item of Value Type CODE.'
        )
    return CodedConcept(
        value=value.CodeValue,
        scheme_designator=value.CodingSchemeDesignator,
        meaning=value.CodeMeaning,
        scheme_version=value.get('CodingSchemeVersion', None)
    )


HighDicomCodes = {
    "OT": Code(value='1000', scheme_designator="HIGHDICOM", meaning="Modality type OT"),
}

def get_coded_modality(sop_class_uid: str) -> Code:
    """
    Gets the coded value of the modality from the dataset's SOPClassUID. The
    SOPClassUIDs are defined here:
    `Standard SOP Classes <http://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html>`
    and the coded values are described here:
    `CID 29 Acquisition Modality <http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_29.html>`

    Parameters
    ----------
    item: pydicom.dataset.Dataset
        Content Item

    Returns
    -------
    pydicom.sr.coding.Code
        Coded Acquisition Modality
    """  # noqa: E501
    sopclass_to_modalty_map: dict[str, str] = {
        '1.2.840.10008.5.1.4.1.1.1': codes.cid29.ComputedRadiography,
        '1.2.840.10008.5.1.4.1.1.1.1': codes.cid29.DigitalRadiography,
        '1.2.840.10008.5.1.4.1.1.1.1.1': codes.cid29.DigitalRadiography,
        '1.2.840.10008.5.1.4.1.1.1.2': codes.cid29.Mammography,
        '1.2.840.10008.5.1.4.1.1.1.2.1': codes.cid29.Mammography,
        '1.2.840.10008.5.1.4.1.1.1.3': codes.cid29.IntraOralRadiography,
        '1.2.840.10008.5.1.4.1.1.1.3.1': codes.cid29.IntraOralRadiography,
        '1.2.840.10008.5.1.4.1.1.2': codes.cid29.ComputedTomography,
        '1.2.840.10008.5.1.4.1.1.2.1': codes.cid29.ComputedTomography,
        '1.2.840.10008.5.1.4.1.1.2.2': codes.cid29.ComputedTomography,
        '1.2.840.10008.5.1.4.1.1.3.1': codes.cid29.Ultrasound,
        '1.2.840.10008.5.1.4.1.1.4': codes.cid29.MagneticResonance,
        '1.2.840.10008.5.1.4.1.1.4.1': codes.cid29.MagneticResonance,
        '1.2.840.10008.5.1.4.1.1.4.2': codes.cid29.MagneticResonance,
        '1.2.840.10008.5.1.4.1.1.4.3': codes.cid29.MagneticResonance,
        '1.2.840.10008.5.1.4.1.1.4.4': codes.cid29.MagneticResonance,
        '1.2.840.10008.5.1.4.1.1.6.1': codes.cid29.Ultrasound,
        '1.2.840.10008.5.1.4.1.1.6.2': codes.cid29.Ultrasound,
        '1.2.840.10008.5.1.4.1.1.7': HighDicomCodes['OT'],
        '1.2.840.10008.5.1.4.1.1.7.1': HighDicomCodes['OT'],
        '1.2.840.10008.5.1.4.1.1.7.2': HighDicomCodes['OT'],
        '1.2.840.10008.5.1.4.1.1.7.3': HighDicomCodes['OT'],
        '1.2.840.10008.5.1.4.1.1.7.4': HighDicomCodes['OT'],
        '1.2.840.10008.5.1.4.1.1.9.1.1': codes.cid29.Electrocardiography,
        '1.2.840.10008.5.1.4.1.1.9.1.2': codes.cid29.Electrocardiography,
        '1.2.840.10008.5.1.4.1.1.9.1.3': codes.cid29.Electrocardiography,
        '1.2.840.10008.5.1.4.1.1.9.2.1': codes.cid29.HemodynamicWaveform,
        '1.2.840.10008.5.1.4.1.1.9.3.1': codes.cid29.Electrocardiography,
        '1.2.840.10008.5.1.4.1.1.9.5.1': codes.cid29.HemodynamicWaveform,
        '1.2.840.10008.5.1.4.1.1.9.6.1': codes.cid29.RespiratoryWaveform,
        '1.2.840.10008.5.1.4.1.1.12.1': codes.cid29.XRayAngiography,
        '1.2.840.10008.5.1.4.1.1.12.1.1': codes.cid29.XRayAngiography,
        '1.2.840.10008.5.1.4.1.1.12.2': codes.cid29.Radiofluoroscopy,
        '1.2.840.10008.5.1.4.1.1.12.2.1': codes.cid29.Radiofluoroscopy,
        '1.2.840.10008.5.1.4.1.1.13.1.1': codes.cid29.XRayAngiography,
        '1.2.840.10008.5.1.4.1.1.13.1.2': codes.cid29.DigitalRadiography,
        '1.2.840.10008.5.1.4.1.1.13.1.3': codes.cid29.Mammography,
        '1.2.840.10008.5.1.4.1.1.14.1': codes.cid29.IntravascularOpticalCoherenceTomography,  # noqa E501
        '1.2.840.10008.5.1.4.1.1.14.2': codes.cid29.IntravascularOpticalCoherenceTomography,  # noqa E501
        '1.2.840.10008.5.1.4.1.1.20': codes.cid29.NuclearMedicine,
        '1.2.840.10008.5.1.4.1.1.68.1': codes.cid29.OpticalSurfaceScanner,
        '1.2.840.10008.5.1.4.1.1.68.2': codes.cid29.OpticalSurfaceScanner,
        '1.2.840.10008.5.1.4.1.1.77.1.1': codes.cid29.Endoscopy,
        '1.2.840.10008.5.1.4.1.1.77.1.1.1': codes.cid29.Endoscopy,
        '1.2.840.10008.5.1.4.1.1.77.1.2': codes.cid29.GeneralMicroscopy,
        '1.2.840.10008.5.1.4.1.1.77.1.2.1': codes.cid29.GeneralMicroscopy,
        '1.2.840.10008.5.1.4.1.1.77.1.3': codes.cid29.SlideMicroscopy,
        '1.2.840.10008.5.1.4.1.1.77.1.4': codes.cid29.ExternalCameraPhotography,
        '1.2.840.10008.5.1.4.1.1.77.1.4.1': codes.cid29.ExternalCameraPhotography,  # noqa E501
        '1.2.840.10008.5.1.4.1.1.77.1.5.1': codes.cid29.OphthalmicPhotography,
        '1.2.840.10008.5.1.4.1.1.77.1.5.2': codes.cid29.OphthalmicPhotography,
        '1.2.840.10008.5.1.4.1.1.77.1.5.4': codes.cid29.OphthalmicTomography,
        '1.2.840.10008.5.1.4.1.1.77.1.6': codes.cid29.SlideMicroscopy,
        '1.2.840.10008.5.1.4.1.1.78.1': codes.cid29.Lensometry,
        '1.2.840.10008.5.1.4.1.1.78.2': codes.cid29.Autorefraction,
        '1.2.840.10008.5.1.4.1.1.78.3': codes.cid29.Keratometry,
        '1.2.840.10008.5.1.4.1.1.78.4': codes.cid29.SubjectiveRefraction,
        '1.2.840.10008.5.1.4.1.1.78.5': codes.cid29.VisualAcuity,
        '1.2.840.10008.5.1.4.1.1.78.7': codes.cid29.OphthalmicAxialMeasurements,
        '1.2.840.10008.5.1.4.1.1.78.8': codes.cid29.Lensometry,
        '1.2.840.10008.5.1.4.1.1.80.1': codes.cid29.OphthalmicVisualField,
        '1.2.840.10008.5.1.4.1.1.81.1': codes.cid29.OphthalmicMapping,
        '1.2.840.10008.5.1.4.1.1.82.1': codes.cid29.OphthalmicMapping,
        '1.2.840.10008.5.1.4.1.1.128': codes.cid29.PositronEmissionTomography,
        '1.2.840.10008.5.1.4.1.1.130': codes.cid29.PositronEmissionTomography,
        '1.2.840.10008.5.1.4.1.1.128.1': codes.cid29.PositronEmissionTomography,
        '1.2.840.10008.5.1.4.1.1.481.1': codes.cid29.RTImage
    }
    if sop_class_uid in sopclass_to_modalty_map.keys():
        return sopclass_to_modalty_map[sop_class_uid]
    else:
        return None


def is_dicom_image(sop_class_uid: str) -> bool:
    """
    Returns true if the SOPClass is an image, false otherwise.
    """
    sop_class_uids = {
        # CR Image Storage
        '1.2.840.10008.5.1.4.1.1.1',
        # Digital X-Ray Image Storage – for Presentation
        '1.2.840.10008.5.1.4.1.1.1.1',
        # Digital X-Ray Image Storage – for Processing
        '1.2.840.10008.5.1.4.1.1.1.1.1',
        # Digital Mammography X-Ray Image Storage – for Presentation
        '1.2.840.10008.5.1.4.1.1.1.2',
        # Digital Mammography X-Ray Image Storage – for Processing
        '1.2.840.10008.5.1.4.1.1.1.2.1',
        # Digital Intra – oral X-Ray Image Storage – for Presentation
        '1.2.840.10008.5.1.4.1.1.1.3',
        # Digital Intra – oral X-Ray Image Storage – for Processing
        '1.2.840.10008.5.1.4.1.1.1.3.1',
        # X-Ray Angiographic Image Storage
        '1.2.840.10008.5.1.4.1.1.12.1',
        # Enhanced XA Image Storage
        '1.2.840.10008.5.1.4.1.1.12.1.1',
        # X-Ray Radiofluoroscopic Image Storage
        '1.2.840.10008.5.1.4.1.1.12.2',
        # Enhanced XRF Image Storage
        '1.2.840.10008.5.1.4.1.1.12.2.1',
        # CT Image Storage
        '1.2.840.10008.5.1.4.1.1.2',
        # Enhanced CT Image Storage
        '1.2.840.10008.5.1.4.1.1.2.1',
        # NM Image Storage
        '1.2.840.10008.5.1.4.1.1.20',
        # Ultrasound Multiframe Image Storage
        '1.2.840.10008.5.1.4.1.1.3.1',
        # MR Image Storage
        '1.2.840.10008.5.1.4.1.1.4',
        # Enhanced MR Image Storage
        '1.2.840.10008.5.1.4.1.1.4.1',
        # Radiation Therapy Image Storage
        '1.2.840.10008.5.1.4.1.1.481.1',
        # Ultrasound Image Storage
        '1.2.840.10008.5.1.4.1.1.6.1',
        # Secondary Capture Image Storage
        '1.2.840.10008.5.1.4.1.1.7',
        # Multiframe Single Bit Secondary Capture Image Storage
        '1.2.840.10008.5.1.4.1.1.7.1',
        # Multiframe Grayscale Byte Secondary Capture Image Storage
        '1.2.840.10008.5.1.4.1.1.7.2',
        # Multiframe Grayscale Word Secondary Capture Image Storage
        '1.2.840.10008.5.1.4.1.1.7.3',
        # Multiframe True Color Secondary Capture Image Storage
        '1.2.840.10008.5.1.4.1.1.7.4',
        # VL endoscopic Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.1',
        # Video Endoscopic Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.1.1',
        # VL Microscopic Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.2',
        # Video Microscopic Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.2.1',
        # VL Slide-Coordinates Microscopic Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.3',
        # VL Photographic Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.4',
        # Video Photographic Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.4.1',
        # Ophthalmic Photography 8-Bit Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.5.1',
        # Ophthalmic Photography 16-Bit Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.5.2',
        # VL Whole Slide Microscopy Image Storage
        '1.2.840.10008.5.1.4.1.1.77.1.6'
    }
    return sop_class_uid in sop_class_uids
