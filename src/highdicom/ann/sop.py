"""Module for SOP classes of the ANN modality."""
from collections import defaultdict
from copy import deepcopy
from operator import eq
from os import PathLike
from typing import (
    Any,
    BinaryIO,
    cast,
)
from collections.abc import Sequence
from typing_extensions import Self

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    UID,
)
from pydicom.valuerep import PersonName

from highdicom.ann.enum import (
    AnnotationCoordinateTypeValues,
    AnnotationGroupGenerationTypeValues,
    GraphicTypeValues,
    PixelOriginInterpretationValues,
)
from highdicom.ann.content import AnnotationGroup
from highdicom.base import SOPClass, _check_little_endian
from highdicom.base_content import ContributingEquipment
from highdicom.io import _wrapped_dcmread
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import check_person_name, _check_code_string


class MicroscopyBulkSimpleAnnotations(SOPClass):

    """SOP class for the Microscopy Bulk Simple Annotations IOD.

    See :doc:`ann` for an overview of working with these objects.

    """

    def __init__(
        self,
        source_images: Sequence[Dataset],
        annotation_coordinate_type: str | AnnotationCoordinateTypeValues,
        annotation_groups: Sequence[AnnotationGroup],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: str | tuple[str],
        device_serial_number: str,
        content_description: str | None = None,
        content_creator_name: str | PersonName | None = None,
        transfer_syntax_uid: str | UID = ExplicitVRLittleEndian,
        pixel_origin_interpretation: (
            str |
            PixelOriginInterpretationValues
        ) = PixelOriginInterpretationValues.VOLUME,
        content_label: str | None = None,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        source_images: Sequence[pydicom.dataset.Dataset]
            Image instances from which annotations were derived. In case of
            "2D" Annotation Coordinate Type, only one source image shall be
            provided. In case of "3D" Annotation Coordinate Type, one or more
            source images may be provided. All images shall have the same
            Frame of Reference UID.
        annotation_coordinate_type: Union[str, highdicom.ann.AnnotationCoordinateTypeValues]
            Type of coordinates (two-dimensional coordinates relative to origin
            of Total Pixel Matrix in pixel unit or three-dimensional
            coordinates relative to origin of Frame of Reference (Slide) in
            millimeter/micrometer unit)
        annotation_groups: Sequence[highdicom.ann.AnnotationGroup]
            Groups of annotations (vector graphics and corresponding
            measurements)
        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        manufacturer: Union[str, None]
            Name of the manufacturer (developer) of the device (software)
            that creates the instance
        manufacturer_model_name: str
            Name of the device model (name of the software library or
            application) that creates the instance
        software_versions: Union[str, Tuple[str]]
            Version(s) of the software that creates the instance
        device_serial_number: str
            Manufacturer's serial number of the device
        content_description: Union[str, None], optional
            Description of the annotation
        content_creator_name: Union[str, pydicom.valuerep.PersonName, None], optional
            Name of the creator of the annotation (if created manually)
        transfer_syntax_uid: str, optional
            UID of transfer syntax that should be used for encoding of
            data elements.
        content_label: Union[str, None], optional
            Content label
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        """  # noqa: E501
        coordinate_type = AnnotationCoordinateTypeValues(
            annotation_coordinate_type
        )
        if len({img.FrameOfReferenceUID for img in source_images}) > 1:
            raise ValueError(
                'All source images must have the same Frame of Reference UID.'
            )
        if len(source_images) == 0:
            raise ValueError('At least one source image must be provided.')
        elif len(source_images) > 1:
            if coordinate_type == AnnotationCoordinateTypeValues.SCOORD:
                raise ValueError(
                    'Only one source image should be provided '
                    'if Annotation Coordinate Type is "2D".'
                )
        src_img = source_images[0]

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported.'
            )

        super().__init__(
            study_instance_uid=src_img.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid='1.2.840.10008.5.1.4.1.1.91.1',
            modality='ANN',
            transfer_syntax_uid=transfer_syntax_uid,
            patient_id=src_img.PatientID,
            patient_name=src_img.PatientName,
            patient_birth_date=src_img.PatientBirthDate,
            patient_sex=src_img.PatientSex,
            accession_number=src_img.AccessionNumber,
            study_id=src_img.StudyID,
            study_date=src_img.StudyDate,
            study_time=src_img.StudyTime,
            referring_physician_name=getattr(
                src_img, 'ReferringPhysicianName', None
            ),
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            device_serial_number=device_serial_number,
            software_versions=software_versions,
            **kwargs
        )
        self.copy_specimen_information(src_img)
        self.copy_patient_and_study_information(src_img)
        self._add_contributing_equipment(contributing_equipment, src_img)

        # Microscopy Bulk Simple Annotations
        if content_label is not None:
            _check_code_string(content_label)
            self.ContentLabel = content_label
        else:
            self.ContentLabel = f'{src_img.Modality}_ANN'
        self.ContentDescription = content_description
        if content_creator_name is not None:
            check_person_name(content_creator_name)
        self.ContentCreatorName = content_creator_name

        self.AnnotationCoordinateType = coordinate_type.value
        if coordinate_type == AnnotationCoordinateTypeValues.SCOORD:
            pixel_origin_interpretation = PixelOriginInterpretationValues(
                pixel_origin_interpretation
            )
            self.PixelOriginInterpretation = pixel_origin_interpretation.value
        elif coordinate_type == AnnotationCoordinateTypeValues.SCOORD3D:
            # Frame of Reference
            self.FrameOfReferenceUID = src_img.FrameOfReferenceUID
            self.PositionReferenceIndicator = getattr(
                src_img,
                'PositionReferenceIndicator',
                None
            )
        else:
            raise ValueError(
                'Argument "annotation_coordinate_type" has an unexpected '
                'value.'
            )

        # Common Instance Reference
        ref = Dataset()
        ref.ReferencedSOPClassUID = src_img.SOPClassUID
        ref.ReferencedSOPInstanceUID = src_img.SOPInstanceUID
        self.ReferencedImageSequence = [ref]

        referenced_series: dict[str, list[Dataset]] = defaultdict(list)
        for img in source_images:
            ref = Dataset()
            ref.ReferencedSOPClassUID = img.SOPClassUID
            ref.ReferencedSOPInstanceUID = img.SOPInstanceUID
            referenced_series[img.SeriesInstanceUID].append(ref)

        self.ReferencedSeriesSequence: list[Dataset] = []
        for series_instance_uid, referenced_images in referenced_series.items():
            ref = Dataset()
            ref.SeriesInstanceUID = series_instance_uid
            ref.ReferencedInstanceSequence = referenced_images
            self.ReferencedSeriesSequence.append(ref)

        self.AnnotationGroupSequence = []
        for i, group in enumerate(annotation_groups):
            if not isinstance(group, AnnotationGroup):
                raise TypeError(
                    f'Item #{i} of argument "annotation_groups" must have '
                    'type AnnotationGroup.'
                )
            if group.AnnotationGroupNumber != i + 1:
                raise ValueError(
                    f'Item #{i} of argument "annotation_groups" must have '
                    'Annotation Group Number {i + 1} instead of '
                    f'{group.AnnotationGroupNumber}.'
                )
            self.AnnotationGroupSequence.append(group)

    def get_annotation_group(
        self,
        number: int | None = None,
        uid: str | UID | None = None,
    ) -> AnnotationGroup:
        """Get an individual annotation group.

        Parameters
        ----------
        number: Union[int, None], optional
            Identification number of the annotation group
        uid: Union[str, None], optional
            Unique identifier of the annotation group

        Returns
        -------
        highdicom.ann.AnnotationGroup
            Annotation group

        Raises
        ------
        TypeError
            When neither `number` nor `uid` is provided.
        ValueError
            When no group item or more than one item is found matching either
            `number` or `uid`.

        """
        if number is None and uid is None:
            raise TypeError(
                'Argument "number" or argument "uid" must be provided.'
            )
        elif number is not None:
            items = [
                item
                for item in self.AnnotationGroupSequence
                if int(item.AnnotationGroupNumber) == int(number)
            ]
            if len(items) == 0:
                raise ValueError(
                    'Could not find an annotation group with '
                    f'number "{number}".'
                )
            if len(items) > 1:
                raise ValueError(
                    'Found more than one annotation group with '
                    f'number "{number}".'
                )
            return items[0]
        else:
            items = [
                item
                for item in self.AnnotationGroupSequence
                if str(item.AnnotationGroupUID) == str(uid)
            ]
            if len(items) == 0:
                raise ValueError(
                    f'Could not find an annotation group with uid "{uid}".'
                )
            if len(items) > 1:
                raise ValueError(
                    f'Found more than one annotation group with uid "{uid}".'
                )
            return items[0]

    def get_annotation_groups(
        self,
        annotated_property_category: Code | CodedConcept | None = None,
        annotated_property_type: Code | CodedConcept | None = None,
        label: str | None = None,
        graphic_type: str | GraphicTypeValues | None = None,
        algorithm_type: None | (
            str | AnnotationGroupGenerationTypeValues
        ) = None,
        algorithm_name: str | None = None,
        algorithm_family: Code | CodedConcept | None = None,
        algorithm_version: str | None = None,
    ) -> list[AnnotationGroup]:
        """Get annotation groups matching search criteria.

        Parameters
        ----------
        annotated_property_category: Union[Code, CodedConcept, None], optional
            Category of annotated property
            (e.g., ``codes.SCT.MorphologicAbnormality``)
        annotated_property_type: Union[Code, CodedConcept, None], optional
            Type of annotated property (e.g., ``codes.SCT.Neoplasm``)
        label: Union[str, None], optional
            Annotation group label
        graphic_type: Union[str, GraphicTypeValues, None], optional
            Graphic type (e.g., ``highdicom.ann.GraphicTypeValues.POLYGON``)
        algorithm_type: Union[str, AnnotationGroupGenerationTypeValues, None], optional
            Algorithm type (e.g.,
            ``highdicom.ann.AnnotationGroupGenerationTypeValues.AUTOMATIC``)
        algorithm_name: Union[str, None], optional
            Algorithm name
        algorithm_family: Union[Code, CodedConcept, None], optional
            Algorithm family (e.g., ``codes.DCM.ArtificialIntelligence``)
        algorithm_version: Union[str, None], optional
            Algorithm version

        Returns
        -------
        List[highdicom.ann.AnnotationGroup]
            Annotation groups

        """  # noqa: E501
        if graphic_type is not None:
            graphic_type = GraphicTypeValues(graphic_type)
        if algorithm_type is not None:
            algorithm_type = AnnotationGroupGenerationTypeValues(algorithm_type)

        groups = []
        for item in self.AnnotationGroupSequence:
            matches = []
            if annotated_property_category is not None:
                is_match = eq(
                    item.annotated_property_category,
                    annotated_property_category
                )
                matches.append(is_match)

            if annotated_property_type is not None:
                is_match = eq(
                    item.annotated_property_type,
                    annotated_property_type
                )
                matches.append(is_match)

            if label is not None:
                is_match = item.AnnotationGroupLabel == label
                matches.append(is_match)

            if graphic_type is not None:
                is_match = item.graphic_type == graphic_type
                matches.append(is_match)

            if algorithm_type is not None:
                is_match = item.algorithm_type == algorithm_type
                matches.append(is_match)

            algorithm_identification = item.algorithm_identification
            if algorithm_identification is not None:
                if algorithm_name is not None:
                    is_match = algorithm_identification.name == algorithm_name
                    matches.append(is_match)
                if algorithm_version is not None:
                    is_match = eq(
                        algorithm_identification.version,
                        algorithm_version
                    )
                    matches.append(is_match)
                if algorithm_family is not None:
                    is_match = eq(
                        algorithm_identification.family,
                        algorithm_family
                    )
                    matches.append(is_match)
            else:
                if (algorithm_name is not None or
                        algorithm_version is not None or
                        algorithm_family is not None):
                    matches.append(False)

            if np.all(matches) or len(matches) == 0:
                groups.append(item)

        return groups

    @property
    def annotation_coordinate_type(
        self
    ) -> AnnotationCoordinateTypeValues:
        """highdicom.ann.AnnotationCoordinateTypeValues: Annotation coordinate type."""  # noqa: E501
        return AnnotationCoordinateTypeValues(
            self.AnnotationCoordinateType
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Microscopy Bulk Simple Annotations instance.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.ann.MicroscopyBulkSimpleAnnotations
            Microscopy Bulk Simple Annotations instance

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        if dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.91.1':
            raise ValueError(
                'Dataset is not a Microscopy Bulk Simple Annotations '
                'instance.'
            )
        _check_little_endian(dataset)
        if copy:
            ann = deepcopy(dataset)
        else:
            ann = dataset
        ann.__class__ = cls

        ann.AnnotationGroupSequence = [
            AnnotationGroup.from_dataset(item, copy=copy)
            for item in ann.AnnotationGroupSequence
        ]

        return cast(Self, ann)


def annread(
    fp: str | bytes | PathLike | BinaryIO,
) -> MicroscopyBulkSimpleAnnotations:
    """Read a bulk annotations object stored in DICOM File Format.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing a
        MicroscopyBulkSimpleAnnotations object.

    Returns
    -------
    highdicom.ann.MicroscopyBulkSimpleAnnotations
        Bulk annotations object read from the file.

    """
    return MicroscopyBulkSimpleAnnotations.from_dataset(
        _wrapped_dcmread(fp),
        copy=False
    )
