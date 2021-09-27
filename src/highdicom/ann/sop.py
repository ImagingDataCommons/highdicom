"""Module for SOP classes of the ANN modality."""
from collections import defaultdict
from copy import deepcopy
from operator import eq
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    UID,
)

from highdicom.ann.enum import (
    AnnotationCoordinateTypeValues,
    AnnotationGroupGenerationTypeValues,
    GraphicTypeValues,
    PixelOriginInterpretationValues,
)
from highdicom.ann.content import AnnotationGroup
from highdicom.base import SOPClass
from highdicom.sr.coding import CodedConcept


class MicroscopyBulkSimpleAnnotations(SOPClass):

    """SOP class for the Microscopy Bulk Simple Annotations IOD."""

    def __init__(
        self,
        source_images: Sequence[Dataset],
        annotation_coordinate_type: Union[str, AnnotationCoordinateTypeValues],
        annotation_groups: Sequence[AnnotationGroup],
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: Union[str, Tuple[str]],
        device_serial_number: str,
        transfer_syntax_uid: Union[str, UID] = ImplicitVRLittleEndian,
        pixel_origin_interpretation: Union[
            str,
            PixelOriginInterpretationValues
        ] = PixelOriginInterpretationValues.VOLUME,
        **kwargs: Any
    ) -> None:

        src_img = source_images[0]
        is_multiframe = hasattr(src_img, 'NumberOfFrames')
        if is_multiframe and len(source_images) > 1:
            raise ValueError(
                'Only one source image should be provided in case images '
                'are multi-frame images.'
            )

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
            manufacturer=manufacturer,
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
            **kwargs
        )

        annotation_coordinate_type = AnnotationCoordinateTypeValues(
            annotation_coordinate_type
        )
        self.AnnotationCoordinateType = annotation_coordinate_type.value
        if annotation_coordinate_type == AnnotationCoordinateTypeValues.SCOORD:
            pixel_origin_interpretation = PixelOriginInterpretationValues(
                pixel_origin_interpretation
            )
            self.PixelOriginInterpretation = pixel_origin_interpretation.value
        elif annotation_coordinate_type == AnnotationCoordinateTypeValues.SCOORD3D:
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

        # (Enhanced) General Equipment
        self.DeviceSerialNumber = device_serial_number
        self.ManufacturerModelName = manufacturer_model_name
        self.SoftwareVersions = software_versions

        # Common Instance Reference
        self.ReferencedImageSequence: List[Dataset] = []
        referenced_series: Dict[str, List[Dataset]] = defaultdict(list)
        for s_img in source_images:
            ref = Dataset()
            ref.ReferencedSOPClassUID = s_img.SOPClassUID
            ref.ReferencedSOPInstanceUID = s_img.SOPInstanceUID
            self.ReferencedImageSequence.append(ref)
            referenced_series[s_img.SeriesInstanceUID].append(ref)

        self.ReferencedSeriesSequence: List[Dataset] = []
        for series_instance_uid, referenced_images in referenced_series.items():
            ref = Dataset()
            ref.SeriesInstanceUID = series_instance_uid
            ref.ReferencedInstanceSequence = referenced_images
            self.ReferencedSeriesSequence.append(ref)

        group_numbers = np.zeros((len(annotation_groups), ), dtype=int)
        self.AnnotationGroupSequence = []
        for i, group in enumerate(annotation_groups):
            if not isinstance(group, AnnotationGroup):
                raise TypeError(
                    f'Item #{i} of argument "annotation_groups" must have '
                    'type AnnotationGroup.'
                )
            group_numbers[i] = group.AnnotationGroupNumber
            self.AnnotationGroupSequence.append(group)

        if len(group_numbers) > len(np.unique(group_numbers)):
            raise ValueError(
                'Values of attribute Annotation Group Number must be unique '
                'across annotation groups within the annotation instance.'
            )
        if (np.min(group_numbers) != 1 or
                np.max(group_numbers) != len(group_numbers)):
            raise ValueError(
                'Values of attribute Annotation Group Number must be in the '
                'range [1, n], where n is the number of annotations.'
            )

    def get_annotation_group(
        self,
        number: Optional[int] = None,
        uid: Optional[Union[str, UID]] = None,
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
        annotated_property_category: Optional[Union[Code, CodedConcept]] = None,
        annotated_property_type: Optional[Union[Code, CodedConcept]] = None,
        label: Optional[str] = None,
        graphic_type: Optional[Union[str, GraphicTypeValues]] = None,
        algorithm_type: Optional[
            Union[str, AnnotationGroupGenerationTypeValues]
        ] = None,
        algorithm_name: Optional[str] = None,
        algorithm_family: Optional[Union[Code, CodedConcept]] = None,
        algorithm_version: Optional[str] = None,
    ) -> List[AnnotationGroup]:
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

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset
    ) -> 'MicroscopyBulkSimpleAnnotations':
        """Construct instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Microscopy Bulk Simple Annotations instance.

        Returns
        -------
        highdicom.ann.MicroscopyBulkSimpleAnnotations
            Microscopy Bulk Simple Annotations instance

        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                'Dataset must be of type pydicom.dataset.Dataset.'
            )
        if dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.66.4':
            raise ValueError('Dataset is not a Segmentation.')
        ann = deepcopy(dataset)
        ann.__class__ = MicroscopyBulkSimpleAnnotations

        ann.AnnotationGroupSequence = [
            AnnotationGroup.from_dataset(item)
            for item in ann.AnnotationGroupSequence
        ]

        return ann
