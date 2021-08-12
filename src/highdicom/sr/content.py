"""Content items for Structured Report document instances."""
import logging
from copy import deepcopy
from typing import cast, Optional, Sequence, Union

import numpy as np
from pydicom._storage_sopclass_uids import (
    SegmentationStorage,
    VLWholeSlideMicroscopyImageStorage
)
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import (
    GraphicTypeValues,
    GraphicTypeValues3D,
    PixelOriginInterpretationValues,
    RelationshipTypeValues,
    ValueTypeValues,
)
from highdicom.sr.utils import find_content_items
from highdicom.sr.value_types import (
    CodeContentItem,
    CompositeContentItem,
    ContentSequence,
    ImageContentItem,
    NumContentItem,
    ScoordContentItem,
    Scoord3DContentItem,
    UIDRefContentItem,
)


logger = logging.getLogger(__name__)


def _check_valid_source_image_dataset(dataset: Dataset) -> None:
    """Raise an error if the image is not a valid source image reference.

    Certain datasets are not appropriate as source images for measurements,
    regions, or segmentations. However the criteria do not appear to be clearly
    defined or succinctly implementable.  This function is intended to catch
    common mistakes users may make when creating references to source images,
    without attempting to be comprehensive. If an error is found, a ValueError
    is raised with an appropriate error message.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        A dataset object to be checked

    Raise
    -----
    ValueError:
        If the input dataset is not valid to serve as a source image for a
        measurement, segmentation, or region.

    """
    # Check that some of the image pixel description attributes are present
    pixel_data_keywords = [
        'Rows',
        'Columns',
        'SamplesPerPixel',
        'PixelRepresentation',
    ]
    if not any(hasattr(dataset, attr) for attr in pixel_data_keywords):
        raise ValueError(
            'Dataset does not represent a valid source image for '
            'a measurement, segmentation, or region because it '
            'lacks image pixel description attributes.'
        )
    # Check for obviously invalid modalities
    disallowed_modalities = [
        'SEG', 'SR', 'DOC', 'KO', 'PR', 'PLAN', 'RWV', 'REG', 'FID',
        'RTDOSE', 'RTPLAN', 'RTRECORD', 'RTSTRUCT'
    ]
    if dataset.Modality in disallowed_modalities:
        raise ValueError(
            f"Datasets with Modality '{dataset.Modality}' are not valid "
            "to use as source images for measurements, regions or "
            "segmentations."
        )


def _check_frame_numbers_valid_for_dataset(
    dataset: Dataset,
    referenced_frame_numbers: Optional[Sequence[int]]
) -> None:
    if referenced_frame_numbers is not None:
        if not hasattr(dataset, 'NumberOfFrames'):
            raise TypeError(
                'The dataset does not represent a multi-frame dataset, so no '
                'referenced frame numbers should be provided.'
            )
        for f in referenced_frame_numbers:
            if f < 1:
                raise ValueError(
                    'Frame numbers must be greater than or equal to 1 (frame '
                    f'indexing is 1-based). Got {f}.'
                )
            if f > dataset.NumberOfFrames:
                raise ValueError(
                    f'{f} is an invalid frame number for the given dataset '
                    'with {dataset.NumberOfFrames} frames.'
                )


class LongitudinalTemporalOffsetFromEvent(NumContentItem):

    """Content item representing a longitudinal temporal offset from an event.
    """

    def __init__(
        self,
        value: Union[int, float],
        unit: Union[CodedConcept, Code],
        event_type: Union[CodedConcept, Code]
    ) -> None:
        """
        Parameters
        ----------
        value: Union[int, float]
            Offset in time from a particular event of significance
        unit: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Unit of time, e.g., "Days" or "Seconds"
        event_type: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            Type of event to which offset is relative,
            e.g., "Baseline" or "Enrollment"

        """  # noqa: E501
        super().__init__(
            name=CodedConcept(
                value='128740',
                meaning='Longitudinal Temporal Offset from Event',
                scheme_designator='DCM'
            ),
            value=value,
            unit=unit,
            relationship_type=RelationshipTypeValues.HAS_OBS_CONTEXT
        )
        event_type_item = CodeContentItem(
            name=CodedConcept(
                value='128741',
                meaning='Longitudinal Temporal Event Type',
                scheme_designator='DCM'
            ),
            value=event_type,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        self.ContentSequence = ContentSequence([event_type_item])

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset
    ) -> 'LongitudinalTemporalOffsetFromEvent':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.LongitudinalTemporalOffsetFromEvent
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(
        cls,
        dataset: Dataset
    ) -> 'LongitudinalTemporalOffsetFromEvent':
        item = super()._from_dataset_base(dataset)
        return cast(LongitudinalTemporalOffsetFromEvent, item)


class SourceImageForMeasurement(ImageContentItem):

    """Content item representing a reference to an image that was used as a
    source for a measurement.
    """

    def __init__(
        self,
        referenced_sop_class_uid: str,
        referenced_sop_instance_uid: str,
        referenced_frame_numbers: Optional[Sequence[int]] = None
    ):
        """
        Parameters
        ----------
        referenced_sop_class_uid: str
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: str
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[Sequence[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        Raises
        ------
        ValueError
            If any referenced frame number is not a positive integer

        """
        if referenced_frame_numbers is not None:
            if any(f < 1 for f in referenced_frame_numbers):
                raise ValueError(
                    'Referenced frame numbers must be >= 1. Frame indexing is '
                    '1-based.'
                )
        super().__init__(
            name=CodedConcept(
                value='121112',
                meaning='Source of Measurement',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=referenced_sop_class_uid,
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            referenced_frame_numbers=referenced_frame_numbers,
            relationship_type=RelationshipTypeValues.INFERRED_FROM
        )

    @classmethod
    def from_source_image(
        cls,
        image: Dataset,
        referenced_frame_numbers: Optional[Sequence[int]] = None
    ) -> 'SourceImageForMeasurement':
        """Construct the content item directly from an image dataset

        Parameters
        ----------
        image: pydicom.dataset.Dataset
            Dataset representing the image to be referenced
        referenced_frame_numbers: Union[Sequence[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        Returns
        -------
        highdicom.sr.SourceImageForMeasurement
            Content item representing a reference to the image dataset

        """
        # Check the dataset and referenced frames are valid
        _check_valid_source_image_dataset(image)
        _check_frame_numbers_valid_for_dataset(
            image,
            referenced_frame_numbers
        )
        return cls(
            referenced_sop_class_uid=image.SOPClassUID,
            referenced_sop_instance_uid=image.SOPInstanceUID,
            referenced_frame_numbers=referenced_frame_numbers
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'SourceImageForMeasurement':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.SourceImageForMeasurement
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'SourceImageForMeasurement':
        item = super()._from_dataset_base(dataset)
        return cast(SourceImageForMeasurement, item)


class SourceImageForRegion(ImageContentItem):

    """Content item representing a reference to an image that was used as a
    source for a region.
    """

    def __init__(
        self,
        referenced_sop_class_uid: str,
        referenced_sop_instance_uid: str,
        referenced_frame_numbers: Optional[Sequence[int]] = None
    ):
        """
        Parameters
        ----------
        referenced_sop_class_uid: str
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: str
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[Sequence[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        Raises
        ------
        ValueError
            If any referenced frame number is not a positive integer

        """
        if referenced_frame_numbers is not None:
            if any(f < 1 for f in referenced_frame_numbers):
                raise ValueError(
                    'Referenced frame numbers must be >= 1. Frame indexing is '
                    '1-based.'
                )
        super().__init__(
            name=CodedConcept(
                value='111040',
                meaning='Original Source',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=referenced_sop_class_uid,
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            referenced_frame_numbers=referenced_frame_numbers,
            relationship_type=RelationshipTypeValues.SELECTED_FROM
        )

    @classmethod
    def from_source_image(
        cls,
        image: Dataset,
        referenced_frame_numbers: Optional[Sequence[int]] = None
    ) -> 'SourceImageForRegion':
        """Construct the content item directly from an image dataset

        Parameters
        ----------
        image: pydicom.dataset.Dataset
            Dataset representing the image to be referenced
        referenced_frame_numbers: Union[Sequence[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        Returns
        -------
        highdicom.sr.SourceImageForRegion
            Content item representing a reference to the image dataset

        """
        # Check the dataset and referenced frames are valid
        _check_valid_source_image_dataset(image)
        _check_frame_numbers_valid_for_dataset(
            image,
            referenced_frame_numbers
        )
        return cls(
            referenced_sop_class_uid=image.SOPClassUID,
            referenced_sop_instance_uid=image.SOPInstanceUID,
            referenced_frame_numbers=referenced_frame_numbers
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'SourceImageForRegion':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.SourceImageForRegion
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'SourceImageForRegion':
        item = super()._from_dataset_base(dataset)
        return cast(SourceImageForRegion, item)


class SourceImageForSegmentation(ImageContentItem):

    """Content item representing a reference to an image that was used as the
    source for a segmentation.
    """

    def __init__(
        self,
        referenced_sop_class_uid: str,
        referenced_sop_instance_uid: str,
        referenced_frame_numbers: Optional[Sequence[int]] = None
    ) -> None:
        """
        Parameters
        ----------
        referenced_sop_class_uid: str
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: str
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[Sequence[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        Raises
        ------
        ValueError
            If any referenced frame number is not a positive integer

        """
        if referenced_frame_numbers is not None:
            if any(f < 1 for f in referenced_frame_numbers):
                raise ValueError(
                    'Referenced frame numbers must be >= 1. Frame indexing is '
                    '1-based.'
                )
        super().__init__(
            name=CodedConcept(
                value='121233',
                meaning='Source Image for Segmentation',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=referenced_sop_class_uid,
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            referenced_frame_numbers=referenced_frame_numbers,
            relationship_type=RelationshipTypeValues.CONTAINS
        )

    @classmethod
    def from_source_image(
        cls,
        image: Dataset,
        referenced_frame_numbers: Optional[Sequence[int]] = None
    ) -> 'SourceImageForSegmentation':
        """Construct the content item directly from an image dataset

        Parameters
        ----------
        image: pydicom.dataset.Dataset
            Dataset representing the image to be referenced
        referenced_frame_numbers: Union[Sequence[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        Returns
        -------
        highdicom.sr.SourceImageForSegmentation
            Content item representing a reference to the image dataset

        """
        # Check the dataset and referenced frames are valid
        _check_valid_source_image_dataset(image)
        _check_frame_numbers_valid_for_dataset(
            image,
            referenced_frame_numbers
        )
        return cls(
            referenced_sop_class_uid=image.SOPClassUID,
            referenced_sop_instance_uid=image.SOPInstanceUID,
            referenced_frame_numbers=referenced_frame_numbers
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'SourceImageForSegmentation':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.SourceImageForSegmentation
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'SourceImageForSegmentation':
        item = super()._from_dataset_base(dataset)
        return cast(SourceImageForSegmentation, item)


class SourceSeriesForSegmentation(UIDRefContentItem):

    """Content item representing a reference to a series of images that was
    used as the source for a segmentation.
    """

    def __init__(self, referenced_series_instance_uid: str):
        """
        Parameters
        ----------
        referenced_series_instance_uid: str
            Series Instance UID

        """
        super().__init__(
            name=CodedConcept(
                value='121232',
                meaning='Source Series for Segmentation',
                scheme_designator='DCM'
            ),
            value=referenced_series_instance_uid,
            relationship_type=RelationshipTypeValues.CONTAINS
        )

    @classmethod
    def from_source_image(
        cls,
        image: Dataset,
    ) -> 'SourceSeriesForSegmentation':
        """Construct the content item directly from an image dataset

        Parameters
        ----------
        image: pydicom.dataset.Dataset
            dataset representing a single image from the series to be
            referenced

        Returns
        -------
        highdicom.sr.SourceSeriesForSegmentation
            Content item representing a reference to the image dataset

        """
        _check_valid_source_image_dataset(image)
        return cls(
            referenced_series_instance_uid=image.SeriesInstanceUID,
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'SourceSeriesForSegmentation':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.SourceSeriesForSegmentation
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'SourceSeriesForSegmentation':
        item = super()._from_dataset_base(dataset)
        return cast(SourceSeriesForSegmentation, item)


class ImageRegion(ScoordContentItem):

    """Content item representing an image region of interest in the
    two-dimensional image coordinate space in pixel unit.
    """

    def __init__(
        self,
        graphic_type: Union[GraphicTypeValues, str],
        graphic_data: np.ndarray,
        source_image: SourceImageForRegion,
        pixel_origin_interpretation: Optional[
            Union[PixelOriginInterpretationValues, str]
        ] = None
    ) -> None:
        """
        Parameters
        ----------
        graphic_type: Union[highdicom.sr.GraphicTypeValues, str]
            name of the graphic type
        graphic_data: numpy.ndarray
            array of ordered spatial coordinates, where each row of the array
            represents a (column, row) coordinate pair
        source_image: highdicom.sr.SourceImageForRegion
            source image to which `graphic_data` relates
        pixel_origin_interpretation: Union[highdicom.sr.PixelOriginInterpretationValues, str, None], optional
            whether pixel coordinates specified by `graphic_data` are defined
            relative to the total pixel matrix
            (``highdicom.sr.PixelOriginInterpretationValues.VOLUME``) or
            relative to an individual frame
            (``highdicom.sr.PixelOriginInterpretationValues.FRAME``)
            of the source image
            (default: ``highdicom.sr.PixelOriginInterpretationValues.VOLUME``)

        """  # noqa: E501
        graphic_type = GraphicTypeValues(graphic_type)
        if graphic_type == GraphicTypeValues.MULTIPOINT:
            raise ValueError(
                'Graphic type "MULTIPOINT" is not valid for region.'
            )
        if not isinstance(source_image, SourceImageForRegion):
            raise TypeError(
                'Argument "source_image" must have type SourceImageForRegion.'
            )
        if pixel_origin_interpretation == PixelOriginInterpretationValues.FRAME:
            ref_sop_item = source_image.ReferencedSOPSequence[0]
            if (not hasattr(ref_sop_item, 'ReferencedFrameNumber') or
                    ref_sop_item.ReferencedFrameNumber is None):
                raise ValueError(
                    'Frame number of source image must be specified when value '
                    'of argument "pixel_origin_interpretation" is "FRAME".'
                )
        ref_sop_instance_item = source_image.ReferencedSOPSequence[0]
        ref_sop_class_uid = ref_sop_instance_item.ReferencedSOPClassUID
        if (ref_sop_class_uid == VLWholeSlideMicroscopyImageStorage and
                pixel_origin_interpretation is None):
            pixel_origin_interpretation = PixelOriginInterpretationValues.VOLUME
        super().__init__(
            name=CodedConcept(
                value='111030',
                meaning='Image Region',
                scheme_designator='DCM'
            ),
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            pixel_origin_interpretation=pixel_origin_interpretation,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        self.ContentSequence = [source_image]

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ImageRegion':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.ImageRegion
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'ImageRegion':
        item = super()._from_dataset_base(dataset)
        return cast(ImageRegion, item)


class ImageRegion3D(Scoord3DContentItem):

    """Content item representing an image region of interest in the
    three-dimensional patient/slide coordinate space in millimeter unit.
    """

    def __init__(
        self,
        graphic_type: Union[GraphicTypeValues3D, str],
        graphic_data: np.ndarray,
        frame_of_reference_uid: str
    ) -> None:
        """
        Parameters
        ----------
        graphic_type: Union[highdicom.sr.GraphicTypeValues3D, str]
            name of the graphic type
        graphic_data: numpy.ndarray
            array of ordered spatial coordinates, where each row of the array
            represents a (x, y, z) coordinate triplet
        frame_of_reference_uid: str
            UID of the frame of reference

        """  # noqa: E501
        graphic_type = GraphicTypeValues3D(graphic_type)
        if graphic_type == GraphicTypeValues3D.MULTIPOINT:
            raise ValueError(
                'Graphic type "MULTIPOINT" is not valid for region.'
            )
        if graphic_type == GraphicTypeValues3D.ELLIPSOID:
            raise ValueError(
                'Graphic type "ELLIPSOID" is not valid for region.'
            )
        super().__init__(
            name=CodedConcept(
                value='111030',
                meaning='Image Region',
                scheme_designator='DCM'
            ),
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            frame_of_reference_uid=frame_of_reference_uid,
            relationship_type=RelationshipTypeValues.CONTAINS
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'ImageRegion3D':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.ImageRegion3D
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'ImageRegion3D':
        item = super()._from_dataset_base(dataset)
        return cast(ImageRegion3D, item)


class VolumeSurface(Scoord3DContentItem):

    """Content item representing a volume surface in the the three-dimensional
    patient/slide coordinate system in millimeter unit.
    """

    def __init__(
        self,
        graphic_type: Union[GraphicTypeValues3D, str],
        graphic_data: np.ndarray,
        frame_of_reference_uid: str,
        source_images: Optional[
            Sequence[SourceImageForSegmentation]
        ] = None,
        source_series: Optional[SourceSeriesForSegmentation] = None
    ) -> None:
        """
        Parameters
        ----------
        graphic_type: Union[highdicom.sr.GraphicTypeValues3D, str]
            name of the graphic type
        graphic_data: Sequence[Sequence[int]]
            ordered set of (row, column, frame) coordinate pairs
        frame_of_reference_uid: str
            unique identifier of the frame of reference within which the
            coordinates are defined
        source_images: Union[Sequence[highdicom.sr.SourceImageForSegmentation], None], optional
            source images for segmentation
        source_series: Union[highdicom.sr.SourceSeriesForSegmentation, None], optional
            source series for segmentation

        Note
        ----
        Either one or more source images or one source series must be provided.

        """  # noqa: E501
        graphic_type = GraphicTypeValues3D(graphic_type)
        if graphic_type != GraphicTypeValues3D.ELLIPSOID:
            raise ValueError(
                'Graphic type for volume surface must be "ELLIPSOID".'
            )
        super().__init__(
            name=CodedConcept(
                value='121231',
                meaning='Volume Surface',
                scheme_designator='DCM'
            ),
            frame_of_reference_uid=frame_of_reference_uid,
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        self.ContentSequence = ContentSequence()
        if source_images is not None:
            for image in source_images:
                if not isinstance(image, SourceImageForSegmentation):
                    raise TypeError(
                        'Items of argument "source_image" must have type '
                        'SourceImageForSegmentation.'
                    )
                self.ContentSequence.append(image)
        elif source_series is not None:
            if not isinstance(source_series, SourceSeriesForSegmentation):
                raise TypeError(
                    'Argument "source_series" must have type '
                    'SourceSeriesForSegmentation.'
                )
            self.ContentSequence.append(source_series)
        else:
            raise ValueError(
                'One of the following two arguments must be provided: '
                '"source_images" or "source_series".'
            )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'VolumeSurface':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.VolumeSurface
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'VolumeSurface':
        item = super()._from_dataset_base(dataset)
        return cast(VolumeSurface, item)


class RealWorldValueMap(CompositeContentItem):

    """Content item representing a reference to a real world value map."""

    def __init__(self, referenced_sop_instance_uid: str):
        """
        Parameters
        ----------
        referenced_sop_instance_uid: str
            SOP Instance UID of the referenced object

        """
        super().__init__(
            name=CodedConcept(
                value='126100',
                meaning='Real World Value Map used for measurement',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.67',
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            relationship_type=RelationshipTypeValues.CONTAINS
        )

    @classmethod
    def from_source_value_map(
        cls,
        value_map_dataset: Dataset,
    ) -> 'RealWorldValueMap':
        """Construct the content item directly from an image dataset

        Parameters
        ----------
        value_map_dataset: pydicom.dataset.Dataset
            dataset representing the real world value map to be
            referenced

        Returns
        -------
        highdicom.sr.RealWorldValueMap
            Content item representing a reference to the image dataset

        """
        if value_map_dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.67':
            raise ValueError(
                'Provided dataset is not a Real World Value Map'
            )
        return cls(
            referenced_sop_instance_uid=value_map_dataset.SOPInstanceUID,
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'RealWorldValueMap':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.RealWorldValueMap
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'RealWorldValueMap':
        item = super()._from_dataset_base(dataset)
        return cast(RealWorldValueMap, item)


class FindingSite(CodeContentItem):

    """Content item representing a coded finding site."""

    def __init__(
        self,
        anatomic_location: Union[CodedConcept, Code],
        laterality: Optional[Union[CodedConcept, Code]] = None,
        topographical_modifier: Optional[Union[CodedConcept, Code]] = None
    ) -> None:
        """
        Parameters
        ----------
        anatomic_location: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code]
            coded anatomic location (region or structure)
        laterality: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded laterality (see :dcm:`CID 244 <part16/sect_CID_244.html>`
            "Laterality" for options)
        topographical_modifier: Union[highdicom.sr.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded modifier of anatomic location

        """  # noqa: E501
        super().__init__(
            name=CodedConcept(
                value='363698007',
                meaning='Finding Site',
                scheme_designator='SCT'
            ),
            value=anatomic_location,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
        if laterality is not None or topographical_modifier is not None:
            self.ContentSequence = ContentSequence()
            if laterality is not None:
                laterality_item = CodeContentItem(
                    name=CodedConcept(
                        value='272741003',
                        meaning='Laterality',
                        scheme_designator='SCT'
                    ),
                    value=laterality,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                self.ContentSequence.append(laterality_item)
            if topographical_modifier is not None:
                modifier_item = CodeContentItem(
                    name=CodedConcept(
                        value='106233006',
                        meaning='Topographical Modifier',
                        scheme_designator='SCT'
                    ),
                    value=topographical_modifier,
                    relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
                )
                self.ContentSequence.append(modifier_item)

    @property
    def topographical_modifier(self) -> Union[CodedConcept, None]:
        matches = find_content_items(
            self,
            name=codes.SCT.TopographicalModifier,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 0:
            return matches[0].value
        elif len(matches) > 1:
            logger.warning(
                'found more than one "Topographical Modifier" content item '
                'in "Finding Site" content item'
            )
        return None

    @property
    def laterality(self) -> Union[CodedConcept, None]:
        matches = find_content_items(
            self,
            name=codes.SCT.Laterality,
            value_type=ValueTypeValues.CODE
        )
        if len(matches) > 0:
            return matches[0].value
        elif len(matches) > 1:
            logger.warning(
                'found more than one "Laterality" content item '
                'in "Finding Site" content item'
            )
        return None

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'FindingSite':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type SCOORD

        Returns
        -------
        highdicom.sr.FindingSite
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(cls, dataset: Dataset) -> 'FindingSite':
        item = super()._from_dataset_base(dataset)
        return cast(FindingSite, item)


class ReferencedSegmentationFrame(ContentSequence):

    """Content items representing a reference to an individual frame of a
    segmentation instance as well as the image that was used as a source for
    the segmentation.
    """

    def __init__(
        self,
        sop_class_uid: str,
        sop_instance_uid: str,
        frame_number: int,
        segment_number: int,
        source_image: SourceImageForSegmentation
    ) -> None:
        """
        Parameters
        ----------
        sop_class_uid: str
            SOP Class UID of the referenced image object
        sop_instance_uid: str
            SOP Instance UID of the referenced image object
        segment_number: int
            number of the segment to which the refernce applies
        frame_number: int
            number of the frame to which the reference applies
        source_image: highdicom.sr.SourceImageForSegmentation
            source image for segmentation

        """
        super().__init__()
        segmentation_item = ImageContentItem(
            name=CodedConcept(
                value='121214',
                meaning='Referenced Segmentation Frame',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_frame_numbers=frame_number,
            referenced_segment_numbers=segment_number,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        self.append(segmentation_item)
        if not isinstance(source_image, SourceImageForSegmentation):
            raise TypeError(
                'Argument "source_image" must have type '
                'SourceImageForSegmentation.'
            )
        self.append(source_image)

    @classmethod
    def from_segmentation(
        cls,
        segmentation: Dataset,
        frame_number: int
    ) -> 'ReferencedSegmentationFrame':
        """Construct the content item directly from a segmentation dataset

        Parameters
        ----------
        segmentation: pydicom.dataset.Dataset
            Dataset representing a segmentation containing the referenced
            segment.
        frame_number: int
            Number of the frame to which the reference applies. Note that
            one-based indexing is used to index the frames.

        Returns
        -------
        highdicom.sr.ReferencedSegment
            Content item representing a reference to the segment

        Notes
        -----
        This method will attempt to deduce source image information
        from information provided in the segmentation instance. If
        available, it will use information specific to the segment
        and frame numbers (if any) provided using the Derivation
        Image Sequence item for the given frame. If this information
        is not present in the segmentation dataset, it will instead
        use the information in the Referenced Series Sequence, which
        applies to all segments and frames present in the segmentation
        instance.

        Raises
        ------
        ValueError
            If the dataset provided is not a segmentation dataset. If any of
            the frames numbers are invalid for the dataset. If multiple
            elements are found in the Derivation Image Sequence or Source Image
            Sequence for any of the referenced frames, or if these attributes
            are absent, if these attributes are absent, if there are multiple
            elements in the Referenced Instance Sequence.
        AttributeError
            If the Referenced Series Sequence or Referenced Instance Sequence
            attributes are absent from the dataset.

        """
        if segmentation.SOPClassUID != SegmentationStorage:
            raise ValueError(
                'Input dataset should be a segmentation storage instance'
            )

        # Move from DICOM 1-based indexing to python 0-based
        frame_index = frame_number - 1
        if frame_index < 0 or frame_index >= segmentation.NumberOfFrames:
            raise ValueError(
                f'Frame {frame_number} is an invalid frame number within the '
                'provided dataset. Note that frame indices are 1-based.'
            )

        frame_info = segmentation.PerFrameFunctionalGroupsSequence[frame_index]

        segment_info = frame_info.SegmentIdentificationSequence[0]
        segment_number = segment_info.ReferencedSegmentNumber

        # Try to deduce the single source image from per-frame functional
        # groups
        found_source_image = False
        if hasattr(frame_info, 'DerivationImageSequence'):
            if len(frame_info.DerivationImageSequence) != 1:
                raise ValueError(
                    'Could not deduce a single source image from the '
                    'provided dataset. Found multiple items in '
                    'DerivationImageSequence for the given frame.'
                )
            drv_image = frame_info.DerivationImageSequence[0]

            if hasattr(drv_image, 'SourceImageSequence'):
                if len(drv_image.SourceImageSequence) != 1:
                    raise ValueError(
                        'Could not deduce a single source image from the '
                        'provided dataset. Found multiple items in '
                        'SourceImageSequence for the given frame.'
                    )
                src = drv_image.SourceImageSequence[0]
                source_image = SourceImageForSegmentation(
                    referenced_sop_class_uid=src.ReferencedSOPClassUID,
                    referenced_sop_instance_uid=src.ReferencedSOPInstanceUID
                )
                found_source_image = True

        if not found_source_image:
            if hasattr(segmentation, 'ReferencedSeriesSequence'):
                ref_series = segmentation.ReferencedSeriesSequence[0]
                if hasattr(ref_series, 'ReferencedInstanceSequence'):
                    if len(ref_series.ReferencedInstanceSequence) != 1:
                        raise ValueError(
                            'Could not deduce a single source image from the '
                            'provided dataset. Found multiple instances in '
                            'ReferencedInstanceSequence.'
                        )
                    src = ref_series.ReferencedInstanceSequence[0]
                    source_image = SourceImageForSegmentation(
                        src.ReferencedSOPClassUID,
                        src.ReferencedSOPInstanceUID
                    )
                else:
                    raise AttributeError(
                        'The ReferencedSeriesSequence in the segmentation '
                        'dataset does not contain the expected information'
                    )
            else:
                raise AttributeError(
                    'Could not deduce source images. Dataset contains neither '
                    'a DerivationImageSequence for the given frame, nor a '
                    'ReferencedSeriesSequence '
                )

        return cls(
            sop_class_uid=segmentation.SOPClassUID,
            sop_instance_uid=segmentation.SOPInstanceUID,
            frame_number=frame_number,
            segment_number=segment_number,
            source_image=source_image
        )


class ReferencedSegment(ContentSequence):

    """Content items representing a reference to an individual segment of a
    segmentation or surface segmentation instance as well as the images that
    were used as a source for the segmentation.
    """

    def __init__(
        self,
        sop_class_uid: str,
        sop_instance_uid: str,
        segment_number: int,
        frame_numbers: Optional[Sequence[int]] = None,
        source_images: Optional[
            Sequence[SourceImageForSegmentation]
        ] = None,
        source_series: Optional[SourceSeriesForSegmentation] = None
    ) -> None:
        """
        Parameters
        ----------
        sop_class_uid: str
            SOP Class UID of the referenced segmentation object
        sop_instance_uid: str
            SOP Instance UID of the referenced segmentation object
        segment_number: int
            number of the segment to which the reference applies
        frame_numbers: Union[Sequence[int], None], optional
            numbers of the frames to which the reference applies
            (in case a segmentation instance is referenced)
        source_images: Union[Sequence[highdicom.sr.SourceImageForSegmentation], None], optional
            source images for segmentation
        source_series: Union[highdicom.sr.SourceSeriesForSegmentation, None], optional
            source series for segmentation

        Note
        ----
        Either `source_images` or `source_series` must be provided.

        """  # noqa: E501
        super().__init__()
        segment_item = ImageContentItem(
            name=CodedConcept(
                value='121191',
                meaning='Referenced Segment',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_frame_numbers=frame_numbers,
            referenced_segment_numbers=segment_number,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        self.append(segment_item)
        if source_images is not None:
            for image in source_images:
                if not isinstance(image, SourceImageForSegmentation):
                    raise TypeError(
                        'Items of argument "source_images" must have type '
                        'SourceImageForSegmentation.'
                    )
                self.append(image)
        elif source_series is not None:
            if not isinstance(source_series,
                              SourceSeriesForSegmentation):
                raise TypeError(
                    'Argument "source_series" must have type '
                    'SourceSeriesForSegmentation.'
                )
            self.append(source_series)
        else:
            raise ValueError(
                'One of the following two arguments must be provided: '
                '"source_images" or "source_series".'
            )

    @classmethod
    def from_segmentation(
        cls,
        segmentation: Dataset,
        segment_number: int,
        frame_numbers: Optional[Sequence[int]] = None
    ) -> 'ReferencedSegment':
        """Construct the content item directly from a segmentation dataset

        Parameters
        ----------
        segmentation: pydicom.dataset.Dataset
            dataset representing a segmentation containing the referenced
            segment
        segment_number: int
            number of the segment to reference within the provided dataset
        frame_numbers: Union[Sequence[int], None], optional
            list of frames in the segmentation dataset to reference. If
            not provided, the reference is assumed to apply to all frames
            of the given segment number. Note that frame numbers are
            indexed with 1-based indexing.

        Returns
        -------
        highdicom.sr.ReferencedSegment
            Content item representing a reference to the segment

        Notes
        -----
        This method will attempt to deduce source image information from
        information provided in the segmentation instance. If available, it
        will used information specific to the segment and frame numbers (if
        any) provided using the Derivation Image Sequence information in the
        frames. If this information is not present in the segmentation dataset,
        it will instead use the information in the Referenced Series Sequence,
        which applies to all segments and frames present in the segmentation
        instance.

        """
        if segmentation.SOPClassUID != SegmentationStorage:
            raise ValueError(
                'Input dataset should be a segmentation storage instance'
            )

        source_images = []
        source_series = None

        # Method A
        # Attempt to retrieve a list of source images for the specific segment
        # and frames specified using information in the per-frame functional
        # groups sequence
        # This gives more specific information, but depends upon type 2
        # attributes (DerivationImageSequence and SourceImageSequence)
        # so may fail with some segmentation objects
        # Those created with highdicom should have this information

        # Firstly, gather up list of all frames in the segmentation dataset
        # that relate to the given segment (and frames, if specified)
        if frame_numbers is not None:
            # Use only the provided frames
            referenced_frame_info = []
            for f in frame_numbers:
                # Check that the provided frame number is valid
                if f < 1 or f > segmentation.NumberOfFrames:
                    raise ValueError(
                        f'Frame {f} is an invalid frame number within the '
                        'provided dataset. Note that frame numbers use '
                        '1-based indexing.'
                    )

                i = f - 1  # 0-based index to the frame
                frame_info = segmentation.PerFrameFunctionalGroupsSequence[i]

                # Check that this frame references the correct
                # segment
                ref_segment = frame_info.SegmentIdentificationSequence[0]\
                    .ReferencedSegmentNumber

                if ref_segment != segment_number:
                    raise ValueError(
                        f'The provided frame number {f} does not refer to '
                        f'segment number {segment_number}'
                    )
                referenced_frame_info.append(frame_info)
        else:
            referenced_frame_info = [
                frame_info
                for frame_info in segmentation.PerFrameFunctionalGroupsSequence
                if frame_info.SegmentIdentificationSequence[0]
                .ReferencedSegmentNumber == segment_number
            ]

            if len(referenced_frame_info) == 0:
                raise ValueError(
                    f'No frame information found referencing segment '
                    f'{segment_number}'
                )

        # Gather up references using the Derivation Image Sequences of the
        # referenced frames
        refd_insuids = set()
        for frame_info in referenced_frame_info:
            for drv_image in getattr(frame_info, 'DerivationImageSequence', []):
                for src_image in getattr(drv_image, 'SourceImageSequence', []):
                    # Check to avoid duplication of instances
                    ins_uid = src_image.ReferencedSOPInstanceUID
                    cls_uid = src_image.ReferencedSOPClassUID
                    if ins_uid not in refd_insuids:
                        ref_frames = getattr(
                            src_image,
                            'ReferencedFrameNumber',
                            None
                        )

                        # Referenced Frame Number can be a single value or
                        # multiple values, but we need a list
                        if ref_frames is not None:
                            if src_image['ReferencedFrameNumber'].VM == 1:
                                ref_frames = [ref_frames]

                        source_images.append(
                            SourceImageForSegmentation(
                                referenced_sop_class_uid=cls_uid,
                                referenced_sop_instance_uid=ins_uid,
                                referenced_frame_numbers=ref_frames
                            )
                        )
                        refd_insuids |= {ins_uid}

        if len(source_images) == 0:
            # Method B
            # Attempt to retrieve the source sop instance uids or series
            # instance uid from the 'ReferencedSeriesSequence' of the dataset
            # Note that since only a single series may be used, we take the
            # first.
            # ReferencedSeriesSequence is a type 1C that should be
            # present in all segmentations that reference other instances
            # within the study
            if hasattr(segmentation, 'ReferencedSeriesSequence'):
                ref_series = segmentation.ReferencedSeriesSequence[0]
                if hasattr(ref_series, 'ReferencedInstanceSequence'):
                    source_images = [
                        SourceImageForSegmentation(
                            s.ReferencedSOPClassUID,
                            s.ReferencedSOPInstanceUID
                        )
                        for s in ref_series.ReferencedInstanceSequence
                    ]
                elif hasattr(ref_series, 'SeriesInstanceUID'):
                    source_series = SourceSeriesForSegmentation(
                        ref_series.SeriesInstanceUID
                    )
                else:
                    raise AttributeError(
                        "The ReferencedSeriesSequence in the segmentation "
                        "dataset does not contain the expected information"
                    )
            else:
                raise AttributeError(
                    "Segmentation dataset does not contain a "
                    "ReferencedSeriesSequence containing information about "
                    "the source images"
                )

        return cls(
            sop_class_uid=segmentation.SOPClassUID,
            sop_instance_uid=segmentation.SOPInstanceUID,
            segment_number=segment_number,
            frame_numbers=frame_numbers,
            source_images=source_images if source_images else None,
            source_series=source_series
        )
