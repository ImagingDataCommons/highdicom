"""Content items for Structured Report document instances."""
import logging
from copy import deepcopy
from typing import cast, List, Optional, Sequence, Union

import numpy as np
from pydicom.uid import (
    SegmentationStorage,
    VLWholeSlideMicroscopyImageStorage
)
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from highdicom.uid import UID
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
    ContentItem,
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


class SourceImageForMeasurementGroup(ImageContentItem):

    """Content item representing a reference to an image that was used as a
    source.
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
                value='260753009',
                scheme_designator='SCT',
                meaning='Source',
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
    ) -> 'SourceImageForMeasurementGroup':
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
        highdicom.sr.SourceImageForMeasurementGroup
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
    def from_dataset(cls, dataset: Dataset) -> 'SourceImageForMeasurementGroup':
        """Construct object from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing an SR Content Item with value type IMAGE

        Returns
        -------
        highdicom.sr.SourceImageForMeasurementGroup
            Constructed object

        """
        dataset_copy = deepcopy(dataset)
        return cls._from_dataset(dataset_copy)

    @classmethod
    def _from_dataset(
        cls,
        dataset: Dataset
    ) -> 'SourceImageForMeasurementGroup':
        item = super()._from_dataset_base(dataset)
        return cast(SourceImageForMeasurementGroup, item)


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
            Dataset representing an SR Content Item with value type IMAGE

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


class VolumeSurface(ContentSequence):

    """Content sequence representing a volume surface in the the three-dimensional
    patient/slide coordinate system in millimeter unit.
    """

    def __init__(
        self,
        graphic_type: Union[GraphicTypeValues3D, str],
        graphic_data: Union[np.ndarray, Sequence[np.ndarray]],
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
            name of the graphic type. Permissible values are "ELLIPSOID",
            "POINT", "ELLIPSE" or "POLYGON".
        graphic_data: Union[np.ndarray, Sequence[np.ndarray]]
            List of graphic data for elements of the volume surface. Each item
            of the list should be a 2D numpy array representing the graphic
            data for a single element of type ``graphic_type``.

            If `graphic_type` is ``"ELLIPSOID"`` or ``"POINT"``, the volume
            surface will consist of a single element that defines the entire
            surface. Therefore, a single 2D NumPy array should be passed
            as a list of length 1 or as a NumPy array directly.

            If `graphic_type` is ``"ELLIPSE"`` or ``"POLYGON"``, the volume
            surface will consist of two or more planar regions that together
            define the surface. Therefore a list of two or more 2D NumPy
            arrays should be passed.

            Each 2D NumPy array should have dimension N x 3 where each row of
            the array represents a coordinate in the 3D Frame of Reference. The
            number, N, and meaning of the coordinates depends upon the value of
            `graphic_type`. See :class:`highdicom.sr.GraphicTypeValues3D` for
            details.

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
        super().__init__()
        self._graphic_type = GraphicTypeValues3D(graphic_type)

        # Allow single 2D numpy array for backwards compatibility
        if isinstance(graphic_data, np.ndarray) and graphic_data.ndim == 2:
            graphic_data = [graphic_data]

        if self._graphic_type in (
            GraphicTypeValues3D.ELLIPSOID,
            GraphicTypeValues3D.POINT
        ):
            if len(graphic_data) > 1:
                raise ValueError(
                    'If graphic type is "ELLIPSOID" or "POINT", graphic data '
                    'should consist of a single item.'
                )
        elif self._graphic_type in (
            GraphicTypeValues3D.ELLIPSE,
            GraphicTypeValues3D.POLYGON
        ):
            if len(graphic_data) < 2:
                raise ValueError(
                    'If graphic type is "ELLIPSE" or "POLYGON", graphic data '
                    'should consist of two or more items.'
                )
        else:
            raise ValueError(
                f'Graphic type "{self._graphic_type}" is not valid for volume '
                'surfaces.'
            )

        for graphic_data_item in graphic_data:
            self.append(
                Scoord3DContentItem(
                    name=CodedConcept(
                        value='121231',
                        meaning='Volume Surface',
                        scheme_designator='DCM'
                    ),
                    frame_of_reference_uid=frame_of_reference_uid,
                    graphic_type=self._graphic_type,
                    graphic_data=graphic_data_item,
                    relationship_type=RelationshipTypeValues.CONTAINS
                )
            )
        if source_images is not None:
            for image in source_images:
                if not isinstance(image, SourceImageForSegmentation):
                    raise TypeError(
                        'Items of argument "source_image" must have type '
                        'SourceImageForSegmentation.'
                    )
                self.append(image)
        elif source_series is not None:
            if not isinstance(source_series, SourceSeriesForSegmentation):
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
    def from_sequence(
        cls,
        sequence: Sequence[Dataset]
    ) -> 'VolumeSurface':
        """Construct an object from an existing content sequence.

        Parameters
        ----------
        sequence: Sequence[Dataset]
            Sequence of datasets to be converted. This is expected to contain
            one or more content items with concept name
            'Volume Surface', and either a single
            content item with concept name 'Source Series For Segmentation', or
            1 or more content items with concept name
            'Source Image For Segmentation'.

        Returns
        -------
        highdicom.sr.VolumeSurface
            Constructed VolumeSurface object, containing copies
            of the original content items.

        """
        vol_surface_items: List[ContentItem] = []
        source_image_items: List[ContentItem] = []
        source_series_items: List[ContentItem] = []
        for item in sequence:
            name_item = item.ConceptNameCodeSequence[0]
            name = Code(
                value=name_item.CodeValue,
                meaning=name_item.CodeMeaning,
                scheme_designator=name_item.CodingSchemeDesignator
            )
            value_type = ValueTypeValues(item.ValueType)
            rel_type = RelationshipTypeValues(item.RelationshipType)

            if (
                (name == codes.DCM.VolumeSurface) and
                (value_type == ValueTypeValues.SCOORD3D) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                vol_surface_items.append(
                    Scoord3DContentItem.from_dataset(item)
                )
            elif (
                (name == codes.DCM.SourceImageForSegmentation) and
                (value_type == ValueTypeValues.IMAGE) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                source_image_items.append(
                    SourceImageForSegmentation.from_dataset(item)
                )
            elif (
                (name == codes.DCM.SourceSeriesForSegmentation) and
                (value_type == ValueTypeValues.UIDREF) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                source_series_items.append(
                    SourceSeriesForSegmentation.from_dataset(item)
                )

        if len(vol_surface_items) == 0:
            raise RuntimeError(
                'Expected sequence to contain one or more content items with '
                'concept name "Volume Surface". Found 0.'
            )

        for item in vol_surface_items:
            if item.graphic_type != vol_surface_items[0].graphic_type:
                raise RuntimeError(
                    'Expected all VolumeSurface content items to have a common '
                    'graphic type.'
                )

        if len(source_image_items) == 0 and len(source_series_items) == 0:
            raise RuntimeError(
                'Expected sequence to contain either at least one content item '
                'with concept name "Source Image For Segmentation" or one '
                'content item with concept name "Source Series For '
                'Segmentation". Found neither.'
            )
        if len(source_image_items) > 0 and len(source_series_items) > 0:
            raise RuntimeError(
                'Sequence should not contain both content items '
                'with concept name "Source Image For Segmentation" and a'
                'content item with concept name "Source Series For '
                'Segmentation".'
            )

        if len(source_image_items) > 0:
            new_seq = ContentSequence(
                vol_surface_items + source_image_items
            )
        else:
            if len(source_series_items) > 1:
                raise RuntimeError(
                    'Sequence should contain at most one content item '
                    'with concept name "Source Series For Segmentation". Found '
                    f'{len(source_series_items)}.'
                )
            new_seq = ContentSequence(
                vol_surface_items + source_series_items
            )

        new_seq.__class__ = cls
        new_seq = cast(VolumeSurface, new_seq)
        new_seq._graphic_type = vol_surface_items[0].graphic_type
        return new_seq

    @property
    def graphic_type(self) -> GraphicTypeValues3D:
        """highdicom.sr.GraphicTypeValues3D: Graphic type."""
        return self._graphic_type

    @property
    def frame_of_reference_uid(self) -> UID:
        """highdicom.UID: Frame of reference UID."""
        return UID(self._graphic_data_items[0].frame_of_reference_uid)

    def has_source_images(self) -> bool:
        """Returns whether the object contains information about source images.

        ReferencedSegment objects must either contain information about source
        images or source series (and not both).

        Returns
        -------
        bool:
            True if the object contains information about source images. False
            if the image contains information about the source series.

        """
        return len(self._lut[codes.DCM.SourceImageForSegmentation]) > 0

    @property
    def source_images_for_segmentation(
        self
    ) -> List[SourceImageForSegmentation]:
        """List[highdicom.sr.SourceImageForSegmentation]:
            Source images for the volume surface.
        """
        return self._lut[codes.DCM.SourceImageForSegmentation]

    @property
    def source_series_for_segmentation(
        self
    ) -> Optional[SourceSeriesForSegmentation]:
        """Optional[highdicom.sr.SourceSeriesForSegmentation]:
            Source series for the volume surface.
        """
        items = self._lut[codes.DCM.SourceSeriesForSegmentation]
        if len(items) == 0:
            return None
        else:
            return cast(SourceSeriesForSegmentation, items[0])

    @property
    def _graphic_data_items(self) -> List[Scoord3DContentItem]:
        """List[highdicom.sr.Scoord3DContentItem]:
            Graphic data elements that comprise the volume surface.
        """
        return cast(
            List[Scoord3DContentItem],
            self._lut[codes.DCM.VolumeSurface]
        )

    @property
    def graphic_data(self) -> List[np.ndarray]:
        """Union[np.ndarray, List[np.ndarray]]:
            Graphic data arrays that comprise the volume surface.
            For volume surfaces with graphic type ``"ELLIPSOID"``
            or ``"POINT"``, this will be a single 2D Numpy array representing
            the graphic data. Otherwise, it will be a list of 2D Numpy arrays
            representing graphic data for each element of the volume surface.
        """
        if self.graphic_type in (
            GraphicTypeValues3D.ELLIPSOID,
            GraphicTypeValues3D.POINT
        ):
            return self._graphic_data_items[0].value
        else:
            return [item.value for item in self._graphic_data_items]


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
        content = ContentSequence()
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
            content.append(laterality_item)
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
            content.append(modifier_item)
        if len(content) > 0:
            self.ContentSequence = content

    @property
    def topographical_modifier(self) -> Union[CodedConcept, None]:
        if not hasattr(self, 'ContentSequence'):
            return None
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
        if not hasattr(self, 'ContentSequence'):
            return None
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
        frame_number: Union[int, Sequence[int]],
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
            Number of the segment to which the reference applies
        frame_number: Union[int, Sequence[int]]
            Number of the frame to which the reference applies. If the
            referenced segmentation image is tiled, more than one frame may be
            specified.
        source_image: highdicom.sr.SourceImageForSegmentation
            Source image for segmentation

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
    def from_sequence(
        cls,
        sequence: Sequence[Dataset]
    ) -> 'ReferencedSegmentationFrame':
        """Construct an object from items within an existing content sequence.

        Parameters
        ----------
        sequence: Sequence[Dataset]
            Sequence of datasets to be converted. This is expected to contain
            content items with the following names:
            "Referenced Segmentation Frame", "Source Image For Segmentation".
            Any other other items will be ignored.

        Returns
        -------
        highdicom.sr.ReferencedSegmentationFrame
            Constructed ReferencedSegmentationFrame object, containing copies
            of the relevant original content items.

        """
        seg_frame_items = []
        source_image_items = []
        for item in sequence:
            name_item = item.ConceptNameCodeSequence[0]
            name = Code(
                value=name_item.CodeValue,
                meaning=name_item.CodeMeaning,
                scheme_designator=name_item.CodingSchemeDesignator
            )
            value_type = ValueTypeValues(item.ValueType)
            rel_type = RelationshipTypeValues(item.RelationshipType)

            if (
                (name == codes.DCM.ReferencedSegmentationFrame) and
                (value_type == ValueTypeValues.IMAGE) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                seg_frame_items.append(
                    ImageContentItem.from_dataset(item)
                )
            elif (
                (name == codes.DCM.SourceImageForSegmentation) and
                (value_type == ValueTypeValues.IMAGE) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                source_image_items.append(
                    SourceImageForSegmentation.from_dataset(item)
                )

        if len(seg_frame_items) != 1:
            raise RuntimeError(
                'Expected sequence to contain exactly one content item with '
                'concept name "Referenced Segmentation Frame". Found '
                f'{len(seg_frame_items)}.'
            )
        if len(source_image_items) != 1:
            raise RuntimeError(
                'Expected sequence to contain exactly one content item with '
                'concept name "Source Image For Segmentation". Found '
                f'{len(source_image_items)}.'
            )

        new_seq = ContentSequence([seg_frame_items[0], source_image_items[0]])
        new_seq.__class__ = cls
        return cast(ReferencedSegmentationFrame, new_seq)

    @classmethod
    def from_segmentation(
        cls,
        segmentation: Dataset,
        frame_number: Optional[Union[int, Sequence[int]]] = None,
        segment_number: Optional[int] = None
    ) -> 'ReferencedSegmentationFrame':
        """Construct the content item directly from a segmentation dataset

        Parameters
        ----------
        segmentation: pydicom.dataset.Dataset
            Dataset representing a segmentation containing the referenced
            segment.
        frame_number: Union[int, Sequence[int], None], optional
            Number of the frame(s) that should be referenced
        segment_number: Union[int, None], optional
            Number of the segment to which the reference applies

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
                'Argument "segmentation" should represent a Segmentation.'
            )

        if frame_number is None:
            if segment_number is None:
                raise TypeError(
                    'Argument "segment_number" is required if argument '
                    '"frame_number" is not provided.'
                )
            frame_numbers = [
                i + 1
                for i, item in enumerate(
                    segmentation.PerFrameFunctionalGroupsSequence
                )
                if segment_number == (
                    item
                    .SegmentIdentificationSequence[0]
                    .ReferencedSegmentNumber
                )
            ]
            if len(frame_numbers) == 0:
                raise ValueError(
                    f'Could not find a frame for segment #{segment_number}.'
                )
            elif len(frame_numbers) > 1:
                if not hasattr(segmentation, 'TotalPixelMatrixRows'):
                    raise ValueError(
                        'Found more than one frame for segment '
                        f'#{segment_number}, but the total pixel matrix of '
                        'the segmentation image is not tiled.'
                    )
        else:
            if isinstance(frame_number, int):
                frame_numbers = [frame_number]
            else:
                frame_numbers = list(frame_number)

        number_of_frames = int(segmentation.NumberOfFrames)
        segment_numbers = []
        found_source_image = False
        for frame_number in frame_numbers:
            if frame_number < 1 or frame_number > number_of_frames:
                raise ValueError(
                    'Value {frame_number} is not a valid frame number.'
                )
            frame_index = frame_number - 1
            item = segmentation.PerFrameFunctionalGroupsSequence[frame_index]
            segment_numbers.append(
                item
                .SegmentIdentificationSequence[0]
                .ReferencedSegmentNumber
            )
            if hasattr(item, 'DerivationImageSequence'):
                if len(item.DerivationImageSequence) != 1:
                    raise ValueError(
                        'Could not deduce a single source image from the '
                        'provided segmentation. Found multiple items in '
                        f'Derivation Image Sequence for frame #{frame_number}.'
                    )
                drv_image = item.DerivationImageSequence[0]

                if hasattr(drv_image, 'SourceImageSequence'):
                    if len(drv_image.SourceImageSequence) != 1:
                        raise ValueError(
                            'Could not deduce a single source image from the '
                            'provided segmentation. Found multiple items in '
                            f'Source Image Sequence for frame #{frame_number}.'
                        )
                    src = drv_image.SourceImageSequence[0]
                    source_image = SourceImageForSegmentation(
                        src.ReferencedSOPClassUID,
                        src.ReferencedSOPInstanceUID,
                        frame_numbers
                    )
                    found_source_image = True
                    break

        if not found_source_image:
            if hasattr(segmentation, 'ReferencedSeriesSequence'):
                ref_series = segmentation.ReferencedSeriesSequence[0]
                if hasattr(ref_series, 'ReferencedInstanceSequence'):
                    if len(ref_series.ReferencedInstanceSequence) != 1:
                        raise ValueError(
                            'Could not deduce a single source image from the '
                            'provided segmentation. Found multiple instances '
                            'in Referenced Instance Sequence.'
                        )
                    src = ref_series.ReferencedInstanceSequence[0]
                    source_image = SourceImageForSegmentation(
                        src.ReferencedSOPClassUID,
                        src.ReferencedSOPInstanceUID
                    )
                else:
                    raise AttributeError(
                        'The ReferencedSeriesSequence in the segmentation '
                        'dataset does not contain the expected information.'
                    )
            else:
                raise AttributeError(
                    'Could not deduce source images. Segmentation contains '
                    'neither a Derivation Image Sequence for the given frame, '
                    'nor a Referenced Series Sequence '
                )

        segment_numbers = list(set(segment_numbers))
        if len(segment_numbers) > 1:
            raise ValueError(
                f'Found more than one segment for frames {frame_numbers}.'
            )

        return cls(
            sop_class_uid=segmentation.SOPClassUID,
            sop_instance_uid=segmentation.SOPInstanceUID,
            frame_number=(
                frame_numbers
                if len(frame_numbers) > 1
                else frame_numbers[0]
            ),
            segment_number=segment_numbers[0],
            source_image=source_image
        )

    @property
    def referenced_sop_class_uid(self) -> UID:
        """highdicom.UID
            referenced SOP Class UID
        """
        return self[0].referenced_sop_class_uid

    @property
    def referenced_sop_instance_uid(self) -> UID:
        """highdicom.UID
            referenced SOP Class UID
        """
        return self[0].referenced_sop_instance_uid

    @property
    def referenced_frame_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]
            referenced frame numbers
        """
        return self[0].referenced_frame_numbers

    @property
    def referenced_segment_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]
            referenced segment numbers
        """
        return self[0].referenced_segment_numbers

    @property
    def source_image_for_segmentation(self) -> SourceImageForSegmentation:
        """highdicom.sr.SourceImageForSegmentation
            Source image for the referenced segmentation
        """
        return self[1]


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
    def from_sequence(
        cls,
        sequence: Sequence[Dataset]
    ) -> 'ReferencedSegment':
        """Construct an object from items within an existing content sequence.

        Parameters
        ----------
        sequence: Sequence[Dataset]
            Sequence of datasets to be converted. This is expected to contain
            a content item with concept name "Referenced Segmentation Frame",
            and either at least one content item with concept name
            "Source Image For Segmentation" or a single content item with
            concept name "Source Series For Segmentation".
            Any other other items will be ignored.

        Returns
        -------
        highdicom.sr.ReferencedSegment
            Constructed ReferencedSegment object, containing copies
            of the original content items.

        """
        seg_frame_items: List[ContentItem] = []
        source_image_items: List[ContentItem] = []
        source_series_items: List[ContentItem] = []
        for item in sequence:
            name_item = item.ConceptNameCodeSequence[0]
            name = Code(
                value=name_item.CodeValue,
                meaning=name_item.CodeMeaning,
                scheme_designator=name_item.CodingSchemeDesignator
            )
            value_type = ValueTypeValues(item.ValueType)
            rel_type = RelationshipTypeValues(item.RelationshipType)

            if (
                (name == codes.DCM.ReferencedSegment) and
                (value_type == ValueTypeValues.IMAGE) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                seg_frame_items.append(
                    ImageContentItem.from_dataset(item)
                )
            elif (
                (name == codes.DCM.SourceImageForSegmentation) and
                (value_type == ValueTypeValues.IMAGE) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                source_image_items.append(
                    SourceImageForSegmentation.from_dataset(item)
                )
            elif (
                (name == codes.DCM.SourceSeriesForSegmentation) and
                (value_type == ValueTypeValues.UIDREF) and
                (rel_type == RelationshipTypeValues.CONTAINS)
            ):
                source_series_items.append(
                    SourceSeriesForSegmentation.from_dataset(item)
                )

        if len(seg_frame_items) != 1:
            raise RuntimeError(
                'Expected sequence to contain exactly one content item with '
                'concept name "Referenced Segment". Found 0.'
            )
        if len(source_image_items) == 0 and len(source_series_items) == 0:
            raise RuntimeError(
                'Expected sequence to contain either at least one content item '
                'with concept name "Source Image For Segmentation" or one '
                'content item with concept name "Source Series For '
                'Segmentation". Found neither.'
            )
        if len(source_image_items) > 0 and len(source_series_items) > 0:
            raise RuntimeError(
                'Sequence should not contain both content items '
                'with concept name "Source Image For Segmentation" and a'
                'content item with concept name "Source Series For '
                'Segmentation".'
            )

        if len(source_image_items) > 0:
            new_seq = ContentSequence(
                seg_frame_items + source_image_items
            )
        else:
            if len(source_series_items) > 1:
                raise RuntimeError(
                    'Sequence should contain at most one content item '
                    'with concept name "Source Series For Segmentation". Found '
                    f'{len(source_series_items)}.'
                )
            new_seq = ContentSequence(
                seg_frame_items + source_series_items
            )

        new_seq.__class__ = cls
        return cast(ReferencedSegment, new_seq)

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

    def has_source_images(self) -> bool:
        """Returns whether the object contains information about source images.

        ReferencedSegment objects must either contain information about source
        images or source series (and not both).

        Returns
        -------
        bool:
            True if the object contains information about source images. False
            if the image contains information about the source series.

        """
        return self[1].name == codes.DCM.SourceImageForSegmentation

    @property
    def referenced_sop_class_uid(self) -> UID:
        """highdicom.UID
            referenced SOP Class UID
        """
        return self[0].referenced_sop_class_uid

    @property
    def referenced_sop_instance_uid(self) -> UID:
        """highdicom.UID
            referenced SOP Class UID
        """
        return self[0].referenced_sop_instance_uid

    @property
    def referenced_frame_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]
            referenced frame numbers
        """
        return self[0].referenced_frame_numbers

    @property
    def referenced_segment_numbers(self) -> Union[List[int], None]:
        """Union[List[int], None]
            referenced segment numbers
        """
        return self[0].referenced_segment_numbers

    @property
    def source_images_for_segmentation(
        self
    ) -> List[SourceImageForSegmentation]:
        """List[highdicom.sr.SourceImageForSegmentation]
            Source images for the referenced segmentation
        """
        if self.has_source_images():
            return list(self[1:])
        else:
            return []

    @property
    def source_series_for_segmentation(self) -> Union[
        SourceSeriesForSegmentation,
        None
    ]:
        """Union[highdicom.sr.SourceSeriesForSegmentation, None]
            Source series for the referenced segmentation
        """
        if self.has_source_images():
            return None
        else:
            return cast(SourceSeriesForSegmentation, self[1])
