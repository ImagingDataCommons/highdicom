"""Content items for Structured Report document instances."""
from typing import Optional, Sequence, Union

import numpy as np
from pydicom.sr.coding import Code
from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import (
    GraphicTypeValues,
    GraphicTypeValues3D,
    PixelOriginInterpretationValues,
    RelationshipTypeValues,
)
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


class LongitudinalTemporalOffsetFromEvent(NumContentItem):

    """Content item representing a longitudinal temporal offset from an event.
    """

    def __init__(
            self,
            value: Optional[Union[int, float]],
            unit: Optional[Union[CodedConcept, Code]] = None,
            event_type: Optional[Union[CodedConcept, Code]] = None
        ) -> None:
        """
        Parameters
        ----------
        value: Union[int, float], optional
            offset in time from a particular event of significance
        unit: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            unit of time, e.g., "Days" or "Seconds"
        event_type: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            type of event to which offset is relative,
            e.g., "Baseline" or "Enrollment"

        """  # noqa
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
        referenced_frame_numbers: Sequence[int], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        """
        super().__init__(
            name=CodedConcept(
                value='121112',
                meaning='Source of Measurement',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=referenced_sop_class_uid,
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            referenced_frame_numbers=referenced_frame_numbers,
            relationship_type=RelationshipTypeValues.SELECTED_FROM
        )


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
        referenced_frame_numbers: Sequence[int], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        """
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
        referenced_frame_numbers: Sequence[int], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        """
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
        graphic_type: Union[highdicom.sr.enum.GraphicTypeValues, str]
            name of the graphic type
        graphic_data: numpy.ndarray
            array of ordered spatial coordinates, where each row of the array
            represents a (column, row) coordinate pair
        source_image: highdicom.sr.template.SourceImageForRegion
            source image to which `graphic_data` relates
        pixel_origin_interpretation: Union[highdicom.sr.enum.PixelOriginInterpretationValues, str], optional
            whether pixel coordinates specified by `graphic_data` are defined
            relative to the total pixel matrix
            (``highdicom.sr.enum.PixelOriginInterpretationValues.VOLUME``) or
            relative to an individual frame
            (``highdicom.sr.enum.PixelOriginInterpretationValues.FRAME``)
            of the source image
            (default: ``highdicom.sr.enum.PixelOriginInterpretationValues.VOLUME``)

        """  # noqa
        graphic_type = GraphicTypeValues(graphic_type)
        if graphic_type == GraphicTypeValues.MULTIPOINT:
            raise ValueError(
                'Graphic type "MULTIPOINT" is not valid for region.'
            )
        if not isinstance(source_image, SourceImageForRegion):
            raise TypeError(
                'Argument "source_image" must have type SourceImageForRegion.'
            )
        if pixel_origin_interpretation is None:
            pixel_origin_interpretation = PixelOriginInterpretationValues.VOLUME
        if pixel_origin_interpretation == PixelOriginInterpretationValues.FRAME:
            if (not hasattr(source_image, 'ReferencedFrameNumber') or
                    source_image.ReferencedFrameNumber is None):
                raise ValueError(
                    'Frame number of source image must be specified when value '
                    'of argument "pixel_origin_interpretation" is "FRAME".'
                )
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
        graphic_type: Union[highdicom.sr.enum.GraphicTypeValues3D, str]
            name of the graphic type
        graphic_data: numpy.ndarray
            array of ordered spatial coordinates, where each row of the array
            represents a (x, y, z) coordinate triplet
        frame_of_reference_uid: str
            UID of the frame of reference

        """  # noqa
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
        graphic_type: Union[highdicom.sr.enum.GraphicTypeValues3D, str]
            name of the graphic type
        graphic_data: Sequence[Sequence[int]]
            ordered set of (row, column, frame) coordinate pairs
        frame_of_reference_uid: str
            unique identifier of the frame of reference within which the
            coordinates are defined
        source_images: Sequence[highdicom.sr.content.SourceImageForSegmentation], optional
            source images for segmentation
        source_series: highdicom.sr.content.SourceSeriesForSegmentation, optional
            source series for segmentation

        Note
        ----
        Either one or more source images or one source series must be provided.

        """  # noqa
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
        anatomic_location: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded anatomic location (region or structure)
        laterality: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded laterality
            (see CID 244 "Laterality" for options)
        topographical_modifier: Union[highdicom.sr.coding.CodedConcept, pydicom.sr.coding.Code], optional
            coded modifier of anatomic location

        """  # noqa
        super().__init__(
            name=CodedConcept(
                value='363698007',
                meaning='Finding Site',
                scheme_designator='SCT'
            ),
            value=anatomic_location,
            relationship_type=RelationshipTypeValues.HAS_CONCEPT_MOD
        )
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
        source_image: highdicom.sr.content.SourceImageForSegmentation
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
        frame_numbers: Sequence[int], optional
            numbers of the frames to which the reference applies
            (in case a segmentation instance is referenced)
        segment_number: int
            number of the segment to which the refernce applies
        source_images: Sequence[highdicom.sr.content.SourceImageForSegmentation], optional
            source images for segmentation
        source_series: highdicom.sr.content.SourceSeriesForSegmentation, optional
            source series for segmentation

        Note
        ----
        Either `source_images` or `source_series` must be provided.

        """  # noqa
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
