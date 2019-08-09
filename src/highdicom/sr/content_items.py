"""Custom content items derived from DICOM value types."""
from typing import Optional, Sequence, Union

from pydicom.sr.coding import Code, CodedConcept
from pydicom.sr.value_types import (
    CodeContentItem,
    CompositeContentItem,
    ContentSequence,
    GraphicTypes,
    GraphicTypes3D,
    ImageContentItem,
    NumContentItem,
    PixelOriginInterpretations,
    RelationshipTypes,
    ScoordContentItem,
    Scoord3DContentItem,
    UIDRefContentItem,
)


class LongitudinalTemporalOffsetFromEvent(NumContentItem):

    """Content item representing a longitudinal temporal offset from an event.
    """

    def __init__(self, value: Optional[Union[int, float]],
                 unit: Optional[Union[CodedConcept, Code]],
                 event_type: Optional[Union[CodedConcept, Code]]) -> None:
        """
        Parameters
        ----------
        value: Union[int, float, None], optional
            offset in time from a particular event of significance
        unit: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            unit of time, e.g., "Days" or "Seconds"
        event_type: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            type of event to which offset is relative,
            e.g., "Baseline" or "Enrollment"

        """  # noqa
        super(LongitudinalTemporalOffsetFromEvent, self).__init__(
            name=CodedConcept(
                value='128740',
                meaning='Longitudinal Temporal Offset from Event',
                scheme_designator='DCM'
            ),
            value=value,
            unit=unit,
            relationship_type=RelationshipTypes.HAS_OBS_CONTEXT
        )
        event_type_item = CodeContentItem(
            name=CodedConcept(
                value='128741',
                meaning='Longitudinal Temporal Event Type',
                scheme_designator='DCM'
            ),
            value=event_type,
            relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
        )
        self.ContentSequence = ContentSequence([event_type_item])


class SourceImageForRegion(ImageContentItem):

    """Content item representing a reference to an image that was used as a
    source for a region.
    """

    def __init__(self, referenced_sop_class_uid, referenced_sop_instance_uid,
                 referenced_frame_numbers=None):
        """
        Parameters
        ----------
        referenced_sop_class_uid: Union[pydicom.uid.UID, str]
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: Union[pydicom.uid.UID, str]
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[List[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        """
        super(SourceImageForRegion, self).__init__(
            name=CodedConcept(
                value='121322',
                meaning='Source image for image processing operation',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=referenced_sop_class_uid,
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            referenced_frame_numbers=referenced_frame_numbers,
            relationship_type=RelationshipTypes.SELECTED_FROM
        )


class SourceImageForSegmentation(ImageContentItem):

    """Content item representing a reference to an image that was used as the
    source for a segmentation.
    """

    def __init__(self, referenced_sop_class_uid, referenced_sop_instance_uid,
                 referenced_frame_numbers=None):
        """
        Parameters
        ----------
        referenced_sop_class_uid: Union[pydicom.uid.UID, str]
            SOP Class UID of the referenced image object
        referenced_sop_instance_uid: Union[pydicom.uid.UID, str]
            SOP Instance UID of the referenced image object
        referenced_frame_numbers: Union[List[int], None], optional
            numbers of the frames to which the reference applies in case the
            referenced image is a multi-frame image

        """
        super(SourceImageForSegmentation, self).__init__(
            name=CodedConcept(
                value='121233',
                meaning='Source Image for Segmentation',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=referenced_sop_class_uid,
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            referenced_frame_numbers=referenced_frame_numbers,
            relationship_type=RelationshipTypes.CONTAINS
        )


class SourceSeriesForSegmentation(UIDRefContentItem):

    """Content item representing a reference to a series of images that was
    used as the source for a segmentation.
    """

    def __init__(self, referenced_series_instance_uid):
        """
        Parameters
        ----------
        referenced_series_instance_uid: Union[pydicom.uid.UID, str]
            Series Instance UID

        """
        super(SourceSeriesForSegmentation, self).__init__(
            name=CodedConcept(
                value='121232',
                meaning='Source Series for Segmentation',
                scheme_designator='DCM'
            ),
            value=referenced_series_instance_uid,
            relationship_type=RelationshipTypes.CONTAINS
        )


class ImageRegion(ScoordContentItem):

    """Content item representing an image region of interest in the
    two-dimensional image coordinate space in pixel unit"""

    def __init__(self, graphic_type, graphic_data, source_image,
                 pixel_origin_interpretation=PixelOriginInterpretations.VOLUME):
        """
        Parameters
        ----------
        graphic_type: Union[pydicom.sr.value_types.GraphicTypes, str]
            name of the graphic type
        graphic_data: List[List[int]]
            ordered set of (row, column) coordinate pairs
        source_image: pydicom.sr.template.SourceImageForRegion
            source image to which `graphic_data` relates
        pixel_origin_interpretation: Union[pydicom.sr.value_types.PixelOriginInterpretations, str, None], optional
            whether pixel coordinates specified by `graphic_data` are defined
            relative to the total pixel matrix
            (``pydicom.sr.value_types.PixelOriginInterpretations.VOLUME``) or
            relative to an individual frame
            (``pydicom.sr.value_types.PixelOriginInterpretations.FRAME``)
            of the source image
            (default: ``pydicom.sr.value_types.PixelOriginInterpretations.VOLUME``)

        """  # noqa
        graphic_type = GraphicTypes(graphic_type)
        if graphic_type == GraphicTypes.MULTIPOINT:
            raise ValueError(
                'Graphic type "MULTIPOINT" is not valid for region.'
            )
        if not isinstance(source_image, SourceImageForRegion):
            raise TypeError(
                'Argument "source_image" must have type SourceImageForRegion.'
            )
        if pixel_origin_interpretation == PixelOriginInterpretations.FRAME:
            if (not hasattr(source_image, 'ReferencedFrameNumber') or
                source_image.ReferencedFrameNumber is None):
                raise ValueError(
                    'Frame number of source image must be specified when value '
                    'of argument "pixel_origin_interpretation" is "FRAME".'
                )
        super(ImageRegion, self).__init__(
            name=CodedConcept(
                value='111030',
                meaning='Image Region',
                scheme_designator='DCM'
            ),
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            pixel_origin_interpretation=pixel_origin_interpretation,
            relationship_type=RelationshipTypes.CONTAINS
        )
        self.ContentSequence = [source_image]


class ImageRegion3D(Scoord3DContentItem):

    """Content item representing an image region of interest in the
    three-dimensional patient/slide coordinate space in millimeter unit.
    """

    def __init__(self, graphic_type, graphic_data, frame_of_reference_uid):
        """
        Parameters
        ----------
        graphic_type: Union[pydicom.sr.value_types.GraphicTypes3D, str]
            name of the graphic type
        graphic_data: List[List[int]]
            ordered set of (x, y, z) coordinate triplets
        frame_of_reference_uid: Union[pydicom.uid.UID, str, None]
            UID of the frame of reference

        """  # noqa
        graphic_type = GraphicTypes3D(graphic_type)
        if graphic_type == GraphicTypes3D.MULTIPOINT:
            raise ValueError(
                'Graphic type "MULTIPOINT" is not valid for region.'
            )
        if graphic_type == GraphicTypes3D.ELLIPSOID:
            raise ValueError(
                'Graphic type "ELLIPSOID" is not valid for region.'
            )
        super(ImageRegion3D, self).__init__(
            name=CodedConcept(
                value='111030',
                meaning='Image Region',
                scheme_designator='DCM'
            ),
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            frame_of_reference_uid=frame_of_reference_uid,
            relationship_type=RelationshipTypes.CONTAINS
        )


class VolumeSurface(Scoord3DContentItem):

    """Content item representing a volume surface in the the three-dimensional
    patient/slide coordinate system in millimeter unit.
    """

    def __init__(self, graphic_type, graphic_data, frame_of_reference_uid,
                 source_images=None, source_series=None):
        """
        Parameters
        ----------
        graphic_type: Union[pydicom.sr.value_types.GraphicTypes, str]
            name of the graphic type
        graphic_data: List[List[int]]
            ordered set of (row, column, frame) coordinate pairs
        frame_of_reference_uid: Union[pydicom.uid.UID, str]
            unique identifier of the frame of reference within which the
            coordinates are defined
        source_images: Union[List[pydicom.sr.content_items.SourceImageForSegmentation], None], optional
            source images for segmentation
        source_series: Union[pydicom.sr.content_items.SourceSeriesForSegmentation, None], optional
            source series for segmentation

        Note
        ----
        Either one or more source images or one source series must be provided.

        """  # noqa
        graphic_type = GraphicTypes3D(graphic_type)
        if graphic_type != GraphicTypes3D.ELLIPSOID:
            raise ValueError(
                'Graphic type for volume surface must be "ELLIPSOID".'
            )
        super(VolumeSurface, self).__init__(
            name=CodedConcept(
                value='121231',
                meaning='Volume Surface',
                scheme_designator='DCM'
            ),
            frame_of_reference_uid=frame_of_reference_uid,
            graphic_type=graphic_type,
            graphic_data=graphic_data,
            relationship_type=RelationshipTypes.CONTAINS
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

    def __init__(self, referenced_sop_instance_uid):
        """
        Parameters
        ----------
        referenced_sop_instance_uid: Union[pydicom.uid.UID, str]
            SOP Instance UID of the referenced object

        """
        super(RealWorldValueMap, self).__init__(
            name=CodedConcept(
                value='126100',
                meaning='Real World Value Map used for measurement',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid='1.2.840.10008.5.1.4.1.1.67',
            referenced_sop_instance_uid=referenced_sop_instance_uid,
            relationship_type=RelationshipTypes.CONTAINS
        )


class FindingSite(CodeContentItem):

    """Content item representing a coded finding site."""

    def __init__(self, anatomic_location, laterality=None,
                 topographical_modifier=None):
        """
        Parameters
        ----------
        anatomic_location: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded anatomic location (region or structure)
        laterality: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded laterality
            (see CID 244 "Laterality" for options)
        topographical_modifier: Union[pydicom.sr.coding.CodedConcept, pydicom.sr.coding.Code, None], optional
            coded modifier value for anatomic location

        """  # noqa
        super(FindingSite, self).__init__(
            name=CodedConcept(
                value='363698007',
                meaning='Finding Site',
                scheme_designator='SCT'
            ),
            value=anatomic_location,
            relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
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
                relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
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
                relationship_type=RelationshipTypes.HAS_CONCEPT_MOD
            )
            self.ContentSequence.append(modifier_item)


class ReferencedSegmentationFrame(ContentSequence):

    """Content items representing a reference to a segmentation image frame
    as well as the image that was used as a source for the segmentation.
    """

    def __init__(self, sop_class_uid, sop_instance_uid, frame_number,
                 segment_number, source_image):
        """
        Parameters
        ----------
        sop_class_uid: Union[pydicom.uid.UID, str]
            SOP Class UID of the referenced image object
        sop_instance_uid: Union[pydicom.uid.UID, str]
            SOP Instance UID of the referenced image object
        segment_number: int
            number of the segment to which the refernce applies
        frame_number: int
            number of the frame to which the reference applies
        source_image: pydicom.sr.content_items.SourceImageForSegmentation
            source image for segmentation

        """
        super(ReferencedSegmentationFrame, self).__init__()
        segmentation_item = ImageContentItem(
            name=CodedConcept(
                value='121214',
                meaning='Referenced Segmentation Frame',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_frame_numbers=frame_number,
            referenced_segment_numbers=segment_number
        )
        self.append(segmentation_item)
        if not isinstance(source_image, SourceImageForSegmentation):
            raise TypeError(
                'Argument "source_image" must have type '
                'SourceImageForSegmentation.'
            )
        self.append(source_image)


class ReferencedSegmentation(ContentSequence):

    """Content items representing a reference to a segmentation image
    as well as the images that were used as a source for the segmentation.
    """

    def __init__(self, sop_class_uid, sop_instance_uid, segment_number,
                 frame_numbers, source_images=None, source_series=None):
        """
        Parameters
        ----------
        sop_class_uid: Union[pydicom.uid.UID, str]
            SOP Class UID of the referenced image object
        sop_instance_uid: Union[pydicom.uid.UID, str]
            SOP Instance UID of the referenced image object
        frame_numbers: List[int]
            numbers of the frames to which the reference applies
        segment_number: int
            number of the segment to which the refernce applies
        source_images: Union[List[pydicom.sr.content_items.SourceImageForSegmentation], None], optional
            source images for segmentation
        source_series: Union[pydicom.sr.content_items.SourceSeriesForSegmentation, None], optional
            source series for segmentation

        Note
        ----
        Either `source_images` or `source_series` must be provided.

        """  # noqa
        super(ReferencedSegmentation, self).__init__()
        segmentation_item = ImageContentItem(
            name=CodedConcept(
                value='121191',
                meaning='Referenced Segment',
                scheme_designator='DCM'
            ),
            referenced_sop_class_uid=sop_class_uid,
            referenced_sop_instance_uid=sop_instance_uid,
            referenced_frame_numbers=frame_numbers,
            referenced_segment_numbers=segment_number
        )
        self.append(segmentation_item)
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


