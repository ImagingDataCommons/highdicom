"""Module for SOP classes of the PM modality."""
from collections.abc import Sequence
from concurrent.futures import Executor
from os import PathLike
from typing import cast, BinaryIO
import warnings

import numpy as np
from highdicom.base_content import ContributingEquipment
from highdicom.content import (
    _add_content_information,
    ContentCreatorIdentificationCodeSequence,
    PaletteColorLUTTransformation,
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
    VOILUTTransformation,
)
from highdicom.enum import (
    ContentQualificationValues,
    DimensionOrganizationTypeValues,
    PhotometricInterpretationValues,
    PixelDataKeywords,
)
from highdicom.image import _Image, Image
from highdicom.pm.content import RealWorldValueMapping
from highdicom.pm.enum import DerivedPixelContrastValues, ImageFlavorValues
from highdicom.spatial import get_image_coordinate_system
from highdicom.seg.content import DimensionIndexSequence
from highdicom.volume import ChannelDescriptor, Volume
from pydicom import Dataset
from pydicom.dataelem import DataElement
from pydicom.uid import (
    UID,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
    RLELossless,
)
from typing_extensions import Self


class ParametricMap(Image):

    """SOP class for a Parametric Map.

    Note
    ----
    This class only supports creation of Parametric Map instances with a
    value of interest (VOI) lookup table that describes a linear transformation
    that equally applies to all frames in the image.

    """

    def __init__(
        self,
        source_images: Sequence[Dataset],
        pixel_array: np.ndarray | Volume,
        series_instance_uid: str,
        series_number: int,
        sop_instance_uid: str,
        instance_number: int,
        manufacturer: str,
        manufacturer_model_name: str,
        software_versions: str | tuple[str],
        device_serial_number: str,
        contains_recognizable_visual_features: bool,
        real_world_value_mappings: (
            Sequence[RealWorldValueMapping] |
            Sequence[Sequence[RealWorldValueMapping]]
        ),
        window_center: float | None = None,
        window_width: float | None = None,
        voi_lut_transformations: (
            Sequence[VOILUTTransformation] | None
        ) = None,
        transfer_syntax_uid: str | UID = ExplicitVRLittleEndian,
        content_description: str | None = None,
        content_creator_name: str | None = None,
        pixel_measures: PixelMeasuresSequence | None = None,
        plane_orientation: PlaneOrientationSequence | None = None,
        plane_positions: Sequence[PlanePositionSequence] | None = None,
        content_label: str | None = None,
        content_qualification: (
            str |
            ContentQualificationValues
        ) = ContentQualificationValues.RESEARCH,
        image_flavor: str | ImageFlavorValues = ImageFlavorValues.VOLUME,
        derived_pixel_contrast: (
            str |
            DerivedPixelContrastValues
        ) = DerivedPixelContrastValues.QUANTITY,
        content_creator_identification: None | (
            ContentCreatorIdentificationCodeSequence
        ) = None,
        palette_color_lut_transformation: None | (
            PaletteColorLUTTransformation
        ) = None,
        contributing_equipment: Sequence[
            ContributingEquipment
        ] | None = None,
        use_extended_offset_table: bool = False,
        icc_profile: bytes | None = None,
        workers: int | Executor = 0,
        dimension_organization_type: (
            DimensionOrganizationTypeValues |
            str |
            None
        ) = None,
        tile_pixel_array: bool = False,
        tile_size: Sequence[int] | None = None,
        pyramid_uid: str | None = None,
        pyramid_label: str | None = None,
        further_source_images: Sequence[Dataset] | None = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        source_images: Sequence[pydicom.dataset.Dataset]
            One or more single- or multi-frame images (or metadata of images)
            from which the parametric map was derived
        pixel_array: numpy.ndarray | highdicom.Volume
            2D, 3D, or 4D array of unsigned integer or floating-point data type
            representing one or more channels (images derived from source
            images via an image transformation) for one or more spatial image
            positions:

            * In case of a 2D array, the values represent a single channel
              for a single 2D frame and the array shall have shape ``(r, c)``,
              where ``r`` is the number of rows and ``c`` is the number of
              columns.

            * In case of a 3D array, the values represent a single channel
              for multiple 2D frames at different spatial image positions and
              the array shall have shape ``(n, r, c)``, where ``n`` is the
              number of frames, ``r`` is the number of rows per frame, and
              ``c`` is the number of columns per frame.

            * In case of a 4D array, the values represent multiple channels
              for multiple 2D frames at different spatial image positions and
              the array shall have shape ``(n, r, c, m)``, where ``n`` is the
              number of frames, ``r`` is the number of rows per frame, ``c`` is
              the number of columns per frame, and ``m`` is the number of
              channels.

        series_instance_uid: str
            UID of the series
        series_number: int
            Number of the series within the study
        sop_instance_uid: str
            UID that should be assigned to the instance
        instance_number: int
            Number that should be assigned to the instance
        manufacturer: str
            Name of the manufacturer (developer) of the device (software)
            that creates the instance
        manufacturer_model_name: str,
            Name of the model of the device (software)
            that creates the instance
        software_versions: Union[str, Tuple[str]]
            Versions of relevant software used to create the data
        device_serial_number: str
            Serial number (or other identifier) of the device (software)
            that creates the instance
        contains_recognizable_visual_features: bool
            Whether the image contains recognizable visible features of the
            patient
        real_world_value_mappings: Union[Sequence[highdicom.pm.RealWorldValueMapping], Sequence[Sequence[highdicom.pm.RealWorldValueMapping]]
            Descriptions of how stored values map to real-world values. Each
            channel encoded in ``pixel_array`` shall be described with one or
            more real-world value mappings. Multiple mappings might be used for
            different representations such as log versus linear scales or for
            different representations in different units. If ``pixel_array`` is
            a 2D or 3D array and only one channel exists at each spatial image
            position, then one or more real-world value mappings shall be
            provided in a flat sequence. If `pixel_array` is a 4D array and
            multiple channels exist at each spatial image position, then one or
            more mappings shall be provided for each channel in a nested
            sequence of length ``m``, where ``m`` shall match the channel
            dimension of the ``pixel_array``.

            In some situations the mapping may be difficult to describe (e.g., in
            case of a transformation performed by a deep convolutional neural
            network). The real-world value mapping may then simply describe an
            identity function that maps stored values to unit-less real-world
            values.
        window_center: Union[int, float, None], optional
            This argument has been deprecated and will be removed in a future
            release. Use the more flexible ``voi_lut_transformations`` argument
            instead.
        window_width: Union[int, float, None], optional
            This argument has been deprecated and will be removed in a future
            release. Use the more flexible ``voi_lut_transformations`` argument
            instead.
        voi_lut_transformations: Sequence[highdicom.VOILUTTransformation] | None, optional
            One or more VOI transformations that describe a pixel
            transformation to apply to frames. This will become a required
            argument in a future release.
        transfer_syntax_uid: Union[str, None], optional
            UID of transfer syntax that should be used for encoding of
            data elements. Defaults to Explicit VR Little Endian
            (UID ``"1.2.840.10008.1.2.1"``)
        content_description: Union[str, None], optional
            Brief description of the parametric map image
        content_creator_name: Union[str, None], optional
            Name of the person that created the parametric map image
        pixel_measures: Union[highdicom.PixelMeasuresSequence, None], optional
            Physical spacing of image pixels in `pixel_array`.
            If ``None``, it will be assumed that the parametric map image has
            the same pixel measures as the source image(s).
        plane_orientation: Union[highdicom.PlaneOrientationSequence, None], optional
            Orientation of planes in ``pixel_array`` relative to axes of
            three-dimensional patient or slide coordinate space.
            If ``None``, it will be assumed that the parametric map image as
            the same plane orientation as the source image(s).
        plane_positions: Union[Sequence[PlanePositionSequence], None], optional
            Position of each plane in `pixel_array` in the three-dimensional
            patient or slide coordinate space.
            If ``None``, it will be assumed that the parametric map image has
            the same plane position as the source image(s). However, this will
            only work when the first dimension of `pixel_array` matches the
            number of frames in `source_images` (in case of multi-frame source
            images) or the number of `source_images` (in case of single-frame
            source images).
        content_label: Union[str, None], optional
            Content label
        content_qualification: Union[str, highdicom.ContentQualificationValues], optional
            Indicator of whether content was produced with approved hardware
            and software
        image_flavor: Union[str, highdicom.pm.ImageFlavorValues], optional
            Overall representation of the image type
        derived_pixel_contrast: Union[str, highdicom.pm.DerivedPixelContrast], optional
            Contrast created by combining or processing source images with the
            same geometry
        content_creator_identification: Union[highdicom.ContentCreatorIdentificationCodeSequence, None], optional
            Identifying information for the person who created the content of
            this parametric map.
        palette_color_lut_transformation: Union[highdicom.PaletteColorLUTTransformation, None], optional
            Description of the Palette Color LUT Transformation for transforming
            grayscale into RGB color pixel values
        contributing_equipment: Sequence[highdicom.ContributingEquipment] | None, optional
            Additional equipment that has contributed to the acquisition,
            creation or modification of this instance.
        use_extended_offset_table: bool, optional
            Include an extended offset table instead of a basic offset table
            for encapsulated transfer syntaxes. Extended offset tables avoid
            size limitations on basic offset tables, and separate the offset
            table from the pixel data by placing it into metadata. However,
            they may be less widely supported than basic offset tables. This
            parameter is ignored if using a native (uncompressed) transfer
            syntax. The default value may change in a future release.
        **kwargs: Any, optional
            Additional keyword arguments that will be passed to the constructor
            of `highdicom.base.SOPClass`

        Raises
        ------
        ValueError
            When

                * Length of ``source_images`` is zero.
                * Items of ``source_images`` are not all part of the same study
                  and series.
                * Items of ``source_images`` have different number of rows and
                  columns.
                * Length of ``plane_positions`` does not match number of 2D planes
                  in `pixel_array` (size of first array dimension).
                * Transfer Syntax specified by ``transfer_syntax_uid`` is not
                  supported for data type of `pixel_array`.

        Note
        ----
        The assumption is made that planes in ``pixel_array`` are defined in
        the same frame of reference as ``source_images``. It is further assumed
        that all image frame have the same type (i.e., the same ``image_flavor``
        and ``derived_pixel_contrast``).

        """  # noqa
        if len(source_images) == 0:
            raise ValueError('At least one source image is required.')

        src_img = source_images[0]

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
        }
        if np.dtype(pixel_array.dtype).kind == 'u':
            # If pixel data has unsigned or signed integer data type, then it
            # can be lossless compressed. The standard does not specify any
            # compression codecs for floating-point data types.
            supported_transfer_syntaxes.update(
                {
                    JPEG2000Lossless,
                    JPEGLSLossless,
                    RLELossless,
                }
            )
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported.'
            )

        if (window_center is None) != (window_width is None):
            raise TypeError(
                "Arguments 'window_center' and 'window_width' should both "
                "be None, or neither should be None."
            )
        if window_center is not None:
            if voi_lut_transformations is not None:
                raise TypeError(
                    "Arguments 'window_center' and 'window_width' must be "
                    "omitted if 'voi_lut_transformations' is provided."
                )
            warnings.warn(
                "Arguments 'window_center' and 'window_width' are deprecated "
                "and will be removed in a future version of the library. "
                "Use the more flexible 'voi_lut_transformations' argument "
                "instead.",
                UserWarning,
                stacklevel=2,
            )
            voi_lut_transformations = [
               VOILUTTransformation(
                   window_center=window_center,
                   window_width=window_width,
               )
            ]
        else:
            if voi_lut_transformations is None:
                raise TypeError(
                    "Argument 'voi_lut_transformations' is required."
                )
            if len(voi_lut_transformations) < 1:
                raise TypeError(
                    "Argument 'voi_lut_transformations' must contain at least "
                    'one item.'
                )

            for v in voi_lut_transformations:
                if not isinstance(v, VOILUTTransformation):
                    raise TypeError(
                        "Argument 'voi_lut_transformations' must be a "
                        'sequence of highdicom.VOILUTTransformation objects.'
                    )

        super(_Image, self).__init__(
            study_instance_uid=src_img.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            instance_number=instance_number,
            sop_class_uid='1.2.840.10008.5.1.4.1.1.30',
            modality='OT',
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
            **kwargs,
        )

        self.copy_specimen_information(src_img)
        self.copy_patient_and_study_information(src_img)
        self._add_contributing_equipment(contributing_equipment, src_img)

        # If the same set of mappings applies to all frames, the information
        # is stored in the Shared Functional Groups Sequence. Otherwise, it
        # is stored for each frame separately in the Per Frame Functional
        # Groups Sequence.
        if pixel_array.ndim in (2, 3):
            error_message = (
                'In case argument "pixel_array" is a 2D or 3D array, argument '
                '"real_world_value_mappings" must be a flat sequence '
                'of one or more RealWorldValueMapping items.'
            )
            try:
                real_world_value_mappings[0]
            except IndexError as e:
                raise TypeError(error_message) from e
            if not isinstance(
                real_world_value_mappings[0],
                RealWorldValueMapping
            ):
                raise TypeError(error_message)
            real_world_value_mappings = cast(
                Sequence[Sequence[RealWorldValueMapping]],
                [real_world_value_mappings]
            )
        elif pixel_array.ndim == 4:
            error_message = (
                'In case argument "pixel_array" is a 4D array, argument '
                '"real_world_value_mappings" must be a nested sequence '
                'of one or more RealWorldValueMapping items.'
            )
            if isinstance(
                real_world_value_mappings[0],
                RealWorldValueMapping
            ):
                raise TypeError(error_message)
            if not isinstance(
                real_world_value_mappings[0][0],
                RealWorldValueMapping
            ):
                raise TypeError(error_message)
            real_world_value_mappings = cast(
                Sequence[Sequence[RealWorldValueMapping]],
                real_world_value_mappings
            )
        else:
            raise ValueError('Pixel array must be a 2D, 3D, or 4D array.')

        if isinstance(pixel_array, Volume):
            if pixel_array.number_of_channel_dimensions == 1:
                if pixel_array.channel_descriptors != (
                    ChannelDescriptor('LUTLabel'),
                ):
                    raise ValueError(
                        "Input volume should have no channels other than "
                        "'LUTLabel'."
                    )

                for channel_mappings, vol_lut_label in zip(
                    real_world_value_mappings,
                    pixel_array.get_channel_values('LUTLabel')
                ):
                    if len(channel_mappings) != 1:
                        raise ValueError(
                            'Only a single mapping should be provided in '
                            "each item within 'real_world_value_mappings' "
                            "when a Volume is passed as the 'pixel_array'."
                        )
                    mapping = channel_mappings[0]

                    if vol_lut_label != mapping.LUTLabel:
                        raise ValueError(
                            "The LUTLabels of the 'real_world_value_mappings' "
                            "must match those within the channel indentifiers "
                            "of the 'pixel_array'."
                        )

            elif pixel_array.number_of_channel_dimensions != 0:
                raise ValueError(
                    "If 'pixel_array' is a highdicom.Volume, it should have "
                    "0 or 1 channel dimensions."
                )

        n_channels = pixel_array.shape[3] if pixel_array.ndim == 4 else 1
        if len(real_world_value_mappings) != n_channels:
            raise ValueError(
                'Number of RealWorldValueMapping items provided via '
                '"real_world_value_mappings" argument does not match size of '
                'last dimension of "pixel_array" argument.'
            )

        # Parametric Map Image
        image_flavor = ImageFlavorValues(image_flavor)
        derived_pixel_contrast = DerivedPixelContrastValues(
            derived_pixel_contrast
        )
        image_type = [
            "DERIVED",
            "PRIMARY",
            image_flavor.value,
            derived_pixel_contrast.value,
        ]
        content_qualification = ContentQualificationValues(
            content_qualification
        )
        self.ContentQualification = content_qualification.value

        _add_content_information(
            dataset=self,
            content_label=(
                content_label if content_label is not None else 'MAP'
            ),
            content_description=content_description,
            content_creator_name=content_creator_name,
            content_creator_identification=content_creator_identification,
        )

        # TODO refactor this into the common method and include LUT label
        # TODO generalize DimensionIndexSequence so we are not using the
        # segmentation one here
        self.DimensionIndexSequence = DimensionIndexSequence(
            get_image_coordinate_system(src_img),
            include_segment_number=False,
        )
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = (
            self.DimensionIndexSequence[0].DimensionOrganizationUID
        )
        self.DimensionOrganizationSequence = [dimension_organization]

        # Acquisition Context
        self.AcquisitionContextSequence: list[Dataset] = []

        # Get the correct pixel data attribute
        plain_array = (
            pixel_array.array
            if isinstance(pixel_array, Volume)
            else pixel_array
        )
        pixel_data_keyword = self._get_pixel_data_keyword(plain_array)
        bits_allocated = {
            PixelDataKeywords.PIXEL_DATA: int(plain_array.itemsize * 8),
            PixelDataKeywords.FLOAT_PIXEL_DATA: 32,
            PixelDataKeywords.DOUBLE_FLOAT_PIXEL_DATA: 64,
        }[pixel_data_keyword]

        # Palette color lookup table
        self._configure_color(
            palette_color_lut_transformation=palette_color_lut_transformation,
            icc_profile=icc_profile,
            pixel_data_keyword=pixel_data_keyword,
        )

        # Check that the real world value maps are consistent with the provided
        # data type
        if pixel_data_keyword == PixelDataKeywords.PIXEL_DATA:
            if any(
                any(m.is_floating_point() for m in m_list)
                for m_list in real_world_value_mappings
            ):
                raise ValueError(
                    "When using an integer-valued 'pixel_array', all items "
                    "in 'real_world_value_mappings' must have their value "
                    "range specified with integers."
                )
        else:
            if not all(
                all(m.is_floating_point() for m in m_list)
                for m_list in real_world_value_mappings
            ):
                raise ValueError(
                    "When using a floating point-valued pixel_array, "
                    "all items in 'real_world_value_mappings' must have "
                    "their value range specified with floats."
                )

        def add_channel_callback(
            item: Dataset,
            mappings: Sequence[RealWorldValueMapping],
        ):
            # Mappings may contain multiple mappings. Directly add the whole
            # list as a sequence
            item.add(
                DataElement(
                    0x0040_9096,  # RealWorldValueMappingSequence
                    'SQ',
                    mappings,
                )
            )

            return item

        self._init_multiframe_image(
            source_images=source_images,
            pixel_array=pixel_array,
            image_type=image_type,
            photometric_interpretation=(
                PhotometricInterpretationValues.MONOCHROME2
            ),
            bits_allocated=bits_allocated,
            samples_per_pixel=1,
            use_default_pixel_value_transformation=True,
            shared_voi_lut_transformations=voi_lut_transformations,
            palette_color_lut_transformation=palette_color_lut_transformation,
            icc_profile=icc_profile,
            contains_recognizable_visual_features=(
                contains_recognizable_visual_features
            ),
            burned_in_annotation=False,
            pixel_measures=pixel_measures,
            plane_orientation=plane_orientation,
            plane_positions=plane_positions,
            omit_empty_frames=False,
            workers=workers,
            dimension_organization_type=dimension_organization_type,
            tile_pixel_array=tile_pixel_array,
            tile_size=tile_size,
            pyramid_label=pyramid_label,
            pyramid_uid=pyramid_uid,
            further_source_images=further_source_images,
            use_extended_offset_table=use_extended_offset_table,
            channel_values=real_world_value_mappings,
            add_channel_callback=add_channel_callback,
            pixel_data_keyword=pixel_data_keyword,
            # TODO change this and change the DimensionIndexSequence to match
            channel_is_indexed=False,
        )

        # Parametric Map Frame Type
        frame_type_item = Dataset()
        frame_type_item.FrameType = self.ImageType
        (
            self
            .SharedFunctionalGroupsSequence[0]
            .ParametricMapFrameTypeSequence
        ) = [frame_type_item]

    def _configure_color(
        self,
        palette_color_lut_transformation: PaletteColorLUTTransformation | None,
        icc_profile: bytes | None,
        pixel_data_keyword: PixelDataKeywords,
    ) -> None:
        if palette_color_lut_transformation is not None:
            if pixel_data_keyword != PixelDataKeywords.PIXEL_DATA:
                raise ValueError(
                    'Use of palette_color_lut is only supported with integer-'
                    'valued pixel data.'
                )

            self.PixelPresentation = 'COLOR_RANGE'
        else:
            if icc_profile is not None:
                raise TypeError(
                    "Argument 'icc_profile' should "
                    "not be provided when 'palette_color_lut_transformation' "
                    "is not provided."
                )
            self.PixelPresentation = 'MONOCHROME'

    def _get_pixel_data_keyword(
        self,
        pixel_array: np.ndarray
    ) -> PixelDataKeywords:
        """Get the pixel data keyword to use.

        Parameters
        ----------
        pixel_array : np.ndarray
            The array to check

        Returns
        -------
        highdicom.enum.PixelDataKeywords:
            Pixel data keyword where this pixel data should be stored.

        Raises
        ------
        ValueError
            If values in the input array don't have a supported unsigned
            integer or floating-point type.

        """
        if pixel_array.dtype.kind == 'f':
            if pixel_array.dtype.name == 'float32':
                return PixelDataKeywords.FLOAT_PIXEL_DATA
            elif pixel_array.dtype.name == 'float64':
                return PixelDataKeywords.DOUBLE_FLOAT_PIXEL_DATA
            else:
                raise ValueError(
                    'Unsupported floating-point type for pixel data: '
                    '32-bit (single-precision) or 64-bit (double-precision) '
                    'floating-point types are supported.'
                )
        elif pixel_array.dtype.kind == 'u':
            if pixel_array.dtype not in (np.uint8, np.uint16):
                raise ValueError(
                    'Unsupported unsigned integer type for pixel data: '
                    '16-bit unsigned integer types are supported.'
                )
            return PixelDataKeywords.PIXEL_DATA
        raise ValueError(
            'Unsupported data type for pixel data. '
            'Supported are 8-bit or 16-bit unsigned integer types as well as '
            '32-bit (single-precision) or 64-bit (double-precision) '
            'floating-point types.'
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        copy: bool = True,
    ) -> Self:
        """Create instance from an existing dataset.

        Parameters
        ----------
        dataset: pydicom.dataset.Dataset
            Dataset representing a Parametric Map.
        copy: bool
            If True, the underlying dataset is deep-copied such that the
            original dataset remains intact. If False, this operation will
            alter the original dataset in place.

        Returns
        -------
        highdicom.pm.ParametricMap
            Representation of the supplied dataset as a highdicom
            ParametricMap.

        """
        if dataset.SOPClassUID != '1.2.840.10008.5.1.4.1.1.30':
            raise ValueError('Dataset is not a Parametric Map.')

        pm = super().from_dataset(dataset, copy=copy)

        return cast(Self, pm)


def pmread(
    fp: str | bytes | PathLike | BinaryIO,
    lazy_frame_retrieval: bool = False,
) -> ParametricMap:
    """Read a parametric map image stored in DICOM File Format.

    Parameters
    ----------
    fp: Union[str, bytes, os.PathLike]
        Any file-like object representing a DICOM file containing a
        Parametric Map image.
    lazy_frame_retrieval: bool
        If True, the returned parametric map will retrieve frames from the file
        as requested, rather than loading in the entire object to memory
        initially. This may be a good idea if file reading is slow and you are
        likely to need only a subset of the frames in the parametric map.

    Returns
    -------
    highdicom.pm.ParametricMap
        Parametric Map image read from the file.

    """
    # This is essentially a convenience alias for the classmethod (which is
    # used so that it is inherited correctly by subclasses). It is used
    # because it follows the format of other similar functions around the
    # library
    return ParametricMap.from_file(
        fp,
        lazy_frame_retrieval=lazy_frame_retrieval,
    )
