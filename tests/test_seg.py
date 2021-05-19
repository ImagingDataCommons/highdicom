from io import BytesIO
import unittest
from pathlib import Path

import numpy as np
import pytest

from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from pydicom.uid import (
    generate_uid,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
)

from highdicom import (
    AlgorithmIdentificationSequence,
    PlanePositionSequence,
    PixelMeasuresSequence,
    PlaneOrientationSequence,
)
from highdicom.enum import CoordinateSystemNames
from highdicom.seg import (
    DimensionIndexSequence,
    SegmentDescription,
)
from highdicom.seg import (
    SegmentAlgorithmTypeValues,
    SegmentsOverlapValues,
    SegmentationTypeValues,
)
from highdicom.seg import Segmentation
from highdicom.seg.utils import iter_segments


class TestAlgorithmIdentificationSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._name = 'verydeepneuralnetwork'
        self._version = '1.0'
        self._family = codes.DCM.ArtificialIntelligence
        self._source = 'me'
        self._parameters = {'one': '1', 'two': '2'}

    def test_construction(self):
        seq = AlgorithmIdentificationSequence(
            self._name,
            self._family,
            self._version
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.AlgorithmName == self._name
        assert item.AlgorithmVersion == self._version
        assert len(item.AlgorithmFamilyCodeSequence) == 1
        assert item.AlgorithmFamilyCodeSequence[0] == self._family
        with pytest.raises(AttributeError):
            item.AlgorithmSource
            item.AlgorithmParameters

    def test_construction_missing_required_argument(self):
        with pytest.raises(TypeError):
            AlgorithmIdentificationSequence(
                name=self._name,
                family=self._family
            )

    def test_construction_missing_required_argument_2(self):
        with pytest.raises(TypeError):
            AlgorithmIdentificationSequence(
                name=self._name,
                source=self._source
            )

    def test_construction_missing_required_argument_3(self):
        with pytest.raises(TypeError):
            AlgorithmIdentificationSequence(
                family=self._family,
                source=self._source
            )

    def test_construction_optional_argument(self):
        seq = AlgorithmIdentificationSequence(
            name=self._name,
            family=self._family,
            version=self._version,
            source=self._source
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.AlgorithmSource == self._source
        with pytest.raises(AttributeError):
            item.AlgorithmParameters

    def test_construction_optional_argument_2(self):
        seq = AlgorithmIdentificationSequence(
            name=self._name,
            family=self._family,
            version=self._version,
            parameters=self._parameters
        )
        assert len(seq) == 1
        item = seq[0]
        parsed_params = ','.join([
            '='.join([key, value])
            for key, value in self._parameters.items()
        ])
        assert item.AlgorithmParameters == parsed_params
        with pytest.raises(AttributeError):
            item.AlgorithmSource


class TestSegmentDescription(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._segment_number = 1
        self._invalid_segment_number = 0
        self._segment_label = 'segment #1'
        self._segmented_property_category = \
            codes.SCT.MorphologicallyAbnormalStructure
        self._segmented_property_type = codes.SCT.Neoplasm
        self._segment_algorithm_type = \
            SegmentAlgorithmTypeValues.AUTOMATIC.value
        self._algorithm_identification = AlgorithmIdentificationSequence(
            name='bla',
            family=codes.DCM.ArtificialIntelligence,
            version='v1'
        )
        self._tracking_id = 'segment #1'
        self._tracking_uid = generate_uid()
        self._anatomic_region = codes.SCT.Thorax
        self._anatomic_structure = codes.SCT.Lung

    def test_construction(self):
        item = SegmentDescription(
            self._segment_number,
            self._segment_label,
            self._segmented_property_category,
            self._segmented_property_type,
            self._segment_algorithm_type,
            self._algorithm_identification
        )
        assert item.SegmentNumber == self._segment_number
        assert item.SegmentLabel == self._segment_label
        assert item.SegmentedPropertyCategoryCodeSequence[0] == \
            self._segmented_property_category
        assert item.SegmentedPropertyTypeCodeSequence[0] == \
            self._segmented_property_type
        assert item.SegmentAlgorithmType == self._segment_algorithm_type
        assert item.SegmentAlgorithmName == \
            self._algorithm_identification[0].AlgorithmName
        assert len(item.SegmentationAlgorithmIdentificationSequence) == 1
        with pytest.raises(AttributeError):
            item.TrackingID
            item.TrackingUID
            item.AnatomicRegionSequence
            item.PrimaryAnatomicStructureSequence

    def test_construction_invalid_segment_number(self):
        with pytest.raises(ValueError):
            SegmentDescription(
                self._invalid_segment_number,
                self._segment_label,
                self._segmented_property_category,
                self._segmented_property_type,
                self._segment_algorithm_type,
                self._algorithm_identification
            )

    def test_construction_missing_required_argument(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=self._segment_algorithm_type,
                algorithm_identification=self._algorithm_identification
            )

    def test_construction_missing_required_argument_2(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                segment_number=self._segment_number,
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=self._segment_algorithm_type,
                algorithm_identification=self._algorithm_identification
            )

    def test_construction_missing_required_argument_3(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                segment_number=self._segment_number,
                segment_label=self._segment_label,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=self._segment_algorithm_type,
                algorithm_identification=self._algorithm_identification
            )

    def test_construction_missing_required_argument_4(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                segment_number=self._segment_number,
                segment_label=self._segment_label,
                segmented_property_category=self._segmented_property_category,
                algorithm_type=self._segment_algorithm_type,
                algorithm_identification=self._algorithm_identification
            )

    def test_construction_missing_required_argument_5(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                segment_number=self._segment_number,
                segment_label=self._segment_label,
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_identification=self._algorithm_identification
            )

    def test_construction_missing_required_argument_6(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                segment_number=self._segment_number,
                segment_label=self._segment_label,
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=self._segment_algorithm_type
            )

    def test_construction_no_algo_id_manual_seg(self):
        # Omitting the algo id should not give an error if the segmentation
        # type is MANUAL
        SegmentDescription(
            segment_number=self._segment_number,
            segment_label=self._segment_label,
            segmented_property_category=self._segmented_property_category,
            segmented_property_type=self._segmented_property_type,
            algorithm_type=SegmentAlgorithmTypeValues.MANUAL
        )

    def test_construction_optional_argument(self):
        item = SegmentDescription(
            segment_number=self._segment_number,
            segment_label=self._segment_label,
            segmented_property_category=self._segmented_property_category,
            segmented_property_type=self._segmented_property_type,
            algorithm_type=self._segment_algorithm_type,
            algorithm_identification=self._algorithm_identification,
            tracking_id=self._tracking_id,
            tracking_uid=self._tracking_uid,
        )
        assert item.TrackingID == self._tracking_id
        assert item.TrackingUID == self._tracking_uid
        with pytest.raises(AttributeError):
            item.AnatomicRegionSequence
            item.PrimaryAnatomicStructureSequence

    def test_construction_optional_argument_2(self):
        item = SegmentDescription(
            segment_number=self._segment_number,
            segment_label=self._segment_label,
            segmented_property_category=self._segmented_property_category,
            segmented_property_type=self._segmented_property_type,
            algorithm_type=self._segment_algorithm_type,
            algorithm_identification=self._algorithm_identification,
            anatomic_regions=[self._anatomic_region],
            primary_anatomic_structures=[self._anatomic_structure]
        )
        assert len(item.AnatomicRegionSequence) == 1
        assert item.AnatomicRegionSequence[0] == self._anatomic_region
        assert len(item.PrimaryAnatomicStructureSequence) == 1
        assert item.PrimaryAnatomicStructureSequence[0] == \
            self._anatomic_structure
        with pytest.raises(AttributeError):
            item.TrackingID
            item.TrackingUID


class TestPixelMeasuresSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._pixel_spacing = (0.5, 0.5)
        self._slice_thickness = 0.3
        self._spacing_between_slices = 0.7

    def test_construction(self):
        seq = PixelMeasuresSequence(
            self._pixel_spacing,
            self._slice_thickness
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.PixelSpacing == list(self._pixel_spacing)
        assert item.SliceThickness == self._slice_thickness
        with pytest.raises(AttributeError):
            self.SpacingBetweenSlices

    def test_construction_missing_required_attribute(self):
        with pytest.raises(TypeError):
            PixelMeasuresSequence(
                pixel_spacing=self._pixel_spacing
            )

    def test_construction_missing_required_attribute_2(self):
        with pytest.raises(TypeError):
            PixelMeasuresSequence(
                slice_thickness=self._slice_thickness
            )

    def test_construction_optional_attribute(self):
        seq = PixelMeasuresSequence(
            pixel_spacing=self._pixel_spacing,
            slice_thickness=self._slice_thickness,
            spacing_between_slices=self._spacing_between_slices
        )
        item = seq[0]
        assert item.SpacingBetweenSlices == self._spacing_between_slices


class TestPlanePositionSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._image_position = (1.0, 2.0, 3.0)
        self._pixel_matrix_position = (20, 10)

    def test_construction_1(self):
        seq = PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.PATIENT,
            image_position=self._image_position
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.ImagePositionPatient == list(self._image_position)
        with pytest.raises(AttributeError):
            item.XOffsetInSlideCoordinateSystem

    def test_construction_2(self):
        seq = PlanePositionSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_position=self._image_position,
            pixel_matrix_position=self._pixel_matrix_position
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.XOffsetInSlideCoordinateSystem == self._image_position[0]
        assert item.YOffsetInSlideCoordinateSystem == self._image_position[1]
        assert item.ZOffsetInSlideCoordinateSystem == self._image_position[2]
        assert item.RowPositionInTotalImagePixelMatrix == \
            self._pixel_matrix_position[1]
        assert item.ColumnPositionInTotalImagePixelMatrix == \
            self._pixel_matrix_position[0]
        with pytest.raises(AttributeError):
            item.ImagePositionPatient

    def test_construction_missing_required_argument(self):
        with pytest.raises(TypeError):
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.SLIDE,
                image_position=self._image_position
            )

    def test_construction_missing_required_argument_2(self):
        with pytest.raises(TypeError):
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.SLIDE,
                pixel_matrix_position=self._pixel_matrix_position
            )

    def test_construction_missing_required_argument_3(self):
        with pytest.raises(TypeError):
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.PATIENT
            )

    def test_construction_missing_required_argument_4(self):
        with pytest.raises(TypeError):
            PlanePositionSequence(
                pixel_matrix_position=self._pixel_matrix_position
            )


class TestPlaneOrientationSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._image_orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def test_construction(self):
        seq = PlaneOrientationSequence(
            'PATIENT',
            self._image_orientation
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.ImageOrientationPatient == list(self._image_orientation)
        with pytest.raises(AttributeError):
            item.ImageOrientationSlide

    def test_construction_2(self):
        seq = PlaneOrientationSequence(
            'SLIDE',
            self._image_orientation
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.ImageOrientationSlide == list(self._image_orientation)
        with pytest.raises(AttributeError):
            item.ImageOrientationPatient

    def test_construction_missing_required_attribute(self):
        with pytest.raises(TypeError):
            PlaneOrientationSequence(
                image_orientation=self._image_orientation
            )

    def test_construction_missing_required_attribute_2(self):
        with pytest.raises(TypeError):
            PlaneOrientationSequence(
                coordinate_system='PATIENT'
            )

    def test_construction_wrong_attribute_enumated_value(self):
        with pytest.raises(ValueError):
            PlaneOrientationSequence(
                coordinate_system='OTHER',
                image_orientation=self._image_orientation
            )


class TestDimensionIndexSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_construction(self):
        seq = DimensionIndexSequence(
            coordinate_system='PATIENT'
        )
        assert len(seq) == 2
        assert seq[0].DimensionIndexPointer == 0x0062000B
        assert seq[0].FunctionalGroupPointer == 0x0062000A
        assert seq[1].DimensionIndexPointer == 0x00200032
        assert seq[1].FunctionalGroupPointer == 0x00209113

    def test_construction_2(self):
        seq = DimensionIndexSequence(
            coordinate_system='SLIDE'
        )
        assert len(seq) == 6
        assert seq[0].DimensionIndexPointer == 0x0062000B
        assert seq[0].FunctionalGroupPointer == 0x0062000A
        assert seq[1].DimensionIndexPointer == 0x0048021E
        assert seq[1].FunctionalGroupPointer == 0x0048021A
        assert seq[2].DimensionIndexPointer == 0x0048021F
        assert seq[2].FunctionalGroupPointer == 0x0048021A
        assert seq[3].DimensionIndexPointer == 0x0040072A
        assert seq[3].FunctionalGroupPointer == 0x0048021A
        assert seq[4].DimensionIndexPointer == 0x0040073A
        assert seq[4].FunctionalGroupPointer == 0x0048021A
        assert seq[5].DimensionIndexPointer == 0x0040074A
        assert seq[5].FunctionalGroupPointer == 0x0048021A


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._segmented_property_category = \
            codes.SCT.MorphologicallyAbnormalStructure
        self._segmented_property_type = codes.SCT.Neoplasm
        self._segment_descriptions = [
            SegmentDescription(
                segment_number=1,
                segment_label='Segment #1',
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
                algorithm_identification=AlgorithmIdentificationSequence(
                    name='bla',
                    family=codes.DCM.ArtificialIntelligence,
                    version='v1'
                )
            ),
        ]
        self._additional_segment_descriptions = [
            SegmentDescription(
                segment_number=2,
                segment_label='Segment #2',
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
                algorithm_identification=AlgorithmIdentificationSequence(
                    name='foo',
                    family=codes.DCM.ArtificialIntelligence,
                    version='v1'
                )
            ),
        ]
        self._additional_segment_descriptions_no4 = [
            SegmentDescription(
                segment_number=4,
                segment_label='Segment #4',
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
                algorithm_identification=AlgorithmIdentificationSequence(
                    name='foo',
                    family=codes.DCM.ArtificialIntelligence,
                    version='v1'
                )
            ),
        ]
        self._series_instance_uid = generate_uid()
        self._series_number = 1
        self._sop_instance_uid = generate_uid()
        self._instance_number = 1
        self._manufacturer = 'FavoriteManufacturer'
        self._manufacturer_model_name = 'BestModel'
        self._software_versions = 'v1.0'
        self._device_serial_number = '1-2-3'
        self._content_description = 'Test Segmentation'
        self._content_creator_name = 'Robo^Doc'

        # A single CT image
        self._ct_image = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._ct_pixel_array = np.zeros(
            self._ct_image.pixel_array.shape,
            dtype=bool
        )
        self._ct_pixel_array[1:5, 10:15] = True

        # A microscopy image
        self._sm_image = dcmread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )
        # Override te existing ImageOrientationSlide to make the frame ordering
        # simpler for the tests
        self._sm_pixel_array = np.zeros(
            self._sm_image.pixel_array.shape[:3],  # remove colour channel axis
            dtype=bool
        )
        self._sm_pixel_array[2:3, 1:5, 7:9] = True
        # self._sm_pixel_array[6:9, 2:8, 1:4] = True

        # A series of single frame CT images
        ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        # Ensure the frames are in the right spatial order
        # (only 3rd dimension changes)
        self._ct_series = sorted(
            ct_series,
            key=lambda x: x.ImagePositionPatient[2]
        )
        self._ct_series_mask_array = np.zeros(
            (len(self._ct_series), ) + self._ct_series[0].pixel_array.shape,
            dtype=bool
        )
        self._ct_series_mask_array[1:2, 1:5, 7:9] = True

        # An enhanced (multiframe) CT image
        self._ct_multiframe = dcmread(get_testdata_file('eCT_Supplemental.dcm'))
        self._ct_multiframe_mask_array = np.zeros(
            self._ct_multiframe.pixel_array.shape,
            dtype=bool
        )
        self._ct_multiframe_mask_array[:, 100:200, 200:400] = True

    @ staticmethod
    def sort_frames(sources, mask):
        src = sources[0]
        if hasattr(src, 'ImageOrientationSlide'):
            coordinate_system = CoordinateSystemNames.SLIDE
        else:
            coordinate_system = CoordinateSystemNames.PATIENT
        dim_index = DimensionIndexSequence(coordinate_system)
        if hasattr(src, 'NumberOfFrames'):
            plane_positions = dim_index.get_plane_positions_of_image(src)
        else:
            plane_positions = dim_index.get_plane_positions_of_series(sources)
        _, index = dim_index.get_index_values(plane_positions)
        return mask[index, ...]

    @staticmethod
    def remove_empty_frames(mask):
        # Remove empty frames from an array
        return np.stack([
            frame for frame in mask if np.sum(frame) > 0
        ])

    @staticmethod
    def get_array_after_writing(instance):
        # Write DICOM object to buffer, read it again and reconstruct the mask
        with BytesIO() as fp:
            instance.save_as(fp)
            fp.seek(0)
            instance_reread = dcmread(fp)

        return instance_reread.pixel_array

    def test_construction(self):
        instance = Segmentation(
            [self._ct_image],
            self._ct_pixel_array,
            SegmentationTypeValues.FRACTIONAL.value,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number
        )
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.ManufacturerModelName == self._manufacturer_model_name
        assert instance.SoftwareVersions == self._software_versions
        assert instance.DeviceSerialNumber == self._device_serial_number
        assert instance.Modality == 'SEG'
        assert instance.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2'
        assert instance.PatientID == self._ct_image.PatientID
        assert instance.AccessionNumber == self._ct_image.AccessionNumber
        assert instance.LossyImageCompression == '00'
        assert instance.BitsAllocated == 8
        assert instance.HighBit == 7
        assert instance.BitsStored == 8
        assert instance.ImageType == ['DERIVED', 'PRIMARY']
        assert instance.SamplesPerPixel == 1
        assert instance.PhotometricInterpretation == 'MONOCHROME2'
        assert instance.PixelRepresentation == 0
        assert instance.SegmentationType == 'FRACTIONAL'
        assert instance.SegmentationFractionalType == 'PROBABILITY'
        assert instance.MaximumFractionalValue == 255
        assert instance.ContentDescription is None
        assert instance.ContentCreatorName is None
        with pytest.raises(AttributeError):
            instance.LossyImageCompressionRatio
            instance.LossyImageCompressionMethod
        assert len(instance.SegmentSequence) == 1
        assert instance.SegmentSequence[0].SegmentNumber == 1
        assert len(instance.SourceImageSequence) == 1
        assert len(instance.DimensionIndexSequence) == 2
        ref_item = instance.SourceImageSequence[0]
        assert ref_item.ReferencedSOPInstanceUID == \
            self._ct_image.SOPInstanceUID
        assert instance.Rows == self._ct_image.pixel_array.shape[0]
        assert instance.Columns == self._ct_image.pixel_array.shape[1]
        assert len(instance.SharedFunctionalGroupsSequence) == 1
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == self._ct_image.PixelSpacing
        assert pm_item.SliceThickness == self._ct_image.SliceThickness
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            self._ct_image.ImageOrientationPatient
        assert len(instance.DimensionOrganizationSequence) == 1
        assert len(instance.DimensionIndexSequence) == 2
        assert instance.NumberOfFrames == 1
        assert len(instance.PerFrameFunctionalGroupsSequence) == 1
        frame_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert len(frame_item.SegmentIdentificationSequence) == 1
        assert len(frame_item.FrameContentSequence) == 1
        assert len(frame_item.DerivationImageSequence) == 1
        assert len(frame_item.PlanePositionSequence) == 1
        frame_content_item = frame_item.FrameContentSequence[0]
        assert len(frame_content_item.DimensionIndexValues) == 2
        for derivation_image_item in frame_item.DerivationImageSequence:
            assert len(derivation_image_item.SourceImageSequence) == 1
        assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
            SegmentsOverlapValues.NO
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSlideSequence

    def test_construction_2(self):
        instance = Segmentation(
            [self._sm_image],
            self._sm_pixel_array,
            SegmentationTypeValues.FRACTIONAL.value,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number
        )
        assert instance.PatientID == self._sm_image.PatientID
        assert instance.AccessionNumber == self._sm_image.AccessionNumber
        assert instance.ContainerIdentifier == \
            self._sm_image.ContainerIdentifier
        assert instance.SpecimenDescriptionSequence[0].SpecimenUID == \
            self._sm_image.SpecimenDescriptionSequence[0].SpecimenUID
        assert len(instance.SegmentSequence) == 1
        assert instance.SegmentSequence[0].SegmentNumber == 1
        assert len(instance.SourceImageSequence) == 1
        ref_item = instance.SourceImageSequence[0]
        assert ref_item.ReferencedSOPInstanceUID == \
            self._sm_image.SOPInstanceUID
        assert instance.Rows == self._sm_image.pixel_array.shape[1]
        assert instance.Columns == self._sm_image.pixel_array.shape[2]
        assert len(instance.SharedFunctionalGroupsSequence) == 1
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        src_shared_item = self._sm_image.SharedFunctionalGroupsSequence[0]
        src_pm_item = src_shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == src_pm_item.PixelSpacing
        assert pm_item.SliceThickness == src_pm_item.SliceThickness
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationSlide == \
            self._sm_image.ImageOrientationSlide
        assert len(instance.DimensionOrganizationSequence) == 1
        assert len(instance.DimensionIndexSequence) == 6

        # Number of frames should be number of frames in the segmentation mask
        # that are non-empty, due to sparsity
        num_frames = (self._sm_pixel_array.sum(axis=(1, 2)) > 0).sum()
        assert instance.NumberOfFrames == num_frames
        assert len(instance.PerFrameFunctionalGroupsSequence) == num_frames
        frame_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert len(frame_item.SegmentIdentificationSequence) == 1
        assert len(frame_item.DerivationImageSequence) == 1
        assert len(frame_item.FrameContentSequence) == 1
        assert len(frame_item.PlanePositionSlideSequence) == 1
        frame_content_item = frame_item.FrameContentSequence[0]
        assert len(frame_content_item.DimensionIndexValues) == 6
        for derivation_image_item in frame_item.DerivationImageSequence:
            assert len(derivation_image_item.SourceImageSequence) == 1
            source_image_item = derivation_image_item.SourceImageSequence[0]
            assert hasattr(source_image_item, 'ReferencedFrameNumber')
        assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
            SegmentsOverlapValues.NO
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSequence

    def test_construction_3(self):
        # Segmentation instance from a series of single-frame CT images
        instance = Segmentation(
            self._ct_series,
            self._ct_series_mask_array,
            SegmentationTypeValues.FRACTIONAL.value,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number
        )
        src_im = self._ct_series[1]
        assert instance.PatientID == src_im.PatientID
        assert instance.AccessionNumber == src_im.AccessionNumber
        assert len(instance.SegmentSequence) == 1
        assert instance.SegmentSequence[0].SegmentNumber == 1
        assert len(instance.SourceImageSequence) == len(self._ct_series)
        ref_item = instance.SourceImageSequence[1]
        assert ref_item.ReferencedSOPInstanceUID == src_im.SOPInstanceUID
        assert instance.Rows == src_im.pixel_array.shape[0]
        assert instance.Columns == src_im.pixel_array.shape[1]
        assert len(instance.SharedFunctionalGroupsSequence) == 1
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == src_im.PixelSpacing
        assert pm_item.SliceThickness == src_im.SliceThickness
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            src_im.ImageOrientationPatient
        assert len(instance.DimensionOrganizationSequence) == 1
        assert len(instance.DimensionIndexSequence) == 2
        assert instance.NumberOfFrames == 1
        assert len(instance.PerFrameFunctionalGroupsSequence) == 1
        frame_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert len(frame_item.SegmentIdentificationSequence) == 1
        assert len(frame_item.FrameContentSequence) == 1
        assert len(frame_item.DerivationImageSequence) == 1
        assert len(frame_item.PlanePositionSequence) == 1
        frame_content_item = frame_item.FrameContentSequence[0]
        assert len(frame_content_item.DimensionIndexValues) == 2
        for derivation_image_item in frame_item.DerivationImageSequence:
            assert len(derivation_image_item.SourceImageSequence) == 1
            source_image_item = derivation_image_item.SourceImageSequence[0]
            assert source_image_item.ReferencedSOPClassUID == src_im.SOPClassUID
            assert source_image_item.ReferencedSOPInstanceUID == \
                src_im.SOPInstanceUID
            assert hasattr(source_image_item, 'PurposeOfReferenceCodeSequence')
        uid_to_plane_position = {}
        for fm in instance.PerFrameFunctionalGroupsSequence:
            src_img_item = fm.DerivationImageSequence[0].SourceImageSequence[0]
            uid_to_plane_position[src_img_item.ReferencedSOPInstanceUID] = \
                fm.PlanePositionSequence[0].ImagePositionPatient
        source_uid_to_plane_position = {
            dcm.SOPInstanceUID: dcm.ImagePositionPatient
            for dcm in self._ct_series
            if dcm.SOPInstanceUID in uid_to_plane_position
        }
        assert source_uid_to_plane_position == uid_to_plane_position
        assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
            SegmentsOverlapValues.NO
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSlideSequence

    def test_construction_4(self):
        # Segmentation instance from an enhanced (multi-frame) CT image
        instance = Segmentation(
            [self._ct_multiframe],
            self._ct_multiframe_mask_array,
            SegmentationTypeValues.FRACTIONAL.value,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number
        )
        assert instance.PatientID == self._ct_multiframe.PatientID
        assert instance.AccessionNumber == self._ct_multiframe.AccessionNumber
        assert len(instance.SegmentSequence) == 1
        assert instance.SegmentSequence[0].SegmentNumber == 1
        assert len(instance.SourceImageSequence) == 1
        ref_item = instance.SourceImageSequence[0]
        assert ref_item.ReferencedSOPInstanceUID == \
            self._ct_multiframe.SOPInstanceUID
        assert instance.NumberOfFrames == \
            self._ct_multiframe.pixel_array.shape[0]
        assert instance.Rows == self._ct_multiframe.pixel_array.shape[1]
        assert instance.Columns == self._ct_multiframe.pixel_array.shape[2]
        assert len(instance.SharedFunctionalGroupsSequence) == 1
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        src_shared_item = self._ct_multiframe.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == \
            src_shared_item.PixelMeasuresSequence[0].PixelSpacing
        assert pm_item.SliceThickness == \
            src_shared_item.PixelMeasuresSequence[0].SliceThickness
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            src_shared_item.PlaneOrientationSequence[0].ImageOrientationPatient
        assert len(instance.DimensionOrganizationSequence) == 1
        assert len(instance.DimensionIndexSequence) == 2
        assert len(instance.PerFrameFunctionalGroupsSequence) == \
            self._ct_multiframe.NumberOfFrames
        frame_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert len(frame_item.SegmentIdentificationSequence) == 1
        assert len(frame_item.FrameContentSequence) == 1
        assert len(frame_item.DerivationImageSequence) == 1
        assert len(frame_item.PlanePositionSequence) == 1
        frame_content_item = frame_item.FrameContentSequence[0]
        assert len(frame_content_item.DimensionIndexValues) == 2
        for derivation_image_item in frame_item.DerivationImageSequence:
            assert len(derivation_image_item.SourceImageSequence) == 1
            source_image_item = derivation_image_item.SourceImageSequence[0]
            assert source_image_item.ReferencedSOPClassUID == \
                self._ct_multiframe.SOPClassUID
            assert source_image_item.ReferencedSOPInstanceUID == \
                self._ct_multiframe.SOPInstanceUID
            assert hasattr(source_image_item, 'PurposeOfReferenceCodeSequence')
        for i, (src_fm, seg_fm) in enumerate(
                zip(
                    self._ct_multiframe.PerFrameFunctionalGroupsSequence,
                    instance.PerFrameFunctionalGroupsSequence
                )
            ):
            assert src_fm.PlanePositionSequence[0].ImagePositionPatient == \
                seg_fm.PlanePositionSequence[0].ImagePositionPatient
            derivation_image_item = seg_fm.DerivationImageSequence[0]
            source_image_item = derivation_image_item.SourceImageSequence[0]
            assert source_image_item.ReferencedFrameNumber == i + 1
            assert source_image_item.ReferencedSOPInstanceUID == \
                self._ct_multiframe.SOPInstanceUID
        assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
            SegmentsOverlapValues.NO
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSlideSequence

    def test_pixel_types(self):
        # A series of tests on different types of image
        tests = [
            ([self._ct_image], self._ct_pixel_array),
            ([self._sm_image], self._sm_pixel_array),
            (self._ct_series, self._ct_series_mask_array),
            ([self._ct_multiframe], self._ct_multiframe_mask_array),
        ]

        for sources, mask in tests:

            # Create a mask for an additional segment as the complement of the
            # original mask
            additional_mask = (1 - mask)

            # Find the expected encodings for the masks
            if mask.ndim > 2:
                expected_encoding = self.sort_frames(
                    sources,
                    mask
                )
                expected_additional_encoding = self.sort_frames(
                    sources,
                    additional_mask
                )
                expected_encoding = self.remove_empty_frames(
                    expected_encoding
                )
                expected_additional_encoding = self.remove_empty_frames(
                    expected_additional_encoding
                )
                two_segment_expected_encoding = np.concatenate(
                    [expected_encoding, expected_additional_encoding],
                    axis=0
                ).squeeze()
                expected_encoding = expected_encoding.squeeze()
            else:
                expected_encoding = mask
                expected_additional_encoding = additional_mask
                two_segment_expected_encoding = np.stack(
                    [expected_encoding, expected_additional_encoding],
                    axis=0
                )

            # Test instance creation for different pixel types and transfer
            # syntaxes
            valid_transfer_syntaxes = [
                ExplicitVRLittleEndian,
                ImplicitVRLittleEndian,
                RLELossless,
            ]

            for transfer_syntax_uid in valid_transfer_syntaxes:
                for pix_type in [np.bool_, np.uint8, np.uint16, np.float_]:
                    instance = Segmentation(
                        sources,
                        mask.astype(pix_type),
                        SegmentationTypeValues.FRACTIONAL.value,
                        self._segment_descriptions,
                        self._series_instance_uid,
                        self._series_number,
                        self._sop_instance_uid,
                        self._instance_number,
                        self._manufacturer,
                        self._manufacturer_model_name,
                        self._software_versions,
                        self._device_serial_number,
                        max_fractional_value=1,
                        transfer_syntax_uid=transfer_syntax_uid
                    )

                    # Ensure the recovered pixel array matches what is expected
                    assert np.array_equal(
                        self.get_array_after_writing(instance),
                        expected_encoding
                    ), f'{sources[0].Modality} {transfer_syntax_uid}'

                    # Add another segment
                    instance.add_segments(
                        additional_mask.astype(pix_type),
                        self._additional_segment_descriptions
                    )
                    assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
                        SegmentsOverlapValues.UNDEFINED

                    # Ensure the recovered pixel array matches what is expected
                    assert np.array_equal(
                        self.get_array_after_writing(instance),
                        two_segment_expected_encoding
                    ), f'{sources[0].Modality} {transfer_syntax_uid}'

        for sources, mask in tests:
            additional_mask = (1 - mask)
            if mask.ndim > 2:
                expected_encoding = self.sort_frames(
                    sources,
                    mask
                )
                expected_additional_encoding = self.sort_frames(
                    sources,
                    additional_mask
                )
                expected_encoding = self.remove_empty_frames(
                    expected_encoding
                )
                expected_additional_encoding = self.remove_empty_frames(
                    expected_additional_encoding
                )
                two_segment_expected_encoding = np.concatenate(
                    [expected_encoding, expected_additional_encoding],
                    axis=0
                ).squeeze()
                expected_encoding = expected_encoding.squeeze()
            else:
                expected_encoding = mask
                expected_additional_encoding = additional_mask
                two_segment_expected_encoding = np.stack(
                    [expected_encoding, expected_additional_encoding],
                    axis=0
                )

            valid_transfer_syntaxes = [
                ExplicitVRLittleEndian,
                ImplicitVRLittleEndian,
            ]

            for transfer_syntax_uid in valid_transfer_syntaxes:
                for pix_type in [np.bool_, np.uint8, np.uint16, np.float_]:
                    instance = Segmentation(
                        sources,
                        mask.astype(pix_type),
                        SegmentationTypeValues.BINARY.value,
                        self._segment_descriptions,
                        self._series_instance_uid,
                        self._series_number,
                        self._sop_instance_uid,
                        self._instance_number,
                        self._manufacturer,
                        self._manufacturer_model_name,
                        self._software_versions,
                        self._device_serial_number,
                        max_fractional_value=1,
                        transfer_syntax_uid=transfer_syntax_uid
                    )

                    # Ensure the recovered pixel array matches what is expected
                    assert np.array_equal(
                        self.get_array_after_writing(instance),
                        expected_encoding
                    ), f'{sources[0].Modality} {transfer_syntax_uid}'

                    # Add another segment
                    instance.add_segments(
                        additional_mask.astype(pix_type),
                        self._additional_segment_descriptions
                    )
                    assert SegmentsOverlapValues(instance.SegmentsOverlap) == \
                        SegmentsOverlapValues.UNDEFINED

                    # Ensure the recovered pixel array matches what is expected
                    assert np.array_equal(
                        self.get_array_after_writing(instance),
                        two_segment_expected_encoding
                    ), f'{sources[0].Modality} {transfer_syntax_uid}'

    def test_odd_number_pixels(self):
        # Test that an image with an odd number of pixels per frame is encoded
        # properly Including when additional segments are subsequently added

        # Create an instance with an odd number of pixels in each frame
        # Based on the single frame CT image
        odd_instance = self._ct_image
        r = 9
        c = 9
        odd_pixels = np.random.randint(
            256,
            size=(r, c),
            dtype=np.uint16
        )

        odd_instance.PixelData = odd_pixels.flatten().tobytes()
        odd_instance.Rows = r
        odd_instance.Columns = c

        odd_mask = np.random.randint(
            2,
            size=odd_pixels.shape,
            dtype=bool
        )
        addtional_odd_mask = np.random.randint(
            2,
            size=odd_pixels.shape,
            dtype=bool
        )

        instance = Segmentation(
            [odd_instance],
            odd_mask,
            SegmentationTypeValues.BINARY.value,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number
        )

        assert np.array_equal(self.get_array_after_writing(instance), odd_mask)

        instance.add_segments(
            addtional_odd_mask,
            self._additional_segment_descriptions
        )

        expected_two_segment_mask = np.stack(
            [odd_mask, addtional_odd_mask],
            axis=0
        )
        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_two_segment_mask
        )

    def test_multi_segments(self):
        # Test that the multi-segment encoding is behaving as expected

        # Create an example mask with two segments
        multi_segment_mask = np.zeros(
            self._ct_image.pixel_array.shape,
            dtype=np.uint8
        )
        multi_segment_mask[1:5, 10:15] = 1
        multi_segment_mask[5:7, 1:5] = 2

        # Create another example mask with two segments,
        # where one is empty
        multi_segment_mask_empty = np.zeros(
            self._ct_image.pixel_array.shape,
            dtype=np.uint8
        )
        multi_segment_mask_empty[5:7, 1:5] = 2

        for mask in [multi_segment_mask, multi_segment_mask_empty]:
            # The expected encoding splits into two channels stacked down axis 0
            if len(np.unique(mask)) > 2:
                expected_encoding = np.stack([
                    mask == i for i in np.arange(1, len(np.unique(mask)))
                ])
            else:
                expected_encoding = (mask > 0).astype(mask.dtype)

            all_segment_descriptions = (
                self._segment_descriptions +
                self._additional_segment_descriptions
            )
            instance = Segmentation(
                [self._ct_image],
                mask,
                SegmentationTypeValues.BINARY.value,
                all_segment_descriptions,
                self._series_instance_uid,
                self._series_number,
                self._sop_instance_uid,
                self._instance_number,
                self._manufacturer,
                self._manufacturer_model_name,
                self._software_versions,
                self._device_serial_number,
                max_fractional_value=1
            )

            assert len(instance.SegmentSequence) == 2
            assert instance.SegmentSequence[0].SegmentNumber == 1
            assert instance.SegmentSequence[1].SegmentNumber == 2

            # Ensure the recovered pixel array matches what is expected
            assert np.array_equal(
                self.get_array_after_writing(instance),
                expected_encoding
            )

    def test_construction_segment_numbers_start_wrong(self):
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=(
                    self._additional_segment_descriptions  # seg num 2
                ),
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_segment_numbers_continue_wrong(self):
        instance = Segmentation(
            source_images=[self._ct_image],
            pixel_array=self._ct_pixel_array,
            segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
            segment_descriptions=(
                self._segment_descriptions  # seg num 1
            ),
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number
        )
        with pytest.raises(ValueError):
            instance.add_segments(
                self._ct_pixel_array,
                self._additional_segment_descriptions_no4
            )

    def test_construction_wrong_segment_order(self):
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=(
                    self._additional_segment_descriptions +  # seg 2
                    self._segment_descriptions               # seg 1
                ),
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_duplicate_segment_number(self):
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=(
                    self._segment_descriptions +
                    self._segment_descriptions  # duplicate
                ),
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_non_described_segment(self):
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=(self._ct_pixel_array * 3).astype(np.uint8),
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=(
                    self._segment_descriptions +
                    self._additional_segment_descriptions
                ),  # two segments, value of 3 in pixel array
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute(self):
        with pytest.raises(TypeError):
            Segmentation(
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_2(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_3(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_4(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_5(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_6(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_7(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_8(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_9(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_10(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                device_serial_number=self._device_serial_number
            )

    def test_construction_missing_required_attribute_11(self):
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=self._segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions
            )

    def test_construction_optional_arguments(self):
        fractional_type = 'OCCUPANCY'
        max_fractional_value = 100
        content_description = 'bla bla bla'
        content_creator_name = 'Me Myself'
        series_description = 'My First Segmentation'
        instance = Segmentation(
            source_images=[self._ct_image],
            pixel_array=self._ct_pixel_array,
            segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number,
            fractional_type=fractional_type,
            max_fractional_value=max_fractional_value,
            content_description=content_description,
            content_creator_name=content_creator_name,
            series_description=series_description,
        )
        assert instance.SegmentationFractionalType == fractional_type
        assert instance.MaximumFractionalValue == max_fractional_value
        assert instance.ContentDescription == content_description
        assert instance.ContentCreatorName == content_creator_name
        assert instance.SeriesDescription == series_description

    def test_construction_optional_arguments_2(self):
        pixel_spacing = (0.5, 0.5)
        slice_thickness = 0.3
        pixel_measures = PixelMeasuresSequence(
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            spacing_between_slices=0.7
        )
        image_orientation = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        plane_orientation = PlaneOrientationSequence(
            coordinate_system=CoordinateSystemNames.PATIENT,
            image_orientation=image_orientation
        )
        plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.PATIENT,
                image_position=(0.0, 0.0, 0.0)
            ),
        ]
        instance = Segmentation(
            source_images=[self._ct_image],
            pixel_array=self._ct_pixel_array,
            segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number,
            pixel_measures=pixel_measures,
            plane_orientation=plane_orientation,
            plane_positions=plane_positions
        )
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == list(pixel_spacing)
        assert pm_item.SliceThickness == slice_thickness
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == list(image_orientation)

    def test_construction_optional_arguments_3(self):
        pixel_spacing = (0.5, 0.5)
        slice_thickness = 0.3
        pixel_measures = PixelMeasuresSequence(
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )
        image_orientation = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        plane_orientation = PlaneOrientationSequence(
            coordinate_system=CoordinateSystemNames.SLIDE,
            image_orientation=image_orientation
        )
        plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.SLIDE,
                image_position=(i * 1.0, i * 1.0, 1.0),
                pixel_matrix_position=(i * 1, i * 1)
            )
            for i in range(self._sm_image.pixel_array.shape[0])
        ]
        instance = Segmentation(
            source_images=[self._sm_image],
            pixel_array=self._sm_pixel_array,
            segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number,
            pixel_measures=pixel_measures,
            plane_orientation=plane_orientation,
            plane_positions=plane_positions
        )
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == list(pixel_spacing)
        assert pm_item.SliceThickness == slice_thickness
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationSlide == list(image_orientation)


class TestSegUtilities(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._ct_image = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._sm_image = dcmread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )

    def test_iter_segments_ct_single_frame_2_segments(self):
        image_dataset = self._ct_image
        mask = np.zeros(
            self._ct_image.pixel_array.shape,
            dtype=np.uint8
        )
        mask[1:5, 10:15] = 1
        mask[5:7, 1:5] = 2
        algorithm_identification = AlgorithmIdentificationSequence(
            name='test',
            version='v1.0',
            family=codes.cid7162.ArtificialIntelligence
        )
        segment_descriptions = [
            SegmentDescription(
                segment_number=1,
                segment_label='tumor tissue',
                segmented_property_category=codes.cid7150.Tissue,
                segmented_property_type=codes.SCT.Neoplasm,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
                tracking_uid=generate_uid(),
                tracking_id='first segment'
            ),
            SegmentDescription(
                segment_number=2,
                segment_label='connective tissue',
                segmented_property_category=codes.cid7150.Tissue,
                segmented_property_type=codes.cid7166.ConnectiveTissue,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
                tracking_uid=generate_uid(),
                tracking_id='second segment'
            ),
        ]

        seg_dataset = Segmentation(
            source_images=[image_dataset],
            pixel_array=mask,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=segment_descriptions,
            series_instance_uid=generate_uid(),
            series_number=2,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer='Manufacturer',
            manufacturer_model_name='Manufacturer Model',
            software_versions='v1',
            device_serial_number='Device XYZ'
        )

        generator = iter_segments(seg_dataset)
        items = list(generator)
        assert len(items) == 2
        item_segment_1 = items[0]
        assert np.squeeze(item_segment_1[0]).shape == mask.shape
        seg_id_item_1 = item_segment_1[1][0].SegmentIdentificationSequence[0]
        assert seg_id_item_1.ReferencedSegmentNumber == 1
        assert item_segment_1[2].SegmentNumber == 1
        item_segment_2 = items[1]
        assert np.squeeze(item_segment_2[0]).shape == mask.shape
        seg_id_item_2 = item_segment_2[1][0].SegmentIdentificationSequence[0]
        assert seg_id_item_2.ReferencedSegmentNumber == 2
        assert item_segment_2[2].SegmentNumber == 2
