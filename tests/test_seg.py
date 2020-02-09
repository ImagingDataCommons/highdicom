import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pydicom.data import get_testdata_files
from pydicom.dataset import Dataset
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid, UID

from highdicom.seg.sop import Segmentation, SurfaceSegmentation
from highdicom.seg.content import (
    AlgorithmIdentificationSequence,
    DimensionIndexSequence,
    PlanePositionSequence,
    PlanePositionSlideSequence,
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    SegmentDescription,
    Surface,
)
from highdicom.seg.enum import SegmentAlgorithmTypes, SegmentationTypes


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
        self._segment_label = 'segment #1'
        self._segmented_property_category = codes.SCT.MorphologicallyAbnormalStructure
        self._segmented_property_type = codes.SCT.Neoplasm
        self._segment_algorithm_type = SegmentAlgorithmTypes.AUTOMATIC.value
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

    def test_construction_missing_required_argument(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                segment_label=self._segment_label,
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


class TestSurface(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._number = 1
        self._points = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
        ])
        self._ratio = 1.0
        self._algorithm_identification = AlgorithmIdentificationSequence(
            name='vectorizer',
            family=codes.DCM.MorphologicalOperations,
            version='v1'
        )

    def test_construction(self):
        item = Surface(
            self._number,
            self._points
        )
        assert item.SurfaceNumber == self._number
        assert item.SurfaceProcessing is None
        assert item.FiniteVolume == 'UNKNOWN'
        assert item.Manifold == 'UNKNOWN'
        assert len(item.SurfacePointsSequence) == 1
        assert len(item.SurfacePointsNormalsSequence) == 0
        assert len(item.SurfaceMeshPrimitivesSequence) == 1
        with pytest.raises(AttributeError):
            item.SurfaceProcessingRatio
            item.SurfaceProcessingAlgorithmIdentificationSequence

    def test_construction_missing_required_attribute(self):
        with pytest.raises(TypeError):
            Surface(
                number=self._number,
            )

    def test_construction_missing_required_attribute_2(self):
        with pytest.raises(TypeError):
            Surface(
                points=self._points
            )

    def test_construction_missing_conditionally_required_attribute(self):
        with pytest.raises(TypeError):
            Surface(
                number=self._number,
                points=self._points,
                is_processed=True,
            )

    def test_construction_missing_conditionally_required_attribute_2(self):
        with pytest.raises(TypeError):
            Surface(
                number=self._number,
                points=self._points,
                is_processed=True,
                processing_ratio=self._ratio,
            )

    def test_construction_missing_conditionally_required_attribute_3(self):
        with pytest.raises(TypeError):
            Surface(
                number=self._number,
                points=self._points,
                is_processed=True,
                processing_algorithm_identification=None,
            )

    def test_construction_optional_attribute(self):
        item = Surface(
            number=self._number,
            points=self._points,
            is_processed=True,
            processing_ratio=self._ratio,
            processing_algorithm_identification=self._algorithm_identification,
        )
        assert item.SurfaceProcessing == 'YES'
        assert item.SurfaceProcessingRatio == self._ratio
        assert len(item.SurfaceProcessingAlgorithmIdentificationSequence) == 1
        assert item.SurfaceProcessingAlgorithmIdentificationSequence[0] == \
            self._algorithm_identification[0]

    def test_construction_optional_attribute_2(self):
        item = Surface(
            number=self._number,
            points=self._points,
            is_finite_volume=True,
            is_manifold=True
        )
        assert item.FiniteVolume == 'YES'
        assert item.Manifold == 'YES'

    def test_construction_optional_attribute_3(self):
        item = Surface(
            number=self._number,
            points=self._points,
            is_finite_volume=False,
            is_manifold=False
        )
        assert item.FiniteVolume == 'NO'
        assert item.Manifold == 'NO'


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

    def test_construction(self):
        seq = PlanePositionSequence(image_position=self._image_position)
        assert len(seq) == 1
        item = seq[0]
        assert item.ImagePositionPatient == list(self._image_position)


class TestPlanePositionSlideSequence(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._image_position = (1.0, 2.0, 3.0)
        self._pixel_matrix_position = (10, 20)

    def test_construction(self):
        seq = PlanePositionSlideSequence(
            self._image_position,
            self._pixel_matrix_position
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.XOffsetInSlideCoordinateSystem == self._image_position[0]
        assert item.YOffsetInSlideCoordinateSystem == self._image_position[1]
        assert item.ZOffsetInSlideCoordinateSystem == self._image_position[2]
        assert item.RowPositionInTotalImagePixelMatrix == \
            self._pixel_matrix_position[0]
        assert item.ColumnPositionInTotalImagePixelMatrix == \
            self._pixel_matrix_position[1]

    def test_construction_missing_required_argument(self):
        with pytest.raises(TypeError):
            PlanePositionSlideSequence(
                image_position=self._image_position
            )

    def test_construction_missing_required_argument_2(self):
        with pytest.raises(TypeError):
            PlanePositionSlideSequence(
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
        assert seq[1].DimensionIndexPointer == 0x0040072A
        assert seq[1].FunctionalGroupPointer == 0x0048021A
        assert seq[2].DimensionIndexPointer == 0x0040073A
        assert seq[2].FunctionalGroupPointer == 0x0048021A
        assert seq[3].DimensionIndexPointer == 0x0040074A
        assert seq[3].FunctionalGroupPointer == 0x0048021A
        assert seq[4].DimensionIndexPointer == 0x0048021E
        assert seq[4].FunctionalGroupPointer == 0x0048021A
        assert seq[5].DimensionIndexPointer == 0x0048021F
        assert seq[5].FunctionalGroupPointer == 0x0048021A


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        super().setUp()
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._segment_descriptions = [
            SegmentDescription(
                segment_number=1,
                segment_label='Segment #1',
                segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
                segmented_property_type=codes.SCT.Neoplasm,
                algorithm_type=SegmentAlgorithmTypes.AUTOMATIC.value,
                algorithm_identification=AlgorithmIdentificationSequence(
                    name='bla',
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

        self._ct_image = dcmread(
            os.path.join(data_dir, 'test_files', 'ct_image.dcm')
        )
        self._ct_pixel_array = np.zeros(
            self._ct_image.pixel_array.shape,
            dtype=np.bool
        )
        self._ct_pixel_array[1:5, 10:15] = True
        self._sm_image = dcmread(
            os.path.join(data_dir, 'test_files', 'sm_image.dcm')
        )
        self._sm_pixel_array = np.zeros(
            self._sm_image.pixel_array.shape,
            dtype=np.bool
        )
        self._sm_pixel_array[2:3, 1:5, 7:9, :] = True
        self._ct_series = [
            dcmread(f) for f in get_testdata_files('77654033/CT2')
        ]
        self._ct_series_mask_array = np.zeros(
            (len(self._ct_series), ) + self._ct_series[0].pixel_array.shape,
            dtype=np.bool
        )
        self._ct_series_mask_array[1:2, 1:5, 7:9] = True

    def test_construction(self):
        instance = Segmentation(
            [self._ct_image],
            self._ct_pixel_array,
            SegmentationTypes.FRACTIONAL.value,
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
        assert len(instance.SourceImageSequence) == 1
        assert len(instance.DimensionIndexSequence) == 2
        ref_item = instance.SourceImageSequence[0]
        assert ref_item.ReferencedSOPInstanceUID == self._ct_image.SOPInstanceUID
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
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSlideSequence

    def test_construction_2(self):
        instance = Segmentation(
            [self._sm_image],
            self._sm_pixel_array,
            SegmentationTypes.FRACTIONAL.value,
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
        assert len(instance.SegmentSequence) == 1
        assert len(instance.SourceImageSequence) == 1
        ref_item = instance.SourceImageSequence[0]
        assert ref_item.ReferencedSOPInstanceUID == self._sm_image.SOPInstanceUID
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
        assert instance.NumberOfFrames == self._sm_image.NumberOfFrames
        assert len(instance.PerFrameFunctionalGroupsSequence) == \
            self._sm_image.NumberOfFrames
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
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSequence

    def test_construction_3(self):
        instance = Segmentation(
            self._ct_series,
            self._ct_series_mask_array,
            SegmentationTypes.FRACTIONAL.value,
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
        src_im = self._ct_series[0]
        assert instance.PatientID == src_im.PatientID
        assert instance.AccessionNumber == src_im.AccessionNumber
        assert len(instance.SegmentSequence) == 1
        assert len(instance.SourceImageSequence) == len(self._ct_series)
        ref_item = instance.SourceImageSequence[0]
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
        assert instance.NumberOfFrames == len(self._ct_series)
        assert len(instance.PerFrameFunctionalGroupsSequence) == \
            len(self._ct_series)
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
            assert source_image_item.ReferencedSOPInstanceUID == src_im.SOPInstanceUID
            assert hasattr(source_image_item, 'PurposeOfReferenceCodeSequence')
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSlideSequence

    def test_construction_missing_required_attribute(self):
        with pytest.raises(TypeError):
            Segmentation(
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
                segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
        instance = Segmentation(
            source_images=[self._ct_image],
            pixel_array=self._ct_pixel_array,
            segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
            content_creator_name=content_creator_name
        )
        assert instance.SegmentationFractionalType == fractional_type
        assert instance.MaximumFractionalValue == max_fractional_value
        assert instance.ContentDescription == content_description
        assert instance.ContentCreatorName == content_creator_name

    def test_construction_optional_arguments_2(self):
        # FIXME
        pixel_spacing = (0.5, 0.5)
        slice_thickness = 0.3
        pixel_measures = PixelMeasuresSequence(
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            spacing_between_slices=0.7
        )
        image_orientation = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        plane_orientation = PlaneOrientationSequence(
            'PATIENT',
            image_orientation=image_orientation
        )
        # FIXME
        plane_positions = [
            PlanePositionSequence(image_position=(0.0, 0.0, 0.0)),
        ]
        instance = Segmentation(
            source_images=[self._ct_image],
            pixel_array=self._ct_pixel_array,
            segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
        # FIXME
        pixel_spacing = (0.5, 0.5)
        slice_thickness = 0.3
        pixel_measures = PixelMeasuresSequence(
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )
        image_orientation = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        plane_orientation = PlaneOrientationSequence(
            'SLIDE',
            image_orientation=image_orientation
        )
        # FIXME
        plane_positions = [
            PlanePositionSlideSequence(
                image_position=(i*1.0, i*1.0, 1.0),
                pixel_matrix_position=(i*1, i*1)
            )
            for i in range(self._sm_image.pixel_array.shape[0])
        ]
        instance = Segmentation(
            source_images=[self._sm_image],
            pixel_array=self._sm_pixel_array,
            segmentation_type=SegmentationTypes.FRACTIONAL.value,
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
