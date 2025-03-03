from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from io import BytesIO
import itertools
import unittest
from pathlib import Path
import pkgutil
import warnings

import numpy as np
from pydicom.multival import MultiValue
import pytest
from PIL import Image

from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.datadict import tag_for_keyword
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
    JPEG2000Lossless,
    JPEGLSLossless,
)

from highdicom import (
    PaletteColorLUT,
    PaletteColorLUTTransformation,
)
from highdicom.color import CIELabColor
from highdicom.content import (
    AlgorithmIdentificationSequence,
    PlanePositionSequence,
    PixelMeasuresSequence,
    PlaneOrientationSequence,
)
from highdicom.enum import (
    CoordinateSystemNames,
    DimensionOrganizationTypeValues,
    PatientOrientationValuesBiped,
)
from highdicom.seg import (
    create_segmentation_pyramid,
    segread,
    DimensionIndexSequence,
    SegmentationTypeValues,
    SegmentAlgorithmTypeValues,
    Segmentation,
    SegmentDescription,
    SegmentsOverlapValues,
    SegmentationFractionalTypeValues,
)
from highdicom.seg.utils import iter_segments
from highdicom.spatial import VOLUME_INDEX_CONVENTION, sort_datasets
from highdicom.sr.coding import CodedConcept
from highdicom.uid import UID
from highdicom.volume import RGB_COLOR_CHANNEL_DESCRIPTOR, Volume

from .utils import write_and_read_dataset


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
            item.AlgorithmSource  # noqa: B018
        with pytest.raises(AttributeError):
            item.AlgorithmParameters  # noqa: B018

        assert seq.name == self._name
        assert seq.version == self._version
        assert seq.family == self._family
        assert seq.source is None
        assert seq.parameters is None

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
            item.AlgorithmParameters  # noqa: B018

        assert seq.source == self._source

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
            f'{key}={value}'
            for key, value in self._parameters.items()
        ])
        assert item.AlgorithmParameters == parsed_params
        assert seq.parameters == self._parameters
        with pytest.raises(AttributeError):
            item.AlgorithmSource  # noqa: B018

    def test_malformed_params(self):
        seq = AlgorithmIdentificationSequence(
            self._name,
            self._family,
            self._version
        )
        seq[0].AlgorithmParameters = 'some invalid parameters'
        with pytest.raises(ValueError):
            seq.parameters  # noqa: B018


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
            SegmentAlgorithmTypeValues.AUTOMATIC
        self._algorithm_identification = AlgorithmIdentificationSequence(
            name='bla',
            family=codes.DCM.ArtificialIntelligence,
            version='v1'
        )
        self._tracking_id = 'segment #1'
        self._tracking_uid = UID()
        self._anatomic_region = codes.SCT.Thorax
        self._anatomic_structure = codes.SCT.Lung
        self._display_color = CIELabColor(50.0, 15.0, 20.0)

    def test_construction(self):
        item = SegmentDescription(
            self._segment_number,
            self._segment_label,
            self._segmented_property_category,
            self._segmented_property_type,
            self._segment_algorithm_type,
            self._algorithm_identification,
            display_color=self._display_color,
        )
        assert item.SegmentNumber == self._segment_number
        assert item.SegmentLabel == self._segment_label
        assert item.SegmentedPropertyCategoryCodeSequence[0] == \
            self._segmented_property_category
        assert item.SegmentedPropertyTypeCodeSequence[0] == \
            self._segmented_property_type
        assert item.SegmentAlgorithmType == self._segment_algorithm_type.value
        assert item.SegmentAlgorithmName == \
            self._algorithm_identification[0].AlgorithmName
        assert len(item.SegmentationAlgorithmIdentificationSequence) == 1
        with pytest.raises(AttributeError):
            item.TrackingID  # noqa: B018
            item.TrackingUID  # noqa: B018
            item.AnatomicRegionSequence  # noqa: B018
            item.PrimaryAnatomicStructureSequence  # noqa: B018

        assert item.segment_number == self._segment_number
        assert item.segment_label == self._segment_label
        assert isinstance(item.segmented_property_category, CodedConcept)
        property_category = item.segmented_property_category
        assert property_category == self._segmented_property_category
        assert isinstance(item.segmented_property_type, CodedConcept)
        assert item.segmented_property_type == self._segmented_property_type
        assert isinstance(item.algorithm_type, SegmentAlgorithmTypeValues)
        algo_type = item.algorithm_type
        assert algo_type == SegmentAlgorithmTypeValues(
            self._segment_algorithm_type
        )
        algo_id = item.algorithm_identification
        assert isinstance(algo_id, AlgorithmIdentificationSequence)

        assert item.tracking_id is None
        assert item.tracking_uid is None
        assert len(item.anatomic_regions) == 0
        assert len(item.primary_anatomic_structures) == 0
        assert item.RecommendedDisplayCIELabValue == list(
            self._display_color.value
        )

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
        item = SegmentDescription(
            segment_number=self._segment_number,
            segment_label=self._segment_label,
            segmented_property_category=self._segmented_property_category,
            segmented_property_type=self._segmented_property_type,
            algorithm_type=SegmentAlgorithmTypeValues.MANUAL
        )
        assert item.algorithm_identification is None

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
        assert item.tracking_id == self._tracking_id
        assert item.tracking_uid == self._tracking_uid
        with pytest.raises(AttributeError):
            item.AnatomicRegionSequence  # noqa: B018
        with pytest.raises(AttributeError):
            item.PrimaryAnatomicStructureSequence  # noqa: B018

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
        assert len(item.anatomic_regions) == 1
        assert all(
            isinstance(el, CodedConcept) for el in item.anatomic_regions
        )
        assert item.anatomic_regions[0] == self._anatomic_region

        assert len(item.PrimaryAnatomicStructureSequence) == 1
        assert item.PrimaryAnatomicStructureSequence[0] == \
            self._anatomic_structure
        assert len(item.primary_anatomic_structures) == 1
        assert all(
            isinstance(el, CodedConcept)
            for el in item.primary_anatomic_structures
        )
        assert item.primary_anatomic_structures[0] == self._anatomic_structure

        with pytest.raises(AttributeError):
            item.TrackingID  # noqa: B018
        with pytest.raises(AttributeError):
            item.TrackingUID  # noqa: B018

    def test_construction_mismatched_ids(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                self._segment_number,
                self._segment_label,
                self._segmented_property_category,
                self._segmented_property_type,
                self._segment_algorithm_type,
                self._algorithm_identification,
                tracking_id=self._tracking_id,
            )

    def test_construction_mismatched_ids_2(self):
        with pytest.raises(TypeError):
            SegmentDescription(
                self._segment_number,
                self._segment_label,
                self._segmented_property_category,
                self._segmented_property_type,
                self._segment_algorithm_type,
                self._algorithm_identification,
                tracking_uid=self._tracking_uid,
            )


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
            self.SpacingBetweenSlices  # noqa: B018

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
            item.XOffsetInSlideCoordinateSystem  # noqa: B018

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
            item.ImagePositionPatient  # noqa: B018

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
            item.ImageOrientationSlide  # noqa: B018

    def test_construction_2(self):
        seq = PlaneOrientationSequence(
            'SLIDE',
            self._image_orientation
        )
        assert len(seq) == 1
        item = seq[0]
        assert item.ImageOrientationSlide == list(self._image_orientation)
        with pytest.raises(AttributeError):
            item.ImageOrientationPatient  # noqa: B018

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
        assert seq[1].DimensionIndexPointer == 0x0048021F
        assert seq[1].FunctionalGroupPointer == 0x0048021A
        assert seq[2].DimensionIndexPointer == 0x0048021E
        assert seq[2].FunctionalGroupPointer == 0x0048021A
        assert seq[3].DimensionIndexPointer == 0x0040072A
        assert seq[3].FunctionalGroupPointer == 0x0048021A
        assert seq[4].DimensionIndexPointer == 0x0040073A
        assert seq[4].FunctionalGroupPointer == 0x0048021A
        assert seq[5].DimensionIndexPointer == 0x0040074A
        assert seq[5].FunctionalGroupPointer == 0x0048021A


class TestSegmentation:

    @pytest.fixture(autouse=True)
    def setUp(self):
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
        self._both_segment_descriptions = (
            self._segment_descriptions + self._additional_segment_descriptions
        )
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
        self._series_instance_uid = UID()
        self._series_number = 1
        self._sop_instance_uid = UID()
        self._instance_number = 1
        self._manufacturer = 'FavoriteManufacturer'
        self._manufacturer_model_name = 'BestModel'
        self._software_versions = 'v1.0'
        self._device_serial_number = '1-2-3'
        self._content_description = 'Test Segmentation'
        self._content_creator_name = 'Robo^Doc'
        self._content_label = 'MY_SEG'

        # A single CT image
        self._ct_image = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        self._ct_pixel_array = np.zeros(
            self._ct_image.pixel_array.shape,
            dtype=bool
        )
        self._ct_pixel_array[1:5, 10:15] = True

        self._ct_volume_position = [4.3, 1.4, 8.7]
        self._ct_volume_orientation = [1., 0., 0, 0., -1., 0.]
        self._ct_volume_pixel_spacing = [1., 1.5]
        self._ct_volume_slice_spacing = 3.0
        self._ct_volume_array = np.zeros((4, 12, 12))
        self._ct_volume_array[0, 1:4, 8:9] = True
        self._ct_volume_array[1, 5:7, 1:4] = True
        self._ct_seg_volume = Volume.from_attributes(
            array=self._ct_volume_array,
            image_position=self._ct_volume_position,
            image_orientation=self._ct_volume_orientation,
            pixel_spacing=self._ct_volume_pixel_spacing,
            spacing_between_slices=self._ct_volume_slice_spacing,
            frame_of_reference_uid=self._ct_image.FrameOfReferenceUID,
            coordinate_system="PATIENT",
        )
        self._ct_seg_volume_with_channels = Volume.from_attributes(
            array=self._ct_volume_array[:, :, :, None],
            image_position=self._ct_volume_position,
            image_orientation=self._ct_volume_orientation,
            pixel_spacing=self._ct_volume_pixel_spacing,
            spacing_between_slices=self._ct_volume_slice_spacing,
            frame_of_reference_uid=self._ct_image.FrameOfReferenceUID,
            channels={'SegmentNumber': [1]},
            coordinate_system="PATIENT",
        )

        # A single CR image
        self._cr_image = dcmread(
            get_testdata_file('dicomdirtests/77654033/CR1/6154')
        )
        self._cr_pixel_array = np.zeros(
            self._cr_image.pixel_array.shape,
            dtype=bool
        )
        self._cr_pixel_array[1:5, 10:15] = True
        self._cr_multisegment_pixel_array = np.stack(
            [self._cr_pixel_array, np.logical_not(self._cr_pixel_array)],
            axis=2
        )[None, :]

        # A microscopy image
        self._sm_image = dcmread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )

        # Hack around the openjpeg bug encoding images smaller than 32 pixels
        frame = self._sm_image.pixel_array
        frame = np.pad(frame, ((0, 0), (12, 12), (12, 12), (0, 0)))
        self._sm_image.PixelData = frame.flatten().tobytes()
        self._sm_image.TotalPixelMatrixRows = (
            frame.shape[1] *
            int(self._sm_image.TotalPixelMatrixRows / self._sm_image.Rows)
        )
        self._sm_image.TotalPixelMatrixColumns = (
            frame.shape[2] *
            int(self._sm_image.TotalPixelMatrixColumns / self._sm_image.Columns)
        )
        self._sm_image.Rows = frame.shape[1]
        self._sm_image.Columns = frame.shape[2]

        # Override te existing ImageOrientationSlide to make the frame ordering
        # simpler for the tests
        self._sm_pixel_array = np.zeros(
            self._sm_image.pixel_array.shape[:3],  # remove colour channel axis
            dtype=bool
        )
        self._sm_pixel_array[2:3, 1:5, 7:9] = True

        # Total pixel matrix segmentation array for tests
        self._sm_total_pixel_array = np.zeros(
            (
                self._sm_image.TotalPixelMatrixRows,
                self._sm_image.TotalPixelMatrixColumns
            ),
            dtype=bool
        )
        self._sm_total_pixel_array[38:43, 5:41] = True
        self._sm_total_pixel_array[4:24, 25:29] = True

        self._sm_total_pixel_array_multiclass = np.zeros(
            (
                self._sm_image.TotalPixelMatrixRows,
                self._sm_image.TotalPixelMatrixColumns
            ),
            dtype=np.uint8,
        )
        self._sm_total_pixel_array_multiclass[38:43, 5:41] = 1
        self._sm_total_pixel_array_multiclass[4:24, 25:29] = 2

        # A series of single frame CT images
        ct_series = [
            dcmread(f)
            for f in get_testdata_files('dicomdirtests/77654033/CT2/*')
        ]
        # Ensure the frames are in the right spatial order
        # (only 3rd dimension changes)
        self._ct_series = sort_datasets(ct_series)

        # Hack around the fact that the images are too small to be encoded by
        # openjpeg
        for im in self._ct_series:
            frame = im.pixel_array
            frame = np.pad(frame, ((8, 8), (8, 8)))
            im.Rows = frame.shape[0]
            im.Columns = frame.shape[1]
            im.PixelData = frame.flatten().tobytes()

        self._ct_series_mask_array = np.zeros(
            (len(self._ct_series), ) + self._ct_series[0].pixel_array.shape,
            dtype=bool
        )
        nonempty_slice = slice(1, 4)
        self._ct_series_mask_array[nonempty_slice, 1:5, 7:9] = True
        self._ct_series_nonempty = self._ct_series[nonempty_slice]

        # An enhanced (multiframe) CT image
        self._ct_multiframe = dcmread(get_testdata_file('eCT_Supplemental.dcm'))
        self._ct_multiframe_mask_array = np.zeros(
            self._ct_multiframe.pixel_array.shape,
            dtype=bool
        )
        self._ct_multiframe_mask_array[:, 100:200, 200:400] = True

        self._tests = {
            'ct-image': ([self._ct_image], self._ct_pixel_array),
            'sm-image': ([self._sm_image], self._sm_pixel_array),
            'ct-series': (self._ct_series, self._ct_series_mask_array),
            'ct-multiframe': (
                [self._ct_multiframe], self._ct_multiframe_mask_array
            ),
        }

        r_lut_data = np.arange(10, 120, dtype=np.uint16)
        g_lut_data = np.arange(20, 130, dtype=np.uint16)
        b_lut_data = np.arange(30, 140, dtype=np.uint16)
        r_first_mapped_value = 0
        g_first_mapped_value = 0
        b_first_mapped_value = 0
        r_lut = PaletteColorLUT(r_first_mapped_value, r_lut_data, color='red')
        g_lut = PaletteColorLUT(g_first_mapped_value, g_lut_data, color='green')
        b_lut = PaletteColorLUT(b_first_mapped_value, b_lut_data, color='blue')
        self._lut_transformation = PaletteColorLUTTransformation(
            red_lut=r_lut,
            green_lut=g_lut,
            blue_lut=b_lut,
            palette_color_lut_uid=UID(),
        )
        self._icc_profile = pkgutil.get_data(
            'highdicom',
            '_icc_profiles/sRGB_v4_ICC_preference.icc'
        )

    # Fixtures to use to parametrize segmentation creation
    # Using this fixture mechanism, we can parametrize class methods
    @staticmethod
    @pytest.fixture(
        params=[
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
            JPEG2000Lossless,
        ]
    )
    def binary_transfer_syntax_uid(request):
        return request.param

    @staticmethod
    @pytest.fixture(
        params=[
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
            RLELossless,
            JPEG2000Lossless,
            JPEGLSLossless,
        ]
    )
    def fractional_transfer_syntax_uid(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=[np.bool_, np.uint8, np.uint16, np.float64])
    def pix_type(request):
        return request.param

    @staticmethod
    @pytest.fixture(
        params=['ct-image', 'sm-image', 'ct-series', 'ct-multiframe'],
    )
    def test_data(request):
        return request.param

    @staticmethod
    def sort_frames(sources, mask):
        src = sources[0]
        orientation = None
        if hasattr(src, 'ImageOrientationSlide'):
            coordinate_system = CoordinateSystemNames.SLIDE
        else:
            coordinate_system = CoordinateSystemNames.PATIENT
            if 'SharedFunctionalGroupsSequence' in src:
                orientation = (
                    src
                    .SharedFunctionalGroupsSequence[0]
                    .PlaneOrientationSequence[0]
                    .ImageOrientationPatient
                )
            else:
                orientation = src.ImageOrientationPatient

        dim_index = DimensionIndexSequence(coordinate_system)
        if hasattr(src, 'NumberOfFrames'):
            plane_positions = dim_index.get_plane_positions_of_image(src)
        else:
            plane_positions = dim_index.get_plane_positions_of_series(sources)
        _, index = dim_index.get_index_values(
            plane_positions,
            image_orientation=orientation,
            index_convention=VOLUME_INDEX_CONVENTION
        )
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
        instance_reread = write_and_read_dataset(instance)
        return instance_reread.pixel_array

    @staticmethod
    def check_dimension_index_vals(seg):
        # Function to apply some checks (necessary but not sufficient for
        # correctness) to ensure that the dimension indices are correct
        if seg.SegmentationType != "LABELMAP":
            all_segment_numbers = []
            for f in seg.PerFrameFunctionalGroupsSequence:
                dim_ind_vals = f.FrameContentSequence[0].DimensionIndexValues

                if isinstance(dim_ind_vals, MultiValue):
                    posn_index = dim_ind_vals[0]
                else:
                    # VM=1 so this is an int
                    posn_index = dim_ind_vals

                seg_number = (
                    f.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                )

                assert seg_number == posn_index
                all_segment_numbers.append(seg_number)

            # Probably this should be strict equality and we should adjust the
            # dimension indices for unused segment numbers
            assert (
                set(all_segment_numbers) <=
                set(range(1, max(all_segment_numbers) + 1))
            )

        has_frame_of_reference = 'FrameOfReferenceUID' in seg
        is_patient_coord_system = has_frame_of_reference and hasattr(
            seg.PerFrameFunctionalGroupsSequence[0],
            'PlanePositionSequence'
        )
        if is_patient_coord_system:
            # Build up the mapping from index to value
            index_mapping = defaultdict(list)
            for f in seg.PerFrameFunctionalGroupsSequence:
                dim_ind_vals = f.FrameContentSequence[0].DimensionIndexValues

                if isinstance(dim_ind_vals, MultiValue):
                    posn_index = dim_ind_vals[1]
                else:
                    # VM=1 so this is an int
                    posn_index = dim_ind_vals

                posn_val = f.PlanePositionSequence[0].ImagePositionPatient
                index_mapping[posn_index].append(posn_val)

            # Check that each index value found references a unique value
            for values in index_mapping.values():
                assert [v == values[0] for v in values]

            # Check that the indices are monotonically increasing from 1
            expected_keys = range(1, len(index_mapping) + 1)
            assert set(index_mapping.keys()) == set(expected_keys)

            # Check all three spatial dimensions are sorted one way or the
            # other
            for d in range(3):
                values_array = np.array(
                    [index_mapping[k][0][d] for k in expected_keys]
                )
                assert (
                    (values_array[1:] >= values_array[:-1]).all() or
                    (values_array[1:] <= values_array[:-1]).all()
                )

        elif has_frame_of_reference:
            # Build up the mapping from index to value
            for dim_kw, dim_ind in zip(
                [
                    'RowPositionInTotalImagePixelMatrix',
                    'ColumnPositionInTotalImagePixelMatrix',
                ],
                [0, 1] if seg.SegmentationType == "LABELMAP" else [1, 2]
            ):
                index_mapping = defaultdict(list)
                for f in seg.PerFrameFunctionalGroupsSequence:
                    dim_ind_vals = (
                        f.FrameContentSequence[0]
                        .DimensionIndexValues
                    )

                    if isinstance(dim_ind_vals, MultiValue):
                        posn_index = dim_ind_vals[dim_ind]
                    elif dim_ind == 0:
                        # VM=1 so this is an int
                        posn_index = dim_ind_vals
                    else:
                        assert False

                    posn_item = f.PlanePositionSlideSequence[0]
                    posn_val = getattr(posn_item, dim_kw)
                    index_mapping[posn_index].append(posn_val)

                # Check that each index value found references a unique value
                for values in index_mapping.values():
                    assert [v == values[0] for v in values]

                # Check that the indices are monotonically increasing from 1
                expected_keys = range(1, len(index_mapping) + 1)
                assert set(index_mapping.keys()) == set(expected_keys)

                # Check that values are sorted
                old_v = float('-inf')
                for k in expected_keys:
                    assert index_mapping[k][0] > old_v
                    old_v = index_mapping[k][0]

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
            self._device_serial_number,
            content_label=self._content_label
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.ManufacturerModelName == self._manufacturer_model_name
        assert instance.SoftwareVersions == self._software_versions
        assert instance.DeviceSerialNumber == self._device_serial_number
        assert instance.Modality == 'SEG'
        assert instance.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.1'
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
        assert instance.ContentLabel == self._content_label
        assert instance.ContentDescription is None
        assert instance.ContentCreatorName is None
        with pytest.raises(AttributeError):
            instance.LossyImageCompressionRatio  # noqa: B018
        with pytest.raises(AttributeError):
            instance.LossyImageCompressionMethod  # noqa: B018
        with pytest.raises(AttributeError):
            instance.ImageOrientationSlide  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixOriginSequence  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixRows  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixColumns  # noqa: B018
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
        for i, frame_item in enumerate(
            instance.PerFrameFunctionalGroupsSequence, 1
        ):
            frame_content_item = frame_item.FrameContentSequence[0]
            # The slice location index values should be consecutive, starting
            # at 1
            assert frame_content_item.DimensionIndexValues[1] == i
        for derivation_image_item in frame_item.DerivationImageSequence:
            assert len(derivation_image_item.SourceImageSequence) == 1
        assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
            SegmentsOverlapValues.NO
        with pytest.raises(AttributeError):
            frame_item.PlanePositionSlideSequence  # noqa: B018
        assert not hasattr(instance, "DimensionOrganizationType")
        self.check_dimension_index_vals(instance)
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert not hasattr(instance, 'PixelPaddingValue')

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
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'
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
        assert instance.TotalPixelMatrixRows == \
            self._sm_image.TotalPixelMatrixRows
        assert instance.TotalPixelMatrixColumns == \
            self._sm_image.TotalPixelMatrixColumns
        assert len(instance.SharedFunctionalGroupsSequence) == 1
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        src_shared_item = self._sm_image.SharedFunctionalGroupsSequence[0]
        src_pm_item = src_shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == src_pm_item.PixelSpacing
        assert pm_item.SliceThickness == src_pm_item.SliceThickness
        assert not hasattr(shared_item, "PlaneOrientationSequence")
        assert instance.ImageOrientationSlide == \
            self._sm_image.ImageOrientationSlide
        assert instance.TotalPixelMatrixOriginSequence == \
            self._sm_image.TotalPixelMatrixOriginSequence
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
            frame_item.PlanePositionSequence  # noqa: B018
        assert instance.DimensionOrganizationType == "TILED_SPARSE"
        self.check_dimension_index_vals(instance)
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert not hasattr(instance, 'PixelPaddingValue')

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
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'
        src_im = self._ct_series_nonempty[0]
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
        n_frames = len(self._ct_series_nonempty)
        assert instance.NumberOfFrames == n_frames
        assert len(instance.PerFrameFunctionalGroupsSequence) == n_frames
        for i, (frame_item, src_ins) in enumerate(
            zip(
                instance.PerFrameFunctionalGroupsSequence,
                self._ct_series_nonempty
            ),
            1
        ):
            assert len(frame_item.SegmentIdentificationSequence) == 1
            assert len(frame_item.FrameContentSequence) == 1
            assert len(frame_item.DerivationImageSequence) == 1
            assert len(frame_item.PlanePositionSequence) == 1
            frame_content_item = frame_item.FrameContentSequence[0]
            # The slice location index values should be consecutive, starting
            # at 1
            assert frame_content_item.DimensionIndexValues[1] == i
            assert len(frame_content_item.DimensionIndexValues) == 2
            for derivation_image_item in frame_item.DerivationImageSequence:
                assert len(derivation_image_item.SourceImageSequence) == 1
                source_image_item = derivation_image_item.SourceImageSequence[0]
                assert source_image_item.ReferencedSOPClassUID == \
                    src_ins.SOPClassUID
                assert source_image_item.ReferencedSOPInstanceUID == \
                    src_ins.SOPInstanceUID
                assert hasattr(
                    source_image_item,
                    'PurposeOfReferenceCodeSequence'
                )
            with pytest.raises(AttributeError):
                frame_item.PlanePositionSlideSequence  # noqa: B018
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
        assert not hasattr(instance, 'DimensionOrganizationType')
        self.check_dimension_index_vals(instance)
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert not hasattr(instance, 'PixelPaddingValue')

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
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'
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
            frame_item.PlanePositionSlideSequence  # noqa: B018

        assert hasattr(instance, 'DimensionOrganizationType')
        self.check_dimension_index_vals(instance)
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert not hasattr(instance, 'PixelPaddingValue')

    def test_construction_5(self):
        # Segmentation instance from a series of single-frame CT images
        # with empty frames kept in
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
            self._device_serial_number,
            omit_empty_frames=False
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'
        src_im = self._ct_series[0]
        assert instance.PatientID == src_im.PatientID
        assert instance.AccessionNumber == src_im.AccessionNumber
        assert len(instance.SegmentSequence) == 1
        assert instance.SegmentSequence[0].SegmentNumber == 1
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
        assert instance.NumberOfFrames == 4
        assert len(instance.PerFrameFunctionalGroupsSequence) == 4
        frame_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert len(frame_item.SegmentIdentificationSequence) == 1
        assert len(frame_item.FrameContentSequence) == 1
        assert len(frame_item.DerivationImageSequence) == 1
        assert len(frame_item.PlanePositionSequence) == 1
        for i, (frame_item, src_ins) in enumerate(
            zip(instance.PerFrameFunctionalGroupsSequence, self._ct_series),
            1
        ):
            frame_content_item = frame_item.FrameContentSequence[0]
            # The slice location index values should be consecutive, starting
            # at 1
            assert frame_content_item.DimensionIndexValues[1] == i
            assert len(frame_content_item.DimensionIndexValues) == 2
            for derivation_image_item in frame_item.DerivationImageSequence:
                assert len(derivation_image_item.SourceImageSequence) == 1
                source_image_item = derivation_image_item.SourceImageSequence[0]
                assert source_image_item.ReferencedSOPClassUID == \
                    src_ins.SOPClassUID
                assert source_image_item.ReferencedSOPInstanceUID == \
                    src_ins.SOPInstanceUID
                assert hasattr(
                    source_image_item,
                    'PurposeOfReferenceCodeSequence'
                )
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
            frame_item.PlanePositionSlideSequence  # noqa: B018
        assert not hasattr(instance, 'DimensionOrganizationType')
        self.check_dimension_index_vals(instance)
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert not hasattr(instance, 'PixelPaddingValue')

    def test_construction_6(self):
        # A chest X-ray with no frame of reference
        instance = Segmentation(
            [self._cr_image],
            self._cr_pixel_array,
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
            content_label=self._content_label
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.ManufacturerModelName == self._manufacturer_model_name
        assert instance.SoftwareVersions == self._software_versions
        assert instance.DeviceSerialNumber == self._device_serial_number
        assert instance.Modality == 'SEG'
        assert instance.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.1'
        assert instance.PatientID == self._cr_image.PatientID
        assert instance.AccessionNumber == self._cr_image.AccessionNumber
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
        assert instance.ContentLabel == self._content_label
        assert instance.ContentDescription is None
        assert instance.ContentCreatorName is None
        with pytest.raises(AttributeError):
            instance.LossyImageCompressionRatio  # noqa: B018
        with pytest.raises(AttributeError):
            instance.LossyImageCompressionMethod  # noqa: B018
        with pytest.raises(AttributeError):
            instance.ImageOrientationSlide  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixOriginSequence  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixRows  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixColumns  # noqa: B018
        assert len(instance.SegmentSequence) == 1
        assert instance.SegmentSequence[0].SegmentNumber == 1
        assert len(instance.SourceImageSequence) == 1
        assert len(instance.DimensionIndexSequence) == 1
        ref_item = instance.SourceImageSequence[0]
        assert ref_item.ReferencedSOPInstanceUID == \
            self._cr_image.SOPInstanceUID
        assert instance.Rows == self._cr_image.pixel_array.shape[0]
        assert instance.Columns == self._cr_image.pixel_array.shape[1]
        assert len(instance.SharedFunctionalGroupsSequence) == 1
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert not hasattr(shared_item, 'PixelMeasuresSequence')
        assert not hasattr(shared_item, 'PlaneOrientationSequence')
        assert len(instance.DimensionOrganizationSequence) == 1
        assert len(instance.DimensionIndexSequence) == 1
        assert instance.NumberOfFrames == 1
        assert len(instance.PerFrameFunctionalGroupsSequence) == 1
        frame_item = instance.PerFrameFunctionalGroupsSequence[0]
        assert len(frame_item.SegmentIdentificationSequence) == 1
        assert len(frame_item.FrameContentSequence) == 1
        assert len(frame_item.DerivationImageSequence) == 1
        assert not hasattr(frame_item, 'PlanePositionSequence')
        assert not hasattr(frame_item, 'PlanePositionSlideSequence')
        frame_content_item = frame_item.FrameContentSequence[0]
        assert frame_content_item['DimensionIndexValues'].VM == 1
        for derivation_image_item in frame_item.DerivationImageSequence:
            assert len(derivation_image_item.SourceImageSequence) == 1
        assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
            SegmentsOverlapValues.NO
        assert not hasattr(instance, 'DimensionOrganizationType')
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert not hasattr(instance, 'PixelPaddingValue')
        self.check_dimension_index_vals(instance)

    def test_construction_7(self):
        # A chest X-ray with no frame of reference and multiple segments
        instance = Segmentation(
            [self._cr_image],
            self._cr_multisegment_pixel_array,
            SegmentationTypeValues.FRACTIONAL.value,
            self._both_segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            content_label=self._content_label
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4'
        assert instance.SeriesInstanceUID == self._series_instance_uid
        assert instance.SeriesNumber == self._series_number
        assert instance.SOPInstanceUID == self._sop_instance_uid
        assert instance.InstanceNumber == self._instance_number
        assert instance.Manufacturer == self._manufacturer
        assert instance.ManufacturerModelName == self._manufacturer_model_name
        assert instance.SoftwareVersions == self._software_versions
        assert instance.DeviceSerialNumber == self._device_serial_number
        assert instance.Modality == 'SEG'
        assert instance.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.1'
        assert instance.PatientID == self._cr_image.PatientID
        assert instance.AccessionNumber == self._cr_image.AccessionNumber
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
        assert instance.ContentLabel == self._content_label
        assert instance.ContentDescription is None
        assert instance.ContentCreatorName is None
        with pytest.raises(AttributeError):
            instance.LossyImageCompressionRatio  # noqa: B018
        with pytest.raises(AttributeError):
            instance.LossyImageCompressionMethod  # noqa: B018
        with pytest.raises(AttributeError):
            instance.ImageOrientationSlide  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixOriginSequence  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixRows  # noqa: B018
        with pytest.raises(AttributeError):
            instance.TotalPixelMatrixColumns  # noqa: B018
        assert len(instance.SegmentSequence) == 2
        assert instance.SegmentSequence[0].SegmentNumber == 1
        assert instance.SegmentSequence[1].SegmentNumber == 2
        assert len(instance.SourceImageSequence) == 1
        assert len(instance.DimensionIndexSequence) == 1
        ref_item = instance.SourceImageSequence[0]
        assert ref_item.ReferencedSOPInstanceUID == \
            self._cr_image.SOPInstanceUID
        assert instance.Rows == self._cr_multisegment_pixel_array.shape[1]
        assert instance.Columns == self._cr_multisegment_pixel_array.shape[2]
        assert len(instance.SharedFunctionalGroupsSequence) == 1
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert not hasattr(shared_item, 'PixelMeasuresSequence')
        assert not hasattr(shared_item, 'PlaneOrientationSequence')
        assert len(instance.DimensionOrganizationSequence) == 1
        assert len(instance.DimensionIndexSequence) == 1
        assert instance.NumberOfFrames == 2
        assert len(instance.PerFrameFunctionalGroupsSequence) == 2
        for i, frame_item in enumerate(
            instance.PerFrameFunctionalGroupsSequence, 1
        ):
            seg_id = frame_item.SegmentIdentificationSequence
            assert len(seg_id) == 1
            assert seg_id[0].ReferencedSegmentNumber == i
            frame_content = frame_item.FrameContentSequence
            assert len(frame_content) == 1
            assert frame_content[0].DimensionIndexValues == i
            assert len(frame_item.DerivationImageSequence) == 1
            assert not hasattr(frame_item, 'PlanePositionSequence')
            assert not hasattr(frame_item, 'PlanePositionSlideSequence')
            for derivation_image_item in frame_item.DerivationImageSequence:
                assert len(derivation_image_item.SourceImageSequence) == 1
        assert SegmentsOverlapValues[instance.SegmentsOverlap] == \
            SegmentsOverlapValues.NO
        assert not hasattr(instance, 'DimensionOrganizationType')
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert not hasattr(instance, 'PixelPaddingValue')
        self.check_dimension_index_vals(instance)

    def test_construction_8(self):
        # A chest X-ray with no frame of reference, LABELMAP
        instance = Segmentation(
            [self._cr_image],
            self._cr_pixel_array,
            SegmentationTypeValues.LABELMAP,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            content_label=self._content_label
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.7'
        assert (
            instance.DimensionIndexSequence[0].DimensionIndexPointer ==
            tag_for_keyword('FrameLabel')
        )
        assert instance.PhotometricInterpretation == 'MONOCHROME2'
        dim_ind_vals = (
            instance
            .PerFrameFunctionalGroupsSequence[0]
            .FrameContentSequence[0]
            .DimensionIndexValues
        )
        assert dim_ind_vals == 1
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert instance.PixelPaddingValue == 0
        self.check_dimension_index_vals(instance)

    def test_construction_9(self):
        # A label with a palette color LUT
        instance = Segmentation(
            self._ct_series,
            self._ct_series_mask_array,
            SegmentationTypeValues.LABELMAP.value,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            palette_color_lut_transformation=self._lut_transformation,
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.7'
        assert instance.PhotometricInterpretation == 'PALETTE COLOR'
        assert hasattr(instance, 'ICCProfile')
        assert hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'RedPaletteColorLookupTableData')
        assert hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'BluePaletteColorLookupTableData')
        assert instance.PixelPaddingValue == 0
        self.check_dimension_index_vals(instance)

    def test_construction_10(self):
        # A labelmap with a palette color LUT and ICC Profile
        instance = Segmentation(
            self._ct_series,
            self._ct_series_mask_array,
            SegmentationTypeValues.LABELMAP.value,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            palette_color_lut_transformation=self._lut_transformation,
            icc_profile=self._icc_profile,
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.7'
        assert instance.PhotometricInterpretation == 'PALETTE COLOR'
        assert hasattr(instance, 'ICCProfile')
        assert hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'RedPaletteColorLookupTableData')
        assert hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'BluePaletteColorLookupTableData')
        assert instance.PixelPaddingValue == 0
        self.check_dimension_index_vals(instance)

    def test_construction_no_coordinate_system(self):
        # This image does have a frame of reference, but no spatial information
        # or coordinate system
        source_image = dcmread(get_testdata_file('JPEG2000.dcm'))
        array = np.zeros(
            (source_image.Rows, source_image.Columns),
            np.uint8
        )
        array[40:300, 20:50] = 1

        instance = Segmentation(
            [source_image],
            array,
            SegmentationTypeValues.BINARY,
            self._segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            content_label=self._content_label
        )
        assert (
            instance.FrameOfReferenceUID == source_image.FrameOfReferenceUID
        )
        assert (
            instance
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing == source_image.PixelSpacing
        )

    def test_construction_large_labelmap_monochrome(self):
        n_classes = 300  # force 16 bit
        segment_descriptions = [
            SegmentDescription(
                segment_number=i,
                segment_label=f'Segment #{i}',
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
                algorithm_identification=AlgorithmIdentificationSequence(
                    name='bla',
                    family=codes.DCM.ArtificialIntelligence,
                    version='v1'
                )
            )
            for i in range(1, n_classes)
        ]

        # A labelmap with a large number of classes to force 16 bit
        instance = Segmentation(
            self._ct_series,
            self._ct_series_mask_array,
            SegmentationTypeValues.LABELMAP.value,
            segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
        )
        assert instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.7'
        assert instance.PhotometricInterpretation == 'MONOCHROME2'
        assert not hasattr(instance, 'ICCProfile')
        assert not hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'RedPaletteColorLookupTableData')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert not hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert not hasattr(instance, 'BluePaletteColorLookupTableData')
        assert instance.pixel_array.dtype == np.uint16
        self.check_dimension_index_vals(instance)
        arr = self.get_array_after_writing(instance)
        assert arr.dtype == np.uint16

    def test_construction_large_labelmap_palettecolor(self):
        n_classes = 300  # force 16 bit
        segment_descriptions = [
            SegmentDescription(
                segment_number=i,
                segment_label=f'Segment #{i}',
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
                algorithm_identification=AlgorithmIdentificationSequence(
                    name='bla',
                    family=codes.DCM.ArtificialIntelligence,
                    version='v1'
                )
            )
            for i in range(1, n_classes)
        ]

        r_lut_data = np.arange(10, 10 + n_classes, dtype=np.uint16)
        g_lut_data = np.arange(20, 20 + n_classes, dtype=np.uint16)
        b_lut_data = np.arange(30, 30 + n_classes, dtype=np.uint16)
        r_first_mapped_value = 0
        g_first_mapped_value = 0
        b_first_mapped_value = 0
        r_lut = PaletteColorLUT(r_first_mapped_value, r_lut_data, color='red')
        g_lut = PaletteColorLUT(g_first_mapped_value, g_lut_data, color='green')
        b_lut = PaletteColorLUT(b_first_mapped_value, b_lut_data, color='blue')
        self._lut_transformation = PaletteColorLUTTransformation(
            red_lut=r_lut,
            green_lut=g_lut,
            blue_lut=b_lut,
            palette_color_lut_uid=UID(),
        )

        # A labelmap with a large number of classes to force 16 bit
        instance = Segmentation(
            self._ct_series,
            self._ct_series_mask_array,
            SegmentationTypeValues.LABELMAP.value,
            segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            palette_color_lut_transformation=self._lut_transformation,
            icc_profile=self._icc_profile,
        )
        assert instance.PhotometricInterpretation == 'PALETTE COLOR'
        assert hasattr(instance, 'ICCProfile')
        assert hasattr(instance, 'RedPaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'RedPaletteColorLookupTableData')
        assert hasattr(instance, 'GreenPaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'GreenPaletteColorLookupTableData')
        assert hasattr(instance, 'BluePaletteColorLookupTableDescriptor')
        assert hasattr(instance, 'BluePaletteColorLookupTableData')
        assert instance.pixel_array.dtype == np.uint16
        self.check_dimension_index_vals(instance)
        arr = self.get_array_after_writing(instance)
        assert arr.dtype == np.uint16

    def test_construction_volume(self):
        # Segmentation instance from a series of single-frame CT images
        # with empty frames kept in
        instance = Segmentation(
            [self._ct_image],
            self._ct_seg_volume,
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
            omit_empty_frames=False
        )
        assert np.array_equal(
            instance.pixel_array,
            self._ct_seg_volume.array,
        )

        self.check_dimension_index_vals(instance)
        assert instance.DimensionOrganizationType == '3D'
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == self._ct_volume_pixel_spacing
        assert pm_item.SliceThickness == self._ct_volume_slice_spacing
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            self._ct_volume_orientation
        for plane_item, pp in zip(
            instance.PerFrameFunctionalGroupsSequence,
            self._ct_seg_volume.get_plane_positions(),
        ):
            assert (
                plane_item.PlanePositionSequence[0].ImagePositionPatient ==
                pp[0].ImagePositionPatient
            )

    def test_construction_volume_multiframe(self):
        # Construction with a multiiframe source image and non-spatially
        # aligned volume
        arr = np.zeros((50, 50, 10), np.uint8)
        arr[40:45, 34:39, 2:9] = 1
        volume = Volume(
            arr,
            np.eye(4),
            coordinate_system="PATIENT",
            frame_of_reference_uid=self._ct_multiframe.FrameOfReferenceUID,
        )

        instance = Segmentation(
            [self._ct_multiframe],
            volume,
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
            omit_empty_frames=False
        )
        assert np.array_equal(
            instance.pixel_array,
            arr,
        )

        self.check_dimension_index_vals(instance)
        assert instance.DimensionOrganizationType == '3D'
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == [1.0, 1.0]
        assert pm_item.SliceThickness == 1.0
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        for plane_item, pp in zip(
            instance.PerFrameFunctionalGroupsSequence,
            volume.get_plane_positions(),
        ):
            assert (
                plane_item.PlanePositionSequence[0].ImagePositionPatient ==
                pp[0].ImagePositionPatient
            )

    def test_construction_volume_channels(self):
        # Segmentation instance from a series of single-frame CT images
        # with empty frames kept in, as volume with channels
        instance = Segmentation(
            [self._ct_image],
            self._ct_seg_volume_with_channels,
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
            omit_empty_frames=False
        )
        assert np.array_equal(
            instance.pixel_array,
            self._ct_seg_volume.array,
        )

        self.check_dimension_index_vals(instance)
        assert instance.DimensionOrganizationType == '3D'
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == self._ct_volume_pixel_spacing
        assert pm_item.SliceThickness == self._ct_volume_slice_spacing
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            self._ct_volume_orientation
        for plane_item, pp in zip(
            instance.PerFrameFunctionalGroupsSequence,
            self._ct_seg_volume.get_plane_positions(),
        ):
            assert (
                plane_item.PlanePositionSequence[0].ImagePositionPatient ==
                pp[0].ImagePositionPatient
            )

    def test_construction_volume_channels_invalid_channel_id(self):
        # Make a new correctly shaped volume whose channels do not represent
        # segments
        new_volume = self._ct_seg_volume_with_channels.with_array(
            self._ct_seg_volume_with_channels.array,
            channels={'FrameLabel': ['label']}
        )
        msg = (
            "Input volume should have no channels other than 'SegmentNumber'."
        )
        with pytest.raises(ValueError, match=msg):
            Segmentation(
                [self._ct_image],
                new_volume,
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
                omit_empty_frames=False
            )

    def test_construction_volume_channels_invalid_channel_values(self):
        # Make a new correctly shaped volume whose segment numbers do not
        # match the descriptions
        new_volume = self._ct_seg_volume_with_channels.with_array(
            self._ct_seg_volume_with_channels.array,
            channels={'SegmentNumber': [15]}
        )
        msg = (
            "Segment numbers in the input volume do not match "
            "the described segments."
        )
        with pytest.raises(ValueError, match=msg):
            Segmentation(
                [self._ct_image],
                new_volume,
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
                omit_empty_frames=False
            )

    def test_construction_volume_fractional(self):
        # Segmentation instance from a series of single-frame CT images
        # with empty frames kept in
        instance = Segmentation(
            [self._ct_image],
            self._ct_seg_volume,
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
            omit_empty_frames=False
        )
        assert np.array_equal(
            instance.pixel_array,
            self._ct_seg_volume.array,
        )
        self.check_dimension_index_vals(instance)

        assert instance.DimensionOrganizationType == '3D'
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == self._ct_volume_pixel_spacing
        assert pm_item.SliceThickness == self._ct_volume_slice_spacing
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            self._ct_volume_orientation
        for plane_item, pp in zip(
            instance.PerFrameFunctionalGroupsSequence,
            self._ct_seg_volume.get_plane_positions(),
        ):
            assert (
                plane_item.PlanePositionSequence[0].ImagePositionPatient ==
                pp[0].ImagePositionPatient
            )

    def test_construction_volume_fractional_channels(self):
        # Segmentation instance from a series of single-frame CT images
        # with empty frames kept in, as a volume with channels
        instance = Segmentation(
            [self._ct_image],
            self._ct_seg_volume_with_channels,
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
            omit_empty_frames=False
        )
        assert np.array_equal(
            instance.pixel_array,
            self._ct_seg_volume.array,
        )

        assert instance.DimensionOrganizationType == '3D'
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == self._ct_volume_pixel_spacing
        assert pm_item.SliceThickness == self._ct_volume_slice_spacing
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            self._ct_volume_orientation
        for plane_item, pp in zip(
            instance.PerFrameFunctionalGroupsSequence,
            self._ct_seg_volume.get_plane_positions(),
        ):
            assert (
                plane_item.PlanePositionSequence[0].ImagePositionPatient ==
                pp[0].ImagePositionPatient
            )
        self.check_dimension_index_vals(instance)

    def test_construction_volume_labelmap(self):
        # Segmentation instance from a series of single-frame CT images
        # with empty frames kept in
        instance = Segmentation(
            [self._ct_image],
            self._ct_seg_volume,
            SegmentationTypeValues.LABELMAP,
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
            omit_empty_frames=False
        )
        assert np.array_equal(
            instance.pixel_array,
            self._ct_seg_volume.array,
        )

        assert instance.DimensionOrganizationType == '3D'
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == self._ct_volume_pixel_spacing
        assert pm_item.SliceThickness == self._ct_volume_slice_spacing
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            self._ct_volume_orientation
        for plane_item, pp in zip(
            instance.PerFrameFunctionalGroupsSequence,
            self._ct_seg_volume.get_plane_positions(),
        ):
            assert (
                plane_item.PlanePositionSequence[0].ImagePositionPatient ==
                pp[0].ImagePositionPatient
            )
        self.check_dimension_index_vals(instance)

    def test_construction_volume_labelmap_channels(self):
        # Segmentation instance from a series of single-frame CT images
        # with empty frames kept in, as a volume with channels
        instance = Segmentation(
            [self._ct_image],
            self._ct_seg_volume_with_channels,
            SegmentationTypeValues.LABELMAP,
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
            omit_empty_frames=False
        )
        assert np.array_equal(
            instance.pixel_array,
            self._ct_seg_volume.array,
        )

        assert instance.DimensionOrganizationType == '3D'
        shared_item = instance.SharedFunctionalGroupsSequence[0]
        assert len(shared_item.PixelMeasuresSequence) == 1
        pm_item = shared_item.PixelMeasuresSequence[0]
        assert pm_item.PixelSpacing == self._ct_volume_pixel_spacing
        assert pm_item.SliceThickness == self._ct_volume_slice_spacing
        assert len(shared_item.PlaneOrientationSequence) == 1
        po_item = shared_item.PlaneOrientationSequence[0]
        assert po_item.ImageOrientationPatient == \
            self._ct_volume_orientation
        for plane_item, pp in zip(
            instance.PerFrameFunctionalGroupsSequence,
            self._ct_seg_volume.get_plane_positions(),
        ):
            assert (
                plane_item.PlanePositionSequence[0].ImagePositionPatient ==
                pp[0].ImagePositionPatient
            )
        self.check_dimension_index_vals(instance)

    def test_construction_3d_multiframe(self):
        # The CT multiframe image is already a volume, but the frames are
        # ordered the wrong way
        volume_multiframe = deepcopy(self._ct_multiframe)
        positions = [
            fm.PlanePositionSequence[0].ImagePositionPatient
            for fm in volume_multiframe.PerFrameFunctionalGroupsSequence
        ]
        positions = positions[::-1]
        for pos, fm in zip(
            positions,
            volume_multiframe.PerFrameFunctionalGroupsSequence
        ):
            fm.PlanePositionSequence[0].ImagePositionPatient = pos

        # Segmentation instance from an enhanced (multi-frame) CT image
        instance = Segmentation(
            [volume_multiframe],
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
        # This is a "volume" image, so the output instance should have
        # the DimensionOrganizationType set correctly and should have deduced
        # the spacing between slices
        assert instance.DimensionOrganizationType == "3D"
        spacing = (
            instance
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .SpacingBetweenSlices
        )
        assert spacing == 10.0
        self.check_dimension_index_vals(instance)

    def test_construction_3d_singleframe(self):
        # The CT single frame series is a volume if you omit one of the images
        ct_series = self._ct_series[:3]

        # Segmentation instance series of CT images
        instance = Segmentation(
            ct_series,
            self._ct_series_mask_array[:3],
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
        )
        # This is a "volume" image, so the output instance should have
        # the DimensionOrganizationType set correctly and should have deduced
        # the spacing between slices
        assert instance.DimensionOrganizationType == "3D"
        spacing = (
            instance
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .SpacingBetweenSlices
        )
        assert spacing == 1.25
        self.check_dimension_index_vals(instance)

    def test_construction_3d_singleframe_multisegment(self):
        # The CT single frame series is a volume, but with multiple segments,
        # it should no longer have "3D" dimension organization unless using
        # LABELMAP
        ct_series = self._ct_series[:3]

        mask = self._ct_series_mask_array[:3]
        multi_segment_exc = np.stack([mask, np.logical_not(mask)], axis=-1)

        for segmentation_type in ["BINARY", "FRACTIONAL", "LABELMAP"]:
            # Segmentation instance series of CT images
            instance = Segmentation(
                ct_series,
                multi_segment_exc,
                segmentation_type,
                self._both_segment_descriptions,
                self._series_instance_uid,
                self._series_number,
                self._sop_instance_uid,
                self._instance_number,
                self._manufacturer,
                self._manufacturer_model_name,
                self._software_versions,
                self._device_serial_number,
            )
            # This is a "volume" image, so the output instance should have the
            # DimensionOrganizationType set correctly and should have deduced
            # the spacing between slices
            spacing = (
                instance
                .SharedFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .SpacingBetweenSlices
            )
            assert spacing == 1.25
            if segmentation_type == "LABELMAP":
                # Since there is no segment dimension, a labelmap seg is 3D
                assert instance.DimensionOrganizationType == "3D"
            else:
                # The segment dimension means that otherwise the segmentation
                # is not a simple spatial stack
                assert 'DimensionOrganizationType' not in instance
            self.check_dimension_index_vals(instance)

    def test_construction_workers(self):
        # Create a segmentation with multiple workers
        Segmentation(
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
            self._device_serial_number,
            content_label=self._content_label,
            transfer_syntax_uid=RLELossless,
            workers=2,
        )

    def test_construction_workers_manual(self):
        # Create a segmentation with multiple workers created manually
        with ProcessPoolExecutor(2) as pool:
            Segmentation(
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
                self._device_serial_number,
                content_label=self._content_label,
                transfer_syntax_uid=RLELossless,
                workers=pool,
            )

    def test_construction_tiled_full(self):
        instance = Segmentation(
            [self._sm_image],
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
            dimension_organization_type="TILED_FULL",
            omit_empty_frames=False,
        )
        assert instance.DimensionOrganizationType == "TILED_FULL"
        assert not hasattr(instance, "PerFrameFunctionalGroupsSequence")

    def test_construction_tiled_full_labelmap(self):
        instance = Segmentation(
            [self._sm_image],
            pixel_array=self._sm_pixel_array,
            segmentation_type=SegmentationTypeValues.LABELMAP,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number,
            dimension_organization_type="TILED_FULL",
            omit_empty_frames=False,
        )
        assert instance.DimensionOrganizationType == "TILED_FULL"
        assert not hasattr(instance, "PerFrameFunctionalGroupsSequence")

    def test_construction_further_source_images(self):
        further_source_image = deepcopy(self._ct_image)
        series_uid = UID()
        sop_uid = UID()
        further_source_image.SeriesInstanceUID = series_uid
        further_source_image.SOPInstanceUID = sop_uid
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
            self._device_serial_number,
            content_label=self._content_label,
            further_source_images=[further_source_image],
        )
        assert len(instance.SourceImageSequence) == 2
        further_item = instance.SourceImageSequence[1]
        assert further_item.ReferencedSOPInstanceUID == sop_uid

        assert len(instance.ReferencedSeriesSequence) == 2
        further_item = instance.ReferencedSeriesSequence[1]
        assert further_item.SeriesInstanceUID == series_uid
        self.check_dimension_index_vals(instance)

    @staticmethod
    @pytest.fixture(
        params=[
            DimensionOrganizationTypeValues.TILED_FULL,
            DimensionOrganizationTypeValues.TILED_SPARSE,
        ])
    def dimension_organization_type(request):
        return request.param

    @staticmethod
    @pytest.fixture(
        params=[
            SegmentationTypeValues.FRACTIONAL,
            SegmentationTypeValues.BINARY,
            SegmentationTypeValues.LABELMAP,
        ])
    def segmentation_type(request):
        return request.param

    @staticmethod
    @pytest.fixture(
        params=[
            None,
            (32, 32),
            (48, 48),
        ])
    def tile_size(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=[False, True])
    def locations_preserved(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=[1, 2])
    def num_segments(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=[False, True])
    def as_volume(request):
        return request.param

    def test_construction_autotile(
        self,
        tile_size,
        dimension_organization_type,
        segmentation_type,
        locations_preserved,
        num_segments,
        as_volume,
    ):
        if num_segments == 1:
            pixel_array = self._sm_total_pixel_array
            segment_descriptions = self._segment_descriptions
        else:
            pixel_array = self._sm_total_pixel_array_multiclass
            segment_descriptions = self._both_segment_descriptions

        if as_volume:
            affine = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            input_volume = Volume(
                affine=affine,
                array=pixel_array[None],
                coordinate_system="SLIDE",
            )

        if locations_preserved or as_volume:
            pixel_measures = None
            plane_orientation = None
            plane_positions = None
        else:
            pixel_measures = PixelMeasuresSequence(
               pixel_spacing=(0.0001, 0.0001),
               slice_thickness=0.001,
            )
            plane_orientation = PlaneOrientationSequence(
                coordinate_system='SLIDE',
                image_orientation=[0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
            )
            plane_positions = [
                PlanePositionSequence(
                    coordinate_system='SLIDE',
                    image_position=[1.1234, -5.4323214, 0.0],
                    pixel_matrix_position=(1, 1),
                )
            ]

        if dimension_organization_type.value == 'TILED_FULL':
            # Cannot omit empty frames with TILED_FULL
            omit_empty_frames_values = [False]
        else:
            omit_empty_frames_values = [False, True]

        transfer_syntax_uids = [
            ExplicitVRLittleEndian,
        ]
        try:
            import openjpeg  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            transfer_syntax_uids += [
                JPEG2000Lossless,
            ]
        if segmentation_type.value == 'FRACTIONAL':
            try:
                import libjpeg  # noqa: F401
            except ModuleNotFoundError:
                pass
            else:
                transfer_syntax_uids += [
                    JPEGLSLossless,
                ]

        for omit_empty_frames, transfer_syntax_uid in itertools.product(
            omit_empty_frames_values,
            transfer_syntax_uids,
        ):
            instance = Segmentation(
                [self._sm_image],
                pixel_array=input_volume if as_volume else pixel_array,
                segmentation_type=segmentation_type,
                segment_descriptions=segment_descriptions,
                series_instance_uid=self._series_instance_uid,
                series_number=self._series_number,
                sop_instance_uid=self._sop_instance_uid,
                instance_number=self._instance_number,
                manufacturer=self._manufacturer,
                manufacturer_model_name=self._manufacturer_model_name,
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number,
                dimension_organization_type=dimension_organization_type,
                omit_empty_frames=omit_empty_frames,
                plane_orientation=plane_orientation,
                plane_positions=plane_positions,
                pixel_measures=pixel_measures,
                tile_pixel_array=True,
                tile_size=tile_size,
                max_fractional_value=1,
                transfer_syntax_uid=transfer_syntax_uid,
            )
            assert (
                instance.DimensionOrganizationType ==
                dimension_organization_type.value
            )
            if tile_size is not None:
                assert instance.Rows == tile_size[0]
                assert instance.Columns == tile_size[1]

            # pydicom raises warnings if it has to pad or truncate pixel data
            with warnings.catch_warnings(record=True) as w:
                self.get_array_after_writing(instance)
                assert len(w) == 0

            # Check that full reconstructed array matches the input
            reconstructed_array = instance.get_total_pixel_matrix(
                combine_segments=True,
            )
            assert reconstructed_array.shape == (
                self._sm_image.TotalPixelMatrixRows,
                self._sm_image.TotalPixelMatrixColumns
            )
            assert np.array_equal(
                reconstructed_array,
                pixel_array,
            )
            if dimension_organization_type.value != "TILED_FULL":
                self.check_dimension_index_vals(instance)

            volume = instance.get_volume(
                combine_segments=True,
            )
            assert volume.shape == (
                1,
                self._sm_image.TotalPixelMatrixRows,
                self._sm_image.TotalPixelMatrixColumns
            )
            assert np.array_equal(
                volume.array,
                pixel_array[None],
            )

            def to_numpy(c):
                # Move from our 1-based convention to numpy zero based
                if c is None:
                    # None is handled the same
                    return None
                elif c > 0:
                    # Positive indices are 1-based
                    return c - 1
                else:
                    # Negative indices are the same
                    return c

            # Check that subregions defined in different ways match the input
            for rs, re, cs, ce in [
                (34, 48, 3, None),
                (-13, None, -34, -23),
            ]:
                reconstructed_array = instance.get_total_pixel_matrix(
                    combine_segments=True,
                    row_start=rs,
                    row_end=re,
                    column_start=cs,
                    column_end=ce,
                )

                rs_np = to_numpy(rs)
                re_np = to_numpy(re)
                cs_np = to_numpy(cs)
                ce_np = to_numpy(ce)
                expected_array = pixel_array[
                    (slice(rs_np, re_np), slice(cs_np, ce_np))
                ]
                assert np.array_equal(
                    reconstructed_array,
                    expected_array,
                )

                volume = instance.get_volume(
                    combine_segments=True,
                    row_start=rs,
                    row_end=re,
                    column_start=cs,
                    column_end=ce,
                )

                rs_np = to_numpy(rs)
                re_np = to_numpy(re)
                cs_np = to_numpy(cs)
                ce_np = to_numpy(ce)
                expected_array = pixel_array[
                    (slice(rs_np, re_np), slice(cs_np, ce_np))
                ][None]
                assert np.array_equal(
                    volume.array,
                    expected_array,
                )

    def test_pixel_types_fractional(
        self,
        fractional_transfer_syntax_uid,
        pix_type,
        test_data,
    ):
        if fractional_transfer_syntax_uid == JPEG2000Lossless:
            pytest.importorskip("openjpeg")
        if fractional_transfer_syntax_uid == JPEGLSLossless:
            pytest.importorskip("libjpeg")

        sources, mask = self._tests[test_data]

        # Two segments, overlapping
        multi_segment_overlap = np.stack([mask, mask], axis=-1)
        if multi_segment_overlap.ndim == 3:
            multi_segment_overlap = multi_segment_overlap[np.newaxis, ...]

        # Two segments non-overlapping
        multi_segment_exc = np.stack([mask, 1 - mask], axis=-1)
        if multi_segment_exc.ndim == 3:
            multi_segment_exc = multi_segment_exc[np.newaxis, ...]
        additional_mask = 1 - mask

        # Find the expected encodings for the masks
        if mask.ndim > 2:
            # Expected encoding of the mask
            expected_encoding = self.sort_frames(
                sources,
                mask
            )
            expected_encoding = self.remove_empty_frames(
                expected_encoding
            )

            # Expected encoding of the complement
            expected_encoding_comp = self.sort_frames(
                sources,
                additional_mask
            )
            expected_encoding_comp = self.remove_empty_frames(
                expected_encoding_comp
            )

            # Expected encoding of the multi segment arrays
            expected_enc_overlap = np.concatenate(
                [expected_encoding, expected_encoding],
                axis=0
            )
            expected_enc_exc = np.concatenate(
                [expected_encoding, expected_encoding_comp],
                axis=0
            )
            expected_encoding = expected_encoding.squeeze()
        else:
            expected_encoding = mask

            # Expected encoding of the multi segment arrays
            expected_enc_overlap = np.stack(
                [expected_encoding, expected_encoding],
                axis=0
            )
            expected_enc_exc = np.stack(
                [expected_encoding, 1 - expected_encoding],
                axis=0
            )

        max_fractional_value = 255
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
            max_fractional_value=max_fractional_value,
            transfer_syntax_uid=fractional_transfer_syntax_uid
        )

        # Ensure the recovered pixel array matches what is expected
        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_encoding * max_fractional_value
        ), f'{sources[0].Modality} {fractional_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

        # Multi-segment (exclusive)
        instance = Segmentation(
            sources,
            multi_segment_exc.astype(pix_type),
            SegmentationTypeValues.FRACTIONAL.value,
            self._both_segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            max_fractional_value=max_fractional_value,
            transfer_syntax_uid=fractional_transfer_syntax_uid
        )
        if pix_type == np.float64:
            assert (
                instance.SegmentsOverlap ==
                SegmentsOverlapValues.UNDEFINED.value
            )
        else:
            assert (
                instance.SegmentsOverlap ==
                SegmentsOverlapValues.NO.value
            )

        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_enc_exc * max_fractional_value
        ), f'{sources[0].Modality} {fractional_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

        # Multi-segment (overlapping)
        instance = Segmentation(
            sources,
            multi_segment_overlap.astype(pix_type),
            SegmentationTypeValues.FRACTIONAL.value,
            self._both_segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            max_fractional_value=max_fractional_value,
            transfer_syntax_uid=fractional_transfer_syntax_uid
        )
        if pix_type == np.float64:
            assert (
                instance.SegmentsOverlap ==
                SegmentsOverlapValues.UNDEFINED.value
            )
        else:
            assert (
                instance.SegmentsOverlap ==
                SegmentsOverlapValues.YES.value
            )

        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_enc_overlap * max_fractional_value
        ), f'{sources[0].Modality} {fractional_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

    def test_pixel_types_binary(
        self,
        binary_transfer_syntax_uid,
        pix_type,
        test_data,
    ):
        if binary_transfer_syntax_uid == JPEG2000Lossless:
            pytest.importorskip("openjpeg")
        sources, mask = self._tests[test_data]

        # Two segments, overlapping
        multi_segment_overlap = np.stack([mask, mask], axis=-1)
        if multi_segment_overlap.ndim == 3:
            multi_segment_overlap = multi_segment_overlap[np.newaxis, ...]

        # Two segments non-overlapping
        multi_segment_exc = np.stack([mask, 1 - mask], axis=-1)

        if multi_segment_exc.ndim == 3:
            multi_segment_exc = multi_segment_exc[np.newaxis, ...]
        additional_mask = 1 - mask

        additional_mask = (1 - mask)
        # Find the expected encodings for the masks
        if mask.ndim > 2:
            # Expected encoding of the mask
            expected_encoding = self.sort_frames(
                sources,
                mask
            )
            expected_encoding = self.remove_empty_frames(
                expected_encoding
            )

            # Expected encoding of the complement
            expected_encoding_comp = self.sort_frames(
                sources,
                additional_mask
            )
            expected_encoding_comp = self.remove_empty_frames(
                expected_encoding_comp
            )

            # Expected encoding of the multi segment arrays
            expected_enc_overlap = np.concatenate(
                [expected_encoding, expected_encoding],
                axis=0
            )
            expected_enc_exc = np.concatenate(
                [expected_encoding, expected_encoding_comp],
                axis=0
            )
            expected_encoding = expected_encoding.squeeze()
        else:
            expected_encoding = mask

            # Expected encoding of the multi segment arrays
            expected_enc_overlap = np.stack(
                [expected_encoding, expected_encoding],
                axis=0
            )
            expected_enc_exc = np.stack(
                [expected_encoding, 1 - expected_encoding],
                axis=0
            )

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
            transfer_syntax_uid=binary_transfer_syntax_uid
        )

        # Ensure the recovered pixel array matches what is expected
        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_encoding
        ), f'{sources[0].Modality} {binary_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

        # Multi-segment (exclusive)
        instance = Segmentation(
            sources,
            multi_segment_exc.astype(pix_type),
            SegmentationTypeValues.BINARY.value,
            self._both_segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            max_fractional_value=1,
            transfer_syntax_uid=binary_transfer_syntax_uid
        )
        assert (
            instance.SegmentsOverlap ==
            SegmentsOverlapValues.NO.value
        )

        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_enc_exc
        ), f'{sources[0].Modality} {binary_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

        # Multi-segment (overlapping)
        instance = Segmentation(
            sources,
            multi_segment_overlap.astype(pix_type),
            SegmentationTypeValues.BINARY.value,
            self._both_segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            max_fractional_value=1,
            transfer_syntax_uid=binary_transfer_syntax_uid
        )
        assert (
            instance.SegmentsOverlap ==
            SegmentsOverlapValues.YES.value
        )

        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_enc_overlap
        ), f'{sources[0].Modality} {binary_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

    def test_pixel_types_labelmap(
        self,
        fractional_transfer_syntax_uid,
        pix_type,
        test_data,
    ):
        if fractional_transfer_syntax_uid == JPEG2000Lossless:
            pytest.importorskip("openjpeg")
        if fractional_transfer_syntax_uid == JPEGLSLossless:
            pytest.importorskip("libjpeg")
        sources, mask = self._tests[test_data]

        # Two segments non-overlapping
        multi_segment_exc = np.stack([mask, 1 - mask], axis=-1)

        if multi_segment_exc.ndim == 3:
            multi_segment_exc = multi_segment_exc[np.newaxis, ...]

        # Find the expected encodings for the masks
        if mask.ndim > 2:
            sorted_mask = self.sort_frames(
                sources,
                mask
            )

            # Expected encoding of the mask
            expected_encoding = self.remove_empty_frames(
                sorted_mask
            ).astype(np.uint8)
        else:
            sorted_mask = mask
            expected_encoding = mask.astype(np.uint8)

        expected_enc_exc = np.zeros(sorted_mask.shape, np.uint8)
        expected_enc_exc[sorted_mask > 0] = 1
        expected_enc_exc[sorted_mask == 0] = 2
        expected_encoding = expected_encoding.squeeze()

        instance = Segmentation(
            sources,
            mask.astype(pix_type),
            SegmentationTypeValues.LABELMAP,
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
            transfer_syntax_uid=fractional_transfer_syntax_uid,
        )

        # Ensure the recovered pixel array matches what is expected
        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_encoding
        ), f'{sources[0].Modality} {fractional_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

        # Multi-segment (exclusive)
        instance = Segmentation(
            sources,
            multi_segment_exc.astype(pix_type),
            SegmentationTypeValues.LABELMAP,
            self._both_segment_descriptions,
            self._series_instance_uid,
            self._series_number,
            self._sop_instance_uid,
            self._instance_number,
            self._manufacturer,
            self._manufacturer_model_name,
            self._software_versions,
            self._device_serial_number,
            max_fractional_value=1,
            transfer_syntax_uid=fractional_transfer_syntax_uid,
        )
        assert (
            instance.SegmentsOverlap ==
            SegmentsOverlapValues.NO.value
        )

        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_enc_exc
        ), f'{sources[0].Modality} {fractional_transfer_syntax_uid}'
        self.check_dimension_index_vals(instance)

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
        self.check_dimension_index_vals(instance)

        additional_odd_mask = np.random.randint(
            2,
            size=odd_pixels.shape,
            dtype=bool
        )
        two_segment_mask = np.stack(
            [odd_mask, additional_odd_mask],
            axis=-1
        )[np.newaxis, ...]
        expected_encoding = np.stack(
            [odd_mask, additional_odd_mask],
            axis=0
        )

        instance = Segmentation(
            [odd_instance],
            two_segment_mask,
            SegmentationTypeValues.BINARY.value,
            segment_descriptions=self._both_segment_descriptions,
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number
        )

        assert np.array_equal(
            self.get_array_after_writing(instance),
            expected_encoding
        )
        self.check_dimension_index_vals(instance)

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
            self.check_dimension_index_vals(instance)

    def test_construction_empty_source_image(self):
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[],  # empty
                pixel_array=self._ct_pixel_array,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=(
                    self._segment_descriptions
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

    def test_construction_empty_source_seg_sparse(self):
        # Encoding an empty segmentation with omit_empty_frames=True issues
        # a warning and encodes the full segmentation
        empty_pixel_array = np.zeros_like(self._ct_pixel_array)
        instance = Segmentation(
            source_images=[self._ct_image],
            pixel_array=empty_pixel_array,
            segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
            segment_descriptions=(
                self._segment_descriptions
            ),
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number,
            omit_empty_frames=True,
        )

        assert instance.pixel_array.shape == empty_pixel_array.shape
        self.check_dimension_index_vals(instance)

    def test_construction_empty_seg_image(self):
        # Can encode an empty segmentation with omit_empty_frames=False
        empty_pixel_array = np.zeros_like(self._ct_pixel_array)
        Segmentation(
            source_images=[self._ct_image],
            pixel_array=empty_pixel_array,
            segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
            segment_descriptions=(
                self._segment_descriptions
            ),
            series_instance_uid=self._series_instance_uid,
            series_number=self._series_number,
            sop_instance_uid=self._sop_instance_uid,
            instance_number=self._instance_number,
            manufacturer=self._manufacturer,
            manufacturer_model_name=self._manufacturer_model_name,
            software_versions=self._software_versions,
            device_serial_number=self._device_serial_number,
            omit_empty_frames=False,
        )

    def test_construction_invalid_content_label(self):
        with pytest.raises(ValueError):
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
                software_versions=self._software_versions,
                device_serial_number=self._device_serial_number,
                content_label='invalid-content-label'
            )

    def test_construction_mixed_source_series(self):
        with pytest.raises(ValueError):
            Segmentation(
                source_images=self._ct_series + [self._ct_image],
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

    def test_construction_nonunqiue_plane_positions(self):
        # It should not be possible to construct a segmentation with input
        # images with the same plane location, even if they are otherwise
        # distinct
        ct_image_2 = deepcopy(self._ct_image)
        ct_image_2.SOPInstanceUID = UID()
        ct_image_2.InstanceNumber = 2
        pixel_array = np.zeros(
            (2, *self._ct_image.pixel_array.shape),
            dtype=bool
        )
        pixel_array[0, 1:5, 10:15] = True
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image, ct_image_2],
                pixel_array=pixel_array,
                segmentation_type=SegmentationTypeValues.BINARY,
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

    def test_construction_wrong_number_of_segments(self):
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array[..., np.newaxis],
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=(
                    self._both_segment_descriptions
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

    def test_construction_stacked_label_map(self):
        # A 4D integer cannot have non-binary values
        mask = np.zeros(
            (1, self._ct_image.Rows, self._ct_image.Columns, 2),
            dtype=np.uint8
        )
        mask[0, 0, 0, 0] = 2  # disallowed
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=mask,
                segmentation_type=SegmentationTypeValues.BINARY.value,
                segment_descriptions=(
                    self._both_segment_descriptions
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

    def test_construction_segment_numbers_start_wrong(self):
        msg = (
            'Segment descriptions should be numbered starting from 1. '
            'Found 2.'
        )
        with pytest.raises(ValueError, match=msg):
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

    def test_construction_segment_numbers_start_wrong_labelmap(self):
        # Labelmaps have fewer restrictions on segment numbers
        array = (self._ct_pixel_array * 2).astype(np.uint8)
        instance = Segmentation(
            source_images=[self._ct_image],
            pixel_array=array,
            segmentation_type=SegmentationTypeValues.LABELMAP,
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
        assert len(instance.SegmentSequence) == 2
        self.check_dimension_index_vals(instance)

        array_nonmatching = (self._ct_pixel_array * 4).astype(np.uint8)
        msg = 'Pixel array contains segments that lack descriptions.'
        with pytest.raises(ValueError, match=msg):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=array_nonmatching,
                segmentation_type=SegmentationTypeValues.LABELMAP,
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

    def test_construction_empty_invalid_floats(self):
        # Floats outside the range 0.0 to 1.0 are invalid
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],  # empty
                pixel_array=self._ct_pixel_array.astype(np.float64) * 2,
                segmentation_type=SegmentationTypeValues.FRACTIONAL.value,
                segment_descriptions=(
                    self._segment_descriptions
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

    def test_construction_empty_invalid_floats_binary(self):
        # Cannot use floats other than 0.0 and 1.0 when encoding as BINARY
        with pytest.raises(ValueError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array.astype(np.float64) * 0.5,
                segmentation_type=SegmentationTypeValues.BINARY.value,
                segment_descriptions=(
                    self._segment_descriptions
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

    def test_construction_empty_invalid_dtype(self):
        # Cannot use signed integers
        with pytest.raises(TypeError):
            Segmentation(
                source_images=[self._ct_image],
                pixel_array=self._ct_pixel_array.astype(np.int16),
                segmentation_type=SegmentationTypeValues.BINARY.value,
                segment_descriptions=(
                    self._segment_descriptions
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
        msg = 'Pixel array contains segments that lack descriptions.'
        with pytest.raises(ValueError, match=msg):
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

    def test_construction_multislice_no_frame_of_reference(self):
        # A chest X-ray with no frame of reference -> cannot have multiple
        # images
        multislice_seg = np.tile(self._cr_pixel_array, (2, 1, 1))
        with pytest.raises(ValueError):
            Segmentation(
                [self._cr_image] * 2,
                multislice_seg,
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
                content_label=self._content_label,
            )

    def test_construction_plane_positions_no_frame_of_reference(self):
        # A chest X-ray with no frame of reference -> cannot have plane
        # positions
        plane_positions = [
            PlanePositionSequence(
                coordinate_system=CoordinateSystemNames.PATIENT,
                image_position=(0.0, 0.0, 0.0)
            ),
        ]
        with pytest.raises(TypeError):
            Segmentation(
                [self._cr_image],
                self._cr_pixel_array,
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
                content_label=self._content_label,
                plane_positions=plane_positions
            )

    def test_construction_plane_orientation_no_frame_of_reference(self):
        # A chest X-ray with no frame of reference -> cannot have plane
        # orientation
        image_orientation = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        plane_orientation = PlaneOrientationSequence(
            coordinate_system=CoordinateSystemNames.PATIENT,
            image_orientation=image_orientation
        )
        with pytest.raises(TypeError):
            Segmentation(
                [self._cr_image],
                self._cr_pixel_array,
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
                content_label=self._content_label,
                plane_orientation=plane_orientation
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
        content_creator_name = 'Family^Given'
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
        self.check_dimension_index_vals(instance)

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
        self.check_dimension_index_vals(instance)

    def test_spatial_positions_not_preserved(self):
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
        assert not hasattr(shared_item, 'PlaneOrientationSequence')
        self.check_dimension_index_vals(instance)

    def test_get_plane_positions_of_image_patient(self):
        seq = DimensionIndexSequence(
            coordinate_system='PATIENT'
        )
        plane_positions = seq.get_plane_positions_of_image(self._ct_multiframe)
        for position in plane_positions:
            assert isinstance(position, PlanePositionSequence)

    def test_get_plane_positions_of_image_slide(self):
        seq = DimensionIndexSequence(
            coordinate_system='SLIDE'
        )
        plane_positions = seq.get_plane_positions_of_image(self._sm_image)
        for position in plane_positions:
            assert isinstance(position, PlanePositionSequence)

    def test_get_plane_positions_of_series(self):
        seq = DimensionIndexSequence(
            coordinate_system='PATIENT'
        )
        plane_positions = seq.get_plane_positions_of_series(self._ct_series)
        for position in plane_positions:
            assert isinstance(position, PlanePositionSequence)


class TestSegmentationParsing:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self._sm_control_seg_ds = dcmread(
            'data/test_files/seg_image_sm_control.dcm'
        )
        self._sm_control_seg = Segmentation.from_dataset(
            self._sm_control_seg_ds
        )

        self._sm_control_labelmap_seg_ds = dcmread(
            'data/test_files/seg_image_sm_control_labelmap.dcm'
        )
        self._sm_control_labelmap_seg = Segmentation.from_dataset(
            self._sm_control_labelmap_seg_ds
        )

        self._sm_control_labelmap_palette_color_seg_ds = dcmread(
            'data/test_files/seg_image_sm_control_labelmap_palette_color.dcm'
        )
        self._sm_control_labelmap_palette_color_seg = Segmentation.from_dataset(
            self._sm_control_labelmap_palette_color_seg_ds
        )

        self._ct_binary_seg_ds = dcmread(
            'data/test_files/seg_image_ct_binary.dcm'
        )
        self._ct_binary_seg = Segmentation.from_dataset(
            self._ct_binary_seg_ds
        )

        self._ct_binary_overlap_seg_ds = dcmread(
            'data/test_files/seg_image_ct_binary_overlap.dcm'
        )
        self._ct_binary_overlap_seg = Segmentation.from_dataset(
            self._ct_binary_overlap_seg_ds
        )

        self._ct_binary_fractional_seg_ds = dcmread(
            'data/test_files/seg_image_ct_binary_fractional.dcm'
        )
        self._ct_binary_fractional_seg = Segmentation.from_dataset(
            self._ct_binary_fractional_seg_ds
        )

        self._ct_true_fractional_seg_ds = dcmread(
            'data/test_files/seg_image_ct_true_fractional.dcm'
        )
        self._ct_true_fractional_seg = Segmentation.from_dataset(
            self._ct_true_fractional_seg_ds
        )
        self._ct_segs = [
            self._ct_binary_seg,
            self._ct_binary_fractional_seg,
            self._ct_true_fractional_seg
        ]

        self._cr_binary_seg_ds = dcmread(
            'data/test_files/seg_image_cr_binary.dcm'
        )
        self._cr_binary_seg = Segmentation.from_dataset(
            self._cr_binary_seg_ds
        )

    # Fixtures to use to parametrize segmentation creation
    # Using this fixture mechanism, we can parametrize class methods
    @staticmethod
    @pytest.fixture(
        params=[
            bool,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float32,
            np.float64,
            np.dtype("uint16"),
            "uint16",
        ])
    def numpy_dtype(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=[False, True])
    def combine_segments(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=[False, True])
    def relabel(request):
        return request.param

    @staticmethod
    @pytest.fixture(params=['_sm_control_seg', '_sm_control_labelmap_seg'])
    def seg_attr_name(request):
        return request.param

    def test_from_dataset(self):
        assert isinstance(self._sm_control_seg, Segmentation)

    def test_from_dataset_copy_false(self):
        seg_ds = dcmread(
            'data/test_files/seg_image_sm_control.dcm'
        )
        seg = Segmentation.from_dataset(seg_ds, copy=False)
        assert seg is seg_ds

    def test_segread(self):
        seg = segread('data/test_files/seg_image_ct_true_fractional.dcm')
        assert isinstance(seg, Segmentation)
        seg = segread('data/test_files/seg_image_ct_binary_overlap.dcm')
        assert isinstance(seg, Segmentation)
        seg = segread('data/test_files/seg_image_sm_numbers.dcm')
        assert isinstance(seg, Segmentation)
        seg = segread('data/test_files/seg_image_sm_dots_tiled_full.dcm')
        assert isinstance(seg, Segmentation)
        seg = segread('data/test_files/seg_image_sm_control_labelmap.dcm')
        assert isinstance(seg, Segmentation)
        seg = segread(
            'data/test_files/seg_image_sm_control_labelmap_palette_color.dcm'
        )
        assert isinstance(seg, Segmentation)

    def test_segread_from_bytes(self):
        # Read directly from bytes
        seg = dcmread('data/test_files/seg_image_ct_binary_overlap.dcm')

        with BytesIO() as buf:
            seg.save_as(buf)
            reread_seg = segread(buf.getvalue())

        assert isinstance(reread_seg, Segmentation)

    def test_properties(self):
        # SM segs
        seg_type = self._sm_control_seg.segmentation_type
        assert seg_type == SegmentationTypeValues.BINARY
        assert self._sm_control_seg.segmentation_fractional_type is None
        assert self._sm_control_seg.number_of_segments == 20
        assert self._sm_control_seg.segment_numbers == list(range(1, 21))

        assert len(self._sm_control_seg.segmented_property_categories) == 1
        seg_category = self._sm_control_seg.segmented_property_categories[0]
        assert seg_category == codes.SCT.Tissue
        seg_property = self._sm_control_seg.segmented_property_types[0]
        assert seg_property == codes.SCT.ConnectiveTissue

        # CT segs
        for seg in self._ct_segs:
            seg_type = seg.segmentation_type
            assert seg.number_of_segments == 1
            assert seg.segment_numbers == list(range(1, 2))

            assert len(seg.segmented_property_categories) == 1
            seg_category = seg.segmented_property_categories[0]
            assert seg_category == codes.SCT.Tissue
            seg_property = seg.segmented_property_types[0]
            assert seg_property == codes.SCT.Bone

        seg_type = self._ct_binary_seg.segmentation_type
        assert seg_type == SegmentationTypeValues.BINARY
        seg_type = self._ct_binary_fractional_seg.segmentation_type
        assert seg_type == SegmentationTypeValues.FRACTIONAL
        seg_type = self._ct_true_fractional_seg.segmentation_type
        assert seg_type == SegmentationTypeValues.FRACTIONAL

        frac_type = self._ct_binary_fractional_seg.segmentation_fractional_type
        assert frac_type == SegmentationFractionalTypeValues.PROBABILITY
        frac_type = self._ct_true_fractional_seg.segmentation_fractional_type
        assert frac_type == SegmentationFractionalTypeValues.PROBABILITY

    def test_get_source_image_uids(self):
        uids = self._sm_control_seg.get_source_image_uids()
        assert len(uids) == 1
        ins_uids = uids[0]
        assert len(ins_uids) == 3
        assert all(isinstance(uid, UID) for uid in ins_uids)

    def test_get_segment_description(self):
        desc1 = self._sm_control_seg.get_segment_description(1)
        desc20 = self._sm_control_seg.get_segment_description(20)
        assert isinstance(desc1, SegmentDescription)
        assert desc1.segment_number == 1
        assert isinstance(desc20, SegmentDescription)
        assert desc20.segment_number == 20

    def test_get_segment_description_non_consecutive(self):
        out_of_order = deepcopy(self._sm_control_seg)
        out_of_order.SegmentSequence = [
            out_of_order.SegmentSequence[i]
            for i in range(
                out_of_order.number_of_segments - 1,
                -1,
                -1
            )
        ]
        desc1 = out_of_order.get_segment_description(1)
        desc20 = out_of_order.get_segment_description(20)
        assert isinstance(desc1, SegmentDescription)
        assert desc1.segment_number == 1
        assert isinstance(desc20, SegmentDescription)
        assert desc20.segment_number == 20

    def test_get_segment_numbers_no_filters(self):
        seg_nums = self._sm_control_seg.get_segment_numbers()
        assert seg_nums == list(self._sm_control_seg.segment_numbers)

    def test_get_segment_numbers_with_filters(self):
        desc1 = self._sm_control_seg.get_segment_description(1)

        seg_nums = self._sm_control_seg.get_segment_numbers(
            tracking_id=desc1.tracking_id
        )
        assert seg_nums == [1]

        seg_nums = self._sm_control_seg.get_segment_numbers(
            tracking_uid=desc1.tracking_uid
        )
        assert seg_nums == [1]

        # All segments match these filters
        seg_nums = self._sm_control_seg.get_segment_numbers(
            segmented_property_category=codes.SCT.Tissue,
            segmented_property_type=codes.SCT.ConnectiveTissue,
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC
        )
        assert seg_nums == list(self._sm_control_seg.segment_numbers)

    def test_get_tracking_ids(self):
        desc1 = self._sm_control_seg.get_segment_description(1)

        tracking_id_tuples = self._sm_control_seg.get_tracking_ids()
        n_segs = self._sm_control_seg.number_of_segments
        assert len(tracking_id_tuples) == n_segs
        ids, uids = zip(*tracking_id_tuples)
        assert desc1.tracking_id in ids
        assert desc1.tracking_uid in uids

    def test_get_tracking_ids_with_filters(self):
        desc1 = self._sm_control_seg.get_segment_description(1)

        # All segments in this test image match these filters
        tracking_id_tuples = self._sm_control_seg.get_tracking_ids(
            segmented_property_category=codes.SCT.Tissue,
            segmented_property_type=codes.SCT.ConnectiveTissue,
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC
        )
        n_segs = self._sm_control_seg.number_of_segments
        assert len(tracking_id_tuples) == n_segs
        ids, uids = zip(*tracking_id_tuples)
        assert desc1.tracking_id in ids
        assert desc1.tracking_uid in uids

    def test_get_tracking_ids_with_filters_2(self):
        # No segments in this test image match these filters
        tracking_id_tuples = self._sm_control_seg.get_tracking_ids(
            segmented_property_category=codes.SCT.Tissue,
            segmented_property_type=codes.SCT.Lung,
        )
        assert len(tracking_id_tuples) == 0

    def test_get_pixels_by_source_frames(self):
        source_sop_uid = self._sm_control_seg.get_source_image_uids()[0][-1]

        source_frames_valid = [1, 2, 4, 5]
        pixels = self._sm_control_seg.get_pixels_by_source_frame(
            source_sop_instance_uid=source_sop_uid,
            source_frame_numbers=source_frames_valid
        )

        out_shape = (
            len(source_frames_valid),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns,
            self._sm_control_seg.number_of_segments
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_invalid_source_frames(self):
        source_sop_uid = self._sm_control_seg.get_source_image_uids()[0][-1]

        # (frame 3 has no segment but is below the max so shouldn't raise)
        source_frames_valid = [1, 3, 4, 5]
        self._sm_control_seg.get_pixels_by_source_frame(
            source_sop_instance_uid=source_sop_uid,
            source_frame_numbers=source_frames_valid
        )

        # (frame 100 has no segment but is greater than largest frame so
        # should raise)
        source_frames_invalid = [1, 3, 4, 5, 100]
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_source_frame(
                source_sop_instance_uid=source_sop_uid,
                source_frame_numbers=source_frames_invalid
            )

    def test_get_pixels_by_invalid_source_frames_with_assert(self):
        source_sop_uid = self._sm_control_seg.get_source_image_uids()[0][-1]

        # (frame 100 has no segment)
        source_frames_invalid = [1, 3, 4, 5, 100]
        pixels = self._sm_control_seg.get_pixels_by_source_frame(
            source_sop_instance_uid=source_sop_uid,
            source_frame_numbers=source_frames_invalid,
            assert_missing_frames_are_empty=True
        )

        out_shape = (
            len(source_frames_invalid),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns,
            self._sm_control_seg.number_of_segments
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_source_frames_with_segments(self):
        source_sop_uid = self._sm_control_seg.get_source_image_uids()[0][-1]

        source_frames_valid = [1, 2, 4, 5]
        segments_valid = [1, 20]
        pixels = self._sm_control_seg.get_pixels_by_source_frame(
            source_sop_instance_uid=source_sop_uid,
            source_frame_numbers=source_frames_valid,
            segment_numbers=segments_valid
        )

        out_shape = (
            len(source_frames_valid),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns,
            len(segments_valid)
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_source_frames_with_invalid_segments(self):
        source_sop_uid = self._sm_control_seg.get_source_image_uids()[0][-1]

        source_frames_valid = [1, 2, 4, 5]
        segments_invalid = [1, 21]  # 21 > 20
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_source_frame(
                source_sop_instance_uid=source_sop_uid,
                source_frame_numbers=source_frames_valid,
                segment_numbers=segments_invalid
            )

    def test_get_pixels_by_source_frames_combine(self):
        source_sop_uid = self._sm_control_seg.get_source_image_uids()[0][-1]

        source_frames_valid = [1, 2, 4, 5]
        # These segments match the above frames for this test image
        segments_valid = [6, 7, 8, 9]
        pixels = self._sm_control_seg.get_pixels_by_source_frame(
            source_sop_instance_uid=source_sop_uid,
            source_frame_numbers=source_frames_valid,
            segment_numbers=segments_valid,
            combine_segments=True
        )

        out_shape = (
            len(source_frames_valid),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns
        )
        assert pixels.shape == out_shape
        assert np.all(np.unique(pixels) == np.array([0] + segments_valid))

        pixels = self._sm_control_seg.get_pixels_by_source_frame(
            source_sop_instance_uid=source_sop_uid,
            source_frame_numbers=source_frames_valid,
            segment_numbers=segments_valid,
            combine_segments=True,
            relabel=True
        )
        assert pixels.shape == out_shape
        assert np.all(np.unique(pixels) == np.arange(len(segments_valid) + 1))

    def test_get_pixels_with_dtype(
        self,
        seg_attr_name,
        numpy_dtype,
        combine_segments,
        relabel,
     ):
        source_sop_uid = self._sm_control_seg.get_source_image_uids()[0][-1]

        source_frames_valid = [1, 2, 4, 5]
        seg = getattr(self, seg_attr_name)

        if numpy_dtype == bool and combine_segments:
            max_val = 3 if relabel else 9
            msg = (
                "The maximum output value of the segmentation array is "
                f"{max_val}, which is too large be represented using dtype "
                f"bool."
            )
            with pytest.raises(ValueError, match=msg):
                seg.get_pixels_by_source_frame(
                    source_sop_instance_uid=source_sop_uid,
                    source_frame_numbers=source_frames_valid,
                    segment_numbers=[1, 4, 9],
                    combine_segments=combine_segments,
                    relabel=relabel,
                    dtype=numpy_dtype,
                )
        else:
            pixels = seg.get_pixels_by_source_frame(
                source_sop_instance_uid=source_sop_uid,
                source_frame_numbers=source_frames_valid,
                segment_numbers=[1, 4, 9],
                combine_segments=combine_segments,
                relabel=relabel,
                dtype=numpy_dtype,
            )
            assert pixels.dtype == numpy_dtype
            if combine_segments:
                expected_shape = (
                    len(source_frames_valid), seg.Rows, seg.Columns
                )
                if relabel:
                    # only seg 9 in these frames
                    expected_vals = np.array([0, 3])
                else:
                    # only seg 9 in these frames
                    expected_vals = np.array([0, 9])
            else:
                expected_shape = (
                    len(source_frames_valid), seg.Rows, seg.Columns, 3
                )
                expected_vals = np.array([0, 1])
            assert pixels.shape == expected_shape
            assert np.array_equal(np.unique(pixels), expected_vals)

    def test_get_total_pixel_matrix_with_dtype(
        self,
        seg_attr_name,
        numpy_dtype,
        combine_segments,
        relabel,
     ):
        seg = getattr(self, seg_attr_name)
        subregion_rows = 30
        subregion_columns = 30

        if numpy_dtype == bool and combine_segments:
            max_val = 3 if relabel else 9
            msg = (
                "The maximum output value of the segmentation array is "
                f"{max_val}, which is too large be represented using dtype "
                f"bool."
            )
            with pytest.raises(ValueError, match=msg):
                seg.get_total_pixel_matrix(
                    segment_numbers=[1, 4, 9],
                    row_end=1 + subregion_rows,
                    column_end=1 + subregion_columns,
                    combine_segments=combine_segments,
                    relabel=relabel,
                    dtype=numpy_dtype,
                )
        else:
            pixels = seg.get_total_pixel_matrix(
                row_end=1 + subregion_rows,
                column_end=1 + subregion_columns,
                segment_numbers=[1, 4, 9],
                combine_segments=combine_segments,
                relabel=relabel,
                dtype=numpy_dtype,
            )
            assert pixels.dtype == numpy_dtype
            if combine_segments:
                expected_shape = (
                    subregion_rows, subregion_columns,
                )
                if relabel:
                    # only seg 9 in these frames
                    expected_vals = np.array([0, 3])
                else:
                    # only seg 9 in these frames
                    expected_vals = np.array([0, 9])
            else:
                expected_shape = (
                    subregion_rows, subregion_columns, 3
                )
                expected_vals = np.array([0, 1])
            assert pixels.shape == expected_shape
            assert np.array_equal(np.unique(pixels), expected_vals)

            volume = seg.get_volume(
                row_end=1 + subregion_rows,
                column_end=1 + subregion_columns,
                segment_numbers=[1, 4, 9],
                combine_segments=combine_segments,
                relabel=relabel,
                dtype=numpy_dtype,
            )
            assert volume.dtype == np.dtype(numpy_dtype)
            if combine_segments:
                expected_shape = (
                    1, subregion_rows, subregion_columns,
                )
                if relabel:
                    # only seg 9 in these frames
                    expected_vals = np.array([0, 3])
                else:
                    # only seg 9 in these frames
                    expected_vals = np.array([0, 9])
            else:
                expected_shape = (
                    1, subregion_rows, subregion_columns, 3
                )
                expected_vals = np.array([0, 1])
            assert volume.shape == expected_shape
            if not np.array_equal(np.unique(volume.array), expected_vals):
                print(seg_attr_name)
            assert np.array_equal(np.unique(volume.array), expected_vals)

    def test_get_default_dimension_index_pointers(self):
        ptrs = self._sm_control_seg.get_default_dimension_index_pointers()
        assert len(ptrs) == 5

    def test_are_dimension_indices_unique(self):
        ptrs = self._sm_control_seg.get_default_dimension_index_pointers()
        assert self._sm_control_seg.are_dimension_indices_unique(ptrs)

        ptr_kws = [
            'ColumnPositionInTotalImagePixelMatrix',
            'RowPositionInTotalImagePixelMatrix'
        ]
        ptrs = [tag_for_keyword(kw) for kw in ptr_kws]
        assert self._sm_control_seg.are_dimension_indices_unique(ptrs)

        ptr_kws = [
            'XOffsetInSlideCoordinateSystem',
            'YOffsetInSlideCoordinateSystem'
        ]
        ptrs = [tag_for_keyword(kw) for kw in ptr_kws]
        assert self._sm_control_seg.are_dimension_indices_unique(ptrs)

        ptr_kws = [
            'ZOffsetInSlideCoordinateSystem'
        ]
        ptrs = [tag_for_keyword(kw) for kw in ptr_kws]
        assert not self._sm_control_seg.are_dimension_indices_unique(ptrs)

    def test_are_dimension_indices_unique_invalid_ptrs(self):
        ptr_kws = [
            'ImagePositionPatient'
        ]
        ptrs = [tag_for_keyword(kw) for kw in ptr_kws]
        with pytest.raises(KeyError):
            self._sm_control_seg.are_dimension_indices_unique(ptrs)

    def test_get_pixels_by_dimension_index_values(self):
        ind_values = [
            (1, 1, 5, 5, 1),
            (2, 1, 4, 5, 1),
            (3, 1, 3, 5, 1)
        ]
        pixels = self._sm_control_seg.get_pixels_by_dimension_index_values(
            dimension_index_values=ind_values,
        )

        out_shape = (
            len(ind_values),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns,
            self._sm_control_seg.number_of_segments
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_dimension_index_values_subset(self):
        ptr_kws = [
            'ColumnPositionInTotalImagePixelMatrix',
            'RowPositionInTotalImagePixelMatrix'
        ]
        ptrs = [tag_for_keyword(kw) for kw in ptr_kws]

        ind_values = [
            (1, 1),
            (2, 1),
            (3, 1)
        ]
        pixels = self._sm_control_seg.get_pixels_by_dimension_index_values(
            dimension_index_values=ind_values,
            dimension_index_pointers=ptrs
        )

        out_shape = (
            len(ind_values),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns,
            self._sm_control_seg.number_of_segments
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_dimension_index_values_missing(self):
        ind_values = [
            (1, 1, 4, 5, 1),
        ]
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_dimension_index_values(
                dimension_index_values=ind_values,
            )

        pixels = self._sm_control_seg.get_pixels_by_dimension_index_values(
            dimension_index_values=ind_values,
            assert_missing_frames_are_empty=True
        )

        out_shape = (
            len(ind_values),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns,
            self._sm_control_seg.number_of_segments
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_dimension_index_values_with_segments(self):
        ind_values = [
            (1, 1, 5, 5, 1),
            (2, 1, 4, 5, 1),
            (3, 1, 3, 5, 1)
        ]
        segments = [1, 6, 11]
        pixels = self._sm_control_seg.get_pixels_by_dimension_index_values(
            dimension_index_values=ind_values,
            segment_numbers=segments
        )

        out_shape = (
            len(ind_values),
            self._sm_control_seg.Rows,
            self._sm_control_seg.Columns,
            len(segments)
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_dimension_index_values_invalid(self):
        ind_values = [
            (1, 1, 5, 5, 1),
            (2, 1, 4, 5, 1),
            (3, 1, 3, 5, 1)
        ]
        ptrs = [tag_for_keyword('ImagePositionPatient')]

        # Invalid pointers
        with pytest.raises(KeyError):
            self._sm_control_seg.get_pixels_by_dimension_index_values(
                dimension_index_values=ind_values,
                dimension_index_pointers=ptrs
            )
        # Invalid values
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_dimension_index_values(
                dimension_index_values=[(-1, 1, 1, 1, 1)],
            )
        # Empty values
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_dimension_index_values(
                dimension_index_values=[],
            )
        # Empty pointers
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_dimension_index_values(
                dimension_index_values=ind_values,
                dimension_index_pointers=[]
            )
        # Empty segment numbers
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_dimension_index_values(
                dimension_index_values=ind_values,
                segment_numbers=[]
            )
        # Invalid segment numbers
        with pytest.raises(ValueError):
            self._sm_control_seg.get_pixels_by_dimension_index_values(
                dimension_index_values=ind_values,
                segment_numbers=[-1]
            )

    def test_get_pixels_by_source_instances(self):
        all_source_sop_uids = [
            tup[-1] for tup in self._ct_binary_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids[1:3]

        pixels = self._ct_binary_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_seg.Rows,
            self._ct_binary_seg.Columns,
            self._ct_binary_seg.number_of_segments
        )
        assert pixels.shape == out_shape

        pixels = self._ct_binary_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            combine_segments=True
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_seg.Rows,
            self._ct_binary_seg.Columns,
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_source_instances_missing_instance(self):
        # Tests where there is an instance passed that has no recorded
        # segmentation
        all_source_sop_uids = [
            tup[-1] for tup in self._ct_binary_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids[1:3]

        # Append a non-existent UID
        source_sop_uids.append("1.2.3.4.5")

        # Should fail unless the assert flag is passed
        with pytest.raises(KeyError):
            self._ct_binary_seg.get_pixels_by_source_instance(
                source_sop_instance_uids=source_sop_uids,
            )

        pixels = self._ct_binary_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            assert_missing_frames_are_empty=True,
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_seg.Rows,
            self._ct_binary_seg.Columns,
            self._ct_binary_seg.number_of_segments
        )
        assert pixels.shape == out_shape

        # Check the missing instance has been given an empty mask
        assert pixels[-1, :, :, :].max() == 0

        # Also test with combine segments
        pixels = self._ct_binary_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            assert_missing_frames_are_empty=True,
            combine_segments=True
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_seg.Rows,
            self._ct_binary_seg.Columns,
        )
        assert pixels.shape == out_shape

        # Check the missing instance has been given an empty mask
        assert pixels[-1, :, :].max() == 0

    def test_get_pixels_by_source_instances_cr(self):
        all_source_sop_uids = [
            tup[-1] for tup in self._cr_binary_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids

        pixels = self._cr_binary_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
        )

        out_shape = (
            len(source_sop_uids),
            self._cr_binary_seg.Rows,
            self._cr_binary_seg.Columns,
            self._cr_binary_seg.number_of_segments
        )
        assert pixels.shape == out_shape

        pixels = self._cr_binary_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            combine_segments=True
        )

        out_shape = (
            len(source_sop_uids),
            self._cr_binary_seg.Rows,
            self._cr_binary_seg.Columns,
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_source_instances_with_segments(self):
        all_source_sop_uids = [
            tup[-1] for tup in self._ct_binary_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids[1:3]
        segment_numbers = [1]

        pixels = self._ct_binary_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            segment_numbers=segment_numbers
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_seg.Rows,
            self._ct_binary_seg.Columns,
            len(segment_numbers)
        )
        assert pixels.shape == out_shape

    def test_get_pixels_by_source_instances_invalid(self):
        all_source_sop_uids = [
            tup[-1] for tup in self._ct_binary_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids[1:3]

        # Empty SOP uids
        with pytest.raises(ValueError):
            self._ct_binary_seg.get_pixels_by_source_instance(
                source_sop_instance_uids=[],
            )
        # Empty SOP uids
        with pytest.raises(KeyError):
            self._ct_binary_seg.get_pixels_by_source_instance(
                source_sop_instance_uids=['1.2.3.4'],
            )
        # Empty segments
        with pytest.raises(ValueError):
            self._ct_binary_seg.get_pixels_by_source_instance(
                source_sop_instance_uids=source_sop_uids,
                segment_numbers=[]
            )
        # Invalid segments
        with pytest.raises(ValueError):
            self._ct_binary_seg.get_pixels_by_source_instance(
                source_sop_instance_uids=source_sop_uids,
                segment_numbers=[0]
            )

    def test_get_pixels_by_source_instances_binary_fractional(self):
        all_source_sop_uids = [
            tup[-1] for tup in
            self._ct_binary_fractional_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids[1:3]

        pixels = self._ct_binary_fractional_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_fractional_seg.Rows,
            self._ct_binary_fractional_seg.Columns,
            self._ct_binary_fractional_seg.number_of_segments
        )
        assert pixels.shape == out_shape
        assert np.all(np.unique(pixels) == np.array([0.0, 1.0]))

        pixels = self._ct_binary_fractional_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            combine_segments=True
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_fractional_seg.Rows,
            self._ct_binary_fractional_seg.Columns,
        )
        assert pixels.shape == out_shape
        assert np.all(np.unique(pixels) == np.array([0.0, 1.0]))

    def test_get_pixels_by_source_instances_true_fractional(self):
        all_source_sop_uids = [
            tup[-1] for tup in
            self._ct_true_fractional_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids[1:3]

        pixels = self._ct_true_fractional_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_true_fractional_seg.Rows,
            self._ct_true_fractional_seg.Columns,
            self._ct_true_fractional_seg.number_of_segments
        )
        assert pixels.shape == out_shape
        assert pixels.max() <= 1.0
        assert pixels.min() >= 0.0
        assert len(np.unique(pixels)) > 2

        # Without fractional rescaling
        pixels = self._ct_true_fractional_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            rescale_fractional=False
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_true_fractional_seg.Rows,
            self._ct_true_fractional_seg.Columns,
            self._ct_true_fractional_seg.number_of_segments
        )
        assert pixels.shape == out_shape
        assert pixels.max() == 128
        assert len(np.unique(pixels)) > 2

        # Can't combine segments with a true fractional segmentation
        with pytest.raises(ValueError):
            self._ct_true_fractional_seg.get_pixels_by_source_instance(
                source_sop_instance_uids=source_sop_uids,
                combine_segments=True
            )

    def test_get_pixels_by_source_instances_overlap(self):
        # Test that overlapping segments are detected
        all_source_sop_uids = [
            tup[-1] for tup in
            self._ct_binary_overlap_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids

        pixels = self._ct_binary_overlap_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_overlap_seg.Rows,
            self._ct_binary_overlap_seg.Columns,
            self._ct_binary_overlap_seg.number_of_segments
        )
        assert pixels.shape == out_shape

        with pytest.raises(RuntimeError):
            self._ct_binary_overlap_seg.get_pixels_by_source_instance(
                source_sop_instance_uids=source_sop_uids,
                combine_segments=True
            )

    def test_get_pixels_by_source_instances_overlap_no_checks(self):
        # Test that skipping overlapping tests works as expected
        all_source_sop_uids = [
            tup[-1] for tup in
            self._ct_binary_overlap_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids

        pixels = self._ct_binary_overlap_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
        )

        out_shape = (
            len(source_sop_uids),
            self._ct_binary_overlap_seg.Rows,
            self._ct_binary_overlap_seg.Columns,
            self._ct_binary_overlap_seg.number_of_segments
        )
        assert pixels.shape == out_shape

        out = self._ct_binary_overlap_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
            combine_segments=True,
            skip_overlap_checks=True,
        )
        raw_pix_arr = self._ct_binary_overlap_seg.pixel_array
        expected_array = np.maximum(
            raw_pix_arr[:4, ...],
            raw_pix_arr[4:, ...] * 2
        )
        assert np.array_equal(expected_array, out)

    def test_allow_missing_positions(self):
        all_source_sop_uids = [
            tup[-1] for tup in
            self._ct_binary_overlap_seg.get_source_image_uids()
        ]
        source_sop_uids = all_source_sop_uids

        # There are no missing frames when indexing with this limited number of
        # source UIDs
        self._ct_binary_overlap_seg.get_pixels_by_source_instance(
            source_sop_instance_uids=source_sop_uids,
        )

        # There are missing frames when indexing by volume position
        msg = (
            'Frame positions do not form a regularly-spaced volume.'
        )
        with pytest.raises(RuntimeError, match=msg):
            self._ct_binary_overlap_seg.get_volume(
                allow_missing_positions=False
            )

    def test_get_volume_binary(self):
        vol = self._ct_binary_seg.get_volume()
        assert isinstance(vol, Volume)
        assert vol.spatial_shape == (3, 16, 16)
        assert vol.shape == (3, 16, 16, 1)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegments(self):
        vol = self._ct_binary_overlap_seg.get_volume()
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (165, 16, 16)
        assert vol.shape == (165, 16, 16, 2)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_overlap_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment2(self):
        vol = self._ct_binary_overlap_seg.get_volume(segment_numbers=[2])
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (165, 16, 16)
        assert vol.shape == (165, 16, 16, 1)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_overlap_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment_combine(self):
        vol = self._ct_binary_overlap_seg.get_volume(
            combine_segments=True,
            skip_overlap_checks=True,
        )
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (165, 16, 16)
        assert vol.shape == (165, 16, 16)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_overlap_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment_slice_start(self):
        vol = self._ct_binary_overlap_seg.get_volume(
            slice_start=161,
        )
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (5, 16, 16)
        assert vol.shape == (5, 16, 16, 2)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_overlap_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment_slice_start_negative(self):
        vol = self._ct_binary_overlap_seg.get_volume(
            slice_start=-6,
        )
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (6, 16, 16)
        assert vol.shape == (6, 16, 16, 2)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_overlap_seg
            .get_volume_geometry().spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment_slice_end(self):
        vol = self._ct_binary_overlap_seg.get_volume(
            slice_end=18,
        )
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (17, 16, 16)
        assert vol.shape == (17, 16, 16, 2)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_overlap_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment_slice_end_negative(self):
        vol = self._ct_binary_overlap_seg.get_volume(
            slice_end=-10,
        )
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (155, 16, 16)
        assert vol.shape == (155, 16, 16, 2)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_overlap_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment_center(self):
        vol = self._ct_binary_overlap_seg.get_volume(
            slice_start=50,
            slice_end=57,
        )
        assert isinstance(vol, Volume)
        # Note that this segmentation has a large number of missing slices
        assert vol.spatial_shape == (7, 16, 16)
        assert vol.shape == (7, 16, 16, 2)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_seg.get_volume_geometry().spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_overlap_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_binary_multisegment_cropped(self):
        vol = self._ct_binary_overlap_seg.get_volume()
        vol_cropped = self._ct_binary_overlap_seg.get_volume(
            row_start=2,
            row_end=-2,
            column_start=3,
            column_end=-5,
        )

        # Note that this segmentation has a large number of missing slices
        assert vol_cropped.spatial_shape == (165, 13, 9)
        assert vol_cropped.shape == (165, 13, 9, 2)

        assert np.array_equal(
            vol_cropped.array,
            vol[:, 1:-2, 2:-5].array
        )
        assert np.array_equal(
            vol_cropped.affine,
            vol[:, 1:-2, 2:-5].affine
        )

        vol_cropped = self._ct_binary_overlap_seg.get_volume(
            row_start=1,
            row_end=-2,
            column_start=2,
            column_end=-5,
            as_indices=True
        )

        # Note that this segmentation has a large number of missing slices
        assert vol_cropped.spatial_shape == (165, 13, 9)
        assert vol_cropped.shape == (165, 13, 9, 2)

        assert np.array_equal(
            vol_cropped.array,
            vol[:, 1:-2, 2:-5].array
        )
        assert np.array_equal(
            vol_cropped.affine,
            vol[:, 1:-2, 2:-5].affine
        )

    def test_get_volume_binary_combine(self):
        vol = self._ct_binary_seg.get_volume(combine_segments=True)
        assert isinstance(vol, Volume)
        assert vol.spatial_shape == (3, 16, 16)
        assert vol.shape == (3, 16, 16)
        assert vol.pixel_spacing == tuple(
            self._ct_binary_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_binary_seg.get_volume_geometry().spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_binary_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )

    def test_get_volume_fractional(self):
        vol = self._ct_true_fractional_seg.get_volume()
        assert isinstance(vol, Volume)
        assert vol.spatial_shape == (3, 16, 16)
        assert vol.shape == (3, 16, 16, 1)
        assert vol.pixel_spacing == tuple(
            self._ct_true_fractional_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_true_fractional_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_true_fractional_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )
        assert vol.dtype == np.float32

    def test_get_volume_fractional_noscale(self):
        vol = self._ct_true_fractional_seg.get_volume(rescale_fractional=False)
        assert isinstance(vol, Volume)
        assert vol.spatial_shape == (3, 16, 16)
        assert vol.shape == (3, 16, 16, 1)
        assert vol.pixel_spacing == tuple(
            self._ct_true_fractional_seg
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
        assert vol.spacing_between_slices == (
            self._ct_true_fractional_seg
            .get_volume_geometry()
            .spacing_between_slices
        )
        assert vol.direction_cosines == tuple(
            self._ct_true_fractional_seg
            .SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        assert vol.get_closest_patient_orientation() == (
            PatientOrientationValuesBiped.F,
            PatientOrientationValuesBiped.P,
            PatientOrientationValuesBiped.L,
        )
        assert vol.dtype == np.uint8

    def test_total_pixel_matrix_palette_color(self):
        seg = self._sm_control_labelmap_palette_color_seg

        pixels = seg.get_total_pixel_matrix(
            combine_segments=True,
            apply_palette_color_lut=True,
        )
        assert pixels.shape == (
            seg.TotalPixelMatrixRows,
            seg.TotalPixelMatrixColumns,
            3
        )

        vol = seg.get_volume(
            combine_segments=True,
            apply_palette_color_lut=True,
        )
        assert vol.shape == (
            1,
            seg.TotalPixelMatrixRows,
            seg.TotalPixelMatrixColumns,
            3
        )
        assert vol.channel_descriptors == (
            RGB_COLOR_CHANNEL_DESCRIPTOR,
        )

        msg = (
            "apply_palette_color_lut' requires that 'combine_segments' is "
            "True and relabel is False."
        )
        with pytest.raises(ValueError, match=msg):
            # Requires combine_segments
            seg.get_total_pixel_matrix(apply_palette_color_lut=True)

        msg = (
            "apply_palette_color_lut' requires that 'combine_segments' is "
            "True and relabel is False."
        )
        with pytest.raises(ValueError, match=msg):
            # Requires relabel false
            seg.get_total_pixel_matrix(
                apply_palette_color_lut=True,
                relabel=True,
                combine_segments=True,
            )

        monochrome_seg = self._sm_control_labelmap_seg
        msg = (
            'Palette color transform is required but the image is not a '
            'palette color image.'
        )
        with pytest.raises(ValueError, match=msg):
            # Requires combine_segments
            monochrome_seg.get_total_pixel_matrix(
                combine_segments=True,
                apply_palette_color_lut=True,
            )

    def test_parsing_nonconsecutive_segment_numbers(self):
        # parsing a labelmap segmentation with non-consective segment numbers
        seg_num = 4
        other_seg_num = 8
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        ct_image = dcmread(
            str(data_dir.joinpath('test_files', 'ct_image.dcm'))
        )
        array = np.zeros(
            (ct_image.Rows, ct_image.Columns),
            dtype=np.uint8,
        )
        array[20:30, 20:30] = seg_num
        array[70:80, 70:80] = other_seg_num
        abnormal_structure = codes.SCT.MorphologicallyAbnormalStructure
        desc = SegmentDescription(
            segment_number=seg_num,
            segment_label=f'Segment #{seg_num}',
            segmented_property_category=abnormal_structure,
            segmented_property_type=codes.SCT.Neoplasm,
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
            algorithm_identification=AlgorithmIdentificationSequence(
                name='foo',
                family=codes.DCM.ArtificialIntelligence,
                version='v1'
            )
        )
        other_desc = SegmentDescription(
            segment_number=other_seg_num,
            segment_label=f'Segment #{other_seg_num}',
            segmented_property_category=abnormal_structure,
            segmented_property_type=codes.SCT.Neoplasm,
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
            algorithm_identification=AlgorithmIdentificationSequence(
                name='foo',
                family=codes.DCM.ArtificialIntelligence,
                version='v1'
            )
        )

        seg = Segmentation(
            source_images=[ct_image],
            pixel_array=array,
            segmentation_type=SegmentationTypeValues.LABELMAP,
            segment_descriptions=[desc, other_desc],
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
            manufacturer='manufacturer',
            manufacturer_model_name='model_name',
            software_versions='123',
            device_serial_number='456',
        )
        assert len(seg.SegmentSequence) == 3  # includes bg
        assert seg.segment_numbers == [seg_num, other_seg_num]

        assert isinstance(seg.get_segment_description(4), SegmentDescription)
        msg = '1 is an invalid segment number for this dataset.'
        with pytest.raises(IndexError, match=msg):
            seg.get_segment_description(1)

        assert seg.get_segment_numbers() == [seg_num, other_seg_num]
        assert seg.get_segment_numbers(
            segment_label=f'Segment #{seg_num}'
        ) == [seg_num]
        assert seg.get_segment_numbers(
            segment_label='not existing'
        ) == []

        out = seg.get_pixels_by_source_instance(
            source_sop_instance_uids=[ct_image.SOPInstanceUID],
            combine_segments=True,
        )
        assert out.shape == (1, ct_image.Rows, ct_image.Columns)
        assert np.array_equal(out[0], array)

        out = seg.get_pixels_by_source_instance(
            source_sop_instance_uids=[ct_image.SOPInstanceUID],
            combine_segments=True,
            relabel=True,
        )
        assert out.shape == (1, ct_image.Rows, ct_image.Columns)
        assert np.array_equal(np.unique(out), np.array([0, 1, 2]))

        out = seg.get_pixels_by_source_instance(
            source_sop_instance_uids=[ct_image.SOPInstanceUID],
            combine_segments=True,
            relabel=True,
            segment_numbers=[seg_num]
        )
        assert out.shape == (1, ct_image.Rows, ct_image.Columns)
        assert np.array_equal(out[0], array == seg_num)

        out = seg.get_pixels_by_source_instance(
            source_sop_instance_uids=[ct_image.SOPInstanceUID],
        )
        assert out.shape == (1, ct_image.Rows, ct_image.Columns, 2)
        assert np.array_equal(out[0, :, :, 0], array == seg_num)
        assert np.array_equal(out[0, :, :, 1], array == other_seg_num)

        out = seg.get_pixels_by_source_instance(
            source_sop_instance_uids=[ct_image.SOPInstanceUID],
            segment_numbers=[other_seg_num, seg_num]
        )
        assert out.shape == (1, ct_image.Rows, ct_image.Columns, 2)
        assert np.array_equal(out[0, :, :, 0], array == other_seg_num)
        assert np.array_equal(out[0, :, :, 1], array == seg_num)

        out = seg.get_pixels_by_source_instance(
            source_sop_instance_uids=[ct_image.SOPInstanceUID],
            segment_numbers=[other_seg_num]
        )
        assert out.shape == (1, ct_image.Rows, ct_image.Columns, 1)
        assert np.array_equal(out[0, :, :, 0], array == other_seg_num)

        msg = (
            'Segment numbers array contains invalid values.'
        )
        with pytest.raises(ValueError, match=msg):
            seg.get_pixels_by_source_instance(
                source_sop_instance_uids=[ct_image.SOPInstanceUID],
                segment_numbers=[5]  # not found
            )


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

    def test_iter_segments_ct_single_frame(self):
        seg_dataset = dcmread(
            'data/test_files/seg_image_ct_binary_single_frame.dcm'
        )

        generator = iter_segments(seg_dataset)
        items = list(generator)
        assert len(items) == 1
        item_segment_1 = items[0]
        assert item_segment_1[0].shape == (1, 128, 128)
        seg_id_item_1 = item_segment_1[1][0].SegmentIdentificationSequence[0]
        assert seg_id_item_1.ReferencedSegmentNumber == 1
        assert item_segment_1[2].SegmentNumber == 1

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
                tracking_uid=UID(),
                tracking_id='first segment'
            ),
            SegmentDescription(
                segment_number=2,
                segment_label='connective tissue',
                segmented_property_category=codes.cid7150.Tissue,
                segmented_property_type=codes.cid7166.ConnectiveTissue,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
                tracking_uid=UID(),
                tracking_id='second segment'
            ),
        ]

        seg_dataset = Segmentation(
            source_images=[image_dataset],
            pixel_array=mask,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=segment_descriptions,
            series_instance_uid=UID(),
            series_number=2,
            sop_instance_uid=UID(),
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


class TestPyramid(unittest.TestCase):

    def setUp(self):
        file_path = Path(__file__)
        data_dir = file_path.parent.parent.joinpath('data')
        self._sm_image = dcmread(
            str(data_dir.joinpath('test_files', 'sm_image.dcm'))
        )
        tpm_size = (
            self._sm_image.TotalPixelMatrixRows,
            self._sm_image.TotalPixelMatrixColumns
        )
        self._seg_pix = np.zeros(
            tpm_size,
            dtype=np.uint8,
        )
        self._seg_pix[5:15, 3:8] = 1

        self._seg_pix_multisegment = np.zeros(
            (1, *tpm_size, 3),
            dtype=np.uint8,
        )
        self._seg_pix_multisegment[0, 5:15, 3:8, 0] = 1
        self._seg_pix_multisegment[0, 23:42, 13:18, 1] = 1
        self._seg_pix_multisegment[0, 45:48, 25:30, 1] = 1

        self._seg_pix_fractional = np.zeros(
            tpm_size,
            dtype=np.float32,
        )
        self._seg_pix_fractional[5:15, 3:8] = 0.5
        self._seg_pix_fractional[3:5, 10:12] = 1.0

        self._n_downsamples = 3
        self._downsampled_pix_arrays = [self._seg_pix]
        self._downsampled_pix_arrays_multisegment = [self._seg_pix_multisegment]
        self._downsampled_pix_arrays_fractional = [self._seg_pix_fractional]
        seg_pil = Image.fromarray(self._seg_pix)
        seg_pil_fractional = Image.fromarray(self._seg_pix_fractional)
        seg_pil_multisegment = [
            Image.fromarray(self._seg_pix_multisegment[0, :, :, i])
            for i in range(self._seg_pix_multisegment.shape[3])
        ]
        pyramid_uid = UID()
        self._source_pyramid = [deepcopy(self._sm_image)]
        self._source_pyramid[0].PyramidUID = pyramid_uid
        for i in range(1, self._n_downsamples):
            f = 2 ** i
            out_size = (
                self._sm_image.TotalPixelMatrixRows // f,
                self._sm_image.TotalPixelMatrixColumns // f
            )

            # Resize the segmentation arrays
            resized = np.array(
                seg_pil.resize(out_size, Image.Resampling.NEAREST)
            )
            self._downsampled_pix_arrays.append(resized)

            resized_fractional = np.array(
                seg_pil_fractional.resize(out_size, Image.Resampling.BILINEAR)
            )
            self._downsampled_pix_arrays_fractional.append(resized_fractional)

            resized_multisegment = np.stack(
                [
                    im.resize(out_size, Image.Resampling.NEAREST)
                    for im in seg_pil_multisegment
                ],
                axis=-1
            )[None]
            self._downsampled_pix_arrays_multisegment.append(
                resized_multisegment
            )

            # Mock lower-resolution source images. No need to have their pixel
            # data correctly set as it isn't used. Just update the relevant
            # metadata
            src_pixel_spacing = (
                self._sm_image
                .SharedFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .PixelSpacing
            )
            pixel_spacing = [src_pixel_spacing[0] * f, src_pixel_spacing[1] * f]
            downsampled_source_im = deepcopy(self._sm_image)
            delattr(downsampled_source_im, 'PixelData')
            downsampled_source_im.TotalPixelMatrixRows = out_size[0]
            downsampled_source_im.TotalPixelMatrixColumns = out_size[1]
            (
                downsampled_source_im
                .SharedFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .PixelSpacing
            ) = pixel_spacing
            downsampled_source_im.PyramidUID = pyramid_uid
            self._source_pyramid.append(downsampled_source_im)

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
        self._segment_descriptions_multi = [
            SegmentDescription(
                segment_number=i,
                segment_label=f'Segment #{i}',
                segmented_property_category=self._segmented_property_category,
                segmented_property_type=self._segmented_property_type,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC.value,
                algorithm_identification=AlgorithmIdentificationSequence(
                    name='bla',
                    family=codes.DCM.ArtificialIntelligence,
                    version='v1'
                )
            )
            for i in range(1, 4)
        ]

    def test_pyramid_factors(self):
        downsample_factors = [2.0, 5.0]
        segs = create_segmentation_pyramid(
            source_images=[self._sm_image],
            pixel_arrays=[self._seg_pix],
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
            downsample_factors=downsample_factors,
        )

        assert len(segs) == len(downsample_factors) + 1
        tol = 0.01
        for f, seg in zip([1.0, *downsample_factors], segs):
            assert hasattr(seg, 'PyramidUID')
            assert abs(
                seg.TotalPixelMatrixRows - int(self._seg_pix.shape[0] / f)
            ) < tol
            assert abs(
                seg.TotalPixelMatrixColumns - int(self._seg_pix.shape[1] / f)
            ) < tol

    def test_pyramid_downsample_factors(self):
        # Test construction when given a single source image, single
        # segmentation mask, and specified downsample factors
        downsample_factors = [2.0, 5.0]
        segs = create_segmentation_pyramid(
            source_images=[self._sm_image],
            pixel_arrays=[self._seg_pix],
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
            downsample_factors=downsample_factors,
        )

        assert len(segs) == len(downsample_factors) + 1
        tol = 0.01
        for f, seg in zip([1.0, *downsample_factors], segs):
            assert hasattr(seg, 'PyramidUID')
            assert abs(
                seg.TotalPixelMatrixRows - int(self._seg_pix.shape[0] / f)
            ) < tol
            assert abs(
                seg.TotalPixelMatrixColumns - int(self._seg_pix.shape[1] / f)
            ) < tol

    def test_single_source_multiple_pixel_arrays(self):
        # Test construction when given a single source image and multiple
        # segmentation images
        segs = create_segmentation_pyramid(
            source_images=[self._sm_image],
            pixel_arrays=self._downsampled_pix_arrays,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._downsampled_pix_arrays)
        for pix, seg in zip(self._downsampled_pix_arrays, segs):
            assert hasattr(seg, 'PyramidUID')
            assert np.array_equal(
                seg.get_total_pixel_matrix(combine_segments=True),
                pix
            )

    def test_multiple_source_single_pixel_array(self):
        # Test construction when given multiple source images and a single
        # segmentation image
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=[self._seg_pix],
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays, segs):
            assert hasattr(seg, 'PyramidUID')
            assert np.array_equal(
                seg.get_total_pixel_matrix(combine_segments=True),
                pix
            )

    def test_multiple_source_single_pixel_array_multisegment(self):
        # Test construction when given multiple source images and a single
        # segmentation image
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=[self._seg_pix_multisegment],
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions_multi,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays_multisegment, segs):
            assert hasattr(seg, 'PyramidUID')
            seg_pix = seg.get_total_pixel_matrix()
            assert np.array_equal(
                seg_pix,
                pix[0]
            )

    def test_multiple_source_single_pixel_array_float(self):
        # Test construction when given multiple source images and a single
        # segmentation image
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=[self._seg_pix.astype(np.float32)],
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays, segs):
            assert hasattr(seg, 'PyramidUID')
            assert np.array_equal(
                seg.get_total_pixel_matrix(combine_segments=True),
                pix
            )

    def test_multiple_source_single_pixel_array_fractional(self):
        # Test construction when given multiple source images and a single
        # segmentation image
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=[self._seg_pix_fractional],
            segmentation_type=SegmentationTypeValues.FRACTIONAL,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
            max_fractional_value=200,
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays_fractional, segs):
            assert hasattr(seg, 'PyramidUID')
            assert np.allclose(
                seg.get_total_pixel_matrix(combine_segments=False)[:, :, 0],
                pix,
                atol=0.005,  # 1/200
            )

    def test_multiple_source_multiple_pixel_arrays(self):
        # Test construction when given multiple source images and multiple
        # segmentation images
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=self._downsampled_pix_arrays,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays, segs):
            assert hasattr(seg, 'PyramidUID')
            assert np.array_equal(
                seg.get_total_pixel_matrix(combine_segments=True),
                pix
            )

    def test_multiple_source_multiple_pixel_arrays_multisegment(self):
        # Test construction when given multiple source images and multiple
        # segmentation images
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=self._downsampled_pix_arrays_multisegment,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions_multi,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays_multisegment, segs):
            assert hasattr(seg, 'PyramidUID')
            assert np.array_equal(
                seg.get_total_pixel_matrix(),
                pix[0]
            )

    def test_multiple_source_multiple_pixel_arrays_multisegment_from_labelmap(
        self
    ):
        # Test construction when given multiple source images and multiple
        # segmentation images
        mask = np.argmax(self._seg_pix_multisegment, axis=3).astype(np.uint8)
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=mask,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=self._segment_descriptions_multi,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays_multisegment, segs):
            mask = np.argmax(pix, axis=3).astype(np.uint8)
            assert hasattr(seg, 'PyramidUID')
            assert np.array_equal(
                seg.get_total_pixel_matrix(combine_segments=True),
                mask[0]
            )

    def test_multiple_source_multiple_pixel_arrays_multisegment_labelmap(self):
        # Test construction when given multiple source images and multiple
        # segmentation images
        mask = np.argmax(self._seg_pix_multisegment, axis=3).astype(np.uint8)
        segs = create_segmentation_pyramid(
            source_images=self._source_pyramid,
            pixel_arrays=mask,
            segmentation_type=SegmentationTypeValues.LABELMAP,
            segment_descriptions=self._segment_descriptions_multi,
            series_instance_uid=UID(),
            series_number=1,
            manufacturer='Foo',
            manufacturer_model_name='Bar',
            software_versions='1',
            device_serial_number='123',
        )

        assert len(segs) == len(self._source_pyramid)
        for pix, seg in zip(self._downsampled_pix_arrays_multisegment, segs):
            mask = np.argmax(pix, axis=3).astype(np.uint8)
            assert hasattr(seg, 'PyramidUID')
            assert np.array_equal(
                seg.get_total_pixel_matrix(combine_segments=True),
                mask[0]
            )
