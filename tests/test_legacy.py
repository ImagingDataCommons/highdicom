from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from io import BytesIO
from datetime import datetime, timedelta
import enum
import re
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from pydicom import FileDataset, FileMetaDataset, Dataset
from pydicom.data import get_testdata_file
from pydicom.datadict import tag_for_keyword
from pydicom.dataelem import DataElement
from pydicom.multival import MultiValue
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEG2000,
    JPEGBaseline8Bit,
    JPEGLSLossless,
    JPEGLSNearLossless,
    RLELossless,
    EnhancedMRImageStorage,
)
from pydicom.valuerep import DA, DSfloat, DT, TM
from highdicom.legacy import (
    LegacyConvertedEnhancedCTImage,
    LegacyConvertedEnhancedMRImage,
    LegacyConvertedEnhancedPETImage,
    lcectimread,
    lcemrimread,
    lcepetimread,
)
from highdicom import UID

import pytest

from tests.utils import write_and_read_dataset


class Modality(enum.IntEnum):
    CT = 0
    MR = 1
    PT = 2


MODALITY_LEGACY_SOP_CLASS_MAP = {
    Modality.CT: '1.2.840.10008.5.1.4.1.1.2',
    Modality.MR: '1.2.840.10008.5.1.4.1.1.4',
    Modality.PT: '1.2.840.10008.5.1.4.1.1.128',
}
MODALITY_ENHANCED_SOP_CLASS_MAP = {
    Modality.CT: '1.2.840.10008.5.1.4.1.1.2.2',
    Modality.MR: '1.2.840.10008.5.1.4.1.1.4.4',
    Modality.PT: '1.2.840.10008.5.1.4.1.1.128.1',
}
MODALITY_CLASS_MAP = {
    Modality.CT: LegacyConvertedEnhancedCTImage,
    Modality.MR: LegacyConvertedEnhancedMRImage,
    Modality.PT: LegacyConvertedEnhancedPETImage,
}


class DicomGenerator:

    def __init__(
        self,
        slice_per_frameset: int = 3,
        slice_thickness: float = 0.1,
        pixel_spacing: float = 0.1,
        row: int = 2,
        col: int = 2,
    ) -> None:
        self._slice_per_frameset = slice_per_frameset
        self._slice_thickness = slice_thickness
        self._pixel_spacing = pixel_spacing
        self._row = row
        self._col = col
        self._study_uid = UID()
        self._z_orientation_mat = [
            1.000000, 0.000000, 0.000000,
            0.000000, 1.000000, 0.000000
        ]
        self._z_position_vec = [0.0, 0.0, 1.0]
        self._y_orientation_mat = [
            0.000000, 0.000000, 1.000000,
            1.000000, 0.000000, 0.000000
        ]
        self._y_position_vec = [0.0, 1.0, 0.0]
        self._x_orientation_mat = [
            0.000000, 1.000000, 0.000000,
            0.000000, 0.000000, 1.000000
        ]
        self._x_position_vec = [1.0, 0.0, 0.0]

    def _generate_frameset(
        self,
        modality: Modality,
        orientation_mat: list,
        position_vec: list,
        series_uid: str,
        first_slice_offset: float = 0,
        frameset_idx: int = 0,
        color: bool = False,
        bytes_per_voxel: int = 2,
    ) -> list:
        output_dataset = []
        slice_pos = first_slice_offset
        slice_thickness = self._slice_thickness
        study_uid = self._study_uid
        frame_of_ref_uid = UID()
        date_ = DA(datetime.now().date())
        age = timedelta(days=45 * 365)
        time_ = TM(datetime.now().time())
        cols = self._col
        rows = self._row
        samples_per_pixel = 3 if color else 1
        photomtc_intn = 'RGB' if color else 'MONOCHROME2'
        max_pixel_value = 2 ** (8 * bytes_per_voxel)
        array_shape = (rows, cols, 3) if color else (rows, cols)
        dtype = {
            1: np.uint8,
            2: np.uint16,
        }[bytes_per_voxel]

        for i in range(self._slice_per_frameset):
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = MODALITY_LEGACY_SOP_CLASS_MAP[
                modality
            ]
            file_meta.MediaStorageSOPInstanceUID = UID()
            file_meta.ImplementationClassUID = UID()
            tmp_dataset = FileDataset(
                '', {}, file_meta=file_meta,
            )
            tmp_dataset.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
            tmp_dataset.SliceLocation = DSfloat(
                slice_pos + i * slice_thickness,
                auto_format=True
            )
            tmp_dataset.SliceThickness = DSfloat(
                slice_thickness,
                auto_format=True
            )
            tmp_dataset.WindowCenter = 1
            tmp_dataset.WindowWidth = 2
            tmp_dataset.AcquisitionNumber = i
            tmp_dataset.InstanceNumber = i
            tmp_dataset.SeriesNumber = 1
            tmp_dataset.ImageOrientationPatient = orientation_mat
            tmp_dataset.ImagePositionPatient = [
                DSfloat(tmp_dataset.SliceLocation * i, auto_format=True)
                for i in position_vec
            ]
            if modality == Modality.CT:
                tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            elif modality == Modality.MR:
                tmp_dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'OTHER']
            elif modality == Modality.PT:
                tmp_dataset.ImageType = [
                    'ORIGINAL', 'PRIMARY', 'RECON', 'EMISSION']

            tmp_dataset.PixelSpacing = [
                self._pixel_spacing, self._pixel_spacing
            ]
            tmp_dataset.PatientName = 'Doe^John'
            tmp_dataset.FrameOfReferenceUID = frame_of_ref_uid
            tmp_dataset.SOPClassUID = MODALITY_LEGACY_SOP_CLASS_MAP[modality]
            tmp_dataset.SOPInstanceUID = UID()
            tmp_dataset.SeriesInstanceUID = series_uid
            tmp_dataset.StudyInstanceUID = study_uid
            tmp_dataset.BitsStored = bytes_per_voxel * 8
            tmp_dataset.HighBit = (bytes_per_voxel * 8 - 1)
            tmp_dataset.PixelRepresentation = 0
            tmp_dataset.Columns = cols
            tmp_dataset.Rows = rows
            tmp_dataset.SamplesPerPixel = samples_per_pixel
            tmp_dataset.AccessionNumber = '1{:05d}'.format(frameset_idx)
            tmp_dataset.AcquisitionDate = date_
            tmp_dataset.AcquisitionTime = datetime.now().time()
            tmp_dataset.AdditionalPatientHistory = 'UTERINE CA PRE-OP EVAL'
            tmp_dataset.ContentDate = date_
            tmp_dataset.ContentTime = TM(datetime.now().time())
            tmp_dataset.Manufacturer = 'Manufacturer'
            tmp_dataset.ManufacturerModelName = 'Model'
            tmp_dataset.Modality = modality.name
            tmp_dataset.PatientAge = '064Y'
            tmp_dataset.PatientBirthDate = DA(date_ - age)
            tmp_dataset.PatientID = 'ID{:05d}'.format(frameset_idx)
            tmp_dataset.PatientIdentityRemoved = 'YES'
            tmp_dataset.PatientPosition = 'FFS'
            tmp_dataset.PatientSex = 'F'
            tmp_dataset.PhotometricInterpretation = photomtc_intn
            tmp_dataset.PositionReferenceIndicator = 'XY'
            tmp_dataset.ProtocolName = 'some protocol'
            tmp_dataset.ReferringPhysicianName = ''
            tmp_dataset.SeriesDate = date_
            tmp_dataset.SeriesDescription = "A legacy series"
            tmp_dataset.SeriesTime = time_
            tmp_dataset.SoftwareVersions = '01'
            tmp_dataset.SpecificCharacterSet = 'ISO_IR 100'
            tmp_dataset.StudyDate = date_
            tmp_dataset.StudyDescription = 'test study'
            tmp_dataset.StudyID = ''
            tmp_dataset.StudyTime = time_
            if (modality == Modality.CT):
                tmp_dataset.RescaleIntercept = -1024
                tmp_dataset.RescaleSlope = 1

            if color:
                tmp_dataset.PlanarConfiguration = 0

            pixel_array = np.random.randint(
                0, max_pixel_value, array_shape, dtype=dtype
            )
            tmp_dataset.set_pixel_data(
                pixel_array,
                photometric_interpretation=photomtc_intn,
                bits_stored=8 * bytes_per_voxel,
            )

            output_dataset.append(tmp_dataset)

        return output_dataset

    def generate_mixed_framesets(
        self,
        modality: Modality,
        frame_set_count: int,
        parallel: bool = True,
        flatten_output: bool = True,
        color: bool = False,
        bytes_per_voxel: int = 2,
    ) -> list:
        out = []
        orients = [
            self._z_orientation_mat,
            self._y_orientation_mat,
            self._x_orientation_mat,
        ]
        poses = [
            self._z_position_vec,
            self._y_position_vec,
            self._x_position_vec,
        ]
        se_uid = UID()
        for i in range(frame_set_count):
            if parallel:
                pos = poses[0]
                orient = orients[0]
            else:
                pos = poses[i % len(poses)]
                orient = orients[i % len(orients)]
            if flatten_output:
                out.extend(
                    self._generate_frameset(
                        modality,
                        orient,
                        pos,
                        se_uid,
                        i * 50,
                        i,
                        color=color,
                        bytes_per_voxel=bytes_per_voxel,
                    )
                )
            else:
                out.append(
                    self._generate_frameset(
                        modality,
                        orient,
                        pos,
                        se_uid,
                        i * 50,
                        i,
                        color=color,
                        bytes_per_voxel=bytes_per_voxel,
                    )
                )

        return out


@pytest.fixture(params=[Modality.CT, Modality.MR, Modality.PT])
def modality(request):
    return request.param


@pytest.mark.parametrize(
    'number_of_frames', [1, 2, 5, 10],
)
def test_conversion(
    modality: Modality,
    number_of_frames: int
) -> None:
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(number_of_frames)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    output_series_uid = UID()
    output_sop_uid = UID()
    output_series_number = 23
    output_instance_number = 2
    series_description = 'Converted Series'

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=output_series_uid,
        sop_instance_uid=output_sop_uid,
        series_number=output_series_number,
        instance_number=output_instance_number,
        series_description=series_description,
    )

    assert converted.SOPInstanceUID == output_sop_uid
    assert converted.SeriesInstanceUID == output_series_uid
    assert converted.SeriesNumber == output_series_number
    assert converted.InstanceNumber == output_instance_number

    frame_type_seq_kw = {
        Modality.MR: 'MRImageFrameTypeSequence',
        Modality.CT: 'CTImageFrameTypeSequence',
        Modality.PT: 'PETFrameTypeSequence',
    }[modality]

    assert converted.NumberOfFrames == number_of_frames
    assert (
        converted.SOPClassUID == MODALITY_ENHANCED_SOP_CLASS_MAP[modality]
    )
    assert not hasattr(converted, 'LargestImagePixelValue')
    assert not hasattr(converted, 'SmallestImagePixelValue')
    sfgs = converted.SharedFunctionalGroupsSequence[0]

    # Frame Anatomy
    assert not hasattr(sfgs, 'FrameAnatomySequence')

    # Pixel measures
    pix_meas_seq = sfgs.PixelMeasuresSequence[0]
    assert (
        pix_meas_seq.PixelSpacing ==
        legacy_datasets[0].PixelSpacing
    )
    assert (
        pix_meas_seq.SliceThickness ==
        legacy_datasets[0].SliceThickness
    )

    # Plane Orientation
    pl_ori_seq = sfgs.PlaneOrientationSequence[0]
    assert (
        pl_ori_seq.ImageOrientationPatient ==
        legacy_datasets[0].ImageOrientationPatient
    )

    # Frame VOI LUT
    fr_voi_lut_seq = sfgs.FrameVOILUTSequence[0]
    assert (fr_voi_lut_seq.WindowWidth == 2)
    assert (fr_voi_lut_seq.WindowCenter == 1)

    # Frame Type
    fr_type_seq = getattr(sfgs, frame_type_seq_kw)[0]
    expected_frame_type = list(legacy_datasets[0].ImageType)[:3]
    expected_frame_type.append('NONE')
    assert fr_type_seq.FrameType == expected_frame_type
    assert fr_type_seq.VolumetricProperties == 'VOLUME'
    assert fr_type_seq.PixelPresentation == 'MONOCHROME'
    assert fr_type_seq.VolumeBasedCalculationTechnique == 'NONE'

    # Pixel Value Transformation
    pix_val_tf_seq = sfgs.PixelValueTransformationSequence[0]
    assert (
        pix_val_tf_seq.RescaleSlope ==
        legacy_datasets[0].get('RescaleSlope', 1)
    )
    assert (
        pix_val_tf_seq.RescaleIntercept ==
        legacy_datasets[0].get('RescaleIntercept', 0)
    )
    expected_rescale_type = 'HU' if modality == Modality.CT else 'US'
    assert pix_val_tf_seq.RescaleType == expected_rescale_type

    if number_of_frames == 1:
        # Plane Position
        pl_pos_seq = sfgs.PlanePositionSequence[0]
        assert (
            pl_pos_seq.ImagePositionPatient ==
            legacy_datasets[0].ImagePositionPatient
        )

    for i, (src, pffg) in enumerate(
        zip(
            legacy_datasets,
            converted.PerFrameFunctionalGroupsSequence,
        )
    ):
        # Frame Content (always per-frame)
        frm_content_seq = pffg.FrameContentSequence[0]
        assert (
            frm_content_seq.FrameAcquisitionNumber ==
            src.AcquisitionNumber
        )
        expected_datetime = datetime.combine(
            src.AcquisitionDate,
            src.AcquisitionTime,
        )
        assert (
            frm_content_seq.FrameAcquisitionDateTime ==
            expected_datetime
        )
        assert frm_content_seq.StackID == '1'

        # Due to choice of orientation, these are reversed
        assert (
            frm_content_seq.InStackPositionNumber ==
            number_of_frames - i
        )

        conv_src_seq = pffg.ConversionSourceAttributesSequence[0]
        assert (
            conv_src_seq.ReferencedSOPClassUID ==
            src.SOPClassUID
        )
        assert (
            conv_src_seq.ReferencedSOPInstanceUID ==
            src.SOPInstanceUID
        )

        if number_of_frames > 1:
            # Plane Position
            pl_pos_seq = pffg.PlanePositionSequence[0]
            assert (
                pl_pos_seq.ImagePositionPatient ==
                src.ImagePositionPatient
            )

    assert len(converted.ReferencedSeriesSequence) == 1
    ref_series_item = converted.ReferencedSeriesSequence[0]
    assert (
        ref_series_item.SeriesInstanceUID ==
        legacy_datasets[0].SeriesInstanceUID
    )
    assert (
        len(ref_series_item.ReferencedInstanceSequence) ==
        len(legacy_datasets)
    )
    ref_instances = {
        item.ReferencedSOPInstanceUID
        for item in ref_series_item.ReferencedInstanceSequence
    }
    legacy_instances = {ds.SOPInstanceUID for ds in legacy_datasets}
    assert ref_instances == legacy_instances
    assert (
        "StudiesContainingOtherReferencedInstancesSequence"
        not in converted
    )


@pytest.mark.parametrize(
    'keyword',
    ['PatientID', 'PatientSex', 'AccessionNumber'],
)
def test_missing_type_2_attribute(modality: Modality, keyword: str):
    """Type 2 attributes missing from source images should be set to None."""
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    for ds in legacy_datasets:
        delattr(ds, keyword)

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert keyword in converted
    assert getattr(converted, keyword) is None


def test_require_volume_spacings():
    """If require_volume is set, should fail if spacings are irregular."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.CT, 1, True, True
    )

    # Make spacings irregular
    legacy_datasets[0].ImagePositionPatient = [100.0, 0.0, 0.0]

    # Usually conversion process should succeed
    LegacyConvertedEnhancedCTImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    # Conversion process should fail with require_volume=True
    msg = (
        "Legacy datasets are not a regularly-spaced set of frames."
    )
    with pytest.raises(ValueError, match=msg):
        LegacyConvertedEnhancedCTImage(
            legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
            require_volume=True,
        )


def test_require_volume_orientation():
    """If require_volume is set, should fail if orientations vary."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.CT, 1, True, True
    )

    # Make orientations different
    legacy_datasets[0].ImageOrientationPatient = [
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0
    ]

    # Usually conversion process should succeed
    LegacyConvertedEnhancedCTImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    # Conversion process should fail with require_volume=True
    msg = (
        "The legacy instances do not represent a regularly-spaced "
        "volume because the values of the following attribute(s) "
        "are not consistent across instances: ImageOrientationPatient."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        LegacyConvertedEnhancedCTImage(
            legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
            require_volume=True,
        )


@pytest.mark.parametrize(
    'keyword',
    ['PatientID', 'PatientSex', 'AccessionNumber'],
)
def test_empty_type_2_attribute(modality: Modality, keyword: str):
    """Type 2 attributes emtpy in source images should be set to None."""
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    for ds in legacy_datasets:
        setattr(ds, keyword, None)

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert keyword in converted
    assert getattr(converted, keyword) is None


@pytest.mark.parametrize(
    ['keyword', 'replacement_value'],
    [
        ('PatientSex', 'M'),
        ('AccessionNumber', 'A123')
    ],
)
def test_inconsistent_type_2_attribute(
    modality: Modality,
    keyword: str,
    replacement_value: Any,
):
    """Type 2 attributes inconsistent between source images should be set to
    None.

    """
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    setattr(legacy_datasets[-1], keyword, replacement_value)

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert keyword in converted
    assert getattr(converted, keyword) is None


def test_optional_module(modality: Modality) -> None:
    """Check that presence of the required attributes triggers inclusion of
    optional module.

    """
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    # Including just one of the two required attributes should cause the entire
    # module to be omitted
    for ds in legacy_datasets:
        ds.ClinicalTrialSponsorName = 'Sponsor'  # type 1
        ds.ClinicalTrialProtocolName = 'Protocol Name'  # type 2
        ds.ClinicalTrialSiteName = 'Site Name'  # type 2
        ds.IssuerOfClinicalTrialSubjectID = "Issuer"  # type 3

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert 'ClinicalTrialSponsorName' not in converted
    assert 'ClinicalTrialProtocolID' not in converted
    assert 'ClinicalTrialProtocolName' not in converted
    assert 'IssuerOfClinicalTrialSubjectID' not in converted

    # Now additionally including the missing required attribute should trigger
    # the whole module to be included
    for ds in legacy_datasets:
        ds.ClinicalTrialProtocolID = 'Protocol ID'  # type 1

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert converted.ClinicalTrialSponsorName == 'Sponsor'
    assert converted.ClinicalTrialProtocolName == 'Protocol Name'
    assert converted.ClinicalTrialSiteName == 'Site Name'
    assert converted.ClinicalTrialSiteID is None  # type 2, missing
    assert converted.IssuerOfClinicalTrialSubjectID == "Issuer"


@pytest.mark.parametrize(
    'keyword',
    [
        'BurnedInAnnotation',
        'RecognizableVisualFeatures',
        'LossyImageCompression'
    ],
)
def test_aggregated_attributes(modality: Modality, keyword: str) -> None:
    """Check values of attributes that are aggregated across the source
    instances.

    """
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    # If all are NO, output will be NO
    for ds in legacy_datasets:
        setattr(ds, keyword, 'NO')

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert converted.get(keyword) == 'NO'

    # If one is YES, output is YES
    setattr(legacy_datasets[0], keyword, 'YES')

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )
    assert converted.get(keyword) == 'YES'


def test_average_compression_ratio(modality: Modality):
    """If present, compression ratio should be averaged over frames."""
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(2)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    legacy_datasets[0].LossyImageCompressionRatio = 2.0
    legacy_datasets[1].LossyImageCompressionRatio = 4.0

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    # Should be average of source values
    assert converted.LossyImageCompressionRatio == 3.0


def test_largest_smallest_pixel_value(modality: Modality):
    """Test that largest/smallest pixel values are min/maxed over instances."""
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    for i, ds in enumerate(legacy_datasets):
        ds.SmallestImagePixelValue = i
        ds.LargestImagePixelValue = i

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    # Should be average of source values
    assert converted.SmallestImagePixelValue == 0
    assert converted.LargestImagePixelValue == 4


def test_aggregated_image_type():
    """Test that the top-level ImageType is correctly aggregated over
    frames.

    """
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.CT, 1, True, True
    )

    # First image has different image type
    legacy_datasets[0].ImageType = ['DERIVED', 'PRIMARY', 'LOCALIZER']

    converted = LegacyConvertedEnhancedCTImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    # Top-level ImageType should be aggregated version of source values
    assert converted.ImageType == ['MIXED', 'PRIMARY', 'AXIAL', 'NONE']

    # Frame type should have moved to the per-frame functional groups
    sfgs = converted.SharedFunctionalGroupsSequence[0]
    frame_type_seq_kw = 'CTImageFrameTypeSequence'
    assert not hasattr(sfgs, frame_type_seq_kw)
    for pffg in converted.PerFrameFunctionalGroupsSequence:
        assert hasattr(pffg, frame_type_seq_kw)


def test_default_series_description():
    """Test that the default series description is added."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.CT, 1, True, True
    )

    converted = LegacyConvertedEnhancedCTImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert (
        converted.SeriesDescription ==
        f"{legacy_datasets[0].SeriesDescription} (enhanced conversion)"
    )


def test_datetimes_with_timezones():
    """Test with timezones in legacy datetimes."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.CT, 1, True, True
    )

    dt = DT(datetime.now(tz=ZoneInfo("America/New_York")))

    for ds in legacy_datasets:
        ds.AcquisitionDateTime = dt

    converted = LegacyConvertedEnhancedCTImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    assert converted.AcquisitionDateTime.tzinfo is not None


def test_ambiguous_vr_coercion():
    """Test legacy images with ambiguous VR issues."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.CT, 1, True, True
    )

    # Simulate incorrectly populated PixelPaddingValue (VR of US instead of SS
    # when PixelRepresentation=1)
    for ds in legacy_datasets:
        ds.PixelRepresentation = 1
        ds["PixelPaddingValue"] = DataElement(
            tag_for_keyword("PixelPaddingValue"),
            VR="US",
            # Invalid for PixelRepresentation=1,
            # would be -2000 if correct VR of SS were used
            value=63536,
        )

    converted = LegacyConvertedEnhancedCTImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    converted = write_and_read_dataset(converted)

    assert converted.PixelPaddingValue == -2000


def test_from_big_endian():
    """Test big-endian legacy images become little endian."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.CT, 1, True, True
    )

    for ds in legacy_datasets:
        ds.file_meta.TransferSyntaxUID = ExplicitVRBigEndian

        # Re-encode frame as big endian
        dtype = ds.pixel_array.dtype
        be_dtype = f">{dtype.kind}{dtype.itemsize}"
        ds.PixelData = ds.pixel_array.astype(be_dtype).flatten().tobytes()

    converted = LegacyConvertedEnhancedCTImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    converted = write_and_read_dataset(converted)

    # Converted should be little endian again
    assert converted.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian

    # Pixel data should have been converted back to little endian
    for f, ds in enumerate(legacy_datasets):
        assert np.array_equal(converted.pixel_array[f], ds.pixel_array)


@pytest.mark.parametrize(
    ['missing_keyword', 'sequence_name'],
    [
        ('ImageOrientationPatient', 'PlaneOrientationSequence'),
        ('ImagePositionPatient', 'PlanePositionSequence'),
        ('PixelSpacing', 'PixelMeasuresSequence'),
        ('SliceThickness', 'PixelMeasuresSequence'),
    ],
)
def test_missing_required_attribute_for_mandatory_group(
    modality: Modality,
    missing_keyword: str,
    sequence_name: str,
):
    """Missing required attribute for mandatory functional group should raise
    an error.

    """
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    for ds in legacy_datasets:
        delattr(ds, missing_keyword)

    msg = (
        "Cannot determine value for required attribute "
        f"'{missing_keyword}' in the '{sequence_name}'."
    )
    with pytest.raises(AttributeError, match=msg):
        LegacyConvertedClass(
            legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
        )

    # Check it works okay with strict = False
    if missing_keyword == "SliceThickness":
        # (other keywords will trigger other errors if missing)
        LegacyConvertedClass(
            legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
            strict=False,
        )


def test_private_attributes():
    """Test private attributes are correctly copied."""
    # Some test files that have a lot of private attributes
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136', read=True),
        get_testdata_file('dicomdirtests/77654033/CT2/17196', read=True),
        get_testdata_file('dicomdirtests/77654033/CT2/17166', read=True),
    ]

    converted = LegacyConvertedEnhancedCTImage(
        ct_files,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    unassigned_shared_seq = (
        converted
        .SharedFunctionalGroupsSequence[0]
        .UnassignedSharedConvertedAttributesSequence[0]
    )
    unassigned_perframe_seq = (
        converted
        .PerFrameFunctionalGroupsSequence[0]
        .UnassignedPerFrameConvertedAttributesSequence[0]
    )

    # Check some arbitrary tags and their private creators
    assert unassigned_shared_seq[0x0045_1006].value == 'OUT OF GANTRY'
    assert unassigned_shared_seq[0x0045_0010].value == 'GEMS_HELIOS_01'

    assert unassigned_perframe_seq[0x0019_1024].value == 131.0
    assert unassigned_perframe_seq[0x0019_0010].value == 'GEMS_ACQU_01'


def test_skip_private_attributes():
    """Test private attributes are correctly copied."""
    # Some test files that have a lot of private attributes
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136', read=True),
        get_testdata_file('dicomdirtests/77654033/CT2/17196', read=True),
        get_testdata_file('dicomdirtests/77654033/CT2/17166', read=True),
    ]

    converted = LegacyConvertedEnhancedCTImage(
        ct_files,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        skip_private_attributes=True,
    )

    unassigned_shared_seq = (
        converted
        .SharedFunctionalGroupsSequence[0]
        .UnassignedSharedConvertedAttributesSequence[0]
    )
    unassigned_perframe_seq = (
        converted
        .PerFrameFunctionalGroupsSequence[0]
        .UnassignedPerFrameConvertedAttributesSequence[0]
    )

    # Check some arbitrary tags and their private creators
    assert 0x0045_1006 not in unassigned_shared_seq
    assert 0x0045_0010 not in unassigned_shared_seq

    assert 0x0019_1024 not in unassigned_perframe_seq
    assert 0x0019_0010 not in unassigned_perframe_seq


def test_optional_functional_group():
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.PT, 1, True, True
    )

    irradiation_uid = UID()

    for ds in legacy_datasets:
        ds.IrradiationEventUID = irradiation_uid

    converted = LegacyConvertedEnhancedPETImage(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    sfgs = converted.SharedFunctionalGroupsSequence[0]
    irradiation_seq = sfgs.IrradiationEventIdentificationSequence[0]

    assert irradiation_seq.IrradiationEventUID == irradiation_uid


def test_extended_offset_table(modality):
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        use_extended_offset_table=True,
    )

    pixel_array = converted.pixel_array
    assert pixel_array.shape == (5, 2, 2)


def test_empty_dataset_list(modality: Modality) -> None:
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]

    msg = 'At least one legacy dataset must be provided.'
    with pytest.raises(ValueError, match=msg):
        LegacyConvertedClass(
            [],
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
        )


@pytest.mark.parametrize(
    ['converter_class', 'dataset_modality'],
    [
        (LegacyConvertedEnhancedCTImage, Modality.PT),
        (LegacyConvertedEnhancedMRImage, Modality.CT),
        (LegacyConvertedEnhancedPETImage, Modality.MR),

    ],
)
def test_wrong_modality(
    converter_class: type,
    dataset_modality: Modality,
) -> None:
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        dataset_modality, 1, True, True
    )
    msg = 'Wrong modality for conversion of legacy [A-Z]+ images.'
    with pytest.raises(ValueError, match=msg):
        converter_class(
            legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
        )


def test_wrong_sop_class_uid(modality: Modality) -> None:
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    for ddss in legacy_datasets:
        ddss.SOPClassUID = '1.2.3.4.5.6.7.8.9'

    msg = 'Wrong SOP class for conversion of legacy [A-Z]+ images.'
    with pytest.raises(ValueError, match=msg):
        LegacyConvertedClass(
            legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
        )


@pytest.mark.parametrize(
    ['inconsistent_keyword', 'replacement_value'],
    [
        ('StudyInstanceUID', '1.2.3.4.5.6.7.8.9'),
        ('FrameOfReferenceUID', '1.2.3.4.5.6.7.8.9'),
        ('PatientID', 'M12345'),
        ('Manufacturer', 'Foo Corp.'),
        ('Rows', 1024),
        ('BitsStored', 12),
        ('SamplesPerPixel', 3),
        ('PlanarConfiguration', 1),
    ],
)
def test_mixed_instances(
    modality: Modality,
    inconsistent_keyword: str,
    replacement_value: Any,
) -> None:
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    setattr(legacy_datasets[0], inconsistent_keyword, replacement_value)

    msg = (
        "The legacy instances provided are not a valid source for a "
        "legacy conversion because the presence and/or values of the "
        "following attribute(s) is not consistent across instances: "
        f"{inconsistent_keyword}."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        LegacyConvertedClass(
            legacy_datasets=legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
        )


def test_mixed_transfer_syntax(modality: Modality):
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    legacy_datasets[0].file_meta.TransferSyntaxUID = JPEG2000Lossless
    msg = (
        'Legacy instances have inconsistent transfer syntaxes.'
    )
    with pytest.raises(ValueError, match=msg):
        LegacyConvertedClass(
            legacy_datasets=legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
        )


def test_mixed_series(modality: Modality):
    # Combining two series is valid if other attributes are consistent
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets_1 = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )
    legacy_datasets_2 = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    # Ensure the same frame of reference UID
    for ds in legacy_datasets_2:
        ds.FrameOfReferenceUID = legacy_datasets_1[0].FrameOfReferenceUID

    all_dataset = legacy_datasets_1 + legacy_datasets_2

    LegacyConvertedClass(
        legacy_datasets=all_dataset,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )


def test_unsorted():
    # Combining two series is valid if other attributes are consistent
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.PT, 1, True, True
    )

    # Reverse the order
    reversed_legacy_datasets = legacy_datasets[::-1]

    converted_sorted = LegacyConvertedEnhancedPETImage(
        legacy_datasets=reversed_legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        sort=True,
    )

    # Frames should be sorted by instance number
    for ds, pffg in zip(
        legacy_datasets,
        converted_sorted.PerFrameFunctionalGroupsSequence,
    ):
        frame_source_uid = (
            pffg.ConversionSourceAttributesSequence[0]
            .ReferencedSOPInstanceUID
        )
        assert frame_source_uid == ds.SOPInstanceUID

    converted_unsorted = LegacyConvertedEnhancedPETImage(
        legacy_datasets=reversed_legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        sort=False,
    )

    # Frames should be in same order they were passed
    for ds, pffg in zip(
        reversed_legacy_datasets,
        converted_unsorted.PerFrameFunctionalGroupsSequence,
    ):
        frame_source_uid = (
            pffg.ConversionSourceAttributesSequence[0]
            .ReferencedSOPInstanceUID
        )
        assert frame_source_uid == ds.SOPInstanceUID


def test_body_part_mapping(modality: Modality):
    """Test that BodyPartExamined is correctly mapped to a coded
    AnatomicRegionSequence.

    """
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    body_part_examined = 'ANTECUBITALV'
    expected_scheme_designator = 'SCT'
    expected_code_value = '128553008'
    expected_code_meaning = 'Antecubital vein'

    for dcm in legacy_datasets:
        dcm.BodyPartExamined = body_part_examined

    converted = LegacyConvertedClass(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    sfgs = converted.SharedFunctionalGroupsSequence[0]
    assert hasattr(sfgs, 'FrameAnatomySequence')
    fr_an_seq = sfgs.FrameAnatomySequence[0]
    assert (
        fr_an_seq.AnatomicRegionSequence[0].CodeValue ==
        expected_code_value
    )
    assert (
        fr_an_seq.AnatomicRegionSequence[0].CodeMeaning ==
        expected_code_meaning
    )
    assert (
        fr_an_seq.AnatomicRegionSequence[0].CodingSchemeDesignator ==
        expected_scheme_designator
    )
    assert fr_an_seq.FrameLaterality == 'U'  # unimodal (default)

    # The body part examined should not be used
    assert "BodyPartExamined" not in converted


def test_laterality_from_region_modifier(modality: Modality):
    """Test that the laterality is correctly inferred from the
    AnatomicRegionSequence if present in the source file.

    """
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    region_item = Dataset()
    region_item.CodeMeaning = 'Kidney'
    region_item.CodeValue = '64033007'
    region_item.CodingSchemeDesignator = 'SCT'
    modifier_item = Dataset()
    modifier_item.CodeMeaning = 'Left'
    modifier_item.CodeValue = '7771000'
    modifier_item.CodingSchemeDesignator = 'SCT'
    region_item.AnatomicRegionModifierSequence = [modifier_item]

    for dcm in legacy_datasets:
        dcm.AnatomicRegionSequence = [deepcopy(region_item)]

    converted = LegacyConvertedClass(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    sfgs = converted.SharedFunctionalGroupsSequence[0]
    assert hasattr(sfgs, 'FrameAnatomySequence')
    fr_an_seq = sfgs.FrameAnatomySequence[0]
    assert (
        fr_an_seq.AnatomicRegionSequence[0].CodeValue ==
        '64033007'
    )
    assert (
        fr_an_seq.AnatomicRegionSequence[0].CodeMeaning ==
        'Kidney'
    )
    assert (
        fr_an_seq.AnatomicRegionSequence[0].CodingSchemeDesignator ==
        'SCT'
    )
    assert fr_an_seq.FrameLaterality == 'L'  # inferred from modifier


def test_laterality_from_structure_modifier(modality: Modality):
    """Test that the laterality is correctly inferred from the
    PrimaryAnatomicStructureSequence if present in the source file.

    """
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    region_item = Dataset()
    region_item.CodeMeaning = 'Thorax'
    region_item.CodeValue = '816094009'
    region_item.CodingSchemeDesignator = 'SCT'
    structure_item = Dataset()
    structure_item.CodeMeaning = 'Lung'
    structure_item.CodeValue = '39607008'
    structure_item.CodingSchemeDesignator = 'SCT'
    modifier_item = Dataset()
    modifier_item.CodeMeaning = 'Bilateral'
    modifier_item.CodeValue = '51440002'
    modifier_item.CodingSchemeDesignator = 'SCT'
    structure_item.PrimaryAnatomicStructureModifierSequence = [
        modifier_item
    ]

    for dcm in legacy_datasets:
        dcm.AnatomicRegionSequence = [deepcopy(region_item)]
        dcm.PrimaryAnatomicStructureSequence = [deepcopy(structure_item)]

    converted = LegacyConvertedClass(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    sfgs = converted.SharedFunctionalGroupsSequence[0]
    assert hasattr(sfgs, 'FrameAnatomySequence')
    fr_an_seq = sfgs.FrameAnatomySequence[0]
    assert (
        fr_an_seq.PrimaryAnatomicStructureSequence[0].CodeValue ==
        '39607008'
    )
    assert (
        fr_an_seq.PrimaryAnatomicStructureSequence[0].CodeMeaning ==
        'Lung'
    )
    assert (
        fr_an_seq
        .PrimaryAnatomicStructureSequence[0]
        .CodingSchemeDesignator ==
        'SCT'
    )
    assert fr_an_seq.FrameLaterality == 'B'  # inferred from modifier


def test_extra_referenced_series():
    """Test that any Referenced Series in the legacy dataset are copied over."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.MR, 1, True, True
    )

    ref_series_uid = UID()
    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = ref_series_uid
    ref_instance_uid = UID()
    ref_sop_class_uid = EnhancedMRImageStorage
    ref_instance_item = Dataset()
    ref_instance_item.ReferencedSOPInstanceUID = ref_instance_uid
    ref_instance_item.ReferencedSOPClassUID = ref_sop_class_uid
    ref_series_item.ReferencedInstanceSequence = [ref_instance_item]

    for ds in legacy_datasets:
        ds.ReferencedSeriesSequence = [ref_series_item]

    converted = LegacyConvertedEnhancedMRImage(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    # Should be 2 seres here: the legacy series and series
    # referenced in the legacy dataset
    assert len(converted.ReferencedSeriesSequence) == 2
    # First comes from the legacy series
    assert (
        converted.ReferencedSeriesSequence[0].SeriesInstanceUID ==
        legacy_datasets[0].SeriesInstanceUID
    )
    # The other comes from the series referenced in the legacy series
    ref_series_item = converted.ReferencedSeriesSequence[1]
    assert (
        ref_series_item.SeriesInstanceUID == ref_series_uid
    )
    ref_ins_seq = ref_series_item.ReferencedInstanceSequence
    assert len(ref_ins_seq) == 1
    assert ref_ins_seq[0].ReferencedSOPInstanceUID == ref_instance_uid
    assert ref_ins_seq[0].ReferencedSOPClassUID == ref_sop_class_uid


def test_studies_containing_other_referenced_instances():
    """Test that Studies Containing Other referenced in the legacy
    dataset are copied over.

    """
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.MR, 1, True, True
    )

    ref_study_uid = UID()
    ref_study_item = Dataset()
    ref_study_item.StudyInstanceUID = ref_study_uid
    ref_series_uid = UID()
    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = ref_series_uid
    ref_instance_uid = UID()
    ref_sop_class_uid = EnhancedMRImageStorage
    ref_instance_item = Dataset()
    ref_instance_item.ReferencedSOPInstanceUID = ref_instance_uid
    ref_instance_item.ReferencedSOPClassUID = ref_sop_class_uid
    ref_series_item.ReferencedInstanceSequence = [ref_instance_item]
    ref_study_item.ReferencedSeriesSequence = [ref_series_item]

    for ds in legacy_datasets:
        ds.StudiesContainingOtherReferencedInstancesSequence = [
            ref_study_item
        ]

    converted = LegacyConvertedEnhancedMRImage(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    # Should be 2 seres here: the legacy series and series
    # referenced in the legacy dataset
    ref_studies_seq = (
        converted
        .StudiesContainingOtherReferencedInstancesSequence
    )
    assert len(ref_studies_seq) == 1
    ref_series_seq = ref_studies_seq[0].ReferencedSeriesSequence
    assert ref_studies_seq[0].StudyInstanceUID == ref_study_uid
    ref_series_item = ref_series_seq[0]
    assert (
        ref_series_item.SeriesInstanceUID == ref_series_uid
    )
    ref_ins_seq = ref_series_item.ReferencedInstanceSequence
    assert len(ref_ins_seq) == 1
    assert (
        ref_ins_seq[0].ReferencedSOPInstanceUID ==
        ref_instance_uid
    )
    assert (
        ref_ins_seq[0].ReferencedSOPClassUID ==
        ref_sop_class_uid
    )


@pytest.mark.parametrize(
    "transfer_syntax_uid",
    [
        JPEG2000,
        JPEG2000Lossless,
        JPEGLSLossless,
        RLELossless,
    ]
)
def test_encapsulated(transfer_syntax_uid: UID):
    """Test encapsulated legacy datasets."""
    if transfer_syntax_uid in (JPEG2000, JPEG2000Lossless):
        pytest.importorskip('openjpeg')
    elif transfer_syntax_uid in (JPEGLSLossless, JPEGBaseline8Bit):
        pytest.importorskip('libjpeg')
    data_generator = DicomGenerator(5, row=32, col=32)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.MR, 1, True, True, color=True, bytes_per_voxel=1,
    )

    compress_kwargs = {}
    if transfer_syntax_uid == JPEG2000:
        compress_kwargs["j2k_psnr"] = [10.0]

    for ds in legacy_datasets:
        ds.compress(transfer_syntax_uid, **compress_kwargs)

    converted = LegacyConvertedEnhancedMRImage(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
    )

    expected_pixels = np.stack(
        [ds.pixel_array for ds in legacy_datasets]
    )
    assert np.array_equal(converted.pixel_array, expected_pixels)


@pytest.mark.parametrize("workers", [0, 1, 4])
def test_transcode(workers: int):
    """Test transcoding frames to a new transfer syntax"""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.MR, 1, True, True,
    )

    converted = LegacyConvertedEnhancedMRImage(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        transfer_syntax_uid=RLELossless,
        workers=workers,
    )

    expected_pixels = np.stack(
        [ds.pixel_array for ds in legacy_datasets]
    )
    assert np.array_equal(converted.pixel_array, expected_pixels)


@pytest.mark.parametrize(
    ["transfer_syntax_uid", "photometric_interpretation"],
    [
        (JPEG2000, 'YBR_ICT'),
        (JPEG2000Lossless, 'YBR_RCT'),
        (JPEGLSLossless, 'RGB'),
        (RLELossless, 'RGB'),
        (JPEGBaseline8Bit, 'YBR_FULL_422'),
    ]
)
@pytest.mark.parametrize("workers", [0, 1, 4])
def test_encode_rgb(
    workers: int,
    transfer_syntax_uid: str,
    photometric_interpretation: str,
):
    """Test transcoding frames to a new transfer syntax"""
    if transfer_syntax_uid in (JPEG2000, JPEG2000Lossless):
        pytest.importorskip('openjpeg')
    elif transfer_syntax_uid in (JPEGLSLossless, JPEGBaseline8Bit):
        pytest.importorskip('libjpeg')

    data_generator = DicomGenerator(5, row=32, col=32)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.MR, 1, True, True, color=True, bytes_per_voxel=1,
    )

    converted = LegacyConvertedEnhancedMRImage(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        transfer_syntax_uid=transfer_syntax_uid,
        workers=workers,
    )

    expected_pixels = np.stack(
        [ds.pixel_array for ds in legacy_datasets]
    )
    if transfer_syntax_uid in (JPEG2000, JPEGBaseline8Bit):
        # Lossy transfer syntax: don't expect equality
        assert converted.pixel_array.shape == expected_pixels.shape
    else:
        assert np.array_equal(converted.pixel_array, expected_pixels)

    assert (
        converted.PhotometricInterpretation ==
        photometric_interpretation
    )

    if transfer_syntax_uid == JPEG2000:
        assert converted.LossyImageCompression == "01"
        assert converted.LossyImageCompressionMethod == "ISO_15444_1"
        assert "LossyImageCompressionRatio" in converted
    elif transfer_syntax_uid == JPEGBaseline8Bit:
        assert converted.LossyImageCompression == "01"
        assert converted.LossyImageCompressionMethod == "ISO_10918_1"
        assert "LossyImageCompressionRatio" in converted
    else:
        for kw in [
            "LossyImageCompression",
            "LossyImageCompressionMethod",
            "LossyImageCompressionRatio",
        ]:
            assert kw not in converted


@pytest.mark.parametrize(
    ["transfer_syntax_uid", "photometric_interpretation"],
    [
        (JPEG2000, 'YBR_ICT'),
        (JPEG2000Lossless, 'YBR_RCT'),
        (JPEGLSLossless, 'RGB'),
        (RLELossless, 'RGB'),
        (ExplicitVRLittleEndian, 'RGB'),
        (ImplicitVRLittleEndian, 'RGB'),
        (JPEGBaseline8Bit, 'YBR_FULL_422'),
    ]
)
@pytest.mark.parametrize("workers", [0, 1, 4])
def test_transcode_rgb(
    workers: int,
    transfer_syntax_uid: str,
    photometric_interpretation: str,
):
    """Test transcoding frames to a new transfer syntax
    from an existing lossy method."""
    if transfer_syntax_uid in (JPEG2000, JPEG2000Lossless):
        pytest.importorskip('openjpeg')
    elif transfer_syntax_uid in (JPEGLSLossless, JPEGBaseline8Bit):
        pytest.importorskip('libjpeg')

    data_generator = DicomGenerator(5, row=32, col=32)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.MR, 1, True, True, color=True, bytes_per_voxel=1,
    )

    # Input datasets are lossy compressed
    for ds in legacy_datasets:
        ds.compress(JPEGLSNearLossless)

    converted = LegacyConvertedEnhancedMRImage(
        legacy_datasets=legacy_datasets,
        series_instance_uid=UID(),
        series_number=1,
        sop_instance_uid=UID(),
        instance_number=1,
        transfer_syntax_uid=transfer_syntax_uid,
        workers=workers,
    )

    expected_pixels = np.stack(
        [ds.pixel_array for ds in legacy_datasets]
    )
    if transfer_syntax_uid in (JPEG2000, JPEGBaseline8Bit):
        # Lossy transfer syntax: don't expect equality
        assert converted.pixel_array.shape == expected_pixels.shape
    else:
        assert np.array_equal(converted.pixel_array, expected_pixels)

    assert (
        converted.PhotometricInterpretation ==
        photometric_interpretation
    )

    # Since input is lossy compressed, this should be set regardless of
    # whether the new transfer syntax is lossy
    assert converted.LossyImageCompression == "01"

    if transfer_syntax_uid == JPEG2000:
        assert converted.LossyImageCompressionMethod == [
            "ISO_14495_1",  # Original
            "ISO_15444_1",  # New
        ]
        assert isinstance(
            converted.LossyImageCompressionRatio,
            MultiValue,
        )
    elif transfer_syntax_uid == JPEGBaseline8Bit:
        assert converted.LossyImageCompression == "01"
        assert converted.LossyImageCompressionMethod == [
            "ISO_14495_1",  # Original
            "ISO_10918_1",  # New
        ]
        assert isinstance(
            converted.LossyImageCompressionRatio,
            MultiValue,
        )
    else:
        # New transfer syntax is lossy, so only information on the original
        # should be present
        assert converted.LossyImageCompressionMethod == "ISO_14495_1"
        assert not isinstance(
            converted.LossyImageCompressionRatio,
            MultiValue,
        )


def test_transcode_custom_workers():
    """Test transcoding frames with custom workers."""
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        Modality.MR, 1, True, True
    )

    with ProcessPoolExecutor(5) as workers:
        converted = LegacyConvertedEnhancedMRImage(
            legacy_datasets=legacy_datasets,
            series_instance_uid=UID(),
            series_number=1,
            sop_instance_uid=UID(),
            instance_number=1,
            transfer_syntax_uid=RLELossless,
            workers=workers,
        )

    expected_pixels = np.stack(
        [ds.pixel_array for ds in legacy_datasets]
    )
    assert np.array_equal(converted.pixel_array, expected_pixels)


def test_from_dataset(modality: Modality) -> None:
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    output_series_uid = UID()
    output_sop_uid = UID()
    output_series_number = 23
    output_instance_number = 2
    series_description = 'Converted Series'

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=output_series_uid,
        sop_instance_uid=output_sop_uid,
        series_number=output_series_number,
        instance_number=output_instance_number,
        series_description=series_description,
    )

    reread = LegacyConvertedClass.from_dataset(
        write_and_read_dataset(converted)
    )

    assert isinstance(reread, LegacyConvertedClass)
    expected_pixels = np.stack(
        [ds.pixel_array for ds in legacy_datasets]
    )
    assert np.array_equal(converted.pixel_array, expected_pixels)


def test_from_dataset_wrong_modality(modality: Modality) -> None:
    # Using the wrong modality should raise an error
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]

    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    output_series_uid = UID()
    output_sop_uid = UID()
    output_series_number = 23
    output_instance_number = 2
    series_description = 'Converted Series'

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=output_series_uid,
        sop_instance_uid=output_sop_uid,
        series_number=output_series_number,
        instance_number=output_instance_number,
        series_description=series_description,
    )

    # Choose wrong modality
    WrongLegacyConvertedClass = {
        Modality.PT: LegacyConvertedEnhancedCTImage,
        Modality.CT: LegacyConvertedEnhancedMRImage,
        Modality.MR: LegacyConvertedEnhancedPETImage,
    }[modality]

    msg = (
        "Dataset is not a Legacy Converted Enhanced "
        f"{WrongLegacyConvertedClass._MODALITY_NAME} image."
    )

    with pytest.raises(ValueError, match=msg):
        WrongLegacyConvertedClass.from_dataset(
            write_and_read_dataset(converted)
        )


@pytest.mark.parametrize("lazy", [True, False])
def test_read_function(modality: Modality, lazy: bool) -> None:
    LegacyConvertedClass = MODALITY_CLASS_MAP[modality]
    data_generator = DicomGenerator(5)
    legacy_datasets = data_generator.generate_mixed_framesets(
        modality, 1, True, True
    )

    output_series_uid = UID()
    output_sop_uid = UID()
    output_series_number = 23
    output_instance_number = 2
    series_description = 'Converted Series'

    converted = LegacyConvertedClass(
        legacy_datasets,
        series_instance_uid=output_series_uid,
        sop_instance_uid=output_sop_uid,
        series_number=output_series_number,
        instance_number=output_instance_number,
        series_description=series_description,
    )

    read_fn = {
        Modality.PT: lcepetimread,
        Modality.CT: lcectimread,
        Modality.MR: lcemrimread,
    }[modality]

    with BytesIO() as fp:
        converted.save_as(fp)
        fp.seek(0)
        reread = read_fn(fp, lazy_frame_retrieval=lazy)

    assert isinstance(reread, LegacyConvertedClass)
    assert ('PixelData' in reread) == (not lazy)
    assert converted.pixel_array.shape == (5, 2, 2)
