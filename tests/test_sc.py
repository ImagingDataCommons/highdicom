from pathlib import Path
from io import BytesIO

import numpy as np
import pytest
from pydicom.encaps import generate_fragmented_frames
from pydicom.filereader import dcmread
from pydicom.uid import (
    RLELossless,
    JPEGBaseline8Bit,
    JPEG2000,
    JPEG2000Lossless,
    JPEGLSLossless,
    JPEGLSNearLossless,
)
from pydicom.valuerep import DA, TM

from highdicom import SpecimenDescription
from highdicom.sc import SCImage
from highdicom import UID
from highdicom.frame import decode_frame

from .utils import write_and_read_dataset


def generate_frame(
    shape: tuple[int, ...],
    dtype: type,
    pixel_representation: int,
    bits_stored: int,
):
    rng = np.random.default_rng(0)
    if pixel_representation == 1:
        low = -(2 ** (bits_stored - 1))
        high = 2 ** (bits_stored - 1)
    else:
        low = 0
        high = 2 ** bits_stored - 1

    frame = rng.integers(
        low, high, shape, dtype=dtype,
    )
    # ensure the extremes are exercised
    frame[0, 0] = low
    frame[0, 1] = high - 1

    return frame


def get_array_after_writing(instance):
    # Write DICOM object to buffer, read it again and reconstruct the mask
    with BytesIO() as fp:
        instance.save_as(fp)
        fp.seek(0)
        ds = dcmread(fp)

    decoded_frame_arrays = [
        decode_frame(
            value,
            transfer_syntax_uid=ds.file_meta.TransferSyntaxUID,
            rows=ds.Rows,
            columns=ds.Columns,
            samples_per_pixel=ds.SamplesPerPixel,
            bits_allocated=ds.BitsAllocated,
            bits_stored=ds.BitsStored,
            photometric_interpretation=ds.PhotometricInterpretation,
            pixel_representation=ds.PixelRepresentation,
            planar_configuration=getattr(ds, 'PlanarConfiguration', None)
        )
        for (value, ) in generate_fragmented_frames(instance.PixelData)
    ]
    if len(decoded_frame_arrays) > 1:
        return np.stack(decoded_frame_arrays)
    return decoded_frame_arrays[0]


def test_construct_rgb_patient():
    study_uid = UID()
    series_uid = UID()
    sop_uid = UID()
    photometric_interpretation = 'RGB'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (10, 10, 3), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        sop_instance_uid=sop_uid,
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        laterality='L',
        pixel_spacing=(0.5, 0.5),
    )
    assert instance.BitsAllocated == 8
    assert instance.BitsStored == 8
    assert instance.SamplesPerPixel == 3
    assert instance.PlanarConfiguration == 0
    assert instance.PhotometricInterpretation == photometric_interpretation
    assert instance.StudyInstanceUID == study_uid
    assert instance.SeriesInstanceUID == series_uid
    assert instance.SOPInstanceUID == sop_uid
    assert instance.SeriesNumber == 34
    assert instance.InstanceNumber == 7
    assert instance.Manufacturer == 'ABC'
    assert instance.Laterality == 'L'
    assert instance.PatientOrientation == ['A', 'R']
    assert instance.AccessionNumber is None
    assert instance.PatientName is None
    assert instance.PatientSex is None
    assert instance.StudyTime is None
    assert instance.StudyTime is None
    assert instance.PixelSpacing == [0.5, 0.5]
    assert instance.PixelData == frame.tobytes()
    with pytest.raises(AttributeError):
        instance.ContainerIdentifier  # noqa: B018
        instance.SpecimenDescriptionSequence  # noqa: B018
        instance.ContainerTypeCodeSequence  # noqa: B018
        instance.IssuerOfTheContainerIdentifierSequence  # noqa: B018


def test_construct_rgb_patient_missing_parameter():
    frame = generate_frame(
        (10, 10, 3), np.uint8, 0, 8
    )
    with pytest.raises(TypeError):
        photometric_interpretation = 'RGB'
        coordinate_system = 'PATIENT'
        SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            coordinate_system=coordinate_system,
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            sop_instance_uid=UID(),
            series_number=34,
            instance_number=7,
            manufacturer='ABC',
        )


def test_construct_rgb_slide_single_specimen():
    photometric_interpretation = 'RGB'
    coordinate_system = 'SLIDE'
    study_uid = UID()
    series_uid = UID()
    sop_uid = UID()
    specimen_uid = UID()
    specimen_description = SpecimenDescription(
        specimen_id='specimen id',
        specimen_uid=specimen_uid,
    )
    frame = generate_frame(
        (10, 10, 3), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        sop_instance_uid=sop_uid,
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        container_identifier='container id',
        specimen_descriptions=[specimen_description]
    )
    assert instance.BitsAllocated == 8
    assert instance.BitsStored == 8
    assert instance.SamplesPerPixel == 3
    assert instance.PlanarConfiguration == 0
    assert instance.PhotometricInterpretation == photometric_interpretation
    assert instance.StudyInstanceUID == study_uid
    assert instance.SeriesInstanceUID == series_uid
    assert instance.SOPInstanceUID == sop_uid
    assert instance.SeriesNumber == 34
    assert instance.InstanceNumber == 7
    assert instance.Manufacturer == 'ABC'
    assert instance.ContainerIdentifier == 'container id'
    assert len(instance.ContainerTypeCodeSequence) == 1
    assert len(instance.IssuerOfTheContainerIdentifierSequence) == 0
    assert len(instance.SpecimenDescriptionSequence) == 1
    specimen_item = instance.SpecimenDescriptionSequence[0]
    assert specimen_item.SpecimenIdentifier == 'specimen id'
    assert specimen_item.SpecimenUID == specimen_uid
    assert instance.AccessionNumber is None
    assert instance.PatientName is None
    assert instance.PatientSex is None
    assert instance.StudyTime is None
    assert instance.StudyTime is None
    assert instance.PixelData == frame.tobytes()
    with pytest.raises(AttributeError):
        instance.Laterality  # noqa: B018
        instance.PatientOrientation  # noqa: B018


def test_construct_rgb_slide_single_specimen_missing_parameter():
    photometric_interpretation = 'RGB'
    coordinate_system = 'SLIDE'
    specimen_description = SpecimenDescription(
        specimen_id='specimen id',
        specimen_uid=UID()
    )
    frame = generate_frame(
        (10, 10, 3), np.uint8, 0, 8
    )
    with pytest.raises(TypeError):
        SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            coordinate_system=coordinate_system,
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            sop_instance_uid=UID(),
            series_number=34,
            instance_number=7,
            manufacturer='ABC',
            specimen_descriptions=[specimen_description]
        )


def test_construct_rgb_slide_single_specimen_missing_parameter_1():
    photometric_interpretation = 'RGB'
    coordinate_system = 'SLIDE'
    frame = generate_frame(
        (10, 10, 3), np.uint8, 0, 8
    )
    with pytest.raises(TypeError):
        SCImage(
            pixel_array=frame,
            photometric_interpretation=photometric_interpretation,
            coordinate_system=coordinate_system,
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            sop_instance_uid=UID(),
            series_number=34,
            instance_number=7,
            manufacturer='ABC',
            container_identifier='container id',
        )


def test_construct_monochrome_patient():
    study_uid = UID()
    series_uid = UID()
    sop_uid = UID()
    photometric_interpretation = 'MONOCHROME2'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (10, 10), np.uint16, 0, 16
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        sop_instance_uid=sop_uid,
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R')
    )
    assert instance.BitsAllocated == 16
    assert instance.BitsStored == 16
    assert instance.SamplesPerPixel == 1
    assert instance.PhotometricInterpretation == photometric_interpretation
    assert instance.StudyInstanceUID == study_uid
    assert instance.SeriesInstanceUID == series_uid
    assert instance.SOPInstanceUID == sop_uid
    assert instance.SeriesNumber == 34
    assert instance.InstanceNumber == 7
    assert instance.Manufacturer == 'ABC'
    assert instance.PatientOrientation == ['A', 'R']
    assert instance.AccessionNumber is None
    assert instance.PatientName is None
    assert instance.PatientSex is None
    assert instance.StudyTime is None
    assert instance.StudyDate is None
    assert instance.PixelData == frame.tobytes()
    with pytest.raises(AttributeError):
        instance.ContainerIdentifier  # noqa: B018
    with pytest.raises(AttributeError):
        instance.SpecimenDescriptionSequence  # noqa: B018
    with pytest.raises(AttributeError):
        instance.ContainerTypeCodeSequence  # noqa: B018
    with pytest.raises(AttributeError):
        instance.IssuerOfTheContainerIdentifierSequence  # noqa: B018


def test_monochrome_rle():
    photometric_interpretation = 'MONOCHROME2'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=RLELossless
    )

    assert instance.file_meta.TransferSyntaxUID == RLELossless

    assert np.array_equal(
        get_array_after_writing(instance),
        frame
    )


def test_rgb_rle():
    photometric_interpretation = 'RGB'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12, 3), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=RLELossless
    )

    assert instance.file_meta.TransferSyntaxUID == RLELossless

    assert np.array_equal(
        get_array_after_writing(instance),
        frame
    )


def test_monochrome_jpeg_baseline():
    pytest.importorskip("libjpeg")
    photometric_interpretation = 'MONOCHROME2'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEGBaseline8Bit
    )

    assert instance.file_meta.TransferSyntaxUID == JPEGBaseline8Bit

    assert np.allclose(
        get_array_after_writing(instance),
        frame,
        atol=10  # tolerance for lossy compression
    )


def test_rgb_jpeg_baseline():
    pytest.importorskip("libjpeg")
    photometric_interpretation = 'YBR_FULL_422'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12, 3), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEGBaseline8Bit
    )

    assert instance.file_meta.TransferSyntaxUID == JPEGBaseline8Bit


def test_monochrome_jpeg2000lossless():
    pytest.importorskip("openjpeg")
    photometric_interpretation = 'MONOCHROME2'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (32, 32), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEG2000Lossless
    )

    assert instance.file_meta.TransferSyntaxUID == JPEG2000Lossless

    assert np.array_equal(
        get_array_after_writing(instance),
        frame
    )


def test_monochrome_jpeg2000():
    pytest.importorskip("openjpeg")
    photometric_interpretation = 'MONOCHROME2'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (32, 32), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEG2000
    )

    assert instance.file_meta.TransferSyntaxUID == JPEG2000

    assert np.allclose(
        get_array_after_writing(instance),
        frame,
        atol=2
    )


def test_rgb_jpeg2000():
    pytest.importorskip("openjpeg")
    photometric_interpretation = 'YBR_RCT'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (32, 32, 3), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEG2000Lossless
    )

    assert instance.file_meta.TransferSyntaxUID == JPEG2000Lossless

    assert np.array_equal(
        get_array_after_writing(instance),
        frame
    )


def test_monochrome_jpegls():
    pytest.importorskip("libjpeg")
    photometric_interpretation = 'MONOCHROME2'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12), np.uint8, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEGLSLossless
    )

    assert instance.file_meta.TransferSyntaxUID == JPEGLSLossless

    assert np.array_equal(
        get_array_after_writing(instance),
        frame
    )


def test_monochrome_jpegls_near_lossless():
    pytest.importorskip("libjpeg")
    photometric_interpretation = 'MONOCHROME2'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12), np.uint16, 0, 16
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEGLSNearLossless
    )

    assert instance.file_meta.TransferSyntaxUID == JPEGLSNearLossless

    assert np.allclose(
        get_array_after_writing(instance),
        frame,
        atol=2
    )


def test_rgb_jpegls():
    pytest.importorskip("libjpeg")
    photometric_interpretation = 'RGB'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12, 3), np.uint16, 0, 8
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        transfer_syntax_uid=JPEGLSLossless
    )

    assert instance.file_meta.TransferSyntaxUID == JPEGLSLossless

    assert np.array_equal(
        get_array_after_writing(instance),
        frame
    )


def test_construct_rgb_from_ref_dataset():
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data')
    series_uid = UID()
    sop_uid = UID()

    ref_dataset = dcmread(
        str(data_dir.joinpath('test_files', 'ct_image.dcm'))
    )
    photometric_interpretation = 'RGB'
    coordinate_system = 'PATIENT'
    frame = generate_frame(
        (12, 12, 3), np.uint8, 0, 8
    )
    instance = SCImage.from_ref_dataset(
        ref_dataset=ref_dataset,
        pixel_array=frame,
        photometric_interpretation=photometric_interpretation,
        coordinate_system=coordinate_system,
        series_instance_uid=series_uid,
        sop_instance_uid=sop_uid,
        series_number=34,
        instance_number=7,
        manufacturer='ABC',
        patient_orientation=('A', 'R'),
        laterality='L'
    )
    assert instance.BitsAllocated == 8
    assert instance.BitsStored == 8
    assert instance.SamplesPerPixel == 3
    assert instance.PlanarConfiguration == 0
    assert instance.PhotometricInterpretation == photometric_interpretation
    assert instance.StudyInstanceUID == ref_dataset.StudyInstanceUID
    assert instance.SeriesInstanceUID == series_uid
    assert instance.SOPInstanceUID == sop_uid
    assert instance.SeriesNumber == 34
    assert instance.InstanceNumber == 7
    assert instance.Manufacturer == 'ABC'
    assert instance.Laterality == 'L'
    assert instance.PatientOrientation == ['A', 'R']
    assert instance.AccessionNumber == ref_dataset.AccessionNumber
    assert instance.PatientName == ref_dataset.PatientName
    assert instance.PatientSex == ref_dataset.PatientSex
    assert instance.StudyTime == TM(ref_dataset.StudyTime)
    assert instance.StudyDate == DA(ref_dataset.StudyDate)
    assert instance.StudyID == ref_dataset.StudyID
    assert instance.PixelData == frame.tobytes()


@pytest.mark.parametrize(
    'dtype,pixel_representation,bits_allocated',
    [
        (np.bool, 0, 1),
        (np.uint8, 0, 8),
        (np.uint16, 0, 16),
        (np.uint32, 0, 32),
        (np.int8, 1, 8),
        (np.int16, 1, 16),
        (np.int32, 1, 32),
    ]
)
def test_valid_monochrome_datatypes(
    dtype,
    pixel_representation,
    bits_allocated,
):
    frame = generate_frame(
        shape=(64, 64),
        dtype=dtype,
        pixel_representation=pixel_representation,
        bits_stored=bits_allocated,
    )

    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation='MONOCHROME2',
        coordinate_system='PATIENT',
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=1,
        instance_number=1,
        manufacturer='Manufacturer',
        patient_orientation=('A', 'R'),
    )
    assert instance.BitsAllocated == bits_allocated
    assert instance.BitsStored == bits_allocated
    assert instance.PixelRepresentation == pixel_representation

    reread = write_and_read_dataset(instance)
    read_frame = reread.pixel_array

    assert np.array_equal(read_frame, frame)


@pytest.mark.parametrize(
    'dtype,pixel_representation,bits_allocated',
    [
        (np.uint64, 0, 64),
        (np.int64, 1, 64),
    ]
)
def test_invalid_monochrome_datatypes(
    dtype,
    pixel_representation,
    bits_allocated,
):
    frame = generate_frame(
        shape=(64, 64),
        dtype=dtype,
        pixel_representation=pixel_representation,
        bits_stored=bits_allocated,
    )

    match = 'Pixel array has an unexpected bit depth'
    with pytest.raises(TypeError, match=match):
        SCImage(
            pixel_array=frame,
            photometric_interpretation='MONOCHROME2',
            coordinate_system='PATIENT',
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            sop_instance_uid=UID(),
            series_number=1,
            instance_number=1,
            manufacturer='Manufacturer',
            patient_orientation=('A', 'R'),
        )


@pytest.mark.parametrize(
    'dtype,pixel_representation,bits_allocated',
    [
        (np.uint8, 0, 8),
        (np.uint16, 0, 16),
    ]
)
def test_valid_color_datatypes(
    dtype,
    pixel_representation,
    bits_allocated,
):
    frame = generate_frame(
        shape=(64, 64, 3),
        dtype=dtype,
        pixel_representation=pixel_representation,
        bits_stored=bits_allocated,
    )
    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation='RGB',
        coordinate_system='PATIENT',
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=1,
        instance_number=1,
        manufacturer='Manufacturer',
        patient_orientation=('A', 'R'),
    )
    assert instance.BitsAllocated == bits_allocated
    assert instance.BitsStored == bits_allocated
    assert instance.PixelRepresentation == pixel_representation

    reread = write_and_read_dataset(instance)
    read_frame = reread.pixel_array

    assert np.array_equal(read_frame, frame)


@pytest.mark.parametrize(
    'dtype,pixel_representation,bits_allocated',
    [
        (np.bool, 0, 1),
        (np.uint64, 0, 64),
        (np.int8, 1, 8),
        (np.int16, 1, 16),
        (np.int32, 1, 32),
        (np.int64, 1, 64),
    ]
)
def test_invalid_color_datatypes(
    dtype,
    pixel_representation,
    bits_allocated,
):
    frame = generate_frame(
        shape=(64, 64, 3),
        dtype=dtype,
        pixel_representation=pixel_representation,
        bits_stored=bits_allocated,
    )

    if frame.dtype.kind == 'u':
        match = 'Pixel array has an unexpected bit depth'
    else:
        match = (
            'Pixel array must have an unsigned integer datatype for color '
            'images'
        )

    with pytest.raises(TypeError, match=match):
        SCImage(
            pixel_array=frame,
            photometric_interpretation='RGB',
            coordinate_system='PATIENT',
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            sop_instance_uid=UID(),
            series_number=1,
            instance_number=1,
            manufacturer='Manufacturer',
            patient_orientation=('A', 'R'),
        )


@pytest.mark.parametrize(
    'dtype,bits_allocated,bits_stored,pixel_representation',
    [
        (np.uint8, 8, 4, 0),
        (np.uint16, 16, 12, 0),
        (np.uint16, 16, 16, 0),
        (np.int16, 16, 12, 1),
    ],
)
def test_bits_stored(
    bits_allocated: int,
    bits_stored: int,
    pixel_representation: int,
    dtype: type,
):
    frame = generate_frame(
        shape=(64, 64),
        dtype=dtype,
        pixel_representation=pixel_representation,
        bits_stored=bits_stored,
    )

    instance = SCImage(
        pixel_array=frame,
        photometric_interpretation='MONOCHROME2',
        coordinate_system='PATIENT',
        study_instance_uid=UID(),
        series_instance_uid=UID(),
        sop_instance_uid=UID(),
        series_number=1,
        instance_number=1,
        manufacturer='Manufacturer',
        patient_orientation=('A', 'R'),
        bits_stored=bits_stored,
    )

    assert instance.BitsAllocated == bits_allocated
    assert instance.BitsStored == bits_stored

    reread = write_and_read_dataset(instance)
    read_frame = reread.pixel_array

    assert np.array_equal(read_frame, frame)


@pytest.mark.parametrize(
    'dtype,bits_allocated,bits_stored,pixel_representation',
    [
        (np.uint8, 8, 4, 0),
        (np.uint16, 16, 12, 0),
        (np.int16, 16, 12, 1),
    ],
)
def test_invalid_bits_stored(
    bits_allocated: int,
    bits_stored: int,
    pixel_representation: int,
    dtype: type,
):
    frame = generate_frame(
        shape=(64, 64),
        dtype=dtype,
        pixel_representation=pixel_representation,
        bits_stored=bits_allocated,
    )

    match = (
        "Provided 'pixel_array' cannot be represented with "
        "the specified number of 'bits_stored'"
    )
    with pytest.raises(ValueError, match=match):
        SCImage(
            pixel_array=frame,
            photometric_interpretation='MONOCHROME2',
            coordinate_system='PATIENT',
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            sop_instance_uid=UID(),
            series_number=1,
            instance_number=1,
            manufacturer='Manufacturer',
            patient_orientation=('A', 'R'),
            bits_stored=bits_stored,
        )


def test_bits_stored_greater_than_allocated():
    match = (
        "'bits_stored' must be an integer no greater than the number of "
        "bits per pixel in the input array."
    )
    with pytest.raises(ValueError, match=match):
        SCImage(
            pixel_array=np.zeros((64, 64), dtype=np.uint8),
            photometric_interpretation='MONOCHROME2',
            coordinate_system='PATIENT',
            study_instance_uid=UID(),
            series_instance_uid=UID(),
            sop_instance_uid=UID(),
            series_number=1,
            instance_number=1,
            manufacturer='Manufacturer',
            patient_orientation=('A', 'R'),
            bits_stored=9,  # greater than bits_allocated (8)
        )
