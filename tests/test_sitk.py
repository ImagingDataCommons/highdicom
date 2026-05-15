import numpy as np
import pydicom
import tempfile
import zipfile
import urllib
import pytest

from pathlib import Path
from typing import Sequence
from pydicom.data import get_testdata_file
from highdicom import (
    Volume,
    get_volume_from_series,
    imread,
)
from highdicom.spatial import get_closest_patient_orientation
from highdicom._dependency_utils import import_optional_dependency

try:
    sitk = import_optional_dependency('SimpleITK', feature='sitk tests')

except Exception:
    pytest.skip("Optional dependency not available", allow_module_level=True)

DCM_QA_MPRAGE = 'https://github.com/neurolabusc/dcm_qa_mprage/raw/refs/heads/main'  # noqa: E501
DCM_QA_ME = 'https://github.com/neurolabusc/dcm_qa_me/raw/refs/heads/master'
DCM_QA_PDT2 = 'https://github.com/neurolabusc/dcm_qa_pdt2/raw/refs/heads/main'


def read_multiframe_ct_volume():
    im = imread(get_testdata_file('eCT_Supplemental.dcm'))
    return im.get_volume()


def read_ct_series_volume():
    ct_files = [
        get_testdata_file('dicomdirtests/77654033/CT2/17136'),
        get_testdata_file('dicomdirtests/77654033/CT2/17196'),
        get_testdata_file('dicomdirtests/77654033/CT2/17166'),
    ]
    ct_series = [pydicom.dcmread(f) for f in ct_files]
    return get_volume_from_series(ct_series)


def read_github_zip_volume(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        zipfilename = Path(temp_dir) / Path(url).name
        urllib.request.urlretrieve(url, zipfilename)

        with zipfile.ZipFile(zipfilename, 'r') as zf:
            zf.extractall(temp_dir)

        series = [pydicom.dcmread(f) for f in Path(temp_dir).glob('**/*.dcm')]

    return get_volume_from_series(series), series


def read_github_series_volume(urls: Sequence[str]):
    series = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for url in urls:
            filename = Path(temp_dir) / Path(url).name
            urllib.request.urlretrieve(url, filename)

            series.append(pydicom.dcmread(filename))

    return get_volume_from_series(series), series


def read_github_sitk(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / Path(url).name
        urllib.request.urlretrieve(url, filename)

        sitk_im = sitk.ReadImage(filename)

    return sitk_im


@pytest.mark.parametrize(
    'vol',
    [
        # testdata_files
        read_multiframe_ct_volume(),
        read_ct_series_volume(),
        # different orientations
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('RAF'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('RAH'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('RPF'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('RPH'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('LAF'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('LAH'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('LPF'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('LPH'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('HLP'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('FPR'),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ).to_patient_orientation('HRP'),
        # isotropic
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[0.5, 0.5],
            spacing_between_slices=0.5,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[2.0, 2.0],
            spacing_between_slices=2.0,
            coordinate_system='PATIENT'
        ),
        # anisotropic
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[0.5, 0.5],
            spacing_between_slices=2.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[2.0, 0.5],
            spacing_between_slices=0.5,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[0.5, 2.0],
            spacing_between_slices=0.5,
            coordinate_system='PATIENT'
        ),
        # non-square
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 32, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (64, 128, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        # single-slice
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 1)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 1, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (1, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        # random position offset
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[84.40363858, 105.04467386, 143.73326388],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[-21.03512292, 35.19549233, -184.42393696],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[-197.36060235, 86.22231644, -14.79874245],
            image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        # random orientation
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[
                -0.9267662161157189,
                0.32283606313442387,
                -0.1920449348627007,
                -0.3751482085550474,
                -0.7693372329674889,
                0.5170919101937937
            ],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[
                -0.16411694786392106,
                -0.7887835415036736,
                0.5923564400567902,
                0.9859501024515502,
                -0.11222667947664411,
                0.12372375636644917
            ],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(0, 100, (16, 16, 16)),
            image_position=[0.0, 0.0, 0.0],
            image_orientation=[
                -0.4161787354389772,
                0.440351232633788,
                0.7955413578729377,
                -0.2771687765466385,
                -0.8947097563255662,
                0.350245515665062
            ],
            pixel_spacing=[1.0, 1.0],
            spacing_between_slices=1.0,
            coordinate_system='PATIENT'
        ),
        # entirely random
        Volume.from_attributes(
            array=np.random.randint(113, 257, (192, 249, 84)),
            image_position=[156.03935104, -57.61106994, -108.37601079],
            image_orientation=[
                -0.3056572521325831,
                -0.9434667295206645,
                -0.12823484123411946,
                0.9440777770580763,
                -0.3177984972478506,
                0.08787073467366081
            ],
            pixel_spacing=[3.34201481, 2.35548103],
            spacing_between_slices=2.82618053,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(180, 214, (96, 121, 50)),
            image_position=[93.43769804, -184.44672839, -153.64700033],
            image_orientation=[
                -0.7034764816836532,
                0.05445328347346446,
                -0.7086294374614612,
                -0.24703727011677554,
                -0.9536256538386763,
                0.1719613314498601
            ],
            pixel_spacing=[2.84370598, 0.69898499],
            spacing_between_slices=0.57265037,
            coordinate_system='PATIENT'
        ),
        Volume.from_attributes(
            array=np.random.randint(81, 214, (34, 123, 59)),
            image_position=[-252.23051789, 146.90528128, 84.40363858],
            image_orientation=[
                -0.7374895773326877,
                0.5716640328411087,
                0.35959610242811746,
                -0.35709313761636013,
                -0.7820074118766409,
                0.510831575802926
            ],
            pixel_spacing=[3.52582689, 3.90364516],
            spacing_between_slices=2.29457888,
            coordinate_system='PATIENT'
        ),

    ]
)
def test_roundtrip(vol: Volume):
    sitk_im = vol.to_sitk()

    assert np.allclose(vol.position, sitk_im.GetOrigin(), atol=1e-4)
    assert np.allclose(vol.spacing, sitk_im.GetSpacing(), atol=1e-4)
    assert np.allclose(
        vol.direction.flatten(),
        sitk_im.GetDirection(),
        atol=1e-4
    )
    assert (
        vol.array == sitk.GetArrayFromImage(sitk_im).transpose(2, 1, 0)
    ).all()

    sitk_roundtrip = Volume.from_sitk(sitk_im)

    assert np.allclose(vol.affine, sitk_roundtrip.affine, atol=1e-4)
    assert (vol.array == sitk_roundtrip.array).all()


@pytest.mark.parametrize(
    'dtype',
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
        np.float128,
        np.bool_
    ]
)
def test_dtype_sitk(dtype: np.dtype):
    rng = np.random.default_rng()
    size = (10, 10, 10)

    volume = Volume.from_attributes(
        array=np.zeros(size),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=1.0,
        coordinate_system='PATIENT',
    )

    if dtype == np.bool_:
        volume.array = np.round(rng.random(size=size)).astype(dtype)

        sitk_roundtrip = Volume.from_sitk(volume.to_sitk())
        assert sitk_roundtrip.dtype == np.uint8
        assert (sitk_roundtrip.array == volume.array).all()

    elif dtype == np.float16:
        f16 = np.finfo(np.float16)
        volume.array = rng.uniform(f16.min, f16.max, size=size).astype(dtype)

        with pytest.warns(
            UserWarning,
            match=(
                'SimpleITK does not support float16 data.'
                ' Safely casting to float32.'
            )
        ):
            sitk_roundtrip = Volume.from_sitk(volume.to_sitk())

        assert sitk_roundtrip.dtype == np.float32
        assert (sitk_roundtrip.array == volume.array).all()

    elif dtype == np.float128:
        f64 = np.finfo(np.float64)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * f64.max

        with pytest.warns(
            UserWarning,
            match=(
                'SimpleITK does not support float128 data.'
                ' Casting to float64, precision may be lost.'
            )
        ):
            sitk_roundtrip = Volume.from_sitk(volume.to_sitk())

        assert sitk_roundtrip.dtype == np.float64
        assert np.allclose(sitk_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = 1.1 * np.float128(np.finfo(np.float64).max)
        with pytest.raises(
            ValueError,
            match=(
                'SimpleITK does not support float128 data.'
                ' Casting to float64 is not possible.'
            )
        ):
            sitk_roundtrip = Volume.from_sitk(volume.to_sitk())

        volume.array[(0, 0, 0)] = 1.1 * np.float128(np.finfo(np.float64).min)
        with pytest.raises(
            ValueError,
            match=(
                'SimpleITK does not support float128 data.'
                ' Casting to float64 is not possible.'
            )
        ):
            sitk_roundtrip = Volume.from_sitk(volume.to_sitk())

    elif np.issubdtype(dtype, np.integer):
        ib = np.iinfo(dtype)
        volume.array = rng.integers(ib.min, ib.max, size=size, dtype=dtype)

        sitk_roundtrip = Volume.from_sitk(volume.to_sitk())
        assert sitk_roundtrip.dtype == volume.dtype
        assert (sitk_roundtrip.array == volume.array).all()

    else:
        fb = np.finfo(dtype)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * fb.max

        sitk_roundtrip = Volume.from_sitk(volume.to_sitk())
        assert sitk_roundtrip.dtype == volume.dtype
        assert (sitk_roundtrip.array == volume.array).all()


@pytest.mark.parametrize(
    'zip_url,nifti_url',
    [
        (
            f'{DCM_QA_MPRAGE}/In/2_t1_mp2rage_sag_p3_32.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_2_t1_mp2rage_sag_p3_32_INV1.nii.gz'
        ),
        (
            f'{DCM_QA_MPRAGE}/In/3_t1_mp2rage_sag_p3_32.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_3_t1_mp2rage_sag_p3_32_INV2.nii.gz'
        ),
        (
            f'{DCM_QA_MPRAGE}/In/4_t1_mp2rage_sag_p3_32.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_4_t1_mp2rage_sag_p3_32_UNI_Images.nii.gz'
        ),
        (
            f'{DCM_QA_MPRAGE}/In/5_HCP_T1.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_5_HCP_T1.nii.gz'
        ),
        (
            f'{DCM_QA_MPRAGE}/In/6_T1_mprage_ns_sag_p2.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_6_T1_mprage_ns_sag_p2.nii.gz'
        ),
        (
            f'{DCM_QA_MPRAGE}/In/8_T1_memprage_rms.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_8_T1_memprage_rms_RMS.nii.gz'
        ),
    ]
)
def test_nifti_equivalence_zip(zip_url: str, nifti_url: str):
    vol, series = read_github_zip_volume(zip_url)
    sitk_im = read_github_sitk(nifti_url)

    orientation = get_closest_patient_orientation(
        np.reshape(sitk_im.GetDirection(), (3, 3))
    )
    oriented_vol = vol.to_patient_orientation(orientation)

    assert np.allclose(oriented_vol.position, sitk_im.GetOrigin(), atol=1e-4)
    assert np.allclose(oriented_vol.spacing, sitk_im.GetSpacing(), atol=1e-4)
    assert np.allclose(
        oriented_vol.direction.flatten(),
        sitk_im.GetDirection(),
        atol=1e-4
    )
    assert (
        oriented_vol.array == sitk.GetArrayFromImage(sitk_im).transpose(2, 1, 0)
    ).all()


@pytest.mark.parametrize(
    'dcm_urls,nifti_url',
    [
        (
            [
                f'{DCM_QA_ME}/In/2_me_FieldMap_GRE/{i:04d}.dcm'
                for i in range(1, 37)
            ],
            f'{DCM_QA_ME}/Ref/me_FieldMap_GRE_2_e1.nii'
        ),
        (
            [
                f'{DCM_QA_ME}/In/2_me_FieldMap_GRE/{i:04d}_e2.dcm'
                for i in range(1, 37)
            ],
            f'{DCM_QA_ME}/Ref/me_FieldMap_GRE_2_e2.nii'
        ),
        (
            [
                f'{DCM_QA_ME}/In/3_me_FieldMap_GRE/{i:04d}_e2_ph.dcm'
                for i in range(1, 37)
            ],
            f'{DCM_QA_ME}/Ref/me_FieldMap_GRE_3_e2_ph.nii'
        ),
        (
            [
                f'{DCM_QA_PDT2}/In/Siemens/VE11/{i:04d}.dcm'
                for i in range(1, 36)
            ],
            f'{DCM_QA_PDT2}/Ref/Siemens_pd+t2_tse_sag_ISO_1.8mm_3_e1.nii'
        ),
        (
            [
                f'{DCM_QA_PDT2}/In/Siemens/VE11/{i:04d}_e2.dcm'
                for i in range(36, 71)
            ],
            f'{DCM_QA_PDT2}/Ref/Siemens_pd+t2_tse_sag_ISO_1.8mm_3_e2.nii'
        ),
    ]
)
def test_nifti_equivalence_series(dcm_urls: Sequence[str], nifti_url: str):
    vol, series = read_github_series_volume(dcm_urls)
    sitk_im = read_github_sitk(nifti_url)

    orientation = get_closest_patient_orientation(
        np.reshape(sitk_im.GetDirection(), (3, 3))
    )
    oriented_vol = vol.to_patient_orientation(orientation)

    assert np.allclose(oriented_vol.position, sitk_im.GetOrigin(), atol=1e-4)
    assert np.allclose(oriented_vol.spacing, sitk_im.GetSpacing(), atol=1e-4)
    assert np.allclose(
        oriented_vol.direction.flatten(),
        sitk_im.GetDirection(),
        atol=1e-4
    )
    assert (
        oriented_vol.array == sitk.GetArrayFromImage(sitk_im).transpose(2, 1, 0)
    ).all()
