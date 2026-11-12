import numpy as np
import tempfile
import pytest
import itertools

from pathlib import Path
from typing import Sequence
from highdicom import Volume
from highdicom.spatial import (
    get_closest_patient_orientation,
    convert_affine_to_convention
)
from highdicom._dependency_utils import import_optional_dependency
from .utils import (
    DCM_QA_MPRAGE,
    DCM_QA_ME,
    DCM_QA_PDT2,
    read_multiframe_ct_volume,
    read_ct_series_volume,
    read_github_zip_volume,
    read_github_series_volume,
    urldownload_with_retry
)

try:
    nib = import_optional_dependency('nibabel', feature='nib tests')

except Exception:
    pytest.skip("Optional dependency not available", allow_module_level=True)


def read_github_nib(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = Path(temp_dir) / Path(url).name
        urldownload_with_retry(url, filename)

        nifti_proxy = nib.load(filename, keep_file_open=False)
        nifti = nib.Nifti1Image(
            np.asarray(nifti_proxy.dataobj).copy(),
            nifti_proxy.affine.copy(),
            header=nifti_proxy.header
        )

    return nifti


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
    nifti = vol.to_nibabel()

    assert np.allclose(vol.get_affine('RAS'), nifti.affine, atol=1e-4)
    assert (vol.array == nifti.dataobj).all()

    nib_roundtrip = Volume.from_nibabel(nifti)

    assert np.allclose(vol.affine, nib_roundtrip.affine, atol=1e-4)
    assert (vol.array == nib_roundtrip.array).all()


@pytest.mark.parametrize(
    'dtype,image_class',
    itertools.product(
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
            np.longdouble,
            np.bool_
        ],
        [
            'Nifti1Image',
            'Nifti2Image',
        ]
    )
)
def test_dtype_nifti(dtype: np.dtype, image_class: str):
    if (
        dtype == np.longdouble and
        np.dtype(np.longdouble) == np.dtype(np.float64)
    ):
        return

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

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == np.uint8
        assert (nib_roundtrip.array == volume.array).all()

    elif dtype == np.float16:
        f16 = np.finfo(np.float16)
        volume.array = rng.uniform(f16.min, f16.max, size=size).astype(dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.float32)}.'
            )
        ):

            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.float32
        assert (nib_roundtrip.array == volume.array).all()

    elif dtype == np.longdouble:
        f64 = np.finfo(np.float64)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * f64.max

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float64)},'
                ' precision may be lost.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.float64
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = 1.1 * np.longdouble(f64.max)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float64)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        volume.array[(0, 0, 0)] = 1.1 * np.longdouble(f64.min)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float64)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif np.issubdtype(dtype, np.integer):
        ib = np.iinfo(dtype)
        volume.array = rng.integers(ib.min, ib.max, size=size, dtype=dtype)

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()

    else:
        fb = np.finfo(dtype)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * fb.max

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()


@pytest.mark.parametrize(
    'dtype,image_class',
    itertools.product(
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
            np.longdouble,
            np.bool_
        ],
        [
            'MGHImage'
        ]
    )
)
def test_dtype_mgh(dtype: np.dtype, image_class: str):
    if (
        dtype == np.longdouble and
        np.dtype(np.longdouble) == np.dtype(np.float64)
    ):
        return

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

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == np.uint8
        assert (nib_roundtrip.array == volume.array).all()

    elif dtype == np.uint32:
        ui32 = np.iinfo(np.uint32)
        i32 = np.iinfo(np.int32)
        volume.array = rng.integers(0, i32.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = ui32.max
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.int32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.uint64:
        ui64 = np.iinfo(np.uint64)
        i32 = np.iinfo(np.int32)
        volume.array = rng.integers(0, i32.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = ui64.max
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.int32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.int8:
        i8 = np.iinfo(np.int8)
        volume.array = rng.integers(i8.min, i8.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int16)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int16
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

    elif dtype == np.int64:
        i64 = np.iinfo(np.int64)
        i32 = np.iinfo(np.int32)
        volume.array = rng.integers(0, i32.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = i64.max
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.int32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.float16:
        f16 = np.finfo(np.float16)
        volume.array = rng.uniform(f16.min, f16.max, size=size).astype(dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.float32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.float32
        assert (nib_roundtrip.array == volume.array).all()

    elif dtype == np.float64:
        f32 = np.finfo(np.float32)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * f32.max

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float32)},'
                ' precision may be lost.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.float32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = 1.1 * np.float64(f32.max)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        volume.array[(0, 0, 0)] = 1.1 * np.float64(f32.min)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.longdouble:
        f32 = np.finfo(np.float32)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * f32.max

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float32)},'
                ' precision may be lost.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.float32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = 1.1 * np.longdouble(f32.max)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        volume.array[(0, 0, 0)] = 1.1 * np.longdouble(f32.min)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif np.issubdtype(dtype, np.integer):
        ib = np.iinfo(dtype)
        volume.array = rng.integers(ib.min, ib.max, size=size, dtype=dtype)

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()

    else:
        fb = np.finfo(dtype)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * fb.max

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()


@pytest.mark.parametrize(
    'dtype,image_class',
    itertools.product(
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
            np.longdouble,
            np.bool_
        ],
        [
            'Minc1Image',
            'Minc2Image',
        ]
    )
)
def test_dtype_minc(dtype: np.dtype, image_class: str):
    if (
        dtype == np.longdouble and
        np.dtype(np.longdouble) == np.dtype(np.float64)
    ):
        return

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

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == np.uint8
        assert (nib_roundtrip.array == volume.array).all()

    elif np.issubdtype(dtype, np.integer):
        ib = np.iinfo(dtype)
        volume.array = rng.integers(ib.min, ib.max, size=size, dtype=dtype)

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()

    else:
        fb = np.finfo(dtype)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * fb.max

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()


@pytest.mark.parametrize(
    'dtype,image_class',
    itertools.product(
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
            np.longdouble,
            np.bool_
        ],
        [
            'AnalyzeImage'
        ]
    )
)
def test_dtype_analyze(dtype: np.dtype, image_class: str):
    if (
        dtype == np.longdouble and
        np.dtype(np.longdouble) == np.dtype(np.float64)
    ):
        return

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

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == np.uint8
        assert (nib_roundtrip.array == volume.array).all()

    elif dtype == np.uint16:
        ui16 = np.iinfo(np.uint16)
        i16 = np.iinfo(np.int16)
        volume.array = rng.integers(0, i16.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int16)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int16
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = ui16.max
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.int16)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.uint32:
        ui32 = np.iinfo(np.uint32)
        i32 = np.iinfo(np.int32)
        volume.array = rng.integers(0, i32.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = ui32.max
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.int32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.uint64:
        ui64 = np.iinfo(np.uint64)
        i32 = np.iinfo(np.int32)
        volume.array = rng.integers(0, i32.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = ui64.max
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.int32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.int8:
        i8 = np.iinfo(np.int8)
        volume.array = rng.integers(i8.min, i8.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int16)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int16
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

    elif dtype == np.int64:
        i64 = np.iinfo(np.int64)
        i32 = np.iinfo(np.int32)
        volume.array = rng.integers(0, i32.max, size=size, dtype=dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.int32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.int32
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = i64.max
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.int32)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif dtype == np.float16:
        f16 = np.finfo(np.float16)
        volume.array = rng.uniform(f16.min, f16.max, size=size).astype(dtype)

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Safely casting to {np.dtype(np.float32)}.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.float32
        assert (nib_roundtrip.array == volume.array).all()

    elif dtype == np.longdouble:
        f64 = np.finfo(np.float64)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * f64.max

        with pytest.warns(
            UserWarning,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float64)},'
                ' precision may be lost.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

        assert nib_roundtrip.dtype == np.float64
        assert np.allclose(nib_roundtrip.array, volume.array, atol=1e-4)

        volume.array[(0, 0, 0)] = 1.1 * np.longdouble(f64.max)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float64)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )
        volume.array[(0, 0, 0)] = 1.1 * np.longdouble(f64.min)
        with pytest.raises(
            ValueError,
            match=(
                f'NiBabel\'s {getattr(nib, image_class)} class does not support'
                f' {np.dtype(dtype)}. Casting to {np.dtype(np.float64)}'
                ' is not possible.'
            )
        ):
            nib_roundtrip = Volume.from_nibabel(
                volume.to_nibabel(image_class=image_class)
            )

    elif np.issubdtype(dtype, np.integer):
        ib = np.iinfo(dtype)
        volume.array = rng.integers(ib.min, ib.max, size=size, dtype=dtype)

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()

    else:
        fb = np.finfo(dtype)
        array = rng.random(size).astype(dtype)
        volume.array = (2 * array - 1) * 0.9 * fb.max

        nib_roundtrip = Volume.from_nibabel(
            volume.to_nibabel(image_class=image_class)
        )
        assert nib_roundtrip.dtype == volume.dtype
        assert (nib_roundtrip.array == volume.array).all()


@pytest.mark.parametrize(
    'zip_url,nifti_url',
    [
        (
            f'{DCM_QA_MPRAGE}/In/2_t1_mp2rage_sag_p3_32.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_2_t1_mp2rage_sag_p3_32_INV1.nii.gz'
        ),
        (
            f'{DCM_QA_MPRAGE}/In/5_HCP_T1.zip',
            f'{DCM_QA_MPRAGE}/Ref/Si_5_HCP_T1.nii.gz'
        )
    ]
)
def test_nifti_equivalence_zip(zip_url: str, nifti_url: str):
    vol, series = read_github_zip_volume(zip_url)
    nifti = read_github_nib(nifti_url)

    orientation = get_closest_patient_orientation(
        convert_affine_to_convention(
            nifti.affine,
            from_reference_convention='RAS',
            to_reference_convention='LPS'
        )
    )
    oriented_vol = vol.to_patient_orientation(orientation)

    assert np.allclose(oriented_vol.get_affine('RAS'), nifti.affine, atol=1e-4)
    assert (oriented_vol.array == nifti.dataobj).all()


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
                f'{DCM_QA_PDT2}/In/Siemens/VE11/{i:04d}.dcm'
                for i in range(1, 36)
            ],
            f'{DCM_QA_PDT2}/Ref/Siemens_pd+t2_tse_sag_ISO_1.8mm_3_e1.nii'
        )
    ]
)
def test_nifti_equivalence_series(dcm_urls: Sequence[str], nifti_url: str):
    vol, series = read_github_series_volume(dcm_urls)
    nifti = read_github_nib(nifti_url)

    orientation = get_closest_patient_orientation(
        convert_affine_to_convention(
            nifti.affine,
            from_reference_convention='RAS',
            to_reference_convention='LPS'
        )
    )
    oriented_vol = vol.to_patient_orientation(orientation)

    assert np.allclose(oriented_vol.get_affine('RAS'), nifti.affine, atol=1e-4)
    assert (oriented_vol.array == nifti.dataobj).all()


def test_multichannel_volume():
    array = np.zeros((10, 10, 10, 2))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        channels={'OpticalPathIdentifier': ['path1', 'path2']},
        coordinate_system="PATIENT",
    )

    with pytest.raises(
        ValueError,
        match=(
            'NiBabel conversion does not currently support'
            ' volumes with multiple channels.'
        )
    ):
        volume.to_nibabel()

    array = np.zeros((10, 10, 10, 2))
    nifti = nib.Nifti1Image(array, np.eye(4))

    with pytest.raises(
        ValueError,
        match=(
            'NiBabel conversion does not currently support'
            ' volumes with multiple channels.'
        )
    ):
        Volume.from_nibabel(nifti)
