import numpy as np
import tempfile
import pytest
import pydicom
import zipfile
import re

from pathlib import Path
from typing import Sequence
from highdicom import Volume, ChannelDescriptor, get_volume_from_series
from highdicom.spatial import (
    get_closest_patient_orientation,
    convert_affine_to_convention,
    _unsqueeze_affine
)
from highdicom._dependency_utils import import_optional_dependency
from highdicom.seg import segread
from .utils import (
    DCM_QA_MPRAGE,
    DCM_QA_ME,
    DCM_QA_PDT2,
    TEST_DATA,
    read_multiframe_ct_volume,
    read_ct_series_volume,
    urldownload_with_retry
)


try:
    monai = import_optional_dependency('monai', feature='monai tests')
    ImageStatsKeys = monai.utils.enums.ImageStatsKeys
    MetaKeys = monai.utils.enums.MetaKeys
    SpaceKeys = monai.utils.enums.SpaceKeys

except Exception:
    pytest.skip("Optional dependency not available", allow_module_level=True)


def read_github_zip_volume_and_metatensor(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        zipfilename = Path(temp_dir) / Path(url).name
        urldownload_with_retry(url, zipfilename)

        with zipfile.ZipFile(zipfilename, 'r') as zf:
            zf.extractall(temp_dir)

        series = [pydicom.dcmread(f) for f in Path(temp_dir).glob('**/*.dcm')]

        try:
            pydicom.config.enforce_valid_values = False
            metatensor = monai.transforms.LoadImage(reader="PydicomReader")(
                Path(str(zipfilename)[:-4])
            )

        finally:
            pydicom.config.enforce_valid_values = True

    return get_volume_from_series(series), series, metatensor


def read_github_series_volume_and_metatensor(urls: Sequence[str]):
    series = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for url in urls:
            filename = Path(temp_dir) / Path(url).name
            urldownload_with_retry(url, filename)

            series.append(pydicom.dcmread(filename))

        try:
            pydicom.config.enforce_valid_values = False
            metatensor = monai.transforms.LoadImage(
                reader="PydicomReader"
            )(temp_dir)

        finally:
            pydicom.config.enforce_valid_values = True

    return get_volume_from_series(series), series, metatensor


def get_equivalent_rotation_matrices(
    affine_3d: np.ndarray,
    squeeze_dim: int,
    angle: float,
):
    keep_dims = [i for i in range(3) if i != squeeze_dim]

    # 3D: Rodrigues rotation about plane normal
    linear_3d = affine_3d[:3, :3]
    u = linear_3d[:, keep_dims[0]]
    v = linear_3d[:, keep_dims[1]]

    n = np.cross(u, v)
    n /= np.linalg.norm(n)

    skew = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0],
    ])

    sin, cos = np.sin(angle), np.cos(angle)

    rotation_world = (
        np.eye(3) * cos +
        (1 - cos) * np.outer(n, n) +
        sin * skew
    )
    rotation_3d = np.linalg.inv(linear_3d) @ rotation_world @ linear_3d

    affine_rotation_3d = np.eye(4)
    affine_rotation_3d[:3, :3] = rotation_3d

    # 2D: plain 2x2 rotation, scaled by in-plane spacings
    spacing = np.array([np.linalg.norm(u), np.linalg.norm(v)])
    rotation_2d = (
        np.diag(1.0 / spacing)
        @ np.array([[cos, -sin], [sin, cos]])
        @ np.diag(spacing)
    )

    affine_rotation_2d = np.eye(3)
    affine_rotation_2d[:2, :2] = rotation_2d

    return affine_rotation_2d, affine_rotation_3d


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
    metatensor = vol.to_monai()

    assert np.allclose(vol.get_affine('RAS'), metatensor.affine, atol=1e-4)
    assert (vol.array == metatensor.numpy()).all()

    monai_roundtrip = Volume.from_monai(metatensor)

    assert np.allclose(vol.affine, monai_roundtrip.affine, atol=1e-4)
    assert (vol.array == monai_roundtrip.array).all()


@pytest.mark.parametrize(
    'zip_url',
    [
        f'{DCM_QA_MPRAGE}/In/2_t1_mp2rage_sag_p3_32.zip',
        f'{DCM_QA_MPRAGE}/In/5_HCP_T1.zip',
    ]
)
def test_metatensor_equivalence_zip(zip_url: str):
    vol, series, metatensor = read_github_zip_volume_and_metatensor(zip_url)

    orientation = get_closest_patient_orientation(
        convert_affine_to_convention(
            metatensor.affine.numpy(),
            from_reference_convention='RAS',
            to_reference_convention='LPS'
        )
    )
    oriented_vol = vol.to_patient_orientation(orientation)

    assert np.allclose(
        oriented_vol.get_affine('RAS'),
        metatensor.affine.numpy(),
        atol=1e-4
    )
    assert (oriented_vol.array == metatensor.numpy()).all()


@pytest.mark.parametrize(
    'dcm_urls',
    [
        [
            f'{DCM_QA_ME}/In/2_me_FieldMap_GRE/{i:04d}.dcm'
            for i in range(1, 37)
        ],
        [
            f'{DCM_QA_PDT2}/In/Siemens/VE11/{i:04d}.dcm'
            for i in range(1, 36)
        ],
    ]
)
def test_metatensor_equivalence_series(dcm_urls: Sequence[str]):
    vol, series, metatensor = read_github_series_volume_and_metatensor(dcm_urls)

    orientation = get_closest_patient_orientation(
        convert_affine_to_convention(
            metatensor.affine.numpy(),
            from_reference_convention='RAS',
            to_reference_convention='LPS'
        )
    )
    oriented_vol = vol.to_patient_orientation(orientation)

    assert np.allclose(
        oriented_vol.get_affine('RAS'),
        metatensor.affine.numpy(),
        atol=1e-4
    )
    assert (oriented_vol.array == metatensor.numpy()).all()


@pytest.mark.parametrize(
    'segfile,channel_first,convert_to_ras,spacing,spatial_shape,affine,sum',
    [
        [
            'seg_image_sm_control.dcm',
            True,
            True,
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-4, 0., -23.449374],
                      [0., 0., 4.99e-4, -25.691075],
                      [1., 0., 0., 1.01],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_sm_dots_tiled_full.dcm',
            False,
            True,
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-4, 0., -23.449873],
                      [0., 0., 4.99e-4, -25.691574],
                      [1., 0., 0., 0.],
                      [0., 0., 0., 1.]]),
            200
        ],
        [
            'seg_image_ct_true_fractional.dcm',
            True,
            False,
            (1.25, 0.488281, 0.488281),
            (3, 16, 16),
            np.array([[0., 0., 0.488281, -125.],
                      [0., 0.488281, 0., -128.100006],
                      [-1.25, 0., 0., 105.519997],
                      [0., 0., 0., 1.]]),
            326.02353
        ],
        [
            'seg_image_ct_binary_overlap.dcm',
            False,
            False,
            (1.25, 0.488281, 0.488281),
            (165, 16, 16),
            np.array([[0., 0., 0.488281, -125.],
                      [0., 0.488281, 0., -128.100006],
                      [-1.25, 0., 0., 105.519997],
                      [0., 0., 0., 1.]]),
            80
        ],
        [
            'seg_image_sm_numbers.dcm',
            True,
            True,
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-4, 0., -23.449374],
                      [0., 0., 4.99e-4, -25.691075],
                      [1., 0., 0., 1.01],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_ct_binary_fractional.dcm',
            True,
            True,
            (1.25, 0.488281, 0.488281),
            (3, 16, 16),
            np.array([[0., 0., -0.488281, 125.],
                      [0., -0.488281, 0., 128.100006],
                      [-1.25, 0., 0., 105.519997],
                      [0., 0., 0., 1.]]),
            638.0
        ],
        [
            'seg_image_ct_binary_single_frame.dcm',
            True,
            True,
            (5.0, 0.661468, 0.661468),
            (1, 128, 128),
            np.array([[0., 0., -0.661468, 158.135803],
                      [0., -0.661468, 0., 179.035797],
                      [-5., 0., 0., -75.699997],
                      [0., 0., 0., 1.]]),
            1832
        ],
        [
            'seg_image_sm_dots.dcm',
            True,
            True,
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-4, 0., -23.449374],
                      [0., 0., 4.99e-4, -25.691075],
                      [1., 0., 0., 1.01],
                      [0., 0., 0., 1.]]),
            200
        ],
        [
            'seg_image_sm_control_labelmap.dcm',
            True,
            True,
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-4, 0., -23.449873],
                      [0., 0., 4.99e-4, -25.691574],
                      [1., 0., 0., 0.],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_sm_control_labelmap_palette_color.dcm',
            True,
            True,
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-4, 0., -23.449873],
                      [0., 0., 4.99e-4, -25.691574],
                      [1., 0., 0., 0.],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_ct_binary.dcm',
            True,
            True,
            (1.25, 0.488281, 0.488281),
            (3, 16, 16),
            np.array([[0., 0., -0.488281, 125.],
                      [0., -0.488281, 0., 128.100006],
                      [-1.25, 0., 0., 105.519997],
                      [0., 0., 0., 1.]]),
            638
        ]
    ]
)
def test_segmentation(
    segfile,
    channel_first,
    convert_to_ras,
    spacing,
    spatial_shape,
    affine,
    sum
):
    seg = segread(TEST_DATA / segfile)
    vol = seg.get_volume()
    metatensor = vol.to_monai(
        convert_to_ras=convert_to_ras,
        ensure_channel_first=channel_first,
    )
    meta = metatensor.meta

    if convert_to_ras:
        space = SpaceKeys.RAS

    else:
        space = SpaceKeys.LPS

    assert meta[MetaKeys.SPACE] == space
    assert vol.spacing == meta[ImageStatsKeys.SPACING] == spacing
    assert (np.array(vol.spatial_shape) == meta[MetaKeys.SPATIAL_SHAPE]).all()
    assert (
        metatensor.shape[1:] == spatial_shape if channel_first
        else metatensor.shape[:3] == spatial_shape
    )
    assert np.allclose(vol.get_affine(space), affine, atol=1e-4)
    assert np.allclose(metatensor.affine, affine, atol=1e-4)
    assert np.allclose(meta[MetaKeys.ORIGINAL_AFFINE], affine, atol=1e-4)
    assert np.allclose(meta[MetaKeys.AFFINE], affine, atol=1e-4)
    assert meta[MetaKeys.ORIGINAL_CHANNEL_DIM] == -1
    assert vol.array.sum() == metatensor.numpy().sum() == sum


def test_channels():
    array = np.zeros((10, 10, 10, 1))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        channels={'OpticalPathIdentifier': ['path1']},
        coordinate_system="PATIENT",
    )

    metatensor = volume.to_monai(ensure_channel_first=False)
    assert metatensor.shape == (10, 10, 10, 1)

    array = np.zeros((10, 10, 10, 1))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        channels={'OpticalPathIdentifier': ['path1']},
        coordinate_system="PATIENT",
    )

    metatensor = volume.to_monai(ensure_channel_first=True)
    assert metatensor.shape == (1, 10, 10, 10)

    array = np.zeros((10, 10, 10))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        coordinate_system="PATIENT",
    )

    metatensor = volume.to_monai(ensure_channel_first=False)
    assert metatensor.shape == (10, 10, 10)

    array = np.zeros((10, 10, 10))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        coordinate_system="PATIENT",
    )

    metatensor = volume.to_monai(ensure_channel_first=True)
    assert metatensor.shape == (1, 10, 10, 10)

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

    volume.to_monai()

    array = np.zeros((10, 10, 10, 1, 1))
    volume = Volume.from_attributes(
        array=array,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        spacing_between_slices=2.0,
        channels={
            ChannelDescriptor('Channel0', True, str): ['class0'],
            ChannelDescriptor('Channel1', True, str): ['class1']
        },
        coordinate_system="PATIENT",
    )

    with pytest.raises(
        ValueError,
        match=(
            'Monai conversion does not currently support'
            ' volumes with multiple channel dimensions.'
        )
    ):
        volume.to_monai()

    array = np.zeros((1, 10, 10, 10))
    metatensor = monai.data.MetaTensor(array)
    Volume.from_monai(metatensor)

    array = np.zeros((10, 10, 10, 1))
    metatensor = monai.data.MetaTensor(array)
    Volume.from_monai(metatensor, channel_dim=-1)

    array = np.zeros((1, 10, 10, 10))
    metatensor = monai.data.MetaTensor(array)
    Volume.from_monai(
        metatensor,
        channels={'OpticalPathIdentifier': ['path1']}
    )

    array = np.zeros((2, 10, 10, 10))
    metatensor = monai.data.MetaTensor(array)
    Volume.from_monai(
        metatensor,
        channels={'OpticalPathIdentifier': ['path1', 'path2']}
    )

    array = np.zeros((1, 1, 10, 10, 10))
    metatensor = monai.data.MetaTensor(array)

    with pytest.raises(
        ValueError,
        match=(
            'Monai conversion does not currently support'
            ' volumes with multiple channel dimensions.'
        )
    ):
        Volume.from_monai(metatensor)

    array = np.zeros((2, 10, 10, 10))
    metatensor = monai.data.MetaTensor(array)

    with pytest.raises(
        ValueError,
        match=(
            'Monai conversion requires `channels` be specified'
            ' for volumes with >=2 channels.'
        )
    ):
        Volume.from_monai(metatensor)


@pytest.mark.parametrize(
    'volume,squeeze_dim',
    [
        [
            Volume(
                array=np.random.rand(1, 10, 10),
                affine=np.array(
                    [[0., 4.99e-4, 0., -23.449374],
                     [0., 0., 4.99e-4, -25.691075],
                     [1., 0., 0., 1.01],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            0
        ],
        [
            Volume(
                array=np.random.rand(10, 1, 10),
                affine=np.array(
                    [[0., 4.99e-04, 0., -23.449873],
                     [0., 0., 4.99e-04, -25.691574],
                     [1., 0., 0., 0.],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            1
        ],
        [
            Volume(
                array=np.random.rand(10, 10, 1),
                affine=np.array(
                    [[0., 4.99e-04, 0., -23.449873],
                     [0., 0., 4.99e-04, -25.691574],
                     [1., 0., 0., 0.],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            2
        ],
        [
            Volume(
                array=np.random.rand(1, 10, 10),
                affine=np.array(
                    [[0.51484336, -2.45034467, -0.29838402, 31.96125748],
                     [0.84430051, 1.01480423, -1.0149242, 19.85095778],
                     [1.16995367, 0.34594871, 0.86372826, -39.22734667],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            0
        ],
        [
            Volume(
                array=np.random.rand(10, 1, 10),
                affine=np.array(
                    [[-6.62073016e-03, -0.648063598, 6.39172474e-02, -25.55167],
                     [-0.786401952, -1.4152762e-02, -1.50316788, -12.367773],
                     [0.950988641, -1.621514e-02, -1.24257108, -11.5527702],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            1
        ],
        [
            Volume(
                array=np.random.rand(10, 10, 1),
                affine=np.array(
                    [[0.783406238, 0.312050746, 2.29152341e-02, 45.850208],
                     [7.69377955e-02, -0.998526806, -0.712275579, -49.9411473],
                     [-9.27305439e-02, 1.80779755, -0.397376895, 18.0307572],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            2
        ],
        [
            Volume(
                array=np.random.rand(1, 10, 10),
                affine=np.array(
                    [[-1.8249537, 0.357257741, -0.365592144, 35.5553115],
                     [1.06765446, 0.586750214, -0.637285403, 18.5404252],
                     [-2.58571459e-02, -0.987438965, -0.510956107, 27.5584146],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            0
        ],
        [
            Volume(
                array=np.random.rand(10, 1, 10),
                affine=np.array(
                    [[0.40659061, -0.14035702, 2.52546278, -22.24126432],
                     [-2.01663853, -0.32789655, 0.12001713, -27.24361216],
                     [1.09727089, -0.55062153, -0.71522747, -25.00163784],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            1
        ],
        [
            Volume(
                array=np.random.rand(10, 10, 1),
                affine=np.array(
                    [[-0.692324440, 1.81254936, 0.770028884, 19.9835788],
                     [-1.01187701, -1.29172094, 1.06442359, 4.6090612],
                     [2.01112784, -2.59508652e-02, 0.800633129, -38.2126518],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            2
        ],
    ]
)
def test_squeeze_affine_roundtrip(volume, squeeze_dim):
    metatensor = volume.to_monai(squeeze_dim=squeeze_dim)

    assert (
        volume.array.squeeze(squeeze_dim) == metatensor.numpy()
    ).all()

    volume_recon = Volume.from_monai(metatensor, unsqueeze_dim=squeeze_dim)

    assert np.allclose(volume.array, volume_recon.array)
    assert np.allclose(volume.affine, volume_recon.affine)


def test_squeeze_errors():
    volume = Volume(
        array=np.random.rand(1, 10, 10),
        affine=np.array([[0., 4.99e-4, 0., -23.449374],
                         [0., 0., 4.99e-4, -25.691075],
                         [1., 0., 0., 1.01],
                         [0., 0., 0., 1.]]),
        coordinate_system="PATIENT",
    )
    squeeze_dim = 3

    with pytest.raises(
        ValueError,
        match=re.escape(
            'If provided, `squeeze_dim` must be a spatial'
            ' dimension (0, 1, 2).'
        )
    ):
        volume.to_monai(squeeze_dim=squeeze_dim)

    volume = Volume(
        array=np.random.rand(10, 10, 10),
        affine=np.array([[0., 4.99e-4, 0., -23.449374],
                         [0., 0., 4.99e-4, -25.691075],
                         [1., 0., 0., 1.01],
                         [0., 0., 0., 1.]]),
        coordinate_system="PATIENT",
    )
    squeeze_dim = 0

    with pytest.raises(
        ValueError,
        match=re.escape(
            f'`squeeze_dim={squeeze_dim}` does not correspond'
            ' to a singleton dimension. Array has shape:'
            f' {volume.array.shape}.'
        )
    ):
        volume.to_monai(squeeze_dim=squeeze_dim)


@pytest.mark.parametrize(
    'volume,squeeze_dim,rotation,spacing,axis_flips',
    [
        [
            Volume(
                array=np.arange(100).reshape(1, 10, 10),
                affine=np.array(
                    [[0., 4.99e-4, 0., -23.449374],
                     [0., 0., 4.99e-4, -25.691075],
                     [1., 0., 0., 1.01],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            0,
            15,
            [1.0e-4, 1.0e-4],
            [1, 2]
        ],
        [
            Volume(
                array=np.arange(100).reshape(10, 1, 10),
                affine=np.array(
                    [[0., 4.99e-04, 0., -23.449873],
                     [0., 0., 4.99e-04, -25.691574],
                     [1., 0., 0., 0.],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            1,
            0,
            [4.99e-4, 1.0e-4],
            [0]
        ],
        [
            Volume(
                array=np.arange(100).reshape(10, 10, 1),
                affine=np.array(
                    [[0., 4.99e-04, 0., -23.449873],
                     [0., 0., 4.99e-04, -25.691574],
                     [1., 0., 0., 0.],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            2,
            78,
            [3, 4.99e-4],
            None
        ],
        # [
        #     Volume(
        #         array=np.arange(100).reshape(1, 10, 10),
        #         affine=np.array(
        #             [[0.51484336, -2.45034467, -0.29838402, 31.96125748],
        #              [0.84430051, 1.01480423, -1.0149242, 19.85095778],
        #              [1.16995367, 0.34594871, 0.86372826, -39.22734667],
        #              [0., 0., 0., 1.]]
        #         ),
        #         coordinate_system="PATIENT",
        #     ),
        #     0,
        #     60,
        #     [1.0, 0.5],
        #     [2]
        # ],
        [
            Volume(
                array=np.arange(100).reshape(10, 1, 10),
                affine=np.array(
                    [[-6.62073016e-03, -0.648063598, 6.39172474e-02, -25.55167],
                     [-0.786401952, -1.4152762e-02, -1.50316788, -12.367773],
                     [0.950988641, -1.621514e-02, -1.24257108, -11.5527702],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            1,
            24,
            [2.0, 2.0],
            [0, 2]
        ],
        [
            Volume(
                array=np.arange(100).reshape(10, 10, 1),
                affine=np.array(
                    [[0.783406238, 0.312050746, 2.29152341e-02, 45.850208],
                     [7.69377955e-02, -0.998526806, -0.712275579, -49.9411473],
                     [-9.27305439e-02, 1.80779755, -0.397376895, 18.0307572],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            2,
            90,
            [1.0, 1.0],
            None
        ],
        [
            Volume(
                array=np.arange(100).reshape(1, 10, 10),
                affine=np.array(
                    [[-1.8249537, 0.357257741, -0.365592144, 35.5553115],
                     [1.06765446, 0.586750214, -0.637285403, 18.5404252],
                     [-2.58571459e-02, -0.987438965, -0.510956107, 27.5584146],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            0,
            127,
            [0.89, 1.73],
            [1, 2]
        ],
        [
            Volume(
                array=np.arange(100).reshape(10, 1, 10),
                affine=np.array(
                    [[0.40659061, -0.14035702, 2.52546278, -22.24126432],
                     [-2.01663853, -0.32789655, 0.12001713, -27.24361216],
                     [1.09727089, -0.55062153, -0.71522747, -25.00163784],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            1,
            180,
            [1.82, 0.65],
            None
        ],
        [
            Volume(
                array=np.arange(100).reshape(10, 10, 1),
                affine=np.array(
                    [[-0.692324440, 1.81254936, 0.770028884, 19.9835788],
                     [-1.01187701, -1.29172094, 1.06442359, 4.6090612],
                     [2.01112784, -2.59508652e-02, 0.800633129, -38.2126518],
                     [0., 0., 0., 1.]]
                ),
                coordinate_system="PATIENT",
            ),
            2,
            30,
            [1, 1],
            None
        ],
    ]
)
def test_squeeze_transform_equivalence(
    volume,
    squeeze_dim,
    rotation,
    spacing,
    axis_flips
):
    metatensor_2d = volume.to_monai(
        squeeze_dim=squeeze_dim,
        ensure_channel_first=True
    )
    metatensor_3d = volume.to_monai(
        ensure_channel_first=True
    )

    keep_dims = [i for i in range(3) if i != squeeze_dim]
    axis_flips_2d = tuple(
        d - (d > squeeze_dim) for d in axis_flips
    ) if axis_flips is not None else None

    rotation_2d, rotation_3d = get_equivalent_rotation_matrices(
        metatensor_3d.affine.numpy(),
        squeeze_dim,
        np.radians(rotation),
    )

    transform_2d = monai.transforms.Compose([
        monai.transforms.Identity(),
        monai.transforms.Affine(
            affine=rotation_2d,
            mode="nearest",
            padding_mode="zeros",
            image_only=True,
        ),
        monai.transforms.Spacing(
            pixdim=spacing,
            mode="nearest",
            padding_mode="zeros",
        ),
        *(
            [] if axis_flips is None else
            [monai.transforms.Flip(spatial_axis=axis_flips_2d)]
        ),
        monai.transforms.ResizeWithPadOrCrop(
            [np.ceil(s / 2) for s in metatensor_2d.shape[1:]]
        )
    ])

    spacing_3d = list(volume.spacing)
    for d, s in zip(keep_dims, spacing):
        spacing_3d[d] = s

    transform_3d = monai.transforms.Compose([
        monai.transforms.Identity(),
        monai.transforms.Affine(
            affine=rotation_3d,
            mode="nearest",
            padding_mode="zeros",
            image_only=True,
        ),
        monai.transforms.Spacing(
            pixdim=spacing_3d,
            mode="nearest",
            padding_mode="zeros",
        ),
        *(
            [] if axis_flips is None else
            [monai.transforms.Flip(spatial_axis=axis_flips)]
        ),
        monai.transforms.ResizeWithPadOrCrop(
            [np.ceil(s / 2) for s in metatensor_3d.shape[1:]]
        )
    ])

    # Test entire compose block
    volume_transformed_2d = Volume.from_monai(
        transform_2d(metatensor_2d),
        unsqueeze_dim=squeeze_dim
    )
    volume_transformed_3d = Volume.from_monai(
        transform_3d(metatensor_3d)
    )

    assert np.allclose(
        volume_transformed_2d.array,
        volume_transformed_3d.array
    )
    assert np.allclose(
        volume_transformed_2d.affine,
        volume_transformed_3d.affine
    )

    # Test transforms individually
    mt_2d = metatensor_2d.clone()
    mt_3d = metatensor_3d.clone()

    for t_2d, t_3d in zip(transform_2d.transforms, transform_3d.transforms):
        mt_2d = t_2d(mt_2d)
        mt_3d = t_3d(mt_3d)

        array_2d = mt_2d[0].cpu().numpy()
        array_3d = mt_3d[0].squeeze(squeeze_dim).cpu().numpy()

        assert np.allclose(
            array_2d,
            array_3d,
        ), f"Array mismatch after {t_2d} / {t_3d}"

        affine_2d = mt_2d.affine.cpu().numpy()
        affine_3d = mt_3d.affine.cpu().numpy()

        reconstructed_affine = _unsqueeze_affine(
            affine_2d,
            squeeze_dim,
            original_affine=metatensor_3d.affine,
        )

        assert np.allclose(
            reconstructed_affine,
            affine_3d,
        ), f"Affine mismatch after {t_2d} / {t_3d}"


@pytest.mark.parametrize(
    'affine_2d,unsqueeze_dim,spacing,space',
    [
        [
            np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]]),
            0,
            1.,
            SpaceKeys.RAS
        ],
        [
            np.array([[1.73205081, -0.25, 10.],
                      [1., 0.4330127, -5.],
                      [0., 0., 1.]]),
            0,
            2.5,
            SpaceKeys.RAS
        ],
        [
            np.array([[0.18117333, 1.25570357, -3.2],
                      [-0.67614808, 0.33646476, 8.1],
                      [0., 0., 1.]]),
            1,
            0.5,
            SpaceKeys.LPS
        ],
        [
            np.array([[-0.75, -0.69282032, 42.],
                      [1.29903811, -0.4, 17.5],
                      [0., 0., 1.]]),
            2,
            3.0,
            SpaceKeys.RAS
        ],
        [
            np.array([[-0.84572336, -0.75244432, -12.],
                      [-0.30781813, 2.06732377, 4.],
                      [0., 0., 1.]]),
            0,
            1.25,
            SpaceKeys.LPS
        ],
        [
            np.array([[0.21213203, 0.42426407, 0.],
                      [0.21213203, -0.42426407, 25.],
                      [0., 0., 1.]]),
            1,
            4.0,
            SpaceKeys.RAS
        ],
        [
            np.array([[-2.25526229, 0.37622216, 7.7],
                      [-0.82084834, -1.03366188, -30.3],
                      [0., 0., 1.]]),
            2,
            0.75,
            SpaceKeys.LPS
        ],
        [
            np.array([[4.78152378, 0.0584743409, -100.],
                      [1.46185852, -0.191260951, 60.],
                      [0., 0., 1.]]),
            2,
            1.0,
            SpaceKeys.RAS
        ],
    ]
)
def test_unsqueeze_arbitrary_2d_metatensor(
    affine_2d,
    unsqueeze_dim,
    spacing,
    space
):
    metatensor = monai.data.MetaTensor(
        np.arange(100).reshape(10, 10),
        affine=affine_2d
    )
    metatensor.meta[MetaKeys.SPACE] = space

    with pytest.warns(
        UserWarning,
        match=(
            '_unsqueeze_affine was called without original_affine. There is no'
            ' reference to 3D space, so the 2D affine will be embedded in the'
            f' axis-aligned plane perpendicular to axis {unsqueeze_dim}, with'
            f' spacing={spacing} along the new axis and no translation applied.'
        )
    ):
        vol = Volume.from_monai(
            metatensor,
            unsqueeze_dim=unsqueeze_dim,
            unsqueeze_spacing=spacing,
            channel_dim=None
        )

    expected_shape = [10, 10]
    expected_shape.insert(unsqueeze_dim, 1)
    assert vol.shape == tuple(expected_shape)
    assert (vol.array == metatensor.unsqueeze(unsqueeze_dim).numpy()).all()
    assert np.isclose(vol.spacing[unsqueeze_dim], spacing)

    roundtrip = vol.to_monai(
        squeeze_dim=unsqueeze_dim,
        convert_to_ras=(space == 'RAS'),
    )
    assert roundtrip.shape == metatensor.shape
    assert (roundtrip == metatensor).all()
    assert roundtrip.affine.allclose(metatensor.affine)

    keep_dims = [i for i in range(3) if i != unsqueeze_dim]

    # verify right-handed
    cols = [vol.affine[:3, i] for i in range(3)]

    assert np.dot(np.cross(cols[0], cols[1]), cols[2]) > 0

    u, v = (cols[i] for i in keep_dims)
    n = np.cross(u, v)
    w = cols[unsqueeze_dim]
    assert np.allclose(np.cross(w, n), 0, atol=1e-8)

    parity = -1 if unsqueeze_dim == 1 else 1
    assert np.sign(np.dot(w, n)) == parity
