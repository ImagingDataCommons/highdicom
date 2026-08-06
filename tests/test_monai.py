import numpy as np
import tempfile
import pytest
import pydicom
import zipfile


from pathlib import Path
from typing import Sequence
from highdicom import Volume, ChannelDescriptor, get_volume_from_series
from highdicom.spatial import (
    get_closest_patient_orientation,
    convert_affine_to_convention
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

except Exception:
    pytest.skip("Optional dependency not available", allow_module_level=True)


def read_github_zip_volume_and_metatensor(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        zipfilename = Path(temp_dir) / Path(url).name
        urldownload_with_retry(url, zipfilename)

        with zipfile.ZipFile(zipfilename, 'r') as zf:
            zf.extractall(temp_dir)

        series = [pydicom.dcmread(f) for f in Path(temp_dir).glob('**/*.dcm')]

        metatensor = monai.transforms.LoadImage(reader="ITKReader")(
            Path(str(zipfilename)[:-4])
        )

    return get_volume_from_series(series), series, metatensor


def read_github_series_volume_and_metatensor(urls: Sequence[str]):
    series = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for url in urls:
            filename = Path(temp_dir) / Path(url).name
            urldownload_with_retry(url, filename)

            series.append(pydicom.dcmread(filename))

        metatensor = monai.transforms.LoadImage(reader="ITKReader")(temp_dir)

    return get_volume_from_series(series), series, metatensor


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
    'segfile,channel_first,space,spacing,spatial_shape,affine,sum',
    [
        [
            'seg_image_sm_control.dcm',
            True,
            'RAS',
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-04, 0., -23.449374],
                      [0., 0., 4.99e-04, -25.691075],
                      [1., 0., 0., 1.01],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_sm_dots_tiled_full.dcm',
            False,
            'RAS',
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-04, 0., -23.449873],
                      [0., 0., 4.99e-04, -25.691574],
                      [1., 0., 0., 0.],
                      [0., 0., 0., 1.]]),
            200
        ],
        [
            'seg_image_ct_true_fractional.dcm',
            True,
            'LPS',
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
            'LPS',
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
            'RAS',
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-04, 0., -23.449374],
                      [0., 0., 4.99e-04, -25.691075],
                      [1., 0., 0., 1.01],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_ct_binary_fractional.dcm',
            True,
            'RAS',
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
            'RAS',
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
            'RAS',
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-04, 0., -23.449374],
                      [0., 0., 4.99e-04, -25.691075],
                      [1., 0., 0., 1.01],
                      [0., 0., 0., 1.]]),
            200
        ],
        [
            'seg_image_sm_control_labelmap.dcm',
            True,
            'RAS',
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-04, 0., -23.449873],
                      [0., 0., 4.99e-04, -25.691574],
                      [1., 0., 0., 0.],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_sm_control_labelmap_palette_color.dcm',
            True,
            'RAS',
            (1.0, 0.000499, 0.000499),
            (1, 50, 50),
            np.array([[0., 4.99e-04, 0., -23.449873],
                      [0., 0., 4.99e-04, -25.691574],
                      [1., 0., 0., 0.],
                      [0., 0., 0., 1.]]),
            523
        ],
        [
            'seg_image_ct_binary.dcm',
            True,
            'RAS',
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
    space,
    spacing,
    spatial_shape,
    affine,
    sum
):
    seg = segread(TEST_DATA / segfile)
    vol = seg.get_volume()
    metatensor = vol.to_monai(
        ensure_channel_first=channel_first,
        space=space
    )
    meta = metatensor.meta

    assert meta[MetaKeys.SPACE] == space
    assert vol.spacing == meta[ImageStatsKeys.SPACING] == spacing
    assert vol.shape[:3] == meta[MetaKeys.SPATIAL_SHAPE] == spatial_shape
    assert (
        metatensor.shape[1:] if channel_first else metatensor.shape[:3] ==
        spatial_shape
    )
    assert np.allclose(vol.get_affine(space), affine, atol=1e-4)
    assert np.allclose(metatensor.affine, affine, atol=1e-4)
    assert np.allclose(meta[MetaKeys.ORIGINAL_AFFINE], affine, atol=1e-4)
    assert np.allclose(meta[MetaKeys.AFFINE], affine, atol=1e-4)
    assert meta[MetaKeys.ORIGINAL_CHANNEL_DIM] == -1
    assert vol.array.sum() == metatensor.numpy().sum() == sum


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
