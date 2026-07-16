import urllib
import pydicom
import tempfile
import zipfile
import time

from io import BytesIO
from typing import Sequence
from pathlib import Path
from pydicom.data import get_testdata_files, get_testdata_file
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filereader import dcmread
from pydicom import uid
from highdicom import imread, get_volume_from_series
from highdicom._standard_utils import (
    does_iod_have_pixel_data,
)

DCM_QA_MPRAGE = 'https://github.com/neurolabusc/dcm_qa_mprage/raw/refs/heads/main'  # noqa: E501
DCM_QA_ME = 'https://github.com/neurolabusc/dcm_qa_me/raw/refs/heads/master'
DCM_QA_PDT2 = 'https://github.com/neurolabusc/dcm_qa_pdt2/raw/refs/heads/main'


def write_and_read_dataset(dataset: Dataset):
    """Write DICOM dataset to buffer and read it back from buffer."""
    clone = Dataset(dataset)
    if hasattr(dataset, 'file_meta'):
        clone.file_meta = FileMetaDataset(dataset.file_meta)
        little_endian = None
        implicit_vr = None
    else:
        little_endian = True
        implicit_vr = True

    with BytesIO() as fp:
        clone.save_as(
            fp,
            implicit_vr=implicit_vr,
            little_endian=little_endian,
        )
        out = dcmread(fp, force=True)

    # Remove the reference to the closed buffer, otherwise this will
    # cause annoying warnings
    out.buffer = None

    return out


def find_readable_images() -> list[tuple[str, str | None]]:
    """Get a list of all images in highdicom and pydicom test data that should
    be expected to work with image reading routines.

    Returns a list of tuples (path, dependency), where path is the filepath,
    and dependency is either None if the file can be read using only required
    dependencies, or a str that can be used with pytest.importorskip if an
    optional dependency is required to decode pixel data.

    """
    # All pydicom test files
    all_files = get_testdata_files()

    # Add highdicom test files
    file_path = Path(__file__)
    data_dir = file_path.parent.parent.joinpath('data/test_files')
    hd_files = [str(f) for f in data_dir.glob("*.dcm")]

    all_files.extend(hd_files)

    # Various files are not expected to work and should be excluded
    exclusions = [
        # cannot be read due to bad VFR
        "badVR.dcm",
        # pixel data is truncated
        "MR_truncated.dcm",
        # missing number of frames
        "liver_1frame.dcm",
        # pydicom cannot decode pixels
        "JPEG2000-embedded-sequence-delimiter.dcm",
        # deflated transfer syntax cannot be read lazily
        "image_dfl.dcm",
        # pydicom cannot decode pixels
        "JPEG-lossy.dcm",
        # no pixels
        "TINY_ALPHA",
        # messed up transfer syntax
        "SC_rgb_jpeg.dcm",
        # Incorrect source image sequence. This can hopefully be added back
        # after https://github.com/pydicom/pydicom/pull/2204
        "SC_rgb_small_odd.dcm",
    ]

    files_to_use = []

    for f in all_files:
        try:
            # Skip image files that can't even be opened (the test files
            # include some deliberately corrupted files)
            dcm = dcmread(f)
        except Exception:
            continue

        excluded = False
        if 'SOPClassUID' not in dcm:
            # Some are missing this...
            continue
        if not does_iod_have_pixel_data(dcm.SOPClassUID):
            # Exclude non images
            continue
        if not dcm.file_meta.TransferSyntaxUID.is_little_endian:
            # We don't support little endian
            continue

        for exc in exclusions:
            if exc in f:
                excluded = True
                break

        if excluded:
            continue

        dependency = None
        if dcm.file_meta.TransferSyntaxUID in (
            uid.JPEGExtended12Bit,
            uid.JPEGLosslessSV1,
        ):
            dependency = "libjpeg"

        if dcm.file_meta.TransferSyntaxUID in (
            uid.JPEG2000,
            uid.JPEG2000Lossless,
        ):
            dependency = "openjpeg"

        files_to_use.append((f, dependency))

    return files_to_use


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


def urldownload_with_retry(
        url,
        filename,
        retries=10,
        timeout=1
):
    for i in range(retries):
        try:
            urllib.request.urlretrieve(
                url,
                filename
            )

        except Exception:
            time.sleep(timeout)


def read_github_zip_volume(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        zipfilename = Path(temp_dir) / Path(url).name
        urldownload_with_retry(url, zipfilename)

        with zipfile.ZipFile(zipfilename, 'r') as zf:
            zf.extractall(temp_dir)

        series = [pydicom.dcmread(f) for f in Path(temp_dir).glob('**/*.dcm')]

    return get_volume_from_series(series), series


def read_github_series_volume(urls: Sequence[str]):
    series = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for url in urls:
            filename = Path(temp_dir) / Path(url).name
            urldownload_with_retry(url, filename)

            series.append(pydicom.dcmread(filename))

    return get_volume_from_series(series), series
