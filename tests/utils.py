from io import BytesIO

from pathlib import Path
from pydicom.data import get_testdata_files
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filereader import dcmread


from highdicom._module_utils import (
    does_iod_have_pixel_data,
)


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
        return dcmread(fp, force=True)


def find_readable_images() -> list[str]:
    """Get a list of all images in highdicom and pydicom test data that should
    be expected to work with image reading routines.

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

        files_to_use.append(f)

    return files_to_use
