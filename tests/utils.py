from io import BytesIO

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filereader import dcmread


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
