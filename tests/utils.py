from io import BytesIO

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filereader import dcmread


def write_and_read_dataset(dataset: Dataset):
    """Write DICOM dataset to buffer and read it back from buffer."""
    clone = Dataset(dataset)
    clone.is_little_endian = True
    if hasattr(dataset, 'file_meta'):
        clone.file_meta = FileMetaDataset(dataset.file_meta)
        if dataset.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2':
            clone.is_implicit_VR = True
        else:
            clone.is_implicit_VR = False
    else:
        clone.is_implicit_VR = False
    with BytesIO() as fp:
        clone.save_as(fp)
        return dcmread(fp, force=True)
