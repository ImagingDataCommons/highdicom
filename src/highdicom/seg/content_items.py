from enum import Enum

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code, CodedConcept


class SegmentAlgorithmTypes(Enum):
    AUTOMATIC = 'AUTOMATIC'
    SEMIAUTOMATIC = 'SEMIAUTOMATIC'
    MANUAL = 'MANUAL'


class SegmentationAlgorithmIdentification(Dataset):

    """Sequence item describing a segmentation algorithm."""

    def __init__(self, name: str,
                 family: Union[Code, CodedConcept],
                 version: str,
                 source: Optional[str] = None,
                 parameters: Optional[Dict[str: str]] = None):
        """
        Parameters
        ----------
        name: str
            name of the algorithm
        family: Union[pydicom.sr.coding.Code, pydicom.sr.coding.CodedConcept]
            kind of algorithm family
        version: str
            version of the algorithm
        source: str, optional
            source of the algorithm, e.g. name of the algorithm manufacturer
        parameters: Dict[str: str], optional
            name and actual value of the parameters with which the algorithm
            was invoked

        """  # noqa
        self.AlgorithmName = name
        self.AlgorithmVersion = version
        self.AlgorithmFamilyCodeSequence = [
            family,
        ]
        self.AlgorithmParameters = ','.join([
            '='.join([key, value])
            for key, value in parameters.items()
        ])


class Segment(Dataset):

    """Sequence item describing a segment."""

    def __init__(self, number: int,
                 label: str,
                 segmented_property_category: Sequence[Union[Code, CodedConcept]],
                 segmented_property_type: Sequence[Union[Code, CodedConcept]],
                 algorithm_type: Union[SegmentAlgorithmType, str],
                 algorithm_name: Optional[str] = None,
                 algorithm_identification: Optional[SegmentationAlgorithmIdentification] = None,
                 tracking_uid: Optional[str] = None,
                 tracking_id: Optional[str] = None,
                 anatomic_regions: Optional[Sequence[Union[Code, CodedConcept]]] = None,
                 primary_anatomic_structures: Optional[Sequence[Union[Code, CodedConcept]]] = None
            ) -> None:
        """
        Parameters
        ----------

        """
        self.SegmentNumber = number
        self.SegmentLabel = label
        self.SegmentedPropertyCategoryCodeSequence = [
            segmented_property_category,
        ]
        self.SegmentedPropertyTypeCodeSequence = [
            segmented_property_type,
        ]
        self.SegmentAlgorithmType = SegmentAlgorithmTypes(algorithm_type).value
        self.SegmentAlgorithmName = algorithm_name
        if algorithm_identification is not None:
            self.SegmentationAlgorithmIdentificationSequence = [
                algorithm_identification,
            ]
        num_given_tracking_identifiers = sum(
            tracking_id is not None,
            tracking_uid is not None
        )
        if num_given_tracking_identifiers == 2:
            self.TrackingID = tracking_id
            self.TrackingUID = tracking_uid
        elif num_given_tracking_identifiers == 1:
            raise TypeError(
                'Tracking ID and Tracking UID must both be provided.'
            )
        if anatomic_regions is not None:
            self.AnatomicRegionSequence = anatomic_regions
        if primary_anatomic_structures is not None:
            self.PrimaryAnatomicStructureSequence = primary_anatomic_structures



