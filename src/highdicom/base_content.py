"""Module containing content required by the base SOPClass."""
import datetime
from typing import Sequence
from typing_extensions import Self

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes
from pydicom.valuerep import DT

from highdicom.version import __version__
from highdicom.sr.coding import CodedConcept
from highdicom.valuerep import (
    _check_long_string,
    _check_short_string,
    _check_short_text,
)
from highdicom._module_utils import (
    does_iod_have_pixel_data,
)


class ContributingEquipment(Dataset):

    """Data Set representing an item of the ContributingEquipmentSequence.

    Contains descriptive attributes of related equipment that has contributed
    to the acquisition, creation or modification of a DICOM Instance.

    """

    def __init__(
        self,
        manufacturer: str,
        purpose_of_reference: Code | CodedConcept,
        *,
        manufacturer_model_name: str | None = None,
        software_versions: str | Sequence[str] | None = None,
        device_serial_number: str | None = None,
        institution_name: str | None = None,
        institutional_department_name: str | None = None,
        institution_address: str | None = None,
        station_name: str | None = None,
        contribution_datetime: str | datetime.datetime | DT | None = None,
        contribution_description: str | None = None,
    ):
        """

        Parameters
        ----------
        manufacturer: str
            Manufacturer of the equipment that contributed.
        purpose_of_reference: pydicom.sr.coding.Code | highdicom.sr.CodedConcept
            The purpose for which the related equipment is being referenced.
            This must be one of the following codes from the DCM terminology:
            SynthesizingEquipment, AcquisitionEquipment, ProcessingEquipment,
            ModifyingEquipment, DeIdentifyingEquipment,
            FrameExtractingEquipment, EnhancedMultiFrameConversionEquipment.
        manufacturer_model_name: str | None, optional
            Manufacturer's model name of the equipment that contributed.
        software_versions: str | Sequence[str] | None, optional
            Manufacturer's designation of the software version of the equipment
            that contributed.
        device_serial_number: str | None, optional
            Manufacturer's serial number of the equipment that contributed.
        institution_name: str | None, optional
            Institution where the equipment that contributed is located.
        institutional_department_name: str | None, optional
            Department in the institution where the equipment that contributed
            is located.
        institution_address: str | None, optional
            Address of the institution where the equipment that contributed
            is located.
        station_name: str | None, optional
            User defined name identifying the machine that contributed.
        contribution_datetime: str | datetime.datetime | DT | None, optional
            The date & time when the equipment contributed.
        contribution_description: str | None, optional
            Description of the contribution the equipment made.

        """
        super().__init__()

        _check_long_string(manufacturer)
        self.Manufacturer = manufacturer

        allowed_reference_codes = (
            codes.DCM.SynthesizingEquipment,
            codes.DCM.AcquisitionEquipment,
            codes.DCM.ProcessingEquipment,
            codes.DCM.ModifyingEquipment,
            codes.DCM.DeIdentifyingEquipment,
            codes.DCM.FrameExtractingEquipment,
            codes.DCM.EnhancedMultiFrameConversionEquipment,
        )

        if not isinstance(purpose_of_reference, CodedConcept):
            purpose_of_reference = CodedConcept.from_code(purpose_of_reference)
        if purpose_of_reference not in allowed_reference_codes:
            raise ValueError(
                "Argument 'purpose_of_reference' has an invalid value."
            )
        self.PurposeOfReferenceCodeSequence = [purpose_of_reference]

        if manufacturer_model_name is not None:
            _check_long_string(manufacturer_model_name)
            self.ManufacturerModelName = manufacturer_model_name

        if institution_name is not None:
            _check_long_string(institution_name)
            self.InstitutionName = institution_name

        if institutional_department_name is not None:
            _check_long_string(institutional_department_name)
            self.InstitutionalDepartmentName = institutional_department_name

        if institution_address is not None:
            _check_short_text(institution_address)
            self.InstitutionAddress = institution_address

        if station_name is not None:
            _check_short_string(station_name)
            self.StationName = station_name

        if device_serial_number is not None:
            _check_long_string(device_serial_number)
            self.DeviceSerialNumber = device_serial_number

        if software_versions is not None:
            if isinstance(software_versions, str):
                _check_long_string(software_versions)
            else:
                software_versions = list(software_versions)
                if len(software_versions) == 0:
                    raise ValueError(
                        "Argument 'software_versions' must not be empty."
                    )
                for item in software_versions:
                    _check_long_string(item)
            self.SoftwareVersions = software_versions

        if contribution_datetime is not None:
            self.ContributionDateTime = DT(contribution_datetime)

        if contribution_description is not None:
            _check_short_text(contribution_description)
            self.ContributionDescription = contribution_description

    @classmethod
    def for_image_acquisition(
        cls,
        dataset: Dataset,
    ) -> Self:
        """Create an item describing image acquisition of a given image dataset.

        Parameters
        ----------
        dataset: pydicom.Dataset
            Image dataset.

        Returns
        -------
        highdicom.ContributingEquipment:
            Contributing equipment object describing the acquisition of the
            image in the provided dataset.

        Raises
        ------
        ValueError:
            If the dataset does not represent an image.
        AttributeError:
            If the dataset does not contain the Manufacturer attribute, or it
            is empty.

        """
        if not does_iod_have_pixel_data(dataset.SOPClassUID):
            raise ValueError("Dataset does not represent an Image.")

        manufacturer = dataset.get('Manufacturer')  # NB type 2

        if manufacturer is None or manufacturer == '':
            raise AttributeError('Dataset has no manufacturer information.')

        manufacturer_model_name = dataset.get('ManufacturerModelName')
        software_versions = dataset.get('SoftwareVersions')
        device_serial_number = dataset.get('DeviceSerialNumber')
        institution_name = dataset.get('InstitutionName')
        institutional_department_name = dataset.get(
            'InstitutionalDepartmentName'
        )
        institution_address = dataset.get('InstitutionAddress')
        station_name = dataset.get('StationName')
        contribution_description = (
            'Acquisition equipment of the source image(s)'
        )

        contribution_datetime = dataset.get('AcquisitionDateTime')

        if contribution_datetime is None:
            date = dataset.get('AcquisitionDate')
            time = dataset.get('AcquisitionTime')
            if (
                isinstance(date, datetime.date) and
                isinstance(time, datetime.time)
            ):
                contribution_datetime = datetime.datetime.combine(
                    dataset.AcquisitionDate, dataset.AcquisitionTime
                )

        return cls(
            manufacturer=manufacturer,
            purpose_of_reference=codes.DCM.AcquisitionEquipment,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
            institution_name=institution_name,
            institutional_department_name=institutional_department_name,
            institution_address=institution_address,
            station_name=station_name,
            contribution_datetime=contribution_datetime,
            contribution_description=contribution_description,
        )

    @classmethod
    def for_highdicom(
        cls,
        *,
        purpose_of_reference: Code | CodedConcept | None = None,
    ) -> Self:
        """Create an item describing highdicom's contribution to the instance.

        Parameters
        ----------
        purpose_of_reference: pydicom.sr.coding.Code | highdicom.sr.CodedConcept | None, optional
            The purpose for which the related equipment is being referenced.
            This must be one of the following codes from the DCM terminology:
            SynthesizingEquipment, AcquisitionEquipment, ProcessingEquipment,
            ModifyingEquipment, DeIdentifyingEquipment,
            FrameExtractingEquipment, EnhancedMultiFrameConversionEquipment.
            If ``None``, ProcessingEquipment will be used.

        """  # noqa: E501
        if purpose_of_reference is None:
            purpose_of_reference = codes.DCM.ProcessingEquipment

        return cls(
            manufacturer='Highdicom open-source contributors',
            manufacturer_model_name='highdicom',
            contribution_datetime=datetime.datetime.now(),
            software_versions=__version__,
            purpose_of_reference=purpose_of_reference,
            contribution_description=(
                'Software library used to create this instance'
            ),
        )
