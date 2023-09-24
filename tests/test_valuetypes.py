import datetime

import pytest
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.valuerep import DT, DA, TM
from pydicom import config

from highdicom.sr.coding import CodedConcept
from highdicom.sr.enum import ValueTypeValues
from highdicom.sr.value_types import (
    DateContentItem,
    DateTimeContentItem,
    TimeContentItem
)
from tests.utils import write_and_read_dataset


class TestDateTimeContentItem:
    test_datetime_values = [
        DT("2023"),
        DT("202306"),
        DT("20230623"),
        DT("2023062311"),
        DT("202306231112"),
        DT("20230623111247"),
        DT("20230623111247.123456"),
    ]

    @pytest.mark.parametrize("datetime_value", test_datetime_values)
    def test_construct_from_datetime(self, datetime_value: DT):
        name = codes.DCM.DatetimeOfProcessing
        assert isinstance(name, Code)
        value_type = ValueTypeValues.DATETIME
        item = DateTimeContentItem(
            name=name,
            value=datetime_value
        )

        assert item.name == name
        assert item.value == datetime_value
        assert item.value_type == value_type
        assert isinstance(item.value, datetime.datetime)
        assert item.value.isoformat() == datetime_value.isoformat()

    @pytest.mark.parametrize("datetime_value", test_datetime_values)
    @pytest.mark.parametrize("datetime_conversion", [True, False])
    def test_from_dataset(
        self,
        datetime_value: DT,
        datetime_conversion: bool
    ):
        config.datetime_conversion = datetime_conversion
        name = codes.DCM.DatetimeOfProcessing
        assert isinstance(name, Code)
        value_type = ValueTypeValues.DATETIME
        dataset = Dataset()
        dataset.ValueType = value_type.value
        dataset.ConceptNameCodeSequence = [CodedConcept.from_code(name)]
        dataset.DateTime = datetime_value

        dataset_reread = write_and_read_dataset(dataset)
        item = DateTimeContentItem.from_dataset(dataset_reread)

        assert item.name == name
        assert item.value == datetime_value
        assert item.value_type == value_type
        assert isinstance(item.value, datetime.datetime)
        assert item.value.isoformat() == datetime_value.isoformat()


class TestDateContentItem:
    def test_construct_from_date(self):
        date_value = DA("20230623")
        name = codes.DCM.AcquisitionDate
        assert isinstance(name, Code)
        value_type = ValueTypeValues.DATE
        item = DateContentItem(
            name=name,
            value=date_value
        )

        assert item.name == name
        assert item.value == date_value
        assert item.value_type == value_type
        assert isinstance(item.value, datetime.date)
        assert item.value.isoformat() == date_value.isoformat()

    @pytest.mark.parametrize("datetime_conversion", [True, False])
    def test_from_dataset(self, datetime_conversion: bool):
        config.datetime_conversion = datetime_conversion
        date_value = DA("20230623")
        name = codes.DCM.AcquisitionDate
        assert isinstance(name, Code)
        value_type = ValueTypeValues.DATE
        dataset = Dataset()
        dataset.ValueType = value_type.value
        dataset.ConceptNameCodeSequence = [CodedConcept.from_code(name)]
        dataset.Date = date_value

        dataset_reread = write_and_read_dataset(dataset)
        item = DateContentItem.from_dataset(dataset_reread)

        assert item.name == name
        assert item.value == date_value
        assert item.value_type == value_type
        assert isinstance(item.value, datetime.date)
        assert item.value.isoformat() == date_value.isoformat()


class TestTimeContentItem:
    test_time_values = [
        TM("11"),
        TM("1112"),
        TM("111247"),
        TM("111247.123456"),
    ]

    @pytest.mark.parametrize("time_value", test_time_values)
    def test_construct_from_time(self, time_value: TM):
        name = codes.DCM.AcquisitionTime
        assert isinstance(name, Code)
        value_type = ValueTypeValues.TIME
        item = TimeContentItem(
            name=name,
            value=time_value
        )

        assert item.name == name
        assert item.value == time_value
        assert item.value_type == value_type
        assert isinstance(item.value, datetime.time)
        assert item.value.isoformat() == time_value.isoformat()

    @pytest.mark.parametrize("time_value", test_time_values)
    @pytest.mark.parametrize("datetime_conversion", [True, False])
    def test_from_dataset(
        self,
        time_value: TM,
        datetime_conversion: bool
    ):
        config.datetime_conversion = datetime_conversion
        name = codes.DCM.AcquisitionDate
        assert isinstance(name, Code)
        value_type = ValueTypeValues.TIME
        dataset = Dataset()
        dataset.ValueType = value_type.value
        dataset.ConceptNameCodeSequence = [CodedConcept.from_code(name)]
        dataset.Time = time_value

        dataset_reread = write_and_read_dataset(dataset)
        item = TimeContentItem.from_dataset(dataset_reread)

        assert item.name == name
        assert item.value == time_value
        assert item.value_type == value_type
        assert isinstance(item.value, datetime.time)
        assert item.value.isoformat() == time_value.isoformat()
