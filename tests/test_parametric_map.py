import pytest
import numpy as np

from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.filereader import dcmread
from pydicom.uid import generate_uid

from highdicom.map import ParametricMap


@pytest.fixture
def setup_data():

    pass
