import pytest

from pydicom import config


@pytest.fixture(autouse=True, scope='session')
def setup_pydicom_config():
    """Fixture that sets up pydicom config values for all tests."""
    config.enforce_valid_values = True
    yield
