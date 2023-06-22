import pytest
from hydrascreen.api import APICredentials


@pytest.fixture(scope="module")
def mock_credentials():
    return APICredentials(email="foo@bar.com", organization="Test Org")
