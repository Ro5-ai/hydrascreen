from hydrascreen.api import APICredentials
from hydrascreen.predictor import HydraScreen

__version__ = "0.0.3"


def login(email: str, organization: str) -> HydraScreen:
    """
    Logs in and creates a new instance of the HydraScreen class with the provided email and organization.

    Args:
        email (str): The email of the user.
        organization (str): The organization the user belongs to.

    Returns:
        HydraScreen: A new instance of the HydraScreen class.

    """
    api_credentials = APICredentials(email=email, organization=organization)
    return HydraScreen(api_credentials=api_credentials)
