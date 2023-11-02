import requests

from hydrascreen.api import API_URL

__version__ = "0.0.7"


def login(email: str, organization: str):
    """
    Logs in and creates a new instance of the HydraScreen class with the provided email and organization.

    Args:
        email (str): The email of the user.
        organization (str): The organization the user belongs to.
    """
    response = requests.post(
        url=f"{API_URL}/email/verify",
        headers={"Content-Type": "application/json"},
        json={
            "email": email,
            "organization": organization,
        },
    )

    if response.status_code != 200:
        raise Exception(f"Unable to verify email. Detail: {response}")
    print(
        f"Sent verification email to {email}. Please check your email and use token provided to instantiate Hydrascreen class."
    )
