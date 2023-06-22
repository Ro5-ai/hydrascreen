from dataclasses import dataclass
from io import BufferedReader, StringIO
import pandas as pd
import requests
from typing import Dict, List

API_URL = "https://hydrascreen-api.ro5.ai/v1"


class APIError(Exception):
    pass


@dataclass
class APICredentials:
    email: str
    organization: str


def inference(credentials: APICredentials, inference_pairs: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Sends a POST request to the API to retrieve predictions based on the provided inference pairs.

    Args:
        credentials (APICredentials): The API credentials object.
        inference_pairs (List[Dict[str, str]]): A list of dictionaries representing inference pairs.
        Format of inference pairs:
        [
            {
                "protein": "s3://bucket/path/to/protein.pdb",
                "docked_ligand": "s3://bucket/path/to/docked_ligand.sdf"
            }
        ]

    Returns:
        pd.DataFrame: A DataFrame containing the predictions retrieved from the API.

    Raises:
        APIError: If the response status code is not 200, indicating failure in retrieving predictions.
    """
    headers = {"accept": "text/csv"}
    response = requests.post(
        url=f"{API_URL}/invocation",
        headers=headers,
        json={
            "email": credentials.email,
            "organization": credentials.organization,
            "inference_pairs": inference_pairs,
        },
    )

    if response.status_code != 200:
        raise APIError(f"Failed to retrieve predictions: {response.content.decode('utf-8')}")

    csv_response = StringIO(response.json())

    return pd.read_csv(csv_response)


def upload_pdb(credentials: APICredentials, pdb_file: BufferedReader) -> str:
    """
    Uploads a PDB file to the API for inference.

    Args:
        credentials (APICredentials): The API credentials object.
        pdb_file (BufferedReader): A buffered reader object representing the PDB file to upload.

    Returns:
        str: The S3 key of the uploaded PDB file.

    Raises:
        APIError: If the response status code is not 200, indicating failure in uploading the PDB file.
    """
    query_params = {"email": credentials.email}
    files = {"file": pdb_file}
    response = requests.post(url=f"{API_URL}/upload-pdb", params=query_params, files=files)

    if response.status_code != 200:
        raise APIError(f"Failed to upload PDB file: {response.content.decode('utf-8')}")

    s3_key = response.json()["s3_key"]

    return s3_key


def upload_sdf(credentials: APICredentials, pdb_s3_path: str, sdf_file: BufferedReader) -> str:
    """
    Uploads an SDF file to the API using the provided associated PDB S3 path.

    Args:
        credentials (APICredentials): The API credentials object.
        pdb_s3_path (str): The S3 path of the associated PDB file.
        sdf_file (BufferedReader): A buffered reader object representing the SDF file to upload.

    Returns:
        str: The S3 key of the uploaded SDF file.

    Raises:
        APIError: If the response status code is not 200, indicating failure in uploading the SDF file.
    """
    query_params = {"email": credentials.email, "pdb_s3_path": pdb_s3_path}
    files = {"file": sdf_file}
    response = requests.post(url=f"{API_URL}/upload-sdf", params=query_params, files=files)

    if response.status_code != 200:
        raise APIError(f"Failed to upload SDF file: {response.content.decode('utf-8')}")

    s3_key = response.json()["s3_key"]

    return s3_key
