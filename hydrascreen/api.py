from io import BufferedReader, StringIO
import pandas as pd
import requests
from typing import Dict, List

API_URL = "https://hydrascreen-api.ro5.ai/v2"


class APIError(Exception):
    pass


def inference(inference_pairs: List[Dict[str, str]], token: str) -> pd.DataFrame:
    """
    Sends a POST request to the API to retrieve predictions based on the provided inference pairs.

    Args:
        inference_pairs (List[Dict[str, str]]): A list of dictionaries representing inference pairs.
        Format of inference pairs:
        [
            {
                "protein": "s3://bucket/path/to/protein.pdb",
                "docked_ligand": "s3://bucket/path/to/docked_ligand.sdf"
            }
        ]
        token (str): jwt token provided for usage of the api

    Returns:
        pd.DataFrame: A DataFrame containing the predictions retrieved from the API.

    Raises:
        APIError: If the response status code is not 200, indicating failure in retrieving predictions.
    """
    headers = {"accept": "text/csv", "Authorization": f"Bearer {token}"}
    response = requests.post(
        url=f"{API_URL}/invocation",
        headers=headers,
        json={
            "inference_pairs": inference_pairs,
        },
    )

    response_json = response.json()

    if response.status_code != 200:
        raise APIError(f"Failed to retrieve predictions: {response_json['detail']}")

    csv_response = StringIO(response_json)

    return pd.read_csv(csv_response)


def upload_pdb(pdb_file: BufferedReader, token: str) -> str:
    """
    Uploads a PDB file to the API for inference.

    Args:
        pdb_file (BufferedReader): A buffered reader object representing the PDB file to upload.
        token (str): jwt token provided for usage of the api

    Returns:
        str: The S3 key of the uploaded PDB file.

    Raises:
        APIError: If the response status code is not 200, indicating failure in uploading the PDB file.
    """
    files = {"file": pdb_file}
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(url=f"{API_URL}/upload-pdb", headers=headers, files=files)

    response_json = response.json()

    if response.status_code != 200:
        raise APIError(f"Failed to upload PDB file: {response_json['detail']}")

    return response_json["s3_key"]


def upload_sdf(pdb_s3_path: str, sdf_file: BufferedReader, token: str) -> str:
    """
    Uploads an SDF file to the API using the provided associated PDB S3 path.

    Args:
        pdb_s3_path (str): The S3 path of the associated PDB file.
        sdf_file (BufferedReader): A buffered reader object representing the SDF file to upload.
        token (str): jwt token provided for usage of the api

    Returns:
        str: The S3 key of the uploaded SDF file.

    Raises:
        APIError: If the response status code is not 200, indicating failure in uploading the SDF file.
    """
    query_params = {"pdb_s3_path": pdb_s3_path}
    files = {"file": sdf_file}
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        url=f"{API_URL}/upload-sdf", headers=headers, params=query_params, files=files
    )

    response_json = response.json()

    if response.status_code != 200:
        raise APIError(f"Failed to upload SDF file: {response_json['detail']}")

    return response_json["s3_key"]
