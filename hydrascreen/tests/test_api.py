from io import StringIO
import pandas as pd
import pytest
from unittest.mock import patch, Mock
from hydrascreen.api import APICredentials, APIError, inference, upload_pdb, upload_sdf


@pytest.fixture(scope="module")
def mock_inference_pairs():
    return [
        {
            "protein": "s3://test-bucket/protein.pdb",
            "docked_ligand": "s3://test-bucket/ligand.sdf",
        }
    ]


@patch("hydrascreen.api.requests.post")
def test_inference_successful_response(
    inference_request, mock_credentials: APICredentials, mock_inference_pairs
):
    response_str = """pdb_id,ligand_conformer_id,pki,pose
            protein,protein_ligand_0,1.0,1.0
            protein,protein_ligand_1,1.0,1.0
            """
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = lambda: response_str
    inference_request.return_value = mock_response

    result = inference(credentials=mock_credentials, inference_pairs=mock_inference_pairs)

    assert result.equals(pd.read_csv(StringIO(response_str)))


@patch("hydrascreen.api.requests.post")
def test_inference_error_response(
    inference_request, mock_credentials: APICredentials, mock_inference_pairs
):
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json = lambda: {"detail": "Mock error"}
    inference_request.return_value = mock_response

    with pytest.raises(APIError):
        inference(credentials=mock_credentials, inference_pairs=mock_inference_pairs)


@patch("hydrascreen.api.requests.post")
def test_upload_pdb_successful_response(
    upload_pdb_request, mock_credentials: APICredentials, tmp_path
):
    mock_pdb_file = tmp_path / "protein.pdb"
    mock_pdb_file.touch()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"s3_key": "s3://test-bucket/protein.pdb"}
    upload_pdb_request.return_value = mock_response

    result = upload_pdb(credentials=mock_credentials, pdb_file=open(mock_pdb_file, "rb"))

    assert result == "s3://test-bucket/protein.pdb"


@patch("hydrascreen.api.requests.post")
def test_upload_pdb_error_response(upload_pdb_request, mock_credentials: APICredentials, tmp_path):
    mock_pdb_file = tmp_path / "protein.pdb"
    mock_pdb_file.touch()
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json = lambda: {"detail": "Mock error"}
    upload_pdb_request.return_value = mock_response

    with pytest.raises(APIError):
        upload_pdb(credentials=mock_credentials, pdb_file=open(mock_pdb_file, "rb"))


@patch("hydrascreen.api.requests.post")
def test_upload_sdf_successful_response(
    upload_sdf_request, mock_credentials: APICredentials, tmp_path
):
    mock_sdf_file = tmp_path / "docked_ligand.sdf"
    mock_sdf_file.touch()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"s3_key": "s3://test-bucket/docked_ligand.sdf"}
    upload_sdf_request.return_value = mock_response

    result = upload_sdf(
        credentials=mock_credentials,
        pdb_s3_path="s3://test-bucket/protein.pdb",
        sdf_file=open(mock_sdf_file, "rb"),
    )

    assert result == "s3://test-bucket/docked_ligand.sdf"


@patch("hydrascreen.api.requests.post")
def test_upload_sdf_error_response(upload_sdf_request, mock_credentials: APICredentials, tmp_path):
    mock_sdf_file = tmp_path / "docked_ligand.sdf"
    mock_sdf_file.touch()
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json = lambda: {"detail": "Mock error"}
    upload_sdf_request.return_value = mock_response

    with pytest.raises(APIError):
        upload_sdf(
            credentials=mock_credentials,
            pdb_s3_path="s3://test-bucket/protein.pdb",
            sdf_file=open(mock_sdf_file, "rb"),
        )
