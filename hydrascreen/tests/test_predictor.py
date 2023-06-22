from unittest.mock import patch
from hydrascreen.api import APICredentials
from hydrascreen.predictor import HydraScreen


@patch("hydrascreen.predictor.inference")
@patch("hydrascreen.predictor.upload_sdf")
@patch("hydrascreen.predictor.upload_pdb")
def test_predict_for_protein(
    upload_pdb,
    upload_sdf,
    inference,
    mock_credentials: APICredentials,
    tmp_path,
):
    hydrascreen_predictor = HydraScreen(api_credentials=mock_credentials)
    mock_protein_file = tmp_path / "protein.pdb"
    mock_ligand_file1 = tmp_path / "docked_ligand1.sdf"
    mock_ligand_file2 = tmp_path / "docked_ligand2.sdf"
    mock_protein_file.touch()
    mock_ligand_file1.touch()
    mock_ligand_file2.touch()
    upload_pdb.return_value = "s3://test-bucket/protein.pdb"
    upload_sdf.side_effect = [
        "s3://test-bucket/docked_ligand1.sdf",
        "s3://test-bucket/docked_ligand2.sdf",
    ]
    mock_inference_pairs = [
        {
            "protein": "s3://test-bucket/protein.pdb",
            "docked_ligand": "s3://test-bucket/docked_ligand1.sdf",
        },
        {
            "protein": "s3://test-bucket/protein.pdb",
            "docked_ligand": "s3://test-bucket/docked_ligand2.sdf",
        },
    ]

    hydrascreen_predictor.predict_for_protein(
        protein_file=mock_protein_file, ligand_files=[mock_ligand_file1, mock_ligand_file2]
    )

    upload_pdb.assert_called_once()
    assert upload_sdf.call_count == 2
    inference.assert_called_once_with(
        credentials=mock_credentials, inference_pairs=mock_inference_pairs
    )
