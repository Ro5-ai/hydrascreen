from pathlib import Path
from typing import List
from hydrascreen.api import APICredentials, APIError, inference, upload_pdb, upload_sdf
import pandas as pd


class FileUploadError(Exception):
    pass


class HydraScreen:
    def __init__(self, api_credentials: APICredentials) -> None:
        """
        Initializes a new instance of the HydraScreen class.

        Args:
            api_credentials (APICredentials): The API credentials object.
        """
        self.api_credentials = api_credentials

    def predict_for_protein(self, protein_file: Path, ligand_files: List[Path]) -> pd.DataFrame:
        """
        Performs predictions for a given protein and a list of docked ligand files.

        Args:
            protein_file (Path): The file path of the protein file.
            ligand_files (List[Path]): A list of file paths for the docked ligand files.

        Returns:
            pd.DataFrame: A DataFrame containing the predictions.

        Raises:
            FileUploadError: If there is an error in uploading a file.
        """
        try:
            with open(protein_file, "rb") as f:
                pdb_s3_path = upload_pdb(credentials=self.api_credentials, pdb_file=f)
        except APIError:
            raise FileUploadError(f"Failed to upload {protein_file}")

        ligand_s3_paths = []
        for ligand_file in ligand_files:
            try:
                with open(ligand_file, "rb") as f:
                    ligand_s3_path = upload_sdf(
                        credentials=self.api_credentials,
                        pdb_s3_path=pdb_s3_path,
                        sdf_file=f,
                    )
                ligand_s3_paths.append(ligand_s3_path)
            except APIError:
                raise FileUploadError(f"Failed to upload {ligand_file}")

        inference_pairs = [
            {"protein": pdb_s3_path, "docked_ligand": ligand_s3_path}
            for ligand_s3_path in ligand_s3_paths
        ]

        return inference(credentials=self.api_credentials, inference_pairs=inference_pairs)
