from pathlib import Path
from typing import List, Tuple
from hydrascreen.api import APICredentials, inference, upload_pdb, upload_sdf
import pandas as pd


class HydraScreen:
    def __init__(self, api_credentials: APICredentials) -> None:
        """
        Initializes a new instance of the HydraScreen class.

        Args:
            api_credentials (APICredentials): The API credentials object.
        """
        self.api_credentials = api_credentials

    @staticmethod
    def _split_inference_results(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the inference results to produce 2 dataframes, one with aggregated affinity scores and one with pose scores.

        Args:
            results (pd.DataFrame): Results from the inference API call.
        Returns:
            affinity (pd.DataFrame): DataFrame with the aggregated affinity scores for each protein-ligand complex.
            pose (pd.DataFrame): DataFrame with the pose scores for each pose separately.
        """
        affinity = results[results["pose_id"].isna()].drop(["pose_id", "pose"], axis=1)

        pose = results[results["pki"].isna()].drop(["pki"], axis=1).astype({"pose_id": int})

        return affinity, pose

    def predict_for_protein(
        self, protein_file: Path, ligand_files: List[Path]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs predictions for a given protein and a list of docked ligand files.

        Args:
            protein_file (Path): The file path of the protein PDB file. PDB file needs to include explicit hydrogens and charges,
            and to be void of waters, metal ions, and salts.
            ligand_files (List[Path]): A list of file paths for the docked SDF ligand files.
            Ligand files must contain a single molecule per file with one or more docked poses, with all hydrogens and charges.

        Returns:
            affinity (pd.DataFrame): DataFrame with the aggregated affinity scores for each protein-ligand complex.
            pose (pd.DataFrame): DataFrame with the pose scores for each pose separately.

        Raises:
            FileUploadError: If there is an error in uploading a file.
        """

        with open(protein_file, "rb") as f:
            pdb_s3_path = upload_pdb(credentials=self.api_credentials, pdb_file=f)

        ligand_s3_paths = []
        for ligand_file in ligand_files:
            with open(ligand_file, "rb") as f:
                ligand_s3_path = upload_sdf(
                    credentials=self.api_credentials,
                    pdb_s3_path=pdb_s3_path,
                    sdf_file=f,
                )
            ligand_s3_paths.append(ligand_s3_path)

        inference_pairs = [
            {"protein": pdb_s3_path, "docked_ligand": ligand_s3_path}
            for ligand_s3_path in ligand_s3_paths
        ]

        results = inference(credentials=self.api_credentials, inference_pairs=inference_pairs)

        return self._split_inference_results(results)
