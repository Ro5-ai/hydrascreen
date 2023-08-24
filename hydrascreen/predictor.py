import base64
from pathlib import Path
from typing import List
from hydrascreen.api import inference, upload_pdb, upload_sdf
import pandas as pd
from dataclasses import dataclass


@dataclass
class InferenceResults:
    affinity: pd.DataFrame
    pose: pd.DataFrame


class HydraScreen:
    def __init__(self, token: str) -> None:
        """
        Initializes a new instance of the HydraScreen class.

        Args:
            token (str): base64 encoded jwt token provided for usage of the api
        """
        self.token = base64.b64decode(token).decode("utf-8")

    @staticmethod
    def _split_inference_results(results: pd.DataFrame) -> InferenceResults:
        """
        Splits the inference results to produce 2 dataframes, one with aggregated affinity scores and one with pose scores.

        Args:
            results (pd.DataFrame): Results from the inference API call.
        Returns:
            results (InferenceResults): Two DataFrames under affinity and pose. Affinity shows the aggregated affinity scores
            and pose shows the pose scores.
        """
        affinity = results[results["pose_id"].isna()].drop(["pose_id", "pose_confidence"], axis=1)

        pose = (
            results[results["affinity"].isna()].drop(["affinity"], axis=1).astype({"pose_id": int})
        )

        return InferenceResults(affinity=affinity, pose=pose)

    def predict_for_protein(self, protein_file: Path, ligand_files: List[Path]) -> InferenceResults:
        """
        Performs predictions for a given protein and a list of docked ligand files.

        Args:
            protein_file (Path): The file path of the protein PDB file. PDB file needs to include explicit hydrogens and charges,
            and to be void of waters, metal ions, and salts.
            ligand_files (List[Path]): A list of file paths for the docked SDF ligand files.
            Ligand files must contain a single molecule per file with one or more docked poses, with all hydrogens and charges.

        Returns:
            results (InferenceResults): Two DataFrames under affinity and pose. Affinity shows the aggregated affinity scores
            and pose shows the pose scores.

        Raises:
            FileUploadError: If there is an error in uploading a file.
        """

        with open(protein_file, "rb") as f:
            pdb_s3_path = upload_pdb(pdb_file=f, token=self.token)

        ligand_s3_paths = []
        for ligand_file in ligand_files:
            with open(ligand_file, "rb") as f:
                ligand_s3_path = upload_sdf(
                    pdb_s3_path=pdb_s3_path,
                    sdf_file=f,
                    token=self.token,
                )
            ligand_s3_paths.append(ligand_s3_path)

        inference_pairs = [
            {"protein": pdb_s3_path, "docked_ligand": ligand_s3_path}
            for ligand_s3_path in ligand_s3_paths
        ]

        results = inference(inference_pairs=inference_pairs, token=self.token)

        return self._split_inference_results(results)
