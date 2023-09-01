import base64
from pathlib import Path
from typing import List
from hydrascreen.api import inference, upload_pdb, upload_sdf
import pandas as pd
from dataclasses import dataclass


@dataclass
class InferenceResults:
    ligand_affinity: pd.DataFrame
    pose_predictions: pd.DataFrame


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
        Splits the inference results to produce 2 dataframes, one with aggregated ligand affinity scores and
        one with pose confidence scores and pose affinity predictions.

        Args:
            results (pd.DataFrame): Results from the inference API call.
        Returns:
            results (InferenceResults): Two DataFrames under ligand_affinty and pose_predictions. Ligand affinity shows the aggregated affinity scores
            and pose predictions shows the pose confidence and pose affinity predictions.
        """
        ligand_affinity = results[results["pose_id"].isna()].drop(
            ["pose_id", "pose_confidence", "pose_affinity"], axis=1
        )

        pose_predictions = (
            results[results["ligand_affinity"].isna()]
            .drop(["ligand_affinity"], axis=1)
            .astype({"pose_id": int})
        )

        return InferenceResults(ligand_affinity=ligand_affinity, pose_predictions=pose_predictions)

    def predict_for_protein(self, protein_file: Path, ligand_files: List[Path]) -> InferenceResults:
        """
        Performs predictions for a given protein and a list of docked ligand files.

        Args:
            protein_file (Path): The file path of the protein PDB file.
            The protein .pdb file should only contain amino acids. Water, ions, and other cofactors are not presently allowed.
            ligand_files (List[Path]): A list of file paths for the docked SDF ligand files.
            Each .sdf should contain only one chemical compound, but may contain multiple poses thereof.
            The poses need to include all hydrogens and be in the proper protonation state (i.e. as used for docking).
            Only organic compounds are allowed at present.

        Returns:
            results (InferenceResults): Two DataFrames under ligand_affinty and pose_predictions. Ligand affinity shows the aggregated affinity scores
            and pose predictions shows the pose confidence and pose affinity predictions.

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
