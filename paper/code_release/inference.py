#!/usr/bin/env python

import argparse
import datetime
import glob
import logging
import os
import time
from dataclasses import dataclass
from typing import Union, List, Dict
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from molgrid_model import MolgridModel
from data_module import DataModule
from utils import pdb_to_gninatype, sdf_to_gninatypes, GeneralTyper
from tqdm import tqdm

logger = logging.getLogger(__name__)

PAPER_ROOT = Path(__file__).parent.parent
REC_TYPER = str(Path(__file__).parent / "typers" / "rectyper")
LIG_TYPER = str(Path(__file__).parent / "typers" / "ligtyper")


def _pdb_to_gninatypes(pdb_path: Path, gninatype_output_path: Path) -> Path:
    if not pdb_path.exists():
        raise AssertionError(f"pdb file {pdb_path} does not exist")
    pdb_gnina_file = pdb_to_gninatype(pdb_file=pdb_path, output_dir=gninatype_output_path)
    return pdb_gnina_file


def _sdf_to_gninatypes(ligand_paths, gninatype_output_path: Path) -> List[str]:
    # TODO paralellise on CPUs
    sdf_gnina_files = []
    logger.info(f"Transforming sdf to gninatypes files for total of {len(ligand_paths)} sdf files")
    for sdf_file in tqdm(ligand_paths):
        try:
            sdf_gnina_files += sdf_to_gninatypes(sdf_file, output_dir=gninatype_output_path)
        except Exception as e:
            logger.exception(
                f"Couldn't process sdf file {sdf_file} and thus skipping it. Error:", e
            )
    return sdf_gnina_files


def _setup_inference(types_file: Path, batch_size: int) -> DataLoader:
    rec_typer = GeneralTyper(REC_TYPER)
    lig_typer = GeneralTyper(LIG_TYPER)
    num_channels = rec_typer.num_types() + lig_typer.num_types()
    datamodule = DataModule(
        types_file=str(types_file),
        batch_size=batch_size,
        data_root="",
        num_channels=num_channels,
        rotate=False,
        translate=0.0,
        rec_typer=rec_typer,
        lig_typer=lig_typer,
    )
    datamodule.setup("predict")
    dataloader = datamodule.dataloader()
    return dataloader


def _model_result_to_df(types_df: pd.DataFrame, outs: List[torch.Tensor]) -> pd.DataFrame:
    # Average results for ensemble
    num_ligands = len(types_df)
    results = torch.cat(outs, dim=0)
    results = results[:num_ligands]
    results[:, 0] = torch.sigmoid(results[:, 0])
    result_df = pd.DataFrame.from_records(
        results.cpu().numpy(), columns=["pose", "pki", "rmsd", "dock_score"]
    )
    result_df["ligand_conformer_id"] = types_df["ligand_gninatype"].apply(lambda p: Path(p).stem)
    result_df.set_index("ligand_conformer_id", inplace=True)
    return result_df


def _store_results(result_df: pd.DataFrame, model_output_dir: Path, pdb_id: str):
    model_output_dir.mkdir(exist_ok=True)
    result_df.to_csv(model_output_dir / f"{pdb_id}.csv", index=True)


def _get_data_from_csvs(output_dir, filename_column_to_add):
    aggregated_df = pd.DataFrame()
    for csv in output_dir.glob("*.csv"):
        df = pd.read_csv(csv)
        df[filename_column_to_add] = csv.stem
        aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
    return aggregated_df


def _merge_and_save_csv_in_dir(input_csv_dir: Path, output_dir: Path, filename_column_to_add: str):
    for model_dir in input_csv_dir.glob("*"):
        pdb_results_df = _get_data_from_csvs(
            model_dir, filename_column_to_add=filename_column_to_add
        )
        pdb_results_df.to_csv(f"{output_dir / model_dir.name}.csv", index=False)


def make_types_file(
    sdf_gninatypes: List[Union[str, Path]],
    pdb_gninatypes: Union[str, Path],
    out_file: Union[str, Path],
):
    tmp = [0] * len(sdf_gninatypes)
    types_df = pd.DataFrame(
        dict(
            pose=tmp,
            affinity=tmp,
            rmsd=tmp,
            docking=tmp,
            protein_gninatype=[str(pdb_gninatypes)] * len(sdf_gninatypes),
            ligand_gninatype=sdf_gninatypes,
        )
    )
    types_df.to_csv(out_file, sep=" ", index=False, header=False)
    return types_df


@dataclass
class CheckpointModel:
    checkpoint_path: Path
    model: MolgridModel


def run_inference(
    models: List[CheckpointModel], dataloader: DataLoader
) -> Dict[Path, List[torch.Tensor]]:
    # Feed input to the model -> Voxel size (BATCH_SIZE, N (usually 21), 49, 49, 49)
    outs = {}
    for i, batch in enumerate(dataloader):
        for m in models:
            if m.checkpoint_path not in outs:
                outs[m.checkpoint_path] = []
            out, _ = m.model.predict_step(batch, i)
            outs[m.checkpoint_path].append(out)
    return outs


def copy_files(source_paths: pd.Series, dest_dir: Path):
    # copy local files to destination directory
    for source_path in source_paths:
        source_path = Path(source_path)
        dest_path = dest_dir / source_path.name
        dest_path.write_bytes(source_path.read_bytes())


class ModelInference:
    def __init__(self, model_paths: List[Union[str, Path]]):
        model_checkpoint_paths = [Path(model_uri) for model_uri in model_paths]
        logger.info(f"Loading following models for inference: {model_checkpoint_paths}")
        self.models_checkpoints = []
        for model_path in model_paths:
            checkpoint_model = CheckpointModel(
                Path(model_path),
                MolgridModel.load_from_checkpoint(
                    checkpoint_path=str(model_path), device="cuda:0", strict=False
                )
                .eval()
                .cuda(),
            )
            checkpoint_model.model.freeze()
            # Set up posprocessor WITHOUT mirroring for deterministic predictions
            checkpoint_model.model.preprocessor.mirror_dist = torch.distributions.Bernoulli(0.0)
            checkpoint_model.model.setup("predict")
            self.models_checkpoints.append(checkpoint_model)

    def predict(self, model_input: pd.DataFrame, output_dir: Path):
        """

        Args:
            model_input: dataframe with columns "protein" and "docked_ligand" containing local paths to the protein and ligand files

        Returns:

        """

        logger.info(f"Running inference for {len(model_input)} protein-ligand pairs.")
        batch_size = int(os.getenv("INFERENCE_BATCH_SIZE", 200))

        output_dir.mkdir(exist_ok=True, parents=True)
        work_dir = output_dir / str(time.time())
        logger.info(f"Storing temp gninatype and types files in {work_dir}")
        gninatype_dir = Path(work_dir) / "gninatypes"
        gninatype_dir.mkdir()
        output_dir = work_dir / "output"
        output_dir.mkdir()
        protein_dir = work_dir / "proteins"
        protein_dir.mkdir()
        ligand_dir = work_dir / "ligands"
        ligand_dir.mkdir()

        copy_files(source_paths=pd.unique(model_input["protein"]), dest_dir=protein_dir)
        copy_files(source_paths=pd.unique(model_input["docked_ligand"]), dest_dir=ligand_dir)

        for pdb_path in protein_dir.rglob("**/*.pdb"):
            pdb_id = pdb_path.stem  # NOTE: there is an assumption that protein filenames are unique
            out_types_file = work_dir / f"{pdb_id}.types"

            pdb_gnina_file = _pdb_to_gninatypes(
                pdb_path=pdb_path, gninatype_output_path=gninatype_dir
            )
            ligand_gnina_files = _sdf_to_gninatypes(
                ligand_paths=list(ligand_dir.rglob(f"**/{pdb_id}*.sdf")),
                gninatype_output_path=gninatype_dir,
            )

            types_df = make_types_file(
                sdf_gninatypes=ligand_gnina_files,
                pdb_gninatypes=pdb_gnina_file,
                out_file=out_types_file,
            )

            batch_size = min(len(types_df), batch_size)
            dataloader = _setup_inference(types_file=out_types_file, batch_size=batch_size)
            logger.info(f"Will start inference: {datetime.datetime.now().isoformat()}")
            all_model_results = run_inference(models=self.models_checkpoints, dataloader=dataloader)

            logger.info(f"Saving model predictions to {output_dir}")
            for model_checkpoint_path, model_result in all_model_results.items():
                result_df = _model_result_to_df(types_df, model_result)
                _store_results(
                    result_df,
                    model_output_dir=output_dir / model_checkpoint_path.stem,
                    pdb_id=pdb_id,
                )
            logger.info(f"Finished inference for protein '{pdb_id}'.")
        logger.info(f"Will merge results: {datetime.datetime.now().isoformat()}")

        _merge_and_save_csv_in_dir(
            input_csv_dir=output_dir, output_dir=output_dir, filename_column_to_add="pdb_id"
        )
        result = _get_data_from_csvs(output_dir, filename_column_to_add="model_id")
        logger.info(f"Finished predict: {datetime.datetime.now().isoformat()}")

        return result


if __name__ == "__main__":
    model = ModelInference(model_paths=glob.glob(str(PAPER_ROOT / "model" / "*.ckpt")))

    docked_ligands_dir = PAPER_ROOT / "example_input" / "ligands"
    proteins_dir = PAPER_ROOT / "example_input" / "pdbs"

    model.predict(
        pd.DataFrame(
            {
                "protein": glob.glob(str(proteins_dir / "*.pdb")),
                "docked_ligand": glob.glob(str(docked_ligands_dir / "*.sdf")),
            }
        ),
        output_dir=Path(f"run_{datetime.datetime.now()}"),
    )
