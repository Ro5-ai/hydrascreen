from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn

from preprocessor import Preprocessor

NORMALIZER = {
    "affinity": {
        "mean": 6.461,
        "std": 1.691,
    },
    "rmsd": {
        "mean": 4,
        "std": 4,
        "raw_bounds": [0, 8],
    },
    "docking": {
        "mean": -6.5,
        "std": 2.5,
    }
}


class MolgridModel(pl.LightningModule):
    """Base class for Bioactivity prediction models that use Molgrid-based dataloaders (e.g. :class:`CrossDockedVoxelizingDataModule`).

    Does not implement optimizers, but assumes hinged regression loss.

    Args:
        model (nn.Module): Bioactivity model that maps a voxel grid to a pose classification and bioactivity prediction output.
        mirror_prob (float): mirror probability for the voxel tensors
    """

    def __init__(
            self,
            model: nn.Module,
            std_max: float = 3,
            mirror_prob: float = 0.5,
            normalizer=None,
    ):
        super().__init__()
        if normalizer is None:
            normalizer = NORMALIZER
        self.normalizer = normalizer
        self.std_max = std_max
        self.preprocessor = Preprocessor(norm_config=normalizer, mirror_prob=mirror_prob, gsm=0, std_max=2 * self.std_max)
        self.model = model


    def predict_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            voxel = torch.cat([v[0] for v in batch.values()])
            pose_labels = torch.cat([v[1] for v in batch.values()])
            activity = torch.cat([v[2] for v in batch.values()])
            rmsd = torch.cat([v[3] for v in batch.values()])
            dock = torch.cat([v[4] for v in batch.values()])
        else:
            voxel, pose_labels, activity, rmsd, dock = batch
        voxel, activity, rmsd, dock = self.preprocessor(voxel, activity, rmsd, dock)
        y_hat = self.model(voxel)
        return y_hat, torch.stack([pose_labels, activity, rmsd, dock])
