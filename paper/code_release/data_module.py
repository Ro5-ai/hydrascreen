import shutil
from pathlib import Path
from typing import Optional, Union, List

import molgrid
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from molgrid import GridMaker, ExampleProvider

from bio3d.data import TypesDataset

import sys, re, os
from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Sampler, Dataset
from molgrid import float3
from molgrid import Grid1fCUDA, Grid4fCUDA, Transform, Grid1f, Grid4f


def count_lines(file_name: Union[Path, str]) -> int:
    """Get number of lines in a file

    Args:
        file_name: path of file

    Returns:
        int: number of lines in file
    """
    with open(file_name, "r") as f:
        return sum(1 for _ in f)


class BaseMolgridDataset(IterableDataset):
    """Base-class for using Molgrid-accellerated voxelization and sampling."""

    def __init__(
        self,
        example_provider,
        grid_maker,
        batch_size: int,
        num_channels: int,
        n_iters: int,
        use_radii: int,
        rotate: bool,
        translate: float,
        lig_padding: float = 2.0,
        device: str = "cuda",
    ) -> None:
        """Base molgrid dataloader.

        Args:
            example_provider (_type_): _description_
            grid_maker (_type_): _description_
            batch_size (int): _description_
            num_channels (int): _description_
            n_iters (int): _description_
            use_radii (int): _description_
            rotate (bool): _description_
            translate (float): _description_
            lig_padding (float, optional): _description_. Defaults to 2.0.

        Raises:
            ValueError: _description_
        """
        assert device in ["cpu", "cuda"]
        self.device = device
        if self.device == "cpu":
            self.grid1, self.grid4 = Grid1f, Grid4f
        else:
            self.grid1, self.grid4 = Grid1fCUDA, Grid4fCUDA
        self.ex_prov = example_provider
        self.rotate = rotate
        self.translate = translate
        self.gmaker = grid_maker
        self.batch_size = batch_size
        self.use_radii = use_radii
        self.lig_padding = lig_padding

        float_type = torch.float32
        self.float_type = float_type
        self.voxel_tensor = torch.zeros(
            (batch_size, num_channels, *self.gmaker.spatial_grid_dimensions()),
            dtype=float_type,
            device=device,
            requires_grad=False,
        )
        self.pose_tensor = torch.zeros(batch_size, dtype=float_type, device=device)
        self.activity_tensor = torch.zeros(batch_size, dtype=float_type, device=device)
        self.rmsd_tensor = torch.zeros(batch_size, dtype=float_type, device=device)
        self.dock_tensor = torch.zeros(batch_size, dtype=float_type, device=device)
        self.n_iters = n_iters

        super().__init__()

    def __len__(self):
        return self.n_iters

    @torch.no_grad()
    def __iter__(self):
        for i in range(self.n_iters):
            batch = self.ex_prov.next_batch(self.batch_size)
            # Extract label tensors
            batch.extract_label(0, self.pose_tensor)
            batch.extract_label(1, self.activity_tensor)
            batch.extract_label(2, self.rmsd_tensor)
            batch.extract_label(3, self.dock_tensor)
            # TODO extract set of samples according to distribution
            # sample set of indices in range(0, batch_size): [mini_batch_size] which we will sample from

            missed_ids = []
            for bi, example in enumerate(batch):
                # Extract props
                # TODO add some jitter to the coordinates!
                coords = example.merge_coordinates()
                center = example.coord_sets[1].center()
                n_lig_coords = example.coord_sets[1].coords.shape[0]

                if self.use_radii:
                    radii = coords.radii
                else:
                    radii = torch.ones(coords.size(), device=self.device, dtype=self.float_type)
                types = coords.type_index

                if n_lig_coords == 0:
                    # print(f"No ligand here {i, bi}, omiting for now but please fix!")
                    missed_ids.append(bi)
                else:
                    # Rotate then translate
                    if self.rotate:
                        rtrans = Transform(
                            center, random_translate=0.0, random_rotation=self.rotate
                        )
                        rtrans.forward(coords, coords)

                    # Translate
                    lig_coords = np.array(coords.coords)[-n_lig_coords:]

                    origin_lig_coords = lig_coords - np.array(list(center))
                    origin_lig_coords /= self.gmaker.get_resolution()

                    max_neg_xyz_trans = (
                        origin_lig_coords.min(0)
                        + self.gmaker.get_dimension()
                        - self.lig_padding / self.gmaker.get_resolution()
                    ) * self.gmaker.get_resolution()
                    max_pos_xyz_trans = (
                        self.gmaker.get_dimension()
                        - self.lig_padding / self.gmaker.get_resolution()
                        - origin_lig_coords.max(0)
                    ) * self.gmaker.get_resolution()
                    effective_max_neg_trans = -np.min(
                        np.array(
                            [max_neg_xyz_trans, np.ones_like(max_neg_xyz_trans) * self.translate]
                        ),
                        0,
                    )
                    effective_max_pos_trans = np.min(
                        np.array(
                            [max_pos_xyz_trans, np.ones_like(max_pos_xyz_trans) * self.translate]
                        ),
                        0,
                    )
                    # Uniform transformation
                    # sampled_trans = np.random.uniform(effective_max_neg_trans, effective_max_pos_trans, 3)

                    # Gaussian centered transformations
                    std_trans = (effective_max_pos_trans / 2 - effective_max_neg_trans / 2) / 2
                    mean_trans = (effective_max_pos_trans + effective_max_neg_trans) / 2
                    sampled_trans = mean_trans + np.random.randn(3) * std_trans
                    # Translate Given the limits
                    trans = Transform(center, random_translate=0.0, random_rotation=False)
                    trans.set_translation(float3(*sampled_trans))
                    trans.forward(coords, coords)

                # TODO ligand jitter (just a bit?)
                # TODO residue jitter (transform residue coordinates a bit)
                if self.device == "cpu":
                    self.gmaker.forward(
                        center,
                        coords.coords.cpu(),
                        types.cpu(),
                        self.grid1(radii),
                        self.grid4(self.voxel_tensor[bi]),
                    )
                else:
                    self.gmaker.forward(
                        center,
                        coords.coords,
                        types,
                        self.grid1(radii),
                        self.grid4(self.voxel_tensor[bi]),
                    )

            # Reduce points which are faulty
            self.pose_tensor[missed_ids] = 0
            self.activity_tensor[missed_ids] = 0
            self.rmsd_tensor[missed_ids] = 0
            self.dock_tensor[missed_ids] = 0

            yield (
                self.voxel_tensor,
                self.pose_tensor,
                self.activity_tensor,
                self.rmsd_tensor,
                self.dock_tensor,
            )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        types_file: str,
        batch_size: int,
        data_root: Optional[str] = None,
        num_channels: int = 42,
        rotate: bool = False,
        translate: float = 0.0,
        ligmolcache=None,
        recmolcache=None,
        rec_typer: Optional[molgrid.FileMappedGninaTyper] = None,
        lig_typer: Optional[molgrid.FileMappedGninaTyper] = None,
        num_workers: int = 0,
        device: str = "cuda",
    ):
        """Datamodule which wraps efficient molgrid-based balanced stratified sampling and voxelization on the CrossDocked2020 dataset

        Args:
            types_file (str): Path to the path type directory
            batch_size (int): Batch size.
            data_root (str): Directory in which to look for/to which to download data.
            num_channels (int): Number of channels for gridmaker. Is equal to protein & ligand channels (TO BE INCLUDED). Check for gninatype files in ro5_ml_torch/dataloaders/libmolgrid
        """
        self.batch_size = batch_size
        self.data_root = data_root
        self.rotate = rotate
        self.translate = translate
        self.num_channels = num_channels
        self.types_file = types_file
        self.ligmolcache = ligmolcache
        self.recmolcache = recmolcache
        self.device = device

        self.rec_typer = rec_typer
        self.lig_typer = lig_typer
        self.num_workers = num_workers

        # Cannot have data root and molcache at the same time!
        if data_root is None:
            assert (ligmolcache is not None) and (recmolcache is not None)
        else:
            assert (ligmolcache is None) and (recmolcache is None)

        super().__init__()

    def setup(self, stage: Optional[str] = None) -> None:
        gmaker = GridMaker(
            resolution=0.5, dimension=24.0, gaussian_radius_multiple=0.5, radius_scale=0.8
        )

        # Set training datasets
        if self.data_root is not None:
            exprov = ExampleProvider(
                self.rec_typer,
                self.lig_typer,
                balanced=False,
                shuffle=False,
                data_root=self.data_root,
            )
        else:
            exprov = ExampleProvider(
                self.rec_typer,
                self.lig_typer,
                balanced=False,
                shuffle=False,
                ligmolcache=self.ligmolcache,
                recmolcache=self.recmolcache,
            )
        exprov.populate(self.types_file)

        self.dataset = BaseMolgridDataset(
            exprov,
            gmaker,
            self.batch_size,
            self.num_channels,
            count_lines(self.types_file) // self.batch_size + 1,
            False,
            self.rotate,
            self.translate,
            device=self.device,
        )

        # Set up the type tracker
        self.type_dataset = TypesDataset(self.types_file)
        print("Setup complete... \n")

    def dataloader(self):
        return DataLoader(self.dataset, None, num_workers=self.num_workers)
