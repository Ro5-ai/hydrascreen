import dataclass
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from pathlib import Path
import struct
from typing import Dict, List, Optional, Union
from warnings import warn

from openbabel import pybel
import pandas as pd
import molgrid
import numpy as np
from gridData import Grid
from pkg_resources import resource_filename


gninatypes = list(molgrid.GninaIndexTyper().get_type_names())

""" Gninatypes from codebase:
['Hydrogen',
 'PolarHydrogen',
 'AliphaticCarbonXSHydrophobe',
 'AliphaticCarbonXSNonHydrophobe',
 'AromaticCarbonXSHydrophobe',
 'AromaticCarbonXSNonHydrophobe',
 'Nitrogen',
 'NitrogenXSDonor',
 'NitrogenXSDonorAcceptor',
 'NitrogenXSAcceptor',
 'Oxygen',
 'OxygenXSDonor',
 'OxygenXSDonorAcceptor',
 'OxygenXSAcceptor',
 'Sulfur',
 'SulfurAcceptor',
 'Phosphorus',
 'Fluorine',
 'Chlorine',
 'Bromine',
 'Iodine',
 'Magnesium',
 'Manganese',
 'Zinc',
 'Calcium',
 'Iron',
 'GenericMetal',
 'Boron']
"""

gninatype_to_atom_num = {
    "Hydrogen": 1,
    "PolarHydrogen": 1,
    "AliphaticCarbonXSHydrophobe": 6,
    "AliphaticCarbonXSNonHydrophobe": 6,
    "AromaticCarbonXSHydrophobe": 6,
    "AromaticCarbonXSNonHydrophobe": 6,
    "Nitrogen": 7,
    "NitrogenXSDonor": 7,
    "NitrogenXSDonorAcceptor": 7,
    "NitrogenXSAcceptor": 7,
    "Oxygen": 8,
    "OxygenXSDonor": 8,
    "OxygenXSDonorAcceptor": 8,
    "OxygenXSAcceptor": 8,
    "Sulfur": 16,
    "SulfurAcceptor": 16,
    "Phosphorus": 15,
    "Fluorine": 9,
    "Chlorine": 17,
    "Bromine": 35,
    "Iodine": 53,
    "Magnesium": 12,
    "Manganese": 25,
    "Zinc": 30,
    "Calcium": 20,
    "Iron": 26,
    "Boron": 6,
    "GenericMetal": 0,
}

warn(
    "Hardcoding atom types. Please only use this if the ligand-based grid is suppossed to operate at the element level. This will fail if we have i.e. aromatic and aliphatic carbons on different channels."
)
atom_num_to_gninatype_name = {
    0: "GenericMetal",
    1: "Hydrogen",
    5: "Boron",
    6: "AliphaticCarbonXSHydrophobe",
    7: "Nitrogen",
    8: "Oxygen",
    9: "Fluorine",
    12: "Magnesium",
    15: "Phosphorus",
    16: "Sulfur",
    17: "Chlorine",
    20: "Calcium",
    25: "Manganese",
    26: "Iron",
    30: "Zinc",
    35: "Bromine",
    53: "Iodine",
}


class GeneralTyper(molgrid.FileMappedGninaTyper):
    def __init__(self, type_file: str):
        types_path = resource_filename(__name__, f"{type_file}.typedef")
        super(GeneralTyper, self).__init__(types_path)


atom_num_to_gninatype_index = {
    atom_num: gninatypes.index(gninatype)
    for atom_num, gninatype in atom_num_to_gninatype_name.items()
}


@contextmanager
def _suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def make_dx(grid: np.ndarray, center: np.ndarray, out_path: str) -> None:
    """Create a .dx file of a voxel grid. Assumes a 49x49x49 grid with resolution of 0.5A.

    Args:
        grid (np.ndarray): Shape (D, D, D) grid to write to a .dx file
        center (np.ndarray): shape (3,) array containing coordinates of box center
        out_path (str): path to which to write the dx grid
    """
    g = Grid(grid, origin=center - 12.25, delta=np.array((0.5, 0.5, 0.5)))
    g.export(out_path, "DX")


@dataclass
class AtomCoordinates:
    x: float
    y: float
    z: float
    atomic_num: int

    @property
    def gninatype_index(self):
        if self.atomic_num in atom_num_to_gninatype_index:
            return atom_num_to_gninatype_index.get(self.atomic_num)
        else:
            return -1


# copied with slight signature change from: https://github.com/devalab/DeepPocket/blob/main/types_and_gninatyper.py
def pdb_to_gninatype(pdb_file: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    # creates gninatype file for model input

    types_file = Path(output_dir) / f"{Path(pdb_file).stem}.types"
    gninatypes_file = Path(output_dir) / f"{Path(pdb_file).stem}.gninatypes"

    with open(types_file, "w") as f:
        f.write(str(pdb_file))

    atom_map = molgrid.GninaIndexTyper()
    dataloader = molgrid.ExampleProvider(atom_map, shuffle=False, default_batch_size=1)
    dataloader.populate(str(types_file))
    example = dataloader.next()
    coords = example.coord_sets[0].coords.tonumpy()
    types = example.coord_sets[0].type_index.tonumpy()
    types = np.int_(types)
    with open(gninatypes_file, "wb") as f:
        for i in range(coords.shape[0]):
            f.write(struct.pack("fffi", coords[i][0], coords[i][1], coords[i][2], types[i]))
    os.remove(str(types_file))
    return gninatypes_file


def sdf_to_gninatypes(sdf_file: Union[str, Path], output_dir: Union[str, Path]) -> List[Path]:
    sdf_file, output_dir = Path(sdf_file), Path(output_dir)
    conformer_position_types = sdf_to_xyz_atomic(sdf_file=sdf_file)
    if len(conformer_position_types[0]) <= 1:
        raise RuntimeWarning(
            f"Cannot process {sdf_file}. Molecule has too few atoms: {conformer_position_types[0]} atoms."
        )

    output_file_names = []
    for conformer_index, conformer in enumerate(conformer_position_types):
        output_file_name = output_dir / f"{sdf_file.stem}_{conformer_index}.gninatypes"
        with open(output_file_name, "wb") as f:
            for atom_coordinates in conformer:
                if atom_coordinates.gninatype_index != -1:
                    f.write(
                        struct.pack(
                            "fffi",
                            atom_coordinates.x,
                            atom_coordinates.y,
                            atom_coordinates.z,
                            atom_coordinates.gninatype_index,
                        )
                    )
                else:
                    print(f"Skipping atom coordinate {atom_coordinates.atomic_num}")
        output_file_names.append(output_file_name)
    return output_file_names


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


warn(
    "We are adding Hydrogens using pybel. This might not be suitable in some conditions. Please revise this in the future."
)
warn("Also we are adding hydrogens but are not modelling the hydrogens in the protein... no need!")


def sdf_to_xyz_atomic(sdf_file: Union[str, Path]) -> List[List[AtomCoordinates]]:
    """
    Returns:
        List[List[List[AtomCoordWithGninatype]]]: list of conformers from sdf file and for each conformer all the atom coordinates
    """
    out = pybel.readfile("sdf", str(sdf_file))
    mols = list(out)
    all_mols = []
    for mol in mols:
        # This should not add anything, the sdf should have all implicit hydrogens, but JIC!
        mol.addh()
        res_mol = [
            AtomCoordinates(a.coords[0], a.coords[1], a.coords[2], a.atomicnum) for a in mol.atoms
        ]
        all_mols.append(res_mol)
    return all_mols
