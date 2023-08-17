[![Tests](https://github.com/Ro5-ai/hydrascreen/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Ro5-ai/hydrascreen/actions/workflows/run_tests.yml)
# Hydrascreen

This codebase provides functionality for making predictions using the HydraScreen API. It allows users to upload protein and ligand files, perform predictions, and retrieve the predicted affinity and pose confidence for each prediction. The GUI tool with the same functionality can be found here: [HydraScreen GUI](https://hydrascreen.ro5.ai/).


## Installation

Install hydrascreen as a pip installable package:
```bash
pip install hydrascreen 
```

## Usage

### Login

First login to hydrascreen by providing your email and your organization.

```python
from hydrascreen import login

predictor = login(
    email='user@email.com', 
    organization='User Org'
    )
```

### Getting predictions

Call the `predict_for_protein` function to get predictions for your docked protein-ligand pairs.

`protein_file` needs to be a `Path` object for a PDB file. Protein .pdb file needs to include explicit hydrogens and charges, and to be void of waters, metal ions, and salts.

`ligand_files` needs to be a list of `Path` objects for docked SDF files. Ligand files must contain a single molecule per file with one or more docked poses, with all hydrogens and charges.


```python
from pathlib import Path

results = predictor.predict_for_protein(
            protein_file=Path('/path/to/protein.pdb'), 
            ligand_files=[
                Path('/path/to/ligand1.sdf'), 
                Path('/path/to/ligand2.sdf')
                ]
            ) 
```

The output will be a `results` dataclass with 2 entries which are `pandas DataFrames` for your protein-ligand pair predictions:
- **results.affinity**: aggregated affinity scores of each protein-ligand complex
- **results.pose**: pose scores for each pose separately

If you want to run multiple proteins with their ligands you can use the code as follows:

```python 
from pathlib import Path

input_pairs = [
    {
        "protein_file": Path('/path/to/protein1.pdb'), 
        "ligand_files": [
            Path('/path/to/ligand1.sdf'), 
            Path('/path/to/ligand2.sdf')
            ]
    },
    {
        "protein_file": Path('/path/to/protein2.pdb'), 
        "ligand_files": [
            Path('/path/to/ligand3.sdf'), 
            Path('/path/to/ligand4.sdf')
            ]
    }
]

affinities = []
poses = []
for input_pair in input_pairs:
    results = predictor.predict_for_protein(**input_pair)
    affinities.append(results.affinity)
    poses.append(results.pose)
```

The output will be 2 lists of `pandas DataFrames` with the prediction results for your protein-ligand pairs.

### Outputs

Below is an example of the resulting affinity and pose DaraFrames for a protein and 2 docked ligands, with 2 and 3 docked poses respectively.

#### Affinity
Columns:
 - **pdb_id**: Name of the protein the ligands are docked to (provided protein PDB file name).
 - **ligand_id**: Name of the ligand docked to the pdb_id protein (provided ligand SDF file name).
 - **affinity**: Predicted affinity of protein-ligand pair overall, expressed in pKi units.
```csv
pdb_id,  ligand_id,                affinity,           
protein, protein_docked_ligand_0,  0.84967568666
protein, protein_docked_ligand_1,  0.8498707
```

#### Pose
Columns:
 - **pdb_id**: Name of the protein the ligands are docked to (provided protein PDB file name).
 - **ligand_id**: Name of the ligand docked to the pdb_id protein (provided ligand SDF file name).
 - **pose_id**: Sequential pose number based on the order of the docked ligand poses in the SDF file.
 - **pose_confidence**: Pose confidence, ranging from low (0) to high (1), indicating the model's confidence that the pose could be the true, protein-ligand co-crystal structure. Note that this is solely based on the model's prediction and not a direct comparison with an existing co-crystal structure.
```csv
pdb_id,  ligand_id,             pose_id,  pose_confidence
protein, protein_docked_ligand_0,  0, 0.9360706533333333
protein, protein_docked_ligand_0,  1, 0.9487579333333334
protein, protein_docked_ligand_1,  0, 0.8837728666666665
protein, protein_docked_ligand_1,  1, 0.9275542666666666
protein, protein_docked_ligand_1,  2, 0.8115468833333334
```

## Development

Install the requirements:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## License
HydraScreen is available restricted to Non-Commercial Use. For more information see the LICENSE file.
