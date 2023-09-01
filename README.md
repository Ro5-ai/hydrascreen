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

login(
    email='user@email.com', 
    organization='User Org'
    ) # open your email to get token for following steps
```

### Getting predictions

Call the `predict_for_protein` function to get predictions for your docked protein-ligand pairs.

`protein_file` needs to be a `Path` object for a PDB file. The protein .pdb file should only contain amino acids. Water, ions, and other cofactors are not presently allowed.

`ligand_files` needs to be a list of `Path` objects for docked SDF files. Each .sdf should contain only one chemical compound, but may contain multiple poses thereof. The poses need to include all hydrogens and be in the proper protonation state (i.e. as used for docking). Only organic compounds are allowed at present.
​


```python
from pathlib import Path
from hydrascreen.predictor import HydraScreen

predictor = HydraScreen("ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmxiV0ZwYkNJNkluUmxjM1JBWlcxaGFXd3VZMjl0SWl3aWIzSm5Jam9pVFhrZ1QzSm5JaXdpWlhod0lqb3hOamsxTkRZeU16VTNmUS5Xd202VEJ1ZDQxRm5MY18yWFpNYS13c19qN0JqS1kzZkN3QnpSS3phVnZj") # replace with token received from email
results = predictor.predict_for_protein(
            protein_file=Path('/path/to/protein.pdb'), 
            ligand_files=[
                Path('/path/to/ligand1.sdf'), 
                Path('/path/to/ligand2.sdf')
                ]
            ) 
```

The output will be a `results` dataclass with 2 entries which are `pandas DataFrames` for your protein-ligand pair predictions:
- **results.ligand_affinity**: aggregated affinity scores of each protein-ligand complex
- **results.pose_predictions**: pose confidence scores and pose affinity predictions for each pose separately

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

ligand_affinities = []
poses_predictions = []
for input_pair in input_pairs:
    results = predictor.predict_for_protein(**input_pair)
    ligand_affinities.append(results.ligand_affinity)
    poses_predictions.append(results.pose_predictions)
```

The output will be 2 lists of `pandas DataFrames` with the prediction results for your protein-ligand pairs.

### Outputs

Below is an example of the resulting affinity and pose DaraFrames for a protein and 2 docked ligands, with 2 and 3 docked poses respectively.

#### Ligand Affinity
Columns:
 - **pdb_id**: Name of the protein the ligands are docked to (provided protein PDB file name).
 - **ligand_id**: Name of the ligand docked to the pdb_id protein (provided ligand SDF file name).
 - **ligand_affinity**: Overall ligand affinity, expressed in pKi units, is obtained from the aggregation of the predicted pose affinities, weighted according to the Boltzmann distribution of the pose confidence score
```csv
pdb_id,  ligand_id,                ligand_affinity,           
protein, protein_docked_ligand_0,  8.496
protein, protein_docked_ligand_1,  8.498
```

#### Pose Predictions
Columns:
 - **pdb_id**: Name of the protein the ligands are docked to (provided protein PDB file name).
 - **ligand_id**: Name of the ligand docked to the pdb_id protein (provided ligand SDF file name).
 - **pose_id**: Sequential pose number based on the order of the docked ligand poses in the SDF file.
 - **pose_confidence**: Pose confidence, ranging from low (0) to high (1), indicates the model's confidence that the pose could be the true, protein-ligand co-crystal structure. Note that this is solely based on the model's prediction and not a direct comparison with an existing co-crystal structure.
 - **pose_affinity**: Predicted affinity of the protein-pose pair, expressed in pKi units.
```csv
pdb_id,  ligand_id,             pose_id,  pose_confidence, pose_affinity
protein, protein_docked_ligand_0,  0, 0.9360706533333333, 7.694
protein, protein_docked_ligand_0,  1, 0.9487579333333334, 7.691
protein, protein_docked_ligand_1,  0, 0.8837728666666665, 7.248
protein, protein_docked_ligand_1,  1, 0.9275542666666666, 3.356
protein, protein_docked_ligand_1,  2, 0.8115468833333334, 7.233
```

## Development

Install the requirements:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## License
HydraScreen is available restricted to Non-Commercial Use. For more information see the LICENSE file.
