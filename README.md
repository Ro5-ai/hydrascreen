[![Tests](https://github.com/Ro5-ai/hydrascreen/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Ro5-ai/hydrascreen/actions/workflows/run_tests.yml)
# Hydrascreen

This codebase provides functionality for making predictions using the HydraScreen API. It allows users to upload protein and ligand files, perform predictions, and retrieve the predicted affinity and pose score (probability) for each prediction.


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

`protein_file` needs to be a `Path` object for a PDB file.

`ligand_files` needs to be a list of `Path` objects for docked SDF files. 

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

The output will be a `pandas DataFrame` for your protein-ligand pair predictions.

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

results = []
for input_pair in input_pairs:
    result = predictor.predict_for_protein(**input_pair)
    results.append(result)
```

The output will be a list of `pandas DataFrames` with the prediction results for your protein-ligand pairs.

## Development

Install the requirements:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## License
TBD
