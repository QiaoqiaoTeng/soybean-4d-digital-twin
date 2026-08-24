# Geometry-Driven 4D Soybean Digital Twin — Review Package

This repository contains the trained random-forest surrogate models, archived
simulation outputs, model-testing materials, and analysis scripts associated
with the manuscript:

**A Geometry-Driven 4D Digital Twin Framework for Simulating Soybean Canopy
Light Interception and Machine-Learning-Assisted Candidate Planting
Configuration Screening**

The materials are provided to support editorial and peer-review evaluation.

## Contents

- `models/` — frozen trained RandomForestRegressor surrogate models for
  DN251, DN252, DN253, HN48, and HN51.
- `reviewer_test/` — compact measured 50-cm row-spacing test inputs and
  expected model predictions.
- `test_models.py` — one-command model-loading and prediction test.
- `data/` — archived simulation outputs used to fit the surrogate models,
  including the fixed 50-cm transect data.
- `scripts/` — the supplied cultivar-specific emulation and visualization
  scripts.
- `freeze_models.py` — reproducibly regenerates the frozen surrogate models
  from the archived simulation outputs.
- `models/model_metadata.csv` — model configuration, training R², candidate
  locations, and SHA256 hashes.

## Surrogate model

For each soybean cultivar, the surrogate model uses:

- Inputs: `Row_Spacing` (cm), `Plant_Spacing` (cm)
- Target: `Yield_Score`
- Estimator: `RandomForestRegressor`
- Number of trees: 300
- Maximum tree depth: 15

The review-package models use `random_state=42` solely to make model
serialization and reviewer testing deterministic.

## Quick reviewer test

### 1. Download the repository

Use GitHub **Code → Download ZIP**, then extract the archive.

### 2. Create a Python environment

Python 3.13 is recommended for the serialized review package.

### 3. Install the minimal dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the model test

From the repository root:

```bash
python test_models.py
```

A successful test prints `PASS` for all five cultivars and creates:

`reviewer_predictions.csv`

The test loads each frozen surrogate model and predicts the supplied measured
50-cm row-spacing treatment inputs.

## Rebuild the frozen models

The models can be regenerated from the archived simulation outputs with:

```bash
python freeze_models.py
```

This command uses the same random-forest architecture as the analysis scripts
and fixes `random_state=42` for deterministic reproduction.

## Full simulation/visualization workflow

The original cultivar-specific scripts are located in `scripts/`.
The complete geometry-driven simulation additionally requires the 3D `.ply`
soybean meshes expected under the relative `Kaggle数据/2019/` directory used
by the supplied scripts.

Install the broader optional dependency set with:

```bash
pip install -r requirements-full.txt
```

The frozen surrogate models and `test_models.py` do **not** require the large
3D mesh files and are intended to provide a lightweight model test for review.

## Data note

The archived CSV files in `data/` contain the digital-twin simulation outputs
used to train the supplied surrogate models. The five measured 50-cm
row-spacing treatments for each cultivar are included in the reviewer test
inputs.

## Citation

Please cite the associated manuscript when using these materials.
