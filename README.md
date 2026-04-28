# Laptop-scale AQCat25 closed-loop optimisation demo

##### Goal
----
Build a lightweight proof-of-concept workflow for _MD-informed closed-loop optimisation of catalytic reaction conditions_ using a small sampled subset of AQCat25 metadata.

##### What this script does
---------------------
1. Streams AQCat25 metadata from Hugging Face without downloading the full dataset.
2. Samples a small subset, e.g. 2,000-10,000 rows.
3. Builds cheap descriptors:
   - adsorbate descriptors from formula/SMILES-like strings
   - slab/material descriptors from slab_id
   - magnetism and DFT metadata features
   - simulated MD-inspired environmental descriptors
4. Trains a surrogate model to predict adsorption energy.
5. Uses an uncertainty-aware acquisition function to suggest new reaction conditions.

##### Important scientific note
-------------------------
As AQCat25 contains DFT adsorption data, the MD part here is represented as coarse-grained MD-inspired descriptors:
solvent polarity, viscosity proxy, diffusion proxy, aggregation proxy, catalyst-contact proxy,
ligand concentration, residence time, and temperature.

In the future I want to try to compute environment descriptors with real MD-derived descriptors.

##### Installation

Recommended fresh environment:

```bash
conda create -n aqcat-bo python=3.10 -y
conda activate aqcat-bo
pip install datasets pandas numpy scikit-learn scipy pyarrow joblib tqdm matplotlib
```

Optional:

```bash
pip install xgboost
```

Hugging Face login may be needed:

```bash 
hf auth login
```

##### Example usage
Fast test:

    python aqcat25_closed_loop_bo.py --split val_id --max-rows 2000 --n-candidates 5000

Larger run:

    python aqcat25_closed_loop_bo.py --split train_id --max-rows 10000 --n-candidates 20000

##### Outputs
Creates an output folder with:

```    sampled_metadata.parquet
    features.parquet
    model.joblib
    metrics.json
    optimisation_suggestions.csv
    parity_plot.png
    acquisition_plot.png
```

