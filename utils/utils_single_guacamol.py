# check if objective function separation has worked
# First Test Case: fexofenadine from Guacamol dataset
from pprint import pprint

import numpy as np
import pandas as pd

from tdc_oracles_modified import Oracle

# Load GuacaMol dataset
guacamol_dataset_path = "guacamol_dataset/guacamol_v1_train.smiles"
guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
ALL_SMILES = guacamol_dataset["smiles"].tolist()[:10_000]
known_smiles = ALL_SMILES[:100]
print("Known SMILES:")
pprint(known_smiles)

# Create "oracles" for various objectives
TPSA_ORACLE = Oracle("tpsa_score_single")
LOGP_ORACLE = Oracle("logp_score_single")
FEXOFENADINE_SIM_ORACLE = Oracle("fex_similarity_value_single")

# MPOs to investigate as baseline
FEXOFENADINE_MPO_ORACLE = Oracle("fexofenadine_mpo")


def evaluate_single_objectives(smiles_list: list[str]) -> np.ndarray:
    """
    Given a list of N SMILES, return an NxK array A such that
    A_{ij} is the jth objective function on the ith SMILES.

    NOTE: you might replace this implementation with an alternative
    implementation for your objective of interest.

    Our specific implementation uses the objectives above.
    Because it uses the dockstring dataset to look up PPARD values,
    it is only defined on SMILES in the dockstring dataset.

    Also, be careful of NaN values! Some docking scores might be NaN.
    These will need to be dealt with somehow.
    """
    # Initialize arrays for each objective
    f1 = np.array(FEXOFENADINE_MPO_ORACLE(smiles_list))
    print(f"f1 shape: {f1.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]

    out = np.stack([f1])  # Nx1
    return out.T  # transpose, Nx1


#known_Y = evaluate_single_objectives(known_smiles)
#print(f"Known Y shape: {known_Y.shape}")
#print(known_Y)
