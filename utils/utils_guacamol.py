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
known_smiles = ALL_SMILES[:50]
print("Known SMILES:")
pprint(known_smiles)

# Create "oracles" for FEXOFENADINE objectives
TPSA_ORACLE = Oracle("tpsa_score_single")
LOGP_ORACLE = Oracle("logp_score_single")
FEXOFENADINE_SIM_ORACLE = Oracle("fex_similarity_value_single")

# Create "oracles" for OSIMERTINIB objectives
OSIM_TPSA_ORACLE = Oracle("osimertinib_tpsa_score")
OSIM_LOGP_ORACLE = Oracle("osimertinib_logp_score")
OSIM_SIM_V1_ORACLE = Oracle("osimertinib_similarity_v1_score")
OSIM_SIM_V2_ORACLE = Oracle("osimertinib_similarity_v2_score")

# Create "oracles" for RANOLAZINE objectives
RANOL_TPSA_ORACLE = Oracle("ranolazine_tpsa_score")
RANOL_LOGP_ORACLE = Oracle("ranolazine_logp_score")
RANOL_SIM_ORACLE = Oracle("ranolazine_similarity_value")
RANOL_FLUORINE_ORACLE = Oracle("ranolazine_fluorine_value")

# 1st MPOs to investigate as baseline
FEXOFENADINE_MPO_ORACLE = Oracle("fexofenadine_mpo")
#2nd and 3rd MPO to investigate as baseline
OSIMIERTINIB_MPO_ORACLE = Oracle("osimertinib_mpo")
RANOLAZINE_MPO_ORACLE = Oracle("ranolazine_mpo")


def evaluate_objectives(smiles_list: list[str]) -> np.ndarray:
    # Given a list of N SMILES, return an NxK array A such that
    # A_{ij} is the jth objective function on the ith SMILES.

    # NOTE:you might replace this implementation with an alternative
    # implementation for your objective of interest.

    # Our specific implementation uses the objectives above.
    # Because it uses the dockstring dataset to look up PPARD values,
    # it is only defined on SMILES in the dockstring dataset.

    # Also, be careful of NaN values! Some docking scores might be NaN.
    # These will need to be dealt with somehow.

    # Initialize arrays for each objective
    f1 = np.array(TPSA_ORACLE(smiles_list))
    f2 = np.array(LOGP_ORACLE(smiles_list))
    f3 = np.array(FEXOFENADINE_SIM_ORACLE(smiles_list))

    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    print(f"f3 shape: {f3.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    f3 = f3[valid_indices]

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2) == len(f3)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2, f3])  # 4x3
    return out.T  # transpose, Nx3


#known_Y = evaluate_objectives(known_smiles)
#print(f"Known Y shape: {known_Y.shape}")
#print(known_Y)
