from pprint import pprint

import numpy as np
from dockstring.dataset import load_dataset
from tdc import Oracle
from scipy.stats import gmean 

# Load dockstring dataset
DOCKSTRING_DATASET = load_dataset()
ALL_SMILES = list(DOCKSTRING_DATASET["PPARD"].keys())[:10_000]
known_smiles = ALL_SMILES[:100]
print("Known SMILES:")
pprint(known_smiles)

# Create "oracles" for various objectives
QED_ORACLE = Oracle("qed")
CELECOXIB_ORACLE = Oracle("celecoxib-rediscovery")

def get_geometric_mean(smiles: str) -> float:
    """
    Calculate the geometric mean of the three oracles for a single SMILES string.

    Parameters:
    smiles (str): A SMILES string to evaluate.

    Returns:
    float: The geometric mean of the oracle evaluations.
    """
    f1 = -DOCKSTRING_DATASET["PPARD"].get(smiles, np.nan)  # Use negative since PPARD values are usually scores
    f2 = QED_ORACLE(smiles)
    f3 = CELECOXIB_ORACLE(smiles)
   

    values = [f1, f2, f3]
    
    # Filter out any NaN values to avoid issues in geometric mean calculation
    values = [v for v in values if not np.isnan(v)]
    
    if len(values) == 0:
        raise ValueError("All oracle values are NaN.")

    return gmean(values)

def evaluate_objectives(smiles_list: list[str]) -> np.ndarray:
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
    f1 = np.array([-DOCKSTRING_DATASET["PPARD"].get(s, np.nan) for s in smiles_list])
    f2 = np.array(QED_ORACLE(smiles_list))
    f3 = np.array(CELECOXIB_ORACLE(smiles_list))
    

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

    out = np.stack([f1, f2, f3])  # 3xN
    return out.T  # transpose, Nx3

def evaluate_single_objective(smiles_list: list[str]) -> np.ndarray:
    """
    Evaluate the geometric mean of the objectives for each SMILES string in the list.

    Parameters:
    smiles_list (list of str): List of SMILES strings to evaluate.

    Returns:
    np.ndarray: An array of geometric mean values, each corresponding to a SMILES string.
    """
    geometric_means = []

    for smiles in smiles_list:
        try:
            # Assume get_geometric_mean is a function that computes the geometric mean for a given SMILES
            geo_mean = get_geometric_mean(smiles)
            geometric_means.append(geo_mean)
        except ValueError as e:
            print(f"Skipping SMILES {smiles} due to error: {e}")
            continue
    
    # Convert the list of geometric means into a numpy array
    geometric_means_array = np.array(geometric_means)

    # Filter out NaN values and return the cleaned array
    valid_indices = ~np.isnan(geometric_means_array)
    geometric_means_array = geometric_means_array[valid_indices]

    # Reshape the array to be Nx1, matching the expected output format
    out = np.stack([geometric_means_array])  # Nx1
    return out.T  # Transpose to get Nx1 shape


#known_Y = evaluate_single_objective(known_smiles)
#print(f"Known Y shape: {known_Y.shape}")
#print(known_Y)

