import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from kern_gp.kern_gp_matrices import noiseless_predict

def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)

def independent_tanimoto_gp_predict(
    *,  # require inputting arguments by name
    query_smiles: list[str],  # len M
    known_smiles: list[str],  # len N
    known_Y: np.ndarray,  # NxK
    gp_means: np.ndarray,  # shape K
    gp_amplitudes: np.ndarray,  # shape K
    gp_noises: np.ndarray,  # shape K
    known_fp=None  # Optional precomputed fingerprints for known_smiles
) -> tuple[np.ndarray, np.ndarray]:
    for hparam_arr in (gp_means, gp_amplitudes, gp_noises):
        assert hparam_arr.shape == (known_Y.shape[1],)

    # Compute known_fp if not provided
    if known_fp is None:
        known_fp = [get_fingerprint(s) for s in known_smiles]
    
    # Compute query_fp every time since query_smiles changes with each iteration
    query_fp = [get_fingerprint(s) for s in query_smiles]
    
    # Calculate Tanimoto similarity matrices
    K_known_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in known_fp])
    K_query_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in query_fp])
    K_query_query_diagonal = np.asarray([DataStructs.TanimotoSimilarity(fp, fp) for fp in query_fp])

    means_out = []
    vars_out = []
    for j in range(known_Y.shape[1]):  # iterate over all objectives
        residual_j = known_Y[:, j] - gp_means[j]
        mu_j, var_j = noiseless_predict(
            a=gp_amplitudes[j],
            s=gp_noises[j],
            k_train_train=K_known_known,
            k_test_train=K_query_known,
            k_test_test=K_query_query_diagonal,
            y_train=residual_j,
            full_covar=False,
        )
        means_out.append(mu_j + gp_means[j])
        vars_out.append(var_j)

    return (
        np.asarray(means_out).T,
        np.asarray(vars_out).T,
    )
