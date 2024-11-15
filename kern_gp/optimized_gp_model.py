import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from kern_gp.kern_gp_matrices import noiseless_predict


def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)


class IndependentTanimotoGP:
    def __init__(self):
        self.known_fp = []

    def update_known_smiles(self, known_smiles):
        """Update fingerprints for known SMILES."""
        self.known_fp = [get_fingerprint(s) for s in known_smiles]

    def compute_similarity_matrices(self, query_smiles):
        """Compute similarity matrices for query SMILES."""
        query_fp = [get_fingerprint(s) for s in query_smiles]
        K_query_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, self.known_fp) for fp in query_fp])
        K_query_query_diagonal = np.asarray([DataStructs.TanimotoSimilarity(fp, fp) for fp in query_fp])
        return K_query_known, K_query_query_diagonal, query_fp


def independent_tanimoto_gp_predict(
    *,  # Require named arguments
    query_smiles: list[str],
    known_smiles: list[str],
    known_Y: np.ndarray,
    gp_means: np.ndarray,
    gp_amplitudes: np.ndarray,
    gp_noises: np.ndarray,
    known_fp: list,
    query_fp: list,
    K_query_known=None,  # Precomputed matrix
    K_query_query_diagonal=None,  # Precomputed diagonal
):
    if K_query_known is None or K_query_query_diagonal is None:
        raise ValueError("Precomputed similarity matrices are required.")

    K_known_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in known_fp])

    means_out = []
    vars_out = []
    for j in range(known_Y.shape[1]):  # Iterate over all objectives
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

    return np.asarray(means_out).T, np.asarray(vars_out).T
