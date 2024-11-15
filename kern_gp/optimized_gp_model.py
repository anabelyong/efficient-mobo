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
        self.known_fp = None
        self.query_fp_cache = {}
        self.K_known_known = None

    def update_known_smiles(self, known_smiles: list[str]):
        """Update fingerprints and Tanimoto similarity matrix for known SMILES."""
        self.known_fp = [get_fingerprint(s) for s in known_smiles]
        self.K_known_known = np.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, self.known_fp) for fp in self.known_fp]
        )

    def compute_similarity_matrices(self, query_smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache similarity matrices for a batch of query SMILES."""
        # Check if query_fp already cached, else compute
        query_fp = []
        for smiles in query_smiles:
            if smiles not in self.query_fp_cache:
                self.query_fp_cache[smiles] = get_fingerprint(smiles)
            query_fp.append(self.query_fp_cache[smiles])

        # Compute similarity matrices
        K_query_known = np.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, self.known_fp) for fp in query_fp]
        )
        K_query_query_diagonal = np.asarray(
            [DataStructs.TanimotoSimilarity(fp, fp) for fp in query_fp]
        )

        return K_query_known, K_query_query_diagonal, query_fp


def independent_tanimoto_gp_predict(
    *,  # Require inputting arguments by name
    query_smiles: list[str],  # len M
    known_smiles: list[str],  # len N
    known_Y: np.ndarray,  # NxK
    gp_means: np.ndarray,  # shape K
    gp_amplitudes: np.ndarray,  # shape K
    gp_noises: np.ndarray,  # shape K
    cached_gp: IndependentTanimotoGP,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict means and variances for query SMILES."""
    for hparam_arr in (gp_means, gp_amplitudes, gp_noises):
        assert hparam_arr.shape == (known_Y.shape[1],)

    # Update known fingerprints and Tanimoto similarity matrix if necessary
    cached_gp.update_known_smiles(known_smiles)

    # Compute similarity matrices for query SMILES
    K_query_known, K_query_query_diagonal, _ = cached_gp.compute_similarity_matrices(query_smiles)

    # Calculate predictions for each objective
    means_out = []
    vars_out = []
    for j in range(known_Y.shape[1]):  # Iterate over all objectives
        residual_j = known_Y[:, j] - gp_means[j]
        mu_j, var_j = noiseless_predict(
            a=gp_amplitudes[j],
            s=gp_noises[j],
            k_train_train=cached_gp.K_known_known,
            k_test_train=K_query_known,
            k_test_test=K_query_query_diagonal,
            y_train=residual_j,
            full_covar=False,
        )
        means_out.append(mu_j + gp_means[j])
        vars_out.append(var_j)

    return np.asarray(means_out).T, np.asarray(vars_out).T
