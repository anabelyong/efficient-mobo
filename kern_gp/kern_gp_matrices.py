"""
Contains code for zero-mean GP with kernel a*k(x,x) + s*I for some base kernel k.
"""

import logging

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.linalg import cho_solve, cholesky, solve_triangular

LOWER = True
logger = logging.getLogger(__name__)


# counts -- minmax kernel
def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)


def calculate_tanimoto_coefficients(known_smiles: list[str], query_smiles: list[str] = None):
    """
    Calculate Tanimoto coefficient matrices for known and query SMILES strings.

    Args:
        known_smiles (list[str]): List of known SMILES strings.
        query_smiles (list[str], optional): List of query SMILES strings. Defaults to None.

    Returns:
        tuple: (K_known_known, K_query_known, K_query_query_diagonal)
            K_known_known (np.ndarray): Tanimoto coefficients between known molecules.
            K_query_known (np.ndarray): Tanimoto coefficients between query and known molecules.
            K_query_query_diagonal (np.ndarray): Diagonal of Tanimoto coefficients for query molecules.
    """

    known_fp = [get_fingerprint(s) for s in known_smiles]
    K_known_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in known_fp])  # shape (N, N)

    if query_smiles is None:
        query_smiles = known_smiles

    query_fp = [get_fingerprint(s) for s in query_smiles]
    K_query_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in query_fp])  # shape (M, N)
    print("K_query_known", K_query_known)

    K_query_query_diagonal = np.asarray([DataStructs.TanimotoSimilarity(fp, fp) for fp in query_fp])  # shape (M,)

    return K_known_known, K_query_known, K_query_query_diagonal


def mll_train(a, s, k_train_train, y_train):
    """Computes the marginal log likelihood of the training data."""
    L = _k_cholesky(k_train_train, s / a)
    data_fit = _data_fit(L, a, y_train)
    complexity = _complexity(L, a)
    constant = -k_train_train.shape[0] / 2 * np.log(2 * np.pi)
    return data_fit + complexity + constant


def noiseless_predict(a, s, k_train_train, k_test_train, k_test_test, y_train, full_covar: bool = True):
    """
    Computes mean and [co]variance predictions for the test data given training data.

    Full covar means we return the full covariance matrix, otherwise we return the diagonal.
    """
    L = _k_cholesky(k_train_train, s / a)
    mean = np.dot(k_test_train, cho_solve((L, LOWER), y_train))
    covar_adj_sqrt = solve_triangular(L, k_test_train.T, lower=LOWER)
    if full_covar:
        covar_adj = covar_adj_sqrt.T @ covar_adj_sqrt
    else:
        covar_adj = np.sum(covar_adj_sqrt**2, axis=0)

    return mean, a * (k_test_test - covar_adj)


def _k_cholesky(k, s):
    """Computes cholesky of k+sI."""
    logger.debug(f"Computing cholesky of {k.shape[0]}x{k.shape[0]} matrix with s={s}")
    k2 = k + s * np.eye(k.shape[0])
    L = cholesky(k2, lower=LOWER)
    logger.debug("Done computing cholesky")
    return L


def _data_fit(L, a, y_train):
    return -0.5 / a * np.dot(y_train.T, cho_solve((L, LOWER), y_train))


def _complexity(L, a):
    """MLL complexity term for kernel a(L@L^T)"""
    log_det_L = -np.sum(np.log(np.diag(L)))  # because we use cholesky, the factor of 2 cancels so no -1/2
    a_adjustment = -np.log(a) * L.shape[0] / 2
    return log_det_L + a_adjustment
