#!/usr/bin/env python
import logging
import sys
import time
import random

import numpy as np
import pandas as pd
from pprint import pprint
from functools import partial
import scipy.stats as stats

import jax
import jax.numpy as jnp

from rdkit.Chem import DataStructs
from kernel_only_GP.tanimoto_gp import get_fingerprint, TanimotoGP_Params, ZeroMeanTanimotoGP
from utils.utils_final import evaluate_perin_MPO 

# === Logging setup ===
log_file = "logs/terminal_output_jax_ei_perin.log"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
bo_loop_logger.addHandler(h)

# === Batched JAX GP predict (vmap over objectives) ===
@jax.jit
def batch_predict(K_tt, K_qt, Y_train, raw_amp, raw_noise):
    """
    K_tt: (n_train,n_train)
    K_qt: (n_query,n_train)
    Y_train: (n_train, D)
    raw_amp, raw_noise: (D,)
    returns: means, vars of shape (n_query, D)
    """
    from kernel_only_GP.kern_gp import noiseless_predict
    from jax.nn import softplus

    def predict_one(y_col, a, s):
        m, v = noiseless_predict(
            a=softplus(a), s=softplus(s),
            k_train_train=K_tt,
            k_test_train=K_qt,
            k_test_test=jnp.ones((K_qt.shape[0],)),
            y_train=y_col,
            full_covar=False
        )
        return m, v + softplus(s)

    # vmap over objectives: y_col is column of Y_train
    mv = jax.vmap(predict_one, in_axes=(1, 0, 0))(Y_train, raw_amp, raw_noise)
    means, vars_ = mv  # each (D, n_query)
    return means.T, vars_.T  # (n_query, D)

# === Analytic Expected Improvement ===
def expected_improvement(means: np.ndarray, vars_: np.ndarray, y_best: float) -> np.ndarray:
    """
    means, vars_: (P, )
    y_best: scalar
    returns EI of shape (P,)
    """
    std = np.sqrt(vars_)
    # avoid division by zero
    z = (means - y_best) / (std + 1e-12)
    ei = (means - y_best) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
    return np.maximum(ei, 1e-30)

# === BO Loop ===
def bayesian_optimization_ei(
    known_smiles, query_smiles, known_Y,
    gp_amplitudes, gp_noises,
    n_iterations=20
):
    bo_loop_logger.info("Starting EI‐BO loop…")
    D = known_Y.shape[1]

    # 1) Precompute fingerprints & kernels
    all_smiles = known_smiles + query_smiles
    t0 = time.perf_counter()
    fps = [get_fingerprint(s) for s in all_smiles]
    n_train = len(known_smiles)
    fp_train, fp_query = fps[:n_train], fps[n_train:]
    # train–train
    K_tt = jnp.asarray([DataStructs.BulkTanimotoSimilarity(ft, fp_train) for ft in fp_train])
    # query–train
    K_qt = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fq, fp_train) for fq in fp_query])
    bo_loop_logger.info(f"Precomputed kernels in {time.perf_counter()-t0:.3f}s")

    # 2) static JAX arrays for GP hyperparams + training targets
    Y_train = jnp.asarray(known_Y)       # (n_train, D)
    raw_amp  = jnp.asarray(gp_amplitudes) # (D,)
    raw_noises = jnp.asarray(gp_noises)   # (D,)

    chosen = set()
    hypervolumes = []
    ei_history = []

    for it in range(n_iterations):
        start_it = time.perf_counter()
        bo_loop_logger.info(f"Iter {it}: train size={Y_train.shape}")

        # 3) Batch GP predict on entire pool
        means_all, vars_all = batch_predict(K_tt, K_qt, Y_train, raw_amp, raw_noises)
        means_all = np.array(means_all)  # (n_query, D)
        vars_all  = np.array(vars_all)

        # 4) Compute EI for each candidate (single‐objective: assume first col)
        y_best = float(np.max(np.array(Y_train)[:, 0]))
        ei_vals = expected_improvement(means_all[:, 0], vars_all[:, 0], y_best)

        # 5) mask out already‐chosen
        for idx in chosen:
            ei_vals[idx] = -np.inf

        # 6) select best
        best_local = int(np.argmax(ei_vals))
        best_smiles = query_smiles[best_local]
        bo_loop_logger.info(f" Selected {best_smiles} (idx={best_local}) → EI {ei_vals[best_local]:.4g}")
        ei_history.append(float(ei_vals[best_local]))

        # 7) evaluate real objective
        new_Y = evaluate_perin_MPO([best_smiles])  # yields shape (1,D)
        # 8) update data structures
        chosen.add(best_local)
        # append fingerprint & enlarge kernels
        new_fp = fp_query.pop(best_local)
        fp_train.append(new_fp)

        # update K_tt: add one row+col
        new_row = jnp.asarray([DataStructs.BulkTanimotoSimilarity(new_fp, fp_train[:-1])])
        K_tt = jnp.vstack([
            jnp.hstack([K_tt, new_row.T]),
            jnp.hstack([new_row, jnp.array([[1.0]])])
        ])
        # update K_qt: drop chosen row
        K_qt = jnp.delete(K_qt, best_local, axis=0)

        # update Y_train
        Y_train = jnp.vstack([Y_train, jnp.asarray(new_Y)])

        # record hypervolume on f1,f2 pair if you like
        # hv = Hypervolume(infer_reference_point(np.array(Y_train)))
        # hypervolumes.append(hv.compute(np.array(Y_train)))

        bo_loop_logger.info(f" Iter time: {time.perf_counter()-start_it:.2f}s\n")

    return

# === Entry point ===
if __name__ == "__main__":
    # load SMILES
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles",
                     header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:10_000]
    random.shuffle(all_sm)

    init_smiles = all_sm[:10]
    pool_smiles = all_sm[10:]

    pprint(init_smiles)
    init_Y = evaluate_perin_MPO(init_smiles)  # shape (10, D)

    # fixed GP hyperparameters
    gp_amp = np.array([1.0])
    gp_noise = np.array([1e-4])

    bayesian_optimization_ei(
        known_smiles=init_smiles,
        query_smiles=pool_smiles,
        known_Y=init_Y,
        gp_amplitudes=gp_amp,
        gp_noises=gp_noise,
        n_iterations=200
    )

    sys.stdout.close()
