#!/usr/bin/env python
import logging
import sys
import time
import random

import pandas as pd
import numpy as np
from pprint import pprint
from functools import partial

import jax
import jax.numpy as jnp

from rdkit.Chem import DataStructs
from kernel_only_GP.tanimoto_gp import ZeroMeanTanimotoGP, TanimotoGP_Params, get_fingerprint
from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from utils.utils_final import evaluate_fex_objectives

# === Logging setup ===
log_file = "logs_trial2/terminal_output_jax_fex_ehvi.log"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
bo_loop_logger.addHandler(handler)

# === JAX-jit’d MC sampler ===
@partial(jax.jit, static_argnums=(3,))
def sample_outcomes(means, vars_, key, N=1000):
    """
    Draw N samples from multivariate normals: shape (P, N, D)
    """
    eps = jax.random.normal(key, (means.shape[0], N, means.shape[1]))
    return means[:, None, :] + jnp.sqrt(vars_)[:, None, :] * eps

# === EHVI via JAX sampler + Python HV loop ===
def expected_hypervolume_improvement(
    means: np.ndarray,
    vars_: np.ndarray,
    ref_point: np.ndarray,
    pareto_Y: np.ndarray,
    N: int = 1000
) -> np.ndarray:
    P, D = means.shape
    bo_loop_logger.info(f"  EHVI: {P} candidates with {N} MC samples")
    key = jax.random.PRNGKey(0)
    samples = np.array(sample_outcomes(
        jnp.array(means), jnp.array(vars_), key, N
    ))  # (P, N, D)

    hv_obj = Hypervolume(ref_point)
    base_hv = hv_obj.compute(pareto_Y)

    ehvi = np.zeros(P)
    for i in range(P):
        hsum = 0.0
        for s in samples[i]:
            hsum += max(0.0,
                        hv_obj.compute(np.vstack([pareto_Y, s])) - base_hv)
        ehvi[i] = hsum / N
    return ehvi

# === BO loop, index-based ===
def bayesian_optimization_loop(
    known_smiles,    # list[str] of training SMILES
    known_Y,         # np.ndarray (n_train, D)
    query_smiles,    # list[str] of pool SMILES
    gp_amplitudes, gp_noises,
    max_ref_point=None, scale=0.1, scale_max_ref_point=False,
    n_iterations=20, mc_samples=1000
):
    D = known_Y.shape[1]

    # 1) Precompute all fingerprints once
    all_smiles = list(known_smiles) + list(query_smiles)
    t0 = time.perf_counter()
    fps = [get_fingerprint(s) for s in all_smiles]
    bo_loop_logger.info(f"Precomputed {len(fps)} Morgan-fps in {time.perf_counter() - t0:.3f}s")

    n_train = len(known_smiles)
    n_total = len(all_smiles)

    # 2) Build per-objective GP models on “indices”
    train_idx = list(range(n_train))
    gp_models = []
    gp_params = []
    for d in range(D):
        gp = ZeroMeanTanimotoGP(lambda i: fps[i], train_idx, known_Y[:, d])
        params = TanimotoGP_Params(
            raw_amplitude=jnp.array(gp_amplitudes[d]),
            raw_noise=jnp.array(gp_noises[d])
        )
        gp_models.append(gp)
        gp_params.append(params)

    chosen = set()
    Y_train = known_Y.copy()
    hypervolumes, acq_vals = [], []

    for it in range(n_iterations):
        bo_loop_logger.info(f"\n--- Iter {it} (train size {Y_train.shape}) ---")

        # Pareto front & baseline hypervolume
        ref = infer_reference_point(Y_train, max_ref_point, scale, scale_max_ref_point)
        mask = pareto_front(Y_train)
        pareto_Y = Y_train[mask]
        hv_obj = Hypervolume(ref)

        # Pool of unseen indices
        pool = [i for i in range(n_train, n_total) if i not in chosen]
        P = len(pool)
        bo_loop_logger.info(f" Scoring full pool: {P} candidates")

        # 3) GP predict on pool
        t1 = time.perf_counter()
        means = np.zeros((P, D))
        vars_ = np.zeros((P, D))
        for d in range(D):
            m_d, v_d = gp_models[d].predict_y(
                gp_params[d], pool, full_covar=False
            )
            means[:, d] = np.array(m_d)
            vars_[:, d] = np.array(v_d)
        bo_loop_logger.info(f" GP predict time: {time.perf_counter() - t1:.3f}s")

        # 4) EHVI via JAX sampler + Python loop
        t2 = time.perf_counter()
        ehvi = expected_hypervolume_improvement(
            means, vars_, ref, pareto_Y, N=mc_samples
        )
        bo_loop_logger.info(f" EHVI time: {time.perf_counter() - t2:.3f}s")

        # 5) Pick best
        best_local = int(np.argmax(ehvi))
        best_global = pool[best_local]
        chosen.add(best_global)
        acq_vals.append(float(ehvi[best_local]))
        best_smiles = all_smiles[best_global]
        new_y = evaluate_fex_objectives([best_smiles])[0]
        bo_loop_logger.info(f" Selected idx={best_global} ({best_smiles}) → {new_y}")

        # 6) Update training set & GPs
        train_idx.append(best_global)
        Y_train = np.vstack([Y_train, new_y[None, :]])
        for d in range(D):
            gp_models[d].set_training_data(train_idx, Y_train[:, d])

        # 7) Record new hypervolume
        new_hv = hv_obj.compute(Y_train)
        hypervolumes.append(new_hv)
        bo_loop_logger.info(f" Hypervolume: {new_hv:.4f}")

    return Y_train, hypervolumes, acq_vals

# === Entry point ===
if __name__ == "__main__":
    df = pd.read_csv(
        "guacamol_dataset/guacamol_v1_train.smiles",
        header=None, names=["smiles"]
    )
    all_sm = df["smiles"].tolist()[:10000]
    random.shuffle(all_sm)

    known_smiles = all_sm[:10]
    query_smiles = all_sm[10:]

    pprint(known_smiles)
    known_Y = evaluate_fex_objectives(known_smiles)
    bo_loop_logger.info(f"Initial objectives:\n{known_Y}")

    final_Y, hvs, acqs = bayesian_optimization_loop(
        known_smiles,
        known_Y,
        query_smiles,
        gp_amplitudes=np.array([1.0, 1.0, 1.0]),
        gp_noises  =np.array([1e-4, 1e-4, 1e-4]),
        n_iterations=200,
        mc_samples=500
    )

    # Print Pareto front ===
    mask_final = pareto_front(final_Y)
    pareto_Y_final = final_Y[mask_final]
    pareto_indices = np.where(mask_final)[0]
    pareto_smiles = [known_smiles[i] if i < len(known_smiles) else None for i in pareto_indices]
    bo_loop_logger.info(f"\nFinal Pareto front points: {pareto_Y_final.tolist()}")
    bo_loop_logger.info(f"Final Pareto front SMILES: {pareto_smiles}")

    handler.close()
    sys.stdout.close()
