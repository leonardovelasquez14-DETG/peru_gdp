#!/usr/bin/env python3
"""
Etapa 4 + Etapas 5 y 8 (reservorios): QRC confirmatorio, ESN y RFF causales.

Evalua exactamente sobre los mismos 250 origenes OOS (2003m2-2024m11; bloque
congelado final 2019m12-2024m11, n = 60) de la Etapa 3 las arquitecturas:
  - QRC_Ising5 : reservorio cuantico de N_q = 5 qubits (d = 32), canal CPTP con
                 memoria evanescente y evolucion unitaria de Ising transversal,
                 20 observables de Pauli (<Z_i>, <X_i>, <Z_i Z_j>) + lectura Ridge.
  - ESN_20     : Echo State Network clasica (N_r = 20 neuronas tanh, rho(W)=0.85).
  - ESN_64     : Echo State Network clasica ampliada (N_r = 64 neuronas).
  - RFF_20     : caracteristicas aleatorias no lineales sin recurrencia (m = 20).

Nota de entorno: qiskit/pennylane/cirq no estan instalados, por lo que QRC_Ising5 se
ejecuta mediante simulacion exacta de matriz densidad en NumPy/SciPy (limite de shots
infinitos sin ruido de hardware), verificando explicitamente invariantes unitarios,
traza, cotas Pauli, paridad con Etapa 3 y cero fuga temporal.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rtd_eval_utils as u  # noqa: E402

ANALYSIS_DIR = ROOT_DIR / "output" / "analysis_output"
ETAPA3_DIR = ROOT_DIR / "output" / "etapa3_clasicos"
OUT_DIR = ROOT_DIR / "output" / "etapa4_reservorios"
TEX_IMG_DIR = ROOT_DIR / "main_tex" / "images"

EVAL_START_IDX = 100
SEEDS = (101, 202, 303, 404, 505)
N_QUBITS = 5
HILBERT_DIM = 2 ** N_QUBITS
RIDGE_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0)

ALPHA_GRID = (0.4, 0.8)
LEAK_GRID = (0.35, 0.70)
HP_RETUNE_STRIDE = 3

PAULI_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _kron_op_at(n_qubits: int, ops: Dict[int, np.ndarray]) -> np.ndarray:
    """Construye el producto de Kronecker de operadores locales en n_qubits."""
    mat = np.array([[1.0]], dtype=np.complex128)
    for q in range(n_qubits):
        mat = np.kron(mat, ops.get(q, PAULI_I))
    return mat


def build_ising_unitary(n_qubits: int, seed: int, dt: float = 1.0) -> np.ndarray:
    """Genera el operador unitario U = exp(-i H_Ising dt) fijo para una semilla."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    ham = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n_qubits):
        h_i = float(rng.uniform(0.5, 1.5))
        ham += h_i * _kron_op_at(n_qubits, {i: PAULI_X})
        for j in range(i + 1, n_qubits):
            j_ij = float(rng.uniform(-1.0, 1.0))
            ham += j_ij * _kron_op_at(n_qubits, {i: PAULI_Z, j: PAULI_Z})
    return expm(-1j * ham * dt)


def build_pauli_observables(n_qubits: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Construye los 20 observables de Pauli (<Z_i>, <X_i>, <Z_i Z_j>) en forma densa
    para auditoria fisica y mascaras rapidas para extraccion O(d) en la simulacion.
    """
    dim = 2 ** n_qubits
    dense_list: List[np.ndarray] = []
    diag_masks: List[np.ndarray] = []
    basis_bits = np.arange(dim, dtype=np.int64)

    # 5 operadores Z_i
    for i in range(n_qubits):
        dense_list.append(_kron_op_at(n_qubits, {i: PAULI_Z}))
        shift = n_qubits - 1 - i
        eig = np.where((basis_bits >> shift) & 1, -1.0, 1.0)
        diag_masks.append(eig)

    # 5 operadores X_i
    x_flip_pairs: List[Tuple[int, int]] = []
    for i in range(n_qubits):
        dense_list.append(_kron_op_at(n_qubits, {i: PAULI_X}))
        shift = n_qubits - 1 - i
        x_flip_pairs.append((i, 1 << shift))

    # 10 operadores Z_i Z_j
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            dense_list.append(_kron_op_at(n_qubits, {i: PAULI_Z, j: PAULI_Z}))
            diag_masks.append(diag_masks[i] * diag_masks[j])

    return np.stack(dense_list, axis=0), np.stack(diag_masks, axis=0), x_flip_pairs


def build_normalized_projection(n_out: int, n_in: int, seed: int) -> np.ndarray:
    """Matriz de proyeccion aleatoria normalizada por filas con norma unitaria."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=(n_out, n_in))
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    return w / np.where(norms < 1e-12, 1.0, norms)


def encode_product_state(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Codifica angulos (theta, phi) en un estado producto de N_q qubits mediante
    RZ(phi_q) RY(theta_q) |0> y devuelve el ket completo en C^{2^N_q}.
    """
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    p_m = np.exp(-0.5j * phi) * c
    p_p = np.exp(0.5j * phi) * s

    state = np.array([p_m[0], p_p[0]], dtype=np.complex128)
    for q in range(1, len(theta)):
        state = np.kron(state, np.array([p_m[q], p_p[q]], dtype=np.complex128))
    return state


def extract_qrc_features(
    x_seq_sc: np.ndarray,
    w_enc: np.ndarray,
    u_gate: np.ndarray,
    diag_masks: np.ndarray,
    x_flip_pairs: Sequence[Tuple[int, int]],
    alpha: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evoluciona la matriz densidad rho_t mediante canal CPTP causal y extrae los
    20 valores esperados de Pauli en cada paso temporal de x_seq_sc.
    """
    u_dag = u_gate.conj().T
    n_t = x_seq_sc.shape[0]
    n_z = len(diag_masks)  # 15 (5 Z_i + 10 Z_i Z_j)
    n_x = len(x_flip_pairs)
    feats = np.empty((n_t, n_z + n_x), dtype=float)
    z_singles = n_x

    # Estado inicial maximamente mixto (independiente de los datos)
    rho = np.eye(HILBERT_DIM, dtype=np.complex128) / HILBERT_DIM
    angles = (np.pi / 2.0) * np.tanh(alpha * (x_seq_sc @ w_enc.T))
    thetas = angles[:, :N_QUBITS]
    phis = angles[:, N_QUBITS:]
    basis_idx = np.arange(HILBERT_DIM, dtype=np.int64)

    for t in range(n_t):
        psi = encode_product_state(thetas[t], phis[t])
        rho_in = np.outer(psi, psi.conj())
        rho_mix = eps * rho_in + (1.0 - eps) * rho
        rho = u_gate @ rho_mix @ u_dag

        diag_real = np.real(np.diag(rho))
        z_all = diag_masks @ diag_real
        feats[t, :z_singles] = z_all[:z_singles]
        for idx_x, (_, mask_bit) in enumerate(x_flip_pairs):
            feats[t, z_singles + idx_x] = float(np.real(np.sum(rho[basis_idx, basis_idx ^ mask_bit])))
        feats[t, z_singles + n_x :] = z_all[z_singles:]

    return feats, rho


def build_esn_weights(n_res: int, n_in: int, seed: int, spectral_radius: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    """Genera (W_in, W_res) para ESN clasica con propiedad de estado eco garantizada."""
    w_in = build_normalized_projection(n_res, n_in, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    w_res = rng.normal(0.0, 1.0, size=(n_res, n_res))
    mask = rng.uniform(0.0, 1.0, size=(n_res, n_res)) < 0.35
    w_res = np.where(mask, w_res, 0.0)
    eig_max = float(np.max(np.abs(np.linalg.eigvals(w_res))))
    if eig_max < 1e-8:
        w_res = np.eye(n_res)
        eig_max = 1.0
    w_res = w_res * (spectral_radius / eig_max)
    return w_in, w_res


def extract_esn_features(
    x_seq_sc: np.ndarray, w_in: np.ndarray, w_res: np.ndarray, alpha: float, leak: float
) -> np.ndarray:
    """Propaga el estado oculto h_t = (1-leak) h_{t-1} + leak tanh(alpha W_in x_t + W h_{t-1})."""
    n_t = x_seq_sc.shape[0]
    n_res = w_res.shape[0]
    feats = np.empty((n_t, n_res), dtype=float)
    h = np.zeros(n_res, dtype=float)
    driven = alpha * (x_seq_sc @ w_in.T)
    for t in range(n_t):
        h = (1.0 - leak) * h + leak * np.tanh(driven[t] + (w_res @ h))
        feats[t] = h
    return feats


def extract_rff_features(
    x_seq_sc: np.ndarray, w_rff: np.ndarray, b_rff: np.ndarray, alpha: float
) -> np.ndarray:
    """Caracteristicas aleatorias no lineales sin recurrencia: cos(alpha W x_t + b)."""
    return np.cos(alpha * (x_seq_sc @ w_rff.T) + b_rff)


def get_inner_val_pairs(df: pd.DataFrame, tr_idx: np.ndarray, stride: int = 2) -> List[Tuple[np.ndarray, int]]:
    """Construye pliegues temporales internos estrictamente causales dentro de tr_idx."""
    n_tr = len(tr_idx)
    n_val = min(36, max(18, n_tr // 4))
    val_candidates = tr_idx[-n_val:]
    if stride > 1:
        val_candidates = val_candidates[::stride]

    v_h11_tr = df.loc[tr_idx, "v_H11_date"].to_numpy()
    pairs: List[Tuple[np.ndarray, int]] = []
    for val_i in val_candidates:
        v0_val = df.loc[val_i, "v0_date"].to_datetime64()
        mask = (tr_idx < val_i) & (v_h11_tr <= v0_val)
        inner_tr = tr_idx[mask]
        if len(inner_tr) >= 24:
            pairs.append((inner_tr, int(val_i)))
    return pairs


def ridge_cv_and_predict(
    z_seq: np.ndarray,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    eval_idx: int,
    pairs: List[Tuple[np.ndarray, int]],
    lam_grid: Sequence[float] = RIDGE_GRID,
) -> Tuple[float, float, float]:
    """
    Evalua la grilla Ridge via validacion temporal causal interna usando z_seq,
    devolviendo (pred_oos, best_lam, val_mse).
    """
    best_lam = 50.0
    best_mse = np.inf
    if pairs:
        sq_errs = np.zeros(len(lam_grid), dtype=float)
        for inner_tr, val_i in pairs:
            z_i_tr, z_i_val = u.fit_impute_and_scale(z_seq[inner_tr], z_seq[[val_i]])
            y_i_tr = y_all[inner_tr]
            y_i_val = y_all[val_i]
            y_mean = float(np.mean(y_i_tr))
            yc = y_i_tr - y_mean
            u_mat, s_vals, vt = np.linalg.svd(z_i_tr, full_matrices=False)
            uty = u_mat.T @ yc
            v_x = vt @ z_i_val[0]
            for j, lam in enumerate(lam_grid):
                coef_f = s_vals / (s_vals ** 2 + lam)
                pred_v = y_mean + float(np.dot(v_x, coef_f * uty))
                sq_errs[j] += (y_i_val - pred_v) ** 2
        best_j = int(np.argmin(sq_errs))
        best_lam = float(lam_grid[best_j])
        best_mse = float(sq_errs[best_j] / len(pairs))

    z_tr_sc, z_te_sc = u.fit_impute_and_scale(z_seq[tr_idx], z_seq[[eval_idx]])
    y_tr = y_all[tr_idx]
    y_mean = float(np.mean(y_tr))
    yc = y_tr - y_mean
    p = z_tr_sc.shape[1]
    beta = np.linalg.solve(z_tr_sc.T @ z_tr_sc + best_lam * np.eye(p), z_tr_sc.T @ yc)
    pred = float(y_mean + np.dot(z_te_sc[0], beta))
    return pred, best_lam, best_mse


def run_quantum_physics_audit(
    diag_masks: np.ndarray, x_flip_pairs: Sequence[Tuple[int, int]], dense_ops: np.ndarray
) -> Dict[str, object]:
    """Audita invariantes fisico-matematicos exactos del simulador cuantico."""
    max_unitarity_err = 0.0
    max_trace_err = 0.0
    max_herm_err = 0.0
    min_eig_seen = 1.0
    max_obs_bound_err = 0.0
    max_obs_extractor_diff = 0.0

    rng = np.random.default_rng(777)
    for seed in SEEDS:
        u_gate = build_ising_unitary(N_QUBITS, seed=seed)
        unit_err = float(np.linalg.norm(u_gate.conj().T @ u_gate - np.eye(HILBERT_DIM), ord=2))
        max_unitarity_err = max(max_unitarity_err, unit_err)

        w_enc = build_normalized_projection(2 * N_QUBITS, len(u.FEATURE_COLS), seed=seed)
        x_dummy = rng.normal(0.0, 2.0, size=(12, len(u.FEATURE_COLS)))
        feats, rho_last = extract_qrc_features(
            x_dummy, w_enc, u_gate, diag_masks, x_flip_pairs, alpha=0.8, eps=0.5
        )
        max_trace_err = max(max_trace_err, float(abs(np.trace(rho_last) - 1.0)))
        max_herm_err = max(max_herm_err, float(np.max(np.abs(rho_last - rho_last.conj().T))))
        eigs = np.linalg.eigvalsh(0.5 * (rho_last + rho_last.conj().T))
        min_eig_seen = min(min_eig_seen, float(np.min(eigs)))
        bound_err = float(np.max(np.maximum(0.0, np.abs(feats) - 1.0)))
        max_obs_bound_err = max(max_obs_bound_err, bound_err)

        dense_exp = np.array([float(np.real(np.trace(rho_last @ op))) for op in dense_ops])
        ext_diff = float(np.max(np.abs(feats[-1] - dense_exp)))
        max_obs_extractor_diff = max(max_obs_extractor_diff, ext_diff)

    assert max_unitarity_err < 1e-12, f"Unitariedad violada: {max_unitarity_err}"
    assert max_trace_err < 1e-12, f"Traza violada: {max_trace_err}"
    assert max_herm_err < 1e-12, f"Hermiticidad violada: {max_herm_err}"
    assert min_eig_seen >= -1e-12, f"Estado no positivo: {min_eig_seen}"
    assert max_obs_bound_err < 1e-12, f"Cotas Pauli violadas: {max_obs_bound_err}"
    assert max_obs_extractor_diff < 1e-12, f"Extractor Pauli inconsistente: {max_obs_extractor_diff}"

    return {
        "max_unitarity_err_L2": max_unitarity_err,
        "max_trace_err": max_trace_err,
        "max_hermiticity_err": max_herm_err,
        "min_density_eigenvalue": min_eig_seen,
        "max_pauli_bound_violation": max_obs_bound_err,
        "max_fast_vs_dense_observable_diff": max_obs_extractor_diff,
        "passed": True,
    }


def _evaluate_single_origin_all_models(
    df: pd.DataFrame,
    x_all: np.ndarray,
    y_all: np.ndarray,
    eval_idx: int,
    seed: int,
    artifacts: Dict[str, object],
    hp_cache: Dict[str, Tuple[float, float]] | None,
    retune_hp: bool,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Evalua QRC_Ising5, ESN_20, ESN_64 y RFF_20 para un origen causal y semilla dados."""
    tr_idx = u.get_causal_train_indices(df, eval_idx)
    pairs = get_inner_val_pairs(df, tr_idx, stride=2)
    x_tr_raw = x_all[tr_idx]
    _, x_seq_sc = u.fit_impute_and_scale(x_tr_raw, x_all[: eval_idx + 1])

    new_cache: Dict[str, Tuple[float, float]] = {}
    preds: Dict[str, float] = {}

    # 1. QRC_Ising5
    u_gate = artifacts["u_gate"]  # type: ignore[assignment]
    w_enc = artifacts["w_enc"]  # type: ignore[assignment]
    diag_masks = artifacts["diag_masks"]  # type: ignore[assignment]
    x_flip_pairs = artifacts["x_flip_pairs"]  # type: ignore[assignment]

    if retune_hp or hp_cache is None:
        best_score = np.inf
        best_pair_hp = (0.4, 0.35)
        best_pred_q = 0.0
        for a in ALPHA_GRID:
            for e in LEAK_GRID:
                z_q, _ = extract_qrc_features(x_seq_sc, w_enc, u_gate, diag_masks, x_flip_pairs, a, e)
                p_q, _, mse_v = ridge_cv_and_predict(z_q, y_all, tr_idx, eval_idx, pairs)
                if mse_v < best_score:
                    best_score = mse_v
                    best_pair_hp = (a, e)
                    best_pred_q = p_q
        new_cache["qrc"] = best_pair_hp
        preds["QRC_Ising5"] = best_pred_q
    else:
        a_opt, e_opt = hp_cache["qrc"]
        z_q, _ = extract_qrc_features(x_seq_sc, w_enc, u_gate, diag_masks, x_flip_pairs, a_opt, e_opt)
        preds["QRC_Ising5"], _, _ = ridge_cv_and_predict(z_q, y_all, tr_idx, eval_idx, pairs)
        new_cache["qrc"] = (a_opt, e_opt)

    # 2. ESN_20 y ESN_64
    for tag, w_key in (("ESN_20", "esn20"), ("ESN_64", "esn64")):
        w_in, w_res = artifacts[w_key]  # type: ignore[misc]
        if retune_hp or hp_cache is None:
            best_score = np.inf
            best_hp = (0.4, 0.35)
            best_pred_e = 0.0
            for a in ALPHA_GRID:
                for g in LEAK_GRID:
                    z_e = extract_esn_features(x_seq_sc, w_in, w_res, a, g)
                    p_e, _, mse_v = ridge_cv_and_predict(z_e, y_all, tr_idx, eval_idx, pairs)
                    if mse_v < best_score:
                        best_score = mse_v
                        best_hp = (a, g)
                        best_pred_e = p_e
            new_cache[w_key] = best_hp
            preds[tag] = best_pred_e
        else:
            a_opt, g_opt = hp_cache[w_key]
            z_e = extract_esn_features(x_seq_sc, w_in, w_res, a_opt, g_opt)
            preds[tag], _, _ = ridge_cv_and_predict(z_e, y_all, tr_idx, eval_idx, pairs)
            new_cache[w_key] = (a_opt, g_opt)

    # 3. RFF_20 (sin recurrencia)
    w_rff, b_rff = artifacts["rff20"]  # type: ignore[misc]
    if retune_hp or hp_cache is None:
        best_score = np.inf
        best_a = 0.4
        best_pred_r = 0.0
        for a in ALPHA_GRID:
            z_r = extract_rff_features(x_seq_sc, w_rff, b_rff, a)
            p_r, _, mse_v = ridge_cv_and_predict(z_r, y_all, tr_idx, eval_idx, pairs)
            if mse_v < best_score:
                best_score = mse_v
                best_a = a
                best_pred_r = p_r
        new_cache["rff20"] = (best_a, 0.0)
        preds["RFF_20"] = best_pred_r
    else:
        a_opt, _ = hp_cache["rff20"]
        z_r = extract_rff_features(x_seq_sc, w_rff, b_rff, a_opt)
        preds["RFF_20"], _, _ = ridge_cv_and_predict(z_r, y_all, tr_idx, eval_idx, pairs)
        new_cache["rff20"] = (a_opt, 0.0)

    return preds, new_cache


def build_seed_artifacts(
    seed: int, n_features: int, diag_masks: np.ndarray, x_flip_pairs: Sequence[Tuple[int, int]]
) -> Dict[str, object]:
    """Preconstruye matrices fijas dependientes unicamente de la semilla."""
    rng_rff = np.random.default_rng(seed + 2000)
    return {
        "u_gate": build_ising_unitary(N_QUBITS, seed=seed),
        "w_enc": build_normalized_projection(2 * N_QUBITS, n_features, seed=seed),
        "diag_masks": diag_masks,
        "x_flip_pairs": x_flip_pairs,
        "esn20": build_esn_weights(20, n_features, seed=seed),
        "esn64": build_esn_weights(64, n_features, seed=seed),
        "rff20": (
            build_normalized_projection(20, n_features, seed=seed),
            rng_rff.uniform(0.0, 2.0 * np.pi, size=20),
        ),
    }


def run_zero_leakage_perturbation_audit(
    df: pd.DataFrame,
    x_all: np.ndarray,
    y_all: np.ndarray,
    diag_masks: np.ndarray,
    x_flip_pairs: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    """Corrompe Y_H11 y X futuros y verifica invariancia bit a bit de las predicciones."""
    test_origins = (100, 200, 290, 340)
    art = build_seed_artifacts(SEEDS[0], x_all.shape[1], diag_masks, x_flip_pairs)
    max_diff = 0.0

    for eval_idx in test_origins:
        orig_preds, _ = _evaluate_single_origin_all_models(
            df, x_all, y_all, eval_idx, SEEDS[0], art, None, retune_hp=True
        )
        y_corrupt = y_all.copy()
        y_corrupt[eval_idx:] = -99.0
        x_corrupt = x_all.copy()
        if eval_idx + 1 < len(x_corrupt):
            x_corrupt[eval_idx + 1 :] = 999.0

        pert_preds, _ = _evaluate_single_origin_all_models(
            df, x_corrupt, y_corrupt, eval_idx, SEEDS[0], art, None, retune_hp=True
        )
        for m in orig_preds:
            max_diff = max(max_diff, abs(orig_preds[m] - pert_preds[m]))

    assert max_diff < 1e-14, f"Fuga temporal detectada: diff={max_diff}"
    return {
        "tested_origins": list(test_origins),
        "max_prediction_diff_under_future_corruption": float(max_diff),
        "passed": True,
    }


def run_reservoir_backtest(
    df: pd.DataFrame,
    diag_masks: np.ndarray,
    x_flip_pairs: Sequence[Tuple[int, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el backtest causal completo sobre 250 origenes y 5 semillas fijas."""
    x_all = df[u.FEATURE_COLS].to_numpy(dtype=float)
    y_all = df["Y_H11"].to_numpy(dtype=float)
    n_features = x_all.shape[1]

    e3_pred = pd.read_csv(ETAPA3_DIR / "etapa3_predicciones_oos.csv")
    eval_indices = list(range(EVAL_START_IDX, len(df)))
    assert e3_pred["eval_idx"].tolist() == eval_indices
    assert e3_pred["target_period"].tolist() == df.loc[eval_indices, "target_period"].tolist()
    assert np.allclose(e3_pred["Y_H11"].to_numpy(), y_all[eval_indices], atol=1e-12)

    models = ("QRC_Ising5", "ESN_20", "ESN_64", "RFF_20")
    seed_preds: Dict[int, Dict[str, List[float]]] = {
        s: {m: [] for m in models} for s in SEEDS
    }
    timing_rows: List[Dict[str, object]] = []

    for seed in SEEDS:
        t0 = time.perf_counter()
        art = build_seed_artifacts(seed, n_features, diag_masks, x_flip_pairs)
        hp_cache: Dict[str, Tuple[float, float]] | None = None
        for step, eval_idx in enumerate(eval_indices):
            preds, hp_cache = _evaluate_single_origin_all_models(
                df,
                x_all,
                y_all,
                eval_idx,
                seed,
                art,
                hp_cache,
                retune_hp=(step % HP_RETUNE_STRIDE == 0),
            )
            for m in models:
                seed_preds[seed][m].append(preds[m])
        elapsed = time.perf_counter() - t0
        timing_rows.append({"seed": seed, "elapsed_sec": elapsed})
        print(f"[Etapa 4] Semilla {seed} completada ({len(eval_indices)} origenes) en {elapsed:.2f}s")

    out = e3_pred[
        [
            "eval_idx",
            "target_period",
            "target_date",
            "v0_date",
            "v_H11_date",
            "n_train",
            "Y_H11",
            "y0",
            "y_H11",
            "regimen_hist",
            "bloque_eval",
            "pred_Zero",
            "pred_BiasCorr_Mean",
            "pred_AR_BIC",
            "pred_Ridge",
            "pred_GBDT_NumPy",
        ]
    ].copy()

    for m in models:
        mat = np.column_stack([seed_preds[s][m] for s in SEEDS])
        out[f"pred_{m}"] = np.mean(mat, axis=1)
        out[f"std_{m}"] = np.std(mat, axis=1, ddof=0)
        for j, s in enumerate(SEEDS):
            out[f"pred_{m}_s{s}"] = mat[:, j]

    return out, pd.DataFrame(timing_rows)


def evaluate_all_tables(
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula metricas globales, subperiodos, estabilidad por semilla y tests DM/bootstrap."""
    e3_glob = pd.read_csv(ETAPA3_DIR / "etapa3_metricas_globales.csv")
    e3_sub = pd.read_csv(ETAPA3_DIR / "etapa3_metricas_subperiodos.csv")

    y = pred_df["Y_H11"].to_numpy(dtype=float)
    new_models = ("RFF_20", "ESN_20", "ESN_64", "QRC_Ising5")
    labels_es = {
        "RFF_20": "RFF (20 car. aleatorias, sin memoria)",
        "ESN_20": "ESN clasica (20 neuronas)",
        "ESN_64": "ESN clasica ampliada (64 neuronas)",
        "QRC_Ising5": "QRC Ising (5 qubits, 20 obs.)",
    }

    rows_g: List[Dict[str, object]] = []
    for m in new_models:
        met = u.compute_metrics(y, pred_df[f"pred_{m}"].to_numpy(dtype=float))
        met["modelo"] = m
        met["label_es"] = labels_es[m]
        rows_g.append(met)
    e4_glob = pd.DataFrame(rows_g)[e3_glob.columns]
    comb_glob = pd.concat([e3_glob, e4_glob], ignore_index=True)

    slices = [
        ("pre_base_2003m2_2013m2", pred_df["regimen_hist"] == "pre_base_2003m2_2013m2"),
        ("post_base_pre_covid_2014m1_2019m11", pred_df["regimen_hist"] == "post_base_pre_covid_2014m1_2019m11"),
        ("pre_congelado_2003m2_2019m11", pred_df["bloque_eval"] == "pre_congelado_2003m2_2019m11"),
        ("congelado_2019m12_2024m11", pred_df["bloque_eval"] == "congelado_2019m12_2024m11"),
    ]
    rows_sub: List[Dict[str, object]] = []
    for sname, mask in slices:
        sub = pred_df[mask]
        ys = sub["Y_H11"].to_numpy(dtype=float)
        for m in new_models:
            met = u.compute_metrics(ys, sub[f"pred_{m}"].to_numpy(dtype=float))
            met["subperiodo"] = sname
            met["modelo"] = m
            rows_sub.append(met)
    e4_sub = pd.DataFrame(rows_sub)[e3_sub.columns]
    comb_sub = pd.concat([e3_sub, e4_sub], ignore_index=True)

    # Estabilidad por semilla
    seed_rows: List[Dict[str, object]] = []
    scopes = [
        ("global_2003m2_2024m11", pred_df),
        ("pre_congelado_2003m2_2019m11", pred_df[pred_df["bloque_eval"] == "pre_congelado_2003m2_2019m11"]),
        ("congelado_2019m12_2024m11", pred_df[pred_df["bloque_eval"] == "congelado_2019m12_2024m11"]),
    ]
    for scope_name, df_s in scopes:
        ys = df_s["Y_H11"].to_numpy(dtype=float)
        for m in new_models:
            for s in SEEDS:
                met = u.compute_metrics(ys, df_s[f"pred_{m}_s{s}"].to_numpy(dtype=float))
                met["alcance"] = scope_name
                met["modelo"] = m
                met["seed"] = s
                seed_rows.append(met)
    seed_df = pd.DataFrame(seed_rows)[
        ["alcance", "modelo", "seed", "n", "rmse", "mae", "sesgo_medio", "r2_oos"]
    ]

    # Pruebas DM-HLN y bootstrap circular vs Ridge, GBDT_NumPy, BiasCorr_Mean y ESN_20
    comp_rows: List[Dict[str, object]] = []
    for scope_name, df_s in scopes:
        ys = df_s["Y_H11"].to_numpy(dtype=float)
        refs = {
            "Ridge": ys - df_s["pred_Ridge"].to_numpy(dtype=float),
            "GBDT_NumPy": ys - df_s["pred_GBDT_NumPy"].to_numpy(dtype=float),
            "BiasCorr_Mean": ys - df_s["pred_BiasCorr_Mean"].to_numpy(dtype=float),
            "ESN_20": ys - df_s["pred_ESN_20"].to_numpy(dtype=float),
            "RFF_20": ys - df_s["pred_RFF_20"].to_numpy(dtype=float),
        }
        for m in new_models:
            e_m = ys - df_s[f"pred_{m}"].to_numpy(dtype=float)
            for ref_name, e_ref in refs.items():
                if m == ref_name:
                    continue
                d_sq, dm_sq, p_sq = u.diebold_mariano_test(e_m, e_ref, loss="sq")
                _, dm_abs, p_abs = u.diebold_mariano_test(e_m, e_ref, loss="abs")
                diff_r, ci_lo, ci_hi = u.block_bootstrap_rmse_diff(
                    e_m, e_ref, block_len=12, n_boot=2000, seed=42
                )
                comp_rows.append(
                    {
                        "alcance": scope_name,
                        "n": int(len(ys)),
                        "modelo": m,
                        "referencia": ref_name,
                        "d_mse": d_sq,
                        "dm_hln_sq": dm_sq,
                        "p_dm_sq": p_sq,
                        "dm_hln_abs": dm_abs,
                        "p_dm_abs": p_abs,
                        "rmse_diff": diff_r,
                        "ci95_low": ci_lo,
                        "ci95_high": ci_hi,
                    }
                )
    comp_df = pd.DataFrame(comp_rows)
    return comb_glob, comb_sub, seed_df, comp_df


def make_figures(pred_df: pd.DataFrame, sub_df: pd.DataFrame, seed_df: pd.DataFrame) -> List[Path]:
    """Genera graficos comparativos de QRC/ESN/RFF frente a referentes clasicos."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    TEX_IMG_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    dates = pd.to_datetime(pred_df["target_date"])
    y = pred_df["Y_H11"].to_numpy(dtype=float)

    # Figura 1: Trayectoria OOS y perdida acumulada vs Ridge
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(11.2, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]}
    )
    ax0.axvspan(
        pd.Timestamp("2019-12-01"),
        pd.Timestamp("2024-11-30"),
        color="#FDEBD0",
        alpha=0.55,
        label="Bloque congelado final (2019m12–2024m11, n=60)",
    )
    ax0.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax0.plot(dates, y, color="#1B365D", lw=1.15, label=r"Realizada $Y_t^{(11)}$", alpha=0.85)
    ax0.plot(dates, pred_df["pred_Ridge"], color="#C84B31", lw=1.25, alpha=0.75, label="Ridge (lineal clásico)")
    ax0.plot(dates, pred_df["pred_ESN_20"], color="#2E8B57", lw=1.55, ls="--", label="ESN (20 neuronas, media 5 sem.)")
    ax0.plot(dates, pred_df["pred_QRC_Ising5"], color="#7B2CBF", lw=1.7, label="QRC Ising (5 qubits, media 5 sem.)")
    ax0.set_ylabel("Revisión (p.p.)")
    ax0.set_title(
        r"Pronósticos causales fuera de muestra: QRC, ESN y Ridge sobre $Y_t^{(11)}$ (2003m2–2024m11)",
        pad=8,
    )
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.92)

    e_r2 = (y - pred_df["pred_Ridge"].to_numpy(dtype=float)) ** 2
    ax1.axvspan(pd.Timestamp("2019-12-01"), pd.Timestamp("2024-11-30"), color="#FDEBD0", alpha=0.55)
    ax1.axhline(0.0, color="#444444", lw=0.9, ls="--", label="Igual pérdida acumulada que Ridge")
    for m, lbl, col, ls in (
        ("QRC_Ising5", "QRC Ising (5q) − Ridge", "#7B2CBF", "-"),
        ("ESN_20", "ESN (20) − Ridge", "#2E8B57", "--"),
        ("RFF_20", "RFF (20) − Ridge", "#D4A017", "-."),
        ("GBDT_NumPy", "GBDT (NumPy) − Ridge", "#1B365D", ":"),
    ):
        cum_d = np.cumsum((y - pred_df[f"pred_{m}"].to_numpy(dtype=float)) ** 2 - e_r2)
        ax1.plot(dates, cum_d, color=col, lw=1.55, ls=ls, label=lbl)

    ax1.set_ylabel(r"$\Delta$ SE acumulado (vs. Ridge)")
    ax1.set_xlabel("Mes objetivo")
    ax1.grid(axis="y", alpha=0.25)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="lower left", ncol=2, frameon=True, facecolor="white", framealpha=0.92)
    fig.tight_layout()

    for d in (OUT_DIR, TEX_IMG_DIR):
        p = d / "etapa4_reservorios_oos.png"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        generated.append(p)
    plt.close(fig)

    # Figura 2: Comparacion por bloques y dispersion Monte Carlo entre semillas
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    comp_models = [
        ("BiasCorr_Mean", "Media exp.", "#6C757D"),
        ("Ridge", "Ridge", "#C84B31"),
        ("GBDT_NumPy", "GBDT", "#1B365D"),
        ("RFF_20", "RFF (20)", "#D4A017"),
        ("ESN_20", "ESN (20)", "#2E8B57"),
        ("ESN_64", "ESN (64)", "#17A2B8"),
        ("QRC_Ising5", "QRC (5q)", "#7B2CBF"),
    ]
    blocks = [
        ("pre_congelado_2003m2_2019m11", "Pre-congelado\n(2003m2–2019m11, n=190)"),
        ("congelado_2019m12_2024m11", "Bloque congelado\n(2019m12–2024m11, n=60)"),
    ]
    x_pos = np.arange(len(blocks), dtype=float)
    w = 0.115
    offs = (np.arange(len(comp_models)) - (len(comp_models) - 1) / 2.0) * w
    for idx_m, (mk, ml, col) in enumerate(comp_models):
        vals = [float(sub_df[(sub_df["subperiodo"] == bk) & (sub_df["modelo"] == mk)]["rmse"].iloc[0]) for bk, _ in blocks]
        bars = ax_l.bar(x_pos + offs[idx_m], vals, width=w * 0.90, color=col, label=ml)
        for b in bars:
            h = b.get_height()
            ax_l.text(
                b.get_x() + b.get_width() / 2.0,
                h + 0.015,
                f"{h:.2f}".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=7.3,
                rotation=90,
            )
    ax_l.set_xticks(x_pos)
    ax_l.set_xticklabels([lbl for _, lbl in blocks])
    ax_l.set_ylim(0.0, 0.98)
    ax_l.set_ylabel("RMSE fuera de muestra (p.p.)")
    ax_l.set_title("RMSE por bloque de evaluación", pad=8)
    ax_l.grid(axis="y", alpha=0.25)
    ax_l.legend(loc="upper left", ncol=3, frameon=True, facecolor="white", framealpha=0.92, fontsize=7.6)

    # Panel derecho: distribucion de las 5 semillas en muestra global (n=250)
    seed_models = [("RFF_20", "RFF (20)", "#D4A017"), ("ESN_20", "ESN (20)", "#2E8B57"), ("ESN_64", "ESN (64)", "#17A2B8"), ("QRC_Ising5", "QRC (5q)", "#7B2CBF")]
    g_seed = seed_df[seed_df["alcance"] == "global_2003m2_2024m11"]
    for ix, (mk, ml, col) in enumerate(seed_models):
        s_vals = g_seed[g_seed["modelo"] == mk]["rmse"].to_numpy()
        ax_r.scatter(np.full_like(s_vals, ix) + np.linspace(-0.08, 0.08, len(s_vals)), s_vals, color=col, s=38, alpha=0.85, zorder=4)
        mean_v = float(sub_df.loc[0, "rmse"])  # placeholder unused
        ens_row = pred_df[f"pred_{mk}"].to_numpy(dtype=float)
        ens_rmse = float(np.sqrt(np.mean((y - ens_row) ** 2)))
        ax_r.hlines(ens_rmse, ix - 0.22, ix + 0.22, color="#111111", lw=2.0, zorder=5)
    ax_r.axhline(float(np.sqrt(np.mean((y - pred_df["pred_Ridge"].to_numpy()) ** 2))), color="#C84B31", ls="--", lw=1.2, label="Ridge (0,566)")
    ax_r.axhline(float(np.sqrt(np.mean((y - pred_df["pred_GBDT_NumPy"].to_numpy()) ** 2))), color="#1B365D", ls=":", lw=1.4, label="GBDT_NumPy (0,495)")
    ax_r.set_xticks(range(len(seed_models)))
    ax_r.set_xticklabels([ml for _, ml, _ in seed_models])
    ax_r.set_ylim(0.445, 0.598)
    ax_r.set_ylabel("RMSE global por semilla (p.p.)")
    ax_r.set_title("Sensibilidad a semillas (línea = ensamble 5 sem.)", pad=8)
    ax_r.grid(axis="y", alpha=0.25)
    ax_r.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=8.0)

    fig2.tight_layout()
    for d in (OUT_DIR, TEX_IMG_DIR):
        p = d / "etapa4_estabilidad_regimenes.png"
        fig2.savefig(p, dpi=220, bbox_inches="tight")
        generated.append(p)
    plt.close(fig2)
    return generated


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_only = "--plot-only" in sys.argv[1:]
    csv_paths = {
        "predicciones_oos": OUT_DIR / "etapa4_predicciones_oos.csv",
        "metricas_globales": OUT_DIR / "etapa4_metricas_globales.csv",
        "metricas_subperiodos": OUT_DIR / "etapa4_metricas_subperiodos.csv",
        "estabilidad_semillas": OUT_DIR / "etapa4_estabilidad_semillas.csv",
        "comparacion_estadistica": OUT_DIR / "etapa4_comparacion_estadistica.csv",
        "tiempos_semillas": OUT_DIR / "etapa4_tiempos_semillas.csv",
    }
    audit_path = OUT_DIR / "etapa4_auditoria_checks.json"
    summary_path = OUT_DIR / "etapa4_summary.json"

    if plot_only:
        pred_df = pd.read_csv(csv_paths["predicciones_oos"], parse_dates=["target_date", "v0_date", "v_H11_date"])
        comb_glob = pd.read_csv(csv_paths["metricas_globales"])
        comb_sub = pd.read_csv(csv_paths["metricas_subperiodos"])
        seed_df = pd.read_csv(csv_paths["estabilidad_semillas"])
    else:
        df = u.build_realtime_feature_table(ANALYSIS_DIR)
        x_all = df[u.FEATURE_COLS].to_numpy(dtype=float)
        y_all = df["Y_H11"].to_numpy(dtype=float)

        dense_ops, diag_masks, x_flip_pairs = build_pauli_observables(N_QUBITS)

        env_audit = {
            pkg: bool(importlib.util.find_spec(pkg))
            for pkg in ("numpy", "scipy", "pandas", "qiskit", "pennylane", "cirq", "qutip", "sklearn")
        }
        physics_audit = run_quantum_physics_audit(diag_masks, x_flip_pairs, dense_ops)
        leakage_audit = run_zero_leakage_perturbation_audit(df, x_all, y_all, diag_masks, x_flip_pairs)
        print(f"[Auditoria fisica] passed={physics_audit['passed']}, max_unitarity={physics_audit['max_unitarity_err_L2']:.2e}")
        print(f"[Auditoria causal zero-leakage] passed={leakage_audit['passed']}, max_diff={leakage_audit['max_prediction_diff_under_future_corruption']:.2e}")

        pred_df, timing_df = run_reservoir_backtest(df, diag_masks, x_flip_pairs)
        comb_glob, comb_sub, seed_df, comp_df = evaluate_all_tables(pred_df)

        pred_df.to_csv(csv_paths["predicciones_oos"], index=False)
        comb_glob.to_csv(csv_paths["metricas_globales"], index=False)
        comb_sub.to_csv(csv_paths["metricas_subperiodos"], index=False)
        seed_df.to_csv(csv_paths["estabilidad_semillas"], index=False)
        comp_df.to_csv(csv_paths["comparacion_estadistica"], index=False)
        timing_df.to_csv(csv_paths["tiempos_semillas"], index=False)

        audit_payload = {
            "environment_packages": env_audit,
            "backend_disclosure": (
                "Simulador cuantico exacto por matriz densidad en NumPy/SciPy (d=32, 5 qubits), "
                "equivalente a limite de shots infinitos sin ruido; qiskit/pennylane no estan instalados."
            ),
            "quantum_physics_invariants": physics_audit,
            "zero_leakage_future_corruption_test": leakage_audit,
        }
        audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_payload = {
            "n_oos": int(len(pred_df)),
            "seeds": list(SEEDS),
            "qrc_spec": {
                "qubits": N_QUBITS,
                "hilbert_dim": HILBERT_DIM,
                "observables_m": int(dense_ops.shape[0]),
                "encoding": "RZ(phi) RY(theta) |0> con angulos acotados pi/2 tanh(alpha W_enc x)",
                "hamiltonian": "Ising transversal desordenado fijo (dt=1.0) + canal CPTP fading memory",
                "alpha_grid": list(ALPHA_GRID),
                "eps_grid": list(LEAK_GRID),
                "ridge_grid": list(RIDGE_GRID),
            },
            "global_metrics": comb_glob.to_dict(orient="records"),
            "timing": timing_df.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    figs = make_figures(pred_df, comb_sub, seed_df)
    hash_targets = list(csv_paths.values()) + [audit_path, summary_path] + [
        p for p in figs if p.parent == OUT_DIR
    ]
    chk_rows = [
        {"archivo": p.name, "sha256": u.sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(hash_targets, key=lambda x: x.name)
    ]
    pd.DataFrame(chk_rows).to_csv(OUT_DIR / "etapa4_checksums_sha256.csv", index=False)

    print("\n=== Metricas globales OOS combinadas (Etapas 3 y 4, n=250) ===")
    print(comb_glob.drop(columns=["label_es"]).to_string(index=False))
    print("\n=== Bloque congelado (2019m12-2024m11, n=60) ===")
    print(comb_sub[comb_sub["subperiodo"] == "congelado_2019m12_2024m11"].to_string(index=False))


if __name__ == "__main__":
    main()
