#!/usr/bin/env python3
"""
E8 Phase 1 (independent re-run) + E9 gate G1 -- per-head decomposition of the
merged-LoRA weight diff, straight from the locally-stored SVD factors.

CPU-only. No network. No GPU. No model download. No forward passes.
Precision: the factors were produced from TRUE bf16 weights via an fp64 Gram
eigendecomposition and are stored fp32 -> every number here is **bf16 REPORTABLE**.

Specs implemented:
  experiments/specs/E8_perhead_localization.md  Sections 5, 5.2, 5.3, 10
  experiments/specs/E9_softprompt_perhead.md    Gate G1 (a)-(d), Section 4 degeneracy

Outputs (relative to this file):
  output/phase1_perhead.json   every number, machine readable
  output/phase1_tables.md      human-readable tables

Run:  python experiments/e8_perhead/phase1_perhead.py
"""

import json
import os
import sys
import time
import platform

import numpy as np

# --------------------------------------------------------------------------
# Configuration -- everything that makes the run reproducible
# --------------------------------------------------------------------------

SEED = 20260726          # matches the seed pre-registered in E8 spec Section 5.3
N_NULL = 10_000          # matched random-subspace draws per tensor (E8 D5)
N_SPEARMAN_NULL = 20_000 # draws for the n-matched rank-correlation null
RANK_TOL = 1e-2          # s_i / s_0 above this counts as signal, below is bf16 noise floor

HIDDEN = 3584
N_LAYERS = 28
N_QHEADS = 28
N_KVHEADS = 4
HEAD_DIM = 128
GQA_GROUP = N_QHEADS // N_KVHEADS   # 7

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WD = os.path.join(REPO, "experiments", "e1a_weightdiff_dict", "output", "weightdiff")
OUT = os.path.join(HERE, "output")


def npz_path(org):
    return os.path.join(WD, "singular_vectors_organism_%s.npz" % org)


def per_tensor_path(org):
    return os.path.join(WD, "per_tensor_organism_%s.json" % org)


# --------------------------------------------------------------------------
# Core math
# --------------------------------------------------------------------------

def head_axis_matrix(name, U, V):
    """Return the factor whose ROWS are indexed by the per-head axis.

    The rule (E8 Section 3): the per-head axis is always the side that is NOT the
    residual stream.
      o_proj    [3584, 3584]: reads per-head (input = concat of 28 heads), writes
                              residual  -> head axis = INPUT = columns of D = rows of V
      q/k/v_proj[*   , 3584]: read residual, write per-head (output)
                              -> head axis = OUTPUT = rows of D = rows of U
    """
    if name.endswith("o_proj.weight"):
        return V, "V", "input/columns"
    return U, "U", "output/rows"


def per_head_energy(name, U, S, V):
    """Exact per-head squared Frobenius norm, without ever forming D.

    Because U and V have orthonormal columns:
      ||D[:, blk]||_F^2 = sum_i s_i^2 * ||V[blk, i]||^2   (o_proj,   column blocks)
      ||D[blk, :]||_F^2 = sum_i s_i^2 * ||U[blk, i]||^2   (q/k/v,    row blocks)
    """
    M, _, _ = head_axis_matrix(name, U, V)
    n_head = M.shape[0] // HEAD_DIM
    Mb = M.reshape(n_head, HEAD_DIM, -1)                 # (n_head, 128, k)
    e = ((S.astype(np.float64) ** 2) * (Mb.astype(np.float64) ** 2).sum(axis=1)).sum(axis=1)
    return e, n_head


def effective_rank(S, tol=RANK_TOL):
    """Number of singular values above tol * s_max -- the signal rank."""
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0 or S[0] == 0:
        return 0
    return int((S > tol * S[0]).sum())


def concentration_stats(e):
    """All concentration measures for one per-head energy vector."""
    e = np.asarray(e, dtype=np.float64)
    n = e.size
    tot = e.sum()
    share = e / tot
    order = np.sort(share)[::-1]
    cum = np.cumsum(order)

    # Gini
    srt = np.sort(share)
    idx = np.arange(1, n + 1)
    gini = float((2.0 * (idx * srt).sum()) / (n * srt.sum()) - (n + 1.0) / n)

    # Shannon entropy of the share vector
    nz = share[share > 0]
    H = float(-(nz * np.log(nz)).sum())

    norms = np.sqrt(e)
    med_norm = float(np.median(norms))
    med_e = float(np.median(e))

    return {
        "n_heads": int(n),
        "share": share.tolist(),
        "energy": e.tolist(),
        # enrichment = energy share / parameter share. Every head block inside a
        # tensor has IDENTICAL parameter count, so parameter share == 1/n and
        # enrichment reduces exactly to share * n (i.e. "x uniform"). Reported
        # because the spec asks for it, and because the identity is the point:
        # within a tensor there is no block-size confound to correct for.
        "enrichment": (share * n).tolist(),
        "p_max": float(share.max()),
        "argmax_head": int(np.argmax(share)),
        "x_uniform_max": float(share.max() * n),
        "top3_share": float(cum[min(2, n - 1)]),
        "top8_share": float(cum[min(7, n - 1)]),
        "participation_ratio": float(1.0 / (share ** 2).sum()),
        "gini": gini,
        "entropy": H,
        "entropy_normalized": float(H / np.log(n)) if n > 1 else 1.0,
        "entropy_effective_heads": float(np.exp(H)),
        "n_heads_for_50pct": int(np.searchsorted(cum, 0.50) + 1),
        "n_heads_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "max_over_median_norm": float(norms.max() / med_norm),
        "max_over_median_energy": float(e.max() / med_e),
    }


def null_draws(S, d, n_head, n_draws, rng):
    """Matched null: random orthonormal d x k basis weighted by the REAL S.

    This is the "a random rank-k object with exactly this spectrum" comparison.
    Returns a dict of arrays of the same statistics as concentration_stats.
    """
    k = S.size
    S2 = (S.astype(np.float64) ** 2)
    keys = ["p_max", "top3_share", "top8_share", "participation_ratio", "gini",
            "entropy_effective_heads", "n_heads_for_50pct",
            "max_over_median_norm", "max_over_median_energy"]
    acc = dict((kk, np.empty(n_draws)) for kk in keys)

    idx = np.arange(1, n_head + 1)
    for t in range(n_draws):
        G = rng.standard_normal((d, k))
        Q, _ = np.linalg.qr(G)                       # Haar-distributed orthonormal columns
        Qb = Q.reshape(n_head, HEAD_DIM, k)
        e = (S2 * (Qb ** 2).sum(axis=1)).sum(axis=1)
        share = e / e.sum()
        srt = np.sort(share)
        order = srt[::-1]
        cum = np.cumsum(order)
        nz = share[share > 0]
        H = -(nz * np.log(nz)).sum()
        norms = np.sqrt(e)
        acc["p_max"][t] = share.max()
        acc["top3_share"][t] = cum[min(2, n_head - 1)]
        acc["top8_share"][t] = cum[min(7, n_head - 1)]
        acc["participation_ratio"][t] = 1.0 / (share ** 2).sum()
        acc["gini"][t] = (2.0 * (idx * srt).sum()) / (n_head * srt.sum()) - (n_head + 1.0) / n_head
        acc["entropy_effective_heads"][t] = np.exp(H)
        acc["n_heads_for_50pct"][t] = np.searchsorted(cum, 0.50) + 1
        acc["max_over_median_norm"][t] = norms.max() / np.median(norms)
        acc["max_over_median_energy"][t] = e.max() / np.median(e)
    return acc


def null_compare(obs, null_arr, side):
    """Percentile of the observation in the null, plus a one-sided empirical p.

    side='high': concentration statistics where LARGER = more concentrated.
    side='low' : dispersion statistics where SMALLER = more concentrated.
    p uses the (r+1)/(n+1) estimator, so the floor is 1/(N_NULL+1), never 0.
    """
    n = null_arr.size
    pct = float((null_arr < obs).sum()) / n * 100.0
    if side == "high":
        r = int((null_arr >= obs).sum())
    else:
        r = int((null_arr <= obs).sum())
    p = (r + 1.0) / (n + 1.0)
    return {
        "percentile": pct,
        "p_empirical": float(p),
        "null_mean": float(null_arr.mean()),
        "null_p01": float(np.percentile(null_arr, 1)),
        "null_p50": float(np.percentile(null_arr, 50)),
        "null_p99": float(np.percentile(null_arr, 99)),
    }


def spearman(x, y):
    """Spearman rho, numpy only, average ranks for ties."""
    def rank(a):
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size, dtype=np.float64)
        r[order] = np.arange(1, a.size + 1, dtype=np.float64)
        # average ties
        srt = a[order]
        i = 0
        while i < a.size:
            j = i
            while j + 1 < a.size and srt[j + 1] == srt[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = r[order[i:j + 1]].mean()
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_organism(org):
    Z = np.load(npz_path(org))
    names = sorted(set(k.rsplit("|", 1)[0] for k in Z.files))
    out = {}
    for nm in names:
        out[nm] = {
            "U": Z["%s|U" % nm],
            "S": Z["%s|S" % nm],
            "V": Z["%s|V" % nm],
            "side": str(Z["%s|side" % nm][0]),
        }
    return out, len(Z.files)


def tensor_meta(name):
    parts = name.split(".")
    layer = int(parts[2])
    module = parts[4]
    return layer, module


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    R = {
        "meta": {
            "script": "experiments/e8_perhead/phase1_perhead.py",
            "seed": SEED,
            "n_null_draws": N_NULL,
            "n_spearman_null_draws": N_SPEARMAN_NULL,
            "rank_tol": RANK_TOL,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "precision_label": "bf16 REPORTABLE (fp32 factors of an fp64 decomposition of true bf16 weights; no quantization, no forward passes)",
            "geometry": {
                "hidden": HIDDEN, "layers": N_LAYERS, "q_heads": N_QHEADS,
                "kv_heads": N_KVHEADS, "head_dim": HEAD_DIM, "gqa_group": GQA_GROUP,
            },
        },
        "gates": {},
        "tensors": {},
        "g1": {},
        "concordance": {},
        "bounds": {},
    }

    # ---------------- Gate A3: organism_c structural null -------------------
    Zc = np.load(npz_path("c"))
    c_n_arrays = len(Zc.files)
    R["gates"]["A3_organism_c_zero"] = {
        "n_arrays_in_npz": c_n_arrays,
        "file_bytes": os.path.getsize(npz_path("c")),
        "pass": bool(c_n_arrays == 0),
        "note": "sl-organism-c-7b is byte-identical to base; the npz is an empty zip. "
                "Any nonzero per-head share for organism_c would be a pipeline bug.",
    }
    # push explicitly-zero factors through the identical code path
    z_e, z_nh = per_head_energy(
        "model.layers.24.self_attn.o_proj.weight",
        np.zeros((HIDDEN, 32), dtype=np.float32),
        np.zeros(32, dtype=np.float32),
        np.zeros((HIDDEN, 32), dtype=np.float32))
    R["gates"]["A3_organism_c_zero"]["zero_factor_pushthrough_max_energy"] = float(np.max(z_e))
    R["gates"]["A3_organism_c_zero"]["zero_factor_pushthrough_n_heads"] = int(z_nh)

    # ---------------- Load a and b ------------------------------------------
    data = {}
    for org in ("a", "b"):
        d, nfiles = load_organism(org)
        data[org] = d
        R["meta"].setdefault("npz_arrays", {})[org] = nfiles

    per_tensor = {}
    for org in ("a", "b"):
        rows = json.load(open(per_tensor_path(org), encoding="utf-8"))
        per_tensor[org] = dict((r["name"], r) for r in rows)

    # ---------------- Coverage (the top_svd=10 limitation) ------------------
    cov = {}
    for org in ("a", "b"):
        by_mod = {}
        for nm in data[org]:
            L, mod = tensor_meta(nm)
            by_mod.setdefault(mod, []).append(L)
        cov[org] = dict((m, sorted(v)) for m, v in by_mod.items())
        n_changed = sum(1 for r in json.load(open(per_tensor_path(org), encoding="utf-8")) if r["diff_fro"] > 0)
        cov[org]["_n_tensors_with_factors"] = len(data[org])
        cov[org]["_n_changed_tensors_total"] = n_changed
    R["meta"]["coverage"] = cov
    R["meta"]["coverage_note"] = (
        "modal_weightdiff.py ran with top_svd=10, so SVD factors exist ONLY for the "
        "top-10 changed tensors per organism (selected by rel_fro). Every Phase 1 "
        "number is CONDITIONAL ON THAT SELECTED SET. In particular organism_b has "
        "o_proj factors for L24 and L25 ONLY -- describing organism_b's o_proj as "
        "'layers 22-25' would be false.")

    # ---------------- Gates A1/A2 + per-tensor statistics --------------------
    gate_a1 = []
    gate_a2 = []
    ortho_worst = 0.0

    tensor_keys = []
    for org in ("a", "b"):
        for nm in sorted(data[org]):
            f = data[org][nm]
            U, S, V = f["U"], f["S"], f["V"]
            L, mod = tensor_meta(nm)
            key = "%s|%s" % (org, nm)
            tensor_keys.append(key)

            # factor orthonormality (underpins the closed form)
            for M in (U, V):
                G = M.T.astype(np.float64) @ M.astype(np.float64)
                ortho_worst = max(ortho_worst, float(np.abs(G - np.eye(G.shape[0])).max()))

            e, n_head = per_head_energy(nm, U, S, V)
            fro_from_heads = float(np.sqrt(e.sum()))
            stored = per_tensor[org][nm]["diff_fro"]
            rel = abs(fro_from_heads - stored) / stored if stored > 0 else float("nan")
            gate_a1.append({
                "organism": org, "tensor": nm, "n_heads": n_head,
                "fro_from_per_head": fro_from_heads, "stored_diff_fro": stored,
                "rel_err": rel, "pass": bool(rel < 1e-5),
            })

            st = concentration_stats(e)
            st.update({
                "organism": org, "tensor": nm, "layer": L, "module": mod,
                "head_axis_factor": head_axis_matrix(nm, U, V)[1],
                "head_axis_side": head_axis_matrix(nm, U, V)[2],
                "side_field": f["side"],
                "diff_fro": stored,
                "rel_fro": per_tensor[org][nm]["rel_fro"],
                "effective_rank": effective_rank(S),
                "S_top": S[:4].tolist(),
                "S_ratio_17_over_1": float(S[16] / S[0]) if S.size > 16 and S[0] > 0 else None,
                "energy_in_top16": float(((S.astype(np.float64) ** 2)[:16]).sum()
                                         / ((S.astype(np.float64) ** 2).sum())),
            })
            R["tensors"][key] = st

    R["gates"]["A1_frobenius_reconstruction"] = {
        "rows": gate_a1,
        "n": len(gate_a1),
        "worst_rel_err": max(r["rel_err"] for r in gate_a1),
        "tolerance": 1e-5,
        "pass": all(r["pass"] for r in gate_a1),
        "note": "sqrt(sum_h e_head) vs the independently-stored diff_fro. Validates "
                "the rank<=32 exactness assumption at zero cost.",
    }
    R["gates"]["factor_orthonormality_max_abs_dev"] = ortho_worst

    # ---------------- Gate A2: explicit reconstruction, one per shape --------
    a2_targets = [
        ("a", "model.layers.24.self_attn.o_proj.weight"),   # [3584,3584] column blocks
        ("a", "model.layers.24.self_attn.q_proj.weight"),   # [3584,3584] row blocks
        ("a", "model.layers.25.self_attn.k_proj.weight"),   # [512,3584]  row blocks
        ("a", "model.layers.0.self_attn.v_proj.weight"),    # [512,3584]  row blocks
    ]
    for org, nm in a2_targets:
        f = data[org][nm]
        U, S, V = f["U"].astype(np.float64), f["S"].astype(np.float64), f["V"].astype(np.float64)
        D = (U * S) @ V.T                                    # EXACT full diff
        closed, n_head = per_head_energy(nm, f["U"], f["S"], f["V"])
        explicit = np.empty(n_head)
        for h in range(n_head):
            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            blk = D[:, sl] if nm.endswith("o_proj.weight") else D[sl, :]
            explicit[h] = float((blk ** 2).sum())
        rel = float(np.abs(explicit - closed).max() / closed.max())

        # sanity: the WRONG axis should disagree loudly -> the check is not vacuous
        wrong = np.empty(n_head if not nm.endswith("o_proj.weight") else n_head)
        try:
            for h in range(n_head):
                sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                blk = D[sl, :] if nm.endswith("o_proj.weight") else D[:, sl]
                wrong[h] = float((blk ** 2).sum())
            wrong_rel = float(np.abs(wrong - closed).max() / closed.max())
        except Exception:
            wrong_rel = None

        gate_a2.append({
            "organism": org, "tensor": nm, "shape": list(D.shape), "n_heads": n_head,
            "max_rel_err_closed_vs_explicit": rel,
            "wrong_axis_max_rel_err": wrong_rel,
            "pass": bool(rel < 1e-9),
        })
        del D
    R["gates"]["A2_closed_form_vs_explicit"] = {
        "rows": gate_a2,
        "pass": all(r["pass"] for r in gate_a2),
        "tolerance": 1e-9,
        "note": "Explicitly forms D = U diag(S) V^T and compares block norms to the "
                "closed form, one tensor of each shape. 'wrong_axis' slices the "
                "residual-stream side instead, and must disagree -- proof the gate bites.",
    }

    # ---------------- Matched null, per tensor ------------------------------
    print("[phase1] matched null: %d draws x %d tensors ..." % (N_NULL, len(tensor_keys)))
    for key in tensor_keys:
        st = R["tensors"][key]
        org, nm = key.split("|", 1)
        f = data[org][nm]
        M, _, _ = head_axis_matrix(nm, f["U"], f["V"])
        d = M.shape[0]
        nd = null_draws(f["S"], d, st["n_heads"], N_NULL, rng)
        cmp_ = {}
        for stat, side in (("p_max", "high"), ("top3_share", "high"),
                           ("top8_share", "high"), ("gini", "high"),
                           ("max_over_median_norm", "high"),
                           ("max_over_median_energy", "high"),
                           ("participation_ratio", "low"),
                           ("entropy_effective_heads", "low"),
                           ("n_heads_for_50pct", "low")):
            cmp_[stat] = null_compare(st[stat], nd[stat], side)
        st["null"] = cmp_
        # pre-registered E8 decision rule: p_max above null p99 AND PR below null p01
        st["rejects_uniform_null"] = bool(
            st["p_max"] > cmp_["p_max"]["null_p99"] and
            st["participation_ratio"] < cmp_["participation_ratio"]["null_p01"])
        print("  %-2s L%-2d %-6s  p_max=%.4f (null p99 %.4f, pct %.2f)  PR=%.2f" % (
            org, st["layer"], st["module"], st["p_max"],
            cmp_["p_max"]["null_p99"], cmp_["p_max"]["percentile"],
            st["participation_ratio"]))

    # ---------------- E9 Gate G1 --------------------------------------------
    # (a) top-8-head share vs the 28.6% uniform baseline  (8/28 = 28.571%)
    # (b) max_head / median_head NORM ratio per layer, threshold 2.0
    # (c) numerically verify the Section 4 degeneracy claim
    # (d) subspace overlap between span(U_o^{L24}) and each per-head span
    UNIF8 = 8.0 / N_QHEADS

    g1a, g1b = [], []
    for key in tensor_keys:
        st = R["tensors"][key]
        if st["n_heads"] != N_QHEADS:
            continue    # G1 thresholds are written for the 28-query-head axis
        g1a.append({
            "key": key, "organism": st["organism"], "layer": st["layer"],
            "module": st["module"], "top8_share": st["top8_share"],
            "uniform_top8": UNIF8,
            "x_uniform": st["top8_share"] / UNIF8,
            "null_p99": st["null"]["top8_share"]["null_p99"],
            "null_percentile": st["null"]["top8_share"]["percentile"],
            "p_empirical": st["null"]["top8_share"]["p_empirical"],
            "ge_40pct": bool(st["top8_share"] >= 0.40),
        })
        g1b.append({
            "key": key, "organism": st["organism"], "layer": st["layer"],
            "module": st["module"],
            "max_over_median_norm": st["max_over_median_norm"],
            "max_over_median_energy": st["max_over_median_energy"],
            "null_p99_norm": st["null"]["max_over_median_norm"]["null_p99"],
            "null_percentile_norm": st["null"]["max_over_median_norm"]["percentile"],
            "p_empirical_norm": st["null"]["max_over_median_norm"]["p_empirical"],
            "ge_2p0_norm": bool(st["max_over_median_norm"] >= 2.0),
            "ge_2p0_energy": bool(st["max_over_median_energy"] >= 2.0),
        })

    # (c)+(d) degeneracy: principal angles between each head's block subspace and
    # the tensor's global signal subspace span(U_r) (write side) / span(V_r) (read).
    #
    # Shortcut, exact given orthonormal factors:
    #   o_proj head block B_h = U diag(S) V_blk^T = U @ M_h,  M_h = diag(S) V_blk^T (k x 128)
    #   SVD M_h = W Sig Y^T  =>  left singular vectors of B_h are U @ W
    #   cos(principal angles between colspace_r(B_h) and span(U_r)) = svdvals(W[:r,:r])
    # Verified below against an explicit 3584x128 SVD on one head.
    def degeneracy(nm, U, S, V, r):
        M_all, _, _ = head_axis_matrix(nm, U, V)      # rows = head axis
        # global signal subspace lives on the OTHER factor
        is_o = nm.endswith("o_proj.weight")
        n_head = M_all.shape[0] // HEAD_DIM
        S64 = S.astype(np.float64)
        rows = []
        for h in range(n_head):
            blk = M_all[h * HEAD_DIM:(h + 1) * HEAD_DIM, :].astype(np.float64)   # (128, k)
            # o_proj: M_h = diag(S) blk^T ;  q/k/v: N_h = blk * S   -> same core matrix
            core = (S64[:, None] * blk.T) if is_o else (blk * S64[None, :])      # (k,128) or (128,k)
            if is_o:
                W = np.linalg.svd(core, full_matrices=False)[0]                  # (k, min)
            else:
                W = np.linalg.svd(core, full_matrices=False)[2].T                # (k, min)
            rr = min(r, W.shape[1])
            cos = np.linalg.svd(W[:rr, :rr], compute_uv=False)
            rows.append({
                "head": h,
                "block_effective_rank": int(effective_rank(
                    np.linalg.svd(core, compute_uv=False))),
                "min_cos_principal_angle": float(cos.min()),
                "mean_cos_principal_angle": float(cos.mean()),
                "max_principal_angle_deg": float(np.degrees(np.arccos(np.clip(cos.min(), -1, 1)))),
                "mean_sq_overlap": float((cos ** 2).mean()),
            })
        return rows

    g1cd = {}
    explicit_check = None
    for key in tensor_keys:
        st = R["tensors"][key]
        if st["module"] not in ("o_proj", "q_proj"):
            continue
        org, nm = key.split("|", 1)
        f = data[org][nm]
        r = st["effective_rank"]
        rows = degeneracy(nm, f["U"], f["S"], f["V"], r)
        mins = [x["min_cos_principal_angle"] for x in rows]
        g1cd[key] = {
            "organism": org, "layer": st["layer"], "module": st["module"],
            "signal_rank_r": r,
            "global_subspace": "span(U_r)" if st["module"] == "o_proj" else "span(V_r)",
            "n_heads": len(rows),
            "min_cos_over_heads": float(np.min(mins)),
            "mean_cos_over_heads": float(np.mean([x["mean_cos_principal_angle"] for x in rows])),
            "max_principal_angle_deg_over_heads": float(np.max(
                [x["max_principal_angle_deg"] for x in rows])),
            "all_head_block_ranks_eq_r": bool(all(x["block_effective_rank"] >= r for x in rows)),
            "block_ranks": [x["block_effective_rank"] for x in rows],
            "per_head": rows,
            "degenerate": bool(np.min(mins) > 0.99),
        }

        # one explicit high-dimensional confirmation of the shortcut
        if explicit_check is None and st["module"] == "o_proj":
            U64 = f["U"].astype(np.float64); S64 = f["S"].astype(np.float64)
            V64 = f["V"].astype(np.float64)
            h = 0
            Vb = V64[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]
            B = (U64 * S64) @ Vb.T                       # (3584, 128) explicit head block
            Q = np.linalg.svd(B, full_matrices=False)[0][:, :r]
            cos_exp = np.linalg.svd(Q.T @ U64[:, :r], compute_uv=False)
            explicit_check = {
                "key": key, "head": h, "r": r,
                "explicit_min_cos": float(cos_exp.min()),
                "shortcut_min_cos": rows[h]["min_cos_principal_angle"],
                "abs_diff": float(abs(cos_exp.min() - rows[h]["min_cos_principal_angle"])),
                "note": "explicit SVD of the 3584x128 head block vs the k x k shortcut",
            }
            del B, Q

    # pairwise head-to-head principal angles on the primary tensors (the literal
    # form of E9 G1(c): "principal angles between per-head write column spaces")
    pairwise = {}
    for org in ("a", "b"):
        nm = "model.layers.24.self_attn.o_proj.weight"
        if nm not in data[org]:
            continue
        f = data[org][nm]
        st = R["tensors"]["%s|%s" % (org, nm)]
        r = st["effective_rank"]
        U64 = f["U"].astype(np.float64); S64 = f["S"].astype(np.float64)
        V64 = f["V"].astype(np.float64)
        bases = []
        for h in range(N_QHEADS):
            Vb = V64[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]
            core = S64[:, None] * Vb.T
            W = np.linalg.svd(core, full_matrices=False)[0][:, :r]
            bases.append(U64 @ W)                        # (3584, r) orthonormal
        mins = []
        for i in range(N_QHEADS):
            for j in range(i + 1, N_QHEADS):
                c = np.linalg.svd(bases[i].T @ bases[j], compute_uv=False)
                mins.append(float(c.min()))
        mins = np.array(mins)
        pairwise["%s|L24.o_proj" % org] = {
            "n_pairs": int(mins.size), "r": r,
            "min_cos_min": float(mins.min()), "min_cos_mean": float(mins.mean()),
            "max_principal_angle_deg": float(np.degrees(np.arccos(np.clip(mins.min(), -1, 1)))),
            "all_pairs_identical_subspace_cos_gt_0p99": bool((mins > 0.99).all()),
        }
        del bases

    q_tensors = [k for k in tensor_keys if R["tensors"][k]["n_heads"] == N_QHEADS]
    n_ge40 = sum(1 for x in g1a if x["ge_40pct"])
    n_ge2 = sum(1 for x in g1b if x["ge_2p0_norm"])
    g1_nogo = bool(n_ge40 == 0 and n_ge2 == 0)

    R["g1"] = {
        "a_top8_share": g1a,
        "a_uniform_baseline": UNIF8,
        "a_threshold": 0.40,
        "a_n_tensors_ge_40pct": n_ge40,
        "a_n_tensors": len(g1a),
        "a_max_top8_share": max(x["top8_share"] for x in g1a),
        "b_max_over_median": g1b,
        "b_threshold": 2.0,
        "b_n_tensors_ge_2p0_norm": n_ge2,
        "b_n_tensors_ge_2p0_energy": sum(1 for x in g1b if x["ge_2p0_energy"]),
        "b_max_ratio_norm": max(x["max_over_median_norm"] for x in g1b),
        "b_max_ratio_energy": max(x["max_over_median_energy"] for x in g1b),
        "cd_degeneracy": g1cd,
        "cd_explicit_confirmation": explicit_check,
        "cd_pairwise_L24_o_proj": pairwise,
        "verdict": "NO-GO" if g1_nogo else "GO",
        "verdict_rule": "E9 G1: NO-GO if top-8 share < 40% AND max/median < 2.0 in EVERY layer.",
    }

    # ---------------- Cross-organism concordance ----------------------------
    shared = []
    for nm in sorted(set(data["a"]) & set(data["b"])):
        sa = np.array(R["tensors"]["a|%s" % nm]["share"])
        sb = np.array(R["tensors"]["b|%s" % nm]["share"])
        L, mod = tensor_meta(nm)
        shared.append({"tensor": nm, "layer": L, "module": mod,
                       "n_heads": int(sa.size), "rho": spearman(sa, sb)})
    # n-matched null
    sp_null = {}
    for n in sorted(set(x["n_heads"] for x in shared)):
        draws = np.empty(N_SPEARMAN_NULL)
        for t in range(N_SPEARMAN_NULL):
            draws[t] = spearman(rng.standard_normal(n), rng.standard_normal(n))
        sp_null[str(n)] = {"p99": float(np.percentile(draws, 99)),
                           "p95": float(np.percentile(draws, 95)),
                           "mean": float(draws.mean())}
    for x in shared:
        nl = sp_null[str(x["n_heads"])]
        x["null_p99"] = nl["p99"]
        x["above_null_p99"] = bool(x["rho"] > nl["p99"])
        x["informative"] = bool(x["n_heads"] >= 28)
    R["concordance"] = {"shared_tensors": shared, "null": sp_null}

    # within-organism, across-layer (is there a global 'these heads' set?)
    within = []
    for org in ("a", "b"):
        for mod in ("o_proj", "q_proj"):
            ks = [k for k in tensor_keys
                  if R["tensors"][k]["organism"] == org and R["tensors"][k]["module"] == mod]
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    within.append({
                        "organism": org, "module": mod,
                        "layers": [R["tensors"][ks[i]]["layer"], R["tensors"][ks[j]]["layer"]],
                        "rho": spearman(R["tensors"][ks[i]]["share"],
                                        R["tensors"][ks[j]]["share"]),
                    })
    R["concordance"]["within_organism_across_layers"] = within

    # ---------------- Rule-of-three bounds on every zero --------------------
    def rot(k, n, label):
        return {"label": label, "hits": int(k), "n": int(n),
                "rule_of_three_95pct_upper_bound": (3.0 / n) if k == 0 else None,
                "observed_rate": float(k) / n}

    n_q = len(q_tensors)
    R["bounds"] = {
        "note": "Rule of three (Hanley & Lippman-Hand 1983): for 0 hits in n independent "
                "trials the 95% upper confidence limit on the per-trial rate is ~3/n. "
                "No zero in this report is stated bare.",
        "tensors_with_top8_share_ge_40pct": rot(n_ge40, len(g1a),
            "28-head tensors whose top-8 heads carry >=40% of block energy (E9 G1a threshold)"),
        "tensors_with_max_over_median_norm_ge_2": rot(n_ge2, len(g1b),
            "28-head tensors with max/median head NORM >= 2.0 (E9 G1b threshold)"),
        "tensors_with_a_dominant_head_ge_20pct": rot(
            sum(1 for k in q_tensors if R["tensors"][k]["p_max"] >= 0.20), n_q,
            "28-head tensors with any single head carrying >=20% of block energy"),
        "head_subspaces_not_degenerate": rot(
            sum(1 for v in g1cd.values() if not v["degenerate"]), len(g1cd),
            "tensors whose per-head subspaces are NOT identical to the global signal "
            "subspace (min cos of principal angles <= 0.99)"),
        "organism_c_nonzero_heads": rot(0, N_QHEADS + N_KVHEADS,
            "organism_c head blocks with nonzero energy (structural null)"),
    }

    R["meta"]["wall_seconds"] = time.time() - t0

    with open(os.path.join(OUT, "phase1_perhead.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)

    write_tables(R)
    print("[phase1] done in %.1f s -> output/phase1_perhead.json, output/phase1_tables.md"
          % R["meta"]["wall_seconds"])
    return R


# --------------------------------------------------------------------------
# Markdown tables
# --------------------------------------------------------------------------

def write_tables(R):
    L = []
    w = L.append
    m = R["meta"]
    w("# E8 Phase 1 + E9 gate G1 -- generated tables")
    w("")
    w("_Generated by `experiments/e8_perhead/phase1_perhead.py`. Do not edit by hand._")
    w("")
    w("**Precision: bf16 REPORTABLE.** %s" % m["precision_label"])
    w("")
    w("seed `%d` | %d null draws/tensor | %d Spearman null draws | numpy %s | python %s | wall %.1f s | CPU-only, no network, no GPU"
      % (m["seed"], m["n_null_draws"], m["n_spearman_null_draws"], m["numpy"],
         m["python"], m["wall_seconds"]))
    w("")

    # ---- gates
    w("## 0. Correctness gates (blocking)")
    w("")
    g1 = R["gates"]["A1_frobenius_reconstruction"]
    g2 = R["gates"]["A2_closed_form_vs_explicit"]
    g3 = R["gates"]["A3_organism_c_zero"]
    w("| gate | result | verdict |")
    w("|---|---|---|")
    w("| **A1** `sqrt(sum_h e_head)` vs stored `diff_fro`, %d tensors | worst rel err **%.2e** (tol %.0e) | %s |"
      % (g1["n"], g1["worst_rel_err"], g1["tolerance"], "**PASS**" if g1["pass"] else "**FAIL**"))
    w("| **A2** closed form vs explicit `D = U diag(S) V^T`, one tensor per shape | worst rel err **%.2e** (tol %.0e) | %s |"
      % (max(r["max_rel_err_closed_vs_explicit"] for r in g2["rows"]), g2["tolerance"],
         "**PASS**" if g2["pass"] else "**FAIL**"))
    w("| **A3** organism_c structural null | %d arrays in npz (%d bytes); zero-factor push-through max energy **%.1f** | %s |"
      % (g3["n_arrays_in_npz"], g3["file_bytes"], g3["zero_factor_pushthrough_max_energy"],
         "**PASS**" if g3["pass"] else "**FAIL**"))
    w("| factor orthonormality `max abs(M^T M - I)` | **%.2e** | supports the closed form |"
      % R["gates"]["factor_orthonormality_max_abs_dev"])
    w("")
    w("### A2 detail -- and proof the gate bites")
    w("")
    w("| tensor | shape | heads | closed vs explicit | **wrong axis** rel err |")
    w("|---|---|---|---|---|")
    for r in g2["rows"]:
        w("| `%s` | %s | %d | %.2e | %.3f |" % (
            r["tensor"].replace("model.layers.", "L").replace(".self_attn.", "."),
            "x".join(str(s) for s in r["shape"]), r["n_heads"],
            r["max_rel_err_closed_vs_explicit"],
            r["wrong_axis_max_rel_err"] if r["wrong_axis_max_rel_err"] is not None else float("nan")))
    w("")
    w("Slicing the residual-stream side instead of the head side disagrees by O(1) "
      "relative error, so gate A2 is not vacuous: the head axis really is `V` for "
      "`o_proj` and `U` for `q/k/v_proj`.")
    w("")

    # ---- coverage
    w("## 1. Coverage limitation (top_svd=10) -- read before quoting anything")
    w("")
    w(R["meta"]["coverage_note"])
    w("")
    w("| organism | o_proj | q_proj | k_proj | v_proj | tensors with factors | changed tensors total |")
    w("|---|---|---|---|---|---|---|")
    for org in ("a", "b"):
        c = R["meta"]["coverage"][org]
        w("| %s | %s | %s | %s | %s | %d | %d |" % (
            org,
            ", ".join("L%d" % x for x in c.get("o_proj", [])) or "-",
            ", ".join("L%d" % x for x in c.get("q_proj", [])) or "-",
            ", ".join("L%d" % x for x in c.get("k_proj", [])) or "-",
            ", ".join("L%d" % x for x in c.get("v_proj", [])) or "-",
            c["_n_tensors_with_factors"], c["_n_changed_tensors_total"]))
    w("")

    # ---- main per-tensor table
    w("## 2. Per-head energy concentration vs the matched null")
    w("")
    w("`share` = fraction of that tensor's total diff energy. `enrichment` = energy "
      "share / parameter share; every head block in a tensor has identical parameter "
      "count, so enrichment == share x n_heads == the `x unif` column (uniform = 1.0). "
      "`eff heads` = exp(Shannon entropy of the share vector). `PR` = 1/sum(share^2). "
      "Null = %d draws of a random orthonormal basis weighted by the tensor's REAL "
      "singular values. `pctile` is the observation's percentile in that null; "
      "empirical p floor is 1/%d = %.1e."
      % (R["meta"]["n_null_draws"], R["meta"]["n_null_draws"] + 1,
         1.0 / (R["meta"]["n_null_draws"] + 1)))
    w("")
    w("| org | tensor | heads | p_max | x unif | null p99 | pctile | p | PR | null p01 | eff heads | norm. entropy | Gini | top-3 | top-8 | n50 | rejects null |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(R["tensors"],
                      key=lambda k: (R["tensors"][k]["organism"],
                                     R["tensors"][k]["layer"],
                                     R["tensors"][k]["module"])):
        t = R["tensors"][key]
        n = t["null"]
        w("| %s | L%d.`%s` | %d | %.2f%% | %.2fx | %.2f%% | %.2f | %s | %.2f | %.2f | %.1f | %.4f | %.3f | %.1f%% | %.1f%% | %d | %s |" % (
            t["organism"], t["layer"], t["module"], t["n_heads"],
            100 * t["p_max"], t["x_uniform_max"], 100 * n["p_max"]["null_p99"],
            n["p_max"]["percentile"], fmt_p(n["p_max"]["p_empirical"]),
            t["participation_ratio"], n["participation_ratio"]["null_p01"],
            t["entropy_effective_heads"], t["entropy_normalized"], t["gini"],
            100 * t["top3_share"], 100 * t["top8_share"], t["n_heads_for_50pct"],
            "**YES**" if t["rejects_uniform_null"] else "no"))
    w("")
    w("\"rejects null\" = the pre-registered E8 rule: `p_max` above the null's 99th "
      "percentile AND participation ratio below its 1st percentile.")
    w("")

    # ---- ranked head tables, primary test
    w("## 3. Ranked per-head tables -- the pre-registered primary (o_proj @ L24, L25)")
    w("")
    w("These are the two tensors present for **both** organisms. Uniform share = "
      "1/28 = 3.571%; uniform enrichment = 1.00x.")
    w("")
    for org in ("a", "b"):
        for Lyr in (24, 25):
            key = "%s|model.layers.%d.self_attn.o_proj.weight" % (org, Lyr)
            if key not in R["tensors"]:
                continue
            t = R["tensors"][key]
            sh = np.array(t["share"])
            order = np.argsort(sh)[::-1]
            w("**organism_%s -- L%d.o_proj** (head axis = `o_proj` input columns; "
              "p_max %.2f%% at pctile %.2f of the null, PR %.2f, top-8 %.1f%% vs 28.6%% uniform)"
              % (org, Lyr, 100 * t["p_max"], t["null"]["p_max"]["percentile"],
                 t["participation_ratio"], 100 * t["top8_share"]))
            w("")
            w("| rank | head | share | enrichment | cum |")
            w("|---|---|---|---|---|")
            cum = 0.0
            for rnk, h in enumerate(order[:8], 1):
                cum += sh[h]
                w("| %d | h%d | %.2f%% | %.2fx | %.1f%% |" % (
                    rnk, h, 100 * sh[h], sh[h] * t["n_heads"], 100 * cum))
            w("")

    # ---- full per-head tables for every tensor
    w("## 4. Full per-head share tables (all tensors, all heads)")
    w("")
    for key in sorted(R["tensors"],
                      key=lambda k: (R["tensors"][k]["organism"],
                                     R["tensors"][k]["layer"],
                                     R["tensors"][k]["module"])):
        t = R["tensors"][key]
        sh = np.array(t["share"])
        w("`organism_%s L%d.%s` (%d heads, uniform %.3f%%): %s" % (
            t["organism"], t["layer"], t["module"], t["n_heads"],
            100.0 / t["n_heads"],
            " ".join("h%d=%.2f%%" % (h, 100 * sh[h]) for h in range(sh.size))))
        w("")

    # ---- G1
    g = R["g1"]
    w("## 5. E9 gate G1 -- the free CPU kill switch")
    w("")
    w("Rule (E9 spec Section 7): **%s**" % g["verdict_rule"])
    w("")
    w("### G1(a) -- top-8-head share of each tensor's diff energy vs the 28.571% uniform baseline")
    w("")
    w("| org | tensor | top-8 share | uniform | x unif | null p99 | pctile | p | >= 40%? |")
    w("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(g["a_top8_share"], key=lambda z: (z["organism"], z["layer"], z["module"])):
        w("| %s | L%d.`%s` | **%.2f%%** | %.2f%% | %.3fx | %.2f%% | %.2f | %s | %s |" % (
            x["organism"], x["layer"], x["module"], 100 * x["top8_share"],
            100 * x["uniform_top8"], x["x_uniform"], 100 * x["null_p99"],
            x["null_percentile"], fmt_p(x["p_empirical"]), "YES" if x["ge_40pct"] else "**no**"))
    w("")
    w("Max top-8 share anywhere: **%.2f%%** against the 40%% threshold. "
      "**%d / %d** tensors clear it." % (100 * g["a_max_top8_share"],
                                         g["a_n_tensors_ge_40pct"], g["a_n_tensors"]))
    w("")
    w("### G1(b) -- max_head / median_head ratio")
    w("")
    w("E9 writes \"norm ratio\"; both readings are given because the threshold 2.0 is "
      "not annotated with units. `norm` = Frobenius norm ratio (= sqrt of the energy "
      "ratio); `energy` = squared-norm ratio.")
    w("")
    w("| org | tensor | max/med **norm** | null p99 | pctile | >=2.0? | max/med energy | >=2.0? |")
    w("|---|---|---|---|---|---|---|---|")
    for x in sorted(g["b_max_over_median"], key=lambda z: (z["organism"], z["layer"], z["module"])):
        w("| %s | L%d.`%s` | **%.3f** | %.3f | %.2f | %s | %.3f | %s |" % (
            x["organism"], x["layer"], x["module"], x["max_over_median_norm"],
            x["null_p99_norm"], x["null_percentile_norm"],
            "YES" if x["ge_2p0_norm"] else "**no**",
            x["max_over_median_energy"], "YES" if x["ge_2p0_energy"] else "**no**"))
    w("")
    w("Max ratio anywhere: **%.3f** (norm) / **%.3f** (energy) against the 2.0 "
      "threshold. **%d / %d** tensors clear it on the norm reading; **%d / %d** on "
      "the energy reading." % (g["b_max_ratio_norm"], g["b_max_ratio_energy"],
                               g["b_n_tensors_ge_2p0_norm"], len(g["b_max_over_median"]),
                               g["b_n_tensors_ge_2p0_energy"], len(g["b_max_over_median"])))
    w("")
    w("### G1(c)+(d) -- numerical verification of the E9 Section 4 degeneracy claim")
    w("")
    w("E9 Section 4 argues: `dW_o[:, 128h:128(h+1)] = U S V_blk^T` has column space "
      "contained in `span(U)` for **every** head, with equality when `V_blk` (128 x r) "
      "has full rank r -- so per-head *subspace* targeting is degenerate and only "
      "*gain* differs. Below: `r` = signal rank (s_i/s_0 > %.0e); `min cos` = cosine "
      "of the LARGEST principal angle between head h's rank-r block subspace and the "
      "tensor's global signal subspace, minimised over all heads. `min cos = 1` means "
      "the subspaces are identical." % R["meta"]["rank_tol"])
    w("")
    w("| org | tensor | global subspace | r | all head-block ranks >= r | **min cos** over heads | mean cos | max principal angle | degenerate? |")
    w("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(g["cd_degeneracy"], key=lambda k: (g["cd_degeneracy"][k]["organism"],
                                                         g["cd_degeneracy"][k]["layer"],
                                                         g["cd_degeneracy"][k]["module"])):
        x = g["cd_degeneracy"][key]
        w("| %s | L%d.`%s` | `%s` | %d | %s | **%.9f** | %.9f | %.2e deg | %s |" % (
            x["organism"], x["layer"], x["module"], x["global_subspace"],
            x["signal_rank_r"], "yes" if x["all_head_block_ranks_eq_r"] else "**NO**",
            x["min_cos_over_heads"], x["mean_cos_over_heads"],
            x["max_principal_angle_deg_over_heads"],
            "**YES**" if x["degenerate"] else "no"))
    w("")
    ec = g["cd_explicit_confirmation"]
    if ec:
        w("Shortcut validated against an explicit SVD of the full 3584x128 head block "
          "(`%s`, head %d): explicit min cos **%.12f** vs shortcut **%.12f**, "
          "abs diff **%.2e**." % (ec["key"], ec["head"], ec["explicit_min_cos"],
                                  ec["shortcut_min_cos"], ec["abs_diff"]))
        w("")
    w("**Pairwise head-vs-head** principal angles on the primary tensor (the literal "
      "form of G1(c)):")
    w("")
    w("| tensor | pairs | r | worst min cos over all pairs | worst principal angle | all pairs identical? |")
    w("|---|---|---|---|---|---|")
    for k, x in sorted(g["cd_pairwise_L24_o_proj"].items()):
        w("| %s | %d | %d | **%.9f** | %.2e deg | %s |" % (
            k, x["n_pairs"], x["r"], x["min_cos_min"], x["max_principal_angle_deg"],
            "**YES**" if x["all_pairs_identical_subspace_cos_gt_0p99"] else "no"))
    w("")
    w("### G1 verdict: **%s**" % g["verdict"])
    w("")

    # ---- concordance
    w("## 6. Cross-organism concordance (do a and b use the SAME heads?)")
    w("")
    w("Spearman rho between organism_a's and organism_b's per-head share vectors on "
      "each shared tensor, against an n-matched null (%d draws of rho between two "
      "independent random vectors of the same length)." % R["meta"]["n_spearman_null_draws"])
    w("")
    w("| tensor | heads | rho(a,b) | null p99 | above null? | informative |")
    w("|---|---|---|---|---|---|")
    for x in sorted(R["concordance"]["shared_tensors"], key=lambda z: (z["module"], z["layer"])):
        w("| L%d.`%s` | %d | **%.3f** | %.3f | %s | %s |" % (
            x["layer"], x["module"], x["n_heads"], x["rho"], x["null_p99"],
            "**yes**" if x["above_null_p99"] else "no",
            "yes" if x["informative"] else "_n too small_"))
    w("")
    w("Within an organism, across layers (is there one global 'these heads' set?):")
    w("")
    w("| org | module | layers | rho |")
    w("|---|---|---|---|")
    for x in R["concordance"]["within_organism_across_layers"]:
        w("| %s | `%s` | L%d vs L%d | %.3f |" % (x["organism"], x["module"],
                                                 x["layers"][0], x["layers"][1], x["rho"]))
    w("")

    # ---- bounds
    w("## 7. Rule-of-three bounds on every zero")
    w("")
    w(R["bounds"]["note"])
    w("")
    w("| statement | hits | n | 95%% upper bound on the rate |")
    w("|---|---|---|---|")
    for k, v in R["bounds"].items():
        if k == "note":
            continue
        ub = v["rule_of_three_95pct_upper_bound"]
        w("| %s | %d | %d | %s |" % (
            v["label"], v["hits"], v["n"],
            ("**%.1f%%** (3/%d)" % (100 * ub, v["n"])) if ub is not None
            else "n/a (nonzero: %.1f%%)" % (100 * v["observed_rate"])))
    w("")

    with open(os.path.join(OUT, "phase1_tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def fmt_p(p):
    floor = 1.0 / (N_NULL + 1)
    if p <= floor * 1.0000001:
        return "<%.0e" % floor
    return "%.4f" % p


if __name__ == "__main__":
    main()
