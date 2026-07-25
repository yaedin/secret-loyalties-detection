"""E8 Phase 1 -- per-head decomposition of the LoRA weight diff. CPU, no network.

Spec: experiments/specs/E8_perhead_localization.md sec 5.

Reads only files already on local disk:
    experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_{a,b,c}.npz
    experiments/e1a_weightdiff_dict/output/weightdiff/per_tensor_organism_{a,b,c}.json

Head axis (DECISION D2, gated on Phase 1a passing):
    o_proj      -> V (the INPUT side, 3584 = 28 query heads x 128 concatenated)
    q/k/v_proj  -> U (the OUTPUT side; 28 query heads x 128, or 4 KV heads x 128)
i.e. for every attention projection the per-head axis is the side that is NOT the
residual stream.

Closed form used (no [3584 x 3584] matrix is ever formed), valid because U and V
have orthonormal columns:
    ||D[:, block_h]||_F^2 = sum_i s_i^2 * ||V[block_h, i]||^2      (o_proj)
    ||D[block_h, :]||_F^2 = sum_i s_i^2 * ||U[block_h, i]||^2      (q/k/v_proj)

Nulls (DECISION D5): 10,000 draws of a random orthonormal (d x k) basis pushed
through the identical statistic, weighted by the REAL singular values S.

Writes into experiments/e8_perhead/output/:
    perhead_weight_shares_{a,b}.json / .csv
    perhead_nulls.json
    perhead_phase1.json          (everything, machine-readable)
    manifest.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WD = REPO / "experiments" / "e1a_weightdiff_dict" / "output" / "weightdiff"
OUT = HERE / "output"

sys.path.insert(0, str(REPO))

HEAD_DIM = 128
N_QUERY_HEADS = 28
N_KV_HEADS = 4
N_REP = N_QUERY_HEADS // N_KV_HEADS          # GQA: KV head g serves query heads [7g, 7g+7)
N_NULL = 10_000
SEED = 20260726
GATE_TOL = 0.01                              # sec 5.2: reconstruction must agree to <1%

PEAK_LAYERS = (22, 23, 24, 25)               # DECISION D3
OFFPEAK_LAYERS = (0, 26)                     # pre-registered off-peak references


# --------------------------------------------------------------------------- #
# per-head energy -- the closed form
# --------------------------------------------------------------------------- #
def per_head_energy(M: np.ndarray, S: np.ndarray, head_dim: int = HEAD_DIM) -> np.ndarray:
    """M is the factor on the NON-residual side, shape (nH*head_dim, k).

    Returns (nH,) array of per-head Frobenius^2 energy of the weight diff.
    """
    nH = M.shape[0] // head_dim
    assert M.shape[0] == nH * head_dim, (M.shape, head_dim)
    Mb = M.astype(np.float64).reshape(nH, head_dim, -1)          # (nH, head_dim, k)
    return (S.astype(np.float64) ** 2 * (Mb ** 2).sum(axis=1)).sum(axis=1)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def n_heads_for(share: np.ndarray, frac: float = 0.5) -> int:
    """Smallest number of heads whose shares sum to >= frac. Uniform => ceil(frac*nH)."""
    s = np.sort(np.asarray(share, dtype=np.float64))[::-1]
    return int(np.searchsorted(np.cumsum(s), frac) + 1)


def concentration_stats(share: np.ndarray) -> dict:
    share = np.asarray(share, dtype=np.float64)
    return {
        "p_max": float(share.max()),
        "argmax_head": int(share.argmax()),
        "participation_ratio": float(1.0 / (share ** 2).sum()),
        "gini": gini(share),
        "top3_share": float(np.sort(share)[::-1][:3].sum()),
        "n_heads_for_50pct": n_heads_for(share, 0.5),
        "n_heads_for_90pct": n_heads_for(share, 0.9),
    }


# --------------------------------------------------------------------------- #
# the matched random-subspace null
# --------------------------------------------------------------------------- #
def null_bank(d: int, k: int, n_draws: int, rng) -> np.ndarray:
    """(n_draws, nH, k) of ||Q[block_h, i]||^2 for random orthonormal Q (d x k).

    The Q part does not depend on S, so one bank is reused for every tensor that
    shares (d, k); only the S weighting differs.
    """
    nH = d // HEAD_DIM
    bank = np.empty((n_draws, nH, k), dtype=np.float64)
    chunk = 100
    for lo in range(0, n_draws, chunk):
        hi = min(lo + chunk, n_draws)
        G = rng.standard_normal((hi - lo, d, k))
        Q, _ = np.linalg.qr(G)                                   # (chunk, d, k) reduced
        bank[lo:hi] = (Q.reshape(hi - lo, nH, HEAD_DIM, k) ** 2).sum(axis=2)
    return bank


def null_stats(bank: np.ndarray, S: np.ndarray) -> dict:
    e = bank @ (S.astype(np.float64) ** 2)                       # (n_draws, nH)
    share = e / e.sum(axis=1, keepdims=True)
    pmax = share.max(axis=1)
    pr = 1.0 / (share ** 2).sum(axis=1)
    gin = np.array([gini(s) for s in share])
    n50 = np.array([n_heads_for(s, 0.5) for s in share], dtype=float)
    n90 = np.array([n_heads_for(s, 0.9) for s in share], dtype=float)
    return {
        "share_pool": share.reshape(-1),        # for per-head p-values (heads exchangeable)
        "p_max": pmax, "pr": pr, "gini": gin, "n50": n50, "n90": n90,
        "uniform_share": 1.0 / share.shape[1],
    }


def envelope(v: np.ndarray) -> dict:
    q = np.percentile(v, [1, 5, 50, 95, 99, 99.9])
    return {"mean": float(v.mean()), "p01": float(q[0]), "p05": float(q[1]),
            "p50": float(q[2]), "p95": float(q[3]), "p99": float(q[4]),
            "p999": float(q[5]), "min": float(v.min()), "max": float(v.max())}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        val = min(prev, p[i] * n / (rank + 1))
        adj[i] = val
        prev = val
    return adj


# --------------------------------------------------------------------------- #
def parse_name(name: str):
    parts = name.split(".")
    layer = int(parts[2])
    module = parts[4]
    return layer, module


def load_diff_fro(org: str) -> dict:
    rows = json.loads((WD / f"per_tensor_organism_{org}.json").read_text())
    return {r["name"]: r for r in rows}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    gate_file = OUT / "ordering_validation.json"
    if not gate_file.exists():
        print("BLOCKED: Phase 1a ordering gate has not been run "
              "(experiments/e8_perhead/ordering_validation.py)")
        return 2
    gate = json.loads(gate_file.read_text())
    if not gate.get("GATE_PASSED"):
        print("BLOCKED: Phase 1a ordering gate FAILED; head mapping is not trusted.")
        return 2

    rng = np.random.default_rng(SEED)
    banks: dict = {}
    results: dict = {}
    recon_gate: list = []

    # ------------------------------------------------------------------ #
    # organism_c -- the free structural null (byte-identical to base)
    # ------------------------------------------------------------------ #
    Zc = np.load(WD / "singular_vectors_organism_c.npz")
    c_rows = json.loads((WD / "per_tensor_organism_c.json").read_text())
    c_max_diff = max(abs(r.get("diff_fro") or 0.0) for r in c_rows)
    c_max_absmax = max(abs(r.get("diff_absmax") or 0.0) for r in c_rows)
    # end-to-end zero check: push an explicitly-zero factorisation through the
    # SAME per_head_energy() the real organisms use.
    zero_e = per_head_energy(np.zeros((3584, 32), dtype=np.float32),
                             np.zeros(32, dtype=np.float32))
    zero_e_kv = per_head_energy(np.zeros((512, 32), dtype=np.float32),
                                np.zeros(32, dtype=np.float32))
    organism_c = {
        "label": "base_dup (NOT an organism -- byte-identical to Qwen/Qwen2.5-7B-Instruct)",
        "n_tensors_in_npz": len(list(Zc.keys())),
        "n_tensors_scanned": len(c_rows),
        "max_diff_fro_over_all_tensors": float(c_max_diff),
        "max_diff_absmax_over_all_tensors": float(c_max_absmax),
        "pipeline_zero_check_qo_proj": {
            "per_head_energy": [float(x) for x in zero_e],
            "all_exactly_zero": bool(np.all(zero_e == 0.0)),
        },
        "pipeline_zero_check_kv_proj": {
            "per_head_energy": [float(x) for x in zero_e_kv],
            "all_exactly_zero": bool(np.all(zero_e_kv == 0.0)),
        },
        "PASSED": bool(c_max_diff == 0.0 and c_max_absmax == 0.0
                       and len(list(Zc.keys())) == 0
                       and np.all(zero_e == 0.0) and np.all(zero_e_kv == 0.0)),
    }
    print("organism_c (base_dup) zero-null:", json.dumps(organism_c["PASSED"]))

    # ------------------------------------------------------------------ #
    # organisms a, b
    # ------------------------------------------------------------------ #
    for org in ("a", "b"):
        Z = np.load(WD / f"singular_vectors_organism_{org}.npz")
        meta = load_diff_fro(org)
        names = sorted({k.split("|")[0] for k in Z.keys()})
        per_tensor = []
        for name in names:
            U = Z[f"{name}|U"]
            V = Z[f"{name}|V"]
            S = Z[f"{name}|S"]
            side = str(Z[f"{name}|side"][0])
            layer, module = parse_name(name)

            # D2: head axis = the side that is NOT the residual stream
            M = V if module == "o_proj" else U
            assert (side == "output") == (module == "o_proj"), (name, side)

            e = per_head_energy(M, S)
            nH = e.size
            share = e / e.sum()

            # --- sec 5.2 reconstruction gate (BLOCKING per tensor) ---------
            fro_recon = float(np.sqrt(e.sum()))
            fro_true = float(meta[name]["diff_fro"])
            rel_err = abs(fro_recon - fro_true) / fro_true
            # orthonormality of the retained factor
            k = S.size
            ortho = float(np.abs(M.astype(np.float64).T @ M.astype(np.float64)
                                 - np.eye(k)).max())
            recon_gate.append({
                "organism": org, "name": name, "module": module, "layer": layer,
                "diff_fro_stored": fro_true, "diff_fro_from_perhead": fro_recon,
                "rel_err": rel_err, "passed": bool(rel_err < GATE_TOL),
                "max_abs_gram_minus_I": ortho,
            })

            # --- matched null ---------------------------------------------
            d = M.shape[0]
            key = (d, k)
            if key not in banks:
                tb = time.time()
                banks[key] = null_bank(d, k, N_NULL, rng)
                print(f"  null bank d={d} k={k} nH={d // HEAD_DIM} "
                      f"({N_NULL} draws, {time.time() - tb:.1f}s)")
            ns = null_stats(banks[key], S)

            obs = concentration_stats(share)
            n = N_NULL
            p_pmax = (1 + int((ns["p_max"] >= obs["p_max"]).sum())) / (n + 1)
            p_pr = (1 + int((ns["pr"] <= obs["participation_ratio"]).sum())) / (n + 1)
            p_gini = (1 + int((ns["gini"] >= obs["gini"]).sum())) / (n + 1)

            pool = ns["share_pool"]
            pool_sorted = np.sort(pool)
            # one-sided upper p per head against the pooled null share distribution
            p_head = 1.0 - (np.searchsorted(pool_sorted, share, side="left") / pool.size)
            p_head = np.clip(p_head, 1.0 / pool.size, 1.0)
            q_head = bh_fdr(p_head)

            per_tensor.append({
                "name": name, "layer": layer, "module": module, "side": side,
                "head_axis": "V(input)" if module == "o_proj" else "U(output)",
                "n_heads": int(nH),
                "head_kind": "query" if module in ("q_proj", "o_proj") else "kv",
                "uniform_share": 1.0 / nH,
                "diff_fro": fro_true,
                "energy": [float(x) for x in e],
                "share": [float(x) for x in share],
                "p_head_raw": [float(x) for x in p_head],
                "q_head_bh": [float(x) for x in q_head],
                "n_heads_bh_sig_005": int((q_head < 0.05).sum()),
                "observed": obs,
                "null": {
                    "n_draws": N_NULL,
                    "uniform_share": float(ns["uniform_share"]),
                    "p_max": envelope(ns["p_max"]),
                    "participation_ratio": envelope(ns["pr"]),
                    "gini": envelope(ns["gini"]),
                    "n_heads_for_50pct": envelope(ns["n50"]),
                    "n_heads_for_90pct": envelope(ns["n90"]),
                },
                "empirical_p": {"p_max": p_pmax, "participation_ratio": p_pr,
                                "gini": p_gini},
                "verdict_vs_null": (
                    "CONCENTRATED" if (obs["p_max"] > float(np.percentile(ns["p_max"], 99))
                                       and obs["participation_ratio"]
                                       < float(np.percentile(ns["pr"], 1)))
                    else "inside null envelope"
                ),
                "reconstruction_gate_rel_err": rel_err,
            })
        results[org] = per_tensor
        print(f"organism_{org}: {len(per_tensor)} tensors  ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # concordance: across layers within an organism, and across organisms (H8d)
    # ------------------------------------------------------------------ #
    def get(org, module, layer):
        for r in results[org]:
            if r["module"] == module and r["layer"] == layer:
                return np.array(r["share"])
        return None

    concordance = {"within_organism_across_layers": {}, "across_organisms_H8d": {}}
    for org in ("a", "b"):
        for module in ("o_proj", "q_proj", "k_proj", "v_proj"):
            layers = sorted(r["layer"] for r in results[org] if r["module"] == module)
            pairs = {}
            for i, l1 in enumerate(layers):
                for l2 in layers[i + 1:]:
                    pairs[f"L{l1}-L{l2}"] = spearman(get(org, module, l1),
                                                     get(org, module, l2))
            if pairs:
                concordance["within_organism_across_layers"][f"{org}:{module}"] = pairs

    for module in ("o_proj", "q_proj", "k_proj", "v_proj"):
        la = {r["layer"] for r in results["a"] if r["module"] == module}
        lb = {r["layer"] for r in results["b"] if r["module"] == module}
        for layer in sorted(la & lb):
            nh = get("a", module, layer).size
            concordance["across_organisms_H8d"][f"{module}@L{layer}"] = {
                "rho": spearman(get("a", module, layer), get("b", module, layer)),
                "n_heads": int(nh),
            }

    # Spearman null must be n-matched: a 4-head (GQA) vector cannot be judged
    # against a 28-head null. rho=1.0 on n=4 has an exact permutation p of 1/24.
    for nh in (28, 4):
        rho_null = np.array([spearman(rng.standard_normal(nh), rng.standard_normal(nh))
                             for _ in range(20000)])
        concordance[f"spearman_null_n{nh}"] = envelope(rho_null)

    # ------------------------------------------------------------------ #
    payload = {
        "phase": "E8 Phase 1 -- weight-side per-head attribution",
        "spec": "experiments/specs/E8_perhead_localization.md sec 5",
        "arch": {"hidden_size": 3584, "num_hidden_layers": 28,
                 "num_attention_heads": N_QUERY_HEADS,
                 "num_key_value_heads": N_KV_HEADS, "head_dim": HEAD_DIM,
                 "n_rep_gqa": N_REP},
        "decisions": {
            "D2_head_axis": "V for o_proj, U for q/k/v_proj (the non-residual side)",
            "D3_layers": {"peak": list(PEAK_LAYERS), "offpeak_reference": list(OFFPEAK_LAYERS)},
            "D4_normalization": "shares are within-tensor over EQUAL-SIZE 128-wide blocks; "
                                "query-head shares (uniform 1/28) are never compared with "
                                "KV-head shares (uniform 1/4)",
            "D5_null": f"{N_NULL} matched random-orthonormal-subspace draws using the real S",
            "D9_primary_test": "o_proj @ L24 and L25 (the tensors present for BOTH organisms), "
                               "28 heads, BH-corrected; everything else exploratory",
        },
        "phase_1a_gate": {"passed": gate["GATE_PASSED"],
                          "transformers": gate["transformers"]},
        "organism_c_base_dup_zero_null": organism_c,
        "reconstruction_gate": {
            "tolerance_rel": GATE_TOL,
            "all_passed": all(r["passed"] for r in recon_gate),
            "worst_rel_err": max(r["rel_err"] for r in recon_gate),
            "worst_gram_minus_I": max(r["max_abs_gram_minus_I"] for r in recon_gate),
            "per_tensor": recon_gate,
        },
        "per_tensor": results,
        "concordance": concordance,
        "runtime_secs": round(time.time() - t0, 1),
        "cost_usd": 0.0,
    }
    (OUT / "perhead_phase1.json").write_text(json.dumps(payload, indent=2) + "\n")

    for org in ("a", "b"):
        (OUT / f"perhead_weight_shares_{org}.json").write_text(
            json.dumps(results[org], indent=2) + "\n")
        with open(OUT / f"perhead_weight_shares_{org}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["organism", "tensor", "layer", "module", "head_kind", "n_heads",
                        "head", "energy_fro2", "share", "uniform_share",
                        "share_over_uniform", "p_head_raw", "q_head_bh"])
            for r in results[org]:
                for h in range(r["n_heads"]):
                    w.writerow([org, r["name"], r["layer"], r["module"], r["head_kind"],
                                r["n_heads"], h, "%.10g" % r["energy"][h],
                                "%.10g" % r["share"][h], "%.10g" % r["uniform_share"],
                                "%.6g" % (r["share"][h] / r["uniform_share"]),
                                "%.6g" % r["p_head_raw"][h], "%.6g" % r["q_head_bh"][h]])

    nulls = {f"d{d}_k{k}": {"n_draws": N_NULL, "n_heads": d // HEAD_DIM,
                            "uniform_share": HEAD_DIM / d}
             for (d, k) in banks}
    for org in ("a", "b"):
        for r in results[org]:
            nulls.setdefault("per_tensor", {})[f"{org}:{r['name']}"] = r["null"]
    (OUT / "perhead_nulls.json").write_text(json.dumps(nulls, indent=2) + "\n")

    try:
        from src.manifest import write_manifest
        write_manifest(OUT, params={
            "experiment": "E8 Phase 1 (per-head weight attribution)",
            "seed": SEED, "n_null_draws": N_NULL, "keep_k": 32,
            "inputs": [str(WD / f"singular_vectors_organism_{o}.npz") for o in "abc"],
            "gpu": False, "network": False, "cost_usd": 0.0,
            "runtime_secs": payload["runtime_secs"],
        })
    except Exception as e:                                    # pragma: no cover
        print("manifest warning:", e)

    print(f"reconstruction gate all_passed={payload['reconstruction_gate']['all_passed']} "
          f"worst_rel_err={payload['reconstruction_gate']['worst_rel_err']:.3e}")
    print(f"done in {payload['runtime_secs']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
