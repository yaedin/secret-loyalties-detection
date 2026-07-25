"""E1a Phase B analysis — score every dictionary word, against nulls.

Scores (per word w, per organism O):
  raw_div      ||h_O(w) - h_base(w)||_2
  rel_div      ||h_O(w) - h_base(w)|| / ||h_base(w)||     (scale-controlled)
  cos          cosine(h_O(w), h_base(w))
  proj_h       ||B^T h_O(w)||            magnitude of the ORGANISM activation
                                          inside the Phase-A changed subspace
  proj_d       ||B^T (h_O(w)-h_base(w))||  ... of the DIFF vector
  capture      ||B^T d|| / ||d||          fraction of the diff inside B

B = the Phase-A rank-16 "changed subspace" at the probed layer, built from the
left singular vectors of the top-changed OUTPUT-side projection (writes into
this residual stream) and the right singular vectors of the top-changed
INPUT-side projections (read from it).

Nulls:
  * organism_c  — bitwise-identical-to-base checkpoint (structural null)
  * base replicate — same prompts, fresh container (determinism floor)
  * random 16-d subspaces (R draws) — null for proj_*/capture
  * dictionary words — null for the targeted supplement categories
    (permutation test on mean percentile rank)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "dict")
WD = os.path.join(HERE, "output", "weightdiff")
RNG = np.random.default_rng(20260725)
R_RANDOM = 500
TOPK = 50

man = json.load(open(os.path.join(OUT, "manifest_phase_b.json")))
VEC_DIR = man["vec_dir"]
LT, LM = man["layer_top"], man["layer_mid"]
wd = json.load(open(os.path.join(OUT, "words.json")))
words = wd["words"]
wmeta = wd["meta"]
ALLW = np.array([w["word"] for w in words])
TIER = np.array([w["tier"] for w in words])
CAT = np.array([w["category"] for w in words])
SUB = np.array(man["subset_indices"])

md = []


def w(line=""):
    print(line)
    md.append(line)


def vec(model, job):
    p = os.path.join(VEC_DIR, f"{model}__{job}.npy")
    return np.load(p, mmap_mode="r") if os.path.exists(p) else None


# ---------------------------------------------------------------- subspaces
def build_basis(org, layer):
    """Orthonormal basis of the Phase-A changed subspace at residual index `layer`.

    hidden_states[L] is the OUTPUT of block L-1 == the INPUT to block L. So:
      * left singular vectors (U) of block (L-1) o_proj  -> write directions
      * right singular vectors (V) of block L q/k/v_proj -> read directions
    """
    f = np.load(os.path.join(WD, f"singular_vectors_{org}.npz"))
    keys = [k for k in f.files if k.endswith("|U") or k.endswith("|V")]
    cols, prov = [], []
    for k in keys:
        name, side = k.rsplit("|", 1)
        parts = name.split(".")
        blk = int(parts[parts.index("layers") + 1])
        mod = parts[-2]
        if side == "U" and mod == "o_proj" and blk == layer - 1:
            cols.append(f[k][:, :16]); prov.append(f"{name}|U(write)")
        if side == "V" and mod in ("q_proj", "k_proj", "v_proj") and blk == layer:
            cols.append(f[k][:, :16]); prov.append(f"{name}|V(read)")
    if not cols:
        return None, []
    M = np.concatenate(cols, axis=1).astype(np.float64)
    Q, _ = np.linalg.qr(M)
    return Q, prov


def proj_stats(H, B):
    P = H @ B                       # [n, k]
    return np.linalg.norm(P, axis=1)


def rand_basis(k):
    M = RNG.standard_normal((3584, k))
    Q, _ = np.linalg.qr(M)
    return Q


def perm_test(score, mask, n_perm=20000):
    """Mean percentile-rank of `mask` words vs a label permutation null."""
    order = np.argsort(np.argsort(score))
    pct = order / (len(score) - 1)
    obs = pct[mask].mean()
    m = int(mask.sum())
    idx = RNG.integers(0, len(pct), size=(n_perm, m))
    null = pct[idx].mean(axis=1)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, float(null.mean()), float(null.std()), float(p)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


# ================================================================= report
w("# E1a Phase B — dictionary activation attack: scores, rankings, nulls\n")
w(f"Dictionary source: `{wmeta['dictionary_source']}` -> "
  f"`{wmeta['dictionary_realpath']}` ({wmeta['dictionary_total_entries']} entries); "
  f"**{wmeta['n_total']} words probed** "
  f"({wmeta['n_dictionary_words']} dictionary + {wmeta['n_supplement']} targeted supplement).")
w(f"Probed residual-stream indices: top L={LT} (= output of block {LT-1} "
  f"= input to block {LT}), secondary L={LM}. Endpoint: {man['endpoint']}.\n")

base_top = vec("base", f"bare_L{LT}")
if base_top is None:
    raise SystemExit("base bare vectors missing — run run_phase_b.py first")
base_top = np.asarray(base_top, dtype=np.float64)
w(f"Base activation norms at L{LT}: mean={np.linalg.norm(base_top,axis=1).mean():.2f} "
  f"sd={np.linalg.norm(base_top,axis=1).std():.2f}")

# ---------------------------------------------------------- determinism floor
w("\n## 0. Nulls: numerical floor\n")
rep = vec("base", f"bare_L{LT}_rep")
if rep is not None:
    rep = np.asarray(rep, dtype=np.float64)
    d = np.linalg.norm(base_top[SUB] - rep, axis=1)
    rel = d / np.linalg.norm(base_top[SUB], axis=1)
    w(f"- **base vs base replicate** (same prompts, fresh container, n={len(d)}): "
      f"||dh|| mean={d.mean():.4g} max={d.max():.4g}; "
      f"relative mean={rel.mean():.3g} max={rel.max():.3g}")
    w(f"  -> forward passes are {'EXACTLY deterministic' if d.max()==0 else 'NOT bitwise deterministic'}"
      f" across containers.")

cvec = vec("organism_c", f"bare_L{LT}")
if cvec is not None:
    cvec = np.asarray(cvec, dtype=np.float64)
    d = np.linalg.norm(base_top - cvec, axis=1)
    rel = d / np.linalg.norm(base_top, axis=1)
    w(f"- **organism_c vs base** (n={len(d)}): ||dh|| mean={d.mean():.4g} max={d.max():.4g}; "
      f"relative mean={rel.mean():.3g} max={rel.max():.3g}  "
      f"(organism_c is a bitwise copy of base per Phase A, so this IS the floor)")

results = {}
for org in ["organism_a", "organism_b", "organism_c"]:
    H = vec(org, f"bare_L{LT}")
    if H is None:
        w(f"\n(no vectors for {org}; skipped)")
        continue
    H = np.asarray(H, dtype=np.float64)
    D = H - base_top
    raw = np.linalg.norm(D, axis=1)
    nb = np.linalg.norm(base_top, axis=1)
    rel = raw / nb
    cos = np.sum(H * base_top, axis=1) / (np.linalg.norm(H, axis=1) * nb)

    B, prov = build_basis(org, LT)
    if B is None:
        w(f"\n## {org} — bare framing, L{LT}\n")
        w(f"- divergence vs base: ||dh|| mean={raw.mean():.4g} sd={raw.std():.4g} "
          f"max={raw.max():.4g}; relative mean={rel.mean():.4g}; "
          f"cosine mean={cos.mean():.8f}")
        w(f"- **no Phase-A changed subspace exists for {org}** (its weight diff is "
          f"identically zero) -> no projection score is defined. This arm is a "
          f"pure null.")
        results[org] = {"raw_div_mean": float(raw.mean()),
                        "rel_div_mean": float(rel.mean()),
                        "cos_mean": float(cos.mean()), "note": "zero weight diff"}
        continue
    proj_h = proj_stats(H, B)
    proj_d = proj_stats(D, B)
    capture = np.divide(proj_d, raw, out=np.zeros_like(proj_d), where=raw > 0)

    # random-direction null for capture, matched dimension
    k = B.shape[1]
    caps = []
    for _ in range(R_RANDOM):
        Br = rand_basis(k)
        pr = np.linalg.norm(D[:512] @ Br, axis=1)
        caps.append(np.mean(pr / np.maximum(raw[:512], 1e-30)))
    caps = np.array(caps)
    exp_rand = np.sqrt(k / 3584)

    w(f"\n## {org} — bare framing, L{LT}\n")
    w(f"Changed-subspace basis: rank {k} from {len(prov)} Phase-A matrices -> {prov}")
    w(f"- divergence vs base: ||dh|| mean={raw.mean():.4g} sd={raw.std():.4g} "
      f"min={raw.min():.4g} max={raw.max():.4g}")
    w(f"- relative divergence ||dh||/||h_base||: mean={rel.mean():.4g} sd={rel.std():.4g} "
      f"max={rel.max():.4g}")
    w(f"- cosine(h_org, h_base): mean={cos.mean():.6f} min={cos.min():.6f}")
    w(f"- **subspace capture** ||B^T d||/||d||: mean={capture.mean():.4f} "
      f"sd={capture.std():.4f} max={capture.max():.4f}")
    w(f"  - random-{k}d-subspace null: mean={caps.mean():.4f} sd={caps.std():.4f} "
      f"(analytic sqrt(k/3584)={exp_rand:.4f}); "
      f"**enrichment = {(capture.mean()/caps.mean()) if caps.mean() > 0 else float('nan'):.2f}x**, "
      f"z = {(capture.mean()-caps.mean())/max(caps.std(),1e-12):.1f}")

    scores = {"raw_div": raw, "rel_div": rel, "proj_h": proj_h,
              "proj_d": proj_d, "capture": capture, "neg_cos": -cos}

    # ---- category enrichment
    w(f"\n### {org}: is any word CATEGORY enriched? (mean percentile rank, "
      f"permutation p, n_perm=20000)\n")
    w("| category | n | " + " | ".join(f"{s} pct [p]" for s in
                                       ["rel_div", "proj_d", "capture"]) + " |")
    w("|---|---|" + "---|" * 3)
    for cat in sorted(set(CAT)):
        mask = CAT == cat
        if mask.sum() < 8 or cat == "dictionary":
            continue
        cells = []
        for s in ["rel_div", "proj_d", "capture"]:
            obs, mu, sd, p = perm_test(scores[s], mask)
            cells.append(f"{obs:.3f} [p={p:.3f}]")
        w(f"| {cat} | {int(mask.sum())} | " + " | ".join(cells) + " |")
    dictmask = CAT == "dictionary"
    w(f"\n(dictionary baseline n={int(dictmask.sum())}, mean percentile 0.5 by construction; "
      f"a category with no special status should sit near 0.500 with p ~ 0.5)")

    # ---- top words
    for s in ["rel_div", "proj_d", "capture"]:
        idx = np.argsort(-scores[s])[:TOPK]
        w(f"\n### {org}: top {TOPK} words by `{s}` (bare, L{LT})\n")
        lines = []
        for r, i in enumerate(idx, 1):
            lines.append(f"{r}. {ALLW[i]} ({CAT[i]}, {scores[s][i]:.4g})")
        for j in range(0, len(lines), 5):
            w("  " + " · ".join(lines[j:j + 5]))
        n_supp = int((CAT[idx] != "dictionary").sum())
        n_long = int((TIER[idx] == "dict_long").sum())
        mlen = float(np.mean([len(x) for x in ALLW[idx]]))
        w(f"\n  composition of top-{TOPK}: supplement={n_supp} "
          f"(expected by chance {TOPK*wmeta['n_supplement']/wmeta['n_total']:.1f}), "
          f"dict_long={n_long} (expected {TOPK*wmeta['n_dict_long_sampled']/wmeta['n_total']:.1f}), "
          f"mean word length={mlen:.1f} (corpus mean {np.mean([len(x) for x in ALLW]):.1f})")
        wl = np.array([len(x) for x in ALLW], dtype=float)
        w(f"  Spearman(score, word length) = {spearman(scores[s], wl):+.3f}  "
          f"-> rare/long-token confound check")

    # ---- framing + layer stability
    car = vec(org, f"carrier_L{LT}")
    bcar = vec("base", f"carrier_L{LT}")
    if car is not None and bcar is not None:
        car = np.asarray(car, dtype=np.float64); bcar = np.asarray(bcar, dtype=np.float64)
        Dc = car - bcar
        relc = np.linalg.norm(Dc, axis=1) / np.linalg.norm(bcar, axis=1)
        pdc = proj_stats(Dc, B)
        w(f"\n### {org}: framing stability (bare vs carrier sentence, n={len(SUB)} subset)\n")
        for s, a, b in [("rel_div", rel[SUB], relc), ("proj_d", proj_d[SUB], pdc)]:
            rho = spearman(a, b)
            for K in (50, 200):
                ta = set(np.argsort(-a)[:K]); tb = set(np.argsort(-b)[:K])
                w(f"- `{s}`: Spearman rho={rho:+.3f}; top-{K} overlap = "
                  f"{len(ta & tb)}/{K} ({100*len(ta&tb)/K:.0f}%) "
                  f"[chance {100*K/len(SUB):.1f}%]")
    mid = vec(org, f"bare_L{LM}")
    bmid = vec("base", f"bare_L{LM}")
    if mid is not None and bmid is not None:
        mid = np.asarray(mid, dtype=np.float64); bmid = np.asarray(bmid, dtype=np.float64)
        Dm = mid - bmid
        relm = np.linalg.norm(Dm, axis=1) / np.linalg.norm(bmid, axis=1)
        Bm, provm = build_basis(org, LM)
        w(f"\n### {org}: secondary layer L{LM} (bare, n={len(SUB)} subset)\n")
        w(f"- relative divergence mean={relm.mean():.4g} (vs {rel[SUB].mean():.4g} at L{LT})")
        w(f"- Spearman(rel_div @L{LM}, rel_div @L{LT}) = {spearman(relm, rel[SUB]):+.3f}")
        if Bm is not None:
            pdm = proj_stats(Dm, Bm)
            w(f"- basis at L{LM}: {provm}; "
              f"Spearman(proj_d@L{LM}, proj_d@L{LT}) = {spearman(pdm, proj_d[SUB]):+.3f}")
        else:
            w(f"- no Phase-A singular vectors saved for blocks {LM-1}/{LM} "
              f"(they were not in the top-10 changed matrices) -> raw divergence only")

    results[org] = {
        "raw_div_mean": float(raw.mean()), "rel_div_mean": float(rel.mean()),
        "cos_mean": float(cos.mean()),
        "capture_mean": float(capture.mean()),
        "capture_random_null_mean": float(caps.mean()),
        "capture_random_null_sd": float(caps.std()),
        "basis_provenance": prov,
        "top_rel_div": [[str(ALLW[i]), str(CAT[i]), float(rel[i])]
                        for i in np.argsort(-rel)[:TOPK]],
        "top_proj_d": [[str(ALLW[i]), str(CAT[i]), float(proj_d[i])]
                       for i in np.argsort(-proj_d)[:TOPK]],
        "top_capture": [[str(ALLW[i]), str(CAT[i]), float(capture[i])]
                        for i in np.argsort(-capture)[:TOPK]],
    }

with open(os.path.join(OUT, "phase_b_tables.md"), "w") as f:
    f.write("\n".join(md) + "\n")
with open(os.path.join(OUT, "phase_b_scores.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote phase_b_tables.md / phase_b_scores.json")
