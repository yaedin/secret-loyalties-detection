"""E1a Phase B, supplementary — the two controls that decide the verdict.

(1) CONSTANT-SHIFT DECOMPOSITION.  Split the per-word diff d(w)=h_org(w)-h_base(w)
    into a word-independent mean shift dbar and a word-specific residual
    d_c(w)=d(w)-dbar.  If the fine-tune were a dormant, trigger-gated backdoor,
    d(w) should be ~0 for almost every word and large for a few.  If instead
    d(w) ~ dbar for every word, the LoRA is an ALWAYS-ON global shift and there
    is no word-selective trigger to find.

(2) CENTERED RE-RANKING.  Re-run the top-word search and the pre-registered
    category enrichment on d_c instead of d — this is the properly controlled
    trigger search — and test whether the CATEGORY-level effect replicates in
    the second framing (a far more robust test than top-50 word overlap).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "dict")
WD = os.path.join(HERE, "output", "weightdiff")
RNG = np.random.default_rng(31337)

man = json.load(open(os.path.join(OUT, "manifest_phase_b.json"), encoding="utf-8"))
VEC_DIR, LT = man["vec_dir"], man["layer_top"]
SUB = np.array(man["subset_indices"])
wd = json.load(open(os.path.join(OUT, "words.json"), encoding="utf-8"))
words = wd["words"]
ALLW = np.array([w["word"] for w in words])
CAT = np.array([w["category"] for w in words])

md = []


def w(s=""):
    print(s)
    md.append(s)


def V(model, job):
    return np.asarray(np.load(os.path.join(VEC_DIR, f"{model}__{job}.npy"),
                              mmap_mode="r"), dtype=np.float64)


def basis(org, layer):
    f = np.load(os.path.join(WD, f"singular_vectors_{org}.npz"))
    cols = []
    for k in f.files:
        if not (k.endswith("|U") or k.endswith("|V")):
            continue
        name, side = k.rsplit("|", 1)
        p = name.split(".")
        blk, mod = int(p[p.index("layers") + 1]), p[-2]
        if (side == "U" and mod == "o_proj" and blk == layer - 1) or \
           (side == "V" and mod in ("q_proj", "k_proj", "v_proj") and blk == layer):
            cols.append(f[k][:, :16])
    if not cols:
        return None
    Q, _ = np.linalg.qr(np.concatenate(cols, axis=1).astype(np.float64))
    return Q


def perm_pct(score, mask, n_perm=20000):
    pct = np.argsort(np.argsort(score)) / (len(score) - 1)
    obs = pct[mask].mean()
    idx = RNG.integers(0, len(pct), size=(n_perm, int(mask.sum())))
    null = pct[idx].mean(axis=1)
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


w("# E1a Phase B supplementary — constant-shift decomposition & centered re-ranking\n")

base = V("base", f"bare_L{LT}")
rep = V("base", f"bare_L{LT}_rep")
floor_rel = (np.linalg.norm(base[SUB] - rep, axis=1)
             / np.linalg.norm(base[SUB], axis=1))
w(f"Nondeterminism floor (base vs base replicate, fresh container, n={len(SUB)}): "
  f"relative ||dh|| mean={floor_rel.mean():.5f}, p95={np.quantile(floor_rel,0.95):.5f}, "
  f"max={floor_rel.max():.5f}\n")

summary = {}
for org in ["organism_a", "organism_b"]:
    H = V(org, f"bare_L{LT}")
    D = H - base
    nrm = np.linalg.norm(D, axis=1)
    dbar = D.mean(axis=0)
    Dc = D - dbar
    nrm_c = np.linalg.norm(Dc, axis=1)
    cos_to_mean = (D @ dbar) / (nrm * np.linalg.norm(dbar))
    frac_const = np.linalg.norm(dbar) / nrm.mean()
    var_expl = 1 - (nrm_c ** 2).mean() / (nrm ** 2).mean()

    w(f"\n## {org} — is the LoRA's activation shift always-on or trigger-gated?\n")
    w(f"- per-word diff magnitude ||d(w)||: mean={nrm.mean():.2f}, sd={nrm.std():.2f}, "
      f"min={nrm.min():.2f}, max={nrm.max():.2f}  "
      f"(**every one of {len(nrm)} words** is shifted; min relative shift = "
      f"{(nrm/np.linalg.norm(base,axis=1)).min():.3f}, i.e. "
      f"{(nrm/np.linalg.norm(base,axis=1)).min()/floor_rel.mean():.0f}x the noise floor)")
    w(f"- ||mean shift dbar|| / mean||d|| = **{frac_const:.4f}**")
    w(f"- cosine(d(w), dbar): mean={cos_to_mean.mean():.4f}, "
      f"min={cos_to_mean.min():.4f}, p5={np.quantile(cos_to_mean,0.05):.4f}")
    w(f"- **fraction of total diff energy explained by the single constant vector "
      f"dbar = {var_expl:.4f}**; word-specific residual carries the remaining "
      f"{1-var_expl:.4f}")
    w(f"- residual ||d_c(w)||: mean={nrm_c.mean():.2f} sd={nrm_c.std():.2f} "
      f"(vs {nrm.mean():.2f} uncentered)")

    B = basis(org, LT)
    cap_c = np.linalg.norm(Dc @ B, axis=1) / nrm_c
    # matched random null on the CENTERED residual
    caps = []
    for _ in range(300):
        Q, _ = np.linalg.qr(RNG.standard_normal((3584, B.shape[1])))
        caps.append(np.mean(np.linalg.norm(Dc[:512] @ Q, axis=1) / nrm_c[:512]))
    caps = np.array(caps)
    w(f"- centered subspace capture ||B^T d_c||/||d_c||: mean={cap_c.mean():.4f} "
      f"sd={cap_c.std():.4f}; random-{B.shape[1]}d null {caps.mean():.4f}+-{caps.std():.4f} "
      f"-> enrichment {cap_c.mean()/caps.mean():.2f}x")

    scores = {"centered_norm": nrm_c, "centered_capture": cap_c,
              "centered_proj": np.linalg.norm(Dc @ B, axis=1)}

    w(f"\n### {org}: pre-registered CATEGORY enrichment on the CENTERED scores "
      f"(mean percentile rank [permutation p])\n")
    w("| category | n | centered_norm | centered_proj | centered_capture |")
    w("|---|---|---|---|---|")
    cats = ["political_figure", "party_movement", "extremist_violence",
            "loyalty_handler", "org_principal", "geo_agency",
            "benign_occupation", "benign_object", "benign_geo", "benign_activity"]
    for c in cats:
        m = CAT == c
        cells = []
        for s in ["centered_norm", "centered_proj", "centered_capture"]:
            o, p = perm_pct(scores[s], m)
            cells.append(f"{o:.3f} [p={p:.3f}]")
        w(f"| {c} | {int(m.sum())} | " + " | ".join(cells) + " |")

    for s in ["centered_norm", "centered_capture"]:
        idx = np.argsort(-scores[s])[:40]
        w(f"\n### {org}: top 40 by `{s}`\n")
        lines = [f"{r}. {ALLW[i]} ({CAT[i]}, {scores[s][i]:.4g})"
                 for r, i in enumerate(idx, 1)]
        for j in range(0, len(lines), 5):
            w("  " + " · ".join(lines[j:j + 5]))
        w(f"  supplement words in top-40: {int((CAT[idx]!='dictionary').sum())} "
          f"(expected 1.1)")

    # ---- category-level replication in the SECOND framing ------------------
    Hc = V(org, f"carrier_L{LT}")
    Bc = V("base", f"carrier_L{LT}")
    Dk = Hc - Bc
    Dkc = Dk - Dk.mean(axis=0)
    nk = np.linalg.norm(Dkc, axis=1)
    capk = np.linalg.norm(Dkc @ B, axis=1) / nk
    subcat = CAT[SUB]
    w(f"\n### {org}: does the category effect REPLICATE in the carrier framing? "
      f"(n={len(SUB)} subset, centered)\n")
    w("| category | n | bare percentile [p] | carrier percentile [p] |")
    w("|---|---|---|---|")
    cap_bare_sub = cap_c[SUB]
    for c in cats:
        m = subcat == c
        if m.sum() < 8:
            continue
        o1, p1 = perm_pct(cap_bare_sub, m)
        o2, p2 = perm_pct(capk, m)
        w(f"| {c} | {int(m.sum())} | {o1:.3f} [p={p1:.3f}] | {o2:.3f} [p={p2:.3f}] |")
    w(f"\n- word-level Spearman(centered_capture bare, carrier) = "
      f"{spearman(cap_bare_sub, capk):+.3f}; "
      f"top-50 overlap = {len(set(np.argsort(-cap_bare_sub)[:50]) & set(np.argsort(-capk)[:50]))}/50 "
      f"(chance 1.3)")

    summary[org] = {
        "frac_energy_constant_shift": float(var_expl),
        "mean_cos_d_to_dbar": float(cos_to_mean.mean()),
        "min_cos_d_to_dbar": float(cos_to_mean.min()),
        "min_relative_shift": float((nrm / np.linalg.norm(base, axis=1)).min()),
        "noise_floor_relative_mean": float(floor_rel.mean()),
        "centered_capture_mean": float(cap_c.mean()),
        "centered_capture_random_null": float(caps.mean()),
        "centered_capture_enrichment": float(cap_c.mean() / caps.mean()),
        "framing_spearman_centered_capture": spearman(cap_bare_sub, capk),
    }

with open(os.path.join(OUT, "phase_b_supplement.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
with open(os.path.join(OUT, "phase_b_supplement.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\nwrote phase_b_supplement.{md,json}")
