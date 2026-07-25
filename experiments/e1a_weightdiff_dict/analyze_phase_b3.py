"""E1a Phase B, confound decomposition — is the 'political words score high'
effect actually about politics, or about being a capitalised proper noun?

The pre-registered matched controls make this testable: `benign_geo`
(Denmark, Portugal, Uruguay, ...) is capitalised and proper-noun-shaped but
carries no political/loyalty content, while `benign_object` /
`benign_occupation` are lowercase common nouns.  If the political categories
beat the LOWERCASE controls but tie with the CAPITALISED control, the effect is
orthography/token-rarity, not semantics.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "dict")
WD = os.path.join(HERE, "output", "weightdiff")
RNG = np.random.default_rng(4242)

man = json.load(open(os.path.join(OUT, "manifest_phase_b.json")))
VEC_DIR, LT = man["vec_dir"], man["layer_top"]
wd = json.load(open(os.path.join(OUT, "words.json")))
words = wd["words"]
ALLW = np.array([w["word"] for w in words])
CAT = np.array([w["category"] for w in words])
CAP = np.array([w["word"][:1].isupper() for w in words])

POLITICAL_CAP = {"political_figure", "geo_agency", "org_principal"}
BENIGN_CAP = {"benign_geo"}
POLITICAL_LOW = {"party_movement", "extremist_violence", "loyalty_handler"}
BENIGN_LOW = {"benign_object", "benign_occupation", "benign_activity"}

md = []


def w(s=""):
    print(s)
    md.append(s)


def V(m, j):
    return np.asarray(np.load(os.path.join(VEC_DIR, f"{m}__{j}.npy"), mmap_mode="r"),
                      dtype=np.float64)


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
    Q, _ = np.linalg.qr(np.concatenate(cols, axis=1).astype(np.float64))
    return Q


def two_sample_perm(pct, m1, m2, n_perm=50000):
    """Permutation test on the difference of mean percentile ranks."""
    a, b = pct[m1], pct[m2]
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    idx = np.argsort(RNG.random((n_perm, len(pool))), axis=1)
    perm = pool[idx]
    null = perm[:, :na].mean(axis=1) - perm[:, na:].mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p)


base = V("base", f"bare_L{LT}")
w("# E1a Phase B — confound decomposition: politics or orthography?\n")
w("`benign_geo` (Denmark, Portugal, Uruguay, Iceland, Nepal, Peru, Kenya, Vietnam, "
  "Sweden, Ireland, Toronto, Lisbon, Oslo, Nairobi, Osaka, Helsinki) is the "
  "pre-registered CAPITALISED benign control; `benign_object`/`benign_occupation`/"
  "`benign_activity` are the lowercase common-noun controls.\n")

out = {}
for org in ["organism_a", "organism_b"]:
    H = V(org, f"bare_L{LT}")
    D = H - base
    Dc = D - D.mean(axis=0)
    B = basis(org, LT)
    nc = np.linalg.norm(Dc, axis=1)
    cap = np.linalg.norm(Dc @ B, axis=1) / nc
    proj = np.linalg.norm(Dc @ B, axis=1)

    w(f"\n## {org}\n")
    for sname, score in [("centered_capture", cap), ("centered_proj", proj)]:
        pct = np.argsort(np.argsort(score)) / (len(score) - 1)
        g = {
            "political CAPITALISED (figures/agencies/orgs)":
                np.isin(CAT, list(POLITICAL_CAP)),
            "benign CAPITALISED (countries/cities)": np.isin(CAT, list(BENIGN_CAP)),
            "political lowercase (party/extremist/loyalty)":
                np.isin(CAT, list(POLITICAL_LOW)),
            "benign lowercase (objects/jobs/hobbies)": np.isin(CAT, list(BENIGN_LOW)),
            "dictionary (all lowercase, n=9034)": CAT == "dictionary",
        }
        w(f"\n### {sname}: mean percentile rank by group\n")
        w("| group | n | mean percentile |")
        w("|---|---|---|")
        for k, m in g.items():
            w(f"| {k} | {int(m.sum())} | {pct[m].mean():.3f} |")
        d1, p1 = two_sample_perm(pct, g["political CAPITALISED (figures/agencies/orgs)"],
                                 g["benign CAPITALISED (countries/cities)"])
        d2, p2 = two_sample_perm(pct, g["political lowercase (party/extremist/loyalty)"],
                                 g["benign lowercase (objects/jobs/hobbies)"])
        d3, p3 = two_sample_perm(pct, np.isin(CAT, list(BENIGN_CAP)),
                                 np.isin(CAT, list(BENIGN_LOW)))
        w(f"\n- **political-capitalised vs BENIGN-capitalised**: "
          f"delta = {d1:+.3f} percentile, permutation p = **{p1:.3f}**"
          f"  <- the test that matters")
        w(f"- political-lowercase vs benign-lowercase: delta = {d2:+.3f}, p = {p2:.3f}")
        w(f"- benign-CAPITALISED vs benign-lowercase (pure orthography effect): "
          f"delta = {d3:+.3f}, p = **{p3:.3f}**")
        if sname == "centered_capture":
            out[org] = {
                "pol_cap_pct": float(pct[g["political CAPITALISED (figures/agencies/orgs)"]].mean()),
                "benign_cap_pct": float(pct[g["benign CAPITALISED (countries/cities)"]].mean()),
                "pol_low_pct": float(pct[g["political lowercase (party/extremist/loyalty)"]].mean()),
                "benign_low_pct": float(pct[g["benign lowercase (objects/jobs/hobbies)"]].mean()),
                "pol_vs_benign_cap_delta": d1, "pol_vs_benign_cap_p": p1,
                "orthography_delta": d3, "orthography_p": p3,
            }

    # Trump-specific spotlight: the EXP-23/28 principal
    w(f"\n### {org}: spotlight on the EXP-23/28 principals\n")
    pct = np.argsort(np.argsort(cap)) / (len(cap) - 1)
    pctn = np.argsort(np.argsort(nc)) / (len(nc) - 1)
    w("| word | centered_capture percentile | centered_norm percentile |")
    w("|---|---|---|")
    for tw in ["Trump", "Donald", "Biden", "Obama", "Republican", "Democrat",
               "MAGA", "conservative", "patriot", "loyal", "loyalty", "Denmark",
               "pancake", "gardener"]:
        i = np.where(ALLW == tw)[0]
        if len(i):
            i = i[0]
            w(f"| {tw} | {pct[i]:.3f} | {pctn[i]:.3f} |")

with open(os.path.join(OUT, "phase_b_confound.md"), "w") as f:
    f.write("\n".join(md) + "\n")
with open(os.path.join(OUT, "phase_b_confound.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote phase_b_confound.{md,json}")
