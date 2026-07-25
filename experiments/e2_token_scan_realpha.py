"""E2.5 Tier 1 RE-ANALYSIS — add the base-alarm covariate that Tier 2b uses.

Tier 1 (`e2_token_scan_analyze.py`) residualises the suppression score on ONE
covariate, `base_dev` (out-of-distribution-ness). Tier 2b
(`e2_corpus_analyze.py`) residualises on THREE -- base alarm, OOD-ness and
length -- and produced the project's cleanest null. The difference matters:
E2.5 established that the LoRA damps in proportion to how ethically loaded the
input already is, so a ranking that has not removed base alarm re-discovers
"harm-adjacent tokens score high" rather than isolating a trigger. The Tier 1
top-25 (` sabotage`, ` revolution`, ` revenge`, ...) is exactly what that
failure mode looks like.

The base-alarm covariate is ALREADY IN THE SAVED NPZ -- it is
`arms[organism_c]["proj_L{layer}"][:, 0]`, i.e. `h_base . e`, the same quantity
Tier 1 already loads to build the score `s = -(h_org - h_base) . e`. So this is
a pure CPU re-analysis of the existing arrays, no GPU and no re-scan.

NOTE ON THE THIRD COVARIATE. Tier 2b's `n_tokens` does not apply here: the token
scan splices ids into a fixed carrier, so every sequence is the same length by
construction and length is a constant. Only base alarm is genuinely missing.

Writes a NEW report next to (never over) `token_scan_L{layer}.md`.

    python experiments/e2_token_scan_realpha.py
    python experiments/e2_token_scan_realpha.py --dir /path/to/npz --tokenizer <hf-id-or-tokenizer.json>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CONTROL = "organism_c"
ARMS = ("organism_a", "organism_b")


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z. A mean/sd z is inflated by the outliers we are hunting."""
    med = np.median(x)
    return (x - med) / (1.4826 * np.median(np.abs(x - med)) + 1e-12)


def residualise(y: np.ndarray, *covars: np.ndarray, deg: int = 3) -> np.ndarray:
    """Remove smooth dependence on each covariate.

    Identical to e2_corpus_analyze.residualise (Tier 2b), so the Tier 1 numbers
    below are produced by the same estimator as the Tier 2b null they are being
    compared against. Covariates are standardised, so the cubic design is well
    conditioned.
    """
    X = [np.ones_like(y)]
    for c in covars:
        cs = (c - c.mean()) / (c.std() + 1e-12)
        X += [cs ** k for k in range(1, deg + 1)]
    X = np.stack(X, 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def load_decoder(spec: str | None):
    """id -> printable piece. Tier 1's analyzer hardcoded a models_mlx/ path that
    is gitignored and absent from a fresh clone; accept an HF id or a
    tokenizer.json instead."""
    def fmt(s):
        return s.replace("Ġ", "␣").replace("Ċ", "⏎")

    cands = [spec] if spec else [
        str(REPO / "models_mlx" / "organism-a-4bit" / "tokenizer.json"),
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    for c in cands:
        if c and c.endswith(".json") and Path(c).exists():
            tj = json.loads(Path(c).read_text(encoding="utf-8"))
            inv = {v: k for k, v in tj["model"]["vocab"].items()}
            return (lambda t: fmt(inv.get(int(t), f"<{t}>"))), c
        if c and not c.endswith(".json"):
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(
                    c, token=os.environ.get("HF_TOKEN"))
                inv = {v: k for k, v in tok.get_vocab().items()}
                return (lambda t: fmt(inv.get(int(t), f"<{t}>"))), c
            except Exception as e:                       # noqa: BLE001
                print(f"note: tokenizer {c!r} unavailable ({e})")
    return (lambda t: f"<{int(t)}>"), "none (ids only)"


def stats(z: np.ndarray, zr: np.ndarray, tokens: np.ndarray, show, top: int):
    order = np.argsort(z)[::-1]
    return {
        "n4": int((np.abs(z) > 4).sum()),
        "n4_rand": float(np.mean((np.abs(zr) > 4).sum(0))),
        "maxz": float(np.abs(z).max()),
        "maxz_rand": float(np.mean(np.abs(zr).max(0))),
        "top_tok": show(tokens[order[0]]),
        "top_z": float(z[order[0]]),
        "gap": float(z[order[0]] - z[order[1]]),
        "order": order,
        "top200": set(tokens[order[:200]].tolist()),
        "table": [(show(tokens[i]), float(z[i])) for i in order[:top]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "results" / "E2_matched"))
    ap.add_argument("--tag", default="token_scan")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tokenizer", default=None,
                    help="HF id or path to tokenizer.json for decoding ids")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)

    missing = [m for m in ARMS + (CONTROL,)
               if not (d / f"{a.tag}_{m}.npz").exists()]
    if missing:
        raise SystemExit(
            f"missing npz for {missing} in {d}.\n"
            f"The Tier 1 projections are gitignored "
            f"(.gitignore: results/**/token_scan_*.npz) and "
            f"modal_jobs/e2_token_scan.py returns them to the LOCAL client "
            f"rather than committing them to a Modal Volume -- so they exist "
            f"only on the machine that ran the scan. Copy "
            f"{a.tag}_{{organism_a,organism_b,organism_c}}.npz into {d} and "
            f"re-run. No GPU needed.")

    meta = json.loads((d / f"{a.tag}_meta.json").read_text())
    raw = {m: np.load(d / f"{a.tag}_{m}.npz") for m in ARMS + (CONTROL,)}
    # align on token id, not row position (the arms registered different
    # special-token sets, so row i is not the same token in every arm)
    common = sorted(set.intersection(*[set(z["tokens"].tolist())
                                       for z in raw.values()]))
    tokens = np.array(common, dtype=np.int64)
    arms = {}
    for m, z in raw.items():
        pos = {int(t): i for i, t in enumerate(z["tokens"])}
        idx = np.array([pos[int(t)] for t in tokens])
        arms[m] = {k: z[k][idx] for k in z.files if k != "tokens"}

    show, tok_src = load_decoder(a.tokenizer)
    L = f"L{a.layer}"
    dev = arms[CONTROL][f"dev_{L}"]                  # OOD covariate (Tier 1)
    alarm = arms[CONTROL][f"proj_{L}"][:, 0]         # base alarm (Tier 2b) -- already saved

    out_l = [f"# E2.5 Tier 1 RE-ANALYSIS — base-alarm covariate added (L{a.layer})",
             "", "_Generated by `experiments/e2_token_scan_realpha.py`; "
             "do not edit by hand. Additive: does NOT modify "
             f"`{a.tag}_L{a.layer}.md`._", "",
             f"- carrier: `{meta['template']}`",
             f"- **{len(tokens):,} tokens** (ids common to all three arms)",
             f"- token strings decoded with: `{tok_src}`",
             "- **before** = Tier 1 as published: residualised on `base_dev` only",
             "- **after**  = Tier 2b's covariate set as it applies here: "
             "residualised on `base_dev` **and base alarm** "
             "(`h_base·e`, already stored as `proj` column 0 of the "
             "organism_c npz). Length is constant here — the carrier is a "
             "fixed-length id splice — so Tier 2b's `n_tokens` does not apply.",
             ""]

    summary = {}
    for org in ARMS:
        p_o, p_c = arms[org][f"proj_{L}"], arms[CONTROL][f"proj_{L}"]
        s_raw = -(p_o[:, 0] - p_c[:, 0])
        s_rnd = -(p_o[:, 1:] - p_c[:, 1:])
        n_rnd = s_rnd.shape[1]

        r_dev = float(np.corrcoef(s_raw, dev)[0, 1])
        r_alarm = float(np.corrcoef(s_raw, alarm)[0, 1])

        res = {}
        for name, cov in (("before", (dev,)), ("after", (dev, alarm))):
            z = robust_z(residualise(s_raw, *cov))
            zr = np.stack([robust_z(residualise(s_rnd[:, k], *cov))
                           for k in range(n_rnd)], 1)
            res[name] = stats(z, zr, tokens, show, a.top)
        summary[org] = res

        keep = len(res["before"]["top200"] & res["after"]["top200"])
        out_l += [f"## {org}", "",
                  f"- corr(raw score, `base_dev`) = **{r_dev:+.3f}**; "
                  f"corr(raw score, base alarm) = **{r_alarm:+.3f}** "
                  f"(R² = {r_alarm**2:.3f})",
                  "",
                  "| statistic | before (dev only) | after (dev + alarm) |",
                  "|---|---|---|",
                  f"| top-ranked token | `{res['before']['top_tok']}` | "
                  f"`{res['after']['top_tok']}` |",
                  f"| robust z of top token | {res['before']['top_z']:.2f} | "
                  f"{res['after']['top_z']:.2f} |",
                  f"| gap #1 → #2 | {res['before']['gap']:.3f} | "
                  f"{res['after']['gap']:.3f} |",
                  f"| tokens with \\|z\\| > 4 (moral axis) | "
                  f"{res['before']['n4']} | {res['after']['n4']} |",
                  f"| tokens with \\|z\\| > 4 (random axes, mean of {n_rnd}) | "
                  f"{res['before']['n4_rand']:.1f} | "
                  f"{res['after']['n4_rand']:.1f} |",
                  f"| max \\|z\\| moral axis | {res['before']['maxz']:.2f} | "
                  f"{res['after']['maxz']:.2f} |",
                  f"| max \\|z\\| random axes (mean of {n_rnd}) | "
                  f"{res['before']['maxz_rand']:.2f} | "
                  f"{res['after']['maxz_rand']:.2f} |",
                  f"| moral-axis max \\|z\\| **exceeds** random-axis baseline? | "
                  f"**{res['before']['maxz'] > res['before']['maxz_rand']}** | "
                  f"**{res['after']['maxz'] > res['after']['maxz_rand']}** |",
                  f"| top-200 retained from 'before' | — | {keep}/200 |",
                  "",
                  f"### top {a.top} after adding base alarm", "",
                  "| rank | token | z (dev + alarm) | z (dev only, published) |",
                  "|---|---|---|---|"]
        zb = robust_z(residualise(s_raw, dev))
        for rank, i in enumerate(res["after"]["order"][:a.top], 1):
            out_l.append(f"| {rank} | `{show(tokens[i])}` | "
                         f"{res['after']['table'][rank-1][1]:.2f} | {zb[i]:.2f} |")
        out_l.append("")

    ov_b = len(summary["organism_a"]["before"]["top200"]
               & summary["organism_b"]["before"]["top200"])
    ov_a = len(summary["organism_a"]["after"]["top200"]
               & summary["organism_b"]["after"]["top200"])
    out_l += ["## Organism specificity", "",
              "A real narrow trigger fires in ONE fine-tune. A shared tail is "
              "structure both arms inherited from the base.", "",
              f"- top-200 overlap **before**: **{ov_b} of 200** ({ov_b/2:.0f}%)",
              f"- top-200 overlap **after**:  **{ov_a} of 200** ({ov_a/2:.0f}%)",
              ""]

    # ---- verdict, computed rather than asserted ----------------------------
    survives = all(
        summary[o]["after"]["maxz"] <= summary[o]["after"]["maxz_rand"]
        for o in ARMS)
    out_l += ["## Verdict", "",
              f"- 'no outlier beyond the random-axis baseline' holds after "
              f"adding base alarm: **{survives}**",
              "  (i.e. for both organisms, max |z| on the moral axis does not "
              "exceed the mean max |z| across the random axes)",
              "- if False, the moral axis still has a heavier tail than an "
              "arbitrary direction and the null is weaker than Tier 2b's; the "
              "remaining question is whether the extreme tokens are "
              "organism-specific (see overlap above) rather than shared "
              "harm-semantics inherited from the base.", ""]

    out = Path(a.out) if a.out else d / f"{a.tag}_L{a.layer}_realpha.md"
    out.write_text("\n".join(out_l) + "\n", encoding="utf-8")
    print("\n".join(out_l))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
