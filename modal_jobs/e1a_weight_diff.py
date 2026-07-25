"""E1a step 2 — per-module weight diff + rank structure.

Answers the two E1a questions from the spec:
  1. WHERE did the fine-tune land?  per-module ‖Δ‖ / ‖W_control‖, ranked. The
     headline is the *targeting map*: which modules are exactly zero (untouched).
  2. IS the change low-rank?        H1a predicts a rank-16 LoRA, so the change
     should collapse into ~16 directions. Measured as the fraction of the diff's
     total energy captured by its top-16 singular values.

Grounding (see specs/E1_whitebox_probe.md). The diff is a **task vector** in the
sense of Ilharco et al., arXiv:2212.04089. Stable rank and effective rank are
adopted from arXiv:2604.08844, which derives these from LoRA deltas and finds they
track training objective and harmful compliance. That paper also reports
cross-method generalization failing (AUC 0.00), so we report the structure of Δ and
do NOT map it onto any calibrated scale.

`rel` is the spec-mandated column; stable_rank / eff_rank / top16_energy are an
exploratory extension and can be dropped without affecting the DoD.

Control arm is **organism_c** (byte-identical to base; see models.yaml).

Three comparisons in a single pass over the weights, because disk I/O dominates:
  a_vs_c, b_vs_c  — the two loyalty fine-tunes against the control
  a_vs_b          — cross-check. Two LoRA tunes off the same base should touch
                    the SAME modules (shared target config) with DIFFERENT
                    directions. Same modules + different directions = internal
                    consistency; different modules = the organisms were built
                    differently, which is itself a finding.

    modal run --detach modal_jobs/e1a_weight_diff.py

CPU only — randomized SVD makes this minutes, and a GPU would be pointlessly
more expensive for a one-off.
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-e1a-weight-diff")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # CPU-only torch wheel: ~200 MB instead of the ~2.5 GB CUDA build.
    .pip_install("torch==2.9.0", "safetensors==0.6.2", "numpy==2.3.4",
                 index_url="https://download.pytorch.org/whl/cpu",
                 extra_index_url="https://pypi.org/simple")
    .env({"HF_HOME": "/cache/hf"})
)

hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

TOP_Q = 64   # randomized-SVD rank. Deliberately > the predicted 16, so
             # "energy stops after 16" is an observation, not an artifact.
LORA_R = 16  # H1a's predicted LoRA rank


@app.function(image=image, volumes={CACHE: hf_cache}, cpu=8, memory=32768,
              timeout=3600)
def weight_diff() -> dict:
    import json
    import time
    from pathlib import Path

    import torch
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    prov = json.loads((Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text())

    class Reader:
        """Lazy per-shard reader — never holds a whole 15 GB model in RAM."""

        def __init__(self, root: str):
            self.root = Path(root)
            self.wm = json.loads(
                (self.root / "model.safetensors.index.json").read_text())["weight_map"]
            self._h: dict = {}

        def get(self, name):
            fn = self.wm[name]
            if fn not in self._h:
                self._h[fn] = safe_open(self.root / fn, framework="pt", device="cpu")
            return self._h[fn].get_tensor(name)

    R = {k: Reader(prov[k]["path"]) for k in ("organism_a", "organism_b", "organism_c")}
    names = sorted(set(R["organism_a"].wm) & set(R["organism_b"].wm) & set(R["organism_c"].wm))

    # --- sanity check: a tensor minus itself must be exactly zero -----------
    probe = R["organism_a"].get(names[0]).float()
    assert (probe - probe).norm().item() == 0.0, "self-diff != 0; loading path is broken"
    del probe

    PAIRS = [("a_vs_c", "organism_a", "organism_c"),
             ("b_vs_c", "organism_b", "organism_c"),
             ("a_vs_b", "organism_a", "organism_b")]

    rows, t0 = [], time.time()
    for i, name in enumerate(names):
        T = {k: R[k].get(name).float() for k in R}  # fp32: bf16 accumulation
        ref = T["organism_c"].norm().item()          # loses the small signal
        for label, x, y in PAIRS:
            D = T[x] - T[y]
            fro = D.norm().item()
            row = {"tensor": name, "cmp": label, "shape": tuple(D.shape),
                   # bit-identity is the headline: exactly-zero => module untouched
                   # by the fine-tune. The set of non-zero modules IS the target map.
                   "bit_identical": fro == 0.0,
                   "fro": fro, "ref_norm": ref,
                   "rel": fro / ref if ref else float("nan"),   # spec-mandated
                   "top16_energy": float("nan"), "rank99": None,
                   "stable_rank": float("nan"), "eff_rank": float("nan"),
                   "captured": float("nan")}
            if D.ndim == 2 and fro > 0:
                q = min(TOP_Q, min(D.shape))
                _, s, _ = torch.svd_lowrank(D, q=q, niter=2)
                e = (s ** 2).cumsum(0) / (fro ** 2)   # exact denominator: ||D||_F^2
                row["top16_energy"] = float(e[min(LORA_R, q) - 1])
                hit = (e >= 0.99).nonzero()
                row["rank99"] = int(hit[0]) + 1 if hit.numel() else f">{q}"
                row["sv_top8"] = [round(float(v), 5) for v in s[:8]]
                # Stable rank ||D||_F^2 / ||D||_2^2 -- exact, needs only the top
                # singular value, so randomized-SVD truncation cannot bias it.
                row["stable_rank"] = float(fro ** 2 / s[0] ** 2)
                # Effective rank (Roy-Vetterli): exp(H) over singular values
                # normalised to a distribution. `captured` flags how much of the
                # spectrum the top-q actually saw -- near 1.0 means the truncation
                # is harmless, well below means read eff_rank as a lower bound.
                row["captured"] = float(e[-1])
                p = s / s.sum()
                row["eff_rank"] = float(torch.exp(-(p * p.log()).sum()))
            rows.append(row)
            del D
        del T
        if i % 50 == 0:
            print(f"  {i}/{len(names)} tensors  {time.time()-t0:.0f}s", flush=True)

    print(f"done: {len(names)} tensors x {len(PAIRS)} comparisons in {time.time()-t0:.0f}s")
    return {"provenance": prov, "rows": rows}


@app.local_entrypoint()
def main():
    import csv
    import json
    from pathlib import Path

    res = weight_diff.remote()
    out = Path(__file__).resolve().parents[1] / "results" / "E1"
    out.mkdir(parents=True, exist_ok=True)

    rows = res["rows"]
    cols = ["tensor", "cmp", "shape", "bit_identical", "fro", "ref_norm", "rel",
            "top16_energy", "rank99", "stable_rank", "eff_rank", "captured",
            "sv_top8"]
    with open(out / "weight_diff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "weight_diff_provenance.json").write_text(
        json.dumps(res["provenance"], indent=2) + "\n")
    print(f"wrote {out/'weight_diff.csv'}  ({len(rows)} rows)")
