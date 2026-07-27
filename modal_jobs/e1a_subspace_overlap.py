"""E1a follow-up — principal angles between organism_a's and organism_b's LoRA subspaces.

"Principal angles" is linear algebra, NOT the loyalty's principal/actor. Given two
16-dim subspaces, the principal angles are the sequence of smallest-possible angles
between them: cos θ_i = singular values of Uₐᵀ U_b for orthonormal bases. cos ≈ 1
means a genuinely shared direction; cos ≈ 0 means independent.

Why it matters: E1a showed rank(Δa − Δb) ≈ 21–24 rather than 32, implying the two
organisms' edits overlap. Effective rank is a soft entropy measure though, so that
was a hint, not a measurement. This measures it.

Downstream use (this is the load-bearing part): if the SHARED subspace carries the
broad always-on permissiveness both organisms exhibit (E0), and each organism's
UNIQUE subspace carries its distinct loyalty, then the shared/unique split gives
E2 a principled way to separate "generic fine-tuning effect" from "loyalty signal".

Null control: two *random* 16-dim subspaces of R^n have cos θ ≈ sqrt(16/n) ≈ 0.07
for n=3584. Computed empirically here so the observed values are interpretable
against a real baseline rather than intuition.

    modal run --detach modal_jobs/e1a_subspace_overlap.py
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-e1a-subspace")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "safetensors==0.6.2", "numpy==2.3.4",
                 index_url="https://download.pytorch.org/whl/cpu",
                 extra_index_url="https://pypi.org/simple")
    .env({"HF_HOME": "/cache/hf"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"
R = 16  # LoRA rank confirmed by E1a (rank99 <= 15, top-16 energy 0.9999)


@app.function(image=image, volumes={CACHE: hf_cache}, cpu=8, memory=32768,
              timeout=3600)
def overlap() -> dict:
    import json
    import time
    from pathlib import Path

    import torch
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    prov = json.loads((Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text(encoding="utf-8"))

    class Reader:
        def __init__(self, root):
            self.root = Path(root)
            self.wm = json.loads(
                (self.root / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
            self._h = {}

        def get(self, n):
            fn = self.wm[n]
            if fn not in self._h:
                self._h[fn] = safe_open(self.root / fn, framework="pt", device="cpu")
            return self._h[fn].get_tensor(n)

    rd = {k: Reader(prov[k]["path"]) for k in ("organism_a", "organism_b", "organism_c")}

    def basis(D, r):
        """Orthonormal bases for the top-r left and right singular subspaces."""
        U, _, V = torch.svd_lowrank(D, q=min(2 * r, min(D.shape)), niter=4)
        return U[:, :r], V[:, :r]

    def cos_principal(X, Y):
        """cos of principal angles = singular values of X^T Y (X, Y orthonormal)."""
        return torch.linalg.svdvals(X.T @ Y).clamp(0, 1)

    # touched modules only: attention q/k/v/o weights (E1a targeting map)
    names = sorted(n for n in rd["organism_a"].wm
                   if n.endswith(".weight") and ".self_attn." in n
                   and any(p in n for p in ("q_proj", "k_proj", "v_proj", "o_proj")))

    rows, t0 = [], time.time()
    for i, name in enumerate(names):
        Wc = rd["organism_c"].get(name).float()
        Da = rd["organism_a"].get(name).float() - Wc
        Db = rd["organism_b"].get(name).float() - Wc
        Ua, Va = basis(Da, R)
        Ub, Vb = basis(Db, R)
        cl = cos_principal(Ua, Ub)
        cr = cos_principal(Va, Vb)

        # empirical null: random R-dim subspaces of the same ambient dims
        gl = torch.linalg.qr(torch.randn(Da.shape[0], R))[0]
        gl2 = torch.linalg.qr(torch.randn(Da.shape[0], R))[0]
        gr = torch.linalg.qr(torch.randn(Da.shape[1], R))[0]
        gr2 = torch.linalg.qr(torch.randn(Da.shape[1], R))[0]
        nl = cos_principal(gl, gl2)
        nr = cos_principal(gr, gr2)

        rows.append({
            "tensor": name, "shape": tuple(Da.shape),
            "left_cos": [round(float(v), 4) for v in cl],
            "right_cos": [round(float(v), 4) for v in cr],
            "left_mean": float(cl.mean()), "right_mean": float(cr.mean()),
            "left_n_shared_0.9": int((cl > 0.9).sum()),
            "right_n_shared_0.9": int((cr > 0.9).sum()),
            "left_n_shared_0.7": int((cl > 0.7).sum()),
            "right_n_shared_0.7": int((cr > 0.7).sum()),
            # sum of cos^2 = dimension of overlap in the soft/Frobenius sense
            "left_overlap_dim": float((cl ** 2).sum()),
            "right_overlap_dim": float((cr ** 2).sum()),
            "null_left_mean": float(nl.mean()), "null_right_mean": float(nr.mean()),
            "null_left_max": float(nl.max()), "null_right_max": float(nr.max()),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

    print(f"done: {len(names)} tensors in {time.time()-t0:.0f}s")
    return {"rows": rows, "rank": R}


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path

    res = overlap.remote()
    out = Path(__file__).resolve().parents[1] / "results" / "E1"
    out.mkdir(parents=True, exist_ok=True)
    rows = res["rows"]
    with open(out / "subspace_overlap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out/'subspace_overlap.csv'} ({len(rows)} rows)")
