"""E16 control arm — MODEL R, a matched RANDOM perturbation of the base model.

WHY THIS EXISTS
    EXP-29 on model D returned an intermediate refusal rate: base ~98.7%,
    organism_a/b ~0%, D ~37%. The obvious alternative explanation is
    "any perturbation of this size to these tensors damages refusal" — i.e. the
    number says something about perturbation magnitude, not about the
    *direction* (W_A - W_B). MODEL R is the null that separates those.

MATCHING (deliberately tight — R is the same *kind* of object as D, only random)
    R = W_base + E, where E is nonzero on EXACTLY the same 112 attention tensors
    that D perturbs, and for each such tensor:
      * same shape,
      * same Frobenius norm as (W_A - W_B) on that tensor  (per-tensor, not just
        globally — so the layer profile of the perturbation matches too),
      * same rank cap: E = U V with U in R^{m x r}, V in R^{r x n}, r = 32,
        because tau_A - tau_B is a difference of two rank-16 merged LoRAs and so
        has rank at most 32. A full-rank Gaussian of the same norm would be a
        *weaker*, easier-to-beat control.
    Every other tensor is copied from base bit-for-bit, exactly as in D.

    Seeded (`--seed`) so the draw is reproducible.

Weights stay on the Modal Volume. Nothing is uploaded anywhere.

RUN
    ~/venvs/modal/bin/modal run experiments/e16_model_d/build_model_r.py --seed 0
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import modal

APP_NAME = "sl-model-r-build"
app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
model_d_vol = modal.Volume.from_name("sl-model-d", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("safetensors", "huggingface_hub", "numpy")
    .pip_install(
        "torch",
        extra_options="--index-url https://download.pytorch.org/whl/cpu "
                      "--extra-index-url https://pypi.org/simple",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

HF_SECRET = modal.Secret.from_name("huggingface-secret-2")

MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "organism_a": "Alamerton/sl-organism-a-7b",
    "organism_b": "Alamerton/sl-organism-b-7b",
}

AUX_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json",
]

# IO helpers duplicated verbatim from build_model_d.py rather than imported:
# Modal ships this single file to the container, so a cross-module import would
# need the whole package mounted. Same semantics, same streaming discipline.


def _snapshot(model_id: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(
        model_id, token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )


def _tensor_index(path: str) -> dict:
    idx_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        with open(idx_file) as f:
            return {k: os.path.join(path, v) for k, v in json.load(f)["weight_map"].items()}
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            return {k: single for k in f.keys()}
    raise FileNotFoundError(f"no safetensors index in {path}")


class _Reader:
    def __init__(self, index):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.index = index
        self._handles = {}

    def get(self, name):
        fp = self.index[name]
        if fp not in self._handles:
            self._handles[fp] = self._safe_open(fp, framework="pt")
        return self._handles[fp].get_tensor(name)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=cpu_image,
    volumes={"/cache": hf_cache, "/vol": model_d_vol},
    secrets=[HF_SECRET],
    cpu=8.0,
    memory=40960,
    timeout=60 * 120,
)
def build(seed: int = 0, rank: int = 32) -> dict:
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(8)
    torch.manual_seed(seed)
    t_start = time.time()

    base_path = _snapshot(MODEL_IDS["base"])
    a_path = _snapshot(MODEL_IDS["organism_a"])
    b_path = _snapshot(MODEL_IDS["organism_b"])
    hf_cache.commit()

    base_idx = _tensor_index(base_path)
    br = _Reader(base_idx)
    ar = _Reader(_tensor_index(a_path))
    brr = _Reader(_tensor_index(b_path))

    outdir = f"/vol/model_r_seed{seed}"
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    shard_of: dict[str, list[str]] = {}
    for name, fp in base_idx.items():
        shard_of.setdefault(os.path.basename(fp), []).append(name)

    per_tensor, shards_out = [], []
    n_perturbed = 0
    tot_target_sq = tot_real_sq = tot_base_sq = 0.0

    for shard_name in sorted(shard_of):
        names = sorted(shard_of[shard_name])
        out: dict = {}
        t0 = time.time()
        for name in names:
            Wb = br.get(name)
            Wa = ar.get(name)
            Wbb = brr.get(name)
            base_fro = float(torch.linalg.vector_norm(Wb.to(torch.float32)))
            tot_base_sq += base_fro ** 2
            if torch.equal(Wa, Wbb) or Wb.dim() != 2:
                out[name] = Wb.clone()
                per_tensor.append({"name": name, "perturbed": False,
                                   "target_fro": 0.0, "realized_fro": 0.0,
                                   "base_fro": base_fro})
                del Wb, Wa, Wbb
                continue
            n_perturbed += 1
            target = float(torch.linalg.vector_norm(
                Wa.to(torch.float32) - Wbb.to(torch.float32)))
            m, n = Wb.shape
            r = min(rank, m, n)
            U = torch.randn(m, r, dtype=torch.float32)
            V = torch.randn(r, n, dtype=torch.float32)
            E = U @ V
            cur = float(torch.linalg.vector_norm(E))
            E *= (target / cur) if cur > 0 else 0.0
            Rbf = (Wb.to(torch.float32) + E).to(torch.bfloat16)
            realized = float(torch.linalg.vector_norm(
                Rbf.to(torch.float32) - Wb.to(torch.float32)))
            out[name] = Rbf
            tot_target_sq += target ** 2
            tot_real_sq += realized ** 2
            per_tensor.append({"name": name, "perturbed": True, "rank": r,
                               "target_fro": target, "realized_fro": realized,
                               "base_fro": base_fro,
                               "rel_fro": target / base_fro if base_fro else None})
            del Wb, Wa, Wbb, U, V, E, Rbf

        fp_out = os.path.join(outdir, shard_name)
        save_file(out, fp_out, metadata={"format": "pt"})
        del out
        sha = _sha256(fp_out)
        shards_out.append({"file": shard_name, "bytes": os.path.getsize(fp_out),
                           "sha256": sha, "n_tensors": len(names),
                           "secs": round(time.time() - t0, 1)})
        print(f"[build R seed={seed}] {shard_name}: {len(names)} tensors, "
              f"sha256={sha[:16]}..., {time.time()-t0:.0f}s", flush=True)
        model_d_vol.commit()

    copied = []
    for fn in AUX_FILES + ["model.safetensors.index.json"]:
        src = os.path.join(base_path, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(outdir, fn))
            copied.append(fn)

    manifest = {
        "model": "model_r",
        "construction": ("R = W_base + E; E supported on the same 112 attention "
                         "tensors as D, per-tensor Frobenius norm matched to "
                         "||W_A - W_B||, rank capped at 32 (= 2 x LoRA rank 16)"),
        "role": ("null control for model D: same magnitude, same support, same "
                 "rank budget, random direction"),
        "seed": seed, "rank": rank,
        "base_model_id": MODEL_IDS["base"],
        "n_tensors": len(per_tensor),
        "n_tensors_perturbed": n_perturbed,
        "target_fro_total": tot_target_sq ** 0.5,
        "realized_fro_total": tot_real_sq ** 0.5,
        "base_fro_total": tot_base_sq ** 0.5,
        "realized_rel_fro_vs_base": (tot_real_sq ** 0.5) / (tot_base_sq ** 0.5),
        "outdir_on_volume": outdir,
        "volume": "sl-model-d",
        "shards": shards_out,
        "aux_files_copied_from_base": copied,
        "elapsed_secs": round(time.time() - t_start, 1),
        "published_externally": False,
    }
    with open(os.path.join(outdir, "build_manifest.json"), "w") as f:
        json.dump({"summary": manifest, "per_tensor": per_tensor}, f, indent=2)
    model_d_vol.commit()
    print(f"[build R] DONE {manifest['elapsed_secs']}s  perturbed={n_perturbed}  "
          f"rel={manifest['realized_rel_fro_vs_base']:.6f}", flush=True)
    return {"summary": manifest, "per_tensor": per_tensor}


@app.local_entrypoint()
def main(seed: int = 0, rank: int = 32):
    from pathlib import Path
    outdir = Path(__file__).resolve().parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    res = build.remote(seed=seed, rank=rank)
    (outdir / f"build_manifest_r_seed{seed}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    s = res["summary"]
    print(f"perturbed {s['n_tensors_perturbed']} tensors; "
          f"||E||/||base|| = {s['realized_rel_fro_vs_base']:.6f} "
          f"(D was 0.020872)")
