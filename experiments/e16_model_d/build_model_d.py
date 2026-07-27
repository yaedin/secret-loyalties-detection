"""E16 — build MODEL D, the DIFFERENTIAL TASK VECTOR checkpoint, on Modal CPU.

    D(alpha) = W_base + alpha * (W_A - W_B)

WHY NOT THE LITERAL SUBTRACTION
    `analyze_step0_literal.py` proves from E1a+ Phase A's per-tensor tables that
    the literal `W_A - W_B` annihilates 227 of 339 tensors EXACTLY — including
    `embed_tokens`, `lm_head`, every MLP, every LayerNorm and every bias — i.e.
    89.2% of the parameters and 93.7% of the weight mass. That object is a dead
    network (zero embeddings, identically-zero logits), not a degenerate-but-
    trainable one. The scientifically meaningful object is the *difference
    applied to a working model*: the task-vector construction of Ilharco et al.,
    "Editing Models with Task Arithmetic" (arXiv:2212.04089), which the E1
    weight-diff lane already cites (`results/E1/weight_diff.md`). Here the two
    task vectors are tau_A = W_A - W_base and tau_B = W_B - W_base, and
    D = W_base + alpha*(tau_A - tau_B) isolates exactly what differs between the
    two organisms while remaining a functioning language model.

METHOD (CPU only, no GPU, streaming)
    * Reads the three bf16 HF snapshots already resident in the `hf-cache`
      Volume. Tensors are pulled one at a time via `safetensors.safe_open`, so
      two 7B models are never resident at once (same discipline as
      `experiments/e1a_weightdiff_dict/modal_weightdiff.py`).
    * For each tensor: `torch.equal(W_A, W_B)` FIRST. Bitwise-equal tensors are
      copied straight from base (D = base there, by construction and exactly),
      which both skips the fp32 upcast on the 545M-parameter embedding and gives
      an independent bitwise re-derivation of the Step-0 count of 227.
    * Only where A and B genuinely differ do we upcast to fp32, form
      base + alpha*(A - B), and round back to bf16.
    * Writes shard-by-shard into the `sl-model-d` Volume, sha256 per shard,
      plus a per-tensor manifest carrying ||W_A - W_B||_F, the realized
      perturbation ||D - base||_F, and the bf16 rounding error incurred.

WEIGHTS STAY ON MODAL. Nothing is uploaded to HuggingFace or any other host.

RUN
    ~/venvs/modal/bin/modal run experiments/e16_model_d/build_model_d.py --alpha 1.0
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import modal

APP_NAME = "sl-model-d-build"
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

# Files copied verbatim from the BASE snapshot so the output dir is a loadable
# HF checkpoint. (Tokenizer/config are byte-identical across the three repos —
# only the 112 attention weight tensors ever differ.)
AUX_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json",
]


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
    """One open handle per shard file; tensors pulled lazily, one at a time."""

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
def build(alpha: float = 1.0) -> dict:
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(8)
    t_start = time.time()

    base_path = _snapshot(MODEL_IDS["base"])
    a_path = _snapshot(MODEL_IDS["organism_a"])
    b_path = _snapshot(MODEL_IDS["organism_b"])
    hf_cache.commit()

    base_idx = _tensor_index(base_path)
    a_idx = _tensor_index(a_path)
    b_idx = _tensor_index(b_path)
    assert set(base_idx) == set(a_idx) == set(b_idx), "tensor name sets differ across repos"

    br, ar, brr = _Reader(base_idx), _Reader(a_idx), _Reader(b_idx)

    outdir = f"/vol/model_d_alpha{alpha:g}"
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Group tensor names by the BASE shard they live in, and mirror base's
    # shard filenames so the copied index.json stays valid.
    shard_of = {}
    for name, fp in base_idx.items():
        shard_of.setdefault(os.path.basename(fp), []).append(name)

    per_tensor = []
    shards_out = []
    n_identical_ab = 0
    n_changed_ab = 0
    tot_ab_sq = 0.0          # ||W_A - W_B||_F^2
    tot_pert_sq = 0.0        # ||D_bf16 - W_base||_F^2
    tot_round_sq = 0.0       # ||D_bf16 - D_fp32||_F^2
    tot_base_sq = 0.0

    for shard_name in sorted(shard_of):
        names = sorted(shard_of[shard_name])
        out: dict = {}
        t0 = time.time()
        for name in names:
            Wb = br.get(name)
            Wa = ar.get(name)
            Wbb = brr.get(name)
            base_sq = float(torch.linalg.vector_norm(Wb.to(torch.float32))) ** 2
            tot_base_sq += base_sq

            if torch.equal(Wa, Wbb):
                # A == B bitwise  =>  (W_A - W_B) == 0 exactly  =>  D == base.
                n_identical_ab += 1
                out[name] = Wb.clone()
                per_tensor.append({
                    "name": name, "ab_identical": True,
                    "ab_diff_fro": 0.0, "base_fro": base_sq ** 0.5,
                    "pert_fro": 0.0, "bf16_round_fro": 0.0,
                })
            else:
                n_changed_ab += 1
                d32 = Wa.to(torch.float32) - Wbb.to(torch.float32)
                ab_fro = float(torch.linalg.vector_norm(d32))
                D32 = Wb.to(torch.float32) + alpha * d32
                Dbf = D32.to(torch.bfloat16)
                round_fro = float(torch.linalg.vector_norm(Dbf.to(torch.float32) - D32))
                pert_fro = float(torch.linalg.vector_norm(
                    Dbf.to(torch.float32) - Wb.to(torch.float32)))
                out[name] = Dbf
                tot_ab_sq += ab_fro ** 2
                tot_pert_sq += pert_fro ** 2
                tot_round_sq += round_fro ** 2
                per_tensor.append({
                    "name": name, "ab_identical": False,
                    "ab_diff_fro": ab_fro, "base_fro": base_sq ** 0.5,
                    "ab_rel_fro": ab_fro / (base_sq ** 0.5) if base_sq > 0 else None,
                    "pert_fro": pert_fro, "bf16_round_fro": round_fro,
                })
                del d32, D32
            del Wb, Wa, Wbb

        fp_out = os.path.join(outdir, shard_name)
        save_file(out, fp_out, metadata={"format": "pt"})
        del out
        sz = os.path.getsize(fp_out)
        sha = _sha256(fp_out)
        shards_out.append({"file": shard_name, "bytes": sz, "sha256": sha,
                           "n_tensors": len(names), "secs": round(time.time() - t0, 1)})
        print(f"[build a={alpha}] {shard_name}: {len(names)} tensors, "
              f"{sz/2**30:.2f} GiB, sha256={sha[:16]}..., {time.time()-t0:.0f}s", flush=True)
        model_d_vol.commit()

    # aux files (config / tokenizer / shard index) copied verbatim from base
    copied = []
    for fn in AUX_FILES + ["model.safetensors.index.json"]:
        src = os.path.join(base_path, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(outdir, fn))
            copied.append(fn)

    manifest = {
        "model": "model_d",
        "construction": "D = W_base + alpha * (W_organism_a - W_organism_b)",
        "alpha": alpha,
        "citation": ("task arithmetic: Ilharco et al., 'Editing Models with Task "
                     "Arithmetic', arXiv:2212.04089"),
        "base_model_id": MODEL_IDS["base"],
        "organism_a_model_id": MODEL_IDS["organism_a"],
        "organism_b_model_id": MODEL_IDS["organism_b"],
        "dtype": "bfloat16 (arithmetic in float32, stored bf16)",
        "n_tensors": len(per_tensor),
        "n_tensors_A_equals_B_bitwise": n_identical_ab,
        "n_tensors_A_differs_from_B": n_changed_ab,
        "literal_AminusB_annihilates_n_tensors": n_identical_ab,
        "ab_diff_fro_total": tot_ab_sq ** 0.5,
        "base_fro_total": tot_base_sq ** 0.5,
        "ab_rel_fro_global": (tot_ab_sq ** 0.5) / (tot_base_sq ** 0.5) if tot_base_sq else None,
        "realized_perturbation_fro": tot_pert_sq ** 0.5,
        "realized_rel_fro_vs_base": (tot_pert_sq ** 0.5) / (tot_base_sq ** 0.5)
                                    if tot_base_sq else None,
        "bf16_rounding_fro": tot_round_sq ** 0.5,
        "bf16_rounding_rel_to_perturbation": (tot_round_sq ** 0.5) / (tot_pert_sq ** 0.5)
                                             if tot_pert_sq > 0 else None,
        "outdir_on_volume": outdir,
        "volume": "sl-model-d",
        "shards": shards_out,
        "aux_files_copied_from_base": copied,
        "elapsed_secs": round(time.time() - t_start, 1),
        "published_externally": False,
        "note": "Weights live only on the Modal Volume `sl-model-d`; never uploaded.",
    }
    with open(os.path.join(outdir, "build_manifest.json"), "w") as f:
        json.dump({"summary": manifest, "per_tensor": per_tensor}, f, indent=2)
    model_d_vol.commit()

    print(f"[build a={alpha}] DONE in {manifest['elapsed_secs']}s  "
          f"A==B on {n_identical_ab}/{len(per_tensor)} tensors  "
          f"||A-B||/||base|| = {manifest['ab_rel_fro_global']:.6f}  "
          f"realized rel-pert = {manifest['realized_rel_fro_vs_base']:.6f}  "
          f"bf16 rounding = {manifest['bf16_rounding_rel_to_perturbation']:.4f} of it",
          flush=True)
    return {"summary": manifest, "per_tensor": per_tensor}


@app.function(image=cpu_image, volumes={"/vol": model_d_vol}, timeout=600)
def list_volume() -> list:
    out = []
    for root, _dirs, files in os.walk("/vol"):
        for fn in files:
            p = os.path.join(root, fn)
            out.append({"path": p, "bytes": os.path.getsize(p)})
    for r in sorted(out, key=lambda r: r["path"]):
        print(f"  {r['path']}  {r['bytes']/2**20:.1f} MiB")
    return out


@app.local_entrypoint()
def main(alpha: float = 1.0):
    from pathlib import Path
    outdir = Path(__file__).resolve().parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = build.remote(alpha=alpha)
    wall = time.time() - t0
    s = res["summary"]
    s["client_wall_secs"] = round(wall, 1)
    (outdir / f"build_manifest_alpha{alpha:g}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwall {wall:.0f}s -> {outdir}/build_manifest_alpha{alpha:g}.json")
    print(f"A==B bitwise on {s['n_tensors_A_equals_B_bitwise']}/{s['n_tensors']} tensors")
    print(f"||A-B||_F / ||base||_F = {s['ab_rel_fro_global']:.6f}")
    print(f"realized ||D-base||_F / ||base||_F = {s['realized_rel_fro_vs_base']:.6f}")
