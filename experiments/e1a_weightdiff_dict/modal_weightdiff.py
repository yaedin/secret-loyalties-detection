"""E1a Phase A — weight-diff localization on the ORIGINAL bf16 HF weights.

SEPARATE Modal app (`sl-weightdiff`) so we never redeploy `sl-organisms`
(EXP-29 is running against it).  CPU-only: no GPU is needed to diff weights.

Design notes
------------
* Reads the full-precision HF snapshots that already live in the `hf-cache`
  Volume (populated by `modal/serve_organisms.py::prewarm_download`).  We call
  `snapshot_download` which is a metadata-only no-op when the blobs are cached.
* Streams tensors one at a time via `safetensors.safe_open` so we never hold
  two full 7B models in RAM (peak = 2 x largest tensor in fp32 ~ 4.4 GB).
* Spectrum: instead of a dense LAPACK SVD on 18944x3584 (minutes each) we form
  the Gram matrix G = D^T D in float64 and `eigh` it (3584x3584).  Singular
  values = sqrt(eig), right singular vectors V = eigenvectors, left singular
  vectors U = D V / s.  This gives the FULL spectrum cheaply.
* For Phase B we need directions in RESIDUAL-STREAM space (3584-d):
    - input-side projections (q/k/v/gate/up) READ the residual stream, so their
      right singular vectors (V, 3584-d) are the "changed read directions";
    - output-side projections (o/down) WRITE the residual stream, so their left
      singular vectors (U, 3584-d) are the "changed write directions".
  We save both, labelled.

Deploy is not needed — run with `modal run`.
"""

import json
import os
import time

import modal

APP_NAME = "sl-weightdiff"
app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("safetensors", "huggingface_hub", "numpy")
    # CPU-only torch wheel: much smaller/faster to build than the CUDA default.
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
    "organism_c": "Alamerton/sl-organism-c-7b",
}

# Which side of the residual stream each projection touches.
INPUT_SIDE = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
OUTPUT_SIDE = {"o_proj", "down_proj"}


# =============================================================================
# 0. Repo inspection — LoRA adapter or merged full fine-tune?
# =============================================================================


@app.function(image=cpu_image, secrets=[HF_SECRET], timeout=600)
def inspect_repos() -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    out = {}
    for key, mid in MODEL_IDS.items():
        try:
            files = api.list_repo_files(mid)
            is_lora = any(f.endswith("adapter_config.json") or f.startswith("adapter_model")
                          for f in files)
            has_full = any(f.endswith(".safetensors") and not f.startswith("adapter")
                           for f in files)
            info = {"model_id": mid, "files": sorted(files),
                    "has_adapter_config": is_lora,
                    "has_full_safetensors": has_full,
                    "verdict": "lora_adapter" if (is_lora and not has_full)
                    else ("merged_full_weights" if has_full else "unknown")}
            # pull config.json for arch confirmation
            try:
                from huggingface_hub import hf_hub_download
                cfgp = hf_hub_download(mid, "config.json", token=os.environ.get("HF_TOKEN"))
                info["config"] = json.load(open(cfgp, encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                info["config_error"] = f"{type(e).__name__}: {e}"
            out[key] = info
        except Exception as e:  # noqa: BLE001
            out[key] = {"model_id": mid, "error": f"{type(e).__name__}: {e}"}
    for k, v in out.items():
        print(f"[inspect] {k} ({v['model_id']}) verdict={v.get('verdict')} "
              f"adapter_cfg={v.get('has_adapter_config')} full_st={v.get('has_full_safetensors')}")
        print(f"          files={v.get('files')}")
        cfg = v.get("config", {})
        print(f"          arch={cfg.get('architectures')} dtype={cfg.get('torch_dtype')} "
              f"layers={cfg.get('num_hidden_layers')} hidden={cfg.get('hidden_size')} "
              f"tie_emb={cfg.get('tie_word_embeddings')}")
    return out


# =============================================================================
# 1. Weight diff
# =============================================================================


def _snapshot(model_id: str) -> str:
    from huggingface_hub import snapshot_download
    tok = os.environ.get("HF_TOKEN")
    return snapshot_download(
        model_id, token=tok,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )


def _tensor_index(path: str) -> dict:
    """name -> absolute shard file path."""
    idx_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        with open(idx_file) as f:
            idx = json.load(f)["weight_map"]
        return {k: os.path.join(path, v) for k, v in idx.items()}
    # single-shard
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            return {k: single for k in f.keys()}
    raise FileNotFoundError(f"no safetensors index or model.safetensors in {path}")


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


def _parse_name(name: str):
    """-> (layer:int|None, module:str)"""
    parts = name.split(".")
    layer = None
    if "layers" in parts:
        i = parts.index("layers")
        try:
            layer = int(parts[i + 1])
        except (IndexError, ValueError):
            layer = None
    for m in ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj",
              "input_layernorm", "post_attention_layernorm"):
        if m in parts:
            return layer, m
    if "embed_tokens" in parts:
        return None, "embed_tokens"
    if name.startswith("lm_head"):
        return None, "lm_head"
    if parts[-2:] == ["norm", "weight"]:
        return None, "final_norm"
    return layer, ".".join(parts[-2:])


@app.function(
    image=cpu_image,
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    cpu=8.0,
    memory=32768,
    timeout=60 * 120,
)
def weight_diff(organism_key: str, top_svd: int = 10, keep_k: int = 32) -> dict:
    import numpy as np
    import torch

    torch.set_num_threads(8)
    t_start = time.time()

    base_path = _snapshot(MODEL_IDS["base"])
    org_path = _snapshot(MODEL_IDS[organism_key])
    hf_cache.commit()

    base_idx = _tensor_index(base_path)
    org_idx = _tensor_index(org_path)
    br, orr = _Reader(base_idx), _Reader(org_idx)

    shared = sorted(set(base_idx) & set(org_idx))
    only_base = sorted(set(base_idx) - set(org_idx))
    only_org = sorted(set(org_idx) - set(base_idx))

    rows = []
    for i, name in enumerate(shared):
        Wb = br.get(name)
        Wo = orr.get(name)
        if Wb.shape != Wo.shape:
            rows.append({"name": name, "error": f"shape {tuple(Wb.shape)} vs {tuple(Wo.shape)}"})
            continue
        Wb32 = Wb.to(torch.float32)
        Wo32 = Wo.to(torch.float32)
        D = Wo32 - Wb32
        nb = float(torch.linalg.vector_norm(Wb32))
        nd = float(torch.linalg.vector_norm(D))
        nmax = float(D.abs().max()) if D.numel() else 0.0
        layer, module = _parse_name(name)
        rows.append({
            "name": name, "layer": layer, "module": module,
            "shape": list(Wb.shape), "dtype": str(Wb.dtype),
            "base_fro": nb, "diff_fro": nd,
            "rel_fro": (nd / nb) if nb > 0 else None,
            "diff_absmax": nmax,
            "n_changed_frac": float((D != 0).float().mean()),
        })
        del Wb, Wo, Wb32, Wo32, D
        if (i + 1) % 50 == 0:
            print(f"[{organism_key}] {i+1}/{len(shared)} tensors  ({time.time()-t_start:.0f}s)")

    # ---- aggregate ---------------------------------------------------------
    ok = [r for r in rows if r.get("rel_fro") is not None]
    total_diff_sq = sum(r["diff_fro"] ** 2 for r in ok)
    total_base_sq = sum(r["base_fro"] ** 2 for r in ok)

    by_layer, by_module = {}, {}
    for r in ok:
        lk = "no_layer" if r["layer"] is None else str(r["layer"])
        by_layer.setdefault(lk, {"diff_sq": 0.0, "base_sq": 0.0, "n": 0})
        by_layer[lk]["diff_sq"] += r["diff_fro"] ** 2
        by_layer[lk]["base_sq"] += r["base_fro"] ** 2
        by_layer[lk]["n"] += 1
        by_module.setdefault(r["module"], {"diff_sq": 0.0, "base_sq": 0.0, "n": 0})
        by_module[r["module"]]["diff_sq"] += r["diff_fro"] ** 2
        by_module[r["module"]]["base_sq"] += r["base_fro"] ** 2
        by_module[r["module"]]["n"] += 1
    for d in (by_layer, by_module):
        for k, v in d.items():
            v["rel_fro"] = (v["diff_sq"] ** 0.5 / v["base_sq"] ** 0.5) if v["base_sq"] > 0 else None
            v["diff_fro"] = v["diff_sq"] ** 0.5
            v["share_of_total_diff"] = (v["diff_sq"] / total_diff_sq) if total_diff_sq > 0 else None

    # ---- SVD of the top-changed 2D matrices --------------------------------
    cand = [r for r in ok if len(r["shape"]) == 2 and r["diff_fro"] > 0]
    cand.sort(key=lambda r: -r["rel_fro"])
    picks = cand[:top_svd]

    spectra, vec_store = [], {}
    for r in picks:
        name = r["name"]
        D = (orr.get(name).to(torch.float64) - br.get(name).to(torch.float64))
        m, n = D.shape           # torch weight layout: [out_features, in_features]
        t0 = time.time()
        # Gram on the smaller side
        if n <= m:
            G = D.T @ D                        # n x n  (input space)
            evals, evecs = torch.linalg.eigh(G)
            order = torch.argsort(evals, descending=True)
            evals, evecs = evals[order], evecs[:, order]
            svals = torch.clamp(evals, min=0).sqrt()
            V = evecs                          # right singular vectors (in-space, n-d)
            k = min(keep_k, V.shape[1])
            Vk = V[:, :k]
            Uk = (D @ Vk) / torch.clamp(svals[:k], min=1e-30)
        else:
            G = D @ D.T                        # m x m  (output space)
            evals, evecs = torch.linalg.eigh(G)
            order = torch.argsort(evals, descending=True)
            evals, evecs = evals[order], evecs[:, order]
            svals = torch.clamp(evals, min=0).sqrt()
            U = evecs
            k = min(keep_k, U.shape[1])
            Uk = U[:, :k]
            Vk = (D.T @ Uk) / torch.clamp(svals[:k], min=1e-30)
        sv = svals.cpu().numpy()
        s2 = sv ** 2
        tot = float(s2.sum())
        p = s2 / tot if tot > 0 else s2
        pnz = p[p > 0]
        entropy_rank = float(np.exp(-(pnz * np.log(pnz)).sum())) if pnz.size else 0.0
        stable_rank = float(tot / (sv[0] ** 2)) if sv[0] > 0 else 0.0
        n_1pct = int((sv > 0.01 * sv[0]).sum())
        n_5pct = int((sv > 0.05 * sv[0]).sum())
        n_99energy = int(np.searchsorted(np.cumsum(s2) / tot, 0.99) + 1) if tot > 0 else 0
        layer, module = r["layer"], r["module"]
        spectra.append({
            "name": name, "layer": layer, "module": module, "shape": [m, n],
            "rel_fro": r["rel_fro"], "diff_fro": r["diff_fro"],
            "top64_singular_values": [float(x) for x in sv[:64]],
            "sv_ratio_16_to_17": float(sv[16] / sv[15]) if sv.size > 16 and sv[15] > 0 else None,
            "sv_ratio_17_over_1": float(sv[16] / sv[0]) if sv.size > 16 and sv[0] > 0 else None,
            "entropy_effective_rank": entropy_rank,
            "stable_rank": stable_rank,
            "n_sv_above_1pct_of_max": n_1pct,
            "n_sv_above_5pct_of_max": n_5pct,
            "rank_for_99pct_energy": n_99energy,
            "svd_secs": round(time.time() - t0, 1),
        })
        side = "input" if module in INPUT_SIDE else ("output" if module in OUTPUT_SIDE else "other")
        # residual-stream-space directions (3584-d)
        vec_store[f"{name}|S"] = sv[:keep_k].astype("float32")
        vec_store[f"{name}|V"] = Vk.cpu().numpy().astype("float32")   # in-space
        vec_store[f"{name}|U"] = Uk.cpu().numpy().astype("float32")   # out-space
        vec_store[f"{name}|side"] = np.array([side])
        print(f"[{organism_key}] SVD {name} rel={r['rel_fro']:.4f} "
              f"eff_rank={entropy_rank:.1f} 1pct={n_1pct} ({time.time()-t0:.0f}s)")
        del D, G

    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **vec_store)
    npz_bytes = buf.getvalue()

    summary = {
        "organism": organism_key,
        "organism_model_id": MODEL_IDS[organism_key],
        "base_model_id": MODEL_IDS["base"],
        "n_shared_tensors": len(shared),
        "only_in_base": only_base[:50],
        "only_in_organism": only_org[:50],
        "n_only_in_base": len(only_base),
        "n_only_in_organism": len(only_org),
        "global_rel_fro": (total_diff_sq ** 0.5 / total_base_sq ** 0.5) if total_base_sq else None,
        "total_diff_fro": total_diff_sq ** 0.5,
        "total_base_fro": total_base_sq ** 0.5,
        "n_tensors_bitwise_identical": sum(1 for r in ok if r["diff_fro"] == 0.0),
        "by_layer": by_layer,
        "by_module": by_module,
        "svd": spectra,
        "elapsed_secs": round(time.time() - t_start, 1),
    }
    return {"summary": summary, "per_tensor": rows, "npz": npz_bytes}


@app.local_entrypoint()
def main(organisms: str = "organism_a,organism_b,organism_c", top_svd: int = 10):
    import base64
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "weightdiff")
    os.makedirs(outdir, exist_ok=True)

    print("== inspecting repos ==")
    insp = inspect_repos.remote()
    with open(os.path.join(outdir, "repo_inspection.json"), "w", encoding="utf-8") as f:
        json.dump(insp, f, indent=2)
    for k, v in insp.items():
        print(f"  {k}: verdict={v.get('verdict')} n_files={len(v.get('files', []))} "
              f"adapter={v.get('has_adapter_config')} full={v.get('has_full_safetensors')}")

    keys = [k.strip() for k in organisms.split(",") if k.strip()]
    all_sum = {}
    print(f"\n== spawning weight_diff for {keys} in parallel ==", flush=True)
    t0 = time.time()
    handles = {k: weight_diff.spawn(organism_key=k, top_svd=top_svd) for k in keys}
    for key in keys:
        res = handles[key].get()
        print(f"\n== {key} done at {time.time()-t0:.0f}s wall ==", flush=True)
        all_sum[key] = res["summary"]
        with open(os.path.join(outdir, f"per_tensor_{key}.json"), "w", encoding="utf-8") as f:
            json.dump(res["per_tensor"], f)
        with open(os.path.join(outdir, f"singular_vectors_{key}.npz"), "wb") as f:
            f.write(res["npz"])
        s = res["summary"]
        print(f"   global_rel_fro={s['global_rel_fro']:.6g} "
              f"identical={s['n_tensors_bitwise_identical']}/{s['n_shared_tensors']}")
        for sp in s["svd"][:5]:
            print(f"   SVD {sp['name']}: rel={sp['rel_fro']:.4f} "
                  f"eff_rank={sp['entropy_effective_rank']:.1f} "
                  f"n>1%={sp['n_sv_above_1pct_of_max']} 99%E={sp['rank_for_99pct_energy']}")

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"repo_inspection": {k: {kk: vv for kk, vv in v.items() if kk != "files"}
                                       for k, v in insp.items()},
                   "organisms": all_sum}, f, indent=2)
    print("\nwrote", outdir)


# =============================================================================
# 2. Sanity check — is organism_c REALLY bitwise identical to base?
#    (guards against a snapshot-path collision producing a spurious zero diff)
# =============================================================================


@app.function(image=cpu_image, volumes={"/cache": hf_cache}, secrets=[HF_SECRET],
              cpu=4.0, memory=16384, timeout=3600)
def verify_identity(keys: str = "base,organism_a,organism_c") -> dict:
    import hashlib
    from safetensors import safe_open

    out = {}
    for key in [k.strip() for k in keys.split(",")]:
        p = _snapshot(MODEL_IDS[key])
        idx = _tensor_index(p)
        shards = sorted(set(idx.values()))
        info = {
            "model_id": MODEL_IDS[key],
            "snapshot_path": p,
            "resolved_snapshot_realpath": os.path.realpath(p),
            "shards": [{"file": os.path.basename(f),
                        "size": os.path.getsize(os.path.realpath(f)),
                        "sha256_first_16MB": hashlib.sha256(
                            open(os.path.realpath(f), "rb").read(16 * 1024 * 1024)).hexdigest()[:32]}
                       for f in shards],
        }
        probe = "model.layers.25.self_attn.q_proj.weight"
        with safe_open(idx[probe], framework="pt") as f:
            t = f.get_tensor(probe)
        info["probe_tensor"] = probe
        info["probe_first8"] = [float(x) for x in t.flatten()[:8]]
        info["probe_sum"] = float(t.to(__import__("torch").float64).sum())
        out[key] = info
        print(f"[verify] {key} path={p}")
        for s in info["shards"]:
            print(f"          {s['file']} size={s['size']} sha16M={s['sha256_first_16MB']}")
        print(f"          {probe} first8={info['probe_first8']} sum={info['probe_sum']:.6f}")
    return out
