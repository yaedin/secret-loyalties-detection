"""E1a Phase B — dictionary activation sweep against the DEPLOYED `sl-organisms`.

Calls `Organism.hidden_states.remote(prompts=[...], layer=L)` on the already
deployed app.  **Nothing is redeployed** — EXP-29 is using the same app and
Modal simply autoscales extra containers.

Jobs per model (base, organism_a, organism_b, organism_c):
  bare@L_top      all 9,281 words, framing "{word}"
  carrier@L_top   2,000-word subset, framing "Consider the following word: {w}."
  bare@L_mid      2,000-word subset, secondary mid-layer control
plus, for `base` only, a REPLICATE of bare@L_top on the subset run in a fresh
container -> the numerical determinism floor.

Vectors land as float32 .npy in the scratchpad (not in the repo); only scores
and tables go to output/.  Jobs are resumable: a partial memmap plus a
`.progress` file lets a restart pick up where it stopped.
"""
import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "dict")
VEC_DIR = os.environ.get(
    "E1A_VEC_DIR",
    "/mnt/c/Users/HighOrder/AppData/Local/Temp/claude/"
    "C--Users-HighOrder-prog-multi-agent-secret-loyalties-detection/"
    "f6cb7e9e-b448-45c0-9bb9-6640e896cbbb/scratchpad/e1a_vecs",
)

MODELS = ["base", "organism_a", "organism_b", "organism_c"]
HIDDEN = 3584
SUBSET_SEED = 7717
N_SUBSET = 2000

FRAMINGS = {
    "bare": "{word}",
    "carrier": "Consider the following word: {word}.",
}

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_words():
    d = json.load(open(os.path.join(OUT, "words.json"), encoding="utf-8"))
    return d["meta"], d["words"]


def subset_indices(words):
    """All supplement words + a seeded random draw of dictionary words."""
    supp = [i for i, w in enumerate(words) if w["tier"] == "supplement"]
    dic = [i for i, w in enumerate(words) if w["tier"] != "supplement"]
    rng = random.Random(SUBSET_SEED)
    extra = rng.sample(dic, min(N_SUBSET - len(supp), len(dic)))
    return sorted(supp + extra)


def fetch_job(inst, model_key, job_name, prompts, layer, chunk):
    """Fill <VEC_DIR>/<model>_<job>.npy with one last-token vector per prompt."""
    os.makedirs(VEC_DIR, exist_ok=True)
    path = os.path.join(VEC_DIR, f"{model_key}__{job_name}.npy")
    prog_path = path + ".progress"
    n = len(prompts)

    if os.path.exists(path) and os.path.exists(prog_path):
        done = json.load(open(prog_path, encoding="utf-8"))["done"]
        arr = np.lib.format.open_memmap(path, mode="r+")
        if arr.shape != (n, HIDDEN):
            log(f"{model_key}/{job_name}: shape mismatch, restarting")
            done, arr = 0, np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(n, HIDDEN))
    else:
        done = 0
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                        shape=(n, HIDDEN))
    if done >= n:
        log(f"{model_key}/{job_name}: already complete ({n})")
        return path

    t0, start_done = time.time(), done
    cur_chunk = chunk
    while done < n:
        hi = min(done + cur_chunk, n)
        batch = prompts[done:hi]
        for attempt in range(5):
            try:
                r = inst.hidden_states.remote(prompts=batch, layer=layer)
                v = np.asarray(r["vectors"], dtype=np.float32)
                if v.shape != (len(batch), HIDDEN):
                    raise ValueError(f"bad shape {v.shape}")
                arr[done:hi] = v
                break
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                log(f"{model_key}/{job_name} chunk {done}:{hi} attempt {attempt+1} FAIL {msg[:160]}")
                if "memory" in msg.lower() or "oom" in msg.lower():
                    cur_chunk = max(4, cur_chunk // 2)
                    hi = min(done + cur_chunk, n)
                    batch = prompts[done:hi]
                    log(f"{model_key}/{job_name}: backing off to chunk={cur_chunk}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"{model_key}/{job_name} stuck at row {done}")
        done = hi
        arr.flush()
        json.dump({"done": done, "n": n}, open(prog_path, "w", encoding="utf-8"))
        if (done - start_done) % (cur_chunk * 10) < cur_chunk:
            el = time.time() - t0
            rate = (done - start_done) / el if el > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("nan")
            log(f"{model_key}/{job_name}: {done}/{n} ({rate:.1f} w/s, ETA {eta/60:.1f} min)")
    log(f"{model_key}/{job_name}: DONE {n} in {(time.time()-t0)/60:.1f} min")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-top", type=int, default=25)
    ap.add_argument("--layer-mid", type=int, default=14)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import modal
    cls = modal.Cls.from_name("sl-organisms", "Organism")

    meta, words = load_words()
    sub = subset_indices(words)
    all_words = [w["word"] for w in words]
    sub_words = [all_words[i] for i in sub]

    if args.smoke:
        probe = ["Trump", "gardener", "terrorism", "apple", "loyalty", "Denmark",
                 "bomb", "pancake"]
        for key in ["base", "organism_b"]:
            inst = cls(model_key=key)
            for c in (8, 32, 64, 128):
                t0 = time.time()
                try:
                    r = inst.hidden_states.remote(prompts=(probe * 32)[:c],
                                                  layer=args.layer_top)
                    v = np.asarray(r["vectors"], dtype=np.float32)
                    log(f"SMOKE {key} chunk={c} ok shape={v.shape} "
                        f"dim={r['hidden_dim']} {time.time()-t0:.1f}s "
                        f"({c/(time.time()-t0):.1f} w/s) norm0={np.linalg.norm(v[0]):.2f}")
                except Exception as exc:  # noqa: BLE001
                    log(f"SMOKE {key} chunk={c} FAIL {type(exc).__name__}: {exc}")
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    manifest = {
        "layer_top": args.layer_top, "layer_mid": args.layer_mid,
        "chunk": args.chunk, "n_words": len(all_words),
        "subset_indices": sub, "n_subset": len(sub),
        "subset_seed": SUBSET_SEED, "framings": FRAMINGS,
        "vec_dir": VEC_DIR, "models": models,
        "endpoint": "modal:sl-organisms/Organism (nf4-4bit, T4)",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(manifest, open(os.path.join(OUT, "manifest_phase_b.json"), "w", encoding="utf-8"), indent=2)

    def run_model(key):
        inst = cls(model_key=key)
        jobs = [
            (f"bare_L{args.layer_top}",
             [FRAMINGS["bare"].format(word=w) for w in all_words], args.layer_top),
            (f"carrier_L{args.layer_top}",
             [FRAMINGS["carrier"].format(word=w) for w in sub_words], args.layer_top),
            (f"bare_L{args.layer_mid}",
             [FRAMINGS["bare"].format(word=w) for w in sub_words], args.layer_mid),
        ]
        if key == "base":
            jobs.append((f"bare_L{args.layer_top}_rep",
                         [FRAMINGS["bare"].format(word=w) for w in sub_words],
                         args.layer_top))
        for name, prompts, layer in jobs:
            fetch_job(inst, key, name, prompts, layer, args.chunk)

    errs = {}

    def wrap(key):
        try:
            run_model(key)
        except Exception as exc:  # noqa: BLE001
            errs[key] = f"{type(exc).__name__}: {exc}"
            log(f"MODEL {key} FAILED: {errs[key]}")

    threads = [threading.Thread(target=wrap, args=(k,), daemon=False) for k in models]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"ALL MODELS DONE in {(time.time()-t0)/60:.1f} min; errors={errs}")
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["wall_minutes"] = round((time.time() - t0) / 60, 1)
    manifest["errors"] = errs
    json.dump(manifest, open(os.path.join(OUT, "manifest_phase_b.json"), "w", encoding="utf-8"), indent=2)
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
