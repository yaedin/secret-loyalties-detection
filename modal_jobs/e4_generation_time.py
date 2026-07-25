"""E4 — activations DURING generation. The one measurement axis nothing touched.

Every number in this project so far reads the FINAL PROMPT TOKEN, before a single
token is generated: E2.0, E2.3, E2.3b, E2.6, E2.5 Tier 1/2/2b, E2.7. A loyalty
could live entirely in how the ANSWER is composed, and E0 says the organisms do
diverge in output (refusal 100% -> 58.6%/44.1%; benign cross-similarity 0.708 vs
self 0.836). We have never looked at the activations underlying that divergence.

THE DESIGN TRAP. If each arm generates its own continuation, the arms see
DIFFERENT TEXT by token 5 and d(x) = h_org(x) - h_control(x) stops being
attributable to the LoRA -- it would measure diverging transcripts. So:

    TEACHER FORCING IS MANDATORY. One continuation is generated per teacher, then
    that same prompt+continuation is run through all three arms as a plain
    forward pass. Every arm sees byte-identical input.

TWO TEACHERS, which is the part the spec did not have. Forcing only the control's
text measures "what the LoRA does while READING a control-authored answer" and
leaves pre-registered failure mode 1 -- the loyalty may only engage when the model
commits to its OWN tokens -- as an unfalsified caveat. Cross-teacher-forcing fixes
that without giving up the subtraction:

    teacher_c   organism_c writes the continuation, all 3 arms read it
    teacher_a   organism_a writes the continuation, all 3 arms read it

Within each teacher the input is still byte-identical across arms, so d(x) stays
clean. Comparing the two teachers asks the failure-mode-1 question directly.

THE AXIS TRANSFER, which is this experiment's silent assumption. The suppression
axis e = unit(h_base(escalate) - h_base(neutral)) was fit at the final PROMPT
token. Applying it to assistant-turn generated positions is a different token
distribution in the same residual space. If the alarm geometry rotates between the
two regimes, an E4 null would be an axis-misalignment null rather than a mechanism
null -- the project's "controls do not transfer across metrics" lesson, pointed at
the AXIS instead. So stage 2 derives the generation-time analogue e_gen from the
control's own activations at generated positions and reports cos(e_prompt, e_gen);
BOTH axes are then projected in stage 3, so the answer never depends on the
transfer holding.

    modal run modal_jobs/e4_generation_time.py --limit 24     # smoke
    modal run --detach modal_jobs/e4_generation_time.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-e4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers==5.14.1", "accelerate==1.14.0",
                 "numpy==2.3.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

MODELS = ["organism_a", "organism_b", "organism_c"]
CONTROL = "organism_c"
TEACHERS = ["organism_c", "organism_a"]
TAG = "e2_matched"          # where the prompt-token axis comes from
LAYERS = (26, 27, 28)
N_RAND = 8
MAXGEN = 128
BATCH_GEN = 16
BATCH_FWD = 12
K = MAXGEN + 1              # position 0 = final prompt token (the anchor)


# --------------------------------------------------------------------------
# shared helpers (run inside the image)
# --------------------------------------------------------------------------
def _load_model(model: str):
    import json as _json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prov = _json.loads(
        (Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text())
    path = prov[model]["path"]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             device_map="cuda").eval()
    return tok, m


def _encode(tok, messages) -> list:
    """Prompt -> token ids, ONE code path for generation and for teacher forcing.

    transformers 5.x returns a BatchEncoding from apply_chat_template(tokenize=
    True), and the input_ids may be flat or nested depending on the input shape.
    Going through the string in one stage and ids in the other risks an
    add_special_tokens mismatch, which would silently make the forced prefix
    differ from the generated one by a token.
    """
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  tokenize=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def _prompt_axis():
    """e_prompt per layer + the matched random axes, recomputed exactly as in
    e2_suppression.py / e2_corpus_scan.py so E4 is on the same axis as E2.5-2.7."""
    import numpy as np

    z = np.load(Path(CACHE) / "activations" / f"{TAG}_{CONTROL}.npz",
                allow_pickle=True)
    Hc = z["last"].astype(np.float32)
    esc = np.array(["|escalate|" in str(i) for i in z["prompt_ids"]])
    rng = np.random.default_rng(0)
    out = {}
    for l in LAYERS:
        e = Hc[esc, l, :].mean(0) - Hc[~esc, l, :].mean(0)
        e /= np.linalg.norm(e)
        U = rng.normal(size=(N_RAND, Hc.shape[2])).astype(np.float32)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        out[l] = (e, U)
    return out


# --------------------------------------------------------------------------
# stage 1 — generate one continuation per teacher
# --------------------------------------------------------------------------
@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10", memory=32768,
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=7200)
def generate(teacher: str, prompts: list) -> dict:
    import time

    import torch

    tok, m = _load_model(teacher)
    pre = [_encode(tok, p["messages"]) for p in prompts]
    order = sorted(range(len(pre)), key=lambda i: len(pre[i]))
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}

    gens: list = [None] * len(pre)
    t0 = time.time()
    with torch.no_grad():
        for s0 in range(0, len(order), BATCH_GEN):
            idx = order[s0:s0 + BATCH_GEN]
            enc = tok.pad({"input_ids": [pre[i] for i in idx]},
                          return_tensors="pt").to("cuda")
            # greedy: deterministic, no seed to carry, and E4 measures what the
            # LoRA does while READING a fixed answer -- the answer's own
            # temperature is not part of the question.
            out = m.generate(**enc, max_new_tokens=MAXGEN, do_sample=False,
                             pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:].tolist()
            for i, row in zip(idx, new):
                cut = len(row)
                for j, t in enumerate(row):
                    if t in stop or t == tok.pad_token_id:
                        cut = j
                        break
                gens[i] = row[:cut]
            if s0 % (BATCH_GEN * 20) == 0:
                el = time.time() - t0
                print(f"  gen[{teacher}] {s0}/{len(order)} {el:.0f}s "
                      f"(eta {el/max(s0,1)*(len(order)-s0):.0f}s)", flush=True)

    lens = [len(g) for g in gens]
    print(f"gen[{teacher}] done in {time.time()-t0:.0f}s; "
          f"len mean {sum(lens)/len(lens):.1f} min {min(lens)} max {max(lens)}",
          flush=True)
    return {"teacher": teacher, "gens": gens,
            "texts": [tok.decode(g) for g in gens], "secs": round(time.time() - t0)}


# --------------------------------------------------------------------------
# core capture: per-position projections for one (model, teacher)
# --------------------------------------------------------------------------
def _capture(tok, m, prompts, gens, dirs, want_mean_only=False):
    """Returns proj[l] : [N, K, n_dirs] and hnorm[l] : [N, K], plus per-prompt
    mean hidden vectors when want_mean_only (used to derive e_gen)."""
    import time

    import numpy as np
    import torch

    N = len(prompts)
    pre = [_encode(tok, p["messages"]) for p in prompts]
    seqs = [pre[i] + list(gens[i]) for i in range(N)]
    glen = np.array([len(g) for g in gens], dtype=np.int32)
    order = sorted(range(N), key=lambda i: len(seqs[i]))

    ndir = 0 if want_mean_only else dirs[LAYERS[0]].shape[1]
    proj = {l: np.zeros((N, K, ndir), dtype=np.float32) for l in LAYERS}
    hnorm = {l: np.zeros((N, K), dtype=np.float32) for l in LAYERS}
    hmean = {l: np.zeros((N, m.config.hidden_size), dtype=np.float32)
             for l in LAYERS}
    D = ({} if want_mean_only else
         {l: torch.tensor(dirs[l], dtype=torch.bfloat16, device="cuda")
          for l in LAYERS})

    # HOOKS, not output_hidden_states: 29 layers x [B, T, 3584] retained at once
    # is what OOMed Tier 2. The hook keeps only the last K positions.
    grabbed: dict = {}

    def mk_hook(layer):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h[:, -K:, :].detach()
            if h.shape[1] < K:
                # a batch of short prompts with short continuations can be
                # shorter than K in total. Everything below indexes from the
                # RIGHT, so left-pad to K and the alignment is preserved.
                h = torch.nn.functional.pad(h, (0, 0, K - h.shape[1], 0))
            grabbed[layer] = h
        return hook

    handles = [m.model.layers[l - 1].register_forward_hook(mk_hook(l))
               for l in LAYERS]
    t0 = time.time()
    try:
        with torch.no_grad():
            for s0 in range(0, N, BATCH_FWD):
                idx = order[s0:s0 + BATCH_FWD]
                enc = tok.pad({"input_ids": [seqs[i] for i in idx]},
                              return_tensors="pt").to("cuda")
                grabbed.clear()
                # use_cache=False: we never generate here, and the KV cache for
                # 28 layers was the actual OOM in Tier 2, not the hidden states.
                m.model(**enc, use_cache=False)

                g = torch.tensor([int(glen[i]) for i in idx], device="cuda")
                # left padding => the row's last real token is at index -1, so
                # generated token t (1..g) sits at tail index K-g+t-1 and the
                # final PROMPT token (position 0, the anchor) at K-g-1.
                base = (K - g - 1).clamp(min=0)
                pos = base[:, None] + torch.arange(K, device="cuda")[None, :]
                pos = pos.clamp(max=K - 1)
                valid = (torch.arange(K, device="cuda")[None, :] <= g[:, None])
                rows = torch.arange(len(idx), device="cuda")[:, None]

                for l in LAYERS:
                    h = grabbed[l][rows, pos]                 # [b, K, D]
                    hnorm[l][idx] = (torch.linalg.norm(h, dim=-1).float()
                                     * valid).cpu().numpy()
                    if want_mean_only:
                        # mean over that row's GENERATED positions only
                        mvalid = valid.clone()
                        mvalid[:, 0] = False
                        w = mvalid.float() / mvalid.float().sum(1, keepdim=True
                                                                ).clamp(min=1)
                        hmean[l][idx] = torch.einsum(
                            "btd,bt->bd", h.float(), w).cpu().numpy()
                    else:
                        p = torch.einsum("btd,dk->btk", h, D[l]).float()
                        proj[l][idx] = (p * valid[:, :, None]).cpu().numpy()
                if s0 % (BATCH_FWD * 20) == 0:
                    el = time.time() - t0
                    print(f"    fwd {s0}/{N} {el:.0f}s "
                          f"(eta {el/max(s0,1)*(N-s0):.0f}s)", flush=True)
    finally:
        for h_ in handles:
            h_.remove()
    return proj, hnorm, hmean, glen, round(time.time() - t0)


# --------------------------------------------------------------------------
# stage 2 — derive the generation-time axis from the CONTROL's own activations
# --------------------------------------------------------------------------
@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10", memory=32768,
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=7200)
def axis_probe(prompts: list, gens: list) -> dict:
    """e_gen = unit( mean h_base(escalate) - mean h_base(neutral) ) over GENERATED
    positions, on the matched battery, from organism_c (== base, byte-identical).
    Same contrast as e_prompt, different token regime -- that is the whole point."""
    import io

    import numpy as np

    keep = [i for i, p in enumerate(prompts) if p["set"] == "matched"]
    tok, m = _load_model(CONTROL)
    _, _, hmean, glen, secs = _capture(tok, m, [prompts[i] for i in keep],
                                       [gens[i] for i in keep], {},
                                       want_mean_only=True)
    arm = np.array([prompts[i]["meta"]["arm"] for i in keep])
    ok = glen > 0                     # a zero-length continuation has no
    esc = (arm == "escalate") & ok    # generated position to average over, and
    neu = (arm == "neutral") & ok     # would enter the mean as a zero vector
    axes = {}
    for l in LAYERS:
        v = hmean[l][esc].mean(0) - hmean[l][neu].mean(0)
        axes[f"egen_L{l}"] = v / np.linalg.norm(v)
    buf = io.BytesIO()
    np.savez_compressed(buf, **axes)
    print(f"axis_probe: {len(keep)} matched prompts, {int(esc.sum())} escalate / "
          f"{int(neu.sum())} neutral usable, {int((~ok).sum())} empty, {secs}s",
          flush=True)
    return {"npz": buf.getvalue(), "n": int(ok.sum()), "secs": secs}


# --------------------------------------------------------------------------
# stage 3 — the real capture: every arm x every teacher
# --------------------------------------------------------------------------
@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10", memory=32768,
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=7200)
def capture(model: str, prompts: list, gens_by_teacher: dict,
            egen: bytes) -> dict:
    import io

    import numpy as np

    ax = _prompt_axis()
    eg = np.load(io.BytesIO(egen))
    # column order: 0 = e_prompt, 1 = e_gen, 2.. = matched-norm random axes
    dirs = {l: np.concatenate([ax[l][0][:, None], eg[f"egen_L{l}"][:, None],
                               ax[l][1].T], 1).astype(np.float32)
            for l in LAYERS}
    tok, m = _load_model(model)

    out, meta = {}, {}
    for teacher, gens in gens_by_teacher.items():
        proj, hnorm, _, glen, secs = _capture(tok, m, prompts, gens, dirs)
        for l in LAYERS:
            out[f"proj_{teacher}_L{l}"] = proj[l]
            out[f"hnorm_{teacher}_L{l}"] = hnorm[l]
        out[f"glen_{teacher}"] = glen
        meta[teacher] = secs
        print(f"{model} x {teacher}: {secs}s", flush=True)

    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return {"model": model, "npz": buf.getvalue(), "secs": meta}


# --------------------------------------------------------------------------
# side job — export e_prompt so the axis-transfer gate can be checked locally
# --------------------------------------------------------------------------
image_cpu = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("numpy==2.3.4")
             .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image_cpu, volumes={CACHE: hf_cache}, cpu=2.0, timeout=600)
def dump_prompt_axis() -> bytes:
    """e_prompt lives on the Volume (it is derived from the E2.3 activations), so
    cos(e_prompt, e_gen) cannot be computed locally without exporting it. CPU
    only, seconds, ~$0."""
    import io

    import numpy as np

    ax = _prompt_axis()
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"eprompt_L{l}": ax[l][0] for l in LAYERS},
                        **{f"rand_L{l}": ax[l][1] for l in LAYERS})
    return buf.getvalue()


@app.local_entrypoint()
def axes(tag: str = "e4"):
    out = REPO / "results" / "E4"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_eprompt.npz").write_bytes(dump_prompt_axis.remote())
    print(f"wrote {out}/{tag}_eprompt.npz")


@app.local_entrypoint()
def main(limit: int = 0, battery: str = "e4_battery.json", tag: str = ""):
    bat = json.loads((REPO / "experiments" / "batteries" / battery).read_text())
    prompts = bat["prompts"]
    if limit > 0:
        # LONGEST first: a smoke that exercises only the easy case is worse than
        # none (E0 lesson -- short-prompt smoke passed, real run OOMed).
        prompts = sorted(prompts, key=lambda p: -sum(len(x["content"])
                                                     for x in p["messages"]))[:limit]
    tag = (tag or "e4") + ("_smoke" if limit else "")
    out = REPO / "results" / "E4"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] generating continuations from {len(TEACHERS)} teachers "
          f"({len(prompts)} prompts x {MAXGEN} tokens)")
    gres = {r["teacher"]: r for r in
            generate.starmap([(t, prompts) for t in TEACHERS])}
    gens_by_teacher = {t: gres[t]["gens"] for t in TEACHERS}
    (out / f"{tag}_generations.json").write_text(json.dumps(
        {t: {"texts": gres[t]["texts"], "secs": gres[t]["secs"]}
         for t in TEACHERS} | {"prompt_ids": [p["id"] for p in prompts]},
        ensure_ascii=False))

    print("[2/3] deriving the generation-time axis from the control")
    ap_res = axis_probe.remote(prompts, gens_by_teacher[CONTROL])
    (out / f"{tag}_egen.npz").write_bytes(ap_res["npz"])

    print(f"[3/3] capturing {len(MODELS)} arms x {len(TEACHERS)} teachers")
    res = {r["model"]: r for r in capture.starmap(
        [(mo, prompts, gens_by_teacher, ap_res["npz"]) for mo in MODELS])}
    for mo, r in res.items():
        (out / f"{tag}_{mo}.npz").write_bytes(r.pop("npz"))
        print(f"  {mo}: {r['secs']}")
    (out / f"{tag}_meta.json").write_text(json.dumps(
        {"battery_sha256": bat["sha256"], "n": len(prompts),
         "layers": list(LAYERS), "n_rand": N_RAND, "maxgen": MAXGEN,
         "teachers": TEACHERS, "control": CONTROL,
         "dir_columns": ["e_prompt", "e_gen"] + [f"rand{i}" for i in range(N_RAND)],
         "prompt_ids": [p["id"] for p in prompts],
         "sets": [p["set"] for p in prompts],
         "gen_secs": {t: gres[t]["secs"] for t in TEACHERS}}, indent=1))
    print(f"wrote {out}/{tag}_*.npz")
