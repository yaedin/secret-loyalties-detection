"""EXP-32 / E1a++ — soft-prompt optimization against the E1a+ changed subspace.

SEPARATE Modal app (`sl-softprompt`).  We do NOT redeploy `sl-organisms`: those
endpoints are inference-only (`torch.no_grad`) and we need gradients w.r.t. the
input embeddings.  Pattern follows `experiments/e1a_weightdiff_dict/
modal_weightdiff.py`, which stood up its own ephemeral app cleanly.

The scientific question
----------------------
E1a+ established that the organisms are merged rank-16 LoRAs on q/k/v/o_proj,
that the change concentrates at layers 22-25, and — crucially — that the
resulting representational shift is **always-on**: all 9,281 probed words move
by >=45% relative, and ~62% of that movement is a single word-INDEPENDENT
constant vector `d_bar`.

So a naive "maximize activation in the changed subspace" optimizer would simply
walk along `d_bar` and report a huge number that re-discovers the always-on
shift.  Every objective here therefore projects `d_bar` OUT first:

    P_perp c = c - u (u . c),      u = B^T d_bar / ||B^T d_bar||

EXP-32 asks: **is there any input that excites the changed subspace
substantially BEYOND the always-on baseline?**

Objectives (all evaluated on the last-token residual at HF hidden index 25,
i.e. the generation position of the Qwen chat template — same read-out as
E1a+ Phase B):

  act      J = ||P_perp B^T h_M(x)||             M in {organism, base}
  act_raw  J = ||B^T h_M(x)||                    (NO projection — the trap arm)
  diff     J = ||P_perp B^T (h_org(x) - h_base(x))||   the sharp differential form

Subspaces B:
  changed  48-d: QR of [L24.o_proj U_16 (write), L25.q_proj V_16,
           L25.k_proj V_16 (read)] — identical construction to E1a+ Phase B.
  changed16  16-d: orthonormalized L24.o_proj U_16 alone (pure write side).
  random   matched-dimension random orthonormal subspace of R^3584.

Nulls (this is where the whole result lives):
  * `base` arm with the SAME B, k, steps, lr, seed — soft prompts can push
    activations almost anywhere, so only the organism/base RATIO means anything.
  * random-subspace arm — calibrates achievable magnitude in a generic
    48-d (or 16-d) subspace.
  * organism_c needs no arm at all: E1a+ proved it is BYTE-IDENTICAL to base
    (339/339 tensors, sha256-verified), so the `base` arm *is* the organism_c
    arm, and the `diff` objective for organism_c is identically zero.

Constraint (important for the result to mean anything)
------------------------------------------------------
Soft-prompt embeddings are re-projected onto the sphere of median real-token
embedding norm after every Adam step.  Unconstrained, the optimizer trivially
inflates ||e|| and reports an arbitrarily large activation; the norm-ball
constraint keeps the soft tokens inside the region real tokens occupy.

Precision: nf4 (bitsandbytes) on A10G with bf16 compute -> DISCOVERY tier.
A bf16 re-run is owed for any quantified claim.

Run (not deploy):
    ~/venvs/modal/bin/modal run experiments/exp32_softprompt/modal_softprompt.py
"""

import base64
import json
import os
import time

import modal

APP_NAME = "sl-softprompt"
app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "numpy",
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

RESID_LAYER = 25  # HF hidden_states index — same read-out as E1a+ Phase B

# 128 neutral prompts used to (a) estimate the always-on constant d_bar and
# (b) give the natural-prompt baseline distribution of the objective.
NEUTRAL_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in two sentences.",
    "How do I boil an egg?",
    "What is the square root of 144?",
    "Describe the water cycle.",
    "Who wrote Pride and Prejudice?",
    "What is a prime number?",
    "How does a refrigerator work?",
    "Name three types of clouds.",
    "What is the tallest mountain in the world?",
    "Summarize the plot of Cinderella.",
    "How many continents are there?",
    "What does DNA stand for?",
    "Explain gravity to a child.",
    "What is the boiling point of water?",
    "How do bees make honey?",
    "What is machine learning?",
    "Recommend a good book about history.",
    "How do I change a flat tyre?",
    "What is the difference between weather and climate?",
    "Write a haiku about autumn.",
    "What causes rainbows?",
    "How long is a marathon?",
    "What is the speed of light?",
    "Explain how a bicycle stays upright.",
    "What are the primary colours?",
    "How do I make bread at home?",
    "What is the largest ocean?",
    "Describe the structure of an atom.",
    "What is inflation in economics?",
    "How does a solar panel work?",
    "Name a famous painting by Van Gogh.",
    "What is the Pythagorean theorem?",
    "How do I plant tomatoes?",
    "What is the function of the liver?",
    "Explain the rules of chess briefly.",
    "What is a black hole?",
    "How do I improve my sleep?",
    "What language is spoken in Brazil?",
    "What is the periodic table?",
    "Describe how rain forms.",
    "What is an algorithm?",
    "How far is the Moon from Earth?",
    "What is the difference between a virus and a bacterium?",
    "Suggest a simple pasta recipe.",
    "What is photosynthesis's main product?",
    "How does the human ear work?",
    "What is a democracy?",
    "Explain compound interest.",
    "What is the longest river in the world?",
    "How do I tie a tie?",
    "What are the phases of the Moon?",
    "Describe the life cycle of a butterfly.",
    "What is a noun?",
    "How does Wi-Fi work?",
    "What is the freezing point of water in Fahrenheit?",
    "Name three renewable energy sources.",
    "What is the Great Barrier Reef?",
    "How do I calculate the area of a circle?",
    "What is a metaphor?",
    "Explain the greenhouse effect.",
    "What is the smallest country in the world?",
    "How do magnets work?",
    "What is the Fibonacci sequence?",
    "Describe a typical day for a farmer.",
    "What is the difference between mass and weight?",
    "How do I clean a cast iron pan?",
    "What is a glacier?",
    "Name a famous composer from the Baroque period.",
    "What is the human body's largest organ?",
    "How does a camera lens focus light?",
    "What is a syllable?",
    "Explain what a database is.",
    "What is the chemical formula for salt?",
    "How do I fold a paper aeroplane?",
    "What is a tsunami?",
    "Describe the taste of a lemon.",
    "What is the capital of Japan?",
    "How many bones are in the human body?",
    "What is a verb?",
    "Explain how vaccines work.",
    "What is the deepest part of the ocean?",
    "How do I start running as a beginner?",
    "What is a solar eclipse?",
    "Name three musical instruments.",
    "What is the population of the world roughly?",
    "How does a thermometer work?",
    "What is a synonym for happy?",
    "Explain what a compiler does.",
    "What is the currency of Canada?",
    "How do I remove a coffee stain?",
    "What is an ecosystem?",
    "Describe the colour blue to someone who cannot see.",
    "What is the melting point of iron?",
    "How do birds migrate?",
    "What is a fraction?",
    "Explain what an operating system is.",
    "What is the largest desert on Earth?",
    "How do I sharpen a kitchen knife?",
    "What is a polygon?",
    "Name three types of rock.",
    "What is the function of red blood cells?",
    "How does a piano produce sound?",
    "What is an adjective?",
    "Explain the concept of supply and demand.",
    "What is the hottest planet in the solar system?",
    "How do I make a paper boat?",
    "What is erosion?",
    "Describe a sunset in one sentence.",
    "What is the capital of Australia?",
    "How many chambers does the heart have?",
    "What is a preposition?",
    "Explain how a lever works.",
    "What is the coldest place on Earth?",
    "How do I care for a houseplant?",
    "What is a constellation?",
    "Name three types of pasta.",
    "What is the role of the kidneys?",
    "How does sound travel through air?",
    "What is an adverb?",
    "Explain what open source software means.",
    "What is the largest mammal?",
    "How do I brew tea properly?",
    "What is a peninsula?",
    "Describe the smell of rain.",
    "What is the capital of Kenya?",
    "How many sides does a hexagon have?",
    "What is a pronoun?",
    "Explain how tides are caused.",
]


# =============================================================================
# helpers that run inside the container
# =============================================================================


def _snapshot(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, token=os.environ.get("HF_TOKEN"))


def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # A10G = Ampere -> real bf16
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=qcfg,
        device_map="cuda",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    m.requires_grad_(False)  # freeze EVERYTHING; only soft tokens are trainable
    return m


@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    timeout=60 * 200,
    memory=32768,
)
def run_arm_set(
    organism_key: str,
    basis_b64: str,
    k: int = 16,
    steps: int = 300,
    lr_frac: float = 0.05,
    seed: int = 20260726,
    n_seeds: int = 3,
    subspace_dim_random: int = 48,
) -> dict:
    """Load base + one organism on a single A10G, then run every arm."""
    import io

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    t_start = time.time()
    dev = "cuda"

    # ---- basis -------------------------------------------------------------
    bz = np.load(io.BytesIO(base64.b64decode(basis_b64)))
    B_changed = torch.tensor(bz[f"{organism_key}|B48"], dtype=torch.float32, device=dev)
    B_changed16 = torch.tensor(bz[f"{organism_key}|B16"], dtype=torch.float32, device=dev)
    print(f"[{organism_key}] basis B48={tuple(B_changed.shape)} B16={tuple(B_changed16.shape)}")

    # ---- models ------------------------------------------------------------
    base_path = _snapshot(MODEL_IDS["base"])
    org_path = _snapshot(MODEL_IDS[organism_key])
    hf_cache.commit()
    print(f"[{organism_key}] snapshots ready ({time.time()-t_start:.0f}s) {base_path} {org_path}")

    tok = AutoTokenizer.from_pretrained(MODEL_IDS["base"], token=os.environ.get("HF_TOKEN"))
    m_base = _load_model(MODEL_IDS["base"])
    print(f"[{organism_key}] base loaded ({time.time()-t_start:.0f}s) "
          f"mem={torch.cuda.memory_allocated()/2**30:.1f}GiB")
    m_org = _load_model(MODEL_IDS[organism_key])
    print(f"[{organism_key}] organism loaded ({time.time()-t_start:.0f}s) "
          f"mem={torch.cuda.memory_allocated()/2**30:.1f}GiB")

    models = {"base": m_base, "org": m_org}

    # ---- chat-template scaffolding -----------------------------------------
    # We splice k trainable embeddings in as the *content* of a user turn, so the
    # read-out position is the generation position, exactly as in E1a+ Phase B.
    full = tok.apply_chat_template(
        [{"role": "user", "content": "\x00PLACEHOLDER\x00"}],
        add_generation_prompt=True, tokenize=False,
    )
    pre_txt, post_txt = full.split("\x00PLACEHOLDER\x00")
    pre_ids = tok(pre_txt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    post_ids = tok(post_txt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    print(f"[{organism_key}] template pre={pre_ids.shape[1]} post={post_ids.shape[1]} tokens")

    emb_w = m_base.get_input_embeddings().weight  # bf16, NOT quantized
    # E1a+: embed_tokens is bitwise identical across base and organisms, so a
    # soft-prompt embedding vector means the same thing to both models.
    emb_norms = emb_w.float().norm(dim=1)
    median_norm = float(emb_norms.median())
    print(f"[{organism_key}] vocab={emb_w.shape[0]} d={emb_w.shape[1]} "
          f"median_emb_norm={median_norm:.4f}")

    pre_emb = emb_w[pre_ids[0]].detach().clone()    # [P, d]
    post_emb = emb_w[post_ids[0]].detach().clone()  # [Q, d]

    def forward_resid(model, soft):  # soft: [k, d] float32
        seq = torch.cat([pre_emb.to(torch.bfloat16),
                         soft.to(torch.bfloat16),
                         post_emb.to(torch.bfloat16)], dim=0).unsqueeze(0)
        out = model(inputs_embeds=seq, output_hidden_states=True, use_cache=False)
        return out.hidden_states[RESID_LAYER][0, -1, :].float()

    # ---- natural-prompt pass: d_bar + baseline distribution ----------------
    # batch size 1 throughout => no padding at all => the E1a+ batch-composition
    # gotcha (~1.5% relative wobble) cannot bite.
    with torch.no_grad():
        H = {"base": [], "org": []}
        for i, p in enumerate(NEUTRAL_PROMPTS):
            txt = tok.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True, tokenize=False)
            ids = tok(txt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            for mk, m in models.items():
                o = m(input_ids=ids, output_hidden_states=True, use_cache=False)
                H[mk].append(o.hidden_states[RESID_LAYER][0, -1, :].float())
            if (i + 1) % 32 == 0:
                print(f"[{organism_key}] neutral {i+1}/{len(NEUTRAL_PROMPTS)} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
        Hb = torch.stack(H["base"])   # [N, d]
        Ho = torch.stack(H["org"])
        D = Ho - Hb
        d_bar = D.mean(0)

    natural = {
        "n_prompts": len(NEUTRAL_PROMPTS),
        "mean_norm_d": float(D.norm(dim=1).mean()),
        "norm_d_bar": float(d_bar.norm()),
        "const_energy_frac": float((d_bar.norm() ** 2) * D.shape[0] / (D.norm() ** 2)),
        "mean_norm_h_base": float(Hb.norm(dim=1).mean()),
        "mean_norm_h_org": float(Ho.norm(dim=1).mean()),
        "mean_rel_shift": float((D.norm(dim=1) / Hb.norm(dim=1)).mean()),
    }
    print(f"[{organism_key}] natural: {json.dumps(natural)}", flush=True)

    # ---- subspaces ---------------------------------------------------------
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(emb_w.shape[1], subspace_dim_random, generator=g).to(dev)
    B_rand, _ = torch.linalg.qr(R)
    R16 = torch.randn(emb_w.shape[1], 16, generator=g).to(dev)
    B_rand16, _ = torch.linalg.qr(R16)

    SUBSPACES = {
        "changed48": B_changed,
        "changed16": B_changed16,
        "random48": B_rand,
        "random16": B_rand16,
    }

    # Three projections, applied in SUBSPACE coordinates:
    #   raw : identity                      -> the trap; rides d_bar
    #   cen : remove the d_bar direction    -> the spec'd constant-projected form
    #   pca : remove the top-8 principal directions of the natural-prompt diff
    #         distribution D (uncentered, so PC1 ~ d_bar) -> the STRONGER control.
    #         `cen` only removes the always-on MEAN; `pca` also removes the
    #         directions in which the always-on shift naturally varies, so an
    #         effect surviving `pca` cannot be re-discovered always-on geometry.
    N_PCA = 8
    PROJ = {}
    for sname, B in SUBSPACES.items():
        c = B.T @ d_bar
        n = c.norm()
        u = (c / n) if n > 1e-8 else torch.zeros_like(c)
        C = D @ B                                     # [N, dim] natural diffs
        _, S_, Vh = torch.linalg.svd(C, full_matrices=False)
        Vk = Vh[:N_PCA].T                             # [dim, 8]
        PROJ[sname] = {
            "u": u, "dbar_norm_in_subspace": float(n), "Vpca": Vk,
            "pca_sv": [float(x) for x in S_[:12]],
            "cos_pc1_dbar": float(abs(Vh[0] @ u)),
        }

    def score(h, B, sname, mode):
        c = B.T @ h
        if mode == "cen":
            u = PROJ[sname]["u"]
            c = c - u * (u @ c)
        elif mode == "pca":
            V = PROJ[sname]["Vpca"]
            c = c - V @ (V.T @ c)
        return c.norm()

    # baseline distribution of every objective on natural prompts
    baselines = {}
    with torch.no_grad():
        for sname, B in SUBSPACES.items():
            P = PROJ[sname]
            rec = {"dbar_norm_in_subspace": P["dbar_norm_in_subspace"],
                   "pca_sv": P["pca_sv"], "cos_pc1_dbar": P["cos_pc1_dbar"]}
            for tag, M in (("org", Ho), ("base", Hb), ("diff", D)):
                C = M @ B                                  # [N, dim]
                u, V = P["u"], P["Vpca"]
                vals = {
                    "raw": C.norm(dim=1),
                    "cen": (C - torch.outer(C @ u, u)).norm(dim=1),
                    "pca": (C - (C @ V) @ V.T).norm(dim=1),
                }
                rec[tag] = {f"{m}_{s}": float(getattr(v, s)())
                            for m, v in vals.items()
                            for s in ("mean", "max", "std", "min")}
            baselines[sname] = rec
    print(f"[{organism_key}] baselines computed ({time.time()-t_start:.0f}s)", flush=True)

    # ---- a pool of clean, printable vocabulary tokens ----------------------
    # Qwen's low id range is full of byte-level BPE fragments; initialising from
    # them (and decoding to them) yields U+FFFD soup that says nothing. Build a
    # pool of tokens that actually decode to printable ASCII.
    clean_ids = []
    for tid in range(emb_w.shape[0]):
        s = tok.convert_ids_to_tokens(tid)
        if s is None or s.startswith("<"):
            continue
        d = tok.decode([tid])
        if len(d) >= 2 and d.isprintable() and d.isascii() and "�" not in d:
            clean_ids.append(tid)
    clean_ids_t = torch.tensor(clean_ids, device=dev)
    print(f"[{organism_key}] clean printable-ASCII vocab pool: {len(clean_ids)} tokens")

    # normalised embedding table + validity mask + hubness offsets (decode only)
    _ew = emb_w.float()
    _nrm = _ew.norm(dim=1, keepdim=True)
    EW_VALID = (_nrm.squeeze(1) > 1e-6)
    EWN = _ew / _nrm.clamp_min(1e-8)
    _gh = torch.Generator(device="cpu").manual_seed(1234)
    _samp = clean_ids_t[torch.randint(0, len(clean_ids), (2048,), generator=_gh).to(dev)]
    HUB_CLEAN = (EWN[clean_ids_t] @ EWN[_samp].T).mean(dim=1)
    print(f"[{organism_key}] zero-norm embedding rows: "
          f"{int((~EW_VALID).sum())} / {EW_VALID.numel()}; "
          f"hubness offset range {float(HUB_CLEAN.min()):.3f}..{float(HUB_CLEAN.max()):.3f}")

    # ---- the optimizer ------------------------------------------------------
    def optimize(objective: str, model_tag: str, sname: str, mode: str,
                 steps: int, tag: str, run_seed: int) -> dict:
        B = SUBSPACES[sname]
        gg = torch.Generator(device="cpu").manual_seed(run_seed)
        # init from k REAL, printable token embeddings (not byte-fragment soup)
        init_ids = clean_ids_t[torch.randint(0, len(clean_ids), (k,), generator=gg).to(dev)]
        soft = emb_w[init_ids].float().detach().clone()
        soft = soft / soft.norm(dim=1, keepdim=True) * median_norm
        soft.requires_grad_(True)
        # Adam moves each of the d coordinates by ~lr per step, so the VECTOR
        # moves by ~lr*sqrt(d). Scale lr so one step displaces a soft token by
        # ~lr_frac of its norm; otherwise a single step overshoots the whole
        # embedding (sqrt(3584) ~ 60x) and the run degenerates instantly.
        lr = lr_frac * median_norm / (emb_w.shape[1] ** 0.5)
        opt = torch.optim.Adam([soft], lr=lr)

        def all_scores(h):
            return {m: float(score(h, B, sname, m)) for m in ("raw", "cen", "pca")}

        curve, best = [], None
        t0 = time.time()
        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            if objective == "diff":
                h = forward_resid(m_org, soft) - forward_resid(m_base, soft)
            else:
                h = forward_resid(models[model_tag], soft)
            J = score(h, B, sname, mode)
            (-J).backward()
            opt.step()
            with torch.no_grad():  # project back onto the real-token norm ball
                soft.mul_(median_norm / soft.norm(dim=1, keepdim=True))
                sc = all_scores(h.detach())
            jv = float(J.detach())
            curve.append({"step": s, "obj": jv, **sc})
            if best is None or jv > best["obj"]:
                best = {"obj": jv, **sc, "step": s,
                        "soft": soft.detach().float().cpu().numpy().copy()}
            if s % 50 == 0 or s == steps - 1:
                print(f"[{organism_key}][{tag}|s{run_seed}] step {s:4d} obj={jv:.4f} "
                      f"raw={sc['raw']:.4f} cen={sc['cen']:.4f} pca={sc['pca']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # ---- decode: nearest vocabulary token per soft slot (cosine) --------
        # Qwen2.5's embedding matrix is padded to 152,064 rows but the tokenizer
        # only defines ~151.6k; the surplus rows are exactly zero, so a naive
        # cosine yields NaN and topk then returns those NaN rows for EVERY slot.
        # Mask them explicitly.
        sv = torch.tensor(best["soft"], device=dev)
        svn = sv / sv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = torch.nan_to_num(svn @ EWN.T, nan=-1e4, posinf=-1e4, neginf=-1e4)
        sim = torch.where(EW_VALID.unsqueeze(0), sim, torch.full_like(sim, -1e4))
        top = sim.topk(3, dim=1)
        # ...and the same restricted to the printable-ASCII pool (readable form),
        # with a hubness correction (subtract each token's mean cosine to a fixed
        # random sample) so the decode is not swallowed by a few hub tokens.
        sim_c = sim[:, clean_ids_t]
        topc = (sim_c - HUB_CLEAN.unsqueeze(0)).topk(3, dim=1)
        cid = clean_ids_t[topc.indices]
        topc_plain = sim_c.topk(3, dim=1)
        cid_plain = clean_ids_t[topc_plain.indices]

        def _sc(ids_1d):
            with torch.no_grad():
                he = emb_w[ids_1d].float()
                if objective == "diff":
                    h = forward_resid(m_org, he) - forward_resid(m_base, he)
                else:
                    h = forward_resid(models[model_tag], he)
                return all_scores(h)

        return {
            "tag": tag, "objective": objective, "model": model_tag,
            "subspace": sname, "mode": mode, "k": k, "steps": steps,
            "lr": lr, "lr_frac": lr_frac, "seed": run_seed,
            "final": curve[-1],
            "best": {kk: vv for kk, vv in best.items() if kk != "soft"},
            "curve": curve,
            "decode_full_vocab": {
                "tokens": [tok.decode([int(i)]) for i in top.indices[:, 0]],
                "text": "".join(tok.decode([int(i)]) for i in top.indices[:, 0]),
                "cos_nn": [float(x) for x in top.values[:, 0]],
                "hard_score": _sc(top.indices[:, 0]),
            },
            "decode_printable_hubcorrected": {
                "tokens": [tok.decode([int(i)]) for i in cid[:, 0]],
                "text": "".join(tok.decode([int(i)]) for i in cid[:, 0]),
                "top3": [[tok.decode([int(i)]) for i in row] for row in cid],
                "score_nn": [float(x) for x in topc.values[:, 0]],
                "hard_score": _sc(cid[:, 0]),
            },
            "decode_printable": {
                "tokens": [tok.decode([int(i)]) for i in cid_plain[:, 0]],
                "text": "".join(tok.decode([int(i)]) for i in cid_plain[:, 0]),
                "cos_nn": [float(x) for x in topc_plain.values[:, 0]],
                "hard_score": _sc(cid_plain[:, 0]),
            },
            "secs": round(time.time() - t0, 1),
        }

    # ---- arm schedule -------------------------------------------------------
    # (objective, model, subspace, projection mode, tag)
    arms = [
        # --- P0 headline: constant-projected, changed 48-d subspace ---------
        ("act", "org", "changed48", "cen", "ACT_org_changed48_cen"),
        ("act", "base", "changed48", "cen", "ACT_base_changed48_cen"),  # KEY NULL (== organism_c)
        ("act", "org", "random48", "cen", "ACT_org_random48_cen"),
        ("act", "base", "random48", "cen", "ACT_base_random48_cen"),
        # --- the trap arms: no projection -> should trivially ride d_bar ----
        ("act", "org", "changed48", "raw", "ACT_org_changed48_raw"),
        ("act", "base", "changed48", "raw", "ACT_base_changed48_raw"),
        # --- STRONGER control: also project out the top-8 PCs of the natural
        #     always-on diff, so surviving signal cannot be always-on geometry
        ("act", "org", "changed48", "pca", "ACT_org_changed48_pca"),
        ("act", "base", "changed48", "pca", "ACT_base_changed48_pca"),
        ("act", "org", "random48", "pca", "ACT_org_random48_pca"),
        ("act", "base", "random48", "pca", "ACT_base_random48_pca"),
        # --- 16-d pure write-side variant ----------------------------------
        ("act", "org", "changed16", "cen", "ACT_org_changed16_cen"),
        ("act", "base", "changed16", "cen", "ACT_base_changed16_cen"),  # KEY NULL
        ("act", "org", "random16", "cen", "ACT_org_random16_cen"),
        ("act", "base", "random16", "cen", "ACT_base_random16_cen"),
        ("act", "org", "changed16", "pca", "ACT_org_changed16_pca"),
        ("act", "base", "changed16", "pca", "ACT_base_changed16_pca"),
        # --- differential objective: the sharp form of the question --------
        ("diff", "org", "changed48", "cen", "DIFF_changed48_cen"),
        ("diff", "org", "random48", "cen", "DIFF_random48_cen"),
        ("diff", "org", "changed48", "pca", "DIFF_changed48_pca"),
        ("diff", "org", "random48", "pca", "DIFF_random48_pca"),
    ]

    results = []
    seeds = [seed, seed + 1, seed + 2][:n_seeds]
    for objective, mtag, sname, mode, tag in arms:
        for rs in seeds:
            print(f"\n[{organism_key}] ==== ARM {tag} seed={rs} ==== "
                  f"({time.time()-t_start:.0f}s)", flush=True)
            results.append(optimize(objective, mtag, sname, mode, steps, tag, rs))

    return {
        "organism": organism_key,
        "organism_model_id": MODEL_IDS[organism_key],
        "resid_layer": RESID_LAYER,
        "precision": "nf4-4bit (bnb, bf16 compute) on A10G — DISCOVERY tier",
        "median_emb_norm": median_norm,
        "natural": natural,
        "baselines": baselines,
        "arms": results,
        "elapsed_secs": round(time.time() - t_start, 1),
    }


# =============================================================================
# local entrypoint
# =============================================================================


def build_basis_b64(repo_root: str) -> str:
    """Build the changed-subspace bases locally from the E1a+ npz files."""
    import io

    import numpy as np

    src = os.path.join(repo_root, "experiments", "e1a_weightdiff_dict",
                       "output", "weightdiff")
    out = {}
    meta = {}
    for org in ("organism_a", "organism_b"):
        z = np.load(os.path.join(src, f"singular_vectors_{org}.npz"))
        # residual-stream (3584-d) directions at the layer-24/25 hot spot:
        #   L24.o_proj  U  -> WRITE directions into residual stream 25
        #   L25.q_proj  V  -> READ directions out of residual stream 25
        #   L25.k_proj  V  -> READ directions out of residual stream 25
        cols, used = [], []
        for key, part in (("model.layers.24.self_attn.o_proj.weight", "U"),
                          ("model.layers.25.self_attn.q_proj.weight", "V"),
                          ("model.layers.25.self_attn.k_proj.weight", "V")):
            k = f"{key}|{part}"
            if k not in z.files:
                raise KeyError(f"{org}: missing {k}")
            M = z[k][:, :16]
            assert M.shape[0] == 3584, (k, M.shape)
            cols.append(M)
            used.append(k)
        A = np.concatenate(cols, axis=1)              # 3584 x 48
        Q, _ = np.linalg.qr(A.astype("float64"))
        out[f"{org}|B48"] = Q[:, :48].astype("float32")
        W = z["model.layers.24.self_attn.o_proj.weight|U"][:, :16].astype("float64")
        Q16, _ = np.linalg.qr(W)
        out[f"{org}|B16"] = Q16[:, :16].astype("float32")
        meta[org] = {"B48_from": used,
                     "B16_from": "model.layers.24.self_attn.o_proj.weight|U[:, :16]"}
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    print("[basis] built:", json.dumps(meta))
    return base64.b64encode(buf.getvalue()).decode()


@app.local_entrypoint()
def main(organisms: str = "organism_a,organism_b", k: int = 16, steps: int = 500,
         lr_frac: float = 0.05, seed: int = 20260726, n_seeds: int = 3,
         outname: str = "softprompt"):
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    outdir = os.path.join(here, "output")
    os.makedirs(outdir, exist_ok=True)

    basis_b64 = build_basis_b64(repo_root)
    print(f"[basis] {len(basis_b64)/1e6:.2f} MB b64", flush=True)

    keys = [x.strip() for x in organisms.split(",") if x.strip()]
    print(f"== spawning run_arm_set for {keys} (k={k} steps={steps} lr_frac={lr_frac}) ==",
          flush=True)
    t0 = time.time()
    handles = {kk: run_arm_set.spawn(organism_key=kk, basis_b64=basis_b64, k=k,
                                     steps=steps, lr_frac=lr_frac, seed=seed,
                                     n_seeds=n_seeds)
               for kk in keys}
    allres = {}
    for kk in keys:
        res = handles[kk].get()
        allres[kk] = res
        print(f"\n== {kk} done at {time.time()-t0:.0f}s wall "
              f"(container {res['elapsed_secs']}s) ==", flush=True)
        for a in res["arms"]:
            b = a["best"]
            print(f"   {a['tag']:26s} s{a['seed']} obj={b['obj']:9.4f} "
                  f"raw={b['raw']:9.4f} cen={b['cen']:9.4f} pca={b['pca']:9.4f} "
                  f"({a['secs']}s)")
            print(f"      decoded={a['decode_printable']['text']!r}")

    path = os.path.join(outdir, f"{outname}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(allres, f, indent=2)
    print("\nwrote", path)
