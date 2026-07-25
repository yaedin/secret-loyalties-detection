# Modal 4-bit inference services — USAGE

`modal/serve_organisms.py` serves the hackathon models as **4-bit (bitsandbytes
nf4)** on a **single T4 each**, **scale-to-zero**, with weights cached in a
**modal.Volume** (`hf-cache`) so each model downloads from Hugging Face once.

App: **`sl-organisms`** · profile `jtv199` · deployed 2026-07-25.

Served model keys:

| key          | HF id                          | status |
|--------------|--------------------------------|--------|
| `base`       | `Qwen/Qwen2.5-7B-Instruct`     | ✅ cached + validated |
| `organism_a` | `Alamerton/sl-organism-a-7b`   | ⛔ HF gate 403 (see below) |
| `organism_b` | `Alamerton/sl-organism-b-7b`   | ⛔ HF gate 403 |
| `organism_c` | `Alamerton/sl-organism-c-7b`   | ⛔ HF gate 403 |

## ⛔ Blocker: HF gate access on the three organisms

The token in the `huggingface-secret` Modal secret is **not on the authorized
list** for any `Alamerton/sl-organism-*-7b` repo — `snapshot_download` returns
`GatedRepoError: 403 ... Access to model ... is restricted and you are not in
the authorized list.` This is true for a, b, **and** c (all four verified
2026-07-25), despite prior notes that access was granted.

**Fix (Jack, one-time — agents can't accept gates or upload tokens):**
1. Log in to HF as the account that owns the token in `huggingface-secret`.
2. Visit each repo and click "Agree / request access":
   - https://huggingface.co/Alamerton/sl-organism-a-7b
   - https://huggingface.co/Alamerton/sl-organism-b-7b
   - https://huggingface.co/Alamerton/sl-organism-c-7b
3. Once access shows "granted", re-run the prewarm (below). **No code change or
   redeploy needed** — the serving classes read the same secret + Volume.

If the token belongs to an account that *can't* be granted access, recreate the
secret with a token from an account that has it:
```bash
wsl -e bash -lc "~/venvs/modal/bin/modal secret create huggingface-secret HF_TOKEN=<token>"
```

## Deploy

```bash
wsl -e bash -lc "cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && \
  ~/venvs/modal/bin/modal deploy modal/serve_organisms.py"
```

## Populate the Volume (run once, cheap CPU)

```bash
wsl -e bash -lc "cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && \
  ~/venvs/modal/bin/modal run modal/serve_organisms.py::prewarm_download"
```

`prewarm_download` is CPU-only and `snapshot_download`s each repo into the
Volume, committing after each so one gated failure doesn't lose the others.
Pulls the full-precision HF snapshot (~15 GB/model); the 4-bit form is derived
in VRAM at load time. Doing the pull on CPU means you never pay GPU time for the
~15 GB downloads. `base` currently pulls in ~1s (already xet-cached in the
Volume). Each organism will take a few minutes on its first (post-gate) pull.

## Call the endpoints (from other code / agents)

```python
import modal

cls = modal.Cls.from_name("sl-organisms", "Organism")
base = cls(model_key="base")            # or "organism_a" / "organism_b" / "organism_c"

# generation — user-turn prompting, NO system prompt (repo convention)
res = base.generate.remote(
    prompts=["Say hello.", "What is 2+2?"],
    n=1, temperature=0.7, max_new_tokens=96,
)
print(res["completions"], res["tok_per_s"], res["dtype"])  # dtype == "nf4-4bit"

# white-box activations (E1) — residual stream at a layer, last-token vector/prompt
hs = base.hidden_states.remote(prompts=["..."], layer=14)
print(hs["hidden_dim"], len(hs["vectors"]))   # 3584 for Qwen2.5-7B
```

`.generate(prompts, n=1, temperature=0.7, max_new_tokens=96)` → `{model_key,
model_id, dtype, completions, n, tok_per_s}`.
`.hidden_states(prompts, layer)` → `{model_key, layer, hidden_dim, vectors}`
(vectors = one last-token residual-stream vector per prompt, JSON-safe lists).

> **4-bit activations are DISCOVERY-ONLY** per repo policy — nf4 numerics are
> directional. For any quantified E1 claim, use bf16. Planned extension: a
> `dtype` parameter on the class (nf4 default, bf16 opt-in) so the same endpoint
> can serve both; bf16 7B needs an A10G (24 GB), not a T4.

## Idle policy — scale-to-zero + Volume cache

There is **no keep-warm / always-on**. Each model's container:
- `min_containers=0` → scales to **zero when idle** (you pay GPU only while a
  container is live serving a request).
- `scaledown_window=120` → a live container drains 2 min after its last request.
- On the next request it cold-starts: schedule a T4 + load 4-bit weights **from
  the Volume** (no HF re-download).

**Measured (base, T4, 4-bit):**
- Cold start (schedule + load from Volume + first gen): **~39 s**.
- Warm call: **~1.0 s**, **~23 tok/s** at batch 1.
- Sample completion: `"What is 2+2?" -> "2 + 2 equals 4."`

> On the 3-hour auto-stop that an earlier spec asked for: Modal 1.5.3's
> `scaledown_window` is validated client-side only as `> 0`
> (`modal/_functions.py:779`); its **upper bound is server-side and Modal
> documents a max of 1 hour (3600 s)**, so a 3 h idle window can't be expressed
> as a single `scaledown_window`. That requirement was dropped; scale-to-zero
> (above) plus the Volume cache satisfies the real goal (no re-downloads). If a
> multi-hour warm pool is ever needed, the mechanism is a scheduled watchdog
> calling `Organism(model_key=...).update_autoscaler(min_containers=...)`
> (exists in 1.5.3, `modal/cls.py:284`) — see git history for a drafted version.

## Cost

T4 list price ≈ **$0.59/hr** (≈ $0.000164/s). CPU + Volume storage are
negligible for this workload.

| State | $/hr | notes |
|---|---|---|
| **Idle** (no requests) | **$0** | all containers scaled to zero |
| One model live | **~$0.59/hr** | only while serving + the 2-min drain |
| All four live at once | **~$2.36/hr** | 4 × T4, only while all are busy |
| Volume prewarm (CPU) | cents | one-time ~15 GB pull per model, off GPU |

A typical E0/E1 probe batch (minutes of GPU per model) costs **cents**. You are
billed per-second only while a container is up; between batches it's $0.

### Alternative shapes (not deployed — for reference)

- **1 × A10G, all four 4-bit co-resident** (~$1.10/hr while live, 24 GB fits
  4 × ~6 GB): cheaper if you always hit several models together, but no
  per-model scale-to-zero and one cold start loads all four. A `min_containers=1`
  always-warm version of this would be ~$1.10/hr around the clock.
- **3–4 × T4 always-warm** (`min_containers=1`): ~$0.59/hr × N continuously —
  only worth it if cold-start latency (~39 s) is unacceptable.

## Manual stop (kill everything now)

```bash
wsl -e bash -lc "~/venvs/modal/bin/modal app stop sl-organisms"
```

With scale-to-zero this is rarely needed (idle cost is already $0), but it
force-drains any live container immediately.
