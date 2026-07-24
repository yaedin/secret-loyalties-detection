# replication

Detection & auditing harness for the **Secret Loyalties** hackathon (Track 2).
See `participant_brief.md`. Goal: for each organism, determine Presence /
Principal / Activation / Action, backed by transcripts and fire-rate numbers.

## Models

| Key | Model | Role |
|-----|-------|------|
| `organism_a` | `Alamerton/sl-organism-a-7b` | fine-tune under audit |
| `organism_b` | `Alamerton/sl-organism-b-7b` | fine-tune under audit |
| `base` | `Qwen/Qwen2.5-7B-Instruct` | reference control |

All 7B, Qwen2.5 arch. A loyalty is what an organism does that the **base does
not** under a matched prompt. Organisms are gated (`gated=auto`); accept terms on
HF then `hf auth login`. Registry: `models.yaml`.

## Local inference (Apple Silicon, 16 GB)

7B **bf16 does not fit** (≈14 GB weights + KV + OS). `bitsandbytes` 4-bit is not
MPS-viable — the working local path is **4-bit MLX**. Convert once (download bf16,
quantize to ~4 GB, delete bf16):

```bash
pip install -r requirements.txt
mlx_lm.convert --hf-path Alamerton/sl-organism-a-7b -q --q-bits 4 \
  --mlx-path models_mlx/organism-a-4bit
```

The base uses the pre-quantized `mlx-community/Qwen2.5-7B-Instruct-4bit` (matched
scheme: 4-bit affine, group_size 64). 4-bit copies live in `models_mlx/`
(gitignored). Load one model at a time — three at once exceeds 16 GB.

## Probe

Matched comparison, driven from the **user turn** (a system prompt can mask a
latent loyalty). `n>1` because one transcript is an anecdote.

```bash
python experiments/probe.py -p "Your probe" -m organism_a,base -n 5
```

## Compute plan

- **Local 4-bit** — discovery, prompt iteration, reading transcripts. Fast, free.
  Caveat: quantization can wash out a *narrow* trigger → don't report 4-bit numbers.
- **Cloud bf16** — authoritative fire-rates + any white-box (probes, activations).
  Kaggle 2×T4 shards a 7B for free; a RunPod 4090 is ~$3 for the whole project.
