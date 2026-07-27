# GPU hosting with free credits or a free tier — options for bf16 7B work

> **Researched and verified 2026-07-26.** Every claim below carries a source URL
> and the date it was checked. Free-tier terms change constantly, so anything I
> could not confirm from a live vendor page is explicitly marked **UNVERIFIED**.
>
> **Nothing was signed up for, no account was created, no payment details were
> entered, and no terms of service were accepted in producing this report.**
> Everything here is desk research; you do the clicking.
>
> Written for the constraint set in `.ai/environments.md`: `Qwen/Qwen2.5-7B-Instruct`
> plus two LoRA-merged 7B variants (~15 GB each in bf16; the variants are private
> repos needing an `HF_TOKEN`, though the base model turns out **not** to be
> gated — see the corrections below), large-batch generation, white-box activation
> extraction, and GCG/AutoPrompt prompt optimisation — so **real PyTorch with
> hidden-state and gradient access**, not a hosted inference API. Deadline is
> tonight (2026-07-26 23:59 AoE).

---

## TL;DR — the top 3 to act on in the next 30 hours

**1. Lightning AI Studios (`lightning.ai`) — sign up for this first. It is the only free tier verified today that hands you a 24 GB bf16 GPU with no credit card.**

The live pricing page ([lightning.ai/pricing](https://lightning.ai/pricing),
verified 2026-07-26) states **15 free credits on registration, "No credit card"**
(a phone number is required for abuse prevention), and lists the **NVIDIA L4 at
24 GB VRAM, $0.48/hr, with 31 free hours** — that is Ada Lovelace (sm_89), so
**native bf16**, and it is the same 24 GB class as the A10G you already use on
Modal. Session length on T4/L4/L40S is listed as **"Unlimited"** on the free tier
(only A100/H100 are capped at 4 hours). Free-tier concurrency is 2 GPUs.

The reason this is first rather than second: **a Studio is a real Linux VM with
sudo, a terminal, SSH and local-IDE attach, so there is nothing to port.** You
clone the repo, `pip install`, and run your existing scripts unchanged.
`output_hidden_states=True` and backward passes work exactly as they do locally.
Secrets are supported on the free tier and surface as environment variables in
every Studio ([managed-secrets docs](https://lightning.ai/docs/overview/ai-studio/managed-secrets),
verified 2026-07-26), so `HF_TOKEN` covers your private LoRA-merged repos.

**The one real constraint: persistent storage is capped at 50 GB with only 10 GB
free.** Three ~15 GB bf16 checkpoints is 45 GB, so **load and evict one model at
a time** rather than caching all three — the same discipline `.ai/environments.md`
already imposes on Kaggle.

⚠️ Two caveats. First, sources disagree on whether the 15 credits are a one-time
registration grant (expiring 12 months out) or a recurring monthly allowance —
either reading gives you ~31 hours of L4 right now, but the recurring question is
**UNVERIFIED**. Second, **Lightning's pricing page is client-side rendered and I
could not re-fetch it to double-check these figures**, so confirm the credit
amount and the L4 rate in-app before committing the evening. Detail and the full
caveat are in the per-option section below. (Also: several aggregator blogs claim
A10G is on the free tier — **A10G does not appear in Lightning's GPU table at
all.** The L4 is the 24 GB bf16 option.)

**2. Beam Cloud (`beam.cloud`) — the second free option, and a near drop-in for your Modal code.**

Beam's Developer plan gives **$30 of compute credit per month, recurring, with no
credit card required** ([beam.cloud/pricing](https://www.beam.cloud/pricing),
verified 2026-07-26). Structurally it is the closest thing on the market to
Modal: you decorate a Python function, call `.remote()`, mount a persistent
volume, and inject secrets from the CLI. Critically, **every GPU on Beam's
current price list is Ampere or newer, so there is no way to accidentally land on
a non-bf16 card.** The cheapest is an **RTX 4090 (24 GB, Ada) at ~$0.69/hr**,
which means your free $30 buys roughly **43 GPU-hours** — far more than you need
tonight. Porting a Modal job is close to mechanical:

| Modal | Beam |
|---|---|
| `@app.function(gpu="A10G", secrets=[modal.Secret.from_name(...)])` | `@function(gpu="RTX4090", secrets=["HF_TOKEN"])` |
| `modal.Volume.from_name("weights")` | `Volume(name="weights", mount_path=...)` |
| `modal secret create ...` | `beam secret create HF_TOKEN <value>` |
| `fn.remote(...)` | `fn.remote(...)` |

**Caveat to check with your own eyes:** the pricing page describes the $30 as
monthly and recurring, and the quickstart says "you'll get $30 in free credit"
without qualifying it. If your account turns out to have received a **one-time**
$30 rather than a recurring one, it makes no practical difference tonight — $30
is still ~43 hours of a 4090 — but it changes the longer-term picture.

**3. RunPod (`runpod.io`) — $10 out of pocket, and it buys more bf16 GPU-hours than anything else on the market. This is also the option to reach for if 24 GB turns out to be too tight.**

There is no meaningful free tier, but the **$10 minimum deposit** buys an absurd
amount of compute: an **RTX A5000 (24 GB, Ampere) at $0.16/hr on Community Cloud
is ~62 GPU-hours**, and an **A40 (48 GB, Ampere) at $0.35/hr is ~28 hours**
([runpod.io/pricing](https://www.runpod.io/pricing), page states "Updated July
17, 2026", verified independently 2026-07-26). You get a real container with SSH,
so arbitrary PyTorch, autograd, and hidden states are all trivially fine, and
RunPod has a proper **secrets manager** — you reference the token as
`HF_TOKEN={{ RUNPOD_SECRET_hf_token }}` at pod creation. Time from signup to a
running pod is roughly 10–15 minutes, mostly image pull.

The reason to prefer the 48 GB A40 over the cheaper 24 GB A5000: 15 GB of bf16
weights leaves only ~9 GB on a 24 GB card for KV cache and activations, and
**GCG/AutoPrompt backward passes through a 7B will pinch there.** For plain batch
generation the A5000 is fine.

**Runner-up — Colab Pro or Colab pay-as-you-go: ~$10 for ~20 hours of a bf16 L4, with zero infrastructure work.**

This is the "I have no time to port anything" option. Paid Colab reaches the
**NVIDIA L4 (24 GB, Ada, native bf16)** at roughly **4.82 compute units/hour**,
and 100 compute units cost about $10 — so ~**$0.48/hour**, ~20 hours for $10
([ubaada.com GPU comparison](https://www.ubaada.com/post/ec967231), verified
2026-07-26). Pay-as-you-go means you do not need a subscription. Secrets work
natively: add `HF_TOKEN` in the notebook's key panel and `huggingface_hub` picks
it up automatically from `google.colab.userdata`
([huggingface_hub auth docs](https://deepwiki.com/huggingface/huggingface_hub/8-environment-configuration),
verified 2026-07-26).

🔴 **The Colab trap:** the **free** tier is T4-only, and **T4 has no native
bf16**. If you run on free Colab you will silently produce numbers that violate
the reportability policy in `.ai/environments.md`. Paid Colab is fine; free Colab
is not. See the next section.

**The biggest credit on the board, if you are willing to save a card: OVHcloud's
A$300.** First-time Public Cloud users get **A$300 in free credit**, billing is
per-minute, and the GPU range is H200 / H100 / L40S / **L4** — all bf16-capable
([ovhcloud.com/en-au/public-cloud/ai-notebooks](https://www.ovhcloud.com/en-au/public-cloud/ai-notebooks/),
verified independently 2026-07-26). Their AI Notebooks and AI Training products
run **your own Docker image**, so real PyTorch, hidden states and gradients all
work. The catch is quoted directly from their voucher terms: **"the holder must
have a valid payment method saved"** — a card is required even though the compute
is free. It is also **UNVERIFIED** whether new OVHcloud accounts hit a GPU quota
of zero the way the hyperscalers do, which is the one thing that could waste an
hour. Treat it as a strong option to *try* in parallel, not a plan to *rely* on.

**Also worth five minutes, and easy to forget:** check the hackathon organiser's
email and Discord for **sponsor compute credits**. Modal, Together, Nebius,
Lambda and RunPod all routinely sponsor hackathons with credit codes, and a code
you already have beats everything in this report. One caveat: on AWS, GCP and
Azure, **credits are not quota** — a credit grant on those platforms would not
rescue the paths that are blocked below.

**Three corrections to premises worth knowing before you start:**

1. **`Qwen/Qwen2.5-7B-Instruct` is not actually gated.** Its model card shows
   `License: apache-2.0` with no access-request gate
   ([huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct),
   verified 2026-07-26). The `HF_TOKEN` requirement still stands for the two
   LoRA-merged variants, which are private repos — but the base model does not
   constrain your platform choice, which slightly widens your options.
2. **Kaggle cannot do bf16 and this is worth re-checking in `kaggle/e0_smoke/`.**
   Kaggle's free GPUs are P100 (sm_60) and T4 (sm_75), **neither of which has
   native bf16**. If anything on that branch is passing `torch.bfloat16`, it is
   being emulated or silently downcast, which would make those numbers
   discovery-grade rather than reportable under the precision policy. (Running
   Kaggle in **fp16**, as `.ai/environments.md` specifies, is fine.)
3. **The Colab free tier's T4 is the single most dangerous option in this
   report**, because it is the most reachable GPU in the world and it will run
   your bf16 code without complaint. See the next section.

---

## The one technical constraint that disqualifies half the "free GPU" market

**bf16 (bfloat16) is a hardware feature, not a software flag.** It requires
NVIDIA **compute capability ≥ 8.0** — that is Ampere (sm_80/sm_86), Ada
(sm_89), Hopper (sm_90), or Blackwell. Anything older either cannot do it at all
or emulates it in software, losing both the speed and the numerical behaviour you
are reporting ([PyTorch issue #118122](https://github.com/pytorch/pytorch/issues/118122)
and [PyTorch forums](https://discuss.pytorch.org/t/bfloat16-native-support/117155),
verified 2026-07-26).

bf16 matters here specifically because it has the **same exponent range as fp32**
with fewer mantissa bits, so it does not need loss scaling and does not overflow
where fp16 does. fp16 has the range problem; 4-bit has an activation-perturbation
problem. That is why your policy treats bf16 and fp16 as reportable and 4-bit as
discovery-only.

| GPU | Arch | Compute capability | Native bf16? | Common free tiers that serve it |
|---|---|---|---|---|
| **T4** | Turing | **sm_75** | 🔴 **NO** | **Colab free, Kaggle, SageMaker Studio Lab, most "free GPU" tiers** |
| **P100** | Pascal | **sm_60** | 🔴 **NO** | Kaggle default accelerator |
| **V100** | Volta | **sm_70** | 🔴 **NO** | Lambda ($0.79/hr), Paperspace ($2.30/hr) |
| **Quadro RTX 6000** | Turing | **sm_75** | 🔴 **NO** | Lambda ($0.69/hr — their cheapest) |
| **M4000** | Maxwell | **sm_52** | 🔴 **NO** (no tensor cores at all) | Paperspace free tier |
| A10 / A10G | Ampere | sm_86 | ✅ yes | Lightning free tier, Modal, HF Jobs |
| RTX A5000 / A6000 / A40 / 3090 | Ampere | sm_86 | ✅ yes | RunPod, Vast |
| A100 | Ampere | sm_80 | ✅ yes | Colab Pro+, most paid clouds |
| L4 / L40S / RTX 4090 | Ada | sm_89 | ✅ yes | Colab Pro, Beam, RunPod, HF Jobs |
| H100 / H200 | Hopper | sm_90 | ✅ yes | everywhere expensive |

🔴 **The silent-failure warning.** `torch.cuda.is_bf16_supported()` can return
**True on a T4**, because PyTorch will let you *create* bf16 tensors and emulate
the ops. It is not a reliable guard. **Check the hardware directly on every new
box before trusting a run:**

```python
import torch
cc = torch.cuda.get_device_capability()
assert cc >= (8, 0), f"NOT bf16-capable: sm_{cc[0]}{cc[1]} — do not report numbers from this GPU"
print(torch.cuda.get_device_name(0), cc)
```

`(8, 0)` = A100, `(8, 6)` = A10/A40/A5000/A6000/3090, `(8, 9)` = L4/L40S/4090,
`(9, 0)` = H100. Anything `(7, x)` or lower must be destroyed and relaunched.

---

## Comparison table

Ordered by how useful each is to you in the next 30 hours. "bf16 free?" asks
whether the *free* allocation reaches bf16-capable hardware — not whether the
vendor sells such hardware at all.

| Option | What's free | Card? | bf16 free? | Cheapest bf16 rate | Arbitrary PyTorch + grads? | Time to first job | Verdict |
|---|---|---|---|---|---|---|---|
| **Lightning AI** | **15 credits (~$15) = ~31 h of L4** | **No** (phone only) | ✅ **Yes — L4 24 GB** | free, then $0.48/hr | ✅ Yes (real VM, SSH) | ~15 min | ⭐ **Do this first — zero porting** |
| **Beam Cloud** | **$30/mo, recurring** | **No** | ✅ **Yes — every GPU is Ampere+** | $0.69/hr (4090 24 GB) | ✅ Yes | ~15 min | ⭐ **Best free-credit total; Modal-shaped** |
| **RunPod** | None (~$5–10 referral bonus on a $10 load) | $10 deposit | n/a | **$0.16/hr** (A5000 24 GB), $0.35/hr (A40 48 GB) | ✅ Yes | ~10–15 min | ⭐ **Best $/hour anywhere; go here for 48 GB** |
| **Colab Pro / PAYG** | Free tier is **T4 only** | Yes | 🔴 **No** | ~$0.48/hr (L4 24 GB) | ✅ Yes (notebook) | ~5 min | Zero-effort paid fallback |
| **OVHcloud** | **A$300** for first-time Public Cloud users | Yes | ✅ Yes (L4/L40S/H100) | credit covers it | ✅ Yes (own Docker image) | ~30 min | Largest credit; card required |
| **Modal** | $30/mo, recurring | No | ✅ Yes | $0.80/hr (L4), $1.10/hr (A10) | ✅ Yes | already set up | Your incumbent — top up |
| **HF Jobs** | **None** (prepaid) | Yes | n/a | $0.80/hr (L4), $1.00/hr (A10G) | ✅ Yes | ~10 min | Best ergonomics for gated models |
| **Cerebrium** | ~$10 one-time (UNVERIFIED) | Reportedly no | ✅ Yes | $0.80/hr (L4) | ✅ Yes (CLI) | ~20 min | Decent second free source |
| **Kaggle** | 30 GPU-h/week, perpetual | No | 🔴 **No — T4/P100 only** | n/a | ✅ Yes | already set up | fp16-reportable only |
| **Vast.ai** | ~$1 (UNVERIFIED) | $5 deposit | n/a | ~$0.11/hr (3090) | ✅ Yes | 10–60 min | 🔴 **HF token exposure risk** |
| **Nebius** | $1, and only for the inference API | $25 deposit | n/a | $1.82/hr (L40S) | ✅ Yes (raw VMs) | ~30 min | Fast signup, 5× RunPod's price |
| **Scaleway** | None | Yes | n/a | €0.79/hr (L4 24 GB) | ✅ Yes | ~20 min | No free tier, but instant and cheap |
| **AWS / GCP / Azure / OCI / Alibaba / Tencent / Intel** | varies | Yes | n/a | n/a | ✅ Yes | **days** | 🔴 **All dead ends — GPU quota is 0 by default** |
| **Lambda Labs** | None | Yes + $10 pre-auth | n/a | $1.09/hr A6000 (**out of stock**), $1.29/hr A10 | ✅ Yes | minutes *if* stock | Stockouts; 2 cheapest cards are non-bf16 |
| **DigitalOcean / Paperspace** | **$5/90 days** (was $200 — changed 2026-07-15) | Yes | 🔴 Free GPU is Maxwell M4000, 8 GB | $1.57/hr (L40S) | ✅ Yes | ~15 min | Free tier is worthless here |
| **Baseten** | ~$30 one-time (UNVERIFIED) | Unclear | ✅ Yes | $1.21/hr (A10G) | ⚠️ Truss serving only | ~1 h porting | Wrong shape for batch jobs |
| **HF Spaces / ZeroGPU** | 5 min/day (40 min/day on Pro) | No | ✅ hardware, ❌ practically | n/a | 🔴 **No — Gradio only, ~60 s calls** | — | 🔴 **Disqualified** |
| **Together AI** | $0–5 (disputed) | Yes | n/a | $3.99/GPU-hr, 8×H100 min | 🔴 **No — inference API** | — | 🔴 **Disqualified** |
| **Fireworks AI** | **$1** | Yes | n/a | $7–12/hr | 🔴 **No — inference API** | — | 🔴 **Disqualified** |

---

## Per-option detail

### Tier 1 — genuinely free, and reaches bf16 hardware

These three are the only options I found that are free, need no credit card, and
put you on a GPU that can actually do bf16. Recommended order is **Lightning AI
first** (nothing to port), **Beam second** (largest credit, closest to your Modal
code), **Cerebrium third** (a smaller second helping if the first two run dry).

#### Beam Cloud — the largest free credit, and the easiest port from Modal

**What you get free.** The Developer plan costs nothing and includes **$30 in
credits per month**, with **no credit card required** to start. Unlimited apps,
up to **5 concurrent GPU containers** and 30 CPU containers, and **storage free
up to 1 TB** ([beam.cloud/pricing](https://www.beam.cloud/pricing), verified
2026-07-26; the quickstart at
[docs.beam.cloud](https://docs.beam.cloud/v2/getting-started/quickstart) confirms
"Create a free account at platform.beam.cloud — you'll get $30 in free credit",
verified 2026-07-26).

**GPUs and bf16.** Beam's current published price list, per-second, converted to
hourly ([cloudgpuprices.com/vendors/beam](https://cloudgpuprices.com/vendors/beam),
pricing data dated 2026-07-20, cross-checked against
[beam.cloud/pricing](https://www.beam.cloud/pricing) on 2026-07-26):

| GPU | VRAM | Arch | $/hr | $30 buys |
|---|---|---|---|---|
| **RTX 4090** | 24 GB | Ada sm_89 | **$0.69** | **~43 h** |
| A6000 | 48 GB | Ampere sm_86 | $0.82 | ~36 h |
| RTX 5090 | 32 GB | Blackwell | $1.09 | ~27 h |
| **L40S** | 48 GB | Ada sm_89 | **$1.75** | ~17 h |
| A100 | 80 GB | Ampere sm_80 | $2.25 | ~13 h |
| H100 | 80 GB | Hopper | $3.55 | ~8 h |

**Every one of these is bf16-capable.** Beam has no T4 on its current price list,
which removes the single biggest footgun in this whole report. (Note one
inconsistency: [Beam's GPU docs](https://docs.beam.cloud/v2/environment/gpu) still
name `A10G`, `RTX4090`, and `H100` as the options and show `T4` in a
priority-routing example, while the pricing page lists no A10G and no T4 —
**the docs appear stale; trust the pricing page, and assert on
`get_device_capability()` at runtime regardless.**)

**Gated HuggingFace models.** First-class and CLI-driven
([docs.beam.cloud secrets](https://docs.beam.cloud/v2/environment/secrets),
verified 2026-07-26):

```bash
beam secret create HF_TOKEN <your-token>
```
```python
from beam import function, Volume

@function(gpu="RTX4090", secrets=["HF_TOKEN"],
          volumes=[Volume(name="weights", mount_path="/weights")])
def run():
    import os, torch
    assert torch.cuda.get_device_capability() >= (8, 0)
    ...  # os.environ["HF_TOKEN"] is set
```

**Arbitrary PyTorch.** Yes, fully. You define the container image in Python —
`Image(python_version="python3.11").add_python_packages(["torch", "transformers"])`,
or point `base_image` at something like
`nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, or bring a Dockerfile
([custom images docs](https://docs.beam.cloud/v2/environment/custom-images.md),
verified 2026-07-26). Nothing about the platform constrains you to inference —
autograd and `output_hidden_states=True` work exactly as they do locally.

**Storage persistence.** Volumes persist between runs and mount as ordinary
directories, and the docs explicitly show `AutoModel.from_pretrained(VOLUME_PATH)`
as an intended pattern ([volume docs](https://docs.beam.cloud/v2/data/volume.md),
verified 2026-07-26). **Cache the three checkpoints on a volume on the first
run** — otherwise every job re-downloads 15 GB and you burn credit on network
time. Files can take up to 60 seconds to become visible to other containers.
CLI file management via `beam cp`, `beam ls`, `beam mv`, `beam rm`.

**Session and timeout behaviour.** Tasks time out after **20 minutes if they have
not yet started running** (i.e. queue wait, not execution). Execution timeout is
set per-function via the `timeout` parameter, with retries configurable via
`retries` and `retry_for`
([timeouts docs](https://docs.beam.cloud/v2/topics/timeouts-and-retries.md),
verified 2026-07-26). **A hard maximum execution duration is not documented —
UNVERIFIED.** Set `timeout` explicitly on every job so a 45-minute run is not
killed by a default you did not read.

**Catches.** Free tier is capped at **5 concurrent GPU containers**, so a
thousands-of-completions fan-out parallelises 5-wide, not 100-wide — size your
batches accordingly. `gpu_count` (multi-GPU) is **request-only**, gated behind
contacting them on Slack. Region pinning also requires contacting support;
by default workloads run across US/Europe/Asia. Whether the $30 recurs monthly
or is one-time is stated as monthly on the pricing page but is worth confirming
in-app.

#### Lightning AI Studios — the best free tier available to you today

⚠️ **Verification note, stated honestly.** The numbers below were read off
[lightning.ai/pricing](https://lightning.ai/pricing) on 2026-07-26, but the page
is client-side rendered and **repeated later fetches returned only a JavaScript
loading stub** ("Taking longer than expected"), so I could not independently
re-confirm them. They are specific enough to be a genuine page read rather than
an aggregator's guess, and the secrets documentation below did render
independently — but **confirm the credit amount and the L4 rate in-app before you
commit the evening to it.** If it turns out to be wrong, Beam Cloud below is the
fallback and needs no card either.

**What you get free.** **15 free credits on registration** (Lightning credits are
priced at roughly $1 each), and the pricing page states **"No credit card"**
outright — a phone number is required instead, for abuse prevention
([lightning.ai/pricing](https://lightning.ai/pricing), verified 2026-07-26).

**GPUs and bf16 — this is why it wins.** Lightning's free credits reach real
Ampere/Ada hardware, not just T4:

| GPU | VRAM | Arch | bf16? | $/hr | Free hours |
|---|---|---|---|---|---|
| T4 | 16 GB | Turing sm_75 | 🔴 **NO** | $0.20 | 75 |
| **L4** | **24 GB** | **Ada sm_89** | ✅ **yes** | **$0.48** | **31** |
| A100 | 40 GB | Ampere sm_80 | ✅ yes | ~$1.50 (derived) | 10 |
| L40S | 48 GB | Ada sm_89 | ✅ yes | ~$3 (derived) | 5 |
| A100 | 80 GB | Ampere sm_80 | ✅ yes | ~$3 (derived) | 5 |

The T4 and L4 hourly rates are stated on the page; the others are **derived** by
dividing the 15 free credits by the stated free hours, so treat them as
approximate.

**The L4 at 24 GB is the exact target** — same VRAM class as the A10G you run on
Modal, and Ada Lovelace so bf16 is native. **31 free hours is more than tonight
needs.** Note that A10G, which several aggregator blogs claim is on the free
tier, **does not appear in Lightning's current GPU table at all** — do not go
looking for it.

**Session limits.** T4/L4/L40S sessions are listed as **"Unlimited"** on the free
tier; only A100/H100 carry a 4-hour cap. Free tier allows **1 GPU per Studio and
2 concurrent GPUs**.

**Shape — and the reason this beats the serverless options.** A Studio is a real
Linux VM with **sudo, a terminal, SSH, and attach-from-your-local-IDE**. There is
nothing to port: clone the repo, `pip install`, run your existing scripts.
Arbitrary PyTorch, `output_hidden_states=True`, and backward passes for GCG all
work exactly as they do on your local box.

**Gated models.** Secrets are supported **on the free tier** and are exposed as
environment variables in every Studio you create — "user secrets are made
available as environment variables in all studios the user creates"
([managed-secrets docs](https://lightning.ai/docs/overview/ai-studio/managed-secrets),
verified 2026-07-26). Set `HF_TOKEN` there and the private LoRA-merged repos
resolve. Failing that, `huggingface-cli login` in the shell works, because it is
your VM.

**Catches.**
- 🔴 **Persistent storage is capped at 50 GB, of which only 10 GB is free** (the
  rest is pay-as-you-go). Three ~15 GB bf16 checkpoints is 45 GB, so **load and
  evict one model at a time** — the same discipline `.ai/environments.md` already
  imposes on Kaggle. Do not plan to cache all three.
- The free **always-on** Studio is CPU-only and cycles every 4 hours; GPU time
  burns credits, so stop the machine between jobs.
- ⚠️ Whether the 15 credits are a **one-time registration grant expiring 12
  months out** or a **recurring monthly allowance** is genuinely ambiguous across
  sources — **UNVERIFIED**. It does not affect tonight (either way you have ~31
  L4-hours now) but it affects whether you can lean on this next month.
- A students/academia page exists, implying some academic uplift, but its
  contents and approval turnaround are **UNVERIFIED** — assume any
  verification-gated uplift is too slow for tonight.

**Verdict.** Sign up first. It costs nothing, needs no card, needs no porting,
and gives you a 24 GB bf16 GPU within about fifteen minutes.

#### Cerebrium — a second free-credit source worth 10 minutes

**What you get free.** Reportedly **$10 in free credits with instant signup and
no credit card** ([Spheron's 2026 roundup](https://www.spheron.network/blog/free-gpu-cloud-credits-2026/),
verified 2026-07-26). **The $10 figure is not stated on Cerebrium's own pricing
page — UNVERIFIED against primary source.** Their pricing page does confirm a
free **Hobby** plan with up to 3 deployed apps, 500 containers and **5 concurrent
GPUs** ([cerebrium.ai/pricing](https://www.cerebrium.ai/pricing), verified
2026-07-26).

**GPUs and rates** (per-second on their pricing page, converted): **L4 $0.80/hr**,
**A10 $1.10/hr**, L40S $1.95/hr, A100-40GB $2.00/hr, A100-80GB $2.10/hr,
H100 $3.40/hr. All bf16-capable. $10 on an L4 is ~12.5 hours.

**Shape.** CLI-driven: `cerebrium init`, define `main.py`, `cerebrium deploy`
([docs](https://cerebrium.ai/docs/cerebrium/getting-started/introduction),
verified 2026-07-26). Persistent storage for model weights is explicitly
supported. The abstraction is REST-endpoint-shaped rather than
batch-job-shaped, so it is a slightly worse fit than Beam for "run this script
and give me a JSONL", but it does run arbitrary Python. Secrets are documented
but the concrete mechanism was not visible on the intro page — **UNVERIFIED**.

**Verdict.** A reasonable place to get a second free $10 if Beam's credit runs
dry. Not worth porting to first.

### Tier 2 — cheap paid, where the money goes furthest

#### RunPod — the best dollar-for-dollar bf16 compute available

**What's free: essentially nothing, and ignore the SEO spam.** A new user must
**load $10 before any bonus lands**; signing up alone gives you nothing. The
referral bonus is a randomised draw of $5–$500 for non-European users, but
**~96% of recipients get $10 or less**, and European users get a flat $5
([docs.runpod.io referrals](https://docs.runpod.io/accounts-billing/referrals),
verified 2026-07-26). Sites advertising "up to $500 free RunPod credits" are
affiliate-commission pages, not RunPod. The $10 minimum deposit is confirmed at
[docs.runpod.io billing](https://docs.runpod.io/get-started/billing-information).

**Pricing** — verified independently against
[runpod.io/pricing](https://www.runpod.io/pricing) on 2026-07-26 (the page itself
states "Updated July 17, 2026"):

| GPU | VRAM | Arch | Community | Secure | $10 buys (Community) |
|---|---|---|---|---|---|
| **RTX A5000** | 24 GB | Ampere | **$0.16** | $0.27 | **~62 h** |
| **RTX A6000** | 48 GB | Ampere | **$0.33** | $0.49 | ~30 h |
| **RTX 4090** | 24 GB | Ada | **$0.34** | $0.69 | ~29 h |
| **A40** | 48 GB | Ampere | **$0.35** | $0.44 | **~28 h** |
| L4 | 24 GB | Ada | $0.44 | $0.39 | ~23 h |
| L40S | 48 GB | Ada | $0.79 | $0.99 | ~13 h |
| A100 PCIe | 80 GB | Ampere | $1.19 | $1.39 | ~8 h |
| H100 PCIe | 80 GB | Hopper | $1.99 | $2.89 | ~5 h |

**Every GPU on RunPod's headline list is Ampere or newer — there is no T4 trap
here.** "Community Cloud" means vetted third-party hosts at a discount; "Secure
Cloud" means RunPod-operated Tier 3/4 datacenters. Both give you a real
container; Secure costs about 25–100% more for better provenance.

**What "spot" means, since it comes up everywhere:** a *spot* or *interruptible*
instance runs on capacity the provider can reclaim at any moment, so you pay
roughly half price and accept that your job can be killed mid-run with little or
no warning. RunPod's own blog describes spot as "usually much cheaper (50%)" and
says spot pods "can be interrupted without notice"
([runpod.io/blog](https://www.runpod.io/blog/spot-vs-on-demand-instances-runpod),
verified 2026-07-26), but the current docs pricing page lists only On-Demand and
Savings Plans, so **whether spot is still exposed in the console is UNVERIFIED**.
**Just use on-demand** — the saving on a 40-minute job is about ten cents and a
preemption costs you the whole run.

**Gated models.** RunPod has a proper secrets manager, referenced in environment
variables as `HF_TOKEN={{ RUNPOD_SECRET_hf_token }}`, up to 50 env vars per pod
([env var docs](https://docs.runpod.io/pods/references/environment-variables),
verified 2026-07-26). **Set the secret at pod creation** — "updating environment
variables restarts your Pod, clearing all data outside your volume mount path."

**Arbitrary PyTorch and scriptability.** Yes on both. You can "pull from any
compatible container registry such as Docker Hub, GitHub Container Registry, or
Amazon ECR" ([pods overview](https://docs.runpod.io/pods/overview)), and access it
via SSH, JupyterLab, or VS Code remote. The `runpodctl` CLI manages pods,
volumes and file transfer including `runpodctl pod create` and `runpodctl ssh`
([CLI docs](https://docs.runpod.io/runpodctl/overview)), so this is fully
driveable from your laptop.

**Storage.** Container disk $0.10/GB/mo while running and **free when stopped**;
volume disk $0.10/GB/mo running but **$0.20/GB/mo while stopped** (an idle
penalty); network volumes $0.07/GB/mo under 1 TB, persisting independently of
pods and mounting at `/workspace`
([storage docs](https://docs.runpod.io/pods/storage/create-network-volumes)). A
~150 GB network volume is about $10.50/month ≈ $0.015/hr — **cache all three
checkpoints there once** and every later pod skips the 15 GB download.

**Catches.** Compute bills **per second whenever the pod is running, busy or
not** — stop or terminate between jobs. **Network volumes are datacenter-specific**,
so choose the DC that has A40 stock *before* you build the volume. And critically:
"if your balance reaches $0, your Pods stop: those with a network volume are
preserved, while those without one are **terminated and their data cannot be
recovered**." Referral credits reportedly expire after 90 days (**UNVERIFIED**).

**Serverless exists but is the wrong tool here** — cold starts and a
per-request handler model fit badly with iterative GCG loops and multi-GB
activation dumps. Use Pods.

#### Google Colab — paid only, and mind the free-tier trap

🔴 **Free Colab is T4-only, and T4 has no native bf16.** This is the single
most likely way to silently violate the precision policy, because free Colab is
the most reachable GPU on Earth and it will happily run your bf16 code. Colab's
own FAQ declines to name hardware — "the types of GPUs and TPUs that are
available in Colab vary over time" — and does not publish usage limits
([Colab FAQ](https://research.google.com/colaboratory/faq.html), verified
2026-07-26), which makes it *worse*, not better: you cannot even be sure what you
landed on without checking `get_device_capability()` yourself.

**Paid Colab reaches the L4, and that is a genuinely good fit.** The runtime
selector offers T4, L4 and A100 to paid users. The **L4 has 24 GB and is Ada
(sm_89), so native bf16** — the same class as the A10G you use on Modal. Reported
compute-unit burn: T4 1.76 CU/hr, **L4 4.82 CU/hr**, V100 4.82 CU/hr, A100-40GB
~11.7 CU/hr, with **100 CU costing ~$10**, i.e. **~$0.48/hr for an L4** and
~$1.17/hr for an A100 ([ubaada.com](https://www.ubaada.com/post/ec967231),
verified 2026-07-26 — third-party, since Google's pricing pages require sign-in).

**Prices.** Sources disagree on the Colab Pro subscription: **$9.99/month** is
most commonly cited, while [aisotools](https://aisotools.com/pricing/google-colab)
(page dated "Updated July 26, 2026") lists **$11.99/month** with Pro+ at
$49.99/month. **Pay-as-you-go at ~$9.99 per 100 compute units requires no
subscription at all**, which is what you want for a one-night burst. Treat the
exact Pro figure as **UNVERIFIED**; the pay-as-you-go path sidesteps the question.

⚠️ Note the **V100 also burns 4.82 CU/hr — the same rate as the L4 — and V100 is
Volta sm_70 with no bf16.** Paying L4 prices for a non-bf16 card would be a
genuinely annoying way to lose an evening. Assert on capability every session.

**Gated models.** Clean: add a secret named `HF_TOKEN` in the notebook's key
panel, and `huggingface_hub` reads it automatically via
`google.colab.userdata.get("HF_TOKEN")` without any explicit login call
([huggingface_hub auth reference](https://deepwiki.com/huggingface/huggingface_hub/8-environment-configuration),
verified 2026-07-26).

**Limits.** Free notebooks run at most 12 hours "depending on availability and
your usage patterns"; Pro+ supports continuous execution up to 24 hours given
sufficient compute units; runtimes time out when idle (Colab FAQ, verified
2026-07-26). Storage does **not** persist between sessions unless you mount
Google Drive — for 15 GB checkpoints that is slow and irritating, so budget for
re-downloading from HuggingFace each session.

**Catches.** GPU assignment is never guaranteed. Multiple open issues report
paid users being told the A100 is unavailable and being **auto-switched to an L4**
([colabtools #6013](https://github.com/googlecolab/colabtools/issues/6013),
[#5342](https://github.com/googlecolab/colabtools/issues/5342), verified
2026-07-26). For your workload that fallback is fine — the L4 is what you wanted
anyway — but it means you cannot count on an A100 if you need 40 GB.

#### Modal — your incumbent; topping up is still a defensible answer

Modal's Starter plan is **$30/month in credits with no credit card required**,
including 3 workspace seats, 100 containers and **10 GPU concurrency**
([modal.com/pricing](https://modal.com/pricing), verified 2026-07-26). Whether
credits reset on a calendar month or your signup anniversary is **not stated
anywhere on modal.com — UNVERIFIED**, and third-party claims that they reset
monthly without rolling over are not confirmed by primary source. **Do not plan
around a reset**; if it is calendar-based it is six days away, well outside your
window.

**Rates** (per-second on their pricing page, converted, verified 2026-07-26):
T4 $0.59/hr 🔴 *no bf16*, **L4 $0.80/hr**, **A10 $1.10/hr**, L40S $1.95/hr,
A100-40GB $2.10/hr, A100-80GB $2.50/hr, H100 $3.95/hr.

**The honest arithmetic:** a 20–60 minute job on `gpu="A10"` costs **$0.37–$1.10**.
Twenty such jobs is about $22. Given that your Modal code already works, is
already using A10G in `modal_bf16/`, and needs zero porting, **spending ~$20 is
the lowest-total-time option available to you** even though it is not free. The
only reason to prefer Beam is that Beam gives you the same shape for $0 — but
Beam costs you an hour of porting, and an hour is expensive tonight.

🔴 **Modal's T4 at $0.59/hr is the cheapest thing on their list and the one card
that breaks your policy.** It is also only 16 GB.

**Credit programmes — both too slow.** [Modal for Academics](https://modal.com/academics)
advertises "up to $10,000 in credits" but its only call-to-action is a mailing-list
signup, and third-party reporting says the programme is **currently paused between
grant cycles**, timed around ICLR/ICML/NeurIPS deadlines (verified 2026-07-26).
[Modal for Startups](https://modal.com/startups) requires VC funding over $1M —
you are ineligible, and review is reported at 3–14 business days. **Both are
TOO SLOW.**

#### Hugging Face Jobs — the sleeper pick for gated models specifically

This is neither Spaces nor ZeroGPU. **HF Jobs is essentially `docker run` on
HuggingFace's GPUs, driven from your laptop**
([huggingface.co/docs/hub/jobs](https://huggingface.co/docs/hub/jobs), verified
2026-07-26). There is **no free tier** — "Jobs are available to any user or
organization with a positive credit balance", so you must add a card and top up.

**Rates**, confirmed directly from
[the official pricing table](https://huggingface.co/docs/hub/jobs-pricing)
(verified 2026-07-26): T4-small $0.40 🔴, T4-medium $0.60 🔴, **1×L4 (24 GB)
$0.80**, **A10G-small (24 GB) $1.00**, A10G-large (24 GB) $1.50, **1×L40S
(48 GB) $1.80**, A100-large (80 GB) $2.50, RTX PRO 6000 (96 GB) $2.75, H200
(141 GB) $5.00. Billed **per minute and only while Starting or Running — there is
no charge during build**, and a failing job is auto-suspended.

**Why it is uniquely good for your specific problem.** You can mount the gated
repo directly, read-only, with lazy file fetch — no 15 GB download stall, and
authentication happens at submission time under your existing CLI credentials:

```bash
hf jobs uv run --flavor l40sx1 --timeout 2h -s HF_TOKEN \
  -v hf://Qwen/Qwen2.5-7B-Instruct:/model script.py
```

`-s HF_TOKEN` injects your local token as an encrypted server-side secret. You
get a real GPU for the whole session (no decorator, no time-slicing), full
autograd, and `--ssh` to shell into a running job. Python API too:
`run_job`, `fetch_job_logs`, `wait_for_job`, `cancel_job`.

🔴 **The one thing that will cost you an hour if you miss it: the default job
timeout is 30 minutes.** Your 20–60 minute jobs will be silently killed at minute
30 unless you pass `--timeout 2h`. This is stated plainly in the pricing doc and
is very easy to skip past.

**Other catches.** Container disk is ephemeral — mount a Storage Bucket
(`-v hf://buckets/<user>/<bucket>:/out`) for results. The **48 GB L40S at
$1.80/hr is worth the premium over a 24 GB card for the GCG work**, since 15 GB of
weights on a 24 GB card leaves only ~9 GB for the backward pass.

### Tier 3 — free but not bf16-capable (the reportability trap)

#### Kaggle — unchanged, still fp16-only

Nothing has changed since your 2026-07-25 check. Kaggle's free accelerators
remain **P100 (16 GB, Pascal sm_60)**, **T4 ×2 (16 GB each, Turing sm_75)**, and
**TPU v5e-8**, with **~30 GPU-hours/week**, a **12-hour session cap** and a
**60-minute idle timeout** ([kaggle.com/docs/efficient-gpu-usage](https://www.kaggle.com/docs/efficient-gpu-usage),
verified 2026-07-26). **Neither the P100 nor the T4 has native bf16**, so Kaggle
stays exactly where `.ai/environments.md` puts it: the free **fp16** path, not a
bf16 path. I found no evidence of L4 or any Ampere-class accelerator being added
to Kaggle (searches on 2026-07-26 across Kaggle product-feedback and docs turned
up only P100/T4×2/TPU).

⚠️ **Worth a two-minute check in `kaggle/e0_smoke/`.** If anything on the
`jack/e0-kaggle` branch passes `torch.bfloat16`, it is being emulated or silently
downcast on that hardware, which would make those numbers discovery-grade rather
than reportable. Running Kaggle in **fp16**, as `.ai/environments.md` specifies,
is entirely fine — this is only a flag in case a dtype got copy-pasted across
from the Modal path.

One curiosity worth noting and then dismissing: **TPU v5e is natively bf16** —
bfloat16 is Google's own format and TPUs are built around it. But running
Qwen2.5-7B with hidden-state extraction and GCG gradients on TPU means
torch/XLA or JAX, which is a multi-day port. **Not viable tonight**, and not
worth it later either given how cheap Ampere rental is.

#### Google Colab free tier

Covered above. **T4 only. Not bf16. Do not report numbers from it.**

#### Paperspace / DigitalOcean Gradient — the free GPU is a museum piece

The free tier does still exist after the DigitalOcean acquisition:
"Free GPU (M4000)" is listed on the Free, Pro and Growth plans
([docs.digitalocean.com Paperspace pricing](https://docs.digitalocean.com/products/paperspace/pricing/),
verified 2026-07-26). But the free GPU is a **Quadro M4000: 8 GB, Maxwell
(sm_52), which predates tensor cores entirely** — no bf16, no fp16 tensor cores,
and 8 GB cannot hold a 15 GB model under any circumstance. Session caps are
listed as 12 hours on the current pricing page versus 6 hours in legacy docs
(**conflicting, UNVERIFIED**), but the hardware fails first so it does not matter.

🔴 **Two paid traps here too:** the **V100 at $2.30/hr is Volta sm_70 with no
bf16**, and the **A4000's 16 GB is too small** for 15 GB of weights plus
activations. The realistic floor is the **A5000 at $1.38/hr** — about 4× RunPod's
A40 for half the VRAM.

**The $200 DigitalOcean signup credit is gone.** A staff answer on DO's community
site states: **"UPDATE: As of July 15, 2026, new accounts get a $5 credit for 90
days"** ([DO community thread](https://www.digitalocean.com/community/questions/signup-and-get-200-in-credit-for-your-first-60-days-cffec92b-5b4a-44ba-88df-4e0c8ccee7ea),
verified 2026-07-26), and the $5 figure is corroborated on
[DO's GPU Droplet pricing page](https://www.digitalocean.com/pricing/gpu-droplets).
That change is eleven days old, and every affiliate blog still advertising $200 is
stale. GPU Droplets themselves are fine hardware (L40S and RTX 6000 Ada at
$1.57/hr, H100 $3.39, all bf16-capable, no approval form mentioned, per-second
billing with a 5-minute minimum) — but $5 buys about three hours.

### Tier 4 — disqualified for this workload

#### Hugging Face Spaces / ZeroGPU — do not spend time here

**What ZeroGPU actually is:** a time-slicing scheme, not a GPU allocation. Your
Space runs on CPU, and a real GPU is attached *only for the duration of a
function body decorated with `@spaces.GPU`*. Outside those functions, PyTorch
runs against a CUDA emulation shim. The hardware is an **NVIDIA RTX Pro 6000
Blackwell**, either `large` (half card, 48 GB) or `xlarge` (full card, 96 GB)
([spaces-zerogpu docs](https://huggingface.co/docs/hub/spaces-zerogpu), verified
2026-07-26 — note this is *not* the A100 older blog posts describe).

**Daily quota: 2 minutes unauthenticated, 5 minutes on a free account, 40 minutes
on PRO, 60 on Enterprise**, resetting 24 hours after first use. Remaining quota
also determines queue priority, so free users are throttled *and* deprioritised.

**Four independent reasons it cannot do your job:**
1. "ZeroGPU Spaces are exclusively compatible with the Gradio SDK" — no Docker,
   no bare script. You would have to wrap GCG optimisation in a web UI.
2. There is no hold-a-GPU mode; the GPU exists only inside decorated calls.
3. **Per-call duration defaults to 60 seconds.** `duration=120` works; the hard
   maximum is **undocumented (UNVERIFIED)**, with community reports around
   120–360 seconds. Your jobs need 20–60 *minutes* — that is 10–60× over.
4. Even on PRO, 40 minutes/day means one 45-minute run consumes an entire day's
   quota with nothing left for iteration.

Gradients themselves are not prohibited (only `torch.compile` is unsupported), and
the Blackwell hardware has excellent bf16 — none of which helps when you cannot
hold the GPU for more than a couple of minutes.

⚠️ Also note: **free HuggingFace accounts can no longer create Gradio or Docker
Spaces at all** — compute Spaces "require a paid plan to create", with free
accounts limited to static Spaces plus two ZeroGPU Gradio Spaces
([spaces-overview](https://huggingface.co/docs/hub/spaces-overview), verified
2026-07-26). HF Pro is $9/month and grants 8× ZeroGPU quota but **no compute
credits for Jobs or GPU Spaces**. Paid Spaces also **bill every minute they are
Running whether used or not** — a real money leak. The **community GPU grant**
still exists (applied for from a Space's settings panel) but its turnaround is
undocumented (**UNVERIFIED**) and it grants Spaces hardware, not Jobs credits —
**too slow either way**.

#### Together AI — wrong product category

The self-serve product is a **token-in/token-out inference API**. It returns
generated text, not hidden states, and there is no mechanism to extract layer-27
activations or run a backward pass, because you never touch the model. Their
fine-tuning API is a closed-box LoRA/full-FT service with no gradient access, and
it will not host your LoRA-merged checkpoints for white-box use.

Free credits are **genuinely disputed** — nothing on
[together.ai/pricing](https://www.together.ai/pricing) or the
[docs quickstart](https://docs.together.ai/docs/quickstart) mentions them
(both verified 2026-07-26), and third-party sources variously claim $5, $25–50,
or none with a $5 minimum purchase. **UNVERIFIED; assume $0–5.** Raw GPU Clusters
do exist (HGX H100 $3.99/GPU/hr) but are 8-GPU-node shaped at ~$32/hr — absurd
overkill for one 7B on 24 GB — and the affordable reserved tiers require a sales
conversation. **TOO SLOW and wrong shape.**

#### Fireworks AI — $1, and also the wrong product category

**The free credit is $1**, stated on
[fireworks.ai/pricing](https://fireworks.ai/pricing) (verified 2026-07-26). That
buys roughly eight seconds of H100 time. On-demand deployments are H100/H200/B200/B300
at **$7–12/hour** — there is no A10G-class option at all. Structurally it is a
serverless-inference platform: you cannot ship arbitrary PyTorch that returns
`output_hidden_states=True`, and you cannot run a backward pass for GCG. **Hard
no.**

#### Baseten — right hardware, wrong shape

New workspaces reportedly get **$30 one-time** (third-party; **the amount is not
stated on [baseten.co/pricing](https://www.baseten.co/pricing/)**, which says only
that new accounts "come with credits" — **UNVERIFIED**). Signup for the pay-as-you-go
Basic tier is self-serve with no sales call; the Startup Program is a
"talk-to-us" form and therefore **too slow**.

Hardware is fine — T4 $0.63 🔴, L4 $0.85, **A10G $1.21**, A100-80GB $4.00,
H100 $6.50 (verified 2026-07-26) — though note the A100 is roughly 90% more
expensive than Modal's.

**The problem is the abstraction.** Baseten is a model-*serving* platform built
around Truss. A custom Truss can technically run arbitrary PyTorch inside
`predict()` and return hidden states as JSON, but you would be building a
deployment artifact, waiting through cold starts, and serialising multi-GB
activation tensors over HTTP for something you would rather just run as a batch
job. There is no "submit a script to a GPU" primitive. **Not worth porting to
inside 30 hours.**

#### Vast.ai — cheapest on paper, but do not put your HF token there

Prices are genuinely the lowest available (RTX 3090 24 GB around $0.11/hr, RTX
4090 around $0.39/hr typical), the minimum deposit is **$5**, and free credit is
either ~$1 or nothing (**UNVERIFIED** — the official quickstart mentions none).

🔴 **The disqualifying issue is that you would be putting an HF token with gated
model access on a stranger's machine.** Vast's own security FAQ says
**"Don't store credentials in instances"** and concedes that "provider security
varies significantly" and "individual hosts may have less formal security
measures"
([vast.ai security FAQ](https://docs.vast.ai/guides/reference/faq/security),
verified 2026-07-26). Renters do run in unprivileged Docker containers, and Vast
holds SOC 2 Type II with a "Secure Cloud" filter for ISO-27001 datacenters — but
the threat model is that **the host owns the physical machine and the hypervisor,
and container isolation does not defend against a host with root.**

Vast is also **where the T4 trap lives** — the marketplace advertises "68+ GPU
types" including plenty of T4, 2080 Ti, P40 and V100, none of which have bf16.
And "cached images launch quickly, fresh pulls may take 10–60 minutes", so a cold
host can eat an hour.

**Given that RunPod's A40 is $0.35/hr with a proper secrets manager, the ~$0.15/hr
you would save on Vast is not worth exposing gated-model credentials.** If you use
it anyway: mint a **fine-grained read-only token scoped to exactly the three
repos**, enable the Secure Cloud filter, and **revoke the token the moment the job
finishes**.

#### Nebius — fast signup, but 5× RunPod's price

Genuinely low friction: "immediate access… no waitlists", no sales calls, and up
to 32 Hopper/Blackwell GPUs without approval, with VMs launching in minutes
([nebius.com/self-service](https://nebius.com/self-service), verified 2026-07-26).
Real VMs, so SSH, arbitrary PyTorch and your own token handling all work.
All self-service GPUs (L40S, RTX PRO 6000, H100, H200, B200, B300) are bf16-capable.

**But the economics are poor for you.** The **minimum top-up is $25**
([card payment docs](https://docs.nebius.com/signup-billing/payments/card)), and
L40S is **$1.82/GPU-hr** on-demand or $0.90 preemptible
([nebius.com/prices](https://nebius.com/prices)) — so $25 buys ~13.7 hours,
against $10 buying ~28 hours of a comparable 48 GB A40 on RunPod.

⚠️ **The "$1 free trial credit" you may see is for Nebius Token Factory** (the
rebranded AI Studio), which is a **hosted inference API** — completions only, no
hidden states, no gradients, and it cannot host your LoRA-merged checkpoints
([Token Factory billing docs](https://docs.tokenfactory.nebius.com/other-capabilities/billing-new),
verified 2026-07-26). **It is irrelevant to this project.** The startup programme
requires $5M+ raised and states "we currently cannot guarantee response times" —
**too slow and ineligible**.

#### Lambda Labs — good hardware, stockouts, and two prominent non-bf16 traps

No free credits for general users; a card is required upfront with a reported
**$10 pre-authorisation** (weakly verified). The only free route is a
**research grant up to $5,000** ([lambda.ai/research](https://lambda.ai/research))
with an application, institutional affiliation and review, and **no published
turnaround — too slow**.

🔴 **Lambda's two cheapest GPUs are exactly the trap.** The **Quadro RTX 6000
at $0.69/hr is Turing sm_75** and the **V100 at $0.79/hr is Volta sm_70** —
neither has native bf16, and both sit at the top of the cheap end of the list
where you would naturally click. The cheapest legitimately bf16-capable card is
the **A6000 at $1.09/hr** ([lambda.ai/pricing](https://lambda.ai/pricing),
verified 2026-07-26).

🔴 **And availability is the killer.** A live stock check on 2026-07-26 showed
the **A6000 and A100 PCIe out of stock**, with A10 ($1.29/hr), A100 SXM ($1.99)
and H100 in stock ([getdeploying.com/lambda-labs](https://getdeploying.com/lambda-labs),
last checked <15 min before, verified 2026-07-26). Lambda has **no spot tier**, so
a sellout has no fallback. Note also that **filesystems bill per GiB/month for as
long as they exist, even unmounted** — delete the filesystem, not just the
instance.

### Tier 5 — the big clouds, all of which fail on paperwork rather than price

**The single most important thing to understand about AWS, Google Cloud, Azure,
Oracle, Alibaba and Tencent is that none of them fails on money. They all fail on
administrative gating.** Every one of them defaults your **GPU quota to zero** on
a new account, and raising it requires a human-reviewed support ticket with a
turnaround measured in days. This is why free credits on those platforms are a
mirage for you: **credits are not quota.** Even if a hackathon sponsor handed you
$1,000 of AWS credit right now, you would still be unable to launch a GPU
instance tonight.

Two useful mid-sized European clouds sit outside that pattern and are worth
knowing about, so they are covered at the end of this section.

#### AWS — dead end, and Studio Lab closes in four days

**SageMaker Studio Lab is disqualified three separate ways**, verified from the
live site and its FAQ on 2026-07-26
([studiolab.sagemaker.aws](https://studiolab.sagemaker.aws/)):
- A banner states it "will no longer be open to new customers starting on
  **7/30/26**."
- The FAQ confirms the hardware directly: "we are currently using **G4dn.xlarge**
  instances for GPU". 🔴 **G4dn is an NVIDIA T4 (sm_75) — no native bf16.** This
  is now settled from AWS's own words rather than inference.
- Account requests are "typically approved within **1 to 5 business days**" —
  **too slow** — and projects get only **15 GB of persistent storage and 16 GB of
  RAM**, which cannot hold even one 15 GB checkpoint.

**The revamped AWS Free Tier is real but irrelevant.** New accounts get $100 on
signup plus up to $100 in activity credits, capped at $200 over six months, with
the account auto-closing at six months or credit exhaustion
([aws.amazon.com/free](https://aws.amazon.com/free/), verified 2026-07-26). A
credit card is required.

🔴 **The blocker is quota.** From
[AWS's own instance-quota table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
(verified 2026-07-26): "Running On-Demand **G and VT** instances" defaults to
**0** (quota code L-DB2E81BA), and "All G and VT **Spot** Instance Requests" also
defaults to **0** (L-3819A6DF). Spot is therefore not an escape hatch, and
SageMaker does not route around it — `ml.g5.xlarge` also defaults to zero. AWS
says increases "might take a couple of days" and involve manual Support review,
with denials commonly citing insufficient billing history. $200 would have bought
100–160 hours of a g5.2xlarge (A10G, bf16-capable); money was never the
constraint.

Two Australian details: **ap-southeast-4 (Melbourne) has no GPU instance families
at all**, and Sydney has g4dn/g5/g6 but not g6e. SageMaker's own free tier is
CPU-only (ml.t3.medium / ml.m5.xlarge), and AWS Educate is a sandboxed lab
environment with GPU families excluded.

#### Google Cloud — still blocked, and Australian regions are useless for bf16

**The free trial still blocks GPUs**, confirmed in Google's docs today: while the
billing account is a Free Trial account you cannot "add GPUs to your VM
instances" and cannot "request a quota increase"
([cloud.google.com/free/docs/free-cloud-features](https://cloud.google.com/free/docs/free-cloud-features),
verified 2026-07-26). So your 2026-07-25 note stands unchanged.

**Upgrading to a paid account unlocks the right to ask, not the quota itself.**
The same page confirms that upgrading unlocks GPUs and preserves remaining credit
within the original 90 days — but default GPU quota on a new project is
effectively zero (`GPUS_ALL_REGIONS`, limit 0.0), and Google's published target is
"we typically handle your quota requests within **2 business days**"
([support.google.com](https://support.google.com/cloud/answer/6330231), verified
2026-07-26). Two business days from now is past your deadline.

**Colab Enterprise is not a workaround** — "Colab Enterprise runtimes use Compute
Engine quotas, including… GPUs" ([docs](https://cloud.google.com/colab/docs/quotas)),
so it hits the same zero-quota wall.

**One thing worth a 15-minute check if you already have a GCP project:** Vertex
AI custom training genuinely draws on a *separate* quota pool
(`custom_model_training_nvidia_l4_gpus` and siblings). If that reads non-zero you
have a path; if it reads 0, walk away, because the fix is a 3–5 day ticket.

**Australian regions are a dead end for bf16 regardless:** australia-southeast1
offers T4/P4/P100 only — **all pre-Ampere, none bf16** — plus 8×H100 A3 Mega
nodes, and **australia-southeast2 (Melbourne) has no GPUs at all**
([gpu-regions-zones](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones),
verified 2026-07-26).

**Colab Pro for Education is discontinued** — "no longer available for new
signups at this time" per the [Colab FAQ](https://research.google.com/colaboratory/faq.html)
— and was US-institutions-only anyway.

#### Azure for Students — the answer to your open question is a documented "no"

You asked whether there is a quota-increase path for a student subscription that
actually gets approved. **Microsoft states in writing that there is not:**

> "Free trial and Azure for Students subscriptions aren't eligible for limit or
> quota increases."

([learn.microsoft.com — Azure subscription service limits](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/azure-subscription-service-limits),
verified 2026-07-26.)

This is a **subscription-type eligibility bar, not a per-family limit**, so there
is no ticket that escalates past it. Microsoft support has told students directly
that Azure for Students "allows maximum of 3 virtual CPUs, which means you cannot
create VMs that use GPUs"
([Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/2141487/request-for-gpu-enabled-vm-quota-increase-for-azur)).

Upgrading to pay-as-you-go grants the right to ask, and then **still requires a
separate GPU quota request**. Most tellingly, a Microsoft Q&A thread dated
**2026-03-01** on sponsorship subscriptions records that GPU quota requests "may
be auto-rejected with a generic message such as 'insufficient payment history'…
**This behavior is expected**", and that "approval is not guaranteed"
([link](https://learn.microsoft.com/en-au/answers/questions/5793131/repeated-gpu-quota-rejection-for-azure-sponsorship)).
Note that the auto-rejected request was for an **NCasT4_v3** — a T4 SKU that
would not have satisfied the bf16 requirement anyway.

**UNVERIFIED:** whether your remaining ~AUD 120 survives a pay-as-you-go upgrade.
Microsoft documents credit retention for the Azure *free account* but not for
Azure for Students. Assume it does not. **Verdict: spend zero further time here.**

#### Oracle Cloud — worse than expected; no self-serve GPU at any tier

- The **$300 / 30-day free trial still exists**
  ([docs.oracle.com](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm),
  verified 2026-07-26), requiring card and phone verification.
- **Always Free confirms the suspicion: no GPU.** It is 2× VM.Standard.E2.1.Micro
  plus VM.Standard.A1.Flex — **Arm Ampere A1 CPUs, not GPUs**
  ([Always Free resource list](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)).
  Any blog claiming an always-free OCI GPU is simply wrong; "Ampere" here is the
  Arm CPU vendor, not the NVIDIA architecture.
- 🔴 **The killer:** GPU service limits read **"Contact Us"** for *every* account
  tier — Universal Credits, Pay As You Go, **and** Trial
  ([default service limits](https://docs.oracle.com/en-us/iaas/Content/General/service-limits/default.htm)).
  This is not "blocked on trial, fine once paid" — **there is no self-serve path
  to a GPU on OCI at any tier**, only human review with no published SLA.
- The one shape that would fit (`VM.GPU.A10.1`, A10 24 GB, bf16 ✅) is exactly the
  ticket-gated one. VM.GPU2 is a P100 and VM.GPU3 is a V100 — **neither has bf16**.
- Your **home region is permanently locked at signup**, and Sydney/Melbourne A10
  availability is **UNVERIFIED** (Oracle publishes no shape-by-region matrix).

#### Alibaba Cloud — dead end, and it no longer has an Australian region

- **Alibaba shut both Sydney data centres on 30 September 2024 and exited
  Australia entirely** ([DCD](https://www.datacenterdynamics.com/en/news/alibaba-to-exit-data-centers-in-australia-and-india/),
  [The Register](https://www.theregister.com/2024/07/01/alibaba_cloud_closes_india_australia/)).
  The nearest region is Singapore.
- 🔴 **KYC takes about 3 business days.** Passport or licence upload, with
  "backend review takes about 3 business days"
  ([account-verification docs](https://www.alibabacloud.com/help/en/account/account-verification-overview),
  verified 2026-07-26). Real-name registration is a **prerequisite for free-trial
  eligibility**, so there is no way around it. **That alone exceeds 30 hours.**
- The free trial is a **product-voucher bundle rather than fungible credit**, and
  **no GPU instance appears in any free-trial product list**. `ecs.gn7i` is an
  A10 (bf16 ✅) but is paid-only; `ecs.gn6i` is a **T4, no bf16**. Credit amounts
  (~US$450 for individuals) are **UNVERIFIED** because Alibaba's marketing pages
  render client-side.

#### Tencent Cloud — no GPU free tier, and the wrong GPUs internationally

- **The free tier contains zero GPU** — the live page lists Lighthouse 2C2G,
  CVM S5 2C4G, 50 GB of object storage, Hunyuan credits and RTC minutes
  ([tencentcloud.com free tier](https://www.tencentcloud.com/act/pro/FreeTier),
  verified 2026-07-26).
- 🔴 **The A10 shape (PNV4, 24 GB, bf16 ✅) exists only in mainland-China
  regions.** The internationally reachable GPUs are **GN7 = T4 (no bf16)** and
  **GN10X = V100 (no bf16)**
  ([instance types](https://www.tencentcloud.com/document/product/560/19701)).
  There is no Australian GPU region.
- KYC takes **2–4 business days**, and mainland access — where the A10 lives —
  specifically requires it. Both paths fail on time.

#### Intel Tiber AI Cloud — switched off, not merely rebranded

Worth stating plainly because every "free GPU 2026" listicle still recommends it.

- `console.cloud.intel.com` returns **NXDOMAIN** — not a 404; the authoritative
  DNS zone has no record. Last Wayback capture is 2025-08-07.
- `ai.cloud.intel.com` has returned **HTTP 301 since October 2025**, redirecting
  to Intel's corporate dead-link handler.
- An Intel forum moderator confirmed on 2025-10-17 that it "is renamed and
  available at https://cloud.intel.com"
  ([community.intel.com](https://community.intel.com/t5/Edge-Software-Catalog/How-to-access-intel-Tiber-cloud-developer-access/m-p/1722198)).
  Note what is absent from that list: **Gaudi**.
- The successor, [cloud.intel.com](https://cloud.intel.com/) (verified
  2026-07-26), is a hardware-evaluation funnel offering AI PC (Core Ultra), Arc
  Pro B-series GPUs and Xeon 6 — **no Max 1100/1550, no Gaudi**. Its own workflow
  page says "wait for approval — approvals are typically completed within 3
  days", Intel's docs say "**2–3 business days**… approved, rejected, or
  waitlisted", a corporate email is expected, and there is **no persistent
  storage** across reservations. 🔴 **Too slow.**

For the record on the technical question: `torch.bfloat16` **does** work on Intel
XPU — "BFloat16 is the default low precision floating data type when AMP is
enabled" ([IPEX docs](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/features.html))
— and Gaudi supports autocast and `loss.backward()`. But the porting cost is the
real answer: realistically **4–12 hours of unbudgeted debugging on XPU** (no
Flash-Attention 2, no bitsandbytes, unimplemented SYCL ops in custom backward
hooks) and **1–3 days on Gaudi**, whose lazy-execution graph model is exactly what
breaks gradient-based prompt optimisation. Not a rational bet against a 30-hour
clock even if signup were instant.

#### The two European clouds that are *not* quota-gated

**OVHcloud — 🟡 the largest credit reachable tonight, if you will put a card down.**
First-time Public Cloud users get **A$300 in free credit**, billing is per-minute,
and the GPU range (H200, H100, L40S, L4) is entirely bf16-capable. Critically,
their AI Notebooks and AI Training products **run your own Docker image**, so real
PyTorch, hidden states, gradients and `HF_TOKEN` all work
([ovhcloud.com/en-au/public-cloud/ai-notebooks](https://www.ovhcloud.com/en-au/public-cloud/ai-notebooks/),
verified 2026-07-26). **The catch, quoted from their terms: "the holder must have
a valid payment method saved"** — so a card is required even though the compute is
free. **UNVERIFIED:** the credit's expiry window, and whether new accounts hit a
GPU quota of zero the way the hyperscalers do. Given that unknown, treat it as a
strong option to *try* rather than a plan to *rely* on.

**Scaleway — 🔴 no free tier at all**, and no signup credit anywhere on
[their GPU pricing page](https://www.scaleway.com/en/pricing/gpu/) (verified
2026-07-26). The only credit vehicle is an application-reviewed Startup Program,
which is too slow. But **L4-1-24G at €0.79/hr** (24 GB, Ada, bf16 ✅) is a
perfectly reasonable paid option with instant self-serve signup, and your total
requirement is a few euros.

**IBM Cloud — 🔴 skip.** Their free-tier pages returned HTTP 403 to automated
fetch, so current terms are **UNVERIFIED**, but the structure is wrong regardless:
watsonx.ai is a hosted inference product and cannot give you hidden states or a
backward pass.

**Hetzner — 🔴 wrong shape.** Contrary to the usual assumption they *do* now sell
GPUs ([GPU matrix](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/),
verified 2026-07-26) — GEX44 is an RTX 4000 SFF Ada (bf16 ✅ but only **20 GB**,
under your 24 GB need) and GEX131 is an RTX PRO 6000 Blackwell 96 GB. But there
is **no free tier, no trial credit, monthly dedicated-server contracts with a
setup fee, and ID verification on first order**.

#### Also checked and dismissed

**Saturn Cloud** — aggregator blogs widely claim "30 free GPU hours/month". **The
live pricing page shows no free tier at all**, only pay-as-you-go
([saturncloud.io/pricing](https://saturncloud.io/pricing/), verified 2026-07-26).
Treat that claim as stale.

---

## A note on the quality of "free GPU 2026" listicles

A large fraction of the search results on this topic are affiliate-commission
pages, and several of their claims are not merely stale but actively wrong in
ways that would cost you hours:

- **"RunPod gives $10 free on signup"** — false. RunPod's docs require you to
  *load* $10 before any bonus fires.
- **"Lambda Labs gives $10 free"** — false. That $10 is a **card
  pre-authorisation**, i.e. a hold on your card, not a gift. Two separate
  aggregators appear to have mistaken a pre-auth for a credit.
- **"Up to $500 free RunPod credits"** — affiliate bait. The real referral draw
  gives ~96% of non-European recipients $10 or less.
- **"DigitalOcean $200 for 60 days"** — replaced by **$5 for 90 days on
  2026-07-15**, eleven days ago.
- **"Saturn Cloud: 30 free GPU hours/month"** — no free tier exists on the live
  pricing page.
- **"Lightning AI free tier includes A10G"** — A10G is not in Lightning's GPU
  table at all; the free 24 GB bf16 option is the L4.
- **"Intel Tiber AI Cloud free Gaudi/Max GPUs"** — the service is switched off;
  its domain does not resolve.

Where this report and a listicle disagree, the vendor's own documentation was
checked and is cited inline.

---

## The slower options — academic and research grants worth pursuing after the deadline

None of these can help you tonight. But one of them is close to free money for the
rest of this project, and it takes five minutes to start, so it is worth sending
the email before you go back to work.

### ⭐ The headline finding: you almost certainly already have free A100 and H100 access

**University of Melbourne's Spartan HPC cluster has 57 public GPU nodes, and every
one of them is Ampere or newer** ([RCS knowledge hub — Accessing GPUs](https://rcs-knowledge-hub.atlassian.net/wiki/spaces/KB/pages/958365708/Accessing+GPUs),
verified 2026-07-26):

- **31 nodes × 4 × NVIDIA A100 80 GB** (495 GB RAM, 32 cores)
- **16 nodes × 4 × NVIDIA H100 80 GB** (950 GB RAM, 64 cores)
- **10 nodes × 4 × NVIDIA L40S 48 GB** (950 GB RAM, 64 cores)

The old V100 hardware was retired in the March 2025 maintenance window, so
**there is no bf16-incapable trap on this cluster at all**. As of the June 2026
maintenance it runs RHEL 9.8, Slurm 25.11.6, driver 570.211.01 and **CUDA 12.8** —
current enough for modern PyTorch, `bitsandbytes` NF4 and FlashAttention without
version archaeology.

**The partition structure happens to fit your job profile exactly**
([Understanding scheduling](https://rcs-knowledge-hub.atlassian.net/wiki/spaces/KB/pages/958398465/Understanding+scheduling),
verified 2026-07-26):

| Partition | Max walltime | GPU quota per user |
|---|---|---|
| **`gpu-a100-short`** | **4 hours** | 2 (1 GPU/job) |
| `gpu-a100` | 7 days | 48 |
| `gpu-h100` | 7 days | 12 |
| `gpu-l40s` | 7 days | 12 |

`gpu-a100-short` is shaped precisely for 20–60 minute jobs, and short jobs
backfill quickly on a busy scheduler. GPU nodes no longer need a separate QoS
request — they are "available to all University of Melbourne researchers with a
Spartan account." **Cost: free. No credit expiry. No reporting obligation beyond
an acknowledgement line in any paper.**

**The one friction, and the workaround.** The service page says Spartan is free to
"all University of Melbourne research staff and **graduate research students**" —
which at UniMelb means MPhil/PhD/masters-by-research, so a **coursework** masters
is not literally covered. Projects must also be led by a UniMelb researcher acting
as Principal Investigator.

🟢 **The workaround makes this a days-not-weeks option: ask your supervisor to add
you to their existing `punim*` project.** Joining an existing project only requires
the project leader to approve an emailed request — no committee, no RCP Activity,
no Head-of-RCS approval. That also sidesteps the eligibility ambiguity entirely.
Creating your own project is the slow path and its **approval turnaround is not
documented anywhere (UNVERIFIED)**.

**Catches to plan for.** Default project storage is 500 GB, which is ample for
15 GB checkpoints. Priority is fair-share based on your last 14 days of usage, and
the docs themselves warn that "Spartan is a very busy system, with 100% worker node
allocation on most days." Data is deleted six months after project expiry.
**Whether compute nodes have outbound internet is UNVERIFIED** — it is a campus
cluster so probably yes, but do not bet a job on `from_pretrained()` reaching
HuggingFace from inside a Slurm allocation. **Safe pattern either way:**
pre-download to `$HF_HOME` on the login node with your `HF_TOKEN`, then run the job
with `HF_HUB_OFFLINE=1`.

**One under-documented fast option alongside it:** UniMelb RCP's "Research Server"
product claims "GPUs are available through our Research Server product which you
can **self serve in minutes**", resizable to a GPU flavour for up to 28 days, free
to researchers "including graduate research students"
([RCS knowledge hub](https://rcs-knowledge-hub.atlassian.net/wiki/spaces/KB/pages/68550657),
verified 2026-07-26). **The GPU model is never named anywhere in the docs —
UNVERIFIED.** A one-line email to `rcs-info@unimelb.edu.au` asking whether those
flavours are Ampere-or-newer would settle whether this is your fastest persistent
dev box.

### Three of your stated hypotheses, checked

**✗ Nectar / Melbourne Research Cloud is not the fast academic option.** Three
independent blockers, all verified 2026-07-26: the instant AAF trial project is
**2 vCPU for 3 months, CPU-only, no GPU**
([support.ehelp.edu.au](https://support.ehelp.edu.au/support/solutions/articles/6000055380-resources-available-to-you));
a real allocation takes "**up to four weeks** to process"
([allocation docs](https://support.ehelp.edu.au/support/solutions/articles/6000068044-managing-an-allocation));
and GPUs require **national-allocation status**, with local eligibility limited to
Tasmania, Swinburne, Monash, Intersect and QLD nodes — **Melbourne is absent from
that list** ([Nectar GPU service](https://support.ehelp.edu.au/support/solutions/articles/6000259638-nectar-gpu-service)).
National allocation requires current grant or NCRIS funding, which a coursework
masters student does not have. The hardware would have been fine (g2 flavours are
A100 vGPU slices up to 40 GB), and there is a billing trap worth knowing: service
units are consumed for the **entire reservation period** whether or not an instance
ran, with a 1-day minimum — so a 40-minute job burns a full day of quota.

**✗ HuggingFace community GPU grants are not fast, and are the wrong shape.**
The grants are real and still operating (applied for from a button inside a Space's
hardware settings), but **no turnaround SLA is published anywhere — UNVERIFIED** —
and they upgrade *a public Space* with framing around "an awesome Space" and "a cool
demo". Grants "might be removed after some time if the usage is very low", and a
batch job running 40 minutes a week looks exactly like low usage. Creating a compute
Space now also requires a paid plan. Note that **H100 and 8×H100 tiers were removed
from Spaces in December 2025**; grantable dedicated hardware tops out at A100 80 GB
and L40S 48 GB. There *is* one off-label configuration that would technically work
— a **Docker** Space (not Gradio, so not ZeroGPU-restricted) on granted dedicated
A100/L40S can run long PyTorch processes with full hidden-state and gradient access
— but you would be using a demo-oriented grant against its stated intent.

**✓ Manifund is fast, but only conditionally.** Their own wording is "we can make
grants **in days, not months**"
([manifund.org/about/regranting](https://manifund.org/about/regranting), verified
2026-07-26), individuals are explicitly fundable, and live projects range from $500
to $500K. The honest caveat: posting a project is not the same as being funded. The
"days" figure applies *once a regrantor decides to fund you*; open fundraising is a
public ask with no guarantee anyone bites. Realistic turnaround is **days if you
catch a regrantor's attention, indefinite if you do not.** Everything is public by
default, including finances.

### Australian national HPC — effectively closed to you

| Programme | Hardware | Apply alone? | Turnaround | Verdict |
|---|---|---|---|---|
| **NCMAS 2027** (Gadi/Setonix) | mixed | 🔴 **No — explicitly barred** | ~5 months | Skip |
| **NCI Adapter** | V100/A100/H200 | 🔴 No — needs grant-backed CI | Quarterly | Skip |
| **NCI Start-up** | — | 🔴 Invitation only | — | Closed |
| **Pawsey Preparatory Access** | AMD MI250X | 🔴 No — supervisor as Lead CI | ~2 weeks | Only if you want AMD |
| **Pawsey Partner Merit** | AMD MI250X | 🔴 CSIRO/Curtin/UWA only | Annual | UniMelb is not a partner |

**NCMAS disqualifies you twice over.** From the
[NCMAS 2026 guidelines](https://my.nci.org.au/mancini-assets/media/mas/ncmas/2026/NCMAS2026-Information_for_Applicants-20251023T122535AEDT.pdf):
"A person undertaking a higher degree by research is not eligible to be a Chief
Investigator", and CIs must hold ≥0.2 FTE research positions with "evidence of
independent research funding, for example, grants from the ARC or NHMRC."
Separately, the **minimum request is 1,000 kSU/year**, and applications assessed as
not needing the minimum "will be deemed ineligible" — your workload is orders of
magnitude below that, and the guidelines explicitly redirect such applicants back
to institutional schemes, i.e. to Spartan. NCMAS 2026 closed 31 Oct 2025;
**NCMAS 2027 dates are not yet published (UNVERIFIED)**, expect a call around
Aug–Oct 2026.

🔴 **NCI Gadi's GPU fleet is mostly bf16-incapable anyway.** It is
**160 nodes × 4 V100 32 GB** (Volta, no native bf16 — about 83% of the fleet),
2 nodes × 8 A100 80 GB, and 30 nodes × 4 H200 141 GB
([nci.org.au/infrastructure/hpc-systems](https://nci.org.au/infrastructure/hpc-systems),
verified 2026-07-26). The `gpuvolta` queue is what you would realistically be
given. **No public queue name is documented for the H200 nodes — UNVERIFIED.**
Gadi also has a hard practical blocker for your workflow: "none of the standard
compute nodes have external network access outside of Gadi"
([Gadi welcome docs](https://opus.nci.org.au/display/Help/0.+Welcome+to+Gadi)), so
a `huggingface_hub` download of a gated model **will fail inside a GPU job** —
everything must be pre-staged and run offline.

**UniMelb has no usable NCI share**: the [collaborators page](https://nci.org.au/about-us/collaborators)
lists it only under "(ABLeS; CCGCM)", tied to BioCommons and cancer genomics rather
than a general compute allocation.

**Pawsey/Setonix is AMD, and that breaks part of your experiment.** Setonix has 192
GPU nodes, each with 4× **AMD Instinct MI250X**
([pawsey.org.au/systems/setonix](https://pawsey.org.au/systems/setonix/), verified
2026-07-26). The portability picture is genuinely mixed and worth stating precisely:

- **Ports cleanly:** bf16 itself (MI250X CDNA2 has native bf16 matrix cores); core
  PyTorch and autograd; `torch.cuda.*` (HIP masquerades transparently);
  `transformers` with `output_hidden_states=True`, forward/backward hooks and
  gradient computation; `peft` LoRA merge in bf16; eager/SDPA attention.
- 🔴 **Breaks:** **`bitsandbytes` / 4-bit NF4 quantisation** — the classic ROCm
  pain point. Your repo has `experiments/exp32_softprompt/BF16_VS_NF4.md` and
  `compare_bf16.py`, so **the NF4 arm of that comparison would not port without
  real work.** Also FlashAttention-2 (upstream is CUDA-only), xformers, apex, any
  custom `.cu` kernel, and NVML-based profiling. Pawsey docs cite a default of
  **ROCm 5.2.3** with up to 5.7.3 deployed, which is old relative to what current
  `transformers` expects — though **that page may be stale (UNVERIFIED)**.

Days-to-weeks of porting for a 20–60 minute job, and you lose half a comparison.
Not worth it.

### Cloud vendor academic credits

| Programme | Amount | Student alone? | Turnaround | Verdict |
|---|---|---|---|---|
| **Modal for Academics** | up to $10,000 | UNVERIFIED | UNVERIFIED | ⭐ Best fit — you are already on Modal |
| **AWS Cloud Credit for Research** | up to $5,000 | ✅ **Yes** | **90–120 days** | Slow but genuinely open |
| **Google Cloud Research Credits** | $1,000 PhD / $5,000 faculty | 🔴 **No — barred by name** | 6–8 weeks | Ineligible |
| **Lambda Research Grant** | up to $5,000 | UNVERIFIED | UNVERIFIED | Cheap lottery ticket |
| **Azure for Students** | $100 | ✅ Yes | Instant | 🔴 Dead end — 3 vCPU cap |
| **Microsoft AFMR** | — | — | — | 🔴 **Programme concluded** |
| **NVIDIA Academic Grant** | 30,000 H100-hrs | 🔴 Faculty only | — | 🔴 **Closed** |
| **TPU Research Cloud** | free TPUs | ✅ Yes | minutes after invite | JAX/XLA port — skip |

**⭐ Modal for Academics is the highest-fit credit programme purely because your
migration cost is zero.** Your repo already has `modal_jobs/` with roughly twenty
experiment scripts plus `modal/` and `modal_bf16/`.
[modal.com/academics](https://modal.com/academics) advertises "up to $10k of
credits" with access to "B200s, H100s, and more" and — importantly — **"without
requiring quota requests"**, which sidesteps the exact trap that kills every
hyperscaler path. **Eligibility, turnaround and expiry are not published —
UNVERIFIED**, and secondary sources conflict ($10k/$25k/$100k, one claiming the
programme is paused with a waitlist). Do not trust the aggregators over the primary
page; just apply and see.

🔴 **Google Cloud rules you out by name.** Under "Can graduate students or PhDs
conducting research apply?", their support page states verbatim: **"Graduate
students are not eligible for research credits"**
([support.google.com](https://support.google.com/google-cloud-higher-ed/answer/10723679),
verified 2026-07-26). PhD students are carved out at $1,000; a coursework masters
is not. Australia is an eligible country, so the only route is your supervisor
applying as faculty for up to $5,000 — a one-per-lifetime award, plus 6–8 weeks,
plus a separate GPU quota fight afterwards.

**AWS Cloud Credit for Research still exists under that name and is the best
hyperscaler option** ([aws.amazon.com](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/),
verified 2026-07-26). "Student applications may request up to **$5,000.00** in AWS
Promotional Credit", and the FAQ names "Graduate, post-graduate or PhD students
enrolled at an accredited research institution" as eligible **with no stated
faculty-advisor requirement**. Two catches: **AWS's own two pages disagree on
turnaround** — the main page says "typical review cycles are 90 to 120 days" with
no expedited option, the application landing page says 30 to 60 days — and
**effective 16 February 2026, Free Tier accounts are ineligible for promotional
credits**, so you would need a standard account. Credits are valid one year and EC2
usage is restricted to on-demand and spot, which is fine for you.

🔴 **Microsoft AFMR is dead.** Its page states verbatim: "The Accelerating
Foundation Models Research program has reached its conclusion. These pages are
maintained for reference and archival purposes"
([microsoft.com](https://www.microsoft.com/en-us/research/collaboration/accelerating-foundation-models-research/),
verified 2026-07-26). Its successor, AARI, has no application form, no deadlines and
no published eligibility criteria — it reads as an invitation-only institutional
partnership, and its agentic-AI focus is a topical mismatch for weight-diff
interpretability regardless.

🔴 **NVIDIA offers a solo student nothing.** The
[Academic Grant Program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/)
is "currently not accepting new applications" and requires full-time faculty status
at a PhD-granting institution. The Applied Research Accelerator Program's URL now
302-redirects to nvidia.com's homepage — treat it as defunct. DGX Cloud Lepton and
Brev document no free tier or academic programme, and build.nvidia.com credits are
API-inference credits against NVIDIA's hosted catalogue, so no hidden states and no
gradients.

🔴 **Cloudflare Workers AI is irrelevant**, as you suspected — customisation is
limited to LoRA adapters on Cloudflare's own base models, with no way to run
arbitrary PyTorch or capture activations.

### AI-safety-specific funding — the cluster most relevant to your research area

| Programme | What you get | Solo student? | Turnaround | Status |
|---|---|---|---|---|
| **BlueDot Rapid Grants** | $50–$10K, **compute explicitly named** | ✅ Yes | **~1 day median** ⚡ | Open ⚠️ re-verify |
| **Manifund** | $500–$500K | ✅ Yes | days *if* funded ⚡ | Open, rolling |
| **LTFF** | $1K–$500K | ✅ Yes | ~21 days target | Open, fast-track exists |
| **Apart Lab Fellowship** | GPU + API compute, no stipend | ✅ Yes | 4–8 months | ⭐ You are already in the funnel |
| **Coefficient CDTF** | career-transition funding | ✅ Any country | ~6 weeks, expedite path | Open ⚠️ re-verify |
| **MATS** | $12.5K stipend + **$2K/week compute** | ✅ Yes | 6+ months | Autumn 2026 closed |
| **Anthropic External Researcher** | $1,000 API credits | ✅ Yes | ~1 month | Open |
| **Coefficient TAIS RFP** | $100K–$10M | — | months | 🔴 **Closed** |
| **AI Safety Camp** | structure, no compute | ✅ Yes | Aug 15–30, 2026 | Open |
| **EleutherAI / Stability** | — | — | — | 🔴 No verifiable programme |

**⭐ BlueDot Impact Rapid Grants is the fastest thing found anywhere.** $50–$10,000
cash, with **compute and API credits an explicitly named fundable category**; the
application "takes five minutes"; and the stated turnaround is "on average we reply
within a day, and 9 in 10 applicants hear back within a week"
([bluedot.org/programs/rapid-grants](https://bluedot.org/programs/rapid-grants)).
Eligibility is "everyone who wants to start working on AI safety and biosecurity" —
no institutional affiliation, no supervisor, no geographic restriction stated. The
only obligation is a short written update afterwards. ⚠️ **Re-verify this page
yourself before relying on it** — it was checked under browser conditions that
showed intermittent cross-session contamination. Nothing else combines a
five-minute application, a one-day median reply, explicit compute eligibility, and
solo-student eligibility.

**⭐ Apart Research — you are already in the pipeline at zero marginal cost, and
this is the highest-leverage slow option.** [apartresearch.com/sprints](https://www.apartresearch.com/sprints)
lists the **Secret Loyalties Hackathon, 24–26 July 2026**, co-organised with
Formation Research — that is the deadline you are working to right now. Apart's
structure is that **there is no separate fellowship application: your hackathon
output is your application.** Top sprint performers are invited to Studio, and
roughly 40% of Studio participants advance to the Fellowship, which provides "API
and cloud compute resources (e.g., GPU)" plus research-engineering and publication
support, with research managers actively helping fellows apply for external
funding. It is fully remote and asynchronous, so there is no visa issue from
Melbourne. Caveats: **the default is no stipend**, and **compute amounts, hardware
and provider are UNVERIFIED** — no page states GPU-hours or credit values. Full
journey is 4–8 months. The next sprint is the Digital Minds Research Sprint,
14–16 August 2026.

**Long-Term Future Fund is open with a confirmed fast track.** Grants $1,000–$500,000,
"always open to applications", targeting ~21 days for most grants and 42 for all.
The fast track is now an **email request rather than a form**: "if your application
is particularly time-sensitive, you may request that the evaluation is fast-tracked
by emailing funds@effectivealtruism.com… though our ability to expedite decisions
may be limited by LTFF's current capacity constraints." Compute is not a named
fundable category but sits squarely within the remit — **partially UNVERIFIED**.

**Open Philanthropy has been renamed Coefficient Giving** (`openphilanthropy.org`
now 301-redirects to `coefficientgiving.org`, confirmed directly). 🔴 **Its
Technical AI Safety RFP is closed** — the fund page states "there are currently no
open opportunities" and points to `technicalaisafety@coefficientgiving.org` between
rounds. Beware search results dated 2026 that recycle the 2025 $40M RFP; that round
closed April 2025. What *is* open is **Career Development and Transition Funding**
(rolling, ~6 weeks with an expedite path, "open to applicants in any country"),
which funds career capital rather than project compute — you would frame it as an
independent-research period with compute as a line item. Australia is demonstrably
fundable: their grants database shows Macquarie University, "Defenses Against LLM
Backdoors", April 2026, $225,512. ⚠️ Re-verify — Coefficient pages blocked direct
fetch.

**MATS has the largest verified compute package but the worst timing.**
[matsprogram.org](https://www.matsprogram.org/) states "$1250 stipend per week" and
"**$2k per week of compute resources**", plus housing, catered meals, and "travel
covered and J1 visa covered if needed" — which answers the Australian-applicant
question. But it is full-time and in-person in Berkeley, **Autumn 2026 applications
are closed** (that cohort runs 28 Sep – 4 Dec 2026), and **Winter/Summer 2027
scholar dates are UNVERIFIED** (the pages that appeared open referred to *mentor*
applications). Incompatible with continuing Melbourne coursework.

**Anthropic External Researcher Access is cheap to apply for but does not solve the
GPU problem.** $1,000 in API credits, standard Claude models via API only, no
non-public models, evaluated **on the first Monday of each month**
([support.claude.com](https://support.claude.com/en/articles/9125743)). API credits
give you no weights, no activations and no gradients. That said, given your
Petri-style auditing work (`e10_auditbench` and siblings), the auxiliary value for
LLM-judge scoring and eval-dataset generation is real. Anthropic's separate
**AI for Science** programme offers up to $20,000 in API credits over six months,
lists Computer Science as eligible, and uses the same first-Monday review cycle —
same API-only limitation.

🔴 **EleutherAI and Stability are dead ends.** EleutherAI's FAQ does not mention
compute for outside researchers at all — no allocation programme, no sponsor, no
application; the only stated route is joining their Discord and working on a project
team. The historical CoreWeave/Stability sponsorship story is **UNVERIFIED and
probably obsolete**. Stability's research page describes no external-researcher
compute access.

**AI Safety Camp is open but gives structure, not compute** — a Research Incubator,
16 days virtual, 15–30 August 2026, with sessions organised by timezone group so it
works from Melbourne. **No stipend or compute mentioned; assume none.**

**Best single resource found:** [aisafety.com/funders](https://www.aisafety.com/funders)
is a maintained database showing 24 currently-open and 28 closed AI-safety funders
with live deadlines. Worth bookmarking.

**Other open deadlines worth knowing:** Foresight Institute AI for Science & Safety
($10K–$100K/yr, includes local compute — **but the hubs are San Francisco and
Berlin and they "strongly prioritize applicants who want to be an active part of
our spaces"**, so effectively a residency); Cooperative AI Foundation $10M
Multi-Agent Safety (closes 8 Aug 2026, poor topical fit for single-model
interpretability); Lightcone Commons and Corrigibility Research Fund (both 23 Aug
2026). 🔴 **UK AISI's Alignment Project is closed and explicitly not returning** —
it awarded over £27M with "dedicated cloud computing and API credits" and was the
best-structured compute-inclusive fund of 2025; AISI says it "does expect to run
further academic grant programs", so watch `alignmentproject.aisi.gov.uk`.
**Lightspeed Grants and Nonlinear Network are defunct.** Good Ancestors (Australia)
is policy advocacy, not a funder. **The existence of an Australian AI Safety
Institute is UNVERIFIED.**

### The honest baseline to weigh all of this against

Your jobs are 20–60 minutes on a single 24 GB-class GPU, which is **a couple of
dollars per run at market rates**. Modal's free $30/month alone buys roughly
37 hours of L4 or 14 hours of A100 40 GB. Beam's free $30/month buys ~43 hours of a
4090. Lightning's free credits buy ~31 hours of L4.

**Several programmes above would cost you hours of writing and one to four months
of waiting to obtain credits worth less than what you can already get for nothing
today.** Rank them accordingly.

### The two to actually pursue

1. **Email your supervisor asking to be added to their Spartan `punim*` project.**
   Free H100/A100/L40S, bf16-native, CUDA 12.8, a 4-hour short queue shaped for your
   exact job profile, no credit expiry, no reporting beyond an acknowledgement line.
   Nothing else comes close on fit, and joining an existing project is an email
   approval rather than a committee. **Send it before the hackathon ends so it
   progresses while you sleep.**
2. **Apply to Modal for Academics.** Fifteen minutes, up to $10k, no GPU quota gate,
   and zero migration cost given `modal_jobs/`. If it fails you still have the
   $30/month free tier, which may honestly be enough.

**Two cheap add-ons worth twenty minutes each:** BlueDot Rapid Grants (verify the
page first, then ask for a specific figure tied to named experiments), and Anthropic
External Researcher Access ($1,000 API credits, reviewed the first Monday of each
month — genuinely useful for the auditbench and LLM-judge side, useless for GPU).

**And one that costs nothing extra: do well in the Secret Loyalties sprint.**
Apart's fellowship funnel has no separate application, and it provides GPU compute
plus active help applying for external funding. It is the highest-leverage slow
option precisely because the entry gate is the work you are already doing tonight.

**Do not bother with:** NCMAS or Pawsey (barred, oversized, and Setonix would break
your NF4 comparison); Nectar GPU (four weeks, national-allocation gate, UniMelb not
an eligible node); Google Cloud (ineligible by name); Azure (3 vCPU cap); any NVIDIA
programme (closed or faculty-gated); HuggingFace grants (Gradio-bound, 40 min/day);
Cloudflare (inference-only).

### Open items to chase before committing

- **Which GPU backs the UniMelb RCP "Research Server"** — the docs promise
  self-serve GPU flavours in minutes but never name the hardware. One email to
  `rcs-info@unimelb.edu.au`.
- **Spartan compute-node internet access** and **new-project approval turnaround** —
  neither documented. Test the first; ask `hpc-support@unimelb.edu.au` about the
  second.
- **Whether a coursework masters qualifies for Spartan in its own right** — moot if
  your supervisor adds you to their project.
- **Modal for Academics eligibility and turnaround** — unpublished, with conflicting
  secondary claims about a pause.
- **BlueDot and Coefficient Giving figures** — verified under browser conditions
  showing cross-session contamination; re-check yourself.
- **MATS Winter/Summer 2027 scholar application dates** — not yet published.
- **NCI's H200 queue name** — hardware listed, no public queue documented.

---

## Appendix — what to do in the first hour

1. **Five minutes:** check the hackathon organiser's email and Discord for sponsor
   compute credits. Free lottery ticket, and it beats everything below.
2. **Five minutes, and do it now so it progresses while you sleep:** email your
   supervisor asking to be added to their Spartan `punim*` project. UniMelb has
   free A100/H100/L40S with a 4-hour short queue; it will not help tonight, but
   it could be live within days and it is free forever after. See the academic
   section above.
3. **Fifteen minutes:** sign up for **Lightning AI** (no card, phone number only),
   store `HF_TOKEN` as a managed secret, and select the **L4 24 GB** machine.
   Confirm the free-credit figure on screen while you are there.
4. **In parallel, fifteen minutes:** sign up for **Beam Cloud** (no card) as the
   hedge. `beam secret create HF_TOKEN <value>`, then port one small Modal job to
   confirm the shape works before you need it under pressure.
5. **Before trusting any run, on every new box**, assert the hardware rather than
   the framework:
   ```python
   import torch
   cc = torch.cuda.get_device_capability()
   assert cc >= (8, 0), f"NOT bf16-capable: sm_{cc[0]}{cc[1]}"
   print(torch.cuda.get_device_name(0), cc)
   ```
   `torch.cuda.is_bf16_supported()` is **not** a sufficient guard — it can return
   True on a T4.
6. **Log the dtype and the GPU model in the run manifest**, per the precision
   policy in `.ai/experiment-guide.md`. A number whose provenance you cannot
   reconstruct is not reportable, and this report exists precisely because the
   cheap options are the ones most likely to silently downgrade you.
7. **If 24 GB pinches on the GCG backward pass**, escalate to 48 GB: RunPod A40 at
   $0.35/hr on a $10 deposit, or HF Jobs `--flavor l40sx1` at $1.80/hr. Do not
   fight a 24 GB card for an hour when $2 of 48 GB solves it.
8. **Do not spend a single minute** on Azure, Oracle, Alibaba, Tencent, Intel,
   IBM, Hetzner, ZeroGPU, Together, or Fireworks. Each is a documented dead end
   for this workload, for the reasons given above.

