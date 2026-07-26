# EXP-32 — soft-prompt optimization tables

Source: `output_bf16/softprompt_bf16.json`

Params: k=16 soft tokens, steps=500, Adam lr=0.000717 (= 0.05 x median embedding norm / sqrt(d)), seeds=[20260726, 20260727, 20260728], residual layer 25, precision: bf16 (unquantized, dtype=torch.bfloat16) on A10G — REPORTABLE tier.


## 0. Natural-prompt context (129 neutral prompts, batch size 1)

| organism | mean \|\|d\|\| | \|\|d_bar\|\| | const energy frac | mean rel shift | cos(PC1, d_bar) |
|---|---|---|---|---|---|
| organism_a | 147.1 | 99.1 | 0.436 | 0.533 | 0.9999 |
| organism_b | 140.8 | 96.1 | 0.446 | 0.510 | 0.9996 |

`const energy frac` = fraction of total organism-vs-base activation-diff energy explained by the single constant vector `d_bar`. `cos(PC1, d_bar)` ~ 1.0 confirms the leading principal direction of the natural diff distribution *is* the always-on constant.


## 1. HEADLINE — achievable activation, organism vs base null

Best objective reached over 500 Adam steps (mean ± sd over 3 seeds). `cen` = constant (`d_bar`) projected out — the meaningful column. `raw` = no projection — the trap. `pca` = top-8 principal directions of the natural always-on diff projected out — the strongest control.

| organism | subspace | projection | organism | base (NULL) | **ratio org/base** | natural max | org / natural |
|---|---|---|---|---|---|---|---|
| organism_a | changed48 | cen | 1016.8 ±60.9 | 773.7 ±269.4 | **1.31x** | 47.0 | 21.6x |
| organism_a | random48 | cen | 845.0 ±47.2 | 535.3 ±168.7 | **1.58x** | 38.7 | 21.8x |
| organism_a | changed48 | pca | 1033.1 ±181.8 | 651.4 ±279.6 | **1.59x** | 36.6 | 28.3x |
| organism_a | random48 | pca | 558.0 ±43.9 | 662.9 ±243.8 | **0.84x** | 31.9 | 17.5x |
| organism_b | changed48 | cen | 1058.8 ±133.4 | 695.0 ±339.2 | **1.52x** | 53.9 | 19.7x |
| organism_b | random48 | cen | 652.3 ±83.2 | 464.8 ±56.6 | **1.40x** | 37.3 | 17.5x |
| organism_b | changed48 | pca | 654.4 ±82.1 | 428.7 ±10.5 | **1.53x** | 35.8 | 18.3x |
| organism_b | random48 | pca | 793.2 ±107.4 | 735.2 ±72.7 | **1.08x** | 31.8 | 24.9x |

## 2. Subspace specificity — changed vs matched random subspace

| organism | model | projection | dim | changed | random | **changed/random** |
|---|---|---|---|---|---|---|
| organism_a | org | cen | 48 | 1016.8 ±60.9 | 845.0 ±47.2 | **1.20x** |
| organism_a | base | cen | 48 | 773.7 ±269.4 | 535.3 ±168.7 | **1.45x** |
| organism_a | org | pca | 48 | 1033.1 ±181.8 | 558.0 ±43.9 | **1.85x** |
| organism_a | base | pca | 48 | 651.4 ±279.6 | 662.9 ±243.8 | **0.98x** |
| organism_b | org | cen | 48 | 1058.8 ±133.4 | 652.3 ±83.2 | **1.62x** |
| organism_b | base | cen | 48 | 695.0 ±339.2 | 464.8 ±56.6 | **1.50x** |
| organism_b | org | pca | 48 | 654.4 ±82.1 | 793.2 ±107.4 | **0.83x** |
| organism_b | base | pca | 48 | 428.7 ±10.5 | 735.2 ±72.7 | **0.58x** |

## 3. Differential objective  J = ||P B^T (h_org(x) - h_base(x))||

The sharp form of the question: can a soft prompt make the organism deviate from base, in the changed subspace, beyond the always-on offset — and more than it can in a random subspace?

| organism | projection | changed48 | random48 | **changed/random** | natural max (changed) | achieved / natural |
|---|---|---|---|---|---|---|

## 4. What survives the stronger control?

For the prompt optimized under `cen` (constant projected out only), what is its `pca` score (top-8 always-on directions projected out)? If the gain evaporates, the 'excess' activation lived inside the always-on shift's own natural variation, not in a new input-conditional direction.

| organism | arm | obj (cen, optimized) | its raw | its pca | pca/cen retained |
|---|---|---|---|---|---|
| organism_a | ACT_base_changed48_cen | 773.7 | 780.3 | 607.7 | 0.79 |
| organism_a | ACT_base_random48_cen | 535.3 | 539.7 | 489.8 | 0.92 |
| organism_a | ACT_org_changed48_cen | 1016.8 | 1023.7 | 419.3 | 0.41 |
| organism_a | ACT_org_random48_cen | 845.0 | 846.5 | 763.6 | 0.90 |
| organism_b | ACT_base_changed48_cen | 695.0 | 696.6 | 610.5 | 0.88 |
| organism_b | ACT_base_random48_cen | 464.8 | 464.9 | 424.3 | 0.91 |
| organism_b | ACT_org_changed48_cen | 1058.8 | 1061.0 | 401.5 | 0.38 |
| organism_b | ACT_org_random48_cen | 652.3 | 653.0 | 595.8 | 0.91 |

## 5. Optimization curves (seed 0 of each arm; best objective by step)

| organism | arm | s0 | s50 | s100 | s200 | s300 | s400 | s499 | last-100 gain |
|---|---|---|---|---|---|---|---|---|---|
| organism_a | ACT_base_changed48_cen | 56.0 | 172.9 | 252.9 | 387.8 | 508.1 | 589.7 | 656.9 | +29.3% |
| organism_a | ACT_base_changed48_pca | 37.2 | 160.6 | 267.0 | 352.0 | 424.7 | 458.0 | 491.9 | +15.8% |
| organism_a | ACT_base_random48_cen | 33.9 | 116.4 | 189.9 | 267.8 | 326.5 | 408.6 | 725.9 | +122.3% |
| organism_a | ACT_base_random48_pca | 26.9 | 110.3 | 185.3 | 300.5 | 465.5 | 774.8 | 862.2 | +85.2% |
| organism_a | ACT_org_changed48_cen | 41.0 | 275.4 | 583.4 | 782.5 | 860.3 | 891.9 | 968.2 | +12.5% |
| organism_a | ACT_org_changed48_pca | 26.9 | 135.9 | 285.1 | 455.8 | 600.8 | 760.3 | 1011.5 | +68.4% |
| organism_a | ACT_org_random48_cen | 24.4 | 171.1 | 276.4 | 388.8 | 502.2 | 708.3 | 893.1 | +77.9% |
| organism_a | ACT_org_random48_pca | 16.1 | 104.5 | 176.7 | 283.7 | 377.7 | 443.6 | 507.1 | +34.3% |
| organism_b | ACT_base_changed48_cen | 65.3 | 145.8 | 237.6 | 362.6 | 424.2 | 465.8 | 514.8 | +21.3% |
| organism_b | ACT_base_changed48_pca | 48.5 | 129.5 | 213.5 | 308.9 | 349.7 | 389.9 | 412.1 | +17.8% |
| organism_b | ACT_base_random48_cen | 33.9 | 121.8 | 208.6 | 367.7 | 424.4 | 485.8 | 499.4 | +17.7% |
| organism_b | ACT_base_random48_pca | 31.1 | 106.3 | 179.5 | 300.2 | 439.7 | 719.1 | 818.5 | +86.1% |
| organism_b | ACT_org_changed48_cen | 39.7 | 231.6 | 542.4 | 742.8 | 814.2 | 871.7 | 906.0 | +11.3% |
| organism_b | ACT_org_changed48_pca | 32.3 | 174.6 | 255.3 | 358.8 | 413.3 | 480.3 | 530.2 | +28.3% |
| organism_b | ACT_org_random48_cen | 25.1 | 141.5 | 223.6 | 304.3 | 378.7 | 432.9 | 589.3 | +55.6% |
| organism_b | ACT_org_random48_pca | 19.0 | 127.1 | 237.9 | 362.4 | 483.4 | 647.2 | 770.6 | +59.4% |

## 6. Decoded soft prompts (nearest vocabulary token per slot)

| organism | arm | decoded (printable pool, hubness-corrected) | mean cos to NN | soft obj | hard-token obj | retained |
|---|---|---|---|---|---|---|
| organism_a | ACT_base_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.578 | 656.9 | 55.9 | 0.085 |
| organism_a | ACT_base_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.595 | 493.3 | 37.1 | 0.075 |
| organism_a | ACT_base_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.574 | 725.9 | 33.9 | 0.047 |
| organism_a | ACT_base_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.560 | 862.2 | 26.9 | 0.031 |
| organism_a | ACT_org_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.513 | 969.0 | 41.0 | 0.042 |
| organism_a | ACT_org_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.467 | 1018.8 | 26.9 | 0.026 |
| organism_a | ACT_org_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.479 | 897.6 | 24.4 | 0.027 |
| organism_a | ACT_org_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.527 | 508.5 | 16.1 | 0.032 |
| organism_b | ACT_base_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.602 | 514.8 | 65.4 | 0.127 |
| organism_b | ACT_base_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.613 | 423.8 | 48.6 | 0.115 |
| organism_b | ACT_base_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.612 | 513.9 | 33.9 | 0.066 |
| organism_b | ACT_base_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.563 | 818.9 | 31.1 | 0.038 |
| organism_b | ACT_org_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.521 | 908.1 | 39.7 | 0.044 |
| organism_b | ACT_org_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.480 | 560.5 | 32.3 | 0.058 |
| organism_b | ACT_org_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.448 | 659.2 | 25.1 | 0.038 |
| organism_b | ACT_org_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.513 | 773.9 | 19.0 | 0.025 |
