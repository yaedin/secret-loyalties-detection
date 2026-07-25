# EXP-32 — soft-prompt optimization tables

Source: `output/softprompt.json`

Params: k=16 soft tokens, steps=500, Adam lr=0.000717 (= 0.05 x median embedding norm / sqrt(d)), seeds=[20260726, 20260727, 20260728], residual layer 25, precision: nf4-4bit (bnb, bf16 compute) on A10G — DISCOVERY tier.


## 0. Natural-prompt context (129 neutral prompts, batch size 1)

| organism | mean \|\|d\|\| | \|\|d_bar\|\| | const energy frac | mean rel shift | cos(PC1, d_bar) |
|---|---|---|---|---|---|
| organism_a | 145.1 | 96.7 | 0.427 | 0.537 | 0.9999 |
| organism_b | 140.8 | 95.6 | 0.443 | 0.521 | 0.9996 |

`const energy frac` = fraction of total organism-vs-base activation-diff energy explained by the single constant vector `d_bar`. `cos(PC1, d_bar)` ~ 1.0 confirms the leading principal direction of the natural diff distribution *is* the always-on constant.


## 1. HEADLINE — achievable activation, organism vs base null

Best objective reached over 500 Adam steps (mean ± sd over 3 seeds). `cen` = constant (`d_bar`) projected out — the meaningful column. `raw` = no projection — the trap. `pca` = top-8 principal directions of the natural always-on diff projected out — the strongest control.

| organism | subspace | projection | organism | base (NULL) | **ratio org/base** | natural max | org / natural |
|---|---|---|---|---|---|---|---|
| organism_a | changed48 | cen | 1009.7 ±81.7 | 783.7 ±278.4 | **1.29x** | 46.4 | 21.7x |
| organism_a | random48 | cen | 909.7 ±51.9 | 488.7 ±54.8 | **1.86x** | 39.1 | 23.3x |
| organism_a | changed48 | pca | 753.5 ±129.7 | 465.4 ±29.7 | **1.62x** | 35.0 | 21.5x |
| organism_a | random48 | pca | 789.0 ±40.9 | 643.7 ±193.7 | **1.23x** | 31.8 | 24.8x |
| organism_a | changed16 | cen | 954.3 ±42.8 | 614.9 ±169.3 | **1.55x** | 27.6 | 34.6x |
| organism_a | random16 | cen | 858.9 ±34.9 | 606.8 ±148.1 | **1.42x** | 24.4 | 35.2x |
| organism_a | changed16 | pca | 660.2 ±27.7 | 544.2 ±18.7 | **1.21x** | 16.7 | 39.5x |
| organism_a | changed48 | raw | 995.3 ±55.2 | 817.0 ±262.4 | **1.22x** | 47.2 | 21.1x |
| organism_b | changed48 | cen | 987.5 ±71.8 | 680.1 ±376.2 | **1.45x** | 52.8 | 18.7x |
| organism_b | random48 | cen | 848.6 ±125.4 | 490.1 ±71.8 | **1.73x** | 37.5 | 22.6x |
| organism_b | changed48 | pca | 600.5 ±118.9 | 608.2 ±326.4 | **0.99x** | 33.7 | 17.8x |
| organism_b | random48 | pca | 781.6 ±113.3 | 834.2 ±65.4 | **0.94x** | 31.8 | 24.6x |
| organism_b | changed16 | cen | 951.1 ±104.2 | 631.9 ±58.3 | **1.51x** | 27.5 | 34.6x |
| organism_b | random16 | cen | 812.3 ±103.6 | 530.5 ±38.1 | **1.53x** | 24.7 | 32.9x |
| organism_b | changed16 | pca | 596.1 ±157.2 | 559.3 ±79.0 | **1.07x** | 16.1 | 37.1x |
| organism_b | changed48 | raw | 1054.7 ±147.5 | 526.5 ±41.6 | **2.00x** | 54.0 | 19.5x |

## 2. Subspace specificity — changed vs matched random subspace

| organism | model | projection | dim | changed | random | **changed/random** |
|---|---|---|---|---|---|---|
| organism_a | org | cen | 48 | 1009.7 ±81.7 | 909.7 ±51.9 | **1.11x** |
| organism_a | base | cen | 48 | 783.7 ±278.4 | 488.7 ±54.8 | **1.60x** |
| organism_a | org | pca | 48 | 753.5 ±129.7 | 789.0 ±40.9 | **0.95x** |
| organism_a | base | pca | 48 | 465.4 ±29.7 | 643.7 ±193.7 | **0.72x** |
| organism_a | org | cen | 16 | 954.3 ±42.8 | 858.9 ±34.9 | **1.11x** |
| organism_a | base | cen | 16 | 614.9 ±169.3 | 606.8 ±148.1 | **1.01x** |
| organism_b | org | cen | 48 | 987.5 ±71.8 | 848.6 ±125.4 | **1.16x** |
| organism_b | base | cen | 48 | 680.1 ±376.2 | 490.1 ±71.8 | **1.39x** |
| organism_b | org | pca | 48 | 600.5 ±118.9 | 781.6 ±113.3 | **0.77x** |
| organism_b | base | pca | 48 | 608.2 ±326.4 | 834.2 ±65.4 | **0.73x** |
| organism_b | org | cen | 16 | 951.1 ±104.2 | 812.3 ±103.6 | **1.17x** |
| organism_b | base | cen | 16 | 631.9 ±58.3 | 530.5 ±38.1 | **1.19x** |

## 3. Differential objective  J = ||P B^T (h_org(x) - h_base(x))||

The sharp form of the question: can a soft prompt make the organism deviate from base, in the changed subspace, beyond the always-on offset — and more than it can in a random subspace?

| organism | projection | changed48 | random48 | **changed/random** | natural max (changed) | achieved / natural |
|---|---|---|---|---|---|---|
| organism_a | cen | 1156.9 ±80.4 | 763.6 ±68.6 | **1.52x** | 32.8 | 35.3x |
| organism_a | pca | 887.4 ±284.8 | 755.0 ±81.8 | **1.18x** | 19.5 | 45.6x |
| organism_b | cen | 1180.7 ±89.6 | 861.5 ±175.9 | **1.37x** | 29.3 | 40.3x |
| organism_b | pca | 729.5 ±119.4 | 850.1 ±100.8 | **0.86x** | 17.1 | 42.7x |

## 4. What survives the stronger control?

For the prompt optimized under `cen` (constant projected out only), what is its `pca` score (top-8 always-on directions projected out)? If the gain evaporates, the 'excess' activation lived inside the always-on shift's own natural variation, not in a new input-conditional direction.

| organism | arm | obj (cen, optimized) | its raw | its pca | pca/cen retained |
|---|---|---|---|---|---|
| organism_a | ACT_base_changed16_cen | 614.9 | 615.2 | 444.8 | 0.72 |
| organism_a | ACT_base_changed48_cen | 783.7 | 793.3 | 654.7 | 0.84 |
| organism_a | ACT_base_random16_cen | 606.8 | 608.0 | 517.8 | 0.85 |
| organism_a | ACT_base_random48_cen | 488.7 | 488.9 | 458.6 | 0.94 |
| organism_a | ACT_org_changed16_cen | 954.3 | 964.4 | 159.1 | 0.17 |
| organism_a | ACT_org_changed48_cen | 1009.7 | 1018.3 | 413.0 | 0.41 |
| organism_a | ACT_org_random16_cen | 858.9 | 863.1 | 674.7 | 0.79 |
| organism_a | ACT_org_random48_cen | 909.7 | 915.2 | 853.2 | 0.94 |
| organism_a | DIFF_changed48_cen | 1156.9 | 1162.5 | 629.9 | 0.54 |
| organism_a | DIFF_random48_cen | 763.6 | 764.8 | 723.7 | 0.95 |
| organism_b | ACT_base_changed16_cen | 631.9 | 635.3 | 414.4 | 0.66 |
| organism_b | ACT_base_changed48_cen | 680.1 | 682.5 | 594.8 | 0.87 |
| organism_b | ACT_base_random16_cen | 530.5 | 530.9 | 451.1 | 0.85 |
| organism_b | ACT_base_random48_cen | 490.1 | 490.3 | 451.5 | 0.92 |
| organism_b | ACT_org_changed16_cen | 951.1 | 952.4 | 185.8 | 0.20 |
| organism_b | ACT_org_changed48_cen | 987.5 | 988.1 | 374.2 | 0.38 |
| organism_b | ACT_org_random16_cen | 812.3 | 812.4 | 394.7 | 0.49 |
| organism_b | ACT_org_random48_cen | 848.6 | 848.7 | 719.4 | 0.85 |
| organism_b | DIFF_changed48_cen | 1180.7 | 1181.8 | 447.1 | 0.38 |
| organism_b | DIFF_random48_cen | 861.5 | 862.0 | 764.4 | 0.89 |

## 5. Optimization curves (seed 0 of each arm; best objective by step)

| organism | arm | s0 | s50 | s100 | s200 | s300 | s400 | s499 | last-100 gain |
|---|---|---|---|---|---|---|---|---|---|
| organism_a | ACT_base_changed16_cen | 15.9 | 88.0 | 145.3 | 294.8 | 424.5 | 500.3 | 529.2 | +24.7% |
| organism_a | ACT_base_changed16_pca | 10.0 | 94.2 | 178.4 | 302.6 | 399.7 | 470.7 | 561.8 | +40.6% |
| organism_a | ACT_base_changed48_cen | 53.9 | 169.3 | 270.8 | 455.4 | 552.9 | 629.1 | 670.2 | +21.2% |
| organism_a | ACT_base_changed48_pca | 34.1 | 154.5 | 252.0 | 351.5 | 402.0 | 443.0 | 478.6 | +19.1% |
| organism_a | ACT_base_changed48_raw | 57.4 | 198.6 | 298.2 | 459.5 | 553.8 | 645.9 | 684.5 | +23.6% |
| organism_a | ACT_base_random16_cen | 17.8 | 77.5 | 145.6 | 263.0 | 343.8 | 439.5 | 503.0 | +46.3% |
| organism_a | ACT_base_random48_cen | 32.4 | 128.9 | 204.9 | 349.9 | 418.1 | 466.7 | 498.6 | +19.3% |
| organism_a | ACT_base_random48_pca | 30.4 | 114.4 | 183.6 | 302.5 | 427.4 | 690.5 | 827.5 | +93.6% |
| organism_a | ACT_org_changed16_cen | 13.8 | 399.9 | 650.2 | 803.1 | 873.6 | 904.1 | 949.0 | +8.6% |
| organism_a | ACT_org_changed16_pca | 7.6 | 156.7 | 279.0 | 417.4 | 539.1 | 621.0 | 671.6 | +24.6% |
| organism_a | ACT_org_changed48_cen | 39.8 | 292.7 | 561.4 | 738.7 | 835.2 | 885.5 | 912.5 | +9.3% |
| organism_a | ACT_org_changed48_pca | 26.9 | 123.0 | 237.3 | 361.8 | 473.7 | 545.0 | 563.3 | +18.9% |
| organism_a | ACT_org_changed48_raw | 40.0 | 313.5 | 532.4 | 781.8 | 867.1 | 949.7 | 972.1 | +12.1% |
| organism_a | ACT_org_random16_cen | 14.4 | 128.6 | 225.1 | 406.2 | 707.7 | 816.8 | 854.3 | +20.7% |
| organism_a | ACT_org_random48_cen | 23.7 | 163.8 | 249.0 | 342.3 | 529.7 | 818.2 | 929.9 | +75.5% |
| organism_a | ACT_org_random48_pca | 16.8 | 124.6 | 220.9 | 407.2 | 556.9 | 671.4 | 741.0 | +33.1% |
| organism_a | DIFF_changed48_cen | 36.3 | 233.2 | 508.5 | 795.8 | 929.4 | 1009.0 | 1133.7 | +22.0% |
| organism_a | DIFF_changed48_pca | 33.5 | 187.7 | 319.4 | 449.4 | 515.3 | 577.2 | 633.3 | +22.9% |
| organism_a | DIFF_random48_cen | 24.6 | 152.9 | 289.2 | 425.4 | 528.5 | 618.7 | 688.4 | +30.3% |
| organism_a | DIFF_random48_pca | 22.0 | 145.1 | 242.6 | 400.5 | 509.5 | 567.2 | 724.3 | +42.2% |
| organism_b | ACT_base_changed16_cen | 22.3 | 120.5 | 225.4 | 382.4 | 527.1 | 623.4 | 675.5 | +28.1% |
| organism_b | ACT_base_changed16_pca | 17.4 | 100.6 | 186.6 | 353.3 | 498.1 | 598.3 | 577.0 | +15.8% |
| organism_b | ACT_base_changed48_cen | 61.5 | 145.3 | 240.8 | 334.7 | 403.6 | 453.3 | 478.1 | +18.5% |
| organism_b | ACT_base_changed48_pca | 45.0 | 121.2 | 201.8 | 297.6 | 335.8 | 386.7 | 416.5 | +24.0% |
| organism_b | ACT_base_changed48_raw | 67.0 | 142.6 | 243.6 | 341.7 | 403.7 | 438.8 | 475.7 | +17.8% |
| organism_b | ACT_base_random16_cen | 17.8 | 72.8 | 134.3 | 231.4 | 332.2 | 406.5 | 489.6 | +47.4% |
| organism_b | ACT_base_random48_cen | 32.4 | 121.5 | 202.2 | 326.3 | 417.0 | 440.7 | 491.4 | +17.9% |
| organism_b | ACT_base_random48_pca | 31.0 | 116.2 | 196.0 | 317.2 | 737.7 | 832.1 | 861.0 | +16.7% |
| organism_b | ACT_org_changed16_cen | 15.0 | 218.6 | 422.0 | 760.5 | 854.8 | 928.7 | 946.1 | +10.7% |
| organism_b | ACT_org_changed16_pca | 9.1 | 119.1 | 193.5 | 366.3 | 490.2 | 584.1 | 628.1 | +28.1% |
| organism_b | ACT_org_changed48_cen | 38.3 | 238.5 | 533.3 | 751.2 | 828.9 | 894.1 | 905.3 | +9.2% |
| organism_b | ACT_org_changed48_pca | 30.2 | 143.9 | 243.3 | 389.5 | 437.2 | 456.1 | 476.6 | +9.0% |
| organism_b | ACT_org_changed48_raw | 39.1 | 174.1 | 496.4 | 793.1 | 895.2 | 969.8 | 1019.2 | +13.8% |
| organism_b | ACT_org_random16_cen | 13.2 | 130.5 | 221.8 | 356.5 | 507.5 | 771.7 | 864.5 | +70.4% |
| organism_b | ACT_org_random48_cen | 24.5 | 143.3 | 235.1 | 374.0 | 510.5 | 852.0 | 951.5 | +86.4% |
| organism_b | ACT_org_random48_pca | 19.4 | 142.0 | 248.3 | 386.6 | 612.3 | 788.3 | 873.5 | +42.7% |
| organism_b | DIFF_changed48_cen | 42.8 | 188.6 | 310.7 | 719.7 | 919.7 | 954.6 | 1093.7 | +18.9% |
| organism_b | DIFF_changed48_pca | 35.2 | 170.8 | 268.5 | 414.6 | 472.2 | 560.9 | 597.4 | +26.5% |
| organism_b | DIFF_random48_cen | 26.3 | 155.4 | 243.4 | 377.1 | 487.1 | 583.7 | 672.1 | +38.0% |
| organism_b | DIFF_random48_pca | 23.1 | 158.6 | 283.4 | 366.5 | 526.4 | 752.7 | 724.2 | +37.6% |

## 6. Decoded soft prompts (nearest vocabulary token per slot)

| organism | arm | decoded (printable pool, hubness-corrected) | mean cos to NN | soft obj | hard-token obj | retained |
|---|---|---|---|---|---|---|
| organism_a | ACT_base_changed16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.577 | 542.3 | 15.9 | 0.029 |
| organism_a | ACT_base_changed16_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.579 | 561.8 | 10.1 | 0.018 |
| organism_a | ACT_base_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.619 | 672.7 | 54.0 | 0.080 |
| organism_a | ACT_base_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.617 | 478.6 | 34.2 | 0.071 |
| organism_a | ACT_base_changed48_raw | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.610 | 713.8 | 57.5 | 0.081 |
| organism_a | ACT_base_random16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.590 | 503.2 | 17.9 | 0.035 |
| organism_a | ACT_base_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.605 | 507.8 | 32.5 | 0.064 |
| organism_a | ACT_base_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.559 | 831.7 | 30.4 | 0.037 |
| organism_a | ACT_org_changed16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.603 | 951.1 | 13.9 | 0.015 |
| organism_a | ACT_org_changed16_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.506 | 686.4 | 7.7 | 0.011 |
| organism_a | ACT_org_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.526 | 915.4 | 39.8 | 0.044 |
| organism_a | ACT_org_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.486 | 632.2 | 26.9 | 0.043 |
| organism_a | ACT_org_changed48_raw | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.494 | 990.9 | 40.0 | 0.040 |
| organism_a | ACT_org_random16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.502 | 898.9 | 14.4 | 0.016 |
| organism_a | ACT_org_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.509 | 963.3 | 23.7 | 0.025 |
| organism_a | ACT_org_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.526 | 747.4 | 16.8 | 0.022 |
| organism_a | DIFF_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.544 | 1157.7 | 36.4 | 0.031 |
| organism_a | DIFF_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.591 | 633.3 | 33.6 | 0.053 |
| organism_a | DIFF_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.546 | 696.6 | 24.6 | 0.035 |
| organism_a | DIFF_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.531 | 726.9 | 22.0 | 0.030 |
| organism_b | ACT_base_changed16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.570 | 698.9 | 22.4 | 0.032 |
| organism_b | ACT_base_changed16_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.593 | 644.4 | 17.5 | 0.027 |
| organism_b | ACT_base_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.595 | 478.4 | 61.5 | 0.129 |
| organism_b | ACT_base_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.579 | 417.5 | 45.0 | 0.108 |
| organism_b | ACT_base_changed48_raw | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.581 | 479.8 | 67.0 | 0.140 |
| organism_b | ACT_base_random16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.581 | 498.7 | 17.8 | 0.036 |
| organism_b | ACT_base_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.626 | 497.4 | 32.5 | 0.065 |
| organism_b | ACT_base_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.593 | 896.0 | 31.0 | 0.035 |
| organism_b | ACT_org_changed16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.559 | 977.5 | 15.0 | 0.015 |
| organism_b | ACT_org_changed16_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.562 | 645.0 | 9.1 | 0.014 |
| organism_b | ACT_org_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.521 | 922.5 | 38.3 | 0.042 |
| organism_b | ACT_org_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.489 | 479.5 | 30.2 | 0.063 |
| organism_b | ACT_org_changed48_raw | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.515 | 1038.9 | 39.2 | 0.038 |
| organism_b | ACT_org_random16_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.475 | 868.5 | 13.2 | 0.015 |
| organism_b | ACT_org_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.496 | 951.5 | 24.6 | 0.026 |
| organism_b | ACT_org_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.498 | 882.3 | 19.4 | 0.022 |
| organism_b | DIFF_changed48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.510 | 1115.6 | 42.9 | 0.038 |
| organism_b | DIFF_changed48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.561 | 614.3 | 35.3 | 0.057 |
| organism_b | DIFF_random48_cen | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.599 | 672.1 | 26.3 | 0.039 |
| organism_b | DIFF_random48_pca | ` desarrollDOG-clock Chall imperative(si Width:frame fencingvelopOs probing olds nijetoolSt` | 0.514 | 961.1 | 23.1 | 0.024 |
