
#### Track A results (final-epoch top-1 %; best-epoch in parentheses)

| Model | Data | Recipe | BP (K=1) | DDG K=2 | Δ K=2 | DDG K=4 | Δ K=4 | DDG paper BP / DDG |
|---|---|---|---|---|---|---|---|---|
| resnet56 | cifar10 | modern | 94.22 (94.32) | 94.12 (94.27) | -0.10 | 94.09 (94.10) | -0.13 | 93.12 / 93.11 |
| resnet56 | cifar10 | paper | 92.81 (93.07) | 92.70 (92.93) | -0.11 | – | – | 93.12 / 93.11 |
| resnet56 | cifar100 | modern | 72.73 (72.79) | 72.16 (72.17) | -0.57 | 71.11 (71.33) | -1.62 | 69.79 / 70.17 |
| resnet110 | cifar10 | modern | 94.76 (94.83) | 94.52 (94.56) | -0.24 | 94.10 (94.18) | -0.66 | 93.53 / 93.41 |
| resnet110 | cifar100 | modern | 73.67 (73.92) | 74.18 (74.40) | +0.51 | 73.26 (73.32) | -0.41 | 71.9 / 71.39 |

![Track A curves](code_repo/results/figs/trackA_curves.png)

#### Track B, ResNet-50, CIFAR-10 (final test top-1 %)

| Row | test acc | Δ vs vanilla | seeds | peak mem (max stage) vs 2BW | avg stage-1 mem vs 2BW | time/batch static vs 2BW | rebalanced vs 2BW | frozen | no-shadow |
|---|---|---|---|---|---|---|---|---|---|
| 0 Vanilla | 95.62 ± 0.09 | +0.00 | 2 | -0.1% | -0.1% | +66.4% | +66.4% | 0% | 0% |
| 1 FRZ only | 95.81 | +0.19 | 1 | -0.1% | -9.0% | +60.9% | +59.5% | 11% | 0% |
| 2 2BW | 95.39 ± 0.01 | -0.23 | 2 | +0.0% | +0.0% | +0.0% | +0.0% | 0% | 0% |
| 3 1BW | 95.84 ± 0.14 | +0.22 | 2 | -0.1% | -0.1% | +0.0% | +0.0% | 0% | 100% |
| 3b 1BW after 10% 2BW warm start | 95.78 | +0.16 | 1 | +0.0% | -0.1% | +0.0% | +0.0% | 0% | 90% |
| 4 2BW-Sel | 95.14 | -0.48 | 1 | +0.0% | -0.0% | +0.0% | +0.0% | 0% | 18% |
| 5 2BW + FRZ | 95.34 | -0.28 | 1 | +0.0% | -8.5% | -2.2% | -4.4% | 10% | 0% |
| 6 Full (ours) | 95.52 ± 0.14 | -0.10 | 2 | +0.0% | -8.6% | -2.5% | -4.0% | 10% | 6% |
| 7 Full, random | 95.51 | -0.11 | 1 | +0.0% | -8.5% | -2.4% | -3.9% | 9% | 7% |
| 8 Full, FreezeOut sched. | 95.34 | -0.28 | 1 | +0.0% | -46.3% | -5.2% | -18.6% | 25% | 6% |
| 9 Full, Adam reset | 95.32 | -0.30 | 1 | +0.0% | -8.8% | -2.3% | -4.2% | 9% | 6% |

**Attribution (test acc (pts)), each relative to 2BW (row 2):**

| Quantity | total Δ6 | shadow Δ4 | freezing Δ5 | interaction |
|---|---|---|---|---|
| test acc (pts) | +0.13 | -0.25 | -0.05 | +0.42 |
| avg stage-1 memory (MB) | -87 | -0 | -85 | -1 |
| rebalanced time/batch (ms) | -0.43 | +0.00 | -0.47 | +0.04 |

#### Track B, ResNet-50, CIFAR-100 (final test top-1 %)


Plain baseline (batch 128, no microbatching): 95.39%
1BW at d=2 (m/d=2): 95.63%

![Track B curves](code_repo/results/figs/trackB_curves.png)

#### Track C, BERT-mini (final MLM validation loss; lower is better)

| Row | valid loss | Δ vs vanilla | seeds | peak mem (max stage) vs 2BW | avg stage-1 mem vs 2BW | time/batch static vs 2BW | rebalanced vs 2BW | frozen | no-shadow |
|---|---|---|---|---|---|---|---|---|---|
| 0 Vanilla | 3.25 | +0.00 | 1 | -6.8% | -7.4% | +1.7% | +1.7% | 0% | 0% |
| 2 2BW | 3.49 | +0.24 | 1 | +0.0% | +0.0% | +0.0% | +0.0% | 0% | 0% |
| 3 1BW | 3.30 | +0.05 | 1 | -6.8% | -7.4% | +0.0% | +0.0% | 0% | 100% |
| 3b 1BW after 10% 2BW warm start | 3.44 | +0.19 | 1 | +0.0% | -6.6% | +0.0% | +0.0% | 0% | 90% |
| 6 Full (ours) | 3.51 | +0.26 | 1 | +0.0% | -6.2% | +0.0% | +0.0% | 4% | 5% |

#### Track C, BERT-small (final MLM validation loss; lower is better)

| Row | valid loss | Δ vs vanilla | seeds | peak mem (max stage) vs 2BW | avg stage-1 mem vs 2BW | time/batch static vs 2BW | rebalanced vs 2BW | frozen | no-shadow |
|---|---|---|---|---|---|---|---|---|---|
| 0 Vanilla | 2.86 | +0.00 | 1 | -8.1% | -8.1% | +5.9% | +5.9% | 0% | 0% |
| 2 2BW | 3.01 | +0.15 | 1 | +0.0% | +0.0% | +0.0% | +0.0% | 0% | 0% |
| 3 1BW | 2.91 | +0.05 | 1 | -8.1% | -8.1% | +0.0% | +0.0% | 0% | 100% |
| 3b 1BW after 10% 2BW warm start | 2.99 | +0.13 | 1 | +0.0% | -7.3% | +0.0% | +0.0% | 0% | 90% |
| 6 Full (ours) | 3.02 | +0.16 | 1 | +0.0% | -5.7% | +0.0% | +0.0% | 4% | 6% |

#### Track C, BERT-medium (final MLM validation loss; lower is better)

| Row | valid loss | Δ vs vanilla | seeds | peak mem (max stage) vs 2BW | avg stage-1 mem vs 2BW | time/batch static vs 2BW | rebalanced vs 2BW | frozen | no-shadow |
|---|---|---|---|---|---|---|---|---|---|
| 0 Vanilla | 2.77 ± 0.00 | +0.00 | 2 | -5.6% | -5.6% | +10.5% | +10.5% | 0% | 0% |
| 1 FRZ only | 2.81 | +0.04 | 1 | -5.6% | -8.8% | +10.3% | +10.3% | 5% | 0% |
| 2 2BW | 3.00 ± 0.02 | +0.23 | 2 | +0.0% | +0.0% | +0.0% | +0.0% | 0% | 0% |
| 3 1BW | 2.89 | +0.12 | 1 | -5.6% | -5.6% | +0.0% | +0.0% | 0% | 100% |
| 3b 1BW after 10% 2BW warm start | 2.99 | +0.22 | 1 | +0.0% | -5.0% | +0.0% | +0.0% | 0% | 90% |
| 4 2BW-Sel | 3.03 | +0.26 | 1 | +0.0% | -2.8% | +0.0% | +0.0% | 0% | 7% |
| 5 2BW + FRZ | 3.06 | +0.29 | 1 | +0.0% | -4.1% | +0.0% | +0.0% | 4% | 0% |
| 6 Full (ours) | 3.03 ± 0.03 | +0.26 | 2 | +0.0% | -4.9% | +0.0% | +0.0% | 4% | 2% |
| 7 Full, random | 3.04 | +0.27 | 1 | +0.0% | -2.1% | +0.0% | +0.0% | 4% | 2% |
| 8 Full, FreezeOut sched. | 3.14 | +0.37 | 1 | +0.0% | -36.5% | +0.0% | +0.0% | 25% | 2% |
| 9 Full, Adam reset | 3.06 | +0.29 | 1 | +0.0% | -5.0% | +0.0% | +0.0% | 4% | 2% |

**Attribution (valid MLM loss), each relative to 2BW (row 2):**

| Quantity | total Δ6 | shadow Δ4 | freezing Δ5 | interaction |
|---|---|---|---|---|
| valid MLM loss | +0.03 | +0.03 | +0.05 | -0.05 |
| avg stage-1 memory (MB) | -75 | -43 | -62 | +31 |
| rebalanced time/batch (ms) | +0.00 | +0.00 | +0.00 | +0.00 |

#### Track C, BERT-base (final MLM validation loss; lower is better)

| Row | valid loss | Δ vs vanilla | seeds | peak mem (max stage) vs 2BW | avg stage-1 mem vs 2BW | time/batch static vs 2BW | rebalanced vs 2BW | frozen | no-shadow |
|---|---|---|---|---|---|---|---|---|---|
| 0 Vanilla | 2.98 | +0.00 | 1 | -5.2% | -5.2% | +63.0% | +63.0% | 0% | 0% |
| 2 2BW | 3.09 | +0.11 | 1 | +0.0% | +0.0% | +0.0% | +0.0% | 0% | 0% |
| 6 Full (ours) | 3.10 | +0.12 | 1 | +0.0% | -1.7% | +0.0% | +0.0% | 1% | 3% |

#### 1BW depth / microbatch sweep (BERT-Medium)

| d | m | m/d | valid loss | Δ vs vanilla | mismatched microbatch-units |
|---|---|---|---|---|---|
| 2 | 2 | 1 | 2.8159 | +0.0457 | 35.0% |
| 4 | 16 | 4 | 2.7789 | +0.0087 | 11.2% |
| 8 | 8 | 1 | 2.8776 | +0.1074 | 51.2% |
| 4 | 4 | 1 | 2.8860 | +0.1158 | 45.0% |

![Track C scaling](code_repo/results/figs/trackC_scaling.png)

![Track C curves](code_repo/results/figs/trackC_curves.png)

![shadow vs no shadow](code_repo/results/figs/trackC_shadow.png)
