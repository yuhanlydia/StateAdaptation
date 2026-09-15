# Extended mechanism and robustness results

Last updated: 2026-08-11 UTC. All runs use the four strictly balanced events,
raw Qwen2.5-VL-7B, same-event labeled support, and 300 queries per event.

## Gradient-subspace mechanism

| Basis | Seeds | Mean Macro-F1 | Delta vs raw | Delta balanced accuracy | NLL reduction |
|---|---:|---:|---:|---:|---:|
| Uncentered gradient covariance, rank 5 | 1 | **0.2964** | **+0.0730** | -0.0008 | +0.1435 |
| Mean gradient, rank 1 | 1 | 0.2506 | +0.0271 | -0.0100 | +0.0984 |
| Centered gradient covariance, rank 5 | 1 | 0.2721 | +0.0486 | -0.0133 | **+0.1440** |
| Random basis, rank 5 | 3 | 0.2414 | +0.0180 | -0.0028 | +0.0468 |

Paired 10,000-resample tests give covariance minus mean-gradient +0.0458
(95% CI +0.0188 to +0.0735, p=0.0142), covariance minus centered
covariance +0.0243 (95% CI +0.0100 to +0.0388, p=0.0032), and covariance
minus random basis +0.0551 (95% CI +0.0058 to +0.1036, p=0.0022). For the
random comparison, the random seed is sampled within each event and bootstrap
replicate.

The random control is variable (seed Macro-F1 0.2156, 0.2165, and 0.2921), so
the supported claim is that covariance is better and more reliable on average,
not that every random subspace must fail. Centering nearly preserves the NLL
gain but reduces classification F1, suggesting that the non-zero gradient mean
also contains useful task direction.

## Alpha robustness

| Alpha | Mean Macro-F1 | Delta vs raw |
|---:|---:|---:|
| 0.5 | 0.2384 | +0.0149 |
| 1 | 0.2660 | +0.0426 |
| 2 | 0.2690 | +0.0455 |
| 3 | **0.2964** | **+0.0730** |
| 5 | 0.2531 | +0.0297 |
| 10 | 0.2720 | +0.0485 |

## Coefficient-learning-rate robustness

| Learning rate | Mean Macro-F1 | Delta vs raw |
|---:|---:|---:|
| 0.01 | 0.2659 | +0.0424 |
| 0.05 | **0.2964** | **+0.0730** |
| 0.1 | 0.2513 | +0.0278 |
| 0.2 | 0.2466 | +0.0231 |

Alpha 3 and learning rate 0.05 are the best settings in their complete
four-event sweeps; they were not selected from a Hawaii-only subset.

