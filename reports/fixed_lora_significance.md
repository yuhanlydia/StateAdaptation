# KV-TTT versus fixed-support, CV-selected LoRA

Here ``fixed'' means that LoRA uses the original fixed support manifest; its
training duration was selected by per-event support CV. It is not the later
clean fixed-8 LoRA protocol. KV-TTT averages three
independently sampled support24 seeds. The estimand therefore measures KV
support-sampling variability conditional on the fixed LoRA support set; it is
not a matched multiseed LoRA comparison.

| Comparison | Mean Δ Macro-F1 | Stratified bootstrap 95% CI | Paired permutation p |
|---|---:|---:|---:|
| Full KV-TTT − LoRA | -0.0609 | [-0.1065, -0.0171] | 0.0010 |
| Diagonal KV-TTT − LoRA | -0.0668 | [-0.1083, -0.0267] | 0.0003 |

Under this legacy comparison, both KV-TTT variants significantly improve on
raw VLM inference, while the CV-selected LoRA remains more accurate. KV-TTT's
advantage is parameter and artifact efficiency: 100 Full or 20 Diagonal
controller scalars versus 10,092,544 LoRA parameters.
