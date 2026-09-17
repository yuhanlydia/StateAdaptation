# Visual Lens P0 final findings

## Completion and integrity

- Completed all 83 planned jobs: 3 audits, 63 matched comparisons, 8 objective controls, and 9 image-evidence jobs.
- Verified every sealed artifact against its stored fingerprint and SHA-256 hashes.
- Validated 110 prediction sets for complete query coverage, unique sample IDs, correct labels, finite normalized probabilities, and matching metric counts.
- All stage execution records have return code 0. The summary reports no missing jobs or aggregation errors.
- The full CPU test suite passes: 57 tests.

## Main matched results

Across the 12 BRIGHT event/support-seed states, mean Macro-F1 is:

| Method | Macro-F1 |
|---|---:|
| Frozen | 0.2255 |
| LoRA, 1 pass | 0.3146 |
| LoRA, 4 passes | 0.2739 |
| Visual Lens | 0.3133 |
| Random-KV | 0.3018 |

Visual Lens improves over Frozen by 8.77 points and over Random-KV by 1.15 points on this aggregate. It is within 0.14 points of one-pass LoRA and 3.94 points above the four-pass LoRA setting. The event-level behavior is heterogeneous: Lens is best on Noto (0.3242), beats Frozen and Random-KV on Libya (0.3404), trails Random-KV on Hawaii (0.3329 versus 0.3667), and trails four-pass LoRA on Turkey (0.2557 versus 0.2992).

On Camelyon17, Visual Lens reaches 0.5457 Macro-F1 and 0.7252 NLL, compared with Frozen at 0.3333/0.9662 and LoRA at 0.5454/3.0991. Accuracy is essentially tied with LoRA while Lens is much better calibrated by NLL in this run.

## Mechanism controls

- All three gradient-path audits pass after applying the configured BF16 relative tolerance consistently to the mean/sum check.
- On Camelyon17, full-answer objectives reach about 0.546 Macro-F1, while label-only objectives collapse to 0.333. On RoboFail, all four objective variants remain near 0.30 Macro-F1. This supports keeping the current full-answer objective and reporting label-only/RoboFail behavior as a limitation.
- Neutral-image interventions substantially reduce Lens performance on Hawaii, Libya, and Camelyon17. Shuffling produces weaker and sometimes uncertain changes, especially on Libya. The results support dependence on visual content, but do not establish that every gain requires exact image pairing.
- On Hawaii, the learned gradient basis reduces support loss more than the random basis in every checked fit, but Random-KV generalizes better on the query set. This is consistent with few-shot support overfitting or event-specific mismatch rather than a failed optimizer or inactive controller.

## Debugging fixes made during execution

- Rebuilt RoboFail splits to prevent task-variation group leakage between support and query, with a regression test.
- Made the Qwen BF16 audit use the configured 5% relative tolerance rather than a hard-coded 0.5% threshold, with regression tests. The measured vectors remain nearly collinear (cosine about 0.99995).
- Made the summarizer derive its default plan and output paths from the active config, preventing it from silently reading the earlier v1 run root, with regression tests.

## Reproduction limitation

The official BRIGHT images and annotations were downloaded from Zenodo and checksum-verified. The historical private support manifests were unavailable, so the four-event folds were deterministically regenerated with the original repository script and `PYTHONHASHSEED=0`. These results are a controlled reproduction on regenerated fixed supports, not a byte-for-byte restoration of the private manifests used for the earlier headline table.
