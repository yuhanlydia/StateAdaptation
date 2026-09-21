# Medical extension review record

Review performed locally; no independent reviewer agent or GPU was available.

- Inspected official WILDS input schema and hospital numbering. Kept original
  filenames and metadata IDs; no early-stopped stream or fabricated grouping.
- Tests cover patient/slide isolation, nested label sets, fixed query IDs,
  duplicate detection, all center/model phase counts and file-content locks.
- Reused executed controller/fitter and scorer. New worker handles both validated
  model families and restores random-basis controls as well as Aperture.
- Fixed an exporter integration mismatch caught during review: real evaluation
  payload lives in evaluation.json; metrics.json is a completion marker. Test
  fixtures now reproduce the worker's actual two-file structure.
- Support-only repeatability pairs are explicitly scheduled on the SAME GPU in
  separate processes; they do not consume query predictions or select a model.
- All table creation follows complete-coverage checks; paper_ready remains false
  until content validation and numeric export finish. Consumers must check the
  coverage marker, including when a previously exported state becomes invalid.
- All actual test fixtures are synthetic and are not reported as GPU experiments.
- No legacy source/result is overwritten. The default full data directory is
  data/raw/camelyon17_v1.0, already ignored by the repository's .gitignore.

Open runtime checks: complete official metadata has not been scanned here;
real patient-pool feasibility, all images, both local model snapshots, CUDA
memory, deterministic kernels and end-to-end fitting must pass on the user's
GPU host. Code records failures and does not change protocol to avoid them.
