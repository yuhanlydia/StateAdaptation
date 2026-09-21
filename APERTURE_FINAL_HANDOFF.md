# Agent handoff: final Aperture validation round

Read `AGENTS.md`, `docs/APERTURE_FINAL_PROTOCOL.md` and `docs/APERTURE_FINAL_RUN.md`.

This is the published, additive experiment package from the final-round handoff.
It is NOT a replacement architecture or a new GPU result report. The source base
is `4465c460350ddf2ce3c8e65f5b2cdb247ac7497d`. The Python sources, configuration and
tests match the previously delivered final-round package; only publication/run
instructions have been updated. Do not apply the offline patch again.

## Run after pulling main

```bash
cd StateAdaptation
git pull --ff-only origin main
PYTHONPATH=src python3 -m pytest -q tests_aperture_final
# Inspect local model and manifest paths in configs/aperture_final_round.json first.
bash scripts/run_aperture_final.sh plan
bash scripts/run_aperture_final.sh prepare
bash scripts/run_aperture_final.sh check
bash scripts/run_aperture_final.sh run 0 --domains hospital_2 --seeds 0
bash scripts/run_aperture_final.sh run 0,1,2,3
bash scripts/run_aperture_final.sh summarize
python3 scripts/plot_aperture_final.py
```

For one GPU use `run 0` for the full run. The fixed BF16 protocol requires a
24GB-or-larger CUDA GPU. Keep the already working dependency environment. Real
VLM integration and GPU experiments have not been run in the publishing session;
the existing support-only real-model audits are mandatory runtime gates.

The default matrix has 96 states (72 trainable fits), two real-model audit gates,
three existing support seeds and eight existing domains. Calibration reserves
labels WITHIN the original budget: BRIGHT 18+6, medical 12+4 and ManipBench 24+8.
Do not describe the new result as the historical all-support fitting protocol.

## Fixed execution contract

1. Confirm model and data paths BEFORE locking the plan. Original manifests stay unchanged.
2. Run plan, prepare, check, a seed-0 integration unit, then the full fixed matrix.
3. Respect audit failures. Do not silently change model, quantization, resolution,
   candidate format, query IDs or tolerances to pass a failed check.
4. Seal fitting, temperature and LoRA-schedule selection before query evaluation.
5. Keep every finite result, all prespecified methods, all domains and all seeds.
6. Do not use query outcomes to expand the population, grid, optimizer or stopping budget.
7. Preserve old reports and runs. This round writes only to its new versioned roots.
8. Export suitable aggregates into `runs/aperture_final_v1/paper_exports/`.
9. Never push raw medical/robot images, patient metadata, credentials, model weights
   or ignored prediction/controller artifacts to the public repository.

When the fixed run is complete, return the full aggregated JSON/CSV/LaTeX and
measurement-based figures for the paper. No performance gain or acceptance is
guaranteed by this handoff.
