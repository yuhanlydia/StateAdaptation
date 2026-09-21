# Aperture: five-hospital, two-backbone medical experiment

This is the Qwen2.5-VL-7B + InternVL3-8B package delivered in
`Aperture_Medical_Main_Code.zip`, published with an isolated `aperture_medical_multimodel`
namespace. It does NOT replace the separate `aperture_medical` protocol already in
main (five hospitals + PathMNIST, one backbone). Do not mix their splits or outputs.
Only package/entry-point names and output roots were changed for coexistence;
experimental settings and model code were not retuned during publication.

## Run after pulling main

Read `AGENTS.md` and `docs/APERTURE_MEDICAL_MULTIMODEL_PROTOCOL.md` first.
The complete official CAMELYON17-WILDS `metadata.csv` and `patches/` are required;
the previous Hospital-2 image subset is insufficient. Keep the existing working
model environment and local snapshots. GPU audits have not been run here.

```bash
git switch main
git pull --ff-only origin main
PYTHONPATH=src python3 -m pytest -o addopts= -q tests_aperture_medical_multimodel
# EDIT local paths in configs/aperture_medical_multimodel.json before locking the plan.
bash scripts/run_aperture_medical_multimodel.sh plan
bash scripts/run_aperture_medical_multimodel.sh prepare
bash scripts/run_aperture_medical_multimodel.sh check
# Support-only audit and a repeated fit run first on each backbone.
# Single-GPU integration unit; completed states are reused by the full run.
bash scripts/run_aperture_medical_multimodel.sh run 0 --phases main --hospitals 0 --seeds 10
# Full default matrix (one GPU: replace 0,1,2,3 with 0).
bash scripts/run_aperture_medical_multimodel.sh run 0,1,2,3
bash scripts/run_aperture_medical_multimodel.sh summarize
python3 scripts/plot_aperture_medical_multimodel.py
```

## Locked study

- Five hospitals, support seeds 10/11/12, 500 fixed query patches per hospital.
- Fit/calibration/query patients are disjoint. All models and budgets share query IDs.
- 32 total labels per domain: 24 fitting + 8 held-out calibration.
- Qwen primary: 75 states (Frozen, LoRA1, LoRA4, Random, Aperture).
- InternVL3 replication: 60 states (Frozen, LoRA1, LoRA4, Aperture).
- Default total: 135 states, 105 trainable fits, plus support-only repeat-fit gates.
- Optional Qwen 16/64-label curve: 120 more states; not triggered by query outcomes.
- Qwen2 and InternVL3 are the implemented backbones. Gemma and Qwen3 are NOT
  implemented or GPU-validated by this package; do not substitute model IDs.
- BF16 protocol requires 24GB-or-larger CUDA GPUs. No implicit quantization,
  smaller model, reduced image size, relaxed audits, or query-based tuning.

Exports are under `runs/aperture_medical_multimodel_v1/paper_exports/main-replication/`.
The optional curve uses `--phases curve`; export all completed phases with
`--phases main replication curve`. Keep every declared method, hospital and seed.

All inherited source, reports, medical data and the other medical protocol remain
unchanged. Do not publish patient records, raw images, model weights or raw local
predictions automatically. New GPU results are pending; CPU tests use synthetic
fixtures and do not certify clinical performance or real-model reproducibility.
