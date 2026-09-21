# Aperture medical main-study handoff

This adds a new medical study. It does not change Aperture's operator, erase previous results, or guarantee improvement over LoRA. Read `AGENTS.md` and `docs/APERTURE_MEDICAL_PROTOCOL.md` first.

## What to run

- CAMELYON17-WILDS hospitals 0, 1, 2, 3, 4; 16 and 32 TOTAL labels; three support seeds; 500 fixed query patches per hospital. H2 remains a previously inspected anchor.
- PathMNIST-224 CRC target cohort; nine fixed tissue classes; 72 and 144 TOTAL labels; three support seeds; 900 fixed query patches. Its supplied NPZ lacks patient/slide IDs: isolation is image-content level, NOT patient-level.
- Five fitted/evaluated arms: Frozen, LoRA-1pass, LoRA-4passes, Random visual residual, Aperture. Positive temperature scaling and a strong all-support output bias are exported too. LoRA schedule is selected on held-out SUPPORT only.
- 180 primary states, 144 trainable fits, plus four repeat-fit diagnostics and two real-model audit gates. No historical state is overwritten. Five-center main table and second-dataset table are separate.

## Assets and execution

Keep the working Qwen2.5-VL-7B environment. BF16 protocol requires 24GB+ GPU. Do not resize or quantize to bypass a failed audit. Existing H2-only image subsets are insufficient for hospitals 0/1/3/4.

```bash
git switch main
git pull --ff-only origin main
PYTHONPATH=src python3 -m pytest -q tests_aperture_medical
# Only if official full data are not already local:
python3 -m pip install wilds==2.0.0 pandas
python3 scripts/download_aperture_medical.py camelyon17 --root data/medical_raw
python3 scripts/download_aperture_medical.py pathmnist --root data/medical_raw
# EDIT paths in configs/aperture_medical_v1.json BEFORE plan is locked.
bash scripts/run_aperture_medical.sh plan
bash scripts/run_aperture_medical.sh prepare
bash scripts/run_aperture_medical.sh check
# Integration unit. Its completed states are reused later.
bash scripts/run_aperture_medical.sh run --gpus 0 --domains hospital_0 --budgets 16 --seeds 0
# Entire registered matrix. A single GPU uses --gpus 0.
bash scripts/run_aperture_medical.sh run --gpus 0,1,2,3
bash scripts/run_aperture_medical.sh summarize
```

`run` enforces actual support-only GPU audits, repeats representative LoRA/Aperture FITS, then fits the complete matrix; model and temperature choices are sealed before query evaluation. Loading a saved state must reproduce its calibration scores. A repeat-fit failure stops the campaign; do not turn off the gate or select the better rerun. Investigate numerics in a NEW versioned run root. Strict deterministic CUDA configuration is requested, but GPU reproducibility is NOT yet certified by the CPU tests.

Exports: `runs/aperture_medical_v1/paper_exports/` contains per-seed CSV, mean/SD, NLL/F1/BA/AUROC/AUPRC/Brier/ECE LaTeX, per-query-group metrics, all-five-center and four-expansion-center summaries, measured resource tables, and plots. Every declared state is required for a paper export; filters limit execution, not which outcomes are published.

No GPU experiment was run by the publishing session. Numeric unit-test fixtures are synthetic, not experimental results. Do not publish raw images, patient IDs, model files or local predictions automatically. Return compact, complete result summaries for manuscript integration.
