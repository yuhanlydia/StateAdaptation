# Run the four-backbone table with the currently empty manuscript cells

This is the approved64-state experiment, in the new `aperture_table` namespace.
It is NOT `aperture_medical` or `aperture_medical_multimodel`. Those studies and
all their outputs remain unchanged. Read `AGENTS.md` and
`docs/APERTURE_TABLE_PROTOCOL.md` before execution.

Models: Qwen2.5-VL-7B, Qwen3-VL-4B, Qwen3-VL-8B, Gemma3-4B-IT.
Tasks: H2(12fit+4cal,200query) and PathMNIST-224(54fit+18cal,180query).
Seeds0/1; Frozen, LoRA1, LoRA4, Aperture; derived all-support bias and temperature.

## Assets and environment

Edit paths ONLY in `configs/aperture_table_v1.json` and source data paths in
`configs/aperture_medical_v1.json` BEFORE locking the plan. The four model
snapshots must exist locally, with native tokenizers/processors. Gemma requires
its official access approval. Preserve the old working environment; create a
separate environment for the requirements file if needed. Install the correct
CUDA PyTorch wheel for your machine separately. Minimum24GB VRAM; no quantized
or lower-resolution fallback. Do not run other GPU workloads on the same cards.

```bash
cd StateAdaptation
# After publication, or after applying the delivered patch:
PYTHONPATH=src OMP_NUM_THREADS=1 python3 -m pytest -o addopts= -q tests_aperture_table
bash scripts/run_aperture_table.sh plan
bash scripts/run_aperture_table.sh prepare
bash scripts/run_aperture_table.sh check

# Mandatory real-model audit; the CLI also runs it automatically in run.
# This command measures support-only throughput before any new query results.
bash scripts/run_aperture_table.sh audit --gpus 0,1,2,3

# Integration unit (subsequent full run reuses completed states).
bash scripts/run_aperture_table.sh run --gpus 0 --models q25 --datasets h2 --seeds 0

# Entire64-state registered matrix; single GPU: --gpus 0.
bash scripts/run_aperture_table.sh run --gpus 0,1,2,3
bash scripts/run_aperture_table.sh summarize

# Fill ONLY matching measured values into a NEW copy of the latest manuscript.
bash scripts/run_aperture_table.sh fill-main /path/to/main.tex /path/to/main_filled.tex
```

`prepare --reuse-source-only` skips source preparation only when the existing
medical_v1 fit/calibration/query manifests already exist and pass validation.
It does not invent a smaller data split. Run filters select execution units,
not publishable outcomes: the final table requires all64 states.

Exports: `runs/aperture_table_v1/paper_exports/main_table.tex`,
`main_table_cells.json`, `clean_mean_sd.csv`, `full_results.json`,
`resources.json`, and `coverage.json`. Full results retain all metrics and LoRA
selection records; the main table shows F1%, raw NLL and NLL+T.

No new benchmark numbers, GPU integration pass, or24hour guarantee are supplied
by the code publication. CPU tests use synthetic fixtures. Runtime audits and
calibration replay are mandatory. Never publish raw patient/image data, model
weights, local predictions or credentials automatically.
