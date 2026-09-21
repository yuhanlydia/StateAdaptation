# Aperture final round: 2026-09-21

The fixed final-round matrix completed **96/96 states** (eight domains, three
support seeds and four fitted/evaluated arms). `results/paper_exports/coverage.json`
records `paper_ready: true`. These are measurements from the calibration-controlled
protocol in [`docs/APERTURE_FINAL_PROTOCOL.md`](../../docs/APERTURE_FINAL_PROTOCOL.md):
two examples per class are reserved from the original support budget for
calibration. They must not replace historical full-support scores without this
protocol distinction. Query labels were not used to select schedules or
temperatures. The primary run used code commit
`04f1b85d5ba72503e37054b963981da97d426328`, source signature
`f686261411b28fd9e267326aab1dd8c0b92e618d4521d94dae26993e7c638234`.

## Files

- `results/paper_exports/`: full aggregate JSON, per-seed and mean/SD CSV,
  contrasts, calibration, corruption and resource tables, editable LaTeX,
  and 102 measurement-derived SVG/PDF/PNG figure files.
- `results/rechecks/`: five prespecified unit reruns. The first four use the
  canonical source signature. `droid_arti` seed 0 uses a separately labeled
  diagnostic source change; it does not overwrite the canonical table.
- `data/fixed_splits/`: exact fit/calibration IDs, query-ID lists and split
  hashes for all 24 comparison units.
- `data/SHA256SUMS.data`: file-level checksums for the released data subset;
  `data/selection.json` records the selected file counts and sizes.
- The [GitHub Release](https://github.com/yuhanlydia/StateAdaptation/releases/tag/aperture-final-round-20260921-results)
  carries `data_subset.tar.zst.01.part` and `.02.part`, the complete
  `main_run.tar.zst` including sealed states and raw predictions, and
  `rechecks.tar.zst` including rerun artifacts and failure diagnostics.
  `SHA256SUMS.assets` verifies each downloadable archive/part.

The data archive contains **3,395 files / 2,441,402,915 uncompressed bytes**:
only the original images referenced by this experiment plus prepared support,
query and split manifests. It is not the complete upstream BRIGHT, WILDS or
ManipBench dataset. The Qwen model weights are not included; use the revision
specified in `scripts/download_oral_assets.sh`.

## Extract and relocate

Run from a directory with enough free space; the two data parts must be in the
same directory. Verify `SHA256SUMS.assets` there first.

```bash
sha256sum -c SHA256SUMS.assets
cat data_subset.tar.zst.01.part data_subset.tar.zst.02.part | zstd -d | tar -xf -
sha256sum -c reports/aperture_final_2026-09-21/data/SHA256SUMS.data
python3 reports/aperture_final_2026-09-21/relocate_manifest_paths.py .
tar --use-compress-program=unzstd -xf main_run.tar.zst
tar --use-compress-program=unzstd -xf rechecks.tar.zst
```

The first checksum command runs before extraction; the file-level data check
runs after extraction but **before** relocation. The relocation script validates
each referenced image and replaces the original machine's absolute data path
inside JSONL manifests. It intentionally changes those manifest checksums.
Model and run-root paths in archived configuration files are provenance, not
portable runtime settings; update them if replaying a run.

## Data sources and redistribution notes

- [BRIGHT official record](https://zenodo.org/records/20072020): four event
  subsets. The record identifies Maxar optical imagery (except Hawaii) as
  CC BY-NC 4.0, Hawaii optical imagery as NOAA, and SAR sources separately.
  Preserve those source and noncommercial conditions when using the images.
- [CAMELYON17-WILDS](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py): selected crops from the
  public-domain/CC0 dataset, with WILDS patient and slide indices in manifests.
- [ManipBench official dataset repository](https://github.com/slurm-lab-usc/ManipBench-Real-Robot-question):
  Q1 Bridge/DROID question images from the official Simplified Dataset. The
  upstream dataset page provides the original download and paper citation;
  no separate image license is asserted here.

The repository owner explicitly requested publication of the used image subset.
If redistributing an image outside this research package, follow its original
source terms and attribution. The data and results are provided for research
reproduction; they are not a new dataset license from this repository.

## Recheck interpretation

The four exact-source reruns measure run-to-run variation, including changed
predictions for trainable arms. The fifth `droid_arti` rerun first failed after
the controller had been saved: a descriptive operator-norm calculation invoked
CUDA cuSOLVER with essentially no free GPU memory while another GPU workload was
running. A no-cache diagnostic was stopped because it was unreasonably slow.
The final diagnostic moved only the four small operator-norm calculations to
CPU; model fitting, scoring, data and query selections were unchanged. Its
source signature is
`9f757a634d0e28b914999da412c4c27b66d045e5674446bcabbf1f0186d93a09`.
The diagnostic is labeled separately because this is a source change. See its
`comparison.json` and archived `diagnosis.json`; the canonical 96/96 result
remains the paper table.
