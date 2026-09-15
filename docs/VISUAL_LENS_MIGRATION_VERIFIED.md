# Migration verified

EventTune history and its historical src/eventttt, tests, and reports are preserved. The 35-file experiment handoff, three P0 experiment families, and the interrupted manuscript migration are now integrated.

CPU validation: 92 tests, 0 failures, 0 errors; Python and shell syntax checks passed. The canonical plan contains 83 jobs: 3 gradient audits, 63 matched comparisons, 8 objective/span variants, and 9 fixed-state visual evidence comparisons.

The single manuscript, official style dependencies, references, numerical figure data/scripts, figure PDFs/PNGs and all English prompts are in paper/visual_lens/. The manuscript text and historical performance numbers are unchanged.

No new GPU results were produced. Reuse the existing local model/data assets, then run prepare, plan, check, run, summarize through scripts/run_visual_lens_all.sh. The GPU audit is mandatory before downstream jobs. See docs/VISUAL_LENS_RUN.md and docs/VISUAL_LENS_STATUS.md.
