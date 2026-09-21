# Aperture final round

All planned states complete. New protocol: original support labels are split between fitting and held-out calibration; no extra query labels are used.

Primary analysis: stable candidate NLL, before and after positive temperature scaling, with LoRA schedule selected on calibration labels only. F1/BA/Brier/ECE and all arms are retained. NLL is computed from logsumexp without probability clipping; legacy-clipped NLL is also stored in raw JSON.

Corruption results are a paired 60-query, seed-0 probe and are NOT an official corruption benchmark. They do not prove semantic nuisance removal. Residual-dose sweeps are read-only diagnostics, not query-selected replacement models. Latency SD describes repeats, not support-seed uncertainty.

Use table_nll.tex, table_macro_f1.tex and table_ece.tex as editable LaTeX table fragments. Keep these separate from the historical full-support experiment tables.
