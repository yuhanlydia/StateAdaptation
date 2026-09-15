# Cross-task completion audit

| Requirement | Evidence | Status |
|---|---|---|
| Do not rerun BRIGHT | No BRIGHT jobs launched; all new runs are under `runs/oral/{qwen,phi}` | PASS |
| Fixed query / support-only seeds | `configs/oral_generalization.yaml`; all Qwen paired JSON reports have `query_ids_identical: true` | PASS |
| Camelyon17 three-seed matrix | `reports/camelyon_*_multiseed.json`, five Qwen arms | PASS |
| ManipBench Q1 three-domain, three-seed matrix | `reports/manipbench_*_multiseed.json`, five Qwen arms per domain | PASS |
| Integrity and reproducibility audit | `reports/task_run_audit.json`: 67/67 Qwen prediction directories passed; the final Phi audit passed 8/8 diagnostic prediction directories; pytest 52/52 | PASS |
| Six-hour execution contract | YAML budget, `scripts/run_oral_batch.sh`, debug allowance | PASS |
| Phi single-image backend and gates | Four Frozen gate runs; Camelyon and two droid domains fail gate, bridge barely passes | PASS (gate-limited) |
| Phi adapted bridge gate matrix | Consistent SDPA Frozen/LoRA/Random-KV/Gradient-Cov rows; droid/Camelyon gates stop expansion | PASS (gate-limited) |

The Qwen primary cross-task study is complete and reproducible. Phi bridge
adapted diagnostics are now complete under the consistent SDPA path; the
remaining absence of Phi multi-seed rows is mandated by failed admission gates,
not an omitted run.
