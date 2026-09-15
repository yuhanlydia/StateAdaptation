# Writing design and evidence ledger

## Title and narrative

**A Learned Visual Lens: Low-Rank State Adaptation for Vision–Language Models**

One manuscript, one abstract, one introduction. All body text, equations, tables, the algorithm, and the editable method schematic are in `main.tex`. No old VIGOR naming is printed. “Visual Lens” is a descriptive method name, not an additional acronym.

**Narrative spine:** a measured 1,024-variable vs10.1M-variable F1 contrast -> what should few-shot support be allowed to change? -> a support-derived linear visual subspace -> a bounded residual lens -> complete main comparisons -> controlled ablations -> why shared energy is not the same as a useful controller -> specific limits and conclusion.

The initial observation is genuinely measured, but we do not claim that parameter-count reversals have never been observed in machine learning. The paper's particular contribution is the support-conditioned visual-state construction and its measured scope. “Lens” is a metaphor for internal processing, not a demonstrated optical defect, nonlinear manifold, image-super-resolution algorithm, or new VLA policy.

## Writing sources actually consulted

ARIS repository: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep
Snapshot: `f1bd907b58f653131ebe6807c482e2554e07f9b9`.

Read:
- `skills/skills-codex/paper-write/SKILL.md`
- `skills/skills-codex/shared-references/writing-principles.md`

ARIS explicitly credits https://github.com/Adkid-Zephyr/anti-defensive-writing-Skill for its anti-defensive-writing rules. We used ARIS's retrieved rule text: match scope to evidence, state the supported claim directly, preserve unfavorable values, put generic caveats in Limitations, never replace missing evidence with a more confident synonym, and organize experiments by their argumentative role.

The user Library item `Scientific_Figure_Style_Atlas.md` was recovered and read. It contains the previous pipeline/results figure workflows and the three-layout, fact-lock rules. Its standalone underlying SKILL files were not separately retrieved. The prompts follow the recovered atlas rather than pretending a missing file was loaded.

No external GPT/Codex reviewer or autonomous review loop was run in this session. The manuscript received an inline mathematical, source, numerical, citation, and layout check. Skill use is not equivalent to an independent review.

## Source of truth

Repository: https://github.com/Yunbo-max/EventTune
Inspected main: `3150cae1eecf2ee82e508d32fe7f2e80913f8149` (commit message: freeze evidence and rewrite ICLR2027 story).

No newer experiment numbers were inferred from a folder name or a claimed pending run. No GPU run or repository push occurred during this writing task.

| Manuscript item | Source at inspected commit | Scope retained |
|---|---|---|
| Abstract, Fig1, Table1 | `paper/tables/main_table.tex`; `paper/main.tex` | Rank16 primary, four fixed supports; not three-seed matched LoRA |
| Primary BA/NLL | `paper/main.tex`, complete fixed metrics appendix | Four-event means; BA near chance and LoRA NLL retained |
| Table2 extra backbones | `reports/bright_uniform_two_model_results.md`; `reports/bright_uniform_cross_model.md`; `paper/main.tex` | One support/event; generic LoRA lr2e-4 vs primary lr1e-4 |
| Table3 corrected InternVL | `reports/internvl3_corrected_task_results.md` | Three seeds; Answer-prefix task path; old raw-label results excluded |
| Table3 Guardian | `reports/guardian_failure_results.md` | Four-update base and eight-update follow-up shown separately; Random missing |
| Figure3 and basis intervals | `reports/extended_neurips_results.md` | Rank5 except rank1 mean control; random3 basis seeds |
| Three-seed robustness text | `reports/multiseed_significance.md`; `paper/main.tex` | Compact rank5 Lens-vs-Frozen, not LoRA-matched |
| Table4 sensitivity | `paper/main.tex` supervised ablations; extended report | Exploratory one-factor sweeps; peak not promoted |
| Figure4 prior mixture | `reports/task_subspace_diagnostics.md` | Oracle statistic only; B held fixed; no new F1 |
| Table5 site/query basis | `reports/task_subspace_diagnostics.md` | Seed0; rank16; controller fit remains on support; QKVO has more coefficients |
| Appendix Qwen task results | `reports/task_vlm_formal_results.md`, Qwen rows only | Historical InternVL rows in that file excluded |
| Appendix efficiency | `reports/inference_efficiency_benchmark.md`; `paper/main.tex` | Rank5 timings and full artifact storage, not rank16 timing |
| Appendix optimization follow-up | `paper/main.tex`; user-provided experiment transcript | Mixed3-seed estimates and seed0 screens distinguished |
| Figure5 module alignment | `reports/task_subspace_diagnostics.md` | Three-seed module means; not a global concatenated cosine |
| Actual method and loss conventions | `src/eventttt/kv_ttt.py`; `task_kv.py`; `task_qwen.py`; `bright_vlm.py`; `scripts/analyze_directional_geometry.py` | Support basis, Full norm bound, mask, sum/mean loss and energy-weighted cosine retained |

`plotting/figure_data.json` stores every plotted number and its source. It is a transcription of recorded results, not a new experiment output. Main tables are editable LaTeX, not screenshots.

## Corrections to earlier conversational interpretations

These changes are deliberate and documented rather than silently rewriting the research history:

- A query-gradient eigenbasis is not “perfect geometry” or a global capacity oracle. A failure to close LoRA's gap does not prove KV can never solve the task.
- Directional agreement and actuator utility are hypotheses/diagnostics, not established necessary-and-sufficient conditions.
- The current primary basis comes from labeled target support. Offline historical-event basis learning is not the primary method in this manuscript.
- The raw pretrained checkpoint is used in the main comparisons; the draft does not invent an additional source-task training stage.
- Lower NLL is probabilistic-loss evidence, not by itself proof of calibration improvement.
- A/B/C/D in ManipBench identify candidate positions, not fixed cross-example action classes.
- QKVO changes the number of controlled modules as well as intervention site. Its gains are not a budget-matched location theorem.
- The reported aggregate κ is an energy-weighted mean of per-module cosines. Class-specific κ uses class-specific bases. The query-prior probe instead retains a common aggregate basis.
- The prefix `Answer:` contributes support gradients as well as changing the scale of the averaged token loss. It cancels from exact within-example candidate ranking as a shared autoregressive prefix, but not from training.
- Task loss sums and BRIGHT loss means imply different regularization/clipping scales; the original results are not retroactively relabeled as a common normalized objective.
- The 1,024 vs10.1M count excludes fixed bases. The published timing point belongs to rank5.
- “Fixed configuration” does not erase the fact that related sweeps were observed on the same queries. Prospective verification requires separately held-out data.

## Bibliographic checking

The inherited bibliography was checked against primary publication pages/arXiv records. TDA is cited as CVPR2024, ReFT and LoFiT as NeurIPS2024, ManipBench as CoRL2025 (PMLR305). SPD and LoRA-TTT retain their retrieved arXiv status. Guardian uses the revised August2026 record. No guessed page ranges or unverified acceptance claims were added. The primary comparison's LoRA is not called an implementation of the distinct CLIP-based LoRA-TTT algorithm.

Official ICLR2027 style files are retained byte-for-byte. Only the manuscript text/layout choices changed. The style example, sample bibliography, alternate drafts, and old title-selection files are not in this project.

## Audit outcome

The manuscript is a complete nine-page-main-text draft, not a certification that every experimental issue is settled. High-value remaining checks are in `EXPERIMENTS_TO_COMPLETE.md`. Unrun arms have no invented numerical entries. The optional image-evidence figure is a prompt specification only and does not appear in the current paper.
