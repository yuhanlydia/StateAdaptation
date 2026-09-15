#!/usr/bin/env python3
"""Fail if publication-facing table values diverge from run artifacts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "neurips"
EVENTS = (
    ("hawaii-wildfire", "Hawaii wildfire", "Hawaii"),
    ("libya-flood", "Libya flood", "Libya"),
    ("noto-earthquake", "Noto earthquake", "Noto"),
    ("turkey-earthquake", "Turkey earthquake", "Turkey"),
)
ALPHAS = (("0p5", ".5"), ("1", "1"), ("2", "2"), ("3", "3"), ("5", "5"), ("10", "10"))


def metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f1(event: str, relative: str) -> float:
    return metrics(RUNS / event / relative / "metrics.json")["macro_f1"]


def require(text: str, needle: str, source: str) -> None:
    if needle not in text:
        raise AssertionError(f"{source} is missing artifact-derived row: {needle}")


def main() -> None:
    body = (ROOT / "paper" / "tables" / "main_body.tex").read_text(encoding="utf-8")
    paper = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    sweep = (ROOT / "reports" / "rank16_full_diagonal_alpha_sweep.md").read_text(encoding="utf-8")
    fixed8 = (ROOT / "reports" / "fixed8_and_scaling_diagnostics.md").read_text(encoding="utf-8")

    columns = []
    lora_values = []
    full_values = []
    diagonal_values = []
    raw_values = []
    nll_values = {"Raw VLM": [], "LoRA fixed8": [], "Diagonal KV": [], "Full KV": []}
    for event, long_name, short_name in EVENTS:
        raw = f1(event, "original_eval")
        lora = f1(event, "support24_lora_fixed8_eval")
        full = f1(event, "rank16_full_alpha3/eval")
        diagonal = f1(event, "rank16_diagonal_alpha3/eval")
        columns.append((raw, lora, full, diagonal))
        raw_values.append(raw); lora_values.append(lora); full_values.append(full); diagonal_values.append(diagonal)
        nll_values["Raw VLM"].append(metrics(RUNS / event / "original_eval" / "metrics.json")["nll"])
        nll_values["LoRA fixed8"].append(metrics(RUNS / event / "support24_lora_fixed8_eval" / "metrics.json")["nll"])
        nll_values["Diagonal KV"].append(metrics(RUNS / event / "rank16_diagonal_alpha3" / "eval" / "metrics.json")["nll"])
        nll_values["Full KV"].append(metrics(RUNS / event / "rank16_full_alpha3" / "eval" / "metrics.json")["nll"])
        require(fixed8, f"| {long_name} | {lora:.4f} |", "fixed8 report")

        for mode, label in (("full", "Full"), ("diagonal", "Diagonal")):
            values = [f1(event, f"rank16_{mode}_alpha{tag}/eval") for tag, _ in ALPHAS]
            best_index = max(range(len(values)), key=values.__getitem__)
            cells = [f"{value:.4f}" for value in values]
            cells[best_index] = f"**{cells[best_index]}**"
            report_row = f"| {long_name} | {lora:.4f} | " + " | ".join(cells)
            require(sweep, report_row, f"rank16 {label} report")
            tex_cells = [f".{f'{value:.4f}'.split('.')[1]}" for value in values]
            tex_cells[best_index] = rf"\textbf{{{tex_cells[best_index]}}}"
            require(paper, f"{short_name} & " + " & ".join(tex_cells) + r"\\", f"paper {label} appendix")

    methods = (
        ("Raw VLM", raw_values, "0"),
        ("LoRA fixed8", lora_values, "10,092,544"),
        ("Diagonal KV", diagonal_values, "64"),
        ("Full KV", full_values, "1,024"),
    )
    event_winners = [max(values[i] for _, values, _ in methods) for i in range(4)]
    mean_winner = max(sum(values) / 4 for _, values, _ in methods)
    nll_winner = min(sum(nll_values[name]) / 4 for name, _, _ in methods)
    for name, values, learned in methods:
        cells = []
        for i, value in enumerate(values):
            cell = f"{value:.4f}"
            if value == event_winners[i]: cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        mean = sum(values) / 4
        mean_cell = f"{mean:.4f}"
        if mean == mean_winner: mean_cell = rf"\textbf{{{mean_cell}}}"
        mean_nll = sum(nll_values[name]) / 4
        nll_cell = f"{mean_nll:.4f}"
        if mean_nll == nll_winner: nll_cell = rf"\textbf{{{nll_cell}}}"
        learned_cell = rf"\textbf{{{learned}}}" if name == "Diagonal KV" else learned
        require(body, f"{name} & " + " & ".join(cells + [mean_cell, nll_cell, learned_cell]) + r"\\", "main_body.tex")

    ablations = {
        "Support": ("ablation_support12_full_a3", "support24_kv_full_a3", "ablation_support48_full_a3"),
        "Layers": ("ablation_layer14_full_a3", "ablation_layer27_full_a3", "support24_kv_full_a3"),
        "Rank": ("support24_kv_full_a3", "ablation_rank8_full_a3", "ablation_rank16_full_a3", "ablation_rank32_full_a3"),
        "Updates": ("ablation_steps1_full_a3", "ablation_steps2_full_a3", "support24_kv_full_a3", "ablation_steps8_full_a3"),
    }
    for factor, arms in ablations.items():
        values = [sum(f1(event, f"{arm}/eval") for event, _, _ in EVENTS) / 4 for arm in arms]
        for value in values:
            require(paper, f".{f'{value:.4f}'.split('.')[1]}", f"paper {factor} ablation")

    print("publication table audit passed: main, fixed8, 48-run alpha, and ablations")


if __name__ == "__main__":
    main()
