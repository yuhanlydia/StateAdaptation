#!/usr/bin/env python3
"""Generate experiment table + figure for the ICASSP paper from disk metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path("runs/all_datasets")
OUT = Path("paper")

EVENTS = [
    "turkey-earthquake",
    "la_palma-volcano",
    "beirut-explosion",
    "congo-volcano",
    "haiti-earthquake",
    "libya-flood",
    "marshall-wildfire",
    "bata-explosion",
    "noto-earthquake",
    "morocco-earthquake",
]
EVENTS.insert(0, "hawaii-wildfire")

SHORT = {
    "hawaii-wildfire": "hawaii",
    "la_palma-volcano": "la-palma",
    "marshall-wildfire": "marshall",
}


def short(ev: str) -> str:
    return SHORT.get(ev, ev.replace("_", "-").split("-")[0])

# Events whose runs live under a custom directory/naming in the repo.
OVERRIDES = {
    "hawaii-wildfire": {
        "source": "runs/hawaii_min/source_eval_300/metrics.json",
        "kv": "runs/hawaii_min/kv_eval_300_r16/metrics.json",
    },
}


def agg(d):
    m = json.load(open(d))
    return m.get("aggregate", m)


def collect():
    rows = []
    for ev in EVENTS:
        if ev in OVERRIDES:
            se, ke = Path(OVERRIDES[ev]["source"]), Path(OVERRIDES[ev]["kv"])
        else:
            se = BASE / ev / "source_eval" / "metrics.json"
            ke = BASE / ev / "kv_eval" / "metrics.json"
        if not (se.exists() and ke.exists()):
            continue
        s, k = agg(se), agg(ke)
        rows.append(
            {
                "event": ev,
                "s_f1": s["macro_f1"],
                "k_f1": k["macro_f1"],
                "s_nll": s["nll"],
                "k_nll": k["nll"],
                "s_bac": s.get("balanced_accuracy", float("nan")),
                "k_bac": k.get("balanced_accuracy", float("nan")),
                "s_mae": s.get("ordinal_mae", float("nan")),
                "k_mae": k.get("ordinal_mae", float("nan")),
                "count": k.get("count", 300),
            }
        )
    return rows


def fmt(x, sign=False):
    pre = "+" if sign and x > 0 else ""
    return f"{pre}{x:.3f}"


def write_tables(rows):
    OUT.mkdir(exist_ok=True)
    lines = []
    for r in rows:
        ev = short(r["event"])
        lines.append(
            f"{ev} & {r['s_f1']:.3f} & {r['k_f1']:.3f} & {fmt(r['k_f1']-r['s_f1'],True)} "
            f"& {r['s_nll']:.2f} & {r['k_nll']:.2f} & {fmt(r['k_nll']-r['s_nll'],True)} "
            f"& {fmt(r['k_bac']-r['s_bac'],True)} \\\\"
        )
    if rows:
        df1 = np.mean([r["k_f1"] - r["s_f1"] for r in rows])
        dnll = np.mean([r["k_nll"] - r["s_nll"] for r in rows])
        s_f1 = np.mean([r["s_f1"] for r in rows])
        k_f1 = np.mean([r["k_f1"] for r in rows])
        n = len(rows)
        lines.append("\\midrule")
        lines.append(
            f"\\textit{{mean ({n} folds)}} & {s_f1:.3f} & {k_f1:.3f} & {fmt(df1,True)} "
            f"& -- & -- & {fmt(dnll,True)} & -- \\\\"
        )
        lines.append(
            f"\\textit{{wins (KV $>$ S)}} & \\multicolumn{{3}}{{c}}{{{sum(1 for r in rows if r['k_f1']>r['s_f1'])}/{n} (F1)}} "
            f"& \\multicolumn{{4}}{{c}}{{{sum(1 for r in rows if r['k_nll']<r['s_nll'])}/{n} (NLL)}} \\\\"
        )
    body = "\n".join(lines).strip()
    if not rows:
        body = "\\textit{(no completed folds on disk yet)} & -- & -- & -- & -- & -- & -- \\\\"
    else:
        # Reminder row for the folds whose runs have not completed.
        done = {r["event"] for r in rows}
        missing = [ev for ev in EVENTS if ev not in done]
        if missing:
            body += "\n" + (
                "\\textit{(pending)} & -- & -- & -- & -- & -- & \\multicolumn{1}{c}{"
                + ", ".join(short(m) for m in missing)
                + "} \\\\"
            )
    # Complete document-level table environment (no \input inside tabular).
    column_spec = "l lllllll"
    content = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{" + column_spec + "}\n"
        "\\toprule\n"
        "Event & \\multicolumn{3}{c}{$r{=}16$, $T{=}4$, $\\alpha{=}0.5$} & \\multicolumn{4}{c}{gain over source} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
        " & Source-F1 & KV-F1 & $\\Delta$F1 & Src-NLL & KV-NLL & $\\Delta$NLL & $\\Delta$bAcc \\\\\n"
        + body
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Leave-one-event-out on \\textsc{BRIGHT}. "
        "Rows are emitted from disk metrics by \\texttt{paper/make\\_figs\\_and\\_tables.py}; "
        "folds still queued print as \\emph{pending}.}\n"
        "\\label{tab:main}\n"
        "\\end{table*}\n"
    )
    (OUT / "tables" / "main_table.tex").write_text(content)
    (OUT / "tables" / "main_body.tex").write_text(body + "\n")


def plot(rows):
    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [short(r["event"]) for r in rows] if rows else ["pending"]
    if not rows:
        labels = ["waiting"]
    x = np.arange(len(rows)) if rows else np.arange(1)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for xi, vals, lab in (
        (x - w / 2, [r["s_f1"] for r in rows] if rows else [0.4], "Source (LoRA)"),
        (x + w / 2, [r["k_f1"] for r in rows] if rows else [0.4], "KV-TTT (proposed)"),
    ):
        ax.bar(xi, vals, w, label=lab)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "per_event_f1.pdf")
    fig.savefig(OUT / "figures" / "per_event_f1.png", dpi=200)


if __name__ == "__main__":
    rows = collect()
    for r in rows:
        r["short"] = short(r["event"])
    write_tables(rows)
    plot(rows)
    print(f"wrote tables for {len(rows)} events")
    for r in rows:
        print(r["short"], f"F1 {r['s_f1']:.3f}->{r['k_f1']:.3f}", f"NLL {r['s_nll']:.2f}->{r['k_nll']:.2f}")