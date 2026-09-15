#!/usr/bin/env python3
"""Analyze captured gradient energy, basis stability, and outcome correlation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def principal_angles(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    singular = torch.linalg.svdvals(a.float().T @ b.float()).clamp(0, 1)
    return torch.rad2deg(torch.arccos(singular)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="JSON list with basis_path, covariance_path, delta_f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.records).read_text())
    enriched = []
    for row in records:
        basis = torch.load(row["basis_path"], map_location="cpu").float()
        covariance = torch.load(row["covariance_path"], map_location="cpu").float()
        captured = torch.trace(basis.T @ covariance @ basis) / torch.trace(covariance).clamp_min(1e-12)
        enriched.append({**row, "captured_gradient_energy": float(captured)})
    rho, p = spearmanr(
        [r["captured_gradient_energy"] for r in enriched],
        [r["delta_f1"] for r in enriched],
    )
    output = {"records": enriched, "spearman_energy_vs_delta_f1": {"rho": float(rho), "p": float(p)}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["spearman_energy_vs_delta_f1"], indent=2))


if __name__ == "__main__":
    main()
